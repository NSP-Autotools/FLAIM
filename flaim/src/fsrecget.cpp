//-------------------------------------------------------------------------
// Desc:	Read a record from the database.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Library Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, contact Novell, Inc.
//
// To contact Novell about this file by physical or electronic mail, 
// you may find current contact information at www.novell.com.
//
// $Id$
//------------------------------------------------------------------------------

#include "flaimsys.h"

#define NUM_FIELDS_IN_ARRAY	16

typedef struct Field_State
{
	FLMBYTE *		pElement;			// Points to the element within block
	FLMUINT			uiFieldType;		// Storage field type
	FLMUINT			uiFieldLen;			// Length of storage if known
	FLMUINT			uiPosInElm;			// Position within the element
	FLMUINT			uiTagNum;			// Field tag/data dictionary number
	FLMUINT			uiLevel;				// Current level number
	FLMUINT			uiEncId;				// EncDef ID if encrypted
	FLMUINT			uiEncFieldLen; 	// Encrypted field length
} FSTATE;

typedef struct DATAPIECE
{
	FLMBYTE *		pData;				// Points to data within the block.
	FLMUINT			uiLength;			// Length of this data piece
	DATAPIECE *		pNext;				// Next data piece or NULL
} DATAPIECE;

typedef struct Temporary_Field
{
	FLMUINT			uiLevel;
	FLMUINT			uiFieldID;
	FLMUINT			uiFieldType;
	FLMUINT			uiFieldLen;
	FLMUINT			uiEncId;
	FLMUINT			uiEncFieldLen;
	DATAPIECE		DataPiece;
} TEMPFIELD;

typedef struct FLDGROUP
{
	TEMPFIELD		pFields[ NUM_FIELDS_IN_ARRAY];
	FLDGROUP *		pNext;
} FLDGROUP;

typedef struct LOCKED_BLOCK
{
	SCACHE *			pSCache;
	LOCKED_BLOCK *	pNext;
} LOCKED_BLOCK;

FSTATIC RCODE FSGetFldOverhead(
	FDB *				pDb,
	FSTATE * 		fState);

/****************************************************************************
Desc: Retrieves a record given a DRN
****************************************************************************/
RCODE FSReadRecord(
	FDB *				pDb,
	LFILE *			pLFile,
	FLMUINT			uiDrn,
	FlmRecord **	ppRecord,
	FLMUINT *		puiRecTransId,
	FLMBOOL *		pbMostCurrent)
{
	RCODE				rc = FERR_OK;
	BTSK				stackBuf[BH_MAX_LEVELS];
	BTSK *			pStack = NULL;
	FLMBYTE			pKeyBuf[DIN_KEY_SIZ];
	FLMBYTE			pDrnBuf[DIN_KEY_SIZ];

	FSInitStackCache( &stackBuf[0], BH_MAX_LEVELS);

	pStack = stackBuf;
	pStack->pKeyBuf = pKeyBuf;

	// Search the B-TREE for the record

	f_UINT32ToBigEndian( (FLMUINT32)uiDrn, pDrnBuf);
	if (RC_OK( rc = FSBtSearch( pDb, pLFile, &pStack, pDrnBuf, 4, 0)))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		
		if ((pStack->uiCmpStatus == BT_EQ_KEY) && (uiDrn != DRN_LAST_MARKER))
		{
			rc = FSReadElement( pDb, &pDb->TempPool, pLFile, uiDrn, pStack, TRUE,
									 ppRecord, puiRecTransId, pbMostCurrent);
		}
	}

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return (rc);
}

/****************************************************************************
Desc: Low-level routine to retrieve and build an internal tree record.
****************************************************************************/
RCODE FSReadElement(
	FDB *					pDb,
	F_Pool *				pPool,
	LFILE *				pLFile,
	FLMUINT				uiDrn,
	BTSK *				pStack,
	FLMBOOL				bOkToPreallocSpace,
	FlmRecord **		ppRecord,
	FLMUINT *			puiRecTransId,
	FLMBOOL *			pbMostCurrent)
{
	RCODE					rc = FERR_OK;
	FlmRecord *			pRecord = NULL;
	FLMBYTE *			pCurElm;
	void *				pvPoolMark = pPool->poolMark();
	FLMUINT				uiElmRecLen;
	FLMUINT				uiFieldLen;
	FLMUINT				uiLowestTransId;
	FLMUINT				uiFieldCount;
	FLMUINT				uiTrueDataSpace;
	FLMUINT				uiFieldPos;
	FLMBOOL				bMostCurrent;
	TEMPFIELD *			pField;
	FLDGROUP *			pFldGroup = NULL;
	FLDGROUP*			pFirstFldGroup = NULL;
	DATAPIECE *			pDataPiece;
	LOCKED_BLOCK *		pLockedBlock = NULL;
	FSTATE				fState;
	FLMUINT				uiEncFieldLen;

	// Initialize variables

	fState.uiLevel = 0;
	uiFieldCount = 0;
	uiTrueDataSpace = 0;
	uiFieldPos = NUM_FIELDS_IN_ARRAY;

	// Check to make sure we are positioned at the first element.

	pCurElm = CURRENT_ELM( pStack);
	if (!BBE_IS_FIRST( pCurElm))
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	uiLowestTransId = FB2UD( &pStack->pBlk[BH_TRANS_ID]);
	bMostCurrent = (pStack->pSCache->uiHighTransID == 0xFFFFFFFF) 
											? TRUE 
											: FALSE;

	// Loop on each element in the record

	for (;;)
	{

		// Setup all variables to process the current element

		uiElmRecLen = BBE_GET_RL( pCurElm);
		if (uiElmRecLen == 0)
		{
			rc = RC_SET( FERR_EOF_HIT);
			break;
		}

		pCurElm += BBE_REC_OFS( pCurElm);
		fState.uiPosInElm = 0;

		// Loop on each field within this element.

		while (fState.uiPosInElm < uiElmRecLen)
		{
			fState.pElement = pCurElm;
			if (RC_BAD( rc = FSGetFldOverhead( pDb, &fState)))
			{
				goto Exit;
			}

			uiFieldLen = fState.uiFieldLen;
			uiEncFieldLen = fState.uiEncFieldLen;

			// Old record info data - skip past for now

			if (fState.uiTagNum == 0)
			{
				fState.uiPosInElm += (uiEncFieldLen ? uiEncFieldLen : uiFieldLen);
				continue;
			}

			if (!pRecord)
			{

				// Create a new data record or use the existing data record.

				if (*ppRecord)
				{
					if ((*ppRecord)->isReadOnly())
					{
						(*ppRecord)->Release();
						*ppRecord = NULL;

						if ((pRecord = f_new FlmRecord) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else
					{

						// Reuse the existing FlmRecord object.

						pRecord = *ppRecord;
						*ppRecord = NULL;
						pRecord->clear();
					}
				}
				else
				{
					if ((pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}

				pRecord->setContainerID( pLFile->uiLfNum);
				pRecord->setID( uiDrn);
				if (pLFile->bMakeFieldIdTable)
				{
					pRecord->enableFieldIdTable();
				}
			}

			// Check if out of fields in the tempoary field group.

			if (uiFieldPos >= NUM_FIELDS_IN_ARRAY)
			{
				FLDGROUP *		pTempFldGroup;

				uiFieldPos = 0;

				// Allocate the first field group from the pool.
				
				if( RC_BAD( rc = pPool->poolAlloc( 
					sizeof( FLDGROUP), (void **)&pTempFldGroup)))
				{
					goto Exit;
				}

				pTempFldGroup->pNext = NULL;
				if (pFldGroup)
				{
					pFldGroup->pNext = pTempFldGroup;
				}
				else
				{
					pFirstFldGroup = pTempFldGroup;
				}

				pFldGroup = pTempFldGroup;
			}

			uiFieldCount++;
			pField = &pFldGroup->pFields[uiFieldPos++];
			pField->uiLevel = fState.uiLevel;
			pField->uiFieldID = fState.uiTagNum;
			pField->uiFieldType = fState.uiFieldType;
			pField->uiFieldLen = fState.uiFieldLen;
			pField->uiEncId = fState.uiEncId;
			pField->uiEncFieldLen = fState.uiEncFieldLen;
			pDataPiece = &pField->DataPiece;

			if (uiFieldLen || uiEncFieldLen)
			{
				FLMUINT	uiDataPos = 0;

				if (fState.uiEncFieldLen)
				{
					uiTrueDataSpace += FLM_ENC_FLD_OVERHEAD;

					// Binary data needs to account for alignment issues.

					if (fState.uiFieldType == FLM_BINARY_TYPE)
					{

						// Adjust for the decrypted data.

						uiTrueDataSpace = ((uiTrueDataSpace + FLM_ALLOC_ALIGN) & 
												 (~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
					}

					uiTrueDataSpace += fState.uiFieldLen + fState.uiEncFieldLen;

					// Store the encrypted field length rather than the
					// decrypted field length This will allow the gathering of
					// the encrypted or decrypted field data to use the same
					// code.

					uiFieldLen = uiEncFieldLen;
				}
				else if (fState.uiFieldLen > 4)
				{

					// Binary data needs to account for alignment issues.

					if (fState.uiFieldType == FLM_BINARY_TYPE)
					{
						if (fState.uiFieldLen >= 0xFF)
						{

							// Align so that the data is aligned - not the length

							uiTrueDataSpace += sizeof(FLMUINT32);
							uiTrueDataSpace = ((uiTrueDataSpace + FLM_ALLOC_ALIGN) &
														(~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
							uiTrueDataSpace -= sizeof(FLMUINT32);
						}
						else
						{
							uiTrueDataSpace = ((uiTrueDataSpace + FLM_ALLOC_ALIGN) &
														(~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
						}
					}

					uiTrueDataSpace += fState.uiFieldLen;

					// Field values with lengths greater than 255 bytes are
					// stored length-preceded. A single byte flags field
					// precedes the length.

					if (fState.uiFieldLen >= 0xFF)
					{
						uiTrueDataSpace += sizeof(FLMUINT32) + 1;
					}
				}

				// Value may start in the next element.

				while (uiDataPos < uiFieldLen)
				{

					// Need to read next element for the value portion?

					if (fState.uiPosInElm >= uiElmRecLen)
					{
						if (BBE_IS_LAST( CURRENT_ELM( pStack)))
						{
							rc = RC_SET( FERR_DATA_ERROR);
							goto Exit;
						}

						// If we are going to the next block, lock down this
						// block because data pointers are pointing to it.

						if (RC_BAD( FSBlkNextElm( pStack)))
						{
							LOCKED_BLOCK *	pLastLockedBlock = pLockedBlock;

							if( RC_BAD( rc = pPool->poolAlloc( sizeof( LOCKED_BLOCK),
								(void **)&pLockedBlock)))
							{
								goto Exit;
							}
							
							ScaHoldCache( pStack->pSCache);
							pLockedBlock->pSCache = pStack->pSCache;
							pLockedBlock->pNext = pLastLockedBlock;

							if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
							{
								rc = (rc == FERR_BT_END_OF_DATA) 
												? RC_SET( FERR_DATA_ERROR) 
												: rc;
								goto Exit;
							}

							if (uiLowestTransId > FB2UD( &pStack->pBlk[BH_TRANS_ID]))
							{
								uiLowestTransId = FB2UD( &pStack->pBlk[BH_TRANS_ID]);
							}

							if (!bMostCurrent)
							{
								bMostCurrent = 
									(pStack->pSCache->uiHighTransID == 0xFFFFFFFF) 
												? TRUE 
												: FALSE;
							}
						}

						pCurElm = CURRENT_ELM( pStack);
						uiElmRecLen = BBE_GET_RL( pCurElm);
						pCurElm += BBE_REC_OFS( pCurElm);
						fState.uiPosInElm = 0;
					}

					// Compare number of bytes left if value <= # bytes left
					// in element

					if (uiFieldLen - uiDataPos <= uiElmRecLen - fState.uiPosInElm)
					{
						FLMUINT	uiDelta = uiFieldLen - uiDataPos;

						pDataPiece->pData = &pCurElm[fState.uiPosInElm];
						pDataPiece->uiLength = uiDelta;
						fState.uiPosInElm += uiDelta;
						pDataPiece->pNext = NULL;
						break;
					}
					else
					{

						// Take what is there and get next element to grab some
						// more.

						FLMUINT			uiBytesToMove = uiElmRecLen - fState.uiPosInElm;
						DATAPIECE *		pNextDataPiece;

						pDataPiece->pData = &pCurElm[fState.uiPosInElm];
						pDataPiece->uiLength = uiBytesToMove;
						fState.uiPosInElm += uiBytesToMove;
						uiDataPos += uiBytesToMove;

						if( RC_BAD( rc = pPool->poolAlloc( sizeof( DATAPIECE),
							(void **)&pNextDataPiece)))
						{
							goto Exit;
						}

						pDataPiece->pNext = pNextDataPiece;
						pDataPiece = pNextDataPiece;
					}
				}
			}
		}

		// Done?

		if (BBE_IS_LAST( CURRENT_ELM( pStack)))
		{
			break;
		}

		// Position to next element

		if (RC_BAD( FSBlkNextElm( pStack)))
		{
			LOCKED_BLOCK *		pLastLockedBlock = pLockedBlock;
			
			if( RC_BAD( rc = pPool->poolAlloc( sizeof( LOCKED_BLOCK),
				(void **)&pLockedBlock)))
			{
				goto Exit;
			}

			ScaHoldCache( pStack->pSCache);
			pLockedBlock->pSCache = pStack->pSCache;
			pLockedBlock->pNext = pLastLockedBlock;

			if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
			{
				if (rc == FERR_BT_END_OF_DATA)
				{
					rc = RC_SET( FERR_DATA_ERROR);
				}

				goto Exit;
			}

			if (uiLowestTransId > FB2UD( &pStack->pBlk[BH_TRANS_ID]))
			{
				uiLowestTransId = FB2UD( &pStack->pBlk[BH_TRANS_ID]);
			}

			if (!bMostCurrent)
			{
				bMostCurrent = (pStack->pSCache->uiHighTransID == 0xFFFFFFFF) 
												? TRUE 
												: FALSE;
			}
		}

		// Corruption Check.

		pCurElm = CURRENT_ELM( pStack);
		if (BBE_IS_FIRST( pCurElm))
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
	}

	if (pRecord)
	{
		void *		pvField;

		if (bOkToPreallocSpace)
		{
			if (RC_BAD( rc = pRecord->preallocSpace( 
				uiFieldCount, uiTrueDataSpace)))
			{
				goto Exit;
			}
		}

		pFldGroup = pFirstFldGroup;

		for (uiFieldPos = 0; uiFieldCount--; uiFieldPos++)
		{
			if (uiFieldPos >= NUM_FIELDS_IN_ARRAY)
			{
				uiFieldPos = 0;
				if ((pFldGroup = pFldGroup->pNext) == NULL)
				{
					break;
				}
			}

			pField = &pFldGroup->pFields[uiFieldPos];

			if (RC_BAD( rc = pRecord->insertLast( pField->uiLevel,
						  pField->uiFieldID, pField->uiFieldType, &pvField)))
			{
				goto Exit;
			}

			if (pField->uiFieldLen)
			{
				FLMBYTE *		pDataPtr;
				FLMBYTE *		pEncDataPtr;

				pDataPiece = &pField->DataPiece;
				if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
							  pField->uiFieldType, pField->uiFieldLen,
							  pField->uiEncFieldLen, pField->uiEncId,
							  (pField->uiEncId ? FLD_HAVE_ENCRYPTED_DATA : 0),
							  &pDataPtr, &pEncDataPtr)))
				{
					goto Exit;
				}

				do
				{
					if (pField->uiEncId)
					{
						f_memcpy( pEncDataPtr, pDataPiece->pData, pDataPiece->uiLength);
						pEncDataPtr += pDataPiece->uiLength;
					}
					else
					{
						f_memcpy( pDataPtr, pDataPiece->pData, pDataPiece->uiLength);
						pDataPtr += pDataPiece->uiLength;
					}

					pDataPiece = pDataPiece->pNext;
				} while (pDataPiece);

				// If the field is encrypted, we must decrypt it here.

				if (pField->uiEncId && !pDb->pFile->bInLimitedMode)
				{
					if (RC_BAD( rc = flmDecryptField( pDb->pDict, pRecord, pvField,
								  pField->uiEncId, &pDb->TempPool)))
					{
						goto Exit;
					}
				}
			}
		}
	}

	if (puiRecTransId)
	{
		*puiRecTransId = uiLowestTransId;
	}

	if (pbMostCurrent)
	{
		*pbMostCurrent = bMostCurrent;
	}

	if (*ppRecord)
	{
		flmAssert( 0);
		(*ppRecord)->Release();
	}
	
	pRecord->sortFieldIdTable();

	*ppRecord = pRecord;
	pRecord = NULL;

Exit:

	// Release all locked down blocks except the current block.

	while (pLockedBlock)
	{
		ScaReleaseCache( pLockedBlock->pSCache, FALSE);
		pLockedBlock = pLockedBlock->pNext;
	}

	pPool->poolReset( pvPoolMark);

	if (pRecord)
	{
		pRecord->Release();
	}

	// You are now positioned to the last element in the record

	return (rc);
}

/****************************************************************************
Desc: Read field overhead (level, field drn, and length information. This
		isolates the complexity of the storage formats.
****************************************************************************/
FSTATIC RCODE FSGetFldOverhead(
	FDB *				pDb,
	FSTATE *			fState)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pFieldOvhd = &fState->pElement[fState->uiPosInElm];
	FLMBYTE *		pElement = fState->pElement;
	FLMBYTE *		pTmp;
	FLMBOOL			bDoesntHaveFieldDef = TRUE;
	FLMUINT			uiFieldLen;
	FLMUINT			uiFieldType = 0;
	FLMUINT			uiTagNum;
	FLMBYTE			ucBaseFlags;
	FLMUINT			uiEncId = 0;
	FLMUINT			uiEncFieldLen = 0;

	if (FOP_IS_STANDARD( pFieldOvhd))
	{
		if (FSTA_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		uiFieldLen = FSTA_FLD_LEN( pFieldOvhd);
		uiTagNum = FSTA_FLD_NUM( pFieldOvhd);
		pFieldOvhd += FSTA_OVHD;
	}
	else if (FOP_IS_OPEN( pFieldOvhd))
	{
		if (FOPE_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		ucBaseFlags = (FLMBYTE) (FOP_GET_FLD_FLAGS( pFieldOvhd++));
		uiTagNum = (FLMUINT) * pFieldOvhd++;

		if (FOP_2BYTE_FLDNUM( ucBaseFlags))
		{
			uiTagNum += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiFieldLen = (FLMUINT) * pFieldOvhd++;
		if (FOP_2BYTE_FLDLEN( ucBaseFlags))
		{
			uiFieldLen += ((FLMUINT) * pFieldOvhd++) << 8;
		}
	}
	else if (FOP_IS_NO_VALUE( pFieldOvhd))
	{
		if (FNOV_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		ucBaseFlags = (FLMBYTE) (FOP_GET_FLD_FLAGS( pFieldOvhd++));
		uiTagNum = (FLMUINT) * pFieldOvhd++;
		if (FOP_2BYTE_FLDNUM( ucBaseFlags))
		{
			uiTagNum += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiFieldLen = uiFieldType = 0;
	}
	else if (FOP_IS_SET_LEVEL( pFieldOvhd))
	{

		// SET THE LEVEL Must be continuous with the next field

		fState->uiLevel -= FSLEV_GET( pFieldOvhd++);
		fState->uiPosInElm = (FLMUINT) (pFieldOvhd - pElement);

		rc = FSGetFldOverhead( pDb, fState);
		goto Exit;
	}
	else if (FOP_IS_TAGGED( pFieldOvhd))
	{
		bDoesntHaveFieldDef = FALSE;

		if (FTAG_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		ucBaseFlags = (FLMBYTE) (FOP_GET_FLD_FLAGS( pFieldOvhd));
		pFieldOvhd++;
		uiFieldType = (FLMUINT) (FTAG_GET_FLD_TYPE( *pFieldOvhd));
		pFieldOvhd++;
		uiTagNum = (FLMUINT) * pFieldOvhd++;

		if (FOP_2BYTE_FLDNUM( ucBaseFlags))
		{
			uiTagNum += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		// When storing the unregistered fields we cleared the high bit to
		// save on storage for VER11. The problem is that if a tag that is
		// not in the unregistered range (FLAIM TAGS) cannot be represented.
		// SO, we will XOR the high bit so 0x0111 is stored as 0x8111 and
		// 0x8222 is stored as 0x0222.

		uiTagNum ^= 0x8000;
		uiFieldLen = (FLMUINT) * pFieldOvhd++;
		if (FOP_2BYTE_FLDLEN( ucBaseFlags))
		{
			uiFieldLen += ((FLMUINT) * pFieldOvhd++) << 8;
		}
	}
	else if (FOP_IS_RECORD_INFO( pFieldOvhd))
	{
		bDoesntHaveFieldDef = FALSE;
		ucBaseFlags = *pFieldOvhd++;
		uiFieldLen = *pFieldOvhd++;

		if (FOP_2BYTE_FLDLEN( ucBaseFlags))
		{
			uiFieldLen += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiTagNum = 0;
	}
	else if (FOP_IS_ENCRYPTED( pFieldOvhd))
	{
		FLMBOOL	bTagSz;
		FLMBOOL	bLenSz;
		FLMBOOL	bENumSz;
		FLMBOOL	bELenSz;

		if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_60)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		bDoesntHaveFieldDef = FALSE;

		if (FENC_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		uiFieldType = (FLMUINT) (FENC_FLD_TYPE( pFieldOvhd));
		bTagSz = FENC_TAG_SZ( pFieldOvhd);
		bLenSz = FENC_LEN_SZ( pFieldOvhd);
		bENumSz = FENC_ETAG_SZ( pFieldOvhd);
		bELenSz = FENC_ELEN_SZ( pFieldOvhd);

		pFieldOvhd += 2;

		uiTagNum = (FLMUINT) * pFieldOvhd++;
		if (bTagSz)
		{
			uiTagNum += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiFieldLen = (FLMUINT) * pFieldOvhd++;
		if (bLenSz)
		{
			uiFieldLen += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiEncId = (FLMUINT) * pFieldOvhd++;
		if (bENumSz)
		{
			uiEncId += ((FLMUINT) * pFieldOvhd++) << 8;
		}

		uiEncFieldLen = (FLMUINT) * pFieldOvhd++;
		if (bELenSz)
		{
			uiEncFieldLen += ((FLMUINT) * pFieldOvhd++) << 8;
		}
	}
	else if (FOP_IS_LARGE( pFieldOvhd))
	{
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		bDoesntHaveFieldDef = FALSE;
		pTmp = pFieldOvhd;

		if (FLARGE_LEVEL( pFieldOvhd))
		{
			fState->uiLevel++;
		}

		pTmp++;

		uiFieldType = FLARGE_FLD_TYPE( pFieldOvhd);
		pTmp++;

		uiTagNum = FLARGE_TAG_NUM( pFieldOvhd);
		pTmp += 2;

		if ((uiFieldLen = FLARGE_DATA_LEN( pFieldOvhd)) <= 0x0000FFFF)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		pTmp += 4;

		if (FLARGE_ENCRYPTED( pFieldOvhd))
		{
			uiEncId = FLARGE_ETAG_NUM( pFieldOvhd);
			pTmp += 2;

			uiEncFieldLen = FLARGE_EDATA_LEN( pFieldOvhd);
			pTmp += 4;
		}

		pFieldOvhd = pTmp;
	}
	else
	{
		flmAssert( 0);
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	if (bDoesntHaveFieldDef)
	{

		// Get the field's storage type.

		if (RC_BAD( fdictGetField( pDb->pDict, uiTagNum, 
				&uiFieldType, NULL, NULL)))
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
	}

	// Set the fState return values

	fState->uiFieldType = uiFieldType;
	fState->uiFieldLen = uiFieldLen;
	fState->uiPosInElm = (FLMUINT) (pFieldOvhd - pElement);
	fState->uiTagNum = uiTagNum;
	fState->uiEncId = uiEncId;
	fState->uiEncFieldLen = uiEncFieldLen;

Exit:

	return (rc);
}
