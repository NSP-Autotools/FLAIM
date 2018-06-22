//-------------------------------------------------------------------------
// Desc:	Key and reference building routines.
// Tabs:	3
//
// Copyright (c) 1990-1992, 1994-2007 Novell, Inc. All Rights Reserved.
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

// Constants for checking to see if numbers are greater than what will fit
// in 32 bits.  the B at the beginning signifies a negative value

static const FLMBYTE gv_ucMinInt32 [6] =
{
	0xB2,0x14,0x74,0x83,0x64,0x8F
};

// Note that the last byte is 0xFF, but it could be 0xF1, 0xF2, etc - it really
// doesn't matter what is in the lower nibble, because the number terminates
// when it sees the F in the high nibble.

static const FLMBYTE gv_ucMaxUInt32 [6] =
{
	0x42,0x94,0x96,0x72,0x95,0xFF
};
							
#define KREF_TBL_SIZE					512			
#define KREF_TBL_THRESHOLD				400
#define KREF_POOL_BLOCK_SIZE			8192
#define KREF_TOTAL_BYTES_THRESHOLD	((KREF_POOL_BLOCK_SIZE * 3) - 250)

#define KY_SWAP(pKrefTbl, leftP, rightP) \
	pTempKref = pKrefTbl[leftP];			  \
	pKrefTbl[leftP] = pKrefTbl[rightP];	  \
	pKrefTbl[rightP] = pTempKref

FSTATIC FLMINT _KrefCompare(
	FLMUINT *			puiQsortFlags, 
	KREF_ENTRY * 		pKreftA, 
	KREF_ENTRY * 		pKreftB);

FSTATIC RCODE KYAddUniqueKeys(
	FDB * 				pDb);

FSTATIC RCODE _KrefQuickSort(
	FLMUINT *			puiQsortFlags,
	KREF_ENTRY **	 	pEntryTbl,
	FLMUINT				uiLowerBounds,
	FLMUINT				uiUpperBounds);

FSTATIC RCODE _KrefKillDups(
	FLMUINT *			puiQsortFlags,
	KREF_ENTRY **	 	pKrefTbl,
	FLMUINT *			puiKrefTotalRV);
	
FSTATIC RCODE flmProcessIndexedFld(
	FDB *					pDb,
	IXD *					pUseIxd,
	IFD *					pIfdChain,
	void **		 		ppPathFlds,
	FLMUINT				uiLeafFieldLevel,
	FLMUINT				uiAction,
	FLMUINT				uiContainerNum,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	FlmRecord *			pRecord,
	void *				pvField);

/****************************************************************************
Desc: Main driver for processing the fields in a record.
****************************************************************************/
RCODE flmProcessRecFlds(
	FDB *				pDb,
	IXD *				pIxd,
	FLMUINT			uiContainerNum,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord,
	FLMUINT			uiAction,
	FLMBOOL			bPurgedFldsOk,
	FLMBOOL *		pbHadUniqueKeys)
{
	RCODE				rc = FERR_OK;
	void *			pathFlds[GED_MAXLVLNUM + 1];
	FLMUINT			uiLeafFieldLevel;
	void *			pvField;
	FLMUINT			uiDbVersion = pDb->pFile->FileHdr.uiVersionNum;

	if ((pvField = pRecord->root()) == NULL)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	for (;;)
	{
		FLMUINT	uiItemType;
		IFD*		pIfdChain;
		FLMUINT	uiTagNum = pRecord->getFieldID( pvField);
		FLMUINT	uiFldState;
		FLMBOOL	bFldEncrypted;
		FLMUINT	uiEncFlags = 0;
		FLMUINT	uiEncId = 0;
		FLMUINT	uiEncState;
		FLMUINT	uiFieldType = pRecord->getDataType( pvField);

		if (RC_BAD( rc = fdictGetField( pDb->pDict, uiTagNum, &uiItemType,
					  &pIfdChain, &uiFldState)))
		{

			// Fill diagnostic error data.

			pDb->Diag.uiInfoFlags |= (FLM_DIAG_FIELD_NUM | FLM_DIAG_FIELD_TYPE);
			pDb->Diag.uiFieldNum = uiTagNum;
			pDb->Diag.uiFieldType = uiFieldType;
			goto Exit;
		}

		// Check for encryption.

		bFldEncrypted = pRecord->isEncryptedField( pvField);
		if (bFldEncrypted)
		{

			// May still proceed if the field is already encrypted.

			uiEncFlags = pRecord->getEncFlags( pvField);

			if (!(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA) &&
				 !pDb->pFile->bInLimitedMode)
			{
				uiEncId = pRecord->getEncryptionID( pvField);

				if (RC_BAD( rc = fdictGetEncInfo( pDb, uiEncId, NULL, &uiEncState)))
				{

					// Fill diagnostic error data.

					pDb->Diag.uiInfoFlags |= (FLM_DIAG_FIELD_NUM | FLM_DIAG_ENC_ID);
					pDb->Diag.uiFieldNum = uiTagNum;
					pDb->Diag.uiEncId = uiEncId;
					goto Exit;
				}

				// Check the state of the Encryption Record.

				if (uiEncState == ITT_ENC_STATE_PURGE)
				{

					// EncDef record has been marked as 'purged'. So, user is
					// not allowed to add new fields that are encrypted with
					// this EncDef Id.

					pDb->Diag.uiInfoFlags |= (FLM_DIAG_FIELD_NUM | FLM_DIAG_ENC_ID);
					pDb->Diag.uiFieldNum = uiTagNum;
					pDb->Diag.uiEncId = uiEncId;
					rc = RC_SET( FERR_PURGED_ENCDEF_FOUND);
					goto Exit;
				}
			}
			else if (!(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA) &&
						pDb->pFile->bInLimitedMode)
			{
				rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
				goto Exit;
			}
		}

		uiLeafFieldLevel = (FLMINT) pRecord->getLevel( pvField);
		pathFlds[uiLeafFieldLevel] = pvField;

		// Check the field state

		if (uiFldState == ITT_FLD_STATE_PURGE && bPurgedFldsOk == FALSE)
		{
			pDb->Diag.uiInfoFlags |= FLM_DIAG_FIELD_NUM;
			pDb->Diag.uiFieldNum = uiTagNum;
			rc = RC_SET( FERR_PURGED_FLD_FOUND);
			goto Exit;
		}
		else if ((uiFldState == ITT_FLD_STATE_CHECKING || 
					 uiFldState == ITT_FLD_STATE_UNUSED) && 
					!(uiAction & KREF_DEL_KEYS) && !(uiAction & KREF_INDEXING_ONLY))
		{
			// Because a occurance of this field was found update the field's
			// state to be 'active'
			
			if (RC_BAD( rc = flmChangeItemState( pDb, uiTagNum,
						  ITT_FLD_STATE_ACTIVE)))
			{
				goto Exit;
			}

			// If this is an encrypted field, see if we need to update the
			// state of the EncDef record too.

			if (bFldEncrypted)
			{
				if ((uiEncState == ITT_ENC_STATE_CHECKING ||
					  uiEncState == ITT_ENC_STATE_UNUSED) &&
					 !(uiAction & KREF_DEL_KEYS) && !(uiAction & KREF_INDEXING_ONLY))
				{
					if (RC_BAD( rc = flmChangeItemState( pDb, uiEncId,
								  ITT_ENC_STATE_ACTIVE)))
					{
						goto Exit;
					}
				}
			}
		}

		if (uiItemType != uiFieldType && uiTagNum < FLM_DICT_FIELD_NUMS)
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			pDb->Diag.uiInfoFlags |= (FLM_DIAG_FIELD_NUM | FLM_DIAG_FIELD_TYPE);
			pDb->Diag.uiFieldNum = uiTagNum;
			pDb->Diag.uiFieldType = uiFieldType;
			goto Exit;
		}

		if (uiFieldType == FLM_BLOB_TYPE)
		{
			if (!(uiAction & KREF_INDEXING_ONLY))
			{
				if (RC_BAD( rc = flmBlobPlaceInTransactionList( pDb, 
					((uiAction & KREF_DEL_KEYS) ? BLOB_DELETE_ACTION : BLOB_ADD_ACTION),
					pRecord, pvField)))
				{
					goto Exit;
				}
			}
		}
		else if (uiFieldType == FLM_NUMBER_TYPE)
		{
			
			// Make sure if the database version is not at least 4.62 that we
			// don't allow numbers that are too large into the database.
			
			if (uiDbVersion < FLM_FILE_FORMAT_VER_4_62)
			{
				const FLMBYTE *	pucDataPtr;
				FLMUINT				uiDataLength = pRecord->getDataLength( pvField);
				
				// All data lengths less than six will be ok.  All data lengths
				// greater than six will be bad.
				
				if (uiDataLength < 6)
				{
					// For numbers whose data length is less than six, the number
					// is guaranteed to fit in 32 bits.
				}
				else if (uiDataLength > 6)
				{
					
					// Any numbers whose data length is greater than six are more
					// than 32 bits.
					
					rc = RC_SET( FERR_64BIT_NUMS_NOT_SUPPORTED);
					goto Exit;
				}
				else	// uiDataLength == 6
				{
				
					// Check for encryption - make sure we have the decrypted data to
					// look at.
			
					if (pRecord->isEncryptedField( pvField) &&
						 !(pRecord->getEncFlags( pvField) & FLD_HAVE_DECRYPTED_DATA))
					{
						rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
						goto Exit;
					}
					pucDataPtr = pRecord->getDataPtr( pvField);
					
					// See if the number is negative.
					
					if ((*pucDataPtr & 0xF0) == 0xB0)
					{
						// The sixth byte should have an F in either the upper or
						// lower nibble.  If it is in the upper nibble, the number
						// will fit in 32 bits.
						
						if ((pucDataPtr [5] & 0xF0) == 0xF0)
						{
							// F was in upper nibble, number will fit in 32 bits.
						}
						else
						{
							// The sixth byte should have an F in the lower nibble, so
							// it is ok to compare all six bytes.  We are looking for
							// something greater than B2,14,74,83,64,8F
							
							if (f_memcmp( pucDataPtr, gv_ucMinInt32, 6) > 0)
							{
								rc = RC_SET( FERR_64BIT_NUMS_NOT_SUPPORTED);
								goto Exit;
							}
						}
					}
					else
					{
						// If the sixth byte is not an F in the high nibble, we
						// have more than a 32 bit value.  If it has an F in the
						// high nibble, check the first five bytes to make sure
						// they are not greater than 42,94,96,72,95
						
						if ((pucDataPtr [5] & 0xF0) != 0xF0 ||
							 f_memcmp( pucDataPtr, gv_ucMaxUInt32, 5) > 0)
						{
							rc = RC_SET( FERR_64BIT_NUMS_NOT_SUPPORTED);
							goto Exit;
						}
					}
				}
			}
		}

		if (pIfdChain)
		{
			if (RC_BAD( rc = flmProcessIndexedFld( pDb, pIxd, pIfdChain, pathFlds,
						  uiLeafFieldLevel, uiAction, uiContainerNum, uiDrn,
						  pbHadUniqueKeys, pRecord, pvField)))
			{
				goto Exit;
			}
		}

		if ((pvField = pRecord->next( pvField)) == NULL)
		{
			break;
		}
	}

Exit:

	// Build and add the compound keys to the KREF table

	if (RC_OK( rc))
	{
		rc = KYBuildCmpKeys( pDb, uiAction, uiContainerNum, uiDrn,
								  pbHadUniqueKeys, pRecord);
	}

	return (rc);
}

/****************************************************************************
Desc: See if a field's path matches the path in the IFD.
****************************************************************************/
FLMBOOL flmCheckIfdPath(
	IFD *				pIfd,
	FlmRecord *		pRecord,
	void **		 	ppPathFlds,
	FLMUINT			uiLeafFieldLevel,
	void *			pvLeafField,
	void **		 	ppvContextField)
{
	FLMBOOL			bMatched = FALSE;
	void *			pvContextField;
	FLMINT			iParentPos;
	FLMUINT *		puiIfdFldPathCToP;

	// Check the field path to see if field is in context.

	pvContextField = pvLeafField;
	puiIfdFldPathCToP = &pIfd->pFieldPathCToP[1];
	iParentPos = (FLMINT) uiLeafFieldLevel - 1;
	
	while (*puiIfdFldPathCToP && iParentPos >= 0)
	{
		pvContextField = ppPathFlds[iParentPos];

		// Check for FLM_ANY_FIELD (wild_tag) and skip it.

		if (*puiIfdFldPathCToP == FLM_ANY_FIELD)
		{

			// Look at next field in IFD path to see if it matches the
			// current field. If it does, continue from there.

			if (*(puiIfdFldPathCToP + 1))
			{
				if (pRecord->getFieldID( pvContextField) == 
					 *(puiIfdFldPathCToP + 1))
				{

					// Skip wild card and field that matched.

					puiIfdFldPathCToP += 2;
				}

				// Go to next field in path being evaluated no matter what.
				// If it didn't match, we continue looking at the wild card.
				// If it did match, we go to the next field in the path.

				iParentPos--;
			}
			else
			{

				// Rest of path is an automatic match - had wildcard at top
				// of IFD path.
				//
				// It's not really necessary to increment this, but it is more
				// efficient because of the comparisons that are done when we
				// exit this loop.
				
				puiIfdFldPathCToP++;
				pvContextField = ppPathFlds[0];
				break;
			}
		}
		else if (pRecord->getFieldID( pvContextField) != *puiIfdFldPathCToP)
		{

			// Field does not match current field in IFD. This jump to Exit
			// will return FALSE. bMatched is FALSE at this point.

			goto Exit;
		}
		else
		{

			// Go up a level in the record and the IFD path - to parent.

			iParentPos--;
			puiIfdFldPathCToP++;
		}
	}

	// If we got to the end of the field path in the IFD, we have a match.

	if (!(*puiIfdFldPathCToP) ||
		 (*puiIfdFldPathCToP == FLM_ANY_FIELD && !(*(puiIfdFldPathCToP + 1))))
	{
		*ppvContextField = pvContextField;
		bMatched = TRUE;
	}

Exit:

	return (bMatched);
}

/****************************************************************************
Desc: Processes a field in a record - indexing, blob, etc.
****************************************************************************/
FSTATIC RCODE flmProcessIndexedFld(
	FDB *				pDb,
	IXD *				pUseIxd,
	IFD *				pIfdChain,
	void **			ppPathFlds,
	FLMUINT			uiLeafFieldLevel,
	FLMUINT			uiAction,
	FLMUINT			uiContainerNum,
	FLMUINT			uiDrn,
	FLMBOOL *		pbHadUniqueKeys,
	FlmRecord *		pRecord,
	void *			pvField)
{
	RCODE					rc = FERR_OK;
	IFD *					pIfd;
	IXD *					pIxd;
	void *				pRootContext;
	const FLMBYTE *	pValue;
	const FLMBYTE *	pExportValue;
	FLMUINT				uiValueLen;
	FLMUINT				uiKeyLen;
	FLMBYTE				pTmpKeyBuf[MAX_KEY_SIZ];

	pTmpKeyBuf[0] = '\0';

	for (pIfd = pIfdChain; pIfd; pIfd = pIfd->pNextInChain)
	{
		if (pUseIxd)
		{
			if (pUseIxd->uiIndexNum == pIfd->uiIndexNum)
			{
				pIxd = pUseIxd;
			}
			else
			{
				continue;
			}
		}
		else
		{
			pIxd = pIfd->pIxd;

			// If index is offline or on a different container, skip it.
			// NOTE: if pIxd->uiContainerNum is zero, the index is indexing
			// ALL containers.

			if (pIxd->uiContainerNum)
			{
				if (pIxd->uiContainerNum != uiContainerNum)
				{
					continue;
				}

				if (pIxd->uiFlags & IXD_OFFLINE)
				{
					if (uiDrn > pIxd->uiLastDrnIndexed)
					{
						continue;
					}

					// Else index the key.

				}
			}
			else
			{
				// uiContainerNum == 0, indexing all containers

				if (pIxd->uiFlags & IXD_OFFLINE)
				{
					if (uiContainerNum > pIxd->uiLastContainerIndexed ||
						 (uiContainerNum == pIxd->uiLastContainerIndexed &&
						 uiDrn > pIxd->uiLastDrnIndexed))
					{
						continue;
					}

					// Else index the key.

				}
			}
		}

		// See if field path matches what is defined in the IFD.

		if (!flmCheckIfdPath( pIfd, pRecord, ppPathFlds, uiLeafFieldLevel,
									pvField, &pRootContext))
		{

			// Skip this field.

			continue;
		}

		// Field passed the path verification. Now output the KEY.

		if (pIfd->uiFlags & IFD_COMPOUND)
		{

			// Compound Key.

			if (RC_BAD( rc = KYCmpKeyAdd2Lst( pDb, pIxd, pIfd, pvField,
						  pRootContext)))
			{
				goto Exit;
			}
		}
		else if (pIfd->uiFlags & IFD_CONTEXT)
		{
			FLMBYTE	KeyBuf[4];

			// Context key (tag number).

			KeyBuf[0] = KY_CONTEXT_PREFIX;
			f_UINT16ToBigEndian( (FLMUINT16) pRecord->getFieldID( 
				pvField), &KeyBuf[1]);

			if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, pIfd,
						  uiAction, uiDrn, pbHadUniqueKeys, KeyBuf, KY_CONTEXT_LEN,
						  TRUE, FALSE, FALSE)))
			{
				goto Exit;
			}
		}
		else if ((pIfd->uiFlags & IFD_SUBSTRING) &&
					(pRecord->getDataType( pvField) == FLM_TEXT_TYPE))
		{
			FLMBOOL	bFirstSubstring = TRUE;
			FLMUINT	uiLanguage = pIxd->uiLanguage;

			// An encrypted field, in limited mode means we use the
			// encrypted data instead.

			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);

				if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, pIfd,
							  uiAction, uiDrn, pbHadUniqueKeys, pValue, uiValueLen,
							  FALSE, bFirstSubstring, TRUE)))
				{
					goto Exit;
				}
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);

				// Loop for each word in the text field adding it to the
				// table.

				while (KYSubstringParse( &pValue, &uiValueLen, pIfd->uiFlags, 
					pIfd->uiLimit, (FLMBYTE*) pTmpKeyBuf, &uiKeyLen) == TRUE)
				{
					if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, 
								  pIfd, uiAction, uiDrn, pbHadUniqueKeys,
								  (FLMBYTE *) pTmpKeyBuf, uiKeyLen, FALSE,
								  bFirstSubstring, FALSE)))
					{
						break;
					}

					if ((uiValueLen == 1 && 
						!(uiLanguage >= FLM_FIRST_DBCS_LANG && 
							uiLanguage <= FLM_LAST_DBCS_LANG)))
					{
						break;
					}

					bFirstSubstring = FALSE;
				}

				if (RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
		else if ((pIfd->uiFlags & IFD_EACHWORD) &&
					(pRecord->getDataType( pvField) == FLM_TEXT_TYPE))
		{

			// An encrypted field, in limited mode means we use the
			// encrypted data instead.

			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);

				if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, pIfd,
							  uiAction, uiDrn, pbHadUniqueKeys, pValue, uiValueLen,
							  FALSE, FALSE, TRUE)))
				{
					goto Exit;
				}
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);

				// Loop for each word in the text field adding it to the
				// table.

				while (KYEachWordParse( &pValue, &uiValueLen, pIfd->uiLimit, 
					(FLMBYTE *) pTmpKeyBuf, &uiKeyLen) == TRUE)
				{
					if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, 
								  pIfd, uiAction, uiDrn, pbHadUniqueKeys,
								  (FLMBYTE*) pTmpKeyBuf, uiKeyLen, FALSE, FALSE, FALSE)))
					{
						break;
					}
				}

				if (RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
		else
		{

			// Index field content - entire field.

			FLMBOOL	bEncryptedKey = FALSE;

			// An encrypted field, in limited mode means we use the
			// encrypted data instead.

			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pExportValue = pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);
				bEncryptedKey = TRUE;
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);
			}

			if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum, pIfd,
						  uiAction, uiDrn, pbHadUniqueKeys, pExportValue, uiValueLen,
						  FALSE, FALSE, bEncryptedKey)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Add an index key to the buffers
****************************************************************************/
RCODE KYAddToKrefTbl(
	FDB *					pDb,
	IXD *					pIxd,
	FLMUINT				uiContainerNum,
	IFD *					pIfd,
	FLMUINT				uiAction,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	const FLMBYTE *	pKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bAlreadyCollated,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bFldIsEncrypted)
{
	RCODE				rc = FERR_OK;
	KREF_ENTRY *	pKref;
	FLMBYTE *		pKrefKey;
	FLMUINT			uiKrefKeyLen;
	FLMUINT			uiSizeNeeded;
	KREF_CNTRL *	pKrefCntrl = &pDb->KrefCntrl;

	// If the table is FULL, commit the keys or expand the table

	if (pKrefCntrl->uiCount == pKrefCntrl->uiKrefTblSize)
	{
		FLMUINT	uiAllocSize;
		FLMUINT	uiOrigKrefTblSize = pKrefCntrl->uiKrefTblSize;

		if (pKrefCntrl->uiKrefTblSize > 0x8000 / sizeof(KREF_ENTRY *))
		{
			pKrefCntrl->uiKrefTblSize += 4096;
		}
		else
		{
			pKrefCntrl->uiKrefTblSize *= 2;
		}

		uiAllocSize = pKrefCntrl->uiKrefTblSize * sizeof(KREF_ENTRY *);

		if (RC_BAD( rc = f_realloc( uiAllocSize, &pKrefCntrl->pKrefTbl)))
		{
			pKrefCntrl->uiKrefTblSize = uiOrigKrefTblSize;
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	// Get the collated key.

	if (bAlreadyCollated)
	{

		// Compound keys are already collated.

		pKrefKey = (FLMBYTE*) pKey;
		uiKrefKeyLen = uiKeyLen;
	}
	else
	{
		pKrefKey = pKrefCntrl->pKrefKeyBuf;
		uiKrefKeyLen = (pIxd->uiContainerNum) 
								? MAX_KEY_SIZ 
								: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);

		if (RC_BAD( rc = KYCollateValue( pKrefKey, &uiKrefKeyLen, pKey, uiKeyLen,
					  pIfd->uiFlags, pIfd->uiLimit, NULL, NULL, 
					  (FLMUINT) ((pIxd->uiLanguage != 0xFFFF) 
					  			? pIxd->uiLanguage 
								: pDb->pFile->FileHdr.uiDefaultLanguage),
					  FALSE, bFirstSubstring, FALSE, NULL, NULL, bFldIsEncrypted)))
		{
			goto Exit;
		}
	}

	// If indexing all containers, add the container number.

	if (!pIxd->uiContainerNum)
	{
		appendContainerToKey( pIxd, uiContainerNum, pKrefKey, &uiKrefKeyLen);
	}

	// Allocate memory for the key's KREF and the key itself. We allocate
	// one extra byte so we can NULL terminate the key below. The extra
	// NULL character is to ensure that the compare in the qsort routine
	// will work.

	uiSizeNeeded = sizeof( KREF_ENTRY) + uiKrefKeyLen + 1;

	if( RC_BAD( rc = pKrefCntrl->pPool->poolAlloc( uiSizeNeeded, 
		(void **)&pKref)))
	{
		goto Exit;
	}
	
	pKrefCntrl->pKrefTbl[pKrefCntrl->uiCount++] = pKref;
	pKrefCntrl->uiTotalBytes += uiSizeNeeded;

	// Fill in all of the fields in the KREF structure.

	flmAssert( pIxd->uiIndexNum > 0 && pIxd->uiIndexNum < FLM_UNREGISTERED_TAGS);
	
	pKref->ui16IxNum = (FLMUINT16) pIxd->uiIndexNum;
	pKref->uiDrn = uiDrn;
	if (uiAction & KREF_DEL_KEYS)
	{
		pKref->uiFlags = ((uiAction & KREF_MISSING_KEYS_OK) 
										? (FLMUINT) (KREF_DELETE_FLAG | KREF_MISSING_OK) 
										: (FLMUINT) (KREF_DELETE_FLAG));
	}
	else
	{
		pKref->uiFlags = 0;
	}

	if (pIxd->uiFlags & IXD_UNIQUE)
	{
		*pbHadUniqueKeys = TRUE;
		pKref->uiFlags |= KREF_UNIQUE_KEY;
	}

	if (bFldIsEncrypted)
	{
		pKref->uiFlags |= KREF_ENCRYPTED_KEY;
	}

	pKref->ui16KeyLen = (FLMUINT16) uiKrefKeyLen;
	pKref->uiTrnsSeq = pKrefCntrl->uiTrnsSeqCntr;

	// Null terminate the key so compare in qsort will work

	pKrefKey[uiKrefKeyLen++] = '\0';

	// Copy the key to just after the KREF structure

	f_memcpy( (FLMBYTE *) (&pKref[1]), pKrefKey, uiKrefKeyLen);

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Encrypt a field in the pRecord.
****************************************************************************/
RCODE flmEncryptField(
	FDICT *			pDict,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiEncId,
	F_Pool *			pPool)
{
	RCODE				rc = FERR_OK;
	F_CCS *			pCcs;
	FLMUINT			uiEncLength;
	FLMBYTE *		pucEncBuffer;
	FLMBYTE *		pucDataBuffer = NULL;
	FLMUINT			uiCheckLength;
	void *			pvMark;
#ifdef FLM_DEBUG
	FLMBOOL			bOk;
	FLMUINT			uiLoop;
#endif

	pvMark = pPool->poolMark();

	if (!pRecord->isEncryptedField( pvField))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FLD_NOT_ENCRYPTED);
		goto Exit;
	}

	pCcs = (F_CCS *) pDict->pIttTbl[uiEncId].pvItem;

	flmAssert( pCcs);

	uiEncLength = pRecord->getEncryptedDataLength( pvField);
	
	if( RC_BAD( rc = pPool->poolAlloc( uiEncLength, (void **)&pucDataBuffer)))
	{
		goto Exit;
	}
	
	pucEncBuffer = (FLMBYTE *) pRecord->getEncryptionDataPtr( pvField);
	uiCheckLength = uiEncLength;

#ifdef FLM_DEBUG

	// Preset the buffer to a known value so we can check it after the
	// encryption. It should NOT be the same!

	f_memset( pucEncBuffer, 'B', uiEncLength);
#endif

	// We copy the data into a buffer that is as large as the encrypted
	// data because the encryption algorithm is expecting to get a buffer
	// that does not need to be padded to the nearest 16 byte boundary.

	f_memcpy( pucDataBuffer, pRecord->getDataPtr( pvField),
				pRecord->getDataLength( pvField));

	if (RC_BAD( rc = pCcs->encryptToStore( pucDataBuffer, uiEncLength,
				  pucEncBuffer, &uiCheckLength)))
	{
		goto Exit;
	}

	if (uiCheckLength != uiEncLength)
	{
		rc = RC_SET( FERR_DATA_SIZE_MISMATCH);
		goto Exit;
	}

#ifdef FLM_DEBUG
	bOk = FALSE;
	for (uiLoop = 0; uiLoop < uiEncLength; uiLoop++)
	{
		if (pucEncBuffer[uiLoop] != 'B')
		{
			bOk = TRUE;
			break;
		}
	}

	if (!bOk)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}
#endif

	pRecord->setEncFlags( pvField,
								FLD_HAVE_DECRYPTED_DATA | FLD_HAVE_ENCRYPTED_DATA);

Exit: 

	pPool->poolReset( pvMark);
	return (rc);
}

/****************************************************************************
Desc:	Decrypt an encrypted field in the pRecord.
****************************************************************************/
RCODE flmDecryptField(
	FDICT *			pDict,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiEncId,
	F_Pool *			pPool)
{
	RCODE				rc = FERR_OK;
	F_CCS *			pCcs;
	FLMUINT			uiEncLength;
	FLMBYTE *		pucEncBuffer = NULL;
	FLMBYTE *		pucDataBuffer = NULL;
	FLMUINT			uiCheckLength;
	void *			pvMark = NULL;

	pvMark = pPool->poolMark();

	if (!pRecord->isEncryptedField( pvField))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FLD_NOT_ENCRYPTED);
		goto Exit;
	}

	pCcs = (F_CCS*) pDict->pIttTbl[uiEncId].pvItem;

	flmAssert( pCcs);

	uiEncLength = pRecord->getEncryptedDataLength( pvField);
	
	if( RC_BAD( rc = pPool->poolAlloc( uiEncLength, (void **)&pucDataBuffer)))
	{
		goto Exit;
	}
	
	pucEncBuffer = (FLMBYTE *) pRecord->getEncryptionDataPtr( pvField);
	uiCheckLength = uiEncLength;

	if (RC_BAD( rc = pCcs->decryptFromStore( pucEncBuffer, uiEncLength,
				  pucDataBuffer, &uiCheckLength)))
	{
		goto Exit;
	}

	if (uiCheckLength != uiEncLength)
	{
		rc = RC_SET( FERR_DATA_SIZE_MISMATCH);
		goto Exit;
	}

	f_memcpy( (void *)pRecord->getDataPtr( pvField), pucDataBuffer,
				pRecord->getDataLength( pvField));

	pRecord->setEncFlags( pvField,
								FLD_HAVE_DECRYPTED_DATA | FLD_HAVE_ENCRYPTED_DATA);

Exit:

	pPool->poolReset( pvMark);
	return (rc);
}

/****************************************************************************
Desc:	Substring-ize the string in a node.
****************************************************************************/
FLMBOOL KYSubstringParse(
	const FLMBYTE **	ppText,			// [in][out] points to text
	FLMUINT *			puiTextLen,		// [in][out] length of text
	FLMUINT				uiIfdFlags,		// [in] flags
	FLMUINT				uiLimitParm,	// [in] Max characters
	FLMBYTE *			pKeyBuf,			// [out] key buffer to fill
	FLMUINT *			puiKeyLen)		// [out] returns length
{
	const FLMBYTE *	pText = *ppText;
	FLMUINT				uiLen = *puiTextLen;
	FLMUINT				uiWordLen = 0;
	FLMUINT				uiLimit = uiLimitParm ? uiLimitParm : IFD_DEFAULT_SUBSTRING_LIMIT;
	FLMUINT				uiFlags = 0;
	FLMUINT				uiLeadingSpace = FLM_COMP_NO_WHITESPACE;
	FLMBOOL				bIgnoreSpaceDefault = (uiIfdFlags & IFD_NO_SPACE) ? TRUE : FALSE;
	FLMBOOL				bIgnoreSpace = TRUE;
	FLMBOOL				bIgnoreDash = (uiIfdFlags & IFD_NO_DASH) ? TRUE : FALSE;
	FLMBOOL				bMinSpaces = (uiIfdFlags & (IFD_MIN_SPACES | IFD_NO_SPACE)) ? TRUE : FALSE;
	FLMBOOL				bNoUnderscore = (uiIfdFlags & IFD_NO_UNDERSCORE) ? TRUE : FALSE;
	FLMBOOL				bFirstCharacter = TRUE;

	// Set uiFlags

	if (bIgnoreSpaceDefault)
	{
		uiFlags |= FLM_COMP_NO_WHITESPACE;
	}

	if (bIgnoreDash)
	{
		uiFlags |= FLM_COMP_NO_DASHES;
	}

	if (bNoUnderscore)
	{
		uiFlags |= FLM_COMP_NO_UNDERSCORES;
	}

	if (uiIfdFlags & IFD_MIN_SPACES)
	{
		uiFlags |= FLM_COMP_COMPRESS_WHITESPACE;
	}

	// The limit must return one more than requested in order for the text
	// to collation routine to set the truncated flag.

	uiLimit++;

	while (uiLen && uiLimit--)
	{
		FLMBYTE		ch = *pText;
		FLMUINT16	ui16WPValue;
		FLMUNICODE	ui16UniValue;
		FLMUINT		uiCharLen;

		if ((ch & ASCII_CHAR_MASK) == ASCII_CHAR_CODE)
		{
			if (ch == ASCII_UNDERSCORE && bNoUnderscore)
			{
				ch = ASCII_SPACE;
			}

			if (ch == ASCII_SPACE && bMinSpaces)
			{
				if (!bIgnoreSpace)
				{
					pKeyBuf[uiWordLen++] = ASCII_SPACE;
				}

				bIgnoreSpace = TRUE;
				pText++;
				uiLen--;
				continue;
			}

			ui16WPValue = (FLMUINT16) ch;
			uiCharLen = 1;
		}
		else
		{
			if ((uiCharLen = flmTextGetValue( pText, uiLen, NULL,
				uiFlags | uiLeadingSpace, &ui16WPValue, &ui16UniValue)) == 0)
			{
				break;
			}

			flmAssert( uiCharLen <= uiLen);
		}

		uiLeadingSpace = 0;
		bIgnoreSpace = bIgnoreSpaceDefault;
		uiLen -= uiCharLen;
		while (uiCharLen--)
		{
			pKeyBuf[uiWordLen++] = *pText++;
		}

		// If on the first word position to start on next character for the
		// next call.

		if (bFirstCharacter)
		{
			bFirstCharacter = FALSE;

			// First character - set return value.

			*ppText = pText;
			*puiTextLen = uiLen;
		}
	}

	pKeyBuf[uiWordLen] = '\0';

	// Case of all spaces - the FALSE will trigger indexing is done.

	*puiKeyLen = (FLMUINT) uiWordLen;
	return ((uiWordLen) ? TRUE : FALSE);
}

/****************************************************************************
Desc:	Keyword-ize the information in a node - node is assumed to be a
		TEXT node.
****************************************************************************/
FLMBOOL KYEachWordParse(
	const FLMBYTE **	pText,
	FLMUINT *			puiTextLen,
	FLMUINT				uiLimitParm,	// [in] Max characters
	FLMBYTE *			pKeyBuf,			// [out] Buffer of at least MAX_KEY_SIZ
	FLMUINT *			puiKeyLen)
{
	const FLMBYTE *	pKey = NULL;
	const FLMBYTE *	pTmpKey;
	FLMUINT				uiLimit = uiLimitParm ? uiLimitParm : IFD_DEFAULT_SUBSTRING_LIMIT;
	FLMUINT				uiLen;
	FLMUINT				uiBytesProcessed = 0;
	FLMBOOL				bSkippingDelim = TRUE;
	FLMBOOL				bHaveWord = FALSE;
	FLMUINT				uiWordLen = 0;
	FLMUINT16			ui16WPValue;
	FLMUNICODE			ui16UniValue;
	FLMUINT				uiCharLen;
	FLMUINT				uiType;

	uiLen = *puiTextLen;
	pTmpKey = *pText;
	
	while ((uiBytesProcessed < uiLen) && (!bHaveWord) && uiLimit)
	{
		uiCharLen = flmTextGetCharType( pTmpKey, uiLen, &ui16WPValue,
												 &ui16UniValue, &uiType);

		// Determine how to handle what we got.

		if (bSkippingDelim)
		{

			// If we were skipping delimiters, and we run into a
			// non-delimiter character, set the bSkippingDelim flag to FALSE
			// to indicate the beginning of a word.

			if (uiType & SDWD_CHR)
			{
				pKey = pTmpKey;
				uiWordLen = uiCharLen;
				bSkippingDelim = FALSE;
				uiLimit--;
			}
		}
		else
		{

			// If we were NOT skipping delimiters, and we run into a
			// delimiter output the word.

			if (uiType & (DELI_CHR | WDJN_CHR))
			{
				bHaveWord = TRUE;
			}
			else
			{
				uiWordLen += uiCharLen;
				uiLimit--;
			}
		}

		// Increment str to skip past what we are pointing at.

		pTmpKey += uiCharLen;
		uiBytesProcessed += uiCharLen;
	}

	*pText = pTmpKey;
	*puiTextLen -= uiBytesProcessed;

	// Return the word, if any.

	if (uiWordLen)
	{
		*puiKeyLen = uiWordLen;
		f_memcpy( pKeyBuf, pKey, uiWordLen);
	}

	return ((uiWordLen) ? TRUE : FALSE);
}

/****************************************************************************
Desc:		Setup routine for the KREF_CNTRL structure for record updates.
			Will check to see if all structures, buffers and memory pools 
			need to be allocated: Kref key buffer, CDL table, KrefTbl and pool.
			The goal is to have only one allocation for most small transactions.
			As of Nov 96, each DB will have its own KREF_CNTRL struture so the
			session temp pool does not have to be used.  This means that the
			CDL and cmpKeys arrays do not have to be allocated for each
			record operation (like we did in the session pool).
****************************************************************************/
RCODE KrefCntrlCheck(
	FDB *				pDb)
{
	RCODE				rc = FERR_OK;			// Set for cleaner code.
	KREF_CNTRL *	pKrefCntrl;

	pKrefCntrl = &pDb->KrefCntrl;

	/* Check if we need to flush between the records and not during
		the processing of a record.  This simplifies how we reuse the memory.
	*/

	if( pKrefCntrl->bKrefSetup)
	{
		if( (pKrefCntrl->uiCount >= KREF_TBL_THRESHOLD)
		 || (pKrefCntrl->uiTotalBytes >= KREF_TOTAL_BYTES_THRESHOLD))

		{
			if( RC_BAD( rc = KYKeysCommit( pDb, FALSE)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		FLMUINT		uiKrefTblSize = KREF_TBL_SIZE * sizeof(KREF_ENTRY *);
		FLMUINT		uiCDLSize = pDb->pDict->uiIfdCnt * sizeof( CDL *);
		FLMUINT		uiIxdSize = pDb->pDict->uiIxdCnt;
		FLMUINT		uiKeyBufSize = MAX_KEY_SIZ + 8;
	
		f_memset( pKrefCntrl, 0, sizeof( KREF_CNTRL));
		pKrefCntrl->bKrefSetup = TRUE;
		if (pDb->uiTransType == FLM_UPDATE_TRANS)
		{
			pKrefCntrl->pPool = &pDb->pFile->krefPool;
			pKrefCntrl->bReusePool = TRUE;
		}
		else
		{
			pKrefCntrl->pPool = &pDb->tmpKrefPool;
			pKrefCntrl->bReusePool = FALSE;
		}

		if (pKrefCntrl->bReusePool)
		{
			pKrefCntrl->pPool->poolReset();
		}
		else
		{
			pKrefCntrl->pPool->poolInit( KREF_POOL_BLOCK_SIZE);
		}

		if( RC_BAD( rc = f_alloc( uiKrefTblSize,
			&pKrefCntrl->pKrefTbl))
		 || (uiCDLSize && RC_BAD( rc = f_calloc( uiCDLSize,
				&pKrefCntrl->ppCdlTbl)))
		 || (uiIxdSize && RC_BAD( rc = f_calloc( uiIxdSize,
				&pKrefCntrl->pIxHasCmpKeys)))
		 || RC_BAD( rc = f_calloc( uiKeyBufSize,
				&pKrefCntrl->pKrefKeyBuf)))
		{
			KrefCntrlFree( pDb);
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pKrefCntrl->uiKrefTblSize = KREF_TBL_SIZE;
	}

	pKrefCntrl->pReset = pKrefCntrl->pPool->poolMark(); 

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Resets or frees the memory associated with the KREF.
****************************************************************************/
void KrefCntrlFree(
	FDB *	pDb)
{
	KREF_CNTRL *	pKrefCntrl = &pDb->KrefCntrl;

	if( pKrefCntrl->bKrefSetup)
	{
		if (pKrefCntrl->bReusePool)
		{
			pKrefCntrl->pPool->poolReset();
		}
		else
		{
			pKrefCntrl->pPool->poolFree();
		}

		if( pKrefCntrl->pKrefTbl)
		{
			f_free( &pKrefCntrl->pKrefTbl);
		}

		if( pKrefCntrl->ppCdlTbl)
		{
			f_free( &pKrefCntrl->ppCdlTbl);
		}

		if( pKrefCntrl->pIxHasCmpKeys)
		{
			f_free( &pKrefCntrl->pIxHasCmpKeys);
		}

		if( pKrefCntrl->pKrefKeyBuf)
		{
			f_free( &pKrefCntrl->pKrefKeyBuf);
		}

		// Just set everyone back to zero.

		f_memset( pKrefCntrl, 0, sizeof(KREF_CNTRL));
	}
}

/****************************************************************************
Desc:	Checks if the current database has any UNIQUE indexes that need to
		checked. Also does duplicate processing for the record.
****************************************************************************/
RCODE KYProcessDupKeys(
	FDB *				pDb,
	FLMBOOL			bHadUniqueKeys)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL *	pKrefCntrl = &pDb->KrefCntrl;
	FLMUINT			uiCurRecKrefCnt;

	pKrefCntrl->uiTrnsSeqCntr++;

	// Sort and remove duplicates from the list of this record.

	uiCurRecKrefCnt = pKrefCntrl->uiCount - pKrefCntrl->uiLastRecEnd;

	if (uiCurRecKrefCnt > 1)
	{
		FLMUINT	uiSortFlags = KY_DUP_CHK_SRT;

		if (RC_BAD( rc = _KrefQuickSort( &uiSortFlags,
					  &pKrefCntrl->pKrefTbl[pKrefCntrl->uiLastRecEnd], 0,
					  uiCurRecKrefCnt - 1)))
		{
			goto Exit;
		}

		// Found any duplicates?

		if (uiSortFlags & KY_DUPS_FOUND)
		{
			if (RC_BAD( rc = _KrefKillDups( &uiSortFlags,
						  &pKrefCntrl->pKrefTbl[pKrefCntrl->uiLastRecEnd],
						  &uiCurRecKrefCnt)))
			{
				goto Exit;
			}

			pKrefCntrl->uiCount = pKrefCntrl->uiLastRecEnd + uiCurRecKrefCnt;
		}
	}

	if (bHadUniqueKeys)
	{

		// Now check the keys for uniquness in table, and database.

		if (RC_BAD( rc = KYAddUniqueKeys( pDb)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Remove anything that was put into the KREF table by the current
		record update operation.
****************************************************************************/
void KYAbortCurrentRecord(
	FDB *			pDb)
{
	flmAssert( pDb->KrefCntrl.bKrefSetup);

	// Reset the CDL and pIxHasCmpKeys tables

	if (pDb->pDict->uiIfdCnt)
	{
		f_memset( pDb->KrefCntrl.ppCdlTbl, 0, 
					 pDb->pDict->uiIfdCnt * sizeof(CDL *));
	}

	if (pDb->pDict->uiIxdCnt)
	{
		f_memset( pDb->KrefCntrl.pIxHasCmpKeys, 0, pDb->pDict->uiIxdCnt);
	}

	pDb->KrefCntrl.uiCount = pDb->KrefCntrl.uiLastRecEnd;
	pDb->KrefCntrl.pPool->poolReset( pDb->KrefCntrl.pReset);
}

/****************************************************************************
Desc:	Commit (write out) all reference lists from the current pDb.  Will
		take care of optimially freeing or resetting memory.
****************************************************************************/
RCODE KYKeysCommit(
	FDB *				pDb,
	FLMBOOL			bCommittingTrans)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL *	pKrefCntrl = &pDb->KrefCntrl;

	// If KrefCntrl has not been initialized, there is no work to do.

	if (pKrefCntrl->bKrefSetup)
	{
		LFILE *				pLFile = NULL;
		FLMUINT				uiTotal = pKrefCntrl->uiLastRecEnd;
		KREF_ENTRY *		pKref;
		KREF_ENTRY **	 	pKrefTbl = pKrefCntrl->pKrefTbl;
		FLMUINT				uiKrefNum;
		FLMUINT				uiLastIxNum;

		// We should not have reached this point if bAbortTrans is TRUE

		flmAssert( RC_OK( pDb->AbortRc));

		// uiTotal and uiLastRecEnd must be the same at this point. If not,
		// we have a bug.

		flmAssert( uiTotal == pKrefCntrl->uiLastRecEnd);

		// Sort the KREF table, if it contains more than one record and
		// key. This will sort all keys from the same index the same.

		if ((uiTotal > 1) && (pKrefCntrl->uiTrnsSeqCntr > 1))
		{
			FLMUINT	uiQsortFlags = KY_FINAL_SRT;

			if (RC_BAD( rc = _KrefQuickSort( &uiQsortFlags, 
				pKrefTbl, 0, uiTotal - 1)))
			{
				goto Exit;
			}
		}

		uiLastIxNum = 0;

		// Loop through the KREF table outputting all keys

		for (uiKrefNum = 0; uiKrefNum < uiTotal; uiKrefNum++)
		{
			pKref = pKrefTbl[uiKrefNum];

			// See if the LFILE changed

			flmAssert( pKref->ui16IxNum > 0 && 
						  pKref->ui16IxNum < FLM_UNREGISTERED_TAGS);

			if (pKref->ui16IxNum != uiLastIxNum)
			{
				uiLastIxNum = pKref->ui16IxNum;
				if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
							  pDb->pFile->bInLimitedMode, uiLastIxNum, &pLFile, NULL,
							  TRUE)))
				{
					goto Exit;
				}
			}

			// Flush the key to the index

			if (RC_BAD( rc = FSRefUpdate( pDb, pLFile, pKref)))
			{
				goto Exit;
			}
		}

		if (bCommittingTrans)
		{
			KrefCntrlFree( pDb);
		}
		else
		{

			// Empty the table out so we can add more keys in this trans.

			pKrefCntrl->pPool->poolReset();
			pKrefCntrl->uiCount = 0;
			pKrefCntrl->uiTotalBytes = 0;
			pKrefCntrl->uiLastRecEnd = 0;
			pKrefCntrl->uiTrnsSeqCntr = 0;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Adds all unique key values. Backs out on any unique error so that
		the transaction may continue.
Note:	All duplicates have been removed as well as matching keys.
****************************************************************************/
FSTATIC RCODE KYAddUniqueKeys(
	FDB *				pDb)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL *	pKrefCntrl = &pDb->KrefCntrl;
	KREF_ENTRY **	pKrefTbl = pKrefCntrl->pKrefTbl;
	KREF_ENTRY *	pKref;
	FLMUINT			uiCurKrefNum;
	FLMUINT			uiPrevKrefNum;
	FLMUINT			uiTargetCount;
	FLMUINT			uiLastIxNum;
	LFILE *			pLFile;
	FLMBOOL			bUniqueErrorHit = FALSE;

	// Unique indexes can't be built in the background

	flmAssert( !(pDb->uiFlags & FDB_BACKGROUND_INDEXING));

	// Start at the first key for this current record checking for all
	// keys that belong to a unique index. We must keep all keys around
	// until the last key is added/delete so that we can back out all of
	// the changes on a unique error.

	for (uiCurKrefNum = pKrefCntrl->uiLastRecEnd, uiLastIxNum = 0,
			  uiTargetCount = pKrefCntrl->uiCount; uiCurKrefNum < uiTargetCount;)
	{
		pKref = pKrefTbl[uiCurKrefNum];

		if (pKref->uiFlags & KREF_UNIQUE_KEY)
		{
			flmAssert( pKref->ui16IxNum > 0 && 
						  pKref->ui16IxNum < FLM_UNREGISTERED_TAGS);

			if (pKref->ui16IxNum != uiLastIxNum)
			{
				uiLastIxNum = pKref->ui16IxNum;
				if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
							  pDb->pFile->bInLimitedMode, uiLastIxNum, &pLFile, NULL)))
				{

					// Return the index offline error - should not happen

					flmAssert( rc != FERR_INDEX_OFFLINE);
					goto Exit;
				}
			}

			// Flush the key to the index.

			if (RC_BAD( rc = FSRefUpdate( pDb, pLFile, pKref)))
			{
				pDb->Diag.uiInfoFlags |= FLM_DIAG_INDEX_NUM;
				pDb->Diag.uiIndexNum = pKref->ui16IxNum;

				// Check only for FERR_NOT_UNIQUE

				if (rc != FERR_NOT_UNIQUE)
				{
					goto Exit;
				}

				bUniqueErrorHit = TRUE;

				// Cycle through again backing out all keys.

				uiTargetCount = uiCurKrefNum;
				uiCurKrefNum = pKrefCntrl->uiLastRecEnd;

				// Make sure uiCurKrefNum is NOT incremented at the top of
				// loop.

				continue;
			}

			// Toggle the delete flag so on unique error we can back out.
			// This sets the ADD to DELETE and the DELETE to ADD (0)

			pKref->uiFlags ^= KREF_DELETE_FLAG;
		}

		uiCurKrefNum++;
	}

	if (bUniqueErrorHit)
	{
		rc = RC_SET( FERR_NOT_UNIQUE);
		pKrefCntrl->uiCount = pKrefCntrl->uiLastRecEnd;
	}
	else
	{

		// Move every key down removing the processed keys.

		for (uiCurKrefNum = uiPrevKrefNum = pKrefCntrl->uiLastRecEnd,
				  uiTargetCount = pKrefCntrl->uiCount; 
				  uiCurKrefNum < uiTargetCount; uiCurKrefNum++)
		{
			pKref = pKrefTbl[uiCurKrefNum];

			if (!(pKref->uiFlags & KREF_UNIQUE_KEY))
			{
				pKrefTbl[uiPrevKrefNum++] = pKrefTbl[uiCurKrefNum];
			}
		}

		pKrefCntrl->uiCount = uiPrevKrefNum;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Compare function used to compare two keys. The compare is different
		depending on the sort pass this is on.
****************************************************************************/
FSTATIC FLMINT _KrefCompare(
	FLMUINT *			puiQsortFlags,
	KREF_ENTRY *		pKrefA,
	KREF_ENTRY *		pKrefB)
{
	FLMUINT	uiMinLen;
	FLMINT	iCompare;

	// Compare (SORT1) #1, (SORT2) #2 - Index Number.

	if ((iCompare = ((FLMINT) pKrefA->ui16IxNum) - 
			((FLMINT) pKrefB->ui16IxNum)) != 0)
	{
		return (iCompare);
	}

	// Compare (SORT1) #2, (SORT2) #3: KEY - including NULL character at
	// end.

	// Comparing the NULL character advoids checking the key length.

	// VISIT: There could be a BUG where key length should be checked, but
	// it has to do with not storing all compound key pieces in the key.
	
	uiMinLen = f_min( pKrefA->ui16KeyLen, pKrefB->ui16KeyLen) + 1;
	if ((iCompare = f_memcmp( &pKrefA[1], &pKrefB[1], uiMinLen)) == 0)
	{
		if (*puiQsortFlags & KY_FINAL_SRT)
		{

			// Compare (SORT2) The DRN so we load by low DRN to high DRN.

			if (pKrefA->uiDrn < pKrefB->uiDrn)
			{
				return (-1);
			}
			else if (pKrefA->uiDrn > pKrefB->uiDrn)
			{
				return (1);
			}

			// Compare (SORT2) Sequence number, so operations occur in
			// correct order. - this will ALWAYS set iCompare to -1 or 1. It
			// is only possible to have different operations here like ADD -
			// DELETE - ADD - DELETE when sorted by uiTrnsSeq. This is why we
			// will set KY_DUPS_FOUND to get rid of duplicates.

			iCompare = ((FLMINT) pKrefA->uiTrnsSeq) - ((FLMINT) pKrefB->uiTrnsSeq);
		}
		else
		{

			// Compare (SORT1) Operation Flag, Delete or Add.

			*puiQsortFlags |= KY_DUPS_FOUND;

			// Sort so the delete elements are first.

			if ((iCompare = ((FLMINT) (pKrefB->uiFlags & KREF_DELETE_FLAG)) -
					 ((FLMINT) (pKrefA->uiFlags & KREF_DELETE_FLAG))) == 0)
			{

				// Exact duplicate - will remove later

				pKrefA->uiFlags |= KREF_EQUAL_FLAG;
				pKrefB->uiFlags |= KREF_EQUAL_FLAG;
			}
			else
			{

				// Data is same but different operation, (delete then an add).

				pKrefA->uiFlags |= KREF_IGNORE_FLAG;
				pKrefB->uiFlags |= KREF_IGNORE_FLAG;
			}
		}
	}

	return (iCompare);
}

/****************************************************************************
Desc:	Quick sort an array of KREF_ENTRY * values.
****************************************************************************/
FSTATIC RCODE _KrefQuickSort(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY **	pEntryTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	KREF_ENTRY *	pCurEntry;
	KREF_ENTRY *	pTempKref;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurEntry = pEntryTbl[uiMIDPos];
	for (;;)
	{
		while( (uiLBPos == uiMIDPos) || ((iCompare = _KrefCompare( puiQsortFlags,
			pEntryTbl[uiLBPos], pCurEntry)) < 0))
		{
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}

			uiLBPos++;
		}

		while ((uiUBPos == uiMIDPos) || (((iCompare = _KrefCompare( 
			puiQsortFlags, pCurEntry, pEntryTbl[uiUBPos])) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}

			uiUBPos--;
		}

		if (uiLBPos < uiUBPos)
		{

			// Interchange [uiLBPos] with [uiUBPos].

			KY_SWAP( pEntryTbl, uiLBPos, uiUBPos);
			uiLBPos++;
			uiUBPos--;
		}
		else
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if (uiLBPos < uiMIDPos)
	{

		// Interchange [uiLBPos] with [uiMIDPos]

		KY_SWAP( pEntryTbl, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if (uiMIDPos < uiUBPos)
	{

		// Interchange [uUBPos] with [uiMIDPos]

		KY_SWAP( pEntryTbl, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos) 
									? uiMIDPos - uiLowerBounds
									: 0;
									
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds) 
									? uiUpperBounds - uiMIDPos
									: 0;

	if (uiLeftItems < uiRightItems)
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems)
		{
			(void) _KrefQuickSort( puiQsortFlags, pEntryTbl, uiLowerBounds,
										 uiMIDPos - 1);
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if (uiLeftItems)
	{

		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems)
		{
			(void) _KrefQuickSort( puiQsortFlags, pEntryTbl, uiMIDPos + 1,
										 uiUpperBounds);
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:	Kill all duplicate references out of the kref list
Note:	This will ONLY work if EVERY kref has been compared to its neighbor.
		We may have to compare every neighbor again if the new quick 
		sort doesn't work.
****************************************************************************/
FSTATIC RCODE _KrefKillDups(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY **	pKrefTbl,
	FLMUINT*			puiKrefTotalRV)
{
	FLMUINT			uiTotal = (*puiKrefTotalRV);
	FLMUINT			uiCurKrefNum;
	KREF_ENTRY *	pCurKref;
	FLMUINT			uiLastUniqueKrefNum = 0;

	for (uiCurKrefNum = 1; uiCurKrefNum < uiTotal; uiCurKrefNum++)
	{
		pCurKref = pKrefTbl[uiCurKrefNum];

		// If the current KREF equals the last unique one, we can remove it
		// from the list by skipping the current entry. To check if they are
		// equal, first look at the KREF_EQUAL_FLAGs on both of them. If
		// both KREFs have this flag set, we still have to call the compare
		// routine. The flags could have been set for two pairs of different
		// keys - such as A, A, B, B. In this sequence of keys, all four
		// KREFs would have the flag set, but the 2nd "A" is not equal to
		// the 1st "B" - thus the need for the call to krefCompare to
		// confirm that the keys are really equal.

		if ((pKrefTbl[uiLastUniqueKrefNum]->uiFlags & KREF_EQUAL_FLAG) &&
			 (pCurKref->uiFlags & KREF_EQUAL_FLAG) && 
				(_KrefCompare( puiQsortFlags, 
					pKrefTbl[uiLastUniqueKrefNum], pCurKref) == 0))
		{

			// If the current KREF had it's ignore flag set, propagate that
			// to the last unique KREF also and remove the current key. This
			// will remove all but the first duplicate key. This is possible
			// because quick sort may not compare every item.

			if (pCurKref->uiFlags & KREF_IGNORE_FLAG)
			{
				pKrefTbl[uiLastUniqueKrefNum]->uiFlags |= KREF_IGNORE_FLAG;
			}
		}
		else
		{

			// Increment to the next slot if we like this kref.

			if (!(pKrefTbl[uiLastUniqueKrefNum]->uiFlags & KREF_IGNORE_FLAG))
			{
				uiLastUniqueKrefNum++;
			}

			// Move the item to the current location.

			pKrefTbl[uiLastUniqueKrefNum] = pCurKref;
		}
	}

	if (!(pKrefTbl[uiLastUniqueKrefNum]->uiFlags & KREF_IGNORE_FLAG))
	{
		uiLastUniqueKrefNum++;
	}

	*puiKrefTotalRV = uiLastUniqueKrefNum;
	return (FERR_OK);
}
