//-------------------------------------------------------------------------
// Desc:	Get index keys from a record.
// Tabs:	3
//
// Copyright (c) 1992-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

typedef struct CMP_KEY_ELM
{
	const FLMBYTE *	pValue;
	FLMUINT				uiValueLen;
	FLMUINT				uiType;
	FLMUINT				uiTagNum;
	CMP_KEY_ELM *		pParent;
	FLMBOOL				bFirstSubstring;
	FLMBOOL				bSubstringComponent;
} CMP_KEY_ELM;

FSTATIC RCODE flmKeyAdd(
	FDB *					pDb,
	IXD *					pIxd,
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	F_Pool *				pPool,
	FLMBOOL				bRemoveDups,
	REC_KEY **	 		ppKeyList);

FSTATIC RCODE flmGetFieldKeys(
	FDB *					pDb,
	IXD *					pIxd,
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	void **		 		ppPathFlds,
	FLMUINT				uiLeafFieldLevel,
	void *				pvField,
	FLMBOOL				bRemoveDups,
	F_Pool *				pPool,
	REC_KEY **	 		ppKeyList,
	FLMBOOL *			bHasCmpKeys);

FSTATIC RCODE flmBuildCompoundKey(
	FDB *					pDb,
	IXD *					pIxd,
	CMP_KEY_ELM *		pCmpKeyElm,
	FLMBOOL				bRemoveDups,
	F_Pool *				pPool,
	FLMUINT				uiContainerNum,
	REC_KEY **		 	ppKeyList);

FSTATIC RCODE flmGetCmpKeyElement(
	FDB *					pDb,
	IXD *					pIxd,
	IFD *					pIfd,
	FLMUINT				uiCdlEntry,
	FLMUINT				uiCompoundPos,
	CMP_KEY_ELM *		pParent,
	FLMBOOL				bRemoveDups,
	F_Pool *				pPool,
	REC_KEY **		 	ppKeyList,
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	FLD_CONTEXT *		pFldContext);

FSTATIC RCODE flmGetCompoundKeys(
	FDB *					pDb,
	IXD *					pIxd,
	FLMBOOL				bRemoveDups,
	F_Pool *				pPool,
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	REC_KEY **	 		ppKeyList);

/****************************************************************************
Desc: This routine adds a key to a key list.
****************************************************************************/
FSTATIC RCODE flmKeyAdd(
	FDB *				pDb,
	IXD *				pIxd,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	F_Pool *			pPool,
	FLMBOOL			bRemoveDups,
	REC_KEY **		ppKeyList)
{
	RCODE				rc = FERR_OK;
	REC_KEY *		pTempRecKey;
	FLMBYTE			Key1Buf[MAX_KEY_SIZ];
	FLMUINT			uiKey1Len;
	FLMBYTE			Key2Buf[MAX_KEY_SIZ];
	FLMUINT			uiKey2Len;

	// First see if the key is already in the list.

	pTempRecKey = *ppKeyList;
	if (pTempRecKey && bRemoveDups)
	{
		if (RC_BAD( rc = KYTreeToKey( pDb, pIxd, pRecord, uiContainerNum, 
			Key1Buf, &uiKey1Len, 0)))
		{
			goto Exit;
		}

		while (pTempRecKey != NULL)
		{

			// Build the collated keys for each key tree in *ppKeyList

			if (RC_BAD( rc = KYTreeToKey( pDb, pIxd, pTempRecKey->pKey,
						  uiContainerNum, Key2Buf, &uiKey2Len, 0)))
			{
				goto Exit;
			}

			// If the key was found, return success - don't add to list.
			// Also, free up the memory pool back to where the key started.

			if (KYKeyCompare( Key1Buf, uiKey1Len, Key2Buf, uiKey2Len) == BT_EQ_KEY)
			{

				// Should return FERR_OK.

				goto Exit;
			}

			pTempRecKey = pTempRecKey->pNextKey;
		}
	}
	
	if( RC_BAD( rc = pPool->poolAlloc( sizeof( REC_KEY), (void **)&pTempRecKey)))
	{
		goto Exit;
	}

	pTempRecKey->pKey = pRecord;
	pRecord->AddRef();
	pTempRecKey->pNextKey = *ppKeyList;
	*ppKeyList = pTempRecKey;
	
Exit:

	return (rc);
}

/****************************************************************************
Desc: This routine gets all of the keys for a field and either saves them
		as an element for a compound key, or saves them into the key list.
****************************************************************************/
FSTATIC RCODE flmGetFieldKeys(
	FDB *				pDb,
	IXD *				pIxd,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	void **			ppPathFlds,
	FLMUINT			uiLeafFieldLevel,
	void *			pvField,
	FLMBOOL			bRemoveDups,
	F_Pool *			pPool,
	REC_KEY **	 	ppKeyList,
	FLMBOOL *		pbHasCmpKeys)
{
	RCODE					rc = FERR_OK;
	IFD *					pIfd;
	FlmRecord *			pFieldRecord = NULL;
	void *				pvRootContext;
	void *				pvValueField;
	const FLMBYTE *	pExportPtr = NULL;
	FLMBYTE *			pImportDataPtr;
	FLMUINT				uiCounter;
	FLMUINT				uiTagNum;
	FLMUINT				uiExportLen;
	FLMUINT				uiFieldType;
	FLMUINT *			puiFieldPath;
	FLMUINT				uiLevel;
	FLMUINT				uiLanguage = pIxd->uiLanguage;
	FLMBYTE *			pbyTmpBuf = NULL;

	uiTagNum = pRecord->getFieldID( pvField);

	// See if the field is defined in this index.

	pIfd = pIxd->pFirstIfd;
	for (uiCounter = 0; uiCounter < pIxd->uiNumFlds; uiCounter++, pIfd++)
	{
		if (pIfd->uiFldNum == uiTagNum)
		{
			if (flmCheckIfdPath( pIfd, pRecord, ppPathFlds, uiLeafFieldLevel,
									  pvField, &pvRootContext))
			{
				// Found one that matches.
				
				break;
			}
		}
	}

	// If the field is not part of the index, return.

	if (uiCounter == pIxd->uiNumFlds)
	{
		goto Exit;
	}

	// At this point, we know the field is part of the index.

	pExportPtr = pRecord->getDataPtr( pvField);
	uiExportLen = pRecord->getDataLength( pvField);
	uiFieldType = pRecord->getDataType( pvField);

	if (pIfd->uiFlags & IFD_COMPOUND)
	{
		rc = KYCmpKeyAdd2Lst( pDb, pIxd, pIfd, pvField, pvRootContext);
		*pbHasCmpKeys = TRUE;
	}
	else
	{
		if ((pFieldRecord = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pFieldRecord->setContainerID( uiContainerNum);

		// Build a record with just this field in it.

		puiFieldPath = pIfd->pFieldPathPToC;
		for (uiLevel = 0; *puiFieldPath; puiFieldPath++)
		{
			if (RC_BAD( rc = pFieldRecord->insertLast( uiLevel++, *puiFieldPath,
						  FLM_CONTEXT_TYPE, &pvValueField)))
			{
				goto Exit;
			}
		}

		if (pIfd->uiFlags & IFD_CONTEXT)
		{

			// Create the field and put it into the key list.

			rc = flmKeyAdd( pDb, pIxd, pFieldRecord, uiContainerNum, pPool,
								bRemoveDups, ppKeyList);
		}
		else if ((pIfd->uiFlags & (IFD_EACHWORD | IFD_SUBSTRING)) &&
					(uiFieldType == FLM_TEXT_TYPE))
		{
			const FLMBYTE *	pText = pExportPtr;
			FLMUINT				uiTextLen = uiExportLen;
			FLMUINT				uiKeyLen;
			FLMBOOL				bReturn;
			FLMBOOL				bFirstSubstring = (pIfd->uiFlags & IFD_SUBSTRING) 
																	? TRUE 
																	: FALSE;

			if (!pbyTmpBuf)
			{
				if (RC_BAD( rc = f_alloc( MAX_KEY_SIZ, &pbyTmpBuf)))
				{
					goto Exit;
				}
			}

			// Get each word out of the key and save as a separate key.

			for (pText = pExportPtr;;)
			{
				bReturn = (pIfd->uiFlags & IFD_EACHWORD) 
									? (FLMBOOL) KYEachWordParse( &pText, &uiTextLen, 
											pIfd->uiLimit, pbyTmpBuf, &uiKeyLen) 
									: (FLMBOOL) KYSubstringParse( &pText, &uiTextLen, 
											pIfd->uiFlags, pIfd->uiLimit, pbyTmpBuf,
											&uiKeyLen);
				if (!bReturn)
				{
					break;
				}

				if (!pFieldRecord)
				{
					if ((pFieldRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}

					pFieldRecord->setContainerID( uiContainerNum);
					puiFieldPath = pIfd->pFieldPathPToC;

					for (uiLevel = 0; *puiFieldPath; puiFieldPath++)
					{
						if (RC_BAD( rc = pFieldRecord->insertLast( uiLevel++,
									  *puiFieldPath, FLM_CONTEXT_TYPE, &pvValueField)))
						{
							goto Exit;
						}
					}
				}

				if (RC_BAD( rc = pFieldRecord->allocStorageSpace( pvValueField,
						FLM_TEXT_TYPE, uiKeyLen, 0, 0, 0, &pImportDataPtr, NULL)))
				{
					goto Exit;
				}

				f_memcpy( pImportDataPtr, pbyTmpBuf, uiKeyLen);

				if ((pIfd->uiFlags & IFD_SUBSTRING) && !bFirstSubstring)
				{
					pFieldRecord->setLeftTruncated( pvValueField, TRUE);
				}

				if (RC_BAD( rc = flmKeyAdd( pDb, pIxd, pFieldRecord, uiContainerNum,
							  pPool, bRemoveDups, ppKeyList)))
				{
					goto Exit;
				}

				pFieldRecord->Release();
				pFieldRecord = NULL;

				if ((pIfd->uiFlags & IFD_SUBSTRING) &&
					 uiTextLen == 1 && !(uiLanguage >= FLM_FIRST_DBCS_LANG && 
							uiLanguage <= FLM_LAST_DBCS_LANG))
				{
					break;
				}

				bFirstSubstring = FALSE;
			}
		}
		else
		{
			if (RC_BAD( rc = pFieldRecord->allocStorageSpace( pvValueField,
						  pRecord->getDataType( pvField), uiExportLen, 0, 0, 0,
						  &pImportDataPtr, NULL)))
			{
				goto Exit;
			}

			f_memcpy( pImportDataPtr, pExportPtr, uiExportLen);

			if (RC_BAD( rc = flmKeyAdd( pDb, pIxd, pFieldRecord, uiContainerNum,
						  pPool, bRemoveDups, ppKeyList)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (pFieldRecord)
	{
		pFieldRecord->Release();
	}

	if (pbyTmpBuf)
	{
		f_free( &pbyTmpBuf);
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine builds a compound key and saves it into the key list.
****************************************************************************/
FSTATIC RCODE flmBuildCompoundKey(
	FDB *				pDb,
	IXD *				pIxd,
	CMP_KEY_ELM *	pCmpKeyElm,
	FLMBOOL			bRemoveDups,
	F_Pool *			pPool,
	FLMUINT			uiContainerNum,
	REC_KEY **		ppKeyList)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRecord = NULL;
	FLMBYTE *		pImportDataPtr;

	// Build fields for each value in the list.

	while (pCmpKeyElm)
	{
		if (pCmpKeyElm->uiTagNum != 0)
		{
			void *		pvValueField;
			FLMUINT		uiExportLen = pCmpKeyElm->uiValueLen;

			if (!pRecord)
			{
				if ((pRecord = f_new FlmRecord) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				pRecord->setContainerID( uiContainerNum);
			}

			// VISIT: Add full path of each key.

			if (RC_BAD( rc = pRecord->insertLast( 0, pCmpKeyElm->uiTagNum,
						  pCmpKeyElm->uiType, &pvValueField)))
			{
				goto Exit;
			}

			// VISIT: Don't know what type.

			if (RC_BAD( rc = pRecord->allocStorageSpace( pvValueField,
						  pCmpKeyElm->uiType, uiExportLen, 0, 0, 0, &pImportDataPtr,
						  NULL)))
			{
				goto Exit;
			}

			if (uiExportLen)
			{
				f_memcpy( pImportDataPtr, pCmpKeyElm->pValue, uiExportLen);
			}

			if (pCmpKeyElm->bSubstringComponent && !pCmpKeyElm->bFirstSubstring)
			{
				pRecord->setLeftTruncated( pvValueField, TRUE);
			}
		}

		pCmpKeyElm = pCmpKeyElm->pParent;
	}

	// Add the key to the key list.

	if (pRecord)
	{
		if (RC_BAD( rc = flmKeyAdd( pDb, pIxd, pRecord, uiContainerNum, pPool,
					  bRemoveDups, ppKeyList)))
		{
			goto Exit;
		}
	}

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine gets an element of a compound key and links it to the
		previous element in the compound key.
****************************************************************************/
FSTATIC RCODE flmGetCmpKeyElement(
	FDB *				pDb,
	IXD *				pIxd,
	IFD *				pIfd,
	FLMUINT			uiCdlEntry,
	FLMUINT			uiCompoundPos,
	CMP_KEY_ELM *	pParent,
	FLMBOOL			bRemoveDups,
	F_Pool *			pPool,
	REC_KEY **		ppKeyList,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	FLD_CONTEXT *	pFldContext)
{
	RCODE				rc = FERR_OK;
	CDL **		 	ppCdlTbl = pDb->KrefCntrl.ppCdlTbl;
	CDL *				pCdl = ppCdlTbl[uiCdlEntry];
	CMP_KEY_ELM 	CmpKeyElm;
	void *			pvField;
	void *			pSaveParentAnchor;
	FLMBYTE *		pbyTmpBuf = NULL;
	IFD *				pNextIfdPiece;
	FLMUINT			uiNextCdlEntry;
	FLMUINT			uiNextPiecePos;
	FLMUINT			uiLanguage = pIxd->uiLanguage;
	FLMBOOL			bBuiltKeyPiece;

	// If there are no values, see if we are on the last IFD.

	CmpKeyElm.pParent = pParent;
	CmpKeyElm.pValue = NULL;
	CmpKeyElm.uiType = 0;
	CmpKeyElm.uiValueLen = 0;
	CmpKeyElm.uiTagNum = 0;
	CmpKeyElm.bFirstSubstring = FALSE;
	CmpKeyElm.bSubstringComponent = FALSE;

	// Determine the next IFD compound piece.

	for (pNextIfdPiece = (IFD*) NULL, uiNextCdlEntry = uiCdlEntry +
		  1, uiNextPiecePos = 0;
	  ((pIfd + uiNextPiecePos)->uiFlags & IFD_LAST) == 0;)
	{
		if ((pIfd + uiNextPiecePos)->uiCompoundPos !=
				 (pIfd + uiNextPiecePos + 1)->uiCompoundPos)
		{
			pNextIfdPiece = pIfd + uiNextPiecePos + 1;
			uiNextCdlEntry = uiCdlEntry + uiNextPiecePos + 1;
			break;
		}

		if (!pCdl)
		{
			pIfd++;
			pCdl = ppCdlTbl[++uiCdlEntry];
			uiNextCdlEntry = uiCdlEntry + 1;
		}
		else
		{
			uiNextPiecePos++;
		}
	}

	pSaveParentAnchor = pFldContext->pParentAnchor;
	bBuiltKeyPiece = FALSE;

	// Loop through all of the values in this IFD.

	while (pCdl || !bBuiltKeyPiece)
	{

		// Restore context values for each iteration.

		pFldContext->pParentAnchor = pSaveParentAnchor;

		// If there is a field to process, verify that its path is relative
		// to the previous non-null compound pieces.

		if (pCdl)
		{
			pvField = pCdl->pField;

			// Validate the current and previous root contexts.

			if (KYValidatePathRelation( pRecord, pCdl->pRootContext, pvField,
												pFldContext, uiCompoundPos) == FERR_FAILURE)
			{

				// This field didn't pass the test, get the next field.

				goto Next_CDL_Node;
			}

			CmpKeyElm.pValue = pRecord->getDataPtr( pvField);
			CmpKeyElm.uiValueLen = pRecord->getDataLength( pvField);
			CmpKeyElm.uiType = pRecord->getDataType( pvField);
			CmpKeyElm.uiTagNum = pRecord->getFieldID( pvField);
			CmpKeyElm.bFirstSubstring = FALSE;
			CmpKeyElm.bSubstringComponent = FALSE;
		}
		else
		{
			pvField = NULL;
		}

		if (pRecord &&
			 (pIfd->uiFlags & (IFD_EACHWORD | IFD_SUBSTRING)) &&
			 CmpKeyElm.uiType == FLM_TEXT_TYPE &&
			 pRecord->getDataLength( pvField))
		{
			const FLMBYTE *	pText = pRecord->getDataPtr( pvField);
			FLMUINT				uiTextLen = pRecord->getDataLength( pvField);
			FLMUINT				uiKeyLen;
			FLMBOOL				bReturn;
			FLMBOOL				bFirstSubstring = (pIfd->uiFlags & IFD_SUBSTRING) 
																		? TRUE 
																		: FALSE;

			if (!pbyTmpBuf)
			{
				if (RC_BAD( rc = f_alloc( MAX_KEY_SIZ, &pbyTmpBuf)))
				{
					goto Exit;
				}
			}

			for (;;)
			{
				bReturn = (pIfd->uiFlags & IFD_EACHWORD) 
										? (FLMBOOL) KYEachWordParse( &pText, &uiTextLen,
												pIfd->uiLimit, pbyTmpBuf, &uiKeyLen) 
										: (FLMBOOL) KYSubstringParse( &pText, &uiTextLen,
												pIfd->uiFlags, pIfd->uiLimit, pbyTmpBuf,
												&uiKeyLen);
				if (!bReturn)
				{
					break;
				}

				CmpKeyElm.pValue = pbyTmpBuf;
				CmpKeyElm.uiValueLen = uiKeyLen;
				CmpKeyElm.uiType = FLM_TEXT_TYPE;
				CmpKeyElm.bFirstSubstring = bFirstSubstring;
				CmpKeyElm.bSubstringComponent = (pIfd->uiFlags & IFD_SUBSTRING) 
																	? TRUE 
																	: FALSE;
				
				if (pIfd->uiFlags & IFD_LAST)
				{
					rc = flmBuildCompoundKey( pDb, pIxd, &CmpKeyElm, bRemoveDups,
													 pPool, uiContainerNum, ppKeyList);
				}
				else
				{
					rc = flmGetCmpKeyElement( pDb, pIxd, (pIfd + 1), uiCdlEntry + 1,
													 uiCompoundPos + 1, &CmpKeyElm,
													 bRemoveDups, pPool, ppKeyList, pRecord,
													 uiContainerNum, pFldContext);
				}

				if (RC_BAD( rc))
				{
					goto Exit;
				}

				if ((pIfd->uiFlags & IFD_SUBSTRING) &&
					 uiTextLen == 1 && !(uiLanguage >= FLM_FIRST_DBCS_LANG && 
							uiLanguage <= FLM_LAST_DBCS_LANG))
				{
					break;
				}

				bFirstSubstring = FALSE;
			}
		}
		else
		{
			CmpKeyElm.bSubstringComponent = FALSE;
			if (pIfd->uiFlags & IFD_CONTEXT)
			{
				CmpKeyElm.uiValueLen = 0;
			}

			if (pIfd->uiFlags & IFD_LAST)
			{
				rc = flmBuildCompoundKey( pDb, pIxd, &CmpKeyElm, bRemoveDups, pPool,
												 uiContainerNum, ppKeyList);
			}
			else
			{
				rc = flmGetCmpKeyElement( pDb, pIxd, pNextIfdPiece, uiNextCdlEntry,
												 uiCompoundPos + 1, &CmpKeyElm, bRemoveDups,
												 pPool, ppKeyList, pRecord, uiContainerNum,
												 pFldContext);
			}

			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}

		bBuiltKeyPiece = TRUE;

Next_CDL_Node:

		// Go to next cdl.

		if (pCdl)
		{
			pCdl = pCdl->pNext;
		}

		// If the CDL list is empty, goto the next IFD if same
		// uiCompoundPos.

		while( (!pCdl) && ((pIfd->uiFlags & IFD_LAST) == 0) &&
				 (pIfd->uiCompoundPos == (pIfd + 1)->uiCompoundPos))
		{
			pIfd++;
			pCdl = ppCdlTbl[++uiCdlEntry];
		}

		// If all nodes failed the validate field path test and this piece
		// of the compound key is required, then goto exit NOW which will
		// not build any key with the previous built key pieces.

		if (!pCdl && !bBuiltKeyPiece && ((pIfd->uiFlags & IFD_OPTIONAL) == 0))
		{
			goto Exit;
		}
	}

Exit:

	if (pbyTmpBuf)
	{
		f_free( &pbyTmpBuf);
	}

	return (rc);
}

/****************************************************************************
Desc: This routine builds all of the compound keys whose elements have
		been previously saved off of the index's IFD structures.
Note:	Already knows that there are compound keys.
****************************************************************************/
FSTATIC RCODE flmGetCompoundKeys(
	FDB *				pDb,
	IXD *				pIxd,
	FLMBOOL			bRemoveDups,
	F_Pool *			pPool,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	REC_KEY **	 	ppKeyList)
{
	RCODE				rc = FERR_OK;
	IFD *				pFirstIfd;
	IFD *				pIfd;
	CDL **	 		ppCdlTbl = pDb->KrefCntrl.ppCdlTbl;
	FLMUINT			uiFirstCdlEntry;
	FLMUINT			uiCdlEntry;
	FLMUINT			wIfdCnt;
	FLMBOOL			bBuildCmpKeys = TRUE;

	pFirstIfd = pIxd->pFirstIfd;
	uiFirstCdlEntry = (FLMUINT) (pFirstIfd - pDb->pDict->pIfdTbl);

	// Make sure we have all of the required fields for key generation.

	for (wIfdCnt = 0, pIfd = pFirstIfd, uiCdlEntry = uiFirstCdlEntry;
		  wIfdCnt < pIxd->uiNumFlds;
		  pIfd++, wIfdCnt++, uiCdlEntry++)
	{

		// If field is not optional and no data list found, there are no
		// compound keys to build.

		FLMUINT	uiCompoundPos;
		FLMBOOL	bHitFound;

		// Loop on each compound field piece looking for REQUIRED field
		// without any data.

		bHitFound = (pIfd->uiFlags & IFD_OPTIONAL) ? TRUE : FALSE;
		uiCompoundPos = pIfd->uiCompoundPos;
		
		for (;;)
		{
			if (!bHitFound)
			{
				if (ppCdlTbl[uiCdlEntry])
				{
					bHitFound = TRUE;
				}
			}

			if ((pIfd->uiFlags & IFD_LAST) ||
				 ((pIfd + 1)->uiCompoundPos != uiCompoundPos))
			{
				break;
			}

			pIfd++;
			uiCdlEntry++;
			wIfdCnt++;
		}

		if (!bHitFound)
		{
			bBuildCmpKeys = FALSE;
			break;
		}
	}

	// Build the individual compound keys.

	if (bBuildCmpKeys)
	{
		FLD_CONTEXT fldContext;

		f_memset( &fldContext, 0, sizeof(FLD_CONTEXT));
		rc = flmGetCmpKeyElement( pDb, pIxd, pFirstIfd, uiFirstCdlEntry, 0, NULL,
										 bRemoveDups, pPool, ppKeyList, pRecord,
										 uiContainerNum, &fldContext);
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine builds all of the keys in a record for a particular
		index in the database.
****************************************************************************/
RCODE flmGetRecKeys(
	FDB *				pDb,
	IXD *				pIxd,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	FLMBOOL			bRemoveDups,
	F_Pool *			pPool,
	REC_KEY **	 	ppKeyList)
{
	RCODE				rc = FERR_OK;
	void *			pvField;
	FlmRecord *		pTmpRec = NULL;
	FLMUINT			uiFieldCount;
	FLMUINT			uiSaveFieldID = pRecord->getFieldID( pRecord->root());
	FLMBOOL			bResetID = FALSE;
	FLMBOOL			bHasCmpKeys = FALSE;
	FLMBOOL			bDictIx = FALSE;
	void *			pathFlds[GED_MAXLVLNUM + 1];
	FLMUINT			uiLeafFieldLevel;

	*ppKeyList = NULL;

	if (pIxd->uiIndexNum == FLM_DICT_INDEX)
	{
		bDictIx = TRUE;

		// Temporary convert the record's tag number to FLM_NAME_TAG so that
		// we will generate the appropriate name key.

		if ((uiSaveFieldID >= FLM_DICT_FIELD_NUMS) &&
			 (uiSaveFieldID <= FLM_LAST_DICT_FIELD_NUM))
		{
			if (pRecord->isReadOnly())
			{
				if ((pTmpRec = pRecord->copy()) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				pRecord = pTmpRec;
			}

			pRecord->setFieldID( pRecord->root(), FLM_NAME_TAG);
			bResetID = TRUE;
		}
	}

	// Process all fields in the tree.

	uiFieldCount = 256;
	for (pvField = pRecord->root(); pvField; pvField = pRecord->next( pvField))
	{

		// Is if field is indexed before building the keys.

		uiLeafFieldLevel = (FLMINT) pRecord->getLevel( pvField);
		pathFlds[uiLeafFieldLevel] = pvField;
		if (RC_BAD( rc = flmGetFieldKeys( pDb, pIxd, pRecord, uiContainerNum,
					  pathFlds, uiLeafFieldLevel, pvField, bRemoveDups, pPool,
					  ppKeyList, &bHasCmpKeys)))
		{
			goto Exit;
		}

		// Release the CPU periodically to prevent CPU hog problems.

		if (uiFieldCount-- == 0)
		{
			f_yieldCPU();
			uiFieldCount = 128;
		}
	}

	// If OK, get the compound keys.

	if (bHasCmpKeys)
	{
		if (RC_BAD( rc = flmGetCompoundKeys( pDb, pIxd, bRemoveDups, pPool,
					  pRecord, uiContainerNum, ppKeyList)))
		{
			goto Exit;
		}
	}

Exit:

	if (pTmpRec)
	{
		pTmpRec->Release();
	}
	else if (bResetID)
	{
		pRecord->setFieldID( pRecord->root(), uiSaveFieldID);
	}

	return (rc);
}
