//-------------------------------------------------------------------------
// Desc:	Rebuild corrupted database.
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

typedef struct Container_Info
{
	FLMUINT	uiHighetstDrnFound;
	FLMUINT	uiHighestNextDrnFound;
	FLMUINT	uiNumNextDrnsFound;
	FLMBOOL	bCountEstimated;
} CONTAINER_INFO, * CONTAINER_INFO_p;

typedef struct RECOV_DICT_REC
{
	FlmRecord *			pRec;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiElmOffset;
	FLMBOOL				bAdded;
	FLMBOOL				bGotFromDataRec;
	RECOV_DICT_REC *	pNext;
} RECOV_DICT_REC;

typedef struct RECOV_DICT_INFO
{
	RECOV_DICT_REC *	pRecovRecs;
	F_Pool				pool;
} RECOV_DICT_INFO;

FSTATIC RCODE bldAdjustNextDrn(
	FDB *					pDb,
	LFILE *				pLFile,
	CONTAINER_INFO *	pContInfo);

FSTATIC RCODE bldRecovData(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	CONTAINER_INFO *	pContainerInfo,
	FLMBOOL				bRecovDictRecs,
	FLMBOOL *			pbStartedTransRV);

FSTATIC RCODE bldCheckBlock(
	STATE_INFO *		pStateInfo,
	HDR_INFO *			pHdrInfo,
	FLMUINT				uiBlkAddress,
	FLMUINT				uiPrevBlkAddress,
	eCorruptionType *	peCorruptionCode);

FSTATIC RCODE bldExtractRecs(
	FDB *						pDb,
	REBUILD_STATE *		pRebuildState,
	LFILE *					pLFile,
	CONTAINER_INFO *		pContInfo,
	FLMUINT					uiBlkAddress,
	LF_HDR *					pLogicalFile,
	FLMBOOL					bRecovDictRecs,
	RECOV_DICT_INFO **	ppRecovDictInfoRV);

FSTATIC RCODE bldGetNextElm(
	REBUILD_STATE *	pRebuildState,
	STATE_INFO *		pStateInfo,
	FLMBOOL *			pbGotNextElmRV,
	FLMBOOL *			pbGotNewBlockRV);

FSTATIC RCODE bldGetOneRec(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	STATE_INFO *		pStateInfo,
	FLMBOOL				bRecovDictRecs,
	FLMBOOL *			pbGotNewBlockRV,
	FLMBOOL *			pbGotRecord);

FSTATIC RCODE bldSaveRecovDictRec(
	FDB *						pDb,
	RECOV_DICT_INFO **	ppRecovDictInfoRV,
	FlmRecord *				pRecord,
	FLMUINT					uiDrn,
	FLMBOOL					bGotFromDataRec,
	FLMUINT					uiBlkAddress,
	FLMUINT					uiElmOffset);

FSTATIC void bldFreeRecovDictInfo(
	RECOV_DICT_INFO *		pRecovDictInfo);

FSTATIC RCODE bldDoDict(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	RECOV_DICT_INFO *	pDictToDo,
	FLMBOOL *			pbStartedTransRV);

FSTATIC RCODE bldDetermineBlkSize(
	const char *		pszSourceDbPath,
	const char *		pszSourceDataDir,
	FLMUINT				uiDbVersion,
	FLMUINT				uiMaxFileSize,
	FLMUINT *			puiBlkSizeRV,
	STATUS_HOOK			fnStatusFunc,
	REBUILD_INFO *		pCallbackData,
	void *				AppArg);
	
/***************************************************************************
Desc:	This routine adds all of the recovered dictionary records to their
		appropriate dictionaries.
****************************************************************************/
FINLINE RCODE bldAddRecovDictRecs(
	FDB *						pDb,
	REBUILD_STATE *		pRebuildState,
	RECOV_DICT_INFO **	ppDictListRV,
	FLMBOOL *				pbStartedTransRV)
{
	RECOV_DICT_INFO *	pDict;

	if( (pDict = *ppDictListRV) != NULL)
	{
		*ppDictListRV = NULL;
		return bldDoDict( pDb, pRebuildState, pDict,	pbStartedTransRV);
	}

	return FERR_OK;
}

/***************************************************************************
Desc: Setup corrupt info structure prior to calling status callback
****************************************************************************/
FINLINE RCODE bldReportReason(
	REBUILD_STATE *	pRebuildState,
	eCorruptionType	eCorruption,
	FLMUINT				uiErrBlkAddress,
	FLMUINT				uiErrElmOffset,
	FLMUINT				uiErrDrn,
	FLMUINT				uiErrElmRecOffset,
	FLMUINT				uiErrFieldNum)
{
	RCODE		rc = FERR_OK;

	if( pRebuildState->fnStatusFunc)
	{
		pRebuildState->CorruptInfo.eCorruption = eCorruption;
		pRebuildState->CorruptInfo.uiErrBlkAddress = uiErrBlkAddress;
		pRebuildState->CorruptInfo.uiErrElmOffset = uiErrElmOffset;
		pRebuildState->CorruptInfo.uiErrDrn = uiErrDrn;
		pRebuildState->CorruptInfo.uiErrElmRecOffset = uiErrElmRecOffset;
		pRebuildState->CorruptInfo.uiErrFieldNum = uiErrFieldNum;
		rc = (*pRebuildState->fnStatusFunc)( FLM_PROBLEM_STATUS,
						(void *)&pRebuildState->CorruptInfo,
						(void *)0, pRebuildState->AppArg);
		pRebuildState->CorruptInfo.eCorruption = FLM_NO_CORRUPTION;
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine determines whether or not a container should be done.
****************************************************************************/
FINLINE FLMBOOL bldDoContainer(
	FLMUINT	uiContainerNum,
	FLMBOOL	bDoDictContainers)
{
	if (!bDoDictContainers)
	{
		switch (uiContainerNum)
		{
			case FLM_DICT_CONTAINER:
				return( FALSE);
			case FLM_DATA_CONTAINER:
				return( TRUE);
			default:
				if (uiContainerNum < FLM_RESERVED_TAG_NUMS)
				{
					return( TRUE);
				}
				else
				{
					return( FALSE);
				}
		}
	}

	return( TRUE);
}

/***************************************************************************
Desc:	Reads through a database, extracts data records from all containers
		and puts them into the database specified by hDb.  It is
		assumed that the new database has the same containers as the old
		database.
****************************************************************************/
RCODE flmDbRebuildFile(
	REBUILD_STATE *	pRebuildState,	// Rebuild state information.
	FLMBOOL				bBadHeader		// Was file's header or log header
												// information bad?
	)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiCurrLf;
	FLMBOOL				bFdbInitialized = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	LFILE *				pLFile;
	FLMUINT				uiTemp;
	FLMUINT				uiContainerNum;
	CONTAINER_INFO *	pContainerInfo = NULL;
	CONTAINER_INFO *	pContInfo;
	FDB *					pDb = (FDB *)pRebuildState->hDb;

	pRebuildState->CorruptInfo.eErrLocale = LOCALE_B_TREE;
	pRebuildState->CorruptInfo.uiErrLfType = LF_CONTAINER;

	bFdbInitialized = TRUE;
	if (RC_BAD( fdbInit( pDb, FLM_UPDATE_TRANS,
									FDB_DONT_RESET_DIAG,
									FLM_AUTO_TRANS | FLM_NO_TIMEOUT, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_allocAlignedBuffer( 
		pRebuildState->pHdrInfo->FileHdr.uiBlockSize,
		&pRebuildState->pBlk)))
	{
		goto Exit;
	}

	// Do a first pass to recover any dictionary items that may not have
	// been added from the dictionary file that was passed into the rebuild
	// function.

	if( RC_BAD( rc = bldRecovData( pDb, pRebuildState, pContainerInfo, TRUE,
								&bStartedTrans)))
	{
		goto Exit;
	}

	// Reset records recovered to zero after dictionary pass.

	pRebuildState->CallbackData.uiRecsRecov = 0;

	// Allocate an array of structures to keep information on each
	// container.

	if( RC_BAD( rc = f_calloc( 
				(FLMUINT)( sizeof( CONTAINER_INFO) * pDb->pDict->uiLFileCnt),
				 &pContainerInfo)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = bldRecovData( pDb, pRebuildState, pContainerInfo, FALSE,
								&bStartedTrans)))
	{
		goto Exit;
	}

	// Adjust the next DRN for all containers so that they are at least as high
	// as the next DRN in the containers we were rebuilding from.

	for( uiCurrLf = 0,
			pContInfo = pContainerInfo,
			pLFile = (LFILE *)pDb->pDict->pLFileTbl;
			uiCurrLf < pDb->pDict->uiLFileCnt;
			uiCurrLf++, pLFile++, pContInfo++)
	{
		if (pLFile->uiLfType == LF_CONTAINER)
		{
			uiContainerNum = pLFile->uiLfNum;
			if (bldDoContainer( uiContainerNum, FALSE))
			{
				if (RC_BAD( rc = bldAdjustNextDrn( pDb, pLFile, pContInfo)))
				{
					goto Exit;
				}
			}
		}
	}

	// Preserve other things in the log header that we ought
	// to try and preserve.

	if( !bBadHeader)
	{
		FLMBYTE *	pucLogHdr = &pDb->pFile->ucUncommittedLogHdr [0];

		// Set the commit count one less than the old database's
		// This is because it will be incremented if the transaction 
		// successfully commits - which will make it exactly right.

		uiTemp = (FLMUINT)FB2UD( &pRebuildState->pLogHdr [LOG_COMMIT_COUNT]) - 1;
		if ((FLMUINT)FB2UD( &pucLogHdr [LOG_COMMIT_COUNT]) < uiTemp)
		{
			UD2FBA( (FLMUINT32)uiTemp, &pucLogHdr [LOG_COMMIT_COUNT]);
		}
	}

	// Signal the we are finished the rebuild

	if (pRebuildState->fnStatusFunc)
	{
		pRebuildState->CallbackData.iDoingFlag = REBUILD_FINISHED;
	
		pRebuildState->CallbackData.bStartFlag = TRUE;
	
		if (RC_BAD( rc = (*pRebuildState->fnStatusFunc)( FLM_REBUILD_STATUS,
												(void *)&pRebuildState->CallbackData,
												(void *)0,
												pRebuildState->AppArg)))
		{
			goto Exit;
		}
		pRebuildState->CallbackData.bStartFlag = FALSE;
	}

Exit:

	if (pContainerInfo)
	{
		f_free( &pContainerInfo);
	}

	if (pRebuildState->pBlk)
	{
		f_freeAlignedBuffer( &pRebuildState->pBlk);
	}

	if (bStartedTrans)
	{
		if (rc == FERR_OK)
		{
			rc = flmCommitDbTrans( pDb, 0, TRUE);
		}
		else
		{
			(void)flmAbortDbTrans( pDb);
		}
	}

	if (bFdbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine adjusts the next DRN for a container so that it
		is at least as high as the DRN in the file we are rebuilding
		from.
****************************************************************************/
FSTATIC RCODE bldAdjustNextDrn(
	FDB *					pDb,
	LFILE *				pLFile,
	CONTAINER_INFO *	pContInfo)
{
	RCODE		rc;
	FLMUINT	uiNextDrn;
	BTSK		StackBuf [BH_MAX_LEVELS];
	FLMBOOL	bUsedStack = FALSE;

	// First see what the next DRN is currently set to

	uiNextDrn = 0;
	if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, FALSE, &uiNextDrn)))
	{
		goto Exit;
	}

	// Adjust the next DRN to be at least as high as the highest DRN
	// in the old databaseor the highest next DRN found int the
	// old database.

	if( uiNextDrn < pContInfo->uiHighetstDrnFound)
	{
		uiNextDrn = pContInfo->uiHighetstDrnFound + 1;
	}

	if( uiNextDrn < pContInfo->uiHighestNextDrnFound)
	{
		uiNextDrn = pContInfo->uiHighestNextDrnFound;
	}

	// Add either 100 or 1000 to next record number - just in case
	// things weren't as accurate as they should have been in the
	// old database.  We want to make sure the next record
	// number is high enough to avoid accidentally reusing any
	// records which may have been used in the old database.

	uiNextDrn += ((pContInfo->uiNumNextDrnsFound != 1) ? 1000 : 100);

	// If there is no root block, the next DRN is stored inside the
	// logical file header.  Otherwise, it is stored in the rightmost
	// element of the B-Tree - the one with a DRN of DRN_LAST_MARKER.

	if( pLFile->uiRootBlk == BT_END)
	{
		// LFILE is up to date from previously calling FSGetNxtDrn

		pLFile->uiNextDrn = uiNextDrn;
		if (RC_BAD( rc = flmLFileWrite( pDb, pLFile)))
		{
			goto Exit;
		}
	}
	else
	{
		BTSK *		pStack = StackBuf;				
		FLMBYTE		KeyBuf [DIN_KEY_SIZ + 4];
		FLMBYTE		DrnMarker [DIN_KEY_SIZ];
		FLMBYTE *	pNextDrnBuf;

		// Set up the stack

		FSInitStackCache( &StackBuf [0], BH_MAX_LEVELS);
		bUsedStack = TRUE;
		pStack->pKeyBuf = KeyBuf;

		// Find the element whose DRN is DRN_LAST_MARKER

		f_UINT32ToBigEndian( (FLMUINT32)DRN_LAST_MARKER, DrnMarker);
		if( RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, DrnMarker,
											 DIN_KEY_SIZ, 0)))
		{
			goto Exit;
		}

		// Log the block before modifying it

		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			goto Exit;
		}

		pNextDrnBuf = CURRENT_ELM( pStack);
		pNextDrnBuf += BBE_GETR_KL( pNextDrnBuf ) + BBE_KEY;

		// Update with the next DRN value and dirty the block

		UD2FBA( (FLMUINT32)uiNextDrn, pNextDrnBuf );
	}

Exit:

	if( bUsedStack)
	{
		FSReleaseStackCache( StackBuf, BH_MAX_LEVELS, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine recovers all records in the database for all
		containers.
*****************************************************************************/
FSTATIC RCODE bldRecovData(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	CONTAINER_INFO *	pContainerInfo,
	FLMBOOL				bRecovDictRecs,
	FLMBOOL *			pbStartedTransRV)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiBytesRead;
	F_SuperFileHdl *	pSFileHdl = pRebuildState->pSFileHdl;
	HDR_INFO *			pHdrInfo = pRebuildState->pHdrInfo;
	LF_HDR				LogicalFile;
	LF_STATS				LfStats;
	FLMBYTE *			pucBlk = pRebuildState->pBlk;
	STATUS_HOOK			fnStatusFunc = pRebuildState->fnStatusFunc;
	REBUILD_INFO *		pCallbackData = &pRebuildState->CallbackData;
	void *				AppArg = pRebuildState->AppArg;
	FLMUINT				uiBlockSize = pHdrInfo->FileHdr.uiBlockSize;
	FLMUINT				uiCurrContainerNum = 0;
	LFILE *				pCurrLFile = NULL;
	CONTAINER_INFO *	pCurrContInfo = NULL;
	FLMUINT				uiBlkContainerNum;
	LFILE *				pBlkLFile;
	RECOV_DICT_INFO *	pRecovDictInfo = NULL;
	FLMUINT				uiFileNumber = 0;
	FLMUINT				uiOffset = 0;
	FLMUINT				uiMaxFileSize = pRebuildState->uiMaxFileSize;
	FLMUINT				uiDbVersion = pRebuildState->pHdrInfo->FileHdr.uiVersionNum;

	// Read through all blocks in the file -- looking for leaf blocks
	// of containers. Read until we get an error or run out of file.

	LogicalFile.pLfStats = &LfStats;
	fnStatusFunc = pRebuildState->fnStatusFunc;
	pCallbackData->iDoingFlag = (FLMINT)((bRecovDictRecs)
												  ? (FLMINT)REBUILD_RECOVER_DICT
												  : (FLMINT)REBUILD_RECOVER_DATA);
	pCallbackData->bStartFlag = TRUE;
	pCallbackData->ui64BytesExamined = 0;
	
	for (;;)
	{
		if (uiOffset >= uiMaxFileSize || !uiFileNumber)
		{
			uiOffset = 0;
			uiFileNumber++;
			
			if( uiFileNumber > MAX_DATA_BLOCK_FILE_NUMBER( uiDbVersion))
			{
				break;
			}
		}

		// Read the block into memory.
		
		if( RC_BAD( rc = pSFileHdl->readBlock( 
			FSBlkAddress( uiFileNumber, uiOffset), uiBlockSize, 
			pucBlk, &uiBytesRead)))
		{
			if( rc == FERR_IO_PATH_NOT_FOUND)
			{
				rc = FERR_OK;
				break;
			}
			
			if( rc == FERR_IO_END_OF_FILE)
			{
				if( !uiBytesRead)
				{

					// Set uiOffset so we will go to the next file.

					uiOffset = uiMaxFileSize;
					continue;
				}
				else
				{
					rc = FERR_OK;
				}
			}
			else
			{
				goto Exit;
			}
		}

		if( fnStatusFunc)
		{
			pCallbackData->ui64BytesExamined += (FLMUINT64)uiBlockSize;
			if (RC_BAD( rc = (*fnStatusFunc)( FLM_REBUILD_STATUS,
											(void *)pCallbackData,
											(void *)0,
											AppArg)))
			{
				goto Exit;
			}
			pCallbackData->bStartFlag = FALSE;
		}

		f_yieldCPU();

		uiBlkAddress = (FLMUINT)GET_BH_ADDR( pucBlk);
		if ((FSGetFileOffset( uiBlkAddress) == uiOffset) &&
			 (BH_GET_TYPE( pucBlk) == BHT_LEAF) &&
			 (pucBlk [BH_LEVEL] == 0) &&
			 ((uiBlkContainerNum = (FLMUINT)FB2UW( &pucBlk [BH_LOG_FILE_NUM])) != 0) &&
			 (bldDoContainer( uiBlkContainerNum, bRecovDictRecs)))
		{
			if (uiBlkContainerNum != uiCurrContainerNum)
			{
				if (RC_BAD( rc = fdictGetContainer( pDb->pDict, uiBlkContainerNum,
											&pBlkLFile)))
				{
					if (rc != FERR_BAD_CONTAINER)
						goto Exit;
					rc = FERR_OK;
					goto Do_Next_Block;
				}
				else
				{
					uiCurrContainerNum = uiBlkContainerNum;
					f_memset( &LogicalFile, 0, sizeof( LF_HDR));
					f_memset( &LfStats, 0, sizeof( LF_STATS));
					LogicalFile.pLfStats = &LfStats;
					LogicalFile.pLFile = pCurrLFile = pBlkLFile;
					if (bRecovDictRecs)
					{
						pCurrContInfo = NULL;
					}
					else
					{
						pCurrContInfo =
							&pContainerInfo [pCurrLFile -
										((LFILE *)pDb->pDict->pLFileTbl)];
					}
					pRebuildState->CorruptInfo.uiErrLfNumber = uiBlkContainerNum;
				}
			}

			// Estimate the number of records in the block if we did't have
			// a count of records in the container.  The loop ignores the
			// possibility that the block may be corrupted -- we are trying
			// to estimate what might have been in the block.  It loops through
			// the elements in the block, looking for those which are marked
			// as FIRST elements.  When it encounters one of these, it will
			// increment the counter.

			if (pCurrContInfo && !pCurrContInfo->bCountEstimated)
			{
				FLMUINT		uiBlkOffset;
				FLMUINT		uiEndOfBlock;
				FLMBYTE *	pucElm;
				FLMUINT		uiElmLen;
				FLMUINT		uiElmKeyLen;
				FLMUINT		uiElmPKCLen;
				FLMUINT		uiNxtBlkAddr;
				FLMBOOL		bIncremented;

				uiEndOfBlock = (FLMUINT)FB2UW( &pucBlk [BH_ELM_END]);
				uiNxtBlkAddr = (FLMUINT)FB2UD( &pucBlk [BH_NEXT_BLK]);

				// If uiEndOfBlock is too big, adjust it down so we can
				// estimate.

				if (uiEndOfBlock > uiBlockSize)
				{
					uiEndOfBlock = uiBlockSize;
				}
				uiBlkOffset = BH_OVHD;
				bIncremented = FALSE;
				while (uiBlkOffset < uiEndOfBlock)
				{
					pucElm = &pucBlk [uiBlkOffset];
					uiElmLen = (FLMUINT)(BBE_LEN( pucElm));
					uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pucElm));
					uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( pucElm));

					// If it is a FIRST element, and it is NOT the LEM
					// element, increment the count.

					if ((BBE_IS_FIRST( pucElm)) &&
						 ((uiElmLen != BBE_LEM_LEN) ||
						  (uiElmKeyLen > 0) ||
						  (uiElmPKCLen > 0) ||
						  (uiBlkOffset + uiElmLen != uiEndOfBlock) ||
						  (uiNxtBlkAddr != BT_END)))
					{
						pCallbackData->uiTotRecs++;
						bIncremented = TRUE;
					}
					uiBlkOffset += uiElmLen;
				}

				// Decrement the estimated count by one if it is a last
				// block - one of the elements in a last block should
				// always be a DRN_LAST_MARKER element.

				if ((bIncremented) && (uiNxtBlkAddr == BT_END))
				{
					pCallbackData->uiTotRecs--;
				}
			}

			// See if we can now extract any records from the block

			uiBlkAddress = FSBlkAddress( uiFileNumber, uiOffset);
			if( RC_BAD( rc = bldExtractRecs( pDb, pRebuildState, pCurrLFile,
														pCurrContInfo, uiBlkAddress,
														&LogicalFile,
														bRecovDictRecs,
														&pRecovDictInfo)))
			{
				goto Exit;
			}
		}

Do_Next_Block:

		uiOffset += uiBlockSize;
	}

	// If we are recovering dictionary records, we need to now add them
	// into the appropriate dictionaries.

	if( bRecovDictRecs)
	{
		if( RC_BAD( rc = bldAddRecovDictRecs( pDb, pRebuildState, 
			&pRecovDictInfo, pbStartedTransRV)))
		{
			goto Exit;
		}
	}

Exit:

	bldFreeRecovDictInfo( pRecovDictInfo);
	return( rc);
}

/***************************************************************************
Desc:	This routine checks a few things in the block header and then
		attempts to decrypt the block so we can read through its elements.
Ret:	0 if block is OK, error code otherwise
*****************************************************************************/
FSTATIC RCODE bldCheckBlock(
	STATE_INFO *		pStateInfo,
	HDR_INFO *			pHdrInfo,
	FLMUINT				uiBlkAddress,
	FLMUINT				uiPrevBlkAddress,
	eCorruptionType *	peCorruptionCode)
{
	RCODE			rc = FERR_OK;

	// Determine where the end of block is -- make sure it is a legal value

	pStateInfo->uiBlkAddress = uiBlkAddress;

	// Must force the block address to be correct so this check will not
	// fail.  We already have previously verified that the offset matches, and
	// we know that we got it from the right block file.  However, the low
	// byte of the block header will not be correct because until we do a
	// block checksum calculation, it will hold the low checksum byte.
	// We don't do a block checksum calculation during rebuild, so at this
	// point, it still holds the low checksum byte.

	SET_BH_ADDR( pStateInfo->pBlk, (FLMUINT32)uiBlkAddress);
	if ((*peCorruptionCode = flmVerifyBlockHeader( pStateInfo, NULL,
										pHdrInfo->FileHdr.uiBlockSize,
										0, uiPrevBlkAddress, FALSE, TRUE)) != FLM_NO_CORRUPTION)
	{
		goto Exit;
	}

	pStateInfo->uiElmOffset = BH_OVHD;

	// Decrypt the block if necessary to check the elements.
	// If encryption is not enabled for the database, or the block
	// is already decrypted, do nothing.

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine traverses all elements within a block, extracting
		whatever records it can from the block.
*****************************************************************************/
FSTATIC RCODE bldExtractRecs(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	LFILE *				pLFile,
	CONTAINER_INFO *	pContInfo,
	FLMUINT				uiBlkAddress,
	LF_HDR *				pLogicalFile,
	FLMBOOL				bRecovDictRecs,
	RECOV_DICT_INFO ** ppRecovDictInfoRV)
{
	RCODE		 			rc = FERR_OK;
	eCorruptionType	eCorruptionCode;
	FLMBOOL		 		bGotNewBlock;
	FLMUINT				uiSaveElmOffset;
	STATE_INFO *		pStateInfo = pRebuildState->pStateInfo;
	FLMBOOL				bStateInitialized = TRUE;
	STATUS_HOOK			fnStatusFunc = pRebuildState->fnStatusFunc;
	void *				AppArg = pRebuildState->AppArg;
	FLMBOOL				bGotRecord;

	// Setup the STATE variable for processing through the block

	flmInitReadState( pStateInfo, &bStateInitialized,
							pRebuildState->pHdrInfo->FileHdr.uiVersionNum,
							pDb, pLogicalFile, 0xFF, BHT_LEAF,
							pRebuildState->pKeyBuffer);
	pStateInfo->pBlk = pRebuildState->pBlk;

	if( (RC_BAD( rc = bldCheckBlock( pStateInfo, pRebuildState->pHdrInfo,
		uiBlkAddress, 0, &eCorruptionCode))) || (eCorruptionCode != FLM_NO_CORRUPTION))
	{
		if( eCorruptionCode != FLM_NO_CORRUPTION)
		{
			(void)bldReportReason( pRebuildState, eCorruptionCode, uiBlkAddress,
								  0, 0, 0xFFFF, 0);
		}
		goto Exit;
	}

	// Go through each element in the block, extracting whatever data
	// we can.  The loop quits if it finds an inconsistency in the block.

	bGotNewBlock = FALSE;
	while ((pStateInfo->uiElmOffset < pStateInfo->uiEndOfBlock) &&
			 (!bGotNewBlock))
	{
		bGotRecord = FALSE;
		if (pRebuildState->pRecord)
		{
			pRebuildState->pRecord->clear();
		}
		uiSaveElmOffset = pStateInfo->uiElmOffset;

		// Get the element and check it

		if( (eCorruptionCode = flmVerifyElement( pStateInfo, FLM_CHK_FIELDS)) != FLM_NO_CORRUPTION)
		{
			if( RC_BAD( rc = bldReportReason( pRebuildState, eCorruptionCode, uiBlkAddress,
								  pStateInfo->uiElmOffset,
								  pStateInfo->uiElmDrn, 0xFFFF, 0)))
			{
				goto Exit;
			}
		}

		// Skip continuation elements -- at this point, it should only
		// be continuation elements which are at the first of the block.
		// This is because the bldGetOneRec routine will traverse through
		// continuation elements for a record.

		else if ((BBE_IS_FIRST( pStateInfo->pElm)) && (pStateInfo->uiCurKeyLen))
		{
			if (pStateInfo->uiElmDrn == DRN_LAST_MARKER)
			{

				if (pStateInfo->uiElmRecLen == 4)
				{
					FLMUINT	uiNxtDrn = (FLMUINT)FB2UD( pStateInfo->pElmRec);

					if (pContInfo)
					{
						pContInfo->uiNumNextDrnsFound++;
						if (uiNxtDrn > pContInfo->uiHighestNextDrnFound)
						{
							pContInfo->uiHighestNextDrnFound = uiNxtDrn;
						}
					}
				}
			}

			// If the element is the FIRST element in a record,
			// see if we can extract a record from it.

			else if( RC_BAD( rc = bldGetOneRec( pDb, pRebuildState, pStateInfo,
							bRecovDictRecs, &bGotNewBlock, &bGotRecord)))
			{
				// If we didn't have enough memory to retrieve the record, just
				// skip it.

				if( rc == FERR_MEM)
				{
					bGotRecord = FALSE;
					if (pRebuildState->pRecord)
					{
						pRebuildState->pRecord->clear();
					}
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}

		// If we didn't get a data record, there was some inconsistency
		// we encountered, so we continue to the next element in the block.

		if( bGotRecord)
		{
			f_yieldCPU();

			if( pContInfo)
			{
				if( pStateInfo->uiElmDrn > pContInfo->uiHighetstDrnFound)
				{
					pContInfo->uiHighetstDrnFound = pStateInfo->uiElmDrn;
				}
			}

			// Add the record to the database

			if( bRecovDictRecs)
			{
				FlmRecord *		pDictRec;
				FLMUINT			uiDictDrn = 0;
				FLMBOOL			bGotFromDataRec;
				CHK_RECORD		ChkRec;

				f_memset( &ChkRec, 0, sizeof( ChkRec));

				// If this is not a dictionary container record, need to do the
				// callback to see if there is dictionary information in this
				// record.

				if( pLFile->uiLfNum == FLM_DICT_CONTAINER)
				{
					pDictRec = pRebuildState->pRecord;
					uiDictDrn = pStateInfo->uiElmDrn;
					bGotFromDataRec = FALSE;
				}
				else
				{
					if( !fnStatusFunc)
					{
						pDictRec = NULL;
					}
					else
					{
						ChkRec.pRecord = pRebuildState->pRecord;
						ChkRec.uiContainer = pLFile->uiLfNum;
						ChkRec.uiDrn = pStateInfo->uiElmDrn;
						if( RC_BAD( rc = (*fnStatusFunc)( FLM_CHECK_RECORD_STATUS,
														(void *)&ChkRec,
														(void *)0,
														AppArg)))
						{
							if( ChkRec.pDictRecSet)
							{
								ChkRec.pDictRecSet->Release();
								ChkRec.pDictRecSet = NULL;
							}
							goto Exit;
						}

						if( ChkRec.pDictRecSet)
						{
							if( (pDictRec = ChkRec.pDictRecSet->first()) != NULL)
							{
								uiDictDrn = pDictRec->getID();
							}
						}
						else
						{
							pDictRec = NULL;
						}
						bGotFromDataRec = TRUE;
					}
				}

				if( !pDictRec)
				{
					rc = FERR_OK;
				}
				else
				{
					for (;;)
					{
						rc = bldSaveRecovDictRec( pDb,
											ppRecovDictInfoRV, pDictRec,
											uiDictDrn, bGotFromDataRec, uiBlkAddress,
											uiSaveElmOffset);

						if( RC_BAD( rc) || !bGotFromDataRec)
						{
							break;
						}
						
						// If bGotFromDataRec is TRUE, there may be more than
						// one that was returned.

						if( (pDictRec = ChkRec.pDictRecSet->next()) == NULL)
						{
							break;
						}

						uiDictDrn = pDictRec->getID();
					}
				}

				if( ChkRec.pDictRecSet)
				{
					ChkRec.pDictRecSet->Release();
					ChkRec.pDictRecSet = NULL;
				}
			}
			else if( pLFile->uiLfNum == FLM_TRACKER_CONTAINER)
			{
				rc = FSRecUpdate( pDb, pLFile,
										pRebuildState->pRecord, pStateInfo->uiElmDrn,
										REC_UPD_ADD);
			}
			else
			{
				rc = flmAddRecord( pDb, pLFile, &pStateInfo->uiElmDrn,
										pRebuildState->pRecord, TRUE,
										FALSE, FALSE, FALSE, NULL);

				if( RC_OK(rc) && fnStatusFunc )
				{
					CHK_RECORD	ChkRec;

					f_memset( &ChkRec, 0, sizeof( ChkRec));
					ChkRec.pRecord = pRebuildState->pRecord;
					ChkRec.uiContainer = pLFile->uiLfNum;
					ChkRec.uiDrn = pStateInfo->uiElmDrn;
					rc = (*fnStatusFunc)( FLM_EXAMINE_RECORD_STATUS,
													(void *)&ChkRec,
													(void *)0,
													AppArg);
					if (ChkRec.pDictRecSet)
					{
						ChkRec.pDictRecSet->Release();
						ChkRec.pDictRecSet = NULL;
					}
				}

				if( pRebuildState->pRecord->isReadOnly())
				{
					pRebuildState->pRecord->Release();
					pRebuildState->pRecord = NULL;
				}
			}

			if( RC_BAD( rc))
			{
				if( (rc == FERR_EXISTS) || (rc == FERR_NOT_UNIQUE))
				{
					eCorruptionCode = (rc == FERR_EXISTS)
									 ? FLM_REBUILD_REC_EXISTS
									 : FLM_REBUILD_KEY_NOT_UNIQUE;
					if (RC_BAD( rc = bldReportReason( pRebuildState,
									eCorruptionCode, uiBlkAddress,
									uiSaveElmOffset, pStateInfo->uiElmDrn, 0xFFFF, 0)))
					{
						goto Exit;
					}
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				// Make sure the tempory memory is freed.
				// Eats up the memory during a rebuild.
				
				pDb->TempPool.poolReset();
				if (!bRecovDictRecs)
				{
					pRebuildState->CallbackData.uiRecsRecov++;
				}
			}
		}

		pStateInfo->uiElmOffset += pStateInfo->uiElmLen;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine gets the next element for a record.  If necessary, it
		will try to go to the next block.
Ret:	TRUE if we got the next element, FALSE otherwise.
*****************************************************************************/
FSTATIC RCODE bldGetNextElm(
	REBUILD_STATE *	pRebuildState,
	STATE_INFO *		pStateInfo,
	FLMBOOL *			pbGotNextElmRV,
	FLMBOOL *			pbGotNewBlockRV)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiSaveDrn;
	FLMUINT				uiBytesRead;
	FLMUINT				uiBlkAddress;
	FLMBYTE *			pBlk = pStateInfo->pBlk;
	HDR_INFO *			pHdrInfo = pRebuildState->pHdrInfo;
	eCorruptionType	eCorruptionCode;
	FLMUINT				uiSaveBlkAddress = pStateInfo->uiBlkAddress;

	*pbGotNextElmRV = FALSE;
	uiSaveDrn = pStateInfo->uiElmDrn;

	// See if we need to go to the next block to get the element

	pStateInfo->uiElmOffset += pStateInfo->uiElmLen;
	if (pStateInfo->uiElmOffset >= pStateInfo->uiEndOfBlock)
	{
		// Get the next block

		*pbGotNewBlockRV = TRUE;
		uiBlkAddress = (FLMUINT)FB2UD( &pBlk [BH_NEXT_BLK]);
		rc = pRebuildState->pSFileHdl->readBlock( uiBlkAddress, 
										 pHdrInfo->FileHdr.uiBlockSize,
										 pBlk, &uiBytesRead);
		if( uiBytesRead < pHdrInfo->FileHdr.uiBlockSize)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
		}

		if( RC_BAD( rc))
		{
			RCODE		TempRc;

			if( rc == FERR_IO_END_OF_FILE ||
				 rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				rc = FERR_OK;
			}

			if( RC_BAD( TempRc = bldReportReason( pRebuildState,
										FLM_BAD_BLK_HDR_NEXT, uiSaveBlkAddress,
										0, 0, 0xFFFF, 0)))
			{
				if( RC_OK( rc))
				{
					rc = TempRc;
				}
			}

			goto Exit;
		}

		// Make sure it is the right type of block

		else if( (RC_BAD( rc = bldCheckBlock( pStateInfo, pRebuildState->pHdrInfo,
			uiBlkAddress, uiSaveBlkAddress, &eCorruptionCode))) || (eCorruptionCode != FLM_NO_CORRUPTION))
		{
			if (eCorruptionCode != FLM_NO_CORRUPTION)
			{
				(void)bldReportReason( pRebuildState, eCorruptionCode,
										uiBlkAddress, 0, 0,
									  0xFFFF, 0);
			}

			goto Exit;
		}
	}

	// Verify other things about the element

	if( (eCorruptionCode = flmVerifyElement( pStateInfo, FLM_CHK_FIELDS)) != FLM_NO_CORRUPTION)
	{
		rc = bldReportReason( pRebuildState, eCorruptionCode,
							pStateInfo->uiBlkAddress, pStateInfo->uiElmOffset,
							pStateInfo->uiElmDrn, 0xFFFF, 0);
		goto Exit;
	}

	// This had better not be a LEM element

	if( pStateInfo->uiCurKeyLen == 0)
	{
		rc = bldReportReason( pRebuildState, FLM_BAD_LEM,
							pStateInfo->uiBlkAddress, pStateInfo->uiElmOffset,
							pStateInfo->uiElmDrn, 0xFFFF, 0);
		goto Exit;
	}

	// This element had better not be the first element

	if( BBE_IS_FIRST( pStateInfo->pElm))
	{
		rc = bldReportReason( pRebuildState, FLM_BAD_FIRST_ELM_FLAG,
							pStateInfo->uiBlkAddress, pStateInfo->uiElmOffset,
							pStateInfo->uiElmDrn, 0xFFFF, 0);
		goto Exit;
	}

	// Must stay on the same DRN while processing the record

	if (pStateInfo->uiElmDrn != uiSaveDrn)
	{
		rc = bldReportReason( pRebuildState, FLM_BAD_CONT_ELM_KEY,
							pStateInfo->uiBlkAddress, pStateInfo->uiElmOffset,
							pStateInfo->uiElmDrn, 0xFFFF, 0);
		goto Exit;
	}

	*pbGotNextElmRV = TRUE;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine retrieves one record from a block -- at the current
		element.  It will follow continuation elements if necessary.  The
		record is returned in pRebuildState->pRecord.  A NULL is returned if some
		inconsistency was encountered.
*****************************************************************************/
FSTATIC RCODE bldGetOneRec(
	FDB *					pDb,
	REBUILD_STATE *	pRebuildState,
	STATE_INFO *		pStateInfo,
	FLMBOOL				bRecovDictRecs,
	FLMBOOL *			pbGotNewBlockRV,
	FLMBOOL *			pbGotRecord)
{
	RCODE					rc = FERR_OK;
	FLMBYTE *			pValue;
	FLMBYTE *			pData;
	FLMBYTE *			pTempValue;
	eCorruptionType	eCorruptionCode;
	FlmRecord *			pRecord = NULL;
	FLMBOOL				bAllocatedRecord = FALSE;
	FLMBOOL				bFieldDone;
	FLMUINT				uiSaveElmRecOffset;
	FLMBOOL				bGotNextElm;
	FLMBOOL				bSkipField = FALSE;
	FLMBOOL				bSkippedField = FALSE;
	FLMUINT				uiSkipToLevel = 0;

	// Setup things to get the record

	if( pRebuildState->pRecord)
	{
		pRebuildState->pRecord->clear();
		pRecord = pRebuildState->pRecord;
	}

	pValue = NULL;
	pTempValue = NULL;
	*pbGotRecord = FALSE;

	// Follow elements until we have traversed all continuation elements
	// or until we discover some inconsistency.

	for (;;)
	{
		uiSaveElmRecOffset = pStateInfo->uiElmRecOffset;
		if ((eCorruptionCode = flmVerifyElmFOP( pStateInfo)) != FLM_NO_CORRUPTION)
		{
			if ((bRecovDictRecs) && (eCorruptionCode == FLM_BAD_ELM_FLD_NUM))
			{
				bSkipField = TRUE;
				bSkippedField = TRUE;
				uiSkipToLevel = pStateInfo->uiFieldLevel;
				bFieldDone =
					(pStateInfo->uiFieldProcessedLen == pStateInfo->uiFieldLen)
					? TRUE
					: FALSE;
			}
			else
			{
				rc = bldReportReason( pRebuildState, eCorruptionCode, pStateInfo->uiBlkAddress,
								pStateInfo->uiElmOffset, pStateInfo->uiElmDrn,
								uiSaveElmRecOffset, pStateInfo->uiFieldNum);
				goto Exit;
			}
		}

		// See if we are starting a new field

		if( (pStateInfo->uiFOPType == FLM_FOP_STANDARD) ||
			 (pStateInfo->uiFOPType == FLM_FOP_OPEN) ||
			 (pStateInfo->uiFOPType == FLM_FOP_TAGGED) ||
			 (pStateInfo->uiFOPType == FLM_FOP_NO_VALUE) ||
			 (pStateInfo->uiFOPType == FLM_FOP_ENCRYPTED))
		{
			// If we skipped a previous field, see if this field is a child
			// or grandchild, etc. of the field that was skipped.  If so, we skip
			// this field as well.  We stop skipping when we come to a field
			// that is a sibling or aunt/uncle to the field that was skipped.

			if( (bSkipField) &&
				 (pStateInfo->uiFieldLevel <= uiSkipToLevel))
			{
				bSkipField = FALSE;
			}

			// See if field should be skipped.

			if( !bSkipField)
			{
				FLMUINT 	uiState;
				FLMUINT	uiDictFieldType;

				if( RC_BAD( fdictGetField( pDb->pDict, pStateInfo->uiFieldNum,
												 &uiDictFieldType, NULL, &uiState)))
				{
					bSkipField = TRUE;
					bSkippedField = TRUE;
					uiSkipToLevel = pStateInfo->uiFieldLevel;
				}
			}

			// If we aren't skipping the field, allocate space for it

			if( !bSkipField)
			{
				void *	pvField;

				if( !pRecord)
				{
					if( (pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}

					bAllocatedRecord = TRUE;
				}

				// Create a new field in the record.

				if( RC_BAD( rc = pRecord->insertLast( pStateInfo->uiFieldLevel,
											pStateInfo->uiFieldNum,
											pStateInfo->uiFieldType, &pvField)))
				{
					goto Exit;
				}

				pStateInfo->pvField = pvField;

				// Allocate space for the field's value and set the field's type.

				if( !pStateInfo->uiFieldLen)
				{
					pValue = NULL;
				}
				else
				{
					if (!pStateInfo->uiEncId)
					{
						if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
												pStateInfo->uiFieldType,
												pStateInfo->uiFieldLen,
												0,
												0,
												0,
												&pValue,
												NULL)))
						{
							goto Exit;
						}
					}
					else
					{
						if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
												pStateInfo->uiFieldType,
												pStateInfo->uiFieldLen,
												pStateInfo->uiEncFieldLen,
												pStateInfo->uiEncId,
												FLD_HAVE_ENCRYPTED_DATA,
												&pData,
												&pValue)))
						{
							goto Exit;
						}
					}
				}
				pTempValue = pValue;
			}
			bFieldDone =
				((!pStateInfo->uiEncId &&
						pStateInfo->uiFieldProcessedLen == pStateInfo->uiFieldLen) ||
				 (pStateInfo->uiEncId &&
				 		pStateInfo->uiFieldProcessedLen == pStateInfo->uiEncFieldLen))
				? TRUE
				: FALSE;
		}
		else if( (pStateInfo->uiFOPType == FLM_FOP_JUMP_LEVEL) ||
					(pStateInfo->uiFOPType == FLM_FOP_NEXT_DRN))
		{
			bFieldDone = FALSE;
		}
		else
		{
			bFieldDone =
				((!pStateInfo->uiEncId &&
						pStateInfo->uiFieldProcessedLen == pStateInfo->uiFieldLen) ||
				 (pStateInfo->uiEncId &&
				 		pStateInfo->uiFieldProcessedLen == pStateInfo->uiEncFieldLen))
				? TRUE
				: FALSE;
		}

		// See if we got some data with this FOP

		if( pValue &&
			 pStateInfo->uiFOPDataLen &&
			 !bSkipField &&
			 pStateInfo->uiFOPType != FLM_FOP_REC_INFO)
		{
			f_memcpy( pTempValue,
								pStateInfo->pFOPData, pStateInfo->uiFOPDataLen);
			pTempValue += pStateInfo->uiFOPDataLen;
		}

		// See if we are done with this field

		if( bFieldDone)
		{
			// Verify the value

			if( pStateInfo->uiFieldLen)
			{
				if( bSkipField || pStateInfo->uiFieldType == 0xFF)
				{
					eCorruptionCode = FLM_NO_CORRUPTION;
				}
				else
				{
					if (!pStateInfo->uiEncId)
					{
						eCorruptionCode = flmVerifyField( pStateInfo, pValue,
															  pStateInfo->uiFieldLen,
															  pStateInfo->uiFieldType);
					}
					
					// Encrypted fields are never supposed to be found in the dictionary, so
					// if we do not have an ITT table, we will not be able to decrypt this field.
					// Therefore, we should be in our first pass.  This field / record will
					// not be used to recover the dictionary.
					
					else if (pDb->pFile->pDictList->pIttTbl)
					{
						if (RC_BAD( rc = flmDecryptField( pDb->pDict, pRecord,
							pStateInfo->pvField, pStateInfo->uiEncId,
							&pDb->TempPool)))
						{
							goto Exit;
						}
							eCorruptionCode = flmVerifyField( pStateInfo, pData,
																  pStateInfo->uiFieldLen,
																  pStateInfo->uiFieldType);
					}
				}

				if( eCorruptionCode != FLM_NO_CORRUPTION)
				{
					rc = bldReportReason( pRebuildState, eCorruptionCode,
										  pStateInfo->uiBlkAddress,
										  pStateInfo->uiElmOffset, pStateInfo->uiElmDrn,
										  uiSaveElmRecOffset,
										  pStateInfo->uiFieldNum);
					goto Exit;
				}
			}

			// Set pValue to NULL for the next field

			pValue = pTempValue = NULL;
		}

		// See if we are at the end of this element

		if( pStateInfo->uiElmRecOffset == pStateInfo->uiElmRecLen)
		{
			// If the last element flag is set, we are done with this
			// record and can return - unless we have a half processed field.

			if( BBE_IS_LAST( pStateInfo->pElm))
			{
				if (!bFieldDone)
				{
					rc = bldReportReason( pRebuildState, FLM_BAD_LAST_ELM_FLAG,
										pStateInfo->uiBlkAddress,
										pStateInfo->uiElmOffset, pStateInfo->uiElmDrn,
										uiSaveElmRecOffset,
										pStateInfo->uiFieldNum);
				}
				else
				{
					pRebuildState->pRecord = pRecord;
					*pbGotRecord = TRUE;
					pRecord = NULL;
					bAllocatedRecord = FALSE;
				}

				goto Exit;
			}
			else
			{
				// Attempt to get the next element

				if( (RC_BAD( rc = bldGetNextElm( pRebuildState, pStateInfo,
					&bGotNextElm, pbGotNewBlockRV))) || !bGotNextElm)
				{
					// Need to set bSkippedField to TRUE so that cleanup will
					// occur at the bottom of this routine.

					bSkippedField = TRUE;
					goto Exit;
				}
			}
		}
	}

Exit:

	if( bSkippedField || RC_BAD( rc))
	{
		if (pRebuildState->pRecord)
		{
			pRebuildState->pRecord->Release();
			pRebuildState->pRecord = NULL;
		}

		*pbGotRecord = FALSE;
	}

	if( pRecord && bAllocatedRecord)
	{
		pRecord->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine recovers any dictionary records from the corrupted file
		that it can recover, attempting to find any that may not have been
		in the dictionary file that was passed into the rebuild routines.
*****************************************************************************/
FSTATIC RCODE bldSaveRecovDictRec(
	FDB *						pDb,						// FDB for newly created database
	RECOV_DICT_INFO **	ppRecovDictInfoRV,	// Recover info
	FlmRecord *				pRecord,					// Dictionary Record
	FLMUINT					uiDrn,					// Dict. record DRN
	FLMBOOL					bGotFromDataRec,		// Was this dictionary record
															// extracted from a data record?
	FLMUINT					uiBlkAddress,			// Block the record was
															// recovered from
	FLMUINT					uiElmOffset				// Offset in block the
															// record was recovered
															// from
	)
{
	RCODE						rc = FERR_OK;
	RECOV_DICT_INFO *		pRecovDictInfo;
	RECOV_DICT_REC *		pRecovDictRec;
	RECOV_DICT_REC *		pNewDictRec = NULL;
	RECOV_DICT_REC *		pPrevDictRec;
	FLMUINT					uiRecType = pRecord->getFieldID( pRecord->root());
	FlmRecord *				pDummyRec;
	LFILE *					pDictLFile;

	// Ignore any record that is not a dictionary record or
	// an unregistered (comment) record.

	if( uiRecType < FLM_RESERVED_TAG_NUMS)
	{
		goto Exit;
	}

	// Determine if the record already exists in the dictionary.  If it
	// does, simply ignore this record - the record that was loaded from
	// the dictionary file takes precedence.

	if( RC_OK( rc = fdictGetContainer( pDb->pDict, FLM_DICT_CONTAINER,
											&pDictLFile)))
	{
		pDummyRec = NULL;
		rc = FSReadRecord( pDb, pDictLFile, uiDrn,
								&pDummyRec, NULL, NULL);
		if( pDummyRec)
		{
			pDummyRec->Release();
		}
	}

	if( rc != FERR_NOT_FOUND)
	{
		goto Exit;
	}

	rc = FERR_OK;

	// If the dictionary was not found in the list, create an entry in the
	// list.

	if( (pRecovDictInfo = *ppRecovDictInfoRV) == NULL)
	{
		if( RC_BAD( rc = f_calloc( 
			(FLMUINT)sizeof( RECOV_DICT_INFO), &pRecovDictInfo)))
		{
			goto Exit;
		}

		pRecovDictInfo->pool.poolInit( 512);
		*ppRecovDictInfoRV = pRecovDictInfo;
	}

	// Determine if the record is already in our list of records.  If so,
	// simply ignore it, or remove the old one.  The old one is removed if
	// it is "less desirable to recover".  Desirability of record types is
	// as follows:
	//
	//   1. Field Definitions && Template definitions take top priority
	//   2. Container Definitions
	//   3. Area Definitions
	//   4. Reserve Definitions
	//   5. Index Definitions
	//   6. Other

	pPrevDictRec = NULL;
	pRecovDictRec = pRecovDictInfo->pRecovRecs;

	while( pRecovDictRec)
	{
		if( pRecovDictRec->pRec->getID() != uiDrn)
		{
			pPrevDictRec = pRecovDictRec;
			pRecovDictRec = pRecovDictRec->pNext;
		}
		else if( bGotFromDataRec && !pRecovDictRec->bGotFromDataRec)
		{
			// Throw away this record and keep the one that was previously
			// recovered from the dictionary container.  Records recovered
			// from the dictionary container are preferred to ones that
			// were extracted from a data record.

			goto Exit;
		}
		else if( !bGotFromDataRec && pRecovDictRec->bGotFromDataRec)
		{
			// Throw away prior dictionary record that was recovered, because
			// we got it from a data record.  This newer one was actually found
			// in the dictionary, so we will keep it in preference to the
			// earlier one that we found in a data record.

			goto Remove_Rec;
		}
		else
		{
			switch (pRecovDictRec->pRec->getFieldID( pRecovDictRec->pRec->root()))
			{
				case FLM_FIELD_TAG:
					goto Exit;

				case FLM_CONTAINER_TAG:
					if( uiRecType == FLM_FIELD_TAG)
					{
Remove_Rec:
						if (pPrevDictRec)
						{
							pPrevDictRec->pNext = pRecovDictRec->pNext;
						}
						else
						{
							pRecovDictInfo->pRecovRecs = pRecovDictRec->pNext;
						}

						// Might as well use the old structure for the new record,
						// so we don't have to allocate it below.
						
						pNewDictRec = pRecovDictRec;

						if( pRecovDictRec->pRec)
						{
							pRecovDictRec->pRec->Release();
							pRecovDictRec->pRec = NULL;
						}

						f_memset( pNewDictRec, 0, sizeof( RECOV_DICT_REC));
					}
					else
					{
						goto Exit;
					}
					break;

				case FLM_AREA_TAG:
					if( uiRecType == FLM_FIELD_TAG ||
						 uiRecType == FLM_CONTAINER_TAG)
					{
						goto Remove_Rec;
					}
					else
					{
						goto Exit;
					}

				case FLM_RESERVED_TAG:
					if( uiRecType == FLM_FIELD_TAG ||
						 uiRecType == FLM_CONTAINER_TAG ||
						 uiRecType == FLM_AREA_TAG)
					{
						goto Remove_Rec;
					}
					else
					{
						goto Exit;
					}

				case FLM_INDEX_TAG:
					if( uiRecType == FLM_FIELD_TAG ||
						 uiRecType == FLM_CONTAINER_TAG ||
						 uiRecType == FLM_AREA_TAG ||
						 uiRecType == FLM_RESERVED_TAG)
					{
						goto Remove_Rec;
					}
					else
					{
						goto Exit;
					}

				default:
					if( uiRecType == FLM_FIELD_TAG ||
						 uiRecType == FLM_CONTAINER_TAG ||
						 uiRecType == FLM_AREA_TAG ||
						 uiRecType == FLM_RESERVED_TAG ||
						 uiRecType == FLM_INDEX_TAG)
					{
						goto Remove_Rec;
					}
					else
					{
						goto Exit;
					}
			}
			break;
		}
	}

	// Link the record into the list of records that need to be
	// recovered.  We don't add it right away, because we want to be sure
	// and do them in a certain order, as follows:
	//
	//   1. Field definitions
	//   2. Container definitions
	//   3. Area definitions
	//   4. Template definitions
	//   5. Index definitions
	//   6. Reserve definitions
	//   7. Other

	if( !pNewDictRec)
	{
		// All elements of pNewDictRec are initialized below.
		
		if( RC_BAD( rc = pRecovDictInfo->pool.poolAlloc(
			sizeof( RECOV_DICT_REC), (void **)&pNewDictRec)))
		{
			goto Exit;
		}

		pNewDictRec->pNext = NULL;
		pNewDictRec->bAdded = FALSE;
	}

	if( (pNewDictRec->pRec = pRecord->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
	}

	pNewDictRec->bGotFromDataRec = bGotFromDataRec;
	pNewDictRec->pRec->setID( uiDrn);
	pNewDictRec->uiBlkAddress = uiBlkAddress;
	pNewDictRec->uiElmOffset = uiElmOffset;
	pPrevDictRec = NULL;
	pRecovDictRec = pRecovDictInfo->pRecovRecs;

	while( pRecovDictRec)
	{
		switch( pRecovDictRec->pRec->getFieldID( pRecovDictRec->pRec->root()))
		{
			case FLM_FIELD_TAG:
				break;

			case FLM_CONTAINER_TAG:
				if( uiRecType == FLM_FIELD_TAG)
				{
					goto Insert_Rec;
				}
				break;

			case FLM_AREA_TAG:
				if( uiRecType == FLM_FIELD_TAG ||
					 uiRecType == FLM_CONTAINER_TAG)
				{
					goto Insert_Rec;
				}
				break;

			case FLM_INDEX_TAG:
				if( uiRecType == FLM_FIELD_TAG ||
					 uiRecType == FLM_CONTAINER_TAG ||
					 uiRecType == FLM_AREA_TAG)
				{
					goto Insert_Rec;
				}
				break;

			case FLM_RESERVED_TAG:
				if( uiRecType == FLM_FIELD_TAG ||
					 uiRecType == FLM_CONTAINER_TAG ||
					 uiRecType == FLM_AREA_TAG ||
					 uiRecType == FLM_INDEX_TAG)
				{
					goto Insert_Rec;
				}
				break;

			default:
				if( uiRecType == FLM_FIELD_TAG ||
					 uiRecType == FLM_CONTAINER_TAG ||
					 uiRecType == FLM_AREA_TAG ||
					 uiRecType == FLM_INDEX_TAG ||
					 uiRecType == FLM_RESERVED_TAG)
				{
					goto Insert_Rec;
				}
				break;
		}

		pPrevDictRec = pRecovDictRec;
		pRecovDictRec = pRecovDictRec->pNext;
	}

Insert_Rec:

	pNewDictRec->pNext = pRecovDictRec;
	if( pPrevDictRec)
	{
		pPrevDictRec->pNext = pNewDictRec;
	}
	else
	{
		pRecovDictInfo->pRecovRecs = pNewDictRec;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine frees all of the recovery dictionary information.
*****************************************************************************/
FSTATIC void bldFreeRecovDictInfo(
	RECOV_DICT_INFO *	pRecovDictInfo)
{
	if( pRecovDictInfo)
	{
		RECOV_DICT_REC *	pDictRec;

		pDictRec = pRecovDictInfo->pRecovRecs;
		while (pDictRec)
		{
			if (pDictRec->pRec)
			{
				pDictRec->pRec->Release();
				pDictRec->pRec = NULL;
			}
			pDictRec = pDictRec->pNext;
		}

		pRecovDictInfo->pool.poolFree();
		f_free( &pRecovDictInfo);
	}
}

/***************************************************************************
Desc:	This routine adds all of the recovered dictionary records for a
		specific dictionary.  It makes sure to do parent dictionaries first.
*****************************************************************************/
FSTATIC RCODE bldDoDict(
	FDB *						pDb,
	REBUILD_STATE *		pRebuildState,
	RECOV_DICT_INFO *		pDictToDo,
	FLMBOOL *				pbStartedTransRV)
{
	RCODE						rc = FERR_OK;
	RECOV_DICT_REC *		pDictRec;
	RECOV_DICT_REC *		pFirstDictRecInTrans = NULL;
	LFILE *					pDictLFile;
	STATUS_HOOK				fnStatusFunc = pRebuildState->fnStatusFunc;
	REBUILD_INFO *			pCallbackData = &pRebuildState->CallbackData;
	FLMBOOL					bHaveLFile;
	FLMBOOL					bAddedAtLeastOne;
	FLMBOOL					bFailedAtLeastOne;
	FlmRecord *				pSaveRec;
	FLMUINT					uiRecordsPerTrans;
	FLMUINT					uiRecordInTrans;
	FLMUINT					uiDictRecId;

	// Commit the current update transaction

	if( *pbStartedTransRV)
	{
		*pbStartedTransRV = FALSE;
		if( RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
		{
			goto Exit;
		}
	}

	// Add all of the records for pDictToDo - one per transaction.

	bHaveLFile = FALSE;
	uiRecordsPerTrans = 100;

Do_Dict_Recs:

	pDictRec = pDictToDo->pRecovRecs;
	bAddedAtLeastOne = FALSE;
	bFailedAtLeastOne = FALSE;
	uiRecordInTrans = 0;

	while (pDictRec)
	{
		if (!pDictRec->bAdded)
		{
			if( !uiRecordInTrans)
			{
				// Start an update transaction

				if( RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
				{
					goto Exit;
				}

				*pbStartedTransRV = TRUE;
				pFirstDictRecInTrans = pDictRec;
			}
			uiRecordInTrans++;

			// Don't optimize and keep pDictLFile as a local variable!

			if( RC_BAD( rc = fdictGetContainer( 
				pDb->pDict, FLM_DICT_CONTAINER, &pDictLFile)))
			{
				rc = FERR_OK;
				goto Exit;
			}
			bHaveLFile = TRUE;

			// Call the status callback to give the application a chance
			// to change the definition record

			if( fnStatusFunc)
			{
				if( RC_BAD( rc = (*fnStatusFunc)( FLM_REBUILD_ADD_DICT_REC_STATUS,
											(void *)pDictRec->pRec,
											(void *)0,
											pRebuildState->AppArg)))
				{
					goto Exit;
				}
			}

			// Add the dictionary record.
			// NOTE: We are deliberately ignoring return codes here - so we
			// can attempt to process as many records as possible.

			uiDictRecId = pDictRec->pRec->getID();

			if (RC_BAD( flmLFileDictUpdate( pDb, &pDictLFile,
											&uiDictRecId, pDictRec->pRec,
											NULL, FALSE, FALSE, NULL, TRUE)))
			{
				*pbStartedTransRV = FALSE;
				(void)flmAbortDbTrans( pDb);
				bFailedAtLeastOne = TRUE;
				uiRecordInTrans = 0;
			}
			else if( (uiRecordInTrans >= uiRecordsPerTrans) || !pDictRec->pNext)
			{
				*pbStartedTransRV = FALSE;
				if (RC_BAD( flmCommitDbTrans( pDb, 0, FALSE)))
				{
					bFailedAtLeastOne = TRUE;
				}
				else
				{
					// Set bAdded to TRUE on all dict recs in transaction.

					while( pFirstDictRecInTrans != pDictRec)
					{
						pFirstDictRecInTrans->bAdded = TRUE;
						pFirstDictRecInTrans = pFirstDictRecInTrans->pNext;
					}

					pDictRec->bAdded = bAddedAtLeastOne = TRUE;
					pRebuildState->CallbackData.uiRecsRecov += uiRecordInTrans;

					if( fnStatusFunc)
					{
						if (RC_BAD( rc = (*fnStatusFunc)( FLM_REBUILD_STATUS,
													(void *)pCallbackData,
													(void *)0,
													pRebuildState->AppArg)))
						{
							goto Exit;
						}
					}

					f_yieldCPU();
				}

				uiRecordInTrans = 0;
			}
		}

		pDictRec = pDictRec->pNext;
	}

	// Retry the add loop until either they all FAIL or they all succeed.
	// This is done to handle record template dependencies - or other
	// dependencies for that matter - that may not have been in the proper
	// order in the list.

	if( (bAddedAtLeastOne || uiRecordsPerTrans > 1) && bFailedAtLeastOne)
	{
		// We MUST do single record transactions from this point on
		// so the commit above will be called if the last pDictRec
		// was added on any prior pass.

		uiRecordsPerTrans = 1;
		goto Do_Dict_Recs;
	}

	// Log the records that failed

	pDictRec = pDictToDo->pRecovRecs;
	pSaveRec = pRebuildState->pRecord;

	while (pDictRec)
	{
		if (!pDictRec->bAdded)
		{
			RCODE	TempRc;

			pRebuildState->pRecord = pDictRec->pRec;
			if (RC_BAD( TempRc = bldReportReason( pRebuildState,
							FLM_DICT_REC_ADD_ERR,
							pDictRec->uiBlkAddress, pDictRec->uiElmOffset,
							pDictRec->pRec->getID(), 0xFFFF, 0)))
			{
				rc = TempRc;
			}
		}
		pDictRec = pDictRec->pNext;
	}
	pRebuildState->pRecord = pSaveRec;

Exit:

	bldFreeRecovDictInfo( pDictToDo);
	if ((RC_OK( rc)) && (!(*pbStartedTransRV)))
	{
		if (RC_OK( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			*pbStartedTransRV = TRUE;
		}
	}

	return( rc);
}
/****************************************************************************
Desc:	Extract create options from file header and log header pieces.
****************************************************************************/
void flmGetCreateOpts(
	FILE_HDR *		pFileHdr,
	FLMBYTE *		pucLogHdr,
	CREATE_OPTS *	pCreateOpts)
{
	f_memset( pCreateOpts, 0, sizeof( CREATE_OPTS));
	if (pFileHdr)
	{
		pCreateOpts->uiBlockSize = pFileHdr->uiBlockSize;
		pCreateOpts->uiVersionNum = pFileHdr->uiVersionNum;
		pCreateOpts->uiDefaultLanguage = pFileHdr->uiDefaultLanguage;
		pCreateOpts->uiAppMajorVer = pFileHdr->uiAppMajorVer;
		pCreateOpts->uiAppMinorVer = pFileHdr->uiAppMinorVer;
	}
	else
	{
		pCreateOpts->uiBlockSize = DEFAULT_BLKSIZ;
		pCreateOpts->uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
		pCreateOpts->uiDefaultLanguage = DEFAULT_LANG;

		// uiAppMajorVer and uiAppMinorVer are already zero.
	}

	if (pucLogHdr)
	{
		pCreateOpts->uiMinRflFileSize =
			(FLMUINT)FB2UD( &pucLogHdr [LOG_RFL_MIN_FILE_SIZE]);
		pCreateOpts->uiMaxRflFileSize =
			(FLMUINT)FB2UD( &pucLogHdr [LOG_RFL_MAX_FILE_SIZE]);
		pCreateOpts->bKeepRflFiles =
			(FLMBOOL)((pucLogHdr [LOG_KEEP_RFL_FILES]) ? TRUE : FALSE);
		pCreateOpts->bLogAbortedTransToRfl =
			(FLMBOOL)((pucLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL]) ? TRUE : FALSE);
	}
	else
	{
		pCreateOpts->uiMinRflFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
		pCreateOpts->uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
		pCreateOpts->bKeepRflFiles = DEFAULT_KEEP_RFL_FILES_FLAG;
		pCreateOpts->bLogAbortedTransToRfl = DEFAULT_LOG_ABORTED_TRANS_FLAG;
	}
}

/****************************************************************************
Desc: 	Rebuilds a damaged database.
Notes:	This routine performs the following actions:  1) A temporary database
		 	is created; 2) A copy of the source database is saved;  3) The source
			database is scanned.  Data records from all containers are extracted
			and stored in the temporary database.  4) When the rebuild is
			complete, the temporary database file is copied over the source
			database file.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbRebuild(
	const char *			pszSourceDbPath,
	const char *			pszSourceDataDir,
	const char *			pszDestDbPath,
	const char *			pszDestDataDir,
	const char *			pszDestRflDir,
	const char *			pszDictPath,
	CREATE_OPTS *			pCreateOpts,
	FLMUINT *				puiTotRecsRV,
	FLMUINT *				puiRecsRecovRV,
	STATUS_HOOK				fnStatusFunc,
	void *					pvStatusData)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = NULL;
	FFILE *					pFile;
	F_SuperFileHdl *		pSFileHdl = NULL;
	F_SuperFileClient *	pSFileClient = NULL;
	FLMBOOL					bFileLocked = FALSE;
	FLMBOOL					bWriteLocked = FALSE;
	REBUILD_STATE *		pRebuildState = NULL;
	HDR_INFO *				pHdrInfo;
	CREATE_OPTS *			pDefaultCreateOpts = NULL;
	FLMUINT					uiTransID;
	IF_LockObject *		pWriteLockObj = NULL;
	IF_LockObject *		pFileLockObj = NULL;
	FLMBOOL					bMutexLocked = FALSE;
	IF_FileHdl *			pCFileHdl = NULL;
	IF_FileHdl *			pLockFileHdl = NULL;
	eLockType				currLockType;
	FLMUINT					uiLockThreadId;
	FLMUINT					uiFileNumber;
	FLMUINT					uiDbVersion = 0;
	FLMBOOL					bUsedFFile = FALSE;
	FLMBOOL					bBadHeader = FALSE;
	F_SEM						hWaitSem = F_SEM_NULL;

	// Allocate a semaphore

	if( RC_BAD( rc = f_semCreate( &hWaitSem)))
	{
		goto Exit;
	}
	
	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// See if there is an FFILE structure for this file
	// May unlock and re-lock the global mutex.

	if( RC_BAD( rc = flmFindFile( pszSourceDbPath, pszSourceDataDir,
		&pFile)))
	{
		goto Exit;
	}

	// If we didn't find an FFILE structure, get an
	// exclusive lock on the file.

	if( !pFile)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Attempt to get an exclusive lock on the file.

		if( RC_BAD( rc = flmCreateLckFile( pszSourceDbPath, &pLockFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = flmCheckFFileState( pFile)))
		{
			goto Exit;
		}

		// The call to flmVerifyFileUse will wait if the file is in
		// the process of being opened by another thread.

		if (RC_BAD( rc = flmVerifyFileUse( gv_FlmSysData.hShareMutex, &pFile)))
		{
			goto Exit;
		}

		// Increment the use count on the FFILE so it will not
		// disappear while we are copying the file.

		if (++pFile->uiUseCount == 1)
		{
			flmUnlinkFileFromNUList( pFile);
		}
		bUsedFFile = TRUE;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// See if the thread already has a file lock.  If so, there
		// is no need to obtain another.  However, we also want to
		// make sure there is no write lock.  If there is,
		// we cannot do the rebuild right now.

		pFile->pFileLockObj->getLockInfo( 0, &currLockType, 
			&uiLockThreadId, NULL);

		if (currLockType == FLM_LOCK_EXCLUSIVE && uiLockThreadId == f_threadId())
		{

			// See if there is already a transaction going.

			pFile->pWriteLockObj->getLockInfo( 0, &currLockType, 
				&uiLockThreadId, NULL);
				
			if (currLockType == FLM_LOCK_EXCLUSIVE && 
				 uiLockThreadId == f_threadId())
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}
		}
		else
		{
			pFileLockObj = pFile->pFileLockObj;
			pFileLockObj->AddRef();
			if (RC_BAD( rc = pFileLockObj->lock( hWaitSem,
				TRUE, FLM_NO_TIMEOUT, 0)))
			{
				goto Exit;
			}
			bFileLocked = TRUE;
		}

		// Lock the write object to eliminate contention with
		// the checkpoint thread.

		pWriteLockObj = pFile->pWriteLockObj;
		pWriteLockObj->AddRef();

		// Only contention here is with the checkpoint thread.
		// Wait forever for the checkpoint thread to give
		// up the lock.
		
		if( RC_BAD( rc = pWriteLockObj->lock( hWaitSem, TRUE, FLM_NO_TIMEOUT, 0)))
		{
			goto Exit;
		}

		bWriteLocked = TRUE;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( REBUILD_STATE), &pRebuildState)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( HDR_INFO), &pRebuildState->pHdrInfo)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( STATE_INFO), &pRebuildState->pStateInfo)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( CREATE_OPTS), &pDefaultCreateOpts)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		LOG_HEADER_SIZE, &pRebuildState->pLogHdr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		MAX_KEY_SIZ, &pRebuildState->pKeyBuffer)))
	{
		goto Exit;
	}

	pRebuildState->AppArg = pvStatusData;
	pRebuildState->fnStatusFunc = fnStatusFunc;
	pHdrInfo = pRebuildState->pHdrInfo;
	
	// Open the database file for reading header information
	
	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( pszSourceDbPath, 
		gv_FlmSysData.uiFileOpenFlags, &pCFileHdl)))
	{
		goto Exit;
	}
	
	rc = flmGetHdrInfo( pCFileHdl, &pHdrInfo->FileHdr,
					&pHdrInfo->LogHdr, pRebuildState->pLogHdr);
					
	pCFileHdl->Release();
	pCFileHdl = NULL;

	if( RC_OK( rc))
	{
		if (!pCreateOpts)
		{
			flmGetCreateOpts( &pHdrInfo->FileHdr, pRebuildState->pLogHdr,
				pDefaultCreateOpts);
			pCreateOpts = pDefaultCreateOpts;
		}
		
		rc = FERR_OK;
		uiDbVersion = pHdrInfo->FileHdr.uiVersionNum;
		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
	}
	else if( rc == FERR_BLOCK_CHECKSUM || rc == FERR_INCOMPLETE_LOG ||
				rc == FERR_DATA_ERROR || 
				(rc == FERR_UNSUPPORTED_VERSION && pHdrInfo->FileHdr.uiVersionNum == 0))
	{
		if( rc == FERR_BLOCK_CHECKSUM || rc == FERR_DATA_ERROR)
		{
			bBadHeader = TRUE;
		}
		
		rc = FERR_OK;
		
		if (!pCreateOpts)
		{
			flmGetCreateOpts( &pHdrInfo->FileHdr,
										 pRebuildState->pLogHdr, pDefaultCreateOpts);
			pCreateOpts = pDefaultCreateOpts;
		}
		uiDbVersion = pHdrInfo->FileHdr.uiVersionNum;
		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
	}
	else if( rc == FERR_UNSUPPORTED_VERSION || rc == FERR_NEWER_FLAIM)
	{
		goto Exit;
	}
	else if( rc == FERR_NOT_FLAIM ||
				!VALID_BLOCK_SIZE( pHdrInfo->FileHdr.uiBlockSize))
	{
		FLMUINT	uiSaveBlockSize;
		FLMUINT	uiCalcBlockSize = 0;
		FLMBYTE	ucFileHdrBuf[ FLM_FILE_HEADER_SIZE];

		uiDbVersion = (FLMUINT)((rc != FERR_NOT_FLAIM)
										? pHdrInfo->FileHdr.uiVersionNum
										: FLM_CUR_FILE_FORMAT_VER_NUM);

		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
		if (!pCreateOpts)
		{
			if (rc != FERR_NOT_FLAIM)
			{
				flmGetCreateOpts( &pHdrInfo->FileHdr,
										 pRebuildState->pLogHdr, pDefaultCreateOpts);
			}
			else
			{
				flmGetCreateOpts( NULL, NULL, pDefaultCreateOpts);
			}

			// Set block size to zero, so we will always take the calculated
			// block size below.

			pDefaultCreateOpts->uiBlockSize = 0;
			pCreateOpts = pDefaultCreateOpts;
		}

		// Try to determine the correct block size

		if (RC_BAD( rc = bldDetermineBlkSize(
			pszSourceDbPath, pszSourceDataDir, uiDbVersion,
			pRebuildState->uiMaxFileSize, &uiCalcBlockSize,
			fnStatusFunc, &pRebuildState->CallbackData,
			pRebuildState->AppArg)))
		{
			goto Exit;
		}

		uiSaveBlockSize = pCreateOpts->uiBlockSize;
		pCreateOpts->uiBlockSize = uiCalcBlockSize;

		// Initialize pHdrInfo->FileHdr to useable values.

		flmInitFileHdrInfo( pCreateOpts, &pHdrInfo->FileHdr, ucFileHdrBuf);

		// Only use the passed-in block size (uiSaveBlockSize) if it
		// was non-zero.

		if (uiSaveBlockSize)
		{
			pCreateOpts->uiBlockSize = uiSaveBlockSize;
		}
	}
	else
	{
		goto Exit;
	}

	// Delete the destination database in case it already exists.

	if( RC_BAD( rc = FlmDbRemove( pszDestDbPath, pszDestDataDir,
								pszDestRflDir, TRUE)))
	{
		if( rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// If no block size has been specified or determined yet, use what we
	// read from the file header.

	if( !pCreateOpts->uiBlockSize)
	{
		pCreateOpts->uiBlockSize = pHdrInfo->FileHdr.uiBlockSize;
	}

	// Open the corrupted database

	if( (pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( (pSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pSFileClient->setup( 
		pszSourceDbPath, pszSourceDataDir, pHdrInfo->FileHdr.uiVersionNum)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pSFileHdl->setup( pSFileClient,
		gv_FlmSysData.pFileHdlCache, gv_FlmSysData.uiFileOpenFlags, 0)))
	{
		goto Exit;
	}

	pRebuildState->pSFileHdl = pSFileHdl;

	// Calculate the file size.

	pRebuildState->CallbackData.ui64DatabaseSize = 0;
	for (uiFileNumber = 1;;uiFileNumber++)
	{
		FLMUINT64	ui64TmpSize;

		if (RC_BAD( pSFileHdl->getFileSize( uiFileNumber, &ui64TmpSize)))
		{
			break;
		}
		
		pRebuildState->CallbackData.ui64DatabaseSize += ui64TmpSize;
	}

	// When creating the new file, set the transaction ID to one greater than it
	// is in the corrupt file.  However, don't let it get greater than about
	// 2 billion - want to leave room for 2 billion transactions in case they
	// were corrupted somehow in our old file.

	uiTransID = ((FLMUINT)FB2UD( &pRebuildState->pLogHdr[
								LOG_CURR_TRANS_ID]) + 1) & 0x7FFFFFFF;

	if (RC_BAD( rc = flmCreateNewFile( pszDestDbPath, pszDestDataDir,
			pszDestRflDir, pszDictPath, NULL, pCreateOpts, 
			uiTransID, (FDB * *)&pRebuildState->hDb, pRebuildState)))
	{
		goto Exit;
	}
	pDb = (FDB *)pRebuildState->hDb;

	// Rebuild the database

	if (RC_BAD( rc = flmDbRebuildFile( pRebuildState, bBadHeader)))
	{
		goto Exit;
	}

Exit:

	// Close the temporary database, if it is still open

	if (pDb)
	{
		FFILE *	pTmpFile;
		FFILE *	pTmpFile1;

		// Get the FFILE pointer for the temporary file before closing it.

		pTmpFile = pDb->pFile;

		(void)FlmDbClose( (HFDB *)&pDb);

		// Force temporary FFILE structure to be cleaned up, if it
		// isn't already gone.  The following code searches for the
		// temporary file in the not-used list.  If it finds it,
		// it will unlink it.

		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		
		pTmpFile1 = gv_FlmSysData.pLrnuFile;
		
		while (pTmpFile1)
		{
			if (pTmpFile1 == pTmpFile)
			{
				flmFreeFile( pTmpFile);
				break;
			}
			pTmpFile1 = pTmpFile1->pNextNUFile;
		}
	}

	if (bUsedFFile)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		
		if (!(--pFile->uiUseCount))
		{
			flmLinkFileToNUList( pFile);
		}
	}
	
	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if( bWriteLocked)
	{
		pWriteLockObj->unlock();
		bWriteLocked = FALSE;
	}

	if( bFileLocked)
	{
		pFileLockObj->unlock();
		bFileLocked = FALSE;
	}

	if( pSFileHdl)
	{
		pSFileHdl->Release();
	}
	
	if( pCFileHdl)
	{
		pCFileHdl->Release();
	}

	if( pWriteLockObj)
	{
		pWriteLockObj->Release();
		pWriteLockObj = NULL;
	}
	
	if( pFileLockObj)
	{
		pFileLockObj->Release();
		pFileLockObj = NULL;
	}

	if( pLockFileHdl)
	{
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	if( pDefaultCreateOpts)
	{
		f_free( &pDefaultCreateOpts);
	}

	if( pRebuildState)
	{
		if( puiTotRecsRV)
		{
			*puiTotRecsRV = pRebuildState->CallbackData.uiTotRecs;
		}

		if( puiRecsRecovRV)
		{
			*puiRecsRecovRV = pRebuildState->CallbackData.uiRecsRecov;
		}

		if( pRebuildState->pStateInfo && pRebuildState->pStateInfo->pRecord)
		{
			pRebuildState->pStateInfo->pRecord->Release();

		}

		if( pRebuildState->pRecord)
		{
			pRebuildState->pRecord->Release();
			pRebuildState->pRecord = NULL;
		}

		if( pRebuildState->pHdrInfo)
		{
			f_free( &pRebuildState->pHdrInfo);
		}

		if( pRebuildState->pStateInfo)
		{
			f_free( &pRebuildState->pStateInfo);
		}

		if( pRebuildState->pLogHdr)
		{
			f_free( &pRebuildState->pLogHdr);
		}

		if( pRebuildState->pKeyBuffer)
		{
			f_free( &pRebuildState->pKeyBuffer);
		}

		f_free( &pRebuildState);
	}
	else
	{
		if( puiTotRecsRV)
		{
			*puiTotRecsRV = 0;
		}

		if( puiRecsRecovRV)
		{
			*puiRecsRecovRV = 0;
		}
	}
	
	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine reads through a database and makes a best guess as to 
		the true block size of the database.
*****************************************************************************/
FSTATIC RCODE bldDetermineBlkSize(
	const char *			pszSourceDbPath,
	const char *			pszSourceDataDir,
	FLMUINT					uiDbVersion,
	FLMUINT					uiMaxFileSize,
	FLMUINT *				puiBlkSizeRV,
	STATUS_HOOK				fnStatusFunc,
	REBUILD_INFO *			pCallbackData,
	void *					AppArg)
{
	RCODE						rc = FERR_OK;
	FLMBYTE					ucBlkHeader [BH_OVHD];
	FLMUINT					uiBytesRead;
	FLMUINT					uiBlkAddress;
	FLMUINT					uiFileNumber = 0;
	FLMUINT					uiOffset = 0;
	FLMUINT					uiCount4K = 0;
	FLMUINT					uiCount8K = 0;
	FLMUINT64				ui64BytesDone = 0;
	F_SuperFileHdl *		pSFileHdl = NULL;
	F_SuperFileClient *	pSFileClient = NULL;

	// Open the corrupted database

	if( (pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( (pSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pSFileClient->setup(
		pszSourceDbPath, pszSourceDataDir, uiDbVersion)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pSFileHdl->setup( pSFileClient,  
		gv_FlmSysData.pFileHdlCache, gv_FlmSysData.uiFileOpenFlags, 0)))
	{
		goto Exit;
	}

	// Start from byte offset 0 in the first file

	pCallbackData->iDoingFlag = REBUILD_GET_BLK_SIZ;
	pCallbackData->bStartFlag = TRUE;
	
	for (;;)
	{
		if (uiOffset >= uiMaxFileSize || !uiFileNumber)
		{
			uiOffset = 0;
			uiFileNumber++;
		}

		if( (RC_OK( rc = pSFileHdl->readBlock( 
			FSBlkAddress( uiFileNumber, uiOffset), BH_OVHD, 
				ucBlkHeader, &uiBytesRead))) || rc == FERR_IO_END_OF_FILE)
		{
			if (RC_OK( rc))
			{
				ui64BytesDone += (FLMUINT64)MIN_BLOCK_SIZE;
			}
			else
			{
				ui64BytesDone += (FLMUINT64)uiBytesRead;
			}
			uiBlkAddress = GET_BH_ADDR( ucBlkHeader);
			if ((uiBytesRead == BH_OVHD) &&
				 (FSGetFileOffset( uiBlkAddress) == uiOffset))
			{
				if (uiOffset % 4096 == 0)
				{
					uiCount4K++;
				}
				
				if (uiOffset % 8192 == 0)
				{
					uiCount8K++;
				}
			}
			
			if (rc != FERR_OK || uiBytesRead < BH_OVHD)
			{

				// Even if the file is not full size, set offset to
				// the maximum file offset so we will attempt to go
				// to the next file at the top of this loop.  If that
				// fails, we will assume we truly are at EOF.

				uiOffset = uiMaxFileSize;
			}
			else
			{
				uiOffset += MIN_BLOCK_SIZE;
			}

			// Call the callback function to report copy progress

			if (fnStatusFunc != NULL)
			{
				pCallbackData->ui64BytesExamined = ui64BytesDone;
				if (RC_BAD( rc = (*fnStatusFunc)( FLM_REBUILD_STATUS,
											(void *)pCallbackData,
											(void *)0,
											AppArg)))
				{
					goto Exit;
				}
				
				pCallbackData->bStartFlag = FALSE;
			}

			f_yieldCPU();
		}
		else
		{
			if( rc == FERR_IO_PATH_NOT_FOUND)
			{
				rc = RC_SET( FERR_IO_END_OF_FILE);
				break;
			}
			
			goto Exit;
		}
	}
	
	if (rc == FERR_IO_END_OF_FILE)
	{
		rc = FERR_OK;
	}

	// If our count of 4K blocks is greater than 66% of the number
	// of 4K blocks that would fit in the database, we will use
	// a 4K block size.  Otherwise, we will use an 8K block size.

	if (uiCount4K > 
		(FLMUINT)(((ui64BytesDone / 
			(FLMUINT64)4096) * (FLMUINT64)66) / (FLMUINT64)100))
	{
		*puiBlkSizeRV = 4096;
	}
	else
	{
		*puiBlkSizeRV = 8192;
	}
	
Exit:

	if( pSFileHdl)
	{
		pSFileHdl->Release();
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}

	return( rc);
}
