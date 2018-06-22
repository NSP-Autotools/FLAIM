//-------------------------------------------------------------------------
// Desc:	Check database for corruptions.
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

FSTATIC RCODE chkGetDictInfo(
	DB_INFO *				pDbInfo);

FSTATIC RCODE chkVerifyBlkChain(
	DB_INFO *				pDbInfo,
	BLOCK_INFO *			pBlkInfo,
	eCorruptionLocale 	eLocale,
	FLMUINT					uiFirstBlkAddr,
	FLMUINT					uiBlkType,
	FLMUINT *				puiBlkCount,
	FLMBOOL *				pbStartOverRV);

FSTATIC RCODE chkVerifyLFHBlocks(
	DB_INFO * 				pDbInfo,
	FLMBOOL *				pbStartOverRV);

FSTATIC RCODE chkVerifyAvailList(
	DB_INFO *				pDbInfo,
	FLMBOOL *				pbStartOverRV);

FSTATIC RCODE chkVerifyTrackerCounts(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT					uiIndexNum,
	FLMUINT					uiKeyCount,
	FLMUINT					uiRefCount);

FSTATIC RCODE chkIsCountIndex(
	STATE_INFO *			pStateInfo,
	FLMUINT					uiIndexNum,
	FLMBOOL *				pbIsCountIndex);

FSTATIC RCODE chkResolveRSetMissingKey(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT					uiIxRefDrn);

FSTATIC RCODE chkVerifyDelNonUniqueRec(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT					uiIndex,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiRecDrn,
	FLMUINT *				puiRecContainerRV,
	FLMBOOL *				pbDelRecRV);

FSTATIC RCODE chkVerifyKeyExists(
	FDB *						pDb,
	LFILE *					pLFile,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiRefDrn,
	FLMBOOL *				pbFoundRV);

FSTATIC RCODE chkAddDelKeyRef(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT					uiIndexNum,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiDrn,
	FLMUINT					uiFlags);

FSTATIC RCODE chkGetKeySource(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT					uiIndex,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiDrn,
	FLMUINT *				puiRecordContainerRV,
	FLMBOOL *				pbKeyInRecRV,
	FLMBOOL *				pbKeyInIndexRV);

FSTATIC RCODE chkReportIxError(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	eCorruptionType		eCorruption,
	FLMUINT					uiErrIx,
	FLMUINT					uiErrDrn,
	FLMBYTE *				pucErrKey,
	FLMUINT					uiErrKeyLen,
	FLMBOOL *				pbFixErrRV);

FSTATIC RCODE chkGetNextRSKey(
	IX_CHK_INFO *			pIxChkInfo);

FSTATIC RCODE	chkResolveIXMissingKey(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo);

FSTATIC RCODE chkRSInit(
	const char *			pszIoPath,
	IF_ResultSet **		ppRSet);

FSTATIC RCODE chkRSFinalize(
	IX_CHK_INFO *			pIxChkInfo,
	FLMUINT64 *				pui64TotalEntries);

FSTATIC FLMINT chkCompareKeySet(
	FLMUINT					uiIxNum1,
	FLMBYTE *				pData1,
	FLMUINT					uiLength1,
	FLMUINT					uiDrn1,
	FLMUINT					uiIxNum2,
	FLMBYTE *				pData2,
	FLMUINT					uiLength2,
	FLMUINT					uiDrn2);

FSTATIC RCODE chkBlkRead(
	DB_INFO *				pDbInfo,
	FLMUINT					uiBlkAddress,
	LFILE *					pLFile,
	FLMBYTE **				ppBlk,
	SCACHE **				ppSCache,
	eCorruptionType *		peCorruption);

FSTATIC RCODE chkVerifyBTrees(
	DB_INFO *				pDbInfo,
	F_Pool *					pPool,
	FLMBOOL *				pbStartOverRV);

FSTATIC RCODE chkReportError(
	DB_INFO *				pDbInfo,
	eCorruptionType		eCorruption,
	eCorruptionLocale		eErrLocale,
	FLMUINT					uiErrLfNumber,
	FLMUINT					uiErrLfType,
	FLMUINT					uiErrBTreeLevel,
	FLMUINT					uiErrBlkAddress,
	FLMUINT					uiErrParentBlkAddress,
	FLMUINT					uiErrElmOffset,
	FLMUINT					uiErrDrn,
	FLMUINT					uiErrElmRecOffset,
	FLMUINT					uiErrFieldNum,
	FLMBYTE *				pBlk);

FSTATIC RCODE chkGetNextRSKey(
	IX_CHK_INFO * 			pIxChkInfo);

FSTATIC RCODE chkVerifyKeyNotUnique(
	STATE_INFO *			pStateInfo,
	FLMUINT					uiIndex,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT *				puiRefCountRV);

FSTATIC RCODE chkStartUpdate(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo);

FSTATIC RCODE chkEndUpdate(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo);

FSTATIC RCODE chkReadBlkFromDisk(
	FILE_HDR *				pFileHdr,
	F_SuperFileHdl *		pSFileHdl,
	FLMUINT					uiFilePos,
	FLMUINT					uiBlkAddress,
	LFILE *					pLFile,
	FFILE *					pFile,
	FLMBYTE *				pBlk);

FSTATIC RCODE chkVerifyElmFields(
	STATE_INFO *			pStateInfo,
	DB_INFO *				pDbInfo,
	IX_CHK_INFO *			pIxChkInfo,
	F_Pool *					pTmpPool,
	FLMUINT *				puiErrElmRecOffsetRV,
	eCorruptionType *		peElmErrCorruptCode);

FSTATIC RCODE chkVerifySubTree(
	DB_INFO *				pDbInfo,
	IX_CHK_INFO *			pIxChkInfo,
	STATE_INFO *			ParentState,
	STATE_INFO *			pStateInfo,
	FLMUINT					uiBlkAddress,
	F_Pool *					pTmpPool,
	FLMBYTE *				pucResetKey,
	FLMUINT					uiResetKeyLen,
	FLMUINT					uiResetDrn);

FSTATIC RCODE chkGetLfInfo(
	DB_INFO *				pDbInfo,
	F_Pool *					pPool,
	LF_STATS *				pLfStats,
	LFILE *					pLFile,
	LF_STATS *				pCurrLfStats,
	FLMBOOL *				pbCurrLfLevelChangedRV);

FSTATIC RCODE chkSetupLfTable(
	DB_INFO *				pDbInfo,
	F_Pool *					pPool);

FSTATIC RCODE chkSetupIxInfo(
	DB_INFO *				pDbInfo,
	IX_CHK_INFO *			pIxInfoRV);

FSTATIC RCODE chkOutputIndexKeys(
	STATE_INFO *			pStateInfo,
	IX_CHK_INFO *			pIxChkInfo,
	IXD *						pIxd,
	REC_KEY *				pKeyList);

/****************************************************************************
Desc:
****************************************************************************/
class F_ChkResultSetCompare : public IF_ResultSetCompare
{
public:

	RCODE FLMAPI compare(
		const void *			pvData1,
		FLMUINT					uiLength1,
		const void *			pvData2,
		FLMUINT					uiLength2,
		FLMINT *					piCompare)
	{
		FLMBYTE *		pucData1 = (FLMBYTE *) pvData1;
		FLMBYTE *		pucData2 = (FLMBYTE *) pvData2;
		FLMUINT			uiIxNum1;
		FLMUINT			uiIxNum2;
		FLMUINT			uiDrn1;
		FLMUINT			uiDrn2;
	
		uiIxNum1 = (FLMUINT) FB2UW( &(pucData1[RS_IX_OFFSET]));
		uiIxNum2 = (FLMUINT) FB2UW( &(pucData2[RS_IX_OFFSET]));
		uiDrn1 = (FLMUINT) FB2UD( &(pucData1[RS_REF_OFFSET]));
		uiDrn2 = (FLMUINT) FB2UD( &(pucData2[RS_REF_OFFSET]));
	
		*piCompare = chkCompareKeySet( 
						uiIxNum1, &(pucData1[RS_KEY_OFFSET]),
						uiLength1 - RS_KEY_OVERHEAD, uiDrn1, uiIxNum2,
						&(pucData2[RS_KEY_OFFSET]),
						uiLength2 - RS_KEY_OVERHEAD, uiDrn2);
	
		return( FERR_OK);
	}
};

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE chkCallProgFunc(
	DB_INFO *			pDbInfo)
{
	if( (pDbInfo->fnStatusFunc) && (RC_OK( pDbInfo->LastStatusRc)))
	{
		pDbInfo->LastStatusRc = (*pDbInfo->fnStatusFunc)( FLM_CHECK_STATUS,
												(void *)pDbInfo->pProgress,
												(void *)0,
												pDbInfo->pProgress->AppArg);
	}
	return( pDbInfo->LastStatusRc);
}

/****************************************************************************
Desc:
****************************************************************************/
class F_ChkSortStatus : public IF_ResultSetSortStatus
{
public:

	F_ChkSortStatus( IX_CHK_INFO * pIxChkInfo)
	{
		m_pIxChkInfo = pIxChkInfo;
	}
	
	RCODE FLMAPI reportSortStatus(
		FLMUINT64				ui64EstTotalUnits,
		FLMUINT64				ui64UnitsDone)
	{
		RCODE						rc = NE_FLM_OK;
		DB_CHECK_PROGRESS *	pProgress = m_pIxChkInfo->pDbInfo->pProgress;
	
		// Set the status values.
	
		pProgress->ui64NumRSUnits = ui64EstTotalUnits;
		pProgress->ui64NumRSUnitsDone = ui64UnitsDone;
	
		// Call the progress callback.
	
		if (RC_BAD( rc = chkCallProgFunc( m_pIxChkInfo->pDbInfo)))
		{
			goto Exit;
		}
	
	Exit:
	
		pProgress->bStartFlag = FALSE;
		return( rc);
	}
		
private:

	IX_CHK_INFO *			m_pIxChkInfo;
};
	
/****************************************************************************
Desc: This routine counts the number of fields in an object table.
****************************************************************************/
FINLINE void chkCountFields(
	FDICT *		pDict,
	FLMUINT *	puiNumFieldsRV)
{
	FLMUINT	uiTblSize = pDict->uiIttCnt;
	ITT *		pItt = pDict->pIttTbl;
	FLMUINT	uiCount = 0;
	FLMUINT	uiCurrObj;

	for (uiCurrObj = 0; uiCurrObj < uiTblSize; uiCurrObj++, pItt++)
	{
		if (ITT_IS_FIELD( pItt))
		{
			uiCount++;
		}
	}
	
	(*puiNumFieldsRV) += uiCount;
}

/****************************************************************************
Desc: Frees memory allocated to an IX_CHK_INFO structure
****************************************************************************/
FINLINE RCODE chkFreeIxInfo(
	IX_CHK_INFO *	pIxInfoRV)
{
	pIxInfoRV->pool.poolFree();
	pIxInfoRV->pRSet->Release();
	pIxInfoRV->pRSet = NULL;
	
	f_free( &(pIxInfoRV->puiIxArray));
	f_memset( pIxInfoRV, 0, sizeof(IX_CHK_INFO));

	return (FERR_OK);
}

/****************************************************************************
Desc:	Checks for physical corruption in a FLAIM database. Note: The
		routine verifies the database by first reading through the database to
		count certain block types which are in linked lists. It then verifies 
		the linked lists. It also verifies the B-TREEs in the database. The 
		reason for the first pass is so that when we verify the linked lists, 
		we can keep ourselves from getting into an infinite loop if there is 
		a loop in the lists.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbCheck(
	HFDB						hDb,
	const char *			pDbFileName,
	const char *			pszDataDir,
	const char *			pRflDir,
	FLMUINT					uiCheckFlags,
	F_Pool *					pPool,
	DB_CHECK_PROGRESS *	pCheckProgress,
	STATUS_HOOK				fnStatusFunc,
	void *					AppArg)
{
	RCODE					rc = FERR_OK;
	F_SuperFileHdl *	pSFileHdl = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiFileEnd;
	FLMUINT				uiBlockSize;
	DB_CHECK_PROGRESS Progress;
	FLMBOOL				bOpenedDb = FALSE;
	FDB *					pDb = (FDB *) hDb;
	FLMBOOL				bIgnore;
	FLMUINT				uiLoop;
	FLMUINT64			ui64TmpSize;
	FLMBOOL				bStartOver;
	FLMBOOL				bOkToCloseTrans = FALSE;
	DB_INFO *			pDbInfo;
	F_Pool				localPool;
	void *				pvDbInfoMark;

	if (hDb != HFDB_NULL && IsInCSMode( hDb))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto ExitCS;
	}

	localPool.poolInit( 512);

	if (!pPool)
	{
		pPool = &localPool;
	}

	if( RC_BAD( rc = pPool->poolCalloc( sizeof( DB_INFO), (void **)&pDbInfo)))
	{
		goto Exit;
	}

	pvDbInfoMark = pPool->poolMark();

	if (hDb == HFDB_NULL)
	{
		if (RC_BAD( rc = FlmDbOpen( pDbFileName, pszDataDir, pRflDir, 0, NULL,
					  &hDb)))
		{
			goto Exit;
		}

		bOpenedDb = TRUE;
		pDb = (FDB *) hDb;
	}

	pDbInfo->fnStatusFunc = fnStatusFunc;
	pDbInfo->LastStatusRc = FERR_OK;
	pDbInfo->pDb = pDb;

	if (pCheckProgress)
	{
		pDbInfo->pProgress = pCheckProgress;
	}
	else
	{
		pDbInfo->pProgress = &Progress;
	}

	f_memset( pDbInfo->pProgress, 0, sizeof(DB_CHECK_PROGRESS));

	pDbInfo->bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, 0, 0, &bIgnore)))
	{
		goto Exit;
	}

	// Initialize the information block and Progress structure. Because
	// FlmDbCheck will start and stop read transactions during its
	// processing we can't allow any existing read transactions to exist.
	// However, it is OK for an update transaction to be in progress. An
	// update transaction will NOT be stopped and restarted. The only
	// reason a read transaction may be stopped and restarted is if we get
	// an old view error - something that cannot normally happen during an
	// update transaction.

	if (flmGetDbTransType( pDb) == FLM_READ_TRANS)
	{

		// If it is an invisible transaction, it may be aborted.

		if (pDb->uiFlags & FDB_INVISIBLE_TRANS)
		{
			if (RC_BAD( rc = flmAbortDbTrans( pDb)))
			{
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( FERR_TRANS_ACTIVE);
			goto Exit;
		}
	}

	// Since we know that the check will start read transactions during
	// its processing, set the flag to indicate that the KRef table should
	// be cleaned up on exit if we are still in a read transaction.

	bOkToCloseTrans = TRUE;
	uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;

	// Check does its own reads using the handle to the file.

	pSFileHdl = pDb->pSFileHdl;
	pDbInfo->pSFileHdl = pSFileHdl;

	// Allocate memory to use for reading through the data blocks.

	if (RC_BAD( rc = f_alloc( uiBlockSize, &pBlk)))
	{
		goto Exit;
	}

Begin_Check:

	// Initialize all statistics in the DB_INFO structure.

	rc = FERR_OK;
	bStartOver = FALSE;

	pPool->poolReset( pvDbInfoMark);
	f_memset( pDbInfo->pProgress, 0, sizeof(DB_CHECK_PROGRESS));
	pDbInfo->pLogicalFiles = NULL;
	pDbInfo->uiFlags = uiCheckFlags;
	pDbInfo->bStartedUpdateTrans = FALSE;
	f_memset( &pDbInfo->FileHdr, 0, sizeof(FILE_HDR));
	pDbInfo->pProgress->AppArg = AppArg;

	// Get the dictionary information for the file

	if (RC_BAD( rc = chkGetDictInfo( pDbInfo)))
	{
		goto Exit;
	}

	for (uiLoop = 1;
		  uiLoop <= MAX_DATA_BLOCK_FILE_NUMBER( pDb->pFile->FileHdr.uiVersionNum);
		  uiLoop++)
	{
		if (RC_BAD( pSFileHdl->getFileSize( uiLoop, &ui64TmpSize)))
		{
			break;
		}

		pDbInfo->pProgress->ui64DatabaseSize += ui64TmpSize;
	}

	// See if we have a valid end of file

	uiFileEnd = pDbInfo->pDb->LogHdr.uiLogicalEOF;
	if (FSGetFileOffset( uiFileEnd) % uiBlockSize != 0)
	{
		if (RC_BAD( rc = chkReportError( pDbInfo, FLM_BAD_FILE_SIZE, LOCALE_NONE,
					  0, 0, 0xFF, uiFileEnd, 0, 0, 0, 0xFFFF, 0, NULL)))
		{
			goto Exit;
		}
	}
	else if (pDbInfo->pProgress->ui64DatabaseSize < FSGetSizeInBytes( pDbInfo->pDb->pFile->uiMaxFileSize,
																			uiFileEnd))
	{
		pDbInfo->pProgress->ui64DatabaseSize = FSGetSizeInBytes( pDbInfo->pDb->pFile->uiMaxFileSize,
																	uiFileEnd);
	}

	// Verify and count the LFH and PCODE blocks, B-Trees, and the AVAIL
	// list.

	if (RC_BAD( rc = chkVerifyLFHBlocks( pDbInfo, &bStartOver)))
	{
		goto Exit;
	}

	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Check the b-trees.

	if (RC_BAD( rc = chkVerifyBTrees( pDbInfo, pPool, &bStartOver)))
	{
		goto Exit;
	}

	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Check the avail list.

	if (RC_BAD( rc = chkVerifyAvailList( pDbInfo, &bStartOver)))
	{
		goto Exit;
	}

	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Signal that we are finished.

	pDbInfo->pProgress->iCheckPhase = CHECK_FINISHED;
	pDbInfo->pProgress->bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}

	pDbInfo->pProgress->bStartFlag = FALSE;

Exit:

	// Pass out any error code returned by the callback.

	if ((RC_OK( rc)) && (RC_BAD( pDbInfo->LastStatusRc)))
	{
		rc = pDbInfo->LastStatusRc;
	}

	if (pDb && pDbInfo->bDbInitialized)
	{

		// Close down the transaction, if one is going

		if (bOkToCloseTrans && flmGetDbTransType( pDb) == FLM_READ_TRANS)
		{
			KrefCntrlFree( pDb);
			(void) flmAbortDbTrans( pDb);
		}

		fdbExit( pDb);
	}

	// Free memory, if allocated

	if (pBlk)
	{
		f_free( &pBlk);
	}

	// Close the database we opened.

	if (bOpenedDb)
	{
		(void) FlmDbClose( &hDb);
	}

ExitCS:

	return (rc);
}

/****************************************************************************
Desc: This routine opens a file and reads its dictionary into memory.
****************************************************************************/
FSTATIC RCODE chkGetDictInfo(
	DB_INFO *	pDbInfo)
{
	RCODE 		rc = FERR_OK;
	FDB *			pDb = pDbInfo->pDb;

	// Close down the transaction, if one is going.

	if (flmGetDbTransType( pDb) != FLM_UPDATE_TRANS)
	{
		if (pDb->uiTransType == FLM_READ_TRANS)
		{
			(void) flmAbortDbTrans( pDb);
		}

		// Start a read transaction on the file to ensure we are connected
		// to the file's dictionary structures.

		if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_READ_TRANS, 0,
					  FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}

		f_memcpy( &pDbInfo->FileHdr, &pDb->pFile->FileHdr, sizeof(FILE_HDR));
		pDbInfo->pProgress->uiVersionNum = pDbInfo->FileHdr.uiVersionNum;
		pDbInfo->pProgress->uiBlockSize = pDbInfo->FileHdr.uiBlockSize;
		pDbInfo->pProgress->uiDefaultLanguage = pDbInfo->FileHdr.uiDefaultLanguage;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	This routine follows all of the blocks in a chain, verifying that
		they are properly linked. It also verifies each block's header.
****************************************************************************/
FSTATIC RCODE chkVerifyBlkChain(
	DB_INFO *			pDbInfo,
	BLOCK_INFO *		pBlkInfo,
	eCorruptionLocale eLocale,
	FLMUINT				uiFirstBlkAddr,
	FLMUINT				uiBlkType,
	FLMUINT *			puiBlkCount,
	FLMBOOL *			pbStartOverRV)
{
	RCODE					rc = FERR_OK;
	eCorruptionType	eCorruption = FLM_NO_CORRUPTION;
	SCACHE *				pSCache = NULL;
	FLMBYTE *				pBlk = NULL;
	FLMUINT				uiPrevBlkAddress;
	FLMUINT				uiBlkCount = 0;
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	FLMUINT64			ui64SaveBytesExamined;
	FDB *					pDb = pDbInfo->pDb;
	FILE_HDR *			pFileHdr = &pDb->pFile->FileHdr;
	FLMUINT				uiVersionNum = pFileHdr->uiVersionNum;
	FLMUINT				uiBlockSize = pFileHdr->uiBlockSize;
	FLMUINT				uiMaxBlocks;
	
	uiMaxBlocks = (FLMUINT)(FSGetSizeInBytes( pDb->pFile->uiMaxFileSize,
							pDb->LogHdr.uiLogicalEOF) / (FLMUINT64) uiBlockSize);

	uiPrevBlkAddress = BT_END;

	// There must be at least ONE block if it is the LFH chain.

	if ((uiBlkType == BHT_LFH_BLK) && (uiFirstBlkAddr == BT_END))
	{
		eCorruption = FLM_BAD_LFH_LIST_PTR;
		(void) chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF, 0, 0, 0,
									 0, 0xFFFF, 0, NULL);
		goto Exit;
	}

	// Read through all of the blocks, verifying them as we go.

Restart_Chain:

	uiBlkCount = 0;
	flmInitReadState( &StateInfo, &bStateInitialized, uiVersionNum, pDb, NULL,
						  (FLMUINT) ((uiBlkType == BHT_FREE) 
						  			? (FLMUINT) 0xFF 
									: (FLMUINT) 0), uiBlkType, NULL);
									
	ui64SaveBytesExamined = pDbInfo->pProgress->ui64BytesExamined;
	StateInfo.uiBlkAddress = uiFirstBlkAddr;
	
	while ((StateInfo.uiBlkAddress != BT_END) && (uiBlkCount < uiMaxBlocks))
	{
		StateInfo.pBlk = NULL;
		if (RC_BAD( rc = chkBlkRead( pDbInfo, StateInfo.uiBlkAddress,
					  StateInfo.pLogicalFile ? StateInfo.pLogicalFile->pLFile : NULL,
				  &pBlk, &pSCache, &eCorruption)))
		{
			if (rc == FERR_OLD_VIEW)
			{
				FLMUINT	uiSaveDictSeq = pDb->pDict->uiDictSeq;

				if (RC_BAD( rc = chkGetDictInfo( pDbInfo)))
				{
					goto Exit;
				}

				// If the dictionary ID changed, start over.

				if (pDb->pDict->uiDictSeq != uiSaveDictSeq)
				{
					*pbStartOverRV = TRUE;
					goto Exit;
				}

				pDbInfo->pProgress->ui64BytesExamined = ui64SaveBytesExamined;
				goto Restart_Chain;
			}

			pBlkInfo->eCorruption = eCorruption;
			pBlkInfo->uiNumErrors++;
			rc = chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF,
									  StateInfo.uiBlkAddress, 0, 0, 0, 0xFFFF, 0, pBlk);
		}

		StateInfo.pBlk = pBlk;
		uiBlkCount++;
		pDbInfo->pProgress->ui64BytesExamined += (FLMUINT64) uiBlockSize;
		if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
		{
			goto Exit;
		}

		f_yieldCPU();

		if (( eCorruption = flmVerifyBlockHeader( &StateInfo, pBlkInfo, 
			uiBlockSize, 0, (uiBlkType == BHT_FREE) ? 0L : uiPrevBlkAddress,
			TRUE, TRUE)) != FLM_NO_CORRUPTION)
		{
			pBlkInfo->eCorruption = eCorruption;
			pBlkInfo->uiNumErrors++;
			chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF,
								StateInfo.uiBlkAddress, 0, 0, 0, 0xFFFF, 0, pBlk);
			goto Exit;
		}

		uiPrevBlkAddress = StateInfo.uiBlkAddress;
		StateInfo.uiBlkAddress = (FLMUINT) FB2UD( &pBlk[BH_NEXT_BLK]);
	}

	if ((StateInfo.uiBlkAddress != BT_END) && (RC_OK( pDbInfo->LastStatusRc)))
	{
		switch (uiBlkType)
		{
			case BHT_LFH_BLK:
				eCorruption = FLM_BAD_LFH_LIST_END;
				break;
			case BHT_PCODE_BLK:
				eCorruption = FLM_BAD_PCODE_LIST_END;
				break;
			case BHT_FREE:
				eCorruption = FLM_BAD_AVAIL_LIST_END;
				break;
		}

		pBlkInfo->eCorruption = eCorruption;
		pBlkInfo->uiNumErrors++;
		chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF,
							uiPrevBlkAddress, 0, 0, 0, 0xFFFF, 0, pBlk);
		goto Exit;
	}

Exit:

	if (puiBlkCount)
	{
		*puiBlkCount = uiBlkCount;
	}

	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
	}

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlk)
	{
		f_free( &pBlk);
	}

	if (RC_OK( rc) && (eCorruption != FLM_NO_CORRUPTION))
	{
		rc = (uiBlkType == BHT_FREE) 
						? RC_SET( FERR_DATA_ERROR) 
						: RC_SET( FERR_DD_ERROR);
	}

	return (rc);
}

/****************************************************************************
Desc: This routine verifies the LFH blocks.
****************************************************************************/
FSTATIC RCODE chkVerifyLFHBlocks(
	DB_INFO *	pDbInfo,
	FLMBOOL *	pbStartOverRV)
{
	RCODE 		rc = FERR_OK;

	pDbInfo->pProgress->uiLfNumber = 0;
	pDbInfo->pProgress->uiLfType = 0;
	pDbInfo->pProgress->iCheckPhase = CHECK_LFH_BLOCKS;
	pDbInfo->pProgress->bStartFlag = TRUE;
	
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}

	pDbInfo->pProgress->bStartFlag = FALSE;

	// Go through the LFH blocks.

	if (RC_BAD( rc = chkVerifyBlkChain( pDbInfo, &pDbInfo->pProgress->LFHBlocks,
				  LOCALE_LFH_LIST, pDbInfo->pDb->pFile->FileHdr.uiFirstLFHBlkAddr,
				  BHT_LFH_BLK, NULL, pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	This routine reads through the blocks in the AVAIL list and
		verifies that we don't have a loop or some other corruption in the 
		list.
****************************************************************************/
FSTATIC RCODE chkVerifyAvailList(
	DB_INFO *	pDbInfo,
	FLMBOOL *	pbStartOverRV)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBlkCount;

	pDbInfo->pProgress->uiLfNumber = 0;
	pDbInfo->pProgress->uiLfType = 0;
	pDbInfo->pProgress->iCheckPhase = CHECK_AVAIL_BLOCKS;
	pDbInfo->pProgress->bStartFlag = TRUE;
	
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}

	pDbInfo->pProgress->bStartFlag = FALSE;

	f_yieldCPU();

	if (RC_BAD( rc = chkVerifyBlkChain( pDbInfo,
				  &pDbInfo->pProgress->AvailBlocks, LOCALE_AVAIL_LIST,
				  pDbInfo->pDb->LogHdr.uiFirstAvailBlkAddr, BHT_FREE, &uiBlkCount,
				  pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

	// See if the block count matches the block count stored in the log
	// header.

	if (uiBlkCount != pDbInfo->pDb->LogHdr.uiAvailBlkCount)
	{
		(void) chkReportError( pDbInfo, FLM_BAD_AVAIL_BLOCK_COUNT,
									 LOCALE_AVAIL_LIST, 0, 0, 0xFF, 0, 0, 0, 0, 0xFFFF,
									 0, NULL);
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Verifies the key and reference counts against the counts that are
		stored in the tracker record.
****************************************************************************/
FSTATIC RCODE chkVerifyTrackerCounts(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiIndexNum,
	FLMUINT				uiKeyCount,
	FLMUINT				uiRefCount)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = pStateInfo->pDb;
	FlmRecord *			pRecord = NULL;
	eCorruptionType	eCorruption;
	FLMUINT				uiTrackerKeyCount = 0;
	FLMUINT				uiTrackerRefCount = 0;
	void *				pvField;

	// Retrieve the tracker record from record cache.

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, FLM_TRACKER_CONTAINER,
				  uiIndexNum, TRUE, NULL, NULL, &pRecord)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		rc = FERR_OK;
	}
	else
	{
		if ((pvField = pRecord->find( pRecord->root(), FLM_KEY_TAG)) != NULL)
		{
			if (RC_BAD( rc = pRecord->getUINT( pvField, &uiTrackerKeyCount)))
			{
				goto Exit;
			}
		}

		if ((pvField = pRecord->find( pRecord->root(), FLM_REFS_TAG)) != NULL)
		{
			if (RC_BAD( rc = pRecord->getUINT( pvField, &uiTrackerRefCount)))
			{
				goto Exit;
			}
		}
	}

	// See if the counts match what we got from the tracker record.

	if (uiKeyCount != uiTrackerKeyCount || uiRefCount != uiTrackerRefCount)
	{

		// Log an error.

		eCorruption = (eCorruptionType) ((uiKeyCount != uiTrackerKeyCount) 
									? FLM_KEY_COUNT_MISMATCH 
									: FLM_REF_COUNT_MISMATCH);

		if (RC_BAD( rc = chkReportError( pIxChkInfo->pDbInfo, eCorruption,
					  LOCALE_INDEX, uiIndexNum, LF_INDEX, 0xFF, 0, 0, 0, 0, 0xFFFF,
					  0, NULL)))
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
Desc:	Determine if an index is an index that keeps key and reference
		counts.
****************************************************************************/
FSTATIC RCODE chkIsCountIndex(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiIndexNum,
	FLMBOOL *		pbIsCountIndex)
{
	RCODE 	rc = FERR_OK;
	FDB *		pDb = pStateInfo->pDb;
	IXD *		pIxd;

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				  uiIndexNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	*pbIsCountIndex = (FLMBOOL) ((pIxd->uiFlags & IXD_COUNT) 
									? (FLMBOOL) TRUE 
									: (FLMBOOL) FALSE);
									
Exit:

	return (rc);
}

/****************************************************************************
Desc: Verifies the current index key against the result set.
****************************************************************************/
RCODE chkVerifyIXRSet(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIxRefDrn)
{
	RCODE				rc = FERR_OK;
	FLMINT			iCmpVal = 0;
	FLMUINT			uiIteration = 0;
	FLMBOOL			bRSetEmpty = FALSE;
	RS_IX_KEY *		pCurrRSKey;
	RS_IX_KEY *		pPrevRSKey;

	if (!pIxChkInfo->pCurrRSKey)
	{
		pIxChkInfo->pCurrRSKey = &pIxChkInfo->IxKey1;
		pIxChkInfo->pPrevRSKey = &pIxChkInfo->IxKey2;
	}

	// Compare index and result set keys

	while (!bRSetEmpty)
	{
		if (pIxChkInfo->bGetNextRSKey)
		{

			// Get the next key from the result set. If the result set is
			// empty, then pIxChkInfo->uiRSKeyLen will be set to zero,
			// forcing the problem to be resolved below.

			if (RC_BAD( rc = chkGetNextRSKey( pIxChkInfo)))
			{
				if (rc == FERR_EOF_HIT || rc == FERR_NOT_FOUND)
				{

					// Set bRSetEmpty to TRUE so that the loop will exit after
					// the current key is resolved. Otherwise, conflict
					// resolution on the current key will be repeated forever
					// (infinite loop).

					bRSetEmpty = TRUE;
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{

				// Updated statistics

				pIxChkInfo->pDbInfo->pProgress->ui64NumKeysExamined++;
			}
		}

		pCurrRSKey = pIxChkInfo->pCurrRSKey;
		pPrevRSKey = pIxChkInfo->pPrevRSKey;

		if (pCurrRSKey->uiRSKeyLen == 0 || bRSetEmpty)
		{

			// We don't have a key because we got an EOF when reading the
			// result set. Need to resolve the fact that the result set does
			// not have a key that is found in the index. Set iCmpVal to 1 to
			// force this resolution.

			iCmpVal = 1;
		}
		else
		{

			// Compare the index key and result set key.

			iCmpVal = chkCompareKeySet( pCurrRSKey->uiRSIxNum,
												&(pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET]),
												pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD,
												pCurrRSKey->uiRSRefDrn,
												pStateInfo->pLogicalFile->pLFile->uiLfNum,
												pStateInfo->pCurKey,
												pStateInfo->uiCurKeyLen, uiIxRefDrn);
		}

		if (iCmpVal < 0)
		{

			// If a comparison is done where the keys from the result set
			// don't match what we got from the index, we will forego
			// verifying the tracker counts. Verifying of tracker counts can
			// only occur if we have an otherwise clean check of the index
			// keys.

			pIxChkInfo->bCheckCounts = FALSE;

			// The result set key is less than the index key. This could mean
			// that the result set key needs to be added to the index.

			if ((RC_BAD( rc = chkResolveIXMissingKey( pStateInfo, pIxChkInfo))) ||
				 (pIxChkInfo->pDbInfo->bReposition))
			{

				// If the key was added to the index (bReposition == TRUE) or
				// we got some other error, we don't want to get the next
				// result set key.

				pIxChkInfo->bGetNextRSKey = FALSE;
				goto Exit;
			}
			else
			{

				// False alarm. The index is missing the key because of a
				// concurrent update. We want to get the next RS key.

				pIxChkInfo->bGetNextRSKey = TRUE;
			}
		}
		else if (iCmpVal > 0)
		{

			// If a comparison is done where the keys from the result set
			// don't match what we got from the index, we will forego
			// verifying the tracker counts. Verifying of tracker counts can
			// only occur if we have an otherwise clean check of the index
			// keys.

			pIxChkInfo->bCheckCounts = FALSE;

			// The result set key is greater than the index key. This could
			// mean that the index key needs to be deleted from the index.
			// Whether we delete the index key or not, we don't need to get
			// the next result set key, but we do want to reposition and get
			// the next index key.

			pIxChkInfo->bGetNextRSKey = FALSE;
			if ((RC_BAD( rc = chkResolveRSetMissingKey( pStateInfo, 
				pIxChkInfo, uiIxRefDrn))) || pIxChkInfo->pDbInfo->bReposition)
			{
				goto Exit;
			}
			break;
		}
		else
		{

			// The index and result set keys are equal. We want to get the
			// next result set and index keys.

			pIxChkInfo->bGetNextRSKey = TRUE;

			// Determine if we have switched indexes. If so, verify the key
			// and reference counts against the counts in the tracker record.

			if (pCurrRSKey->uiRSIxNum != pPrevRSKey->uiRSIxNum)
			{
				if (pIxChkInfo->bCheckCounts)
				{

					// Verify the key and reference counts against tracker
					// record.

					if (RC_BAD( rc = chkVerifyTrackerCounts( pStateInfo, pIxChkInfo,
								  pPrevRSKey->uiRSIxNum, pIxChkInfo->uiRSIxKeyCount,
								  pIxChkInfo->uiRSIxRefCount)))
					{
						goto Exit;
					}
				}

				// Determine if the new index is one that supports counts.

				if (RC_BAD( rc = chkIsCountIndex( pStateInfo, pCurrRSKey->uiRSIxNum,
							  &pIxChkInfo->bCheckCounts)))
				{
					goto Exit;
				}

				if (pIxChkInfo->bCheckCounts)
				{

					// Set the counts to one.

					pIxChkInfo->uiRSIxKeyCount = 1;
					pIxChkInfo->uiRSIxRefCount = 1;
				}
			}
			else
			{
				if (pIxChkInfo->bCheckCounts)
				{

					// Always increment the reference count.

					pIxChkInfo->uiRSIxRefCount++;

					// See if the key changed.

					if (pCurrRSKey->uiRSKeyLen != pPrevRSKey->uiRSKeyLen ||
						 (pCurrRSKey->uiRSKeyLen > RS_KEY_OFFSET &&
						 f_memcmp( &pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET],
									 &pPrevRSKey->pucRSKeyBuf[RS_KEY_OFFSET],
									 pCurrRSKey->uiRSKeyLen - RS_KEY_OFFSET) != 0))
					{
						pIxChkInfo->uiRSIxKeyCount++;
					}
					else
					{

						// If the keys are the same, at least the DRNs better
						// be different.

						flmAssert( pCurrRSKey->uiRSRefDrn != pPrevRSKey->uiRSRefDrn);
					}
				}
			}
			break;
		}

		// Call the yield function periodically

		uiIteration++;
		if (!(uiIteration & 0x1F))
		{
			f_yieldCPU();
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Retrieves the next key from the sorted result set
****************************************************************************/
RCODE chkGetNextRSKey(
	IX_CHK_INFO *	pIxChkInfo)
{
	RCODE				rc = FERR_OK;
	RS_IX_KEY *		pCurrRSKey;

	// Swap current and last key pointers - this allows us to always keep
	// the last key without having to memcpy the keys.

	pCurrRSKey = pIxChkInfo->pCurrRSKey;
	pIxChkInfo->pCurrRSKey = pIxChkInfo->pPrevRSKey;
	pIxChkInfo->pPrevRSKey = pCurrRSKey;
	pCurrRSKey = pIxChkInfo->pCurrRSKey;

	// Get the next key

	if (RC_BAD( rc = pIxChkInfo->pRSet->getNext( pCurrRSKey->pucRSKeyBuf,
				  MAX_KEY_SIZ + RS_KEY_OVERHEAD, &pCurrRSKey->uiRSKeyLen)))
	{
		goto Exit;
	}

	// Verify that the key meets the minimum length requirements

	flmAssert( pCurrRSKey->uiRSKeyLen >= RS_KEY_OVERHEAD);

	// Extract the index number and reference DRN

	pCurrRSKey->uiRSIxNum = 
		(FLMUINT) FB2UW( &(pCurrRSKey->pucRSKeyBuf[RS_IX_OFFSET]));
		
	pCurrRSKey->uiRSRefDrn = 
		(FLMUINT) FB2UD( &(pCurrRSKey->pucRSKeyBuf[RS_REF_OFFSET]));

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Resolves the case of a key found in the result set but not in the
		current index.
****************************************************************************/
RCODE chkResolveIXMissingKey(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bKeyInRec;
	FLMBOOL				bKeyInIndex;
	FLMBOOL				bFixCorruption = FALSE;
	RS_IX_KEY *			pCurrRSKey = pIxChkInfo->pCurrRSKey;

	// Determine if the record generates the key and if the key is found in
	// the index.

	if (RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
				  pCurrRSKey->uiRSIxNum, &(pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET]),
				  (FLMUINT) (pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
				  pCurrRSKey->uiRSRefDrn, NULL, &bKeyInRec, &bKeyInIndex)))
	{
		if (rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	// If the record does not generate the key or the key+ref is in the
	// index, the index is not corrupt.

	if (!bKeyInRec || bKeyInIndex)
	{
		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
		goto Exit;
	}

	// Otherwise, the index is corrupt; Update statistics
	
	pIxChkInfo->pDbInfo->pProgress->ui64NumRecKeysNotFound++;
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

	// Report the error

	if (RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
				  FLM_DRN_NOT_IN_KEY_REFSET, pCurrRSKey->uiRSIxNum,
				  pCurrRSKey->uiRSRefDrn, &(pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET]),
				  (FLMUINT) (pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
				  &bFixCorruption)))
	{
		goto Exit;
	}

	// Exit if the application does not want to repair the corruption.

	if (!bFixCorruption)
	{

		// Set the logical corruption flag

		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		goto Exit;
	}

	// Fix the corruption; Update statistics
	
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;

	// Add the key

	if (RC_OK( rc = chkAddDelKeyRef( pStateInfo, pIxChkInfo,
				 pCurrRSKey->uiRSIxNum, &(pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET]),
				 (FLMUINT) (pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
				 pCurrRSKey->uiRSRefDrn, 0)))
	{
		pIxChkInfo->pDbInfo->bReposition = TRUE;
		goto Exit;
	}
	else
	{
		if (rc == FERR_NOT_UNIQUE)
		{

			// A subsequent record probably also generates this key, but the
			// index is a unique index so we were not allowed to add the
			// missing key + ref to the index. This record should probably be
			// deleted.

			if (RC_OK( rc = chkResolveNonUniqueKey( pStateInfo, pIxChkInfo,
						 pCurrRSKey->uiRSIxNum,
						 &(pCurrRSKey->pucRSKeyBuf[RS_KEY_OFFSET]),
						 (FLMUINT) (pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
						 pCurrRSKey->uiRSRefDrn)))
			{
				pIxChkInfo->pDbInfo->bReposition = TRUE;
				goto Exit;
			}
		}
		else
		{

			// Set the logical corruption flag

			pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Resolves the case of a key found in the current index but not in
		the result set.
****************************************************************************/
FSTATIC RCODE chkResolveRSetMissingKey(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiIxRefDrn)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bKeyInRec;
	FLMBOOL				bKeyInIndex;
	FLMBOOL				bFixCorruption = FALSE;

	// See if the key is found in the index and/or generated by the record.

	if (RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
				  pStateInfo->pLogicalFile->pLFile->uiLfNum, pStateInfo->pCurKey,
				  pStateInfo->uiCurKeyLen, uiIxRefDrn, NULL, &bKeyInRec,
				  &bKeyInIndex)))
	{
		if (rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	// If the key is generated by the record or the key is not found in the
	// index, the index is not corrupt.

	if (bKeyInRec || !bKeyInIndex)
	{
		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
		goto Exit;
	}

	// Otherwise, the index is corrupt; Update statistics
	
	pIxChkInfo->pDbInfo->pProgress->ui64NumKeysNotFound++;
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

	// Report the error

	if (RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
				  FLM_IX_KEY_NOT_FOUND_IN_REC,
				  pStateInfo->pLogicalFile->pLFile->uiLfNum, uiIxRefDrn,
				  pStateInfo->pCurKey, pStateInfo->uiCurKeyLen, &bFixCorruption)))
	{
		goto Exit;
	}

	// Exit if the application does not want to repair the corruption.

	if (!bFixCorruption)
	{

		// Set the logical corruption flag

		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		goto Exit;
	}

	// Fix the corruption; Update statistics
	
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;

	// Delete the reference from the index

	if (RC_OK( rc = chkAddDelKeyRef( pStateInfo, pIxChkInfo,
				 pStateInfo->pLogicalFile->pLFile->uiLfNum, pStateInfo->pCurKey,
				 pStateInfo->uiCurKeyLen, uiIxRefDrn, KREF_DELETE_FLAG)))
	{
		pIxChkInfo->pDbInfo->bReposition = TRUE;
	}
	else
	{

		// Set the logical corruption flag

		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Resolves the case of multiple references associated with a key in a
		unique index.
****************************************************************************/
RCODE chkResolveNonUniqueKey(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = pStateInfo->pDb;
	LFILE *			pRecLFile = NULL;
	FLMBOOL			bDeleteRec = FALSE;
	FLMUINT			uiRecContainer;
	RCODE				rc2 = FERR_OK;
	FLMBOOL			bFixCorruption = FALSE;
	FlmRecord *		pOldRecord = NULL;

	// Verify that the record violates the constraints of the unique index
	// and should be deleted.

	if (RC_BAD( rc = chkVerifyDelNonUniqueRec( pStateInfo, pIxChkInfo, uiIndex,
				  pucKey, uiKeyLen, uiDrn, &uiRecContainer, &bDeleteRec)))
	{
		goto Exit;
	}

	if (bDeleteRec)
	{

		// Update statistics

		pIxChkInfo->pDbInfo->pProgress->ui64NumNonUniqueKeys++;
		pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

		// Report the error

		if (RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
					  FLM_NON_UNIQUE_ELM_KEY_REF, uiIndex, uiDrn, pucKey, uiKeyLen,
					  &bFixCorruption)))
		{
			goto Exit;
		}

		if (!bFixCorruption)
		{

			// Set the logical corruption flag

			pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
			goto Exit;
		}

		// Delete the record that generated the non-unique reference. ;
		// Update statistics
		
		pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;

		// Start an update transaction, if necessary.

		if (RC_BAD( rc = chkStartUpdate( pStateInfo, pIxChkInfo)))
		{
			goto Exit;
		}

		// Re-verify that the record should be deleted.

		if (RC_BAD( rc = chkVerifyDelNonUniqueRec( pStateInfo, pIxChkInfo,
					  uiIndex, pucKey, uiKeyLen, uiDrn, &uiRecContainer, 
					  &bDeleteRec)))
		{
			goto Exit;
		}

		if (bDeleteRec == TRUE)
		{
			void *		pvMark;

			// Mark the temporary pool.

			pvMark = pDb->TempPool.poolMark();

			// Call the internal delete function, passing boolean flags
			// indicating that missing keys should not prevent the record
			// deletion and that the record validator callback should not be
			// called.

			if (RC_BAD( rc = fdictGetContainer( pDb->pDict, uiRecContainer,
						  &pRecLFile)))
			{
				goto Exit;
			}

			rc = flmDeleteRecord( pDb, pRecLFile, uiDrn, &pOldRecord, TRUE);

			if (gv_FlmSysData.UpdateEvents.pEventCBList)
			{
				flmUpdEventCallback( pDb, F_EVENT_DELETE_RECORD, (HFDB) pDb, rc,
										  uiDrn, uiRecContainer, NULL, pOldRecord);
			}

			// Reset the temporary pool. The flmDeleteRecord call
			// allocates space for the record that is being deleted.
			
			pDb->TempPool.poolReset( pvMark);

			if (RC_BAD( rc))
			{

				// If the record had already been deleted, continue the check
				// without reporting the error.

				if (rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;

					// Update statistics

					pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
				}
				else
				{

					// Set the logical corruption flag

					pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
				}

				goto Exit;
			}

			// Update statistics

			pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
		}
	}
	else
	{

		// Increment the conflict counter

		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
	}

Exit:

	// End the update. chkEndUpdate will be a no-op if an update
	// transaction was not started.

	rc2 = chkEndUpdate( pStateInfo, pIxChkInfo);

	if (pOldRecord)
	{
		pOldRecord->Release();
	}

	return ((RCODE) ((rc != FERR_OK) ? (RCODE) rc : (RCODE) rc2));
}

/****************************************************************************
Desc:	Verifies that the specified record should be deleted because it
		generates key(s) which violate the constraints of a unique index.
****************************************************************************/
FSTATIC RCODE chkVerifyDelNonUniqueRec(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRecDrn,
	FLMUINT *		puiRecContainerRV,
	FLMBOOL *		pbDelRecRV)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bKeyInRec;
	FLMBOOL			bRecRefdByKey;
	FLMUINT			uiRefCount;
	FLMUINT			uiRecContainer;

	*pbDelRecRV = FALSE;
	*puiRecContainerRV = 0;

	// See if the key is found in the index and/or generated by the record.

	if (RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo, uiIndex, pucKey,
				  uiKeyLen, uiRecDrn, &uiRecContainer, &bKeyInRec, &bRecRefdByKey)))
	{
		if (rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	*puiRecContainerRV = uiRecContainer;

	if (bKeyInRec == TRUE)
	{

		// Verify that the key is not unique

		if (RC_BAD( rc = chkVerifyKeyNotUnique( pStateInfo, uiIndex, pucKey,
					  uiKeyLen, &uiRefCount)))
		{
			goto Exit;
		}

		if (uiRefCount > 1)
		{

			// The unique index has multiple references for the specified
			// key. Since the current record generates a non-unique key, it
			// should be deleted even if it is not one of the records
			// referenced by the key. Of course, if it is already referenced
			// by the key, deleting the record will reduce the number of
			// references associated with the key by one.

			*pbDelRecRV = TRUE;
		}
		else if (uiRefCount == 1 && bRecRefdByKey == FALSE)
		{

			// The unique index already has a key corresponding to the key
			// being generated by the current record. However, the record is
			// not referenced from the unique index. The record should still
			// be deleted since it generates a non-unique key.

			*pbDelRecRV = TRUE;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Determines if a key is generated by the current record and/or if
		the key is found in the current index
****************************************************************************/
FSTATIC RCODE chkGetKeySource(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn,
	FLMUINT *		puiRecordContainerRV,
	FLMBOOL *		pbKeyInRecRV,
	FLMBOOL *		pbKeyInIndexRV)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRecord = NULL;
	FDB *				pDb = pStateInfo->pDb;
	LFILE *			pLFile;
	LFILE *			pIxLFile;
	REC_KEY *		pKeys = NULL;
	REC_KEY *		pTempKey = NULL;
	IXD *				pIxd;
	FLMBYTE			ucRecKeyBuf[MAX_KEY_SIZ];
	FLMUINT			uiRecKeyLen;
	FLMUINT			uiKeyCount;
	FLMBOOL			bResetKRef = FALSE;
	void *			pIxPoolMark;
	void *			pDbPoolMark;
	FLMUINT			uiContainerNum;

	// Initialize return values.

	*pbKeyInRecRV = FALSE;
	*pbKeyInIndexRV = FALSE;

	if (puiRecordContainerRV)
	{
		*puiRecordContainerRV = 0;
	}

	// Initialize variables

	pIxPoolMark = pIxChkInfo->pool.poolMark();

	// Need to mark the DB's temporary pool. The index code allocates
	// memory for new CDL entries from the DB pool. If the pool is not
	// reset, it grows during the check and becomes VERY large.

	pDbPoolMark = pDb->TempPool.poolMark();

	// Set up the KRef so that flmGetRecKeys will work

	if (RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	bResetKRef = TRUE;

	// Get the LFile and IXD of the index

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				  uiIndex, &pIxLFile, &pIxd)))
	{

		// Return FERR_INDEX_OFFLINE error.

		goto Exit;
	}

	if ((uiContainerNum = pIxd->uiContainerNum) == 0)
	{

		// Container number is always the last two bytes of the key.

		flmAssert( uiKeyLen > getIxContainerPartLen( pIxd));
		uiContainerNum = getContainerFromKey( pucKey, uiKeyLen);
	}

	// Get the LFile of the record that caused the error

	if (RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainerNum, &pLFile)))
	{
		goto Exit;
	}

	// Set the record container return value

	if (puiRecordContainerRV)
	{
		*puiRecordContainerRV = uiContainerNum;
	}

	// See if the key is in the index.

	if (RC_BAD( rc = chkVerifyKeyExists( pDb, pIxLFile, pucKey, uiKeyLen, uiDrn,
				  pbKeyInIndexRV)))
	{
		goto Exit;
	}

	// Read the record

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, pLFile->uiLfNum, uiDrn, FALSE,
				  NULL, NULL, &pRecord)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		// NOTE: Deliberately not bringing in to cache if not found there.

		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDrn, &pRecord, NULL, NULL)))
		{
			if (rc == FERR_NOT_FOUND)
			{
				*pbKeyInRecRV = FALSE;
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if (pRecord)
	{

		// Generate record keys

		if (RC_BAD( rc = flmGetRecKeys( pDb, pIxd, pRecord,
					  pRecord->getContainerID(), TRUE, &(pIxChkInfo->pool), &pKeys)))
		{
			goto Exit;
		}

		uiKeyCount = 0;
		pTempKey = pKeys;
		while (pTempKey != NULL)
		{

			// Build the collated keys for each key tree.

			if (RC_BAD( rc = KYTreeToKey( pDb, pIxd, pTempKey->pKey,
						  pTempKey->pKey->getContainerID(), ucRecKeyBuf, &uiRecKeyLen,
						  0)))
			{
				goto Exit;
			}

			if (KYKeyCompare( pucKey, uiKeyLen, ucRecKeyBuf,
									uiRecKeyLen) == BT_EQ_KEY)
			{
				*pbKeyInRecRV = TRUE;
				break;
			}

			pTempKey = pTempKey->pNextKey;
			uiKeyCount++;

			// Release the CPU periodically to prevent CPU hog problems.

			f_yieldCPU();
		}
	}

Exit:

	if (pKeys)
	{
		pTempKey = pKeys;
		while (pTempKey)
		{
			pTempKey->pKey->Release();
			pTempKey = pTempKey->pNextKey;
		}
	}

	if (pRecord)
	{
		pRecord->Release();
	}

	// Remove any keys added to the KRef

	if (bResetKRef)
	{
		KYAbortCurrentRecord( pDb);
	}

	// Reset the DB's temporary pool

	pDb->TempPool.poolReset( pDbPoolMark);

	// Reset the index check pool

	pIxChkInfo->pool.poolReset( pIxPoolMark);
	return (rc);
}

/****************************************************************************
Desc: Verify that a key is (or is not) found in an index.
****************************************************************************/
FSTATIC RCODE chkVerifyKeyExists(
	FDB *			pDb,
	LFILE *		pLFile,
	FLMBYTE *	pucKey,
	FLMUINT		uiKeyLen,
	FLMUINT		uiRefDrn,
	FLMBOOL *	pbFoundRV)
{
	RCODE			rc = FERR_OK;
	BTSK			stackBuf[BH_MAX_LEVELS];
	BTSK *		stack = stackBuf;
	FLMUINT		uiDinDomain = DIN_DOMAIN( uiRefDrn) + 1;
	FLMBYTE		ucBtKeyBuf[MAX_KEY_SIZ];
	DIN_STATE	dinState;
	FLMUINT		uiTmpDrn;

	*pbFoundRV = FALSE;
	f_memset( &dinState, 0, sizeof(DIN_STATE));

	// Initialize stack cache.

	FSInitStackCache( &(stackBuf[0]), BH_MAX_LEVELS);
	stack = stackBuf;
	stack->pKeyBuf = ucBtKeyBuf;

	// Search for the key.

	if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &stack, pucKey, uiKeyLen,
				  uiDinDomain)))
	{
		goto Exit;
	}

	if (stack->uiCmpStatus == BT_EQ_KEY)
	{
		uiTmpDrn = uiRefDrn;

		// Reading the current element, position to or after uiTmpDrn

		rc = FSRefSearch( stack, &dinState, &uiTmpDrn);

		// If the entry was not found, returns FERR_FAILURE

		if (rc == FERR_OK)
		{
			*pbFoundRV = TRUE;
		}
		else if (rc != FERR_FAILURE)
		{
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}

Exit:

	// Free the stack cache

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return (rc);
}

/****************************************************************************
Desc:	Compares a composite key (index, ref, key) for equality.
Note:	Since index references are sorted in decending order, a composite key
		with a lower ref DRN will sort after a key with a higher ref DRN.
****************************************************************************/
FLMINT chkCompareKeySet(
	FLMUINT		uiIxNum1,
	FLMBYTE *	pData1,
	FLMUINT		uiLength1,
	FLMUINT		uiDrn1,
	FLMUINT		uiIxNum2,
	FLMBYTE *	pData2,
	FLMUINT		uiLength2,
	FLMUINT		uiDrn2)
{
	FLMINT		iCmpVal = 0;
	FLMUINT		uiMinLen;

	// Compare index numbers

	if (uiIxNum1 > uiIxNum2)
	{
		iCmpVal = 1;
		goto Exit;
	}
	else if (uiIxNum1 < uiIxNum2)
	{
		iCmpVal = -1;
		goto Exit;
	}

	// Compare keys

	uiMinLen = (FLMUINT) (uiLength1 < uiLength2) ? uiLength1 : uiLength2;
	iCmpVal = f_memcmp( pData1, pData2, uiMinLen);
	if (iCmpVal == 0)
	{

		// Compare references

		if (uiLength1 == uiLength2)
		{

			// A key with a lower ref DRN will sort after a key with a higher
			// ref DRN.

			if (uiDrn1 > uiDrn2)
			{
				iCmpVal = -1;
			}
			else if (uiDrn1 < uiDrn2)
			{
				iCmpVal = 1;
			}
			else
			{
				iCmpVal = 0;
				goto Exit;
			}
		}
		else if (uiLength1 > uiLength2)
		{
			iCmpVal = 1;
		}
		else
		{
			iCmpVal = -1;
		}
	}
	else
	{
		iCmpVal = (FLMINT) ((iCmpVal > 0) ? (FLMINT) 1 : (FLMINT) -1);
	}

Exit:

	return (iCmpVal);
}

/****************************************************************************
Desc: This routine adds or deletes an index key and/or reference.
****************************************************************************/
FSTATIC RCODE chkAddDelKeyRef(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndexNum,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn,
	FLMUINT			uiFlags)
{
	RCODE				rc = FERR_OK;
	RCODE				rc2 = FERR_OK;
	FLMBYTE			ucKeyBuf[sizeof(KREF_ENTRY) + MAX_KEY_SIZ];
	KREF_ENTRY *	pKrefEntry = (KREF_ENTRY *) (&ucKeyBuf[0]);
	IXD *				pIxd;
	LFILE *			pLFile;
	FLMBOOL			bStartedUpdate = FALSE;
	FLMBOOL			bKeyInRec;
	FLMBOOL			bKeyInIndex;

	// Start an update transaction, if necessary

	if (RC_BAD( rc = chkStartUpdate( pStateInfo, pIxChkInfo)))
	{
		goto Exit;
	}

	bStartedUpdate = TRUE;

	// Look up the LFILE and IXD for the index.

	if (RC_BAD( rc = fdictGetIndex( pStateInfo->pDb->pDict,
			pStateInfo->pDb->pFile->bInLimitedMode, uiIndexNum, &pLFile, &pIxd)))
	{

		// Shouldn't get FERR_INDEX_OFFLINE in here.

		goto Exit;
	}

	// Verify that the state has not changed

	if (RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo, uiIndexNum, pucKey,
				  uiKeyLen, uiDrn, NULL, &bKeyInRec, &bKeyInIndex)))
	{
		goto Exit;
	}

	if ((bKeyInIndex == TRUE && ((uiFlags & KREF_DELETE_FLAG) != 0)) ||
		 (bKeyInIndex == FALSE && uiFlags == 0))
	{

		// Setup the KrefEntry structure

		flmAssert( uiIndexNum > 0 && uiIndexNum < FLM_UNREGISTERED_TAGS);
		f_memcpy( &(ucKeyBuf[sizeof(KREF_ENTRY)]), pucKey, uiKeyLen);
		pKrefEntry->ui16KeyLen = (FLMUINT16) uiKeyLen;
		pKrefEntry->ui16IxNum = (FLMUINT16) uiIndexNum;
		pKrefEntry->uiDrn = uiDrn;
		pKrefEntry->uiTrnsSeq = 1;
		pKrefEntry->uiFlags = uiFlags;

		if ((pIxd->uiFlags & IXD_UNIQUE) != 0)
		{

			// Do not allow duplicate keys to be added to a unique index.

			pKrefEntry->uiFlags |= KREF_UNIQUE_KEY;
		}

		// Add or delete the key/reference.

		if (RC_BAD( rc = FSRefUpdate( pStateInfo->pDb, pLFile, pKrefEntry)))
		{
			goto Exit;
		}

		// Update statistics

		pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
	}

Exit:

	// End the update.

	if (bStartedUpdate == TRUE)
	{
		if (RC_BAD( rc2 = chkEndUpdate( pStateInfo, pIxChkInfo)))
		{
			goto Exit;
		}
	}

	rc = (RCODE) ((rc != FERR_OK) ? (RCODE) rc : (RCODE) rc2);

	return (rc);
}

/****************************************************************************
Desc:	Populates the CORRUPT_INFO structure and calls the user's callback
		routine.
****************************************************************************/
FSTATIC RCODE chkReportIxError(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	eCorruptionType	eCorruption,
	FLMUINT				uiErrIx,
	FLMUINT				uiErrDrn,
	FLMBYTE *			pucErrKey,
	FLMUINT				uiErrKeyLen,
	FLMBOOL *			pbFixErrRV)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = pStateInfo->pDb;
	F_Pool *				pTmpPool;
	IXD *					pIxd;
	LFILE *				pLFile;
	void *				pIxPoolMark;
	void *				pDbPoolMark = NULL;
	FLMBOOL				bResetKRef = FALSE;
	CORRUPT_INFO		CorruptInfo;
	FLMUINT				uiContainerNum;

	f_memset( &CorruptInfo, 0, sizeof(CORRUPT_INFO));

	// Mark the index check pool

	pIxPoolMark = pIxChkInfo->pool.poolMark();
	pTmpPool = &(pIxChkInfo->pool);

	// Need to mark the DB's temporary pool. The index code allocates
	// memory for new CDL entries from the DB pool. If the pool is not
	// reset, it grows during the check and becomes VERY large.

	pDbPoolMark = pDb->TempPool.poolMark();

	// Set up the KRef so that flmGetRecKeys will work

	if (RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	bResetKRef = TRUE;

	// Report the error

	CorruptInfo.eErrLocale = LOCALE_INDEX;
	CorruptInfo.eCorruption = eCorruption;
	CorruptInfo.uiErrLfNumber = uiErrIx;
	CorruptInfo.uiErrDrn = uiErrDrn;
	CorruptInfo.uiErrElmOffset = pStateInfo->uiElmOffset;

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				  uiErrIx, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// Generate the key tree using the key that caused the error
	
	if (RC_BAD( rc = flmIxKeyOutput( pIxd, pucErrKey, uiErrKeyLen,
				  &(CorruptInfo.pErrIxKey), FALSE)))
	{
		goto Exit;
	}

	// Get the LFile

	if ((uiContainerNum = pIxd->uiContainerNum) == 0)
	{

		// Container number is always the last two bytes of the key.

		flmAssert( uiErrKeyLen > getIxContainerPartLen( pIxd));
		uiContainerNum = getContainerFromKey( pucErrKey, uiErrKeyLen);
	}

	// Get the LFile

	if (RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainerNum, &pLFile)))
	{
		goto Exit;
	}

	// Read the record

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, pLFile->uiLfNum, uiErrDrn,
				  FALSE, NULL, NULL, &(CorruptInfo.pErrRecord))))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Check_Error;
		}

		// NOTE: Deliberately not bringing in to cache if not found there.

		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiErrDrn,
					  &(CorruptInfo.pErrRecord), NULL, NULL)))
		{
Check_Error:

			// Record may have been deleted or cannot be returned because of
			// an old view error.

			if (rc == FERR_NOT_FOUND)
			{
				rc = FERR_OK;
			}
			else if (FlmErrorIsFileCorrupt( rc))
			{
				pIxChkInfo->pDbInfo->pProgress->bPhysicalCorrupt = TRUE;
				rc = FERR_OK;
				goto Exit;
			}
			else
			{
				goto Exit;
			}
		}
	}

	// Generate index keys for the current index and record

	if (CorruptInfo.pErrRecord != NULL)
	{
		if (RC_BAD( rc = flmGetRecKeys( pDb, pIxd, CorruptInfo.pErrRecord,
					  CorruptInfo.pErrRecord->getContainerID(), TRUE, pTmpPool,
					  &(CorruptInfo.pErrRecordKeyList))))
		{
			goto Exit;
		}
	}

	*pbFixErrRV = FALSE;
	if ((pIxChkInfo->pDbInfo->fnStatusFunc) &&
		 (RC_OK( pIxChkInfo->pDbInfo->LastStatusRc)))
	{
		pIxChkInfo->pDbInfo->LastStatusRc = 
			(*pIxChkInfo->pDbInfo->fnStatusFunc) ( FLM_PROBLEM_STATUS, 
				(void *) &CorruptInfo, (void *) pbFixErrRV, 
				pIxChkInfo->pDbInfo->pProgress->AppArg);
	}

Exit:

	if (CorruptInfo.pErrRecord)
	{
		CorruptInfo.pErrRecord->Release();
	}

	if (CorruptInfo.pErrIxKey)
	{
		CorruptInfo.pErrIxKey->Release();
	}

	if (CorruptInfo.pErrRecordKeyList)
	{
		REC_KEY *		pTempKey = CorruptInfo.pErrRecordKeyList;

		while (pTempKey)
		{
			pTempKey->pKey->Release();
			pTempKey = pTempKey->pNextKey;
		}
	}

	// Remove any keys added to the KRef

	if (bResetKRef)
	{
		KYAbortCurrentRecord( pDb);
	}

	// Reset the DB's temporary pool
	
	pDb->TempPool.poolReset( pDbPoolMark);

	// Reset the index check pool

	pIxChkInfo->pool.poolReset( pIxPoolMark);
	return (rc);
}

/****************************************************************************
Desc: This routine verifies that a key is not unique
****************************************************************************/
FSTATIC RCODE chkVerifyKeyNotUnique(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT *		puiRefCountRV)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = pStateInfo->pDb;
	FlmRecord *		pKeyTree = NULL;
	IXD *				pIxd;
	FLMUINT			uiRefDrn;

	*puiRefCountRV = 0;

	// Get the IXD

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				  uiIndex, NULL, &pIxd)))
	{
		goto Exit;
	}

	// This routine should not be called unless the index is a unique
	// index.

	flmAssert( ((pIxd->uiFlags & IXD_UNIQUE) != 0));

	// Generate the key tree from the collation key.

	if (RC_BAD( rc = flmIxKeyOutput( pIxd, pucKey, uiKeyLen, &pKeyTree, FALSE)))
	{
		goto Exit;
	}

	// Count up to the first two references for the key.

	if (RC_BAD( rc = FlmKeyRetrieve( (HFDB) pDb, uiIndex,
				  pKeyTree->getContainerID(), pKeyTree, 0, FO_EXACT, 
				  NULL, &uiRefDrn)))
	{

		// If the key is NOT found, the problem no longer exists.

		if ((rc == FERR_NOT_FOUND) ||
			 (rc == FERR_BOF_HIT) ||
			 (rc == FERR_EOF_HIT))
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	// Found at least one reference.

	*puiRefCountRV = 1;

	// Go exclusive of the last key/reference found to see if there are
	// more references for the key.

	if (RC_BAD( rc = FlmKeyRetrieve( (HFDB) pDb, uiIndex,
				  pKeyTree->getContainerID(), pKeyTree, uiRefDrn,
				  FO_KEY_EXACT | FO_EXCL, NULL, &uiRefDrn)))
	{
		if ((rc == FERR_NOT_FOUND) ||
			 (rc == FERR_BOF_HIT) ||
			 (rc == FERR_EOF_HIT))
		{
			rc = FERR_OK;
		}

		goto Exit;
	}

	// May be more references, but it is sufficient to know that there are
	// at least two.

	*puiRefCountRV = 2;
	
Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE chkStartUpdate(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo)
{
	RCODE				rc = FERR_OK;
	RCODE				rc2 = FERR_OK;
	FDB *				pDb = pStateInfo->pDb;
	FLMBOOL			bAbortedReadTrans = FALSE;

	if (flmGetDbTransType( pDb) == FLM_READ_TRANS)
	{

		// Free the KrefCntrl

		KrefCntrlFree( pDb);

		// Abort the read transaction

		if (RC_BAD( rc = flmAbortDbTrans( pDb)))
		{
			goto Exit;
		}

		bAbortedReadTrans = TRUE;

		// Try to start an update transaction

		if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS,
					  FLM_NO_TIMEOUT, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}

		pIxChkInfo->pDbInfo->bStartedUpdateTrans = TRUE;
	}

	if (RC_BAD( pIxChkInfo->pDbInfo->LastStatusRc))
	{
		rc = pIxChkInfo->pDbInfo->LastStatusRc;
		goto Exit;
	}

Exit:

	// If something went wrong after the update transaction was started,
	// abort the transaction.

	if (RC_BAD( rc))
	{
		if (pIxChkInfo->pDbInfo->bStartedUpdateTrans == TRUE)
		{
			(void) flmAbortDbTrans( pDb);
			pIxChkInfo->pDbInfo->bStartedUpdateTrans = FALSE;
		}
	}

	// Re-start the read transaction.

	if (bAbortedReadTrans == TRUE &&
		 pIxChkInfo->pDbInfo->bStartedUpdateTrans == FALSE)
	{
		rc2 = flmBeginDbTrans( pDb, FLM_READ_TRANS, 0, FLM_DONT_POISON_CACHE);
	}

	rc = (RCODE) ((rc != FERR_OK) ? (RCODE) rc : (RCODE) rc2);
	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE chkEndUpdate(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo)
{
	RCODE 			rc = FERR_OK;
	RCODE 			rc2 = FERR_OK;

	if (pIxChkInfo->pDbInfo->bStartedUpdateTrans == TRUE)
	{

		// Commit the update transaction that was started. If the
		// transaction started by the application, do not commit it.

		if (RC_BAD( rc = flmCommitDbTrans( pStateInfo->pDb, 0, FALSE)))
		{
			goto Exit;
		}

		pIxChkInfo->pDbInfo->bStartedUpdateTrans = FALSE;
	}

Exit:

	// Re-start read transaction

	if (flmGetDbTransType( pStateInfo->pDb) == FLM_NO_TRANS)
	{
		rc2 = flmBeginDbTrans( pStateInfo->pDb, FLM_READ_TRANS, 0,
									 FLM_DONT_POISON_CACHE);
	}

	rc = (RCODE) ((rc != FERR_OK) ? (RCODE) rc : (RCODE) rc2);
	return (rc);
}

/****************************************************************************
Desc:	Initializes a result set for use by the logical check code
****************************************************************************/
RCODE chkRSInit(
	const char *		pIoPath,
	IF_ResultSet **	ppRSet)
{
	RCODE								rc = FERR_OK;
	IF_ResultSet *					pRSet = NULL;
	F_ChkResultSetCompare *		pCompare = NULL;

	if( RC_BAD( rc = FlmAllocResultSet( &pRSet)))
	{
		goto Exit;
	}
	
	if( (pCompare = f_new F_ChkResultSetCompare) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pRSet->setupResultSet( pIoPath, pCompare, 0)))
	{
		goto Exit;
	}

	*ppRSet = pRSet;
	pRSet = NULL;

Exit:

	if( pRSet)
	{
		pRSet->Release();
	}
	
	if( pCompare)
	{
		pCompare->Release();
	}
	
	return (rc);
}

/****************************************************************************
Desc:	Sorts a result set.
****************************************************************************/
RCODE chkRSFinalize(
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT64 *		pui64TotalEntries)
{
	RCODE						rc = FERR_OK;
	DB_CHECK_PROGRESS *	pProgress = pIxChkInfo->pDbInfo->pProgress;
	DB_CHECK_PROGRESS		saveInfo;
	F_ChkSortStatus		chkSortStatus( pIxChkInfo);

	// Save the current check phase information.

	f_memcpy( &saveInfo, pProgress, sizeof(DB_CHECK_PROGRESS));

	// Set information for the result set sort phase.

	pProgress->iCheckPhase = CHECK_RS_SORT;
	pProgress->bStartFlag = TRUE;
	pProgress->ui64NumRSUnits = 0;
	pProgress->ui64NumRSUnitsDone = 0;

	if (RC_BAD( rc = pIxChkInfo->pRSet->finalizeResultSet( 
		&chkSortStatus, pui64TotalEntries)))
	{
		goto Exit;
	}

Exit:

	f_memcpy( pProgress, &saveInfo, sizeof( DB_CHECK_PROGRESS));
	pProgress->bStartFlag = TRUE;

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE chkBlkRead(
	DB_INFO *			pDbInfo,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FLMBYTE **			ppBlk,
	SCACHE **			ppSCache,
	eCorruptionType *	peCorruption)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = pDbInfo->pDb;
	FILE_HDR *	pFileHdr = &pDb->pFile->FileHdr;

	if (*ppSCache)
	{
		ScaReleaseCache( *ppSCache, FALSE);
		*ppSCache = NULL;
		*ppBlk = NULL;
	}
	else if (*ppBlk)
	{
		f_free( ppBlk);
		*ppBlk = NULL;
	}

	if (pDb->uiKilledTime)
	{
		rc = RC_SET( FERR_OLD_VIEW);
		goto Exit;
	}

	// Get the block from cache.

	if (RC_OK( rc = ScaGetBlock( pDb, pLFile, 0, uiBlkAddress, NULL, ppSCache)))
	{
		*ppBlk = (*ppSCache)->pucBlk;
	}
	else
	{

		// Try to read the block directly from disk.

		FLMUINT		uiBlkLen = pFileHdr->uiBlockSize;
		FLMUINT		uiTransID;
		FLMBYTE *	pucBlk;
		FLMUINT		uiLastReadTransID;
		FLMUINT		uiPrevBlkAddr;
		FLMUINT		uiFilePos;

		// If we didn't get a corruption error, jump to exit.

		if (!FlmErrorIsFileCorrupt( rc))
		{
			goto Exit;
		}

		// Allocate memory for the block.

		if (RC_BAD( rc = f_calloc( uiBlkLen, ppBlk)))
		{
			goto Exit;
		}

		pucBlk = *ppBlk;

		uiFilePos = uiBlkAddress;
		uiTransID = pDb->LogHdr.uiCurrTransID;
		uiLastReadTransID = 0xFFFFFFFF;

		// Follow version chain until we find version we need.

		for (;;)
		{
			if (RC_BAD( rc = chkReadBlkFromDisk( pFileHdr, pDbInfo->pSFileHdl,
						  uiFilePos, uiBlkAddress, pLFile, pDb->pFile, pucBlk)))
			{
				goto Exit;
			}

			// See if we can use the current version of the block, or if we
			// must go get a previous version.

			if ((FLMUINT) FB2UD( &pucBlk[BH_TRANS_ID]) <= uiTransID)
			{
				break;
			}

			// If the transaction ID is greater than or equal to the last
			// one we read, we have a corruption. This test will keep us from
			// looping around forever.

			if ((FLMUINT) FB2UD( &pucBlk[BH_TRANS_ID]) >= uiLastReadTransID)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				goto Exit;
			}

			uiLastReadTransID = (FLMUINT) FB2UD( &pucBlk[BH_TRANS_ID]);

			// Block is too new, go for next older version. If previous
			// block address is same as current file position or zero, we
			// have a problem.

			uiPrevBlkAddr = (FLMUINT) FB2UD( &pucBlk[BH_PREV_BLK_ADDR]);
			if ((uiPrevBlkAddr == uiFilePos) || (!uiPrevBlkAddr))
			{
				rc = (pDb->uiKilledTime) 
								? RC_SET( FERR_OLD_VIEW) 
								: RC_SET( FERR_DATA_ERROR);
				goto Exit;
			}

			uiFilePos = uiPrevBlkAddr;
		}

		// See if we even got the block we thought we wanted.

		if (GET_BH_ADDR( pucBlk) != uiBlkAddress)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
	}

Exit:

	*peCorruption = FLM_NO_CORRUPTION;
	if (RC_BAD( rc))
	{
		switch (rc)
		{
			case FERR_DATA_ERROR:
				*peCorruption = FLM_COULD_NOT_SYNC_BLK;
				break;
			case FERR_BLOCK_CHECKSUM:
				*peCorruption = FLM_BAD_BLK_CHECKSUM;
				break;
			default:
				break;
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE chkReadBlkFromDisk(
	FILE_HDR *			pFileHdr,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiFilePos,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FFILE *				pFile,
	FLMBYTE *			pBlk)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiBytesRead;
	FLMUINT	uiBlkLen = pFileHdr->uiBlockSize;

	if (RC_BAD( rc = pSFileHdl->readBlock( uiFilePos, uiBlkLen, pBlk,
				  &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = RC_SET( FERR_DATA_ERROR);
		}

		goto Exit;
	}

	if (uiBytesRead < uiBlkLen)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// Verify the block checksum BEFORE decrypting or using any data.

	if (RC_BAD( rc = BlkCheckSum( pBlk, CHECKSUM_CHECK, uiBlkAddress, uiBlkLen)))
	{
		goto Exit;
	}

	// If this is an index block it may be encrypted, we need to decrypt
	// it before we can use it. The function ScaDecryptBlock will check if
	// the index is encrypted first. If not, it will return.

	if (pLFile && pLFile->uiLfType == LF_INDEX)
	{
		if (RC_BAD( rc = ScaDecryptBlock( pFile, pBlk)))
		{
			goto Exit;
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE chkVerifyElmFields(
	STATE_INFO *		pStateInfo,
	DB_INFO *			pDbInfo,
	IX_CHK_INFO *		pIxChkInfo,
	F_Pool *				pTmpPool,
	FLMUINT *			puiErrElmRecOffsetRV,
	eCorruptionType *	peElmCorruptCode)
{
	FLMBYTE *			pValue = pStateInfo->pValue;
	FLMBYTE *			pData = pStateInfo->pData;
	FLMBYTE *			pTempValue;
	FlmRecord *			pRecord = pStateInfo->pRecord;
	FLMBOOL				bKRefAbortRec = FALSE;
	FLMUINT				uiSaveElmRecOffset = 0;
	void *				pDbPoolMark = NULL;
	void *				pKeyMark = NULL;
	FLMBOOL				bResetDbPool = FALSE;
	RCODE					rc = FERR_OK;
	void *				pvField = pStateInfo->pvField;
	REC_KEY *			pKeyList = NULL;
	REC_KEY *			pTmpKey;

	*peElmCorruptCode = FLM_NO_CORRUPTION;

	pTempValue = pValue 
						? (FLMBYTE *) &pValue[pStateInfo->uiFieldProcessedLen] 
						: NULL;

	while( (*peElmCorruptCode == FLM_NO_CORRUPTION) &&
			 (pStateInfo->uiElmRecOffset < pStateInfo->uiElmRecLen))
	{
		uiSaveElmRecOffset = pStateInfo->uiElmRecOffset;
		if ((*peElmCorruptCode = flmVerifyElmFOP( pStateInfo)) != FLM_NO_CORRUPTION)
		{
			break;
		}

		if (!pStateInfo->bElmRecOK)
		{
			pValue = pTempValue = NULL;
			if (pRecord)
			{
				pRecord->clear();
			}

			continue;
		}

		switch (pStateInfo->uiFOPType)
		{
			case FLM_FOP_CONT_DATA:
			{
				if ((pTempValue != NULL) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue, pStateInfo->pFOPData, pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;
			}
			
			case FLM_FOP_STANDARD:
			case FLM_FOP_OPEN:
			case FLM_FOP_TAGGED:
			case FLM_FOP_NO_VALUE:
			case FLM_FOP_LARGE:
			{
				if (pValue)
				{
					pValue = pTempValue = NULL;
				}

				if (pvField)
				{
					pvField = NULL;
				}

				if (!pRecord)
				{
					if ((pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}

				if (RC_BAD( rc = pRecord->insertLast( pStateInfo->uiFieldLevel,
							  pStateInfo->uiFieldNum, pStateInfo->uiFieldType, &pvField)))
				{
					goto Exit;
				}

				if (pStateInfo->uiFieldLen)
				{
					if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
								  pStateInfo->uiFieldType, pStateInfo->uiFieldLen, 0, 0, 0,
								  &pValue, NULL)))
					{
						goto Exit;
					}

					pTempValue = pValue;
				}

				if ((pTempValue) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue, pStateInfo->pFOPData, pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;
			}
			
			case FLM_FOP_ENCRYPTED:
			{
				if (pValue)
				{
					pValue = pTempValue = NULL;
				}

				if (pData)
				{
					pData = NULL;
				}

				if (pvField)
				{
					pvField = NULL;
				}

				if (!pRecord)
				{
					if ((pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}

				if (RC_BAD( rc = pRecord->insertLast( pStateInfo->uiFieldLevel,
							  pStateInfo->uiFieldNum, pStateInfo->uiFieldType, &pvField
							  )))
				{
					goto Exit;
				}

				if (pStateInfo->uiFieldLen)
				{
					if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
								  pStateInfo->uiFieldType, pStateInfo->uiFieldLen,
								  pStateInfo->uiEncFieldLen, pStateInfo->uiEncId,
								  FLD_HAVE_ENCRYPTED_DATA, &pData, &pValue)))
					{
						goto Exit;
					}

					pTempValue = pValue;
				}

				if ((pTempValue) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue, pStateInfo->pFOPData,
								pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;
			}

			case FLM_FOP_JUMP_LEVEL:
			{
				break;
			}

			default:
			{
				*peElmCorruptCode = FLM_BAD_FOP;
			}
		}

		if ((!pStateInfo->uiEncId && 
			  pStateInfo->uiFieldProcessedLen == pStateInfo->uiFieldLen) ||
			 (pStateInfo->uiEncId && 
			  pStateInfo->uiFieldProcessedLen == pStateInfo->uiEncFieldLen))
		{

			// The whole field has been retrieved. Verify the field and graft
			// it into the record being built.

			if (pValue && (pDbInfo->uiFlags & FLM_CHK_FIELDS))
			{
				if (pStateInfo->uiFieldType == 0xFF)
				{

					// Hit Rec Info object - don't care what's in it - must
					// not assert, because this would kill our ability to check
					// older versions of the database which have REC_INFO data
					// in them.

					*peElmCorruptCode = FLM_NO_CORRUPTION;
				}
				else
				{
					if (!pStateInfo->uiEncId)
					{
						*peElmCorruptCode = flmVerifyField( pStateInfo, pValue,
																	  pStateInfo->uiFieldLen,
																	  pStateInfo->uiFieldType);
					}
					else
					{

						// Decrypt the field and store the decrypted data.

						if (!pStateInfo->pDb->pFile->bInLimitedMode)
						{
							if (RC_BAD( rc = flmDecryptField( pStateInfo->pDb->pDict,
									pRecord, pvField, pStateInfo->uiEncId, pTmpPool)))
							{
								goto Exit;
							}

							*peElmCorruptCode = flmVerifyField( pStateInfo, pData,
									pStateInfo->uiFieldLen, pStateInfo->uiFieldType);
						}
						else
						{

							// If we can't decrypt the field, then just pass it
							// for now.

							*peElmCorruptCode = FLM_NO_CORRUPTION;
						}
					}
				}
			}
			else
			{
				*peElmCorruptCode = FLM_NO_CORRUPTION;
			}

			pValue = pTempValue = NULL;
		}

		// If this is the last element of the record, verify the record's
		// keys.

		if (BBE_IS_LAST( pStateInfo->pElm) &&
			 (pStateInfo->uiElmRecOffset == pStateInfo->uiElmRecLen))
		{
			pValue = pTempValue = NULL;

			if (!pDbInfo->pProgress->bPhysicalCorrupt &&
				 (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING))
			{
				FLMUINT	uiLoop;
				IXD *		pIxd;

				if (pStateInfo->pLogicalFile->pLFile->uiLfType == LF_CONTAINER)
				{

					// Need to mark the DB's temporary pool. The index code
					// allocates memory for new CDL entries from the DB pool.
					// If the pool is not reset, it grows during the check and
					// becomes VERY large.

					pDbPoolMark = pDbInfo->pDb->TempPool.poolMark();
					bResetDbPool = TRUE;

					// Set up the KRef table so that flmGetRecKeys will work
					// correctly.

					if (RC_BAD( rc = KrefCntrlCheck( pStateInfo->pDb)))
					{
						goto Exit;
					}

					bKRefAbortRec = TRUE;

					for (uiLoop = 0; uiLoop < pIxChkInfo->uiIxCount; uiLoop++)
					{
						if (RC_BAD( rc = fdictGetIndex( pStateInfo->pDb->pDict,
									  pStateInfo->pDb->pFile->bInLimitedMode,
									  pIxChkInfo->puiIxArray[uiLoop], NULL, &pIxd, TRUE)))
						{
							goto Exit;
						}

						if (pIxd->uiFlags & IXD_OFFLINE)
						{
							continue;
						}

						if (pIxd->uiContainerNum == pStateInfo->pLogicalFile->pLFile->uiLfNum ||
								!pIxd->uiContainerNum)
						{

							// Mark the field pool so that it can be reset after
							// the record keys have been generated and output.

							pKeyMark = pTmpPool->poolMark();

							// Build the record keys for the current index. Do
							// not remove duplicate keys. The result set will
							// remove any duplicates.

							if (RC_BAD( rc = flmGetRecKeys( pStateInfo->pDb, pIxd,
									pRecord, pStateInfo->pLogicalFile->pLFile->uiLfNum, 
									FALSE, pTmpPool, &pKeyList)))
							{
								goto Exit;
							}

							// If the record generated keys for the current
							// index, output the keys to the result set.

							if (pKeyList)
							{
								if (RC_BAD( rc = chkOutputIndexKeys( pStateInfo,
											  pIxChkInfo, pIxd, pKeyList)))
								{
									goto Exit;
								}

								pTmpKey = pKeyList;
								while (pTmpKey)
								{
									pTmpKey->pKey->Release();
									pTmpKey->pKey = NULL;
									pTmpKey = pTmpKey->pNextKey;
								}

								pKeyList = NULL;
							}

							// Reset the field pool

							pTmpPool->poolReset( pKeyMark);
						}
					}

					// Clean up any keys that may have been added to the KRef
					// table.

					KYAbortCurrentRecord( pStateInfo->pDb);
					bKRefAbortRec = FALSE;

					// Reset the DB's temporary pool

					pDbInfo->pDb->TempPool.poolReset( pDbPoolMark);
					bResetDbPool = FALSE;
				}
			}

			if (pRecord)
			{
				pRecord->clear();
			}

			pValue = pTempValue = NULL;
			pTmpPool->poolReset( NULL);
		}

		if (*peElmCorruptCode != FLM_NO_CORRUPTION)
		{
			pStateInfo->bElmRecOK = FALSE;
		}
	}

Exit:

	// Clean up any keys that may have been added to the KRef table. This
	// is a fail-safe case to clean up the KRef in case KYKeysCommit didn't
	// get called above.

	if (bKRefAbortRec)
	{
		KYAbortCurrentRecord( pStateInfo->pDb);
	}

	// Free any keys in the key list

	if (pKeyList)
	{
		pTmpKey = pKeyList;
		while (pTmpKey)
		{
			pTmpKey->pKey->Release();
			pTmpKey->pKey = NULL;
			pTmpKey = pTmpKey->pNextKey;
		}
	}

	// Reset the DB's temporary pool

	if (bResetDbPool)
	{
		pDbInfo->pDb->TempPool.poolReset( pDbPoolMark);
	}

	if (*peElmCorruptCode != FLM_NO_CORRUPTION || RC_BAD( rc))
	{
		pValue = pTempValue = NULL;
		pTmpPool->poolReset( NULL);
		if (pRecord)
		{
			pRecord->clear();
		}
	}

	pStateInfo->pValue = pValue;
	pStateInfo->pData = pData;
	pStateInfo->pvField = pvField;
	pStateInfo->pRecord = pRecord;

	if (*peElmCorruptCode != FLM_NO_CORRUPTION)
	{
		*puiErrElmRecOffsetRV = uiSaveElmRecOffset;
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine checks all of the blocks/links in a sub-tree of a
		B-TREE. It calls itself recursively whenever it descends a level in the
		tree.
****************************************************************************/
FSTATIC RCODE chkVerifySubTree(
	DB_INFO *			pDbInfo,
	IX_CHK_INFO *		pIxChkInfo,
	STATE_INFO *		pParentState,
	STATE_INFO *		pStateInfo,
	FLMUINT				uiBlkAddress,
	F_Pool *				pTmpPool,
	FLMBYTE *			pucResetKey,
	FLMUINT				uiResetKeyLen,
	FLMUINT				uiResetDrn)
{
	RCODE					rc = FERR_OK;
	SCACHE *				pSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiLevel = pStateInfo->uiLevel;
	FLMUINT				uiBlkType = pStateInfo->uiBlkType;
	FLMUINT				uiLfType = pStateInfo->pLogicalFile->pLFile->uiLfType;
	FLMUINT				uiBlockSize = pDbInfo->pDb->pFile->FileHdr.uiBlockSize;
	FLMUINT				uiParentBlkAddress;
	FLMBYTE *			pChildBlkAddr;
	FLMUINT				uiChildBlkAddress;
	FLMUINT				uiPrevNextBlkAddress;
	eCorruptionType	eElmCorruptCode;
	eCorruptionType	eBlkCorruptionCode = FLM_NO_CORRUPTION;
	eCorruptionType	eLastCorruptCode = FLM_NO_CORRUPTION;
	FLMUINT				uiNumErrors = 0;
	FLMUINT				uiErrElmRecOffset = 0;
	FLMUINT64			ui64SaveKeyCount = 0;
	FLMUINT64			ui64SaveKeyRefs = 0;
	BLOCK_INFO			SaveBlkInfo;
	BLOCK_INFO *		pBlkInfo;
	FLMBOOL				bProcessElm;
	FLMBOOL				bCountElm;
	FLMBOOL				bDescendToChildBlocks;
	FLMINT				iCompareStatus;
	eCorruptionType	eHdrCorruptCode;

	// Setup the state information.

	pStateInfo->pBlk = NULL;
	pStateInfo->uiBlkAddress = uiBlkAddress;
	uiPrevNextBlkAddress = pStateInfo->uiNextBlkAddr;
	uiParentBlkAddress = (pParentState) ? pParentState->uiBlkAddress : BT_END;

	f_yieldCPU();

	// Read the sub-tree root block into memory.

	bDescendToChildBlocks = TRUE;
	if (RC_BAD( rc = chkBlkRead( pDbInfo, uiBlkAddress,
				  pStateInfo->pLogicalFile ? pStateInfo->pLogicalFile->pLFile : NULL,
			  &pBlk, &pSCache, &eBlkCorruptionCode)))
	{
		if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
		{
			uiNumErrors++;
			eLastCorruptCode = eBlkCorruptionCode;
			chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_B_TREE,
								pDbInfo->pProgress->uiLfNumber,
								pDbInfo->pProgress->uiLfType, uiLevel, uiBlkAddress,
								uiParentBlkAddress, 0, 0, 0xFFFF, 0, pBlk);
			if (eBlkCorruptionCode == FLM_BAD_BLK_CHECKSUM)
			{
				bDescendToChildBlocks = FALSE;

				// Allow to continue the check, but if this is a non-leaf
				// block a non-zero eBlkCorruptionCode will prevent us from
				// descending to child blocks. Set rc to SUCCESS so we won't
				// goto Exit below.

				rc = FERR_OK;
			}
			else if (eBlkCorruptionCode == FLM_COULD_NOT_SYNC_BLK)
			{
				eLastCorruptCode = eBlkCorruptionCode;

				// Need the goto here, because rc is changed to SUCCESS, and
				// the goto below would get skipped.

				rc = FERR_OK;
				goto fix_state;
			}
		}
		else if (rc == FERR_OLD_VIEW)
		{
			pDbInfo->bReposition = TRUE;
		}

		// If rc was not changed to SUCCESS above, goto Exit.

		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	pStateInfo->pBlk = pBlk;

	// Verify the block header; Don't re-count the block if we are resetting.
	
	if (!uiResetKeyLen)
	{
		pDbInfo->pProgress->ui64BytesExamined += (FLMUINT64) uiBlockSize;
		pBlkInfo = &pStateInfo->BlkInfo;
	}
	else
	{
		pBlkInfo = NULL;
	}

	chkCallProgFunc( pDbInfo);

	// Check the block header.

	if (( eHdrCorruptCode = flmVerifyBlockHeader( pStateInfo, pBlkInfo,
					uiBlockSize, (pParentState == NULL) ? BT_END : 0,
					(pParentState == NULL) ? BT_END : pParentState->uiLastChildAddr,
					TRUE, TRUE)) == FLM_NO_CORRUPTION)
	{

		// Verify the previous block's next block address -- it should equal
		// the current block's address.

		if ((uiPrevNextBlkAddress) && (uiPrevNextBlkAddress != uiBlkAddress))
		{
			eHdrCorruptCode = FLM_BAD_PREV_BLK_NEXT;
		}
	}

	if (eHdrCorruptCode != FLM_NO_CORRUPTION)
	{
		eLastCorruptCode = eHdrCorruptCode;
		uiNumErrors++;
		chkReportError( pDbInfo, eHdrCorruptCode, LOCALE_B_TREE,
							pDbInfo->pProgress->uiLfNumber,
							pDbInfo->pProgress->uiLfType, uiLevel, uiBlkAddress,
							uiParentBlkAddress, 0, 0, 0xFFFF, 0, pBlk);
	}

	// Go through the elements in the block.

	pStateInfo->uiElmOffset = BH_OVHD;
	while ((pStateInfo->uiElmOffset < pStateInfo->uiEndOfBlock) &&
			 (RC_OK( pDbInfo->LastStatusRc)))
	{

		// If we are resetting, save any statistical information so we can
		// back it out if we need to.

		if (uiResetKeyLen)
		{
			ui64SaveKeyCount = pStateInfo->ui64KeyCount;
			ui64SaveKeyRefs = pStateInfo->ui64KeyRefs;
			f_memcpy( &SaveBlkInfo, &pStateInfo->BlkInfo, sizeof(BLOCK_INFO));
			bCountElm = FALSE;
			bProcessElm = FALSE;
		}
		else
		{
			bCountElm = TRUE;
			bProcessElm = TRUE;
		}

		pStateInfo->BlkInfo.ui64ElementCount++;

		if ((eElmCorruptCode = flmVerifyElement( 
			pStateInfo, pDbInfo->uiFlags)) != FLM_NO_CORRUPTION)
		{

			// Report any errors in the element.

			eLastCorruptCode = eElmCorruptCode;
			uiNumErrors++;
			if (RC_BAD( rc = chkReportError( pDbInfo, eElmCorruptCode,
						  LOCALE_B_TREE, pDbInfo->pProgress->uiLfNumber,
						  pDbInfo->pProgress->uiLfType, uiLevel, uiBlkAddress,
						  uiParentBlkAddress, pStateInfo->uiElmOffset,
						  pStateInfo->uiElmDrn, 0xFFFF, 0, pBlk)))
			{
				break;
			}
		}

		// Keep track of the number of continuation elements.

		if ((uiBlkType == BHT_LEAF) &&
			 (!BBE_IS_FIRST( pStateInfo->pElm)) &&
			 (pStateInfo->uiElmLen != BBE_LEM_LEN))
		{
			pStateInfo->BlkInfo.ui64ContElementCount++;
			pStateInfo->BlkInfo.ui64ContElmBytes += pStateInfo->uiElmLen;
		}

		// Do some further checking.

		if (eElmCorruptCode != FLM_NO_CORRUPTION)
		{
			pStateInfo->bElmRecOK = FALSE;
		}
		else
		{

			// See if we are resetting

			iCompareStatus = 0;
			if (uiResetKeyLen && pStateInfo->bValidKey && 
				 ((!pStateInfo->uiCurKeyLen) || 
					((iCompareStatus = flmCompareKeys( pStateInfo->pCurKey, 
								pStateInfo->uiCurKeyLen, pucResetKey, 
								uiResetKeyLen)) >= 0)))
			{
				if (iCompareStatus > 0)
				{
					if (uiBlkType == BHT_LEAF)
					{
						uiResetKeyLen = 0;
						pucResetKey = NULL;
						uiResetDrn = 0;
						bCountElm = TRUE;
					}

					bProcessElm = TRUE;
				}
				else if (uiLfType == LF_INDEX)
				{
					FLMBYTE *		pTmpElm = pStateInfo->pElm;
					FLMUINT			uiLowestDrn = FSGetDomain( &pTmpElm,
																	  pStateInfo->uiElmOvhd);

					if (uiResetDrn >= uiLowestDrn)
					{
						bProcessElm = TRUE;
						bCountElm = TRUE;
					}
				}
				else
				{

					// Processing a container

					bProcessElm = TRUE;
				}
			}

			if (uiBlkType == BHT_LEAF)
			{

				// No need to parse LEM element.

				if ((pStateInfo->uiCurKeyLen != 0) && (pStateInfo->bValidKey))
				{
					if (uiLfType == LF_CONTAINER)
					{
						if (pStateInfo->uiElmDrn != DRN_LAST_MARKER)
						{
							if (RC_BAD( rc = chkVerifyElmFields( pStateInfo, pDbInfo,
										  pIxChkInfo, pTmpPool, &uiErrElmRecOffset,
										  &eElmCorruptCode)))
							{
								goto Exit;
							}
						}
					}
					else if (bProcessElm)
					{
						uiErrElmRecOffset = 0xFFFF;
						if (!pDbInfo->pProgress->bPhysicalCorrupt &&
							 (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING))
						{
							if ((RC_BAD( rc = flmVerifyIXRefs( pStateInfo,
								pIxChkInfo, uiResetDrn, &eElmCorruptCode))) ||
								 pDbInfo->bReposition)
							{
								goto Exit;
							}
						}
						else
						{
							if ((RC_BAD( rc = flmVerifyIXRefs( pStateInfo, NULL,
											  uiResetDrn, &eElmCorruptCode))) ||
								 pDbInfo->bReposition)
							{
								goto Exit;
							}
						}
					}
				}

				if (bProcessElm)
				{
					uiResetKeyLen = 0;
					pucResetKey = NULL;
					uiResetDrn = 0;

					if (eElmCorruptCode != FLM_NO_CORRUPTION)
					{

						// Report any errors in the element.

						eLastCorruptCode = eElmCorruptCode;
						uiNumErrors++;
						chkReportError( pDbInfo, eElmCorruptCode, LOCALE_B_TREE,
											pDbInfo->pProgress->uiLfNumber,
											pDbInfo->pProgress->uiLfType, uiLevel,
											uiBlkAddress, uiParentBlkAddress,
											pStateInfo->uiElmOffset, pStateInfo->uiElmDrn,
											uiErrElmRecOffset, pStateInfo->uiFieldNum, pBlk);

						if (RC_BAD( pDbInfo->LastStatusRc))
						{
							break;
						}
					}
				}
			}
			else
			{
				if (uiBlkType == BHT_NON_LEAF_DATA)
				{
					pChildBlkAddr = &pStateInfo->pElm[BNE_DATA_CHILD_BLOCK];
				}
				else
				{
					pChildBlkAddr = &pStateInfo->pElm[BNE_CHILD_BLOCK];
				}

				uiChildBlkAddress = (FLMUINT) FB2UD( pChildBlkAddr);

				// Check the child sub-tree -- NOTE, this is a recursive call.
				// First see if we have a pucResetKey that we want to position
				// to. If so, make sure we are positioned to it before
				// descending to the child block.

				if (bProcessElm)
				{
					if (!bDescendToChildBlocks)
					{
						rc = FERR_OK;
					}
					else
					{
						rc = chkVerifySubTree( pDbInfo, pIxChkInfo, pStateInfo,
													 (pStateInfo - 1), uiChildBlkAddress,
													 pTmpPool, pucResetKey, uiResetKeyLen,
													 uiResetDrn);
					}

					if ((RC_BAD( rc)) ||
						 (RC_BAD( pDbInfo->LastStatusRc)) ||
						 (pDbInfo->bReposition))
					{
						goto Exit;
					}

					// Once we reach the key, set it to an empty to key so that
					// we will always descend to the child block after this
					// point.

					uiResetKeyLen = 0;
					pucResetKey = NULL;
					uiResetDrn = 0;
				}

				// Save the child block address in the level information.

				pStateInfo->uiLastChildAddr = uiChildBlkAddress;
			}
		}

		// If we were resetting on this element, restore the statistics to
		// what they were before.

		if (!bCountElm)
		{
			pStateInfo->ui64KeyCount = ui64SaveKeyCount;
			pStateInfo->ui64KeyRefs = ui64SaveKeyRefs;
			f_memcpy( &pStateInfo->BlkInfo, &SaveBlkInfo, sizeof(BLOCK_INFO));
		}

		// Go to the next element.

		pStateInfo->uiElmOffset += pStateInfo->uiElmLen;
	}

	// Verify that we ended exactly on the end of the block.

	if ((eLastCorruptCode == FLM_NO_CORRUPTION) &&
		 (pStateInfo->uiEndOfBlock >= BH_OVHD) &&
		 (pStateInfo->uiEndOfBlock <= uiBlockSize) &&
		 (pStateInfo->uiElmOffset > pStateInfo->uiEndOfBlock))
	{
		eLastCorruptCode = FLM_BAD_ELM_END;
		uiNumErrors++;
		chkReportError( pDbInfo, eLastCorruptCode, LOCALE_B_TREE,
							pDbInfo->pProgress->uiLfNumber,
							pDbInfo->pProgress->uiLfType, uiLevel, uiBlkAddress,
							uiParentBlkAddress, pStateInfo->uiElmOffset, 0, 0xFFFF, 0,
							pBlk);
	}

	// Verify that the last key in the block matches the parent's key.

	if ((eLastCorruptCode == FLM_NO_CORRUPTION) &&
		 (pParentState) &&
		 (RC_OK( pDbInfo->LastStatusRc)))
	{
		if (pStateInfo->bValidKey &&
			 pParentState->bValidKey &&
			 (flmCompareKeys( pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
					pParentState->pCurKey, pParentState->uiCurKeyLen) != 0))
		{
			eLastCorruptCode = FLM_BAD_PARENT_KEY;
			uiNumErrors++;
			chkReportError( pDbInfo, eLastCorruptCode, LOCALE_B_TREE,
								pDbInfo->pProgress->uiLfNumber,
								pDbInfo->pProgress->uiLfType, uiLevel, uiBlkAddress,
								uiParentBlkAddress, 0, 0, 0xFFFF, 0, pBlk);
		}
	}

fix_state:

	// If the block could not be verified, set the level's next block
	// address and last child address to zero to indicate that we really
	// aren't sure we're at the right place in this level in the B-TREE.

	if (eLastCorruptCode != FLM_NO_CORRUPTION)
	{
		pStateInfo->BlkInfo.eCorruption = eLastCorruptCode;
		pStateInfo->BlkInfo.uiNumErrors += uiNumErrors;

		// Reset all child block states.

		for (;;)
		{
			pStateInfo->uiNextBlkAddr = 0;
			pStateInfo->uiLastChildAddr = 0;
			pStateInfo->bValidKey = FALSE;
			pStateInfo->uiElmLastFlag = 0xFF;

			// Quit when the leaf level has been reached.

			if (pStateInfo->uiLevel == 0)
			{
				break;
			}

			pStateInfo--;
		}
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlk)
	{
		f_free( &pBlk);
	}

	pStateInfo->pBlk = NULL;
	return (rc);
}

/****************************************************************************
Desc:	This routine reads the LFH areas from disk to make sure they are up
		to date in memory.
****************************************************************************/
FSTATIC RCODE chkGetLfInfo(
	DB_INFO *	pDbInfo,
	F_Pool *		pPool,
	LF_STATS *	pLfStats,
	LFILE *		pLFile,
	LF_STATS *	pCurrLfStats,
	FLMBOOL *	pbCurrLfLevelChangedRV)
{
	RCODE					rc = FERR_OK;
	SCACHE *				pSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiSaveLevel;
	eCorruptionType	eBlkCorruptionCode;

	// Read in the block containing the logical file header.

	if (RC_BAD( rc = chkBlkRead( pDbInfo, pLFile->uiBlkAddress, pLFile, &pBlk,
				  &pSCache, &eBlkCorruptionCode)))
	{

		// Log the error.

		if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
		{
			chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_LFH_LIST, 0, 0,
								0xFF, pLFile->uiBlkAddress, 0, 0, 0, 0xFFFF, 0, pBlk);
		}

		goto Exit;
	}

	// Copy the LFH from the block to the LFILE.

	uiSaveLevel = pLfStats->uiNumLevels;
	if (RC_BAD( rc = flmBufferToLFile( &pBlk[pLFile->uiOffsetInBlk], pLFile,
				  pLFile->uiBlkAddress, pLFile->uiOffsetInBlk)))
	{
		goto Exit;
	}

	// Read root block to get the number of levels in the B-TREE

	if (pLFile->uiRootBlk == BT_END)
	{
		pLfStats->uiNumLevels = 0;
	}
	else
	{
		if (RC_BAD( rc = chkBlkRead( pDbInfo, pLFile->uiRootBlk, pLFile, &pBlk,
					  &pSCache, &eBlkCorruptionCode)))
		{
			if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
			{
				chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_B_TREE,
									pLFile->uiLfNum, pLFile->uiLfType, 0xFF,
									pLFile->uiRootBlk, 0, 0, 0, 0xFFFF, 0, pBlk);
			}

			goto Exit;
		}

		pLfStats->uiNumLevels = (FLMUINT) (pBlk[BH_LEVEL]) + 1;

		// Make sure that the level extracted from the block is valid.

		if (pBlk[BH_LEVEL] >= BH_MAX_LEVELS)
		{
			chkReportError( pDbInfo, FLM_BAD_BLK_HDR_LEVEL, LOCALE_B_TREE,
								pLFile->uiLfNum, pLFile->uiLfType,
								(FLMUINT) (pBlk[BH_LEVEL]), pLFile->uiRootBlk, 0, 0, 0,
								0xFFFF, 0, pBlk);

			// Force pLfStats->uiNumLevels to 1 so that we don't crash

			pLfStats->uiNumLevels = 1;
		}
	}

	// If the number of levels changed, reset the level information on this
	// logical file.

	if (uiSaveLevel != pLfStats->uiNumLevels && pLfStats->uiNumLevels)
	{
		if (pLfStats->uiNumLevels > uiSaveLevel)
		{
			if( RC_BAD( rc = pPool->poolCalloc( 
				(FLMUINT) (sizeof(LEVEL_INFO) * pLfStats->uiNumLevels),
				(void **)&pLfStats->pLevelInfo)))
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}

		if (pCurrLfStats == pLfStats)
		{
			*pbCurrLfLevelChangedRV = TRUE;
		}
	}

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if (pBlk)
	{
		f_free( &pBlk);
	}

	return (rc);
}

/****************************************************************************
Desc:	This routine allocates and initializes the LF table (array of
		LF_HDR structures).
****************************************************************************/
FSTATIC RCODE chkSetupLfTable(
	DB_INFO *	pDbInfo,
	F_Pool *		pPool)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCnt;
	FLMUINT		uiNumIndexes = 0;
	FLMUINT		uiIxStart;
	FLMUINT		uiNumDataCont = 0;
	FLMUINT		uiNumDictCont = 0;
	FLMUINT		uiIxOffset;
	FLMUINT		uiDataOffset;
	FLMUINT		uiDictOffset;
	FDB *			pDb = pDbInfo->pDb;
	LFILE *		pLFile;
	LFILE *		pTmpLFile;
	LF_HDR *		pLogicalFile;
	LF_STATS *	pLfStats;

	// Set up the table such that the dictionary is checked first, followed
	// by data containers, and then indexes. This is necessary for the
	// logical (index) check to work. The data records must be extracted
	// before the indexes are checked so that the temporary result set,
	// used during the logical check, can be built.

	pDbInfo->pProgress->uiNumFields = 0;
	pDbInfo->pProgress->uiNumIndexes = 0;
	pDbInfo->pProgress->uiNumContainers = 0;
	
	pDbInfo->pProgress->uiNumLogicalFiles = (FLMUINT) ((pDb->pDict) 
											? (FLMUINT) pDb->pDict->uiLFileCnt 
											: (FLMUINT) 0);

	// Determine the number of fields.

	if (pDb->pDict)
	{
		chkCountFields( pDb->pDict, &pDbInfo->pProgress->uiNumFields);

		for (uiCnt = 0, pLFile = (LFILE *) pDb->pDict->pLFileTbl;
			  uiCnt < pDb->pDict->uiLFileCnt;
			  uiCnt++, pLFile++)
		{
			if (pLFile->uiLfType == LF_INDEX)
			{
				pDbInfo->pProgress->uiNumIndexes++;
				uiNumIndexes++;
			}
			else
			{
				pDbInfo->pProgress->uiNumContainers++;
				if (pLFile->uiLfNum == FLM_DICT_CONTAINER)
				{
					uiNumDictCont++;
				}
				else
				{
					uiNumDataCont++;
				}
			}
		}
	}

	// Allocate memory for each LFILE, then set up each LFILE.

	if (!pDbInfo->pProgress->uiNumLogicalFiles)
	{
		pDbInfo->pLogicalFiles = NULL;
		pDbInfo->pProgress->pLfStats = NULL;
	}
	else
	{
		if( RC_BAD( rc = pPool->poolCalloc(
			(FLMUINT)(sizeof(LF_HDR) * pDbInfo->pProgress->uiNumLogicalFiles),
			(void **)&pDbInfo->pLogicalFiles)))
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = pPool->poolCalloc( 
			(FLMUINT)(sizeof(LF_STATS) * pDbInfo->pProgress->uiNumLogicalFiles),
			(void **)&pDbInfo->pProgress->pLfStats)))
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		uiDictOffset = 0;
		uiDataOffset = uiNumDictCont;
		uiIxOffset = uiDataOffset + uiNumDataCont;
		uiIxStart = uiIxOffset;

		for (uiCnt = 0, pTmpLFile = (LFILE *) pDb->pDict->pLFileTbl;
			  uiCnt < pDbInfo->pProgress->uiNumLogicalFiles;
			  uiCnt++, pTmpLFile++)
		{
			if (pTmpLFile->uiLfType == LF_INDEX)
			{
				FLMUINT	uiTmpIxOffset = uiIxOffset;

				// Indexes need to be in order from lowest to highest because
				// the result set is sorted that way.

				while (uiTmpIxOffset > uiIxStart)
				{
					if (pDbInfo->pLogicalFiles[uiTmpIxOffset - 1].pLFile->uiLfNum <
							 pTmpLFile->uiLfNum)
					{
						break;
					}

					f_memcpy( &pDbInfo->pLogicalFiles[uiTmpIxOffset],
								&pDbInfo->pLogicalFiles[uiTmpIxOffset - 1],
								sizeof(LF_HDR));
								
					f_memcpy( &pDbInfo->pProgress->pLfStats[uiTmpIxOffset],
								&pDbInfo->pProgress->pLfStats[uiTmpIxOffset - 1],
								sizeof(LF_STATS));
								
					pDbInfo->pLogicalFiles [uiTmpIxOffset].pLfStats =
								&pDbInfo->pProgress->pLfStats [uiTmpIxOffset];

					uiTmpIxOffset--;
				}

				pLogicalFile = &(pDbInfo->pLogicalFiles[uiTmpIxOffset]);
				pLfStats = &(pDbInfo->pProgress->pLfStats[uiTmpIxOffset]);
				uiIxOffset++;
			}
			else
			{
				if (pTmpLFile->uiLfNum == FLM_DICT_CONTAINER)
				{
					pLogicalFile = &(pDbInfo->pLogicalFiles[uiDictOffset]);
					pLfStats = &(pDbInfo->pProgress->pLfStats[uiDictOffset]);
					uiDictOffset++;
				}
				else
				{
					pLogicalFile = &(pDbInfo->pLogicalFiles[uiDataOffset]);
					pLfStats = &(pDbInfo->pProgress->pLfStats[uiDataOffset]);
					uiDataOffset++;
				}
			}

			pLogicalFile->pLfStats = pLfStats;

			// Copy the LFILE information - so we can return the information
			// even after the database has been closed.
			
			if( RC_BAD( rc = pPool->poolAlloc( sizeof( LFILE), (void **)&pLFile)))
			{
				goto Exit;
			}
			
			pLogicalFile->pLFile = pLFile;

			// Copy the LFILE structure so we can get enough information to
			// read them from disk, then read them from disk so we have a
			// read-consistent view of them.

			f_memcpy( pLFile, pTmpLFile, sizeof(LFILE));
			if (RC_BAD( rc = flmLFileRead( pDb, pLFile)))
			{
				goto Exit;
			}

			pLfStats->uiLfType = pLFile->uiLfType;
			if (pLFile->uiLfType == LF_INDEX)
			{
				pLfStats->uiIndexNum = pLFile->uiLfNum;
				pLfStats->uiContainerNum = 0;
			}
			else
			{
				pLfStats->uiIndexNum = 0;
				pLfStats->uiContainerNum = pLFile->uiLfNum;
			}

			// If the logical file is an index, get pointers to the index
			// definition and its field definitions.

			if (pLFile->uiLfType == LF_INDEX)
			{
				IXD *		pTmpIxd;
				IFD *		pTmpIfd;

				if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
							  pDb->pFile->bInLimitedMode, pLFile->uiLfNum, NULL,
							  &pTmpIxd, TRUE)))
				{
					if (rc == FERR_BAD_IX)
					{
						chkReportError( pDbInfo, FLM_BAD_PCODE_IXD_TBL,
											LOCALE_IXD_TBL, pLFile->uiLfNum,
											pLFile->uiLfType, 0xFF, 0, 0, 0, 0, 0xFFFF, 0,
											NULL);
						rc = RC_SET( FERR_PCODE_ERROR);
					}

					goto Exit;
				}

				pTmpIfd = pTmpIxd->pFirstIfd;

				// Copy the IXD and IFD information - so we can return the
				// information even after the database has been closed.
				
				if( RC_BAD( rc = pPool->poolAlloc( 
					(FLMUINT)(sizeof(IXD) + sizeof(IFD) * pTmpIxd->uiNumFlds),
					(void **)&pLogicalFile->pIxd)))
				{
					goto Exit;
				}

				pLogicalFile->pIfd = (IFD *)((void *)(&pLogicalFile->pIxd[1]));
				f_memcpy( pLogicalFile->pIxd, pTmpIxd, sizeof(IXD));
				f_memcpy( pLogicalFile->pIfd, pTmpIfd,
							sizeof(IFD) * pTmpIxd->uiNumFlds);
				pLfStats->uiContainerNum = pLogicalFile->pIxd->uiContainerNum;
			}

			// Get the current number of levels in the logical file and
			// allocate an array of LEVEL_INFO structures for the levels.

			pLfStats->uiNumLevels = 0;
			if (RC_BAD( rc = chkGetLfInfo( pDbInfo, pPool, pLfStats, pLFile, NULL,
						  NULL)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	This routine checks all of the B-TREES in the database -- all
		indexes and containers.
****************************************************************************/
RCODE chkVerifyBTrees(
	DB_INFO *		pDbInfo,
	F_Pool *			pPool,
	FLMBOOL *		pbStartOverRV)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = pDbInfo->pDb;
	FLMUINT					uiCurrLf;
	FLMUINT					uiCurrLevel;
	FLMBYTE *				pKeyBuffer = NULL;
	FLMUINT					uiKeysAllocated = 0;
	STATE_INFO				State[BH_MAX_LEVELS];
	FLMBOOL					bStateInitialized[BH_MAX_LEVELS];
	FLMBYTE					ucResetKeyBuff[MAX_KEY_SIZ];
	FLMBYTE *				pucResetKey = NULL;
	FLMUINT					uiResetKeyLen = 0;
	FLMUINT					uiResetDrn = 0;
	LF_HDR *					pLogicalFile;
	LF_STATS *				pLfStats;
	LFILE *					pLFile;
	FLMUINT					uiSaveDictSeq;
	FLMUINT					uiTmpLf;
	LF_STATS *				pTmpLfStats;
	F_Pool					tmpPool;
	FLMBOOL					bRSFinalized = FALSE;
	DB_CHECK_PROGRESS *	pProgress = pDbInfo->pProgress;
	IX_CHK_INFO				IxChkInfo;
	IX_CHK_INFO *			pIxChkInfo = NULL;
	void *					pvPoolMark;
	FILE_HDR *				pFileHdr = &pDb->pFile->FileHdr;

	for (uiCurrLevel = 0; uiCurrLevel < BH_MAX_LEVELS; uiCurrLevel++)
	{
		bStateInitialized[uiCurrLevel] = FALSE;
	}

	if (*pbStartOverRV)
	{
		goto Exit;
	}

	pvPoolMark = pPool->poolMark();
	uiSaveDictSeq = pDb->pDict->uiDictSeq;

	if (RC_BAD( rc = chkSetupLfTable( pDbInfo, pPool)))
	{
		goto Exit;
	}

	if (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING)
	{
		if (RC_BAD( rc = chkSetupIxInfo( pDbInfo, &IxChkInfo)))
		{
			goto Exit;
		}

		pIxChkInfo = &IxChkInfo;
	}

	// Loop through all of the logical files in the database and perform a
	// structural and logical check.

	uiCurrLf = 0;
	while (uiCurrLf < pDbInfo->pProgress->uiNumLogicalFiles)
	{
		pProgress->uiCurrLF = uiCurrLf + 1;
		pLogicalFile = &pDbInfo->pLogicalFiles[uiCurrLf];
		pLfStats = &pDbInfo->pProgress->pLfStats[uiCurrLf];
		pLFile = pLogicalFile->pLFile;
		if (pLFile->uiRootBlk == BT_END)
		{
			rc = FERR_OK;
			uiCurrLf++;
			uiResetKeyLen = 0;
			pucResetKey = NULL;
			uiResetDrn = 0;
			continue;
		}

		// Allocate space to hold the keys, if not already allocated.

		if (uiKeysAllocated < pLfStats->uiNumLevels)
		{

			// If there is already a key allocated, deallocate it

			if (pKeyBuffer)
			{
				f_free( &pKeyBuffer);
				uiKeysAllocated = 0;
			}

			if (RC_BAD( rc = f_alloc( pLfStats->uiNumLevels * MAX_KEY_SIZ,
						  &pKeyBuffer)))
			{
				goto Exit;
			}

			uiKeysAllocated = pLfStats->uiNumLevels;
		}

		// Setup PROGRESS_CHECK_INFO structure

		pProgress->iCheckPhase = CHECK_B_TREE;
		pProgress->bStartFlag = TRUE;
		pProgress->uiLfNumber = pLFile->uiLfNum;
		pProgress->uiLfType = pLFile->uiLfType;

		if (pLFile->uiLfType == LF_INDEX)
		{
			pProgress->bUniqueIndex = (pLogicalFile->pIxd->uiFlags & IXD_UNIQUE) 
															? TRUE 
															: FALSE;
		}

		if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
		{
			break;
		}

		pProgress->bStartFlag = FALSE;

		// Initialize the state information for each level of the B-TREE.

		for (uiCurrLevel = 0; uiCurrLevel < pLfStats->uiNumLevels; uiCurrLevel++)
		{

			// If we are resetting to a particular key, save the statistics
			// which were gathered so far.

			if (uiResetKeyLen)
			{

				// Save the statistics which were gathered.

				pLfStats->pLevelInfo[uiCurrLevel].ui64KeyCount = 
						State[uiCurrLevel].ui64KeyCount;
						
				f_memcpy( &pLfStats->pLevelInfo[uiCurrLevel].BlockInfo,
							&State[uiCurrLevel].BlkInfo, sizeof(BLOCK_INFO));
			}

			flmInitReadState( &State[uiCurrLevel], &bStateInitialized[uiCurrLevel],
								  pFileHdr->uiVersionNum, pDb, pLogicalFile,
								  uiCurrLevel, (FLMUINT) ((!uiCurrLevel) 
								  						? (FLMUINT) BHT_LEAF 
														: (FLMUINT) BHT_NON_LEAF),
								  &pKeyBuffer[uiCurrLevel * MAX_KEY_SIZ]);

			if (!uiResetKeyLen)
			{
				State[uiCurrLevel].uiLastChildAddr = BT_END;
				State[uiCurrLevel].uiElmLastFlag = TRUE;
			}
			else
			{

				// Restore the statistics which were gathered so far.

				State[uiCurrLevel].ui64KeyCount = 
					pLfStats->pLevelInfo[uiCurrLevel].ui64KeyCount;
					
				f_memcpy( &State[uiCurrLevel].BlkInfo,
							&pLfStats->pLevelInfo[uiCurrLevel].BlockInfo,
							sizeof(BLOCK_INFO));
			}
		}

		// Need to finalize the result set used by the logical check. If the
		// current logical file is an index and the result set has not been
		// finalized, call chkRSFinalize.

		if (!pDbInfo->pProgress->bPhysicalCorrupt &&
			 (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING) &&
			 bRSFinalized == FALSE &&
			 pLFile->uiLfType == LF_INDEX)
		{
			FLMUINT64	ui64NumRSKeys = 0;

			// Finalize the result set.

			if (RC_BAD( rc = chkRSFinalize( pIxChkInfo, &ui64NumRSKeys)))
			{
				goto Exit;
			}

			// Reset uiNumKeys to reflect the number of keys in the result
			// set now that all duplicates have been eliminated.

			if (pDbInfo->pProgress->ui64NumKeys > ui64NumRSKeys)
			{
				pDbInfo->pProgress->ui64NumDuplicateKeys = 
					pDbInfo->pProgress->ui64NumKeys - ui64NumRSKeys;
			}

			pDbInfo->pProgress->ui64NumKeys = ui64NumRSKeys;

			// Set bRSFinalized to TRUE so that subsequent passes will not
			// attempt to finalize the result set again.

			bRSFinalized = TRUE;
		}

		// Call chkVerifySubTree to check the B-TREE starting at the root
		// block.

		tmpPool.poolInit( 512);
		pDbInfo->bReposition = FALSE;
		
		rc = chkVerifySubTree( pDbInfo, pIxChkInfo, NULL,
									 &State[pLfStats->uiNumLevels - 1],
									 pLFile->uiRootBlk, &tmpPool, pucResetKey,
									 uiResetKeyLen, uiResetDrn);
		tmpPool.poolFree();

		if (rc == FERR_OLD_VIEW)
		{

			// If it is a read transaction, reset.

			if (flmGetDbTransType( pDb) == FLM_READ_TRANS)
			{

				// Free the KrefCntrl

				KrefCntrlFree( pDb);

				// Abort the read transaction

				if (RC_BAD( rc = flmAbortDbTrans( pDb)))
				{
					goto Exit;
				}

				// Try to start a new read transaction

				if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_READ_TRANS, 0,
							  FLM_DONT_POISON_CACHE)))
				{
					goto Exit;
				}
			}

			rc = FERR_OK;
			pDbInfo->bReposition = TRUE;
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// We may get told to reposition if we had to repair an index or we
		// got an old view error.

		if (pDbInfo->bReposition)
		{

			// If the dictionary has changed we must start all over.

			if (pDb->pDict->uiDictSeq != uiSaveDictSeq)
			{
				*pbStartOverRV = TRUE;
				goto Exit;
			}

			// Save the current key at the bottom level of the B-Tree. This
			// is the point we want to try to reset to. Don't change the
			// reset key if the current key length is zero - this may have
			// occurred because of some error - we want to keep moving
			// forward in the file if at all possible.

			if (State[0].uiCurKeyLen)
			{
				uiResetKeyLen = State[0].uiCurKeyLen;
				pucResetKey = &ucResetKeyBuff[0];
				uiResetDrn = State[0].uiCurrIxRefDrn;
				f_memcpy( pucResetKey, State[0].pCurKey, uiResetKeyLen);
			}

			// Re-read each logical file's LFH information.

			pProgress->ui64DatabaseSize = FSGetSizeInBytes( 
				pDb->pFile->uiMaxFileSize, pDb->LogHdr.uiLogicalEOF);

			// Reread each of the LFH blocks and update the root block
			// address and other pertinent information for each logical file.

			for (uiTmpLf = 0, pTmpLfStats = pDbInfo->pProgress->pLfStats;
				  uiTmpLf < pDbInfo->pProgress->uiNumLogicalFiles;
				  uiTmpLf++, pTmpLfStats++)
			{
				FLMBOOL	bCurrLfLevelChanged = FALSE;

				if (RC_BAD( rc = chkGetLfInfo( pDbInfo, pPool, pTmpLfStats,
							  pDbInfo->pLogicalFiles[uiTmpLf].pLFile, pLfStats,
							  &bCurrLfLevelChanged)))
				{
					goto Exit;
				}

				// If the number of levels for the current logical file
				// changed, reset things so we will recheck the entire logical
				// file.

				if (bCurrLfLevelChanged)
				{
					pucResetKey = NULL;
					uiResetKeyLen = 0;
				}
			}

			continue;
		}

		// Verify that all of the levels' next block address's are BT_END.

		if (RC_OK( pDbInfo->LastStatusRc))
		{
			for (uiCurrLevel = 0;
				  uiCurrLevel < pLfStats->uiNumLevels;
				  uiCurrLevel++)
			{

				// Save the statistics which were gathered.

				pLfStats->pLevelInfo[uiCurrLevel].ui64KeyCount = 
					State[uiCurrLevel].ui64KeyCount;
					
				f_memcpy( &pLfStats->pLevelInfo[uiCurrLevel].BlockInfo,
							&State[uiCurrLevel].BlkInfo, sizeof(BLOCK_INFO));

				// Make sure the last block had a NEXT block address of
				// BT_END.

				if ((State[uiCurrLevel].uiNextBlkAddr) &&
					 (State[uiCurrLevel].uiNextBlkAddr != BT_END))
				{
					chkReportError( pDbInfo, FLM_BAD_LAST_BLK_NEXT, LOCALE_B_TREE,
										pDbInfo->pProgress->uiLfNumber,
										pDbInfo->pProgress->uiLfType, uiCurrLevel, 0, 0,
										0, 0, 0xFFFF, 0, NULL);
				}
			}
		}

		if (RC_BAD( pDbInfo->LastStatusRc))
		{
			break;
		}

		uiCurrLf++;
		pucResetKey = NULL;
		uiResetKeyLen = 0;
		uiResetDrn = 0;
	}

	// If index check was requested, no structural corruptions were
	// detected, and this is the last logical file, need to make sure that
	// the result set is empty.

	if (RC_OK( rc) &&
		 !pDbInfo->pProgress->bPhysicalCorrupt &&
		 (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING) &&
		 bRSFinalized == TRUE &&
		 uiCurrLf == pDbInfo->pProgress->uiNumLogicalFiles)
	{
		for (;;)
		{
			if (RC_BAD( rc = chkGetNextRSKey( pIxChkInfo)))
			{
				if (rc == FERR_EOF_HIT || rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;
					break;
				}

				goto Exit;
			}
			else
			{

				// Updated statistics

				pIxChkInfo->pDbInfo->pProgress->ui64NumKeysExamined++;

				if (RC_BAD( rc = chkResolveIXMissingKey( &(State[0]), pIxChkInfo)))
				{
					goto Exit;
				}
			}
		}
	}

	// Clear the unique index flag

	pProgress->bUniqueIndex = FALSE;

Exit:

	// Clear the pRecord for each level in the state array.

	for (uiCurrLevel = 0; uiCurrLevel < BH_MAX_LEVELS; uiCurrLevel++)
	{
		if (bStateInitialized[uiCurrLevel] && State[uiCurrLevel].pRecord)
		{
			State[uiCurrLevel].pRecord->Release();
			State[uiCurrLevel].pRecord = NULL;
		}
	}

	// Cleanup any temporary index check files

	if (pIxChkInfo != NULL)
	{
		chkFreeIxInfo( pIxChkInfo);
	}

	if (pKeyBuffer)
	{
		f_free( &pKeyBuffer);
	}

	if (RC_OK( rc) && RC_BAD( pDbInfo->LastStatusRc))
	{
		rc = pDbInfo->LastStatusRc;
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE chkReportError(
	DB_INFO *			pDbInfo,
	eCorruptionType	eCorruption,
	eCorruptionLocale eErrLocale,
	FLMUINT				uiErrLfNumber,
	FLMUINT				uiErrLfType,
	FLMUINT				uiErrBTreeLevel,
	FLMUINT				uiErrBlkAddress,
	FLMUINT				uiErrParentBlkAddress,
	FLMUINT				uiErrElmOffset,
	FLMUINT				uiErrDrn,
	FLMUINT				uiErrElmRecOffset,
	FLMUINT				uiErrFieldNum,
	FLMBYTE *			pBlk)
{
	CORRUPT_INFO	CorruptInfo;
	FLMBOOL			bFixErr;

	CorruptInfo.eCorruption = eCorruption;
	CorruptInfo.eErrLocale = eErrLocale;
	CorruptInfo.uiErrLfNumber = uiErrLfNumber;
	CorruptInfo.uiErrLfType = uiErrLfType;
	CorruptInfo.uiErrBTreeLevel = uiErrBTreeLevel;
	CorruptInfo.uiErrBlkAddress = uiErrBlkAddress;
	CorruptInfo.uiErrParentBlkAddress = uiErrParentBlkAddress;
	CorruptInfo.uiErrElmOffset = uiErrElmOffset;
	CorruptInfo.uiErrDrn = uiErrDrn;
	CorruptInfo.uiErrElmRecOffset = uiErrElmRecOffset;
	CorruptInfo.uiErrFieldNum = uiErrFieldNum;
	CorruptInfo.pBlk = pBlk;
	CorruptInfo.pErrIxKey = NULL;
	CorruptInfo.pErrRecord = NULL;
	CorruptInfo.pErrRecordKeyList = NULL;
	if ((pDbInfo->fnStatusFunc) && (RC_OK( pDbInfo->LastStatusRc)))
	{
		bFixErr = FALSE;
		pDbInfo->LastStatusRc = (*pDbInfo->fnStatusFunc)
			(FLM_PROBLEM_STATUS, (void *) &CorruptInfo, (void *) &bFixErr,
			pDbInfo->pProgress->AppArg);
	}

	if (eCorruption != FLM_OLD_VIEW)
	{
		pDbInfo->pProgress->bPhysicalCorrupt = TRUE;
		pDbInfo->uiFlags &= ~FLM_CHK_INDEX_REFERENCING;
	}

	return (pDbInfo->LastStatusRc);
}

/****************************************************************************
Desc: Initializes an IX_CHK_INFO structure
****************************************************************************/
FSTATIC RCODE chkSetupIxInfo(
	DB_INFO *		pDbInfo,
	IX_CHK_INFO *	pIxInfoRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiIxCount = 0;
	FLMUINT			uiIxNum = 0;
	IXD *				pIxd;
	LFILE *			pLFile;
	char				szTmpIoPath[ F_PATH_MAX_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];
	FDB *				pDb = pDbInfo->pDb;

	f_memset( pIxInfoRV, 0, sizeof(IX_CHK_INFO));
	pIxInfoRV->pool.poolInit( 512);
	pIxInfoRV->pDbInfo = pDbInfo;

	// Set up the result set path

	if (RC_BAD( rc = flmGetTmpDir( szTmpIoPath)))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
				pDb->pFile->pszDbPath, szTmpIoPath, szBaseName)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

	// Initialize the result set. The result set will be used to build an
	// ordered list of keys for comparision to the database's indexes.

	if (RC_BAD( rc = chkRSInit( szTmpIoPath, &pIxInfoRV->pRSet)))
	{
		goto Exit;
	}

	// Build list of all indexes

	uiIxCount = 0;
	if (pDb->pDict)
	{
		uiIxCount += pDb->pDict->uiIxdCnt;
	}

	// Allocate memory to save each index number and its associated
	// container number.

	if (RC_BAD( rc = f_alloc( (FLMUINT) ((sizeof(FLMUINT) * uiIxCount) +
					  (sizeof(FLMBOOL) * uiIxCount)), &(pIxInfoRV->puiIxArray))))
	{
		goto Exit;
	}

	// Save the index numbers into the array.

	uiIxCount = 0;
	if (pDb->pDict)
	{
		for (uiIxNum = 0, pIxd = (IXD *) pDb->pDict->pIxdTbl;
			  uiIxNum < pDb->pDict->uiIxdCnt;
			  uiIxNum++, pIxd++)
		{
			pIxInfoRV->puiIxArray[uiIxCount] = pIxd->uiIndexNum;
			uiIxCount++;
		}
	}

	if (RC_OK( fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
				 FLM_DICT_INDEX, &pLFile, NULL)))
	{
		pIxInfoRV->puiIxArray[uiIxCount] = FLM_DICT_INDEX;
		uiIxCount++;
	}

	pIxInfoRV->uiIxCount = uiIxCount;
	pIxInfoRV->bGetNextRSKey = TRUE;

Exit:

	// Clean up any memory on error exit.

	if (RC_BAD( rc))
	{
		pIxInfoRV->pool.poolFree();
		if (pIxInfoRV->puiIxArray)
		{
			f_free( &(pIxInfoRV->puiIxArray));
		}
	}

	return (rc);
}

/****************************************************************************
Desc: Outputs keys to the temporary result set
****************************************************************************/
FSTATIC RCODE chkOutputIndexKeys(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	IXD *				pIxd,
	REC_KEY *		pKeyList)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiKeyLen;
	REC_KEY *	pKey;
	FLMBYTE		ucBuf[MAX_KEY_SIZ + RS_KEY_OVERHEAD];

	pKey = pKeyList;
	while (pKey)
	{

		// Set the index and reference

		UW2FBA( (FLMUINT16) pIxd->uiIndexNum, &(ucBuf[RS_IX_OFFSET]));
		UD2FBA( (FLMUINT32) pStateInfo->uiElmDrn, &(ucBuf[RS_REF_OFFSET]));

		// Convert the key tree to a collation key

		if (RC_BAD( rc = KYTreeToKey( pIxChkInfo->pDbInfo->pDb, pIxd, pKey->pKey,
					  pKey->pKey->getContainerID(), &(ucBuf[RS_KEY_OVERHEAD]),
					  &uiKeyLen, 0)))
		{
			goto Exit;
		}

		// Add the composite key (index, ref, key) to the result set

		if (RC_BAD( rc = pIxChkInfo->pRSet->addEntry( ucBuf,
					  uiKeyLen + RS_KEY_OVERHEAD)))
		{
			goto Exit;
		}

		// Update statistics. Note that uiNumKeys will reflect the total
		// number of keys generated by records, including any duplicate
		// keys. This value is updated to reflect the correct number of keys
		// once the result set has been finalized.

		pIxChkInfo->pDbInfo->pProgress->ui64NumKeys++;
		pKey = pKey->pNextKey;
	}

Exit:

	return (rc);
}
