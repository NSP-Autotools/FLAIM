//-------------------------------------------------------------------------
// Desc:	Dictionary updates.
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

FSTATIC RCODE FDDBuildIndex(
	FDB *			pDb,
	FLMUINT		uiIndexNum,
	FLMBOOL		bDoInBackground,
	FLMBOOL		bCreateSuspended,
	FLMBOOL *	pbLogCompleteIndexSet);

FSTATIC RCODE flmFreeBlockChain(
	FDB *				pDb,
	FLMUINT			uiStartAddr,
	FLMUINT *		puiCount,
	FLMUINT *		puiEndAddr);

FSTATIC RCODE flmFreeLFileBlocks(
	FDB *				pDb,
	LFILE *			pLFile);

FSTATIC RCODE flmFreeContainerBlocks(
	FDB *				pDb,
	LFILE *			pLFile);

FSTATIC RCODE flmFreeIndexBlocks(
	FDB *			pDb,
	LFILE *			pLFile,
	FLMBOOL			bInvalidateLFile);

FSTATIC RCODE flmRetrieveTrackerRec(
	FDB *		pDb,
	FLMUINT			uiDrn,
	FLMBOOL			bExact,
	FlmRecord **	ppRecord);

FSTATIC RCODE flmDeleteTrackerRec(
	FDB *			pDb,
	FLMUINT			uiDrn);

FSTATIC RCODE flmModifyTrackerRec(
	FDB *		pDb,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord);

FSTATIC RCODE FLMAPI flmMaintThread(
	IF_Thread *		pThread);

FSTATIC RCODE fdictRemoveIndexes(
	FDB *				pDb,
	FLMUINT			uiContainer);

FSTATIC RCODE DDVerifyModField(
	FDB *				pDb,
	FlmRecord *		pOldRecord,
	FlmRecord *		pNewRecord);

FSTATIC RCODE DDVerifyModEncDef(
	FDB *				pDb,
	FlmRecord *		pOldRecord,
	FlmRecord *		pNewRecord);

FSTATIC RCODE flmAddIndexFixup(
	FDB *				pDb,
	FLMUINT			uiIndexNum,
	FLMUINT			uiLastDrnIndexed,
	FLMUINT			uiLastContainerIndexed);

/***************************************************************************
Desc:		Verify that the action being attempted in a dictionary update
			is valid.  Caller should check if container in valid range.
****************************************************************************/
RCODE flmLFileDictUpdate(
	FDB *			pDb,
	LFILE **		ppDictLFile,
	FLMUINT *	puiDrnRV,
	FlmRecord *	pNewRecord,
	FlmRecord *	pOldRecord,
	FLMBOOL		bDoInBackground,
	FLMBOOL		bCreateSuspended,
	FLMBOOL *	pbLogCompleteIndexSet,
	FLMBOOL		bRebuildOp)
{
	RCODE			rc = FERR_OK;
	LFILE *		pLFile;
	LFILE *		pDictIxLFile = NULL;
	LFILE			LFile;
	FLMUINT		uiNewRecordType;
	FLMUINT		uiOldRecordType;
	char			szNativeBuf1[ 32];
	FLMUINT		uiLen;
	void *		pvField;

	// Flush any index keys before making any changes to dictionary items.
	// Dictionary changes may change index definitions, etc.

	if( RC_BAD(rc = KYKeysCommit( pDb, FALSE)))
	{
		goto Exit;
	}

	// Also need to flush out any index counts before making changes to
	// dictionary items - due to the fact that index definitions may get
	// dropped, added, etc.

	if (RC_BAD( rc = FSCommitIxCounts( pDb)))
	{
		goto Exit;
	}

	// Clear out the cdl table in case it changes

	KrefCntrlFree( pDb);
	uiNewRecordType = uiOldRecordType = 0;

	if( pNewRecord)
	{
		pvField = pNewRecord->root();
		uiNewRecordType = pNewRecord->getFieldID( pvField);
	}

	if( pOldRecord)
	{
		pvField = pOldRecord->root();
		uiOldRecordType = pOldRecord->getFieldID(pvField);
	}

	if( !pNewRecord)
	{
		// Cannot delete a field if it is referenced from an index.

		if( uiOldRecordType == FLM_FIELD_TAG )
		{
			if( RC_BAD( rc = flmCheckDictFldRefs( pDb->pDict, *puiDrnRV)))
			{
				goto Exit;
			}

			// Determine if this field is qualified to be deleted.
			// Those rules are:
			//
			//   1) it must have a state value of 'unused'
			//   2) or a state of 'purge' (FlmDbSweep or recover only)

			if( (pvField = pOldRecord->find( pOldRecord->root(),
				FLM_STATE_TAG)) == NULL)
			{
				rc = RC_SET( FERR_CANNOT_DEL_ITEM);
				goto Exit;
			}

			uiLen = sizeof( szNativeBuf1);
			if( RC_BAD( rc = pOldRecord->getNative( pvField, szNativeBuf1, &uiLen)))
			{
				goto Exit;
			}

			if( f_strnicmp( szNativeBuf1, "unus", 4) == 0 ||
				(f_strnicmp( szNativeBuf1, "purg", 4) == 0 &&
					pDb->bFldStateUpdOk))
			{
				;		// It's okay to delete this field or record template.
			}
			else
			{
				rc = RC_SET( FERR_CANNOT_DEL_ITEM);
				goto Exit;
			}
		}
		else if ( uiOldRecordType == FLM_ENCDEF_TAG )
		{
			if( RC_BAD( rc = flmCheckDictEncDefRefs( pDb->pDict, *puiDrnRV)))
			{
				goto Exit;
			}

			// Determine if this record is qualified to be deleted.
			// Those rules are:
			//
			//   1) it must have a state value of 'unused'
			//   2) or a state of 'purge' (FlmDbSweep or recover only)

			if( (pvField = pOldRecord->find( pOldRecord->root(),
				FLM_STATE_TAG)) == NULL)
			{
				rc = RC_SET( FERR_CANNOT_DEL_ITEM);
				goto Exit;
			}

			uiLen = sizeof( szNativeBuf1);
			if( RC_BAD( rc = pOldRecord->getNative( pvField, szNativeBuf1, &uiLen)))
			{
				goto Exit;
			}

			if( f_strnicmp( szNativeBuf1, "unus", 4) == 0 ||
				(f_strnicmp( szNativeBuf1, "purg", 4) == 0 &&
					pDb->bFldStateUpdOk))
			{
				;		// It's okay to delete this field or record template.
			}
			else
			{
				rc = RC_SET( FERR_CANNOT_DEL_ITEM);
				goto Exit;
			}
		}

	}
	else if( pNewRecord && pOldRecord)
	{
		// We cannot allow changing the type of dictionary definition

		if( uiOldRecordType != uiNewRecordType)
		{
			rc = RC_SET( FERR_CANNOT_MOD_DICT_REC_TYPE);
			goto Exit;
		}

		// If modifying a field or encryption definition record,
		// cannot change the type.
		if ( uiOldRecordType == FLM_FIELD_TAG)
		{
			if (RC_BAD( rc = DDVerifyModField( pDb, pOldRecord, pNewRecord)))
			{
				goto Exit;
			}
		}
		else if ( uiOldRecordType == FLM_ENCDEF_TAG)
		{
			if (RC_BAD( rc = DDVerifyModEncDef( pDb, pOldRecord, pNewRecord)))
			{

			}
		}
	}

	if( RC_BAD( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				FLM_DICT_INDEX, &pDictIxLFile, NULL)))
	{
		goto Exit;
	}

	// Make sure new records do not contain a key.
	if( !pOldRecord )
	{
		if (uiNewRecordType == FLM_ENCDEF_TAG && !bRebuildOp &&
			 !(pDb->uiFlags & FDB_REPLAYING_RFL))
		{
			// Cannot add a record with a key field already present.
			if( (pNewRecord->find( pNewRecord->root(),
										  FLM_KEY_TAG)) != NULL)
			{
				rc = RC_SET( FERR_CANNOT_SET_KEY);
				goto Exit;
			}
		}
	}

	// fdictRecUpdate checks for duplicates (invalid ADD).
	// It also makes sure we are not trying to change the key on EncDef records.
	// It will make sure new keys are generated for new EncDef records.

	if( RC_BAD( rc = fdictRecUpdate( pDb, *ppDictLFile, pDictIxLFile,
									puiDrnRV, pNewRecord, pOldRecord, bRebuildOp)))
	{
		goto Exit;
	}

	// Now take care of adding, deleting or rebuilding indexes & containers.

	if( !pOldRecord)
	{
		FLMBOOL			bRereadLFiles = FALSE;

		// Create a NEW LFILE for indexes and containers.

		if( (uiNewRecordType == FLM_INDEX_TAG ) ||
			(uiNewRecordType == FLM_CONTAINER_TAG ))
		{
			if( RC_BAD(rc = flmLFileCreate( pDb, &LFile, *puiDrnRV,
				(uiNewRecordType == FLM_INDEX_TAG) ? LF_INDEX : LF_CONTAINER)))
			{
				goto Exit;
			}
			pLFile = &LFile;
			bRereadLFiles = TRUE;
		}

		// Special optimization just for adding a new object to the dictionary.
		if( RC_BAD( rc = flmAddRecordToDict( pDb, pNewRecord,
						*puiDrnRV, bRereadLFiles)))
		{
			goto Exit;
		}

		if( uiNewRecordType == FLM_INDEX_TAG)
		{
			// Build the indexes

			if( RC_BAD( rc = FDDBuildIndex( pDb, *puiDrnRV,
				bDoInBackground, bCreateSuspended, pbLogCompleteIndexSet)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		// Delete Record

		if( !pNewRecord)
		{
			if( uiOldRecordType == FLM_INDEX_TAG )
			{
				if( RC_BAD( rc = fdictGetIndex(
						pDb->pDict, pDb->pFile->bInLimitedMode,
						*puiDrnRV, &pLFile, NULL, TRUE)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = flmFreeIndexBlocks( pDb, pLFile, TRUE)))
				{
					goto Exit;
				}

				// Remove from index fixup list if we are deleting an index.
				// It is impossible to be deleting an index in a background
				// thread, so if there is a fixup, it is here because the
				// index was added during this transaction (in the background).
				// If the transaction aborts the IXD will simply go away, and
				// there will be no need to fix it up.

				if (pDb->pIxdFixups)
				{
					IXD_FIXUP *	pIxdFixup = pDb->pIxdFixups;
					IXD_FIXUP *	pPrevIxdFixup = NULL;

					while (pIxdFixup && pIxdFixup->uiIndexNum != *puiDrnRV)
					{
						pPrevIxdFixup = pIxdFixup;
						pIxdFixup = pIxdFixup->pNext;
					}

					if (pIxdFixup)
					{
						if (pPrevIxdFixup)
						{
							pPrevIxdFixup->pNext = pIxdFixup->pNext;
						}
						else
						{
							pDb->pIxdFixups = pIxdFixup->pNext;
						}
						f_free( &pIxdFixup);
					}
				}
			}
			else if( uiOldRecordType == FLM_CONTAINER_TAG )
			{
				if( RC_BAD( rc = fdictGetContainer( pDb->pDict, *puiDrnRV, &pLFile)))
				{
					goto Exit;
				}

				// Remove indexes associated with this container.

				if (RC_BAD( rc = fdictRemoveIndexes( pDb, *puiDrnRV)))
				{
					goto Exit;
				}

				// Need to remove all records from record cache
				// for this container!

				flmRcaRemoveContainerRecs( pDb, *puiDrnRV);

				if( RC_BAD( rc = flmFreeContainerBlocks( pDb, pLFile)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			// Modify Record

			if( uiOldRecordType == FLM_INDEX_TAG )
			{
				if( RC_BAD(rc = fdictGetIndex(
					pDb->pDict, pDb->pFile->bInLimitedMode,
					*puiDrnRV, &pLFile, NULL, TRUE)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = flmFreeIndexBlocks( pDb, pLFile, FALSE)))
				{
					goto Exit;
				}
			}
		}

		// On delete or modify index make sure something is in the stop list.

		if( uiOldRecordType == FLM_INDEX_TAG &&
			!(pDb->uiFlags & FDB_REPLAYING_RFL))

		{
			if( RC_BAD( rc = flmAddToStopList( pDb, pOldRecord->getID())))
			{
				goto Exit;
			}
		}

		// Create a whole new internal dictionary.

		if( RC_BAD( rc = fdictCreateNewDict( pDb)))
		{
			goto Exit;
		}
		
		// Re-get the dictionary LFILE
		
		if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
			FLM_DICT_CONTAINER, ppDictLFile)))
		{
			goto Exit;
		}
		

		if( pNewRecord && uiOldRecordType == FLM_INDEX_TAG)
		{
			//  Rebuild the index

			if( RC_BAD( rc = FDDBuildIndex( pDb, *puiDrnRV,
				bDoInBackground, bCreateSuspended, pbLogCompleteIndexSet)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc );
}

/***************************************************************************
Desc:		This routine checks to see if a field is referenced in an index
			or record template definitions.
****************************************************************************/
RCODE flmCheckDictFldRefs(
	FDICT *		pDict,
	FLMUINT		uiFieldNum)
{
	RCODE			rc = FERR_OK;
	IFD *			pIfd;
	FLMUINT *	pFieldPathTbl;
	FLMUINT		uiFldPathsCnt;

	// Does the field have an IFD reference.

	if( RC_BAD( rc = fdictGetField( pDict, uiFieldNum, NULL, &pIfd, NULL)))
	{
		goto Exit;
	}

	if( pIfd)
	{
		rc = RC_SET( FERR_CANNOT_DEL_ITEM);
		goto Exit;
	}

	// Look through all of the field paths.

	pFieldPathTbl = pDict->pFldPathsTbl;
	uiFldPathsCnt = pDict->uiFldPathsCnt;

	for( ; uiFldPathsCnt--; pFieldPathTbl++)
	{
		if( *pFieldPathTbl == uiFieldNum)
		{
			rc = RC_SET( FERR_CANNOT_DEL_ITEM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		This routine checks to see if an encryption definition is referenced
			in an index.
****************************************************************************/
RCODE flmCheckDictEncDefRefs(
	FDICT *		pDict,
	FLMUINT		uiEncId)
{
	RCODE			rc = FERR_OK;
	IXD *			pIxd;
	FLMUINT		uiIxdCnt;

	// Is the Encryption Definition referenced in any index?

	for( pIxd = pDict->pIxdTbl, uiIxdCnt = pDict->uiIxdCnt; uiIxdCnt--; pIxd++)
	{
		if( pIxd->uiEncId &&  pIxd->uiEncId == uiEncId)
		{
			rc = RC_SET( FERR_CANNOT_DEL_ITEM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Build an index.
****************************************************************************/
FSTATIC RCODE FDDBuildIndex(
	FDB *			pDb,
	FLMUINT		uiIndexNum,
	FLMBOOL		bDoInBackground,
	FLMBOOL		bCreateSuspended,
	FLMBOOL *	pbLogCompleteIndexSet)
{
	RCODE   		rc = FERR_OK;
	LFILE *		pLFile;
	IXD *			pIxd;

	// Flush any keys and free the tables because they may grow!

	if( RC_BAD( rc = KYKeysCommit( pDb, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
	  	goto Exit;
	}

	if( RC_BAD( rc = fdictGetIndex(
					pDb->pDict, pDb->pFile->bInLimitedMode,
					uiIndexNum, &pLFile, &pIxd)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileIndexBuild( pDb, pLFile, pIxd, bDoInBackground,
		bCreateSuspended, pbLogCompleteIndexSet)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: 	Read all records from a container building all active indexes
****************************************************************************/
RCODE flmLFileIndexBuild(
	FDB *				pDb,
	LFILE *			pIxLFile,
	IXD *				pIxd,
	FLMBOOL			bDoInBackground,
	FLMBOOL			bCreateSuspended,
	FLMBOOL *		pbLogCompleteIndexSet)
{
	RCODE				rc = FERR_OK;

	if( pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		// Don't index now.  The RFL function INDEX_SET will 
		// generate index keys.

		if( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_3_02 &&
			pDb->pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_51)
		{
			if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, 
						pIxd->uiIndexNum, 1, 0, TRANS_ID_OFFLINE, FALSE)))
			{
				goto Exit;
			}

			goto Exit;
		}
	}

	//	Start the indexing thread if building in the background
	// and NOT a unique index.

	if( bDoInBackground && !(pIxd->uiFlags & IXD_UNIQUE))
	{
		if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, 
			pIxd->uiIndexNum, 1, 0, TRANS_ID_OFFLINE, 
			bCreateSuspended)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmLFileWrite( pDb, pIxLFile)))
		{
			goto Exit;
		}

		pIxd->uiFlags |= IXD_OFFLINE;

		if( bCreateSuspended)
		{
			pIxd->uiFlags |= IXD_SUSPENDED;
		}
		else if( !(pDb->uiFlags & FDB_REPLAYING_RFL))
		{
			// Don't add the index to the start list if we are replaying
			// the RFL.  The indexing threads will be started once
			// recovery is complete.

			if( RC_BAD( rc = flmAddToStartList( pDb, pIxd->uiIndexNum)))
			{
				goto Exit;
			}
		}

		goto Exit;
	}
	
	// uiIndexToBeUpdated better be zero at this point since we 
	// are not working in the background.

	if( RC_BAD( rc = flmIndexSetOfRecords( pDb, pIxd->uiIndexNum, 0, 1, 
			DRN_LAST_MARKER, pDb->fnStatus, pDb->StatusData,
			pDb->fnIxCallback, pDb->IxCallbackData,
			NULL, NULL)))
	{
		goto Exit;
	}

	if( pbLogCompleteIndexSet)
	{ 
		*pbLogCompleteIndexSet = TRUE;
	}

Exit:

	// Need to convert FERR_NOT_UNIQUE to FERR_IX_FAILURE so that we
	// can force the transaction to abort.  Normally, FERR_NOT_UNIQUE is
	// an error that does not require the transaction to abort.  However,
	// in this case, we are generating an index and it is not possible
	// to continue the transaction - because we may have modified disk
	// blocks.

	if( rc == FERR_NOT_UNIQUE)
	{
		rc = RC_SET( FERR_IX_FAILURE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Index a set of records.  Called when recovering or restoring from
		roll-forward log.
****************************************************************************/
RCODE flmDbIndexSetOfRecords(
	HFDB				hDb,
	FLMUINT			uiIxNum,
	FLMUINT			uiContainerNum,
	FLMUINT			uiStartDrn, 
	FLMUINT			uiEndDrn)
{
	FLMBOOL		bStartedAutoTrans = FALSE;
	FDB *			pDb = (FDB *)hDb;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
		FDB_TRANS_GOING_OK, 0, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmIndexSetOfRecords( pDb, uiIxNum, 
			uiContainerNum, uiStartDrn, uiEndDrn, 
			NULL, NULL, NULL, NULL, NULL, NULL)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

	// Unlock all memory that has been locked by FLAIM 

	(void)fdbExit( pDb);
	return( rc);
}

/***************************************************************************
Desc:		Logs information about an index being built
****************************************************************************/
void flmLogIndexingProgress(
	FLMUINT		uiIndexNum,
	FLMUINT		uiLastDrn)
{
	flmLogMessage( 
		F_DEBUG_MESSAGE,
		FLM_YELLOW,
		FLM_BLACK,
		uiLastDrn
			? "Indexing progress: Index %u is offline.  Last record processed = %u."
			: "Indexing progress: Index %u is online.",
		(unsigned)uiIndexNum, (unsigned)uiLastDrn);
}

/***************************************************************************
Desc:		Index a set of records or until time runs out.
****************************************************************************/
RCODE flmIndexSetOfRecords(
	FDB *					pDb,
	FLMUINT				uiIxNum,
	FLMUINT				uiContainerNum, // 0 is passed in when doing all containers.
												 // It will only be 0 when adding an index in
												 // the foreground.
	FLMUINT				uiStartDrn,
	FLMUINT				uiEndDrn,
	STATUS_HOOK			fnStatus,
	void *				StatusData,
	IX_CALLBACK			fnIxCallback,
	void *				IxCallbackData,
	FINDEX_STATUS *	pIndexStatus,
	FLMBOOL *			pbHitEnd,
	IF_Thread *			pThread,
	FlmRecord *			pReusableRec)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiDrn;
	FLMUINT				uiLastDrn = 0;
	IXD *					pIxd = NULL;
	LFILE *				pDataLFile;
	FlmRecord *			pRecord = NULL;
	FlmRecord *			pModRecord = NULL;
	IF_LockObject *	pFileLockObj = pDb->pFile->pFileLockObj;
	FLMBOOL				bHitEnd = FALSE;
	FLMBOOL				bDataRecordRead;
	FLMUINT				uiCurrTime;
	FLMUINT				uiLastStatusTime = 0;
	FLMUINT				uiStartTime;
	FLMUINT				uiMinTU;
	FLMUINT				uiStatusIntervalTU;
	FLMUINT				uiRecsProcessed = 0;
	FLMBOOL				bUpdateTracker = FALSE;
	FLMBOOL				bHadUniqueKeys;
	FLMBOOL				bRelinquish = FALSE;
	BTSK  				stackBuf [BH_MAX_LEVELS];
	BTSK *				pStack = &stackBuf [0];
	FLMBYTE				ucKeyBuf [DIN_KEY_SIZ];
	FLMBYTE				ucSearchKey [DIN_KEY_SIZ];
	FLMBOOL				bDoAllContainers = FALSE;
	ITT *					pItt;
	F_Pool				ReadPool;
	void *				pvTmpPoolMark = pDb->TempPool.poolMark();

	ReadPool.poolInit( 8192);
	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);

	uiMinTU = FLM_MILLI_TO_TIMER_UNITS( 500);
	uiStatusIntervalTU = FLM_SECS_TO_TIMER_UNITS( 10);
	uiStartTime = FLM_GET_TIMER();

	if( !pReusableRec)
	{
		if( (pReusableRec = f_new FlmRecord) != NULL)
		{
			if( RC_BAD( pReusableRec->preallocSpace( 512, 1024 * 64)))
			{
				pReusableRec->Release();
				pReusableRec = NULL;
			}
		}
	}
	else
	{
		pReusableRec->AddRef();
	}

	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetIndex(
		pDb->pDict, pDb->pFile->bInLimitedMode,
		uiIxNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}
	flmAssert( (pIxd->uiFlags & IXD_SUSPENDED) == 0);

	if (!pIxd->uiContainerNum && !uiContainerNum)
	{
		flmAssert( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_50);
		bDoAllContainers = TRUE;
	}

	for (;;)
	{
		if( !bDoAllContainers)
		{
			// NOTE: Background indexing always comes through this
			// code path, because a non-zero index number is passed
			// in, even for cross-container indexes.
			// NOTE ALSO: We always take this path for single-container
			// indexes too, because bDoAllContainers is always FALSE
			// for them.

#ifdef FLM_DEBUG
			if( !pIxd->uiContainerNum)
			{
				flmAssert( pDb->pFile->FileHdr.uiVersionNum >= 
							  FLM_FILE_FORMAT_VER_4_50);
			}
#endif

			uiContainerNum = (FLMUINT)(pIxd->uiContainerNum
													 ? pIxd->uiContainerNum
													 : uiContainerNum);

			if (RC_BAD( rc = fdictGetContainer( pDb->pDict,
										uiContainerNum, &pDataLFile)))
			{
				goto Exit;
			}
		}
		else
		{
			uiStartDrn = 1;
			uiEndDrn = 0xFFFFFFFF;
			if (uiContainerNum == FLM_DATA_CONTAINER)
			{
				bHitEnd = TRUE;
				break;
			}
			uiContainerNum++;
			while (uiContainerNum < pDb->pDict->uiIttCnt)
			{
				pItt = &pDb->pDict->pIttTbl [uiContainerNum];
				if (ITT_IS_CONTAINER( pItt))
				{
					pDataLFile = (LFILE *)pItt->pvItem;
					break;
				}
				else
				{
					uiContainerNum++;
				}
			}

			if (uiContainerNum >= pDb->pDict->uiIttCnt)
			{
				uiContainerNum = FLM_DATA_CONTAINER;
				pDataLFile = &pDb->pDict->pLFileTbl[ LFILE_DATA_CONTAINER_OFFSET];
			}
		}

		uiLastDrn = 0;
		pStack->pKeyBuf = ucKeyBuf;
		f_UINT32ToBigEndian( (FLMUINT32)uiStartDrn, ucSearchKey);

		if (RC_BAD( rc = FSBtSearch( pDb, pDataLFile, &pStack,
									ucSearchKey, 4, 0)))
		{
			goto Exit;
		}
		if( pStack->uiCmpStatus == BT_END_OF_DATA ||
			 pStack->uiBlkAddr == BT_END)
		{
			bHitEnd = TRUE;
			goto Next_Container;
		}
		pStack->uiFlags = NO_STACK;

		// Stack points to lowest leaf element (not root)

		for (;;)
		{
			if ((uiDrn = f_bigEndianToUINT32( ucKeyBuf)) == DRN_LAST_MARKER)
			{
				bHitEnd = TRUE;
				break;
			}

			// NOTE: uiEndDrn will be either DRN_LAST_MARKER, which means index
			// as far as we can, or it will be some value that we are to index
			// up to - set only when recovering or restoring from the RFL.

			if( uiDrn > uiEndDrn)
			{
				break;
			}

			// Check first if the record is in cache.

#ifdef FLM_DEBUG
			if( pReusableRec)
			{
				flmAssert( !pReusableRec->isCached());
			}
#endif
			bDataRecordRead = FALSE;
			if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, uiContainerNum,
				uiDrn, FALSE, NULL, NULL, &pRecord)))
			{
				if( rc != FERR_NOT_FOUND)
				{
					goto Exit;
				}

				// If records are equal, no need to do anything.
				// In fact the code would not work properly because the call
				// to Release might free the record, in which case m_pRec would
				// be pointing to freed space.  The AddRef would then be done on a
				// freed record.

				if( pRecord != pReusableRec)
				{
					// NOTE: If not found in cache, we deliberately don't bring it
					// in to cache.  Read the data record into memory, then build 
					// and commit its keys. 

					if( pRecord)
					{
						pRecord->Release();
					}
	
					pRecord = pReusableRec;
					if( pRecord)
					{
						pRecord->AddRef();
					}
				}

				if( RC_BAD( rc = FSReadElement( pDb, &ReadPool, pDataLFile,
					uiDrn, pStack, FALSE, &pRecord, NULL, NULL)))
				{
					goto Exit;
				}


				flmAssert( !pRecord->isCached());
				bDataRecordRead = TRUE;
			}

			bHadUniqueKeys = FALSE;
			if( RC_BAD(rc = flmProcessRecFlds( pDb, pIxd, uiContainerNum, uiDrn,
									pRecord, KREF_ADD_KEYS | KREF_INDEXING_ONLY,
									TRUE, &bHadUniqueKeys)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = KYProcessDupKeys( pDb, bHadUniqueKeys)))
			{
				goto Exit;
			}

			if( pIndexStatus)
			{
				pIndexStatus->uiKeysProcessed += 
					(FLMUINT)(pDb->KrefCntrl.uiCount - pDb->KrefCntrl.uiLastRecEnd);
			}

			KYFinishCurrentRecord( pDb);

			if (RC_BAD( rc = KYFlushKeys( pDb)))
			{
				goto Exit;
			}

			// The indexing code uses the temporary pool to allocate CDL table
			// entries.  Now that we are finished indexing the current record,
			// we need to reset the pool to free the CDL allocations.

			pDb->TempPool.poolReset( pvTmpPoolMark);

			// See if there is an indexing callback

			if (fnIxCallback)
			{
				if (pModRecord)
				{
					pModRecord->Release();
					pModRecord = NULL;
				}

				if (RC_BAD( rc = (*fnIxCallback)( (HFDB)pDb, uiIxNum, uiContainerNum,
									uiDrn, pRecord, &pModRecord, IxCallbackData)))
				{
					goto Exit;
				}

				// If the callback function sent back a changed record, we need
				// to do the modification to the record.  NOTE: This can only be
				// done AFTER adding the keys for the new index on the old record.
				// This call to FlmRecordModify will then change the keys if
				// necessary.

				if (pModRecord)
				{
					if (RC_BAD( rc = FlmRecordModify( (HFDB)pDb, uiContainerNum, uiDrn,
												pModRecord, 0)))
					{
						goto Exit;
					}

					// Need to readjust stack after modifying record.

					FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
					FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
					pStack = stackBuf;
					pStack->pKeyBuf = ucKeyBuf;
					f_UINT32ToBigEndian( (FLMUINT32)uiDrn, ucSearchKey);
					if (RC_BAD( rc = FSBtSearch( pDb, pDataLFile, &pStack,
												ucSearchKey, 4, 0)))
					{
						goto Exit;
					}

					// Should have found the record! We just modified it!

					if (pStack->uiCmpStatus != BT_EQ_KEY ||
						 pStack->uiBlkAddr == BT_END)
					{
						flmAssert( 0);
						rc = RC_SET( FERR_BTREE_ERROR);
						goto Exit;
					}
					pStack->uiFlags = NO_STACK;
					bDataRecordRead = FALSE;
				}
			}

			uiLastDrn = uiDrn;
			uiRecsProcessed++;

			if( pIndexStatus)
			{
				pIndexStatus->uiRecordsProcessed++;
				pIndexStatus->uiLastRecordIdIndexed = uiLastDrn;
			}

			// Get the current time

			uiCurrTime = FLM_GET_TIMER();

			// Break out if someone is waiting for an update transaction.

			if( pThread)
			{
				if( pThread->getShutdownFlag())
				{
					bRelinquish = TRUE;
					break;
				}

				if( pFileLockObj->getWaiterCount())
				{
					// See if our minimum run time has elapsed

					if( FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) >= uiMinTU)
					{
						if( uiRecsProcessed < 50)
						{
							// If there are higher priority waiters in the lock queue,
							// we want to relinquish.

							if( pDb->pFile->pFileLockObj->haveHigherPriorityWaiter( 
								FLM_BACKGROUND_LOCK_PRIORITY))
							{
								bRelinquish = TRUE;
								break;
							}
						}
						else
						{
							bRelinquish = TRUE;
							break;
						}
					}
				}
				else
				{
					// Even if no one has requested a lock for a long time, we
					// still want to periodically commit our transaction so 
					// we won't lose more than uiMaxCPInterval timer units worth
					// of work if we crash.  We will run until we exceed the checkpoint
					// interval and we see that someone (the checkpoint thread) is
					// waiting for the write lock.

					if( FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) > 
						gv_FlmSysData.uiMaxCPInterval && 
						pDb->pFile->pWriteLockObj->getWaiterCount())
					{
						bRelinquish = TRUE;
						break;
					}
				}
			}

			if( FLM_ELAPSED_TIME( uiCurrTime, uiLastStatusTime) >= 
				uiStatusIntervalTU)
			{
				uiLastStatusTime = uiCurrTime;
				if( fnStatus)
				{
					if( RC_BAD( rc = (*fnStatus)( FLM_INDEXING_STATUS, 
							(void *) uiLastDrn, 
							(void *) 0, 
							(void *) StatusData)))
					{
						goto Exit;
					}
				}

				// Send indexing completed event notification

				if( gv_FlmSysData.UpdateEvents.pEventCBList)
				{
					flmDoEventCallback( F_EVENT_UPDATES, 
							F_EVENT_INDEXING_COMPLETE, (void *)uiIxNum, 
							(void *)uiLastDrn);
				}

				// Log a progress message

				flmLogIndexingProgress( uiIxNum, uiLastDrn);
			}

			if (bDataRecordRead)
			{
				if (RC_BAD( rc = FSBtNextElm( pDb, pDataLFile, pStack)))
				{
					if (rc != FERR_BT_END_OF_DATA)
					{
						goto Exit;
					}
					rc = FERR_OK;
					bHitEnd = TRUE;
					break;
				}
			}
			else
			{
				// Need to go to the next record.

				if (RC_BAD( rc = FSNextRecord( pDb, pDataLFile, pStack)))
				{
					if (rc != FERR_BT_END_OF_DATA)
					{
						goto Exit;
					}
					rc = FERR_OK;
					bHitEnd = TRUE;
					break;
				}
			}
		}

Next_Container:

		if( !bDoAllContainers || bRelinquish)
		{
			break;
		}

		FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
		FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
		pStack = stackBuf;
	}

	if (RC_BAD( rc = KYKeysCommit( pDb, TRUE)))
	{
		goto Exit;
	}

	if (bHitEnd && !pIxd->uiContainerNum && !bDoAllContainers)
	{
		if (uiContainerNum != FLM_DATA_CONTAINER)
		{
			bHitEnd = FALSE;

			// Increment container number.  NOTE: This may not be
			// a valid container number, but we will determine the
			// next valid container number, the next time this
			// function is called by the background indexing
			// thread.

			uiContainerNum++;

			// Write a DRN of zero into the tracker so that the
			// next time in, we will start at the beginning
			// of this container.

			uiLastDrn = 0;
			bUpdateTracker = TRUE;
		}
	}
	
	// If at the end, change trans ID to the current transaction.

	if( bHitEnd)
	{
		// Create a new dictionary that will correctly reflect the
		// on-line status of the index once the transaction commits.

		if( (pDb->uiFlags & FDB_UPDATED_DICTIONARY) == 0)
		{
			if( RC_BAD( rc = fdictCloneDict( pDb)))
			{
				goto Exit;
			}
		}

		// NOTE: Always re-get the IXD prior to updating it, since the
		// dictionary may have been updated either by our thread (see
		// above call to flmCloneDict) or by another thread participating
		// in a shared transaction.

		if( RC_BAD( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIxNum, NULL, &pIxd, TRUE)))
		{
			goto Exit;
		}
		pIxd->uiFlags &= ~(IXD_SUSPENDED | IXD_OFFLINE);
		pIxd = NULL;

		if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, 
					uiIxNum, 0xFFFFFFFF,
					DRN_LAST_MARKER, pDb->LogHdr.uiCurrTransID, FALSE)))
		{
			goto Exit;
		}
	}
	else if( uiRecsProcessed || bUpdateTracker)
	{
		if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, 
					uiIxNum, uiContainerNum,
					uiLastDrn, TRANS_ID_OFFLINE, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	// We want to make one last call if we are in the foreground or if
	// we actually did some indexing.

	if( !pThread)
	{
		if( gv_FlmSysData.UpdateEvents.pEventCBList)
		{
			flmDoEventCallback( F_EVENT_UPDATES, 
					F_EVENT_INDEXING_COMPLETE, (void *)uiIxNum, 
					(void *)(bHitEnd ? 0 : uiLastDrn));
		}

		flmLogIndexingProgress( uiIxNum, bHitEnd ? 0 : uiLastDrn);
	}
	else if( uiLastDrn)
	{
		if( gv_FlmSysData.UpdateEvents.pEventCBList)
		{
			flmDoEventCallback( F_EVENT_UPDATES, 
					F_EVENT_INDEXING_COMPLETE, (void *)uiIxNum, 
					(void *)uiLastDrn);
		}

		flmLogIndexingProgress( uiIxNum, uiLastDrn);
	}

	if( fnStatus)
	{
		(void) (*fnStatus)( FLM_INDEXING_STATUS,
					(void *) uiLastDrn, 
					(void *) 0, 
					(void *) StatusData);
	}

	if( pbHitEnd)
	{
		*pbHitEnd = bHitEnd;
	}

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	KrefCntrlFree( pDb);
	pDb->TempPool.poolReset( pvTmpPoolMark);

	if( pReusableRec)
	{
		flmAssert( !pReusableRec->isCached());
		pReusableRec->Release();
	}

	if( pRecord)
	{
		pRecord->Release();
	}

	if (pModRecord)
	{
		pModRecord->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Add fixups for indexes that are created or indexed during the
		transaction.
****************************************************************************/
FSTATIC RCODE flmAddIndexFixup(
	FDB *		pDb,
	FLMUINT	uiIndexNum,
	FLMUINT	uiLastDrnIndexed,
	FLMUINT	uiLastContainerIndexed)
{
	RCODE			rc = FERR_OK;
	IXD_FIXUP *	pIxdFixup;

	// See if this index is in our fixup list.

	pIxdFixup = pDb->pIxdFixups;
	while (pIxdFixup && pIxdFixup->uiIndexNum != uiIndexNum)
	{
		pIxdFixup = pIxdFixup->pNext;
	}

	if (!pIxdFixup)
	{
		if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( IXD_FIXUP), &pIxdFixup)))
		{
			goto Exit;
		}

		pIxdFixup->pNext = pDb->pIxdFixups;
		pDb->pIxdFixups = pIxdFixup;
		pIxdFixup->uiIndexNum = uiIndexNum;
	}

	pIxdFixup->uiLastContainerIndexed = uiLastContainerIndexed;
	pIxdFixup->uiLastDrnIndexed = uiLastDrnIndexed;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Set information in the tracker record for the index.
****************************************************************************/
RCODE flmSetIxTrackerInfo(
	FDB *					pDb, 
	FLMUINT				uiIndexNum,
	FLMUINT				uiLastContainerIndexed,
	FLMUINT				uiLastDrnIndexed,
	FLMUINT				uiOnlineTransId,
	FLMBOOL 				bSuspended)
{
	RCODE				rc = FERR_OK;
	LFILE	 *			pLFile;
	FlmRecord *		pRecord = NULL;
	FLMUINT32		ui32LastDrnIndexed = (FLMUINT32)uiLastDrnIndexed;
	FLMUINT32		ui32LastContainerIndexed = (FLMUINT32)uiLastContainerIndexed;
	FLMUINT32		ui32OnlineTransId = (FLMUINT32)uiOnlineTransId;

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
		FLM_TRACKER_CONTAINER, &pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
		FLM_TRACKER_CONTAINER, uiIndexNum, TRUE, NULL, pLFile, &pRecord)))
	{
		if( rc != FERR_NOT_FOUND )
		{
			goto Exit;
		}

		// Create the record for the first time. 

		if( (pRecord = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pRecord->insertLast( 0, FLM_INDEX_TAG,
			FLM_CONTEXT_TYPE, NULL)))
		{
			goto Exit;
		}
	}

	// Make writable.

	if( pRecord && pRecord->isReadOnly())
	{
		FlmRecord *	pTmpRec;
		
		if( (pTmpRec = pRecord->copy()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		pRecord->Release();
		pRecord = pTmpRec;
	}

	if( RC_BAD( rc = flmModField( pRecord, FLM_LAST_DRN_INDEXED_TAG,
		&ui32LastDrnIndexed, 4, FLM_NUMBER_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmModField( pRecord, FLM_LAST_CONTAINER_INDEXED_TAG,
		&ui32LastContainerIndexed, 4, FLM_NUMBER_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmModField( pRecord, FLM_ONLINE_TRANS_ID_TAG,
		&ui32OnlineTransId, 4, FLM_NUMBER_TYPE)))
	{
		goto Exit;
	}

	if( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_51)
	{
		FLMUINT32	ui32IndexSuspended = bSuspended ? 1 : 0;
		if( RC_BAD( rc = flmModField( pRecord, FLM_STATE_TAG,
			&ui32IndexSuspended, 4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Update the record.

	if( RC_BAD( rc = FSRecUpdate( pDb, pLFile, pRecord, uiIndexNum, 0)))
	{
		goto Exit;
	}

	pRecord->setID( uiIndexNum);
	pRecord->setContainerID( FLM_TRACKER_CONTAINER);

	if( RC_BAD( rc = flmRcaInsertRec( pDb, pLFile, uiIndexNum, pRecord)))
	{
		goto Exit;
	}

	// Update the uiLastDrnIndexed and uiLastContainerIndexed
	// in the IXD structure, but be sure to save
	// the fields in a FIXUP structure so that the IXD can be
	// fixed up when we commit.

	if (RC_BAD( rc = flmAddIndexFixup( pDb, uiIndexNum,
		uiLastDrnIndexed, uiLastContainerIndexed)))
	{
		goto Exit;
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:	Get the last drn indexed value in an index tracker record.
		Returns DRN_LAST_MARKER if the tracker record is not there
		or the last drn indexed field is not there.
****************************************************************************/
RCODE flmGetIxTrackerInfo(
	FDB *					pDb, 
	FLMUINT				uiIndexNum,
	FLMUINT *			puiLastContainerIndexed,
	FLMUINT *			puiLastDrnIndexed,
	FLMUINT *			puiOnlineTransId,
	FLMBOOL *			pbSuspended)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLastContainerIndexed = 0xFFFFFFFF;
	FLMUINT			uiLastDrnIndexed = DRN_LAST_MARKER;
	FLMUINT			uiOnlineTransId = TRANS_ID_ALWAYS_ONLINE;
	FLMUINT			uiIndexSuspended = 0;
	LFILE	 *			pLFile;
	FlmRecord *		pRecord = NULL;
	void *			pvField;

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, FLM_TRACKER_CONTAINER, &pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiIndexNum, &pRecord, NULL, NULL)))
	{
		if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		rc = FERR_OK;
	}
	else
	{
		if( (pvField = pRecord->find( pRecord->root(),
										FLM_LAST_CONTAINER_INDEXED_TAG, 1)) != NULL)
		{
			if( RC_BAD( rc = pRecord->getUINT( pvField, &uiLastContainerIndexed)))
			{
				goto Exit;
			}
		}

		if( (pvField = pRecord->find( pRecord->root(),
										FLM_LAST_DRN_INDEXED_TAG, 1)) != NULL)
		{
			if( RC_BAD( rc = pRecord->getUINT( pvField, &uiLastDrnIndexed)))
			{
				goto Exit;
			}
		}

		if( (pvField = pRecord->find( pRecord->root(),
										FLM_ONLINE_TRANS_ID_TAG, 1)) != NULL)
		{
			if( RC_BAD( rc = pRecord->getUINT( pvField, &uiOnlineTransId)))
			{
				goto Exit;
			}
		}

		if( (pvField = pRecord->find( pRecord->root(), 
			FLM_STATE_TAG, 1)) != NULL)
		{
			if( RC_BAD( rc = pRecord->getUINT( pvField, &uiIndexSuspended)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	if( puiLastContainerIndexed)
	{
		*puiLastContainerIndexed = uiLastContainerIndexed;
	}

	if( puiLastDrnIndexed)
	{
		*puiLastDrnIndexed = uiLastDrnIndexed;
	}

	if( puiOnlineTransId)
	{
		*puiOnlineTransId = uiOnlineTransId;
	}

	if( pbSuspended)
	{
		*pbSuspended = uiIndexSuspended ? TRUE : FALSE;
	}

	return( rc);
}

/***************************************************************************
Desc:		See if any IXD structures need indexing in the background.
****************************************************************************/
RCODE flmStartBackgrndIxThrds(
	FDB *			pDb)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bStartedAutoTrans;
	FLMUINT			uiLoop;
	IXD *				pIxd;

	bStartedAutoTrans = FALSE;
	if( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
		FDB_TRANS_GOING_OK, 0, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	for( uiLoop = 0, pIxd = pDb->pDict->pIxdTbl;
		uiLoop < pDb->pDict->uiIxdCnt;
		uiLoop++, pIxd++)
	{
		// Restart any indexes that are off-line but not suspended

		if( (pIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) == IXD_OFFLINE)
		{
			flmAssert( flmBackgroundIndexGet( pDb->pFile,
				pIxd->uiIndexNum, FALSE) == NULL);

			if( RC_BAD( rc = flmStartIndexBuild( pDb, pIxd->uiIndexNum)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( bStartedAutoTrans)
	{
		(void)flmAbortDbTrans( pDb);
	}
	
	fdbExit( pDb);
	return( rc);
}

/***************************************************************************
Desc:		Frees blocks of a container
****************************************************************************/
FSTATIC RCODE flmFreeLFileBlocks(
	FDB *			pDb,
	LFILE *		pLFile)
{
	RCODE			rc = FERR_OK;
	BTSK *		pStack;
	BTSK *		pOrigStack;
	SCACHE *		pSCache = NULL;
	FLMUINT		uiBlkAddr;
	BTSK			stackBuf[ BH_MAX_LEVELS];
	FLMBYTE		ucKeyBuf[ MAX_KEY_SIZ];
	FLMUINT		uiBlocksDeleted = 0;
	FLMUINT		uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;
	FLMUINT		uiLoopCounter = 0;

	FSInitStackCache( &stackBuf[ 0], BH_MAX_LEVELS);
	pStack = pOrigStack = stackBuf;
	pStack->pKeyBuf = ucKeyBuf;
	ucKeyBuf[ 0] = 0;
	
	if( RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, ucKeyBuf, 1, 0)))
	{
		goto Exit;
	}

	// After calling FSBtSearch, the bottom block in the stack will still be
	// held.  We release it here, because it will be read again in the
	// loops below.

	FSReleaseBlock( pStack, FALSE);

	// Only need to delete blocks if there is something in the tree 

	if( pStack->uiCmpStatus != BT_END_OF_DATA)
	{
		// Stack points to lowest leaf element (not root)
		// Increment for do-while loop and then post decrement.
		// This is a tricky algorithm!

		pStack++;
		do
		{
			pStack--;
			uiBlkAddr = pStack->uiBlkAddr;

			do
			{
				f_yieldCPU();

				// Read block so we can free it 

				if( RC_BAD(rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
										uiBlkAddr, NULL, &pSCache)))
				{
					goto Exit;
				}

				// Get next block address before blkFree creams it out 

				uiBlkAddr = FB2UD( &pSCache->pucBlk[ BH_NEXT_BLK]);

				// The call to FSBlockFree also releases the cache whether
				// it succeeds or not.  That is why we set pSCache to NULL after
				// calling it.

				rc = FSBlockFree( pDb, pSCache);
				pSCache = NULL;
				if( RC_BAD( rc))
				{
					goto Exit;
				}

				// See if we should do a callback

				uiBlocksDeleted++;
				uiLoopCounter++;
				if( pDb->fnStatus && uiLoopCounter == 50)
				{
					uiLoopCounter = 0;
					if( RC_BAD( rc = pDb->fnStatus( FLM_DELETING_STATUS,
											(void *)uiBlocksDeleted,
											(void *)uiBlockSize,
											pDb->StatusData)))
					{
						goto Exit;
					}
				}
			} while( uiBlkAddr != BT_END);
	
		} while( pStack != pOrigStack);
	}

Exit:

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Removes all keys in the index that refer to the specified
		container.
****************************************************************************/
RCODE flmRemoveContainerKeys(
	FDB *			pDb,
	FLMUINT		uiIndexNum,
	FLMUINT		uiContainerNum)
{
	RCODE			rc = FERR_OK;
	BTSK			stack[ BH_MAX_LEVELS];
	BTSK *		pStack;
	FLMBYTE *	pucKeyBufs = NULL;
	LFILE *		pIxLFile;
	FLMUINT		uiLoopCounter = 0;
	FLMUINT		uiElementsTraversed = 0;
	FLMUINT		uiContainerPartLen;
	IXD *			pIxd;

	FSInitStackCache( &stack[0], BH_MAX_LEVELS);

	if( RC_BAD( rc = f_alloc( MAX_KEY_SIZ * 2, &pucKeyBufs)))
	{
		goto Exit;
	}

	// Get the index LFILE.

	if (RC_BAD( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIndexNum, &pIxLFile, &pIxd, TRUE)))
	{
		goto Exit;
	}

	pStack = stack;
	pStack->pKeyBuf = &pucKeyBufs [MAX_KEY_SIZ];
	pucKeyBufs [0] = 0;

	if (RC_BAD( rc = FSBtSearch( pDb, pIxLFile, &pStack, pucKeyBufs, 1, 0)))
	{
		goto Exit;
	}

	// If the B-Tree was empty, we are done.

	if( pStack->uiBlkAddr == BT_END)
	{
		goto Exit;	// Should return FERR_OK
	}

	uiContainerPartLen = getIxContainerPartLen( pIxd);

	// Traverse through the entire index, deleting elements that have this
	// container in their key.

	for (;;)
	{
		// Key must have at least the number of bytes for the container
		// at the end of the key.

		if (pStack->uiKeyLen <= uiContainerPartLen)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_BTREE_ERROR);
			goto Exit;
		}

		if( uiContainerNum == getContainerFromKey(
				pStack->pKeyBuf, pStack->uiKeyLen))
		{
			FLMBYTE *	pucCurElm;

			if (RC_BAD( rc = FSBtDelete( pDb, pIxLFile, &pStack)))
			{
				goto Exit;
			}

			// Delete all continuation elements as well.

			pucCurElm = CURRENT_ELM( pStack);
			while (!BBE_IS_FIRST( pucCurElm))
			{
				if (RC_BAD( rc = FSBtDelete( pDb, pIxLFile, &pStack)))
				{
					goto Exit;
				}

				pucCurElm = CURRENT_ELM( pStack);
				uiElementsTraversed++;
			}

			// See if we are at the end of the index.

			if (pStack->uiBlkAddr == BT_END)
			{
				goto Exit;	// Should return FERR_OK
			}

			// Scan to current position to rebuild the key.

			if (RC_BAD(rc = FSBtScanTo( pStack, NULL, 0,  0)))
			{
				goto Exit;
			}
		}
		else
		{
			// Go to the next element in the B-Tree.

			if( RC_BAD( rc = FSBtNextElm( pDb, pIxLFile, pStack)))
			{
				if( rc != FERR_BT_END_OF_DATA)
				{
					goto Exit;
				}

				// rc was FERR_BT_END_OF_DATA, we are at end of index.

				rc = FERR_OK;
				break;
			}
		}

		uiElementsTraversed++;
		uiLoopCounter++;

		if (pDb->fnStatus && uiLoopCounter == 50)
		{
			uiLoopCounter = 0;
			if (RC_BAD( rc = pDb->fnStatus( FLM_DELETING_KEYS,
									(void *)uiIndexNum,
									(void *)uiElementsTraversed,
									pDb->StatusData)))
			{
				goto Exit;
			}
		}
	}

Exit:

	FSReleaseStackCache( stack, BH_MAX_LEVELS, FALSE);

	if (pucKeyBufs)
	{
		f_free( &pucKeyBufs);
	}

	return( rc);
}

/***************************************************************************
Desc:	Remove indexes associated with a container.  Also, for indexes that
		span all containers, remove any keys/references in the index that
		refer to this container.
****************************************************************************/
FSTATIC RCODE fdictRemoveIndexes(
	FDB *		pDb,
	FLMUINT		uiContainer)
{
	RCODE		rc = FERR_OK;
	IXD *		pIxd;
	FLMUINT	uiCnt;

	// Go through all of the IXDs

	for( pIxd = pDb->pDict->pIxdTbl, uiCnt = 0;
		  uiCnt < pDb->pDict->uiIxdCnt;
		  uiCnt++, pIxd++)
	{
		if( pIxd->uiContainerNum == uiContainer)
		{
			// Index is on this container, remove the index.
			// NOTE: For now, this case is not allowed.  Indexes on a container
			// must be deleted prior to deleting the container.

			rc = RC_SET( FERR_MUST_DELETE_INDEXES);
			goto Exit;
		}
		else if( !pIxd->uiContainerNum)
		{

			// Index is on all containers, must traverse through the index and
			// remove all keys that refer to this container.

			if( RC_BAD( rc = flmRemoveContainerKeys( pDb, pIxd->uiIndexNum,
										uiContainer)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Check that the field type is not changed.
****************************************************************************/
FSTATIC RCODE DDVerifyModField(
	FDB *				pDb,
	FlmRecord *		pOldRecord,
	FlmRecord *		pNewRecord
	)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNewFldInfo, uiOldFldInfo;
	void *		pvField;
	char			szNativeBuf1[ 32];
	char			szNativeBuf2[ 32];
	FLMUINT		uiLen;

	// Fields must have the same type.  May only change the field name.
	// Set Default in case no type.
	
	uiNewFldInfo = uiOldFldInfo = FLM_CONTEXT_TYPE;	
	if( (pvField = pNewRecord->find( pNewRecord->root(), FLM_TYPE_TAG)) != NULL)
	{
		if( RC_BAD(rc = DDGetFieldType( pNewRecord, 
			pvField, &uiNewFldInfo)))
		{
			goto Exit;
		}
	}

	if( (pvField = pOldRecord->find( pOldRecord->root(), 
												FLM_TYPE_TAG)) != NULL)
	{
		if( RC_BAD(rc = DDGetFieldType( pOldRecord, pvField, &uiOldFldInfo)))
		{
			goto Exit;
		}
		if( uiNewFldInfo != uiOldFldInfo)
		{
			rc = RC_SET( FERR_CANNOT_MOD_FIELD_TYPE);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_MISSING_FIELD_TYPE);
		goto Exit;
	}

	// Determine if the field state is changing
	//   1) The new record must have a state value of 'checking' or 'purge'
	//   2) or a state of 'unused' (FlmDbSweep or recover only)

	if( !pDb->bFldStateUpdOk)
	{
		szNativeBuf1[ 0] = 0;
		if( (pvField = pOldRecord->find( pOldRecord->root(),
													FLM_STATE_TAG)) != NULL)
		{
			uiLen = sizeof( szNativeBuf1);
			if( RC_BAD( rc = pOldRecord->getNative( 
				pvField, szNativeBuf1, &uiLen)))
			{
				goto Exit;
			}
		}

		szNativeBuf2[ 0] = 0;
		if( (pvField = pNewRecord->find( pNewRecord->root(),
													FLM_STATE_TAG)) != NULL)
		{
			uiLen = sizeof( szNativeBuf2);
			if( RC_BAD( rc = pNewRecord->getNative( pvField,
																 szNativeBuf2,
																 &uiLen)))
			{
				goto Exit;
			}
		}

		if( f_strnicmp( szNativeBuf1, szNativeBuf2, 4) != 0 &&
			 f_strnicmp( szNativeBuf2, "chec", 4) != 0 &&
			 f_strnicmp( szNativeBuf2, "purg", 4) != 0)
		{
			rc = RC_SET( FERR_CANNOT_MOD_FIELD_STATE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Check that the encryption type and key have not changed.
****************************************************************************/
FSTATIC RCODE DDVerifyModEncDef(
	FDB *				pDb,
	FlmRecord *		pOldRecord,
	FlmRecord *		pNewRecord
	)
{
	RCODE			rc = FERR_OK;
	void *		pvOldField;
	void *		pvNewField;
	FLMUINT		uiOldEncType;
	FLMUINT		uiNewEncType;
	char			szNativeBuf1[ 32];
	char			szNativeBuf2[ 32];
	FLMUINT		uiLen;
	FLMUINT		uiOldKeyLen;
	FLMUINT		uiNewKeyLen;
	
	if ((pvOldField = pOldRecord->find( pOldRecord->root(),
												  FLM_TYPE_TAG)) == NULL)
	{
		rc = RC_SET( FERR_MISSING_ENC_TYPE);
		goto Exit;
	}

	// Verify that the encryption algorithm does not change and that
	// the key is not being changed.
	if (RC_BAD( rc = DDGetEncType( pOldRecord, pvOldField, &uiOldEncType)))
	{
		goto Exit;
	}

	if( (pvNewField = pNewRecord->find( pNewRecord->root(), 
													FLM_TYPE_TAG)) == NULL)
	{
		rc = RC_SET( FERR_MISSING_ENC_TYPE);
		goto Exit;
	}

	if (RC_BAD( rc = DDGetEncType( pNewRecord, pvNewField, &uiNewEncType)))
	{
		goto Exit;
	}

	// Make sure we are not trying to change the encryption type.
	if (uiNewEncType != uiOldEncType)
	{
		rc = RC_SET( FERR_CANNOT_MOD_ENC_TYPE);
		goto Exit;
	}

	// Make sure the key is not being modified
	if( (pvOldField = pOldRecord->find( pOldRecord->root(), 
													FLM_KEY_TAG)) == NULL)
	{
		rc = RC_SET( FERR_MISSING_ENC_KEY);
		goto Exit;
	}

	if ((pvNewField = pNewRecord->find( pNewRecord->root(),
												  FLM_KEY_TAG)) == NULL)
	{
		rc = RC_SET( FERR_MISSING_ENC_KEY);
		goto Exit;
	}
							
	uiOldKeyLen = pOldRecord->getDataLength( pvOldField);
	
	uiNewKeyLen = pNewRecord->getDataLength( pvNewField);

	if (uiOldKeyLen != uiNewKeyLen)
	{
		rc = RC_SET( FERR_CANNOT_CHANGE_KEY);
		goto Exit;
	}
	
	if (uiOldKeyLen == 0)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_BAD_ENC_KEY);
		goto Exit;
	}
	
	if (f_memcmp(pOldRecord->getDataPtr(pvOldField),
					 pNewRecord->getDataPtr(pvNewField),
					 uiNewKeyLen) != 0)
	{
		rc = RC_SET( FERR_CANNOT_CHANGE_KEY);
		goto Exit;
	}

	/* Determine if the record state is changing
		1) The new record must have a state value of 'checking' or 'purge'
		2) or a state of 'unused' (FlmDbSweep or recover only) */

	if( !pDb->bFldStateUpdOk)
	{
		szNativeBuf1[ 0] = 0;
		if( (pvOldField = pOldRecord->find(	pOldRecord->root(),
													FLM_STATE_TAG)) != NULL)
		{
			uiLen = sizeof( szNativeBuf1);
			if( RC_BAD( rc = pOldRecord->getNative( pvOldField,
																 szNativeBuf1,
																 &uiLen)))
			{
				goto Exit;
			}
		}

		szNativeBuf2[ 0] = 0;
		if( (pvNewField = pNewRecord->find( pNewRecord->root(),
														FLM_STATE_TAG)) != NULL)
		{
			uiLen = sizeof( szNativeBuf2);
			if( RC_BAD( rc = pNewRecord->getNative( pvNewField,
																 szNativeBuf2,
																 &uiLen)))
			{
				goto Exit;
			}
		}

		if( f_strnicmp( szNativeBuf1, szNativeBuf2, 4) != 0 &&
			 f_strnicmp( szNativeBuf2, "chec", 4) != 0 &&
			 f_strnicmp( szNativeBuf2, "purg", 4) != 0)
		{
			rc = RC_SET( FERR_CANNOT_MOD_ENC_STATE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE flmFreeContainerBlocks(
	FDB *			pDb,
	LFILE *		pLFile)
{
	RCODE			rc = FERR_OK;
	LFILE			tmpLFile;

	flmAssert( pLFile->uiLfType == LF_CONTAINER);
	
	if( RC_BAD( rc = flmFreeLFileBlocks( pDb, pLFile)))
	{
		goto Exit;
	}

	f_memcpy( &tmpLFile, pLFile, sizeof( LFILE));
	tmpLFile.uiLfType = LF_INVALID;

	// Update the LFile
	
	if( RC_BAD( rc = flmLFileWrite( pDb, &tmpLFile)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Frees up all index blocks
****************************************************************************/
FSTATIC RCODE flmFreeIndexBlocks(
	FDB *			pDb,
	LFILE *		pLFile,
	FLMBOOL		bInvalidateLFile)
{
	RCODE				rc = FERR_OK;
	BTSK				stackBuf[ BH_MAX_LEVELS];
	SCACHE *			pSCache = NULL;
	FLMUINT			uiDrn;
	FlmRecord *		pRec = NULL;
	void *			pvFld;
	FFILE *			pFile = pDb->pFile;
	LFILE *			pTrackerLFile;
	LFILE				tmpLFile;
	char				szTmpBuf[ 32];

	flmAssert( pLFile->uiLfType == LF_INDEX);
	FSInitStackCache( &stackBuf[ 0], BH_MAX_LEVELS);
	
	// Delete the index tracker record, if any

	if( RC_BAD( rc = flmDeleteTrackerRec( pDb, pLFile->uiLfNum)))
	{
		goto Exit;
	}

	if( pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_51)
	{
		// Background deletion is not supported.  Must delete
		// the blocks of the LFILE now.

		if( RC_BAD( rc = flmFreeLFileBlocks( pDb, pLFile)))
		{
			goto Exit;
		}
	}
	else
	{
		BTSK *		pStack;
		BTSK *		pOrigStack;
		FLMBYTE		ucKeyBuf[ MAX_KEY_SIZ];

		// Delete the index in the background

		pStack = pOrigStack = stackBuf;
		pStack->pKeyBuf = ucKeyBuf;
		ucKeyBuf[ 0] = 0;

		if( RC_BAD( rc = FSBtSearch( pDb, pLFile, 
			&pStack, ucKeyBuf, 1, 0)))
		{
			goto Exit;
		}

		// After calling FSBtSearch, the bottom block in the stack 
		// will still be held.  We release it here.

		FSReleaseBlock( pStack, FALSE);

		// If the tree is empty, we don't need to delete anything

		if( pStack->uiCmpStatus != BT_END_OF_DATA)
		{
			// Build the tracker record for the background deletion
			// thread and signal the thread that it has work to do.

			if( (pRec = f_new FlmRecord) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 0, FLM_DELETE_TAG,
				FLM_TEXT_TYPE, NULL)))
			{
				goto Exit;
			}

			pStack++;
			do
			{
				pStack--;

				if( RC_BAD( rc = pRec->insertLast( 1, FLM_BLOCK_CHAIN_TAG,
					FLM_TEXT_TYPE, &pvFld)))
				{
					goto Exit;
				}

				f_sprintf( szTmpBuf, "%u", (unsigned)pStack->uiBlkAddr);
				if( RC_BAD( rc = pRec->setNative( pvFld, szTmpBuf)))
				{
					goto Exit;
				}

			} while( pStack != pOrigStack);

			// Get the tracker LFILE

			if( RC_BAD( rc = fdictGetContainer( 
				pDb->pDict, FLM_TRACKER_CONTAINER, &pTrackerLFile)))
			{
				goto Exit;
			}

			// Find an available DRN in the tracker container

			uiDrn = 0;
			if( RC_BAD( rc = FSGetNextDrn( pDb, 
				pTrackerLFile, FALSE, &uiDrn)))
			{
				goto Exit;
			}

			if( uiDrn <= 0x0000FFFF)
			{
				// Skip past the reserved and unregistered ID range

				uiDrn = 0x00010000;
			}

			pRec->setID( uiDrn);
			pRec->setContainerID( FLM_TRACKER_CONTAINER);

			// Add the record to the tracker container

			if( RC_BAD( rc = FSRecUpdate( pDb, pTrackerLFile, 
				pRec, uiDrn, 0)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = flmRcaInsertRec( pDb, pTrackerLFile, uiDrn, pRec)))
			{
				goto Exit;
			}

			pRec->Release();
			pRec = NULL;

			// Signal the maintenance thread that it has work to do

			f_semSignal( pFile->hMaintSem);
		}
	}

	f_memcpy( &tmpLFile, pLFile, sizeof( LFILE));

	if( bInvalidateLFile)
	{
		tmpLFile.uiLfType = LF_INVALID;
	}
	else
	{
		tmpLFile.uiRootBlk = BT_END;
	}

	// Update the LFile
	
	if( RC_BAD( rc = flmLFileWrite( pDb, &tmpLFile)))
	{
		goto Exit;
	}

Exit:

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( pRec)
	{
		pRec->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE flmFreeBlockChain(
	FDB *			pDb,
	FLMUINT		uiStartAddr,
	FLMUINT *	puiCount,
	FLMUINT *	puiEndAddr)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pSCache = NULL;
	FLMUINT		uiBlkAddr = uiStartAddr;
	FLMUINT		uiBlocksFreed = 0;
	FLMUINT		uiNumToDelete = *puiCount;

	// Make sure an update transaction is going

	if( flmGetDbTransType( pDb) != FLM_UPDATE_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Free blocks in the chain until we hit the end or meet
	// our quota.

	while( uiBlkAddr != BT_END && uiBlocksFreed < uiNumToDelete)
	{
		if( RC_BAD(rc = ScaGetBlock( pDb, NULL, BHT_FREE,
			uiBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}

		// Get next block address before blkFree creams it out 

		uiBlkAddr = FB2UD( &pSCache->pucBlk[ BH_NEXT_BLK]);

		// The call to FSBlockFree also releases the cache whether
		// it succeeds or not.  That is why we set pSCache to NULL after
		// calling it.

		rc = FSBlockFree( pDb, pSCache);
		pSCache = NULL;

		if( RC_BAD( rc))
		{
			goto Exit;
		}

		uiBlocksFreed++;
	}

	*puiEndAddr = uiBlkAddr;
	*puiCount = uiBlocksFreed;

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE flmMaintFreeBlockChain(
	FDB *				pDb,
	FLMUINT			uiTrackerDrn,
	FLMUINT 			uiBlocksToDelete,
	FLMUINT			uiExpectedEndAddr,
	FLMUINT64 *		pui64BlocksFreed)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiStartAddr;
	FLMUINT			uiEndAddr = 0;
	void *			pvBlkChain;
	FlmRecord *		pTrackerRec = NULL;
	FlmRecord *		pTmpRec;
	FLMUINT			uiBlocksFreed = 0;
	FLMUINT			uiTmpCount;
	FLMBOOL			bUpdateTrackerRec = FALSE;

	// Make sure an update transaction is going and that a
	// non-zero number of blocks was specified

	if( flmGetDbTransType( pDb) != FLM_UPDATE_TRANS || !uiBlocksToDelete)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Retrieve the record

	if( RC_BAD( rc = flmRetrieveTrackerRec( pDb, 
		uiTrackerDrn, TRUE, &pTrackerRec)))
	{
		flmAssert( 0);
		goto Exit;
	}

	// Make a writeable copy of the record

	if( pTrackerRec->isReadOnly())
	{
		if( (pTmpRec = pTrackerRec->copy()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pTrackerRec->Release();
		pTrackerRec = pTmpRec;
		pTmpRec = NULL;
	}

	while( uiBlocksFreed < uiBlocksToDelete)
	{
		// Process the block chain(s)
		
		if( (pvBlkChain = pTrackerRec->find( 
			pTrackerRec->root(), FLM_BLOCK_CHAIN_TAG)) == NULL)
		{
			if( RC_BAD( rc = flmDeleteTrackerRec( 
				pDb, pTrackerRec->getID())))
			{
				goto Exit;
			}

			bUpdateTrackerRec = FALSE;
			break;
		}

		if( RC_BAD( rc = pTrackerRec->getUINT( 
			pvBlkChain, &uiStartAddr)))
		{
			goto Exit;
		}

		uiTmpCount = uiBlocksToDelete - uiBlocksFreed;

		if( RC_BAD( rc = flmFreeBlockChain( 
			pDb, uiStartAddr, &uiTmpCount, &uiEndAddr)))
		{
			goto Exit;
		}

		uiBlocksFreed += uiTmpCount;
		flmAssert( uiBlocksFreed <= uiBlocksToDelete);

		if( uiEndAddr == BT_END)
		{
			// If we hit the end of the block chain, remove
			// it from the tracker record

			pTrackerRec->remove( pvBlkChain);
			bUpdateTrackerRec = TRUE;
		}
		else
		{
			// More work to do on the current block chain.  Update
			// the tracker.

			if( RC_BAD( rc = pTrackerRec->setUINT( pvBlkChain, uiEndAddr)))
			{
				goto Exit;
			}
			bUpdateTrackerRec = TRUE;
		}
	}

	if( bUpdateTrackerRec)
	{
		if( RC_BAD( rc = flmModifyTrackerRec( 
			pDb, pTrackerRec->getID(), pTrackerRec)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pDb->pFile->pRfl->logBlockChainFree( 
		uiTrackerDrn, uiBlocksFreed, uiEndAddr)))
	{
		goto Exit;
	}

	if( uiExpectedEndAddr)
	{
		if( uiBlocksToDelete != uiBlocksFreed ||
			uiEndAddr != uiExpectedEndAddr)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_BAD_RFL_PACKET);
			goto Exit;
		}
	}

Exit:

	if( pui64BlocksFreed)
	{
		*pui64BlocksFreed += uiBlocksFreed;
	}

	if( pTrackerRec)
	{
		pTrackerRec->Release();
	}

	return( rc);
}


/***************************************************************************
Desc: This routine begins a thread that will perform misc. background work
		assigned to the tracker.
*****************************************************************************/
RCODE flmStartMaintThread(
	FFILE *		pFile)
{
	RCODE			rc = FERR_OK;
	char			szThreadName[ F_PATH_MAX_SIZE];
	char			szBaseName[ F_FILENAME_SIZE];

	flmAssert( !pFile->pMaintThrd);

	// Generate the thread name

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
		pFile->pszDbPath, szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( szThreadName, 
		"Maintenance (%s)", szBaseName);

	// Initialize the tracker status structure

	f_memset( &pFile->maintStatus, 0, sizeof( FMAINT_STATUS));

	// Start the tracker thread.

	if( RC_BAD( rc = f_threadCreate( &pFile->pMaintThrd,
		flmMaintThread, szThreadName, 
		0, 0, pFile, NULL, 32000)))
	{
		goto Exit;
	}

	// Signal the thread so that it will look for any
	// work that may already be waiting

	f_semSignal( pFile->hMaintSem);

Exit:

	if( RC_BAD( rc))
	{
		if( pFile->pMaintThrd)
		{
			pFile->pMaintThrd->Release();
			pFile->pMaintThrd = NULL;
		}
	}

	return( rc);
}

/***************************************************************************
Desc: 
*****************************************************************************/
FSTATIC RCODE flmRetrieveTrackerRec(
	FDB *				pDb,
	FLMUINT			uiDrn,
	FLMBOOL			bExact,
	FlmRecord **	ppRecord)
{
	RCODE				rc = FERR_OK;
	F_Pool			readPool;
	LFILE *			pTrackerLFile;
	BTSK  			stackBuf[ BH_MAX_LEVELS];
	BTSK *			pStack = &stackBuf[ 0];
	FLMBYTE			ucKeyBuf[ DIN_KEY_SIZ];
	FLMBYTE			ucSearchKey[ DIN_KEY_SIZ];
	FlmRecord *		pTrackerRec = NULL;
	FLMUINT			uiFoundDrn;

	FSInitStackCache( &stackBuf[ 0], BH_MAX_LEVELS);
	readPool.poolInit( 8192);

	// Retrieve a tracker record for processing

	if( RC_BAD( rc = fdictGetContainer( 
		pDb->pDict, FLM_TRACKER_CONTAINER, &pTrackerLFile)))
	{
		goto Exit;
	}

	pStack->pKeyBuf = ucKeyBuf;
	f_UINT32ToBigEndian( (FLMUINT32)uiDrn, ucSearchKey);

	if( RC_BAD( rc = FSBtSearch( 
		pDb, pTrackerLFile, &pStack, ucSearchKey, 4, 0)))
	{
		goto Exit;
	}

	if( pStack->uiCmpStatus == BT_END_OF_DATA ||
		 pStack->uiBlkAddr == BT_END)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	pStack->uiFlags = NO_STACK;

	// Stack points to leaf element

	if( (uiFoundDrn = f_bigEndianToUINT32( ucKeyBuf)) == DRN_LAST_MARKER)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	if( bExact)
	{
		if( uiFoundDrn != uiDrn)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}
	else
	{
		uiDrn = uiFoundDrn;
	}

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, 
		FLM_TRACKER_CONTAINER, uiDrn, FALSE, NULL, 
		NULL, &pTrackerRec)))
	{
		if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		if( RC_BAD( rc = FSReadElement( pDb, &readPool, 
			pTrackerLFile, uiDrn, pStack, FALSE, &pTrackerRec,
			NULL, NULL)))
		{
			goto Exit;
		}
	}

	if( *ppRecord)
	{
		(*ppRecord)->Release();
	}

	*ppRecord = pTrackerRec;
	pTrackerRec = NULL;

Exit:

	if( pTrackerRec)
	{
		pTrackerRec->Release();
	}

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc);
}

/***************************************************************************
Desc: 
*****************************************************************************/
FSTATIC RCODE flmDeleteTrackerRec(
	FDB *			pDb,
	FLMUINT		uiDrn)
{
	RCODE			rc = FERR_OK;
	LFILE *		pTrackerLFile;

	if( RC_BAD( rc = fdictGetContainer( 
		pDb->pDict, FLM_TRACKER_CONTAINER, &pTrackerLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FSRecUpdate( pDb, pTrackerLFile, NULL, uiDrn, 0)))
	{
		if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		rc = FERR_OK;
	}

	if( RC_BAD( rc = flmRcaRemoveRec( pDb, FLM_TRACKER_CONTAINER, uiDrn)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE flmModifyTrackerRec(
	FDB *					pDb, 
	FLMUINT				uiDrn,
	FlmRecord *			pRecord)
{
	RCODE				rc = FERR_OK;
	LFILE	 *			pLFile;

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
		FLM_TRACKER_CONTAINER, &pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FSRecUpdate( pDb, pLFile, pRecord, uiDrn, 0)))
	{
		goto Exit;
	}

	pRecord->setID( uiDrn);
	pRecord->setContainerID( FLM_TRACKER_CONTAINER);

	if( RC_BAD( rc = flmRcaInsertRec( pDb, pLFile, uiDrn, pRecord)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: 
*****************************************************************************/
FSTATIC RCODE FLMAPI flmMaintThread(
	IF_Thread *		pThread)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bDbInitialized = FALSE;
	FFILE *				pFile = (FFILE *)pThread->getParm1();
	LFILE *				pTrackerLFile;
	FDB *					pDb = NULL;
	FLMUINT				uiLastDrn;
	FlmRecord *			pTrackerRec = NULL;
	FMAINT_STATUS *	pStatus = NULL;

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);

	if( pThread->getShutdownFlag())
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = flmOpenFile( pFile,
		NULL, NULL, NULL, 0, TRUE, NULL, NULL,
		pFile->pszDbPassword, &pDb)))
	{
		// If the file is being closed, this is not an error.

		if( pFile->uiFlags & DBF_BEING_CLOSED)
		{
			rc = FERR_OK;
		}
		goto Exit;
	}

	bDbInitialized = TRUE;
	if( RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, 0, 0, &bStartedTrans)))
	{
		goto Exit;
	}

	pStatus = &pFile->maintStatus;

	for(;;)
	{
		// Set the thread's status

		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);

		// Set the tracker status

		pStatus->eDoing = FLM_MAINT_IDLE;

		// Wait for work to become available

		if( RC_BAD( rc = f_semWait( pFile->hMaintSem, F_WAITFOREVER)))
		{
			goto Exit;
		}

		uiLastDrn = 0x00010000;
		for( ;;)
		{
			// See if we should shut down. 

			if( pThread->getShutdownFlag())
			{
				goto Exit;
			}

			// Obtain the file lock

			pStatus->eDoing = FLM_MAINT_WAITING_FOR_LOCK;

			flmAssert( !(pDb->uiFlags & FDB_HAS_FILE_LOCK));
			if( RC_BAD( rc = pDb->pFile->pFileLockObj->lock( pDb->hWaitSem, 
				TRUE, FLM_NO_TIMEOUT, FLM_BACKGROUND_LOCK_PRIORITY,
				pDb->pDbStats ? &pDb->pDbStats->LockStats : NULL)))
			{
				if( rc == FERR_IO_FILE_LOCK_ERR)
				{
					// This would only happen if we were signaled to shut down.
					// So, it's ok to exit

					flmAssert( pThread->getShutdownFlag());
					rc = FERR_OK;
				}
				goto Exit;
			}

			// The lock needs to be marked as implicit so that flmCommitDbTrans
			// will unlock the file and allow the next update transaction to
			// begin before all writes are complete.

			pDb->uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);

			// If there are higher priority waiters in the lock queue,
			// or we are being told to shut down, we want to relinquish.

			if( pThread->getShutdownFlag() ||
				pDb->pFile->pFileLockObj->haveHigherPriorityWaiter( 
					FLM_BACKGROUND_LOCK_PRIORITY))
			{
				if( RC_BAD( rc = pDb->pFile->pFileLockObj->unlock()))
				{
					goto Exit;
				}

				pDb->uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
				continue;
			}

			// Start an update transaction 

			if( RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT, 
				FLM_DONT_POISON_CACHE)))
			{
				if( rc == FERR_IO_FILE_LOCK_ERR)
				{
					// This would only happen if we were signaled to shut down.
					// So, it's ok to exit

					flmAssert( pThread->getShutdownFlag());
					rc = FERR_OK;
				}
				goto Exit;
			}
			bStartedTrans = TRUE;

			if( RC_BAD( rc = fdictGetContainer( 
				pDb->pDict, FLM_TRACKER_CONTAINER, &pTrackerLFile)))
			{
				goto Exit;
			}

			pStatus->eDoing = FLM_MAINT_LOOKING_FOR_WORK;

			// Get a tracker record to process

			if( RC_BAD( rc = flmRetrieveTrackerRec( pDb, 
				uiLastDrn, FALSE, &pTrackerRec)))
			{
				if( rc != FERR_EOF_HIT)
				{
					goto Exit;
				}

				rc = FERR_OK;
				(void)flmAbortDbTrans( pDb);
				bStartedTrans = FALSE;
				break;
			}
			else
			{
				uiLastDrn = pTrackerRec->getID();
				switch( pTrackerRec->getFieldID( pTrackerRec->root()))
				{
					case FLM_DELETE_TAG:
					{
						pStatus->eDoing = FLM_MAINT_FREEING_BLOCKS;

						if( RC_BAD( rc = flmMaintFreeBlockChain( 
							pDb, pTrackerRec->getID(), 25, 0, 
							&pDb->pFile->maintStatus.ui64BlocksFreed)))
						{
							goto Exit;
						}

						break;
					}

					default:
					{
						// Don't know what to do with this record, so we
						// will skip it

						uiLastDrn++;
						break;
					}
				}
				
				// Commit the transaction

				pStatus->eDoing = FLM_MAINT_ENDING_TRANS;

				if( RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
				{
					goto Exit;
				}

				bStartedTrans = FALSE;
			}
		}
	}

Exit:

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);
	pFile->maintStatus.eDoing = FLM_MAINT_TERMINATED;

	if( pTrackerRec)
	{
		pTrackerRec->Release();
	}

	if( bStartedTrans)
	{
		(void)flmAbortDbTrans( pDb);
		bStartedTrans = FALSE;
	}

	if( pDb && pDb->uiFlags & FDB_HAS_FILE_LOCK)
	{
		pDb->pFile->pFileLockObj->unlock();
		pDb->uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
		bDbInitialized = FALSE;
	}

	if( pDb)
	{
		(void)FlmDbClose( (HFDB *)&pDb);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FLMEXP RCODE FLMAPI FlmMaintenanceStatus(
	HFDB					hDb,
	FMAINT_STATUS *	pMaintStatus)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bMutexLocked = FALSE;
	FDB *			pDb = NULL;

	pDb = (FDB *)hDb;
	if( RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
							FDB_TRANS_GOING_OK, 0, NULL)))
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	f_memcpy( pMaintStatus, &pDb->pFile->maintStatus, 
		sizeof( FMAINT_STATUS));

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if( pDb)
	{
		fdbExit( pDb);
	}

	return( rc);
}
