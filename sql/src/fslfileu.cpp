//------------------------------------------------------------------------------
// Desc:	Routines to perform dictionary updates.
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

/****************************************************************************
Desc:	Copies an existing dictionary to a new dictionary.
****************************************************************************/
RCODE F_Db::dictClone( void)
{
	RCODE		rc = NE_SFLM_OK;
	F_Dict *	pNewDict = NULL;

	// Allocate a new FDICT structure

	if ((pNewDict = f_new F_Dict) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	// Nothing to do is not a legal state.

	if (!m_pDict)
	{
		flmAssert( 0);
		m_pDict = pNewDict;
		goto Exit;
	}

	// Copy the dictionary.

	if (RC_BAD( rc = pNewDict->cloneDict( m_pDict)))
	{
		goto Exit;
	}

	m_pDatabase->lockMutex();
	unlinkFromDict();
	m_pDatabase->unlockMutex();
	m_pDict = pNewDict;
	pNewDict = NULL;
	m_uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if (RC_BAD( rc) && pNewDict)
	{
		pNewDict->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Logs information about an index being built
****************************************************************************/
void flmLogIndexingProgress(
	FLMUINT		uiIndexNum,
	FLMUINT64	ui64LastRowId)
{
	IF_LogMessageClient *	pLogMsg = NULL;
	char							szMsg[ 128];

	if( (pLogMsg = flmBeginLogMessage( SFLM_GENERAL_MESSAGE)) != NULL)
	{
		pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
		if (ui64LastRowId)
		{
			f_sprintf( (char *)szMsg,
				"Indexing progress: Index %u is offline.  Last row processed = %I64u.",
				(unsigned)uiIndexNum, ui64LastRowId);
		}
		else
		{
			f_sprintf( (char *)szMsg,
				"Indexing progress: Index %u is online.",
				(unsigned)uiIndexNum);
		}
		pLogMsg->appendString( szMsg);
	}
	flmEndLogMessage( &pLogMsg);
}

/****************************************************************************
Desc:	Index a set of documents or until time runs out.
****************************************************************************/
RCODE F_Db::indexSetOfRows(
	FLMUINT					uiIndexNum,
	FLMUINT64				ui64StartRowId,
	FLMUINT64				ui64EndRowId,
	IF_IxStatus *			pIxStatus,
	IF_IxClient *			pIxClient,
	SFLM_INDEX_STATUS *	pIndexStatus,
	FLMBOOL *				pbHitEnd,
	IF_Thread *				pThread)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT64		ui64RowId;
	FLMUINT64		ui64LastRowId = 0;
	F_INDEX *		pIndex = NULL;
	F_TABLE *		pTable;
	IF_LockObject *
						pDatabaseLockObj = m_pDatabase->m_pDatabaseLockObj;
	FLMBOOL			bHitEnd = FALSE;
	FLMUINT			uiCurrTime;
	FLMUINT			uiLastStatusTime = 0;
	FLMUINT			uiStartTime;
	FLMUINT			uiMinTU;
	FLMUINT			uiStatusIntervalTU;
	FLMUINT64		ui64RowsProcessed = 0;
	FLMBOOL			bRelinquish = FALSE;
	FLMBYTE			ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT			uiKeyLen;
	void *			pvTmpPoolMark = m_tempPool.poolMark();
	F_Btree *		pbtree = NULL;
	FLMBOOL			bNeg;
	FLMUINT			uiBytesProcessed;
	F_Row *			pRow = NULL;

	uiMinTU = FLM_MILLI_TO_TIMER_UNITS( 500);
	uiStatusIntervalTU = FLM_SECS_TO_TIMER_UNITS( 10);
	uiStartTime = FLM_GET_TIMER();

	if (RC_BAD( rc = krefCntrlCheck()))
	{
		goto Exit;
	}

	pIndex = m_pDict->getIndex( uiIndexNum);
	flmAssert( pIndex);
	
	flmAssert( !(pIndex->uiFlags & IXD_SUSPENDED));

	// Get a btree

	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	pTable = m_pDict->getTable( pIndex->uiTableNum);
	flmAssert( pTable);

	if (RC_BAD( rc = pbtree->btOpen( this, &pTable->lfInfo,
								FALSE, TRUE)))
	{
		goto Exit;
	}

	uiKeyLen = sizeof( ucKey);
	if (RC_BAD( rc = flmNumber64ToStorage( ui64StartRowId, &uiKeyLen,
									ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pbtree->btLocateEntry(
								ucKey, sizeof( ucKey), &uiKeyLen, FLM_INCL)))
	{
		if (rc == NE_SFLM_EOF_HIT || rc == NE_SFLM_NOT_FOUND)
		{
			rc = NE_SFLM_OK;
			bHitEnd = TRUE;
			goto Commit_Keys;
		}

		goto Exit;
	}
	
	for (;;)
	{
		
		// See what row we're on
	
		if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
									&ui64RowId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}

		if (ui64RowId > ui64EndRowId)
		{
			break;
		}
		
		if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
									pIndex->uiTableNum, ui64RowId, &pRow)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = buildKeys( pIndex, pTable, pRow, TRUE, NULL)))
		{
			goto Exit;
		}

		// See if there is an indexing callback

		if (pIxClient)
		{
			if (RC_BAD( rc = pIxClient->doIndexing( this, uiIndexNum,
								pIndex->uiTableNum, pRow)))
			{
				goto Exit;
			}
		}

		ui64LastRowId = ui64RowId;
		ui64RowsProcessed++;

		if (pIndexStatus)
		{
			pIndexStatus->ui64RowsProcessed++;
			pIndexStatus->ui64LastRowIndexed = ui64LastRowId;
		}

		// Get the current time

		uiCurrTime = FLM_GET_TIMER();

		// Break out if someone is waiting for an update transaction.

		if (pThread)
		{
			if (pThread->getShutdownFlag())
			{
				bRelinquish = TRUE;
				break;
			}

			if (pDatabaseLockObj->getWaiterCount())
			{
				// See if our minimum run time has elapsed

				if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) >= uiMinTU)
				{
					if (ui64RowsProcessed < 50)
					{
						// If there are higher priority waiters in the lock queue,
						// we want to relinquish.

						if (pDatabaseLockObj->haveHigherPriorityWaiter(
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

				if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) >
					gv_SFlmSysData.uiMaxCPInterval &&
					m_pDatabase->m_pWriteLockObj->getWaiterCount())
				{
					bRelinquish = TRUE;
					break;
				}
			}
		}

		if (FLM_ELAPSED_TIME( uiCurrTime, uiLastStatusTime) >=
					uiStatusIntervalTU)
		{
			uiLastStatusTime = uiCurrTime;
			if( pIxStatus)
			{
				if( RC_BAD( rc = pIxStatus->reportIndex( ui64LastRowId)))
				{
					goto Exit;
				}
			}

			// Send indexing completed event notification

			if( gv_SFlmSysData.EventHdrs[ SFLM_EVENT_UPDATES].pEventCBList)
			{
				flmDoEventCallback( SFLM_EVENT_UPDATES,
						SFLM_EVENT_INDEXING_PROGRESS, this, f_threadId(),
						0, uiIndexNum, ui64LastRowId,
						NE_SFLM_OK);
			}

			// Log a progress message

			flmLogIndexingProgress( uiIndexNum, ui64LastRowId);
		}

		// Need to go to the next row.

		if( RC_BAD( rc = pbtree->btNextEntry(
									ucKey, sizeof( ucKey), &uiKeyLen)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				bHitEnd = TRUE;
				break;
			}
			goto Exit;
		}
	}

Commit_Keys:

	if (RC_BAD( rc = keysCommit( TRUE)))
	{
		goto Exit;
	}

	// If at the end, change index state.

	if (bHitEnd)
	{
		if (RC_BAD( rc = setIxStateInfo( uiIndexNum, 0, 0)))
		{
			goto Exit;
		}

		// setIxStateInfo may have changed to a new dictionary, so pIxd is no
		// good after this point

		pIndex = NULL;
	}
	else if (ui64RowsProcessed)
	{
		if (RC_BAD( rc = setIxStateInfo( uiIndexNum, ui64LastRowId,
										IXD_OFFLINE)))
		{
			goto Exit;
		}

		// setIxStateInfo may have changed to a new dictionary, so pIndex is no
		// good after this point

		pIndex = NULL;
	}
	
	// Log the rows that were indexed, if any
	
	if (ui64LastRowId)
	{
		if (RC_BAD( rc = m_pDatabase->m_pRfl->logIndexSet( this, uiIndexNum,
									ui64StartRowId, ui64LastRowId)))
		{
			goto Exit;
		}
	}

Exit:

	// We want to make one last call if we are in the foreground or if
	// we actually did some indexing.

	if (gv_SFlmSysData.EventHdrs[ SFLM_EVENT_UPDATES].pEventCBList)
	{
		flmDoEventCallback( SFLM_EVENT_UPDATES,
				SFLM_EVENT_INDEXING_PROGRESS, this, f_threadId(),
				0, uiIndexNum,
				(FLMUINT64)(bHitEnd ? (FLMUINT64)0 : ui64LastRowId),
				NE_SFLM_OK);
	}

	flmLogIndexingProgress( uiIndexNum,
		(FLMUINT64)(bHitEnd ? (FLMUINT64)0 : ui64LastRowId));

	if (pIxStatus)
	{
		(void) pIxStatus->reportIndex( ui64LastRowId);
	}

	if (pbHitEnd)
	{
		*pbHitEnd = bHitEnd;
	}

	krefCntrlFree();
	m_tempPool.poolReset( pvTmpPoolMark);

	if (pbtree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}

/****************************************************************************
Desc:	Set information in the index definition row.
****************************************************************************/
RCODE F_Db::setIxStateInfo(
	FLMUINT		uiIndexNum,
	FLMUINT64	ui64LastRowIndexed,
	FLMUINT		uiState)
{
	RCODE				rc = NE_SFLM_OK;
	IXD_FIXUP *		pIxdFixup;
	F_INDEX *		pIndex;
	FLMBOOL			bMustAbortOnError = FALSE;
	F_Row *			pRow = NULL;

	pIndex = m_pDict->getIndex( uiIndexNum);
	flmAssert( pIndex);

	// See if this index is in our fixup list.

	pIxdFixup = m_pIxdFixups;
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
		pIxdFixup->pNext = m_pIxdFixups;
		m_pIxdFixups = pIxdFixup;
		pIxdFixup->uiIndexNum = uiIndexNum;
		pIxdFixup->ui64LastRowIndexed = pIndex->ui64LastRowIndexed;
	}

	bMustAbortOnError = TRUE;

	// Update the last row indexed, if it changed.

	if (pIxdFixup->ui64LastRowIndexed != ui64LastRowIndexed)
	{
		pIxdFixup->ui64LastRowIndexed = ui64LastRowIndexed;

		// First, retrieve the index definition row.
		
		if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
									SFLM_TBLNUM_INDEXES, pIndex->ui64DefRowId, &pRow)))
		{
			goto Exit;
		}

		if (ui64LastRowIndexed)
		{
			if (RC_BAD( rc = pRow->setUINT64( this, SFLM_COLNUM_INDEXES_LAST_ROW_INDEXED,
												ui64LastRowIndexed)))
			{
				goto Exit;
			}
		}
		else
		{
			pRow->setToNull( this, SFLM_COLNUM_INDEXES_LAST_ROW_INDEXED);
		}
	}

	// If IXD_SUSPENDED is set, then IXD_OFFLINE must also be set.
	// There are places in the code that only check for IXD_OFFLINE
	// that don't care if the index is also suspended.

	if (uiState & IXD_SUSPENDED)
	{
		uiState = IXD_SUSPENDED | IXD_OFFLINE;
	}
	else if (uiState & IXD_OFFLINE)
	{
		uiState = IXD_OFFLINE;
	}
	else
	{
		uiState = 0;
	}

	// See if we need to change state.

	if ((pIndex->uiFlags & (IXD_SUSPENDED | IXD_OFFLINE)) != uiState)
	{
		const char *	pszStateStr;
		FLMUINT			uiStateStrLen;

		if (uiState & IXD_SUSPENDED)
		{
			pszStateStr = SFLM_INDEX_SUSPENDED_STR;
		}
		else if (uiState & IXD_OFFLINE)
		{
			pszStateStr = SFLM_INDEX_OFFLINE_STR;
		}
		else
		{
			pszStateStr = SFLM_INDEX_ONLINE_STR;
		}

		// At this point we know we need to change the state.  That means we need
		// to create a new dictionary, if we have not already done so.

		if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
		{
			if (RC_BAD( rc = dictClone()))
			{
				goto Exit;
			}

			// Get a pointer to the new F_INDEX
			
			pIndex = m_pDict->getIndex( uiIndexNum);
		}

		// Retrieve the index definition row if it was not fetched above.
		
		if (!pRow)
		{
			if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
										SFLM_TBLNUM_INDEXES, pIndex->ui64DefRowId, &pRow)))
			{
				goto Exit;
			}
		}
		uiStateStrLen = f_strlen( pszStateStr);
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_INDEXES_INDEX_STATE,
										pszStateStr, uiStateStrLen, uiStateStrLen)))
		{
			goto Exit;
		}

		// Put the state into the F_INDEX.

		pIndex->uiFlags = (pIndex->uiFlags & (~(IXD_SUSPENDED | IXD_OFFLINE))) |
							 uiState;
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:	See if any F_INDEX structures need indexing in the background.
****************************************************************************/
RCODE F_Db::startBackgroundIndexing( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;
	FLMUINT		uiIndexNum;
	F_INDEX *	pIndex;

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	if (m_eTransType != SFLM_NO_TRANS)
	{
		if (!okToCommitTrans())
		{
			rc = RC_SET( NE_SFLM_ABORT_TRANS);
			goto Exit;
		}
	}
	else
	{

		// Need to have at least a read transaction going.

		if (RC_BAD( rc = beginTrans( SFLM_READ_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	for (uiIndexNum = 1, pIndex = m_pDict->m_pIndexTbl;
		  uiIndexNum <= m_pDict->m_uiHighestIndexNum;
		  uiIndexNum++, pIndex++)
	{

		// Restart any indexes that are off-line but not suspended

		if ((pIndex->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED)) == IXD_OFFLINE)
		{
			flmAssert( flmBackgroundIndexGet( m_pDatabase,
									uiIndexNum, FALSE) == NULL);

			if (RC_BAD( rc = startIndexBuild( uiIndexNum)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (bStartedTrans)
	{
		(void)abortTrans();
	}

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_Database::startMaintThread( void)
{
	RCODE			rc = NE_SFLM_OK;
	char			szThreadName[ F_PATH_MAX_SIZE];
	char			szBaseName[ 32];

	flmAssert( !m_pMaintThrd);
	flmAssert( m_hMaintSem == F_SEM_NULL);

	// Generate the thread name

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( 
		m_pszDbPath, szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( (char *)szThreadName, "Maintenance (%s)", (char *)szBaseName);

	// Create the maintenance semaphore

	if( RC_BAD( rc = f_semCreate( &m_hMaintSem)))
	{
		goto Exit;
	}

	// Start the thread.

	if( RC_BAD( rc = gv_SFlmSysData.pThreadMgr->createThread( &m_pMaintThrd,
		F_Database::maintenanceThread, szThreadName,
		0, 0, this, NULL, 32000)))
	{
		goto Exit;
	}

	// Signal the thread to check for any queued work

	f_semSignal( m_hMaintSem);

Exit:

	if( RC_BAD( rc))
	{
		if( m_hMaintSem != F_SEM_NULL)
		{
			f_semDestroy( &m_hMaintSem);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::beginBackgroundTrans(
	IF_Thread *		pThread)
{
	RCODE		rc = NE_SFLM_OK;

RetryLock:

	// Obtain the file lock

	flmAssert( !(m_uiFlags & FDB_HAS_FILE_LOCK));

	if( RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->lock( m_hWaitSem,
		TRUE, FLM_NO_TIMEOUT, FLM_BACKGROUND_LOCK_PRIORITY,
		m_pDbStats ? &m_pDbStats->LockStats : NULL)))
	{
		if( rc == NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT)
		{
			// This would only happen if we were signaled to shut down.
			// So, it's ok to exit

			flmAssert( pThread->getShutdownFlag());
		}
		goto Exit;
	}

	// The lock needs to be marked as implicit so that commitTrans
	// will unlock the database and allow the next update transaction to
	// begin before all writes are complete.

	m_uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);

	// If there are higher priority waiters in the lock queue,
	// we want to relinquish.

	if( m_pDatabase->m_pDatabaseLockObj->haveHigherPriorityWaiter(
			FLM_BACKGROUND_LOCK_PRIORITY))
	{
		if( pThread->getShutdownFlag())
		{
			goto Exit;
		}
		if( RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->unlock()))
		{
			goto Exit;
		}

		m_uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
		goto RetryLock;
	}

	// If we are shutting down, relinquish and exit.

	if( pThread->getShutdownFlag())
	{
		rc = RC_SET( NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = beginTrans( 
		SFLM_UPDATE_TRANS, FLM_NO_TIMEOUT, SFLM_DONT_POISON_CACHE)))
	{
		if( rc == NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT)
		{
			// This would only happen if we were signaled to shut down.
			// So, it's ok to exit

			flmAssert( pThread->getShutdownFlag());
		}
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		if( m_uiFlags & FDB_HAS_FILE_LOCK)
		{
			(void)m_pDatabase->m_pDatabaseLockObj->unlock();
			m_uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Thread that will delete block chains from deleted indexes and
		tables in the background.
****************************************************************************/
RCODE SQFAPI F_Database::maintenanceThread(
	IF_Thread *		pThread)
{
	RCODE					rc = NE_SFLM_OK;
	F_Database *		pDatabase = (F_Database *)pThread->getParm1();
	F_Db *				pDb;
	F_Row *				pRow;
	FLMUINT64			ui64MaintRowId;
	FLMBOOL				bStartedTrans;
	FLMBOOL				bShutdown;
	F_DbSystem *		pDbSystem;
	FSTableCursor *	pTableCursor;
	FLMUINT				uiBlkAddress;
	FLMBOOL				bIsNull;
	FLMUINT				uiBlocksToFree;
	FLMUINT				uiBlocksFreed;

Retry:
	
	rc = NE_SFLM_OK;
	pDb = NULL;
	pRow = NULL;
	bStartedTrans = FALSE;
	bShutdown = FALSE;
	pDbSystem = NULL;
	pTableCursor = NULL;

	if( (pDbSystem = f_new F_DbSystem) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);

	if( RC_BAD( rc = pDbSystem->internalDbOpen( pDatabase, &pDb)))
	{
		// If the file is being closed, this is not an error.

		if( pDatabase->getFlags() & DBF_BEING_CLOSED)
		{
			rc = NE_SFLM_OK;
			bShutdown = TRUE;
		}
		goto Exit;
	}
	pDbSystem->Release();
	pDbSystem = NULL;

	if ((pTableCursor = f_new FSTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	for( ;;)
	{
		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);
		
		if( RC_BAD( rc = pDb->beginBackgroundTrans( pThread)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
			
		pTableCursor->resetCursor();
		if (RC_BAD( rc = pTableCursor->setupRange( pDb, SFLM_TBLNUM_BLOCK_CHAINS,
												1, FLM_MAX_UINT64, FALSE)))
		{
			goto Exit;
		}
		
		// Free up to 25 blocks per transaction.
		
		uiBlocksToFree = 25;
		while (uiBlocksToFree)
		{

			if (RC_BAD( rc = pTableCursor->nextRow( pDb, &pRow, &ui64MaintRowId)))
			{
				if (rc != NE_SFLM_EOF_HIT)
				{
					RC_UNEXPECTED_ASSERT( rc);
					goto Exit;
				}
				rc = NE_SFLM_OK;
				break;
			}
			if (RC_BAD( rc = pRow->getUINT( pDb,
						SFLM_COLNUM_BLOCK_CHAINS_BLOCK_ADDRESS, &uiBlkAddress,
						&bIsNull)))
			{
				goto Exit;
			}
			if (bIsNull)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
				goto Exit;
			}
			
			if( RC_BAD( rc = pDb->maintBlockChainFree(
				ui64MaintRowId, uiBlkAddress, uiBlocksToFree, 0, &uiBlocksFreed)))
			{
				goto Exit;
			}
			uiBlocksToFree -= uiBlocksFreed;
		}

		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->commitTrans( 0, FALSE)))
		{
			goto Exit;
		}

		pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);
		f_semWait( pDatabase->m_hMaintSem, F_WAITFOREVER);
			
		if (pThread->getShutdownFlag())
		{
			bShutdown = TRUE;
			goto Exit;
		}
	}

Exit:

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);

	if (pDbSystem)
	{
		pDbSystem->Release();
	}
	
	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	if (pDb)
	{
		pDb->Release();
		pDb = NULL;
	}

	if (!bShutdown)
	{
		flmAssert( RC_BAD( rc));
		f_sleep( 250);
		f_semSignal( pDatabase->m_hMaintSem);
		goto Retry;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::maintBlockChainFree(
	FLMUINT64		ui64MaintRowId,
	FLMUINT			uiStartBlkAddr,
	FLMUINT 			uiBlocksToFree,
	FLMUINT			uiExpectedEndBlkAddr,
	FLMUINT *		puiBlocksFreed)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiBlocksFreed = 0;
	FLMUINT		uiEndBlkAddr = 0;
	F_Row *		pRow = NULL;
	FLMUINT		uiRflToken = 0;

	// Make sure an update transaction is going and that a
	// non-zero number of blocks was specified

	if( getTransType() != SFLM_UPDATE_TRANS || !uiBlocksToFree)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}

	m_pDatabase->m_pRfl->disableLogging( &uiRflToken);
	
	if( RC_BAD( rc = btFreeBlockChain( 
		this, NULL, uiStartBlkAddr, uiBlocksToFree, 
		&uiBlocksFreed, &uiEndBlkAddr, NULL)))
	{
		goto Exit;
	}

	flmAssert( uiBlocksFreed <= uiBlocksToFree);

	if (!uiEndBlkAddr)
	{
		if (RC_BAD( rc = deleteRow( SFLM_TBLNUM_BLOCK_CHAINS, ui64MaintRowId,
									FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
										SFLM_TBLNUM_BLOCK_CHAINS, ui64MaintRowId,
										&pRow)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->setUINT( this,
					SFLM_COLNUM_BLOCK_CHAINS_BLOCK_ADDRESS, uiEndBlkAddr)))
		{
			goto Exit;
		}
	}

	if (uiExpectedEndBlkAddr)
	{
		if (uiBlocksToFree != uiBlocksFreed ||
			 uiEndBlkAddr != uiExpectedEndBlkAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
			goto Exit;
		}
	}

	if (uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}
	
	if( RC_BAD( rc = m_pDatabase->m_pRfl->logBlockChainFree( 
		this, ui64MaintRowId, uiStartBlkAddr, uiEndBlkAddr, uiBlocksFreed)))
	{
		goto Exit;
	}

	if (puiBlocksFreed)
	{
		*puiBlocksFreed = uiBlocksFreed;
	}

Exit:

	if (uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}
	
	if (pRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}

