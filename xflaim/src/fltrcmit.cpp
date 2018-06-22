//------------------------------------------------------------------------------
// Desc:	Contains routines for committing a transaction.
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
Desc:	This routine commits an active transaction for a particular
		database.
****************************************************************************/
RCODE F_Db::commitTrans(
	FLMUINT		uiNewLogicalEOF,		// New logical end-of-file.  This is only
												// set by the reduceSize function when
												// it is truncating the file.
	FLMBOOL		bForceCheckpoint,		// Force a checkpoint?
	FLMBOOL *	pbEmpty
	)
{
	RCODE	  			rc = NE_XFLM_OK;
	XFLM_DB_HDR *	pUncommittedDbHdr;
	FLMUINT			uiCPFileNum = 0;
	FLMUINT			uiCPOffset = 0;
	FLMUINT64		ui64TransId = 0;
	FLMBOOL			bTransEndLogged;
	FLMBOOL			bForceCloseOnError = FALSE;
	eDbTransType	eSaveTransType;
	FLMBOOL			bOkToLogAbort = TRUE;
	FLMUINT			uiCollection;
	FLMUINT64		ui64DocId;
	FLMUINT64		ui64NodeId;
	FLMBOOL			bIndexAfterCommit = FALSE;
	F_Rfl *			pRfl = m_pDatabase->m_pRfl;
	FLMUINT			uiRflToken = 0;
	F_COLLECTION *	pCollection;
	FLMUINT			uiLfNum;
		
	// Not allowed to commit temporary databases.

	flmAssert( !m_pDatabase->m_bTempDb);

	// See if we even have a transaction going.

	if (m_eTransType == XFLM_NO_TRANS)
	{
		goto Exit;	// Will return NE_XFLM_OK.
	}

	// See if we have a transaction going which should be aborted.

	if (!okToCommitTrans())
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit1;
	}

	// If we are in a read transaction we can skip most of the stuff
	// below because no updates would have occurred.  This will help
	// improve performance.

	if (m_eTransType == XFLM_READ_TRANS)
	{
		if (m_bKrefSetup)
		{

			// krefCntrlFree could be called w/o checking bKrefSetup because
			// it checks the flag, but it is more optimal to check the
			// flag before making the call because most of the time it will
			// be false.

			krefCntrlFree();
		}

		goto Exit1;
	}

	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);

	// Disable DB header writes

	pRfl->clearDbHdrs();

	// At this point, we know we have an update transaction.

	ui64TransId = m_ui64CurrTransID;
#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( m_pDatabase, ui64TransId, 0, 0, NE_XFLM_OK, "TCmit");
#endif

	// End any pending input operations

	if( m_pDatabase->m_pPendingInput)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit1;
	}

	// Call documentDone for any documents in the document list

	for (;;)
	{
		m_pDatabase->m_DocumentList.getNode( 
			0, &uiCollection, &ui64DocId, &ui64NodeId);

		if (!uiCollection)
		{
			break;
		}

		if (RC_BAD( rc = documentDone( uiCollection, ui64DocId)))
		{
			goto Exit1;
		}
	}

	// Flush any dirty cache nodes

	if( RC_BAD( rc = flushDirtyNodes()))
	{
		goto Exit1;
	}
	
	// Write out any LFILE changes for collections
	
	flmAssert( m_pDict);
	
	// Only need to do collections.  Nothing from the LFH of
	// an index is stored in memory except for the root block
	// address, and whenever that is changed, we get a new
	// dictionary.  Since the new dictionary will be discarded
	// in that case, there is nothing to restore for an index.

	uiLfNum = 0;
	while ((pCollection = m_pDict->getNextCollection(
									uiLfNum, TRUE)) != NULL)
	{
		if (pCollection->bNeedToUpdateNodes)
		{
			if (RC_BAD( rc = m_pDatabase->lFileWrite( this, pCollection,
										&pCollection->lfInfo)))
			{
				goto Exit;
			}
		}
		uiLfNum = pCollection->lfInfo.uiLfNum;
	}

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = keysCommit( TRUE)))
	{
		flmLogError( rc, "calling keysCommit from commitTrans");
		goto Exit1;
	}

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	if (!m_bHadUpdOper)
	{
		bOkToLogAbort = FALSE;
		pRfl->enableLogging( &uiRflToken);

		rc = pRfl->logEndTransaction( this, RFL_TRNS_COMMIT_PACKET, TRUE);

		// Even though we didn't have any update operations, there may have
		// been operations during the transaction (i.e., query operations)
		// that initialized the KREF in order to generate keys.

		krefCntrlFree();

		// Restore everything as if the transaction never happened.

		if (pbEmpty)
		{
			*pbEmpty = TRUE;
		}

		// Even if the transaction is empty, there could be "uncommitted" nodes that
		// were created, but never set to "dirty" - hence the m_bHadUpdOper flag
		// would never have been set.  So we need to call freeModifiedNodes to get
		// rid of them.

		m_pDatabase->freeModifiedNodes( this, ui64TransId - 1);
		goto Exit1;
	}

	// Re-enable RFL logging

	pRfl->enableLogging( &uiRflToken);

	// Log commit record to roll-forward log

	bOkToLogAbort = FALSE;
	if (RC_BAD( rc = pRfl->logEndTransaction(
		this, RFL_TRNS_COMMIT_PACKET, FALSE, &bTransEndLogged)))
	{
		goto Exit1;
	}
	bForceCloseOnError = TRUE;

	// Reinitialize the log header.  If the local dictionary was updated
	// during the transaction, increment the local dictionary ID so that
	// other concurrent users will know that it has been modified and
	// that they need to re-read it into memory.

	// If we are in recovery mode, see if we need to force
	// a checkpoint with what we have so far.  We force a
	// checkpoint on one of two conditions:

	// 1. If it appears that we have a buildup of dirty cache
	//		blocks.  We force a checkpoint on this condition
	//		because it will be more efficient than replacing
	//		cache blocks one at a time.
	//		We check for this condition by looking to see if
	//		our LRU block is not used and it is dirty.  That is
	//		a pretty good indicator that we have a buildup
	//		of dirty cache blocks.
	// 2.	We are at the end of the roll-forward log.  We
	//		want to force a checkpoint here to complete the
	//		recovery phase.

	if (m_uiFlags & FDB_REPLAYING_RFL)
	{
		// If we are in the middle of upgrading, and are forcing
		// a checkpoint, use the file number and offset that were
		// set in the F_Db.

		if ((m_uiFlags & FDB_UPGRADING) && bForceCheckpoint)
		{
			uiCPFileNum = m_uiUpgradeCPFileNum;
			uiCPOffset = m_uiUpgradeCPOffset;
		}
		else
		{
			FLMUINT	uiCurrTime;

			uiCurrTime = (FLMUINT)FLM_GET_TIMER();
			f_mutexLock( gv_XFlmSysData.hShareMutex);
			if (FLM_ELAPSED_TIME( uiCurrTime, m_pDatabase->m_uiLastCheckpointTime) >= 
					gv_XFlmSysData.uiMaxCPInterval ||
				!gv_XFlmSysData.uiMaxCPInterval ||
			   (gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache &&
			    (m_pDatabase->m_uiDirtyCacheCount +
				  m_pDatabase->m_uiLogCacheCount) * m_pDatabase->m_uiBlockSize >
				  gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache) ||
			   pRfl->atEndOfLog() ||
			   bForceCheckpoint)
			{
				bForceCheckpoint = TRUE;
				uiCPFileNum = pRfl->getCurrFileNum();
				uiCPOffset = pRfl->getCurrReadOffset();
			}
			f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		}
	}

	// Move information collected in the pDb into the
	// uncommitted DB header.  Other things that need to be
	// set have already been set in the uncommitted log header
	// at various places in the code.

	// Mutex does not have to be locked while we do this because
	// the update transaction is the only one that ever accesses
	// the uncommitted log header buffer.

	pUncommittedDbHdr = &m_pDatabase->m_uncommittedDbHdr;

	// Set the new logical EOF if passed in.

	if (uiNewLogicalEOF)
	{
		m_uiLogicalEOF = uiNewLogicalEOF;
	}
	pUncommittedDbHdr->ui32LogicalEOF = (FLMUINT32)m_uiLogicalEOF;

	// Increment the commit counter.

	pUncommittedDbHdr->ui64TransCommitCnt++;

	// Set the last committed transaction ID

	if (bTransEndLogged || (m_uiFlags & FDB_REPLAYING_COMMIT))
	{
		pUncommittedDbHdr->ui64LastRflCommitID = ui64TransId;
	}

	// Write the header

	pRfl->commitDbHdrs( pUncommittedDbHdr,
							&m_pDatabase->m_checkpointDbHdr);

	// Commit any node cache.

	m_pDatabase->commitNodeCache();

	// Push the IXD_FIXUP values back into the IXD

	if (m_pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;
		IXD *			pIxd;
		RCODE			tmpRc;

		pIxdFixup = m_pIxdFixups;
		while (pIxdFixup)
		{
			if (RC_BAD( tmpRc = m_pDict->getIndex( pIxdFixup->uiIndexNum,
								NULL, &pIxd, TRUE)))
			{
				RC_UNEXPECTED_ASSERT( tmpRc);
				pIxd = NULL;
			}

			if (pIxd)
			{
				pIxd->ui64LastDocIndexed = pIxdFixup->ui64LastDocIndexed;
			}
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		m_pIxdFixups = NULL;
	}

	// Set the update transaction ID back to zero only
	// AFTER we know the transaction has safely committed.

	m_pDatabase->lockMutex();

	f_memcpy( &m_pDatabase->m_lastCommittedDbHdr, pUncommittedDbHdr,
					sizeof( XFLM_DB_HDR));

	if (m_uiFlags & FDB_UPDATED_DICTIONARY)
	{
		
		// Link the new local dictionary to its file.
		// Since the new local dictionary will be linked at the head
		// of the list of FDICT structures, see if the FDICT currently
		// at the head of the list is unused and can be unlinked.

		if (m_pDatabase->m_pDictList && !m_pDatabase->m_pDictList->getUseCount())
		{
			m_pDatabase->m_pDictList->unlinkFromDatabase();
		}
		m_pDict->linkToDatabase( m_pDatabase);
	}
	m_pDatabase->unlockMutex();

	// Log blocks must be released after the last committed database header
	// has been updated.  If done before, there is a slight window of 
	// opportunity for a caller to attempt to start a read transaction.  Between
	// the time releaseLogBlocks unlocks the block cache mutex and when
	// the committed header is copied into m_pDatabase, the read transaction
	// could see the prior committed transaction ID and use that as the basis
	// for its transaction.  In the mean time, releaseLogBlocks may release
	// versions of the blocks that would be needed to satisfy the new reader's
	// (incorrect) view of the database.
	
	m_pDatabase->releaseLogBlocks();
	
Exit1:

	if (RC_BAD( rc))
	{
		// Since we failed to commit, do an abort.  We are purposely not
		// checking the return code from flmAbortDbTrans because we already
		// have an error return code.  If we attempted to log the transaction
		// to the RFL and failed, we don't want to try to log an abort packet.
		// The RFL code has already reset the log back to the starting point
		// of the transaction, thereby discarding all operations.

		(void)abortTrans( bOkToLogAbort);
		eSaveTransType = XFLM_NO_TRANS;

		// Do we need to force all handles to close?

		if (bForceCloseOnError)
		{
			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all FDBs linked to the FFILE
			// and set the FFILE's "must close" flag.  This will cause any
			// subsequent operations on the database to fail until all
			// handles have been closed.

			m_pDatabase->setMustCloseFlags( rc, FALSE);
		}
	}
	else
	{
		eSaveTransType = m_eTransType;
		if (m_eTransType == XFLM_UPDATE_TRANS)
		{
			if (gv_XFlmSysData.EventHdrs [XFLM_EVENT_UPDATES].pEventCBList)
			{
				flmTransEventCallback( XFLM_EVENT_COMMIT_TRANS, this, rc,
							ui64TransId);
			}

			// Do the indexing work before we unlock the db.
			
			if (m_pIxStopList || m_pIxStartList)
			{
				// Must not call indexingAfterCommit until after
				// completeTransWrites.  Otherwise, there is a potential
				// deadlock condition where indexingAfterCommit is
				// waiting on an indexing thread to quit, but that
				// thread is waiting to be signaled by this thread that
				// writes are completed.  However, indexingAfterCommit
				// also must only be called while the database is still
				// locked.  If we were to leave the database locked for
				// every call to completeTransWrites, however, we would
				// lose the group commit capability.  Hence, we opt to
				// only lose it when there are actual indexing operations
				// to start or stop - which should be very few transactions.
				// That is what the bIndexAfterCommit flag is for.
				
				bIndexAfterCommit = TRUE;
			}
		}
	}

	// Unlock the database, if the update transaction is still going.
	// NOTE: We check m_eTransType because it may have been reset
	// to XFLM_NO_TRANS up above if abortTrans was called.

	if (m_eTransType == XFLM_UPDATE_TRANS)
	{
		if (RC_BAD( rc))
		{

			// SHOULD NEVER HAPPEN - because it would have been taken
			// care of above - abortTrans would have been called and
			// m_eTransType would no longer be XFLM_UPDATE_TRANS.

			RC_UNEXPECTED_ASSERT( rc);
			pRfl->completeTransWrites( this, FALSE, TRUE);
		}
		else if (!bForceCheckpoint)
		{
			if (bIndexAfterCommit)
			{
				rc = pRfl->completeTransWrites( this, TRUE, FALSE);
				indexingAfterCommit();
				unlinkFromTransList( TRUE);
			}
			else
			{
				rc = pRfl->completeTransWrites( this, TRUE, TRUE);
			}
		}
		else
		{
			// Do checkpoint, if forcing.  Before doing the checkpoint
			// we have to make sure the roll-forward log writes
			// complete.  We don't want to unlock the DB while the
			// writes are happening in this case - thus, the FALSE
			// parameter to completeTransWrites.

			if (RC_OK( rc = pRfl->completeTransWrites( this, TRUE, FALSE)))
			{
				bForceCloseOnError = FALSE;
				rc = m_pDatabase->doCheckpoint( m_hWaitSem, m_pDbStats, m_pSFileHdl,
						(m_uiFlags & FDB_DO_TRUNCATE) ? TRUE : FALSE,
						TRUE, XFLM_CP_TIME_INTERVAL_REASON,
						uiCPFileNum, uiCPOffset);
			}

			if (bIndexAfterCommit)
			{
				indexingAfterCommit();
			}

			unlinkFromTransList( TRUE);
		}

		if (RC_BAD( rc) && bForceCloseOnError)
		{

			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all F_Db objects linked to the
			// F_Database object and set the F_Database object's "must close"
			// flag.  This will cause any subsequent operations on the
			// database to fail until all handles have been closed.

			m_pDatabase->setMustCloseFlags( rc, FALSE);
		}
	}
	else
	{

		// Unlink the database from the transaction list

		unlinkFromTransList( FALSE);
	}

	if (m_pDbStats && eSaveTransType != XFLM_NO_TRANS)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &m_TransStartTime, &ui64ElapMilli);
		m_pDbStats->bHaveStats = TRUE;
		if (eSaveTransType == XFLM_READ_TRANS)
		{
			m_pDbStats->ReadTransStats.CommittedTrans.ui64Count++;
			m_pDbStats->ReadTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
		else
		{
			m_pDbStats->UpdateTransStats.CommittedTrans.ui64Count++;
			m_pDbStats->UpdateTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	// Update stats

	if (m_pStats)
	{
		(void)flmStatUpdate( &m_Stats);
	}

Exit:

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Commits an active transaction.
*END************************************************************************/
RCODE XFLAPI F_Db::transCommit(
	FLMBOOL *	pbEmpty)	// may be NULL
{
	RCODE	rc = NE_XFLM_OK;

	if (m_eTransType == XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if (RC_BAD( m_AbortRc))
	{
		rc = RC_SET( NE_XFLM_ABORT_TRANS);
		goto Exit;
	}

	if (pbEmpty)
	{
		*pbEmpty = FALSE;
	}

	rc = commitTrans( 0, FALSE, pbEmpty);

Exit:

	if( RC_OK( rc))
	{
		rc = checkState( __FILE__, __LINE__);
	}

	return( rc);
}
