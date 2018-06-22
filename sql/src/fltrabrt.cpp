//------------------------------------------------------------------------------
// Desc:	Contains routines for aborting a transaction.
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
Desc:	This routine aborts an active transaction for a particular
		database.  If the database is open via a server, a message is
		sent to the server to abort the transaction.  Otherwise, the
		transaction is rolled back locally.
****************************************************************************/
RCODE F_Db::abortTrans(
	FLMBOOL			bOkToLogAbort)
{
	RCODE					rc = NE_SFLM_OK;
	eDbTransType		eSaveTransType;
	SFLM_DB_HDR *		pLastCommittedDbHdr;
	SFLM_DB_HDR *		pUncommittedDbHdr;
	FLMBOOL				bDumpedCache = FALSE;
	FLMBOOL				bKeepAbortedTrans;
	FLMUINT64			ui64TransId;
	F_Rfl *				pRfl = m_pDatabase->m_pRfl;
	RCODE					tmpRc;

	// Should never be calling on a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Get transaction type

	if (m_eTransType == SFLM_NO_TRANS)
	{
		goto Exit;	// Will return SUCCESS.
	}

	// No recovery required if it is a read transaction.

	if (m_eTransType == SFLM_READ_TRANS)
	{
		if (m_bKrefSetup)
		{
			// krefCntrlFree could be called w/o checking bKrefSetup because
			// it checks the flag, but it is more optimal to check the
			// flag before making the call because most of the time it will
			// be false.

			krefCntrlFree();
		}

		goto Unlink_From_Trans;
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( m_pDatabase, m_ui64CurrTransID,
			0, 0, NE_SFLM_OK, "TAbrt");
#endif

	// Disable DB header writes

	pRfl->clearDbHdrs();

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	pLastCommittedDbHdr = &m_pDatabase->m_lastCommittedDbHdr;
	pUncommittedDbHdr = &m_pDatabase->m_uncommittedDbHdr;
	ui64TransId = m_ui64CurrTransID;

	// Free up all keys associated with this database.  This is done even
	// if we didn't have any update operations because the KREF may
	// have been initialized by key generation operations performed
	// by cursors, etc.

	krefCntrlFree();

	if (m_bHadUpdOper)
	{

		// Dump any start and stop indexing stubs that should be aborted.

		indexingAfterAbort();

		// Log the abort record to the rfl file, or throw away the logged
		// records altogether, depending on the LOG_KEEP_ABORTED_TRANS_IN_RFL
		// flag.  If the RFL volume is bad, we will not attempt to keep this
		// transaction in the RFL.

		if (!pRfl->seeIfRflVolumeOk())
		{
			bKeepAbortedTrans = FALSE;
		}
		else
		{
			bKeepAbortedTrans =
				(pUncommittedDbHdr->ui8RflKeepAbortedTrans)
				? TRUE
				: FALSE;
		}
	}
	else
	{
		bKeepAbortedTrans = FALSE;
	}

	// Log an abort transaction record to the roll-forward log or
	// throw away the entire transaction, depending on the
	// bKeepAbortedTrans flag.

	// If the transaction is being "dumped" because of a failed commit,
	// don't log anything to the RFL.

	if (bOkToLogAbort)
	{
#ifdef FLM_DEBUG
		if( pRfl->isLoggingEnabled())
		{
			flmAssert( m_ui64CurrTransID == pRfl->getCurrTransID());
		}
#endif

		if (RC_BAD( rc = pRfl->logEndTransaction(
			this, RFL_TRNS_ABORT_PACKET, !bKeepAbortedTrans)))
		{
			goto Exit1;
		}
	}
#ifdef FLM_DEBUG
	else
	{
		// If bOkToLogAbort is FALSE, this always means that either a
		// commit failed while trying to log an end transaction packet or a
		// commit packet was logged and the transaction commit subsequently
		// failed for some other reason.  In either case, the RFL should be
		// in a good state, with its current transaction ID reset to 0.  If
		// not, either bOkToLogAbort is being used incorrectly by the caller
		// or there is a bug in the RFL logic.

		flmAssert( pRfl->getCurrTransID() == 0);
	}
#endif

	// If there were no operations in the transaction, restore
	// everything as if the transaction never happened.

	// Even empty transactions can have modified rows to clean up
	// so we need to call this no matter what.

	m_pDatabase->freeModifiedRows( this, m_ui64CurrTransID - 1);

	if (!m_bHadUpdOper)
	{

		// Pretend we dumped cache - shouldn't be any to worry about at
		// this point.

		bDumpedCache = TRUE;
		goto Exit1;
	}

	// Dump ALL modified cache blocks associated with the DB.
	// NOTE: This needs to be done BEFORE the call to flmGetDbHdrInfo
	// below, because that call will change pDb->m_ui64CurrTransID,
	// and that value is used by freeModifiedRows.

	m_pDatabase->freeModifiedBlocks( m_ui64CurrTransID);
	bDumpedCache = TRUE;

	// Reset the Db header from the last committed DB header in pFile.

	getDbHdrInfo( pLastCommittedDbHdr);
	if (RC_BAD( rc = physRollback( (FLMUINT)pUncommittedDbHdr->ui32RblEOF,
				 m_pDatabase->m_uiFirstLogBlkAddress, FALSE, 0)))
	{
		goto Exit1;
	}

	m_pDatabase->lockMutex();

	// Put the new transaction ID into the log header even though
	// we are not committing.  We want to keep the transaction IDs
	// incrementing even though we aborted.

	pLastCommittedDbHdr->ui64CurrTransID = ui64TransId;

	// Preserve where we are at in the roll-forward log.  Even though
	// the transaction aborted, we may have kept it in the RFL instead of
	// throw it away.

	pLastCommittedDbHdr->ui32RflCurrFileNum =
		pUncommittedDbHdr->ui32RflCurrFileNum;
	pLastCommittedDbHdr->ui32RflLastTransOffset =
		pUncommittedDbHdr->ui32RflLastTransOffset;
	f_memcpy( pLastCommittedDbHdr->ucLastTransRflSerialNum,
				 pUncommittedDbHdr->ucLastTransRflSerialNum,
				 SFLM_SERIAL_NUM_SIZE);
	f_memcpy( pLastCommittedDbHdr->ucNextRflSerialNum,
				 pUncommittedDbHdr->ucNextRflSerialNum,
				 SFLM_SERIAL_NUM_SIZE);

	// The following items tell us where we are at in the roll-back log.
	// During a transaction we may log blocks for the checkpoint or for
	// read transactions.  So, even though we are aborting this transaction,
	// there may be other things in the roll-back log that we don't want
	// to lose.  These items should not be reset until we do a checkpoint,
	// which is when we know it is safe to throw away the entire roll-back log.

	pLastCommittedDbHdr->ui32RblEOF =
		pUncommittedDbHdr->ui32RblEOF;
	pLastCommittedDbHdr->ui32RblFirstCPBlkAddr =
		pUncommittedDbHdr->ui32RblFirstCPBlkAddr;

	m_pDatabase->unlockMutex();

	pRfl->commitDbHdrs( pLastCommittedDbHdr,
							&m_pDatabase->m_checkpointDbHdr);

Exit1:

	// Dump cache, if not done above.

	if (!bDumpedCache)
	{
		m_pDatabase->freeModifiedBlocks( m_ui64CurrTransID);
		m_pDatabase->freeModifiedRows( this, m_ui64CurrTransID - 1);
		bDumpedCache = TRUE;
	}

	// Throw away IXD_FIXUPs

	if (m_pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;

		pIxdFixup = m_pIxdFixups;
		while (pIxdFixup)
		{
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		m_pIxdFixups = NULL;
	}

	if (m_eTransType == SFLM_UPDATE_TRANS &&
		 gv_SFlmSysData.EventHdrs[ SFLM_EVENT_UPDATES].pEventCBList)
	{
		flmTransEventCallback( SFLM_EVENT_ABORT_TRANS, this, rc,
						ui64TransId);
	}

Unlink_From_Trans:

	eSaveTransType = m_eTransType;

	if (m_uiFlags & FDB_HAS_WRITE_LOCK)
	{
		if (RC_BAD( tmpRc = pRfl->completeTransWrites( this, FALSE, FALSE)))
		{
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}
		}
	}

	if (eSaveTransType == SFLM_UPDATE_TRANS)
	{

		// Before unlocking, restore collection information.

		if (m_uiFlags & FDB_UPDATED_DICTIONARY)
		{
			m_pDatabase->lockMutex();
			flmAssert( m_pDict);
			unlinkFromDict();
			if (m_pDatabase->m_pDictList)
			{

				// Link the F_Db to the right F_Dict object so it will
				// fixup the correct F_COLLECTION structures.

				linkToDict( m_pDatabase->m_pDictList);
			}
			m_pDatabase->unlockMutex();
		}

		if (m_pDict)
		{
			F_TABLE *		pTable;
			FLMUINT			uiLfNum;
#ifdef FLM_DEBUG
			F_INDEX *		pIndex;
			FLMUINT			uiRootBlk;
#endif

			// Only need to do tables.  Nothing from the LFH of
			// an index is stored in memory except for the root block
			// address, and whenever that is changed, we get a new
			// dictionary.  Since the new dictionary will be discarded
			// in that case, there is nothing to restore for an index.

			for (uiLfNum = 1, pTable = m_pDict->m_pTableTbl;
				  uiLfNum <= m_pDict->m_uiHighestTableNum;
				  uiLfNum++, pTable++)
			{
				if (pTable->uiTableNum)
				{
#ifdef FLM_DEBUG
					uiRootBlk = pTable->lfInfo.uiRootBlk;
#endif
					if (RC_BAD( tmpRc = m_pDatabase->lFileRead( this, &pTable->lfInfo)))
					{
						if (RC_OK( rc))
						{
							rc = tmpRc;
						}
					}
#ifdef FLM_DEBUG
					else
					{
						// Make sure root block did not change - should not
						// have because root block changes are done by creating
						// a new dictionary, and we have already discarded
						// any new dictionary.  Hence, root block address should
						// be the same in memory as it is no disk.
	
						flmAssert( uiRootBlk == pTable->lfInfo.uiRootBlk);
					}
#endif
				}
			}

			// Do indexes in debug mode to make sure uiRootBlk is correct

#ifdef FLM_DEBUG
			for (uiLfNum = 1, pIndex = m_pDict->m_pIndexTbl;
				  uiLfNum <= m_pDict->m_uiHighestIndexNum;
				  uiLfNum++, pIndex++)
			{
				if (pIndex->uiIndexNum)
				{
					uiRootBlk = pIndex->lfInfo.uiRootBlk;
					if (RC_BAD( tmpRc = m_pDatabase->lFileRead( this, &pIndex->lfInfo)))
					{
						if (RC_OK( rc))
						{
							rc = tmpRc;
						}
					}
					else
					{
						// Make sure root block did not change - should not
						// have because root block changes are done by creating
						// a new dictionary, and we have already discarded
						// any new dictionary.  Hence, root block address should
						// be the same in memory as it is no disk.
	
						flmAssert( uiRootBlk == pIndex->lfInfo.uiRootBlk);
					}
				}
			}
#endif
		}
	}

	// Unlink the database from the transaction list.

	unlinkFromTransList( FALSE);

	if (m_pDbStats)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &m_TransStartTime, &ui64ElapMilli);
		m_pDbStats->bHaveStats = TRUE;
		if (eSaveTransType == SFLM_READ_TRANS)
		{
			m_pDbStats->ReadTransStats.AbortedTrans.ui64Count++;
			m_pDbStats->ReadTransStats.AbortedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
		else
		{
			m_pDbStats->UpdateTransStats.AbortedTrans.ui64Count++;
			m_pDbStats->UpdateTransStats.AbortedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	if (m_pStats)
	{
		(void)flmStatUpdate( &m_Stats);
	}

Exit:

	m_AbortRc = NE_SFLM_OK;
	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Aborts an active transaction.
*END************************************************************************/
RCODE F_Db::transAbort( void)
{
	RCODE	rc = NE_SFLM_OK;

	if (m_eTransType == SFLM_NO_TRANS)
	{
		rc = RC_SET( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	rc = abortTrans();

Exit:

	if (RC_OK( rc))
	{
		rc = checkState( __FILE__, __LINE__);
	}

	return( rc);
}
