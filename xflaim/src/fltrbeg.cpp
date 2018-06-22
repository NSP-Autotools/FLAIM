//------------------------------------------------------------------------------
// Desc:	Contains routines for starting a transaction.
// Tabs:	3
//
// Copyright (c) 1991, 1994-2007 Novell, Inc. All Rights Reserved.
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
Desc:	This routine unlinks an F_Db from a transaction's list of F_Dbs.
****************************************************************************/
void F_Db::unlinkFromTransList(
	FLMBOOL		bCommitting)
{
	flmAssert( m_pIxdFixups == NULL);
	if( m_eTransType != XFLM_NO_TRANS)
	{
		if (m_uiFlags & FDB_HAS_WRITE_LOCK)
		{

			// If this is a commit operation and we have a commit callback,
			// call the callback function before unlocking the DIB.

			if (bCommitting && m_pCommitClient)
			{
				m_pCommitClient->commit( this);
			}
			unlockExclusive();
		}

		m_pDatabase->lockMutex();
		if (m_pDict)
		{
			unlinkFromDict();
		}

		// Unlink the transaction from the F_Database if it is a read transaction.

		if (m_eTransType == XFLM_READ_TRANS)
		{
			if (m_pNextReadTrans)
			{
				m_pNextReadTrans->m_pPrevReadTrans = m_pPrevReadTrans;
			}
			else if (!m_uiKilledTime)
			{
				m_pDatabase->m_pLastReadTrans = m_pPrevReadTrans;
			}
			if (m_pPrevReadTrans)
			{
				m_pPrevReadTrans->m_pNextReadTrans = m_pNextReadTrans;
			}
			else if (m_uiKilledTime)
			{
				m_pDatabase->m_pFirstKilledTrans = m_pNextReadTrans;
			}
			else
			{
				m_pDatabase->m_pFirstReadTrans = m_pNextReadTrans;
			}

			// Zero out so it will be zero for next transaction begin.

			m_uiKilledTime = 0;
		}
		else
		{

			// Reset to NULL or zero for next update transaction.

			m_pIxStartList = m_pIxStopList = NULL;
			flmAssert( !m_pIxdFixups);
		}

		m_pDatabase->unlockMutex();
		m_eTransType = XFLM_NO_TRANS;
		m_uiFlags &= (~(FDB_UPDATED_DICTIONARY |
								FDB_DONT_KILL_TRANS |
								FDB_DONT_POISON_CACHE |
								FDB_SWEEP_SCHEDULED));
		flmAssert( !m_uiDirtyNodeCount);
	}
}

/****************************************************************************
Desc:	This routine reads a database's dictionary.  This is called only
		when we did not have a dictionary off of the F_Database object -
		which will be the first transaction after a database is opened.
****************************************************************************/
RCODE F_Db::readDictionary( void)
{
	RCODE	rc = NE_XFLM_OK;

	if (RC_BAD( rc = dictOpen()))
	{
		goto Exit;
	}

	m_pDatabase->lockMutex();

	// At this point, we will not yet have opened the database for
	// general use, so there is no way that any other thread can have
	// created a dictionary yet.

	flmAssert( !m_pDatabase->m_pDictList);

	// Link the new local dictionary to its file structure.

	m_pDict->linkToDatabase( m_pDatabase);
	m_pDatabase->unlockMutex();

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine starts a transaction for the specified database.  The
		transaction may be part of an overall larger transaction.
****************************************************************************/
RCODE F_Db::beginTrans(
	eDbTransType	eTransType,
	FLMUINT			uiMaxLockWait,
	FLMUINT			uiFlags,
	XFLM_DB_HDR *	pDbHdr)
{
	RCODE				rc = NE_XFLM_OK;
	XFLM_DB_HDR *	pLastCommittedDbHdr;
	F_Rfl *			pRfl = m_pDatabase->m_pRfl;
	FLMUINT			uiRflToken = 0;
	FLMBOOL			bMutexLocked = FALSE;

	// Should not be calling on a temporary database

	flmAssert( !m_pDatabase->m_bTempDb);

	// Check the state of the database engine

	if( RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Initialize a few things - as few as is necessary to avoid
	// unnecessary overhead.

	m_AbortRc = NE_XFLM_OK;
	pLastCommittedDbHdr = &m_pDatabase->m_lastCommittedDbHdr;
	m_bKrefSetup = FALSE;
	m_eTransType = eTransType;
	m_uiThreadId = (FLMUINT)f_threadId();
	m_uiTransCount++;

	// Link the F_Db to the database's most current F_Dict structure,
	// if there is one.  Also, if it is a read transaction, link the F_Db
	// into the list of read transactions off of the F_Database object.

	m_pDatabase->lockMutex();
	bMutexLocked = TRUE;

	if (m_pDatabase->m_pDictList)
	{
		// Link the F_Db to the right F_Dict object

		linkToDict( m_pDatabase->m_pDictList);
	}

	// If it is a read transaction, link into the list of
	// read transactions off of the F_Database object.  Until we
	// get the DB header transaction ID below, we set ui64CurrTransID
	// to zero and link this transaction in at the beginning of the
	// list.

	if (eTransType == XFLM_READ_TRANS)
	{
		getDbHdrInfo( pLastCommittedDbHdr);

		// Link in at the end of the transaction list.

		m_pNextReadTrans = NULL;
		if ((m_pPrevReadTrans = m_pDatabase->m_pLastReadTrans) != NULL)
		{
			// Make sure transaction IDs are always in ascending order.  They
			// should be at this point.

			flmAssert( m_pDatabase->m_pLastReadTrans->m_ui64CurrTransID <=
							m_ui64CurrTransID);
			m_pDatabase->m_pLastReadTrans->m_pNextReadTrans = this;
		}
		else
		{
			m_pDatabase->m_pFirstReadTrans = this;
		}
		m_pDatabase->m_pLastReadTrans = this;
		m_uiInactiveTime = 0;

		if (uiFlags & XFLM_DONT_KILL_TRANS)
		{
			m_uiFlags |= FDB_DONT_KILL_TRANS;
		}
		else
		{
			m_uiFlags &= ~FDB_DONT_KILL_TRANS;
		}
		
		if (pDbHdr)
		{
			f_memcpy( pDbHdr, &m_pDatabase->m_lastCommittedDbHdr,
						sizeof( XFLM_DB_HDR));
		}
	}

	m_pDatabase->unlockMutex();
	bMutexLocked = FALSE;

	if (uiFlags & XFLM_DONT_POISON_CACHE)
	{
		m_uiFlags |= FDB_DONT_POISON_CACHE;
	}
	else
	{
		m_uiFlags &= ~FDB_DONT_POISON_CACHE;
	}

	// Put an exclusive lock on the database if we are not in a read
	// transaction.  Read transactions require no lock.

	if (eTransType != XFLM_READ_TRANS)
	{
		// Set the m_bHadUpdOper to TRUE for all transactions to begin with.
		// Many calls to beginTrans are internal, and we WANT the
		// normal behavior at the end of the transaction when it is
		// committed or aborted.  The only time this flag will be set
		// to FALSE is when the application starts the transaction as
		// opposed to an internal starting of the transaction.

		m_bHadUpdOper = TRUE;

		// Initialize the count of blocks changed to be 0

		m_uiBlkChangeCnt = 0;

		if (RC_BAD( rc = lockExclusive( uiMaxLockWait)))
		{
			goto Exit;
		}

		flmAssert( !m_pDatabase->m_DocumentList.m_uiLastCollection);
		flmAssert( !m_pDatabase->m_DocumentList.m_ui64LastDocument);

		// If there was a problem with the RFL volume, we must wait
		// for a checkpoint to be completed before continuing.
		// The checkpoint thread looks at this same flag and forces
		// a checkpoint.  If it completes one successfully, it will
		// reset this flag.
		// Also, if the last forced checkpoint had a problem
		// (pFile->CheckpointRc != NE_XFLM_OK), we don't want to
		// start up a new update transaction until it is resolved.

		if( !pRfl->seeIfRflVolumeOk() || RC_BAD( m_pDatabase->m_CheckpointRc))

		{
			rc = RC_SET( NE_XFLM_MUST_WAIT_CHECKPOINT);
			goto Exit;
		}

		// Set the first log block address to zero.

		m_pDatabase->m_uiFirstLogBlkAddress = 0;

		// Header must be read before opening roll forward log file to make
		// sure we have the most current log file and log options.

		f_memcpy( &m_pDatabase->m_uncommittedDbHdr, pLastCommittedDbHdr,
			sizeof( XFLM_DB_HDR));
		getDbHdrInfo( pLastCommittedDbHdr);

		// Need to increment the current checkpoint for update transactions
		// so that it will be correct when we go to mark cache blocks.

		if (m_uiFlags & FDB_REPLAYING_RFL)
		{
			// During recovery we need to set the transaction ID to the
			// transaction ID that was logged.

			m_ui64CurrTransID = pRfl->getCurrTransID();
		}
		else
		{
			m_ui64CurrTransID++;
		}

		// Link F_Db to the most current local dictionary, if there
		// is one.

		m_pDatabase->lockMutex();
		if (m_pDatabase->m_pDictList != m_pDict &&
			 m_pDatabase->m_pDictList)
		{
			linkToDict( m_pDatabase->m_pDictList);
		}
		m_pDatabase->unlockMutex();

		// Set the transaction EOF to the current file EOF

		m_uiTransEOF = m_uiLogicalEOF;

		// Put the transaction ID into the uncommitted log header.

		m_pDatabase->m_uncommittedDbHdr.ui64CurrTransID = m_ui64CurrTransID;

		if (pDbHdr)
		{
			f_memcpy( pDbHdr, &m_pDatabase->m_uncommittedDbHdr,
							sizeof( XFLM_DB_HDR));
		}
	}

	// Set up to collect statistics.  We only do this at transaction
	// begin and not on any other type of operation.  So this is the
	// only time when an F_Db will sense that statistics have been
	// turned on or off.

	if (!gv_XFlmSysData.Stats.bCollectingStats)
	{
		m_pStats = NULL;
		m_pDbStats = NULL;
	}
	else
	{
		m_pStats = &m_Stats;

		// Statistics are being collected for the system.  Therefore,
		// if we are not currently collecting statistics in the
		// session, start.  If we were collecting statistics, but the
		// start time was earlier than the start time in the system
		// statistics structure, reset the statistics in the session.

		if (!m_Stats.bCollectingStats)
		{
			flmStatStart( &m_Stats);
		}
		else if (m_Stats.uiStartTime < gv_XFlmSysData.Stats.uiStartTime)
		{
			flmStatReset( &m_Stats, FALSE);
		}
		(void)flmStatGetDb( &m_Stats, m_pDatabase,
						0, &m_pDbStats, NULL, NULL);
		m_pLFileStats = NULL;
	}

	if (m_pDbStats)
	{
		f_timeGetTimeStamp( &m_TransStartTime);
	}

	// If we do not have a dictionary, read it in from disk.
	// NOTE: This should only happen when we are first opening
	// the database.

	if (!m_pDict)
	{
		if (eTransType != XFLM_READ_TRANS)
		{
			pRfl->disableLogging( &uiRflToken);
		}

		if (RC_BAD( rc = readDictionary()))
		{
			goto Exit;
		}
	}

Exit:

	if( bMutexLocked)
	{
		m_pDatabase->unlockMutex();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if (eTransType != XFLM_READ_TRANS)
	{
		if (RC_OK( rc))
		{
			rc = pRfl->logBeginTransaction( this);
		}
#ifdef FLM_DBG_LOG
		flmDbgLogUpdate( m_pDatabase, m_ui64CurrTransID,
				0, 0, rc, "TBeg");
#endif
	}

	if (eTransType == XFLM_UPDATE_TRANS &&
		 gv_XFlmSysData.EventHdrs [XFLM_EVENT_UPDATES].pEventCBList)
	{
		flmTransEventCallback( XFLM_EVENT_BEGIN_TRANS, this, rc,
					(FLMUINT)(RC_OK( rc)
								 ? m_ui64CurrTransID
								 : (FLMUINT64)0));
	}

	if (RC_BAD( rc))
	{
		// If there was an error, unlink the database from the transaction
		// structure as well as from the FDICT structure.  Also dump any nodes
	    // that are already in the cache.

		unlinkFromTransList( FALSE);

		if (m_pStats)
		{
			(void)flmStatUpdate( &m_Stats);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine starts a transaction for the specified database.  It uses
		the transaction information in the passed in pDb.
****************************************************************************/
RCODE F_Db::beginTrans(
	F_Db *	pDb)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bMutexLocked = FALSE;

	// Should not be calling on a temporary database

	flmAssert( !m_pDatabase->m_bTempDb);
	
	// pDb better be running a read transaction.
	
	flmAssert( pDb->m_eTransType == XFLM_READ_TRANS);

	if( RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Initialize a few things - as few as is necessary to avoid
	// unnecessary overhead.

	m_AbortRc = NE_XFLM_OK;
	m_bKrefSetup = FALSE;
	m_eTransType = XFLM_READ_TRANS;
	m_uiThreadId = (FLMUINT)f_threadId();
	m_uiTransCount++;

	// Link the F_Db to the database's most current F_Dict structure,
	// if there is one.  Also, if it is a read transaction, link the F_Db
	// into the list of read transactions off of the F_Database object.

	m_pDatabase->lockMutex();
	bMutexLocked = TRUE;
	
	// Link to the same dictionary as pDb.
	
	linkToDict( pDb->m_pDict);

	// If it is a read transaction, link into the list of
	// read transactions off of the F_Database object.  Until we
	// get the DB header transaction ID below, we set ui64CurrTransID
	// to zero and link this transaction in at the beginning of the
	// list.

	getDbHdrInfo( pDb);

	// Link into the transaction list right after the point where
	// pDb is linked in.  We need to keep transaction IDs in ascending
	// order.
	
	m_pPrevReadTrans = pDb;
	if ((m_pNextReadTrans = pDb->m_pNextReadTrans) != NULL)
	{
		m_pNextReadTrans->m_pPrevReadTrans = this;
	}
	else
	{
		m_pDatabase->m_pLastReadTrans = this;
	}
	pDb->m_pNextReadTrans = this;

	m_uiInactiveTime = 0;
	
	if (pDb->m_uiFlags & FDB_DONT_KILL_TRANS)
	{
		m_uiFlags |= FDB_DONT_KILL_TRANS;
	}
	else
	{
		m_uiFlags &= ~FDB_DONT_KILL_TRANS;
	}
	if (pDb->m_uiFlags & FDB_DONT_POISON_CACHE)
	{
		m_uiFlags |= FDB_DONT_POISON_CACHE;
	}
	else
	{
		m_uiFlags &= ~FDB_DONT_POISON_CACHE;
	}

	m_pDatabase->unlockMutex();
	bMutexLocked = FALSE;

	// Set up to collect statistics.  We only do this at transaction
	// begin and not on any other type of operation.  So this is the
	// only time when an F_Db will sense that statistics have been
	// turned on or off.

	if (!gv_XFlmSysData.Stats.bCollectingStats)
	{
		m_pStats = NULL;
		m_pDbStats = NULL;
	}
	else
	{
		m_pStats = &m_Stats;

		// Statistics are being collected for the system.  Therefore,
		// if we are not currently collecting statistics in the
		// session, start.  If we were collecting statistics, but the
		// start time was earlier than the start time in the system
		// statistics structure, reset the statistics in the session.

		if (!m_Stats.bCollectingStats)
		{
			flmStatStart( &m_Stats);
		}
		else if (m_Stats.uiStartTime < gv_XFlmSysData.Stats.uiStartTime)
		{
			flmStatReset( &m_Stats, FALSE);
		}
		(void)flmStatGetDb( &m_Stats, m_pDatabase,
						0, &m_pDbStats, NULL, NULL);
		m_pLFileStats = NULL;
	}

	if (m_pDbStats)
	{
		f_timeGetTimeStamp( &m_TransStartTime);
	}

Exit:

	if( bMutexLocked)
	{
		m_pDatabase->unlockMutex();
	}

	if (RC_BAD( rc))
	{
		// If there was an error, unlink the database from the transaction
		// structure as well as from the FDICT structure.  Also dump any nodes
	    // that are already in the cache.

		unlinkFromTransList( FALSE);

		if (m_pStats)
		{
			(void)flmStatUpdate( &m_Stats);
		}
	}

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Starts a transaction.
*END************************************************************************/
RCODE XFLAPI F_Db::transBegin(
	eDbTransType	eTransType,
		// [IN] Specifies the type of transaction to begin.
		// Possible values are:
		//
		// XFLM_READ_TRANS:  Begins a read transaction.
		// XFLM_UPDATE_TRANS:  Begins an update transaction.
	FLMUINT			uiMaxLockWait,
		// [IN] Maximum lock wait time.  Specifies the amount of time
		// to wait for lock requests occuring during the transaction
		// to be granted.  Valid values are 0 through 255 seconds.  Zero
		// is used to specify no-wait locks.
	FLMUINT			uiFlags,
		// Transaction flags.
	XFLM_DB_HDR *	pDbHdr
		// [IN] 2K buffer
		// [OUT] Returns the DB header for the file.
	)
{
	RCODE		rc = NE_XFLM_OK;

	// Verify the transaction type.

	if (eTransType != XFLM_UPDATE_TRANS && eTransType != XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_TYPE);
		goto Exit;
	}

	// Verify the transaction flags

	if ((uiFlags & XFLM_DONT_KILL_TRANS) && eTransType != XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_TYPE);
		goto Exit;
	}

	// Can't start an update transaction on a database that
	// is locked in shared mode.

	if (eTransType == XFLM_UPDATE_TRANS && (m_uiFlags & FDB_FILE_LOCK_SHARED))
	{
		rc = RC_SET( NE_XFLM_SHARED_LOCK);
		goto Exit;
	}

	// If the database is not running a transaction, start one.

	if (m_eTransType != XFLM_NO_TRANS)
	{

		// Cannot nest transactions.

		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	if (RC_BAD( rc = beginTrans( eTransType,
								uiMaxLockWait, uiFlags, pDbHdr)))
	{
		goto Exit;
	}

	m_bHadUpdOper = FALSE;

Exit:

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Starts a transaction.
*END************************************************************************/
RCODE XFLAPI F_Db::transBegin(
	IF_Db *	pDb
		// [IN] Start a transaction that has the same view as whatever
		// transaction is running on this database.  NOTE: If pDb is
		// running an update transaction, it is illegal for another pDb
		// to also run an update transaction, so such a request would fail.
	)
{
	RCODE		rc = NE_XFLM_OK;
	
	// Database cannot already be running a transaction.

	if (m_eTransType != XFLM_NO_TRANS)
	{

		// Cannot nest transactions.

		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}
	
	// Verify the transaction type.

	if (((F_Db *)pDb)->m_eTransType != XFLM_READ_TRANS)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_TRANS_TYPE);
		goto Exit;
	}

	if (RC_BAD( rc = beginTrans( (F_Db *)pDb)))
	{
		goto Exit;
	}
	m_bHadUpdOper = FALSE;

Exit:

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Obtains a a lock on the database.
*END************************************************************************/
RCODE XFLAPI F_Db::dbLock(
	eLockType		lockType,
		// [IN] Type of lock request - must be FLM_LOCK_EXCLUSIVE or
		// FLM_LOCK_SHARED
	FLMINT			iPriority,
		// [IN] Priority to be assigned to lock.
	FLMUINT			uiTimeout
		// [IN] Seconds to wait for lock to be granted.  FLM_NO_TIMEOUT
		// means that it will wait forever for the lock to be granted.
	)
{
	RCODE	rc = NE_XFLM_OK;

	// lockType better be exclusive or shared

	if (lockType != FLM_LOCK_EXCLUSIVE && lockType != FLM_LOCK_SHARED)
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Nesting of locks is not allowed - this test also keeps this call from
	// being executed inside an update transaction that implicitly acquired
	// the lock.

	if (m_uiFlags &
			(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED | FDB_FILE_LOCK_IMPLICIT))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Attempt to acquire the lock.

	if (RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->lock( m_hWaitSem,
		(FLMBOOL)((lockType == FLM_LOCK_EXCLUSIVE)
									  ? (FLMBOOL)TRUE
									  : (FLMBOOL)FALSE),
									  uiTimeout, iPriority, 
									  m_pDbStats ? &m_pDbStats->LockStats : NULL)))
	{
		goto Exit;
	}
	m_uiFlags |= FDB_HAS_FILE_LOCK;
	if (lockType == FLM_LOCK_SHARED)
	{
		m_uiFlags |= FDB_FILE_LOCK_SHARED;
	}

Exit:

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Releases a lock on the database
*END************************************************************************/
RCODE XFLAPI F_Db::dbUnlock( void)
{
	RCODE	rc = NE_XFLM_OK;

	// If we don't have an explicit lock, can't do the unlock.  It is
	// also illegal to do the unlock during an update transaction.

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK) ||
		 (m_uiFlags & FDB_FILE_LOCK_IMPLICIT) ||
		 (m_eTransType == XFLM_UPDATE_TRANS))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Unlock the file.

	if (RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->unlock()))
	{
		goto Exit;
	}

	// Unset the flags that indicated the file was explicitly locked.

	m_uiFlags &= (~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED));

Exit:

	if (RC_OK( rc))
	{
		rc = checkState( __FILE__, __LINE__);
	}

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Returns information about current and pending locks on the
		 database.
*END************************************************************************/
RCODE XFLAPI F_Db::getLockInfo(
	FLMINT			iPriority,
		// [IN] A count of all locks with a priority >= to this priority
		// level will be returned in pLockInfo.
	eLockType *		pCurrLockType,
	FLMUINT *		puiThreadId,
	FLMUINT *		puiNumExclQueued,
	FLMUINT *		puiNumSharedQueued,
	FLMUINT *		puiPriorityCount
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	m_pDatabase->m_pDatabaseLockObj->getLockInfo( iPriority,
								pCurrLockType, puiThreadId,
								puiNumExclQueued, puiNumSharedQueued,
								puiPriorityCount);

Exit:

	return( rc);
}

/*API~***********************************************************************
Desc : Returns information about the lock held by the specified database
		 handle.
*END************************************************************************/
RCODE XFLAPI F_Db::getLockType(
	eLockType *		pLockType,
	FLMBOOL *		pbImplicit)
{
	RCODE		rc = NE_XFLM_OK;

	if (pLockType)
	{
		*pLockType = FLM_LOCK_NONE;
	}

	if (pbImplicit)
	{
		*pbImplicit = FALSE;
	}

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	if (m_uiFlags & FDB_HAS_FILE_LOCK)
	{
		if (pLockType)
		{
			if (m_uiFlags & FDB_FILE_LOCK_SHARED)
			{
				*pLockType = FLM_LOCK_SHARED;
			}
			else
			{
				*pLockType = FLM_LOCK_EXCLUSIVE;
			}
		}

		if (pbImplicit)
		{
			*pbImplicit = (m_uiFlags & FDB_FILE_LOCK_IMPLICIT)
								? TRUE
								: FALSE;
		}
	}

Exit:

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Forces a checkpoint on the database.
*END************************************************************************/
RCODE XFLAPI F_Db::doCheckpoint(
	FLMUINT	uiTimeout
		// [IN] Seconds to wait to obtain lock on the database.
		// FLM_NO_TIMEOUT means that it will wait forever for
		// the lock to be granted.
	)
{
	RCODE		rc = NE_XFLM_OK;

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Start an update transaction.  Must not already be one going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// If we get to this point, we need to start a transaction on the
	// database.

	if( RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS, uiTimeout)))
	{
		goto Exit;
	}

	// Commit the transaction, forcing it to be checkpointed.

	m_bHadUpdOper = FALSE;
	if (RC_BAD( rc = commitTrans( 0, TRUE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine locks a database for exclusive access.
****************************************************************************/
RCODE F_Db::lockExclusive(
	FLMUINT	uiMaxLockWait)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bGotFileLock = FALSE;

	flmAssert( !m_pDatabase->m_bTempDb);

	// There must NOT be a shared lock on the file.

	if (m_uiFlags & FDB_FILE_LOCK_SHARED)
	{
		rc = RC_SET( NE_XFLM_SHARED_LOCK);
		goto Exit;
	}

	// Must acquire an exclusive file lock first, if it hasn't been
	// acquired.

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if (RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->lock(
			m_hWaitSem, TRUE, uiMaxLockWait, 0, 
			m_pDbStats ? &m_pDbStats->LockStats : NULL)))
		{
			goto Exit;
		}
		bGotFileLock = TRUE;
		m_uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
	}

	if (RC_OK( rc = m_pDatabase->dbWriteLock( m_hWaitSem, m_pDbStats)))
	{
		m_uiFlags |= FDB_HAS_WRITE_LOCK;
	}

Exit:

	if (rc == NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT)
	{
		if (bGotFileLock)
		{
			(void)m_pDatabase->m_pDatabaseLockObj->unlock();
			m_uiFlags &= (~(FDB_HAS_FILE_LOCK |
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}

		if (m_eTransType != XFLM_NO_TRANS)
		{

			// Unlink the DB from the transaction.

			unlinkFromTransList( FALSE);
		}
	}
	else if (RC_BAD( rc))
	{
		if (bGotFileLock)
		{
			(void)m_pDatabase->m_pDatabaseLockObj->unlock();
			m_uiFlags &= (~(FDB_HAS_FILE_LOCK |
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine unlocks a database that was previously locked
		using the lockExclusive routine.
****************************************************************************/
void F_Db::unlockExclusive( void)
{
	flmAssert( !m_pDatabase->m_bTempDb);

	// If we have the write lock, unlock it first.

	flmAssert( m_uiFlags & FDB_HAS_WRITE_LOCK);

	m_pDatabase->dbWriteUnlock();
	m_uiFlags &= ~FDB_HAS_WRITE_LOCK;

	// Give up the file lock, if it was acquired implicitly.

	if (m_uiFlags & FDB_FILE_LOCK_IMPLICIT)
	{
		(void)m_pDatabase->m_pDatabaseLockObj->unlock();
		m_uiFlags &= (~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT));
	}

}
