//-------------------------------------------------------------------------
// Desc:	Begin transaction
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
Desc:	Starts a transaction.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbTransBegin(
	HFDB			hDb,
	FLMUINT		uiTransType,
	FLMUINT		uiMaxLockWait,
	FLMBYTE *	pucHeader)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bIgnore;
	FLMUINT		uiFlags = FLM_GET_TRANS_FLAGS( uiTransType);
	FDB *			pDb = (FDB *)hDb;

	uiTransType = FLM_GET_TRANS_TYPE( uiTransType);

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE		Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			if( RC_BAD( rc = Wire.doTransOp(
				FCS_OP_TRANSACTION_BEGIN, uiTransType, uiFlags,
				uiMaxLockWait, pucHeader)))
			{
				goto Exit;
			}
		}

		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// Verify the transaction type.

	if (( uiTransType != FLM_UPDATE_TRANS) &&
		 ( uiTransType != FLM_READ_TRANS))
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS);
		goto Exit;
	}

	// Verify the transaction flags

	if( (uiFlags & FLM_DONT_KILL_TRANS) && uiTransType != FLM_READ_TRANS)
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS);
		goto Exit;
	}

	// Can't start an update transaction on a database that
	// is locked in shared mode.

	if ((uiTransType == FLM_UPDATE_TRANS) &&
		 (pDb->uiFlags & FDB_FILE_LOCK_SHARED))
	{
		rc = RC_SET( FERR_PERMISSION);
		goto Exit;
	}

	// If the database has an invisible transaction going, abort it
	// before going any further - we don't want application transactions
	// to be nested under invisible transactions.  Application transactions
	// take precedence over invisible transactions.

	if ((pDb->uiTransType != FLM_NO_TRANS) &&
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		if (RC_BAD( rc = flmAbortDbTrans( pDb)))
		{
			goto Exit;
		}
	}

	// If the database is not running a transaction, start one.
	// Otherwise, start a nested transaction - first verifying that
	// the transation type matches.

	if (pDb->uiTransType == FLM_NO_TRANS)
	{
		FLMUINT		uiBytesRead;

		if( pucHeader)
		{
			if( RC_BAD( rc = pDb->pSFileHdl->readBlock( 
				0, 2048, pucHeader, &uiBytesRead)))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = flmBeginDbTrans( pDb, uiTransType, 
			uiMaxLockWait, uiFlags,
			pucHeader ? &pucHeader [16] : NULL)))
		{
			goto Exit;
		}
		pDb->bHadUpdOper = FALSE;
	}
	else
	{
		// Cannot nest transactions.

		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

Exit:

	flmExit( FLM_DB_TRANS_BEGIN, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	Commits an active transaction.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbTransCommit(
	HFDB			hDb,
	FLMBOOL *	pbEmpty)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE		Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp(	FCS_OP_TRANSACTION_COMMIT, 0, 0, 0);
		}
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If there is an invisible transaction going, it should not be
	// commitable by an application.

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if( RC_BAD( pDb->AbortRc))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	if (pbEmpty)
	{
		*pbEmpty = FALSE;
	}
	rc = flmCommitDbTrans( pDb, 0, FALSE, pbEmpty);

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_TRANS_COMMIT, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	Aborts an active transaction.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbTransAbort(
	HFDB			hDb)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE	Wire( pDb->pCSContext, pDb);
		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp( FCS_OP_TRANSACTION_ABORT, 0, 0, 0);
		}
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If there is an invisible transaction going, it should not be
	// abortable by an application.

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}
	
	rc = flmAbortDbTrans( pDb);

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_TRANS_ABORT, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : Returns the type of the current database transaction.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetTransType(
	HFDB			hDb,
	FLMUINT *	puiTransTypeRV)
{
	RCODE		   rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pDb->pCSContext, pDb);

		// Send a request to get the transaction type.

		if( RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_TRANS, FCS_OP_TRANSACTION_GET_TYPE)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response.
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		
		*puiTransTypeRV = Wire.getTransType();
		rc = Wire.getRCode();
		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (!pDb)
	{
		rc = RC_SET( FERR_BAD_HDL);
		goto Exit;
	}

	fdbUseCheck( pDb);
	pDb->uiInitNestLevel++;
	(void)flmResetDiag( pDb);

	// If the transaction is an internal transaction that is invisible to
	// the application, return FLM_NO_TRANS.  Application is not supposed
	// see invisible transactions.

	*puiTransTypeRV = (FLMUINT)(((pDb->uiTransType == FLM_NO_TRANS) ||
										  (pDb->uiFlags & FDB_INVISIBLE_TRANS))
										  			? (FLMUINT)FLM_NO_TRANS
													: pDb->uiTransType);

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

Exit:

	flmExit( FLM_DB_GET_TRANS_TYPE, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : Retrieves the current transaction number of a database
Notes: This routine should only be called only from within an update
		 transaction since read transactions are not assigned a transaction
		 number.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetTransId(
	HFDB				hDb,
	FLMUINT *		puiTrNumRV)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		
		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send a request to get the transaction ID.

		if (RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_GET_TRANS_ID)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		*puiTrNumRV = Wire.getTransId();

		rc = Wire.getRCode();
		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	*puiTrNumRV = pDb->LogHdr.uiCurrTransID;

Exit:

	flmExit( FLM_DB_GET_TRANS_ID, pDb, rc);
	return( rc);
}


/****************************************************************************
Desc: 	Retrieves the last commit sequence number of a database.
Notes: 	Whenever a transaction is committed, FLAIM increments the commit
		 	sequence number to indicate that the database has been modified.
			An application may use this routine to determine if the database
			has been modified.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetCommitCnt(
	HFDB				hDb,
	FLMUINT *		puiCommitCount)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send a request to get the commit count

		if (RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_DATABASE, FCS_OP_GET_COMMIT_CNT)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response.
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		*puiCommitCount = (FLMUINT)Wire.getCount();

		rc = Wire.getRCode();
		goto ExitCS;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto ExitCS;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if (pDb->uiTransType != FLM_NO_TRANS)
	{
		if (flmCheckBadTrans( pDb))
		{
			rc = RC_SET( FERR_ABORT_TRANS);
			goto Exit;
		}
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	*puiCommitCount = (FLMUINT)FB2UD(
			&pDb->pFile->ucLastCommittedLogHdr [LOG_COMMIT_COUNT]);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit:
ExitCS:

	flmExit( FLM_DB_GET_COMMIT_CNT, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	Forces a checkpoint on the database.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbCheckpoint(
	HFDB			hDb,
	FLMUINT		uiTimeout)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bStartedTrans;

	bStartedTrans = FALSE;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_CHECKPOINT)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiTimeout)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	// Start an update transaction.  Must not already be one going.

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
									  0, uiTimeout | FLM_AUTO_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Commit the transaction, forcing it to be checkpointed.

	bStartedTrans = FALSE;
	pDb->bHadUpdOper = FALSE;
	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	if (bStartedTrans)
	{
		(void)flmAbortDbTrans( pDb);
	}

	flmExit( FLM_DB_CHECKPOINT, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	This routine unlinks an FDB from a transaction's list of FDBs.
****************************************************************************/
void flmUnlinkDbFromTrans(
	FDB *			pDb,
	FLMBOOL		bCommitting)
{
	FFILE *	pFile = pDb->pFile;

	flmAssert( pDb->pIxdFixups == NULL);
	if( pDb->uiTransType != FLM_NO_TRANS)
	{
		if (pDb->uiFlags & FDB_HAS_WRITE_LOCK)
		{

			// If this is a commit operation and we have a commit callback,
			// call the callback function before unlocking the DIB.

			if( bCommitting && pDb->fnCommit)
			{
				FLMBOOL	bSavedInvisTrans;

				CB_ENTER( pDb, &bSavedInvisTrans);
				pDb->fnCommit( (HFDB)pDb, pDb->pvCommitData);
				CB_EXIT( pDb, bSavedInvisTrans);
			}

			dbUnlock( pDb);
		}

		f_mutexLock( gv_FlmSysData.hShareMutex);
		if (pDb->pDict)
		{
			flmUnlinkFdbFromDict( pDb);
		}

		// Unlink the transaction from the FFILE if it is a read
		// transaction

		if( pDb->uiTransType == FLM_READ_TRANS)
		{
			if (pDb->pNextReadTrans)
			{
				pDb->pNextReadTrans->pPrevReadTrans = pDb->pPrevReadTrans;
			}
			else if (!pDb->uiKilledTime)
			{
				pFile->pLastReadTrans = pDb->pPrevReadTrans;
			}
			if (pDb->pPrevReadTrans)
			{
				pDb->pPrevReadTrans->pNextReadTrans = pDb->pNextReadTrans;
			}
			else if (pDb->uiKilledTime)
			{
				pFile->pFirstKilledTrans = pDb->pNextReadTrans;
			}
			else
			{
				pFile->pFirstReadTrans = pDb->pNextReadTrans;
			}

			// Zero out so it will be zero for next transaction begin.

			pDb->uiKilledTime = 0;
		}
		else
		{
			// Reset to NULL or zero for next update transaction.

			pDb->pBlobList = NULL;
			pDb->pIxStartList = pDb->pIxStopList = NULL;
			flmAssert( pDb->pIxdFixups == NULL);
		}

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		pDb->uiTransType = FLM_NO_TRANS;
		pDb->uiFlags &= (~(FDB_UPDATED_DICTIONARY |
								FDB_INVISIBLE_TRANS |
								FDB_DONT_KILL_TRANS |
								FDB_DONT_POISON_CACHE));
	}
}

/****************************************************************************
Desc:	This routine starts a transaction for the specified database.  The
		transaction may be part of an overall larger transaction.
****************************************************************************/
RCODE flmBeginDbTrans(
	FDB *			pDb,
	FLMUINT		uiTransType,
	FLMUINT		uiMaxLockWait,
	FLMUINT		uiFlags,
	FLMBYTE *	pucLogHdr)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMBOOL		bMutexLocked = FALSE;
	FLMBYTE *	pucLastCommittedLogHdr;
	DB_STATS *	pDbStats = pDb->pDbStats;

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// Initialize a few things - as few as is necessary to avoid
	// unnecessary overhead.

	pDb->eAbortFuncId = FLM_UNKNOWN_FUNC;
	pDb->AbortRc = FERR_OK;
	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	pDb->KrefCntrl.bKrefSetup = FALSE;
	pDb->uiTransType = uiTransType;
	pDb->uiThreadId = (FLMUINT)f_threadId();
	pDb->uiTransCount++;

	// Link the FDB to the file's most current FDICT structure,
	// if there is one.
	//
	// Also, if it is a read transaction, link the FDB
	// into the list of read transactions off of
	// the FFILE structure.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	if (pFile->pDictList)
	{

		// Link the FDB to the FDICT.

		flmLinkFdbToDict( pDb, pFile->pDictList);
	}

	// If it is a read transaction, link into the list of
	// read transactions off of the FFILE structure.  Until we
	// get the log header transaction ID below, we set uiCurrTransID
	// to zero and link this transaction in at the beginning of the
	// list.

	if (uiTransType == FLM_READ_TRANS)
	{
		flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);

		// Link in at the end of the transaction list.

		pDb->pNextReadTrans = NULL;
		if ((pDb->pPrevReadTrans = pFile->pLastReadTrans) != NULL)
		{

			// Make sure transaction IDs are always in ascending order.  They
			// should be at this point.

			flmAssert( pFile->pLastReadTrans->LogHdr.uiCurrTransID <=
							pDb->LogHdr.uiCurrTransID);
			pFile->pLastReadTrans->pNextReadTrans = pDb;
		}
		else
		{
			pFile->pFirstReadTrans = pDb;
		}
		pFile->pLastReadTrans = pDb;
		pDb->uiInactiveTime = 0;

		if( uiFlags & FLM_DONT_KILL_TRANS)
		{
			pDb->uiFlags |= FDB_DONT_KILL_TRANS;
		}
		else
		{
			pDb->uiFlags &= ~FDB_DONT_KILL_TRANS;
		}
		
		if (pucLogHdr)
		{
			f_memcpy( pucLogHdr, &pDb->pFile->ucLastCommittedLogHdr[0],
						LOG_HEADER_SIZE);
		}
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	if( uiFlags & FLM_DONT_POISON_CACHE)
	{
		pDb->uiFlags |= FDB_DONT_POISON_CACHE;
	}
	else
	{
		pDb->uiFlags &= ~FDB_DONT_POISON_CACHE;
	}

	// Put an exclusive lock on the database if we are not in a read
	// transaction.  Read transactions require no lock.

	if (uiTransType != FLM_READ_TRANS)
	{
		flmAssert( pDb->pIxStats == NULL);

		// Set the bHadUpdOper to TRUE for all transactions to begin with.
		// Many calls to flmBeginDbTrans are internal, and we WANT the
		// normal behavior at the end of the transaction when it is
		// committed or aborted.  The only time this flag will be set
		// to FALSE is when the application starts the transaction as
		// opposed to an internal starting of the transaction.

		pDb->bHadUpdOper = TRUE;

		// Initialize the count of blocks changed to be 0

		pDb->uiBlkChangeCnt = 0;

		if (RC_BAD( rc = dbLock( pDb, uiMaxLockWait)))
		{
			goto Exit;
		}

		// If there was a problem with the RFL volume, we must wait
		// for a checkpoint to be completed before continuing.
		// The checkpoint thread looks at this same flag and forces
		// a checkpoint.  If it completes one successfully, it will
		// reset this flag.
		//
		// Also, if the last forced checkpoint had a problem
		// (pFile->CheckpointRc != FERR_OK), we don't want to
		// start up a new update transaction until it is resolved.

		if (!pFile->pRfl->seeIfRflVolumeOk() ||
			 RC_BAD( pFile->CheckpointRc))
		{
			rc = RC_SET( FERR_MUST_WAIT_CHECKPOINT);
			goto Exit;
		}

		// Set the first log block address to zero.

		pFile->uiFirstLogBlkAddress = 0;

		// Header must be read before opening roll forward log file to make
		// sure we have the most current log file and log options.

		f_memcpy( pFile->ucUncommittedLogHdr, pucLastCommittedLogHdr,
			LOG_HEADER_SIZE);
		flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);

		// Need to increment the current checkpoint for update transactions
		// so that it will be correct when we go to mark cache blocks.

		if (pDb->uiFlags & FDB_REPLAYING_RFL)
		{
			// During recovery we need to set the transaction ID to the
			// transaction ID that was logged.

			pDb->LogHdr.uiCurrTransID = pFile->pRfl->getCurrTransID();
		}
		else
		{
			pDb->LogHdr.uiCurrTransID++;
		}
		f_mutexLock( gv_FlmSysData.hShareMutex);

		// Link FDB to the most current local dictionary, if there
		// is one.

		if (pFile->pDictList != pDb->pDict && pFile->pDictList)
		{
			flmLinkFdbToDict( pDb, pFile->pDictList);
		}
		pFile->uiUpdateTransID = pDb->LogHdr.uiCurrTransID;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		// Set the transaction EOF to the current file EOF

		pDb->uiTransEOF = pDb->LogHdr.uiLogicalEOF;

		// Put the transaction ID into the uncommitted log header.

		UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
					&pFile->ucUncommittedLogHdr [LOG_CURR_TRANS_ID]);

		if (pucLogHdr)
		{
			f_memcpy( pucLogHdr, &pDb->pFile->ucUncommittedLogHdr [0],
							LOG_HEADER_SIZE);
		}
	}

	if (pDbStats)
	{
		f_timeGetTimeStamp( &pDb->TransStartTime);
	}

	// If we do not have a dictionary, read it in from disk.
	// NOTE: This should only happen when we are first opening
	// the database.

	if (!pDb->pDict)
	{
		flmAssert( pDb->pFile->uiFlags & DBF_BEING_OPENED);
	
		if (RC_BAD( rc = fdictRebuild( pDb)))
		{
			if (pDb->pDict)
			{
				flmFreeDict( pDb->pDict);
				pDb->pDict = NULL;
			}
			
			goto Exit;
		}
	
		f_mutexLock( gv_FlmSysData.hShareMutex);
	
		// At this point, we will not yet have opened the database for
		// general use, so there is no way that any other thread can have
		// created a dictionary yet.
	
		flmAssert( pDb->pFile->pDictList == NULL);
	
		// Link the new local dictionary to its file structure.
	
		flmLinkDictToFile( pDb->pFile, pDb->pDict);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (uiTransType != FLM_READ_TRANS)
	{
		if (RC_OK( rc))
		{
			rc = pFile->pRfl->logBeginTransaction( pDb);
		}
#ifdef FLM_DBG_LOG
		flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
				0, 0, rc, "TBeg");
#endif
	}

	if( uiTransType == FLM_UPDATE_TRANS &&
		 gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmTransEventCallback( F_EVENT_BEGIN_TRANS, (HFDB)pDb, rc,
					(FLMUINT)(RC_OK( rc)
								 ? pDb->LogHdr.uiCurrTransID
								 : (FLMUINT)0));
	}

	if (RC_BAD( rc))
	{
		// If there was an error, unlink the database from the transaction
		// structure as well as from the FDICT structure.

		flmUnlinkDbFromTrans( pDb, FALSE);

		if (pDb->pStats)
		{
			(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine aborts an active transaction for a particular
		database.  If the database is open via a server, a message is
		sent to the server to abort the transaction.  Otherwise, the
		transaction is rolled back locally.
****************************************************************************/
RCODE flmAbortDbTrans(
	FDB *				pDb,
	FLMBOOL			bOkToLogAbort)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiTransType;
	FLMBYTE *		pucLastCommittedLogHdr;
	FLMBYTE *		pucUncommittedLogHdr;
	FLMBOOL			bDumpedCache = FALSE;
	DB_STATS *		pDbStats = pDb->pDbStats;
	FLMBOOL			bKeepAbortedTrans;
	FLMUINT			uiTransId;
	FLMBOOL			bInvisibleTrans;

	// Get transaction type

	if ((uiTransType = pDb->uiTransType) == FLM_NO_TRANS)
	{
		goto Exit;
	}

	// No recovery required if it is a read transaction.

	if (uiTransType == FLM_READ_TRANS)
	{

		if( pDb->KrefCntrl.bKrefSetup)
		{
			// KrefCntrlFree could be called w/o checking bKrefSetup because
			// it checks the flag, but it is more optimal to check the
			// flag before making the call because most of the time it will
			// be false.

			KrefCntrlFree( pDb);
		}

		goto Unlink_From_Trans;
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			0, 0, FERR_OK, "TAbrt");
#endif

	pFile->pRfl->clearLogHdrs();

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];
	uiTransId = pDb->LogHdr.uiCurrTransID;

	// Free up all keys associated with this database.  This is done even
	// if we didn't have any update operations because the KREF may
	// have been initialized by key generation operations performed
	// by cursors, etc.

	KrefCntrlFree( pDb);

	// Free any index counts we may have allocated.

	FSFreeIxCounts( pDb);
	
	if (pDb->bHadUpdOper)
	{
		// Dump any BLOB structures that should be aborted.

		FBListAfterAbort( pDb);

		// Dump any start and stop indexing stubs that should be aborted.

		flmIndexingAfterAbort( pDb);

		// Log the abort record to the rfl file, or throw away the logged
		// records altogether, depending on the LOG_KEEP_ABORTED_TRANS_IN_RFL
		// flag.  If the RFL volume is bad, we will not attempt to keep this
		// transaction in the RFL.

		if (!pFile->pRfl->seeIfRflVolumeOk())
		{
			bKeepAbortedTrans = FALSE;
		}
		else
		{
			bKeepAbortedTrans =
				(pucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL])
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

	if( bOkToLogAbort)
	{
		flmAssert( pDb->LogHdr.uiCurrTransID == pFile->pRfl->getCurrTransID());
		if (RC_BAD( rc = pFile->pRfl->logEndTransaction(
									RFL_TRNS_ABORT_PACKET, !bKeepAbortedTrans)))
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

		flmAssert( pFile->pRfl->getCurrTransID() == 0);
	}
#endif

	// If there were no operations in the transaction, restore
	// everything as if the transaction never happened.

	if (!pDb->bHadUpdOper)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		pFile->uiUpdateTransID = 0;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		// Pretend we dumped cache - shouldn't be any to worry about at
		// this point.

		bDumpedCache = TRUE;
		goto Exit1;
	}

	// Dump ALL modified cache blocks associated with the DB.
	// NOTE: This needs to be done BEFORE the call to flmGetLogHdrInfo
	// below, because that call will change pDb->LogHdr.uiCurrTransID,
	// and that value is used by flmRcaAbortTrans.

	ScaFreeModifiedBlocks( pDb);
	flmRcaAbortTrans( pDb);
	bDumpedCache = TRUE;

	// Reset the LogHdr from the last committed log header in pFile.

	flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);
	if (RC_BAD( rc = flmPhysRollback( pDb,
				 (FLMUINT)FB2UD( &pucUncommittedLogHdr [LOG_ROLLBACK_EOF]),
				 pFile->uiFirstLogBlkAddress, FALSE, 0)))
	{
		goto Exit1;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Put the new transaction ID into the log header even though
	// we are not committing.  We want to keep the transaction IDs
	// incrementing even though we aborted.

	UD2FBA( (FLMUINT32)uiTransId,
			&pucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);

	// Preserve where we are at in the roll-forward log.  Even though
	// the transaction aborted, we may have kept it in the RFL instead of
	// throw it away.

	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_FILE_NUM],
				 &pucUncommittedLogHdr [LOG_RFL_FILE_NUM], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET],
				 &pucUncommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM],
				 &pucUncommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM],
				 F_SERIAL_NUM_SIZE);
	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM],
				 &pucUncommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM],
				 F_SERIAL_NUM_SIZE);

	// The following items tell us where we are at in the roll-back log.
	// During a transaction we may log blocks for the checkpoint or for
	// read transactions.  So, even though we are aborting this transaction,
	// there may be other things in the roll-back log that we don't want
	// to lose.  These items should not be reset until we do a checkpoint,
	// which is when we know it is safe to throw away the entire roll-back log.

	f_memcpy( &pucLastCommittedLogHdr [LOG_ROLLBACK_EOF],
				 &pucUncommittedLogHdr [LOG_ROLLBACK_EOF], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR],
				 &pucUncommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR], 4);

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	pFile->pRfl->commitLogHdrs( pucLastCommittedLogHdr,
							pFile->ucCheckpointLogHdr);

Exit1:

	// Dump cache, if not done above.

	if (!bDumpedCache)
	{
		ScaFreeModifiedBlocks( pDb);
		flmRcaAbortTrans( pDb);
		bDumpedCache = TRUE;
	}

	// Throw away IXD_FIXUPs

	if (pDb->pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;

		pIxdFixup = pDb->pIxdFixups;
		while (pIxdFixup)
		{
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		pDb->pIxdFixups = NULL;
	}

	if (uiTransType != FLM_READ_TRANS &&
		 gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmTransEventCallback( F_EVENT_ABORT_TRANS, (HFDB)pDb, rc,
						uiTransId);
	}

Unlink_From_Trans:

	bInvisibleTrans = (pDb->uiFlags & FDB_INVISIBLE_TRANS) ? TRUE : FALSE;
	if (pDb->uiFlags & FDB_HAS_WRITE_LOCK)
	{
		RCODE	tmpRc;

		if (RC_BAD( tmpRc = pFile->pRfl->completeTransWrites( pDb, FALSE, FALSE)))
		{
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}
		}
	}

	// Unlink the database from the transaction
	// structure as well as from the FLDICT structure.

	flmUnlinkDbFromTrans( pDb, FALSE);

	if (pDbStats)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &pDb->TransStartTime, &ui64ElapMilli);
		pDbStats->bHaveStats = TRUE;
		if (uiTransType == FLM_READ_TRANS)
		{
			pDbStats->ReadTransStats.AbortedTrans.ui64Count++;
			pDbStats->ReadTransStats.AbortedTrans.ui64ElapMilli +=
					ui64ElapMilli;
			if (bInvisibleTrans)
			{
				pDbStats->ReadTransStats.InvisibleTrans.ui64Count++;
				pDbStats->ReadTransStats.InvisibleTrans.ui64ElapMilli +=
					ui64ElapMilli;
			}
		}
		else
		{
			pDbStats->UpdateTransStats.AbortedTrans.ui64Count++;
			pDbStats->UpdateTransStats.AbortedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	if (pDb->pStats)
	{
		(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine commits an active transaction for a particular
		database.  If the database is open via a server, a message is
		sent to the server to commit the transaction.  Otherwise, the
		transaction is committed locally.
****************************************************************************/
RCODE flmCommitDbTrans(
	FDB *				pDb,
	FLMUINT			uiNewLogicalEOF,
	FLMBOOL			bForceCheckpoint,
	FLMBOOL *		pbEmpty)
{
	RCODE	  			rc = FERR_OK;
	FLMBYTE *		pucUncommittedLogHdr;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiCPFileNum = 0;
	FLMUINT			uiCPOffset = 0;
	FLMUINT			uiTransId = 0;
	FLMBOOL			bTransEndLogged;
	FLMBOOL			bForceCloseOnError = FALSE;
	FLMBOOL			bOkToLogAbort = TRUE;
	DB_STATS *		pDbStats = pDb->pDbStats;
	FLMUINT			uiTransType;
	FLMBOOL			bInvisibleTrans = FALSE;
	FLMBOOL			bIndexAfterCommit = FALSE;

	pDb->uiFlags |= FDB_COMMITTING_TRANS;

	// See if we even have a transaction going.

	if ((uiTransType = pDb->uiTransType) == FLM_NO_TRANS)
	{
		goto Exit;	// Will return FERR_OK.
	}

	// See if we have a transaction going which should be aborted.

	if (flmCheckBadTrans( pDb))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	// If we are in a read transaction we can skip most of the stuff
	// below because no updates would have occurred.  This will help
	// improve performance.

	if (uiTransType == FLM_READ_TRANS)
	{

		if( pDb->KrefCntrl.bKrefSetup)
		{
			// KrefCntrlFree could be called w/o checking bKrefSetup because
			// it checks the flag, but it is more optimal to check the
			// flag before making the call because most of the time it will
			// be false.

			KrefCntrlFree( pDb);
		}
		goto Exit1;
	}

	// At this point, we know we have an update transaction.

	pFile->pRfl->clearLogHdrs();

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			0, 0, FERR_OK, "TCmit");
#endif
	uiTransId = pDb->LogHdr.uiCurrTransID;

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	if (!pDb->bHadUpdOper)
	{
		bOkToLogAbort = FALSE;
		rc = pFile->pRfl->logEndTransaction( RFL_TRNS_COMMIT_PACKET, TRUE);

		// Even though we didn't have any update operations, there may have
		// been operations during the transaction (i.e., query operations)
		// that initialized the KREF in order to generate keys.

		KrefCntrlFree( pDb);

		// Restore everything as if the transaction never happened.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		pFile->uiUpdateTransID = 0;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		if (pbEmpty)
		{
			*pbEmpty = TRUE;
		}
		goto Exit1;
	}

	// Log commit record to roll-forward log

	bOkToLogAbort = FALSE;
	if (RC_BAD( rc = pFile->pRfl->logEndTransaction(
								RFL_TRNS_COMMIT_PACKET, FALSE, &bTransEndLogged)))
	{
		goto Exit1;
	}
	bForceCloseOnError = TRUE;

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = KYKeysCommit( pDb, TRUE)))
	{
		flmLogError( rc, "calling KYKeysCommit from flmCommitDbTrans");
		goto Exit1;
	}

	if (RC_BAD( rc = FSCommitIxCounts( pDb)))
	{
		flmLogError( rc, "calling FSCommitIxCounts from flmCommitDbTrans");
		goto Exit1;
	}

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

	if ( pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		// If we are in the middle of upgrading, and are forcing
		// a checkpoint, use the file number and offset that were
		// set in the FDB.

		if ((pDb->uiFlags & FDB_UPGRADING) && bForceCheckpoint)
		{
			uiCPFileNum = pDb->uiUpgradeCPFileNum;
			uiCPOffset = pDb->uiUpgradeCPOffset;
		}
		else
		{
			SCACHE *		pTmpSCache;
			F_Rfl *		pRfl = pFile->pRfl;

			f_mutexLock( gv_FlmSysData.hShareMutex);
			pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUCache;

			// Test for buildup of dirty cache blocks.

			if( (pTmpSCache && !pTmpSCache->uiUseCount &&
				  (pTmpSCache->ui16Flags & 
						(CA_DIRTY | CA_LOG_FOR_CP | CA_WRITE_TO_LOG))) ||
				pRfl->atEndOfLog() || bForceCheckpoint)
			{
				bForceCheckpoint = TRUE;
				uiCPFileNum = pRfl->getCurrFileNum();
				uiCPOffset = pRfl->getCurrReadOffset();
			}
			
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}
	}

	// Move information collected in the pDb->LogHdr into the
	// uncommitted log header.  Other things that need to be
	// set have already been set in the uncommitted log header
	// at various places in the code.

	// Mutex does not have to be locked while we do this because
	// the update transaction is the only one that ever accesses
	// the uncommitted log header buffer.

	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];

	// Set the new logical EOF if passed in.

	if( uiNewLogicalEOF)
	{
		pDb->LogHdr.uiLogicalEOF = uiNewLogicalEOF;
	}
	
	UD2FBA( (FLMUINT32)pDb->LogHdr.uiLogicalEOF,
		&pucUncommittedLogHdr [LOG_LOGICAL_EOF]);

	// Increment the commit counter.

	flmIncrUint( &pucUncommittedLogHdr [LOG_COMMIT_COUNT], 1);

	// Set the last committed transaction ID

	if( (bTransEndLogged || (pDb->uiFlags & FDB_REPLAYING_COMMIT)) &&
		pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_31)
	{
		UD2FBA( (FLMUINT32)uiTransId, 
			&pucUncommittedLogHdr [LOG_LAST_RFL_COMMIT_ID]);
	}

	// Write the header

	pFile->pRfl->commitLogHdrs( pucUncommittedLogHdr,
							pFile->ucCheckpointLogHdr);

	// Commit any record cache.

	flmRcaCommitTrans( pDb);

	// Push the IXD_FIXUP values back into the IXD

	if (pDb->pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;
		IXD *			pIxd;

		pIxdFixup = pDb->pIxdFixups;
		while (pIxdFixup)
		{
			if( RC_BAD( fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				pIxdFixup->uiIndexNum, NULL, &pIxd, TRUE)))
			{
				flmAssert( 0);
				pIxd = NULL;
			}

			if( pIxd)
			{
				pIxd->uiLastContainerIndexed = pIxdFixup->uiLastContainerIndexed;
				pIxd->uiLastDrnIndexed = pIxdFixup->uiLastDrnIndexed;
			}
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		pDb->pIxdFixups = NULL;
	}

	// Set the update transaction ID back to zero only 
	// AFTER we know the transaction has safely committed.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_memcpy( pFile->ucLastCommittedLogHdr, pucUncommittedLogHdr,
					LOG_HEADER_SIZE);
	pFile->uiUpdateTransID = 0;
	ScaReleaseLogBlocks( pFile);
	if (pDb->uiFlags & FDB_UPDATED_DICTIONARY)
	{
		// Link the new local dictionary to its file.
		// Since the new local dictionary will be linked at the head
		// of the list of FDICT structures, see if the FDICT currently
		// at the head of the list is unused and can be unlinked.

		if ((pFile->pDictList) && (!pFile->pDictList->uiUseCount))
		{
			flmUnlinkDict( pFile->pDictList);
		}
		flmLinkDictToFile( pFile, pDb->pDict);
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit1:

	// If the local dictionary was updated during this transaction,
	// link the new local dictionary structures to their file - or free
	// them if there was an error.

	if (pDb->uiFlags & FDB_UPDATED_DICTIONARY)
	{
		if( RC_BAD( rc) && pDb->pDict)
		{
			// Unlink the FDB from the FDICT. - Shouldn't have
			// to lock semaphore, because the DICT is NOT linked
			// to the FFILE.

			flmAssert( pDb->pDict->pFile == NULL);
			flmUnlinkFdbFromDict( pDb);
		}
	}

	if (RC_BAD( rc))
	{

		// Since we failed to commit, do an abort.  We are purposely not
		// checking the return code from flmAbortDbTrans because we already
		// have an error return code.  If we attempted to log the transaction
		// to the RFL and failed, we don't want to try to log an abort packet.
		// The RFL code has already reset the log back to the starting point 
		// of the transaction, thereby discarding all operations.

		pDb->uiFlags &= ~FDB_COMMITTING_TRANS;
		(void)flmAbortDbTrans( pDb, bOkToLogAbort);
		uiTransType = FLM_NO_TRANS;

		// Do we need to force all handles to close?

		if( bForceCloseOnError)
		{

			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all FDBs linked to the FFILE
			// and set the FFILE's "must close" flag.  This will cause any
			// subsequent operations on the database to fail until all
			// handles have been closed.

			flmSetMustCloseFlags( pFile, rc, FALSE);
		}
	}
	else
	{
		bInvisibleTrans = (pDb->uiFlags & FDB_INVISIBLE_TRANS) ? TRUE : FALSE;
		if (uiTransType == FLM_UPDATE_TRANS)
		{
			if (gv_FlmSysData.UpdateEvents.pEventCBList)
			{
				flmTransEventCallback( F_EVENT_COMMIT_TRANS, (HFDB)pDb, rc,
							uiTransId);
			}

			// Do the BLOB and indexing work before we unlock the db.

			FBListAfterCommit( pDb);
			
			if (pDb->pIxStopList || pDb->pIxStartList)
			{
				
				// Must not call flmIndexingAfterCommit until after
				// completeTransWrites.  Otherwise, there is a potential
				// deadlock condition where flmIndexingAfterCommit is
				// waiting on an indexing thread to quit, but that
				// thread is waiting to be signaled by this thread that
				// writes are completed.  However, flmIndexingAfterCommit
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
	// NOTE: We check uiTransType because it may have been reset
	// to FLM_NO_TRANS up above if flmAbortDbTrans was called.

	if (uiTransType == FLM_UPDATE_TRANS)
	{
		if (RC_BAD( rc))
		{

			// SHOULD NEVER HAPPEN - because it would have been taken
			// care of above - flmAbortDbTrans would have been called and
			// uiTransType would no longer be FLM_UPDATE_TRANS.

			flmAssert( 0);
			(void)pFile->pRfl->completeTransWrites( pDb, FALSE, TRUE);
		}
		else if( !bForceCheckpoint)
		{
			if( bIndexAfterCommit)
			{
				rc = pFile->pRfl->completeTransWrites( pDb, TRUE, FALSE);
				flmIndexingAfterCommit( pDb);
				flmUnlinkDbFromTrans( pDb, TRUE);
			}
			else
			{
				rc = pFile->pRfl->completeTransWrites( pDb, TRUE, TRUE);
			}
		}
		else
		{

			// Do checkpoint, if forcing.  Before doing the checkpoint
			// we have to make sure the roll-forward log writes
			// complete.  We don't want to unlock the DB while the
			// writes are happening in this case - thus, the FALSE
			// parameter to completeTransWrites.

			if (RC_OK( rc = pFile->pRfl->completeTransWrites( pDb, TRUE, FALSE)))
			{
				bForceCloseOnError = FALSE;
				rc = ScaDoCheckpoint( pDbStats, pDb->pSFileHdl, pFile,
						(pDb->uiFlags & FDB_DO_TRUNCATE) ? TRUE : FALSE,
						TRUE, CP_TIME_INTERVAL_REASON,
						uiCPFileNum, uiCPOffset);
			}
			
			if (bIndexAfterCommit)
			{
				flmIndexingAfterCommit( pDb);
			}
			
			flmUnlinkDbFromTrans( pDb, TRUE);
		}

		if (RC_BAD( rc) && bForceCloseOnError)
		{

			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all FDBs linked to the FFILE
			// and set the FFILE's "must close" flag.  This will cause any
			// subsequent operations on the database to fail until all
			// handles have been closed.

			flmSetMustCloseFlags( pFile, rc, FALSE);
		}
	}
	else
	{

		// Unlink the database from the transaction
		// structure as well as from the FDICT structure.

		flmUnlinkDbFromTrans( pDb, FALSE);
	}

	if (pDbStats && uiTransType != FLM_NO_TRANS)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &pDb->TransStartTime, &ui64ElapMilli);
		pDbStats->bHaveStats = TRUE;
		if (uiTransType == FLM_READ_TRANS)
		{
			pDbStats->ReadTransStats.CommittedTrans.ui64Count++;
			pDbStats->ReadTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
			if (bInvisibleTrans)
			{
				pDbStats->ReadTransStats.InvisibleTrans.ui64Count++;
				pDbStats->ReadTransStats.InvisibleTrans.ui64ElapMilli +=
					ui64ElapMilli;
			}
		}
		else
		{
			pDbStats->UpdateTransStats.CommittedTrans.ui64Count++;
			pDbStats->UpdateTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	// Update stats

	if (pDb->pStats)
	{
		(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
	}

Exit:

	pDb->uiFlags &= ~FDB_COMMITTING_TRANS;
	return( rc);
}
