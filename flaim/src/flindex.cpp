//-------------------------------------------------------------------------
// Desc:	Routines for managing indexes.
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE FLMAPI flmBackgroundIndexBuildThrd(
	IF_Thread *			pThread);

FSTATIC void stopBackgroundIndexThread(
	FDB *					pDb,
	FLMUINT				uiIndexNum,
	FLMBOOL				bWait,
	FLMBOOL *			pbStopped);

FSTATIC RCODE flmIndexStatusCS(
	FDB *					pDb,
	FLMUINT				uiIndexNum,
	FINDEX_STATUS *	pIndexStatus);

/****************************************************************************
Desc : Return the status of the index.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmIndexStatus(
	HFDB					hDb,
	FLMUINT				uiIndexNum,
	FINDEX_STATUS *	pIndexStatus)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bStartedAutoTrans = FALSE;
	FLMUINT 				uiLastDrnIndexed;
	FDB *					pDb = (FDB *)hDb;
	F_BKGND_IX *		pBackgroundIx;
	FLMBOOL				bSuspended;
	FLMBOOL				bMutexLocked = FALSE;

	flmAssert( pIndexStatus != NULL);

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmIndexStatusCS( pDb, uiIndexNum, pIndexStatus);
		goto Exit;
	}

	if( RC_BAD( rc = fdbInit( (FDB *)hDb, FLM_READ_TRANS,
							FDB_TRANS_GOING_OK, 0, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	pBackgroundIx = flmBackgroundIndexGet( pDb->pFile, uiIndexNum, TRUE);
	if( pBackgroundIx)
	{
		f_memcpy( pIndexStatus, &pBackgroundIx->indexStatus, 
			sizeof( FINDEX_STATUS));
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		flmAssert( pIndexStatus->uiIndexNum == uiIndexNum);
	}
	else
	{
		IXD *		pIxd;
		FLMBOOL	bTrackerIxSuspended;

		if( RC_BAD( rc = fdictGetIndex( 
			pDb->pDict, pDb->pFile->bInLimitedMode,
			uiIndexNum,NULL, &pIxd, TRUE)))
		{
			goto Exit;
		}

		bSuspended = (pIxd->uiFlags & IXD_SUSPENDED)
												? TRUE
												: FALSE;

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Get the index state from the tracker

		if( RC_BAD( rc = flmGetIxTrackerInfo( pDb, uiIndexNum, 
			NULL, &uiLastDrnIndexed, NULL, &bTrackerIxSuspended)))
		{
			if( rc == FERR_NOT_FOUND)
			{
				rc = RC_SET( FERR_BAD_IX);
			}
			goto Exit;
		}

		// Sanity check

#ifdef FLM_DEBUG
		if( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_51 &&
			bSuspended != bTrackerIxSuspended)
		{
			flmAssert( 0);
		}
#endif

		// Populate the index status structure.

		f_memset( pIndexStatus, 0, sizeof( FINDEX_STATUS));
		pIndexStatus->uiIndexNum = uiIndexNum;
		pIndexStatus->uiLastRecordIdIndexed = uiLastDrnIndexed;
		pIndexStatus->bSuspended = bSuspended;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}
	flmExit( FLM_INDEX_STATUS, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : Return the number of the next index.  Pass in zero to get the
		 first index.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmIndexGetNext(
	HFDB			hDb,
	FLMUINT *	puiIndexNum)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bStartedAutoTrans = FALSE;
	IXD *			pIxd;

	flmAssert( puiIndexNum != NULL);

	if( IsInCSMode( hDb))
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
			FCS_OPCLASS_INDEX, FCS_OP_INDEX_GET_NEXT)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_INDEX_ID, *puiIndexNum)))
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

		*puiIndexNum = Wire.getIndexId();
		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if( RC_BAD( rc = fdbInit( (FDB *)hDb, FLM_READ_TRANS,
							FDB_TRANS_GOING_OK, 0, &bStartedAutoTrans)))
	{
		goto Exit;
	}
	(void) fdictGetNextIXD( pDb->pDict, *puiIndexNum, &pIxd);
	if( pIxd && pIxd->uiIndexNum < FLM_RESERVED_TAG_NUMS)
	{
		*puiIndexNum = pIxd->uiIndexNum;
	}
	else
	{
		rc = RC_SET( FERR_EOF_HIT);
	}

Exit:

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}
	flmExit( FLM_INDEX_GET_NEXT, pDb, rc);

	return( rc);
}

/****************************************************************************
Desc : Suspend the selected index from doing any key updates on records
		 that are equal or higher than the next record ID value
		 in the container that the index references.  If the index is offline
		 then the background process will be suspended.  If the index is
		 online then it will be suspended.  If the index is already 
		 suspended FERR_OK will be returned.  A suspended index is not
		 persistant if the database goes down.  
Notes: An update transaction will be started if necessary.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmIndexSuspend(
	HFDB			hDb,
	FLMUINT		uiIndexNum)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	IXD *			pIxd;
	FLMUINT		uiHighestRecId;
	FLMUINT		uiContainerNum;
	FLMBOOL		bSuspended;
	FLMBOOL		bStartedTrans = FALSE;
	LFILE *		pLFile;

	if( IsInCSMode( hDb))
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
			FCS_OPCLASS_INDEX, FCS_OP_INDEX_SUSPEND)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_INDEX_ID, uiIndexNum)))
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

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
		FDB_TRANS_GOING_OK, FLM_AUTO_TRANS | FLM_NO_TIMEOUT, &bStartedTrans)))
	{
		goto Exit;
	}

	// See if the index is valid

	if( RC_BAD( rc = fdictGetIndex(
		pDb->pDict,
		pDb->pFile->bInLimitedMode,
		uiIndexNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	if( pIxd->uiFlags & IXD_UNIQUE)
	{
		// Can't suspend unique indexes
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( pIxd->uiFlags & IXD_SUSPENDED)
	{
		// Index is already suspended.
		goto Exit;
	}

	// Get the current index info from the tracker

	if( RC_BAD( rc = flmGetIxTrackerInfo( pDb,
		uiIndexNum, &uiContainerNum, &uiHighestRecId, NULL, &bSuspended)))
	{
		goto Exit;
	}
	flmAssert( !bSuspended);

	// Get information about the container(s) being indexed

	if( !(pIxd->uiFlags & IXD_OFFLINE))
	{
		if ((uiContainerNum = pIxd->uiContainerNum) == 0)
		{
			// The index was on-line and up-to-date.  For an index that
			// crosses all containers, we will suspend on the highest DRN of
			// the FLM_DATA_CONTAINER.

			uiContainerNum = FLM_DATA_CONTAINER;
		}

		if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
			uiContainerNum, &pLFile)))
		{
			goto Exit;
		}

		uiHighestRecId = 0;
		if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, FALSE, &uiHighestRecId)))
		{
			goto Exit;
		}

		// Decrement uiHighestRecId by 1 to correctly reflect the
		// last record that was indexed.

		flmAssert( uiHighestRecId != 0);
		uiHighestRecId--;
	}

	// There may be a background thread still assigned to the
	// index even though the index may be "on-line."  This is because
	// the background thread may have just commited a transaction that
	// transitioned the index from off-line to on-line, but the thread
	// has not yet exited (even though it will not do any more work
	// to update the index).  We want to wait for the thread to terminate
	// before our transaction is allowed to commit.  This is so that if
	// we immediately call resume, it won't find the yet-to-terminate
	// thread still running in the background.

	if( !(pDb->uiFlags & FDB_REPLAYING_RFL))
	{
		if( RC_BAD( rc = flmAddToStopList( pDb, uiIndexNum)))
		{
			goto Exit;
		}
	}

	flmAssert( uiContainerNum != 0xFFFFFFFF);

	if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, 
		uiIndexNum, uiContainerNum, uiHighestRecId, 
		TRANS_ID_OFFLINE, TRUE)))
	{
		goto Exit;
	}

	// Create a new dictionary

	if( !(pDb->uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if( RC_BAD( rc = fdictCloneDict( pDb)))
		{
			goto Exit;
		}

		// Get a pointer to the new IXD

		if( RC_BAD( rc = fdictGetIndex( pDb->pDict,
			pDb->pFile->bInLimitedMode,
			uiIndexNum, NULL, &pIxd, TRUE)))
		{
			goto Exit;
		}
	}

	// Update the IXDs flags so that the current update
	// transaction will see the correct state of the index.
	// Old read transactions will continue to use a prior
	// version of the dictionary.

	pIxd->uiFlags |= (IXD_SUSPENDED | IXD_OFFLINE);

	// Log the suspend packet to the RFL

	if( RC_BAD( rc = pDb->pFile->pRfl->logIndexSuspendOrResume( 
		uiIndexNum, RFL_INDEX_SUSPEND_PACKET)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

	flmExit( FLM_INDEX_SUSPEND, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : If the index was suspended, restart the background process that
		 will get the index up to date so that it will eventually be online.
		 Returns FERR_OK with no change if the index is already online.
Notes: An update transaction will be started if necessary.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmIndexResume(
	HFDB			hDb,
	FLMUINT		uiIndexNum)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = (FDB *)hDb;
	IXD *				pIxd;
	FLMUINT 			uiLastContainerIndexed;
	FLMUINT 			uiLastDrnIndexed;
	FLMUINT 			uiOnlineTransId;
	FLMBOOL			bStartedTrans = FALSE;

	if( IsInCSMode( hDb))
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
			FCS_OPCLASS_INDEX, FCS_OP_INDEX_RESUME)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_INDEX_ID, uiIndexNum)))
		{
			goto Transmission_Error;
		}

		// Send the "auto-online" flag (only needed for 
		// backwards compatibility)
		
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_BOOLEAN, 1)))
		{
			goto Transmission_Error;
		}

		// Send a priority of high (only needed for
		// backwards compatibility)

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1, 1)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response.
	
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

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
		FDB_TRANS_GOING_OK, FLM_AUTO_TRANS | FLM_NO_TIMEOUT, &bStartedTrans)))
	{
		goto Exit;
	}

	// See if the index is valid

	if( RC_BAD( rc = fdictGetIndex(
		pDb->pDict,
		pDb->pFile->bInLimitedMode,
		uiIndexNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	if( pIxd->uiFlags & IXD_UNIQUE)
	{
		// Can't suspend or resume unique indexes

		flmAssert( !(pIxd->uiFlags & (IXD_SUSPENDED | IXD_OFFLINE)));
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( !(pIxd->uiFlags & (IXD_SUSPENDED | IXD_OFFLINE)))
	{
		// Index is already on-line

		goto Exit;
	}

	// If we're in limited mode and this is an encrypted index,
	// it can't be resumed
	if (pDb->pFile->bInLimitedMode && pIxd->uiEncId)
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( !(pIxd->uiFlags & IXD_SUSPENDED))
	{
		// Index is not suspended.  It is offline (see test
		// above), but a thread should already be building the 
		// index, or it better be in the start list.

#ifdef FLM_DEBUG
		if (flmBackgroundIndexGet( pDb->pFile, 
				uiIndexNum, FALSE) == NULL)
		{
			F_BKGND_IX *	pBackgroundIx;

			for( pBackgroundIx = pDb->pIxStartList;
					pBackgroundIx;
					pBackgroundIx = pBackgroundIx->pNext)
			{
				if( pBackgroundIx->indexStatus.uiIndexNum == uiIndexNum)
				{
					break;
				}
			}
			flmAssert( pBackgroundIx);
		}
#endif

		goto Exit;
	}

	// Better not have a background thread running, or it better be
	// in the stop list - because its state shows suspended.

#ifdef FLM_DEBUG
	if (flmBackgroundIndexGet( pDb->pFile, uiIndexNum, FALSE) != NULL)
	{
		F_BKGND_IX *	pBackgroundIx;

		for( pBackgroundIx = pDb->pIxStopList;
				pBackgroundIx;
				pBackgroundIx = pBackgroundIx->pNext)
		{
			if( pBackgroundIx->indexStatus.uiIndexNum == uiIndexNum)
			{
				break;
			}
		}
		flmAssert( pBackgroundIx);
	}
#endif

	// Get the tracker info

	if( RC_BAD( rc = flmGetIxTrackerInfo( pDb, uiIndexNum, 
		&uiLastContainerIndexed, &uiLastDrnIndexed, &uiOnlineTransId,
		NULL)))
	{
		goto Exit;
	}

	// Update the tracker info so that the index state will
	// be changed to "unsuspended."

	if( RC_BAD( rc = flmSetIxTrackerInfo( pDb, uiIndexNum, 
		uiLastContainerIndexed, uiLastDrnIndexed, 
		uiOnlineTransId, FALSE)))
	{
		goto Exit;
	}

	// Add an entry to the start list so that an indexing thread
	// will be started when this transaction commits.

	if( !(pDb->uiFlags & FDB_REPLAYING_RFL))
	{
		if( RC_BAD( rc = flmAddToStartList( pDb, uiIndexNum)))
		{
			goto Exit;
		}
	}

	// Create a new dictionary.

	if( !(pDb->uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if( RC_BAD( rc = fdictCloneDict( pDb)))
		{
			goto Exit;
		}

		// Get a pointer to the new IXD

		if( RC_BAD( rc = fdictGetIndex(
			pDb->pDict,
			pDb->pFile->bInLimitedMode,
			uiIndexNum, NULL, &pIxd, TRUE)))
		{
			goto Exit;
		}
	}

	// Update the IXDs flags so that the current update
	// transaction will see the correct state of the index.
	// Old read transactions will continue to use a prior
	// version of the dictionary.

	pIxd->uiFlags &= ~IXD_SUSPENDED;
	pIxd->uiFlags |= IXD_OFFLINE;

	// Log the resume packet to the RFL

	if( RC_BAD( rc = pDb->pFile->pRfl->logIndexSuspendOrResume( 
		uiIndexNum, RFL_INDEX_RESUME_PACKET)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

	flmExit( FLM_INDEX_RESUME, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:		Add the index to the stop list of background threads.
****************************************************************************/
RCODE	flmAddToStopList(
	FDB *					pDb,
	FLMUINT				uiIndexNum)
{
	RCODE					rc = FERR_OK;
	F_BKGND_IX *		pBackgroundIx;
	F_BKGND_IX *		pNextBackgroundIx;

	// We'd better not be replaying the RFL

	flmAssert( !(pDb->uiFlags & FDB_REPLAYING_RFL));

	// First look in the start list and remove any index matches.
	// This is need if you add an index and drop 
	// it within the same transaction.

	for( pBackgroundIx = pDb->pIxStartList;
			pBackgroundIx; pBackgroundIx = pNextBackgroundIx)
	{
		pNextBackgroundIx = pBackgroundIx->pNext;

		if( pBackgroundIx->indexStatus.uiIndexNum == uiIndexNum)
		{
			if( pNextBackgroundIx)
			{
				pNextBackgroundIx->pPrev = pBackgroundIx->pPrev;
			}

			if( pBackgroundIx->pPrev)
			{
				pBackgroundIx->pPrev->pNext = pNextBackgroundIx;
			}
			else
			{
				pDb->pIxStartList = pNextBackgroundIx;
			}

			f_free( &pBackgroundIx);
		}
	}

	// See if we already have an entry in the stop list for the index.  There
	// is no reason to have the index in the list more than once.

	for( pBackgroundIx = pDb->pIxStopList;
			pBackgroundIx; pBackgroundIx = pNextBackgroundIx)
	{
		pNextBackgroundIx = pBackgroundIx->pNext;

		if( pBackgroundIx->indexStatus.uiIndexNum == uiIndexNum)
		{
			goto Exit;  // Should return FERR_OK
		}
	}

	// Allocate and add the thread structure to the pFile thread list.

	if( RC_BAD( rc = f_calloc( 
		(FLMUINT)( sizeof( F_BKGND_IX)), &pBackgroundIx)))
	{
		goto Exit;
	}

	pBackgroundIx->indexStatus.uiIndexNum  = uiIndexNum;
	pBackgroundIx->pPrev = NULL;
	if( (pBackgroundIx->pNext = pDb->pIxStopList) != NULL)
	{
		pDb->pIxStopList->pPrev = pBackgroundIx;
	}
	pDb->pIxStopList = pBackgroundIx;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Add the index to the start list of background threads.
****************************************************************************/
RCODE	flmAddToStartList(
	FDB *					pDb,
	FLMUINT				uiIndexNum)
{
	RCODE					rc = FERR_OK;
	F_BKGND_IX *		pBackgroundIx;
	F_BKGND_IX *		pNextBackgroundIx;

	// We'd better not be replaying the RFL

	flmAssert( !(pDb->uiFlags & FDB_REPLAYING_RFL));

	// Look in the start list to make sure we don't already
	// have an entry for this index.  We don't want to
	// start more than one thread per index.  The background
	// indexing code is not structured to handle multiple build
	// threads on the same index.

	// NOTE: We don't want to remove any entries in the stop
	// list corresponding to this index.  The reason for this
	// is the index may have been deleted, re-added, deleted,
	// modified, etc. several times during the transaction.
	// We want to make sure that an existing background indexing
	// thread is terminated and a new one is started.  The stop
	// list is always processed first at transaction commit time.
	// Then new indexing threads (in the start list) are started.

	for( pBackgroundIx = pDb->pIxStartList;
			pBackgroundIx; pBackgroundIx = pNextBackgroundIx)
	{
		pNextBackgroundIx = pBackgroundIx->pNext;

		if( pBackgroundIx->indexStatus.uiIndexNum == uiIndexNum)
		{
			goto Exit; // Should return FERR_OK
		}
	}

	// Allocate and add the thread structure to the pDb thread list.

	if( RC_BAD( rc = f_calloc( 
		(FLMUINT)( sizeof( F_BKGND_IX)), &pBackgroundIx)))
	{
		goto Exit;
	}

	pBackgroundIx->indexStatus.uiIndexNum = uiIndexNum;
	pBackgroundIx->pPrev = NULL;
	if( (pBackgroundIx->pNext = pDb->pIxStartList) != NULL)
	{
		pDb->pIxStartList->pPrev = pBackgroundIx;
	}
	pDb->pIxStartList = pBackgroundIx;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		After Abort and before we unlock, stop and start all indexing.
****************************************************************************/
void flmIndexingAfterAbort(
	FDB *					pDb)
{
	F_BKGND_IX *		pStartIx;
	F_BKGND_IX *		pStopIx;
	F_BKGND_IX *		pNextIx;

	pStopIx = pDb->pIxStopList;
	pDb->pIxStopList = NULL;
	for( ; pStopIx; pStopIx = pNextIx)
	{
		pNextIx = pStopIx->pNext;
		f_free( &pStopIx);
	}

	pStartIx = pDb->pIxStartList;
	pDb->pIxStartList = NULL;
	for( ; pStartIx; pStartIx = pNextIx)
	{
		pNextIx = pStartIx->pNext;
		f_free( &pStartIx);
	}
}

/****************************************************************************
Desc:		Stops a background indexing thread
Notes:	This routine DOES NOT assume that the global mutex is locked.  It
			will lock and unlock the mutex as needed.
****************************************************************************/
FSTATIC void stopBackgroundIndexThread(
	FDB *				pDb,
	FLMUINT			uiIndexNum,
	FLMBOOL			bWait,
	FLMBOOL *		pbStopped)
{
	F_BKGND_IX *	pBackgroundIx;
	FLMUINT			uiThreadId;
	FLMBOOL			bMutexLocked = FALSE;

	if( pbStopped)
	{
		*pbStopped = FALSE;
	}

	for( ;;)
	{
		// Lock the global mutex

		if( !bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		// Get the background index

		if( (pBackgroundIx = flmBackgroundIndexGet( pDb->pFile,
			uiIndexNum, TRUE, &uiThreadId)) == NULL)
		{
			if( pbStopped)
			{
				*pbStopped = TRUE;
			}
			goto Exit;
		}

		// Set the thread's shutdown flag first.

		gv_FlmSysData.pThreadMgr->setThreadShutdownFlag( uiThreadId);

		// Unlock the global mutex

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// The thread may be waiting to start a transaction. 

		pDb->pFile->pFileLockObj->timeoutLockWaiter( uiThreadId);
		pDb->pFile->pWriteLockObj->timeoutLockWaiter( uiThreadId);
		
		if( !bWait)
		{
			break;
		}

		// Wait for the thread to terminate

		f_sleep( 50);
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:		After commit and before we unlock, stop and start all indexing.
****************************************************************************/
void flmIndexingAfterCommit(
	FDB *					pDb)
{
	F_BKGND_IX *		pStartIx;
	F_BKGND_IX *		pStopIx;
	F_BKGND_IX *		pNextIx;
	FLMBOOL				bThreadsActive;
	FLMBOOL				bStopped;

	// Signal all background indexing threads in the stop list
	// to shutdown.  Poll until all have terminated.

	for( ;;)
	{
		bThreadsActive = FALSE;
		for( pStopIx = pDb->pIxStopList; pStopIx; pStopIx = pStopIx->pNext)
		{
			stopBackgroundIndexThread( pDb, pStopIx->indexStatus.uiIndexNum, 
				FALSE, &bStopped);
			if( !bStopped)
			{
				bThreadsActive = TRUE;
			}
		}

		if( !bThreadsActive)
		{
			break;
		}

		f_sleep( 50);
	}

	// Now that all of the threads have been stopped, discard the stop list

	pStopIx = pDb->pIxStopList;
	pDb->pIxStopList = NULL;
	for( ; pStopIx; pStopIx = pNextIx)
	{
		pNextIx = pStopIx->pNext;
		f_free( &pStopIx);
	}

	// Start threads listed in the index start list.

	pStartIx = pDb->pIxStartList;
	pDb->pIxStartList = NULL;
	for( ; pStartIx; pStartIx = pNextIx)
	{
		pNextIx = pStartIx->pNext;
		(void)flmStartIndexBuild( pDb, pStartIx->indexStatus.uiIndexNum);
		f_free( &pStartIx);
	}
}	

/****************************************************************************
Desc:		Thread that will build an index in the background.
			Caller will create a pDb to use.  This pDb must be
			freed at the conclusion of the routine.
****************************************************************************/
RCODE flmStartIndexBuild(
	FDB *				pDb,
	FLMUINT			uiIndexNum)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiGMT;
	IXD *				pIxd;
	F_BKGND_IX *	pBackgroundIx = NULL;
	char				szThreadName[ F_PATH_MAX_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];

	f_timeGetSeconds( &uiGMT );
	
	if( flmBackgroundIndexGet( pDb->pFile, uiIndexNum, FALSE) != NULL)
	{
		// There is already a background thread running on this index.

		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	if( RC_BAD( rc = fdictGetIndex(
		pDb->pDict,
		pDb->pFile->bInLimitedMode,
		uiIndexNum, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// Allocate the background thread and index status strucutures.

	if( RC_BAD( rc = f_calloc(
		(FLMUINT)sizeof( F_BKGND_IX), &pBackgroundIx)))
	{
		goto Exit;
	}

	pBackgroundIx->pFile = pDb->pFile;
	pBackgroundIx->indexStatus.bSuspended = FALSE;
	pBackgroundIx->indexStatus.uiIndexNum = uiIndexNum;
	pBackgroundIx->indexStatus.uiStartTime = uiGMT;
	pBackgroundIx->indexStatus.uiLastRecordIdIndexed = pIxd->uiLastDrnIndexed;
	pBackgroundIx->indexStatus.uiKeysProcessed = 0;
	pBackgroundIx->indexStatus.uiRecordsProcessed = 0;
	pBackgroundIx->indexStatus.uiTransactions = 0;

	pBackgroundIx->uiIndexingAction = FTHREAD_ACTION_INDEX_OFFLINE;
	pBackgroundIx->pPrev = NULL;
	pBackgroundIx->pNext = NULL;

	// Generate the thread name

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
		pDb->pFile->pszDbPath, szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( (char *)szThreadName, "BldIX %u (%s)", 
		(unsigned)uiIndexNum, szBaseName);

	// Start the thread in the background indexing thread group.
	// The new thread will cleanup pBackgroundIx on termination.

	if( RC_BAD( rc = f_threadCreate( NULL,
						flmBackgroundIndexBuildThrd, szThreadName,
						gv_uiBackIxThrdGroup, uiIndexNum,
						(void *)pBackgroundIx, NULL, 24000)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc) && pBackgroundIx)
	{
		f_free( &pBackgroundIx);
	}

	return( rc);
}	

/****************************************************************************
Desc:		Thread that will build an index in the background.
			Caller will create a pDb to use.  This pDb must be
			freed at the conclusion of the routine.
****************************************************************************/
FSTATIC RCODE FLMAPI flmBackgroundIndexBuildThrd(
	IF_Thread *			pThread)
{
	RCODE					rc = FERR_OK;
	IXD *					pIxd;
	F_BKGND_IX * 		pBackgroundIx = (F_BKGND_IX *)pThread->getParm1();
	FLMBOOL				bStartedTrans;
	FLMBOOL				bDbInitialized;
	FLMUINT				uiContainerNum;
	FLMUINT				uiFirstDrn;
	FLMUINT				uiIndexNum;
	FDB *					pDb = NULL;
	FLMBOOL				bForcedShutdown = FALSE;
	FLMBOOL				bHitEnd;
	FINDEX_STATUS		savedIxStatus;
	FlmRecord *			pReusableRec = NULL;
	char					szMsg[ 128];
	FLMINT				iErrorLine = 0;
	FLMBOOL				bLimitedMode = FALSE;

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);

	if( (pReusableRec = f_new FlmRecord) != NULL)
	{
		if( RC_BAD( pReusableRec->preallocSpace( 512, 1024 * 64)))
		{
			pReusableRec->Release();
			pReusableRec = NULL;
		}
	}

Loop_Again:

	rc = FERR_OK;
	uiIndexNum = pBackgroundIx->indexStatus.uiIndexNum;
	flmAssert( pThread->getThreadAppId() == uiIndexNum);
	bDbInitialized = FALSE;
	bStartedTrans = FALSE;
	pDb = NULL;

	// We could loop forever on flmOpenFile errors, check if we should exit.

	if( pThread->getShutdownFlag())
	{
		bForcedShutdown = TRUE;
		goto Exit;
	}

	if( RC_BAD( rc = flmOpenFile( pBackgroundIx->pFile,
		NULL, NULL, NULL, 0, TRUE, NULL, NULL,
		pBackgroundIx->pFile->pszDbPassword, &pDb)))
	{

		// If the file is being closed, this is not an error.

		if( pBackgroundIx->pFile->uiFlags & DBF_BEING_CLOSED)
		{
			bForcedShutdown = TRUE;
			rc = FERR_OK;
		}
		else
		{
			iErrorLine = (FLMINT)__LINE__;
		}
		goto Exit;
	}

	flmAssert( pDb->pSFileHdl);

	bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, 0, 0, &bStartedTrans)))
	{
		iErrorLine = (FLMINT)__LINE__;
		goto Exit;
	}
	flmAssert( !bStartedTrans);
	pDb->uiFlags |= FDB_BACKGROUND_INDEXING;

	for(;;)
	{
		// Set the thread's status

		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);

		// See if we should shut down. 

		if( pThread->getShutdownFlag())
		{
			bForcedShutdown = TRUE;
			break;
		}

		// Obtain the file lock

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
				bForcedShutdown = TRUE;
				rc = FERR_OK;
			}
			else
			{
				iErrorLine = (FLMINT)__LINE__;
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
			if (RC_BAD( rc = pDb->pFile->pFileLockObj->unlock()))
			{
				iErrorLine = (FLMINT)__LINE__;
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
				bForcedShutdown = TRUE;
				rc = FERR_OK;
			}
			else
			{
				iErrorLine = (FLMINT)__LINE__;
			}
			goto Exit;
		}
		bStartedTrans = TRUE;

		if( RC_BAD( rc = fdictGetIndex( pDb->pDict, pDb->pFile->bInLimitedMode,
			uiIndexNum, NULL, &pIxd, TRUE)))
		{
			// Index may have been deleted by another transaction, or
			// there may have been some other error.

			iErrorLine = (FLMINT)__LINE__;
			goto Exit;
		}

		// If we're running in limited mode, then we can't mess with encrypted
		// indexes.  On the other hand, since the index is marked as offline,
		// but not suspended, this thread has to exist, or else it will cause
		// all kinds of problems elsewhere.  So, in such a case, we will simply
		// sleep in an inifinite loop until the shutdown flag is set.
		// (We consider this acceptable becase running in limited mode is not
		// the norm, and Flaim probably won't be up for very long in this mode.)

		if( pDb->pFile->bInLimitedMode && pIxd->uiEncId)
		{
			bLimitedMode = TRUE;
			goto Exit;
		}

		// Look up the tracker info to determine where we need to being indexing

		if (RC_BAD( rc = flmGetIxTrackerInfo( pDb,
			pBackgroundIx->indexStatus.uiIndexNum, &uiContainerNum,
			&uiFirstDrn, NULL, &pBackgroundIx->indexStatus.bSuspended)))
		{
			iErrorLine = (FLMINT)__LINE__;
			goto Exit;
		}

		// If the index is suspended, this thread should have been
		// shut down.  The suspending thread will keep the database
		// locked until we exit.  So, if we have the database locked,
		// the index better not be suspended.

		flmAssert( !pBackgroundIx->indexStatus.bSuspended &&
			!(pIxd->uiFlags & IXD_SUSPENDED));

		if (pIxd->uiContainerNum)
		{
			uiContainerNum = pIxd->uiContainerNum;
			if( uiFirstDrn == DRN_LAST_MARKER)
			{
				goto Exit;
			}
		}
		else
		{
			if( uiFirstDrn == DRN_LAST_MARKER && uiContainerNum == 0xFFFFFFFF)
			{
				goto Exit;
			}
			else
			{
				// The container number from the tracker record
				// may not be a real container.
				// Determine what the next actual container number is.

				if (uiContainerNum != FLM_DATA_CONTAINER)
				{
					while( uiContainerNum < pDb->pDict->uiIttCnt)
					{
						ITT *	pItt = &pDb->pDict->pIttTbl [uiContainerNum];
						if (ITT_IS_CONTAINER( pItt))
						{
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
					}
				}
			}
		}

		// Setup the DRN range we want to index.

		uiFirstDrn++;
		flmAssert( pIxd->uiLastDrnIndexed == uiFirstDrn - 1);

		// Set the thread's status

		pThread->setThreadStatus( "Indexing %u:%u", 
			(unsigned)uiContainerNum, (unsigned)uiFirstDrn);

		// Read and index up to the highest drn (or record higher than uiEndDrn)
		// or until time runs out.  The 500 is millisecs to take for the transaction.

		f_memcpy( &savedIxStatus, 
			&pBackgroundIx->indexStatus, sizeof( FINDEX_STATUS));

		if( RC_BAD( rc = flmIndexSetOfRecords( pDb,
			uiIndexNum, uiContainerNum, uiFirstDrn, DRN_LAST_MARKER,
			NULL, NULL, NULL, NULL,
			&pBackgroundIx->indexStatus, &bHitEnd, pThread, pReusableRec)))
		{
			// Lock the mutex while copying the saved index status back to
			// the main index status so that someone requesting the index status
			// won't see the status while the memcpy is in progress.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_memcpy( &pBackgroundIx->indexStatus, 
				&savedIxStatus, sizeof( FINDEX_STATUS));
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			iErrorLine = (FLMINT)__LINE__;
			goto Exit;
		}

		if( pBackgroundIx->indexStatus.uiRecordsProcessed - 
			savedIxStatus.uiRecordsProcessed)
		{
			if( RC_BAD( rc = pDb->pFile->pRfl->logIndexSet( uiIndexNum,
						uiContainerNum, uiFirstDrn, 
						pBackgroundIx->indexStatus.uiLastRecordIdIndexed)))
			{
				iErrorLine = (FLMINT)__LINE__;
				goto Exit;
			}
		}

		// Commit the transaction (even if we didn't do any indexing work).

		if( RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
		{
			iErrorLine = (FLMINT)__LINE__;
			goto Exit;
		}

		bStartedTrans = FALSE;
		pBackgroundIx->indexStatus.uiTransactions++;

		if( bHitEnd)
		{
			// flmIndexSetOfRecords brought the index on-line

			if( gv_FlmSysData.UpdateEvents.pEventCBList)
			{
				flmDoEventCallback( F_EVENT_UPDATES, 
						F_EVENT_INDEXING_COMPLETE, (void *)uiIndexNum, 
						(void *)0);
			}

			// Log a message

			flmLogIndexingProgress( uiIndexNum, 0);
			break;
		}
	}

Exit:

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);

	if( bStartedTrans)
	{
		(void)flmAbortDbTrans( pDb);
		bStartedTrans = FALSE;
	}

	if( pDb && pDb->uiFlags & FDB_HAS_FILE_LOCK)
	{
		(void)pDb->pFile->pFileLockObj->unlock();
		pDb->uiFlags &= ~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
		bDbInitialized = FALSE;
	}

	if( pDb)
	{
		(void)FlmDbClose( (HFDB *) &pDb);
	}

	if( RC_BAD(rc) && !bForcedShutdown)
	{
		if (rc == FERR_MEM || rc == FERR_IO_DISK_FULL ||
			rc == FERR_MUST_WAIT_CHECKPOINT)
		{
			// Log the error

			f_sprintf( (char *)szMsg,
				"Background indexing thread %u (index %u)",
				(unsigned)pThread->getThreadId(), (unsigned)uiIndexNum);
			flmLogError( rc, (char *)szMsg, __FILE__, iErrorLine);

			// Sleep a half second and try again.

			pThread->sleep( 500);
			goto Loop_Again;
		}
		else
		{
			f_sprintf( (char *)szMsg,
				"Background indexing thread %u (index %u) -- unrecoverable error.",
				(unsigned)pThread->getThreadId(), (unsigned)uiIndexNum);
			flmLogError( rc, (char *)szMsg, __FILE__, iErrorLine);
		}
	}

	if( pReusableRec)
	{
		flmAssert( pReusableRec->getRefCount() == 1);
		pReusableRec->Release();
	}

	if( bLimitedMode)
	{
		flmAssert( RC_OK( rc));

		for (;;)
		{
			if( pThread->getShutdownFlag())
			{
				break;
			}

			pThread->sleep( 1000);
		}
	}

	// Set the thread's app ID to 0, so that it will not
	// be found now that the thread is terminating (we don't
	// want flmBackgroundIndexGet to find the thread).

	pThread->setThreadAppId( 0);

	// Free the background index structure

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_free( &pBackgroundIx);
	pThread->setParm1( NULL);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	return( rc);
}

/****************************************************************************
Desc: Looks for a background indexing thread that is running with 
		a matching action and value.
Note:	The shared semaphore must be locked on the outside while 
		calling this routine and accessing anything within the F_BKGND_IX
		structure.
****************************************************************************/
F_BKGND_IX * flmBackgroundIndexGet(
	FFILE *			pFile,
	FLMUINT			uiIndexNum,
	FLMBOOL			bMutexLocked,
	FLMUINT *		puiThreadId)
{
	RCODE				rc = FERR_OK;
	IF_Thread *		pThread;
	FLMUINT			uiThreadId;
	F_BKGND_IX *	pBackgroundIx = NULL;

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}

	uiThreadId = 0;
	for( ;;)
	{
		if( RC_BAD( rc = gv_FlmSysData.pThreadMgr->getNextGroupThread( 
			&pThread, gv_uiBackIxThrdGroup, &uiThreadId)))
		{
			if( rc == FERR_NOT_FOUND)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				flmAssert( 0);
			}
		}

		if( pThread->getThreadAppId())
		{
			F_BKGND_IX *		pTmpIx = NULL;

			pTmpIx = (F_BKGND_IX *)pThread->getParm1();
			if( pTmpIx->indexStatus.uiIndexNum == uiIndexNum &&
				pTmpIx->pFile == pFile)
			{
				flmAssert( pThread->getThreadAppId() == uiIndexNum);
				pBackgroundIx = pTmpIx;
				pThread->Release();
				if( puiThreadId)
				{
					*puiThreadId = uiThreadId;
				}
				break;
			}
		}
		pThread->Release();
	}

	if( !bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	return( pBackgroundIx);
}

/****************************************************************************
Desc : Return the status of the index (via C/S protocol)
****************************************************************************/
FSTATIC RCODE flmIndexStatusCS(
	FDB *					pDb,
	FLMUINT				uiIndexNum,
	FINDEX_STATUS *	pIndexStatus)
{
	RCODE					rc = FERR_OK;
	CS_CONTEXT *		pCSContext = pDb->pCSContext;
	FCL_WIRE				Wire( pCSContext, pDb);
	void *				pvMark = pCSContext->pool.poolMark();

	// Set the temporary pool

	Wire.setPool( &pCSContext->pool);

	// Send a request to do the update

	if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_INDEX, 
		FCS_OP_INDEX_GET_STATUS)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_INDEX_ID, uiIndexNum)))
	{
		goto Transmission_Error;
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

	if (RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fcsExtractIndexStatus( Wire.getHTD(), pIndexStatus)))
	{
		goto Exit;
	}

Exit:

	pCSContext->pool.poolReset( pvMark);
	return( rc);

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}
