//-------------------------------------------------------------------------
// Desc:	Record add, modify, and delete routines.
// Tabs:	3
//
// Copyright (c) 1991-1992, 1994-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmDoUpdateCS(
	FDB *			pDb,
	FLMUINT		uiOp,
	FLMUINT		uiContainer,
	FLMUINT *	puiDrn,
	FlmRecord *	pRecord,
	FLMUINT		uiAutoTrans);

/****************************************************************************
Desc:	Internal routine for doing updates via client/server
****************************************************************************/
FSTATIC RCODE flmDoUpdateCS(
	FDB *			pDb,
	FLMUINT		uiOp,
	FLMUINT		uiContainer,
	FLMUINT *	puiDrn,
	FlmRecord *	pRecord,
	FLMUINT		uiAutoTrans)
{
	RCODE		rc = FERR_OK;
	CS_CONTEXT *pCSContext = pDb->pCSContext;
	FCL_WIRE	Wire( pCSContext, pDb);
	
	// Send a request to do the update

	if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_RECORD, uiOp)))
	{
		goto Exit;
	}

	if (uiContainer)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_CONTAINER_ID, uiContainer)))
		{
			goto Transmission_Error;
		}
	}

	if (*puiDrn)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_DRN, *puiDrn)))
		{
			goto Transmission_Error;
		}
	}

	if (pRecord)
	{
		if (RC_BAD( rc = Wire.sendRecord( WIRE_VALUE_RECORD, pRecord)))
		{
			goto Transmission_Error;
		}
	}

	if (uiAutoTrans)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_AUTOTRANS, uiAutoTrans)))
		{
			goto Transmission_Error;
		}
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

	if (uiOp == FCS_OP_RECORD_ADD)
	{
		*puiDrn = Wire.getDrn();
	}

Exit:
	return( rc);

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/************************************************************************
Desc : Outputs an update event callback.
*************************************************************************/
void flmUpdEventCallback(
	FDB *			pDb,
	FEventType	eEventType,
	HFDB			hDb,
	RCODE			rc,
	FLMUINT		uiDrn,
	FLMUINT		uiContainer,
	FlmRecord *	pNewRecord,
	FlmRecord *	pOldRecord)
{
	FLM_UPDATE_EVENT	UpdEvent;

	UpdEvent.uiThreadId = f_threadId();
	UpdEvent.hDb = hDb;
	UpdEvent.uiTransID = (FLMUINT)( (pDb->uiTransType)
													? pDb->LogHdr.uiCurrTransID
													: 0);
	UpdEvent.rc = rc;
	UpdEvent.uiDrn = uiDrn;
	UpdEvent.uiContainer = uiContainer;
	UpdEvent.pNewRecord = pNewRecord;
	UpdEvent.pOldRecord = pOldRecord;
	flmDoEventCallback( F_EVENT_UPDATES, eEventType, &UpdEvent, NULL);
}

/****************************************************************************
Desc : Adds a record to a container.
Notes: If an index definition record is added to the dictionary container,
		 the index will be built automatically when the transaction commits.
		 When a unique index is added to the database, FLAIM needs to verify
		 that each key in the proposed index is indeed a unique key.  However,
		 this verification does NOT occur until the dictionary transaction
		 commits and the index is actually built.  If FLAIM discovers that the
		 keys in an index are not unique, the transaction commit will fail and
		 return an error.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmRecordAdd(
	HFDB			hDb,
	FLMUINT		uiContainer,
	FLMUINT *	puiDrn,
	FlmRecord *	pRecord,
	FLMUINT		uiAutoTrans)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiDrn = 0;
	FDB *			pDb = (FDB *)hDb;
	LFILE *		pLFile;
	FLMBOOL		bStartedAutoTrans = FALSE;
	FLMBOOL		bLogCompleteIndexSet = FALSE;
	DB_STATS *	pDbStats = NULL;
	F_TMSTAMP	StartTime;

	if( puiDrn)
	{
		uiDrn = *puiDrn;
	}

	if( uiContainer == FLM_TRACKER_CONTAINER)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmDoUpdateCS( pDb, FCS_OP_RECORD_ADD, uiContainer,
									&uiDrn, pRecord, uiAutoTrans);
		goto ExitCS;
	}

	if( RC_BAD( rc = fdbInit( (FDB *)hDb, FLM_UPDATE_TRANS,
										FDB_TRANS_GOING_OK,
										uiAutoTrans, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	if( (pDbStats = pDb->pDbStats) != NULL)
	{
		f_timeGetTimeStamp( &StartTime);
	}

	// Make sure we have a valid record

	if( !pRecord)
	{
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// We cannot add a record that is marked read-only, because it
	// is probably already in the cache under a different record ID.

	if( pRecord->isReadOnly())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		goto Exit;
	}

	rc = flmAddRecord( pDb, pLFile, &uiDrn, pRecord, FALSE,
				(uiAutoTrans & FLM_DO_IN_BACKGROUND) ? TRUE : FALSE,
				(uiAutoTrans & FLM_SUSPENDED) ? TRUE : FALSE,
				(FLMBOOL)((uiAutoTrans & FLM_DONT_INSERT_IN_CACHE) ? FALSE : TRUE),
				&bLogCompleteIndexSet);

Exit:

	rc = FB_OperationEnd( pDb, rc);
	if( RC_OK( rc))
	{
		if( RC_OK( rc = pDb->pFile->pRfl->logUpdate( 
				uiContainer, uiDrn, uiAutoTrans, NULL, pRecord)) && 
			 bLogCompleteIndexSet &&
			 pDb->pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_51)
		{

			// Log the fact that we indexed everything so the redo will also
			// index all data records in the container.

			rc = pDb->pFile->pRfl->logIndexSet( uiDrn, 0, 1, 0xFFFFFFFF);
		}
	}

	if( pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->RecordAdds.ui64ElapMilli);
		pDbStats->RecordAdds.ui64Count++;
		pDbStats->bHaveStats = TRUE;
	}

	if( gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmUpdEventCallback( pDb, F_EVENT_ADD_RECORD, hDb, rc, uiDrn,
								uiContainer, pRecord, NULL);
	}

	// If started an automatic transaction end it.

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

ExitCS:

	if( puiDrn)
	{
		*puiDrn = uiDrn;
	}

	flmExit( FLM_RECORD_ADD, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:		Internal interface routine for database add operation.
			Must be in a valid transaction.
Note:		Internal name is used so that later we can call this within FLAIM.
****************************************************************************/
RCODE	flmAddRecord(
	FDB *	 		pDb,
	LFILE *		pLFile,
	FLMUINT *	puiDrn,						// Record Number to set AND return.
	FlmRecord * pRecord,						// Record to add, must NOT be NULL.
	FLMBOOL		bBatchProcessing,			// Set to TRUE if called by REBUILD.
	FLMBOOL		bDoInBackground,
	FLMBOOL		bCreateSuspended,
	FLMBOOL		bKeepInCache,
	FLMBOOL *	pbLogCompleteIndexSet)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiDrn = 0;
	FLMBOOL		bProcessedKeys = FALSE;
	FLMUINT		uiLfNum = pLFile->uiLfNum;
	FLMUINT		uiAddAppendFlags = REC_UPD_ADD;
	FLMBOOL		bHadUniqueKeys;

	if( puiDrn)
	{
		uiDrn = *puiDrn;
	}

	if( pDb->uiFlags & FDB_COMMITTING_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Assert that the record is not read-only

	flmAssert( !pRecord->isReadOnly());

	// Set up for indexing.

	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	if( !pRecord)
	{
		rc = RC_SET( FERR_NULL_RECORD);
		goto Exit;
	}

	// If this is a dictionary record then one routine takes care of it.

	if( pLFile->uiLfNum == FLM_LOCAL_DICT_CONTAINER)
	{
		if( RC_OK( rc = flmLFileDictUpdate( pDb, &pLFile, &uiDrn,
					pRecord, NULL, bDoInBackground, bCreateSuspended,
					pbLogCompleteIndexSet)))
		{
			if( puiDrn)
			{
				*puiDrn = uiDrn;
			}
		}

		goto Exit;
	}
	else
	{
		if( !uiDrn || uiDrn == DRN_LAST_MARKER)
		{
			if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, FALSE, &uiDrn)))
			{
#ifdef FLM_DBG_LOG
				uiDrn = 0;
#endif
				goto Exit;
			}
			
			uiAddAppendFlags |= REC_UPD_NEW_RECORD;
		}
	}

	// Add the records keys, and then the record.  NOTE: If the bBatchProcessing
	// flag is set to TRUE, we are being called as part of a rebuild.  In this
	// case, we do NOT want BLOBs to be reprocessed.  Also, we do not want
	// the QF job list to be generated (it could get very large - and all of
	// the entries are stored in a single record).  So we set the
	// KREF_INDEXING_ONLY flag.  This flag will prohibit the processing of
	// BLOB data.  Also, it will prohibit constructing a QF job list.  Instead
	// of the QF job list, the entries will be fed directly to QuickFinder and
	// then processed when KYKeysCommit is called.

	bProcessedKeys = TRUE;
	bHadUniqueKeys = FALSE;
	if( RC_BAD( rc = flmProcessRecFlds( pDb, NULL, pLFile->uiLfNum, uiDrn, pRecord,
										(FLMUINT)((bBatchProcessing)
											? (KREF_ADD_KEYS | KREF_INDEXING_ONLY)
											: KREF_ADD_KEYS),
											FALSE, &bHadUniqueKeys)))
	{
		goto Exit;
	}
											
	// NOTE: The LFile table may have changed locations if the dictionary 
	// was update because of a change in a field state
	
	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiLfNum, &pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD(	rc = FSRecUpdate( pDb, pLFile, pRecord,
											uiDrn, uiAddAppendFlags)))
	{
		goto Exit;
	}

	if( puiDrn)
	{
		*puiDrn = uiDrn;
	}

	// Sort and check keys for uniqueness.
	
	if( RC_BAD( rc = KYProcessDupKeys( pDb, bHadUniqueKeys)))
	{
		// Remove the record that was added because of the error.

		RCODE		rc1 = FSRecUpdate( pDb, pLFile, NULL, uiDrn, REC_UPD_DELETE);
		if( RC_BAD(rc1))
		{
			rc = (rc == FERR_NOT_UNIQUE) ? rc1 : rc;
		}
		goto Exit;
	}

	// Insert record into cache

	pRecord->setID( uiDrn);
	pRecord->setContainerID( pLFile->uiLfNum);
	if (bKeepInCache)
	{
		if( RC_BAD( rc = flmRcaInsertRec( pDb, pLFile, uiDrn, pRecord)))
		{
			// Remove the record that was added because of the error.

			FSRecUpdate( pDb, pLFile, NULL, uiDrn, REC_UPD_DELETE);
			goto Exit;
		}
	}

	// Don't make this call until we are sure of success - because we want to
	// be able to back things out of KREF table.

	KYFinishCurrentRecord( pDb);

Exit:

	if( RC_BAD( rc) && bProcessedKeys)
	{
		KYAbortCurrentRecord( pDb);
	}
	
#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pDb->pFile->uiFFileId, pDb->LogHdr.uiCurrTransID, 
		uiLfNum, uiDrn, rc, "RAdd");
#endif

	return( rc);
}

/****************************************************************************
Desc : Modifies a record within a container.
Notes: If an index definition record is modified in the dictionary container,
		 the index B-TREE will be deleted and rebuilt automatically when the
		 transaction commits.  If an index definition record is changed into
		 a field definition record or a container definition record, the index
		 will be automatically deleted when the transaction commits.  When a
		 non-unique index is changed to a unique index, or the fields in a
		 unique index are changed, FLAIM needs to verify that each key in the
		 proposed index is indeed a unique key.  However, this verification
		 does NOT occur until the dictionary transaction commits and the
		 index is actually rebuilt.  If FLAIM discovers that the keys in the
		 proposed index are not unique, the transaction commit will fail and
		 return FERR_NOT_UNIQUE.

		 If a container definition record is changed into a field definition
		 record, the container will be deleted automatically when the
		 transaction commits.  If a container definition record is changed
		 into an index definition record, the container will be automatically
		 deleted and the index will be automatically built when the transaction
		 commits.

		 Only the name and state of field definition records can be modified.
		 Changing a field type or changing a field definition record into an
		 index definition record is not allowed.  For information on changing
		 the state of a field, see the Dictionary Syntax document.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmRecordModify(
	HFDB		   hDb,
	FLMUINT		uiContainer,
	FLMUINT		uiDrn,
	FlmRecord *	pRecord,
	FLMUINT		uiAutoTrans
	)
{
	RCODE			rc = FERR_OK;
	RCODE			rc1;
	FDB *			pDb = (FDB *)hDb;
	FlmRecord *	pOldRecord = NULL;
	LFILE *		pLFile;
	FLMBOOL		bStartedAutoTrans = FALSE;
	FLMBOOL		bProcessedKeys = FALSE;
	FLMBOOL		bLogCompleteIndexSet = FALSE;
	FLMBOOL		bHadUniqueKeys;
	DB_STATS *	pDbStats = NULL;
	F_TMSTAMP	StartTime;

	if( uiContainer == FLM_TRACKER_CONTAINER)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmDoUpdateCS( pDb, FCS_OP_RECORD_MODIFY, uiContainer,
									&uiDrn, pRecord, uiAutoTrans);
		goto ExitCS;
	}

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
			FDB_TRANS_GOING_OK, uiAutoTrans, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	if( pDb->uiFlags & FDB_COMMITTING_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	if ((pDbStats = pDb->pDbStats) != NULL)
	{
		f_timeGetTimeStamp( &StartTime);
	}

	// Make sure we have a valid record

	if( !pRecord)
	{
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// We cannot modify a record that is marked read-only.

	if( pRecord->isReadOnly())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( !uiDrn || (uiDrn == (FLMUINT) DRN_LAST_MARKER))
	{
		rc = RC_SET( FERR_BAD_DRN);
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	// DICTIONARY RECORD MODIFY

	if( uiContainer == FLM_LOCAL_DICT_CONTAINER) 
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
			uiContainer, uiDrn, FALSE, NULL, NULL, &pOldRecord)))
		{

			// NOTE: Deliberately not reading in to cache if not found.

			if (rc != FERR_NOT_FOUND)
			{
				goto Exit;
			}
			if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDrn, &pOldRecord, NULL, NULL)))
			{
				goto Exit;
			}
		}

		// Sanity check -- make sure that the new and old records point at
		// different objects.

		flmAssert( pRecord != pOldRecord);

		if( RC_BAD( rc = flmLFileDictUpdate( pDb, &pLFile, &uiDrn, 
				pRecord, pOldRecord,
				(uiAutoTrans & FLM_DO_IN_BACKGROUND) ? TRUE : FALSE, 
				(uiAutoTrans & FLM_SUSPENDED) ? TRUE : FALSE,
				&bLogCompleteIndexSet)))
		{
			goto Exit;
		}

		pRecord->setID( uiDrn);
		pRecord->setContainerID( uiContainer);
		if( RC_BAD( rc = flmRcaInsertRec( pDb, pLFile, uiDrn, pRecord)))
		{
			goto Exit;
		}

		goto Exit;
	}

	// First read the old record, and delete it's keys.
	// To do this we need to be able to read any purged fields and delete any
	// keys they build.

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
		uiContainer, uiDrn, FALSE, NULL, NULL, &pOldRecord)))
	{

		// NOTE: Deliberately not reading in to cache if not found.

		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDrn, &pOldRecord, NULL, NULL)))
		{
			goto Exit;
		}
	}

	// Sanity check -- make sure that the new and old records point at
	// different objects.

	flmAssert( pRecord != pOldRecord);

	bProcessedKeys = TRUE;
	bHadUniqueKeys = FALSE;
	if( RC_BAD( rc = flmProcessRecFlds( pDb, NULL, uiContainer, uiDrn, pOldRecord,
													KREF_DEL_KEYS | KREF_IN_MODIFY,
													TRUE,		// Purged Field OK
													&bHadUniqueKeys)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmProcessRecFlds( pDb, NULL, uiContainer, uiDrn, pRecord,
													KREF_ADD_KEYS | KREF_IN_MODIFY,
													FALSE,
													&bHadUniqueKeys)))
	{
		goto Exit;
	}
	
	// NOTE: The LFile table may have changed locations if the dictionary 
	// was update because of a change in a field state
	
	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FSRecUpdate( pDb, pLFile, pRecord, uiDrn, REC_UPD_MODIFY)))
	{
		 goto Exit;
	}

	// Finish up the keys adding the unique keys to the indexes.

	if( RC_BAD( rc = KYProcessDupKeys( pDb, bHadUniqueKeys)))
	{
		// Undo the record that was modified - replace with original record.
		if( RC_BAD( rc1 = FSRecUpdate( pDb, pLFile, 
			pOldRecord, uiDrn, REC_UPD_MODIFY)))
		{
			rc = (rc == FERR_NOT_UNIQUE) ? rc1 : rc;
		}
		goto Exit;
	}

	// Insert record into cache
	
	pRecord->setID( uiDrn);
	pRecord->setContainerID( uiContainer);
	
	if( RC_BAD( rc = flmRcaInsertRec( pDb, pLFile, uiDrn, pRecord)))
	{
		if ( rc != FERR_MEM)
		{
			flmAssert( 0);
		}
		
		// Undo the record that was modified - replace with original record.
		
		FSRecUpdate( pDb, pLFile, pOldRecord, uiDrn, REC_UPD_MODIFY);
		goto Exit;
	}

	// Don't make this call until we are sure of success - because we want to
	// be able to back things out of KREF table.

	KYFinishCurrentRecord( pDb);

Exit:

	if( RC_BAD( rc) && bProcessedKeys)
	{
		KYAbortCurrentRecord( pDb);
	}

	// Add the BLOB entries to the blob list.
	
	rc = FB_OperationEnd( pDb, rc);

	if( RC_OK( rc))
	{
		if( RC_OK( rc = pDb->pFile->pRfl->logUpdate( 
				uiContainer, uiDrn, uiAutoTrans, pOldRecord, pRecord)) && 
			 bLogCompleteIndexSet &&
			 pDb->pFile->FileHdr.uiVersionNum <= FLM_FILE_FORMAT_VER_4_51)
		{
			// Log the fact that we indexed everything so the redo will also
			// index all data records in the container.

			rc = pDb->pFile->pRfl->logIndexSet( uiDrn, 0, 1, 0xFFFFFFFF);
		}
	}

	if( pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->RecordModifies.ui64ElapMilli);
		pDbStats->RecordModifies.ui64Count++;
		pDbStats->bHaveStats = TRUE;
	}

	if( gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmUpdEventCallback( pDb, F_EVENT_MODIFY_RECORD, hDb, rc, uiDrn,
								uiContainer, pRecord, pOldRecord);
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pDb->pFile->uiFFileId, 
		pDb->LogHdr.uiCurrTransID, uiContainer, uiDrn, rc, "RMod");
#endif

	// If started an automatic transaction, end it.

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

	if( pOldRecord)
	{
		pOldRecord->Release();
		pOldRecord = NULL;
	}

ExitCS:

	flmExit( FLM_RECORD_MODIFY, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : Deletes a record from a container.
Notes: If an index definition record or a container definition record is
		 deleted from the dictionary container, the index B-TREE container or
		 container B-TREE will be deleted automatically when the transaction
		 commits.  Field definition records can only be deleted from the
		 dictionary using this routine if the field is not in use.  For more
		 information on deletion of field definitions, see the Dictionary Syntax
		 document.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmRecordDelete(
	HFDB	 		hDb,
	FLMUINT		uiContainer,
	FLMUINT		uiDrn,
	FLMUINT		uiAutoTrans
	)
{
	RCODE				rc = FERR_OK;
	LFILE *			pLFile;
	FDB *				pDb = (FDB *)hDb;
	FLMBOOL			bStartedAutoTrans;
	FlmRecord *		pOldRecord;
	FlmRecord **	ppOldRecord;
	DB_STATS *		pDbStats = NULL;
	F_TMSTAMP		StartTime;

	if( uiContainer == FLM_TRACKER_CONTAINER)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmDoUpdateCS( pDb, FCS_OP_RECORD_DELETE, uiContainer,
									&uiDrn, NULL, uiAutoTrans);
		goto ExitCS;
	}

	bStartedAutoTrans = FALSE;
	pOldRecord = NULL;
	ppOldRecord = NULL;

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS, FDB_TRANS_GOING_OK,
											uiAutoTrans, &bStartedAutoTrans)))
	{
		goto Exit;
	}

	if( (pDbStats = pDb->pDbStats) != NULL)
	{
		f_timeGetTimeStamp( &StartTime);
	}

	if( !uiDrn || uiDrn == (FLMUINT) DRN_LAST_MARKER)
	{
		rc = RC_SET( FERR_BAD_DRN);
		goto Exit;
	}

	if( RC_BAD(rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		goto Exit;
	}

	if( gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		// Do not have flmDeleteRecord fetch the old version of the record
		// unless an event callback is registered.
		
		ppOldRecord = &pOldRecord;
	}
	
	// NOTE: pLFile should NOT be used after this call, because flmDeleteRecord
	// could actually change its position in memory due to field changes.

	if (RC_BAD( rc = flmDeleteRecord( pDb, pLFile, uiDrn, ppOldRecord, FALSE)))
	{
		goto Exit;
	}

Exit:

	if( RC_OK( rc))
	{
		rc = pDb->pFile->pRfl->logUpdate( uiContainer, uiDrn, 
			uiAutoTrans, NULL, NULL);
	}

	if( pDbStats)
	{
		flmAddElapTime( &StartTime, &pDbStats->RecordDeletes.ui64ElapMilli);
		pDbStats->RecordDeletes.ui64Count++;
		pDbStats->bHaveStats = TRUE;
	}

	if( gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmUpdEventCallback( pDb, F_EVENT_DELETE_RECORD, hDb, rc, uiDrn,
								uiContainer, NULL, pOldRecord);
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pDb->pFile->uiFFileId, 
		pDb->LogHdr.uiCurrTransID, uiContainer, uiDrn, rc, "RDel");
#endif

	// If started an automatic transaction, end it.

	if( bStartedAutoTrans)
	{
		rc = flmEndAutoTrans( pDb, rc);
	}

	if( pOldRecord)
	{
		pOldRecord->Release();
		pOldRecord = NULL;
	}

ExitCS:

	flmExit( FLM_RECORD_DELETE, pDb, rc);

	return( rc);
}

/************************************************************************
Desc : Deletes a record from a container.
Notes: This is the internal call that corresponds to FlmRecordDelete.
		 It may also be called by the database checking code when
		 repairing an index.
*************************************************************************/
RCODE	flmDeleteRecord(
	FDB *				pDb,					// Operation context
	LFILE *			pLFile,				// LFILE of container to delete from.
	FLMUINT			uiDrn,				// DRN of the record to be deleted.
	FlmRecord **	ppOldRecord,		// Returns old record, if not null.
	FLMBOOL			bMissingKeysOK		// Is it OK for keys to be missing in the
												// indexes that are to be updated?
												// should we report when index entries
												// aren't found? TRUE=don't report,
												// FALSE=do report.
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pOldRecord = NULL;
	FLMUINT			uiContainer = pLFile->uiLfNum;
	FLMUINT			uiAction;
	FLMBOOL			bHadUniqueKeys;
	FLMBOOL			bProcessedKeys = FALSE;

	if( pDb->uiFlags & FDB_COMMITTING_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
		uiContainer, uiDrn, FALSE, NULL, NULL, &pOldRecord)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDrn, 
			&pOldRecord, NULL, NULL)))
		{
			goto Exit;
		}
	}

	if( uiContainer == FLM_LOCAL_DICT_CONTAINER )
	{
		if( RC_OK( rc = flmLFileDictUpdate( pDb, &pLFile, 
			&uiDrn, NULL, pOldRecord, FALSE, FALSE, NULL)))
		{
			rc = flmRcaRemoveRec( pDb, uiContainer, uiDrn);
		}
		
		goto Exit;
	}

	// First read the record, then delete it's keys and then it.

	uiAction = (FLMUINT)((bMissingKeysOK)
							  ? (FLMUINT)(KREF_DEL_KEYS | KREF_MISSING_KEYS_OK)
							  : (FLMUINT)(KREF_DEL_KEYS));
	bProcessedKeys = TRUE;
	bHadUniqueKeys = FALSE;
	if( RC_BAD(rc = flmProcessRecFlds( pDb, NULL, uiContainer,
													uiDrn, pOldRecord, uiAction,
													TRUE,
													&bHadUniqueKeys)))
	{
		goto Exit;
	}
	
	// NOTE: The LFile table may have changed locations if the dictionary 
	// was updated because of a change in a field state
	
	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = KYProcessDupKeys( pDb, bHadUniqueKeys)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FSRecUpdate( pDb, pLFile, NULL, uiDrn, REC_UPD_DELETE)))
	{
		goto Exit;
	}

	// Remove the record from cache

	if( RC_BAD( rc = flmRcaRemoveRec( pDb, uiContainer, uiDrn)))
	{
		(void)FSRecUpdate( pDb, pLFile, pOldRecord, uiDrn, REC_UPD_ADD);
		goto Exit;
	}

	// Don't make this call until we are sure of success - because we want to
	// be able to back things out of KREF table.

	KYFinishCurrentRecord( pDb);

Exit:

	if (RC_BAD( rc) && bProcessedKeys)
	{
		KYAbortCurrentRecord( pDb);
	}

	if( ppOldRecord)
	{
		*ppOldRecord = pOldRecord;
	}
	else if( pOldRecord)
	{
		pOldRecord->Release();
	}

	// Add the BLOB entries to the blob list.
	
	rc = FB_OperationEnd( pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:		Returns the next DRN that record ADD would return.  The database
			must be in an existing update transaction.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmReserveNextDrn(
	HFDB			hDb,
	FLMUINT		uiContainer,
	FLMUINT *	puiDrnRV)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	LFILE *		pLFile;
	FLMBOOL		bIgnore;
	FLMUINT		uiDrn = 0;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send the request

		if( RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_RECORD, FCS_OP_RESERVE_NEXT_DRN)))
		{
			goto ExitCS;
		}

		if( uiContainer)
		{
			if (RC_BAD( rc = Wire.sendNumber(
				WIRE_VALUE_CONTAINER_ID, uiContainer)))
			{
				goto Transmission_Error;
			}
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response

		if( RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto ExitCS;
		}

		*puiDrnRV = Wire.getDrn();
		goto ExitCS;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto ExitCS;
	}

	bIgnore = FALSE;					// Set to shut up compiler.

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
										FDB_TRANS_GOING_OK,	// byFlags
										0, 						// wAutoTrans
										&bIgnore)))				// bStartedAutoTrans
	{
		goto Exit;
	}

	if( pDb->uiFlags & FDB_COMMITTING_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	if( RC_BAD( fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
#ifdef FLM_DBG_LOG
		uiDrn = 0;
#endif
		goto Exit;
	}
	uiDrn = (FLMUINT) 0;					// Must initialize before call.
	if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, TRUE, &uiDrn)))
	{
#ifdef FLM_DBG_LOG
		uiDrn = 0;
#endif
		goto Exit;
	}

	*puiDrnRV = uiDrn;						// Set return value.

Exit:

	if (RC_OK( rc))
	{
		rc = pDb->pFile->pRfl->logUpdatePacket( 
			RFL_RESERVE_DRN_PACKET, uiContainer, *puiDrnRV, 0);
	}

	if( gv_FlmSysData.UpdateEvents.pEventCBList)
	{
		flmUpdEventCallback( pDb, F_EVENT_RESERVE_DRN, hDb, rc, *puiDrnRV,
								uiContainer, NULL, NULL);
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pDb->pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			uiContainer, uiDrn, rc, "RDrn");
#endif

ExitCS:

	flmExit( FLM_RESERVE_NEXT_DRN, pDb, rc);

	return( rc);
}

/****************************************************************************
Desc:		Searches for an available DRN in the dictionary container.
			Differs from FlmReserveNextDrn in that it will attempt to reuse
			dictionary DRNS.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmFindUnusedDictDrn(
	HFDB					hDb,
	FLMUINT				uiStartDrn,
	FLMUINT				uiEndDrn,
	FLMUINT *			puiDrnRV)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore = FALSE;
	FDICT *		pDict;
	FLMUINT		uiCurrDrn;
	FLMUINT		uiStopSearch;

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS, FDB_TRANS_GOING_OK,
		0, &bIgnore)))
	{
		*puiDrnRV = (FLMUINT)-1;
		goto Exit;
	}

	// Search through the ITT table looking for the first occurance
	// of ITT_EMPTY_SLOT
	
	pDict = pDb->pDict;
	uiCurrDrn = f_max( uiStartDrn, 1);
	uiStopSearch = f_min( uiEndDrn, pDict->uiIttCnt - 1);
	
	while (uiCurrDrn <= uiStopSearch)	
	{
		if (pDict->pIttTbl[ uiCurrDrn].uiType == ITT_EMPTY_SLOT)
		{
			break;
		}
		else
		{
			uiCurrDrn++;
		}	
	}

	if (uiCurrDrn > uiEndDrn)
	{
		rc = RC_SET( FERR_NO_MORE_DRNS);
		goto Exit;
	}

	*puiDrnRV = uiCurrDrn;

Exit:

	fdbExit( pDb);
	return( rc);
}
