//-------------------------------------------------------------------------
// Desc:	Routines for database configuration.
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmDbGetSizes(
	FDB *						pDb,
	FLMUINT64 *				pui64DbFileSize,
	FLMUINT64 *				pui64RollbackFileSize,
	FLMUINT64 *				pui64RflFileSize);

void flmGetCPInfo(
	void *					pFilePtr,
	CHECKPOINT_INFO *		pCheckpointInfo);

/*******************************************************************************
Desc:	 Sets indexing callback function
*******************************************************************************/
FLMEXP void FLMAPI FlmSetIndexingCallback(
	HFDB						hDb,
	IX_CALLBACK				fnIxCallback,
	void *					pvAppData)
{
	((FDB *)hDb)->fnIxCallback = fnIxCallback;
	((FDB *)hDb)->IxCallbackData = pvAppData;
}

/*******************************************************************************
Desc:	 Returns indexing callback function
*******************************************************************************/
FLMEXP void FLMAPI FlmGetIndexingCallback(
	HFDB						hDb,
	IX_CALLBACK *			pfnIxCallback,
	void **					ppvAppData)
{
	if (pfnIxCallback)
	{
		*pfnIxCallback = ((FDB *)hDb)->fnIxCallback;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB *)hDb)->IxCallbackData;
	}
}

/*******************************************************************************
Desc : Configures a callback function which allows validation of records
		 before they are returned to the user or committed to the
		 database.
Notes: This function stores a pointer to a callback function which is
		 called whenever a record is added, deleted, modified or
		 retrieved.  This allows an application to validate record operations
		 before they are committed to the database (update operations)
		 or before records are returned to the application (read operations).
		 By default, no record validation is performed by FLAIM.
*******************************************************************************/
FLMEXP void FLMAPI FlmSetRecValidatorHook(
	HFDB						hDb,
	REC_VALIDATOR_HOOK   fnRecValidatorHook,
	void *					pvAppData)
{
	((FDB *)hDb)->fnRecValidator = fnRecValidatorHook;
	((FDB *)hDb)->RecValData = pvAppData;
}

/*******************************************************************************
Desc : Returns to the user the sessions current Rec Validator Hook values.
*******************************************************************************/
FLMEXP void FLMAPI FlmGetRecValidatorHook(
	HFDB						hDb,
	REC_VALIDATOR_HOOK * pfnRecValidatorHook,
	void **					ppvAppData)
{
	if (pfnRecValidatorHook)
	{
		*pfnRecValidatorHook = ((FDB *)hDb)->fnRecValidator;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB *)hDb)->RecValData;
	}
}

/*******************************************************************************
Desc : Configures a callback function which is called to return general
		 purpose information.
*******************************************************************************/
FLMEXP void FLMAPI FlmSetStatusHook(
	HFDB						hDb,
	STATUS_HOOK    		fnStatusHook,
	void *					pvAppData)
{
	((FDB *)hDb)->fnStatus = fnStatusHook;
	((FDB *)hDb)->StatusData = pvAppData;
}

/*******************************************************************************
Desc : Returns to the user the session's current status hook values.
*******************************************************************************/
FLMEXP void FLMAPI FlmGetStatusHook(
	HFDB						hDb,
	STATUS_HOOK *			pfnStatusHook,
	void **					ppvAppData)
{
	if (pfnStatusHook)
	{
		*pfnStatusHook = ((FDB *)hDb)->fnStatus;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB *)hDb)->StatusData;
	}
}

/*******************************************************************************
Desc:	Allows an application to configure various options for a database.
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmDbConfig(
	HFDB						hDb,
	eDbConfigType			eConfigType,
	void *					pvValue1,
	void *		   		pvValue2)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = (FDB *)hDb;
	FFILE *					pFile = pDb->pFile;
	LFILE *					pLFile;
	FLMBOOL					bDbInitialized = FALSE;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bDbLocked = FALSE;
	
	// Process the client/server request

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_CONFIG)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TYPE, (FLMUINT)eConfigType)))
		{
			goto Transmission_Error;
		}

		switch( eConfigType)
		{
			case FDB_SET_APP_VERSION:
			case FDB_RFL_KEEP_FILES:
			case FDB_KEEP_ABORTED_TRANS_IN_RFL:
			case FDB_AUTO_TURN_OFF_KEEP_RFL:
			case FDB_SET_APP_DATA:
			{
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER2,
					(FLMUINT)pvValue1)))
				{
					goto Transmission_Error;
				}
				break;
			}

			case FDB_RFL_DIR:
			{
				FLMUNICODE *	puzRflDir;

				if( RC_BAD( rc = fcsConvertNativeToUnicode( 
					Wire.getPool(), (const char *)pvValue1, &puzRflDir)))
				{
					goto Transmission_Error;
				}

				if( RC_BAD( rc = Wire.sendString( 
					WIRE_VALUE_FILE_PATH, puzRflDir)))
				{
					goto Transmission_Error;
				}
				break;
			}

			case FDB_RFL_FILE_LIMITS:
			{
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1,
					(FLMUINT)pvValue1)))
				{
					goto Transmission_Error;
				}

				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER2,
					(FLMUINT)pvValue2)))
				{
					goto Transmission_Error;
				}
				break;
			}

			case FDB_FILE_EXTEND_SIZE:
			case FDB_RFL_FOOTPRINT_SIZE:
			case FDB_RBL_FOOTPRINT_SIZE:
			{
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1,
					(FLMUINT)pvValue1)))
				{
					goto Transmission_Error;
				}
				break;
			}

			case FDB_RFL_ROLL_TO_NEXT_FILE:
			{
				break;
			}

			default:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
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

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// Process the local (non-C/S) request

	switch( eConfigType)
	{
		case FDB_RFL_KEEP_FILES:
		{
			FLMBOOL	bKeepFiles = (FLMBOOL)(pvValue1 ? TRUE : FALSE);

			// This operation is not legal for pre 4.3 databases.

			if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Make sure we don't have a transaction going

			if( pDb->uiTransType != FLM_NO_TRANS)
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}

			// Make sure there is no active backup running

			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( pFile->bBackupActive)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				rc = RC_SET( FERR_BACKUP_ACTIVE);
				goto Exit;
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Need to lock the database but not start a transaction yet.

			if( !(pDb->uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
			{
				if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0,
											FLM_NO_TIMEOUT)))
				{
					goto Exit;
				}
				
				bDbLocked = TRUE;
			}

			// If we aren't changing the keep flag, jump to exit without doing
			// anything.

			if ((bKeepFiles &&
				  pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]) ||
				 (!bKeepFiles &&
				  !pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]))
			{
				goto Exit;	// Will return FERR_OK;
			}

			// Force a checkpoint and roll to the next RFL file numbers.
			// When changing from keep to no-keep or vice versa, we need to
			// go to a new RFL file so that the new RFL file gets new
			// serial numbers and a new keep or no-keep flag.

			if (RC_BAD( rc = FlmDbCheckpoint( hDb, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			f_memcpy( pFile->ucUncommittedLogHdr,
						 pFile->ucLastCommittedLogHdr,
						 LOG_HEADER_SIZE);
			pFile->ucUncommittedLogHdr [LOG_KEEP_RFL_FILES] =
				(FLMBYTE)((bKeepFiles) ? (FLMBYTE)1 : (FLMBYTE)0);

			// Force a new RFL file - this will also write out the entire
			// log header - including the changes we made above.

			if (RC_BAD( rc = pFile->pRfl->finishCurrFile( pDb, TRUE)))
			{
				goto Exit;
			}
			
			// Update the RFL size
		
			if( bKeepFiles)
			{
				FLMUINT64		ui64RflDiskUsage;
				
				if( RC_BAD( rc = flmRflCalcDiskUsage( 
					pFile->pRfl->getRflDirPtr(), pFile->pRfl->getDbPrefixPtr(),
					pFile->FileHdr.uiVersionNum, &ui64RflDiskUsage)))
				{
					goto Exit;
				}
				
				f_mutexLock( gv_FlmSysData.hShareMutex);
				pFile->ui64RflDiskUsage = ui64RflDiskUsage;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}
			
			break;
		}

		case FDB_RFL_DIR:
		{
			const char *	pszNewRflDir = (const char *)pvValue1;

			// This operation is not legal for pre 4.3 databases.

			if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Make sure we don't have a transaction going

			if( pDb->uiTransType != FLM_NO_TRANS)
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}

			// Make sure there is no active backup running

			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( pFile->bBackupActive)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				rc = RC_SET( FERR_BACKUP_ACTIVE);
				goto Exit;
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Make sure the path exists and that it is a directory
			// rather than a file.

			if (pszNewRflDir && *pszNewRflDir)
			{
				if( !gv_FlmSysData.pFileSystem->isDir( pszNewRflDir))
				{
					rc = RC_SET( FERR_IO_INVALID_PATH);
					goto Exit;
				}
			}

			// Need to lock the database because we can't change the RFL
			// directory until after the checkpoint has completed.  The
			// checkpoint code will unlock the transaction, but not the
			// file if we have an explicit lock.  We need to do this to
			// prevent another transaction from beginning before we have
			// changed the RFL directory.

			if( !(pDb->uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
			{
				if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0,
											FLM_NO_TIMEOUT)))
				{
					goto Exit;
				}
				bDbLocked = TRUE;
			}

			// Force a checkpoint and roll to the next RFL file numbers.  Both
			// of these steps are necessary to ensure that we won't have to do
			// any recovery using the current RFL file - because we do not
			// move the current RFL file to the new directory.  Forcing the
			// checkpoint ensures that we have no transactions that will need
			// to be recovered if we were to crash.  Rolling the RFL file number
			// ensures that no more transactions will be logged to the current
			// RFL file.

			if (RC_BAD( rc = FlmDbCheckpoint( hDb, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			// Force a new RFL file.

			if (RC_BAD( rc = pFile->pRfl->finishCurrFile( pDb, FALSE)))
			{
				goto Exit;
			}

			// Set the RFL directory to the new value now that we have
			// finished the checkpoint and rolled to the next RFL file.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			rc = pFile->pRfl->setRflDir( pszNewRflDir);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}

		case FDB_RFL_FILE_LIMITS:
		{
			FLMUINT	uiMinRflSize = (FLMUINT)pvValue1;
			FLMUINT	uiMaxRflSize = (FLMUINT)pvValue2;

			// Make sure the limits are valid.

			if (pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
			{

				// Maximum must be enough to hold at least one packet plus
				// the RFL header.  Minimum must not be greater than the
				// maximum.  NOTE: Minimum and maximum are allowed to be
				// equal, but in all cases, maximum takes precedence over
				// minimum.  We will first NOT exceed the maximum.  Then,
				// if possible, we will go above the minimum.

				if (uiMaxRflSize < RFL_MAX_PACKET_SIZE + 512)
				{
					uiMaxRflSize = RFL_MAX_PACKET_SIZE + 512;
				}
				if (uiMaxRflSize > gv_FlmSysData.uiMaxFileSize)
				{
					uiMaxRflSize = gv_FlmSysData.uiMaxFileSize;
				}
				if (uiMinRflSize > uiMaxRflSize)
				{
					uiMinRflSize = uiMaxRflSize;
				}
			}

			// Start an update transaction.  Must not already be one going.

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
											  0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS,
											  &bStartedTrans)))
			{
				goto Exit;
			}

			// Commit the transaction.

			UD2FBA( (FLMUINT32)uiMinRflSize,
				&pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
			if (pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
			{
				UD2FBA( (FLMUINT32)uiMaxRflSize,
					&pFile->ucUncommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);
			}
			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;
		}

		case FDB_RFL_ROLL_TO_NEXT_FILE:
		{
			// This operation is not legal for pre 4.3 databases.

			if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// NOTE: finishCurrFile will not roll to the next file if the current
			// file has not been created.

			if (RC_BAD( rc = pFile->pRfl->finishCurrFile( pDb, FALSE)))
			{
				goto Exit;
			}
			break;
		}

		case FDB_SET_APP_VERSION:
		{
			FLMUINT		uiOldMajorVer;
			FLMUINT		uiOldMinorVer;

			// Start an update transaction.

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
				0, FLM_AUTO_TRANS | FLM_NO_TIMEOUT, &bStartedTrans)))
			{
				goto Exit;
			}

			// Set the version.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			uiOldMajorVer = pFile->FileHdr.uiAppMajorVer;
			pFile->FileHdr.uiAppMajorVer = (FLMUINT)pvValue1;
			uiOldMinorVer = pFile->FileHdr.uiAppMinorVer;
			pFile->FileHdr.uiAppMinorVer = (FLMUINT)pvValue2;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Commit the transaction.  NOTE: This will always cause
			// us to write out the application version numbers, because
			// we always write out the prefix - first 512 bytes.

			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				// Undo the changes made above

				f_mutexLock( gv_FlmSysData.hShareMutex);
				pFile->FileHdr.uiAppMajorVer = uiOldMajorVer;
				pFile->FileHdr.uiAppMinorVer = uiOldMinorVer;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;
		}

		case FDB_KEEP_ABORTED_TRANS_IN_RFL:
		case FDB_AUTO_TURN_OFF_KEEP_RFL:
		{

			// These operations are not legal for pre 4.3 databases.

			if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Start an update transaction.  Must not already be one going.

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
											  0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS,
											  &bStartedTrans)))
			{
				goto Exit;
			}

			// Change the uncommitted log header

			if (eConfigType == FDB_KEEP_ABORTED_TRANS_IN_RFL)
			{
				pFile->ucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL] =
									(FLMBYTE)(pvValue1
												 ? (FLMBYTE)1
												 : (FLMBYTE)0);
			}
			else
			{
				pFile->ucUncommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL] =
									(FLMBYTE)(pvValue1
												 ? (FLMBYTE)1
												 : (FLMBYTE)0);
			}

			// Commit the transaction.

			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;
		}

		case FDB_FILE_EXTEND_SIZE:
		{
			pFile->uiFileExtendSize = (FLMUINT)pvValue1;
			break;
		}

		case FDB_RFL_FOOTPRINT_SIZE:
		{
			pFile->uiRflFootprintSize = 
				(FLMUINT)f_roundUp( (FLMUINT)pvValue1, 512);
			break;
		}
		
		case FDB_RBL_FOOTPRINT_SIZE:
		{
			pFile->uiRblFootprintSize = 
				(FLMUINT)f_roundUp( (FLMUINT)pvValue1, pFile->FileHdr.uiBlockSize);
			break;
		}

		case FDB_SET_APP_DATA:
		{
			pDb->pvAppData = pvValue1;
			break;
		}

		case FDB_SET_COMMIT_CALLBACK:
		{
			pDb->fnCommit = (COMMIT_FUNC)((FLMUINT)pvValue1);
			pDb->pvCommitData = pvValue2;
			break;
		}
		
		case FDB_SET_RFL_SIZE_THRESHOLD:
		{
			if( RC_BAD( rc = flmSetRflSizeThreshold( hDb, (FLMUINT)pvValue1, 
				FLM_MAX_UINT, FLM_MAX_UINT)))
			{
				goto Exit;
			}
			
			break;
		}

		case FDB_SET_RFL_SIZE_EVENT_INTERVALS:
		{
			FLMUINT			uiTimeInterval = (FLMUINT)pvValue1;
			FLMUINT			uiSizeInterval = (FLMUINT)pvValue2;
			
			if( RC_BAD( rc = flmSetRflSizeThreshold( hDb, FLM_MAX_UINT, 
				uiTimeInterval, uiSizeInterval)))
			{
				goto Exit;
			}
			
			break;
		}
		
		case FDB_ENABLE_FIELD_ID_TABLE:
		{
			if (pDb->pDict)
			{
				if (RC_BAD( rc = fdictGetContainer( pDb->pDict, (FLMUINT)pvValue1,
										&pLFile)))
				{
					goto Exit;
				}
				pLFile->bMakeFieldIdTable = (FLMBOOL)((FLMUINT)pvValue2);
			}
			else if (pDb->pFile->pDictList)
			{
				if (RC_BAD( rc = fdictGetContainer( pDb->pFile->pDictList,
										(FLMUINT)pvValue1, &pLFile)))
				{
					goto Exit;
				}
				pLFile->bMakeFieldIdTable = (FLMBOOL)((FLMUINT)pvValue2);
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

Exit:

	if( bStartedTrans)
	{
		flmAbortDbTrans( pDb);
	}

	if( bDbLocked)
	{
		FlmDbUnlock( hDb);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns database, rollback, and rollforward sizes.  We are guaranteed
		to be inside an update transaction at this point.
****************************************************************************/
FSTATIC RCODE flmDbGetSizes(
	FDB *					pDb,
	FLMUINT64 *			pui64DbFileSize,
	FLMUINT64 *			pui64RollbackFileSize,
	FLMUINT64 *			pui64RflFileSize)
{
	RCODE					rc = FERR_OK;
	FFILE *				pFile = pDb->pFile;
	FLMUINT				uiDbVersion = pFile->FileHdr.uiVersionNum;
	FLMUINT				uiEndAddress;
	FLMUINT				uiLastFileNumber;
	FLMUINT64			ui64LastFileSize;
	char					szTmpName[ F_PATH_MAX_SIZE];
	char					szRflDir[ F_PATH_MAX_SIZE];
	char					szPrefix[ F_FILENAME_SIZE];
	IF_FileHdl *		pFileHdl = NULL;
	IF_DirHdl *			pDirHdl = NULL;

	// Better be inside an update transaction at this point.

	flmAssert( pDb->uiTransType == FLM_UPDATE_TRANS);

	// See if they want the database files sizes.

	if (pui64DbFileSize)
	{
		uiEndAddress = pDb->LogHdr.uiLogicalEOF;
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( uiLastFileNumber >= 1 &&
					  uiLastFileNumber <= MAX_DATA_BLOCK_FILE_NUMBER( uiDbVersion));

		// Get the actual size of the last file.

		if (RC_BAD( rc = pDb->pSFileHdl->getFileSize( uiLastFileNumber,
										&ui64LastFileSize)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				if (uiLastFileNumber > 1)
				{
					rc = FERR_OK;
					ui64LastFileSize = 0;
				}
				else
				{

					// Should always be a data file #1

					flmAssert( 0);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// One of two situations exists with respect to the last
		// file: 1) it has not been fully written out yet (blocks
		// are still cached, or 2) it has been written out and
		// extended beyond what the logical EOF shows.  We want
		// the larger of these two possibilities.

		if (FSGetFileOffset( uiEndAddress) > ui64LastFileSize)
		{
			ui64LastFileSize = FSGetFileOffset( uiEndAddress);
		}

		if (uiLastFileNumber == 1)
		{

			// Only one file - use last file's size.

			*pui64DbFileSize = ui64LastFileSize;
		}
		else
		{

			// Size is the sum of full size for all files except the last one,
			// plus the calculated (or actual) size of the last one.

			(*pui64DbFileSize) = (FLMUINT64)(uiLastFileNumber - 1) *
											 (FLMUINT64)pFile->uiMaxFileSize +
											 ui64LastFileSize;
		}
	}

	// See if they want the rollback files sizes.

	if (pui64RollbackFileSize)
	{
		uiEndAddress = (FLMUINT)FB2UD(
								&pFile->ucUncommittedLogHdr [LOG_ROLLBACK_EOF]);
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( !uiLastFileNumber ||
					  (uiLastFileNumber >=
							FIRST_LOG_BLOCK_FILE_NUMBER( uiDbVersion) &&
					   uiLastFileNumber <=
							MAX_LOG_BLOCK_FILE_NUMBER( uiDbVersion)));

		// Get the size of the last file number.

		if (RC_BAD( rc = pDb->pSFileHdl->getFileSize( uiLastFileNumber,
										&ui64LastFileSize)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				if (uiLastFileNumber)
				{
					rc = FERR_OK;
					ui64LastFileSize = 0;
				}
				else
				{

					// Should always have rollback file #0

					flmAssert( 0);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// If the EOF offset for the last file is greater than the
		// actual file size, use it instead of the actual file size.

		if (FSGetFileOffset( uiEndAddress) > ui64LastFileSize)
		{
			ui64LastFileSize = FSGetFileOffset( uiEndAddress);
		}

		// Special case handling here because rollback file numbers start with
		// zero and then skip to a file number that is one beyond the
		// highest data file number - so the calculation for file size needs
		// to account for this.

		if (!uiLastFileNumber)
		{
			*pui64RollbackFileSize = ui64LastFileSize;
		}
		else
		{
			FLMUINT	uiFirstLogFileNum =
							FIRST_LOG_BLOCK_FILE_NUMBER( uiDbVersion);

			// Add full size of file zero plus a full size for every file
			// except the last one.

			(*pui64RollbackFileSize) = (FLMUINT64)(uiLastFileNumber -
																	uiFirstLogFileNum + 1) *
												 (FLMUINT64)pFile->uiMaxFileSize +
												 ui64LastFileSize;
		}
	}

	// See if they want the roll-forward log file sizes.

	if (pui64RflFileSize)
	{
		char *	pszDbFileName = pFile->pszDbPath;

		*pui64RflFileSize = 0;
		if (uiDbVersion < FLM_FILE_FORMAT_VER_4_3)
		{

			// For pre-4.3 versions, only need to get the size for one
			// RFL file.

			if (RC_BAD( rc = rflGetFileName( uiDbVersion, pszDbFileName,
														NULL, 1, szTmpName)))
			{
				goto Exit;
			}

			// Open the file and get its size.

			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( szTmpName,
			  						gv_FlmSysData.uiFileOpenFlags, &pFileHdl)))
			{
				if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
				{
					rc = FERR_OK;
					ui64LastFileSize = 0;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = pFileHdl->size( &ui64LastFileSize)))
				{
					goto Exit;
				}
			}
			
			if (pFileHdl)
			{
				pFileHdl->Release();
				pFileHdl = NULL;
			}
			
			*pui64RflFileSize = ui64LastFileSize;
		}
		else
		{

			// For 4.3 and greater, need to scan the RFL directory for
			// RFL files.  The call below to rflGetDirAndPrefix is done
			// to get the prefix.  It will not return the correct
			// RFL directory name, because we are passing in a NULL
			// RFL directory path (which may or may not be correct).
			// That's OK, because we get the RFL directory directly
			// from the F_Rfl object anyway.

			if (RC_BAD( rc = rflGetDirAndPrefix( uiDbVersion, pszDbFileName,
										NULL, szRflDir, szPrefix)))
			{
				goto Exit;
			}

			// We need to get the RFL directory from the F_Rfl object.

			f_strcpy( szRflDir, pFile->pRfl->getRflDirPtr());

			// See if the directory exists.  If not, we are done.

			if (gv_FlmSysData.pFileSystem->isDir( szRflDir))
			{

				// Open the directory and scan for RFL files.

				if (RC_BAD( rc = gv_FlmSysData.pFileSystem->openDir(
					szRflDir, "*", &pDirHdl)))
				{
					goto Exit;
				}
				for (;;)
				{
					if (RC_BAD( rc = pDirHdl->next()))
					{
						if (rc == FERR_IO_NO_MORE_FILES)
						{
							rc = FERR_OK;
							break;
						}
						else
						{
							goto Exit;
						}
					}
					pDirHdl->currentItemPath( szTmpName);

					// If the item looks like an RFL file name, get
					// its size.

					if (!pDirHdl->currentItemIsDir() &&
						  rflGetFileNum( uiDbVersion, szPrefix, szTmpName,
												&uiLastFileNumber))
					{

						// Open the file and get its size.

						if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile(
							szTmpName, gv_FlmSysData.uiFileOpenFlags, &pFileHdl)))
						{
							if( rc == FERR_IO_PATH_NOT_FOUND ||
								 rc == FERR_IO_INVALID_PATH)
							{
								rc = FERR_OK;
								ui64LastFileSize = 0;
							}
							else
							{
								goto Exit;
							}
						}
						else
						{
							if( RC_BAD( rc = pFileHdl->size( &ui64LastFileSize)))
							{
								goto Exit;
							}
						}
						
						if (pFileHdl)
						{
							pFileHdl->Release();
							pFileHdl = NULL;
						}
						
						(*pui64RflFileSize) += ui64LastFileSize;
					}
				}
			}
		}
	}

Exit:
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	if (pDirHdl)
	{
		pDirHdl->Release();
	}
	return( rc);
}

/*******************************************************************************
Desc:	Returns information about a particular database.
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetConfig(
	HFDB					hDb,
	eDbGetConfigType	eGetConfigType,
	void *				pvValue1,
	void *				pvValue2,
	void *				pvValue3)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = (FDB *)hDb;
	FFILE *				pFile = pDb->pFile;
	FLMBOOL				bDbInitialized = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiTransType = FLM_NO_TRANS;
	CHECKPOINT_INFO *	pCheckpointInfo;

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);
		CREATE_OPTS			createOpts;

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_GET_CONFIG)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TYPE, 
			(FLMUINT)eGetConfigType)))
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

		switch( eGetConfigType)
		{
			case FDB_GET_VERSION:
			{
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)pvValue1) = createOpts.uiVersionNum;
				break;
			}
			
			case FDB_GET_BLKSIZ:
			{
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)pvValue1) = createOpts.uiBlockSize;
				break;
			}
			
			case FDB_GET_DEFAULT_LANG:
			{
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)pvValue1) = createOpts.uiDefaultLanguage;
				break;
			}
			
			case FDB_GET_PATH:
			case FDB_GET_RFL_DIR:
			{
				char *		pszPath;
				F_Pool *		pPool = Wire.getPool();
				void *		pvMark = pPool->poolMark();

				if( RC_BAD( rc = fcsConvertUnicodeToNative( pPool,
					(FLMUNICODE *)Wire.getFilePath(), &pszPath)))
				{
					goto Exit;
				}
				
				f_strcpy( (char *)pvValue1, pszPath);
				pPool->poolReset( pvMark);
				break;
			}
			
			case FDB_GET_TRANS_ID:
			case FDB_GET_RFL_FILE_NUM:
			case FDB_GET_RFL_HIGHEST_NU:
			case FDB_GET_LAST_BACKUP_TRANS_ID:
			case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
			case FDB_GET_FILE_EXTEND_SIZE:
			case FDB_GET_RFL_FOOTPRINT_SIZE:
			case FDB_GET_RBL_FOOTPRINT_SIZE:
			case FDB_GET_APP_DATA:
			case FDB_GET_NEXT_INC_BACKUP_SEQ_NUM:
			case FDB_GET_DICT_SEQ_NUM:
			case FDB_GET_FFILE_ID:
			case FDB_GET_MUST_CLOSE_RC:
			{
				*((FLMUINT *)pvValue1) = (FLMUINT)Wire.getNumber1();
				break;
			}
			
			case FDB_GET_RFL_FILE_SIZE_LIMITS:
			{
				if (pvValue1)
				{
					*((FLMUINT *)pvValue1) = (FLMUINT)Wire.getNumber1();
				}
				
				if (pvValue2)
				{
					*((FLMUINT *)pvValue2) = (FLMUINT)Wire.getNumber2();
				}
				
				break;
			}
			
			case FDB_GET_RFL_KEEP_FLAG:
			case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
			case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
			{
				*((FLMBOOL *)pvValue1) = Wire.getBoolean();
				break;
			}
			
			case FDB_GET_CHECKPOINT_INFO:
			{
				rc = fcsExtractCheckpointInfo( Wire.getHTD(), 
							(CHECKPOINT_INFO *)pvValue1);
				break;
			}
			
			case FDB_GET_LOCK_HOLDER:
			{
				rc = fcsExtractLockUser( Wire.getHTD(), FALSE, 
					((F_LOCK_USER *)pvValue1));
				break;
			}
			
			case FDB_GET_LOCK_WAITERS:
			{
				rc = fcsExtractLockUser( Wire.getHTD(), TRUE, ((void *)pvValue1));
				break;
			}
			
			case FDB_GET_SERIAL_NUMBER:
			{
				f_memcpy( (FLMBYTE *)pvValue1, 
					Wire.getSerialNum(), F_SERIAL_NUM_SIZE);
				break;
			}
			
			case FDB_GET_SIZES:
			{
				if (pvValue1)
				{
					*((FLMUINT64 *)pvValue1) = (FLMUINT64)Wire.getNumber1();
				}
				
				if (pvValue2)
				{
					*((FLMUINT64 *)pvValue2) = (FLMUINT64)Wire.getNumber2();
				}
				
				if (pvValue3)
				{
					*((FLMUINT64 *)pvValue3) = (FLMUINT64)Wire.getNumber3();
				}
				
				break;
			}

			default:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				break;
			}
		}

		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (eGetConfigType == FDB_GET_RFL_FILE_NUM ||
				eGetConfigType == FDB_GET_RFL_HIGHEST_NU ||
				eGetConfigType == FDB_GET_RFL_FILE_SIZE_LIMITS ||
				eGetConfigType == FDB_GET_RFL_KEEP_FLAG ||
				eGetConfigType == FDB_GET_LAST_BACKUP_TRANS_ID ||
				eGetConfigType == FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP ||
				eGetConfigType == FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG ||
				eGetConfigType == FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG ||
				eGetConfigType == FDB_GET_SIZES ||
				eGetConfigType == FDB_GET_NEXT_INC_BACKUP_SEQ_NUM)
	{
		uiTransType = FLM_UPDATE_TRANS;
	}

	bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, uiTransType,
					FDB_TRANS_GOING_OK | FDB_DONT_RESET_DIAG, 
					FLM_NO_TIMEOUT | FLM_AUTO_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	switch( eGetConfigType)
	{
		case FDB_GET_VERSION:
		{
			*((FLMUINT *)pvValue1) = pFile->FileHdr.uiVersionNum;
			break;
		}
		
		case FDB_GET_BLKSIZ:
		{
			*((FLMUINT *)pvValue1) = pFile->FileHdr.uiBlockSize;
			break;
		}
		
		case FDB_GET_DEFAULT_LANG:
		{
			*((FLMUINT *)pvValue1) = pFile->FileHdr.uiDefaultLanguage;
			break;
		}
		
		case FDB_GET_PATH:
		{
			if( RC_BAD( rc = flmGetFilePath( pFile, ((char *)pvValue1))))
			{
				goto Exit;
			}
			break;
		}
		
		case FDB_GET_TRANS_ID:
		{
			if (pDb->uiTransType != FLM_NO_TRANS)
			{
				*((FLMUINT *)pvValue1) = pDb->LogHdr.uiCurrTransID;
			}
			else if (pDb->uiFlags & FDB_HAS_FILE_LOCK)
			{
	
				// Get last committed value.
	
				*((FLMUINT *)pvValue1) = 
						FB2UD( &pFile->ucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);
			}
			else
			{
				*((FLMUINT *)pvValue1) = 0;
			}
			break;
		}
		
		case FDB_GET_CHECKPOINT_INFO:
		{
			pCheckpointInfo = (CHECKPOINT_INFO *)pvValue1;
			f_mutexLock( gv_FlmSysData.hShareMutex);
			flmGetCPInfo( pFile, pCheckpointInfo);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
		
		case FDB_GET_LOCK_HOLDER:
		{
			F_LOCK_USER *		pLockUser = (F_LOCK_USER *)pvValue1;
			
			if (pFile->pFileLockObj)
			{
				rc = pFile->pFileLockObj->getLockInfo( 0, NULL, 
					&pLockUser->uiThreadId, &pLockUser->uiThreadId); 
			}
			else
			{
				((F_LOCK_USER *)pvValue1)->uiThreadId = 0;
				((F_LOCK_USER *)pvValue1)->uiTime = 0;
			}
			break;
		}
		
		case FDB_GET_LOCK_WAITERS:
		{
			if (pFile->pFileLockObj)
			{
				rc = pFile->pFileLockObj->getLockQueue( (F_LOCK_USER **)pvValue1);
			}
			else
			{
				*((F_LOCK_USER **)pvValue1) = NULL;
			}
			break;
		}
	
		case FDB_GET_LOCK_WAITERS_EX:
		{
			IF_LockInfoClient * pLockInfo = (IF_LockInfoClient *)pvValue1;
	
			if (pFile->pFileLockObj)
			{
				rc = pFile->pFileLockObj->getLockInfo( pLockInfo);
			}
			else
			{
				pLockInfo->setLockCount( 0);
			}
			break;
		}
	
		case FDB_GET_RFL_DIR:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_strcpy( (char *)pvValue1, pDb->pFile->pRfl->getRflDirPtr());
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FDB_GET_RFL_FILE_NUM:
		{
			FLMUINT		uiLastCPFile;
			FLMUINT		uiLastTransFile;
	
			// Get the CP and last trans RFL file numbers.  Need to
			// return the higher of the two.  No need to lock the
			// mutex because we are in an update transaction.
	
			uiLastCPFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
				LOG_RFL_LAST_CP_FILE_NUM]);
	
			uiLastTransFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
				LOG_RFL_FILE_NUM]);
	
			*((FLMUINT *)pvValue1) = uiLastCPFile > uiLastTransFile
										 ? uiLastCPFile
										 : uiLastTransFile;
			break;
		}
	
		case FDB_GET_RFL_HIGHEST_NU:
		{
			FLMUINT		uiLastCPFile;
			FLMUINT		uiLastTransFile;
	
			// Get the CP and last trans RFL file numbers.  Need to
			// return the lower of the two minus 1.
	
			uiLastCPFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
				LOG_RFL_LAST_CP_FILE_NUM]);
	
			uiLastTransFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
				LOG_RFL_FILE_NUM]);
	
			*((FLMUINT *)pvValue1) =
				(FLMUINT)((uiLastCPFile < uiLastTransFile
							? uiLastCPFile
							: uiLastTransFile) - 1);
			break;
		}
	
		case FDB_GET_RFL_FILE_SIZE_LIMITS:
		{
			if (pvValue1)
			{
				*((FLMUINT *)pvValue1) = (FLMUINT)FB2UD(
						&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
			}
			if (pvValue2)
			{
				if (pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
				{
					*((FLMUINT *)pvValue2) = (FLMUINT)FB2UD(
						&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);
				}
				else
				{
					*((FLMUINT *)pvValue2) = (FLMUINT)FB2UD(
						&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
				}
			}
			break;
		}
	
		case FDB_GET_RFL_KEEP_FLAG:
		{
			*((FLMBOOL *)pvValue1) =
					pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_RFL_FILES]
					? TRUE
					: FALSE;
			break;
		}
	
		case FDB_GET_LAST_BACKUP_TRANS_ID:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			*((FLMUINT *)pvValue1) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr [LOG_LAST_BACKUP_TRANS_ID]);
			break;
		}
	
		case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			*((FLMUINT *)pvValue1) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr[ LOG_BLK_CHG_SINCE_BACKUP]);
			break;
		}
	
		case FDB_GET_SERIAL_NUMBER:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_memcpy( (FLMBYTE *)pvValue1, 
				&pDb->pFile->ucLastCommittedLogHdr [LOG_DB_SERIAL_NUM],
				F_SERIAL_NUM_SIZE);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				*((FLMBOOL *)pvValue1) = FALSE;
			}
			else
			{
				*((FLMBOOL *)pvValue1) =
					pDb->pFile->ucUncommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL]
					? TRUE
					: FALSE;
			}
			break;
		}
	
		case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				*((FLMBOOL *)pvValue1) = FALSE;
			}
			else
			{
				*((FLMBOOL *)pvValue1) =
					pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL]
					? TRUE
					: FALSE;
			}
			break;
		}
	
		case FDB_GET_SIZES:
		{
			rc = flmDbGetSizes( pDb, (FLMUINT64 *)pvValue1, (FLMUINT64 *)pvValue2,
											 (FLMUINT64 *)pvValue3);
			break;
		}
	
		case FDB_GET_FILE_EXTEND_SIZE:
		{
			*((FLMUINT *)pvValue1) = pDb->pFile->uiFileExtendSize;
			break;
		}
	
		case FDB_GET_RFL_FOOTPRINT_SIZE:
		{
			*((FLMUINT *)pvValue1) = pDb->pFile->uiRflFootprintSize;
			break;
		}

		case FDB_GET_RBL_FOOTPRINT_SIZE:
		{
			*((FLMUINT *)pvValue1) = pDb->pFile->uiRblFootprintSize;
			break;
		}
		
		case FDB_GET_APP_DATA:
		{
			*((void **)pvValue1) = pDb->pvAppData;
			break;
		}
	
		case FDB_GET_NEXT_INC_BACKUP_SEQ_NUM:
		{
			if (pDb->pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
			*((FLMUINT *)pvValue1) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);
			break;
		}
	
		case FDB_GET_DICT_SEQ_NUM:
		{
			if( pDb->pDict)
			{
				*((FLMUINT *)pvValue1) = pDb->pDict->uiDictSeq;
			}
			else
			{
				*((FLMUINT *)pvValue1) = pDb->pFile->pDictList->uiDictSeq;
			}
			break;
		}
		
		case FDB_GET_FFILE_ID:
		{
			*((FLMUINT *)pvValue1) = pDb->pFile->uiFFileId;
			break;
		}
		
		case FDB_GET_MUST_CLOSE_RC:
		{
			*((RCODE *)pvValue1) = pDb->pFile->rcMustClose;
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
		}
	}

Exit:

	if( bStartedTrans)
	{
		flmAbortDbTrans( pDb);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc: Retrieves the Checkpoint info for the pFile passed in.  This assumes the
		hShareMutex has already been locked.
*****************************************************************************/
void flmGetCPInfo(
	void *					pFilePtr,
	CHECKPOINT_INFO *		pCheckpointInfo)
{
	FFILE *	pFile;
	FLMUINT	uiElapTime;
	FLMUINT	uiCurrTime;
				
	flmAssert( pFilePtr);
	flmAssert( pCheckpointInfo);

	pFile = (FFILE *)pFilePtr;

	f_memset( pCheckpointInfo, 0, sizeof( CHECKPOINT_INFO));
	if (pFile->pCPInfo)
	{
		pCheckpointInfo->bRunning = pFile->pCPInfo->bDoingCheckpoint;
		if (pCheckpointInfo->bRunning)
		{
			if (pFile->pCPInfo->uiStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();

				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							pFile->pCPInfo->uiStartTime);
				pCheckpointInfo->uiRunningTime = 
					FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
			}
			else
			{
				pCheckpointInfo->uiRunningTime = 0;
			}
			pCheckpointInfo->bForcingCheckpoint =
				pFile->pCPInfo->bForcingCheckpoint;
			if (pFile->pCPInfo->uiForceCheckpointStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();
				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							pFile->pCPInfo->uiForceCheckpointStartTime);
				pCheckpointInfo->uiForceCheckpointRunningTime =
					FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
			}
			else
			{
				pCheckpointInfo->uiForceCheckpointRunningTime = 0;
			}
			
			pCheckpointInfo->iForceCheckpointReason =
				pFile->pCPInfo->iForceCheckpointReason;
			pCheckpointInfo->bWritingDataBlocks =
				pFile->pCPInfo->bWritingDataBlocks;
			pCheckpointInfo->uiLogBlocksWritten =
				pFile->pCPInfo->uiLogBlocksWritten;
			pCheckpointInfo->uiDataBlocksWritten =
				pFile->pCPInfo->uiDataBlocksWritten;
		}
		
		pCheckpointInfo->uiBlockSize =
			(FLMUINT)pFile->FileHdr.uiBlockSize;
		pCheckpointInfo->uiDirtyCacheBytes = 
			pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;
			
		if (pFile->pCPInfo->uiStartWaitTruncateTime)
		{
			uiCurrTime = FLM_GET_TIMER();

			uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
						pFile->pCPInfo->uiStartWaitTruncateTime);
			pCheckpointInfo->uiWaitTruncateTime = 
				FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
		}
		else
		{
			pCheckpointInfo->uiWaitTruncateTime = 0;
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE flmSetRflSizeThreshold(
	HFDB			hDb,
	FLMUINT		uiSizeThreshold,
	FLMUINT		uiTimeInterval,
	FLMUINT		uiSizeInterval)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FFILE *		pFile = pDb->pFile;
	FLMBOOL		bDbInitialized = FALSE;
	FLMBOOL		bStartedTrans = FALSE;
			
	// Start an update transaction.  Must not already be one going.

	bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
		0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
			
	if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_61)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}
	
	// Set the size threshold and event intervals

	if( uiSizeThreshold == FLM_MAX_UINT)
	{
		uiSizeThreshold = FB2UD( 
			&pFile->ucUncommittedLogHdr [LOG_RFL_DISK_SPACE_THRESHOLD]);
	}
	else
	{
		UD2FBA( (FLMUINT32)uiSizeThreshold,
			&pFile->ucUncommittedLogHdr [LOG_RFL_DISK_SPACE_THRESHOLD]);
	}
	
	if( uiTimeInterval == FLM_MAX_UINT)
	{
		uiTimeInterval = FB2UD( 
			&pFile->ucUncommittedLogHdr [LOG_RFL_LIMIT_TIME_FREQ]);
	}
	else
	{
		UD2FBA( (FLMUINT32)uiTimeInterval,
			&pFile->ucUncommittedLogHdr [LOG_RFL_LIMIT_TIME_FREQ]);
	}
	
	if( uiSizeInterval == FLM_MAX_UINT)
	{
		uiSizeInterval = FB2UD( 
			&pFile->ucUncommittedLogHdr [LOG_RFL_LIMIT_SPACE_FREQ]);
	}
	else
	{
		UD2FBA( (FLMUINT32)uiSizeInterval,
			&pFile->ucUncommittedLogHdr [LOG_RFL_LIMIT_SPACE_FREQ]);
	}
		
	// Log the change to the RFL

	if( RC_BAD( rc = pFile->pRfl->logSizeEventConfig( 
		pDb->LogHdr.uiCurrTransID, uiSizeThreshold, uiTimeInterval,
		uiSizeInterval)))
	{
		goto Exit;
	}

	// Commit the transaction.

	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
	{
		goto Exit;
	}
	
	bStartedTrans = FALSE;
	
Exit:

	if( bStartedTrans)
	{
		flmAbortDbTrans( pDb);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}
	
	return( rc);
}
