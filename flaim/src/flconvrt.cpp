//-------------------------------------------------------------------------
// Desc:	Database upgrade.
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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
Desc:	Upgrades a database to the latest FLAIM version.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbUpgrade(
	HFDB				hDb,		
	FLMUINT			uiNewVersion,
	STATUS_HOOK		fnStatusCallback,
	void *			UserData)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bStartedTrans = FALSE;
	FLMBOOL		bWroteVersion = FALSE;
	FLMBOOL		bLockedDatabase = FALSE;
	FLMBOOL		bInitedDb = FALSE;
	FLMUINT		uiOldVersion = 0;
	FFILE *		pFile = pDb->pFile;
	F_Rfl *		pRfl = pFile->pRfl;
	FLMBYTE *	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];
	FLMUINT		uiRflFileNum = 0;
	FLMUINT		uiAddr;
	FLMUINT		uiMaxFileSize;
	FLMUINT16	ui16Tmp;
	FLMBOOL		bExpandingFileCount = FALSE;
	FLMBOOL		bLoggingWasOff = FALSE;
	FLMBOOL		bRestoreLoggingOffFlag = FALSE;
	FLMUINT		uiSaveTransId;
	FLMBYTE *	pucWrappingKey = NULL;
	FLMUINT32	ui32KeyLen = 0;

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// Lock the database if not already locked.
	// Cannot lose exclusive access between the checkpoint and
	// the update transaction that does the conversion.

	if( (pDb->uiFlags & FDB_HAS_FILE_LOCK) == 0)
	{
		if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bLockedDatabase = TRUE;
	}

	// Cannot have any transaction already going.

	if( pDb->uiTransType != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// NOTE: Don't get the current version number until AFTER obtaining
	// the exclusive lock - to make sure nobody else can or will do
	// an upgrade while we are in here.

	uiOldVersion = pFile->FileHdr.uiVersionNum;

	switch (uiOldVersion)
	{
		case FLM_FILE_FORMAT_VER_3_0:
		case FLM_FILE_FORMAT_VER_3_02:
		case FLM_FILE_FORMAT_VER_4_0:
		case FLM_FILE_FORMAT_VER_4_3:
		case FLM_FILE_FORMAT_VER_4_31:
		case FLM_FILE_FORMAT_VER_4_50:
		case FLM_FILE_FORMAT_VER_4_51:
		case FLM_FILE_FORMAT_VER_4_60:

			// Upgrades from these versions are supported.

			break;
		default:
			rc = RC_SET( FERR_UNALLOWED_UPGRADE);
			goto Exit;
	}

	switch (uiNewVersion)
	{
		case FLM_FILE_FORMAT_VER_4_3:
		case FLM_FILE_FORMAT_VER_4_31:
		case FLM_FILE_FORMAT_VER_4_50:
		case FLM_FILE_FORMAT_VER_4_51:
		case FLM_FILE_FORMAT_VER_4_60:
		case FLM_CUR_FILE_FORMAT_VER_NUM:
		{
			// Verify that we can do the upgrade

			if (uiNewVersion < uiOldVersion)
			{
				rc = RC_SET( FERR_UNALLOWED_UPGRADE);
				goto Exit;
			}
			else if (uiNewVersion == uiOldVersion)
			{
				// No need to do upgrade - already there.

				goto Exit;
			}
			
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_UNALLOWED_UPGRADE);
			goto Exit;
		}
	}

	// Save the state of RFL logging flag.

	bLoggingWasOff = pRfl->loggingIsOff();

	// Change state of logging OFF to TRUE - don't want anything
	// logged during conversion except for the upgrade packet.

	pRfl->setLoggingOffState( TRUE);
	bRestoreLoggingOffFlag = TRUE;
	pDb->uiFlags |= FDB_UPGRADING;

	uiSaveTransId =
		(FLMUINT)FB2UD( &pDb->pFile->ucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);

	// Flush everything do disk so that the roll forward log is empty.
	// The upgrade doesn't put any special data in the roll forward log
	// so if the roll forward log had stuff in it, it would roll forward
	// on data that was a newer version - and never work!
	// Start an update transaction and commit it, forcing it to be
	// checkpointed.

	bInitedDb = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
									  0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS,
									  &bStartedTrans)))
	{
		goto Exit;
	}

	// Don't want this transaction to change the transaction ID because
	// we are only trying to force a checkpoint.  We don't want to change
	// the transaction ID until we have actually done the convert.

	UD2FBA( (FLMUINT32)uiSaveTransId,
				&pDb->pFile->ucUncommittedLogHdr [LOG_CURR_TRANS_ID]);
	pDb->LogHdr.uiCurrTransID = uiSaveTransId;

	// Set up things in the FDB to indicate where we should move the
	// checkpoint file number and offset to.  If we are in the middle
	// of a recovery or restore operation, move the pointer forward
	// to just BEFORE the upgrade packet.  Down below when we do the
	// checkpoint at the end of the upgrade, we will move the pointer
	// forward to just AFTER the upgrade packet.

	if (pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		pDb->uiUpgradeCPFileNum = pRfl->getCurrFileNum();
		pDb->uiUpgradeCPOffset = pRfl->getCurrPacketAddress();

	}

	// Commit the transaction, forcing it to be checkpointed.

	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
	{
		goto Exit;
	}
	bStartedTrans = FALSE;

	// Start an update transaction for the conversion.

	if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS,
										FLM_NO_TIMEOUT, 0)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	// Make sure that commit does something.

	pDb->bHadUpdOper = TRUE;

	// If version is prior to 4.0, upgrade the non-leaf blocks in
	// container B-Trees.

	if (uiOldVersion < FLM_FILE_FORMAT_VER_4_0 && 
		 uiNewVersion >= FLM_FILE_FORMAT_VER_4_0)
	{

		// Upgrade non-leaf blocks in container B-Trees.

		if (RC_BAD( rc = FSVersionConversion40( pDb, 
				FLM_CUR_FILE_FORMAT_VER_NUM, fnStatusCallback, UserData)))
		{
			goto Exit;
		}
	}

	// If versions is pre-4.3, upgrade log header and RFL stuff.

	if (uiOldVersion < FLM_FILE_FORMAT_VER_4_3 && 
		 uiNewVersion >= FLM_FILE_FORMAT_VER_4_3)
	{

		// Initialize backup options

		UD2FBA( 0, &pucUncommittedLogHdr [LOG_LAST_BACKUP_TRANS_ID]);
		UD2FBA( 0, &pucUncommittedLogHdr [LOG_BLK_CHG_SINCE_BACKUP]);
		UD2FBA( 1, &pucUncommittedLogHdr [LOG_INC_BACKUP_SEQ_NUM]);

		// Initialize unused parts to zero.

		UW2FBA( 0, &pucUncommittedLogHdr [LOG_NU_152_153]);

		// Set maximum RFL file size

		UD2FBA( DEFAULT_MAX_RFL_FILE_SIZE,
				&pucUncommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);

		// Set the database serial number

		if (RC_BAD( rc = f_createSerialNumber( 
									&pucUncommittedLogHdr [LOG_DB_SERIAL_NUM])))
		{
			goto Exit;
		}

		// Set the RFL serial number

		if (RC_BAD( rc = f_createSerialNumber( 
							&pucUncommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM])))
		{
			goto Exit;
		}

		// Set the "next" RFL serial number

		if (RC_BAD( rc = f_createSerialNumber( 
									&pucUncommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM])))
		{
			goto Exit;
		}

		// Set the incremental backup serial number

		if (RC_BAD( rc = f_createSerialNumber( 
								&pucUncommittedLogHdr [LOG_INC_BACKUP_SERIAL_NUM])))
		{
			goto Exit;
		}

		// At this point, the last checkpoint offset and file number
		// should be the same as the last transaction offset and
		// file number.

		flmAssert( FB2UD( &pucUncommittedLogHdr [LOG_RFL_FILE_NUM]) ==
					  FB2UD( &pucUncommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]));
		flmAssert( FB2UD( &pucUncommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]) ==
					  FB2UD( &pucUncommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]));

		// Set the transaction offset to zero so that we will be forced to
		// write the serial numbers into the RFL file on the next transaction
		// begin operation.

		UD2FBA( 0, &pucUncommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);

		// Keep log files should be FALSE at this point - pre 4.3
		// versions did not keep RFL files.  Also, we should still
		// be on log file #1.  However, if this is not the case,
		// we need to deal with it and set up to roll to the
		// next log file. - We don't want to lose the current
		// RFL file, but we need to have the serial numbers written
		// out, so our only option is to go to the next one.

		uiRflFileNum = FB2UD( &pucUncommittedLogHdr [LOG_RFL_FILE_NUM]);
		if (pucUncommittedLogHdr [LOG_KEEP_RFL_FILES])
		{
			flmIncrUint( &pucUncommittedLogHdr [LOG_RFL_FILE_NUM], 1);
		}
		else
		{

			// Checkpoint offset better already be 512 if we are not keeping
			// RFL files - due to the checkpoint we executed above.

			flmAssert(
				FB2UD( &pucUncommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]) == 512);
		}

		// Set the maximum file size.  If we currently have more than
		// one data file or rollback file, we must use the old file size
		// limit.

		uiMaxFileSize = gv_FlmSysData.uiMaxFileSize;

		// A file number greater than one in the logical EOF means
		// we have multiple data files.

		uiAddr = (FLMUINT)FB2UD( &pucUncommittedLogHdr [LOG_LOGICAL_EOF]);
		if (FSGetFileNumber( uiAddr) > 1)
		{
			uiMaxFileSize = MAX_FILE_SIZE_VER40;
		}

		ui16Tmp = (FLMUINT16)(uiMaxFileSize >> 16);
		flmAssert( ui16Tmp);
		UW2FBA( ui16Tmp, &pucUncommittedLogHdr [LOG_MAX_FILE_SIZE]);
		bExpandingFileCount = TRUE;
	}

	if (uiOldVersion < FLM_FILE_FORMAT_VER_4_31 && 
		 uiNewVersion >= FLM_FILE_FORMAT_VER_4_31)
	{

		// NOTE: We could really set the LOG_LAST_RFL_COMMIT_ID to anything,
		// because the new transaction begin packet will not be logged until
		// after we do this conversion, and it will log whatever we put
		// in here.

		UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
			&pucUncommittedLogHdr [LOG_LAST_RFL_COMMIT_ID]);
	}

#ifdef FLM_USE_NICI

	if (uiOldVersion < FLM_FILE_FORMAT_VER_4_60 && 
		 uiNewVersion >= FLM_FILE_FORMAT_VER_4_60)
	{

		if( RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
		{
			goto Exit;
		}
		bStartedTrans = FALSE;

		if (RC_BAD( rc = FlmEnableEncryption( hDb, &pucWrappingKey, &ui32KeyLen)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS,
											FLM_NO_TIMEOUT, 0)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

#endif

	// NOTE: THIS TEST SHOULD BE DONE ONLY AFTER ALL CHANGES THAT COULD
	// CAUSE BLOCKS TO BE LOGGED TO THE ROLLBACK LOG HAVE BEEN DONE.
	// IT SHOULD ALSO BE DONE BEFORE WE CHANGE THE VERSION NUMBER ON
	// THE DATABASE.

	if (bExpandingFileCount)
	{

		// Force any keys to be committed so that all blocks that
		// are going to be modified will have been modified by this
		// point.

		if (RC_BAD( rc = KYKeysCommit( pDb, FALSE)))
		{
			goto Exit;
		}

		// If this conversion requires a change in the total number
		// of files then the file number scheme has changed.  However,
		// we cannot do it if we have more than a single rollback file,
		// because the new file numbering scheme would be wrong - it
		// would not support accessing the additional rollback file
		// correctly.
		// NOTE: Could have multiple rollback files if some aspect
		// of the conversion that occurred prior to this caused us to
		// have multiple rollback files.

		uiAddr = (FLMUINT)FB2UD( &pucUncommittedLogHdr [LOG_ROLLBACK_EOF]);
		if (FSGetFileNumber( uiAddr))
		{
			rc = RC_SET( FERR_CANNOT_CONVERT);
			goto Exit;
		}
	}

	// NOTE: By this point, all conversions should be complete, except for
	// committing and changing the version number.

	// Log the upgrade packet to the RFL if we are not in the middle of a
	// restore or recovery.  Will need to re-enable logging temporarily
	// and then turn it back off after logging the packet.
	// NOTE: We can only do this if converting from 4.30 or greater.
	// Makes no sense before that because prior to 4.30 there was no
	// possibility of keeping RFL files and doing a restore from them.

	if (!(pDb->uiFlags & FDB_REPLAYING_RFL) && 
		 uiOldVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		// We would have turned logging OFF above, so we need to
		// turn it back on here.

		pRfl->setLoggingOffState( FALSE);

		// Log the upgrade packet.

		rc = pRfl->logUpgrade( pDb->LogHdr.uiCurrTransID, uiOldVersion,
									  pucWrappingKey, ui32KeyLen);

		// Turn logging back off.

		pRfl->setLoggingOffState( TRUE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// Change the FLAIM version number to the new version number.

	pFile->FileHdr.uiVersionNum = uiNewVersion;
	UW2FBA( (FLMUINT16)uiNewVersion, &pucUncommittedLogHdr [LOG_FLAIM_VERSION]);

	// Change the FLAIM version number on disk

	if( RC_BAD( rc = flmWriteVersionNum( pDb->pSFileHdl, uiNewVersion)))
	{
		goto Exit;
	}
	bWroteVersion = TRUE;

	// Commit and force a checkpoint by passing TRUE.
	// Set up things in the FDB to indicate where we should move the
	// checkpoint file number and offset to.  If we are in the middle
	// of a recovery or restore operation, move the pointer forward
	// to just AFTER the upgrade packet.

	if (pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		pDb->uiUpgradeCPFileNum = pRfl->getCurrFileNum();
		pDb->uiUpgradeCPOffset = pRfl->getCurrReadOffset();
	}
	if( RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
	{
		goto Exit;
	}
	bStartedTrans = FALSE;
		
	// Set up to use a new RFL directory.  Must do this only after
	// setting FileHdr.uiVersionNum above.

	if (uiOldVersion < FLM_FILE_FORMAT_VER_4_3 &&
		 uiNewVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		char	szTmpName [F_PATH_MAX_SIZE];

		// Close the current RFL file.

		pRfl->closeFile();

		// At this point, there should be no RFL directory.

		flmAssert( pRfl->isRflDirSameAsDbDir());
		(void)pRfl->setRflDir( NULL);

		// Attempt to delete the old RFL file - should only be
		// one and it should be file #1 - we could assert that
		// uiRflFileNum == 1, but we will be more lenient.

		if (RC_OK( rc = rflGetFileName( uiOldVersion, pFile->pszDbPath,
									NULL, uiRflFileNum, szTmpName)))
		{
			gv_FlmSysData.pFileSystem->deleteFile( szTmpName);
		}
	}

Exit:

	if (bStartedTrans)
	{

		// Failure condition, we jumped to exit

		UW2FBA( (FLMUINT16)uiOldVersion,
			&pucUncommittedLogHdr [LOG_FLAIM_VERSION]);
		pFile->FileHdr.uiVersionNum = uiOldVersion;

		// Change the FLAIM version number on disk to the original.

		if (bWroteVersion)
		{

			// Need to initialize the database to do the following write.  
			// We don't care about the transaction, which will be aborted
			// a couple of lines down.

			(void) fdbInit( (FDB *)hDb, FLM_UPDATE_TRANS,
				0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS, &bStartedTrans);
			(void) flmWriteVersionNum( pDb->pSFileHdl, uiOldVersion);
			(void) fdbExit( pDb);
		}
		(void) flmAbortDbTrans( pDb);
	}
	if (bInitedDb)
	{
		flmExit( FLM_DB_UPGRADE, pDb, rc);
	}

	if (bRestoreLoggingOffFlag)
	{
		pRfl->setLoggingOffState( bLoggingWasOff);
	}

	// Turn off the upgrade flag, in case it was turned on above.

	pDb->uiFlags &= (~(FDB_UPGRADING));

	if (bLockedDatabase)
	{
		(void) FlmDbUnlock( hDb);
	}

	if (pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}

	return( rc );
}

/****************************************************************************
Desc : Adds an encryption key to the database.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmEnableEncryption(
	HFDB				hDb,
	FLMBYTE **		ppucWrappingKeyRV,
	FLMUINT32 *		pui32KeyLen)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FFILE *		pFile = pDb->pFile;
	F_Rfl *		pRfl = pFile->pRfl;
	FLMBYTE *	pucWrappingKey = NULL;
	FLMUINT32	ui32KeyLen = 0;
	FLMBYTE *	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];
	FLMUINT		uiFlags = FLM_GET_TRANS_FLAGS( FLM_UPDATE_TRANS);
	FLMBOOL		bTransBegun = FALSE;

	// We must start our own transaction.  Then we will force a checkpoint
	// when we commit the transaction

	if ( pDb->uiTransType != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Begin an update transaction.

	if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS,
		FLM_NO_TIMEOUT, uiFlags)))
	{
		goto Exit;
	}

	bTransBegun = TRUE;
	
	// If we don't have a wrapping key, then create one.  Normally
	// this would be the case, since we are enabling encryption,
	// but the test is "just to be sure" we don't
	// overwrite an existing key.

	if (!pFile->pDbWrappingKey)
	{
		if ((pFile->pDbWrappingKey = f_new F_CCS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	
		if( RC_BAD( rc = pFile->pDbWrappingKey->init( TRUE, FLM_NICI_AES)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pFile->pDbWrappingKey->generateWrappingKey()))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pFile->pDbWrappingKey->getKeyToStore( &pucWrappingKey,
		&ui32KeyLen, pFile->pszDbPassword, NULL, FALSE)))
	{
		goto Exit;
	}
	
	f_memcpy( &pucUncommittedLogHdr[ LOG_DATABASE_KEY], pucWrappingKey, ui32KeyLen);
	UW2FBA( ui32KeyLen, &pucUncommittedLogHdr[ LOG_DATABASE_KEY_LEN]);
	
	pFile->rcLimitedCode = FERR_OK;
	pFile->bInLimitedMode = FALSE;
	pFile->bHaveEncKey = TRUE;
	
	// Log the upgrade packet.  NOTE that if this is part of a standard DB
	// upgrade this packet will not be logged.  The upgrade will be logged by
	// the FlmDbUpgrade function.  No need to log it twice.

	if( RC_BAD( rc = pRfl->logEnableEncryption( pDb->LogHdr.uiCurrTransID,
		pucWrappingKey, ui32KeyLen)))
	{
		goto Exit;
	}

	// Commit the transaction - force a checkpoint!

	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE, NULL)))
	{
		goto Exit;
	}

	bTransBegun = FALSE;

	if( ppucWrappingKeyRV)
	{
		// It is now the responsibility of the caller to
		// free this buffer!!
		
		*ppucWrappingKeyRV = pucWrappingKey;
		pucWrappingKey = NULL;
	}
	
	if( pui32KeyLen)
	{
		*pui32KeyLen = ui32KeyLen;
	}

Exit:

	if( bTransBegun)
	{
		RCODE		rcTmp = FERR_OK;

		rcTmp = flmAbortDbTrans( pDb);
		if (RC_OK( rc))
		{
			rc = rcTmp;
		}
	}

	if( pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}

	return( rc);
}

/****************************************************************************
Desc:	Changes the way the database key is stored in the database.  Either
		it is encrypted using a password or it is wrapped in the server key.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbWrapKey(
	HFDB					hDb,
	const char *		pszPassword)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FFILE *		pFile = pDb->pFile;
	F_Rfl *		pRfl = pFile->pRfl;
	FLMBYTE *	pucWrappingKey = NULL;
	FLMUINT32	ui32KeyLen = 0;
	FLMBYTE *	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];
	FLMUINT		uiFlags = FLM_GET_TRANS_FLAGS( FLM_UPDATE_TRANS);
	FLMBOOL		bTransBegun = FALSE;
	FLMBOOL		bLoggingIsOff = pRfl->loggingIsOff();
	FLMBOOL		bLockedDatabase = FALSE;

	// Lock the database if not already locked.
	// Cannot lose exclusive access between the checkpoint and
	// the update transaction that does the conversion.

	if( (pDb->uiFlags & FDB_HAS_FILE_LOCK) == 0)
	{
		if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bLockedDatabase = TRUE;
	}

	// Turn off logging.  We only want to log the wrap key packet.

	pRfl->setLoggingOffState( TRUE);

	// We must will start our own transaction.  Then we will force a checkpoint
	// when we commit the transaction

	if ( pDb->uiTransType != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Begin an update transaction.

	if( RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS,
		FLM_NO_TIMEOUT, uiFlags)))
	{
		goto Exit;
	}
	pDb->bHadUpdOper = TRUE;

	bTransBegun = TRUE;

	// The wrapping key MUST exist!

	flmAssert( pFile->bHaveEncKey);

	if (!pFile->pDbWrappingKey)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( RC_BAD( rc = pFile->pDbWrappingKey->getKeyToStore( &pucWrappingKey,
			&ui32KeyLen, pszPassword, NULL, FALSE)))
	{
		goto Exit;
	}
	
	f_memcpy( &pucUncommittedLogHdr[ LOG_DATABASE_KEY], pucWrappingKey, ui32KeyLen);
	UW2FBA( ui32KeyLen, &pucUncommittedLogHdr[ LOG_DATABASE_KEY_LEN]);
	
	// Turn on logging.  We only want to log the wrap key packet.

	pRfl->setLoggingOffState( FALSE);

	// Log a wrapped key packet to record that the key has been wrapped/encrypted.

	if( RC_BAD( rc = pRfl->logWrappedKey( pDb->LogHdr.uiCurrTransID,
		pucWrappingKey, ui32KeyLen)))
	{
		goto Exit;
	}

	// Turn off logging.  We only want to log the wrap key packet.

	pRfl->setLoggingOffState( TRUE);

	// Commit the transaction - force a checkpoint!

	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE, NULL)))
	{
		goto Exit;
	}

	bTransBegun = FALSE;

	// Delete the old password
	
	if (pFile->pszDbPassword)
	{
		f_free( &pFile->pszDbPassword);
	}

	// Store the new password
	
	if ( pszPassword)
	{
		if (RC_BAD( rc = f_calloc( f_strlen( pszPassword) + 1, 
			&pFile->pszDbPassword)))
		{
			goto Exit;
		}
		
		f_memcpy( pFile->pszDbPassword, pszPassword, f_strlen( pszPassword));
	}

Exit:

	if( bTransBegun)
	{
		RCODE		rcTmp = FERR_OK;

		rcTmp = flmAbortDbTrans( pDb);
		if (RC_OK( rc))
		{
			rc = rcTmp;
		}
	}

	// Restore logging to its original state

	pRfl->setLoggingOffState( bLoggingIsOff);

	if( bLockedDatabase)
	{
		(void) FlmDbUnlock( hDb);
	}

	if( pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}

	return( rc);
}
