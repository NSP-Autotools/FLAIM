//-------------------------------------------------------------------------
// Desc:	Backup and restore routines.
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

typedef struct
{
	char						szPath[ F_PATH_MAX_SIZE];
	IF_MultiFileHdl *		pMultiFileHdl;
	FLMUINT64				ui64Offset;
	void *					pvAppData;
	RCODE						rc;
} BACKER_HOOK_STATE;

#define FLM_BACKER_SIGNATURE_OFFSET			0
#define		FLM_BACKER_SIGNATURE				"!DB_BACKUP_FILE!"
#define		FLM_BACKER_SIGNATURE_SIZE		16
#define FLM_BACKER_VERSION_OFFSET			16
#define		FLM_BACKER_VERSION_1_0_1		101
#define		FLM_BACKER_VERSION				FLM_BACKER_VERSION_1_0_1
#define FLM_BACKER_DB_BLOCK_SIZE_OFFSET	20
#define FLM_BACKER_BFMAX_OFFSET				24
#define FLM_BACKER_MTU_OFFSET					28
#define FLM_BACKER_TIME_OFFSET				32
#define FLM_BACKER_DB_NAME_OFFSET			36
#define FLM_BACKER_BACKUP_TYPE_OFFSET		40
#define FLM_BACKER_NEXT_INC_SERIAL_NUM		44
#define FLM_BACKER_DB_VERSION					60

// The backer MTU size must be a multiple of the largest
// supported block size.  Additionally, it must be at least
// 2 * FLM_BACKER_MAX_DB_BLOCK_SIZE.  DO NOT CHANGE THE MTU
// SIZE UNLESS YOU ALSO BUMP THE BACKUP VERSION.

#define FLM_BACKER_MTU_SIZE				((FLMUINT) 1024 * 512) 
#define FLM_BACKER_MAX_FILE_SIZE			((FLMUINT) 1024 * 1024 * 1024 * 2) // 2 Gigabytes
#define FLM_BACKER_MIN_DB_BLOCK_SIZE	((FLMUINT) 2 * 1024)
#define FLM_BACKER_MAX_DB_BLOCK_SIZE	((FLMUINT) 16 * 1024)

// Backup block header

#define FLM_BACKER_BLK_HDR_SIZE			((FLMUINT) 8)
#define FLM_BACKER_BLK_ADDR_OFFSET		0
#define FLM_BACKER_BLK_SIZE_OFFSET		4

FSTATIC RCODE flmDefaultBackerWriteHook(
	void *					pvBuffer,
	FLMUINT					uiBytesToWrite,
	void *					pvCallbackData);

FSTATIC RCODE flmRestoreFile(
	F_Restore *				pRestoreObj,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszPassword,
	F_SuperFileHdl **		ppSFile,
	FLMBOOL					bIncremental,
	FLMUINT *				puiDbVersion,
	FLMUINT *				puiNextIncSeqNum,
	FLMBOOL *				pbRflPreserved,
	eRestoreActionType *	peRestoreAction,
	FLMBOOL *				pbOKToRetry,
	FLMBYTE **				ppucKeyToSave,
	FLMBYTE *				pucKeyToUse,
	FLMUINT *				puiKeyLen);

/*******************************************************************************
Desc:
*******************************************************************************/
class	F_BackerStream : public F_Object
{
public:

	F_BackerStream( void);
	~F_BackerStream( void);

	RCODE setup(
		FLMUINT				uiMTUSize,
		F_Restore *			pRestoreObj);

	RCODE setup(
		FLMUINT				uiMTUSize,
		BACKER_WRITE_HOOK	fnWrite,
		void *				pvCallbackData);

	RCODE startThreads( void);

	void shutdownThreads( void);

	RCODE read(
		FLMUINT				uiLength,
		FLMBYTE *			pucData,
		FLMUINT *			puiBytesRead = NULL);

	RCODE write(
		FLMUINT				uiLength,
		FLMBYTE *			pucData,
		FLMUINT *			puiBytesWritten = NULL);

	RCODE flush( void);

	FINLINE FLMUINT64 getByteCount( void)
	{
		// Returns the total number of bytes read or written.

		return( m_ui64ByteCount);
	}

	FINLINE FLMUINT getMTUSize( void)
	{
		return( m_uiMTUSize);
	}

private:

	RCODE signalThread( void);

	RCODE _setup( void);

	static RCODE FLMAPI readThread(
		IF_Thread *			pThread);

	static RCODE FLMAPI writeThread(
		IF_Thread *			pThread);

	FLMBOOL					m_bSetup;
	FLMBOOL					m_bFirstRead;
	FLMUINT					m_uiBufOffset;
	FLMUINT64				m_ui64ByteCount;
	F_Restore *				m_pRestoreObj;
	F_SEM						m_hDataSem;
	F_SEM						m_hIdleSem;
	IF_Thread *				m_pThread;
	RCODE						m_rc;
	FLMBYTE *				m_pucInBuf;
	FLMUINT *				m_puiInOffset;
	FLMBYTE *				m_pucOutBuf;
	FLMUINT *				m_puiOutOffset;
	FLMBYTE *				m_pucBufs[ 2];
	FLMUINT					m_uiOffsets[ 2];
	FLMUINT					m_uiMTUSize;
	FLMUINT					m_uiPendingIO;
	BACKER_WRITE_HOOK		m_fnWrite;
	void *					m_pvCallbackData;
};

/*******************************************************************************
Desc:		Prepares FLAIM to backup a database.
Notes:	Only one backup of a particular database can be active at any time
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmDbBackupBegin(
	HFDB			hDb,
	FBackupType	eBackupType,
	FLMBOOL		bHotBackup,
	HFBACKUP *	phBackup
	)
{
	FDB *			pDb = (FDB *)hDb;
	FBak *		pFBak = NULL;
	FLMBOOL		bBackupFlagSet = FALSE;
	FLMUINT		uiLastCPFileNum; 
	FLMUINT		uiLastTransFileNum;
	FLMUINT		uiDbVersion;
	FLMUINT		uiTmp;
	FLMBYTE *	pLogHdr;
	RCODE			rc = FERR_OK;
	FLMUINT		uiTransType = bHotBackup ? FLM_READ_TRANS : FLM_UPDATE_TRANS;

	// Initialize the handle

	*phBackup = HFBACKUP_NULL;

	// Make sure we are not being called inside a transaction

	if( RC_BAD( rc = FlmDbGetTransType( hDb, &uiTmp)))
	{
		goto Exit;
	}

	if( uiTmp != FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure a valid backup type has been specified

	if( RC_BAD( rc = FlmDbGetConfig( hDb, 
		FDB_GET_VERSION, (FLMUINT *)&uiDbVersion)))
	{
		goto Exit;
	}

	if( uiDbVersion < FLM_FILE_FORMAT_VER_4_3 &&
		eBackupType != FLM_FULL_BACKUP)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// See if a backup is currently running against the database.  If so,
	// return an error.  Otherwise, set the backup flag on the FFILE.

	if( !IsInCSMode( hDb))
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		if( pDb->pFile->bBackupActive)
		{
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			rc = RC_SET( FERR_BACKUP_ACTIVE);
			goto Exit;
		}
		else
		{
			bBackupFlagSet = TRUE;
			pDb->pFile->bBackupActive = TRUE;
		}
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
	else
	{
		if( RC_BAD( rc = fcsSetBackupActiveFlag( hDb, TRUE)))
		{
			goto Exit;
		}
		bBackupFlagSet = TRUE;
	}

	// Allocate the backup handle

	if( RC_BAD( rc = f_calloc( sizeof( FBak), &pFBak)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_allocAlignedBuffer( 2048, &pFBak->pucDbHeader)))
	{
		goto Exit;
	}

	pFBak->hDb = hDb;
	pFBak->uiDbVersion = uiDbVersion;

	// Set the C/S mode flag
	
	pFBak->bCSMode = IsInCSMode( hDb);

	// Start a transaction

	if( RC_BAD( rc = FlmDbTransBegin( hDb, 
		uiTransType | FLM_DONT_KILL_TRANS | FLM_DONT_POISON_CACHE,
		FLM_NO_TIMEOUT, pFBak->pucDbHeader)))
	{
		goto Exit;
	}

	pFBak->bTransStarted = TRUE;
	pFBak->uiTransType = uiTransType;
	pLogHdr = &pFBak->pucDbHeader[ DB_LOG_HEADER_START];

	// Don't allow an incremental backup to be performed
	// if a full backup has not yet been done.

	if( eBackupType == FLM_INCREMENTAL_BACKUP &&
		 FB2UD( &pLogHdr[ LOG_LAST_BACKUP_TRANS_ID]) == 0)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	pFBak->eBackupType = eBackupType;

	// Set the next incremental backup serial number.  This is
	// done regardless of the backup type to prevent the wrong
	// set of incremental backup files from being applied
	// to a database.

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		if( !pFBak->bCSMode)
		{
			if( RC_BAD( rc = f_createSerialNumber( 
				pFBak->ucNextIncSerialNum)))
			{
				goto Exit;
			}
		}
		else
		{
			fdbInitCS( pDb);
			rc = fcsCreateSerialNumber( 
				pDb->pCSContext, pFBak->ucNextIncSerialNum);
			fdbExit( pDb);

			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}

	// Get the incremental sequence number from the log header
	
	pFBak->uiIncSeqNum = FB2UD( &pLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);

	// Get version 4.3+ values from the header

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		// Determine the transaction ID of the last backup

		pFBak->uiLastBackupTransId =
			FB2UD( &pLogHdr[ LOG_LAST_BACKUP_TRANS_ID]);

		// Get the block change count

		pFBak->uiBlkChgSinceLastBackup = 
			FB2UD( &pLogHdr[ LOG_BLK_CHG_SINCE_BACKUP]);
	}

	// Get the current transaction ID

	if( RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_TRANS_ID,
		(void *)&pFBak->uiTransId)))
	{
		goto Exit;
	}

	// Get the logical end of file

	pFBak->uiLogicalEOF = FB2UD( &pLogHdr[ LOG_LOGICAL_EOF]);

	// Get the first required RFL file needed by the restore.

	uiLastCPFileNum = FB2UD( &pLogHdr[ LOG_RFL_LAST_CP_FILE_NUM]);
	uiLastTransFileNum = FB2UD( &pLogHdr[ LOG_RFL_FILE_NUM]);

	flmAssert( uiLastCPFileNum <= uiLastTransFileNum);

	pFBak->uiFirstReqRfl = uiLastCPFileNum < uiLastTransFileNum
								? uiLastCPFileNum
								: uiLastTransFileNum;

	flmAssert( pFBak->uiFirstReqRfl);

	// Get the database block size

	if( RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_BLKSIZ, 
		&pFBak->uiBlockSize)))
	{
		goto Exit;
	}

	// Get the database path

	if( RC_BAD( rc = FlmDbGetConfig( hDb, 
		FDB_GET_PATH, pFBak->ucDbPath)))
	{
		goto Exit;
	}

	*phBackup = pFBak;

Exit:

	if( RC_BAD( rc))
	{
		if( pFBak)
		{
			if( pFBak->bTransStarted)
			{
				(void)FlmDbTransAbort( hDb);
			}

			if( pFBak->pucDbHeader)
			{
				f_freeAlignedBuffer( &pFBak->pucDbHeader);
			}

			f_free( &pFBak);
		}

		if( bBackupFlagSet)
		{
			if( !IsInCSMode( hDb))
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);
				pDb->pFile->bBackupActive = FALSE;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}
			else
			{
				(void)fcsSetBackupActiveFlag( hDb, FALSE);
			}
		}
	}

	return( rc);
}

/****************************************************************************
Desc : Returns information about a backup
****************************************************************************/
FLMEXP RCODE FLMAPI FlmBackupGetConfig(
	HFBACKUP					hBackup,
	eBackupGetConfigType	eConfigType,
	void *					pvValue1,
	void *					// pvValue2
	)
{
	RCODE			rc = FERR_OK;
	FBak *		pFBak = (FBak *)hBackup;

	switch( eConfigType)
	{
		case FBAK_GET_BACKUP_TRANS_ID:
		{
			*((FLMUINT *)pvValue1) = pFBak->uiTransId;
			break;
		}

		case FBAK_GET_LAST_BACKUP_TRANS_ID:
		{
			*((FLMUINT *)pvValue1) = pFBak->uiLastBackupTransId;
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc : Streams the contents of a database to the write hook supplied by
		 the application.
Notes: This routine attempts to create a backup of a database without
		 excluding any readers or updaters.  However, if the backup runs
		 too long in an environment where extensive updates are happening,
		 an old view error could be returned.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbBackup(
	HFBACKUP					hBackup,
	const char *			pszBackupPath,
	const char *			pszPassword,
	BACKER_WRITE_HOOK		fnWrite,
	STATUS_HOOK				fnStatus,
	void *					pvAppData,
	FLMUINT *				puiIncSeqNum)
{
	FDB *						pDb = NULL;
	FLMBOOL					bDbInitialized = FALSE;
	FLMBOOL					bFullBackup = TRUE;
	FLMINT					iFileNum;
	FLMUINT					uiBlkAddr;
	FLMUINT					uiTime;
	SCACHE *					pSCache = NULL;
	BACKER_HOOK_STATE		hookState;
	FLMBYTE *				pLogHdr;
	DB_BACKUP_INFO			backupInfo;
	FLMUINT					uiBlockFileOffset;
	FLMUINT					uiActualBlkSize;
	FLMUINT					uiCount;
	FLMUINT					uiBlockCount;
	FLMUINT					uiBlockCountLastCB = 0;
	FLMUINT					uiBytesToPad;
	void *					pvCallbackData;
	FBak *					pFBak = (FBak *)hBackup;
	FLMUINT					uiBlockSize = pFBak->uiBlockSize;
	F_BackerStream *		pBackerStream = NULL;
	FLMBYTE *				pucBlkBuf = NULL;
	FLMUINT					uiBlkBufOffset;
	FLMUINT					uiBlkBufSize;
	FLMUINT					uiMaxCSBlocks;
	FLMUINT					uiCPTransOffset;
	FLMUINT					uiMaxFileSize;
	RCODE						rc = FERR_OK;
	F_CCS *					pDbKey;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pszPassword);
#endif

	pDb = (FDB *)(pFBak->hDb);
	if( puiIncSeqNum)
	{
		*puiIncSeqNum = 0;
	}

	f_memset( &hookState, 0, sizeof( BACKER_HOOK_STATE));

	// Make sure a backup attempt has not been made with this
	// backup handle.

	if( pFBak->bCompletedBackup)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( pFBak->backupRc))
	{
		rc = pFBak->backupRc;
		goto Exit;
	}

	// Look at the backup type

	if( pFBak->eBackupType == FLM_INCREMENTAL_BACKUP)
	{
		if( puiIncSeqNum)
		{
			*puiIncSeqNum = pFBak->uiIncSeqNum;
		}
		
		bFullBackup = FALSE;
	}


	// Set up the callback

	if( !fnWrite)
	{
		fnWrite = flmDefaultBackerWriteHook;
	}

	if( fnWrite == flmDefaultBackerWriteHook)
	{
		if( !pszBackupPath)
		{
			rc = RC_SET( FERR_INVALID_PARM);
			goto Exit;
		}

		f_strcpy( hookState.szPath, pszBackupPath);
		hookState.pvAppData = pvAppData;
		pvCallbackData = &hookState;
	}
	else
	{
		pvCallbackData = pvAppData;
	}

	// Allocate and initialize the backer stream object

	if( (pBackerStream = f_new F_BackerStream) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBackerStream->setup( FLM_BACKER_MTU_SIZE,
		fnWrite, pvCallbackData)))
	{
		goto Exit;
	}

	// Allocate a temporary buffer

	uiBlkBufSize = FLM_BACKER_MTU_SIZE;
	uiMaxCSBlocks = uiBlkBufSize / uiBlockSize;
	if( RC_BAD( rc = f_alloc( uiBlkBufSize, &pucBlkBuf)))
	{
		goto Exit;
	}

	// Setup the status callback info

	f_memset( &backupInfo, 0, sizeof( DB_BACKUP_INFO));

	// Setup the backup file header

	uiBlkBufOffset = 0;
	f_memset( pucBlkBuf, 0, uiBlockSize);
	f_memcpy( &pucBlkBuf[ FLM_BACKER_SIGNATURE_OFFSET],
		FLM_BACKER_SIGNATURE, FLM_BACKER_SIGNATURE_SIZE);

	UD2FBA( FLM_BACKER_VERSION,
		&pucBlkBuf[ FLM_BACKER_VERSION_OFFSET]);
	UD2FBA( (FLMUINT32)uiBlockSize,
		&pucBlkBuf[ FLM_BACKER_DB_BLOCK_SIZE_OFFSET]);
	uiMaxFileSize = flmGetMaxFileSize( pFBak->uiDbVersion,
								&pFBak->pucDbHeader [DB_LOG_HEADER_START]);
	UD2FBA( (FLMUINT32)uiMaxFileSize,
		&pucBlkBuf[ FLM_BACKER_BFMAX_OFFSET]);
	UD2FBA( (FLMUINT32)FLM_BACKER_MTU_SIZE,
		&pucBlkBuf[ FLM_BACKER_MTU_OFFSET]);
	f_timeGetSeconds( &uiTime);
	UD2FBA( (FLMUINT32)uiTime,
		&pucBlkBuf[ FLM_BACKER_TIME_OFFSET]);

	uiCount = f_strlen( (const char *)pFBak->ucDbPath);
	if( uiCount <= 3)
	{
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET] =
			pFBak->ucDbPath[ uiCount - 6];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 1] =
			pFBak->ucDbPath[ uiCount - 5];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 2] =
			pFBak->ucDbPath[ uiCount - 4];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 3] = '\0';
	}

	UD2FBA( (FLMUINT32)pFBak->eBackupType,
		&pucBlkBuf[ FLM_BACKER_BACKUP_TYPE_OFFSET]);

	// Set the next incremental serial number in the backup's
	// header so that it can be put into the database's log header
	// after the backup has been restored.

	f_memcpy( &pucBlkBuf[ FLM_BACKER_NEXT_INC_SERIAL_NUM],
		pFBak->ucNextIncSerialNum, F_SERIAL_NUM_SIZE);

	// Set the database version number

	UD2FBA( (FLMUINT32)pFBak->uiDbVersion,
		&pucBlkBuf[ FLM_BACKER_DB_VERSION]);

	uiBlkBufOffset += uiBlockSize;

	// Copy the database header into the backup's buffer

	f_memset( &pucBlkBuf[ uiBlkBufOffset], 0, uiBlockSize);
	f_memcpy( &pucBlkBuf[ uiBlkBufOffset],
		pFBak->pucDbHeader, F_TRANS_HEADER_SIZE);
	pLogHdr = &pucBlkBuf[ uiBlkBufOffset + DB_LOG_HEADER_START];
	uiBlkBufOffset += uiBlockSize;

	// Fix up the log header

	if( !pLogHdr[ LOG_KEEP_RFL_FILES] || pFBak->uiDbVersion < FLM_FILE_FORMAT_VER_4_3)
	{
		pLogHdr[ LOG_KEEP_RFL_FILES] = 0;

		// Put zero in as the last transaction offset so that the current
		// RFL file will be created if it does not exist after the database
		// is restored.  This has basically the same effect as setting the
		// offset to 512 if the RFL file has already been created.

		UD2FBA( (FLMUINT32)0, &pLogHdr[ LOG_RFL_LAST_TRANS_OFFSET]);
		uiCPTransOffset = 512;

		// Create new serial numbers for the RFL.  We don't want anyone
		// to be able to branch into a "no-keep" RFL sequence.

		if( pFBak->uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
		{
			if (RC_BAD( rc = f_createSerialNumber(
								&pLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM])))
			{
				goto Exit;
			}

			if (RC_BAD( rc = f_createSerialNumber(
										&pLogHdr [LOG_RFL_NEXT_SERIAL_NUM])))
			{
				goto Exit;
			}
		}
	}
	else
	{
		uiCPTransOffset = FB2UD( &pLogHdr[ LOG_RFL_LAST_TRANS_OFFSET]);
		if( !uiCPTransOffset)
		{
			uiCPTransOffset = 512;
		}
	}

	// Shroud the database key (stored in the log header) in the password
	// so we can restore this backup to a different server

	if ( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_60 &&
		  pszPassword && *pszPassword &&
		  FB2UW( &pLogHdr[ LOG_DATABASE_KEY_LEN]) > 0)
	{
		FLMBYTE *	pucTmpBuf = NULL;
		FLMUINT32	ui32KeyLen = 0;

		pDbKey = pDb->pFile->pDbWrappingKey;
		
		if( RC_BAD( rc = pDbKey->getKeyToStore( &pucTmpBuf, &ui32KeyLen,
			pszPassword, NULL, FALSE)))
		{
			goto Exit;
		}

		// Verify that the field in the log header is long enough to
		// hold the key.
		
		if( ui32KeyLen > FLM_MAX_DB_ENC_KEY_LEN)
		{
			rc = RC_SET_AND_ASSERT( FERR_BAD_ENC_KEY);
			goto Exit;
		}

		// Verify that the field in the log header is long enough to
		// hold the key.
		
		if( ui32KeyLen > FLM_MAX_DB_ENC_KEY_LEN)
		{
			f_free( &pucTmpBuf);
			rc = RC_SET_AND_ASSERT( FERR_BAD_ENC_KEY);
			goto Exit;
		}
		
		UW2FBA( ui32KeyLen, &pLogHdr[ LOG_DATABASE_KEY_LEN]);
		f_memcpy( &pLogHdr[ LOG_DATABASE_KEY], pucTmpBuf, ui32KeyLen);
		f_free( &pucTmpBuf);
	}

	// Set the CP offsets to the last trans offsets.  This is done
	// because the backup could actually read dirty (committed) blocks
	// from the cache, resulting in a backup set that contains blocks
	// that are more recent than the ones currently on disk.

	f_memcpy( &pLogHdr[ LOG_RFL_LAST_CP_FILE_NUM],
		&pLogHdr[ LOG_RFL_FILE_NUM], 4);

	f_memcpy( &pLogHdr[ LOG_LAST_CP_TRANS_ID],
		&pLogHdr[ LOG_CURR_TRANS_ID], 4);

	UD2FBA( (FLMUINT32)uiCPTransOffset, &pLogHdr[ LOG_RFL_LAST_CP_OFFSET]);
	UD2FBA( (FLMUINT32) uiBlockSize, &pLogHdr[ LOG_ROLLBACK_EOF]);
	UD2FBA( 0, &pLogHdr[ LOG_PL_FIRST_CP_BLOCK_ADDR]);

	// Compute the log header checksum

	UW2FBA( (FLMUINT16)lgHdrCheckSum( pLogHdr, FALSE),
		&pLogHdr[ LOG_HDR_CHECKSUM]);

	// Output the header

	if( RC_BAD( rc = pBackerStream->write( uiBlkBufOffset, pucBlkBuf)))
	{
		goto Exit;
	}

	// There is no way to quickly compute the actual number of bytes
	// that will be written to the backup.  This is due, in part, to the
	// fact that the backup compresses unused space out of blocks before
	// storing them.

	backupInfo.ui64BytesToDo = FSGetSizeInBytes( uiMaxFileSize,
											pFBak->uiLogicalEOF);

	// Initialize the FDB

	pDb = (FDB *)(pFBak->hDb);
	if( !pFBak->bCSMode)
	{
		FLMBOOL		bDummy;

		bDbInitialized = TRUE;
		if( RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, 
			FDB_TRANS_GOING_OK, 0, &bDummy)))
		{
			goto Exit;
		}

		flmAssert( !bDummy);
	}
	else
	{
		bDbInitialized = TRUE;
		fdbInitCS( pDb);
	}

	uiBlockFileOffset = 0;
	uiBlockCount = 0;
	iFileNum = 1;

	for( ;;)
	{
		if( uiBlockFileOffset >= uiMaxFileSize)
		{
			uiBlockFileOffset = 0;
			iFileNum++;
		}

		uiBlkAddr = FSBlkAddress( iFileNum, uiBlockFileOffset);
		if( !FSAddrIsBelow( uiBlkAddr, pFBak->uiLogicalEOF))
		{
			break;
		}

		if( !pFBak->bCSMode)
		{
			// Get the block

			if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
				uiBlkAddr, NULL, &pSCache)))
			{
				goto Exit;
			}

			if( bFullBackup || 
				FB2UD( &pSCache->pucBlk[ BH_TRANS_ID]) > 
					pFBak->uiLastBackupTransId)
			{
				uiBlkBufOffset = 0;
				uiActualBlkSize = getEncryptSize( pSCache->pucBlk);
				if( uiActualBlkSize < BH_OVHD)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_DATA_ERROR);
					goto Exit;
				}
	
				// Output the backup header for the block

				UD2FBA( (FLMUINT32)uiBlkAddr, 
					&pucBlkBuf[ FLM_BACKER_BLK_ADDR_OFFSET]);

				UD2FBA( (FLMUINT32)uiActualBlkSize,
					&pucBlkBuf[ FLM_BACKER_BLK_SIZE_OFFSET]);

				uiBlkBufOffset += FLM_BACKER_BLK_HDR_SIZE;

				// Copy the block into the block buffer and compute the checksum

				f_memcpy( &pucBlkBuf[ uiBlkBufOffset], 
					pSCache->pucBlk, uiActualBlkSize);

				// If this is an encrypted block, we need to make sure it gets encrypted.
				if ( pucBlkBuf[ BH_ENCRYPTED + uiBlkBufOffset])
				{
					if (RC_BAD( rc = ScaEncryptBlock( pSCache->pFile,
																&pucBlkBuf[uiBlkBufOffset],
																uiActualBlkSize,
																pSCache->pFile->
																				FileHdr.uiBlockSize)))
					{
						goto Exit;
					}
				}

				BlkCheckSum( &pucBlkBuf[ uiBlkBufOffset],
					CHECKSUM_SET, uiBlkAddr, uiBlockSize);

				uiBlkBufOffset += uiActualBlkSize;

				// Write the block to the backup stream

				if( RC_BAD( rc = pBackerStream->write( 
					uiBlkBufOffset, pucBlkBuf)))
				{
					goto Exit;
				}

				uiBlockCount++;
			}

			ScaReleaseCache( pSCache, FALSE);
			pSCache = NULL;
			uiBlockFileOffset += uiBlockSize;
		}
		else
		{
			rc = RC_SET( FERR_ILLEGAL_OP);
			goto Exit;
		}

		// Call the status callback

		if( fnStatus && (uiBlockCount - uiBlockCountLastCB) > 100)
		{
			backupInfo.ui64BytesDone = FSGetSizeInBytes( uiMaxFileSize,
													uiBlkAddr);
			if( RC_BAD( rc = fnStatus( FLM_DB_BACKUP_STATUS, 
				(void *)&backupInfo, NULL, pvAppData)))
			{
				goto Exit;
			}

			uiBlockCountLastCB = uiBlockCount;
		}
	}

	// Output the end-of-backup marker

	f_memset( pucBlkBuf, 0xFF, sizeof( FLM_BACKER_BLK_HDR_SIZE));
	if( RC_BAD( rc = pBackerStream->write( FLM_BACKER_BLK_HDR_SIZE, 
		pucBlkBuf)))
	{
		goto Exit;
	}

	// Pad the backup so that FlmDbRestore will never read more
	// data from the input stream than the backup wrote to it.

	uiBytesToPad = (FLMUINT32)(pBackerStream->getMTUSize() - 
		(FLMUINT)(pBackerStream->getByteCount() %
				(FLMUINT64)pBackerStream->getMTUSize()));

	if( uiBytesToPad < pBackerStream->getMTUSize())
	{
		f_memset( pucBlkBuf, 0, uiBytesToPad);
		if( RC_BAD( rc = pBackerStream->write( uiBytesToPad, pucBlkBuf)))
		{
			goto Exit;
		}
	}

	// Because of the double buffering, we need to have one empty
	// buffer at the end of the file.

	f_memset( pucBlkBuf, 0, pBackerStream->getMTUSize());
	if( RC_BAD( rc = pBackerStream->write( pBackerStream->getMTUSize(), 
		pucBlkBuf)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBackerStream->flush()))
	{
		goto Exit;
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	if( pBackerStream)
	{
		pBackerStream->Release();
	}

	// Call the status callback now that the background 
	// thread has terminated.

	if( RC_OK( rc) && fnStatus)
	{
		backupInfo.ui64BytesDone = backupInfo.ui64BytesToDo;
		(void)fnStatus( FLM_DB_BACKUP_STATUS, 
			(void *)&backupInfo, NULL, pvAppData);
	}

	if( hookState.pMultiFileHdl)
	{
		hookState.pMultiFileHdl->Release();
	}

	if( pucBlkBuf)
	{
		f_free( &pucBlkBuf);
	}

	if( RC_OK( rc))
	{
		pFBak->bCompletedBackup = TRUE;
	}

	pFBak->backupRc = rc;
	return( rc);
}

/****************************************************************************
Desc : Ends the backup, updating the log header if needed.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbBackupEnd(
	HFBACKUP *		phBackup)
{
	RCODE				rc = FERR_OK;
	FBak *			pFBak = (FBak *)*phBackup;
	FDB *				pDb = (FDB *)pFBak->hDb;
	FLMBOOL			bDbInitialized = FALSE;
	FLMBOOL			bStartedTrans = FALSE;
	FLMBYTE *		pLogHdr = NULL;

	// End the transaction

	flmAssert( pFBak->uiTransType != FLM_NO_TRANS);
	if( RC_BAD( rc = FlmDbTransAbort( (HFDB)pDb)))
	{
		goto Exit;
	}
	pFBak->uiTransType = FLM_NO_TRANS;
	pFBak->bTransStarted = FALSE;

	// Update log header fields

	if( pFBak->bCompletedBackup && 
		pFBak->uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		// Start an update transaction.

		if( !pFBak->bCSMode)
		{
			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
				0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS, &bStartedTrans)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = FlmDbTransBegin( (HFDB)pDb, 
				FLM_UPDATE_TRANS,	FLM_NO_TIMEOUT, pFBak->pucDbHeader)))
			{
				goto Exit;
			}
			pLogHdr = &pFBak->pucDbHeader[ DB_LOG_HEADER_START]; 
			bStartedTrans = TRUE;
		}

		// Update the log header fields.

		if( !pFBak->bCSMode)
		{
			UD2FBA( (FLMUINT32)pFBak->uiTransId,
				&pDb->pFile->ucUncommittedLogHdr [LOG_LAST_BACKUP_TRANS_ID]);
		}
		else
		{
			UD2FBA( (FLMUINT32)pFBak->uiTransId, 
				&pLogHdr[ LOG_LAST_BACKUP_TRANS_ID]);
		}

		// Since there may have been transactions during the backup,
		// we need to take into account the number of blocks that have
		// changed during the backup when updating the LOG_BLK_CHG_SINCE_BACKUP
		// statistic.

		if( !pFBak->bCSMode)
		{
			flmDecrUint(
				&pDb->pFile->ucUncommittedLogHdr [LOG_BLK_CHG_SINCE_BACKUP],
				pFBak->uiBlkChgSinceLastBackup);
		}
		else
		{
			flmDecrUint(
				&pLogHdr [LOG_BLK_CHG_SINCE_BACKUP],
				pFBak->uiBlkChgSinceLastBackup);
		}

		// Bump the incremental backup sequence number

		if( pFBak->eBackupType == FLM_INCREMENTAL_BACKUP)
		{
			if( !pFBak->bCSMode)
			{
				flmIncrUint(
					&pDb->pFile->ucUncommittedLogHdr [LOG_INC_BACKUP_SEQ_NUM], 1);
			}
			else
			{
				flmIncrUint( &pLogHdr [LOG_INC_BACKUP_SEQ_NUM], 1);
			}
		}

		// Always change the incremental backup serial number.  This is
		// needed so that if the user performs a full backup, runs some
		// transactions against the database, performs another full backup,
		// and then performs an incremental backup we will know that the
		// incremental backup cannot be restored against the first full
		// backup.

		if( !pFBak->bCSMode)
		{
			f_memcpy(
				&pDb->pFile->ucUncommittedLogHdr [LOG_INC_BACKUP_SERIAL_NUM],
				pFBak->ucNextIncSerialNum, F_SERIAL_NUM_SIZE);
		}
		else
		{
			f_memcpy( &pLogHdr[ LOG_INC_BACKUP_SERIAL_NUM], 
				pFBak->ucNextIncSerialNum, F_SERIAL_NUM_SIZE);
		}

		// Commit the transaction and perform a checkpoint so that the
		// modified log header values will be written.

		if( !pFBak->bCSMode)
		{
			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
		}
		else
		{
			if( RC_BAD( rc = fcsDbTransCommitEx( (HFDB)pDb, TRUE, 
				pFBak->pucDbHeader)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
		}
	}

Exit:

	// Abort the active transaction (if any)

	if( bStartedTrans)
	{
		if( !pFBak->bCSMode)
		{
			flmAbortDbTrans( pDb);
		}
		else
		{
			FlmDbTransAbort( (HFDB)pDb);
		}
	}

	// Release the FDB

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	// Free the backup handle

	if( pFBak->pucDbHeader)
	{
		f_freeAlignedBuffer( &pFBak->pucDbHeader);
	}

	f_free( &pFBak);

	// Clear the handle

	*phBackup = HFBACKUP_NULL;

	// Unset the backup flag

	if( !IsInCSMode( (HFDB)pDb))
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		pDb->pFile->bBackupActive = FALSE;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
	else
	{
		(void)fcsSetBackupActiveFlag( (HFDB)pDb, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	Restores a database and supporting files.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbRestore(
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszBackupPath,
	const char *			pszRflDir,
	const char *			pszPassword,
	F_Restore *				pRestoreObj)
{
	RCODE						rc = FERR_OK;
	HFDB						hDb = HFDB_NULL;
	IF_FileHdl *			pFileHdl = NULL;
	IF_FileHdl *			pLockFileHdl = NULL;
	F_SuperFileHdl *		pSFile = NULL;
	char						szBasePath[ F_PATH_MAX_SIZE];
	char						szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT					uiDbVersion;
	FLMUINT					uiNextIncNum;
	eRestoreActionType	eRestoreAction;
	FLMBOOL					bRflPreserved;
	FLMBOOL					bMutexLocked = FALSE;
	FFILE *					pFile = NULL;
	F_FSRestore *			pFSRestoreObj = NULL;
	FLMBOOL					bOKToRetry;
	FLMBYTE *				pucDbKey = NULL;
	FLMUINT					uiKeyLen = 0;
	char *					pszTempPassword = NULL;

	// Set up the callback

	if( !pRestoreObj)
	{
		if( (pFSRestoreObj = f_new F_FSRestore) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pFSRestoreObj->setup( pszDbPath,
			pszBackupPath, pszRflDir)))
		{
			goto Exit;
		}
		pRestoreObj = pFSRestoreObj;
	}

	// Get the base path

	flmGetDbBasePath( szBasePath, pszDbPath, NULL);

	// Force the file to close if it is not used by another thread

	(void)FlmConfig( FLM_CLOSE_FILE, (void *)pszDbPath,
								(void *)pszDataDir);

	gv_FlmSysData.pFileHdlCache->closeUnusedFiles();

	// Lock the global mutex

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Free any unused structures that have been unused for the maximum
	// amount of time.  May unlock and re-lock the global mutex.

	flmCheckNUStructs( 0);

	// Look up the file using flmFindFile to see if the file is already open.
	// May unlock and re-lock the global mutex..

	if( RC_BAD( rc = flmFindFile( pszDbPath, pszDataDir, &pFile)))
	{
		goto Exit;
	}

	// If the file is open, we cannot perform a restore

	if( pFile)
	{
		rc = RC_SET( FERR_ACCESS_DENIED);
		pFile = NULL;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		goto Exit;
	}

	// Allocate the FFILE.  This will prevent other threads from opening the
	// database while the restore is being performed.

	if( RC_BAD( rc = flmAllocFile( pszDbPath, pszDataDir, NULL, &pFile)))
	{
		goto Exit;
	}

	// Remove the FFILE from the NU list -- it was put in the NU list by
	// flmAllocFile.  We don't want the FFILE to disappear while we are
	// doing the restore because it happens to age out of the NU list.

	flmAssert( !pFile->uiUseCount);
	flmUnlinkFileFromNUList( pFile);

	// Unlock the global mutex

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	// Create a lock file.  If this fails, it could indicate
	// that the destination database exists and is in use by another
	// process.

	f_sprintf( szTmpPath, "%s.lck", szBasePath);
	if( RC_BAD( rc = flmCreateLckFile( szTmpPath, &pLockFileHdl)))
	{
		goto Exit;
	}

	// Create the control file

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->createFile( 
		pszDbPath, FLM_IO_RDWR, &pFileHdl)))
	{
		goto Exit;
	}

	// Open the backup set

	if( RC_BAD( rc = pRestoreObj->openBackupSet()))
	{
		goto Exit;
	}

	// Make a copy of the password as flmRestoreFile may zero-out the original
	// after unshrouding the key.

	if ( pszPassword)
	{
		if ( RC_BAD( rc = f_alloc( 
			f_strlen( pszPassword) + 1, &pszTempPassword)))
		{
			goto Exit;
		}

		f_strcpy( pszTempPassword, pszPassword);
	}

	// Restore the data in the backup set

	if( RC_BAD( rc = flmRestoreFile( pRestoreObj, pszDbPath, pszDataDir,
		pszTempPassword, &pSFile, FALSE, &uiDbVersion, &uiNextIncNum,
		&bRflPreserved, &eRestoreAction, NULL, &pucDbKey, NULL, &uiKeyLen)))
	{
		goto Exit;
	}


	// See if we should continue

	if( eRestoreAction == RESTORE_ACTION_STOP)
	{
		goto Exit;
	}

	// Close the backup set

	if( RC_BAD( rc = pRestoreObj->close()))
	{
		goto Exit;
	}

	// Apply any available incremental backups.  uiNextIncNum will be 0 if
	// the database version does not support incremental backups.

	if( uiNextIncNum && uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		FLMUINT		uiCurrentIncNum;

		for( ;;)
		{
			uiCurrentIncNum = uiNextIncNum;
			if( RC_BAD( rc = pRestoreObj->openIncFile( uiCurrentIncNum)))
			{
				if( rc == FERR_IO_PATH_NOT_FOUND)
				{
					rc = FERR_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = flmRestoreFile( pRestoreObj, pszDbPath, pszDataDir, 
					pszTempPassword, &pSFile, TRUE, &uiDbVersion, &uiNextIncNum,
					&bRflPreserved, &eRestoreAction, &bOKToRetry, NULL,
					pucDbKey, &uiKeyLen)))
				{
					RCODE		tmpRc;

					if( !bOKToRetry)
					{
						// Cannot retry the operation or continue ... the
						// database is in an unknown state.

						goto Exit;
					}

					if (RC_BAD( tmpRc = pRestoreObj->status( RESTORE_ERROR,
										0, (void *)(FLMUINT)rc, NULL, NULL, &eRestoreAction)))
					{
						rc = tmpRc;
						goto Exit;
					}

					if( eRestoreAction == RESTORE_ACTION_RETRY ||
						eRestoreAction == RESTORE_ACTION_CONTINUE)
					{
						// Abort the current file (if any)

						if( RC_BAD( rc = pRestoreObj->abortFile()))
						{
							goto Exit;
						}

						if( eRestoreAction == RESTORE_ACTION_CONTINUE)
						{
							// Break out and begin processing the RFL

							break;
						}

						// Otherwise, retry opening the incremental file

						uiNextIncNum = uiCurrentIncNum;
						continue;
					}
					goto Exit;
				}

				// See if we should continue

				if( eRestoreAction == RESTORE_ACTION_STOP)
				{
					goto Exit;
				}

				// Close the current file

				if( RC_BAD( rc = pRestoreObj->close()))
				{
					goto Exit;
				}
			}
		}
	}
	
	// Force everything out to disk

	if( RC_BAD( rc = pSFile->flush()))
	{
		goto Exit;
	}

	pSFile->Release();
	pSFile = NULL;

	// Don't do anything with the RFL if the preserve flag
	// isn't set.

	if( !bRflPreserved)
	{
		if( pFSRestoreObj == pRestoreObj)
		{
			pFSRestoreObj->Release();
			pFSRestoreObj = NULL;
		}
		pRestoreObj = NULL;
	}

	// Open the file and apply any available RFL files.  The
	// lock file handle is passed to the flmOpenFile call so
	// that we don't have to give up our lock until the
	// restore is complete.  Also, we don't want to resume
	// any indexing at this point.  By not resuming the indexes,
	// we can perform a DB diff of two restored databases that
	// should be identical without having differences in the
	// tracker container due to background indexing.

	rc = flmOpenFile( pFile, pszDbPath, pszDataDir,
		pszRflDir, FO_DONT_RESUME_BACKGROUND_THREADS,
		TRUE, pRestoreObj, pLockFileHdl, NULL, (FDB **)&hDb);
		
	pLockFileHdl = NULL;
	pFile = NULL;

	if( RC_BAD( rc))
	{
		goto Exit;
	}

	// Close the database

	(void)FlmDbClose( &hDb);
	(void)FlmConfig( FLM_CLOSE_FILE, (void *)pszDbPath,
				(void *)pszDataDir);

Exit:

	if( pSFile)
	{
		// Need to release the super file handle before cleaning up the
		// FFILE because the super file still has a reference to the
		// FFILE's file ID list.

		pSFile->Release();
	}

	if( pFile)
	{
		// We should only get to this point if rc is bad.

		flmAssert( RC_BAD( rc));

		if( !bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		rc = flmNewFileFinish( pFile, rc);
		flmFreeFile( pFile);

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	if( pLockFileHdl)
	{
		pLockFileHdl->Release();
	}

	if( pFSRestoreObj)
	{
		pFSRestoreObj->Release();
	}

	if (pucDbKey)
	{
		f_free( &pucDbKey);
	}

	if ( pszTempPassword)
	{
		f_free( &pszTempPassword);
	}

	// If restore failed, remove all database files (excluding RFL files)

	if( RC_BAD( rc))
	{
		(void)FlmDbRemove( pszDbPath, pszDataDir, NULL, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc : Restores a full or incremental backup
****************************************************************************/
FSTATIC RCODE flmRestoreFile(
	F_Restore *				pRestoreObj,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszPassword,
	F_SuperFileHdl **		ppSFile,
	FLMBOOL					bIncremental,
	FLMUINT *				puiDbVersion,
	FLMUINT *				puiNextIncSeqNum,
	FLMBOOL *				pbRflPreserved,
	eRestoreActionType *	peRestoreAction,
	FLMBOOL *				pbOKToRetry,
	FLMBYTE **				ppucKeyToSave,
	FLMBYTE *				pucKeyToUse,
	FLMUINT *				puiKeyLen)
{
	RCODE						rc = FERR_OK;
	FLMUINT					uiBytesWritten;
	FLMUINT					uiLogicalEOF;
	FLMUINT					uiBlkAddr;
	FLMUINT					uiBlockCount = 0;
	FLMUINT					uiActualBlkSize;
	FLMUINT					uiBlockSize;
	FLMUINT					uiDbVersion;
	FLMUINT					uiMaxFileSize;
	FLMUINT					uiBackupMaxFileSize;
	FLMUINT					uiPriorBlkFile = 0;
	FLMUINT					uiSectorSize;
	FLMBYTE *				pLogHdr;
	FLMBYTE					ucIncSerialNum[ F_SERIAL_NUM_SIZE];
	FLMBYTE					ucNextIncSerialNum[ F_SERIAL_NUM_SIZE];
	FLMUINT					uiIncSeqNum;
	FLMBYTE *				pucBlkBuf = NULL;
	FLMBYTE					ucLowChecksumByte;
	FLMUINT					uiBlkBufSize;
	FLMUINT					uiPriorBlkAddr = 0;
	BYTE_PROGRESS			byteProgress;
	FBackupType				eBackupType;
	F_BackerStream *		pBackerStream = NULL;
	F_CCS *					pTmpCCS = NULL;
	F_SuperFileHdl *		pSFile = NULL;
	F_SuperFileClient *	pSFileClient = NULL;

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pszPassword);
#endif

	// Initialize the "ok-to-retry" flag

	if( pbOKToRetry)
	{
		*pbOKToRetry = TRUE;
	}

	// Initialize the progress struct

	f_memset( &byteProgress, 0, sizeof( BYTE_PROGRESS));

	// Set up the backer stream object

	if( (pBackerStream = f_new F_BackerStream) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBackerStream->setup( FLM_BACKER_MTU_SIZE, pRestoreObj)))
	{
		goto Exit;
	}
	
	// Get the sector size

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->getSectorSize( 
		pszDbPath, &uiSectorSize)))
	{
		goto Exit;
	}

	// Allocate a temporary buffer.  Try to align the buffer on a sector
	// boundary to avoid memcpy operatons in the file system.

	uiBlkBufSize = FLM_BACKER_MTU_SIZE;
	if( uiSectorSize)
	{
		uiBlkBufSize = (((uiBlkBufSize / uiSectorSize) + 1) * uiSectorSize);
	}
	
	if( RC_BAD( rc = f_allocAlignedBuffer( uiBlkBufSize, (void **)&pucBlkBuf)))
	{
		goto Exit;
	}

	// Read and verify the backup header

	if( RC_BAD( rc = pBackerStream->read( FLM_BACKER_MIN_DB_BLOCK_SIZE,
		pucBlkBuf)))
	{
		goto Exit;
	}

	if( FB2UD( &pucBlkBuf[ FLM_BACKER_VERSION_OFFSET]) !=	FLM_BACKER_VERSION)
	{
		rc = RC_SET_AND_ASSERT( FERR_UNSUPPORTED_VERSION);
		goto Exit;
	}

	if( f_strncmp( (const char *)&pucBlkBuf[ FLM_BACKER_SIGNATURE_OFFSET],
		FLM_BACKER_SIGNATURE, FLM_BACKER_SIGNATURE_SIZE) != 0)
	{
		rc = RC_SET_AND_ASSERT( FERR_UNSUPPORTED_VERSION);
		goto Exit;
	}
	
	uiBlockSize = (FLMUINT)FB2UW( &pucBlkBuf[ FLM_BACKER_DB_BLOCK_SIZE_OFFSET]);
	if( uiBlockSize > FLM_BACKER_MAX_DB_BLOCK_SIZE)
	{
		rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Get the maximum file size from the backup header.
	
	uiBackupMaxFileSize = (FLMUINT)FB2UD( &pucBlkBuf[ FLM_BACKER_BFMAX_OFFSET]);

	// Make sure the MTU is correct

	if( FB2UD( &pucBlkBuf[ FLM_BACKER_MTU_OFFSET]) != FLM_BACKER_MTU_SIZE)
	{
		rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Make sure the backup type is correct

	eBackupType = (FBackupType)FB2UD( 
		&pucBlkBuf[ FLM_BACKER_BACKUP_TYPE_OFFSET]);

	if( (eBackupType == FLM_INCREMENTAL_BACKUP && !bIncremental) ||
		(eBackupType == FLM_FULL_BACKUP && bIncremental))
	{
		// Do not allow an incremental backup to be restored directly.  The
		// only way to restore an incremental backup is to provide the 
		// incremental files when requested by FlmDbRestore.  Also, we don't
		// want to allow the user to mistakenly hand us a full backup when
		// we are expecting an incremental backup.

		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Grab the "next" incremental backup serial number

	f_memcpy( ucNextIncSerialNum, 
		&pucBlkBuf[ FLM_BACKER_NEXT_INC_SERIAL_NUM], 
		F_SERIAL_NUM_SIZE);

	// Get the database version from the backup header

	uiDbVersion = FB2UD( &pucBlkBuf[ FLM_BACKER_DB_VERSION]);
	if( puiDbVersion)
	{
		*puiDbVersion = uiDbVersion;
	}

	// Seek to the database header block

	if( uiBlockSize > FLM_BACKER_MIN_DB_BLOCK_SIZE)
	{
		if( RC_BAD( rc = pBackerStream->read( 
			uiBlockSize - FLM_BACKER_MIN_DB_BLOCK_SIZE, pucBlkBuf)))
		{
			goto Exit;
		}
	}

	// Read the database header block from the backup

	if( RC_BAD( rc = pBackerStream->read( uiBlockSize, pucBlkBuf)))
	{
		goto Exit;
	}

	// Sanity check - make sure the block size in the backup header
	// is the same as the size in the database header

	if( uiBlockSize != 
		FB2UD( &pucBlkBuf[ FLAIM_HEADER_START + DB_BLOCK_SIZE]))
	{
		rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Get a pointer to the log header and verify the checksum

	pLogHdr = &pucBlkBuf[ DB_LOG_HEADER_START];
	if( lgHdrCheckSum( pLogHdr, TRUE))
	{
		rc = RC_SET( FERR_BLOCK_CHECKSUM);
		goto Exit;
	}

	// Compare the database version in the log header with
	// the one extracted from the backup header

	if( (FLMUINT)FB2UW( &pLogHdr[ LOG_FLAIM_VERSION]) != uiDbVersion)
	{
		rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
		goto Exit;
	}
	uiMaxFileSize = flmGetMaxFileSize( uiDbVersion, pLogHdr);

	// Make sure the maximum block file size matches what was read from the
	// backup header.
	
	if( uiBackupMaxFileSize != uiMaxFileSize)
	{
		rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Allocate a super file object
	
	if( (pSFile = *ppSFile) != NULL)
	{
		pSFile->AddRef();
	}
	else
	{
		flmAssert( !bIncremental);
		
		if( (pSFile = f_new F_SuperFileHdl) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if( (pSFileClient = f_new F_SuperFileClient) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = pSFileClient->setup( pszDbPath, pszDataDir, uiDbVersion)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = pSFile->setup( pSFileClient, 
			gv_FlmSysData.pFileHdlCache, gv_FlmSysData.uiFileOpenFlags,
			gv_FlmSysData.uiFileCreateFlags)))
		{
			goto Exit;
		}

		*ppSFile = pSFile;
		(*ppSFile)->AddRef();
	}
	
	// Unshroud the database key (stored in the log header) using the
	// password the user gave us.  (Note: this only re-writes the data
	// in the log header.  It's up to the database open call to actually
	// create the F_CCS object when it initializes the FFILE.)

	if( pszPassword && *pszPassword &&
		  FB2UD( &pLogHdr[ LOG_FLAIM_VERSION]) >= FLM_FILE_FORMAT_VER_4_60 &&
		  FB2UW( &pLogHdr[ LOG_DATABASE_KEY_LEN]) > 0 )
	{
		FLMBYTE *	pucTmpBuf = NULL;
		FLMUINT32	ui32KeyLen = 0;
		FLMUINT		uiNewChecksum;

		if ( (pTmpCCS = f_new F_CCS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if ( RC_BAD( rc = pTmpCCS->init( TRUE, FLM_NICI_AES)))
		{
			goto Exit;
		}

		ui32KeyLen = FB2UW( &pLogHdr [ LOG_DATABASE_KEY_LEN]);
		
		if( RC_BAD( rc = pTmpCCS->setKeyFromStore( &pLogHdr[ LOG_DATABASE_KEY],
			ui32KeyLen, pszPassword, NULL, FALSE)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pTmpCCS->getKeyToStore( &pucTmpBuf, &ui32KeyLen,
			NULL, NULL, FALSE)))
		{
			goto Exit;
		}

		// Verify that the field in the log header is long enough to
		// hold the key.
		
		if( ui32KeyLen > FLM_MAX_DB_ENC_KEY_LEN)
		{
			f_free( &pucTmpBuf);
			rc = RC_SET_AND_ASSERT( FERR_BAD_ENC_KEY);
			goto Exit;
		}

		UW2FBA( ui32KeyLen, &pLogHdr[ LOG_DATABASE_KEY_LEN]);
		f_memcpy( &pLogHdr[LOG_DATABASE_KEY], pucTmpBuf, ui32KeyLen);
		
		uiNewChecksum = lgHdrCheckSum( pLogHdr, FALSE);
		
		UW2FBA( (FLMUINT16)uiNewChecksum, &pLogHdr[ LOG_HDR_CHECKSUM]);
		f_free( &pucTmpBuf);
		
		pTmpCCS->Release();
		pTmpCCS = NULL;

		if( ppucKeyToSave)
		{
			// Need to allocate a buffer to save the key in
			
			if ( RC_BAD( rc = f_alloc( ui32KeyLen, ppucKeyToSave)))
			{
				goto Exit;
			}

			f_memcpy( *ppucKeyToSave, &pLogHdr[ LOG_DATABASE_KEY], ui32KeyLen);
			*puiKeyLen = ui32KeyLen;
		}
	}
	else if( pucKeyToUse)
	{
		UW2FBA( (FLMUINT16)*puiKeyLen, &pLogHdr[ LOG_DATABASE_KEY_LEN]);
		f_memcpy( &pLogHdr[ LOG_DATABASE_KEY], pucKeyToUse, *puiKeyLen);
	}

	// Get the logical EOF from the log header

	uiLogicalEOF = FB2UD( &pLogHdr[ LOG_LOGICAL_EOF]);

	// Are RFL files being preserved?

	if( pbRflPreserved)
	{
		*pbRflPreserved = pLogHdr[ LOG_KEEP_RFL_FILES]
								? TRUE
								: FALSE;
	}

	// Get the incremental backup sequence number

	uiIncSeqNum = FB2UD( &pLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);
	*puiNextIncSeqNum = uiIncSeqNum;

	if( bIncremental)
	{
		(*puiNextIncSeqNum)++;
	}

	// Get information about the incremental backup

	if( bIncremental)
	{
		FLMBYTE			ucTmpSerialNum[ F_SERIAL_NUM_SIZE];
		FLMBYTE			ucTmpSeqNum[ 4];
		FLMUINT			uiTmp;

		f_memcpy( ucIncSerialNum, &pLogHdr[ LOG_INC_BACKUP_SERIAL_NUM],
			F_SERIAL_NUM_SIZE);

		// Compare the incremental backup sequence number to the value in the
		// database's log header.

		if( RC_BAD( rc = 	pSFile->readOffset( 0, DB_LOG_HEADER_START + LOG_INC_BACKUP_SEQ_NUM,
			4, ucTmpSeqNum, &uiTmp)))
		{
			goto Exit;
		}

		if( FB2UD( &ucTmpSeqNum[ 0]) != uiIncSeqNum)
		{
			rc = RC_SET( FERR_INVALID_FILE_SEQUENCE);
			goto Exit;
		}

		// Compare the incremental backup serial number to the value in the
		// database's log header.

		if( RC_BAD( rc = 	pSFile->readOffset( 0, DB_LOG_HEADER_START + LOG_INC_BACKUP_SERIAL_NUM,
			F_SERIAL_NUM_SIZE, ucTmpSerialNum, &uiTmp)))
		{
			goto Exit;
		}

		if( f_memcmp( ucIncSerialNum, 
			ucTmpSerialNum, F_SERIAL_NUM_SIZE) != 0)
		{
			rc = RC_SET( FERR_SERIAL_NUM_MISMATCH);
			goto Exit;
		}

		// Increment the incremental backup sequence number

		UD2FBA( (FLMUINT32)(uiIncSeqNum + 1),
			&pLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);
	}

	// At the start of a backup, either incremental or full, 
	// we generate a new incremental serial number.  This is needed so
	// that if the user performs a full backup, runs some transactions
	// against the database, performs another full backup, and then
	// performs an incremental backup we will know that the incremental
	// backup cannot be restored against the first full backup.

	// Since the new serial number is not written to the database's log
	// header until after the backup completes, we need to put the
	// new serial number in the log header during the restore.  In doing
	// so, the log header will contain the correct serial number for a
	// subsequent incremental backup that may have been made.

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		f_memcpy( &pLogHdr[ LOG_INC_BACKUP_SERIAL_NUM],
			ucNextIncSerialNum, F_SERIAL_NUM_SIZE);
	}

	// Re-calculate the log header checksum

	UW2FBA( (FLMUINT16)lgHdrCheckSum( pLogHdr, FALSE),
		&pLogHdr[ LOG_HDR_CHECKSUM]);

	pLogHdr = NULL;

	// Set the "ok-to-retry" flag

	if( pbOKToRetry)
	{
		*pbOKToRetry = FALSE;
	}

	// Write the database header

	if( RC_BAD( rc = pSFile->writeBlock( FSBlkAddress( 0, 0), 
		uiBlockSize, pucBlkBuf, &uiBytesWritten)))
	{
		goto Exit;
	}

	// The status callback will give a general idea of how much work
	// is left to do.  We don't have any way to get the total size
	// of the stream to give a correct count, so a close estimate
	// will have to suffice.

	byteProgress.ui64BytesToDo = FSGetSizeInBytes( uiMaxFileSize,
												uiLogicalEOF);

	// Write the blocks in the backup file to the database

	for( ;;)
	{
		if( RC_BAD( rc = pBackerStream->read( FLM_BACKER_BLK_HDR_SIZE, 
			pucBlkBuf)))
		{
			goto Exit;
		}

		uiBlockCount++;
		uiBlkAddr = FB2UD( &pucBlkBuf[ FLM_BACKER_BLK_ADDR_OFFSET]);
		uiActualBlkSize = FB2UD( &pucBlkBuf[ FLM_BACKER_BLK_SIZE_OFFSET]);

		// Are we done?

		if( uiBlkAddr == 0xFFFFFFFF)
		{
			break;
		}

		if( !uiBlkAddr || 
			!FSAddrIsBelow( uiBlkAddr, uiLogicalEOF) ||
			(uiPriorBlkAddr && !FSAddrIsBelow( uiPriorBlkAddr, uiBlkAddr)))
		{
			rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
			goto Exit;
		}

		// Read and process the block

		if( uiActualBlkSize > uiBlockSize || uiActualBlkSize < BH_OVHD)
		{
			rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
			goto Exit;
		}

		if( RC_BAD( rc = pBackerStream->read( uiActualBlkSize, pucBlkBuf)))
		{
			goto Exit;
		}

		if( (GET_BH_ADDR( pucBlkBuf) & 0xFFFFFF00) != (uiBlkAddr & 0xFFFFFF00))
		{
			rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
			goto Exit;
		}
		
		if( uiActualBlkSize < uiBlockSize)
		{
			f_memset( &pucBlkBuf[ uiActualBlkSize], 0,
				uiBlockSize - uiActualBlkSize);
		}

		// Verify the checksum

		ucLowChecksumByte = pucBlkBuf[ BH_CHECKSUM_LOW];
		
		if( RC_BAD( rc = BlkCheckSum( pucBlkBuf, CHECKSUM_CHECK,
			uiBlkAddr, uiBlockSize)))
		{
			if( rc == FERR_BLOCK_CHECKSUM)
			{
				rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
			}
			
			goto Exit;
		}
		
		pucBlkBuf[ BH_CHECKSUM_LOW] = ucLowChecksumByte;

		// Write the block to the database

		if( RC_BAD( rc = pSFile->writeBlock( uiBlkAddr,  
			uiBlockSize, pucBlkBuf, &uiBytesWritten)))
		{
			if( rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				// Create a new block file

				if( FSGetFileNumber( uiBlkAddr) != (uiPriorBlkFile + 1))
				{
					rc = RC_SET_AND_ASSERT( FERR_INCONSISTENT_BACKUP);
					goto Exit;
				}

				if( RC_BAD( rc = pSFile->createFile( 
					FSGetFileNumber( uiBlkAddr))))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pSFile->writeBlock( uiBlkAddr,  
					uiBlockSize, pucBlkBuf, &uiBytesWritten)))
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		uiPriorBlkAddr = uiBlkAddr;
		uiPriorBlkFile = FSGetFileNumber( uiBlkAddr);

		if( (uiBlockCount & 0x7F) == 0x7F)
		{
			byteProgress.ui64BytesDone = FSGetSizeInBytes( uiMaxFileSize,
														uiBlkAddr);
			if( RC_BAD( rc = pRestoreObj->status( RESTORE_PROGRESS, 0,
				(void *)&byteProgress, NULL, NULL, peRestoreAction)))
			{
				goto Exit;
			}

			if( *peRestoreAction == RESTORE_ACTION_STOP)
			{
				rc = RC_SET( FERR_USER_ABORT);
				goto Exit;
			}
		}
	}

	// Call the status callback one last time.

	byteProgress.ui64BytesDone = byteProgress.ui64BytesToDo;
	if( RC_BAD( rc = pRestoreObj->status( RESTORE_PROGRESS, 0,
		(void *)&byteProgress, NULL, NULL, peRestoreAction)))
	{
		goto Exit;
	}

	if( *peRestoreAction == RESTORE_ACTION_STOP)
	{
		// It is safe to jump to exit at this point

		goto Exit;
	}

Exit:

	if( pucBlkBuf)
	{
		f_freeAlignedBuffer( (void **)&pucBlkBuf);
	}

	if( pBackerStream)
	{
		pBackerStream->Release();
	}

	if( pTmpCCS)
	{
		pTmpCCS->Release();
	}
	
	if( pSFile)
	{
		pSFile->Release();
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Default hook for creating a backup file set
****************************************************************************/
FSTATIC RCODE flmDefaultBackerWriteHook(
	void *		pvBuffer,
	FLMUINT		uiBytesToWrite,
	void *		pvCallbackData)
{
	BACKER_HOOK_STATE *		pState = (BACKER_HOOK_STATE *)pvCallbackData;
	FLMUINT						uiBytesWritten;
	RCODE							rc = pState->rc;

	if( RC_BAD( rc))
	{
		goto Exit;
	}

	if( !pState->pMultiFileHdl)
	{
		// Remove any existing backup files

		if( RC_BAD( rc = FlmAllocMultiFileHdl( &pState->pMultiFileHdl)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pState->pMultiFileHdl->deleteMultiFile( 
			pState->szPath)) && rc != FERR_IO_PATH_NOT_FOUND &&
			rc != FERR_IO_INVALID_PATH)
		{
			pState->pMultiFileHdl->Release();
			pState->pMultiFileHdl = NULL;
			goto Exit;
		}

		if( RC_BAD( rc = pState->pMultiFileHdl->createFile( pState->szPath)))
		{
			pState->pMultiFileHdl->Release();
			pState->pMultiFileHdl = NULL;
			goto Exit;
		}
	}

	rc = pState->pMultiFileHdl->write( pState->ui64Offset, 
		uiBytesToWrite, pvBuffer, &uiBytesWritten);
	pState->ui64Offset += uiBytesWritten;

Exit:

	if( RC_BAD( rc))
	{
		pState->rc = rc;
		if( pState->pMultiFileHdl)
		{
			pState->pMultiFileHdl->Release();
			pState->pMultiFileHdl = NULL;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_BackerStream::F_BackerStream( void)
{
	m_bSetup = FALSE;
	m_bFirstRead = TRUE;
	m_ui64ByteCount = 0;
	m_uiBufOffset = 0;
	m_pRestoreObj = NULL;
	m_hDataSem = F_SEM_NULL;
	m_hIdleSem = F_SEM_NULL;
	m_pThread = NULL;
	m_rc = FERR_OK;
	m_pucInBuf = NULL;
	m_puiInOffset = NULL;
	m_pucOutBuf = NULL;
	m_puiOutOffset = NULL;
	m_pucBufs[ 0] = NULL;
	m_pucBufs[ 1] = NULL;
	m_uiOffsets[ 0] = 0;
	m_uiOffsets[ 1] = 0;
	m_uiMTUSize = 0;
	m_uiPendingIO = 0;
	m_fnWrite = NULL;
	m_pvCallbackData = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_BackerStream::~F_BackerStream( void)
{
	shutdownThreads();
	
	if( m_hDataSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hDataSem);
	}

	if( m_hIdleSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hIdleSem);
	}

	if( m_pucBufs[ 0])
	{
		f_free( &m_pucBufs[ 0]);
	}

	if( m_pucBufs[ 1])
	{
		f_free( &m_pucBufs[ 1]);
	}
}

/****************************************************************************
Desc: Start any background threads
****************************************************************************/
RCODE F_BackerStream::startThreads( void)
{
	RCODE		rc = FERR_OK;

	if( m_pThread)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// The semaphore handles better be null

	flmAssert( m_hDataSem == F_SEM_NULL);
	flmAssert( m_hIdleSem == F_SEM_NULL);

	// Create a semaphore to signal the background thread
	// that data is available

	if( RC_BAD( rc = f_semCreate( &m_hDataSem)))
	{
		goto Exit;
	}

	// Create a semaphore to signal when the background thread
	// is idle

	if( RC_BAD( rc = f_semCreate( &m_hIdleSem)))
	{
		goto Exit;
	}

	// Start the thread

	if( m_fnWrite)
	{
		if( RC_BAD( rc = f_threadCreate( &m_pThread,
			F_BackerStream::writeThread, "backup",
			0, 0, (void *)this)))
		{
			goto Exit;
		}
	}
	else if( m_pRestoreObj)
	{
		if( RC_BAD( rc = f_threadCreate( &m_pThread,
			F_BackerStream::readThread, "restore",
			0, 0, (void *)this)))
		{
			goto Exit;
		}
	}
	else
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Shut down any background threads
****************************************************************************/
void F_BackerStream::shutdownThreads( void)
{
	if( m_pThread)
	{
		// Shut down the background read or write thread.

		m_pThread->setShutdownFlag();
		f_semSignal( m_hDataSem);
		f_threadDestroy( &m_pThread);

		// Now that the thread has terminated, it is safe
		// to destroy the data and idle semaphores.

		f_semDestroy( &m_hDataSem);
		f_semDestroy( &m_hIdleSem);
	}
}

/****************************************************************************
Desc: Setup method to use the backer stream as an input stream
****************************************************************************/
RCODE F_BackerStream::setup(
	FLMUINT				uiMTUSize,
	F_Restore *			pRestoreObj)
{
	RCODE			rc = FERR_OK;

	flmAssert( pRestoreObj);
	flmAssert( !m_bSetup);

	m_pRestoreObj = pRestoreObj;
	m_uiMTUSize = uiMTUSize;

	if( RC_BAD( rc = _setup()))
	{
		goto Exit;
	}

	// Fire up the background threads

	if( RC_BAD( rc = startThreads()))
	{
		goto Exit;
	}

	m_bSetup = TRUE;

Exit:
	
	return( rc);
}

/****************************************************************************
Desc: Setup method to use the backer stream as an output stream
****************************************************************************/
RCODE F_BackerStream::setup(
	FLMUINT				uiMTUSize,
	BACKER_WRITE_HOOK	fnWrite,
	void *				pvCallbackData)
{
	RCODE			rc = FERR_OK;

	flmAssert( fnWrite);
	flmAssert( !m_bSetup);

	m_fnWrite = fnWrite;
	m_uiMTUSize = uiMTUSize;
	m_pvCallbackData = pvCallbackData;

	if( RC_BAD( rc = _setup()))
	{
		goto Exit;
	}

	// Fire up the background threads

	if( RC_BAD( rc = startThreads()))
	{
		goto Exit;
	}

	m_bSetup = TRUE;

Exit:
	
	return( rc);
}

/****************************************************************************
Desc: Performs setup operations common to read and write streams
****************************************************************************/
RCODE F_BackerStream::_setup( void)
{
	RCODE			rc = FERR_OK;

	// Allocate a buffer for reading or writing blocks

	if( (m_uiMTUSize < (2 * FLM_BACKER_MAX_DB_BLOCK_SIZE)) ||
		m_uiMTUSize % FLM_BACKER_MAX_DB_BLOCK_SIZE)
	{
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// Allocate buffers for reading or writing

	if( RC_BAD( rc = f_alloc( m_uiMTUSize, &m_pucBufs[ 0])))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( m_uiMTUSize, &m_pucBufs[ 1])))
	{
		goto Exit;
	}

	m_pucInBuf = m_pucBufs[ 0];
	m_puiInOffset = &m_uiOffsets[ 0];

	m_pucOutBuf = m_pucBufs[ 1];
	m_puiOutOffset = &m_uiOffsets[ 1];

Exit:

	return( rc);
}
	
/****************************************************************************
Desc: Reads data from the input stream
****************************************************************************/
RCODE F_BackerStream::read(
	FLMUINT				uiLength,
	FLMBYTE *			pucData,
	FLMUINT *			puiBytesRead)
{
	FLMUINT		uiBufSize;
	FLMBYTE *	pucBuf;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiBytesAvail;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetup);
	flmAssert( m_pRestoreObj);
	flmAssert( uiLength);

	if( m_bFirstRead)
	{
		m_bFirstRead = FALSE;

		// Prime the pump.  Call signalThread twice ... once to
		// get the first chunk of data and a second time to have
		// the background thread pre-fetch the next chunk.  A backup
		// will always have at least two MTU data chunks, so we should
		// never get an IO_END_OF_FILE error.  If we do, the restore
		// operation needs to abort (which will happen because the
		// error will be returned to the caller of this routine).

		if( RC_BAD( rc = signalThread()) ||
			RC_BAD( rc = signalThread()))
		{
			goto Exit;
		}
	}

	while( uiLength)
	{
		uiBufSize = *m_puiOutOffset;
		pucBuf = m_pucOutBuf;

		uiBytesAvail = uiBufSize - m_uiBufOffset;
		flmAssert( uiBytesAvail);

		if( uiBytesAvail < uiLength)
		{
			f_memcpy( &pucData[ uiBytesRead], 
				&pucBuf[ m_uiBufOffset], uiBytesAvail);
			m_uiBufOffset += uiBytesAvail;
			uiBytesRead += uiBytesAvail;
			uiLength -= uiBytesAvail;
		}
		else
		{
			f_memcpy( &pucData[ uiBytesRead],
				&pucBuf[ m_uiBufOffset], uiLength);
			m_uiBufOffset += uiLength;
			uiBytesRead += uiLength;
			uiLength = 0;
		}

		if( m_uiBufOffset == uiBufSize)
		{
			m_uiBufOffset = 0;
			if( RC_BAD( rc = signalThread()))
			{
				// Since we are reading MTU-sized units and the restore
				// code knows when to stop reading, we should never
				// get an IO_END_OF_FILE error back from a call to
				// signalThread().  If we do, we need to return the
				// error to the caller (FlmDbRestore).
				goto Exit;
			}
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}
	m_ui64ByteCount += (FLMUINT64)uiBytesRead;
	return( rc);
}

/****************************************************************************
Desc: Writes data to the output stream
****************************************************************************/
RCODE F_BackerStream::write(
	FLMUINT				uiLength,
	FLMBYTE *			pucData,
	FLMUINT *			puiBytesWritten)
{
	FLMUINT		uiMaxWriteSize;
	FLMUINT		uiBytesWritten = 0;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetup);
	flmAssert( m_fnWrite);
	flmAssert( uiLength);

	while( uiLength)
	{
		uiMaxWriteSize = m_uiMTUSize - *m_puiInOffset;
		flmAssert( uiMaxWriteSize);

		if( uiMaxWriteSize < uiLength)
		{
			f_memcpy( &m_pucInBuf[ *m_puiInOffset], 
				&pucData[ uiBytesWritten], uiMaxWriteSize);
			(*m_puiInOffset) += uiMaxWriteSize;
			uiBytesWritten += uiMaxWriteSize;
			uiLength -= uiMaxWriteSize;
		}
		else
		{
			f_memcpy( &m_pucInBuf[ *m_puiInOffset],
				&pucData[ uiBytesWritten], uiLength);
			(*m_puiInOffset) += uiLength;
			uiBytesWritten += uiLength;
			uiLength = 0;
		}
	
		if( (*m_puiInOffset) == m_uiMTUSize)
		{
			if( RC_BAD( rc = signalThread()))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	m_ui64ByteCount += (FLMUINT64)uiBytesWritten;
	return( rc);
}

/****************************************************************************
Desc: Flushes any pending writes to the output stream
****************************************************************************/
RCODE F_BackerStream::flush( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetup);

	if( m_fnWrite && m_pThread)
	{
		if( *m_puiInOffset)
		{
			if( RC_BAD( rc = signalThread()))
			{
				goto Exit;
			}
		}

		// Wait for the background thread to become idle.  When it
		// does, we know that all writes have completed.

		if( RC_BAD( rc = f_semWait( m_hIdleSem, F_WAITFOREVER)))
		{
			goto Exit;
		}

		// If the background thread set an error code, we need to return it.

		rc = m_rc;

		// At this point, we know the background thread is either waiting
		// for the data semaphore to be signaled or it has exited due to
		// an error.  We need to re-signal the idle semaphore so that 
		// other F_BackerStream calls (i.e., additional calls to 
		// flush, etc.) will not block waiting for it to be signaled
		// since it won't be signaled by the background thread until
		// after the data semaphore has been signaled again.

		f_semSignal( m_hIdleSem);

		// Jump to exit if we have a bad rc

		if( RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Signals the read or write thread indicating that data is needed or
		that data is available.
****************************************************************************/
RCODE F_BackerStream::signalThread( void)
{
	FLMBYTE *	pucTmp;
	FLMUINT *	puiTmp;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetup);

	// Return an error if we don't have a thread.

	if( !m_pThread)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Wait for the thread to become idle

	if( RC_BAD( rc = f_semWait( m_hIdleSem, F_WAITFOREVER)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_rc))
	{
		// If m_rc is bad, we know that the background thread has
		// exited and will not signal the idle semaphore again.
		// Thus, we will re-signal the idle semaphore so that if the
		// code using this class happens to call flush() or some
		// other method that waits on the idle semaphore, we
		// won't wait forever on something that will never happen.

		f_semSignal( m_hIdleSem);

		// Check the error code

		if( rc != FERR_IO_END_OF_FILE)
		{
			goto Exit;
		}
	}

	pucTmp = m_pucOutBuf;
	puiTmp = m_puiOutOffset;

	m_pucOutBuf = m_pucInBuf;
	m_puiOutOffset = m_puiInOffset;

	m_pucInBuf = pucTmp;
	m_puiInOffset = puiTmp;

	*(m_puiInOffset) = 0;

	// If m_rc is bad, the background thread has terminated.

	if( RC_OK( m_rc))
	{
		// Signal the thread to read or write data
		// NOTE: The background thread will never be decrementing
		// m_uiPendingIO while we are incrementing it because it
		// will be blocked on m_hDataSem waiting for it to
		// be signaled.

		m_uiPendingIO++;
		f_semSignal( m_hDataSem);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: This thread reads data in the background
****************************************************************************/
RCODE FLMAPI F_BackerStream::readThread(
	IF_Thread *			pThread)
{
	RCODE					rc = FERR_OK;
	F_BackerStream *	pBackerStream = (F_BackerStream *)pThread->getParm1();

	for( ;;)
	{
		f_semSignal( pBackerStream->m_hIdleSem);

		if( RC_BAD( rc = f_semWait( pBackerStream->m_hDataSem,
			F_WAITFOREVER)))
		{
			goto Exit;
		}

		if( !pBackerStream->m_uiPendingIO && 
			pThread->getShutdownFlag())
		{
			break;
		}

		if( RC_BAD( rc = pBackerStream->m_pRestoreObj->read(
			pBackerStream->m_uiMTUSize, pBackerStream->m_pucInBuf,
			pBackerStream->m_puiInOffset)))
		{
			goto Exit;
		}

		flmAssert( pBackerStream->m_uiPendingIO);
		pBackerStream->m_uiPendingIO--;
	}

Exit:

	pBackerStream->m_rc = rc;
	pBackerStream->m_uiPendingIO = 0;
	f_semSignal( pBackerStream->m_hIdleSem);
	return( rc);
}

/****************************************************************************
Desc:	This thread writes data in the background
****************************************************************************/
RCODE FLMAPI F_BackerStream::writeThread(
	IF_Thread *			pThread)
{
	RCODE					rc = FERR_OK;
	F_BackerStream *	pBackerStream = (F_BackerStream *)pThread->getParm1();

	for( ;;)
	{
		f_semSignal( pBackerStream->m_hIdleSem);

		if( RC_BAD( rc = f_semWait( pBackerStream->m_hDataSem,
			F_WAITFOREVER)))
		{
			goto Exit;
		}

		if( *(pBackerStream->m_puiOutOffset))
		{
			if( RC_BAD( rc = pBackerStream->m_fnWrite( 
				pBackerStream->m_pucOutBuf, *(pBackerStream->m_puiOutOffset),
				pBackerStream->m_pvCallbackData)))
			{
				goto Exit;
			}

			// Reset *puiOutOffset so that we won't re-write
			// the same data again if we receive a shut-down
			// signal.

			*(pBackerStream->m_puiOutOffset) = 0;
		}

		// Need to put the thread shutdown check here
		// so that if the thread is signaled twice,
		// once to do final work and once to shut down,
		// we will actually do the work before we exit.

		if( pThread->getShutdownFlag())
		{
			break;
		}
	}

Exit:

	pBackerStream->m_rc = rc;
	pBackerStream->m_uiPendingIO = 0;
	f_semSignal( pBackerStream->m_hIdleSem);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_FSRestore::F_FSRestore() 
{
	m_pFileHdl = NULL;
	m_pMultiFileHdl = NULL;
	m_ui64Offset = 0;
	m_bSetupCalled = FALSE;
	m_szDbPath[ 0] = 0;
	m_uiDbVersion = 0;
	m_szBackupSetPath[ 0] = 0;
	m_szRflDir[ 0] = 0;
	m_bOpen = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_FSRestore::~F_FSRestore() 
{
	if( m_bOpen)
	{
		(void)close();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::setup(
	const char *		pucDbPath,
	const char *		pucBackupSetPath,
	const char *		pucRflDir)
{
	flmAssert( !m_bSetupCalled);
	flmAssert( pucDbPath);
	flmAssert( pucBackupSetPath);

	f_strcpy( m_szDbPath, pucDbPath);
	f_strcpy( m_szBackupSetPath, pucBackupSetPath);

	if( pucRflDir)
	{
		f_strcpy( m_szRflDir, pucRflDir);
	}
	

	m_bSetupCalled = TRUE;
	return( FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openBackupSet( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pMultiFileHdl);

	if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pMultiFileHdl->openFile( m_szBackupSetPath)))
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openRflFile(
	FLMUINT			uiFileNum)
{
	RCODE				rc = FERR_OK;
	char				szRflPath[ F_PATH_MAX_SIZE];
	char				szDbPrefix[ F_FILENAME_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];
	FLMBYTE *		pBuf = NULL;
	FILE_HDR			fileHdr;
	LOG_HDR			logHdr;
	IF_FileHdl *	pFileHdl = NULL;

	flmAssert( m_bSetupCalled);
	flmAssert( uiFileNum);
	flmAssert( !m_pFileHdl);

	// Read the database header to determine the version number
	
	if( !m_uiDbVersion)
	{
		if (RC_BAD( rc = f_alloc( 2048, &pBuf)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( 
			m_szDbPath, FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pFileHdl)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmReadAndVerifyHdrInfo( NULL, pFileHdl,
			pBuf, &fileHdr, &logHdr, NULL)))
		{
			goto Exit;
		}

		pFileHdl->Release();
		pFileHdl = NULL;

		m_uiDbVersion = fileHdr.uiVersionNum;
	}

	// Generate the log file name.

	if( RC_BAD( rc = rflGetDirAndPrefix( 
		m_uiDbVersion, m_szDbPath, m_szRflDir, szRflPath, szDbPrefix)))
	{
		goto Exit;
	}

	rflGetBaseFileName( m_uiDbVersion, szDbPrefix, uiFileNum, szBaseName);
	gv_FlmSysData.pFileSystem->pathAppend( szRflPath, szBaseName);

	// Open the file.

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( 
		szRflPath, gv_FlmSysData.uiFileOpenFlags, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_bOpen = TRUE;
	m_ui64Offset = 0;

Exit:

	if( pBuf)
	{
		f_free( &pBuf);
	}

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::openIncFile(
	FLMUINT			uiFileNum)
{
	RCODE			rc = FERR_OK;
	char			szIncPath[ F_PATH_MAX_SIZE];
	char			szIncFile[ F_FILENAME_SIZE];

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pMultiFileHdl);

	// Since this is a non-interactive restore, we will "guess"
	// that incremental backups are located in the same parent
	// directory as the main backup set.  We will further assume
	// that the incremental backup sets have been named XXXXXXXX.INC,
	// where X is a hex digit.

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( m_szBackupSetPath, 
		szIncPath, NULL)))
	{
		goto Exit;
	}

	f_sprintf( szIncFile, "%08X.INC", (unsigned)uiFileNum);
	gv_FlmSysData.pFileSystem->pathAppend( szIncPath, szIncFile);

	if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pMultiFileHdl->openFile( szIncPath)))
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesRead = 0;

	flmAssert( m_bSetupCalled);
	flmAssert( m_pFileHdl || m_pMultiFileHdl);

	if( m_pMultiFileHdl)
	{
		if( RC_BAD( rc = m_pMultiFileHdl->read( m_ui64Offset, 
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pFileHdl->read( (FLMUINT)m_ui64Offset,
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}
	
	f_assert( uiBytesRead <= uiLength);

Exit:

	m_ui64Offset += uiBytesRead;

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FSRestore::close( void)
{
	flmAssert( m_bSetupCalled);

	if( m_pMultiFileHdl)
	{
		m_pMultiFileHdl->Release();
		m_pMultiFileHdl = NULL;
	}

	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	m_bOpen = FALSE;
	m_ui64Offset = 0;

	return( FERR_OK);
}
