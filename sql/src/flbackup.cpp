//------------------------------------------------------------------------------
// Desc:	Backup and restore Routines
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

// Typedefs

typedef struct
{
	FLMUINT64		ui64BytesToDo;
	FLMUINT64		ui64BytesDone;
} DB_BACKUP_INFO, * DB_BACKUP_INFO_p;

// Local classes

class	F_BackerStream : public F_Object
{
public:

	F_BackerStream( void);
	~F_BackerStream( void);

	RCODE setup(
		FLMUINT					uiMTUSize,
		IF_RestoreClient *	pRestoreObj);

	RCODE setup(
		FLMUINT					uiMTUSize,
		IF_BackupClient *		pClient);

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

	// Methods

	RCODE signalThread( void);

	RCODE _setup( void);

	static RCODE SQFAPI readThread(
		IF_Thread *			pThread);

	static RCODE SQFAPI writeThread(
		IF_Thread *			pThread);

	// Data

	FLMBOOL					m_bSetup;
	FLMBOOL					m_bFirstRead;
	FLMBOOL					m_bFinalRead;
	FLMUINT					m_uiBufOffset;
	FLMUINT64				m_ui64ByteCount;
	IF_RestoreClient *	m_pRestoreObj;
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
	IF_BackupClient *		m_pClient;
};

// Constants

#define FLM_BACKER_SIGNATURE_OFFSET			0
#define		FLM_BACKER_SIGNATURE				"!DB_BACKUP_FILE!"
#define		FLM_BACKER_SIGNATURE_SIZE		16
#define FLM_BACKER_VERSION_OFFSET			16
#define		FLM_BACKER_VERSION_5_0_0		500
#define		FLM_BACKER_VERSION				FLM_BACKER_VERSION_5_0_0
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

#define FLM_BACKER_BLK_HDR_SIZE			((FLMUINT) 4)
#define FLM_BACKER_BLK_ADDR_OFFSET		0

// Local prototypes

FSTATIC RCODE flmRestoreFile(
	IF_RestoreClient *	pRestoreObj,
	IF_RestoreStatus *	pRestoreStatus,
	F_SuperFileHdl *		pSFile,
	FLMBOOL					bIncremental,
	FLMUINT *				puiDbVersion,
	FLMUINT *				puiNextIncSeqNum,
	FLMBOOL *				pbRflPreserved,
	eRestoreAction *		peAction,
	FLMBOOL *				pbOKToRetry);

// Functions

/***************************************************************************
Desc : Prepares FLAIM to backup a database.
Notes: Only one backup of a particular database can be active at any time
*END************************************************************************/
RCODE F_Db::backupBegin(
	eDbBackupType	eBackupType,
	eDbTransType	eTransType,
	FLMUINT			uiMaxLockWait,
	F_Backup **		ppBackup)
{
	F_Backup *	pBackup = NULL;
	FLMBOOL		bBackupFlagSet = FALSE;
	FLMUINT		uiLastCPFileNum;
	FLMUINT		uiLastTransFileNum;
	FLMUINT		uiDbVersion;
	SFLM_DB_HDR *	pDbHdr;
	RCODE			rc = NE_SFLM_OK;

	// Initialize the handle

	*ppBackup = NULL;

	// Make sure we are not being called inside a transaction

	if( getTransType() != SFLM_NO_TRANS)
	{
		rc = RC_SET( NE_SFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// Verify that the application has specified a valid transaction type.

	if( eTransType != SFLM_READ_TRANS && eTransType != SFLM_UPDATE_TRANS)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Make sure a valid backup type has been specified

	uiDbVersion = getDbVersion();

	// See if a backup is currently running against the database.  If so,
	// return an error.  Otherwise, set the backup flag on the FFILE.

	m_pDatabase->lockMutex();
	if( m_pDatabase->m_bBackupActive)
	{
		m_pDatabase->unlockMutex();
		rc = RC_SET( NE_SFLM_BACKUP_ACTIVE);
		goto Exit;
	}
	else
	{
		bBackupFlagSet = TRUE;
		m_pDatabase->m_bBackupActive = TRUE;
	}
	m_pDatabase->unlockMutex();

	// Allocate the backup handle

	if( (pBackup = f_new F_Backup) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	pBackup->m_pDb = this;
	pBackup->m_uiDbVersion = uiDbVersion;

	// Start a transaction

	if( RC_BAD( rc = beginTrans( eTransType, uiMaxLockWait,
		SFLM_DONT_KILL_TRANS | SFLM_DONT_POISON_CACHE, &pBackup->m_dbHdr)))
	{
		goto Exit;
	}

	pBackup->m_bTransStarted = TRUE;
	pBackup->m_eTransType = eTransType;
	pDbHdr = &pBackup->m_dbHdr;

	// Don't allow an incremental backup to be performed
	// if a full backup has not yet been done.

	if( eBackupType == SFLM_INCREMENTAL_BACKUP &&
		 pDbHdr->ui64LastBackupTransID == 0)
	{
		rc = RC_SET( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}

	pBackup->m_eBackupType = eBackupType;

	// Set the next incremental backup serial number.  This is
	// done regardless of the backup type to prevent the wrong
	// set of incremental backup files from being applied
	// to a database.

	if( RC_BAD( rc = f_createSerialNumber(
		pBackup->m_ucNextIncSerialNum)))
	{
		goto Exit;
	}

	// Get the incremental sequence number from the DB header

	pBackup->m_uiIncSeqNum = (FLMUINT)pDbHdr->ui32IncBackupSeqNum;

	// Determine the transaction ID of the last backup

	pBackup->m_ui64LastBackupTransId = pDbHdr->ui64LastBackupTransID;

	// Get the block change count

	pBackup->m_uiBlkChgSinceLastBackup =
		(FLMUINT)pDbHdr->ui32BlksChangedSinceBackup;

	// Get the current transaction ID

	pBackup->m_ui64TransId = pBackup->m_pDb->getTransID();

	// Get the logical end of file

	pBackup->m_uiLogicalEOF = (FLMUINT)pDbHdr->ui32LogicalEOF;

	// Get the first required RFL file needed by the restore.

	uiLastCPFileNum = (FLMUINT)pDbHdr->ui32RflLastCPFileNum;
	uiLastTransFileNum = (FLMUINT)pDbHdr->ui32RflCurrFileNum;

	flmAssert( uiLastCPFileNum <= uiLastTransFileNum);

	pBackup->m_uiFirstReqRfl = uiLastCPFileNum < uiLastTransFileNum
								? uiLastCPFileNum
								: uiLastTransFileNum;

	flmAssert( pBackup->m_uiFirstReqRfl);

	// Get the database block size

	pBackup->m_uiBlockSize = getBlockSize();

	// Get the database path

	(void)getDbControlFileName( pBackup->m_szDbPath,
										sizeof( pBackup->m_szDbPath));

	// Done

	*ppBackup = pBackup;
	pBackup = NULL;

Exit:

	if( RC_BAD( rc))
	{
		if( pBackup)
		{
			if( pBackup->m_bTransStarted)
			{
				abortTrans();
			}

			pBackup->Release();
		}

		if( bBackupFlagSet)
		{
			m_pDatabase->lockMutex();
			m_pDatabase->m_bBackupActive = FALSE;
			m_pDatabase->unlockMutex();
		}
	}

	return( rc);
}

/****************************************************************************
Desc : Constructor
****************************************************************************/
F_Backup::F_Backup()
{
	m_pDb = NULL;
	m_bTransStarted = FALSE;
	reset();
}

/****************************************************************************
Desc : Destructor
****************************************************************************/
F_Backup::~F_Backup()
{
	endBackup();
}

/****************************************************************************
Desc : Reset member variables to their initial state
****************************************************************************/
void F_Backup::reset( void)
{
	if( m_bTransStarted)
	{
		m_pDb->abortTrans();
		m_bTransStarted = FALSE;
	}

	m_pDb = NULL;
	m_eTransType = SFLM_NO_TRANS;
	m_ui64TransId = 0;
	m_ui64LastBackupTransId = 0;
	m_uiDbVersion = 0;
	m_uiBlkChgSinceLastBackup = 0;
	m_uiBlockSize = 0;
	m_uiLogicalEOF = 0;
	m_uiFirstReqRfl = 0;
	m_uiIncSeqNum = 0;
	m_bCompletedBackup = FALSE;
	m_eBackupType = SFLM_FULL_BACKUP;
	m_backupRc = NE_SFLM_OK;
}

/****************************************************************************
Desc : Streams the contents of a database to the write hook supplied by
		 the application.
Notes: This routine attempts to create a backup of a database without
		 excluding any readers or updaters.  However, if the backup runs
		 too long in an environment where extensive updates are happening,
		 an old view error could be returned.
****************************************************************************/
RCODE F_Backup::backup(
	const char *			pszBackupPath,
	const char *			pszPassword,
	IF_BackupClient *		ifpClient,
	IF_BackupStatus *		ifpStatus,
	FLMUINT *				puiIncSeqNum)
{
	FLMBOOL					bFullBackup = TRUE;
	FLMINT					iFileNum;
	FLMUINT					uiBlkAddr;
	FLMUINT					uiTime;
	F_CachedBlock *		pSCache = NULL;
	SFLM_DB_HDR *			pDbHdr;
	DB_BACKUP_INFO			backupInfo;
	FLMUINT					uiBlockFileOffset;
	FLMUINT					uiCount;
	FLMUINT					uiBlockCount;
	FLMUINT					uiBlockCountLastCB = 0;
	FLMUINT					uiBytesToPad;
	F_BackerStream *		pBackerStream = NULL;
	FLMBYTE *				pucBlkBuf = NULL;
	FLMUINT					uiBlkBufOffset;
	FLMUINT					uiBlkBufSize;
	FLMUINT					uiMaxCSBlocks;
	FLMUINT					uiCPTransOffset;
	FLMUINT					uiMaxFileSize;
	FLMBOOL					bReleaseClient = FALSE;
	FLMBOOL					bMustUnlock = FALSE;
	RCODE						rc = NE_SFLM_OK;

	if( puiIncSeqNum)
	{
		*puiIncSeqNum = 0;
	}

	// Setup the status callback info

	f_memset( &backupInfo, 0, sizeof( DB_BACKUP_INFO));

	// Make sure a backup attempt has not been made with this
	// backup handle.

	if( m_bCompletedBackup)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
		goto Exit;
	}

	if( RC_BAD( m_backupRc))
	{
		rc = m_backupRc;
		goto Exit;
	}

	// Look at the backup type

	if( m_eBackupType == SFLM_INCREMENTAL_BACKUP)
	{
		if( puiIncSeqNum)
		{
			*puiIncSeqNum = m_uiIncSeqNum;
		}

		bFullBackup = FALSE;
	}

	// Set up the callback

	if( !ifpClient)
	{
		if( !pszBackupPath)
		{
			rc = RC_SET( NE_SFLM_INVALID_PARM);
			goto Exit;
		}

		ifpClient = f_new F_DefaultBackupClient( pszBackupPath);
		
		if (ifpClient == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}
		
		bReleaseClient = TRUE;
	}

	// Allocate and initialize the backer stream object

	if( (pBackerStream = f_new F_BackerStream) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBackerStream->setup( FLM_BACKER_MTU_SIZE,	ifpClient)))
	{
		goto Exit;
	}

	// Allocate a temporary buffer

	uiBlkBufSize = FLM_BACKER_MTU_SIZE;
	uiMaxCSBlocks = uiBlkBufSize / m_uiBlockSize;
	if( RC_BAD( rc = f_alloc( uiBlkBufSize, &pucBlkBuf)))
	{
		goto Exit;
	}

	// Setup the backup file header

	uiBlkBufOffset = 0;
	f_memset( pucBlkBuf, 0, m_uiBlockSize);
	f_memcpy( &pucBlkBuf[ FLM_BACKER_SIGNATURE_OFFSET],
		FLM_BACKER_SIGNATURE, FLM_BACKER_SIGNATURE_SIZE);

	UD2FBA( FLM_BACKER_VERSION,
		&pucBlkBuf[ FLM_BACKER_VERSION_OFFSET]);
	UD2FBA( (FLMUINT32)m_uiBlockSize,
		&pucBlkBuf[ FLM_BACKER_DB_BLOCK_SIZE_OFFSET]);
	uiMaxFileSize = (FLMUINT)m_dbHdr.ui32MaxFileSize;
	UD2FBA( (FLMUINT32)uiMaxFileSize,
		&pucBlkBuf[ FLM_BACKER_BFMAX_OFFSET]);
	UD2FBA( (FLMUINT32)FLM_BACKER_MTU_SIZE,
		&pucBlkBuf[ FLM_BACKER_MTU_OFFSET]);
	f_timeGetSeconds( &uiTime);
	UD2FBA( (FLMUINT32)uiTime,
		&pucBlkBuf[ FLM_BACKER_TIME_OFFSET]);

	uiCount = f_strlen( m_szDbPath);

	if( uiCount <= 3)
	{
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET] = 
			(FLMBYTE)m_szDbPath[ uiCount - 6];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 1] = 
			(FLMBYTE)m_szDbPath[ uiCount - 5];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 2] = 
			(FLMBYTE)m_szDbPath[ uiCount - 4];
		pucBlkBuf[ FLM_BACKER_DB_NAME_OFFSET + 3] = '\0';
	}

	UD2FBA( (FLMUINT32)m_eBackupType,
		&pucBlkBuf[ FLM_BACKER_BACKUP_TYPE_OFFSET]);

	// Set the next incremental serial number in the backup's
	// header so that it can be put into the database's log header
	// after the backup has been restored.

	f_memcpy( &pucBlkBuf[ FLM_BACKER_NEXT_INC_SERIAL_NUM],
		m_ucNextIncSerialNum, SFLM_SERIAL_NUM_SIZE);

	// Set the database version number

	UD2FBA( (FLMUINT32)m_uiDbVersion,
		&pucBlkBuf[ FLM_BACKER_DB_VERSION]);

	uiBlkBufOffset += m_uiBlockSize;

	// Copy the database header into the backup's buffer

	f_memset( &pucBlkBuf[ uiBlkBufOffset], 0, m_uiBlockSize);
	f_memcpy( &pucBlkBuf[ uiBlkBufOffset], &m_dbHdr, sizeof( SFLM_DB_HDR));
	pDbHdr = (SFLM_DB_HDR *)(&pucBlkBuf[ uiBlkBufOffset]);
	uiBlkBufOffset += m_uiBlockSize;

	// Fix up the log header

	if( !pDbHdr->ui8RflKeepFiles)
	{
		// Put zero in as the last transaction offset so that the current
		// RFL file will be created if it does not exist after the database
		// is restored.  This has basically the same effect as setting the
		// offset to 512 if the RFL file has already been created.

		pDbHdr->ui32RflLastTransOffset = 0;
		uiCPTransOffset = 512;

		// Create new serial numbers for the RFL.  We don't want anyone
		// to be able to branch into a "no-keep" RFL sequence.

		if (RC_BAD( rc = f_createSerialNumber(
							pDbHdr->ucLastTransRflSerialNum)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = f_createSerialNumber(
									pDbHdr->ucNextRflSerialNum)))
		{
			goto Exit;
		}
	}
	else
	{
		uiCPTransOffset = (FLMUINT)pDbHdr->ui32RflLastTransOffset;
		if( !uiCPTransOffset)
		{
			uiCPTransOffset = 512;
		}
	}

	// Set the CP offsets to the last trans offsets.  This is done
	// because the backup could actually read dirty (committed) blocks
	// from the cache, resulting in a backup set that contains blocks
	// that are more recent than the ones currently on disk.

	pDbHdr->ui32RflLastCPFileNum = pDbHdr->ui32RflCurrFileNum;
	pDbHdr->ui64RflLastCPTransID = pDbHdr->ui64CurrTransID;
	pDbHdr->ui32RflLastCPOffset = (FLMUINT32)uiCPTransOffset;
	pDbHdr->ui32RblEOF = (FLMUINT32)m_uiBlockSize;
	pDbHdr->ui32RblFirstCPBlkAddr = 0;

	// If a password was used, wrap the database key in that password
	if (pszPassword && *pszPassword)
	{
		FLMBYTE *	pucTmp = NULL;
		
		// Need to get a lock on the database - mostly to prevent the very
		// unlikely possibility of another thread attempting to use the
		// database key at the same time we are.
		// (Carson found this in his random testing when one thread did
		// a wrapKey while another did a backup.)
		
		if ((m_pDb->m_uiFlags & FDB_HAS_FILE_LOCK) == 0)
		{
			if	(RC_BAD( rc = m_pDb->dbLock(FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}
			bMustUnlock = TRUE;
		}
		rc = m_pDb->getDatabase()->m_pWrappingKey->getKeyToStore( &pucTmp,
								 &pDbHdr->ui32DbKeyLen,
								 (FLMBYTE *)pszPassword, NULL);
		if (bMustUnlock)
		{
			m_pDb->dbUnlock();
			bMustUnlock = FALSE;
		}
		if (RC_BAD( rc))
		{
			if (pucTmp)
			{
				f_free( &pucTmp);
			}
			goto Exit;
		}
		
		f_memcpy( pDbHdr->ucDbKey, pucTmp, pDbHdr->ui32DbKeyLen);
		f_free(  &pucTmp);
	}

	// Header should already be in native format.

	flmAssert( !hdrIsNonNativeFormat( pDbHdr));

	// Calculate and set the CRC

	pDbHdr->ui32HdrCRC = calcDbHdrCRC( pDbHdr);

	// Output the header

	if( RC_BAD( rc = pBackerStream->write( uiBlkBufOffset, pucBlkBuf)))
	{
		goto Exit;
	}

	// There is no way to quickly compute the actual number of bytes
	// that will be written to the backup.  This is due, in part, to the
	// fact that the backup compresses unused space out of blocks before
	// storing them.

	backupInfo.ui64BytesToDo = FSGetSizeInBytes( uiMaxFileSize, m_uiLogicalEOF);
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
		if( !FSAddrIsBelow( uiBlkAddr, m_uiLogicalEOF))
		{
			break;
		}

		// Get the block

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, NULL,
			uiBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}

		if( bFullBackup ||
			 pSCache->getBlockPtr()->ui64TransID > m_ui64LastBackupTransId)
		{
			uiBlkBufOffset = 0;
			if ((FLMUINT)pSCache->getBlockPtr()->ui16BlkBytesAvail >
								m_uiBlockSize - blkHdrSize( pSCache->getBlockPtr()))
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
				goto Exit;
			}

			// Output the backup header for the block

			UD2FBA( (FLMUINT32)uiBlkAddr,
				&pucBlkBuf[ FLM_BACKER_BLK_ADDR_OFFSET]);

			uiBlkBufOffset += FLM_BACKER_BLK_HDR_SIZE;

			// Copy the block into the block buffer and prepare it
			// for writing.

			f_memcpy( &pucBlkBuf[ uiBlkBufOffset],
				pSCache->getBlockPtr(), m_uiBlockSize);

			// Encrypt the block if needed.
			
			if (RC_BAD( rc = m_pDb->m_pDatabase->encryptBlock( m_pDb->m_pDict,
															&pucBlkBuf[ uiBlkBufOffset])))
			{
				goto Exit;
			}

			if (RC_BAD( rc = flmPrepareBlockToWrite( m_uiBlockSize,
				(F_BLK_HDR *)(&pucBlkBuf [uiBlkBufOffset]))))
			{
				goto Exit;
			}

			uiBlkBufOffset += m_uiBlockSize;

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
		uiBlockFileOffset += m_uiBlockSize;

		// Call the status callback

		if ((uiBlockCount - uiBlockCountLastCB) > 100)
		{
			if( ifpStatus)
			{
				backupInfo.ui64BytesDone = FSGetSizeInBytes( uiMaxFileSize,
														uiBlkAddr);
				if( RC_BAD( rc = ifpStatus->backupStatus(
														backupInfo.ui64BytesToDo,
														backupInfo.ui64BytesDone)))
				{
					goto Exit;
				}
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
		(pBackerStream->getByteCount() % pBackerStream->getMTUSize()));

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

	if( pBackerStream)
	{
		pBackerStream->Release();
	}

	// Call the status callback now that the background
	// thread has terminated.

	if( RC_OK( rc) && ifpStatus)
	{
		backupInfo.ui64BytesDone = backupInfo.ui64BytesToDo;
		(void)ifpStatus->backupStatus( backupInfo.ui64BytesToDo,
											  backupInfo.ui64BytesDone);
	}

	if( pucBlkBuf)
	{
		f_free( &pucBlkBuf);
	}

	if( RC_OK( rc))
	{
		m_bCompletedBackup = TRUE;
	}

	if ( bReleaseClient)
	{
		ifpClient->Release();
	}

	m_backupRc = rc;
	return( rc);
}

/****************************************************************************
Area : MISC
Desc : Ends the backup, updating the log header if needed.
****************************************************************************/
RCODE F_Backup::endBackup( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;

	if( !m_bCompletedBackup)
	{
		goto Exit;
	}

	// End the transaction

	flmAssert( m_eTransType != SFLM_NO_TRANS);
	if( RC_BAD( rc = m_pDb->abortTrans()))
	{
		goto Exit;
	}
	m_eTransType = SFLM_NO_TRANS;
	m_bTransStarted = FALSE;

	// Start an update transaction.

	if( RC_BAD( rc = m_pDb->beginTrans( SFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Update log header fields

	m_pDb->m_pDatabase->m_uncommittedDbHdr.ui64LastBackupTransID =
		m_ui64TransId;

	// Since there may have been transactions during the backup,
	// we need to take into account the number of blocks that have
	// changed during the backup when updating the
	// ui32BlksChangedSinceBackup statistic.

	m_pDb->m_pDatabase->m_uncommittedDbHdr.ui32BlksChangedSinceBackup -=
		(FLMUINT32)m_uiBlkChgSinceLastBackup;

	// Bump the incremental backup sequence number

	if( m_eBackupType == SFLM_INCREMENTAL_BACKUP)
	{
		m_pDb->m_pDatabase->m_uncommittedDbHdr.ui32IncBackupSeqNum++;
	}

	// Always change the incremental backup serial number.  This is
	// needed so that if the user performs a full backup, runs some
	// transactions against the database, performs another full backup,
	// and then performs an incremental backup we will know that the
	// incremental backup cannot be restored against the first full
	// backup.

	f_memcpy(
		m_pDb->m_pDatabase->m_uncommittedDbHdr.ucIncBackupSerialNum,
		m_ucNextIncSerialNum, SFLM_SERIAL_NUM_SIZE);

	// Commit the transaction and perform a checkpoint so that the
	// modified log header values will be written.

	bStartedTrans = FALSE;
	if( RC_BAD( m_pDb->commitTrans( 0, TRUE)))
	{
		goto Exit;
	}

Exit:

	// Abort the active transaction (if any)

	if( bStartedTrans)
	{
		m_pDb->abortTrans();
	}

	// Unset the backup flag

	if( m_pDb)
	{
		m_pDb->m_pDatabase->lockMutex();
		m_pDb->m_pDatabase->m_bBackupActive = FALSE;
		m_pDb->m_pDatabase->unlockMutex();
	}

	// Clear the object

	reset();

	// Done.

	return( rc);
}

/****************************************************************************
Desc:		Restores a database from backup
Notes:	This routine does not restore referenced BLOBs.
****************************************************************************/
RCODE F_DbSystem::dbRestore(
	const char *			pszDbPath,
		// [IN] Path of database that is being restored.  This is the
		// same path format that FlmDbCreate expects
		// (i.e., c:\flaim\flm.db).
	const char *			pszDataDir,
		// [IN] Directory where data files are located.
	const char *			pszRflDir,
		// [IN] RFL log file directory.  NULL can be passed to indicate
		// that the files are located in the same directory as the
		// database (specified above).
	const char *			pszBackupPath,
		// [IN] Directory and name of the backup file set.
		// This parameter is required only if the default
		// BACKER_READ_HOOK is used.  Otherwise, NULL can be
		// passed as the value of this parameter.
	const char *			pszPassword,
		// [IN] Password that was used durning the backup
	IF_RestoreClient *	pRestoreObj,
		// [IN] Object to be used to read data from the backup set.
	IF_RestoreStatus *	pRestoreStatus)
		// [IN] Object for reporting the status of the restore
		// operation
{
	RCODE					rc = NE_SFLM_OK;
	IF_FileHdl *		pFileHdl = NULL;
	IF_FileHdl *		pLockFileHdl = NULL;
	F_SuperFileHdl *	pSFile = NULL;
	F_SuperFileClient	SFileClient;	
	FLMBYTE				szBasePath[ F_PATH_MAX_SIZE];
	char					szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT				uiDbVersion;
	FLMUINT				uiNextIncNum;
	eRestoreAction		eAction = SFLM_RESTORE_ACTION_CONTINUE; // default action...
	FLMBOOL				bRflPreserved;
	FLMBOOL				bMutexLocked = FALSE;
	F_Db *				pDb = NULL;
	F_Database *		pDatabase = NULL;
	F_FSRestore *		pFSRestoreObj = NULL;
	FLMBOOL				bOKToRetry;

	// Set up the callback

	if( !pRestoreObj)
	{
		if( !pszBackupPath || *pszBackupPath == 0)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_PARM);
			goto Exit;
		}

		if( (pFSRestoreObj = f_new F_FSRestore) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pFSRestoreObj->setup( pszDbPath,
			pszBackupPath, pszRflDir)))
		{
			goto Exit;
		}

		// Note: If we wanted to be absolutely correct, we'd do an AddRef on
		// pFSRestoreObj because there's going to be two pointers pointing at
		// it.  It really doesn't matter in this case, though because
		// pFSRestoreObj is local to this function and will get deleted before
		// the function exits.

		pRestoreObj = (IF_RestoreClient *)pFSRestoreObj;
	}

	// Get the base path

	flmGetDbBasePath( (char *)szBasePath, pszDbPath, NULL);

	// Lock the global mutex

	f_mutexLock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Look up the file using findDatabase to see if the file is already open.
	// May unlock and re-lock the global mutex..

	if( RC_BAD( rc = findDatabase( pszDbPath, pszDataDir, &pDatabase)))
	{
		goto Exit;
	}

	// If the database is open, we cannot perform a restore

	if( pDatabase)
	{
		rc = RC_SET( NE_SFLM_DATABASE_OPEN);
		pDatabase = NULL;
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		goto Exit;
	}

	// Allocate the F_Database object.  This will prevent other threads from
	// opening the database while the restore is being performed.

	if( RC_BAD( rc = allocDatabase( pszDbPath, pszDataDir, FALSE, &pDatabase)))
	{
		goto Exit;
	}

	// Unlock the global mutex

	f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	// Create a lock file.  If this fails, it could indicate
	// that the destination database exists and is in use by another
	// process.

	f_sprintf( szTmpPath, "%s.lck", szBasePath);
	if( RC_BAD( rc = flmCreateLckFile( szTmpPath, &pLockFileHdl)))
	{
		goto Exit;
	}

	// Create the control file and set up the super file object

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->createFile( 
		pszDbPath, FLM_IO_RDWR, &pFileHdl)))
	{
		goto Exit;
	}

	// Allocate a super file object
	// NOTE: Do not use extended cache for this super-file object.

	if( (pSFile = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = SFileClient.setup( pszDbPath, pszDataDir)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pSFile->setup( &SFileClient, gv_SFlmSysData.pFileHdlCache,
		gv_SFlmSysData.uiFileOpenFlags, gv_SFlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	// Open the backup set

	if( RC_BAD( rc = pRestoreObj->openBackupSet()))
	{
		goto Exit;
	}

	// Restore the data in the backup set

	if( RC_BAD( rc = flmRestoreFile( pRestoreObj, pRestoreStatus,
		pSFile, FALSE, &uiDbVersion, &uiNextIncNum, &bRflPreserved,
		&eAction, NULL)))
	{
		goto Exit;
	}

	// See if we should continue

	if( eAction == SFLM_RESTORE_ACTION_STOP)
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

	if( uiNextIncNum)
	{
		FLMUINT	uiCurrentIncNum;

		for( ;;)
		{
			uiCurrentIncNum = uiNextIncNum;
			if( RC_BAD( rc = pRestoreObj->openIncFile( uiCurrentIncNum)))
			{
				if( rc == NE_FLM_IO_PATH_NOT_FOUND)
				{
					rc = NE_SFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = flmRestoreFile( pRestoreObj, pRestoreStatus,
					pSFile, TRUE, &uiDbVersion, &uiNextIncNum, &bRflPreserved,
					&eAction, &bOKToRetry)))
				{
					RCODE		tmpRc;

					if( !bOKToRetry)
					{
						// Cannot retry the operation or continue ... the
						// database is in an unknown state.

						goto Exit;
					}

					if( pRestoreStatus)
					{
						if( RC_BAD( tmpRc = 
							pRestoreStatus->reportError( &eAction, rc)))
						{
							rc = tmpRc;
							goto Exit;
						}
					}

					if( eAction == SFLM_RESTORE_ACTION_RETRY ||
						eAction == SFLM_RESTORE_ACTION_CONTINUE)
					{
						// Abort the current file (if any)

						if( RC_BAD( rc = pRestoreObj->abortFile()))
						{
							goto Exit;
						}

						if( eAction == SFLM_RESTORE_ACTION_CONTINUE)
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

				if( eAction == SFLM_RESTORE_ACTION_STOP)
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
		pRestoreObj = NULL;
		pRestoreStatus = NULL;
	}

	// Open the file and apply any available RFL files.  The
	// lock file handle is passed to the openDatabase call so
	// that we don't have to give up our lock until the
	// restore is complete.  Also, we don't want to resume
	// any indexing at this point.  By not resuming the indexes,
	// we can perform a DB diff of two restored databases that
	// should be identical without having differences in the
	// tracker container due to background indexing.

	rc = openDatabase( pDatabase,
		pszDbPath, pszDataDir,
		pszRflDir, pszPassword, SFLM_DONT_RESUME_THREADS,
		TRUE, pRestoreObj, pRestoreStatus, pLockFileHdl, &pDb);
	pLockFileHdl = NULL;

	if( RC_BAD( rc))
	{
		pDatabase = NULL;		
		goto Exit;
	}
	
	// If a password was needed to open the database, we need to clear it so it
	//can be opened without a password.
	
	if (pszPassword && pszPassword[0] != 0)
	{
		if (RC_BAD( rc = pDb->wrapKey()))
		{
			goto Exit;
		}
	}

	// Close the database

	pDb->Release();
	pDb = NULL;

Exit:

	if( pSFile)
	{
		// Need to release the super file handle before cleaning up the
		// FFILE because the super file still has a reference to the
		// FFILE's file ID list.

		pSFile->Release();
	}

	if( pDatabase)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		if (RC_BAD( rc))
		{
			pDatabase->newDatabaseFinish( rc);
		}

		if( !pDatabase->m_uiOpenIFDbCount)
		{
			pDatabase->freeDatabase();
		}

		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}

	if( pDb)
	{
		pDb->Release();
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

	// If restore failed, remove all database files (excluding RFL files)

	if( RC_BAD( rc))
	{
		dropDatabase( pszDbPath, pszDataDir, NULL, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc : Restores a full or incremental backup
*END************************************************************************/
FSTATIC RCODE flmRestoreFile(
	IF_RestoreClient *	pRestoreObj,
	IF_RestoreStatus *	pRestoreStatus,
	F_SuperFileHdl *		pSFile,
	FLMBOOL					bIncremental,
	FLMUINT *				puiDbVersion,
	FLMUINT *				puiNextIncSeqNum,
	FLMBOOL *				pbRflPreserved,
	eRestoreAction *		peAction,
	FLMBOOL *				pbOKToRetry)
{
	FLMUINT				uiBytesWritten;
	FLMUINT				uiLogicalEOF;
	FLMUINT				uiBlkAddr;
	FLMUINT				uiBlockCount = 0;
	FLMUINT				uiBlockSize;
	FLMUINT				uiDbVersion;
	FLMUINT				uiMaxFileSize;
	FLMUINT				uiBackupMaxFileSize;
	FLMUINT				uiPriorBlkFile = 0;
	FLMUINT				uiSectorSize;
	SFLM_DB_HDR *			pDbHdr;
	FLMBYTE				ucIncSerialNum[ SFLM_SERIAL_NUM_SIZE];
	FLMBYTE				ucNextIncSerialNum[ SFLM_SERIAL_NUM_SIZE];
	FLMUINT				uiIncSeqNum;
	FLMBYTE *			pucBlkBuf = NULL;
	char					szPath[ F_PATH_MAX_SIZE];
	FLMUINT				uiBlkBufSize;
	FLMUINT				uiPriorBlkAddr = 0;
	FLMUINT64			ui64BytesToDo = 0;
	FLMUINT64			ui64BytesDone = 0;
	eDbBackupType		eBackupType;
	F_BackerStream *	pBackerStream = NULL;
	RCODE					rc = NE_SFLM_OK;
	FLMUINT32			ui32CRC;
	F_BLK_HDR *			pBlkHdr;

	// Initialize the "ok-to-retry" flag

	if( pbOKToRetry)
	{
		*pbOKToRetry = TRUE;
	}

	// Set up the backer stream object

	if( (pBackerStream = f_new F_BackerStream) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBackerStream->setup( FLM_BACKER_MTU_SIZE, pRestoreObj)))
	{
		goto Exit;
	}

	// Get the path of the .DB file (file 0).

	if( RC_BAD( rc = pSFile->getFilePath( 0, szPath)))
	{
		goto Exit;
	}

	// Get the sector size

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->getSectorSize(
		szPath, &uiSectorSize)))
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
		rc = RC_SET( NE_SFLM_UNSUPPORTED_VERSION);
		goto Exit;
	}

	if( f_strncmp( (const char *)&pucBlkBuf[ FLM_BACKER_SIGNATURE_OFFSET],
		FLM_BACKER_SIGNATURE, FLM_BACKER_SIGNATURE_SIZE) != 0)
	{
		rc = RC_SET( NE_SFLM_UNSUPPORTED_VERSION);
		goto Exit;
	}

	uiBlockSize = (FLMUINT)FB2UW( &pucBlkBuf[ FLM_BACKER_DB_BLOCK_SIZE_OFFSET]);
	if( uiBlockSize > FLM_BACKER_MAX_DB_BLOCK_SIZE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Get the maximum file size from the backup header.

	uiBackupMaxFileSize = (FLMUINT)FB2UD( &pucBlkBuf[ FLM_BACKER_BFMAX_OFFSET]);

	// Make sure the MTU is correct

	if( FB2UD( &pucBlkBuf[ FLM_BACKER_MTU_OFFSET]) != FLM_BACKER_MTU_SIZE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Make sure the backup type is correct

	eBackupType = (eDbBackupType)FB2UD(
		&pucBlkBuf[ FLM_BACKER_BACKUP_TYPE_OFFSET]);

	if( (eBackupType == SFLM_INCREMENTAL_BACKUP && !bIncremental) ||
		(eBackupType == SFLM_FULL_BACKUP && bIncremental))
	{
		// Do not allow an incremental backup to be restored directly.  The
		// only way to restore an incremental backup is to provide the
		// incremental files when requested by FlmDbRestore.  Also, we don't
		// want to allow the user to mistakenly hand us a full backup when
		// we are expecting an incremental backup.

		rc = RC_SET( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Grab the "next" incremental backup serial number

	f_memcpy( ucNextIncSerialNum,
		&pucBlkBuf[ FLM_BACKER_NEXT_INC_SERIAL_NUM],
		SFLM_SERIAL_NUM_SIZE);

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

	pDbHdr = (SFLM_DB_HDR *)pucBlkBuf;

	// Calculate the CRC before doing any conversions.

	ui32CRC = calcDbHdrCRC( pDbHdr);

	// Convert to native platform format, if necessary.

	if (hdrIsNonNativeFormat( pDbHdr))
	{
		convertDbHdr( pDbHdr);
	}

	// Validate the checksum

	if (ui32CRC != pDbHdr->ui32HdrCRC)
	{
		rc = RC_SET( NE_SFLM_HDR_CRC);
		goto Exit;
	}

	if( uiBlockSize != (FLMUINT)pDbHdr->ui16BlockSize)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Compare the database version in the DB header with
	// the one extracted from the backup header

	if( (FLMUINT)pDbHdr->ui32DbVersion != uiDbVersion)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
		goto Exit;
	}
	uiMaxFileSize = (FLMUINT)pDbHdr->ui32MaxFileSize;

	// Set the database version number and block size into the
	// super file handle.  We only do this if the file being restored
	// is the full backup.  It will always be first in the restore sequence,
	// and thus we only need to set these values into the super file handle
	// at that time.

	if( !bIncremental)
	{
//		JMC - FIXME: commented out due to missing functionality in flaimtk.h
//		pSFile->setBlockSize( uiBlockSize);
	}

	// Make sure the maximum block file size matches what was read from the
	// backup header.

	if( uiBackupMaxFileSize != uiMaxFileSize)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
		goto Exit;
	}

	// Get the logical EOF from the log header

	uiLogicalEOF = (FLMUINT)pDbHdr->ui32LogicalEOF;

	// Are RFL files being preserved?

	if( pbRflPreserved)
	{
		*pbRflPreserved = pDbHdr->ui8RflKeepFiles
								? TRUE
								: FALSE;
	}

	// Get the incremental backup sequence number

	uiIncSeqNum = (FLMUINT)pDbHdr->ui32IncBackupSeqNum;
	*puiNextIncSeqNum = uiIncSeqNum;

	if( bIncremental)
	{
		(*puiNextIncSeqNum)++;
	}

	// Get information about the incremental backup

	if( bIncremental)
	{
		FLMUINT	uiTmp;
		SFLM_DB_HDR	dbHdr;

		f_memcpy( ucIncSerialNum, pDbHdr->ucIncBackupSerialNum,
			SFLM_SERIAL_NUM_SIZE);

		// Compare the incremental backup sequence number to the value in the
		// database's DB header.

		if( RC_BAD( rc = pSFile->readBlock( 0, sizeof( SFLM_DB_HDR),
											&dbHdr, &uiTmp)))
		{
			goto Exit;
		}
		if (hdrIsNonNativeFormat( &dbHdr))
		{
			convertDbHdr( &dbHdr);
		}

		if( (FLMUINT)dbHdr.ui32IncBackupSeqNum != uiIncSeqNum)
		{
			rc = RC_SET( NE_SFLM_INVALID_FILE_SEQUENCE);
			goto Exit;
		}

		// Compare the incremental backup serial number to the value in the
		// database's log header.

		if( f_memcmp( ucIncSerialNum, dbHdr.ucIncBackupSerialNum,
			SFLM_SERIAL_NUM_SIZE) != 0)
		{
			rc = RC_SET( NE_SFLM_SERIAL_NUM_MISMATCH);
			goto Exit;
		}

		// Increment the incremental backup sequence number

		pDbHdr->ui32IncBackupSeqNum = (FLMUINT32)(uiIncSeqNum + 1);
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

	f_memcpy( pDbHdr->ucIncBackupSerialNum,
			ucNextIncSerialNum, SFLM_SERIAL_NUM_SIZE);

	// DB Header is in native format.  Set the CRC
	// before writing it out.

	pDbHdr->ui32HdrCRC = calcDbHdrCRC( pDbHdr);
	pDbHdr = NULL;

	// Set the "ok-to-retry" flag

	if( pbOKToRetry)
	{
		*pbOKToRetry = FALSE;
	}

	// Write the database header

	if( RC_BAD( rc = pSFile->writeBlock( 0,
		uiBlockSize, pucBlkBuf, &uiBytesWritten)))
	{
		goto Exit;
	}

	// The status callback will give a general idea of how much work
	// is left to do.  We don't have any way to get the total size
	// of the stream to give a correct count, so a close estimate
	// will have to suffice.
	ui64BytesToDo = FSGetSizeInBytes( uiMaxFileSize, uiLogicalEOF);

	// Write the blocks in the backup file to the database

	for (;;)
	{
		if( RC_BAD( rc = pBackerStream->read( FLM_BACKER_BLK_HDR_SIZE,
			pucBlkBuf)))
		{
			goto Exit;
		}

		uiBlockCount++;
		uiBlkAddr = FB2UD( &pucBlkBuf[ FLM_BACKER_BLK_ADDR_OFFSET]);

		// Are we done?

		if( uiBlkAddr == 0xFFFFFFFF)
		{
			break;
		}

		if( !uiBlkAddr ||
			!FSAddrIsBelow( uiBlkAddr, uiLogicalEOF) ||
			(uiPriorBlkAddr && !FSAddrIsBelow( uiPriorBlkAddr, uiBlkAddr)))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
			goto Exit;
		}

		// Read and process the block

		if( RC_BAD( rc = pBackerStream->read( uiBlockSize, pucBlkBuf)))
		{
			goto Exit;
		}

		pBlkHdr = (F_BLK_HDR *)pucBlkBuf;

		// Convert the entire block to native format if necessary.

		if (RC_BAD( rc = flmPrepareBlockForUse( uiBlockSize, pBlkHdr)))
		{
			if (rc == NE_SFLM_BLOCK_CRC)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
			}
			goto Exit;
		}
		if( (FLMUINT)pBlkHdr->ui32BlkAddr != uiBlkAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
			goto Exit;
		}

		// Prepare the block for writing.

		if (RC_BAD( rc = flmPrepareBlockToWrite( uiBlockSize, pBlkHdr)))
		{
			goto Exit;
		}

		// Write the block to the database

		if( RC_BAD( rc = pSFile->writeBlock( uiBlkAddr,
			uiBlockSize, pucBlkBuf, &uiBytesWritten)))
		{
			if( rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME)
			{
				// Create a new block file

				if( FSGetFileNumber( uiBlkAddr) != (uiPriorBlkFile + 1))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_INCONSISTENT_BACKUP);
					goto Exit;
				}

				if( RC_BAD( rc = pSFile->createFile( FSGetFileNumber( uiBlkAddr))))
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

		if( pRestoreStatus && (uiBlockCount & 0x7F) == 0x7F)
		{
			ui64BytesDone = FSGetSizeInBytes( uiMaxFileSize, uiBlkAddr);
			if( RC_BAD( rc = pRestoreStatus->reportProgress( peAction,
																ui64BytesToDo,
																ui64BytesDone)))
			{
				goto Exit;
			}

			if( *peAction == SFLM_RESTORE_ACTION_STOP)
			{
				rc = RC_SET( NE_SFLM_USER_ABORT);
				goto Exit;
			}
		}
	}

	if( pRestoreStatus)
	{
		// Call the status callback one last time.

		ui64BytesDone = ui64BytesToDo;
		if( RC_BAD( rc = pRestoreStatus->reportProgress( peAction, ui64BytesToDo,
																	 ui64BytesDone)))
		{
			goto Exit;
		}

		if( *peAction == SFLM_RESTORE_ACTION_STOP)
		{
			// It is safe to jump to exit at this point

			goto Exit;
		}
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

	return( rc);
}

/****************************************************************************
Desc: Constructor
****************************************************************************/
F_DefaultBackupClient::F_DefaultBackupClient(
	const char *	pszBackupPath)
{
	m_pMultiFileHdl = NULL;
	m_ui64Offset = 0;
	m_rc = NE_SFLM_OK;

	f_strncpy( m_szPath, pszBackupPath, F_PATH_MAX_SIZE - 1);	
}

/****************************************************************************
Desc: Destructor
****************************************************************************/
F_DefaultBackupClient::~F_DefaultBackupClient()
{
	if (m_pMultiFileHdl)
	{
		m_pMultiFileHdl->closeFile();
		m_pMultiFileHdl->Release();
	}
}


/****************************************************************************
Desc: Default hook for creating a backup file set
****************************************************************************/
RCODE F_DefaultBackupClient::WriteData(
	const void *		pvBuffer,
	FLMUINT				uiBytesToWrite)
{
	FLMUINT				uiBytesWritten;
	RCODE					rc = m_rc;

	if( RC_BAD( rc))
	{
		goto Exit;
	}

	if( m_pMultiFileHdl == 0)
	{
		// Remove any existing backup files
		
		if( RC_BAD( rc = FlmAllocMultiFileHdl( &m_pMultiFileHdl)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pMultiFileHdl->deleteMultiFile( m_szPath)) &&
			rc != NE_FLM_IO_PATH_NOT_FOUND &&
			rc != NE_FLM_IO_INVALID_FILENAME)
		{
			m_pMultiFileHdl->Release();
			m_pMultiFileHdl = NULL;
			goto Exit;
		}

		if( RC_BAD( rc = m_pMultiFileHdl->createFile( m_szPath)))
		{
			m_pMultiFileHdl->Release();
			m_pMultiFileHdl = NULL;
			goto Exit;
		}
	}

	rc = m_pMultiFileHdl->write( m_ui64Offset,
		uiBytesToWrite, (FLMBYTE *)pvBuffer, &uiBytesWritten);
	m_ui64Offset += uiBytesWritten;

Exit:

	if( RC_BAD( rc))
	{
		m_rc = rc;
		if( m_pMultiFileHdl)
		{
			m_pMultiFileHdl->Release();
			m_pMultiFileHdl = NULL;
		}
	}

	return( rc);
}

// F_BackerStream methods

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_BackerStream::F_BackerStream( void)
{
	m_bSetup = FALSE;
	m_bFirstRead = TRUE;
	m_bFinalRead = FALSE;
	m_ui64ByteCount = 0;
	m_uiBufOffset = 0;
	m_pRestoreObj = NULL;
	m_hDataSem = F_SEM_NULL;
	m_hIdleSem = F_SEM_NULL;
	m_pThread = NULL;
	m_rc = NE_SFLM_OK;
	m_pucInBuf = NULL;
	m_puiInOffset = NULL;
	m_pucOutBuf = NULL;
	m_puiOutOffset = NULL;
	m_pucBufs[ 0] = NULL;
	m_pucBufs[ 1] = NULL;
	m_uiOffsets[ 0] = 0;
	m_uiOffsets[ 1] = 0;
	m_uiMTUSize = 0;
	m_pClient = NULL;
}

/****************************************************************************
Desc: Destructor
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
	RCODE		rc = NE_SFLM_OK;

	if( m_pThread)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
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

	if( m_pClient)
	{
		if( RC_BAD( rc = gv_SFlmSysData.pThreadMgr->createThread( &m_pThread,
			F_BackerStream::writeThread, "backup",
			0, 0, (void *)this)))
		{
			goto Exit;
		}
	}
	else if( m_pRestoreObj)
	{
		if( RC_BAD( rc = gv_SFlmSysData.pThreadMgr->createThread( &m_pThread,
			F_BackerStream::readThread, "restore",
			0, 0, (void *)this)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
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
		m_pThread->stopThread();
		m_pThread->Release();
		m_pThread = NULL;

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
	FLMUINT					uiMTUSize,
	IF_RestoreClient *	pRestoreObj)
{
	RCODE			rc = NE_SFLM_OK;

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
	FLMUINT					uiMTUSize,
	IF_BackupClient *		pClient)
{
	RCODE			rc = NE_SFLM_OK;

	flmAssert( pClient);
	flmAssert( !m_bSetup);

	m_pClient = pClient;
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
Desc: Performs setup operations common to read and write streams
****************************************************************************/
RCODE F_BackerStream::_setup( void)
{
	RCODE			rc = NE_SFLM_OK;

	// Allocate a buffer for reading or writing blocks

	if( (m_uiMTUSize < (2 * FLM_BACKER_MAX_DB_BLOCK_SIZE)) ||
		m_uiMTUSize % FLM_BACKER_MAX_DB_BLOCK_SIZE)
	{
		rc = RC_SET( NE_SFLM_INVALID_PARM);
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
	RCODE			rc = NE_SFLM_OK;

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
Desc:		Writes data to the output stream
****************************************************************************/
RCODE F_BackerStream::write(
	FLMUINT				uiLength,
	FLMBYTE *			pucData,
	FLMUINT *			puiBytesWritten)
{
	FLMUINT		uiMaxWriteSize;
	FLMUINT		uiBytesWritten = 0;
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetup);
	flmAssert( m_pClient);
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
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetup);

	if( m_pClient && m_pThread)
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
Desc:	Signals the read or write thread indicating that data is needed or
		that data is available.
****************************************************************************/
RCODE F_BackerStream::signalThread( void)
{
	FLMBYTE *	pucTmp;
	FLMUINT *	puiTmp;
	RCODE			rc = NE_SFLM_OK;

	flmAssert( m_bSetup);

	// Return an error if we don't have a thread.

	if( !m_pThread)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
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

		if( rc == NE_FLM_IO_END_OF_FILE && !m_bFinalRead)
		{
			m_bFinalRead = TRUE;
		}
		else
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

	if( !m_bFinalRead)
	{
		// Signal the thread to read or write data

		f_semSignal( m_hDataSem);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: This thread reads data in the background
****************************************************************************/
RCODE F_BackerStream::readThread(
	IF_Thread *			pThread)
{
	F_BackerStream *	pBackerStream = (F_BackerStream *)pThread->getParm1();
	RCODE					rc = NE_SFLM_OK;

	for( ;;)
	{
		f_semSignal( pBackerStream->m_hIdleSem);

		if( RC_BAD( rc = f_semWait( pBackerStream->m_hDataSem,
			F_WAITFOREVER)))
		{
			goto Exit;
		}

		if( pThread->getShutdownFlag())
		{
			break;
		}

		if( RC_BAD( rc = pBackerStream->m_pRestoreObj->read(
			pBackerStream->m_uiMTUSize, pBackerStream->m_pucInBuf,
			pBackerStream->m_puiInOffset)))
		{
			goto Exit;
		}
	}

Exit:

	pBackerStream->m_rc = rc;
	f_semSignal( pBackerStream->m_hIdleSem);
	return( rc);
}

/****************************************************************************
Desc: This thread writes data in the background
****************************************************************************/
RCODE SQFAPI F_BackerStream::writeThread(
	IF_Thread *			pThread)
{
	F_BackerStream *	pBackerStream = (F_BackerStream *)pThread->getParm1();
	RCODE					rc = NE_SFLM_OK;

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
			if( RC_BAD( rc = pBackerStream->m_pClient->WriteData(
				pBackerStream->m_pucOutBuf, *(pBackerStream->m_puiOutOffset))))
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
	f_semSignal( pBackerStream->m_hIdleSem);
	return( rc);
}
