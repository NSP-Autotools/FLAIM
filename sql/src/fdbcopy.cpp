//------------------------------------------------------------------------------
// Desc:	This file contains the F_DbSystem::dbCopy method.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

// Local prototypes

typedef struct Copied_Name *	COPIED_NAME_p;

typedef struct Copied_Name
{
	char				szPath[ F_PATH_MAX_SIZE];
	COPIED_NAME_p	pNext;
} COPIED_NAME;

typedef struct
{
	FLMUINT64		ui64BytesToCopy;
	FLMUINT64		ui64BytesCopied;
	FLMBOOL			bNewSrcFile;
	char				szSrcFileName[ F_PATH_MAX_SIZE];
	char				szDestFileName[ F_PATH_MAX_SIZE]; 
} DB_COPY_INFO, * DB_COPY_INFO_p;


FSTATIC RCODE flmCopyFile(
	DB_COPY_INFO *			pDbCopyInfo,
	COPIED_NAME **			ppCopiedListRV,
	FLMBOOL					bOkToTruncate,
	IF_DbCopyStatus *		ifpStatus);

/****************************************************************************
Desc:		Copies a database, including roll-forward log files.
****************************************************************************/
RCODE F_DbSystem::dbCopy(
	const char *			pszSrcDbName,
			// [IN] Name of source database to be copied.
	const char *			pszSrcDataDir,
			// [IN] Name of source data directory.
	const char *			pszSrcRflDir,
			// [IN] RFL directory of source database. NULL can be
			// passed to indicate that the log files are located
			// in the same directory as the other database files.
	const char *			pszDestDbName,
			// [IN] Destination name of database - will be overwritten if it
			// already exists.
	const char *			pszDestDataDir,
			// [IN] Name of destination data directory.
	const char *			pszDestRflDir,
			// [IN] RFL directory of destination database. NULL can be
			// passed to indicate that the log files are to be located
			// in the same directory as the other database files.
	IF_DbCopyStatus *		ifpStatus)
			// [IN] Status callback interface.
{
	RCODE				rc = NE_SFLM_OK;
	F_Db *			pDb = NULL;
	FLMBOOL			bDbLocked = FALSE;

	// Make sure the destination database is closed

	if (RC_BAD( rc = checkDatabaseClosed( pszDestDbName, pszDestDataDir)))
	{
		goto Exit;
	}

	// Open the source database so we can force a checkpoint.

	if (RC_BAD( rc = openDatabase( pszSrcDbName, pszSrcDataDir, pszSrcRflDir,
									 NULL, 0, &pDb)))
	{
		goto Exit;
	}

	// Need to lock the database, because we want to do a checkpoint
	// and then the copy immediately after without letting other
	// threads have the opportunity to get in and update the
	// database.

	if (RC_BAD( rc = pDb->dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bDbLocked = TRUE;

	// Force a checkpoint

	if (RC_BAD( rc = pDb->doCheckpoint( FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}

	// Once we get this far, we have exclusive access to the database
	// and we have forced a checkpoint.  The database's contents are
	// guaranteed to be on disk at this point, and they will not
	// change.

	rc = copyDb( pszSrcDbName, pszSrcDataDir, pszSrcRflDir,
						pszDestDbName, pszDestDataDir, pszDestRflDir,
						ifpStatus);

Exit:

	// Unlock and close the database

	if (bDbLocked)
	{
		pDb->dbUnlock();
	}

	if (pDb)
	{
		pDb->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Copy a database's files, including roll-forward log files.
*****************************************************************************/
RCODE F_DbSystem::copyDb(
	const char *		pszSrcDbName,
	const char *		pszSrcDataDir,
	const char *		pszSrcRflDir,
	const char *		pszDestDbName,
	const char *		pszDestDataDir,
	const char *		pszDestRflDir,
	IF_DbCopyStatus *	ifpStatus)
{
	RCODE						rc = NE_SFLM_OK;
	DB_COPY_INFO			DbCopyInfo;
	F_SuperFileHdl *		pSrcSFileHdl = NULL;
	F_SuperFileHdl *		pDestSFileHdl = NULL;
	F_SuperFileClient *	pSrcSFileClient = NULL;
	F_SuperFileClient *	pDestSFileClient = NULL;
	FLMUINT					uiFileNumber;
	FLMUINT					uiHighFileNumber;
	FLMUINT					uiHighLogFileNumber;
	FLMUINT64				ui64FileSize;
	F_Database *			pDatabase = NULL;
	FLMBOOL					bMutexLocked = FALSE;
	IF_FileHdl *			pLockFileHdl = NULL;
	IF_FileHdl *			pTmpFileHdl = NULL;
	IF_DirHdl *				pDirHdl = NULL;
	FLMBOOL					bDatabaseLocked = FALSE;
	FLMBOOL					bWriteLocked = FALSE;
	IF_LockObject *		pWriteLockObj = NULL;
	IF_LockObject *		pDatabaseLockObj = NULL;
	COPIED_NAME *			pCopiedList = NULL;
	FLMBOOL					bUsedDatabase = FALSE;
	eLockType				currLockType;
	FLMUINT					uiThreadId;
	FLMUINT					uiNumExclQueued;
	FLMUINT					uiNumSharedQueued;
	FLMUINT					uiPriorityCount;
	char *					pszActualSrcRflPath = NULL;
	char *					pszActualDestRflPath = NULL;
	FLMBOOL					bCreatedDestRflDir = FALSE;
	FLMBOOL					bWaited;
	F_SEM						hWaitSem = F_SEM_NULL;

	f_memset( &DbCopyInfo, 0, sizeof( DbCopyInfo));

	// Should not do anything if the source and destination names
	// are the same.

	if (f_stricmp( pszSrcDbName, pszDestDbName) == 0)
	{
		goto Exit;
	}
	
	// Create a "wait" semaphore
	
	if( RC_BAD( rc = f_semCreate( &hWaitSem)))
	{
		goto Exit;
	}

	// Allocate memory for paths we don't want to push onto the stack.

	if (RC_BAD( rc = f_calloc( F_PATH_MAX_SIZE * 2,
		&pszActualSrcRflPath)))
	{
		goto Exit;
	}

	pszActualDestRflPath = &pszActualSrcRflPath[ F_PATH_MAX_SIZE];

	// Set up the super file object for the source database.
	// Must at least open the control file.

	if( (pSrcSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pSrcSFileClient->setup( pszSrcDbName, pszSrcDataDir)))
	{
		goto Exit;
	}

	if( (pSrcSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pSrcSFileHdl->setup( pSrcSFileClient, 
		gv_SFlmSysData.pFileHdlCache, gv_SFlmSysData.uiFileOpenFlags,
		gv_SFlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	// Lock the destination database, if not already locked.
	// This is so we can overwrite it without necessarily
	// deleting it.  May unlock and re-lock the global mutex.

	f_mutexLock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = TRUE;

retry:

	if (RC_BAD( rc = F_DbSystem::findDatabase( pszDestDbName,
								pszDestDataDir, &pDatabase)))
	{
		goto Exit;
	}

	// If we didn't find an FFILE structure, get an
	// exclusive lock on the file.

	if (!pDatabase)
	{
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Attempt to get an exclusive lock on the file.

		if (RC_BAD( rc = flmCreateLckFile( pszDestDbName, &pLockFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		// The call to verifyOkToUse will wait if the database is in
		// the process of being opened by another thread.

		if (RC_BAD( rc = pDatabase->verifyOkToUse( &bWaited)))
		{
			goto Exit;
		}
		
		if (bWaited)
		{
			goto retry;
		}

		// Increment the open count on the F_Database object so it will not
		// disappear while we are copying the database.

		pDatabase->incrOpenCount();
		bUsedDatabase = TRUE;

		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Lock the destination file object and transaction
		// object, if not already locked.

		pDatabase->m_pDatabaseLockObj->getLockInfo( (FLMINT)0,
										&currLockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued,
										&uiPriorityCount);
		if (currLockType != FLM_LOCK_EXCLUSIVE ||
			 uiThreadId != f_threadId())
		{
			pDatabaseLockObj = pDatabase->m_pDatabaseLockObj;
			pDatabaseLockObj->AddRef();
			
			if (RC_BAD( rc = pDatabaseLockObj->lock( 
				hWaitSem, TRUE, FLM_NO_TIMEOUT, 0)))
			{
				goto Exit;
			}
			bDatabaseLocked = TRUE;
		}

		// Lock the write object, if not already locked

		pDatabase->m_pWriteLockObj->getLockInfo( (FLMINT)0,
										&currLockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued,
										&uiPriorityCount);
		if (currLockType != FLM_LOCK_EXCLUSIVE ||
			 uiThreadId != (FLMUINT)f_threadId())
		{
			pWriteLockObj = pDatabase->m_pWriteLockObj;
			pWriteLockObj->AddRef();

			// Only contention here is with the checkpoint thread - wait
			// forever until the checkpoint thread gives it up.

			if (RC_BAD( rc = pDatabase->dbWriteLock( hWaitSem, NULL, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}
			bWriteLocked = TRUE;
		}
	}

	// Set up the super file object for the destination database.

	if( (pDestSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pDestSFileClient->setup( pszDestDbName, pszDestDataDir)))
	{
		goto Exit;
	}

	if( (pDestSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pDestSFileHdl->setup( pDestSFileClient,
		gv_SFlmSysData.pFileHdlCache, gv_SFlmSysData.uiFileOpenFlags,
		gv_SFlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	// See how many files we have and calculate the total size.

	uiHighFileNumber = 0;
	for (;;)
	{
		if ((RC_BAD( rc = pSrcSFileHdl->getFileSize(
			uiHighFileNumber, &ui64FileSize))) || !ui64FileSize )
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME ||
				 !ui64FileSize)
			{
				// If the control file doesn't exist, we will return
				// path not found.

				if (!uiHighFileNumber)
				{
					goto Exit;
				}
				uiHighFileNumber--;
				rc = NE_SFLM_OK;
				break;
			}
			goto Exit;
		}

		DbCopyInfo.ui64BytesToCopy += ui64FileSize;
		if (uiHighFileNumber == MAX_DATA_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiHighFileNumber++;
	}

	// See how many rollback log files we have, and calculate
	// their total size.

	uiHighLogFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER;
	for (;;)
	{
		if ((RC_BAD( rc = pSrcSFileHdl->getFileSize(
			uiHighLogFileNumber, &ui64FileSize))) || !ui64FileSize)
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME ||
				 !ui64FileSize)
			{
				if (uiHighLogFileNumber ==
							FIRST_LOG_BLOCK_FILE_NUMBER)
				{
					uiHighLogFileNumber = 0;
				}
				else
				{
					uiHighLogFileNumber--;
				}
				rc = NE_SFLM_OK;
				break;
			}
			goto Exit;
		}

		DbCopyInfo.ui64BytesToCopy += ui64FileSize;
		if (uiHighLogFileNumber == MAX_LOG_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiHighLogFileNumber++;
	}

	// Get the sizes of the roll-forward log files

	if (RC_BAD( rc = rflGetDirAndPrefix( pszSrcDbName,
		pszSrcRflDir, pszActualSrcRflPath)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->openDir(
								pszActualSrcRflPath, (char *)"*", &pDirHdl)))
	{
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = pDirHdl->next()))
		{
			if (rc == NE_FLM_IO_NO_MORE_FILES)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}

		// If the current file is an RFL file, increment ui64BytesToCopy

		if (rflGetFileNum( pDirHdl->currentItemName(), &uiFileNumber))
		{
			DbCopyInfo.ui64BytesToCopy += (FLMUINT64)pDirHdl->currentItemSize();
		}
	}

	pDirHdl->Release();
	pDirHdl = NULL;

	// Close all file handles in the source and destination

	pSrcSFileHdl->releaseFiles();
	pDestSFileHdl->releaseFiles();

	// Copy the database files.

	for (uiFileNumber = 0; uiFileNumber <= uiHighFileNumber; uiFileNumber++)
	{

		// Get the source file path and destination file path.

		if( RC_BAD( rc = pSrcSFileHdl->getFilePath(
			uiFileNumber, DbCopyInfo.szSrcFileName)))
		{
			goto Exit;
		}
		if( RC_BAD( rc = pDestSFileHdl->getFilePath(
			uiFileNumber, DbCopyInfo.szDestFileName)))
		{
			goto Exit;
		}

		DbCopyInfo.bNewSrcFile = TRUE;
		if (RC_BAD( rc = flmCopyFile(
			&DbCopyInfo, &pCopiedList, TRUE, ifpStatus)))
		{
			goto Exit;
		}
	}

	// Copy the additional rollback log files, if any.

	for (uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER;
		  uiFileNumber <= uiHighLogFileNumber; uiFileNumber++)
	{

		// Get the source file path and destination file path.

		if (RC_BAD( rc = pSrcSFileHdl->getFilePath( uiFileNumber,
									DbCopyInfo.szSrcFileName)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pDestSFileHdl->getFilePath( uiFileNumber,
			DbCopyInfo.szDestFileName)))
		{
			goto Exit;
		}

		DbCopyInfo.bNewSrcFile = TRUE;
		if (RC_BAD( rc = flmCopyFile( 
			&DbCopyInfo, &pCopiedList, TRUE, ifpStatus)))
		{
			goto Exit;
		}
	}

	// Copy the RFL files

	// Create the destination RFL directory, if needed.  The purpose of this
	// code is two-fold: 1) We want to keep track of the fact that we tried
	// to create the destination RFL directory so we can try to remove it
	// if the copy fails; 2) If the destination RFL directory path specifies
	// a directory with existing files, we want to remove them.

	if( RC_BAD( rc = rflGetDirAndPrefix( pszDestDbName,
		pszDestRflDir, pszActualDestRflPath)))
	{
		goto Exit;
	}

	if( RC_OK( gv_SFlmSysData.pFileSystem->doesFileExist( pszActualDestRflPath)))
	{
		if( gv_SFlmSysData.pFileSystem->isDir( pszActualDestRflPath))
		{
			// Remove the existing directory and all files, etc.

			(void)gv_SFlmSysData.pFileSystem->removeDir(
				pszActualDestRflPath, TRUE);
		}
		else
		{
			(void)gv_SFlmSysData.pFileSystem->deleteFile( pszActualDestRflPath);
		}
	}

	// Try to create the destination RFL directory.  This might fail if
	// another process was accessing the directory for some reason
	// (i.e., from a command prompt), when we tried to remove it above.
	// We really don't care if the call to CreateDir is sucessful, because
	// when we try to create the destination files (below), the FLAIM file
	// file system code will try to create any necessary directories.

	(void)gv_SFlmSysData.pFileSystem->createDir( pszActualDestRflPath);
	bCreatedDestRflDir = TRUE;

	// Copy the RFL files.  NOTE:  We need to copy all of the RFL files
	// in the source RFL directory so that they will be available
	// when performing a database restore operation.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->openDir(
								pszActualSrcRflPath, (char *)"*", &pDirHdl)))
	{
		goto Exit;
	}

	for (;;)
	{
		if( RC_BAD( rc = pDirHdl->next()))
		{
			if (rc == NE_FLM_IO_NO_MORE_FILES)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}

		// If the current file is an RFL file, copy it to the destination

		if( rflGetFileNum( pDirHdl->currentItemName(), &uiFileNumber))
		{
			// Get the source file path and the destination file path.

			if (RC_BAD( rc = rflGetFileName( pszSrcDbName,
											pszSrcRflDir, uiFileNumber,
											DbCopyInfo.szSrcFileName)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = rflGetFileName( pszDestDbName,
										pszDestRflDir, uiFileNumber,
										DbCopyInfo.szDestFileName)))
			{
				goto Exit;
			}

			DbCopyInfo.bNewSrcFile = TRUE;
			if (RC_BAD( rc = flmCopyFile(
				&DbCopyInfo, &pCopiedList, TRUE, ifpStatus)))
			{
				goto Exit;
			}
		}
	}

	pDirHdl->Release();
	pDirHdl = NULL;

Exit:

	if (bUsedDatabase)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		pDatabase->decrOpenCount();
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	// Unlock the database, if it is locked.

	if (bWriteLocked)
	{
		pDatabase->dbWriteUnlock();
		bWriteLocked = FALSE;
	}

	if (bDatabaseLocked)
	{
		RCODE	rc3;

		if (RC_BAD( rc3 = pDatabaseLockObj->unlock()))
		{
			if (RC_OK( rc))
				rc = rc3;
		}
		
		bDatabaseLocked = FALSE;
	}

	if (pWriteLockObj)
	{
		pWriteLockObj->Release();
		pWriteLockObj = NULL;
	}

	if (pDatabaseLockObj)
	{
		pDatabaseLockObj->Release();
		pDatabaseLockObj = NULL;
	}

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->closeFile();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	if (pTmpFileHdl)
	{
		pTmpFileHdl->Release();
	}

	if (pDirHdl)
	{
		pDirHdl->Release();
	}

	// Free all the names of files that were copied.
	// If the copy didn't finish, try to delete any files
	// that were copied.

	while (pCopiedList)
	{
		COPIED_NAME_p	pNext = pCopiedList->pNext;

		// If the overall copy failed, delete the copied file.

		if (RC_BAD( rc))
		{
			(void)gv_SFlmSysData.pFileSystem->deleteFile( pCopiedList->szPath);
		}

		f_free( &pCopiedList);
		pCopiedList = pNext;
	}

	if (RC_BAD( rc) && bCreatedDestRflDir)
	{
		(void)gv_SFlmSysData.pFileSystem->removeDir( pszActualDestRflPath);
	}

	if (pszActualSrcRflPath)
	{
		f_free( &pszActualSrcRflPath);
	}
	
	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	if( pSrcSFileClient)
	{
		pSrcSFileClient->Release();
	}

	if( pSrcSFileHdl)
	{
		pSrcSFileHdl->Release();
	}

	if( pDestSFileClient)
	{
		pDestSFileClient->Release();
	}

	if( pDestSFileHdl)
	{
		pDestSFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Copy a file that is one of the files of the database.
*****************************************************************************/
FSTATIC RCODE flmCopyFile(
	DB_COPY_INFO *			pDbCopyInfo,
	COPIED_NAME **			ppCopiedListRV,
	FLMBOOL					bOkToTruncate,
	IF_DbCopyStatus *		ifpStatus)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucBuffer = NULL;
	IF_FileHdl *	pSrcFileHdl = NULL;
	IF_FileHdl *	pDestFileHdl = NULL;
	FLMUINT			uiBufferSize = 32768;
	FLMUINT			uiBytesToRead;
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;
	FLMUINT			uiOffset;
	FLMBOOL			bCreatedDestFile = FALSE;

	// Open the source file.

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->openFile( 
		pDbCopyInfo->szSrcFileName,
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
		&pSrcFileHdl)))
	{
		goto Exit;
	}

	// Get a file handle for the destination file.
	// First attempt to open the destination file.  If it does
	// not exist, attempt to create it.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->openFile( pDbCopyInfo->szDestFileName,
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
			&pDestFileHdl)))
	{
		if (rc != NE_FLM_IO_PATH_NOT_FOUND &&
			 rc != NE_FLM_IO_INVALID_FILENAME)
		{
			goto Exit;
		}

		if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->createFile( 
								pDbCopyInfo->szDestFileName,
								FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE |
								FLM_IO_CREATE_DIR | FLM_IO_DIRECT,
								&pDestFileHdl)))
		{
			goto Exit;
		}
		bCreatedDestFile = TRUE;
	}

	// Allocate a buffer for reading and writing.

	if (RC_BAD( rc = f_alloc( uiBufferSize, &pucBuffer)))
	{
		goto Exit;
	}

	// Read from source file until we hit EOF in the file or
	// we hit the end offset.

	uiOffset = 0;
	for (;;)
	{
		uiBytesToRead = (FLMUINT)((0xFFFFFFFF - uiOffset >=
											uiBufferSize)
										 ? uiBufferSize
										 : (FLMUINT)(0xFFFFFFFF - uiOffset));

		// Read data from source file.

		if (RC_BAD( rc = pSrcFileHdl->read( uiOffset, uiBytesToRead,
									pucBuffer, &uiBytesRead)))
		{
			if (rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_SFLM_OK;
				if (!uiBytesRead)
				{
					break;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// Write data to destination file.

		if (RC_BAD( rc = pDestFileHdl->write( uiOffset,
									uiBytesRead, pucBuffer, &uiBytesWritten)))
		{
			goto Exit;
		}

		// Do callback to report progress.

		if (ifpStatus)
		{
			pDbCopyInfo->ui64BytesCopied += (FLMUINT64)uiBytesWritten;
			if (RC_BAD( rc = ifpStatus->dbCopyStatus(
					pDbCopyInfo->ui64BytesToCopy,
					pDbCopyInfo->ui64BytesCopied,
					pDbCopyInfo->bNewSrcFile,
					pDbCopyInfo->szSrcFileName,
					pDbCopyInfo->szDestFileName)))
			{
				goto Exit;
			}
			pDbCopyInfo->bNewSrcFile = FALSE;
		}

		if (0xFFFFFFFF - uiBytesWritten < uiOffset)
		{
			uiOffset = 0xFFFFFFFF;
			break;
		}

		uiOffset += uiBytesWritten;

		// Quit once we reach the end offset or we read fewer bytes
		// than we asked for.

		if (uiBytesRead < uiBytesToRead)
		{
			break;
		}
	}

	// If we overwrote the destination file, as opposed to creating
	// it, truncate it in case it was larger than the number of
	// bytes we actually copied.

	if (!bCreatedDestFile && bOkToTruncate)
	{
		if (RC_BAD( rc = pDestFileHdl->truncateFile( uiOffset)))
		{
			goto Exit;
		}
	}

	// If the copy succeeded, add the destination name to a list
	// of destination files.  This is done so we can clean up
	// copied files if we fail somewhere in the overall database
	// copy.

	if (ppCopiedListRV)
	{
		COPIED_NAME_p	pCopyName;

		if( RC_BAD( rc = f_alloc( (FLMUINT)sizeof( COPIED_NAME), &pCopyName)))
		{
			goto Exit;
		}
		f_strcpy( pCopyName->szPath, pDbCopyInfo->szDestFileName);
		pCopyName->pNext = *ppCopiedListRV;
		*ppCopiedListRV = pCopyName;
	}

Exit:

	if (pucBuffer)
	{
		f_free( &pucBuffer);
	}

	if (pSrcFileHdl)
	{
		pSrcFileHdl->Release();
	}

	if (pDestFileHdl)
	{
		pDestFileHdl->flush();
		pDestFileHdl->Release();
	}

	// Attempt to delete the destination file if
	// we didn't successfully copy it.

	if (RC_BAD( rc))
	{
		(void)gv_SFlmSysData.pFileSystem->deleteFile( pDbCopyInfo->szDestFileName);
	}

	return( rc);
}
