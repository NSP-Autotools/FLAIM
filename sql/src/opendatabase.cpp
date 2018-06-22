//------------------------------------------------------------------------------
// Desc:	Contains the F_DbSystem::openDatabase method.
// Tabs:	3
//
// Copyright (c) 1990-2007 Novell, Inc. All Rights Reserved.
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

#define MAX_DIRTY_PERCENT	70

FSTATIC void flmFreeCPInfo(
	CP_INFO **	ppCPInfoRV);

FSTATIC RCODE SQFAPI flmCPThread(
	IF_Thread *		pThread);
	
/***************************************************************************
Desc:	Does most of the actual work of opening an existing database, but
		 doesn't	provide COM interfaces...
****************************************************************************/
RCODE F_DbSystem::openDatabase(
	const char *	pszDbFileName,
	const char *	pszDataDir,
	const char *	pszRflDir,
	const char *	pszPassword,
	FLMUINT			uiOpenFlags,
	F_Db **			ppDb)
{
	RCODE			rc = NE_SFLM_OK;

	*ppDb = NULL;
	if (!pszDbFileName || *pszDbFileName == 0)
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	// Open the file

	if (RC_BAD( rc = openDatabase( NULL, pszDbFileName, pszDataDir, pszRflDir,
			pszPassword, uiOpenFlags, FALSE, NULL, NULL, NULL, ppDb)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Constructor for F_Db object.
****************************************************************************/
F_Db::F_Db(
	FLMBOOL	bInternalOpen)
{
	m_pDatabase = NULL;
	m_pDict = NULL;
	m_pNextForDatabase = NULL;
	m_pPrevForDatabase = NULL;
	m_pvAppData = NULL;
	m_uiThreadId = 0;
	m_bMustClose = FALSE;
	m_pSFileHdl = NULL;
	m_uiFlags = bInternalOpen ? FDB_INTERNAL_OPEN : 0;
	m_uiTransCount = 0;
	m_eTransType = SFLM_NO_TRANS;
	m_AbortRc = NE_SFLM_OK;
	m_ui64CurrTransID = 0;
	m_uiFirstAvailBlkAddr = 0;
	m_uiLogicalEOF = 0;
	m_uiUpgradeCPFileNum = 0;
	m_uiUpgradeCPOffset = 0;
	m_uiTransEOF = 0;
	f_memset( &m_TransStartTime, 0, sizeof( m_TransStartTime));
	m_bHadUpdOper = FALSE;
	m_uiBlkChangeCnt = 0;
	m_pIxdFixups = NULL;
	m_pNextReadTrans = NULL;
	m_pPrevReadTrans = NULL;
	m_uiInactiveTime = 0;
	m_uiKilledTime = 0;
	m_bItemStateUpdOk = FALSE;
	m_pDeleteStatus = NULL;
	m_pIxClient = NULL;
	m_pIxStatus = NULL;
	m_pCommitClient = NULL;
	m_pStats = NULL;
	m_pDbStats = NULL;
	m_pLFileStats = NULL;
	m_uiLFileAllocSeq = 0;
	f_memset( &m_Stats, 0, sizeof( m_Stats));
	m_bStatsInitialized = TRUE;
	m_pIxStartList = NULL;
	m_pIxStopList = NULL;
	m_pCachedBTree = NULL;
	m_pKeyColl = NULL;
	m_uiDirtyRowCount = 0;
	m_hWaitSem = F_SEM_NULL;
	
	m_bKrefSetup = FALSE;
	m_pKrefTbl = NULL;
	m_uiKrefTblSize = 0;
	m_uiKrefCount = 0;
	m_uiTotalKrefBytes = 0;
	m_pucKrefKeyBuf = NULL;
	m_pKrefPool = NULL;
	m_bReuseKrefPool = FALSE;
	m_bKrefCompoundKey = FALSE;
	m_pKrefReset = NULL;

	m_tmpKrefPool.poolInit( 8192);
	m_tempPool.poolInit( SFLM_MAX_KEY_SIZE * 4);	
}

/***************************************************************************
Desc:	Allocates and initializes an F_Db object for a database which
		is to be opened or created.
****************************************************************************/
RCODE F_DbSystem::allocDb(
	F_Db **	ppDb,
	FLMBOOL	bInternalOpen)
{
	RCODE		rc = NE_SFLM_OK;
	F_Db *	pDb = NULL;

	*ppDb = NULL;
	
	// Allocate the F_Db object.
	
	if ((pDb = f_new F_Db( bInternalOpen)) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_semCreate( &pDb->m_hWaitSem)))
	{
		goto Exit;
	}

	*ppDb = pDb;
	pDb = NULL;

Exit:

	if (pDb)
	{
		pDb->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: This routine performs all of the necessary steps to complete
		a create or open of a database, including notifying other threads
		waiting for the open or create to complete.
NOTE:	If RC_BAD( rc), this routine will delete the F_Db object.
****************************************************************************/
void F_Db::completeOpenOrCreate(
	RCODE		rc,
	FLMBOOL	bNewDatabase
	)
{
	if (RC_OK( rc))
	{

		// If this is a newly created F_Database, we need to notify any
		// threads waiting for the database to be created or opened that
		// the create or open is now complete.

		if (bNewDatabase)
		{
			f_mutexLock( gv_SFlmSysData.hShareMutex);
			m_pDatabase->newDatabaseFinish( NE_SFLM_OK);
			f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		}
	}
	else
	{
		F_Database *	pDatabase = m_pDatabase;

		// Temporarily increment the open count on the F_Database structure
		// so that it will NOT be freed when pDb is freed below.

		if (bNewDatabase)
		{
			f_mutexLock( gv_SFlmSysData.hShareMutex);
			pDatabase->m_uiOpenIFDbCount++;
			f_mutexUnlock( gv_SFlmSysData.hShareMutex);
		}

		// NOTE: Cannot access this F_Db object after this!
		// Must do this before potentially deleting the F_Database object
		// below, so that the F_Db object will unlink itself from
		// the F_Database object.
		Release();

		// If we allocated the F_Database object, notify any
		// waiting threads.

		if (bNewDatabase)
		{
			f_mutexLock( gv_SFlmSysData.hShareMutex);

			// Decrement the use count to compensate for the increment
			// that occurred above.

			pDatabase->m_uiOpenIFDbCount--;

			// If this is a newly created F_Database, we need to notify any
			// threads waiting for the database to be created or opened that
			// the create or open is now complete.

			pDatabase->newDatabaseFinish( rc);
			pDatabase->freeDatabase();
			f_mutexUnlock ( gv_SFlmSysData.hShareMutex);
		}
	}
}

/****************************************************************************
Desc:	Returns the length of the base part of a database name.  If the
		name ends with a '.' or ".db", this will not be included in the
		returned length.
****************************************************************************/
void flmGetDbBasePath(
	char *			pszBaseDbName,
	const char *	pszDbName,
	FLMUINT *		puiBaseDbNameLen)
{
	FLMUINT			uiBaseLen = f_strlen( pszDbName);

	if( uiBaseLen <= 3 || 
		f_stricmp( &pszDbName[ uiBaseLen - 3], ".db") != 0)
	{
		if( pszDbName[ uiBaseLen - 1] == '.')
		{
			uiBaseLen--;
		}
	}
	else
	{
		uiBaseLen -= 3;
	}

	f_memcpy( pszBaseDbName, pszDbName, uiBaseLen);
	pszBaseDbName[ uiBaseLen] = 0;

	if( puiBaseDbNameLen)
	{
		*puiBaseDbNameLen = uiBaseLen;
	}
}

/****************************************************************************
Desc: This routine will open a database, returning a pointer to an F_Db
		object that can be used to access it.
****************************************************************************/
RCODE F_DbSystem::openDatabase(
	F_Database *			pDatabase,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszPassword,
	FLMUINT					uiOpenFlags,
	FLMBOOL					bInternalOpen,
	IF_RestoreClient *	pRestoreObj,
	IF_RestoreStatus *	pRestoreStatus,
	IF_FileHdl *			pLockFileHdl,
	F_Db **					ppDb)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bNewDatabase = FALSE;
	FLMBOOL			bMutexLocked = FALSE;
	F_Db *			pDb = NULL;
	FLMBOOL			bNeedToOpen = FALSE;

	// Allocate and initialize an F_Db object.

	if (RC_BAD( rc = allocDb( &pDb, bInternalOpen)))
	{
		goto Exit;
	}

	f_mutexLock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Free any unused structures that have been unused for the maximum
	// amount of time.  May unlock and re-lock the global mutex.

	checkNotUsedObjects();

	// Look up the file using findDatabase to see if we already
	// have the file open.

	if (!pDatabase)
	{
		bNeedToOpen = TRUE;

		// May unlock and re-lock the global mutex.

		if (RC_BAD( rc = findDatabase( pszDbPath, pszDataDir, &pDatabase)))
		{
			goto Exit;
		}
	}

	if (pDatabase)
	{
		if( RC_BAD( rc = pDatabase->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}
	}

	if (!pDatabase)
	{
		if (RC_BAD( rc = allocDatabase( pszDbPath, pszDataDir, FALSE, &pDatabase)))
		{
			goto Exit;
		}
		flmAssert( !pLockFileHdl);
		bNewDatabase = TRUE;
	}
	else if( pLockFileHdl)
	{
		flmAssert( pDatabase);
		flmAssert( !pDatabase->m_uiOpenIFDbCount);
		flmAssert( pDatabase->m_uiFlags & DBF_BEING_OPENED);

		pDatabase->m_pLockFileHdl = pLockFileHdl;

		// Set to NULL to prevent lock file from being released below

		pLockFileHdl = NULL;

		bNewDatabase = TRUE;
		bNeedToOpen = TRUE;
	}
	else
	{
		FLMBOOL	bWaited = FALSE;
		flmAssert( !pLockFileHdl);

		if (RC_BAD( rc = pDatabase->verifyOkToUse( &bWaited)))
		{
			goto Exit;
		}

		if (bWaited)
		{
			bNewDatabase = FALSE;
			bNeedToOpen = FALSE;
		}
	}

	// Link the F_Db object to the F_Database object.

	rc = pDb->linkToDatabase( pDatabase);
	f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = FALSE;
	if (RC_BAD(rc))
	{
		goto Exit;
	}

	(void)flmStatGetDb( &pDb->m_Stats, pDatabase,
							0, &pDb->m_pDbStats, NULL, NULL);

	if (bNeedToOpen)
	{
		if (RC_BAD( rc = pDatabase->physOpen( 
			pDb, pszDbPath, pszRflDir, pszPassword, uiOpenFlags,
			bNewDatabase, pRestoreObj, pRestoreStatus)))
		{
			goto Exit;
		}
	}

	// Start a checkpoint thread

	if( bNewDatabase && !(uiOpenFlags & SFLM_DONT_REDO_LOG))
	{
		flmAssert( !pDatabase->m_pCPThrd);
		flmAssert( !pDatabase->m_pMaintThrd);

		if( RC_BAD( rc = pDatabase->startCPThread()))
		{
			goto Exit;
		}

		if( !(uiOpenFlags & SFLM_DONT_RESUME_THREADS))
		{
			if( RC_BAD( rc = pDb->startBackgroundIndexing()))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDatabase->startMaintThread()))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}

	if (pLockFileHdl)
	{
		pLockFileHdl->Release();
	}

	if (pDb)
	{
		// completeOpenOrCreate will delete pDb if RC_BAD( rc)

		pDb->completeOpenOrCreate( rc, bNewDatabase);

		if (RC_BAD( rc))
		{
			pDb = NULL;
		}
	}
	*ppDb = pDb;
	return( rc);
}

/****************************************************************************
Desc: This routine checks to see if it is OK for another F_Db to use an
		F_Database object.
		If so, it increments the database's use counter.  NOTE: This routine
		assumes that the calling routine has locked the global mutex.
****************************************************************************/
RCODE F_Database::verifyOkToUse(
	FLMBOOL *	pbWaited)
{
	RCODE    rc = NE_SFLM_OK;
	F_SEM		hWaitSem = F_SEM_NULL;

	// Can't open the database if it is being closed by someone else.

	if (m_uiFlags & DBF_BEING_CLOSED)
	{
		rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
		goto Exit;
	}

	// If the file is in the process of being opened by another
	// thread, wait for the open to complete.

	if (m_uiFlags & DBF_BEING_OPENED)
	{
		if( RC_BAD( rc = f_semCreate( &hWaitSem)))
		{
			goto Exit;
		}
		
		*pbWaited = TRUE;
		if (RC_BAD( rc = flmWaitNotifyReq( 
			gv_SFlmSysData.hShareMutex, hWaitSem, &m_pOpenNotifies, (void *)0)))
		{
			// If flmWaitNotifyReq returns a bad RC, assume that the other
			// thread will unlock and free the F_Database object.  This
			// routine should only unlock the object if an error occurs at
			// some other point.

			goto Exit;
		}
	}
	else
	{
		*pbWaited = FALSE;
	}

Exit:

	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE flmCreateLckFile(
	const char *	pszFilePath,
	IF_FileHdl **	ppLockFileHdlRV)
{
	RCODE				rc = NE_SFLM_OK;
	char				szLockPath [F_PATH_MAX_SIZE];
	char				szDbBaseName [F_FILENAME_SIZE];
	char *			pszFileExt;
	IF_FileHdl *	pLockFileHdl = NULL;
	char				szFilePathStr[ F_PATH_MAX_SIZE];

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString( 
		pszFilePath, szFilePathStr)))
	{
		goto Exit;
	}

	// Extract the 8.3 name and put a .lck extension on it to create
	// the full path for the .lck file.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( 
		szFilePathStr, szLockPath, szDbBaseName)))
	{
		goto Exit;
	}
	pszFileExt = &szDbBaseName [0];
	while ((*pszFileExt) && (*pszFileExt != '.'))
		pszFileExt++;
	f_strcpy( pszFileExt, ".lck");

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
		szLockPath, szDbBaseName)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->createLockFile( 
		szLockPath, &pLockFileHdl)))
	{
		goto Exit;
	}

	*ppLockFileHdlRV = (IF_FileHdl *)pLockFileHdl;
	pLockFileHdl = NULL;
	
Exit:

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->closeFile();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}
	
	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE F_Database::getExclAccess(
	const char *	pszFilePath)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bNotifyWaiters = FALSE;
	FLMBOOL			bMutexLocked = FALSE;
	F_SEM				hWaitSem = F_SEM_NULL;

	// If m_pLockFileHdl is non-NULL, it means that we currently
	// have the database locked with a lock file.  There is no need to make
	// this test inside a mutex lock, because the lock file handle can only
	// be set to NULL when the use count goes to zero, meaning that the thread
	// that sets it to NULL will be the only thread accessing it.

	// However, it is possible that two or more threads will simultaneously
	// test m_pLockFileHdl and discover that it is NULL.  In that case,
	// we allow one thread to proceed and attempt to get a lock on the database
	// while the other threads wait to be notified of the results of the
	// attempt to lock the database.

	if (m_pLockFileHdl)
	{
		goto Exit;
	}

	lockMutex();
	bMutexLocked = TRUE;

	if (m_bBeingLocked)
	{
		// If the database is in the process of being locked by another
		// thread, wait for the lock to complete.  NOTE: flmWaitNotifyReq will
		// re-lock the mutex before returning.
		
		if( RC_BAD( rc = f_semCreate( &hWaitSem)))
		{
			goto Exit;
		}

		rc = flmWaitNotifyReq( m_hMutex, hWaitSem, &m_pLockNotifies, (void *)0);
		goto Exit;
	}

	// No other thread was attempting to lock the database, so
	// set this thread up to make the attempt.  Other threads
	// coming in at this point will be required to wait and
	// be notified of the results.

	m_bBeingLocked = TRUE;
	bNotifyWaiters = TRUE;
	unlockMutex();
	bMutexLocked = FALSE;
	if (RC_BAD( rc = flmCreateLckFile( pszFilePath, &m_pLockFileHdl)))
	{
		goto Exit;
	}

Exit:

	if (bNotifyWaiters)
	{
		FNOTIFY *	pNotify;
		F_SEM			hSem;

		// Notify any thread waiting on the lock what its status is.

		if( !bMutexLocked)
		{
			lockMutex();
			bMutexLocked = TRUE;
		}

		pNotify = m_pLockNotifies;
		while (pNotify)
		{
			*(pNotify->pRc) = rc;
			hSem = pNotify->hSem;
			pNotify = pNotify->pNext;
			f_semSignal( hSem);
		}

		m_bBeingLocked = FALSE;
		m_pLockNotifies = NULL;
		unlockMutex();
		bMutexLocked = FALSE;
	}

	if( bMutexLocked)
	{
		unlockMutex();
	}
	
	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine checks to see if it is OK for another FDB to use a file.
		If so, it increments the file's use counter.  NOTE: This routine
		assumes that the global mutex is NOT locked.
****************************************************************************/
RCODE F_Database::physOpen(
	F_Db *					pDb,
	const char *			pszFilePath,
	const char *			pszRflDir,
	const char *			pszPassword,
	FLMUINT					uiOpenFlags,
	FLMBOOL					bNewDatabase,		// Is this a new F_Database object?
	IF_RestoreClient *	pRestoreObj,
	IF_RestoreStatus *	pRestoreStatus)
{
	RCODE	rc = NE_SFLM_OK;

	// See if we need to read in the database header.  If the database was
	// already open (bNewDatabase == FALSE), we don't need to again.

	if (bNewDatabase)
	{

		// Read in the database header.

		if (RC_BAD( rc = readDbHdr( pDb->m_pDbStats, pDb->m_pSFileHdl,
											 (FLMBYTE *)pszPassword,
											 (uiOpenFlags & SFLM_ALLOW_LIMITED_MODE) ? TRUE : FALSE)))
		{
			goto Exit;
		}

		// Allocate the pRfl object.  Could not do this until this point
		// because we need to have the version number, block size, etc.
		// setup in the database header.

		flmAssert( !m_pRfl);

		if ((m_pRfl = f_new F_Rfl) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}

		if (RC_BAD( rc = m_pRfl->setup( this, pszRflDir)))
		{
			goto Exit;
		}
	}

	// We must have exclusive access.  Create a lock file for that
	// purpose, if there is not already a lock file.

	if (!m_pLockFileHdl)
	{
		if (RC_BAD( rc = getExclAccess( pszFilePath)))
		{
			goto Exit;
		}
	}

	// Do a recovery to ensure a consistent database
	// state before going any further.  The FO_DONT_REDO_LOG
	// flag is used ONLY by the VIEW program.

	if (bNewDatabase && !(uiOpenFlags & SFLM_DONT_REDO_LOG))
	{
		if (RC_BAD( rc = doRecover( pDb, pRestoreObj, pRestoreStatus)))
		{
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		(void)pDb->m_pSFileHdl->releaseFiles();
	}

	return( rc);
}

/****************************************************************************
Desc: This routine finishes up after creating a new F_Database object.  It
		will notify any other threads waiting for the operation to complete
		of the status of the operation.
****************************************************************************/
void F_Database::newDatabaseFinish(
	RCODE			OpenRc)  // Return code to send to other threads that are
								// waiting for the open to complete.
{
	FNOTIFY *	pNotify;
	F_SEM			hSem;

	// Notify anyone waiting on the operation what its status is.

	pNotify = m_pOpenNotifies;
	while (pNotify)
	{
		*(pNotify->pRc) = OpenRc;
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}

	m_pOpenNotifies = NULL;
	m_uiFlags &= (~(DBF_BEING_OPENED));
}

/****************************************************************************
Desc:		This routine is used to see if a file is already in use somewhere.
			This is only called for files which are opened directly by the
			application.
Notes:	This routine assumes that the global mutex is locked, but it
			may unlock and re-lock the mutex if needed.
****************************************************************************/
RCODE F_DbSystem::findDatabase(
	const char *	pszDbPath,
	const char *	pszDataDir,
	F_Database **	ppDatabase)
{
	RCODE				rc = NE_SFLM_OK;
	F_BUCKET *		pBucket;
	FLMUINT			uiBucket;
	FLMBOOL			bMutexLocked = TRUE;
	F_Database *	pDatabase;
	char				szDbPathStr1 [F_PATH_MAX_SIZE];
	char				szDbPathStr2 [F_PATH_MAX_SIZE];
	F_SEM				hWaitSem = F_SEM_NULL;

	*ppDatabase = NULL;

	// Normalize the path to a string before looking for it.
	// NOTE: On non-UNIX, non-WIN platforms, this will basically convert
	// the string to upper case.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString( 
		pszDbPath, szDbPathStr1)))
	{
		goto Exit;
	}

Retry:

	*ppDatabase = NULL;

	if( !bMutexLocked)
	{
		f_mutexLock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = TRUE;
	}

	pBucket = gv_SFlmSysData.pDatabaseHashTbl;
	uiBucket = f_strHashBucket( szDbPathStr1, pBucket, FILE_HASH_ENTRIES);
	pDatabase = (F_Database *)pBucket [uiBucket].pFirstInBucket;
	while (pDatabase)
	{
		// Compare the strings.  On non-Unix platforms we must use
		// f_stricmp, because case does not matter for file names
		// on those platforms.

#ifdef FLM_UNIX
		if( f_strcmp( szDbPathStr1, pDatabase->m_pszDbPath) == 0)
#else
		if( f_stricmp( szDbPathStr1, pDatabase->m_pszDbPath) == 0)
#endif
		{

			// Make sure data paths match.

			if (pszDataDir && *pszDataDir)
			{
				if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString(
					pszDataDir, szDbPathStr2)))
				{
					goto Exit;
				}
				
				if (pDatabase->m_pszDataDir)
				{
					// f_stricmp must be used on non-unix platforms because file
					// names are case insensitive on those platforms.
#ifdef FLM_UNIX
					if (f_strcmp( pDatabase->m_pszDataDir, szDbPathStr2) != 0)
#else
					if (f_stricmp( pDatabase->m_pszDataDir, szDbPathStr2) != 0)
#endif
					{
						rc = RC_SET( NE_SFLM_DATA_PATH_MISMATCH);
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET( NE_SFLM_DATA_PATH_MISMATCH);
					goto Exit;
				}
			}
			else if (pDatabase->m_pszDataDir)
			{
				rc = RC_SET( NE_SFLM_DATA_PATH_MISMATCH);
				goto Exit;
			}
			*ppDatabase = pDatabase;
			break;
		}
		pDatabase = pDatabase->m_pNext;
	}

	if (*ppDatabase && pDatabase->m_uiFlags & DBF_BEING_CLOSED)
	{
		if( RC_BAD( rc = f_semCreate( &hWaitSem)))
		{
			goto Exit;
		}
		
		// Put ourselves into the notify list and then re-try
		// the lookup when we wake up

		if (RC_BAD( rc = flmWaitNotifyReq( gv_SFlmSysData.hShareMutex, hWaitSem,
			&pDatabase->m_pCloseNotifies, (void *)0)))
		{
			goto Exit;
		}
		
		f_semDestroy( &hWaitSem);

		// The mutex will be locked at this point.  Re-try the lookup.
		// IMPORTANT NOTE: pDatabase will have been destroyed by this
		// time.  DO NOT use it for anything!

		goto Retry;
	}

Exit:

	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	// Make sure the global mutex is re-locked before exiting

	if( !bMutexLocked)
	{
		f_mutexLock( gv_SFlmSysData.hShareMutex);
	}
	

	return( rc);
}

/****************************************************************************
Desc: Make sure a database is NOT open.  If it is, return an error.
****************************************************************************/
RCODE F_DbSystem::checkDatabaseClosed(
	const char *	pszDbName,
	const char *	pszDataDir)
{
	RCODE				rc = NE_SFLM_OK;
	F_Database *	pDatabase;

	f_mutexLock( gv_SFlmSysData.hShareMutex);
	rc = findDatabase( pszDbName, pszDataDir, &pDatabase);
	f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	if (RC_BAD( rc))
	{
		goto Exit;
	}
	if (pDatabase)
	{
		rc = RC_SET( NE_SFLM_DATABASE_OPEN);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Constructor for F_Database object.
****************************************************************************/
F_Database::F_Database(
	FLMBOOL	bTempDb)
{
	m_krefPool.poolInit( DEFAULT_KREF_POOL_BLOCK_SIZE * 8);
	m_pNext = NULL;
	m_pPrev = NULL;
	m_pFirstSQLQuery = NULL;
	m_pLastSQLQuery = NULL;
	m_uiBlockSize = 0;
	m_uiDefaultLanguage = 0;
	m_uiMaxFileSize = 0;
	m_uiOpenIFDbCount = 0;
	m_bTempDb = bTempDb;
	m_pFirstDb = NULL;
	m_pszDbPath = NULL;
	m_pszDataDir = NULL;
	m_pSCacheList = NULL;
	m_pFirstRow = NULL;
	m_pLastRow = NULL;
	m_pLastDirtyRow = NULL;
	m_pPendingWriteList = NULL;
	m_pLastDirtyBlk = NULL;
	m_pFirstInLogList = NULL;
	m_pLastInLogList = NULL;
	m_uiLogListCount = 0;
	m_pFirstInNewList = NULL;
	m_pLastInNewList = NULL;
	m_uiNewCount = 0;
	m_uiDirtyCacheCount = 0;
	m_uiLogCacheCount = 0;
	m_ppBlocksDone = NULL;
	m_uiBlocksDoneArraySize = 0;
	m_uiBlocksDone = 0;
	m_pTransLogList = NULL;
	m_pOpenNotifies = NULL;
	m_pCloseNotifies = NULL;
	m_pDictList = NULL;
	m_bMustClose = FALSE;
	m_rcMustClose = NE_SFLM_OK;
	m_uiSigBitsInBlkSize = 0;
	if (!bTempDb)
	{
		m_uiFileExtendSize = SFLM_DEFAULT_FILE_EXTEND_SIZE;
	}
	else
	{
		m_uiFileExtendSize = 65536;
	}
	
	m_pRfl = NULL;
	
	f_memset( &m_lastCommittedDbHdr, 0, sizeof( m_lastCommittedDbHdr));
	f_memset( &m_checkpointDbHdr, 0, sizeof( m_checkpointDbHdr));
	f_memset( &m_uncommittedDbHdr, 0, sizeof( m_uncommittedDbHdr));
	
	m_pBufferMgr = NULL;
	m_pCurrLogBuffer = NULL;
	m_uiCurrLogWriteOffset = 0;
	m_uiCurrLogBlkAddr = 0;
	m_pDbHdrWriteBuf = NULL;
	m_pucUpdBuffer = NULL;
	m_uiUpdBufferSize = 0;
	m_uiUpdByteCount = 0;
	m_uiUpdCharCount = 0;
	m_ePendingDataType = SFLM_UNKNOWN_TYPE;
	m_pPendingBTree = NULL;
	m_pucBTreeTmpBlk = NULL;
	m_pucBTreeTmpDefragBlk = NULL;
	m_pucEntryArray = NULL;
	m_pucSortedArray = NULL;
	m_pucBtreeBuffer = NULL;
	m_pucReplaceStruct = NULL;
	m_pDatabaseLockObj = NULL;
	m_pWriteLockObj = NULL;
	m_pLockFileHdl = NULL;
	m_pLockNotifies = NULL;
	m_bBeingLocked = FALSE;
	m_pFirstReadTrans = NULL;
	m_pLastReadTrans = NULL;
	m_pFirstKilledTrans = NULL;
	m_uiFirstLogBlkAddress = 0;
	m_uiFirstLogCPBlkAddress = 0;
	m_uiLastCheckpointTime = 0;
	m_pCPThrd = NULL;
	m_pCPInfo = NULL;
	m_CheckpointRc = NE_SFLM_OK;
	m_uiBucket = 0;
	m_uiFlags = 0;
	m_bBackupActive = FALSE;
	m_pMaintThrd = NULL;
	m_hMaintSem = F_SEM_NULL;
	m_bAllowLimitedMode = FALSE;
	m_bInLimitedMode = FALSE;
	m_pszDbPasswd = NULL;
	m_pWrappingKey = NULL;
	m_bHaveEncKey = FALSE;
	m_rcLimitedCode = NE_SFLM_OK;
	m_hMutex = F_MUTEX_NULL;
}

/****************************************************************************
Desc:	This destructor frees all of the structures associated with an
		F_Database object.
		Whoever called this routine has already determined that it is safe
		to do so.
Notes:	The global mutex is assumed to be locked when entering the
			routine.  It may be unlocked and re-locked before the routine
			exits, however.
****************************************************************************/
F_Database::~F_Database()
{
	FNOTIFY *	pCloseNotifies;
	F_Dict *    pDict;
	F_Dict *		pTmpDict;

	// At this point, the use count better be zero

	flmAssert( !m_uiOpenIFDbCount);

	// Shut down all background threads before shutting down the CP thread.

	shutdownDatabaseThreads();

	if (m_pRfl)
	{
		m_pRfl->closeFile();
	}

	// At this point, the use count better be zero

	flmAssert( !m_uiOpenIFDbCount);

	// Unlock the mutex

	f_mutexUnlock( gv_SFlmSysData.hShareMutex);

	// Shut down the checkpoint thread

	if( m_pCPThrd)
	{
		m_pCPThrd->stopThread();
		m_pCPThrd->Release();
		m_pCPThrd = NULL;
	}

	// Unlink all of the F_Dict objects that are connected to the
	// database.

	lockMutex();
	while (m_pDictList)
	{
		m_pDictList->unlinkFromDatabase();
	}
	unlockMutex();

	// Take the file out of its name hash bucket, if any.

	if (m_uiBucket != 0xFFFF)
	{
		f_mutexLock( gv_SFlmSysData.hShareMutex);
		if (m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		else
		{
			gv_SFlmSysData.pDatabaseHashTbl[ m_uiBucket].pFirstInBucket = m_pNext;
		}

		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
		m_uiBucket = 0xFFFF;
		
		// After this point, we should not need to keep the global mutex locked
		// because the F_Database is no longer visible to any thread to find in
		// the hash table.
	
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}
	
	// Shouldn't have any queries at this point.  But we will be nice in case
	// we do and will unlink the queries from the list

	flmAssert( !m_pFirstSQLQuery);
	while (m_pFirstSQLQuery)
	{
		SQLQuery *	pQuery = m_pFirstSQLQuery;

		m_pFirstSQLQuery = m_pFirstSQLQuery->m_pNext;
		pQuery->m_pPrev = NULL;
		pQuery->m_pNext = NULL;
		pQuery->m_pDatabase = NULL;
	}

	// Free the RFL data, if any.

	if (m_pRfl)
	{
		m_pRfl->Release();
		m_pRfl = NULL;
	}

	flmAssert( m_pOpenNotifies == NULL);
	m_pOpenNotifies = NULL;

	// Save pCloseNotifies -- we will notify any waiters once the
	// F_Database has been freed.

	pCloseNotifies = m_pCloseNotifies;

	// Free any dictionary usage structures associated with the database.

	pDict = m_pDictList;
	while (pDict)
	{
		pTmpDict = pDict;
		pDict = pDict->getNext();
		pTmpDict->Release();
	}
	m_pDictList = NULL;

	// Free any shared cache associated with the database.
	// IMPORTANT NOTE:
	// Cannot have the global mutex locked when these are called because
	// these routines lock the block cache mutex and the node cache mutex.
	// If both the global mutex and the block or node cache mutexes are to be
	// locked, the rule is that the block or node cache mutex must be locked
	// before locking the global mutex.  This is because neededByReadTrans
	// will end up doing it in this order - when neededByReadTrans is called
	// either the block or node cache mutex is already locked, and it will
	// additionally lock the global mutex.  Since that order is already
	// required, we cannot have anyone else attempting to lock the mutexes
	// in a different order.
	
	freeBlockCache();
	freeRowCache();
	
	// Release the lock objects.

	if (m_pWriteLockObj)
	{
		m_pWriteLockObj->Release();
		m_pWriteLockObj = NULL;
	}

	if (m_pDatabaseLockObj)
	{
		m_pDatabaseLockObj->Release();
		m_pDatabaseLockObj = NULL;
	}

	// Close and delete the lock file.

	if (m_pLockFileHdl)
	{
		(void)m_pLockFileHdl->closeFile();
		m_pLockFileHdl->Release();
		m_pLockFileHdl = NULL;
	}

	// Free the write buffer managers.

	if (m_pBufferMgr)
	{
		m_pBufferMgr->Release();
		m_pBufferMgr = NULL;
	}

	// Free the log header write buffer

	if (m_pDbHdrWriteBuf)
	{
		f_freeAlignedBuffer( (void **)&m_pDbHdrWriteBuf);
	}

	// Free the update buffer

	if (m_pucUpdBuffer)
	{
		f_free( &m_pucUpdBuffer);
		m_uiUpdBufferSize = 0;
	}
	
	m_krefPool.poolFree();

	if (m_ppBlocksDone)
	{
		f_free( &m_ppBlocksDone);
		m_uiBlocksDoneArraySize = 0;
	}

	// Notify waiters that the F_Database is gone

	while (pCloseNotifies)
	{
		F_SEM		hSem;

		*(pCloseNotifies->pRc) = NE_SFLM_OK;
		hSem = pCloseNotifies->hSem;
		pCloseNotifies = pCloseNotifies->pNext;
		f_semSignal( hSem);
	}

	f_free( &m_pszDbPath);
	
	// Encryption stuff
	if (m_pszDbPasswd)
	{
		f_free( &m_pszDbPasswd);
	}
	if (m_pWrappingKey)
	{
		delete m_pWrappingKey;
	}
	
	flmAssert( !m_pFirstRow && !m_pLastRow && !m_pLastDirtyRow);
	
	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
	
	// Global mutex is still expected to be locked at this point
	
	f_mutexLock( gv_SFlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	This frees an F_Database object.
Note:	The global mutex is assumed to be locked when entering the
		routine.  It may be unlocked and re-locked during the destructor,
		however.  For this reason, this routine should be called instead of
		directly deleting a database object - i.e., delete pDatabase.
****************************************************************************/
void F_Database::freeDatabase( void)
{

	// See if another thread is in the process of freeing
	// this F_Database.  It is possible for this to happen, since
	// the monitor thread may have selected this F_Database to be
	// freed because it has been unused for a period of time.
	// At the same time, a foreground thread could have called
	// FlmConfig to close all unused F_Databases.  Since the
	// destructor for the F_Database object
	// may unlock and re-lock the mutex, there is a small window
	// of opportunity for both threads to try to free the same
	// F_Database. -- Therefore, we must do this check while the
	// mutex is still locked.

	if (m_uiFlags & DBF_BEING_CLOSED)
	{
		return;
	}

	// Set the DBF_BEING_CLOSED flag

	m_uiFlags |= DBF_BEING_CLOSED;
	Release();
}

/****************************************************************************
Desc: This routine sets up a new F_Database object, allocating member
		variables, linking into lists, etc.
		NOTE: This routine assumes that the global mutex has already
		been locked. It may unlock it temporarily if there is an error,
		but will always relock it before exiting.
****************************************************************************/
RCODE F_Database::setupDatabase(
	const char *		pszDbPath,
	const char *		pszDataDir)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiAllocLen;
	FLMUINT			uiDbNameLen;
	FLMUINT			uiDirNameLen;
	char				szDbPathStr[ F_PATH_MAX_SIZE];
	char				szDataDirStr[ F_PATH_MAX_SIZE];

	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString( 
		pszDbPath, szDbPathStr)))
	{
		goto Exit;
	}
	uiDbNameLen = f_strlen( szDbPathStr) + 1;

	if( pszDataDir && *pszDataDir)
	{
		if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString( 
			pszDataDir, szDataDirStr)))
		{
			goto Exit;
		}
		uiDirNameLen = f_strlen( szDataDirStr) + 1;

	}
	else
	{
		szDataDirStr[0] = 0;
		uiDirNameLen = 0;
	}

	
	if (RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}


	uiAllocLen = (FLMUINT)(uiDbNameLen + uiDirNameLen);
	if (RC_BAD( rc = f_alloc( uiAllocLen, &m_pszDbPath)))
	{
		goto Exit;
	}
	
	// Allocate a buffer for writing the DB header
	// If we are a temporary database, there is no need
	// for this allocation.

	if (!m_bTempDb)
	{
		if( RC_BAD( rc = f_allocAlignedBuffer( 
			SFLM_MAX_BLOCK_SIZE, (void **)&m_pDbHdrWriteBuf)))
		{
			goto Exit;
		}
	}

	// Setup the write buffer managers.
	
	if( RC_BAD( rc = FlmAllocIOBufferMgr( MAX_PENDING_WRITES, 
		MAX_WRITE_BUFFER_BYTES, FALSE, &m_pBufferMgr)))
	{
		goto Exit;
	}

	// Initialize members of F_Database object.

	m_uiBucket = 0xFFFF;
	m_uiFlags = DBF_BEING_OPENED;

	// Copy the database name and directory.
	// NOTE: uiDbNameLen includes the null terminating byte.
	// and uiDirNameLen includes the null terminating byte.

	f_memcpy( m_pszDbPath, szDbPathStr, uiDbNameLen);
	if (uiDirNameLen)
	{
		m_pszDataDir = m_pszDbPath + uiDbNameLen;
		f_memcpy( m_pszDataDir, szDataDirStr, uiDirNameLen);
	}

	// Link the file into the various lists it needs to be linked into.

	if (RC_BAD( rc = linkToBucket()))
	{
		goto Exit;
	}

	// Allocate a lock object for write locking.
	
	if( RC_BAD( rc = FlmAllocLockObject( &m_pWriteLockObj)))
	{
		goto Exit;
	}

	// Allocate a lock object for file locking.
	
	if( RC_BAD( rc = FlmAllocLockObject( &m_pDatabaseLockObj)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine allocates a new F_Database object and links it
		into its hash buckets.
		NOTE: This routine assumes that the global mutex has already
		been locked. It may unlock it temporarily if there is an error,
		but will always relock it before exiting.
****************************************************************************/
RCODE F_DbSystem::allocDatabase(
	const char *	pszDbPath,
	const char *	pszDataDir,
	FLMBOOL			bTempDb,
	F_Database **	ppDatabase)
{
	RCODE				rc = NE_SFLM_OK;
	F_Database *	pDatabase = NULL;

	if ((pDatabase = f_new F_Database( bTempDb)) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pDatabase->setupDatabase( pszDbPath, pszDataDir)))
	{
		goto Exit;
	}

	*ppDatabase = pDatabase;

Exit:

	if (RC_BAD( rc))
	{
		if (pDatabase)
		{
			pDatabase->freeDatabase();
		}
	}
	return( rc);
}

/***************************************************************************
Desc: This routine reads the header information for an existing
		flaim database and makes sure we have a valid database.
*****************************************************************************/
RCODE F_Database::readDbHdr(
	SFLM_DB_STATS *		pDbStats,
	F_SuperFileHdl *		pSFileHdl,
	FLMBYTE *				pszPassword,
	FLMBOOL					bAllowLimited)
{
	RCODE				rc = NE_SFLM_OK;
	IF_FileHdl *	pCFileHdl = NULL;

	if (RC_BAD( rc = pSFileHdl->getFileHdl( 0, FALSE, &pCFileHdl)))
	{
		goto Exit;
	}

	// Read and verify the database header.

	if (RC_BAD( rc = flmReadAndVerifyHdrInfo( pDbStats, pCFileHdl,
									&m_lastCommittedDbHdr)))
	{
		goto Exit;
	}
	m_uiBlockSize = (FLMUINT)m_lastCommittedDbHdr.ui16BlockSize;
	m_uiDefaultLanguage = (FLMUINT)m_lastCommittedDbHdr.ui8DefaultLanguage;
	m_uiMaxFileSize = (FLMUINT)m_lastCommittedDbHdr.ui32MaxFileSize;
	m_uiSigBitsInBlkSize = calcSigBits( m_uiBlockSize);

	// Initialize the master database key from the database header
	m_bAllowLimitedMode = bAllowLimited;

	if (pszPassword && *pszPassword)
	{
		if (m_pszDbPasswd)
		{
			f_free( &m_pszDbPasswd);
		}
		if ( RC_BAD( rc = f_alloc( 
			(f_strlen( (const char *)pszPassword) + 1), &m_pszDbPasswd)))
		{
			goto Exit;
		}
		
		f_strcpy( (char *)m_pszDbPasswd, (const char *)pszPassword);
	}

	if ((m_pWrappingKey = f_new F_CCS()) == NULL)
	{
		RC_SET( rc = NE_SFLM_MEM);
		goto Exit;
	}
		
	if( RC_OK( rc = m_pWrappingKey->init( TRUE, SFLM_AES_ENCRYPTION)))
	{
		// If the key was encrypted in a password, then the pszPassword parameter better
		// be the key used to encrypt it.  If the key was not encrypted in a password,
		// then pszPassword parameter should be NULL.
		rc = m_pWrappingKey->setKeyFromStore(
								m_lastCommittedDbHdr.ucDbKey,
								pszPassword, NULL);
	}
	
	if (RC_BAD( rc))
	{
		// NE_SFLM_UNSUPPORTED_FEATURE is returned when we've been compiled
		// without NICI support
		if ((rc == NE_SFLM_UNSUPPORTED_FEATURE) || bAllowLimited)
		{
			m_bInLimitedMode = TRUE;
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}
	// Note that we might still end up in limited mode if we can't verify all the keys
	// that are stored in the dictionary.

Exit:

	// Need to close the .db file so that we can set the block size.
	// This will allow direct I/O to be used when accessing the file later.

	if (pCFileHdl)
	{
		(void)pSFileHdl->releaseFiles();
//		JMC - FIXME: commented out due to missing functionality in flaimtk.h
//		pSFileHdl->setBlockSize( m_uiBlockSize);
	}

	return( rc);
}

/***************************************************************************
Desc: This routine frees a CP_INFO structure and all associated data.
*****************************************************************************/
FSTATIC void flmFreeCPInfo(
	CP_INFO **	ppCPInfoRV)
{
	CP_INFO *	pCPInfo;

	if ((pCPInfo = *ppCPInfoRV) != NULL)
	{
		if (pCPInfo->pSFileHdl)
		{
			pCPInfo->pSFileHdl->Release();
		}

		if (pCPInfo->bStatsInitialized)
		{
			flmStatFree( &pCPInfo->Stats);
		}

		if( pCPInfo->hWaitSem != F_SEM_NULL)
		{
			f_semDestroy( &pCPInfo->hWaitSem);
		}

		f_free( ppCPInfoRV);
	}
}

/***************************************************************************
Desc: This routine begins a thread that will do checkpoints for the
		passed in database.  It gives the thread its own FLAIM session and its
		own handle to the database.
*****************************************************************************/
RCODE F_Database::startCPThread( void)
{
	RCODE						rc = NE_SFLM_OK;
	CP_INFO *				pCPInfo = NULL;
	char						szThreadName[ F_PATH_MAX_SIZE];
	char						szBaseName[ 32];
	F_SuperFileClient *	pSFileClient = NULL;

	// Allocate a CP_INFO structure that will be passed into the
	// thread when it is created.

	if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( CP_INFO)), &pCPInfo)))
	{
		goto Exit;
	}
	pCPInfo->pDatabase = this;
	
	// Create a "wait" semaphore
	
	if( RC_BAD( rc = f_semCreate( &pCPInfo->hWaitSem)))
	{
		goto Exit;
	}

	// Allocate a super file handle.

	if( (pCPInfo->pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	if( (pSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pSFileClient->setup( m_pszDbPath, m_pszDataDir)))
	{
		goto Exit;
	}

	// Set up the super file

	if (RC_BAD( rc = pCPInfo->pSFileHdl->setup( pSFileClient,
		gv_SFlmSysData.pFileHdlCache, gv_SFlmSysData.uiFileOpenFlags,
		gv_SFlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	if (m_lastCommittedDbHdr.ui32DbVersion)
	{
//		JMC - FIXME: commented out due to missing functionality in flaimtk.h
//		pCPInfo->pSFileHdl->setBlockSize( m_uiBlockSize);
	}

	f_memset( &pCPInfo->Stats, 0, sizeof( SFLM_STATS));
	pCPInfo->bStatsInitialized = TRUE;

	// Generate the thread name

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( m_pszDbPath, 
		szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( (char *)szThreadName, "Checkpoint (%s)", (char *)szBaseName);

	// Start the checkpoint thread.

	if (RC_BAD( rc = gv_SFlmSysData.pThreadMgr->createThread( &m_pCPThrd,
		flmCPThread, szThreadName, gv_SFlmSysData.uiCheckpointThreadGroup,
		0, pCPInfo, NULL, 32000)))
	{
		goto Exit;
	}

	m_pCPInfo = pCPInfo;
	pCPInfo = NULL;

Exit:

	if( pCPInfo)
	{
		flmFreeCPInfo( &pCPInfo);
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc: Try to perform a checkpoint on the database.  Returns TRUE if we need
		to terminate.
****************************************************************************/
FLMBOOL F_Database::tryCheckpoint(
	IF_Thread *			pThread,
	CP_INFO *			pCPInfo)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bTerminate = FALSE;
	FLMBOOL				bForceCheckpoint;
	eForceCPReason		eForceReason;
	FLMUINT				uiCurrTime;
	SFLM_DB_STATS *	pDbStats;

	// See if we should terminate the thread.

	if (pThread->getShutdownFlag())
	{
		// Set terminate flag to TRUE and then see if
		// we have been set up to do one final checkpoint
		// to flush dirty buffers to disk.

		bTerminate = TRUE;
	}

	// Determine if we need to force a checkpoint.

	bForceCheckpoint = FALSE;
	eForceReason = SFLM_CP_NO_REASON;
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();
	
	if (bTerminate)
	{
		bForceCheckpoint = TRUE;
		eForceReason = SFLM_CP_SHUTTING_DOWN_REASON;
	}
	else if (!m_pRfl->seeIfRflVolumeOk() || RC_BAD( m_CheckpointRc))
	{
		bForceCheckpoint = TRUE;
		eForceReason = SFLM_CP_RFL_VOLUME_PROBLEM;
	}
	else if ((FLM_ELAPSED_TIME( uiCurrTime, m_uiLastCheckpointTime) >=
				 gv_SFlmSysData.uiMaxCPInterval) ||
				(!gv_SFlmSysData.uiMaxCPInterval))
	{
		bForceCheckpoint = TRUE;
		eForceReason = SFLM_CP_TIME_INTERVAL_REASON;
	}

	if (gv_SFlmSysData.Stats.bCollectingStats)
	{

		// Statistics are being collected for the system.  Therefore,
		// if we are not currently collecting statistics in the
		// start.  If we were collecting statistics, but the
		// start time was earlier than the start time in the system
		// statistics structure, reset the statistics.

		if (!pCPInfo->Stats.bCollectingStats)
		{
			flmStatStart( &pCPInfo->Stats);
		}
		else if (pCPInfo->Stats.uiStartTime <
					gv_SFlmSysData.Stats.uiStartTime)
		{
			flmStatReset( &pCPInfo->Stats, FALSE);
		}
		(void)flmStatGetDb( &pCPInfo->Stats, this,
						0, &pDbStats, NULL, NULL);
	}
	else
	{
		pDbStats = NULL;
	}

	// Lock write object - If we are forcing a checkpoint
	// wait until we get the lock.  Otherwise, if we can't get
	// the lock without waiting, don't do anything.

	if (bForceCheckpoint ||
		 (gv_SFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache &&
		  (m_uiDirtyCacheCount + m_uiLogCacheCount) * m_uiBlockSize >
			gv_SFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache))
	{
		if (RC_BAD( rc = dbWriteLock( pCPInfo->hWaitSem, pDbStats, FLM_NO_TIMEOUT)))
		{

			// THIS SHOULD NEVER HAPPEN BECAUSE dbWriteLock will
			// wait forever for the lock!

			RC_UNEXPECTED_ASSERT( rc);
			goto Exit;
		}
		pThread->setThreadStatusStr( "Forcing checkpoint");

		// Must wait for any RFL writes to complete.

		(void)m_pRfl->seeIfRflWritesDone( pCPInfo->hWaitSem, TRUE);
	}
	else
	{
		if (RC_BAD( dbWriteLock( pCPInfo->hWaitSem, pDbStats, 0)))
		{
			goto Exit;
		}

		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);

		// See if we actually need to do the checkpoint.  If the
		// current transaction ID and the last checkpoint transaction
		// ID are the same, no updates have occurred that would require
		// a checkpoint to take place.

		if (m_lastCommittedDbHdr.ui64RflLastCPTransID ==
			 m_lastCommittedDbHdr.ui64CurrTransID ||
			 !m_pRfl->seeIfRflWritesDone( pCPInfo->hWaitSem, FALSE))
		{
			dbWriteUnlock();
			goto Exit;
		}
	}

	// Do the checkpoint.

	(void)doCheckpoint( pCPInfo->hWaitSem, pDbStats, pCPInfo->pSFileHdl, FALSE,
						bForceCheckpoint, eForceReason, 0, 0);
	if (pDbStats)
	{
		(void)flmStatUpdate( &pCPInfo->Stats);
	}

	dbWriteUnlock();

	// Set the thread's status

	pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);

Exit:

	return( bTerminate);
}

/****************************************************************************
Desc: This routine functions as a thread.  It monitors open files and
		frees up files which have been closed longer than the maximum
		close time.
****************************************************************************/
FSTATIC RCODE SQFAPI flmCPThread(
	IF_Thread *		pThread)
{
	CP_INFO *			pCPInfo = (CP_INFO *)pThread->getParm1();
	F_Database *		pDatabase = pCPInfo->pDatabase;

	pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);
	for (;;)
	{
		f_sleep( 1000);
		if (pDatabase->tryCheckpoint( pThread, pCPInfo))
		{
			break;
		}
	}

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);

	flmFreeCPInfo( &pCPInfo);
	return( NE_SFLM_OK);
}

/****************************************************************************
Desc: Recover a database on startup.
****************************************************************************/
RCODE F_Database::doRecover(
	F_Db *					pDb,
	IF_RestoreClient *	pRestoreObj,
	IF_RestoreStatus *	pRestoreStatus)
{
	RCODE				rc = NE_SFLM_OK;
	SFLM_DB_HDR *	pLastCommittedDbHdr;

	// At this point, m_lastCommittedDbHdr contains the header
	// that was read from disk, which will be the state of the
	// header as of the last completed checkpoint.  Therefore,
	// we copy it into m_checkpointDbHdr.

	pLastCommittedDbHdr = &m_lastCommittedDbHdr;
	f_memcpy( &m_checkpointDbHdr, pLastCommittedDbHdr, sizeof( SFLM_DB_HDR));

	// Do a physical rollback on the database to restore the last
	// checkpoint.

	if (RC_BAD( rc = pDb->physRollback(
							(FLMUINT)pLastCommittedDbHdr->ui32RblEOF,
							(FLMUINT)pLastCommittedDbHdr->ui32RblFirstCPBlkAddr,
							TRUE,
							pLastCommittedDbHdr->ui64RflLastCPTransID)))
	{
		goto Exit;
	}
	pLastCommittedDbHdr->ui32RblFirstCPBlkAddr = 0;
	pLastCommittedDbHdr->ui32RblEOF = (FLMUINT32)m_uiBlockSize;
	if (RC_BAD( rc = writeDbHdr( pDb->m_pDbStats, pDb->m_pSFileHdl,
								pLastCommittedDbHdr,
								&m_checkpointDbHdr, TRUE)))
	{
		goto Exit;
	}

	// Set uiFirstLogCPBlkAddress to zero to indicate that no
	// physical blocks have been logged for the current checkpoint.
	// The above call to flmPhysRollback will have set the log header
	// to the same thing.

	m_uiFirstLogCPBlkAddress = 0;

	// Set the checkpointDbHdr to be the same as the log header

	f_memcpy( &m_checkpointDbHdr, pLastCommittedDbHdr, sizeof( SFLM_DB_HDR));

	// Open roll forward log and redo the transactions that
	// occurred since the last checkpoint, if any.

	if( RC_BAD( rc = m_pRfl->recover( pDb, pRestoreObj, pRestoreStatus)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the open database statement.  The "OPEN DATABASE" keywords
//			have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processOpenDatabase( void)
{
	RCODE					rc = NE_SFLM_OK;
	char					szDatabaseName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDatabaseNameLen;
	char					szDataDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDataDirNameLen;
	char					szRflDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiRflDirNameLen;
	F_DbSystem			dbSystem;
	char					szPassword [300];
	FLMUINT				uiPasswordLen;
	FLMUINT				uiFlags;
	char					szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT				uiTokenLineOffset;
	
	// SYNTAX: OPEN DATABASE databasename
	// [DATA_DIR=<DataDirName>] [RFL_DIR=<RflDirName>]
	// [PASSWORD=<password>] [ALLOW_LIMITED]

	// Whitespace must follow the "OPEN DATABASE"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the database name.

	if (RC_BAD( rc = getUTF8String( FALSE, TRUE, (FLMBYTE *)szDatabaseName,
							sizeof( szDatabaseName),
							&uiDatabaseNameLen, NULL, NULL)))
	{
		goto Exit;
	}
	
	szDataDirName [0] = 0;
	szRflDirName [0] = 0;
	szPassword [0] = 0;
	uiFlags = 0;
	
	// See if there are any options
	
	for (;;)
	{
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
									&uiTokenLineOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if (f_stricmp( szToken, SFLM_DATA_DIR_STR) == 0)
		{
			if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szDataDirName,
									sizeof( szDataDirName),
									&uiDataDirNameLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, SFLM_RFL_DIR_STR) == 0)
		{
			if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szRflDirName,
									sizeof( szRflDirName),
									&uiRflDirNameLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, SFLM_PASSWORD_STR) == 0)
		{
			if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szPassword,
									sizeof( szPassword),
									&uiPasswordLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, SFLM_ALLOW_LIMITED_STR) == 0)
		{
			uiFlags |= SFLM_ALLOW_LIMITED_MODE;
		}
		else
		{
			// Move the line offset back to the beginning of the token
			// so it can be processed by the next SQL statement in the
			// stream.
			
			m_uiCurrLineOffset = uiTokenLineOffset;
			break;
		}
	}
	
	if (m_pDb)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}
	
	// Open the database
	
	if (RC_BAD( rc = dbSystem.openDatabase( szDatabaseName, szDataDirName,
										szRflDirName, szPassword, uiFlags, &m_pDb)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

