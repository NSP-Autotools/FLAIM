//------------------------------------------------------------------------------
// Desc:	Contains the destructor for the F_Db object.
// Tabs:	3
//
// Copyright (c) 1990-1992, 1995-2007 Novell, Inc. All Rights Reserved.
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
Desc:	Destructor for F_Db object
****************************************************************************/
F_Db::~F_Db()
{
	if( m_eTransType != SFLM_NO_TRANS)
	{
		// Someone forgot to close their transaction!

		RC_UNEXPECTED_ASSERT( NE_SFLM_TRANS_ACTIVE);
		transAbort();
	}

	// Free the super file.

	if (m_pSFileHdl)
	{
		// Opened files will be released back to the
		// file handle manager

		m_pSFileHdl->Release();
	}

	// Free up statistics.

	if (m_bStatsInitialized)
	{
		flmStatFree( &m_Stats);
	}

	// Return the cached B-Tree (if any) to the
	// global pool

	if( m_pCachedBTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &m_pCachedBTree);
	}
	
	if( m_pKrefTbl)
	{
		f_free( &m_pKrefTbl);
		m_uiKrefTblSize = 0;
	}
	
	if( m_pucKrefKeyBuf)
	{
		f_free( &m_pucKrefKeyBuf);
	}

	if (m_pKeyColl)
	{
		m_pKeyColl->Release();
	}

	if (m_pIxClient)
	{
		m_pIxClient->Release();
	}

	if (m_pIxStatus)
	{
		m_pIxStatus->Release();
	}

	if (m_pDeleteStatus)
	{
		m_pDeleteStatus->Release();
	}

	if (m_pCommitClient)
	{
		m_pCommitClient->Release();
	}
	
	if (m_hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hWaitSem);
	}
	
	m_tmpKrefPool.poolFree();
	m_tempPool.poolFree();

	// Unlink the F_Db from the F_Database and F_Dict structures.
	// IMPORTANT NOTE: The call to unlinkFromDatabase needs to
	// be the last thing that is done, because it may unlock
	// and relock the mutex.

	if (m_pDatabase)
	{
		m_pDatabase->lockMutex();
		unlinkFromDict();
		m_pDatabase->unlockMutex();
		
		f_mutexLock( gv_SFlmSysData.hShareMutex);
		unlinkFromDatabase();
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	Wait for a specific database to close
****************************************************************************/
RCODE F_DbSystem::waitToClose(
	const char *		pszDbPath)
{
	RCODE					rc = NE_SFLM_OK;
	F_BUCKET *			pBucket;
	FLMUINT				uiBucket;
	F_Database *		pDatabase = NULL;
	char					szDbPathStr1[ F_PATH_MAX_SIZE];
	FLMBOOL				bMutexLocked = FALSE;
	F_SEM					hWaitSem = F_SEM_NULL;

	if( RC_BAD( rc = f_semCreate( &hWaitSem)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathToStorageString( 
		pszDbPath, szDbPathStr1)))
	{
		goto Exit;
	}

Retry:

	if( !bMutexLocked)
	{
		f_mutexLock( gv_SFlmSysData.hShareMutex);
		bMutexLocked = TRUE;
	}

	pBucket = gv_SFlmSysData.pDatabaseHashTbl;
	uiBucket = f_strHashBucket( szDbPathStr1, pBucket, FILE_HASH_ENTRIES);
	pDatabase = (F_Database *)pBucket [uiBucket].pFirstInBucket;
	
	while( pDatabase)
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
			break;
		}
		
		pDatabase = pDatabase->m_pNext;
	}
	
	if( !pDatabase)
	{
		// Didn't find a matching database.  We are done.
		
		goto Exit;
	}

	// If the file is in the process of being opened by another
	// thread, wait for the open to complete.

	if( pDatabase->m_uiFlags & DBF_BEING_OPENED)
	{
		flmWaitNotifyReq( gv_SFlmSysData.hShareMutex, hWaitSem,
			&pDatabase->m_pOpenNotifies, NULL);
		goto Retry;
	}
	else
	{
		// The database is open.  Put ourselves into the close notify list
		// so that we will wake up when the database has been closed.

		if( RC_BAD( rc = flmWaitNotifyReq(
			gv_SFlmSysData.hShareMutex, hWaitSem, 
			&pDatabase->m_pCloseNotifies, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}
	
	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	return( rc);
}
