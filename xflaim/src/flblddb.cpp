//------------------------------------------------------------------------------
// Desc:	Routines for rebuilding a corrupted database.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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
Desc:	
****************************************************************************/
typedef union
{
	FLMBYTE *				pucBlk;
	F_BLK_HDR *				pBlkHdr;
	F_BTREE_BLK_HDR *		pBTreeBlkHdr;
} F_BLOCK_UNION;

/****************************************************************************
Desc:	
****************************************************************************/
typedef struct
{
	FLMUINT				uiBlockSize;
	FLMUINT				uiCollection;
	FLMUINT64 			ui64ElmNodeId;
	FLMUINT				uiElmNumber;
	FLMBYTE *			pucElm;
	FLMUINT 				uiElmLen;
	FLMBYTE *			pucElmKey;
	FLMUINT 				uiElmKeyLen;
	FLMBYTE *			pucElmData;
	FLMUINT 				uiElmDataLen;
	FLMUINT				uiOverallDataLen;
	FLMUINT				uiDataOnlyBlkAddr;
	FLMUINT32			ui32NextBlkInChain;
	FLMUINT32			ui32BlkAddr;
	FLMUINT				uiNumKeysInBlk;
} F_ELM_INFO;

/****************************************************************************
Desc:	
****************************************************************************/
typedef struct
{
	FLMUINT				uiFileNumber;
	FLMUINT				uiFileOffset;
	FLMUINT				uiBlockSize;
	FLMUINT				uiBlockBytes;
	FLMUINT				uiCurOffset;
	F_ELM_INFO 			elmInfo;
	F_BLOCK_UNION		blkUnion;
} F_SCAN_STATE;

// Local function prototypes

FSTATIC void flmGetCreateOpts(
	XFLM_DB_HDR *				pDbHdr,
	XFLM_CREATE_OPTS *		pCreateOpts);

FSTATIC FLMINT32 bldGetElmInfo(
	F_BTREE_BLK_HDR *			pBlkHdr,
	FLMUINT						uiBlockSize,
	FLMUINT						uiElmNumber,
	F_ELM_INFO *				pElmInfo);
	
/****************************************************************************
Desc:	
****************************************************************************/
FINLINE void bldFreeCachedNode(
	F_CachedNode **		ppCachedNode)
{
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	(*ppCachedNode)->decrNodeUseCount();
	delete *ppCachedNode;
	*ppCachedNode = NULL;
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:	
****************************************************************************/
class F_RebuildNodeIStream : public IF_IStream
{
public:

	F_RebuildNodeIStream()
	{
		m_bOpen = FALSE;
		m_pucFirstElmBlk = NULL;
		m_pucTmpBlk = NULL;
		m_pCurState = NULL;
		m_pDbRebuild = NULL;
	}
	
	~F_RebuildNodeIStream()
	{
		closeStream();
	}

	RCODE openStream(
		F_DbRebuild *		pRebuild,
		FLMBOOL				bRecovDictionary);

	RCODE XFLAPI closeStream( void);
	
	RCODE XFLAPI read(
		void *				pvBuffer,
		FLMUINT				uiBytesToRead,
		FLMUINT *			puiBytesRead);

	RCODE readNode(
		FLMUINT32			ui32BlkAddr,
		FLMUINT				uiElmNumber,
		F_CachedNode **	ppCachedNode,
		FLMBYTE *			pucIV);

	RCODE getNextNode(
		F_CachedNode **	ppCachedNode,
		F_ELM_INFO *		pElmInfo,
		FLMBYTE *			pucIV);
	
	RCODE getNextNodeInfo(
		F_ELM_INFO *		pElmInfo,
		F_NODE_INFO *		pNodeInfo);

private:

	RCODE readBlock(
		FLMBOOL				bCount,
		FLMUINT				uiFileNumber,
		FLMUINT				uiFileOffset,
		F_SCAN_STATE *		pScanState);
	
	RCODE readContinuationElm( void);

	RCODE readNextFirstElm( void);

	RCODE readNextSequentialBlock(
		F_SCAN_STATE *		pScanState);

	RCODE readFirstDataOnlyBlock( void);

	RCODE readNextDataOnlyBlock( void);

	FINLINE FLMBOOL doCollection(
		FLMUINT	uiCollectionNum,
		FLMBOOL	bDoDictCollections)
	{
		if( !bDoDictCollections)
		{
			if( isDictCollection( uiCollectionNum))
			{
				return( FALSE);
			}
			
			return( (uiCollectionNum == XFLM_DATA_COLLECTION ||
						uiCollectionNum <= XFLM_MAX_COLLECTION_NUM)
						? TRUE
						: FALSE);
		}
		else
		{
			return( isDictCollection( uiCollectionNum));
		}
	}

	F_DbRebuild *			m_pDbRebuild;
	FLMBYTE *				m_pucFirstElmBlk;
	FLMBYTE *				m_pucTmpBlk;

	F_SCAN_STATE			m_firstElmState;
	F_SCAN_STATE			m_tmpState;
	F_SCAN_STATE *			m_pCurState;
	FLMBOOL					m_bOpen;
	FLMBOOL					m_bRecovDictionary;
};

/***************************************************************************
Desc:	Comparison object for node result sets
***************************************************************************/
class F_NodeResultSetCompare : public IF_ResultSetCompare
{
	inline RCODE XFLAPI compare(
		const void *			pvData1,
		FLMUINT					uiLength1,
		const void *			pvData2,
		FLMUINT					uiLength2,
		FLMINT *					piCompare)
	{
		FLMBYTE *				pucData1 = (FLMBYTE *)pvData1;
		FLMBYTE *				pucData2 = (FLMBYTE *)pvData2;
		FLMUINT					uiCollection1;
		FLMUINT					uiCollection2;
		FLMUINT64				ui64NodeId1;
		FLMUINT64				ui64NodeId2;

#ifdef FLM_DEBUG
		flmAssert( uiLength1 == uiLength2);
#else
		F_UNREFERENCED_PARM( uiLength1);
		F_UNREFERENCED_PARM( uiLength2);
#endif

		if( *pucData1 < *pucData2)
		{
			*piCompare = -1;
		}
		else if( *pucData1 > *pucData2)
		{
			*piCompare = 1;
		}
		else
		{
			uiCollection1 = f_bigEndianToUINT32( &pucData1[ 1]);
			uiCollection2 = f_bigEndianToUINT32( &pucData2[ 1]);

			if( uiCollection1 < uiCollection2)
			{
				*piCompare = -1;
			}
			else if( uiCollection1 > uiCollection2)
			{
				*piCompare = 1;
			}
			else
			{
				ui64NodeId1 = f_bigEndianToUINT64( &pucData1[ 5]);
				ui64NodeId2 = f_bigEndianToUINT64( &pucData2[ 5]);

				if( ui64NodeId1 < ui64NodeId2)
				{
					*piCompare = -1;
				}
				else if( ui64NodeId1 > ui64NodeId2)
				{
					*piCompare = 1;
				}
				else
				{
					*piCompare = 0;
				}
			}
		}

		return( NE_XFLM_OK);
	}

	virtual FLMINT XFLAPI AddRef( void)
	{
		return( IF_ResultSetCompare::AddRef());
	}

	virtual FLMINT XFLAPI Release( void)
	{
		return( IF_ResultSetCompare::Release());
	}
};

/****************************************************************************
Desc : Rebuilds a damaged database.
****************************************************************************/
RCODE F_DbRebuild::dbRebuild(
	const char *			pszSourceDbPath,
		// [IN] Source database path.  This parameter specifies
		// the path and file name of the database which is to be
		// rebuilt.
	const char *			pszSourceDataDir,
		// [IN] Source directory for data files.
	const char *			pszDestDbPath,
		// [IN] Destination database path.  This parameter specifies
		// the path and file name of a database to be created during
		// the rebuild.
	const char *			pszDestDataDir,
		// [IN] Destination directory for data files.
	const char *			pszDestRflDir,
		// [IN] Destination database's RFL path.  This parameter specifies
		// the path of the destination RFL directory.  NULL can be passed
		// if the RFL files are stored in the same directory as the other
		// destination database files.
	const char *			pszDictPath,
		// [IN] Dictionary path.  Specifies the path of the
		// data dictionary file to be used when rebuilding the
		// database.  The file should contain a copy of the
		// data dictionary that was used when the source
		// database was originally created.
	const char *			pszPassword,
		// [IN] Pointer to a password to use when extracting the database key if
		// the database is encrypted.
	XFLM_CREATE_OPTS *	pCreateOpts,
		// [IN] Create options.  Specifies the create options
		// which should be used when creating the temporary
		// database.  Once the rebuild is complete, the
		// temporary database file is copied over the source
		// database file.  Any create options specified for the
		// temporary database are inherited by the rebuilt source
		// database.
	FLMUINT64 *				pui64TotNodes,
		// [OUT] Number of nodes in the source database.
		// An estimate of the number of nodes that were in the
		// source database is returned.  This may not be exact.
	FLMUINT64 *				pui64NodesRecov,
		// [OUT] Number of nodes recovered.
	FLMUINT64 *				pui64DiscardedDocs,
		// [OUT] Number of quarantined nodes.
	IF_DbRebuildStatus *	ifpDbRebuild
		// [IN] Pointer to a user-provided interface.  NULL may be passed as the
		// value of this parameter if status reporting is not needed.
	)
{
	RCODE						rc = NE_XFLM_OK;
	F_Database *			pDatabase = NULL;
	FLMBOOL					bDatabaseLocked = FALSE;
	FLMBOOL					bWriteLocked = FALSE;
	IF_LockObject *		pWriteLockObj = NULL;
	IF_LockObject *		pDatabaseLockObj = NULL;
	FLMBOOL					bMutexLocked = FALSE;
	IF_FileHdl *			pLockFileHdl = NULL;
	eLockType				currLockType;
	FLMUINT					uiThreadId;
	FLMUINT					uiNumExclQueued;
	FLMUINT					uiNumSharedQueued;
	FLMUINT					uiPriorityCount;
	FLMBOOL					bUsedDatabase = FALSE;
 	FLMBOOL					bWaited;
	FLMBYTE *				pucWrappingKey = NULL;
	F_SEM						hWaitSem = F_SEM_NULL;
	FLMUINT					uiRflToken = 0;
	IF_CCS *					pWrappingKey = NULL;
	FLMUINT32				ui32KeyLen;
	F_SuperFileClient		SFileClient;
	
	if( RC_BAD( rc = f_semCreate( &hWaitSem)))
	{
		goto Exit;
	}

	f_mutexLock( gv_XFlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	m_bBadHeader = FALSE;
	m_cbrc = NE_XFLM_OK;

Retry:

	// See if there is a database object for this file
	// May unlock and re-lock the global mutex.
	
	if( RC_BAD( rc = gv_pXFlmDbSystem->findDatabase( 
		pszSourceDbPath, pszSourceDataDir, &pDatabase)))
	{
		goto Exit;
	}

	// If we didn't find a database object, get an exclusive lock on the file.

	if( !pDatabase)
	{
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Attempt to get an exclusive lock on the file.

		if( RC_BAD( rc = flmCreateLckFile( pszSourceDbPath, &pLockFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pDatabase->checkState( __FILE__, __LINE__)))
		{
			goto Exit;
		}

		// The call to flmVerifyFileUse will wait if the file is in
		// the process of being opened by another thread.

		if( RC_BAD( rc = pDatabase->verifyOkToUse( &bWaited)))
		{
			goto Exit;
		}
		
		if( bWaited)
		{
			goto Retry;
		}
		
		// Increment the open count on the Database so it will not
		// disappear while we are rebuilding the file.

		pDatabase->incrOpenCount();
		bUsedDatabase = TRUE;
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// See if the thread already has a file lock.  If so, there
		// is no need to obtain another.  However, we also want to
		// make sure there is no write lock.  If there is,
		// we cannot do the rebuild right now.

		pDatabase->m_pDatabaseLockObj->getLockInfo( 0, &currLockType,
			&uiThreadId, &uiNumExclQueued, &uiNumSharedQueued, &uiPriorityCount);
			
		if( currLockType == FLM_LOCK_EXCLUSIVE && uiThreadId == f_threadId())
		{
			// See if there is already a transaction going.

			pDatabase->m_pWriteLockObj->getLockInfo( (FLMINT)0, &currLockType,
				&uiThreadId, &uiNumExclQueued, &uiNumSharedQueued, 
				&uiPriorityCount);
				
			if( currLockType == FLM_LOCK_EXCLUSIVE && 
				 uiThreadId == f_threadId())
			{
				rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
				goto Exit;
			}
		}
		else
		{
			pDatabaseLockObj = pDatabase->m_pDatabaseLockObj;
			pDatabaseLockObj->AddRef();
			
			if (RC_BAD( rc = pDatabaseLockObj->lock( hWaitSem, 
				TRUE, FLM_NO_TIMEOUT, 0)))
			{
				goto Exit;
			}
			
			bDatabaseLocked = TRUE;
		}

		// Lock the write object to eliminate contention with
		// the checkpoint thread.

		pWriteLockObj = pDatabase->m_pWriteLockObj;
		pWriteLockObj->AddRef();

		// Only contention here is with the checkpoint thread.
		// Wait forever for the checkpoint thread to give
		// up the lock.

		if( RC_BAD( rc = pDatabase->dbWriteLock( hWaitSem)))
		{
			goto Exit;
		}
		bWriteLocked = TRUE;
	}

	f_memset( &m_dbHdr, 0, sizeof( XFLM_DB_HDR));
	f_memset( &m_createOpts, 0, sizeof( XFLM_CREATE_OPTS));
	f_memset( &m_callbackData, 0, sizeof( XFLM_REBUILD_INFO));
	f_memset( &m_corruptInfo, 0, sizeof( XFLM_CORRUPT_INFO));
	m_pRebuildStatus = ifpDbRebuild;
	m_uiLastStatusTime = 0;

	// Open the damaged database

	if( (m_pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = SFileClient.setup( pszSourceDbPath, pszSourceDataDir, 0)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pSFileHdl->setup( &SFileClient,
		gv_XFlmSysData.pFileHdlCache, gv_XFlmSysData.uiFileOpenFlags,
		gv_XFlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	// Check the header information to see if we were in the middle
	// of a previous copy.

	if( RC_BAD( rc = flmGetHdrInfo( m_pSFileHdl, &m_dbHdr)))
	{
		m_bBadHeader = TRUE;

		if( rc == NE_XFLM_HDR_CRC)
		{
			m_bBadHeader = TRUE;
			rc = NE_XFLM_OK;
			
			if (!pCreateOpts)
			{
				flmGetCreateOpts( &m_dbHdr, &m_createOpts);
				pCreateOpts = &m_createOpts;
			}
		}
		else if( rc == NE_XFLM_UNSUPPORTED_VERSION || rc == NE_XFLM_NEWER_FLAIM)
		{
			goto Exit;
		}
		else if( rc == NE_XFLM_NOT_FLAIM ||
			!gv_pXFlmDbSystem->validBlockSize( m_dbHdr.ui16BlockSize))
		{
			FLMUINT	uiSaveBlockSize;
			FLMUINT	uiCalcBlockSize;

			if( !pCreateOpts)
			{
				if( rc != NE_XFLM_NOT_FLAIM)
				{
					flmGetCreateOpts( &m_dbHdr, &m_createOpts);
				}
				else
				{
					flmGetCreateOpts( NULL, &m_createOpts);
				}

				// Set block size to zero, so we will always take the calculated
				// block size below.

				m_createOpts.ui32BlockSize = 0;
				pCreateOpts = &m_createOpts;
			}

			// Try to determine the correct block size.

			if( RC_BAD( rc = determineBlkSize( &uiCalcBlockSize)))
			{
				goto Exit;
			}

			uiSaveBlockSize = pCreateOpts->ui32BlockSize;
			pCreateOpts->ui32BlockSize = (FLMUINT32)uiCalcBlockSize;

			// Initialize the database header to useable values.

			flmInitDbHdr( pCreateOpts, FALSE, FALSE, &m_dbHdr);
			
			// Only use the passed-in block size (uiSaveBlockSize) if it
			// was non-zero.

			if( uiSaveBlockSize)
			{
				pCreateOpts->ui32BlockSize = (FLMUINT32)uiSaveBlockSize;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		if( !pCreateOpts)
		{
			flmGetCreateOpts( &m_dbHdr, &m_createOpts);
			pCreateOpts = &m_createOpts;
		}
	}		 

	// If the corrupt database has a key, and we are built with NICI,
	// we will extract it.

	if( m_dbHdr.ui32DbKeyLen)
	{
		if( RC_BAD( rc = flmAllocCCS( &pWrappingKey)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pWrappingKey->init( TRUE, FLM_NICI_AES)))
		{
			goto Exit;
		}

		// If the key was encrypted in a password, then the pszPassword
		// parameter better be the key used to encrypt it.  If the key was
		// not encrypted in a password, then pszPassword parameter should be NULL.

		if( RC_BAD( rc = pWrappingKey->setKeyFromStore( m_dbHdr.DbKey,
			(FLMBYTE *)pszPassword, NULL)))
		{
			goto Exit;
		}
	}

	// Delete the destination database in case it already exists.

	if( RC_BAD( rc = gv_pXFlmDbSystem->dbRemove( pszDestDbPath, pszDestDataDir,
		pszDestRflDir, TRUE)))
	{
		if( rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// If no block size has been specified or determined yet, use what we
	// read from the database header.

	if( !pCreateOpts->ui32BlockSize)
	{
		pCreateOpts->ui32BlockSize = (FLMUINT32)m_dbHdr.ui16BlockSize;
	}

	// Create the destination database

	if( RC_BAD( rc = gv_pXFlmDbSystem->dbCreate( pszDestDbPath, pszDestDataDir,
		pszDestRflDir, pszDictPath, NULL, pCreateOpts,
		(IF_Db **)&m_pDb)))
	{
		goto Exit;
	}
	
	// Check for a key from the corrupt database.
	
	if( pWrappingKey)
	{
		if( RC_BAD( rc = m_pDb->transBegin( 
			XFLM_UPDATE_TRANS, FLM_NO_TIMEOUT, 0)))
		{
			goto Exit;
		}

		ui32KeyLen = XFLM_MAX_ENC_KEY_SIZE;
		if( RC_BAD( rc = pWrappingKey->getKeyToStore( 
			&pucWrappingKey, (FLMUINT32 *)&ui32KeyLen,
			(FLMBYTE *)pszPassword, NULL)))
		{
			goto Exit;
		}
		
		f_memcpy( m_pDb->m_pDatabase->m_uncommittedDbHdr.DbKey,
					 pucWrappingKey, ui32KeyLen);
					 
		m_pDb->m_pDatabase->m_uncommittedDbHdr.ui32DbKeyLen = ui32KeyLen;
		f_free( &pucWrappingKey);
		pucWrappingKey = NULL;

		// Write out the log header

		if (RC_BAD( rc = m_pDb->m_pDatabase->writeDbHdr( m_pDb->m_pDbStats, 
			m_pDb->m_pSFileHdl, &m_pDb->m_pDatabase->m_uncommittedDbHdr,
			NULL, TRUE)))
		{
			goto Exit;
		}

		m_pDb->m_bHadUpdOper = TRUE;
		
		if( RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}
		
	// Need to close the database and re-open it to make the new key active
	// and/or to disable background threads.
	
	m_pDb->Release();
	m_pDb = NULL;
	
	// Open the destination database without starting the background threads.
	// We don't want the background threads to be active while the restore
	// is taking place, because RFL logging is disabled for the duration
	// of the restore.  We don't want the background threads to try to
	// start transactions in this state.

	if( RC_BAD( rc = gv_pXFlmDbSystem->openDb( pszDestDbPath, pszDestDataDir,
		pszDestRflDir, pszPassword, XFLM_DONT_RESUME_THREADS, (IF_Db **)&m_pDb)))
	{
		goto Exit;
	}
	
	// Set the rebuild flag
	
	m_pDb->m_uiFlags |= FDB_REBUILDING_DATABASE;

	// Disable RFL logging

	m_pDb->m_pDatabase->m_pRfl->disableLogging( &uiRflToken);

	// Rebuild the database
	
	if( RC_BAD( rc = rebuildDatabase()))
	{
		goto Exit;
	}

Exit:

	if( pucWrappingKey)
	{
		f_free( pucWrappingKey);
	}
	
	if( pWrappingKey)
	{
		pWrappingKey->Release();
	}

	if( uiRflToken)
	{
		m_pDb->m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if( bUsedDatabase)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_XFlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		pDatabase->decrOpenCount();
	}
	
	if( bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	// Unlock the file, if it is locked.

	if( bWriteLocked)
	{
		pDatabase->dbWriteUnlock();
		bWriteLocked = FALSE;
	}

	if( bDatabaseLocked)
	{
		RCODE	rc3;

		if( RC_BAD( rc3 = pDatabaseLockObj->unlock()))
		{
			if (RC_OK( rc))
			{
				rc = rc3;
			}
		}
		
		bDatabaseLocked = FALSE;
	}

	if( pWriteLockObj)
	{
		pWriteLockObj->Release();
		pWriteLockObj = NULL;
	}
	
	if( pDatabaseLockObj)
	{
		pDatabaseLockObj->Release();
		pDatabaseLockObj = NULL;
	}

	if( pLockFileHdl)
	{
		pLockFileHdl->closeFile();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	if( m_pDb)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}

	if( m_pSFileHdl)
	{
		m_pSFileHdl->Release();
		m_pSFileHdl = NULL;
	}

	if( pui64TotNodes)
	{
		*pui64TotNodes = m_callbackData.ui64TotNodes;
	}

	if( pui64NodesRecov)
	{
		*pui64NodesRecov = m_callbackData.ui64NodesRecov;
	}

	if( pui64DiscardedDocs)
	{
		*pui64DiscardedDocs = m_callbackData.ui64DiscardedDocs;
	}
		
	if( hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &hWaitSem);
	}

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_DbRebuild::getDatabaseSize( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiFileNumber = 1;
	FLMUINT64	ui64FileSize;

	m_callbackData.ui64FileSize = 0;

	while (uiFileNumber <= MAX_DATA_BLOCK_FILE_NUMBER)
	{

		// Get the actual size of the last file.

		if (RC_BAD( rc = m_pSFileHdl->getFileSize( uiFileNumber,
										&ui64FileSize)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
				 rc == NE_FLM_IO_INVALID_FILENAME)
			{
				if (uiFileNumber > 1)
				{
					rc = NE_XFLM_OK;
				}
				else
				{

					// Should always be a data file #1

					RC_UNEXPECTED_ASSERT( rc);
				}
			}
			goto Exit;
		}
		else
		{
			m_callbackData.ui64FileSize += ui64FileSize;
		}
		uiFileNumber++;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_DbRebuild::rebuildDatabase( void)
{
	RCODE					rc = NE_XFLM_OK;
	RCODE					rc2;
	FLMBOOL				bStartedTrans = FALSE;

	m_corruptInfo.ui32ErrLocale = XFLM_LOCALE_B_TREE;
	m_corruptInfo.ui32ErrLfType = XFLM_LF_COLLECTION;

	m_callbackData.ui64NodesRecov = 0;
	m_callbackData.ui64DiscardedDocs = 0;
	
	// Do a first pass to recover any dictionary items that may not have
	// been added from the dictionary file that was passed into the rebuild
	// function.

	if( m_dbHdr.ui32DbVersion < XFLM_VER_5_12)
	{
		rc = RC_SET( NE_XFLM_UNSUPPORTED_VERSION);
		goto Exit;
	}

	if (RC_BAD( rc = getDatabaseSize()))
	{
		goto Exit;
	}

	// Recover the dictionary

	m_callbackData.i32DoingFlag = REBUILD_RECOVER_DICT;
	m_callbackData.bStartFlag = TRUE;
	m_callbackData.ui64BytesExamined = 0;

	if( RC_BAD( rc = recoverNodes( TRUE)))
	{
		goto Exit;
	}
	
	// Reset nodes recovered to zero after the dictionary pass
	
	m_callbackData.ui64TotNodes = 0;
	m_callbackData.ui64NodesRecov = 0;

	// Recover data

	m_callbackData.i32DoingFlag = REBUILD_RECOVER_DATA;
	m_callbackData.bStartFlag = TRUE;
	m_callbackData.ui64BytesExamined = 0;

	if( RC_BAD( rc = recoverNodes( FALSE)))
	{
		goto Exit;
	}

	// Try and preserve other things in the log header

	if( !m_bBadHeader)
	{
		F_Database *	pDatabase;
		XFLM_DB_HDR *	pDbHdr;

		if( RC_BAD( m_pDb->transBegin( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
		pDatabase = m_pDb->getDatabase();
		pDbHdr = pDatabase->getUncommittedDbHdr();

		// Set the commit count one less than the old database's
		// This is because it will be incremented if the transaction 
		// successfully commits - which will make it exactly right.

		if( pDbHdr->ui64TransCommitCnt < m_dbHdr.ui64TransCommitCnt - 1)
		{
			pDbHdr->ui64TransCommitCnt = m_dbHdr.ui64TransCommitCnt - 1;
		}

		bStartedTrans = FALSE;
		if( RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	rc2 = reportStatus( TRUE);
	if (RC_OK( rc))
	{
		rc = rc2;
	}

	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_DbRebuild::recoverNodes(
	FLMBOOL						bRecoverDictionary)
{
	RCODE							rc = NE_XFLM_OK;
	IF_ResultSet *				pRootRSet = NULL;
	IF_ResultSet *				pNonRootRSet = NULL;
	F_CachedNode *				pRecovRoot = NULL;
	F_RebuildNodeIStream *	pIStream = NULL;
	FLMUINT64					ui64RootCount;
	FLMUINT64					ui64NonRootCount;
	FLMUINT64					ui64RSPosition;
	FLMUINT64					ui64NodeId;
	FLMUINT64					ui64TransStartPos = 0;
	FLMUINT64					ui64LastAbortPos = 0;
	F_ELM_INFO					elmInfo;
	FLMUINT						uiRSetEntrySize;
	FLMUINT						uiBlockAddr;
	FLMUINT						uiElmNumber;
	FLMUINT						uiTime;
	FLMUINT						uiTransStartTime = 0;
	FLMUINT						uiMaxTransTime;
	IF_ResultSetCompare *	pCompareRSEntry = NULL;
	FLMBOOL						bStartedTrans = FALSE;
	F_NODE_INFO					nodeInfo;
	FLMBYTE						ucIV[ 16];
	FLMBYTE						ucRSetBuffer[ REBUILD_RSET_ENTRY_SIZE];
	XFLM_REBUILD_INFO			transStartRebuildInfo;

	// Allocate result sets

	if( (pCompareRSEntry = f_new F_NodeResultSetCompare) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocResultSet( &pRootRSet)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRootRSet->setupResultSet( ".", pCompareRSEntry, 0,
		TRUE, FALSE)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocResultSet( &pNonRootRSet)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNonRootRSet->setupResultSet( ".", pCompareRSEntry, 0,
		TRUE, FALSE)))
	{
		goto Exit;
	}
	
	// Allocate and configure the rebuild input stream
	
	if( (pIStream = f_new F_RebuildNodeIStream) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIStream->openStream( this, bRecoverDictionary)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Recover nodes from the source database
	
	for( ;;)
	{ 
		if( RC_BAD( rc = pIStream->getNextNodeInfo( &elmInfo, &nodeInfo)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}

		if( !nodeInfo.ui64ParentId)
		{
			buildRSetEntry( 
				getRSetPrefix( nodeInfo.uiNameId),
				nodeInfo.uiCollection, nodeInfo.ui64NodeId,
				elmInfo.ui32BlkAddr, elmInfo.uiElmNumber, ucRSetBuffer);

			if( RC_BAD( rc = pRootRSet->addEntry( ucRSetBuffer, 
				sizeof( ucRSetBuffer))))
			{
				goto Exit;
			}
		}
		else
		{
			buildRSetEntry( 
				0,
				nodeInfo.uiCollection, nodeInfo.ui64NodeId,
				elmInfo.ui32BlkAddr, elmInfo.uiElmNumber, ucRSetBuffer);

			if( RC_BAD( rc = pNonRootRSet->addEntry( ucRSetBuffer,
				sizeof( ucRSetBuffer))))
			{
				goto Exit;
			}
		}

		m_callbackData.ui64TotNodes++;
	}
	
	bStartedTrans = FALSE;
	if( RC_BAD( rc = m_pDb->transAbort()))
	{
		goto Exit;
	}

	// Sort the result sets

	if( RC_BAD( rc = pRootRSet->finalizeResultSet( NULL, &ui64RootCount)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pNonRootRSet->finalizeResultSet( NULL, &ui64NonRootCount)))
	{
		goto Exit;
	}
	
	m_callbackData.ui64TotNodes = ui64RootCount + ui64NonRootCount;
	uiMaxTransTime = FLM_SECS_TO_TIMER_UNITS( 30);

	// Add the nodes to the destination database

	for( ui64RSPosition = 0; ui64RSPosition < ui64RootCount; ui64RSPosition++)
	{
Retry:
		uiTime = FLM_GET_TIMER();

		if( !bStartedTrans)
		{
			if( RC_BAD( rc = m_pDb->transBegin( XFLM_UPDATE_TRANS)))
			{
				goto Exit;
			}

			bStartedTrans = TRUE;
			ui64TransStartPos = ui64RSPosition;
			uiTransStartTime = FLM_GET_TIMER();
			f_memcpy( &transStartRebuildInfo, &m_callbackData,
				sizeof( XFLM_REBUILD_INFO));
		}

		if( RC_BAD( rc = pRootRSet->setPosition( ui64RSPosition)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pRootRSet->getCurrent( ucRSetBuffer, 
			sizeof( ucRSetBuffer), &uiRSetEntrySize)))
		{
			goto Exit;
		}
		
		extractRSetEntry( ucRSetBuffer, NULL, NULL, 
			&uiBlockAddr, &uiElmNumber);

		if( RC_BAD( rc = pIStream->readNode( (FLMUINT32)uiBlockAddr, 
			uiElmNumber, &pRecovRoot, ucIV)))
		{
			if( RC_BAD( m_cbrc))
			{
				rc = m_cbrc;
				goto Exit;
			}

			rc = NE_XFLM_OK;
			continue;
		}

		ui64NodeId = pRecovRoot->getNodeId();

		if( pRecovRoot->getCollection() == XFLM_DICT_COLLECTION)
		{
			if( ui64NodeId == XFLM_DICTINFO_DOC_ID)
			{
				// No need to recover the dictinfo document
				// since it will be rebuilt as we add nodes
				// to the destination database

				m_callbackData.ui64NodesRecov++;
				continue;
			}
		}

		if( ui64LastAbortPos && ui64RSPosition == ui64LastAbortPos)
		{
			bStartedTrans = FALSE;
			if( RC_BAD( rc = m_pDb->transCommit()))
			{
				goto Exit;
			}

			ui64LastAbortPos = 0;
			m_callbackData.ui64DiscardedDocs++;
			continue;
		}

		if( RC_BAD( rc = recoverTree( pIStream, 
			pNonRootRSet, NULL, pRecovRoot, ucIV)))
		{
			if( RC_BAD( m_cbrc))
			{
				rc = m_cbrc;
				goto Exit;
			}

			bStartedTrans = FALSE;
			if( RC_BAD( rc = m_pDb->transAbort()))
			{
				goto Exit;
			}

			ui64LastAbortPos = ui64RSPosition;
			ui64RSPosition = ui64TransStartPos;
			f_memcpy( &m_callbackData, &transStartRebuildInfo, 
				sizeof( XFLM_REBUILD_INFO));

			if( RC_BAD( rc = reportStatus( TRUE)))
			{
				goto Exit;
			}

			if( !ui64RSPosition)
			{
				continue;
			}

			goto Retry;
		}

		if( FLM_ELAPSED_TIME( uiTime, uiTransStartTime) >= uiMaxTransTime)
		{
			bStartedTrans = FALSE;
			if( RC_BAD( rc = m_pDb->transCommit()))
			{
				goto Exit;
			}
		}
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if( pRecovRoot)
	{
		bldFreeCachedNode( &pRecovRoot);
	}
	
	if( pIStream)
	{
		pIStream->Release();
	}

	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}
	
	if( pRootRSet)
	{
		pRootRSet->Release();
	}
	
	if( pNonRootRSet)
	{
		pNonRootRSet->Release();
	}

	if( pCompareRSEntry)
	{
		pCompareRSEntry->Release();
		pCompareRSEntry = NULL;
	}

	return( RC_BAD( rc) ? rc : reportStatus( TRUE));
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_DbRebuild::recoverTree(
	F_RebuildNodeIStream *	pIStream,
	IF_ResultSet *				pNonRootRSet,
	F_DOMNode *					pParentNode,
	F_CachedNode *				pRecovCachedNode,
	FLMBYTE *					pucNodeIV)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64NodeId = pRecovCachedNode->getNodeId();
	FLMUINT64			ui64ChildId;
	F_DOMNode *			pNode = NULL;
	F_DOMNode *			pAttrNode = NULL;
	F_CachedNode *		pRecovChild = NULL;
	eDomNodeType		eRecovNodeType = pRecovCachedNode->getNodeType();
	FLMUINT				uiCollection = pRecovCachedNode->getCollection();
	FLMUINT				uiBlockAddr;
	FLMUINT				uiElmNumber;
	FLMBYTE				ucTmpRSetBuffer[ REBUILD_RSET_ENTRY_SIZE];
	FLMBYTE				ucChildIV[ 16];

	// Create the node

	if( !pParentNode)
	{
		if( eRecovNodeType == DOCUMENT_NODE)
		{
			if( RC_BAD( rc = m_pDb->createDocument( uiCollection,
				(IF_DOMNode **)&pNode, &ui64NodeId)))
			{
				goto Exit;
			}
		}
		else if( eRecovNodeType == ELEMENT_NODE)
		{
			if( RC_BAD( rc = m_pDb->createRootElement( uiCollection,
				pRecovCachedNode->getNameId(), 
				(IF_DOMNode **)&pNode, &ui64NodeId)))
			{
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pParentNode->createNode( m_pDb, eRecovNodeType, 
			pRecovCachedNode->getNameId(), XFLM_LAST_CHILD,
			(IF_DOMNode **)&pNode, &ui64NodeId)))
		{
			goto Exit;
		}
	}

	// Set the value
	
	if( pRecovCachedNode->getDataLength())
	{
		if( pRecovCachedNode->getModeFlags() & FDOM_VALUE_ON_DISK)
		{
			FLMUINT	uiEncDefId = pRecovCachedNode->getEncDefId();
			FLMBYTE	ucTmpBuf[ FLM_ENCRYPT_CHUNK_SIZE];
			FLMUINT	uiBytesRead;

			for( ;;)
			{
				if( RC_BAD( rc = pIStream->read( ucTmpBuf,
					sizeof( ucTmpBuf), &uiBytesRead)))
				{
					if( rc != NE_XFLM_EOF_HIT)
					{
						goto Exit;
					}

					rc = NE_XFLM_OK;

					if( !uiBytesRead)
					{
						break;
					}
				}

				if( uiEncDefId)
				{
					if( RC_BAD( rc = m_pDb->decryptData( uiEncDefId, pucNodeIV,
						ucTmpBuf, uiBytesRead, ucTmpBuf, sizeof( ucTmpBuf))))
					{
						goto Exit;
					}
				}
					
				if( RC_BAD( rc = pNode->setStorageValue( m_pDb, ucTmpBuf, 
					uiBytesRead, uiEncDefId, FALSE)))
				{
					goto Exit;
				}

				if( uiBytesRead < sizeof( ucTmpBuf))
				{
					break;
				}
			}

			if( RC_BAD( rc = pNode->setStorageValue( m_pDb, 
				NULL, 0, uiEncDefId, TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pNode->setStorageValue( m_pDb,
				pRecovCachedNode->getDataPtr(), pRecovCachedNode->getDataLength(),
				pRecovCachedNode->getEncDefId(), TRUE)))
			{
				goto Exit;
			}
		}
	}

	// Add attributes
	
	if( pRecovCachedNode->m_uiAttrCount)
	{
		FLMUINT			uiLoop;
		F_AttrItem *	pAttrItem;
		
		for (uiLoop = 0; uiLoop < pRecovCachedNode->m_uiAttrCount; uiLoop++)
		{
			pAttrItem = pRecovCachedNode->m_ppAttrList [uiLoop];
			if( RC_BAD( rc = pNode->createAttribute( m_pDb, 
				pAttrItem->m_uiNameId, (IF_DOMNode **)&pAttrNode)))
			{
				goto Exit;
			}
			
			if( pAttrItem->getAttrDataLength())
			{
				if( RC_BAD( rc = pAttrNode->setStorageValue( m_pDb,
					pAttrItem->getAttrDataPtr(), pAttrItem->getAttrDataLength(),
					pAttrItem->getAttrEncDefId(), TRUE)))
				{
					goto Exit;
				}
			}

			if( pAttrItem->getAttrModeFlags())
			{
				if( RC_BAD( rc = pAttrNode->addModeFlags( m_pDb, 
					pAttrItem->getAttrModeFlags())))
				{
					goto Exit;
				}
			}
		}
	}

	// Set the node's flags

	if( pRecovCachedNode->getPersistentFlags())
	{
		if( RC_BAD( rc = pNode->addModeFlags( m_pDb, 
			pRecovCachedNode->getPersistentFlags())))
		{
			goto Exit;
		}
	}

	m_callbackData.ui64NodesRecov++;

	// Recover the node's children (if any)

	ui64ChildId = pRecovCachedNode->getFirstChildId();

	while( ui64ChildId)
	{
		buildRSetEntry( 0, uiCollection, ui64ChildId, 0, 0, ucTmpRSetBuffer);

		if( RC_BAD( rc = pNonRootRSet->findMatch( ucTmpRSetBuffer, 
			REBUILD_RSET_ENTRY_SIZE, ucTmpRSetBuffer, NULL)))
		{
			if( rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}

			goto Exit;
		}

		extractRSetEntry( ucTmpRSetBuffer, NULL, NULL, 
			&uiBlockAddr, &uiElmNumber);

		if( RC_BAD( rc = pIStream->readNode( (FLMUINT32)uiBlockAddr, 
			uiElmNumber, &pRecovChild, ucChildIV)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = recoverTree( pIStream,
			pNonRootRSet, pNode, pRecovChild, ucChildIV)))
		{
			goto Exit;
		}

		ui64ChildId = pRecovChild->getNextSibId();
	}

	// Recover the annotation node (if any)

	if( pRecovCachedNode->getAnnotationId())
	{
		buildRSetEntry( 0, 
			uiCollection, pRecovCachedNode->getAnnotationId(),
			0, 0, ucTmpRSetBuffer);

		if( RC_BAD( rc = pNonRootRSet->findMatch( ucTmpRSetBuffer, 
			REBUILD_RSET_ENTRY_SIZE, ucTmpRSetBuffer, NULL)))
		{
			if( rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}

			goto Exit;
		}

		extractRSetEntry( ucTmpRSetBuffer, NULL, NULL, 
			&uiBlockAddr, &uiElmNumber);

		if( RC_BAD( rc = pIStream->readNode( (FLMUINT32)uiBlockAddr, 
			uiElmNumber, &pRecovChild, ucChildIV)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = recoverTree( pIStream,
			pNonRootRSet, pNode, pRecovChild, ucChildIV)))
		{
			goto Exit;
		}
	}

	// Set the metavalue

	if( pRecovCachedNode->getMetaValue())
	{
		if( RC_BAD( rc = pNode->setMetaValue( m_pDb, 
			pRecovCachedNode->getMetaValue())))
		{
			goto Exit;
		}
	}

	// Set the prefix ID

	if( pRecovCachedNode->getPrefixId())
	{
		if( RC_BAD( rc = pNode->setPrefixId( m_pDb,
			pRecovCachedNode->getPrefixId())))
		{
			goto Exit;
		}
	}

	// Call document done if this was a root node

	if( !pParentNode)
	{
		if( RC_BAD( rc = m_pDb->documentDone( pNode)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		if( pNode)
		{
			(void)pNode->deleteNode( m_pDb);
		}

		m_callbackData.ui64DiscardedDocs++;
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( pRecovChild)
	{
		bldFreeCachedNode( &pRecovChild);
	}

	return( rc);
}

/***************************************************************************
Desc:	Function to extract information about the current element
***************************************************************************/
FSTATIC FLMINT32 bldGetElmInfo(
	F_BTREE_BLK_HDR *	pBlkHdr,
	FLMUINT				uiBlockSize,
	FLMUINT				uiElmNumber,
	F_ELM_INFO *		pElmInfo)
{
	FLMINT32				i32ErrCode = 0;
	FLMBYTE *			pucElm = NULL;
	FLMUINT				uiElmLen = 0;
	FLMUINT				uiElmKeyLen = 0;
	FLMUINT				uiElmDataLen = 0;
	FLMUINT				uiOverallDataLen = 0;
	FLMUINT				uiDataOnlyBlkAddr = 0;
	FLMBYTE *			pucElmKey = NULL;
	FLMBYTE *			pucElmData = NULL;
	FLMBYTE *			pucBlkEnd;
	FLMBOOL				bNeg;
	FLMUINT				uiNumKeys = pBlkHdr->ui16NumKeys;
	FLMUINT				uiBytesProcessed;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT64			ui64ElmNodeId = 0;
	
	if( uiElmNumber >= uiNumKeys)
	{
		flmAssert( 0);
		i32ErrCode = FLM_BAD_ELM_OFFSET;
		goto Exit;
	}
	
	pui16OffsetArray = (FLMUINT16 *)((FLMBYTE *)pBlkHdr +	
								sizeofBTreeBlkHdr( pBlkHdr));
	pucElm = (FLMBYTE *)pBlkHdr + bteGetEntryOffset( pui16OffsetArray, uiElmNumber);
	pucBlkEnd = (FLMBYTE *)pBlkHdr + uiBlockSize;

	switch( pBlkHdr->stdBlkHdr.ui8BlkType)
	{
		case BT_LEAF:
		{
			if( pucElm + 2 > pucBlkEnd)
			{
				i32ErrCode = FLM_BAD_ELM_LEN;
				goto Exit;
			}
			
			uiElmKeyLen = FB2UW( pucElm);
			uiElmLen = uiElmKeyLen + 2;
			break;
		}
		
		case BT_LEAF_DATA:
		{
			FLMBYTE		ucFlags = *pucElm;
			FLMBYTE *	pucPtr = &pucElm[ 1];
			
			if( ucFlags & BTE_FLAG_KEY_LEN)
			{
				if( pucPtr + 2 > pucBlkEnd)
				{
					i32ErrCode = FLM_BAD_ELM_LEN;
					goto Exit;
				}
				
				uiElmKeyLen = FB2UW( pucPtr);
				uiElmLen = uiElmKeyLen + 2;
				pucPtr += 2;
			}
			else
			{
				if( pucPtr > pucBlkEnd)
				{
					i32ErrCode = FLM_BAD_ELM_LEN;
					goto Exit;
				}
				
				uiElmKeyLen = *pucPtr;
				uiElmLen = uiElmKeyLen + 1;
				pucPtr++;
			}

			if( ucFlags & BTE_FLAG_DATA_LEN)
			{
				if( pucPtr + 2 > pucBlkEnd)
				{
					i32ErrCode = FLM_BAD_ELM_LEN;
					goto Exit;
				}
				
				uiElmDataLen = FB2UW( pucPtr);
				uiElmLen += (uiElmDataLen + 2);
				pucPtr += 2;
			}
			else
			{
				if( pucPtr > pucBlkEnd)
				{
					i32ErrCode = FLM_BAD_ELM_LEN;
					goto Exit;
				}
				
				uiElmDataLen = *pucPtr;
				uiElmLen += uiElmDataLen + 1;
				pucPtr++;
			}

			if( ucFlags & BTE_FLAG_OA_DATA_LEN)
			{
				uiOverallDataLen = FB2UD( pucPtr);
				uiElmLen += 4;
				pucPtr += 4;
			}

			pucElmKey = pucPtr;
			pucPtr += uiElmKeyLen;
			
			pucElmData = pucPtr;
			pucPtr += uiElmDataLen;

			if( bteDataBlockFlag( pucElm))
			{
				if( uiElmDataLen != 4)
				{
					i32ErrCode = FLM_BAD_ELM_LEN;
					goto Exit;
				}

				uiDataOnlyBlkAddr = FB2UD( pucElmData);
			}

			break;
		}

		default:
		{
			i32ErrCode = FLM_BAD_BLK_TYPE;
			goto Exit;
		}
	}
	
	if( pucElm + uiElmLen >	pucBlkEnd)
	{
		i32ErrCode = FLM_BAD_ELM_LEN;
		goto Exit;
	}

	if( uiElmKeyLen)
	{
		if( RC_BAD( flmCollation2Number( uiElmKeyLen, pucElmKey,
			&ui64ElmNodeId, &bNeg, &uiBytesProcessed)))
		{
			i32ErrCode = FLM_BAD_ELM_KEY;
			goto Exit;
		}

		if( bNeg || uiBytesProcessed != uiElmKeyLen || !ui64ElmNodeId)
		{
			i32ErrCode = FLM_BAD_ELM_KEY;
			goto Exit;
		}
	}
	else
	{
		// If the key length is zero, then this MUST be the last block!
		
		if( pBlkHdr->stdBlkHdr.ui32NextBlkInChain)
		{
			i32ErrCode = FLM_BAD_ELM_KEY;
			goto Exit;
		}
	}
	
	if( !uiOverallDataLen)
	{
		uiOverallDataLen = uiElmDataLen;
	}
	
Exit:
	
	pElmInfo->uiCollection = pBlkHdr->ui16LogicalFile;
	pElmInfo->uiBlockSize = uiBlockSize;
	pElmInfo->uiElmNumber = uiElmNumber;
	pElmInfo->pucElm = pucElm;
	pElmInfo->uiElmLen = uiElmLen;
	pElmInfo->pucElmKey = pucElmKey;
	pElmInfo->uiElmKeyLen = uiElmKeyLen;
	pElmInfo->pucElmData = pucElmData;
	pElmInfo->uiElmDataLen = uiElmDataLen;
	pElmInfo->uiOverallDataLen = uiOverallDataLen;
	pElmInfo->uiDataOnlyBlkAddr = uiDataOnlyBlkAddr;
	pElmInfo->ui64ElmNodeId = ui64ElmNodeId;
	pElmInfo->ui32BlkAddr = pBlkHdr->stdBlkHdr.ui32BlkAddr;
	pElmInfo->ui32NextBlkInChain = pBlkHdr->stdBlkHdr.ui32NextBlkInChain;
	pElmInfo->uiNumKeysInBlk = pBlkHdr->ui16NumKeys;

	return( i32ErrCode);
}

/****************************************************************************
Desc:	Extract create options from the DB header.
****************************************************************************/
FSTATIC void flmGetCreateOpts(
	XFLM_DB_HDR *			pDbHdr,
	XFLM_CREATE_OPTS *	pCreateOpts)
{
	f_memset( pCreateOpts, 0, sizeof( XFLM_CREATE_OPTS));
	if( pDbHdr)
	{
		pCreateOpts->ui32BlockSize = (FLMUINT32)pDbHdr->ui16BlockSize;
		pCreateOpts->ui32VersionNum = pDbHdr->ui32DbVersion;
		pCreateOpts->ui32DefaultLanguage = pDbHdr->ui8DefaultLanguage;
		pCreateOpts->ui32MinRflFileSize = pDbHdr->ui32RflMinFileSize;
		pCreateOpts->ui32MaxRflFileSize = pDbHdr->ui32RflMaxFileSize;
		pCreateOpts->bKeepRflFiles = (FLMBOOL)(pDbHdr->ui8RflKeepFiles
															? TRUE
															: FALSE);
		pCreateOpts->bLogAbortedTransToRfl =
			(FLMBOOL)(pDbHdr->ui8RflKeepAbortedTrans
						 ? TRUE
						 : FALSE);
	}
	else
	{
		pCreateOpts->ui32BlockSize = XFLM_DEFAULT_BLKSIZ;
		pCreateOpts->ui32VersionNum = XFLM_CURRENT_VERSION_NUM;
		pCreateOpts->ui32DefaultLanguage = XFLM_DEFAULT_LANG;
		pCreateOpts->ui32MinRflFileSize = XFLM_DEFAULT_MIN_RFL_FILE_SIZE;
		pCreateOpts->ui32MaxRflFileSize = XFLM_DEFAULT_MAX_RFL_FILE_SIZE;
		pCreateOpts->bKeepRflFiles = XFLM_DEFAULT_KEEP_RFL_FILES_FLAG;
		pCreateOpts->bLogAbortedTransToRfl = XFLM_DEFAULT_LOG_ABORTED_TRANS_FLAG;
	}
}

/****************************************************************************
Desc : Rebuilds a damaged database.
Notes: All the important stuff is handled by the F_DbRebuild::dbRebuild
		 function (below).  All this call does is create an F_DbRebuild object,
		 call dbRebuild on it, delete the obj and return the RCODE.
****************************************************************************/
RCODE XFLAPI F_DbSystem::dbRebuild(
	const char *				pszSourceDbPath,
	const char *				pszSourceDataDir,
	const char *				pszDestDbPath,
	const char *				pszDestDataDir,
	const char *				pszDestRflDir,
	const char *				pszDictPath,
	const char *				pszPassword,	
	XFLM_CREATE_OPTS *		pCreateOpts,
	FLMUINT64 *					pui64TotNodes,
	FLMUINT64 *					pui64NodesRecov,
	FLMUINT64 *					pui64DiscardedDocs,
	IF_DbRebuildStatus *		ifpDbRebuild)
{
	RCODE				rc = NE_XFLM_OK;
	F_DbRebuild *	pRbldObj = NULL;

	if( (pRbldObj = f_new F_DbRebuild) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	rc = pRbldObj->dbRebuild( pszSourceDbPath, pszSourceDataDir,
									  pszDestDbPath, pszDestDataDir, pszDestRflDir,
									  pszDictPath, pszPassword, pCreateOpts, 
									  pui64TotNodes, pui64NodesRecov, pui64DiscardedDocs,
									  ifpDbRebuild);

Exit:

	if( pRbldObj)
	{
		pRbldObj->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::openStream(
	F_DbRebuild *		pDbRebuild,
	FLMBOOL				bRecovDictionary)
{
	RCODE			rc = NE_XFLM_OK;
	
	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	m_pDbRebuild = pDbRebuild;
	m_pDbRebuild->AddRef();
	m_bRecovDictionary = bRecovDictionary;
	
	f_memset( &m_firstElmState, 0, sizeof( F_SCAN_STATE));
	f_memset( &m_tmpState, 0, sizeof( F_SCAN_STATE));

	if( RC_BAD( rc = f_alloc( m_pDbRebuild->getBlockSize(),
		&m_pucFirstElmBlk)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_alloc( m_pDbRebuild->getBlockSize(),
		&m_pucTmpBlk)))
	{
		goto Exit;
	}

	m_firstElmState.blkUnion.pucBlk = m_pucFirstElmBlk;
	m_tmpState.blkUnion.pucBlk = m_pucTmpBlk;
	m_pCurState = NULL;
	m_bOpen = TRUE;
	
Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);	
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::closeStream( void)
{
	if( m_pucFirstElmBlk)
	{
		f_free( &m_pucFirstElmBlk);
	}
	
	if( m_pucTmpBlk)
	{
		f_free( &m_pucTmpBlk);
	}

	if( m_pDbRebuild)
	{
		m_pDbRebuild->Release();
		m_pDbRebuild = NULL;
	}

	m_pCurState = NULL;
	m_bOpen = FALSE;
	
	f_memset( &m_firstElmState, 0, sizeof( F_SCAN_STATE));
	f_memset( &m_tmpState, 0, sizeof( F_SCAN_STATE));
	
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readBlock(
	FLMBOOL						bCount,
	FLMUINT						uiFileNumber,
	FLMUINT						uiFileOffset,
	F_SCAN_STATE *				pScanState)
{
	RCODE							rc = NE_XFLM_OK;
	F_Dict *						pDict;
	FLMUINT						uiBlockSize = m_pDbRebuild->getBlockSize();
	FLMUINT						uiBlkEnd;
	FLMUINT16					ui16BlkBytesAvail;
	FLMUINT32					ui32CRC;
	F_BLK_HDR *					pBlkHdr = pScanState->blkUnion.pBlkHdr;
	FLMBYTE *					pucBlk = pScanState->blkUnion.pucBlk;
	
	if( RC_BAD( rc = m_pDbRebuild->m_pSFileHdl->readBlock( 
		FSBlkAddress( uiFileNumber, uiFileOffset), 
		uiBlockSize, pucBlk, NULL)))
	{
		goto Exit;
	}
	if (bCount)
	{
		m_pDbRebuild->incrBytesExamined();
	}

	// Determine if we should convert the block here.  Calculation of CRC
	// should be on unconverted block.

	ui16BlkBytesAvail = pBlkHdr->ui16BlkBytesAvail;
	
	if( blkIsNonNativeFormat( pBlkHdr))
	{
		convert16( &ui16BlkBytesAvail);
	}
	
	if( ui16BlkBytesAvail > (uiBlockSize - blkHdrSize( pBlkHdr)))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	uiBlkEnd = (blkIsNewBTree( pBlkHdr)
					? uiBlockSize
					: uiBlockSize - (FLMUINT)ui16BlkBytesAvail);
					
	// Compute the checksum and convert the block
		
	ui32CRC = calcBlkCRC( pBlkHdr, uiBlkEnd);
	
	if( blkIsNonNativeFormat( pBlkHdr))
	{
		convertBlk( uiBlockSize, pBlkHdr);
	}
	
	// Does the checksum match?
	
	if( ui32CRC != pBlkHdr->ui32BlkCRC)
	{
		rc = RC_SET( NE_XFLM_BLOCK_CRC);
		goto Exit;
	}
	
	// Make sure the transaction ID looks valid

	if( !m_pDbRebuild->m_bBadHeader &&
		pBlkHdr->ui64TransID > 1 &&
		pBlkHdr->ui64TransID > m_pDbRebuild->m_dbHdr.ui64LastRflCommitID)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// If this is a data-only block with a next sibling,
	// the block should be full

	if( pBlkHdr->ui8BlkType == BT_DATA_ONLY && 
		 pBlkHdr->ui32NextBlkInChain &&
		 pBlkHdr->ui16BlkBytesAvail)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
		
	// Decrypt the block
	
	if( isEncryptedBlk( pBlkHdr))
	{
		if( RC_BAD( rc = m_pDbRebuild->m_pDb->getDictionary( &pDict)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDbRebuild->m_pDb->getDatabase()->decryptBlock(
			pDict, pucBlk)))
		{
			goto Exit;
		}
	}
	
	pScanState->uiFileNumber = uiFileNumber;
	pScanState->uiFileOffset = uiFileOffset;
	pScanState->uiBlockSize = uiBlockSize;
	pScanState->uiBlockBytes = uiBlockSize - pBlkHdr->ui16BlkBytesAvail;
	pScanState->uiCurOffset = 0;
	f_memset( &pScanState->elmInfo, 0, sizeof( F_ELM_INFO));

Exit:

	return( RC_BAD( rc) ? rc : m_pDbRebuild->reportStatus());
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readNextSequentialBlock(
	F_SCAN_STATE *		pScanState)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiBlkCollectionNum;
	FLMUINT				uiBlockSize = m_pDbRebuild->getBlockSize();
	FLMUINT				uiTryNextCount = 0;

	if( pScanState->uiFileNumber > MAX_DATA_BLOCK_FILE_NUMBER)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	for( ;;)
	{
		if( pScanState->uiFileOffset + uiBlockSize >= 
				m_pDbRebuild->m_dbHdr.ui32MaxFileSize ||
			!pScanState->uiFileNumber)
		{
TryNextFile:

			pScanState->uiFileOffset = 0;
			pScanState->uiFileNumber++;
			
			if( pScanState->uiFileNumber > MAX_DATA_BLOCK_FILE_NUMBER || 
				uiTryNextCount > 5)
			{
				pScanState->uiFileNumber = MAX_DATA_BLOCK_FILE_NUMBER + 1;
				rc = RC_SET( NE_XFLM_EOF_HIT);
				goto Exit;
			}
		}
		else
		{
			pScanState->uiFileOffset += uiBlockSize;
		}
		
		if( RC_BAD( rc = readBlock( TRUE, pScanState->uiFileNumber, 
			pScanState->uiFileOffset, pScanState)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE ||
				 rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				uiTryNextCount++;
				goto TryNextFile;
			}
			else if( rc == NE_XFLM_DATA_ERROR || rc == NE_XFLM_BLOCK_CRC)
			{
				rc = NE_XFLM_OK;
				continue;
			}
			
			goto Exit;
		}

		// Determine if this is a block that should be processed
	
		if( pScanState->blkUnion.pBlkHdr->ui32BlkAddr ==
					FSBlkAddress( pScanState->uiFileNumber, pScanState->uiFileOffset) &&
			 (pScanState->blkUnion.pBlkHdr->ui8BlkType == BT_LEAF || 
		     pScanState->blkUnion.pBlkHdr->ui8BlkType == BT_LEAF_DATA) &&
			 pScanState->blkUnion.pBTreeBlkHdr->ui8BlkLevel == 0 &&
			 (uiBlkCollectionNum = pScanState->blkUnion.pBTreeBlkHdr->ui16LogicalFile) != 0 &&
			 isContainerBlk( pScanState->blkUnion.pBTreeBlkHdr) &&
			 doCollection( uiBlkCollectionNum, m_bRecovDictionary))
		{

			// Make sure the block end looks correct
			
			if( pScanState->blkUnion.pBlkHdr->ui16BlkBytesAvail > 
				uiBlockSize - blkHdrSize( pScanState->blkUnion.pBlkHdr))
			{
				if( RC_BAD( rc = m_pDbRebuild->reportCorruption( 
					FLM_BAD_BLK_HDR_BLK_END, 
					pScanState->blkUnion.pBlkHdr->ui32BlkAddr, 0, 0)))
				{
					goto Exit;
				}
				
				continue;
			}
		
			if( !m_bRecovDictionary)
			{
				if( RC_BAD( rc = m_pDbRebuild->m_pDb->getDict()->getCollection( 
					uiBlkCollectionNum, NULL)))
				{
					if( rc != NE_XFLM_BAD_COLLECTION)
					{
						goto Exit;
					}
					
					rc = NE_XFLM_OK;
					continue;
				}
			}
			
			break;
		}
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readNextFirstElm( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMINT32				i32ErrCode = 0;

	m_pCurState = NULL;

GetNextElement:

	if( !m_firstElmState.uiFileNumber || 
		  m_firstElmState.elmInfo.uiElmNumber + 1 >= 
				m_firstElmState.blkUnion.pBTreeBlkHdr->ui16NumKeys)
	{
		if( RC_BAD( rc = readNextSequentialBlock( &m_firstElmState)))
		{
			goto Exit;
		}
	}
	else
	{
		m_firstElmState.elmInfo.uiElmNumber++;
	}
		
	// Extract information about the element
	
	if( (i32ErrCode = bldGetElmInfo( m_firstElmState.blkUnion.pBTreeBlkHdr, 
		m_firstElmState.uiBlockSize, m_firstElmState.elmInfo.uiElmNumber, 
		&m_firstElmState.elmInfo)) != 0)
	{
			if( RC_BAD( rc = m_pDbRebuild->reportCorruption( i32ErrCode,
			FSBlkAddress( m_firstElmState.uiFileNumber, m_firstElmState.uiFileOffset),
			m_firstElmState.elmInfo.uiElmNumber, 
			m_firstElmState.elmInfo.ui64ElmNodeId)))
		{
			goto Exit;
		}
		
		goto GetNextElement;
	}
	
	if( !bteFirstElementFlag( m_firstElmState.elmInfo.pucElm) ||
		 !m_firstElmState.elmInfo.uiElmKeyLen)
	{
		goto GetNextElement;
	}

	if( m_firstElmState.elmInfo.uiDataOnlyBlkAddr)
	{
		if( RC_BAD( rc = readFirstDataOnlyBlock()))
		{
			if( RC_BAD( m_pDbRebuild->m_cbrc))
			{
				rc = m_pDbRebuild->m_cbrc;
				goto Exit;
			}

			goto GetNextElement;
		}
	}
	else
	{
		m_pCurState = &m_firstElmState;
		m_pCurState->uiCurOffset = 0;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readContinuationElm( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMINT32		i32ErrCode = 0;
	
	if( m_pCurState->elmInfo.uiElmNumber + 1 >= 
		 m_pCurState->blkUnion.pBTreeBlkHdr->ui16NumKeys)
	{
		F_BLK_HDR *		pBlkHdr = m_pCurState->blkUnion.pBlkHdr;

		if( RC_BAD( rc = readBlock( FALSE,
			FSGetFileNumber( pBlkHdr->ui32NextBlkInChain), 
			FSGetFileOffset( pBlkHdr->ui32NextBlkInChain), &m_tmpState)))
		{
			goto Exit;
		}

		if( m_tmpState.blkUnion.pBlkHdr->ui64TransID <
			m_firstElmState.blkUnion.pBlkHdr->ui64TransID)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		m_pCurState = &m_tmpState;
	}
	else
	{
		m_pCurState->elmInfo.uiElmNumber++;
	}
		
	// Extract information about the element
	
	if( (i32ErrCode = bldGetElmInfo( 
		m_pCurState->blkUnion.pBTreeBlkHdr, m_pCurState->uiBlockSize, 
		m_pCurState->elmInfo.uiElmNumber, &m_pCurState->elmInfo)) != 0)
	{
		if( RC_BAD( rc = m_pDbRebuild->reportCorruption( i32ErrCode,
			FSBlkAddress( m_pCurState->uiFileNumber, m_pCurState->uiFileOffset),
			m_pCurState->elmInfo.uiElmNumber, m_pCurState->elmInfo.ui64ElmNodeId)))
		{
			goto Exit;
		}

		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	if( bteFirstElementFlag( m_pCurState->elmInfo.pucElm))
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}
	
	// This had better not be a LEM element.

	if( !m_pCurState->elmInfo.uiElmKeyLen) 
	{
		m_pDbRebuild->reportCorruption( FLM_BAD_LEM,
			m_pCurState->elmInfo.ui32BlkAddr, 
			m_pCurState->elmInfo.uiElmNumber,
			m_pCurState->elmInfo.ui64ElmNodeId);
		goto Exit;
	}

	if( m_pCurState->elmInfo.ui64ElmNodeId != 
		 m_firstElmState.elmInfo.ui64ElmNodeId)
	{
		m_pDbRebuild->reportCorruption( FLM_BAD_CONT_ELM_KEY,
			m_pCurState->elmInfo.ui32BlkAddr, 
			m_pCurState->elmInfo.uiElmNumber, 
			m_pCurState->elmInfo.ui64ElmNodeId);
		goto Exit;
	}

	m_pCurState->uiCurOffset = 0;

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readFirstDataOnlyBlock( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucTmpBuf[ 8];
	F_ELM_INFO *	pElmInfo = &m_firstElmState.elmInfo;

	flmAssert( pElmInfo->uiDataOnlyBlkAddr);

	if( RC_BAD( rc = readBlock( FALSE,
		FSGetFileNumber( pElmInfo->uiDataOnlyBlkAddr),
		FSGetFileOffset( pElmInfo->uiDataOnlyBlkAddr), &m_tmpState)))
	{
		goto Exit;
	}

	if( m_tmpState.blkUnion.pBlkHdr->ui64TransID !=
		m_firstElmState.blkUnion.pBlkHdr->ui64TransID)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
		
	m_pCurState = &m_tmpState;
	m_pCurState->uiCurOffset = blkHdrSize( m_tmpState.blkUnion.pBlkHdr);
	m_pCurState->elmInfo.uiCollection = pElmInfo->uiCollection;
	m_pCurState->elmInfo.ui64ElmNodeId = pElmInfo->ui64ElmNodeId;
	m_pCurState->elmInfo.uiOverallDataLen = pElmInfo->uiOverallDataLen;

	// Since this is the first block in the data-only chain, the key
	// is stored in the first few bytes after the block header.  Make
	// sure it matches the key in m_firstElmState.

	if( RC_BAD( rc = read( ucTmpBuf, 2, NULL)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
		}

		goto Exit;
	}

	if( FB2UW( ucTmpBuf) != m_firstElmState.elmInfo.uiElmKeyLen)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
	}

	if( RC_BAD( rc = read( ucTmpBuf, 
		m_firstElmState.elmInfo.uiElmKeyLen, NULL)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
		}

		goto Exit;
	}

	if( f_memcmp( ucTmpBuf, m_firstElmState.elmInfo.pucElmKey,
		m_firstElmState.elmInfo.uiElmKeyLen) != 0)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::readNextDataOnlyBlock( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT32		ui32NextBlkAddr = m_pCurState->blkUnion.pBlkHdr->ui32NextBlkInChain;

	if( !ui32NextBlkAddr)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = readBlock( FALSE,
		FSGetFileNumber( ui32NextBlkAddr), 
		FSGetFileOffset( ui32NextBlkAddr), &m_tmpState)))
	{
		goto Exit;
	}

	if( m_tmpState.blkUnion.pBlkHdr->ui64TransID !=
		m_firstElmState.blkUnion.pBlkHdr->ui64TransID)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	m_pCurState = &m_tmpState;
	m_pCurState->uiCurOffset = blkHdrSize( m_tmpState.blkUnion.pBlkHdr);
	m_pCurState->elmInfo.uiCollection = m_firstElmState.elmInfo.uiCollection;
	m_pCurState->elmInfo.ui64ElmNodeId = m_firstElmState.elmInfo.ui64ElmNodeId;
	m_pCurState->elmInfo.uiOverallDataLen = m_firstElmState.elmInfo.uiOverallDataLen;

Exit:

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiBytesRead = 0;
	FLMUINT			uiBytesToCopy;
	F_BLK_HDR *		pBlkHdr;
	F_ELM_INFO *	pElmInfo;
	FLMBYTE *		pucBuffer = (FLMBYTE *)pvBuffer;
	
	flmAssert( m_bOpen);
	flmAssert( m_pCurState);

	while( uiBytesRead < uiBytesToRead)
	{
		pBlkHdr = m_pCurState->blkUnion.pBlkHdr;
		pElmInfo = &m_pCurState->elmInfo;

		if( pBlkHdr->ui8BlkType != BT_DATA_ONLY)
		{
			uiBytesToCopy = pElmInfo->uiElmDataLen - m_pCurState->uiCurOffset;
		}
		else
		{
			uiBytesToCopy = m_pCurState->uiBlockBytes - m_pCurState->uiCurOffset;
		}
						  
		uiBytesToCopy = (uiBytesToRead - uiBytesRead) < uiBytesToCopy
									? (uiBytesToRead - uiBytesRead)
									: uiBytesToCopy;

		if( !uiBytesToCopy)
		{
			if( pBlkHdr->ui8BlkType != BT_DATA_ONLY)
			{
				if( RC_BAD( rc = readContinuationElm()))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = readNextDataOnlyBlock()))
				{
					goto Exit;
				}
			}
			
			continue;
		}

		if( pucBuffer)
		{
			if( pBlkHdr->ui8BlkType != BT_DATA_ONLY)
			{
				f_memcpy( pucBuffer, 
					&pElmInfo->pucElmData[ m_pCurState->uiCurOffset], 
					uiBytesToCopy);
			}
			else
			{
				f_memcpy( pucBuffer, 
					&m_pCurState->blkUnion.pucBlk[ m_pCurState->uiCurOffset], 
					uiBytesToCopy);
			}

			pucBuffer += uiBytesToCopy;
		}

		m_pCurState->uiCurOffset += uiBytesToCopy;
		uiBytesRead += uiBytesToCopy;
	}
	
	if( uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}
	
	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::getNextNode(
	F_CachedNode **	ppCachedNode,
	F_ELM_INFO *		pElmInfo,
	FLMBYTE *			pucIV)
{
	RCODE					rc = NE_XFLM_OK;
	F_CachedNode *		pCachedNode = NULL;

	if( ppCachedNode)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		if( *ppCachedNode)
		{
			pCachedNode = *ppCachedNode;
			*ppCachedNode = NULL;
			pCachedNode->decrNodeUseCount();
			pCachedNode->resetNode();
			pCachedNode->incrNodeUseCount();
		}
		else
		{
			if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocNode( 
				&pCachedNode, TRUE)))
			{
				f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
				goto Exit;
			}

			pCachedNode->incrNodeUseCount();
		}
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	// Read the node

	for( ;;)
	{
		if( RC_BAD( rc = readNextFirstElm()))
		{
			goto Exit;
		}

		if( pElmInfo)
		{
			f_memcpy( pElmInfo, &m_firstElmState.elmInfo, sizeof( F_ELM_INFO));
		}

		if( !ppCachedNode)
		{
			break;
		}

		if( RC_OK( rc = pCachedNode->readNode( m_pDbRebuild->m_pDb, 
			m_firstElmState.elmInfo.uiCollection, 
			m_firstElmState.elmInfo.ui64ElmNodeId, this, 
			m_firstElmState.elmInfo.uiOverallDataLen, pucIV)))
		{
			break;
		}

		if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_MEM)
		{
			goto Exit;
		}

		rc = NE_XFLM_OK;

		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		pCachedNode->decrNodeUseCount();
		pCachedNode->resetNode();
		pCachedNode->incrNodeUseCount();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	if( ppCachedNode)
	{
		*ppCachedNode = pCachedNode;
		pCachedNode = NULL;
	}

Exit:

	if( pCachedNode)
	{
		bldFreeCachedNode( &pCachedNode);
	}

	return( rc);
}

/***************************************************************************
Desc:
*****************************************************************************/
RCODE F_RebuildNodeIStream::getNextNodeInfo(
	F_ELM_INFO *		pElmInfo,
	F_NODE_INFO *		pNodeInfo)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiTmpFlags;

	// Read the node info

	for( ;;)
	{
		if( RC_BAD( rc = readNextFirstElm()))
		{
			goto Exit;
		}

		f_memcpy( pElmInfo, &m_firstElmState.elmInfo, sizeof( F_ELM_INFO));

		if( RC_OK( rc = flmReadNodeInfo( 
			m_firstElmState.elmInfo.uiCollection,
			m_firstElmState.elmInfo.ui64ElmNodeId, 
			this, m_firstElmState.elmInfo.uiOverallDataLen, FALSE,
			pNodeInfo, &uiTmpFlags)))
		{
			break;
		}

		if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_MEM)
		{
			goto Exit;
		}

		rc = NE_XFLM_OK;
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine retrieves one node starting at the current element.
*****************************************************************************/
RCODE F_RebuildNodeIStream::readNode(
	FLMUINT32					ui32BlkAddr,
	FLMUINT						uiElmNumber,
	F_CachedNode **			ppCachedNode,
	FLMBYTE *					pucIV)
{
	RCODE							rc = NE_XFLM_OK;
	F_CachedNode *				pCachedNode = NULL;
	FLMINT32						i32ErrCode = 0;

	m_pCurState = NULL;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( *ppCachedNode)
	{
		pCachedNode = *ppCachedNode;
		*ppCachedNode = NULL;
		pCachedNode->decrNodeUseCount();
		pCachedNode->resetNode();
		pCachedNode->incrNodeUseCount();
	}
	else
	{
		if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocNode( 
			&pCachedNode, TRUE)))
		{
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			goto Exit;
		}

		pCachedNode->incrNodeUseCount();
	}
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);

	if( RC_BAD( rc = readBlock( FALSE, FSGetFileNumber( ui32BlkAddr),
		FSGetFileOffset( ui32BlkAddr), &m_firstElmState)))
	{
		goto Exit;
	}
		
	m_pCurState = &m_firstElmState;

	// Extract information about the element
	
	if( (i32ErrCode = bldGetElmInfo( m_firstElmState.blkUnion.pBTreeBlkHdr, 
		m_firstElmState.uiBlockSize, uiElmNumber, &m_firstElmState.elmInfo)) != 0)
	{
		if( RC_BAD( rc = m_pDbRebuild->reportCorruption( i32ErrCode,
			FSBlkAddress( m_firstElmState.uiFileNumber, m_firstElmState.uiFileOffset),
			m_firstElmState.elmInfo.uiElmNumber, 
			m_firstElmState.elmInfo.ui64ElmNodeId)))
		{
			goto Exit;
		}

		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	if( !bteFirstElementFlag( m_firstElmState.elmInfo.pucElm))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	if( m_firstElmState.elmInfo.uiDataOnlyBlkAddr)
	{
		if( RC_BAD( rc = readFirstDataOnlyBlock()))
		{
			goto Exit;
		}
	}
	else
	{
		m_pCurState = &m_firstElmState;
		m_pCurState->uiCurOffset = 0;
	}

	// Read the node

	if( RC_BAD( rc = pCachedNode->readNode( m_pDbRebuild->m_pDb, 
		m_pCurState->elmInfo.uiCollection, 
		m_pCurState->elmInfo.ui64ElmNodeId, this, 
		m_pCurState->elmInfo.uiOverallDataLen, pucIV)))
	{
		goto Exit;
	}

	*ppCachedNode = pCachedNode;
	pCachedNode = NULL;
	
Exit:

	if( pCachedNode)
	{
		bldFreeCachedNode( &pCachedNode);
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine reads through a database and makes a best guess as to 
		the true block size of the database.
*****************************************************************************/
RCODE F_DbRebuild::determineBlkSize(
	FLMUINT *			puiBlkSizeRV)
{
	RCODE					rc = NE_XFLM_OK;
	F_BLK_HDR			blkHdr;
	FLMUINT				uiBytesRead;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiFileNumber = 0;
	FLMUINT				uiOffset = 0;
	FLMUINT				uiCount4K = 0;
	FLMUINT				uiCount8K = 0;
	FLMUINT64			ui64BytesDone = 0;
	IF_FileHdl *		pFileHdl = NULL;

	// Start from byte offset 0 in the first file.

	m_callbackData.i32DoingFlag = REBUILD_GET_BLK_SIZ;
	m_callbackData.bStartFlag = TRUE;

	for (;;)
	{
		if( uiOffset >= m_dbHdr.ui32MaxFileSize || !uiFileNumber)
		{
TryNextFile:
			uiOffset = 0;
			uiFileNumber++;
		}

		if( RC_BAD( rc = pFileHdl->read( uiOffset, 
			SIZEOF_STD_BLK_HDR, &blkHdr, &uiBytesRead)))
		{
			if( rc == NE_XFLM_EOF_HIT ||
				 rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				goto TryNextFile;
			}
			
			goto Exit;
		}

		ui64BytesDone += (FLMUINT64)XFLM_MIN_BLOCK_SIZE;
		uiBlkAddress = (FLMUINT)blkHdr.ui32BlkAddr;

		// If the block address does not match up, try converting it.

		if( FSGetFileOffset( uiBlkAddress) != uiOffset)
		{
			convert32( &blkHdr.ui32BlkAddr);
			uiBlkAddress = (FLMUINT)blkHdr.ui32BlkAddr;
		}
		
		if( FSGetFileOffset( uiBlkAddress) == uiOffset)
		{
			if( uiOffset % 4096 == 0)
			{
				if( ++uiCount4K >= 1000)
				{
					break;
				}
			}

			if( uiOffset % 8192 == 0)
			{
				if( ++uiCount8K >= 1000)
				{
					break;
				}
			}
		}
		
		uiOffset += XFLM_MIN_BLOCK_SIZE;

		if( RC_BAD( rc = reportStatus()))
		{
			goto Exit;
		}
	}

	if( uiCount8K > uiCount4K)
	{
		*puiBlkSizeRV = 8192;
	}
	else
	{
		*puiBlkSizeRV = 4096;
	}
	
Exit:

	return( rc);
}
