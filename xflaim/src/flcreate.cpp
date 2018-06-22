//------------------------------------------------------------------------------
// Desc:	Create a database.
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

/*API~***********************************************************************
Desc:	Creates a new FLAIM database.
*END************************************************************************/
RCODE XFLAPI F_DbSystem::dbCreate(
	const char *			pszFilePath,
		// [IN] Full path file name of the database which is to be created.
	const char *			pszDataDir,
		// [IN] Directory for data files.
	const char *			pszRflDir,
		// [IN] RFL directory.  NULL indicates that the RFL files should
		// be put in the same directory as the database.
	const char *			pszDictFileName,
		// [IN] Full path of a file containing dictionary definitions to be
		// imported into the dictionary collection during database
		// creation.  This is only used if pszDictBuf is NULL.  If
		// both pszDictFileName and pszDictBuf are NULL, the database
		// will be created with an empty dictionary.
	const char *			pszDictBuf,
		// [IN] Buffer containing dictionary definitions in external XML
		// format.  If the value of this parameter is NULL,
		// pszDictFileName will be used.
	XFLM_CREATE_OPTS *	pCreateOpts,
		// [IN] Create options for the database.  All members of the
		// structure should be initialized through a call to f_memset:
		//
		//		f_memset( pCreateOpts, 0, sizeof( XFLM_CREATE_OPTS));
		//
		// Once initialized, the values of specific members can be set to
		// reflect the options desired when the database is created (such
		// as block size and various roll-forward logging options).  If
		// NULL is passed as the value of this parameter, default options
		// will be used.
	FLMBOOL					bTempDb,
		// [IN] Flag indicating whether this is a temporary database.
		// Should try to minimize writing to disk if this is the case.
	IF_Db **					ppDb
		// [OUT] Pointer to a database object.  If the creation is
		// successful, the database object will be initialized.
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = NULL;
	F_Database *	pDatabase = NULL;
	FLMBOOL			bDatabaseCreated = FALSE;
	FLMBOOL			bNewDatabase = FALSE;
	FLMBOOL			bMutexLocked = FALSE;
	FLMUINT			uiRflToken = 0;

	// Make sure the path looks valid
	
	if (!pszFilePath || !pszFilePath [0])
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	// Allocate and initialize an F_Db structure.

	if (RC_BAD( rc = allocDb( &pDb, FALSE)))
	{
		goto Exit;
	}

	f_mutexLock( gv_XFlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	for( ;;)
	{

		// See if we already have the file open.
		// May unlock and re-lock the global mutex.

		if (RC_BAD( rc = findDatabase( pszFilePath, pszDataDir, &pDatabase)))
		{
			goto Exit;
		}

		// Didn't find the database

		if (!pDatabase)
		{
			break;
		}

		// See if file is open or being opened.

		if (pDatabase->m_uiOpenIFDbCount || (pDatabase->m_uiFlags & DBF_BEING_OPENED))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}

		// Free the F_Database object.  May temporarily unlock the global mutex.
		// For this reason, we must call findDatabase again (see above) after
		// freeing the object.

		pDatabase->freeDatabase();
		pDatabase = NULL;
	}

	// Allocate a new F_Database object

	if (RC_BAD( rc = allocDatabase( pszFilePath, pszDataDir, bTempDb, &pDatabase)))
	{
		goto Exit;
	}

	bNewDatabase = TRUE;
	pDatabase->m_uiMaxFileSize = gv_XFlmSysData.uiMaxFileSize;

	// Link the F_Db object to the F_Database object.

	rc = pDb->linkToDatabase( pDatabase);
	
	f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	// If the database has not already been created, do so now.

	if (RC_OK( gv_XFlmSysData.pFileSystem->doesFileExist( pszFilePath)))
	{
		rc = RC_SET( NE_XFLM_FILE_EXISTS);
		goto Exit;
	}

	// Create the .db file.

	pDb->m_pSFileHdl->setMaxAutoExtendSize( gv_XFlmSysData.uiMaxFileSize);
	pDb->m_pSFileHdl->setExtendSize( pDb->m_pDatabase->m_uiFileExtendSize);
	
	if (RC_BAD( rc = pDb->m_pSFileHdl->createFile( 0)))
	{
		goto Exit;
	}
	bDatabaseCreated = TRUE;

	(void)flmStatGetDb( &pDb->m_Stats, pDatabase,
						0, &pDb->m_pDbStats, NULL, NULL);

	// We must have exclusive access.  Create a lock file for that
	// purpose, if there is not already a lock file.
	// NOTE: No need for a lock file if this is a temporary database.

	if (!bTempDb)
	{
		if (RC_BAD( rc = pDatabase->getExclAccess( pszFilePath)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = pDb->initDbFiles( pszRflDir, pszDictFileName,
											 pszDictBuf, pCreateOpts)))
	{
		goto Exit;
	}

	// Disable RFL logging (m_pRfl was initialized in initDbFiles)

	if( pDatabase->m_pRfl)
	{
		pDatabase->m_pRfl->disableLogging( &uiRflToken);
	}

	// Set FFILE stuff to same state as a completed checkpoint.

	pDatabase->m_uiFirstLogCPBlkAddress = 0;
	pDatabase->m_uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();

	// Create a checkpoint thread - no need if this is a temporary
	// database.

	if (!bTempDb)
	{
		if (RC_BAD( rc = pDatabase->startCPThread()))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pDatabase->startMaintThread()))
		{
			goto Exit;
		}
	}
		
Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	}

	if (pDb)
	{
		pDb->completeOpenOrCreate( rc, bNewDatabase);

		// completeOpenOrCreate will delete pDb if RC_BAD( rc)

		if (RC_BAD( rc))
		{
			pDb = NULL;
		}
		else
		{
			*ppDb = (IF_Db *)pDb;
			pDb = NULL;	// This isn't strictly necessary, but it makes it
							// obvious that we are no longer using the object.
		}
	}

	if (RC_BAD( rc))
	{
		if (bDatabaseCreated)
		{
			gv_pXFlmDbSystem->dbRemove( pszFilePath, pszDataDir, pszRflDir, TRUE);
		}
	}
	else if( uiRflToken)
	{
		pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	return( rc);
}

/****************************************************************************
Desc:	Create a database - initialize all physical areas & data dictionary.
****************************************************************************/
RCODE F_Db::initDbFiles(
	const char *			pszRflDir,
	const char *			pszDictFileName,	// Name of dictionary file.  This
														// is only used if pszDictBuf is
														// NULL.  If both pszDictFileName
														// and pszDictBuf are NULL, the
														// file will be craeted with an empty
														// dictionary.
	const char *			pszDictBuf,			// Buffer containing dictionary in
														// XML format.  If NULL,
														// pszDictFileName will be used.
	XFLM_CREATE_OPTS *	pCreateOpts)		// Create options for the database.
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				bTransStarted = FALSE;
	FLMBYTE *			pucBuf = NULL;
	FLMUINT				uiBlkSize;
	FLMUINT				uiWriteBytes;
	FLMUINT				uiRflToken = 0;
	XFLM_DB_HDR *		pDbHdr;
	F_BLK_HDR *			pBlkHdr;
	F_CachedBlock *	pSCache = NULL;
	FLMBYTE *			pucWrappingKey = NULL;
	FLMUINT32			ui32KeyLen = 0;

	// Determine what size of buffer to allocate.

	uiBlkSize = (FLMUINT)(pCreateOpts
								 ? flmAdjustBlkSize( (FLMUINT)pCreateOpts->ui32BlockSize)
								 : (FLMUINT)XFLM_DEFAULT_BLKSIZ);

	// Allocate a buffer for writing.

	if (RC_BAD( rc = f_alloc( (FLMUINT)uiBlkSize, &pucBuf)))
	{
		goto Exit;
	}

	// Initialize the database header structure.

	pDbHdr = (XFLM_DB_HDR *)pucBuf;
	flmInitDbHdr( pCreateOpts, TRUE, m_pDatabase->m_bTempDb, pDbHdr);
	m_pDatabase->m_uiBlockSize = (FLMUINT)pDbHdr->ui16BlockSize;
	m_pDatabase->m_uiDefaultLanguage = (FLMUINT)pDbHdr->ui8DefaultLanguage;
	m_pDatabase->m_uiMaxFileSize = (FLMUINT)pDbHdr->ui32MaxFileSize;
	m_pDatabase->m_uiSigBitsInBlkSize = calcSigBits( uiBlkSize);

	f_memcpy( &m_pDatabase->m_lastCommittedDbHdr, pDbHdr, sizeof( XFLM_DB_HDR));

	// Create the first block file.

	if (!m_pDatabase->m_bTempDb)
	{
		if (RC_BAD( rc = m_pSFileHdl->createFile( 1)))
		{
			goto Exit;
		}
	}
	
	if( RC_OK( rc = createDbKey()))
	{
		if (RC_BAD( rc = m_pDatabase->m_pWrappingKey->getKeyToStore(
								&pucWrappingKey,
								&ui32KeyLen,
								m_pDatabase->m_pszDbPasswd,
								NULL)))
		{
			goto Exit;
		}
	
		f_memcpy( m_pDatabase->m_lastCommittedDbHdr.DbKey,
					 pucWrappingKey,
					 ui32KeyLen);
		m_pDatabase->m_lastCommittedDbHdr.ui32DbKeyLen = ui32KeyLen;
	
		m_pDatabase->m_rcLimitedCode = NE_XFLM_OK;
		m_pDatabase->m_bInLimitedMode = FALSE;
		m_pDatabase->m_bHaveEncKey = TRUE;
	}
	else if( rc == NE_XFLM_ENCRYPTION_UNAVAILABLE)
	{
		rc = NE_XFLM_OK;
		m_pDatabase->m_rcLimitedCode = NE_XFLM_ENCRYPTION_UNAVAILABLE;
		m_pDatabase->m_bInLimitedMode = TRUE;
		m_pDatabase->m_bHaveEncKey = FALSE;
	}
	else
	{
		goto Exit;
	}

	// Write out the log header

	if (RC_BAD( rc = m_pDatabase->writeDbHdr( m_pDbStats, m_pSFileHdl,
							&m_pDatabase->m_lastCommittedDbHdr,
							NULL, TRUE)))
	{
		goto Exit;
	}

	// Initialize and output the first LFH block

	if (m_pDatabase->m_bTempDb)
	{
		getDbHdrInfo( &m_pDatabase->m_lastCommittedDbHdr);
		if (RC_BAD( rc = m_pDatabase->createBlock( this, &pSCache)))
		{
			goto Exit;
		}
		pBlkHdr = (F_BLK_HDR *)pSCache->m_pBlkHdr;
		m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr = pBlkHdr->ui32BlkAddr;
	}
	else
	{
		// Copy the Db header to the checkpointDbHdr buffer.
		// This is now the first official checkpoint version of the log
		// header.  It must be copied to the checkpointDbHdr buffer so that
		// it will not be lost in subsequent calls to flmWriteDbHdr.

		f_memcpy( &m_pDatabase->m_checkpointDbHdr, &m_pDatabase->m_lastCommittedDbHdr,
						sizeof( XFLM_DB_HDR));

		f_memset( pucBuf, 0, uiBlkSize);
		pBlkHdr = (F_BLK_HDR *)pucBuf;
		pBlkHdr->ui32BlkAddr = m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr;
		pBlkHdr->ui64TransID = 0;
	}
	pBlkHdr->ui8BlkType = BT_LFH_BLK;
	pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)(uiBlkSize - SIZEOF_STD_BLK_HDR);
	blkSetNativeFormat( pBlkHdr);

	if (!m_pDatabase->m_bTempDb)
	{
		pBlkHdr->ui32BlkCRC = calcBlkCRC( pBlkHdr, SIZEOF_STD_BLK_HDR);
		if (RC_BAD( rc = m_pSFileHdl->writeBlock( pBlkHdr->ui32BlkAddr,
			uiBlkSize, pucBuf, &uiWriteBytes)))
		{
			goto Exit;
		}

		// Force things to disk.

		if (RC_BAD( rc = m_pSFileHdl->flush()))
		{
			goto Exit;
		}
	}

	// Allocate the pRfl object.  Could not do this until this point
	// because we need to have the version number, block size, etc.
	// setup in the database header

	flmAssert( !m_pDatabase->m_pRfl);

	if (!m_pDatabase->m_bTempDb)
	{
		if ((m_pDatabase->m_pRfl = f_new F_Rfl) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pDatabase->m_pRfl->setup( m_pDatabase, pszRflDir)))
		{
			goto Exit;
		}
		
		// Disable RFL logging

		m_pDatabase->m_pRfl->disableLogging( &uiRflToken);

		// Start an update transaction and populate the dictionary.
		// This also creates the default collections and indexes.

		if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bTransStarted = TRUE;

		if (RC_BAD( rc = dictCreate( pszDictFileName, pszDictBuf)))
		{
			goto Exit;
		}

		// Because the checkpoint thread has not yet been created,
		// flmCommitDbTrans will force a checkpoint when it completes,
		// ensuring a consistent database state.

		bTransStarted = FALSE;
		if (RC_BAD( rc = commitTrans( 0, TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		// The uncommitted header must have all of the stuff from the committed header
		// in order for this to be able to work as if an update transaction was in
		// progress.

		f_memcpy( &m_pDatabase->m_uncommittedDbHdr, &m_pDatabase->m_lastCommittedDbHdr,
			sizeof( XFLM_DB_HDR));
	}

Exit:

	// Free the temporary buffer, if it was allocated.

	f_free( &pucBuf);
	
	if (pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}

	if (bTransStarted)
	{
		abortTrans();
	}

	if( uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}
