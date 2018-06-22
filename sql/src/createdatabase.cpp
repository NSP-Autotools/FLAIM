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

/****************************************************************************
Desc:	Create a database - initialize all physical areas & data dictionary.
****************************************************************************/
RCODE F_Db::initDbFiles(
	const char *			pszRflDir,
	SFLM_CREATE_OPTS *	pCreateOpts)		// Create options for the database.
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				bTransStarted = FALSE;
	FLMBYTE *			pucBuf = NULL;
	FLMUINT				uiBlkSize;
	FLMUINT				uiWriteBytes;
	FLMUINT				uiRflToken = 0;
	SFLM_DB_HDR *		pDbHdr;
	F_BLK_HDR *			pBlkHdr;
	F_CachedBlock *	pSCache = NULL;
	FLMBYTE *			pucWrappingKey = NULL;
#ifdef FLM_USE_NICI
	FLMUINT32			ui32KeyLen = 0;
#endif

	// Determine what size of buffer to allocate.

	uiBlkSize = (FLMUINT)(pCreateOpts
								 ? flmAdjustBlkSize( pCreateOpts->uiBlockSize)
								 : (FLMUINT)SFLM_DEFAULT_BLKSIZ);

	// Allocate a buffer for writing.

	if (RC_BAD( rc = f_alloc( (FLMUINT)uiBlkSize, &pucBuf)))
	{
		goto Exit;
	}

	// Initialize the database header structure.

	pDbHdr = (SFLM_DB_HDR *)pucBuf;
	flmInitDbHdr( pCreateOpts, TRUE, m_pDatabase->m_bTempDb, pDbHdr);
	m_pDatabase->m_uiBlockSize = (FLMUINT)pDbHdr->ui16BlockSize;
	m_pDatabase->m_uiDefaultLanguage = (FLMUINT)pDbHdr->ui8DefaultLanguage;
	m_pDatabase->m_uiMaxFileSize = (FLMUINT)pDbHdr->ui32MaxFileSize;
	m_pDatabase->m_uiSigBitsInBlkSize = calcSigBits( uiBlkSize);

	f_memcpy( &m_pDatabase->m_lastCommittedDbHdr, pDbHdr, sizeof( SFLM_DB_HDR));

	// Create the first block file.

	if (!m_pDatabase->m_bTempDb)
	{
		if (RC_BAD( rc = m_pSFileHdl->createFile( 1)))
		{
			goto Exit;
		}
	}
	
#ifdef FLM_USE_NICI
	if (RC_BAD( rc = createDbKey()))
	{
		goto Exit;
	}
	
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

	m_pDatabase->m_rcLimitedCode = NE_SFLM_OK;
	m_pDatabase->m_bInLimitedMode = FALSE;
	m_pDatabase->m_bHaveEncKey = TRUE;

#else
	m_pDatabase->m_rcLimitedCode = NE_SFLM_ENCRYPTION_UNAVAILABLE;
	m_pDatabase->m_bInLimitedMode = TRUE;
	m_pDatabase->m_bHaveEncKey = FALSE;
#endif

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
						sizeof( SFLM_DB_HDR));

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
		if (RC_BAD( rc = m_pSFileHdl->writeBlock(
									(FLMUINT)pBlkHdr->ui32BlkAddr,
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
			rc = RC_SET( NE_SFLM_MEM);
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

		if (RC_BAD( rc = beginTrans( SFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}
		bTransStarted = TRUE;

		if (RC_BAD( rc = dictCreate()))
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
			sizeof( SFLM_DB_HDR));
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

//------------------------------------------------------------------------------
// Desc:	Creates a new database.
//------------------------------------------------------------------------------
RCODE F_DbSystem::createDatabase(
	const char *			pszFilePath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	SFLM_CREATE_OPTS *	pCreateOpts,
	FLMBOOL					bTempDb,
	F_Db **					ppDb)
{
	RCODE				rc = NE_SFLM_OK;
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

	f_mutexLock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Free any unused structures that have been unused for the maximum
	// amount of time.  May unlock and re-lock the global mutex.

	checkNotUsedObjects();

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

	// Link the F_Db object to the F_Database object.

	rc = pDb->linkToDatabase( pDatabase);
	f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	bMutexLocked = FALSE;
	if (RC_BAD( rc))
	{
		goto Exit;
	}

	// If the database has not already been created, do so now.

	// Determine what to set file block size to.

	if (pCreateOpts != NULL)
	{
//		JMC - FIXME: commented out due to missing function in ftk
//		pDb->m_pSFileHdl->setBlockSize(
//			flmAdjustBlkSize( pCreateOpts->uiBlockSize));
	}
	else
	{
//		JMC - FIXME: commented out due to missing function in ftk
//		pDb->m_pSFileHdl->setBlockSize( SFLM_DEFAULT_BLKSIZ);
	}

	if (RC_OK( gv_SFlmSysData.pFileSystem->doesFileExist( pszFilePath)))
	{
		rc = RC_SET( NE_SFLM_DATABASE_EXISTS);
		goto Exit;
	}

	// Create the .db file.

	pDb->m_pSFileHdl->setMaxAutoExtendSize( gv_SFlmSysData.uiMaxFileSize);
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

	if (RC_BAD( rc = pDb->initDbFiles( pszRflDir, pCreateOpts)))
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
		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
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
			*ppDb = pDb;
			pDb = NULL;	// This isn't strictly necessary, but it makes it
							// obvious that we are no longer using the object.
		}
	}

	if (RC_BAD( rc))
	{
		if (bDatabaseCreated)
		{
			F_DbSystem	dbSystem;

			dbSystem.dropDatabase( pszFilePath, pszDataDir, pszRflDir, TRUE);
		}
	}
	else if( uiRflToken)
	{
		pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the create database statement.  The "CREATE DATABASE" keywords
//			have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processCreateDatabase( void)
{
	RCODE					rc = NE_SFLM_OK;
	char					szDatabaseName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDatabaseNameLen;
	char					szDataDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDataDirNameLen;
	char					szRflDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiRflDirNameLen;
	char					szLanguage [10];
	FLMUINT				uiLanguageLen;
	SFLM_CREATE_OPTS	createOpts;
	F_DbSystem			dbSystem;
	char					szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT				uiTokenLineOffset;
	
	// SYNTAX: CREATE DATABASE databasename [CREATE_OPTIONS( <ParamName>=value,...)]

	// Whitespace must follow the "CREATE DATABASE"

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
	
	createOpts.uiBlockSize = SFLM_DEFAULT_BLKSIZ;
	createOpts.uiVersionNum = SFLM_CURRENT_VERSION_NUM;
	createOpts.uiMinRflFileSize = SFLM_DEFAULT_MIN_RFL_FILE_SIZE;
	createOpts.uiMaxRflFileSize = SFLM_DEFAULT_MAX_RFL_FILE_SIZE;
	createOpts.bKeepRflFiles = SFLM_DEFAULT_KEEP_RFL_FILES_FLAG;
	createOpts.bLogAbortedTransToRfl = SFLM_DEFAULT_LOG_ABORTED_TRANS_FLAG;
	createOpts.uiDefaultLanguage = SFLM_DEFAULT_LANG;
	
	// See if there are other create options.
	
	// Whitespace must follow the database name

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// See if there are "CREATE_OPTIONS"

	if (RC_BAD( rc = haveToken( "create_options", TRUE)))
	{
		if (rc == NE_SFLM_NOT_FOUND)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
	
		// Left paren must follow "CREATE_OPTIONS"
		
		if (RC_BAD( rc = haveToken( "(", FALSE, SQL_ERR_EXPECTING_LPAREN)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
											&uiTokenLineOffset, NULL)))
		{
			goto Exit;
		}
		
		// See if the list of create options is empty.
		
		if (f_stricmp( szToken, ")") == 0)
		{
			goto Create_Database;
		}
		
		// Get all of the create options
		
		for (;;)
		{
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
			else if (f_stricmp( szToken, SFLM_BLOCK_SIZE_STR) == 0)
			{
				if (RC_BAD( rc = getUINT( TRUE, &createOpts.uiBlockSize)))
				{
					goto Exit;
				}
				if (!dbSystem.validBlockSize( createOpts.uiBlockSize))
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset,
							SQL_ERR_INVALID_BLOCK_SIZE,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, SFLM_MIN_RFL_SIZE_STR) == 0)
			{
				if (RC_BAD( rc = getUINT( TRUE, &createOpts.uiMinRflFileSize)))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, SFLM_MAX_RFL_SIZE_STR) == 0)
			{
				if (RC_BAD( rc = getUINT( TRUE, &createOpts.uiMaxRflFileSize)))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, SFLM_KEEP_RFL_STR) == 0)
			{
				if (RC_BAD( rc = getBool( TRUE, &createOpts.bKeepRflFiles)))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, SFLM_LOG_ABORTED_TRANS_STR) == 0)
			{
				if (RC_BAD( rc = getBool( TRUE, &createOpts.bLogAbortedTransToRfl)))
				{
					goto Exit;
				}
			}
			else if (f_stricmp( szToken, SFLM_LANGUAGE_STR) == 0)
			{
				if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szLanguage,
										sizeof( szLanguage),
										&uiLanguageLen, NULL, NULL)))
				{
					goto Exit;
				}
				createOpts.uiDefaultLanguage = f_languageToNum( szLanguage);
				
				// If the default language was returned, make sure that the
				// string corresponds to it.
				
				if (createOpts.uiDefaultLanguage == FLM_US_LANG)
				{
					if (f_stricmp( szLanguage, "US") != 0)
					{
						setErrInfo( m_uiCurrLineNum,
								m_uiCurrLineOffset,
								SQL_ERR_INVALID_LANGUAGE,
								m_uiCurrLineFilePos,
								m_uiCurrLineBytes);
						rc = RC_SET( NE_SFLM_INVALID_SQL);
						goto Exit;
					}
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_INVALID_DB_CREATE_PARAM,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			
			// Option must be followed by a comma or right paren
			
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
												&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
			
			if (f_stricmp( szToken, ")") == 0)
			{
				break;
			}
			else if (f_stricmp( szToken, ",") != 0)
			{
				setErrInfo( m_uiCurrLineNum,
						uiTokenLineOffset,
						SQL_ERR_EXPECTING_COMMA,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			
			// Comma must be followed by the next option
			
			if (RC_BAD( rc = getToken( szToken, sizeof( szToken), FALSE,
												&uiTokenLineOffset, NULL)))
			{
				goto Exit;
			}
		}
	}
	
Create_Database:
	
	if (m_pDb)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}
	
	// Create the database
	
	if (RC_BAD( rc = dbSystem.createDatabase( szDatabaseName, szDataDirName,
										szRflDirName, &createOpts, &m_pDb)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

