//-------------------------------------------------------------------------
// Desc:	Create database.
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

FSTATIC RCODE flmInitNewFile(
	FDB *					pDb,
	const char *		pszRflDir,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	FLMUINT				uiTransID);

FSTATIC RCODE flmInitFileHdrs(
	FDB *					pDb,
	CREATE_OPTS *		pCreateOpts,
	FLMUINT				uiBlkSize,
	FLMUINT				uiTransID,
	FLMBYTE *			pInitBuf);

/****************************************************************************
Desc : Creates a new FLAIM database.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbCreate(
	const char *		pszDbFileName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	HFDB *				phDbRV)
{
	RCODE					rc = FERR_OK;
	CS_CONTEXT *		pCSContext;

	if( phDbRV)
	{
		*phDbRV = HFDB_NULL;
	}

	if( !pszDbFileName || !pszDbFileName[ 0])
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
		goto Exit;
	}

	if (RC_BAD( rc = flmGetCSConnection(
								pszDbFileName, &pCSContext)))
	{
		goto Exit;
	}

	if (pCSContext)
	{

		if( RC_BAD( rc = flmOpenOrCreateDbClientServer( pszDbFileName,
			pszDataDir, pszRflDir, 0, pszDictFileName,
			pszDictBuf, pCreateOpts, FALSE, pCSContext, (FDB * *)phDbRV)))
		{
			(void)flmCloseCSConnection( &pCSContext);
		}
		goto Exit;
	}

	if( RC_BAD( rc = flmCreateNewFile( pszDbFileName, pszDataDir, pszRflDir,
		pszDictFileName, pszDictBuf, pCreateOpts,
		0, (FDB * *)phDbRV)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine creates a FLAIM file.
****************************************************************************/
RCODE flmCreateNewFile(
	const char *			pszFilePath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszDictFileName,
	const char *			pszDictBuf,
	CREATE_OPTS *			pCreateOpts,
	FLMUINT					uiTransID,
	FDB * *					ppDb,
	REBUILD_STATE *		pRebuildState)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = NULL;
	FFILE *					pFile;
	FLMUINT					uiMaxFileSize;
	FLMBOOL					bFileCreated = FALSE;
	FLMBOOL					bNewFile = FALSE;
	FLMBOOL					bMutexLocked = FALSE;
	F_SuperFileClient *	pSFileClient = NULL;

	if( ppDb)
	{
		*ppDb = NULL;
	}

#ifndef FLM_USE_NICI
	F_UNREFERENCED_PARM( pRebuildState);
#endif

	// Allocate and initialize an FDB structure.

	if (RC_BAD( rc = flmAllocFdb( &pDb)))
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Free any unused structures that have been unused for the maximum
	// amount of time.  May unlock and re-lock the global mutex.

	flmCheckNUStructs( 0);

	for( ;;)
	{
		// See if we already have the file open.
		// May unlock and re-lock the global mutex.

		if (RC_BAD( rc = flmFindFile( pszFilePath, pszDataDir, &pFile)))
		{
			goto Exit;
		}

		// Didn't find the file

		if( !pFile)
		{
			break;
		}

		// See if file is being used, is being opened, or has any dependent
		// files being used.

		if (pFile->uiUseCount || (pFile->uiFlags & DBF_BEING_OPENED))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}

		// Free the FFILE structure.  May temporarily unlock the global mutex.
		// For this reason, we must call flmFindFile again (see above) after
		// calling flmFreeFile.

		flmFreeFile( pFile);
		pFile = NULL;
	}

	// Allocate a new FFILE structure.

	if (RC_BAD( rc = flmAllocFile( pszFilePath, pszDataDir, NULL, &pFile)))
	{
		goto Exit;
	}
	bNewFile = TRUE;

	if (pCreateOpts != NULL)
	{
		pFile->FileHdr.uiBlockSize = flmAdjustBlkSize( pCreateOpts->uiBlockSize);
		pFile->FileHdr.uiVersionNum = pCreateOpts->uiVersionNum;
	}
	else
	{
		pFile->FileHdr.uiBlockSize = DEFAULT_BLKSIZ;
		pFile->FileHdr.uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
	}

	// Link the FDB to the file.

	rc = flmLinkFdbToFile( pDb, pFile);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	if (RC_BAD( rc))
	{
		goto Exit;
	}

#ifdef FLM_USE_NICI

	// Create a new F_CCS object for the database wrapping key if the new
	// database version is at least ver 4.60
	
	if (!pCreateOpts || pCreateOpts->uiVersionNum >= FLM_FILE_FORMAT_VER_4_60)
	{
		if ((pFile->pDbWrappingKey = f_new F_CCS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (RC_BAD( rc = pFile->pDbWrappingKey->init( TRUE,
																	 FLM_NICI_AES)))
		{
			goto Exit;
		}
		
		// Only generate a key when this is not part of a rebuild operation or
		// the original database version was less than 4.60
		
		if (!pRebuildState || pRebuildState->pHdrInfo->FileHdr.uiVersionNum <
										FLM_FILE_FORMAT_VER_4_60)
		{
			if (RC_BAD( rc = pFile->pDbWrappingKey->generateWrappingKey()))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pFile->pDbWrappingKey->setKeyFromStore(
				&pRebuildState->pLogHdr[ LOG_DATABASE_KEY],
				FB2UW(&pRebuildState->pLogHdr[ LOG_DATABASE_KEY_LEN]),
				NULL, NULL, FALSE)))
			{
				goto Exit;
			}
		}
		pFile->bHaveEncKey = TRUE;
	}
#endif

	if (RC_OK( gv_FlmSysData.pFileSystem->doesFileExist( pszFilePath)))
	{
		rc = RC_SET( FERR_FILE_EXISTS);
		goto Exit;
	}

	// Allocate the super file object

	flmAssert( !pDb->pSFileHdl);
	flmAssert( pFile->FileHdr.uiVersionNum);

	if( pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{
		uiMaxFileSize = gv_FlmSysData.uiMaxFileSize;
	}
	else
	{
		uiMaxFileSize = MAX_FILE_SIZE_VER40;
	}
	
	if( (pDb->pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( (pSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pSFileClient->setup( 
		pFile->pszDbPath, pFile->pszDataDir, pFile->FileHdr.uiVersionNum)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDb->pSFileHdl->setup( pSFileClient, 
		gv_FlmSysData.pFileHdlCache, gv_FlmSysData.uiFileOpenFlags,
		gv_FlmSysData.uiFileCreateFlags)))
	{
		goto Exit;
	}

	// Create the .db file.

	if( RC_BAD( rc = pDb->pSFileHdl->createFile( 0)))
	{
		goto Exit;
	}
	bFileCreated = TRUE;

	(void)flmStatGetDb( &pDb->Stats, pFile,
						0, &pDb->pDbStats, NULL, NULL);

	// We must have exclusive access.  Create a lock file for that
	// purpose, if there is not already a lock file.

	if( RC_BAD( rc = flmGetExclAccess( pszFilePath, pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmInitNewFile( pDb, pszRflDir, pszDictFileName,
											 pszDictBuf, pCreateOpts, uiTransID)))
	{
		goto Exit;
	}

	// Set FFILE stuff to same state as a completed checkpoint.

	pFile->uiFirstLogCPBlkAddress = 0;
	pFile->uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();

	// Create a checkpoint thread

	if( RC_BAD( rc = flmStartCPThread( pFile)))
	{
		goto Exit;
	}

	// Start the database monitor thread
	
	if( RC_BAD( rc = flmStartDbMonitorThread( pFile)))
	{
		goto Exit;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	rc = flmCompleteOpenOrCreate( &pDb, rc, bNewFile, pDb ? TRUE : FALSE);

	if( RC_BAD( rc))
	{
		if( bFileCreated)
		{
			(void)gv_FlmSysData.pFileSystem->deleteFile( pszFilePath);
		}
	}
	else if( ppDb)
	{
		*ppDb = pDb;
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Create a database - initialize all physical areas & data dictionary.
****************************************************************************/
FSTATIC RCODE flmInitNewFile(
	FDB *					pDb,
	const char *		pszRflDir,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	FLMUINT				uiTransID)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pBuf = NULL;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiBlkSize;
	FLMUINT			uiBufSize;
	FLMUINT			bTransStarted = FALSE;

	// Determine what size of buffer to allocate.

	if (pCreateOpts != NULL)
	{
		uiBlkSize = flmAdjustBlkSize( pCreateOpts->uiBlockSize);
	}
	else
	{
		uiBlkSize = DEFAULT_BLKSIZ;
	}

	// Initialize the database file header.

	uiBufSize = (FLMUINT)((uiBlkSize < 2048)
								 ? (FLMUINT)2048
								 : (FLMUINT)uiBlkSize);

	if( RC_BAD( rc = f_allocAlignedBuffer( uiBufSize, 
		(void **)&pBuf)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmInitFileHdrs( pDb, pCreateOpts, uiBlkSize, 
		uiTransID, pBuf)))
	{
		goto Exit;
	}

	// Allocate the pRfl object.  Could not do this until this point
	// because we need to have the version number, block size, etc.
	// setup in the pFile->FileHdr.

	flmAssert( !pFile->pRfl);
	if ((pFile->pRfl = f_new F_Rfl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pFile->pRfl->setup( pFile, pszRflDir)))
	{
		goto Exit;
	}

	// The following code starts an update transaction on the new DB so
	// we can get it built.

	if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
	{
		goto Exit;
	}
	bTransStarted = TRUE;

	if( RC_BAD( rc = fdictCreate( pDb, pszDictFileName, pszDictBuf)))
	{
		goto Exit;
	}

	// Because the checkpoint thread has not yet been created,
	// flmCommitDbTrans will force a checkpoint when it completes,
	// ensuring a consistent database state.

	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
	{
		goto Exit;
	}
	bTransStarted = FALSE;

Exit:

	// Free the temporary buffer, if one was allocated.

	if (pBuf)
	{
		f_freeAlignedBuffer( (void **)&pBuf);
	}

	if (bTransStarted)
	{
		(void)flmAbortDbTrans( pDb);
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine initializes a new database file.  The first block
		in the file is strictly for prefix and other fixed information
		about the file.  After the prefix, we initialize the fixed log
		segment.  Finally, we initialize the first database block --
		which contains the physical file header record.
*****************************************************************************/
FSTATIC RCODE flmInitFileHdrs(
	FDB *		  		pDb,
	CREATE_OPTS *	pCreateOpts,
	FLMUINT			uiBlkSize,
	FLMUINT			uiTransID,
	FLMBYTE *		pucInitBuf)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMBYTE *		pucLastCommittedLogHdr;
	FLMUINT			uiLogicalEOF;
	FLMUINT			uiMinRflFileSize;
	FLMUINT			uiMaxRflFileSize;
	FLMBYTE *		pucBuf = NULL;

	// Initialize the FFILE structure and first 2048 bytes/blk of the file.

	f_memset( pucInitBuf, 0, uiBlkSize);
	flmInitFileHdrInfo( pCreateOpts, &pFile->FileHdr,
						 &pucInitBuf [FLAIM_HEADER_START]);
	if (pCreateOpts)
	{
		flmSetFilePrefix( pucInitBuf, pCreateOpts->uiAppMajorVer,
								pCreateOpts->uiAppMinorVer);
	}
	else
	{
		flmSetFilePrefix( pucInitBuf, 0, 0);
	}

	if (RC_BAD( rc = pDb->pSFileHdl->writeBlock( 0L, uiBlkSize, 
		pucInitBuf, NULL)))
	{
		goto Exit;
	}

	// Set the logical EOF.
	// Reserve two blocks for pre-4.3 - one for LFH, one for PCODE
	// Reserve only room for LFH in 4.3 and above.

	uiLogicalEOF = (FLMUINT)((pFile->FileHdr.uiVersionNum >=
									  FLM_FILE_FORMAT_VER_4_3)
									 ? (FLMUINT)pFile->FileHdr.uiFirstLFHBlkAddr +
												uiBlkSize
									 : (FLMUINT)pFile->FileHdr.uiFirstLFHBlkAddr +
												uiBlkSize * 2);

	// Initialize and output the log header.

	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	f_memset( pucLastCommittedLogHdr, 0, LOG_HEADER_SIZE);

	UD2FBA( (FLMUINT32)uiTransID,
			  &pucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);

	UD2FBA( (FLMUINT32)1, &pucLastCommittedLogHdr [LOG_RFL_FILE_NUM]);

	// Putting a zero in this value tells the RFL code that the
	// RFL file should be created - overwriting it if it already
	// exists.

	UD2FBA( (FLMUINT32)0, &pucLastCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);
	UD2FBA( (FLMUINT32)1, &pucLastCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]);
	UD2FBA( (FLMUINT32)512, &pucLastCommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]);
	UD2FBA( (FLMUINT32)0, &pucLastCommittedLogHdr [LOG_LAST_RFL_FILE_DELETED]);
	pucLastCommittedLogHdr [LOG_KEEP_RFL_FILES] =
			(FLMBYTE)((pCreateOpts && pCreateOpts->bKeepRflFiles)
						 ? (FLMBYTE)1
						 : (FLMBYTE)0);
	pucLastCommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL] = 0;
	pucLastCommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL] =
			(FLMBYTE)((pCreateOpts && pCreateOpts->bLogAbortedTransToRfl)
						 ? (FLMBYTE)1
						 : (FLMBYTE)0);
	UD2FBA( ((FLMUINT32)uiBlkSize),
				&pucLastCommittedLogHdr [LOG_ROLLBACK_EOF]);
	UD2FBA( (FLMUINT32)0,
				&pucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]);
	UW2FBA( (FLMUINT16) pDb->pFile->FileHdr.uiVersionNum, 
				&pucLastCommittedLogHdr [LOG_FLAIM_VERSION]);

	uiMinRflFileSize = (FLMUINT)((pCreateOpts &&
											pCreateOpts->uiMinRflFileSize)
										  ? pCreateOpts->uiMinRflFileSize
										  : (FLMUINT)DEFAULT_MIN_RFL_FILE_SIZE);

	if( pDb->pFile->FileHdr.uiVersionNum >= FLM_FILE_FORMAT_VER_4_3)
	{
		FLMUINT16	ui16Tmp;

		uiMaxRflFileSize = (FLMUINT)((pCreateOpts &&
												pCreateOpts->uiMaxRflFileSize)
											  ? pCreateOpts->uiMaxRflFileSize
											  : (FLMUINT)DEFAULT_MAX_RFL_FILE_SIZE);

		// Make sure the RFL size limits are valid.
		// Maximum must be enough to hold at least one packet plus
		// the RFL header.  Minimum must not be greater than the
		// maximum.  NOTE: Minimum and maximum are allowed to be
		// equal, but in all cases, maximum takes precedence over
		// minimum.  We will first NOT exceed the maximum.  Then,
		// if possible, we will go above the minimum.

		if (uiMaxRflFileSize < RFL_MAX_PACKET_SIZE + 512)
		{
			uiMaxRflFileSize = RFL_MAX_PACKET_SIZE + 512;
		}
		if (uiMaxRflFileSize > gv_FlmSysData.uiMaxFileSize)
		{
			uiMaxRflFileSize = gv_FlmSysData.uiMaxFileSize;
		}
		if (uiMinRflFileSize > uiMaxRflFileSize)
		{
			uiMinRflFileSize = uiMaxRflFileSize;
		}
		UD2FBA( (FLMUINT32)uiMinRflFileSize,
				&pucLastCommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
		UD2FBA( (FLMUINT32)uiMaxRflFileSize,
				&pucLastCommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);

		// Set the database serial number

		f_createSerialNumber(
				&pucLastCommittedLogHdr[ LOG_DB_SERIAL_NUM]);

		// Set the "current" RFL serial number - will be stamped into the RFL
		// file when it is first created.

		f_createSerialNumber(
				&pucLastCommittedLogHdr[ LOG_LAST_TRANS_RFL_SERIAL_NUM]);

		// Set the "next" RFL serial number

		f_createSerialNumber(
				&pucLastCommittedLogHdr[ LOG_RFL_NEXT_SERIAL_NUM]);

		// Set the incremental backup serial number and sequence number

		f_createSerialNumber(
				&pucLastCommittedLogHdr[ LOG_INC_BACKUP_SERIAL_NUM]);

		UD2FBA( 1, &pucLastCommittedLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);

		// Set the file size limits

		pFile->uiMaxFileSize = gv_FlmSysData.uiMaxFileSize;
		ui16Tmp = (FLMUINT16)(gv_FlmSysData.uiMaxFileSize >> 16);
		UW2FBA( ui16Tmp, &pucLastCommittedLogHdr [LOG_MAX_FILE_SIZE]);
	}
	else
	{
		UD2FBA( (FLMUINT32)uiMinRflFileSize,
				&pucLastCommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
		pFile->uiMaxFileSize = MAX_FILE_SIZE_VER40;
	}

	// The defines better not have changed to be less than two blocks - in
	// case we create more than one block below.

	flmAssert( pFile->uiMaxFileSize >=
					pFile->FileHdr.uiBlockSize * 2);

	// Create the first block file.

	if (RC_BAD( rc = pDb->pSFileHdl->createFile( 1)))
	{
		goto Exit;
	}

	// The following 0xFFFF initializations are done to make this four
	// bytes compatible with how it used to be initialized.  These bytes
	// will ALWAYS be set to something else when the log header is
	// written out.

	UW2FBA( (FLMUINT16)0xFFFF,
			&pucLastCommittedLogHdr [LOG_HDR_CHECKSUM]);
	UD2FBA( (FLMUINT32)BT_END,
			&pucLastCommittedLogHdr [LOG_PF_FIRST_BACKCHAIN]);
	UD2FBA( (FLMUINT32)BT_END,
			&pucLastCommittedLogHdr [LOG_PF_AVAIL_BLKS]);
	UD2FBA( (FLMUINT32)uiLogicalEOF,
			&pucLastCommittedLogHdr [LOG_LOGICAL_EOF]);

	// Write out the database wrapping key
	if (pDb->pFile->pDbWrappingKey)
	{
		FLMUINT32	ui32KeyLen = 0;

		if (RC_BAD( rc = pDb->pFile->pDbWrappingKey->getKeyToStore(
						&pucBuf, &ui32KeyLen, NULL, NULL, FALSE)))
		{
			goto Exit;
		}
		
		// Verify that the field in the log header is long enough to
		// hold the key.
		
		if( ui32KeyLen > FLM_MAX_DB_ENC_KEY_LEN)
		{
			f_free( &pucBuf);
			rc = RC_SET_AND_ASSERT( FERR_BAD_ENC_KEY);
			goto Exit;
		}

		UW2FBA(ui32KeyLen, &pucLastCommittedLogHdr[ LOG_DATABASE_KEY_LEN]);
		
		f_memcpy( &pucLastCommittedLogHdr[LOG_DATABASE_KEY], pucBuf,
					 ui32KeyLen);
		f_free( &pucBuf);
	}

	if (RC_BAD( rc = flmWriteLogHdr( pDb->pDbStats, pDb->pSFileHdl,
							pFile, pucLastCommittedLogHdr, NULL, TRUE)))
	{
		goto Exit;
	}

	// Copy the log header to the ucCheckpointLogHdr buffer.
	// This is now the first official checkpoint version of the log
	// header.  It must be copied to the ucCheckpointLogHdr buffer so that
	// it will not be lost in subsequent calls to flmWriteLogHdr.

	f_memcpy( pFile->ucCheckpointLogHdr, pucLastCommittedLogHdr,
					LOG_HEADER_SIZE);

	// Initialize and output the first LFH block

	f_memset( pucInitBuf, 0, uiBlkSize); 
	SET_BH_ADDR( pucInitBuf, (FLMUINT32)pFile->FileHdr.uiFirstLFHBlkAddr);
	pucInitBuf [BH_TYPE] = BHT_LFH_BLK;
	UD2FBA( (FLMUINT32)BT_END,  &pucInitBuf [BH_PREV_BLK]);
	UD2FBA( (FLMUINT32)BT_END,  &pucInitBuf [BH_NEXT_BLK]);
	UW2FBA( (FLMUINT16)BH_OVHD, &pucInitBuf [BH_ELM_END]);
	UD2FBA( (FLMUINT32)uiTransID, &pucInitBuf [BH_TRANS_ID]);

	BlkCheckSum( pucInitBuf, CHECKSUM_SET,
						pFile->FileHdr.uiFirstLFHBlkAddr,
							uiBlkSize);
	pDb->pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
	pDb->pSFileHdl->setExtendSize( pFile->uiFileExtendSize);
	if( RC_BAD( rc = pDb->pSFileHdl->writeBlock(
				pFile->FileHdr.uiFirstLFHBlkAddr, uiBlkSize, pucInitBuf, NULL)))
	{
		goto Exit;
	}

	// Initialize and output the first pcode block.

	if (pFile->FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{
		FLMUINT	uiPcodeAddr;

		f_memset( pucInitBuf, 0, uiBlkSize);
		uiPcodeAddr = pFile->FileHdr.uiFirstLFHBlkAddr + uiBlkSize;
		SET_BH_ADDR( pucInitBuf, (FLMUINT32)uiPcodeAddr);
		pucInitBuf [BH_TYPE] = BHT_PCODE_BLK;
		UD2FBA( (FLMUINT32)BT_END,  &pucInitBuf [BH_PREV_BLK]);
		UD2FBA( (FLMUINT32)BT_END,  &pucInitBuf [BH_NEXT_BLK]);
		UW2FBA( (FLMUINT16)BH_OVHD, &pucInitBuf [BH_ELM_END]);
		UD2FBA( (FLMUINT32)uiTransID, &pucInitBuf [BH_TRANS_ID]);

		BlkCheckSum( pucInitBuf, CHECKSUM_SET, uiPcodeAddr, uiBlkSize);
		pDb->pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
		pDb->pSFileHdl->setExtendSize( pFile->uiFileExtendSize);
		if (RC_BAD( rc = pDb->pSFileHdl->writeBlock( 
			uiPcodeAddr, uiBlkSize, pucInitBuf, NULL)))
		{
			goto Exit;
		}
	}

	// Force things to disk.

	if (RC_BAD( rc = pDb->pSFileHdl->flush()))
	{
		goto Exit;
	}
Exit:
	if (pucBuf)
	{
		f_free( &pucBuf);
	}
	return( rc);
}
