//------------------------------------------------------------------------------
// Desc:	Check a database for corruptions
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
Desc:	Constructor
****************************************************************************/
F_DbCheck::~F_DbCheck()
{
	if (m_pXRefRS)
	{
		m_pXRefRS->Release();
		m_pXRefRS = NULL;
	}
	
	// Cleanup any temporary index check files

	if (m_pIxRSet)
	{
		m_pIxRSet->Release();
	}
	f_free( &m_puiIxArray);

	if (m_pDb)
	{
		m_pDb->Release();
	}

	if (m_pDbInfo)
	{
		m_pDbInfo->Release();
	}
	
	(void)closeAndDeleteResultSetDb();

	if (m_pRandGen)
	{
		m_pRandGen->Release();
	}
	
	if (m_pBtPool)
	{
		m_pBtPool->Release();
	}
	if (m_pBlkEntries)
	{
		f_free( &m_pBlkEntries);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_DbCheck::createAndOpenResultSetDb( void)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_CREATE_OPTS	createOpts;

	if (m_pResultSetDb)
	{
		if (RC_BAD( rc = closeAndDeleteResultSetDb()))
		{
			goto Exit;
		}
	}

	f_memset( &createOpts, 0, sizeof( XFLM_CREATE_OPTS));
	for (;;)
	{

		// Generate a random file name
		
		f_sprintf( m_szResultSetDibName,
					  "%d.db", (int)m_pRandGen->getUINT32( 100, 20000));
		
		if (RC_OK( rc = gv_pXFlmDbSystem->dbCreate( 
				m_szResultSetDibName, NULL, NULL, NULL, NULL, 
				&createOpts, TRUE, (IF_Db **)&m_pResultSetDb)))
		{
			break;
		}
		if (rc == NE_XFLM_FILE_EXISTS || rc == NE_FLM_IO_ACCESS_DENIED)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// Shouldn't have an RFL object - don't want anything
	// logged by this database.
	
	flmAssert( !m_pResultSetDb->getDatabase()->m_pRfl);

Exit:

	return rc;
}

/****************************************************************************
Desc:	Close the database file and delete it.
****************************************************************************/
RCODE F_DbCheck::closeAndDeleteResultSetDb( void)
{
	RCODE				rc = NE_XFLM_OK;
	
	if (m_pResultSetDb)
	{
		if (m_pResultSetDb->getTransType() != XFLM_NO_TRANS)
		{
			m_pResultSetDb->transAbort();
		}
		m_pResultSetDb->Release();
		m_pResultSetDb = NULL;
	}

	if (RC_BAD( rc = gv_pXFlmDbSystem->dbRemove( 
		m_szResultSetDibName, NULL, NULL, TRUE)))
	{
		goto Exit;
	}

	f_memset( m_szResultSetDibName, 0, sizeof( m_szResultSetDibName));

Exit:

	return rc;
}

/****************************************************************************
Desc:	Method to get a new F_BtResultSet object.  This method will create a
		new collection for the result set to use before handing it off.
****************************************************************************/
RCODE F_DbCheck::getBtResultSet(
	F_BtResultSet **	ppBtRSet
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCollection;
	F_BtResultSet *	pBtRSet = NULL;
	F_Database *		pDatabase = NULL;

	// If there is already a BtResultSet, let's get rid of it.
	
	if (*ppBtRSet)
	{
		(*ppBtRSet)->Release();
		*ppBtRSet = NULL;
	}

	// Create the new BtResultSet object first.
	
	if ((pBtRSet = f_new F_BtResultSet( m_pResultSetDb, m_pBtPool)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pDatabase = m_pResultSetDb->m_pDatabase;

	for (;;)
	{

		// Now create a new collection.  Randomly select a collection number to use.
		
		uiCollection = m_pRandGen->getUINT32( 100, XFLM_MAX_COLLECTION_NUM);
	
		// Check to see if it already exists.
		
		if (RC_OK( rc = pDatabase->lFileCreate( m_pResultSetDb,
			&pBtRSet->m_Collection.lfInfo, &pBtRSet->m_Collection, uiCollection,
			XFLM_LF_COLLECTION, FALSE, TRUE)))
		{
			break;
		}
		
		if (rc != NE_XFLM_EXISTS)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
	}

	*ppBtRSet = pBtRSet;
	pBtRSet = NULL;

Exit:

	if (pBtRSet)
	{
		pBtRSet->Release();
	}
	
	return rc;
}

/****************************************************************************
Desc : Checks for physical corruption in a FLAIM database.
DNote: The routine verifies the database by first reading through
		 the database to count certain block types which are in linked lists.
		 It then verifies the linked lists.  It also verifies the B-TREEs
		 in the database.  The reason for the first pass is so that when we
		 verify the linked lists, we can keep ourselves from getting into
		 an infinite loop if there is a loop in the lists.
****************************************************************************/
RCODE F_DbCheck::dbCheck(
	const char *		pszDbFileName,
		// [IN] Full path and file name of the database which
		// is to be checked.  NULL can be passed as the value of
		// this parameter if pDb is non-NULL.
	const char *		pszDataDir,
		// [IN] Directory for data files.
	const char *		pszRflDir,
		// [IN] RFL directory.  NULL can be passed as the value of
		// this parameter to indicate that the log files are located
		// in the same directory as the database or if pDb is non-NULL.
	const char *		pszPassword,
		// [IN] Database password. Needed to open the database if the database
		// key has been wrapped in a password. NULL by default.
	FLMUINT				uiFlags,
		// [IN] Check flags.  Possible flags include:
		//
		//		XFLM_ONLINE. This flag instructs the check to repair any
		//		index corruptions it finds.  The database must have been
		//		opened in read/write mode in order for the check to
		//		successfully repair corruptions.  An update transaction
		//		will be started whenever a corruption is repaired.
		//
		//		XFLM_DO_LOGICAL_CHECK.  This flag instructs the check to
		//		perform a logical check of the databases's indexes
		//		in addition to the structural check.
		//
		//		XFLM_SKIP_DOM_LINK_CHECK.  This flag instructs the check to skip
		//		verifying the DOM links.  This check can take quite a long time
		//		to execute.
		//
		//		XFLM_ALLOW_LIMITED_MODE. This flag instructs the check to allow
		//		the database to be opened in limited mode if the database key is
		//		wrapped in a password and the password we pass is	incorrect 
		//		(or non-existent).
	IF_DbInfo **			ppDbInfo,
		// [IN] Pointer to a DB_INFO structure which is used to store
		// statistics collected during the database check.
	IF_DbCheckStatus *	pDbCheckStatus
		// [IN] Status interface.  Functions in this interface are called 
		// periodically to iform the calling application of the progress
		// being made.  This allows the application to monitor and/or display
		// the progress of the database check.  NULL may be passed as the
		// value of this parameter if the callback feature is not needed.
	)
{
	RCODE								rc = NE_XFLM_OK;
	FLMBYTE *						pBlk = NULL;
	FLMUINT							uiFileEnd;
	FLMUINT							uiBlockSize;
	FLMUINT							uiLoop;
	FLMUINT64						ui64TmpSize;
	FLMBOOL							bStartOver;
	FLMBOOL							bOkToCloseTrans = FALSE;
	FLMBOOL							bAllowLimitedMode =  ( uiFlags & XFLM_ALLOW_LIMITED_MODE)
																		? TRUE
																		: FALSE;

	if (RC_BAD( rc = gv_pXFlmDbSystem->dbOpen( pszDbFileName, pszDataDir,
		pszRflDir, pszPassword, bAllowLimitedMode, (IF_Db **)&m_pDb)))
	{
		goto Exit;
	}

	if ((m_pDbInfo = f_new F_DbInfo) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (ppDbInfo)
	{
		*ppDbInfo = m_pDbInfo;
		(*ppDbInfo)->AddRef();
	}

	m_pDbCheckStatus = pDbCheckStatus;
	m_LastStatusRc = NE_XFLM_OK;

	// Get the file size...

	if (uiFlags & XFLM_SKIP_DOM_LINK_CHECK)
	{
		m_bSkipDOMLinkCheck = TRUE;
	}

	// Initialize the information block and Progress structure.

	// Since we know that the check will start read transactions
	// during its processing, set the flag to indicate that the KRef table
	// should be cleaned up on exit if we are still in a read transaction.

	bOkToCloseTrans = TRUE;
	uiBlockSize = m_pDb->m_pDatabase->getBlockSize();

	// Allocate memory to use for reading through the data blocks.

	if( RC_BAD( rc = f_alloc( uiBlockSize, &pBlk)))
	{
		goto Exit;
	}

	if ((m_pBtPool = f_new F_BtPool) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = m_pBtPool->btpInit()))
	{
		goto Exit;
	}
	
	// Setup the result set database.
	
	if( RC_BAD( rc = FlmAllocRandomGenerator( &m_pRandGen)))
	{
		goto Exit;
	}

	m_pRandGen->setSeed( 9768);

	if (RC_BAD( rc = createAndOpenResultSetDb()))
	{
		goto Exit;
	}

Begin_Check:

	// Initialize all statistics in the DB_INFO structure.

	rc = NE_XFLM_OK;
	bStartOver = FALSE;

	m_pDbInfo->m_ui64FileSize = 0;
	m_pDbInfo->freeLogicalFiles();
	m_bPhysicalCorrupt = FALSE;
	m_bIndexCorrupt = FALSE;
	m_uiFlags = uiFlags;
	m_bStartedUpdateTrans = FALSE;
	f_memset( &m_pDbInfo->m_AvailBlocks, 0, sizeof( BLOCK_INFO));
	f_memset( &m_pDbInfo->m_LFHBlocks, 0, sizeof( BLOCK_INFO));

	f_memset( &m_Progress, 0, sizeof( XFLM_PROGRESS_CHECK_INFO));

	/* Get the dictionary information for the file. */

	if (RC_BAD( rc = getDictInfo()))
	{
		goto Exit;
	}

	m_Progress.ui64BytesExamined = 0;

	for (uiLoop = 1;
		  uiLoop <= MAX_DATA_BLOCK_FILE_NUMBER;
		  uiLoop++)
	{
		if (RC_BAD( m_pDb->m_pSFileHdl->getFileSize( uiLoop, &ui64TmpSize)))
		{
			break;
		}
		
		m_Progress.ui64FileSize += ui64TmpSize;
	}

	// See if we have a valid end of file

	uiFileEnd = m_pDb->m_uiLogicalEOF;
	if (FSGetFileOffset( uiFileEnd) % uiBlockSize != 0)
	{
		if (RC_BAD( rc = chkReportError( FLM_BAD_FILE_SIZE, XFLM_LOCALE_NONE,
			0, 0, 0xFF, (FLMUINT32)uiFileEnd, 0, 0, 0)))
		{
			goto Exit;
		}
	}
	else if (m_Progress.ui64FileSize <
					FSGetSizeInBytes( m_pDb->m_pDatabase-> getMaxFileSize(),
																		uiFileEnd))
	{
		m_Progress.ui64FileSize =
					FSGetSizeInBytes( m_pDb->m_pDatabase->getMaxFileSize(),
																uiFileEnd);
	}

	m_pDbInfo->m_ui64FileSize = m_Progress.ui64FileSize;

	// Verify the LFH blocks, B-Trees, and the AVAIL list.

	if( RC_BAD( rc = verifyLFHBlocks( &bStartOver)))
	{
		goto Exit;
	}
	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Check the b-trees.
	
	if (RC_BAD( rc = verifyBTrees( &bStartOver)))
	{
		goto Exit;
	}
	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Check the avail list.

	if (RC_BAD( rc = verifyAvailList( &bStartOver)))
	{
		goto Exit;
	}
	if (bStartOver)
	{
		goto Begin_Check;
	}

Exit:

	if ((m_bPhysicalCorrupt || m_bIndexCorrupt) &&
		 !gv_pXFlmDbSystem->errorIsFileCorrupt( rc))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
	}

	if (RC_OK( rc) && RC_BAD( m_LastStatusRc))
	{
		rc = m_LastStatusRc;
	}

	if (m_pDb)
	{
		// Close down the transaction, if one is going

		if( bOkToCloseTrans &&
			m_pDb->getTransType( ) == XFLM_READ_TRANS)
		{
			m_pDb->krefCntrlFree();
			m_pDb->transAbort();
		}
	}
	
	// Free memory, if allocated

	if (pBlk)
	{
		f_free( &pBlk);
	}

	// Close the FLAIM database we opened.

	if (m_pDb)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine opens a file and reads its dictionary into memory.
*****************************************************************************/
RCODE F_DbCheck::getDictInfo()
{
	RCODE	rc = NE_XFLM_OK;

	// Close down the transaction, if one is going.

	if (m_pDb->getTransType() != XFLM_UPDATE_TRANS)
	{
		if (m_pDb->getTransType() == XFLM_READ_TRANS)
		{
			(void)m_pDb->transAbort();
		}

		// Start a read transaction on the file to ensure we are connected
		// to the file's dictionary structures.

		if (RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS,
			FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE, &m_pDbInfo->m_dbHdr)))
		{
			goto Exit;
		}
	}
	else
	{
		f_memcpy( &m_pDbInfo->m_dbHdr, m_pDb->m_pDatabase->getUncommittedDbHdr(),
					sizeof( XFLM_DB_HDR));
	}

Exit:
	return( rc);
}


/********************************************************************
Desc: This routine follows all of the blocks in a chain, verifying
		that they are properly linked.  It also verifies each block's
		header.
*********************************************************************/
RCODE F_DbCheck::verifyBlkChain(
	BLOCK_INFO *	pBlkInfo,
	FLMUINT			uiLocale,
	FLMUINT			uiFirstBlkAddr,
	FLMUINT			uiBlkType,
	FLMBOOL *		pbStartOverRV
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMINT32				i32VerifyCode = 0;
	F_CachedBlock *	pSCache = NULL;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT				uiPrevBlkAddress;
	FLMUINT				uiBlkCount = 0;
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	FLMUINT64			ui64SaveBytesExamined;
	FLMUINT				uiBlockSize = m_pDb->m_pDatabase->getBlockSize();
	FLMUINT				uiMaxBlocks = (FLMUINT)(FSGetSizeInBytes(
										m_pDb->m_pDatabase->getMaxFileSize(),
											m_pDb->m_uiLogicalEOF) /
											(FLMUINT64)uiBlockSize);

	uiPrevBlkAddress = 0;

	/* There must be at least ONE block if it is the LFH chain. */

	if ((uiBlkType == BT_LFH_BLK) && (uiFirstBlkAddr == 0))
	{
		i32VerifyCode = FLM_BAD_LFH_LIST_PTR;
		(void)chkReportError( i32VerifyCode,
									 (FLMUINT32)uiLocale,
									 0,
									 0,
									 0xFF,
									 0,
									 0,
									 0,
									 0);
		goto Exit;
	}

	/* Read through all of the blocks, verifying them as we go. */

Restart_Chain:
	uiBlkCount = 0;
	flmInitReadState( &StateInfo,
							&bStateInitialized,
							(FLMUINT)m_pDb->m_pDatabase->
											m_lastCommittedDbHdr.ui32DbVersion,
							m_pDb,
							NULL,
							(FLMUINT)((uiBlkType == BT_FREE)
											? (FLMUINT)0xFF
											: (FLMUINT)0),
							uiBlkType,
							NULL);
	
	ui64SaveBytesExamined = m_Progress.ui64BytesExamined;
	StateInfo.ui32BlkAddress = (FLMUINT32)uiFirstBlkAddr;

	while ((StateInfo.ui32BlkAddress != 0) && (uiBlkCount < uiMaxBlocks))
	{
		StateInfo.pBlkHdr = NULL;
		if( RC_BAD( rc = blkRead( StateInfo.ui32BlkAddress, &pBlkHdr,
			&pSCache, &i32VerifyCode)))
		{
			if (rc == NE_XFLM_OLD_VIEW)
			{
				FLMUINT	uiSaveDictSeq = m_pDb->m_pDict->getDictSeq();

				if (RC_BAD( rc = getDictInfo()))
					goto Exit;

				// If the dictionary ID changed, start over.

				if (m_pDb->m_pDict->getDictSeq() != uiSaveDictSeq)
				{
					*pbStartOverRV = TRUE;
					goto Exit;
				}

				m_Progress.ui64BytesExamined = ui64SaveBytesExamined;
				goto Restart_Chain;
			}
			pBlkInfo->i32ErrCode = i32VerifyCode;
			pBlkInfo->uiNumErrors++;
			rc = chkReportError( i32VerifyCode,
										(FLMUINT32)uiLocale,
										0,
										0,
										0xFF,
										StateInfo.ui32BlkAddress,
										0,
										0,
										0);
		}
		StateInfo.pBlkHdr = pBlkHdr;
		uiBlkCount++;
		m_Progress.ui64BytesExamined += (FLMUINT64)uiBlockSize;
		if (RC_BAD( rc = chkCallProgFunc()))
		{
			goto Exit;
		}

		f_yieldCPU();

		if ((i32VerifyCode = flmVerifyBlockHeader( &StateInfo,
															  pBlkInfo,
															  uiBlockSize,
															  0xFFFFFFFF,
															  uiPrevBlkAddress,
															  TRUE)) != 0)
		{
			pBlkInfo->i32ErrCode = i32VerifyCode;
			pBlkInfo->uiNumErrors++;
			chkReportError( i32VerifyCode,
								 (FLMUINT32)uiLocale,
								 0,
								 0,
								 0xFF,
								 StateInfo.ui32BlkAddress,
								 0,
								 0,
								 0);
			goto Exit;
		}
		uiPrevBlkAddress = StateInfo.ui32BlkAddress;
		StateInfo.ui32BlkAddress = pBlkHdr->ui32NextBlkInChain;
	}
	if (StateInfo.ui32BlkAddress != 0 && RC_OK( m_LastStatusRc))
	{
		switch (uiBlkType)
		{
			case BT_LFH_BLK:
				i32VerifyCode = FLM_BAD_LFH_LIST_END;
				break;
			case BT_FREE:
				i32VerifyCode = FLM_BAD_AVAIL_LIST_END;
				break;
		}
		pBlkInfo->i32ErrCode = i32VerifyCode;
		pBlkInfo->uiNumErrors++;
		chkReportError( i32VerifyCode,
							 (FLMUINT32)uiLocale,
							 0,
							 0,
							 0xFF,
							 (FLMUINT32)uiPrevBlkAddress,
							 0,
							 0,
							 0);
		goto Exit;
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if( pBlkHdr)
	{
		f_free( &pBlkHdr);
	}

	if (RC_OK(rc) && (i32VerifyCode != 0))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine verifies the LFH blocks.
*****************************************************************************/
RCODE F_DbCheck::verifyLFHBlocks(
	FLMBOOL *	pbStartOverRV)
{
	RCODE	rc = NE_XFLM_OK;

	m_Progress.ui32LfNumber = 0;
	m_Progress.ui32LfType = 0;
	m_Progress.i32CheckPhase = XFLM_CHECK_LFH_BLOCKS;
	m_Progress.bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc()))
	{
		goto Exit;
	}
	m_Progress.bStartFlag = FALSE;

	f_yieldCPU();

	// Go through the LFH blocks.

	if (RC_BAD( rc = verifyBlkChain( 
			&m_pDbInfo->m_LFHBlocks, XFLM_LOCALE_LFH_LIST,
			(FLMUINT)m_pDb->m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr,
			BT_LFH_BLK, pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine reads through the blocks in the AVAIL list and verifies
		that we don't have a loop or some other corruption in the list.
*****************************************************************************/
RCODE F_DbCheck::verifyAvailList(
	FLMBOOL *	pbStartOverRV)
{
	RCODE		rc = NE_XFLM_OK;

	m_Progress.ui32LfNumber = 0;
	m_Progress.ui32LfType = 0;
	m_Progress.i32CheckPhase = XFLM_CHECK_AVAIL_BLOCKS;
	m_Progress.bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc()))
	{
		goto Exit;
	}
	m_Progress.bStartFlag = FALSE;

	f_yieldCPU();
 
	if (RC_BAD( rc = verifyBlkChain( &m_pDbInfo->m_AvailBlocks,
								XFLM_LOCALE_AVAIL_LIST,
								m_pDb->m_uiFirstAvailBlkAddr,
								BT_FREE, pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns the next B-Tree information that was collected during the
		database check.
****************************************************************************/
void XFLAPI F_DbInfo::getBTreeInfo(
	FLMUINT			uiNthLogicalFile,
	FLMUINT *		puiLfNum,
	eLFileType *	peLfType,
	FLMUINT *		puiRootBlkAddress,
	FLMUINT *		puiNumLevels
	)
{
	LF_HDR *	pLfHdr;

	if (uiNthLogicalFile < m_uiNumLogicalFiles)
	{
		pLfHdr = &m_pLogicalFiles[ uiNthLogicalFile];
		*puiLfNum = pLfHdr->uiLfNum;
		*peLfType = pLfHdr->eLfType;
		*puiRootBlkAddress = pLfHdr->uiRootBlk;
		*puiNumLevels = pLfHdr->uiNumLevels;
	}
	else
	{
		flmAssert( 0);
		*puiLfNum = 0;
		*puiRootBlkAddress = 0;
		*puiNumLevels = 0;
	}
}

/****************************************************************************
Desc:	Returns block information on the specified b-tree and level within
		that b-tree.
****************************************************************************/
void XFLAPI F_DbInfo::getBTreeBlockStats(
	FLMUINT		uiNthLogicalFile,
	FLMUINT		uiLevel,
	FLMUINT64 *	pui64KeyCount,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT64 *	pui64ElementCount,
	FLMUINT64 *	pui64ContElementCount,
	FLMUINT64 *	pui64ContElmBytes,
	FLMUINT *	puiBlockCount,
	FLMINT32 *	pi32LastError,
	FLMUINT *	puiNumErrors
	)
{
	LF_HDR *	pLfHdr;

	if (uiNthLogicalFile < m_uiNumLogicalFiles &&
		 uiLevel < m_pLogicalFiles [uiNthLogicalFile].uiNumLevels)
	{
		pLfHdr = &m_pLogicalFiles[ uiNthLogicalFile];
		*pui64KeyCount = pLfHdr->pLevelInfo [uiLevel].ui64KeyCount;
		*pui64BytesUsed = pLfHdr->pLevelInfo [uiLevel].BlockInfo.ui64BytesUsed;
		*pui64ElementCount = pLfHdr->pLevelInfo [uiLevel].BlockInfo.ui64ElementCount;
		*pui64ContElementCount = pLfHdr->pLevelInfo [uiLevel].BlockInfo.ui64ContElementCount;
		*pui64ContElmBytes = pLfHdr->pLevelInfo [uiLevel].BlockInfo.ui64ContElmBytes;
		*puiBlockCount = pLfHdr->pLevelInfo [uiLevel].BlockInfo.uiBlockCount;
		*pi32LastError = pLfHdr->pLevelInfo [uiLevel].BlockInfo.i32ErrCode;
		*puiNumErrors = pLfHdr->pLevelInfo [uiLevel].BlockInfo.uiNumErrors;
	}
	else
	{
		flmAssert( 0);
		*pui64KeyCount = 0;
		*pui64BytesUsed = 0;
		*pui64ElementCount = 0;
		*pui64ContElementCount = 0;
		*pui64ContElmBytes = 0;
		*puiBlockCount = 0;
		*pi32LastError = 0;
		*puiNumErrors = 0;
	}
}

/****************************************************************************
Desc:	Checks for physical corruption in a FLAIM database.
Note:	The routine verifies the database by first reading through
		the database to count certain block types which are in linked lists.
		It then verifies the linked lists.  It also verifies the B-TREEs
		in the database.  The reason for the first pass is so that when we
		verify the linked lists, we can keep ourselves from getting into
		an infinite loop if there is a loop in the lists.
****************************************************************************/
RCODE XFLAPI F_DbSystem::dbCheck(
	const char *			pszDbFileName,
		// [IN] Full path and file name of the database which
		// is to be checked.  NULL can be passed as the value of
		// this parameter if pDb is non-NULL.
	const char *			pszDataDir,
		// [IN] Directory for data files.
	const char *			pszRflDir,
		// [IN] RFL directory.  NULL can be passed as the value of
		// this parameter to indicate that the log files are located
		// in the same directory as the database or if pDb is non-NULL.
	const char *			pszPassword,
		// [IN] Database password. This is necessary to open the database if
		// the key has been wrapped in a password. NULL by default.
	FLMUINT					uiFlags,
		// [IN] Check flags.  Possible flags include:
		//
		//		XFLM_ONLINE. This flag instructs the check to repair any
		//		index corruptions it finds.  The database must have been
		//		opened in read/write mode in order for the check to
		//		successfully repair corruptions.  An update transaction
		//		will be started whenever a corruption is repaired.
		//
		//		XFLM_DO_LOGICAL_CHECK.  This flag instructs the check to
		//		perform a logical check of the databases's indexes
		//		in addition to the structural check.
		//
		//		XFLM_SKIP_DOM_LINK_CHECK.  This flag instructs the check to skip
		//		verifying the DOM links.  This check can take quite a long time
		//		to execute.
		//
		//		XFLM_ALLOW_LIMITED_MODE. This flag instructs the check to allow
		//		the database to be opened in limited mode if the database key is
		//		wrapped in a password and the password we pass is	incorrect 
		//		(or non-existent).
	IF_DbInfo **			ppDbInfo,
		// [IN] Pointer to a DB_INFO structure which is used to store
		// statistics collected during the database check.
	IF_DbCheckStatus *	pDbCheckStatus
		// [IN] Status interface.  Functions in this interface are called 
		// periodically to iform the calling application of the progress
		// being made.  This allows the application to monitor and/or display
		// the progress of the database check.  NULL may be passed as the
		// value of this parameter if the callback feature is not needed.
	)
{
	RCODE			rc = NE_XFLM_OK;
	F_DbCheck *	pCheckObj = NULL;

	if ((pCheckObj = f_new F_DbCheck) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	rc = pCheckObj->dbCheck( pszDbFileName, pszDataDir, pszRflDir, pszPassword,
		uiFlags, ppDbInfo, pDbCheckStatus);

Exit:

	if (pCheckObj)
	{
		pCheckObj->Release();
	}
	return( rc);
}

