//------------------------------------------------------------------------------
// Desc:	Reduce the database size by move 'N' free blocks the the end of
// 	  	the file and truncating the file.
// Tabs:	3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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
Desc : Reduces the size of a FLAIM database file.
Notes: The size of the database file is reduced by freeing a specified
		 number of blocks from the available (unused) block list.  The blocks
		 are moved to the end of the file and the file is truncated.  If the
		 available block list is empty, FLAIM will attemp to add blocks to
		 the list by freeing log extent blocks.
****************************************************************************/
RCODE F_Db::reduceSize(
	FLMUINT			uiCount,
	FLMUINT *		puiCount)
{
	RCODE					rc = NE_SFLM_OK;
	F_Rfl *				pRfl = m_pDatabase->m_pRfl;
	FLMUINT				uiLogicalEOF;
	FLMUINT				uiBlkAddr;
	FLMUINT				uiNumBlksMoved;
	FLMUINT				uiBlkSize;
	F_LARGEST_BLK_HDR	blkHdr;
	FLMINT				iType;
	FLMBOOL				bFlagSet;
	FLMBOOL				bTransActive = FALSE;
	FLMBOOL				bLockedDb = FALSE;
	FLMUINT				uiRflToken = 0;

	uiNumBlksMoved = 0;

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Make sure we are NOT in a database transaction.

	if (m_eTransType != SFLM_NO_TRANS)
	{
		rc = RC_SET( NE_SFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// There must NOT be a shared lock on the file.

	if (m_uiFlags & FDB_FILE_LOCK_SHARED)
	{
		rc = RC_SET( NE_SFLM_SHARED_LOCK);
		goto Exit;
	}

	// Must acquire an exclusive file lock first, if it hasn't been
	// acquired.

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if (RC_BAD( rc = m_pDatabase->m_pDatabaseLockObj->lock( m_hWaitSem,
			TRUE, FLM_NO_TIMEOUT, 0, m_pDbStats ? &m_pDbStats->LockStats : NULL)))
		{
			goto Exit;
		}
		
		bLockedDb = TRUE;
		m_uiFlags |= FDB_HAS_FILE_LOCK;
	}
	
	// Disable RFL logging - don't want anything logged during reduce 
	// except for the reduce packet.	
	
	pRfl->disableLogging( &uiRflToken);

	// Keep looping to here until the count is satisfied or there
	// are not any more log extent blocks to turn into avail blks.
	// The loop does a begin transaction - move blocks - set logical
	// EOF and commits the transaction.  During the commit if there are
	// not any avail blocks left then a log extent (if any) will be turned
	// into more avail blocks and we can do this again with more avail
	// blocks.

	// Start a database transaction

	if( RC_BAD(rc = beginTrans( SFLM_UPDATE_TRANS, 
		FLM_NO_TIMEOUT, SFLM_DONT_POISON_CACHE)))
	{
		goto Exit;
	}
	bTransActive = TRUE;

	// Make sure that commit does something.

	m_bHadUpdOper = TRUE;

	uiBlkSize = m_pDatabase->m_uiBlockSize;

	// Get the logical end of file and use internally.
	// Loop until there are not any more free blocks left or the
	// input count is matched.  Switch on each block type found
	// while backing up through the file.

	uiLogicalEOF = m_uiLogicalEOF;

	while (m_uiFirstAvailBlkAddr &&
			 (!uiCount || uiNumBlksMoved < uiCount))
	{

		// Read the last block and determine block type.

		if( FSGetFileOffset( uiLogicalEOF) == 0)
		{
			IF_FileHdl *	pFileHdl;
			FLMUINT			uiFileNumber = FSGetFileNumber( uiLogicalEOF) - 1;
			FLMUINT64		ui64FileSize;
			FLMUINT64		ui64Temp;

			if (RC_BAD( rc = m_pSFileHdl->getFileHdl(
										uiFileNumber, TRUE, &pFileHdl)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pFileHdl->size( &ui64FileSize)))
			{
				goto Exit;
			}

			// Adjust to a block bounds.

			ui64Temp = (ui64FileSize / uiBlkSize) * uiBlkSize;
			if( ui64Temp < ui64FileSize)
			{
				ui64FileSize = ui64Temp + uiBlkSize;
			}
			
			uiLogicalEOF = FSBlkAddress( uiFileNumber, (FLMUINT)ui64FileSize);
		}

		uiBlkAddr = uiLogicalEOF - uiBlkSize;
		if (RC_BAD( rc = readBlkHdr( uiBlkAddr,
									(F_BLK_HDR *)&blkHdr, &iType)))
		{
			goto Exit;
		}

		switch (iType)
		{
			case	BT_FREE:
				rc = m_pDatabase->freeAvailBlk( this, uiBlkAddr);
				break;

			case BT_LEAF:
			case BT_NON_LEAF:
			case BT_NON_LEAF_COUNTS:
			case BT_LEAF_DATA:
			case BT_DATA_ONLY:
				rc = m_pDatabase->moveBtreeBlk( this, uiBlkAddr,
								(FLMUINT)blkHdr.all.BTreeBlkHdr.ui16LogicalFile,
								getBlkLfType( &blkHdr.all.BTreeBlkHdr));
				break;

			case BT_LFH_BLK:
				rc = m_pDatabase->moveLFHBlk( this, uiBlkAddr);
				break;
				
			default:
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				break;
		}

		if (RC_BAD(rc))
		{
			goto Exit;
		}

		uiNumBlksMoved++;

		// Adjust the logical EOF to the new value.
		// This is complex when dealing with block files.

		if (FSGetFileOffset( uiLogicalEOF) == 0)
		{
			FLMUINT			uiFileNumber = FSGetFileNumber( uiLogicalEOF);
			FLMUINT64		ui64FileOffset;
			IF_FileHdl *	pFileHdl;

			if (uiFileNumber <= 1)
			{
				break;
			}

			// Leave the current file at zero bytes and move to the
			// previous store file.

			uiFileNumber--;

			// Compute the end of the previous block file.

			if (RC_BAD( rc = m_pSFileHdl->getFileHdl(
										uiFileNumber, TRUE, &pFileHdl)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pFileHdl->size( &ui64FileOffset)))
			{
				goto Exit;
			}

			uiLogicalEOF = FSBlkAddress( uiFileNumber, (FLMUINT)ui64FileOffset);
		}
		uiLogicalEOF -= uiBlkSize;
	}

	// Log the reduce packet to the RFL if we are not in the middle of a
	// restore or recovery.  Will need to re-enable logging temporarily
	// and then turn it back off after logging the packet.

	pRfl->enableLogging( &uiRflToken);

	// Log the reduce.

	if( RC_BAD( rc = pRfl->logReduce( this, uiCount)))
	{
		goto Exit;
	}

	// Turn logging back off.

	pRfl->disableLogging( &uiRflToken);

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	// Commit the transaction.

	if (m_uiFlags & FDB_DO_TRUNCATE)
	{
		bFlagSet = TRUE;
	}
	else
	{
		bFlagSet = FALSE;
		m_uiFlags |= FDB_DO_TRUNCATE;
	}

	bTransActive = FALSE;
	rc = commitTrans( uiLogicalEOF, TRUE);

	if (!bFlagSet)
	{
		m_uiFlags &= (~(FDB_DO_TRUNCATE));
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc) && bTransActive)
	{
		(void)abortTrans();
		uiNumBlksMoved = 0;
	}

	if (puiCount)
	{
		// May be more than the count requested.

		*puiCount = uiNumBlksMoved;
	}

	if (uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if (bLockedDb)
	{
		(void)m_pDatabase->m_pDatabaseLockObj->unlock();
		m_uiFlags &= ~FDB_HAS_FILE_LOCK;
	}

	return( rc);
}

/****************************************************************************
Desc:	Read the block header and return the type of block it is
****************************************************************************/
RCODE F_Db::readBlkHdr(
	FLMUINT		uiBlkAddress,
	F_BLK_HDR *	pBlkHdr,
	FLMINT *		piType
	)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiBytesRead;
	FLMUINT					uiNumLooks;
	IF_FileHdl *			pTmpFileHdl = NULL;
	F_CachedBlock *		pBlkSCache;
	SFLM_LFILE_STATS *	pLFileStats;
	F_TMSTAMP				StartTime;
	FLMUINT64				ui64ElapMilli = 0;

	// First see if the block is in cache.
	// Previous writes may not have been forced out to cache.

	if (RC_BAD( rc = m_pDatabase->getBlock( this, NULL, uiBlkAddress,
								&uiNumLooks, &pBlkSCache)))
	{
		goto Exit;
	}

	if (pBlkSCache)		// If found in cache ...
	{
		f_memcpy( pBlkHdr, pBlkSCache->getBlockPtr(), SIZEOF_LARGEST_BLK_HDR);
		ScaReleaseCache( pBlkSCache, FALSE);
	}
	else
	{
		if (m_pDbStats)
		{
			ui64ElapMilli = 0;
			f_timeGetTimeStamp( &StartTime);
		}

		if (RC_OK( rc = m_pSFileHdl->getFileHdl(
								FSGetFileNumber( uiBlkAddress), TRUE, &pTmpFileHdl)))
		{
			rc = pTmpFileHdl->read( FSGetFileOffset( uiBlkAddress),
					SIZEOF_LARGEST_BLK_HDR, pBlkHdr, &uiBytesRead);
		}

		if (m_pDbStats)
		{
			flmAddElapTime( &StartTime, &ui64ElapMilli);
			if (RC_BAD( rc))
			{
				m_pDbStats->bHaveStats = TRUE;
				m_pDbStats->uiReadErrors++;
			}
		}

		// Convert the block header if necessary.

		if (blkIsNonNativeFormat( pBlkHdr))
		{
			convertBlkHdr( pBlkHdr);
		}

		if (m_pDbStats && RC_OK( rc))
		{
			FLMUINT					uiLFileNum;
			SFLM_BLOCKIO_STATS *	pBlockIOStats;

			uiLFileNum = (FLMUINT)((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile;
			if (!uiLFileNum)
			{
				pLFileStats = NULL;
			}
			else
			{
				if( RC_BAD( flmStatGetLFile( m_pDbStats,
								uiLFileNum,
								getBlkLfType((F_BTREE_BLK_HDR *)pBlkHdr),
								0,
								&pLFileStats, NULL, NULL)))
				{
					pLFileStats = NULL;
				}
			}
			if ((pBlockIOStats = flmGetBlockIOStatPtr( m_pDbStats,
											pLFileStats, (FLMBYTE *)pBlkHdr)) != NULL)
			{
				m_pDbStats->bHaveStats = TRUE;
				if (pLFileStats)
				{
					pLFileStats->bHaveStats = TRUE;
				}
				pBlockIOStats->BlockReads.ui64ElapMilli += ui64ElapMilli;
				pBlockIOStats->BlockReads.ui64Count++;
				pBlockIOStats->BlockReads.ui64TotalBytes += SIZEOF_LARGEST_BLK_HDR;
			}
		}

		if (RC_BAD( rc))
		{
			if (rc != NE_FLM_IO_END_OF_FILE && rc != NE_SFLM_MEM)
			{
				m_pSFileHdl->releaseFiles();
			}
			goto Exit;
		}
	}

	if (piType)
	{
		*piType = (FLMINT)pBlkHdr->ui8BlkType;
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Find where in the b-tree a matching block is located.  Move to
			a free block and change all pointers to the block.
Notes:	Some of this code could be called in movePcodeLFHBlk but we have
			to worry about if the block is a root or right most leaf block.
****************************************************************************/
RCODE F_Database::moveBtreeBlk(
	F_Db *		pDb,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiLfNumber,
	eLFileType	eLfType)
{
	RCODE							rc;
	F_INDEX *					pIndex;
	LFILE *						pLFile;
	F_TABLE *					pTable = NULL;
	F_CachedBlock *			pFreeSCache = NULL;
	FLMBOOL						bReleaseCache = FALSE;
	F_Btree *					pbtree = NULL;
	FLMBOOL						bHaveCounts;
	FLMBOOL						bHaveData;
	IXKeyCompare				compareObject;
	IF_ResultSetCompare *	pCompareObject = NULL;
	
	// Get an F_Btree object

	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
	{
		goto Exit;
	}

	if (eLfType == SFLM_LF_TABLE)
	{
		if ((pTable = pDb->m_pDict->getTable( uiLfNumber)) == NULL)
		{
			rc = RC_SET( NE_SFLM_INVALID_TABLE_NUM);
			goto Exit;
		}
		pLFile = &pTable->lfInfo;
		bHaveCounts = FALSE;
		bHaveData = TRUE;
	}
	else
	{
		if ((pIndex = pDb->m_pDict->getIndex( uiLfNumber)) == NULL)
		{
			rc = RC_SET( NE_SFLM_INVALID_INDEX_NUM);
			goto Exit;
		}
		pLFile = &pIndex->lfInfo;
		bHaveCounts = (pIndex->uiFlags & IXD_ABS_POS) ? TRUE : FALSE;
		bHaveData = (pIndex->uiNumDataComponents) ? TRUE : FALSE;

		pCompareObject = &compareObject;
		compareObject.setIxInfo( pDb, pIndex);
	}

	// Need to make sure that LFILE is up to date.
	// Force reading it in.

	if (RC_BAD( rc = lFileRead( pDb, pLFile)))
	{
		goto Exit;
	}

	// If we are moving the root block, create a new dictionary so
	// that the LFILE can be modified safely to point to the new
	// root block.

	if (pLFile->uiRootBlk == uiBlkAddr)
	{

		// Create a new dictionary

		if (!(pDb->m_uiFlags & FDB_UPDATED_DICTIONARY))
		{
			if (RC_BAD( rc = pDb->dictClone()))
			{
				goto Exit;
			}

			// Re-get the LFile

			if (eLfType == SFLM_LF_TABLE)
			{
				pTable = pDb->m_pDict->getTable( uiLfNumber);
				flmAssert( pTable);
				pLFile = &pTable->lfInfo;
			}
			else
			{
				pIndex = pDb->m_pDict->getIndex( uiLfNumber);
				flmAssert( pIndex);
				pLFile = &pIndex->lfInfo;
				pCompareObject = &compareObject;
				compareObject.setIxInfo( pDb, pIndex);
			}
		}
	}

	if (RC_BAD( rc = pbtree->btOpen( pDb, pLFile, bHaveCounts, bHaveData,
												pCompareObject)))
	{
		goto Exit;
	}

	// Get the next free block.

	if (RC_BAD( rc = blockUseNextAvail( pDb, &pFreeSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	// Move B-tree block to free block.

	if (RC_BAD( rc = pbtree->btMoveBlock( (FLMUINT32)uiBlkAddr,
								(FLMUINT32)pFreeSCache->m_uiBlkAddress)))
	{
		goto Exit;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pFreeSCache, FALSE);
	}

	if (pbtree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:	Find where anb LFH input block is located.  Move to
		a free block and change all pointers to the block.
****************************************************************************/
RCODE F_Database::moveLFHBlk(
	F_Db *		pDb,
	FLMUINT		uiBlkAddr)
{
	RCODE					rc;
	F_CachedBlock *	pSCache;
	FLMBOOL				bReleaseCache = FALSE;
	F_CachedBlock *	pFreeSCache = NULL;
	FLMBOOL				bReleaseCache2 = FALSE;
	F_BLK_HDR *			pBlkHdr;
	F_LF_HDR *			pLfHdr;
	F_BLK_HDR *			pFreeBlkHdr;
	FLMUINT				uiLeftBlkAddr;
	FLMUINT				uiRightBlkAddr;
	FLMUINT				uiFreeBlkAddr;
	FLMUINT				uiSavePriorBlkImgAddr;
	FLMUINT				uiPos;
	FLMUINT				uiEndPos;
	F_TABLE *			pTmpTable;
	F_INDEX *			pTmpIndex;
	LFILE *				pTmpLFile;

	if (RC_BAD( rc = getBlock( pDb, NULL, uiBlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}
	pBlkHdr = pSCache->m_pBlkHdr;

	// Get left and rigth block addresses.
	// Get next avail block and move data over

	uiLeftBlkAddr  = (FLMUINT)pBlkHdr->ui32PrevBlkInChain;
	uiRightBlkAddr = (FLMUINT)pBlkHdr->ui32NextBlkInChain;

	if (RC_BAD( rc = blockUseNextAvail( pDb, &pFreeSCache)))
	{
		goto Exit;
	}
	bReleaseCache2 = TRUE;

	pFreeBlkHdr = pFreeSCache->m_pBlkHdr;
	uiFreeBlkAddr = (FLMUINT)pFreeBlkHdr->ui32BlkAddr;

	// The free block has been logged and set to dirty in
	// blockUseNextAvail().
	// BUT, need to preserve prior image block address - it should
	// NOT be copied over from the block we are switching with.

	uiSavePriorBlkImgAddr = (FLMUINT)pFreeBlkHdr->ui32PriorBlkImgAddr;

	f_memcpy( pFreeBlkHdr, pBlkHdr, m_uiBlockSize);
	pFreeBlkHdr->ui32BlkAddr = (FLMUINT32)uiFreeBlkAddr;

	// Restore the saved previous transaction ID and block address.

	pFreeBlkHdr->ui32PriorBlkImgAddr = (FLMUINT32)uiSavePriorBlkImgAddr;

	// Fix up any LFile entries that were pointing to the
	// original LFH block

	uiPos = SIZEOF_STD_BLK_HDR;
	uiEndPos = blkGetEnd( m_uiBlockSize, SIZEOF_STD_BLK_HDR, pFreeBlkHdr);

	// Create a new dictionary

	if (!(pDb->m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if (RC_BAD( rc = pDb->dictClone()))
		{
			goto Exit;
		}
	}

	// Iterate over the set of LFiles in the block and
	// update their LFH block addresses

	pLfHdr = (F_LF_HDR *)((FLMBYTE *)pFreeBlkHdr + SIZEOF_STD_BLK_HDR);
	while (uiPos < uiEndPos)
	{
		if (pLfHdr->ui32LfType != (FLMUINT32)SFLM_LF_INVALID)
		{
			if (pLfHdr->ui32LfType == (FLMUINT32)SFLM_LF_TABLE)
			{
				if ((pTmpTable = pDb->m_pDict->getTable(
								(FLMUINT)pLfHdr->ui32LfNum)) == NULL)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
					goto Exit;
				}
				pTmpLFile = &pTmpTable->lfInfo;
			}
			else
			{
				flmAssert( pLfHdr->ui32LfType == (FLMUINT32)SFLM_LF_INDEX);
				if ((pTmpIndex = pDb->m_pDict->getIndex(
											(FLMUINT)pLfHdr->ui32LfNum)) == NULL)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
					goto Exit;
				}
				pTmpLFile = &pTmpIndex->lfInfo;
			}

			pTmpLFile->uiBlkAddress = uiFreeBlkAddr;
		}
		uiPos += sizeof( F_LF_HDR);
		pLfHdr++;
	}

	// Done with both blocks.

	ScaReleaseCache( pFreeSCache, FALSE);
	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache2 = bReleaseCache = FALSE;

	// Read left and right blocks and adjust their
	// pointers to point to the new block.
	// This doesn't matter what level of the b-tree
	// you are on.

	if (uiLeftBlkAddr)
	{
		if (RC_BAD( rc = getBlock( pDb, NULL, uiLeftBlkAddr,
									NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;
		if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pSCache->m_pBlkHdr->ui32NextBlkInChain = (FLMUINT32)uiFreeBlkAddr;
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	if (uiRightBlkAddr)
	{
		if (RC_BAD( rc = getBlock( pDb, NULL, uiRightBlkAddr,
									NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;
		if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
		{
			goto Exit;
		}

		pSCache->m_pBlkHdr->ui32PrevBlkInChain = (FLMUINT32)uiFreeBlkAddr;
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if (bReleaseCache2)
	{
		ScaReleaseCache( pFreeSCache, FALSE);
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}


/****************************************************************************
Desc:	Free the input avail block.  Link the block out of the free list
****************************************************************************/
RCODE F_Database::freeAvailBlk(
	F_Db *		pDb,
	FLMUINT		uiBlkAddr)
{
	RCODE					rc = NE_SFLM_OK;
	F_LARGEST_BLK_HDR	blkHdr;
	FLMUINT				uiPrevBlkAddr;
	FLMUINT				uiNextBlkAddr;
	F_CachedBlock *	pSCache;

	// Check for first avail block condition.

	if (uiBlkAddr == pDb->m_uiFirstAvailBlkAddr)
	{
		if (RC_OK( rc = blockUseNextAvail( pDb, &pSCache)))
		{
			ScaReleaseCache( pSCache, FALSE);
		}
		goto Exit;
	}

	// Not first block, unlink from list.

	// Read the block header and get pointers

	if (RC_BAD( rc = pDb->readBlkHdr( uiBlkAddr,
									(F_BLK_HDR *)&blkHdr, (FLMINT *)0)))
	{
		goto Exit;
	}
	uiPrevBlkAddr = (FLMUINT)blkHdr.all.stdBlkHdr.ui32PrevBlkInChain;
	uiNextBlkAddr = (FLMUINT)blkHdr.all.stdBlkHdr.ui32NextBlkInChain;

	// Read the previous block, if any

	if (uiPrevBlkAddr)
	{
		if (RC_BAD( rc = getBlock( pDb, NULL, uiPrevBlkAddr,
									NULL, &pSCache)))
		{
			goto Exit;
		}

		// Log the block before modifying it.

		if (RC_OK( rc = logPhysBlk( pDb, &pSCache)))
		{
			pSCache->m_pBlkHdr->ui32NextBlkInChain = (FLMUINT32)uiNextBlkAddr;
		}
		ScaReleaseCache( pSCache, FALSE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// Read the next block, if any.

	if (uiNextBlkAddr)
	{
		if (RC_BAD( rc = getBlock( pDb, NULL, uiNextBlkAddr,
										NULL, &pSCache)))
		{
			goto Exit;
		}

		// Log the block before modifying it.

		if (RC_OK( rc = logPhysBlk( pDb, &pSCache)))
		{
			pSCache->m_pBlkHdr->ui32PrevBlkInChain = (FLMUINT32)uiPrevBlkAddr;
		}
		ScaReleaseCache( pSCache, FALSE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}
