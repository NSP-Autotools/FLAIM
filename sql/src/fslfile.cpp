//------------------------------------------------------------------------------
// Desc:	Routines for reading and writing logical file headers.
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

FSTATIC FLMUINT FSLFileFindEmpty(
	FLMUINT			uiBlockSize,
	F_BLK_HDR *		pBlkHdr);

/***************************************************************************
Desc: Searches a block for an empty LFH slot.  This is called whenever
		a new logical file is create so we re-use the slots.
Ret:	0-Empty slot not found
		non-zero - offset in the block of the empty slot
***************************************************************************/
FSTATIC FLMUINT FSLFileFindEmpty(
	FLMUINT		uiBlockSize,
	F_BLK_HDR *	pBlkHdr)
{
	FLMUINT		uiPos = SIZEOF_STD_BLK_HDR;
	FLMUINT		uiEndPos = blkGetEnd( uiBlockSize, SIZEOF_STD_BLK_HDR,
									pBlkHdr);
	F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)pBlkHdr + SIZEOF_STD_BLK_HDR);

	while (uiPos < uiEndPos)
	{
		if (pLfHdr->ui32LfType == (FLMUINT32)SFLM_LF_INVALID)
		{
			break;
		}

		uiPos += sizeof( F_LF_HDR);
		pLfHdr++;
	}
	return( (uiPos < uiEndPos) ? uiPos : 0);
}

/***************************************************************************
Desc:	Retrieves the logical file header record from disk & updates LFILE.
		Called when it is discovered that the LFH for a
		particular logical file is out of date.
*****************************************************************************/
RCODE F_Database::lFileRead(
	F_Db *	pDb,
	LFILE *	pLFile)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache;
	FLMBOOL				bReleaseCache = FALSE;

	// Read in the block containing the logical file header

	if (RC_BAD( rc = getBlock( pDb, NULL,
								pLFile->uiBlkAddress, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	// Copy the LFH from the block to the LFILE

	FSLFileIn( (FLMBYTE *)(pSCache->m_pBlkHdr) + pLFile->uiOffsetInBlk,
			pLFile, pLFile->uiBlkAddress, pLFile->uiOffsetInBlk);

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Update the LFH data on disk.
*****************************************************************************/
RCODE F_Database::lFileWrite(
	F_Db *	pDb,
	LFILE *	pLFile)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache;
	FLMBOOL				bReleaseCache = FALSE;
	F_LF_HDR *			pLfHdr;
#ifdef DEBUG
	F_CachedBlock *	pTmpSCache = NULL;
#endif

	flmAssert( !pDb->m_pDatabase->m_pRfl || 
				  !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	// Read in the block containing the logical file header

	if (RC_BAD( rc = getBlock( pDb, NULL,
							pLFile->uiBlkAddress, NULL, &pSCache)))
	{
		goto Exit;
	}
	bReleaseCache = TRUE;

	// Log the block before modifying it

	if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	// Now modify the block and set its status to dirty

	pLfHdr = (F_LF_HDR *)((FLMBYTE *)(pSCache->m_pBlkHdr) +
									pLFile->uiOffsetInBlk);

	// If deleted, fill with 0, except for type - it is set below

	if (pLFile->eLfType == SFLM_LF_INVALID)
	{
		f_memset( pLfHdr, 0, sizeof( F_LF_HDR));
		pLfHdr->ui32LfType = (FLMUINT32)SFLM_LF_INVALID;
	}
	else
	{

#ifdef DEBUG
		if (RC_BAD( rc = getBlock( pDb, NULL,
								pLFile->uiRootBlk, NULL, &pTmpSCache)))
		{
			goto Exit;
		}

		if (!isRootBlk( (F_BTREE_BLK_HDR *)pTmpSCache->m_pBlkHdr))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		}
#endif

		pLfHdr->ui32LfType = (FLMUINT32)pLFile->eLfType;
		pLfHdr->ui32RootBlkAddr = (FLMUINT32)pLFile->uiRootBlk;
		pLfHdr->ui32LfNum = (FLMUINT32)pLFile->uiLfNum;
		pLfHdr->ui32EncDefNum = (FLMUINT32)pLFile->uiEncDefNum;

		if (pLFile->eLfType == SFLM_LF_TABLE)
		{
			pLfHdr->ui64NextRowId = pLFile->ui64NextRowId;
		}
		else
		{
			flmAssert( pLFile->eLfType == SFLM_LF_INDEX);
			pLfHdr->ui64NextRowId = 0;
		}
	}
	pLFile->bNeedToWriteOut = FALSE;
	
	// If the LFILE was deleted, we need to set pLFile->uiLfNum to zero.
	// This should happen only AFTER the LFILE update is logged, because
	// logLFileUpdate relies on pLFile->uiLfNum being non-zero.

	if (pLFile->eLfType == SFLM_LF_INVALID)
	{
		pLFile->uiLfNum = 0;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

#ifdef DEBUG
	if (pTmpSCache)
	{
		ScaReleaseCache( pTmpSCache, FALSE);
	}
#endif

	return( rc);
}

/***************************************************************************
Desc:	Creates and initializes a LFILE structure on disk and in memory.
*****************************************************************************/
RCODE F_Database::lFileCreate(
	F_Db *			pDb,
	LFILE *			pLFile,
	FLMUINT			uiLfNum,
	eLFileType		eLfType,
	FLMBOOL			bCounts,
	FLMBOOL			bHaveData,
	FLMUINT			uiEncDefNum)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pNewSCache = NULL;
	F_CachedBlock *	pSCache = NULL;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT				uiBlkAddress = 0;
	FLMUINT				uiNextBlkAddress;
	FLMUINT				uiEndPos = 0;
	FLMUINT				uiPos = 0;
	FLMBOOL				bReleaseCache2 = FALSE;
	FLMBOOL				bReleaseCache = FALSE;
	F_Btree *			pbTree = NULL;

	flmAssert( !pDb->m_pDatabase->m_pRfl || 
				  !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if (eLfType == SFLM_LF_TABLE)
	{
		if( bCounts)
		{
			// Force bCounts to be FALSE in this case
			flmAssert( 0);
			bCounts = FALSE;
		}

		if( !bHaveData)
		{
			// Force bHaveData to be TRUE in this case
			flmAssert( 0);
			bHaveData = TRUE;
		}
	}

	// Find an available slot to create the LFH -- follow the linked list
	// of LFH blocks to find one.

	uiNextBlkAddress = (FLMUINT)m_uncommittedDbHdr.ui32FirstLFBlkAddr;

	// Better be at least one LFH block.

	if (uiNextBlkAddress == 0)
	{
		rc = RC_SET( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	while (uiNextBlkAddress != 0)
	{
		if (bReleaseCache)
		{
			ScaReleaseCache( pSCache, FALSE);
			bReleaseCache = FALSE;
		}
		uiBlkAddress = uiNextBlkAddress;

		if (RC_BAD( rc = getBlock( pDb, NULL,
								uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pBlkHdr = pSCache->m_pBlkHdr;
		uiNextBlkAddress = (FLMUINT)pBlkHdr->ui32NextBlkInChain;
		uiEndPos = blkGetEnd( m_uiBlockSize, SIZEOF_STD_BLK_HDR, pBlkHdr);
		if ((uiPos = FSLFileFindEmpty( m_uiBlockSize, pBlkHdr)) != 0)
		{
			break;
		}
	}

	// If we did not find a deleted slot we can use, see if there
	// is room for a new logical file in the last block
	// in the chain.  If not, allocate a new block.

	if (uiPos == 0)
	{
		uiEndPos = blkGetEnd( m_uiBlockSize, SIZEOF_STD_BLK_HDR, pBlkHdr);

		// Allocate new block?

		if (uiEndPos + sizeof( F_LF_HDR) >= m_uiBlockSize)
		{
			if (RC_BAD( rc = createBlock( pDb, &pNewSCache)))
			{
				goto Exit;
			}
			bReleaseCache2 = TRUE;

			pBlkHdr = pNewSCache->m_pBlkHdr;
			uiNextBlkAddress = (FLMUINT)pBlkHdr->ui32BlkAddr;

			// Modify the new block's next pointer and other fields.

			pBlkHdr->ui32NextBlkInChain = 0;
			pBlkHdr->ui32PrevBlkInChain = (FLMUINT32)uiBlkAddress;
			pBlkHdr->ui8BlkType = BT_LFH_BLK;
			pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)(m_uiBlockSize -
																	SIZEOF_STD_BLK_HDR);

			if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
			{
				goto Exit;
			}
			pSCache->m_pBlkHdr->ui32NextBlkInChain = (FLMUINT32)uiNextBlkAddress;

			// Set everything up so we are pointing to the new block.

			ScaReleaseCache( pSCache, FALSE);
			pSCache = pNewSCache;
			bReleaseCache2 = FALSE;
			uiEndPos = blkGetEnd( m_uiBlockSize, SIZEOF_STD_BLK_HDR,
								pSCache->m_pBlkHdr);
			uiBlkAddress = uiNextBlkAddress;
		}

		// Modify the end of block pointer -- log block before modifying.

		uiPos = uiEndPos;
		uiEndPos += sizeof( F_LF_HDR);
	}

	// Call memset to ensure unused bytes are zero.
	// pBlkHdr, uiPos and uiEndPos should ALL be set.

	if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}
	pBlkHdr = pSCache->m_pBlkHdr;
	f_memset( (FLMBYTE *)(pBlkHdr) + uiPos, 0, sizeof( F_LF_HDR));
	flmAssert( uiEndPos >= SIZEOF_STD_BLK_HDR &&
				  uiEndPos <= m_uiBlockSize);
	pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)(m_uiBlockSize - uiEndPos);

	// Done with block in this routine

	ScaReleaseCache( pSCache, FALSE);
	bReleaseCache = FALSE;

	// Set the variables in the LFILE structure to later save to disk

	pLFile->uiLfNum = uiLfNum;
	pLFile->eLfType = eLfType;
	
	// NOTE: call to btCreate below sets pLFile->uiRootBlk
	
	// pLFile->uiRootBlk = 0;
	pLFile->uiBlkAddress = uiBlkAddress;
	pLFile->uiOffsetInBlk = uiPos;
	pLFile->uiEncDefNum = uiEncDefNum;
	if (eLfType == SFLM_LF_TABLE)
	{
		pLFile->ui64NextRowId = 1;
		pLFile->bNeedToWriteOut = TRUE;
	}
	else
	{
		pLFile->ui64NextRowId = 0;
	}

	// Get the btree...
	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pbTree)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pbTree->btCreate( pDb, pLFile, bCounts, bHaveData)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = lFileWrite( pDb, pLFile)))
	{
		goto Exit;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	if (bReleaseCache2)
	{
		ScaReleaseCache( pNewSCache, FALSE);
	}
	if (pbTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pbTree);
	}
	return( rc);
}

/***************************************************************************
Desc:	Delete a logical file.
*****************************************************************************/
RCODE F_Database::lFileDelete(
	F_Db *			pDb,
	LFILE *			pLFile,
	FLMBOOL			bCounts,
	FLMBOOL			bHaveData)
{
	RCODE				rc = NE_SFLM_OK;
	F_Btree *		pbTree = NULL;
	FLMUINT			uiLoop;
	FLMUINT			uiBlkChains[ BH_MAX_LEVELS];
	FLMUINT			uiChainCount;
	F_Row *			pRow = NULL;

	flmAssert( pDb->m_uiFlags & FDB_UPDATED_DICTIONARY);

	// Get a btree

	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pbTree)))
	{
		goto Exit;
	}

	// Delete the logical file's B-Tree blocks.
	// If there is no root block, no need to do anything.

	flmAssert( pLFile->uiRootBlk);

	if (RC_BAD( rc = pbTree->btOpen( pDb, pLFile, bCounts, bHaveData)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pbTree->btGetBlockChains( uiBlkChains, &uiChainCount)))
	{
		goto Exit;
	}

	// Indexes and tables are always deleted in the background.
	// Add a row to the block chain table for each chain found.

	for (uiLoop = 0; uiLoop < uiChainCount; uiLoop++)
	{
		if (pRow)
		{
			pRow->ReleaseRow();
			pRow = NULL;
		}
		
		// Create the row
		
		if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( pDb,
				SFLM_TBLNUM_BLOCK_CHAINS, &pRow)))
		{
			goto Exit;
		}
		
		// Set the block address column in the row.
		
		if (RC_BAD( rc = pRow->setUINT( pDb,
				SFLM_COLNUM_BLOCK_CHAINS_BLOCK_ADDRESS,
				uiBlkChains [uiLoop])))
		{
			goto Exit;
		}
	}

	// Signal the maintenance thread that it has work to do

	f_semSignal( m_hMaintSem);

	// Delete the LFILE entry.

	pLFile->uiRootBlk = 0;
	pLFile->eLfType = SFLM_LF_INVALID;
	
	if( RC_BAD( rc = lFileWrite( pDb, pLFile)))
	{
		goto Exit;
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}
	if( pbTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pbTree);
	}

	return( rc);
}

