//------------------------------------------------------------------------------
// Desc:	This class handles all of operations on a given B-Tree.
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC FLMUINT btGetEntryDataLength(
	FLMBYTE *			pucEntry,
	const FLMBYTE **	ppucDataRV,
	FLMUINT *			puiOADataLengthRV,
	FLMBOOL *			pbDOBlockRV);

FSTATIC RCODE btGetEntryData(
	FLMBYTE *		pucEntry,
	FLMBYTE *		pucBufferRV,
	FLMUINT			uiBufferSize,
	FLMUINT *		puiLenDataRV);

/***************************************************************************
Desc:	Constructor
****************************************************************************/
F_Btree::F_Btree( void)
{
	m_bOpened = FALSE;
	m_pStack = NULL;
	m_uiStackLevels = 0;
	m_uiRootLevel = 0;
	f_memset(m_Stack, 0, sizeof(m_Stack));
	m_pLFile = NULL;
	m_pDb = NULL;
	m_bTempDb = FALSE;
	m_pucTempBlk = NULL;
	m_pucTempDefragBlk = NULL;
	m_bCounts = FALSE;
	m_bData = TRUE;		// Default
	m_bSetupForRead = FALSE;
	m_bSetupForWrite = FALSE;
	m_bSetupForReplace = FALSE;
	m_uiBlockSize = 0;
	m_uiDefragThreshold = 0;
	m_uiOverflowThreshold = 0;
	m_pReplaceInfo = NULL;
	m_pReplaceStruct = NULL;
	m_uiReplaceLevels = 0;
	m_ui64CurrTransID = 0;
	m_ui64LastBlkTransId = 0;
	m_ui64PrimaryBlkTransId = 0;
	m_uiBlkChangeCnt = 0;
	m_ui64LowTransId = FLM_MAX_UINT64;
	m_bMostCurrent = FALSE;
	m_uiDataLength = 0;
	m_uiPrimaryDataLen = 0;
	m_uiOADataLength = 0;
	m_uiDataRemaining = 0;
	m_uiOADataRemaining = 0;
	m_uiOffsetAtStart = 0;
	m_bDataOnlyBlock = FALSE;
	m_bOrigInDOBlocks = FALSE;
	m_ui32PrimaryBlkAddr = 0;
	m_uiPrimaryOffset = 0;
	m_ui32DOBlkAddr = 0;
	m_ui32CurBlkAddr = 0;
	m_uiCurOffset = 0;
	m_pucDataPtr = NULL;
	m_bFirstRead = FALSE;
	m_pSCache = NULL;
	m_pBlkHdr = NULL;
	m_uiSearchLevel = BH_MAX_LEVELS;
	m_pNext = NULL;
	m_pCompare = NULL;
}

/***************************************************************************
Desc:	Destructor
****************************************************************************/
F_Btree::~F_Btree( void)
{
	if ( m_bOpened)
	{
		btClose();
	}
}

/***************************************************************************
Desc: Function to create a new (empty) B-Tree.  To do this, we create the
		root block.  Upon return, the Root block address and the block address
		will be set in the LFile.
****************************************************************************/
RCODE F_Btree::btCreate(
	F_Db *					pDb,					// In
	LFILE *					pLFile,				// In/Out
	FLMBOOL					bCounts,				// In
	FLMBOOL					bData					// In
	)
{
	RCODE						rc = NE_SFLM_OK;
	F_CachedBlock *		pSCache = NULL;
	F_BTREE_BLK_HDR *		pBlkHdr = NULL;
	FLMUINT16 *				pui16Offset;
	FLMBYTE *				pucEntry;
	FLMBYTE					ucLEMEntry[ 3];
	FLMUINT					uiFlags = 0;
	FLMUINT					uiLEMSize;

	// We can't create a new Btree if we have already been initialized.
	
	if (m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify that we are in an update transaction.
	if (pDb->m_eTransType != SFLM_UPDATE_TRANS && !pDb->m_pDatabase->m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Initialize the returned root block address to 0 incase of an error.
	pLFile->uiRootBlk = 0;

	// Call createBlock to create a new block
	if (RC_BAD( rc = pDb->m_pDatabase->createBlock( pDb, &pSCache)))
	{
		goto Exit;
	}

	// Save the new block address as the root block address
	pLFile->uiRootBlk = pSCache->m_uiBlkAddress;

	// Save the block address and identify the block as the root block.
	if (RC_BAD( rc = btOpen( pDb, pLFile, bCounts, bData)))
	{
		goto Exit;
	}
	pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;

	setRootBlk( pBlkHdr);
	pBlkHdr->ui16LogicalFile = (FLMUINT16)pLFile->uiLfNum;
	setBlkLfType( pBlkHdr, pLFile->eLfType);
	pBlkHdr->ui8BlkLevel = 0;
	pBlkHdr->stdBlkHdr.ui8BlkType = (bData ? BT_LEAF_DATA : BT_LEAF);

	pBlkHdr->stdBlkHdr.ui32PrevBlkInChain = 0;
	pBlkHdr->stdBlkHdr.ui32NextBlkInChain = 0;
	
	if (pLFile->uiEncDefNum)
	{
		setBlockEncrypted( (F_BLK_HDR *)pBlkHdr);
	}

	// Insert a LEM into the block
	uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;

	if (RC_BAD( rc = buildAndStoreEntry( (bData ? BT_LEAF_DATA : BT_LEAF),
		uiFlags, NULL, 0, NULL, 0, 0, 0, 0, &ucLEMEntry[0],
		3, &uiLEMSize)))
	{
		goto Exit;
	}

	pui16Offset = BtOffsetArray((FLMBYTE *)pBlkHdr, 0);
	pucEntry = (FLMBYTE *)pBlkHdr + m_uiBlockSize - uiLEMSize;

	bteSetEntryOffset( pui16Offset, 0, (FLMUINT16)(pucEntry - (FLMBYTE *)pBlkHdr));
	f_memcpy( pucEntry, ucLEMEntry, uiLEMSize);

	// Offset Entry & 2 byte LEM
	pBlkHdr->stdBlkHdr.ui16BlkBytesAvail = (FLMUINT16)(m_uiBlockSize -
															sizeofBTreeBlkHdr(pBlkHdr) -
															uiLEMSize - 2);
	pBlkHdr->ui16HeapSize = pBlkHdr->stdBlkHdr.ui16BlkBytesAvail;

	// There is one entry now.
	pBlkHdr->ui16NumKeys = 1;

Exit:

	if (pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: Btree initialization function.
****************************************************************************/
RCODE F_Btree::btOpen(
	F_Db *						pDb,
	LFILE *						pLFile,
 	FLMBOOL						bCounts,
	FLMBOOL						bData,
	IF_ResultSetCompare *	pCompare)
{
	RCODE				rc = NE_SFLM_OK;
	F_Database *	pDatabase= pDb->m_pDatabase;

	if ( m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	if (pDb->m_eTransType == SFLM_NO_TRANS && !pDatabase->m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}


	if( !pLFile->uiRootBlk)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
		goto Exit;
	}

	m_pLFile = pLFile;
	m_uiBlockSize = pDatabase->m_uiBlockSize;
	m_uiDefragThreshold = m_uiBlockSize / 20;
	m_uiOverflowThreshold = (m_uiBlockSize * 8) / 5;
	m_bCounts = bCounts;
	m_bData = bData;
	m_pDb = pDb;
	m_bTempDb = pDatabase->m_bTempDb;
	m_pReplaceInfo = NULL;
	m_uiReplaceLevels = 0;
	m_ui64CurrTransID = 0;
	m_ui64LastBlkTransId = 0;
	m_ui64PrimaryBlkTransId = 0;
	m_uiBlkChangeCnt = 0;
	m_uiSearchLevel = BH_MAX_LEVELS;

	m_bSetupForRead = FALSE;
	m_bSetupForWrite = FALSE;
	m_bSetupForReplace = FALSE;

	// Buffer is required to hold at least the maximum number of
	// offsets possible given the block size with a minimum entry size of
	// 5 bytes.  Each offset is 2 bytes.  We only need this buffer when we
	// are in an update transaction.

	if (pDb->m_eTransType == SFLM_UPDATE_TRANS || m_bTempDb)
	{
		m_uiBufferSize = m_uiBlockSize * 2;
	}
	else
	{
		m_uiBufferSize = 0;
	}

	// If we are in an update transaction, there are certain buffers that
	// are used.  Make sure these are allocated.

	if( (pDb->m_eTransType == SFLM_UPDATE_TRANS || m_bTempDb) &&
		!pDatabase->m_pucUpdBuffer)
	{
		// Buffer should be at least 80% of two blocks.  To make sure the other
		// structures being allocated here are aligned, we allocate 2 times the
		// database's block size for the update buffer.

		pDatabase->m_uiUpdBufferSize = m_uiBlockSize * 2;
		flmAssert( pDatabase->m_uiUpdBufferSize >= m_uiOverflowThreshold);
		
		if( RC_BAD( rc = f_alloc(
							pDatabase->m_uiUpdBufferSize +
							(m_uiBlockSize * 2) +
							m_uiBufferSize +
							sizeof( BTREE_REPLACE_STRUCT) * BH_MAX_LEVELS,
							&pDatabase->m_pucUpdBuffer)))
		{
			goto Exit;
		}

		// Temporary buffers for the F_Btree class will immediately follow
		// the update buffer.

		pDatabase->m_pucBTreeTmpBlk =
						pDatabase->m_pucUpdBuffer + pDatabase->m_uiUpdBufferSize;
		pDatabase->m_pucBTreeTmpDefragBlk =
						pDatabase->m_pucBTreeTmpBlk + pDatabase->m_uiBlockSize;
		pDatabase->m_pucBtreeBuffer =
						pDatabase->m_pucBTreeTmpDefragBlk +	pDatabase->m_uiBlockSize;
		pDatabase->m_pucReplaceStruct =
						pDatabase->m_pucBtreeBuffer + m_uiBufferSize;
	}

	// NOTE: These temporary buffers may be NULL.  They are only allocated the
	// first time we detect that we are operating inside an update
	// transaction - because we need to make sure that only one thread
	// actually does the allocation.  The assumption here is that the
	// buffers are only ever used during update operations, which will
	// *always* be inside update transactions.

	m_pucTempBlk = pDatabase->m_pucBTreeTmpBlk;
	m_pucTempDefragBlk = pDatabase->m_pucBTreeTmpDefragBlk;
	m_pucBuffer = pDatabase->m_pucBtreeBuffer;
	m_pReplaceStruct = (BTREE_REPLACE_STRUCT *)pDatabase->m_pucReplaceStruct;
	
	flmAssert( !m_pCompare);
	if ((m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_bOpened = TRUE;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc: Btree close function
****************************************************************************/
void F_Btree::btClose()
{
	FLMUINT			uiLoop;

	// Ok to close multiple times.
	if (!m_bOpened)
	{
		// Btree is not open.  Just return.
		return;
	}

	m_pLFile = NULL;
	m_pDb = NULL;
	m_bTempDb = FALSE;

	for (uiLoop = 0; uiLoop < BH_MAX_LEVELS; uiLoop++)
	{
		m_Stack[ uiLoop].pucKeyBuf = NULL;
	}

	// Release any blocks still held in the stack.
	btRelease();

	if (m_pSCache)
	{
		flmAssert( 0);
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}
	
	if (m_pCompare)
	{
		m_pCompare->Release();
		m_pCompare = NULL;
	}

	m_bOpened = FALSE;
}

/***************************************************************************
Desc: Delete the entire tree
****************************************************************************/
RCODE F_Btree::btDeleteTree(
	IF_DeleteStatus *		ifpDeleteStatus)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiNumLevels;
	FLMUINT		puiBlkAddrs[ BH_MAX_LEVELS];
	FLMUINT		uiLoop;

	flmAssert( m_bOpened);

	// Verify the transaction type

	if (m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Fill up uiBlkAddrs and calculate the number of levels.

	if( RC_BAD( rc = btGetBlockChains( puiBlkAddrs, &uiNumLevels)))
	{
		goto Exit;
	}

	// Iterate over the list of block chains and free all of the blocks

	for( uiLoop = 0; uiLoop < uiNumLevels; uiLoop++)
	{
		if( RC_BAD( rc = btFreeBlockChain( 
			m_pDb, m_pLFile, puiBlkAddrs[ uiLoop], 0, NULL, NULL,
			ifpDeleteStatus)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE btFreeBlockChain(
	F_Db *					pDb,
	LFILE *					pLFile,
	FLMUINT					uiStartAddr,
	FLMUINT					uiBlocksToFree,
	FLMUINT *				puiBlocksFreed,
	FLMUINT *				puiEndAddr,
	IF_DeleteStatus *		ifpDeleteStatus)
{
	RCODE					rc = NE_SFLM_OK;
	F_Database *		pDatabase = pDb->getDatabase();
	F_CachedBlock *	pCurrentBlk = NULL;
	F_CachedBlock *	pDOSCache = NULL;
	FLMBYTE *			pBlk;
	FLMBYTE *			pucEntry;
	FLMUINT				uiEntryNum;
	FLMUINT				uiDOBlkAddr;
	FLMBYTE				ucDOBlkAddr[ 4];
	FLMUINT				uiStatusCounter = 0;
	FLMUINT				uiNextBlkAddr = 0;
	FLMUINT				uiCurrentBlkAddr = 0;
	FLMUINT				uiTreeBlocksFreed = 0;
	FLMUINT				uiDataBlocksFreed = 0;
	FLMBOOL				bFreeAll = FALSE;

	if( !uiBlocksToFree)
	{
		bFreeAll = TRUE;
	}

	// Verify the transaction type

	if( pDb->getTransType() != SFLM_UPDATE_TRANS)
	{
		rc = RC_SET_AND_ASSERT( pDb->getTransType() == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Now, go through the chain and delete the blocks...

	uiCurrentBlkAddr = uiStartAddr;
	while( uiCurrentBlkAddr)
	{
		if( !bFreeAll && uiTreeBlocksFreed >= uiBlocksToFree)
		{
			break;
		}

		if( RC_BAD( pDatabase->getBlock( pDb, pLFile,
								uiCurrentBlkAddr, NULL, &pCurrentBlk)))
		{
			goto Exit;
		}

		pBlk = (FLMBYTE *)pCurrentBlk->getBlockPtr();
		uiNextBlkAddr = ((F_BLK_HDR *)pBlk)->ui32NextBlkInChain;

		// If this is a leaf block, then there may be entries 
		// with data-only references that will need to be cleaned up too.

		if( getBlkType( pBlk) == BT_LEAF_DATA)
		{
			for( uiEntryNum = 0;
				  uiEntryNum < ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys;
				  uiEntryNum++)
			{
				pucEntry = BtEntry( pBlk, uiEntryNum);

				if( bteDataBlockFlag( pucEntry))
				{
					// Get the data-only block address

					if( RC_BAD( rc = btGetEntryData( 
						pucEntry, &ucDOBlkAddr[ 0], 4, NULL)))
					{
						goto Exit;
					}

					uiDOBlkAddr = bteGetBlkAddr( (FLMBYTE *)&ucDOBlkAddr[ 0]);
					while( uiDOBlkAddr)
					{
						if( RC_BAD(
							rc = pDatabase->getBlock( pDb, pLFile,
												  uiDOBlkAddr, NULL, &pDOSCache)))
						{
							goto Exit;
						}

						uiDOBlkAddr = pDOSCache->getBlockPtr()->ui32NextBlkInChain;
						rc = pDatabase->blockFree( pDb, pDOSCache);
						pDOSCache = NULL;

						if (RC_BAD( rc))
						{
							goto Exit;
						}

						uiDataBlocksFreed++;
					}
				}
			}
		}

		rc = pDatabase->blockFree( pDb, pCurrentBlk);
		pCurrentBlk = NULL;

		if( RC_BAD( rc))
		{
			goto Exit;
		}

		if( ifpDeleteStatus && pLFile && ++uiStatusCounter >= 25)
		{
			uiStatusCounter = 0;
			if( RC_BAD( rc = ifpDeleteStatus->reportDelete( 
					uiTreeBlocksFreed + uiDataBlocksFreed,
					pDatabase->getBlockSize())))
			{
				goto Exit;
			}
		}

		uiTreeBlocksFreed++;
		uiCurrentBlkAddr = uiNextBlkAddr;
	}

	if( puiBlocksFreed)
	{
		*puiBlocksFreed = uiTreeBlocksFreed;
	}

	if( puiEndAddr)
	{
		*puiEndAddr = uiCurrentBlkAddr;
	}

Exit:

	if( pDOSCache)
	{
		ScaReleaseCache( pDOSCache, FALSE);
	}

	if( pCurrentBlk)
	{
		ScaReleaseCache( pCurrentBlk, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Returns the address of the first block at each level of the tree
Note:	puiBlockAddrs is assumed to point to a buffer that can store
		BH_MAX_LEVELS FLMUINT values
****************************************************************************/
RCODE F_Btree::btGetBlockChains(
	FLMUINT *	puiBlockAddrs,
	FLMUINT *	puiNumLevels)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiNumLevels = 0;
	FLMUINT				uiLFileNum;
	FLMBOOL				bIsIndex;
	FLMUINT32			ui32NextBlkAddr;
	F_CachedBlock *	pCurrentBlk = NULL;
	FLMBYTE *			pucBlk;
	FLMBYTE *			pucEntry;

	flmAssert( m_bOpened);

	// Verify the transaction type

	if( m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Fill puiBlockAddrs and calculate the number of levels.
	// NOTE: Normally, level 0 is the leaf level.  In this function,
	// puiBlockAddrs[ 0] is the ROOT and puiBlockAddrs[ uiNumLevels - 1] 
	// is the LEAF!

	ui32NextBlkAddr = (FLMUINT32)m_pLFile->uiRootBlk;
	uiLFileNum = m_pLFile->uiLfNum;
	bIsIndex = m_pLFile->eLfType == SFLM_LF_INDEX ? TRUE : FALSE;

	while( ui32NextBlkAddr)
	{
		puiBlockAddrs[ uiNumLevels++] = ui32NextBlkAddr;

		if( RC_BAD( m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
										ui32NextBlkAddr, NULL, &pCurrentBlk)))
		{
			goto Exit;
		}

		pucBlk = (FLMBYTE *)pCurrentBlk->m_pBlkHdr;

		if( getBlkType( pucBlk) == BT_LEAF || getBlkType( pucBlk) == BT_LEAF_DATA)
		{
			ui32NextBlkAddr = 0;
		}
		else
		{
			// The child block address is the first part of the entry

			pucEntry = BtEntry( pucBlk, 0);
			ui32NextBlkAddr = bteGetBlkAddr( pucEntry);
		}

		// Release the current block

		ScaReleaseCache( pCurrentBlk, FALSE);
		pCurrentBlk = NULL;
	}

	*puiNumLevels = uiNumLevels;

Exit:

	if( pCurrentBlk)
	{
		ScaReleaseCache( pCurrentBlk, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: Function to insert an entry into the Btree.
****************************************************************************/
RCODE F_Btree::btInsertEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bFirst,
	FLMBOOL				bLast,
	FLMUINT32 *			pui32BlkAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE				rc = NE_SFLM_OK;
	F_BLK_HDR *		pBlkHdr;
	FLMBYTE			pucDOAddr[ 4];
	
	if ( !m_bOpened || m_bSetupForRead || m_bSetupForReplace ||
		  (m_bSetupForWrite && bFirst))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	flmAssert( m_uiSearchLevel >= BH_MAX_LEVELS);
	flmAssert( !m_pDb->m_pDatabase->m_pRfl || 
				  !m_pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if( !uiKeyLen)
	{
		rc = RC_SET( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Verify the transaction type
	
	if( m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Be sure to clear the Data Only flag.
	
	if( bFirst)
	{
		m_bDataOnlyBlock = FALSE;
	}

	if( bLast)
	{

		// We need to locate where we should insert the new entry.
		
		rc = findEntry( pucKey, uiKeyLen, FLM_EXACT);

		// Should not find this entry.  If we get back anything other than
		// an NE_SFLM_NOT_FOUND, then there is a problem.
		
		if( rc != NE_SFLM_NOT_FOUND)
		{
			if( RC_OK( rc))
			{
				rc = RC_SET( NE_SFLM_NOT_UNIQUE);
			}
			goto Exit;
		}
	}

	if( bFirst && (!bLast || (uiKeyLen + uiDataLen > m_uiOverflowThreshold)))
	{
		// If bLast is not set, then we will setup to store the data in
		// data only blocks.  The assumption is that whenever we don't see bLast
		// set when starting an insert, then the data is so large that it must
		// be placed in a chain of Data Only blocks.  There is no way for me to
		// check the final size of the data ahead of time, so I rely on the
		// calling routine to figure this out for me.

		// Get one empty block to begin with.

		flmAssert( m_pSCache == NULL);
		if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock( m_pDb, &m_pSCache)))
		{
			goto Exit;
		}

		// The data going in will be stored in Data-only blocks.
		// Setup the block header...

		pBlkHdr = m_pSCache->m_pBlkHdr;
		pBlkHdr->ui8BlkType = BT_DATA_ONLY;
		pBlkHdr->ui32PrevBlkInChain = 0;
		pBlkHdr->ui32NextBlkInChain = 0;

		if( m_pLFile->uiEncDefNum)
		{
			((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum = (FLMUINT32)m_pLFile->uiEncDefNum;
			setBlockEncrypted( pBlkHdr);
		}

		pBlkHdr->ui16BlkBytesAvail =
				(FLMUINT16)(m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr));

		m_uiDataRemaining = m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr);
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = TRUE;
		m_bSetupForWrite = TRUE;
		m_ui32DOBlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;
		m_ui32CurBlkAddr = m_ui32DOBlkAddr;
	}

	if( m_bDataOnlyBlock)
	{
		if( RC_BAD( rc = storeDataOnlyBlocks( pucKey, uiKeyLen, bFirst,
			pucData, uiDataLen)))
		{
			goto Exit;
		}
	}

	if( bLast)
	{
		const FLMBYTE *		pucLocalData;
		FLMUINT					uiLocalDataLen;
		F_ELM_UPD_ACTION		eAction;

		if( m_bDataOnlyBlock)
		{
			// build an entry that points to the DO block.

			UD2FBA( m_ui32DOBlkAddr, pucDOAddr);
			pucLocalData = &pucDOAddr[0];
			uiLocalDataLen = m_uiOADataLength;
			eAction = ELM_INSERT_DO;

		}
		else
		{
			pucLocalData = pucData;
			uiLocalDataLen = uiDataLen;
			eAction = ELM_INSERT;
		}

		if( RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucLocalData,
			uiLocalDataLen, eAction)))
		{
			goto Exit;
		}

		if( pui32BlkAddr)
		{
			*pui32BlkAddr = m_ui32PrimaryBlkAddr;
		}

		if( puiOffsetIndex)
		{
			*puiOffsetIndex = m_uiCurOffset;
		}

		m_bSetupForWrite = FALSE;
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Function to remove an entry into the Btree.
****************************************************************************/
RCODE F_Btree::btRemoveEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucValue = NULL;
	FLMUINT			uiLen = 0;

	if ( !m_bOpened)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify the Txn type
	if (m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	flmAssert( !m_pDb->m_pDatabase->m_pRfl || 
				  !m_pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	btResetBtree();

	// We need to locate where we should remove the entry.
	if (RC_BAD( rc = findEntry( pucKey,
										 uiKeyLen,
										 FLM_EXACT)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucValue, uiLen,
			ELM_REMOVE)))
	{
		goto Exit;
	}

Exit:

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc:	Function to provide a streaming interface for replacing large 
		data elements.
****************************************************************************/
RCODE F_Btree::btReplaceEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bFirst,
	FLMBOOL				bLast,
	FLMBOOL				bTruncate,
	FLMUINT32 *			pui32BlkAddr,
	FLMUINT *			puiOffsetIndex)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucEntry;
	F_BLK_HDR *			pBlkHdr;
	const FLMBYTE *	pucLocalData = NULL;
	FLMUINT				uiOADataLength = 0;
	FLMBYTE				pucDOAddr[ 4];

	if( !m_bOpened || m_bSetupForRead || m_bSetupForWrite ||
		  (m_bSetupForReplace && bFirst))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	flmAssert( m_uiSearchLevel >= BH_MAX_LEVELS);
	flmAssert( !m_pDb->m_pDatabase->m_pRfl || 
				  !m_pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if (!uiKeyLen)
	{
		rc = RC_SET( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Verify the Txn type
	if (m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}


	// Be sure to clear the Data Only flags
	
	if( bFirst)
	{
		m_bDataOnlyBlock = FALSE;
		m_bOrigInDOBlocks = FALSE;
	}

	if( bFirst || bLast)
	{

		// We need to locate the entry we want to replace
		
		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT, NULL,
			pui32BlkAddr, puiOffsetIndex)))
		{
			goto Exit;
		}

		// We must first determine if the existing entry is stored
		// in data only blocks.
		
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
								  m_pStack->uiCurOffset);

		btGetEntryDataLength( pucEntry, &pucLocalData,
			&uiOADataLength, &m_bOrigInDOBlocks);

	}

	if( bFirst && (!bLast || (bLast && !bTruncate && m_bOrigInDOBlocks) ||
			(uiKeyLen + uiDataLen > m_uiOverflowThreshold)))
	{
		// If bLast is not set, then we will setup to store the data in
		// data only blocks.
		
		m_bDataOnlyBlock = TRUE;
		flmAssert( m_pSCache == NULL);
		
		if( m_bOrigInDOBlocks)
		{
			// Need to get the first DO block, and work from there.
			
			m_ui32DOBlkAddr = bteGetBlkAddr( pucLocalData);
			
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_ui32DOBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
		}
		else
		{
			// Get one empty block to begin with
			
			if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock( m_pDb, &m_pSCache)))
			{
				goto Exit;
			}

			// The data going in will be stored in Data-only blocks.
			// Setup the block header...
			
			pBlkHdr = m_pSCache->m_pBlkHdr;
			pBlkHdr->ui8BlkType = BT_DATA_ONLY;
			pBlkHdr->ui32PrevBlkInChain = 0;
			pBlkHdr->ui32NextBlkInChain = 0;

			if (m_pLFile->uiEncDefNum)
			{
				((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum = (FLMUINT32)m_pLFile->uiEncDefNum;
				setBlockEncrypted( pBlkHdr);
			}

			pBlkHdr->ui16BlkBytesAvail =
						(FLMUINT16)(m_uiBlockSize - 
							sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr));
		}

		m_uiDataRemaining =	m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr);
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = TRUE;
		m_bSetupForReplace = TRUE;
		m_ui32DOBlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;
		m_ui32CurBlkAddr = m_ui32DOBlkAddr;
	}

	if( m_bDataOnlyBlock)
	{
		if( !bTruncate && !m_bOrigInDOBlocks)
		{
			bTruncate = TRUE;
		}

		// May need to skip over the key that is stored in the first DO block.
		// We only want to do this the first time in here.  The test to determine
		// if this is our first time in this block is to see if the m_uiDataLength
		// is equal to the m_uiDataRemaining.  They would only be the same on the
		// first time for each DO block.
		
		if( m_bOrigInDOBlocks && m_pSCache &&
			m_pSCache->m_pBlkHdr->ui32PrevBlkInChain == 0 && !m_uiDataLength)
		{
			m_uiDataRemaining -= (uiKeyLen + 2);
		}

		if( RC_BAD( rc = replaceDataOnlyBlocks( pucKey, uiKeyLen,
			!m_bOrigInDOBlocks && bFirst, pucData, uiDataLen, bLast,
			bTruncate)))
		{
			goto Exit;
		}
	}

	// If we were writing to Data Only Blocks and we are not truncating the
	// data, then we are done here.
	
	if( m_bDataOnlyBlock && !bTruncate)
	{
		if( bLast && (uiOADataLength <= m_uiOADataLength))
		{
			bTruncate = TRUE;
		}
		else
		{
			goto Exit;
		}
	}

	// Only replace the entry on the last call.
	
	if( bLast)
	{
		FLMUINT					uiLocalDataLen;
		F_ELM_UPD_ACTION		eAction;

		if (m_bDataOnlyBlock)
		{
			// build an entry that points to the DO block.

			UD2FBA( m_ui32DOBlkAddr, pucDOAddr);

			pucLocalData = &pucDOAddr[0];
			uiLocalDataLen = m_uiOADataLength;
			eAction = ELM_REPLACE_DO;

		}
		else
		{
			pucLocalData = pucData;
			uiLocalDataLen = uiDataLen;
			eAction = ELM_REPLACE;
		}

		if( RC_BAD( rc = updateEntry( pucKey, uiKeyLen, pucLocalData,
			uiLocalDataLen, eAction, bTruncate)))
		{
			goto Exit;
		}
	}

Exit:

	if (RC_OK( rc))
	{
		if (pui32BlkAddr)
		{
			*pui32BlkAddr = m_ui32PrimaryBlkAddr;
		}

		if (puiOffsetIndex)
		{
			*puiOffsetIndex = m_uiCurOffset;
		}
	}

	if( bLast)
	{
		m_bSetupForReplace = FALSE;
	}

	if (m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Function to search the Btree for a specific key.
****************************************************************************/
RCODE F_Btree::btLocateEntry(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyBufSize,
	FLMUINT *		puiKeyLen,
	FLMUINT			uiMatch,
	FLMUINT *		puiPosition,		// May be NULL
	FLMUINT *		puiDataLength,
	FLMUINT32 *		pui32BlkAddr,
	FLMUINT *		puiOffsetIndex)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucEntry = NULL;

	flmAssert( pucKey && uiKeyBufSize && puiKeyLen);

	if (!m_bOpened || m_bSetupForWrite || m_bSetupForReplace)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	m_bSetupForRead = FALSE;

	// Verify the Txn type
	
	if (m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// Find the entry we are interested in.
	if (RC_BAD(rc = findEntry( pucKey,
										*puiKeyLen,
										uiMatch,
										puiPosition,
										pui32BlkAddr,
										puiOffsetIndex)))
	{
		goto Exit;
	}

	m_ui64LowTransId = m_pStack->pBlkHdr->stdBlkHdr.ui64TransID;
	m_bMostCurrent = (m_pStack->pSCache->m_ui64HighTransID == FLM_MAX_UINT64)
											? TRUE
											: FALSE;

	m_ui32PrimaryBlkAddr =	m_pStack->ui32BlkAddr;
	m_uiPrimaryOffset =		m_pStack->uiCurOffset;
	m_ui32CurBlkAddr =		m_ui32PrimaryBlkAddr;
	m_uiCurOffset =			m_uiPrimaryOffset;

	// Point to the entry...
	pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
							  m_pStack->uiCurOffset);

	// Return the optional data length - get the overall data length only.
	if (puiDataLength &&
		m_pStack->pSCache->m_pBlkHdr->ui8BlkType == BT_LEAF_DATA)
	{
		btGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}
	else if (puiDataLength)
	{
		*puiDataLength = 0;
	}

	if( RC_BAD( rc = setupReadState( m_pStack->pSCache->m_pBlkHdr, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such as
	// in the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL, 
	// we will pass back the key we actually found.

	if( uiMatch != FLM_EXACT)
	{
		if( RC_BAD( rc = setReturnKey( pucEntry, 
			m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, pucKey, puiKeyLen,
			uiKeyBufSize)))
		{
			goto Exit;
		}
	}

	m_bFirstRead =		FALSE;
	m_bSetupForRead = TRUE;

Exit:

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Method to get the data after a call to btLocateEntry, btNextEntry,
		btPrevEntry, btFirstEntry or btLastEntry.
****************************************************************************/
RCODE F_Btree::btGetEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT					uiKeyLen,
	FLMBYTE *				pucData,
	FLMUINT					uiDataBufSize,
	FLMUINT *				puiDataLen
	)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucEntry;

	if( !m_bOpened || !m_bSetupForRead || 
		 m_bSetupForWrite || m_bSetupForReplace)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	if( puiDataLen)
	{
		*puiDataLen = 0;
	}

	// Is there anything there to get?
	
	if( m_uiOADataRemaining == 0)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}

	// If the transaction Id or the Block Change Count has changed,
	// we must re-sync ourselves.

	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		// Test to see if we really need to re-sync ...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// We must call btLocateEntry so we can re-initialize the read
			
			if( !m_bFirstRead)
			{
				if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize,
					&uiKeyLen, FLM_EXACT)))
				{
					goto Exit;
				}

				// Will need a new version of this block.
				
				ScaReleaseCache( m_pSCache, FALSE);
				m_pSCache = NULL;
			}
			else
			{
				rc = RC_SET(NE_SFLM_BTREE_BAD_STATE);
				goto Exit;
			}
		}
	}

	// Get the current block.  It is either a DO or a Btree block.
	
	if( m_pSCache == NULL)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}
	}

	updateTransInfo( m_pSCache->m_pBlkHdr->ui64TransID,
						  m_pSCache->m_ui64HighTransID);

	// Now to find where we were the last time through.
	
	if( !m_bDataOnlyBlock)
	{
		pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr,
								  m_uiCurOffset);

		btGetEntryDataLength( pucEntry, &m_pucDataPtr, NULL, NULL);
	}
	else
	{
		m_pucDataPtr = (FLMBYTE *)m_pSCache->m_pBlkHdr +
							sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr);

		// May need to skip over the key that is stored in the first DO block.
		// We only want to do this the first time in here.  The test to determine
		// if this is our first time in this block is to see if the m_uiDataLength
		// is equal to the m_uiDataRemaining.  They would only be the same on the
		// first time for each DO block.
		
		if( m_pSCache && m_pSCache->m_pBlkHdr->ui32PrevBlkInChain == 0)
		{
			FLMUINT16	ui16KeyLen = FB2UW( m_pucDataPtr);

			// Key lengths should be the same
			
			flmAssert( uiKeyLen == (FLMUINT)ui16KeyLen);

			m_pucDataPtr += (ui16KeyLen + 2);
		}
	}

	m_pucDataPtr += (m_uiDataLength - m_uiDataRemaining);
	
	if( RC_BAD( rc = extractEntryData( pucKey, uiKeyLen, pucData,
		uiDataBufSize, puiDataLen)))
	{
		goto Exit;
	}

	// Mark that we have completed our first read operation.
	// No more read synchronization allowed.
	
	m_bFirstRead = TRUE;

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Function to locate the next entry in the Btree.  The key buffer and
		actual size is passed in.
****************************************************************************/
RCODE F_Btree::btNextEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlkAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucEntry = NULL;
	FLMBOOL			bAdvanced = FALSE;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify the transaction type
	
	if( m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure we are looking at btree  block.  If the m_bDataOnlyBlock
	// flag is set, then the block address in m_ui32CurBlkAddr is a
	// data only block.  We must reset it to the primary block address.
	
	if( m_bDataOnlyBlock)
	{
		m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
	}
	else
	{
		// If the entry did not reference a DO block, then we need to
		// reset the primary block and offset with where we currently
		// are incase the current block is further ahead.  This saves time
		// so that we don't have to scan past old blocks we are not intereseted
		// in.
		
		m_ui32PrimaryBlkAddr = m_ui32CurBlkAddr;
		m_uiPrimaryOffset = m_uiCurOffset;
		m_ui64PrimaryBlkTransId = m_ui64LastBlkTransId;
	}

	// Do we need to resynchronize?
	
	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		// Test to see if we really need to re-sync ...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// Will need a new version of this block.
			
			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;

			// Doing a find with FLM_EXCL will result in our being positioned at
			// the next entry.
			
			if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
				FLM_EXCL, puiDataLength)))
			{
				goto Exit;
			}
			
			bAdvanced = TRUE;
		}
	}

	// Get the current block if we need it.
	
	if( !m_pSCache)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}
	}

	// If we have already advanced due to a resynch, then we don't need to call
	// the advanceToNextElement function, however, we do need to get the
	// current entry.
	
	if( bAdvanced)
	{
		pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr, m_uiCurOffset);
	}
	else
	{
		for (;;)
		{
			// Advance to the next entry in the block.  We don't have a stack so
			// don't advance it.
			
			if( RC_BAD( rc = advanceToNextElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr, m_uiCurOffset);

			if( m_bData)
			{
				if( bteFirstElementFlag(pucEntry))
				{
					break;
				}
			}
			else
			{
				break;
			}
		}
	}

	// Return the optional data length - get the overall data length only.
	
	if( puiDataLength)
	{
		btGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}

	if( RC_BAD( rc = setupReadState( m_pSCache->m_pBlkHdr, pucEntry)))
	{
		goto Exit;
	}

	// Incase the returning key is not what was originally requested, such as in
	// the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL,
	// we will pass back the key we actually found.
	
	if( RC_BAD( rc = setReturnKey( pucEntry, m_pSCache->m_pBlkHdr->ui8BlkType,
		pucKey, puiKeyLen, uiKeyBufSize)))
	{
		goto Exit;
	}

	if( pui32BlkAddr)
	{
		*pui32BlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_uiCurOffset;
	}

	m_bFirstRead = FALSE;

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Function to get the previous entry in the Btree.
****************************************************************************/
RCODE F_Btree::btPrevEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlkAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucEntry = NULL;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify the transaction type
	
	if( m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// Make sure we are looking at the first block of the
	// current entry.  Reading of the entry could have moved us
	// to another block, or if it was in a DO block, we would be
	// looking at the wrong block altogether.
	
	m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
	m_uiCurOffset = m_uiPrimaryOffset;
	m_ui64LastBlkTransId = m_ui64PrimaryBlkTransId;

	// Do we need to resynchronize?
	
	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		// Test to see if we really need to re-sync ...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// Will need a new version of this block.
			
			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;

			// Doing a find with FLM_INCL will allow for the possibility that
			// the original entry is no longer there.  We will still have
			// to backup to the previous entry.
			
			if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
				FLM_INCL, puiDataLength)))
			{
				goto Exit;
			}
		}
	}

	if( !m_pSCache)
	{
		// Fetch the current block, then backup from there.
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}
	}

	for (;;)
	{
		// Backup to the previous entry in the block.
		
		if( RC_BAD( rc = backupToPrevElement( FALSE)))
		{
			goto Exit;
		}

		// Get the entry, size etc.
		
		pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr, m_uiCurOffset);

		if( m_bData)
		{
			if( bteFirstElementFlag( pucEntry))
			{
				break;
			}
		}
		else
		{
			break;
		}
	}

	// Return the optional data length - get the overall data length only.
	
	if( puiDataLength)
	{
		btGetEntryDataLength( pucEntry, NULL, puiDataLength, NULL);
	}

	if( RC_BAD( rc = setupReadState( m_pSCache->m_pBlkHdr, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such as in
	// the case of FLM_FIRST, FLM_LAST, FLM_EXCL and possibly FLM_INCL,
	// we will pass back the key we actually found.

	if( RC_BAD( rc = setReturnKey( pucEntry, m_pSCache->m_pBlkHdr->ui8BlkType,
		pucKey, puiKeyLen, uiKeyBufSize)))
	{
		goto Exit;
	}

	if( pui32BlkAddr)
	{
		*pui32BlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_uiCurOffset;
	}

	m_bFirstRead = FALSE;

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Locate the first entry in the Btree and return the key.
****************************************************************************/
RCODE F_Btree::btFirstEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlkAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE					rc = NE_SFLM_OK;

	m_Stack[ 0].pucKeyBuf = pucKey;

	if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
		FLM_FIRST, NULL, puiDataLength, pui32BlkAddr, puiOffsetIndex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Locate the last entry in the Btree and return the key.
****************************************************************************/
RCODE F_Btree::btLastEntry(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiDataLength,
	FLMUINT32 *				pui32BlkAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE				rc = NE_SFLM_OK;

	m_Stack[ 0].pucKeyBuf = pucKey;

	if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
		FLM_LAST, NULL, puiDataLength, pui32BlkAddr, puiOffsetIndex)))
	{
		if( rc == NE_SFLM_BOF_HIT)
		{
			rc = RC_SET( NE_SFLM_EOF_HIT);
		}
		
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: Function to search the Btree for a specific key.
****************************************************************************/
RCODE F_Btree::btPositionTo(
	FLMUINT					uiPosition,
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyBufSize,
	FLMUINT *				puiKeyLen)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucEntry = NULL;

	flmAssert( pucKey && uiKeyBufSize && puiKeyLen);

	m_bSetupForRead = FALSE;

	if( !m_bOpened || !m_bCounts)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify the transaction type
	
	if( m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// Find the entry we are interested in.
	
	if( RC_BAD(rc = positionToEntry( uiPosition)))
	{
		goto Exit;
	}

	m_ui64LowTransId = m_pStack->pBlkHdr->stdBlkHdr.ui64TransID;
	m_bMostCurrent = (m_pStack->pSCache->m_ui64HighTransID == FLM_MAX_UINT64)
																? TRUE
																: FALSE;

	m_ui32PrimaryBlkAddr = m_pStack->ui32BlkAddr;
	m_uiPrimaryOffset = m_pStack->uiCurOffset;
	m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
	m_uiCurOffset = m_uiPrimaryOffset;

	// Point to the entry ...
	
	pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

	if( RC_BAD( rc = setupReadState( m_pStack->pSCache->m_pBlkHdr, pucEntry)))
	{
		goto Exit;
	}

	// In case the returning key is not what was originally requested, such
	// as in the case of FLM_FIRST, FLM_LAST, FLM_EXCL and 
	// possibly FLM_INCL, we will pass back the key we actually found.
	
	if( RC_BAD( rc = setReturnKey( pucEntry,
		m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, pucKey, puiKeyLen,
		uiKeyBufSize)))
	{
		goto Exit;
	}

	m_bFirstRead = FALSE;
	m_bSetupForRead = TRUE;

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method to get the actual poisition of the entry.  Note: Must be
		maintaining counts in the Btree AND also have located to an entry
		first.  The key that is passed in is used only if we have to
		resynchronize due to a transaction change.
****************************************************************************/
RCODE F_Btree::btGetPosition(
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT *		puiPosition)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiKeyBufSize = uiKeyLen;
	FLMUINT			uiLocalKeyLen = uiKeyLen;

	if( !m_bOpened || !m_bSetupForRead || !m_bCounts)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Verify the transaction type
	
	if( m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	*puiPosition = 0;

	m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
	m_uiCurOffset = m_uiPrimaryOffset;
	m_ui64LastBlkTransId = m_ui64PrimaryBlkTransId;

	// Do we need to resynchronize?
	
	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		// Test to see if we really need to re-sync ...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// We can get the position easily if we have to re-sync.
			
			if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, &uiLocalKeyLen,
				FLM_EXACT, puiPosition)))
			{
				goto Exit;
			}

			// Will need a new version of this block.
			
			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;
		}
	}
	else
	{
		// To calculate the position, we will have to reconstruct the stack.
		
		m_pStack = &m_Stack[ m_uiStackLevels - 1];
		for (;;)
		{
			// Get the block at this level.
			
			flmAssert( m_pStack->ui32BlkAddr);
			flmAssert( m_pStack->pSCache == NULL);

			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_pStack->ui32BlkAddr, NULL, &m_pStack->pSCache)))
			{
				goto Exit;
			}

			m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;

			*puiPosition += countRangeOfKeys( m_pStack, 0, m_pStack->uiCurOffset);

			if( (getBlkType( (FLMBYTE *)m_pStack->pBlkHdr) == BT_LEAF) ||
				 (getBlkType( (FLMBYTE *)m_pStack->pBlkHdr) == BT_LEAF_DATA))
			{
				break;
			}
			else
			{
				// Next level down. (stack is inverted).
				
				m_pStack--;
			}
		}
	}

Exit:

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method to rewind back to the beginning of the current entry.
****************************************************************************/
RCODE F_Btree::btRewind(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyBufSize,
	FLMUINT *			puiKeyLen)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache = NULL;

	if( !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
	m_uiCurOffset = m_uiPrimaryOffset;
	m_ui64LastBlkTransId = m_ui64PrimaryBlkTransId;

	// Do we need to resync?
	
	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		flmAssert( m_pSCache == NULL);

		// Test to see if we really need to re-sync ...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
				  m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// Won't need this block anymore.
			
			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;

			if( RC_BAD( rc = btLocateEntry( pucKey, uiKeyBufSize, puiKeyLen,
													  FLM_EXACT)))
			{
				goto Exit;
			}
			
			goto Exit;
		}
	}

	m_uiOADataRemaining = m_uiOADataLength;	// Track the overall length progress
	m_uiDataLength = m_uiPrimaryDataLen;		// Restore the primary block data length
	m_uiDataRemaining = m_uiDataLength;			// Track the local entry progress

	if( m_bDataOnlyBlock)
	{
		m_ui32CurBlkAddr = m_ui32DOBlkAddr;

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32DOBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		m_ui64LastBlkTransId = pSCache->m_pBlkHdr->ui64TransID;

		// Local amount of data in this block
		
		m_uiDataRemaining = m_uiBlockSize -
										sizeofDOBlkHdr((F_BLK_HDR *)pSCache->m_pBlkHdr) -
										pSCache->m_pBlkHdr->ui16BlkBytesAvail;

		// Keep the actual local data size for later.
		
		m_uiDataLength = m_uiDataRemaining;

		// Now release the DO Block.  We will get it again when we need it.
		
		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

	m_bFirstRead = FALSE;
	m_bSetupForRead = TRUE;

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Method for computing the number of keys and blocks between two points 
		in the Btree.  The key count is inclusive of the two end points and
		the block count is exclusive of the two end points.
****************************************************************************/
RCODE F_Btree::btComputeCounts(
	F_Btree *		pUntilBtree,
	FLMUINT64 *		pui64BlkCount,
	FLMUINT64 *		pui64KeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlkFullness)
{
	RCODE			rc = NE_SFLM_OK;

	if( !m_bSetupForRead || !pUntilBtree->m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// Ensure that both Btrees are from the same container.
	
	if( m_pLFile->uiRootBlk != pUntilBtree->m_pLFile->uiRootBlk)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
		goto Exit;
	}

	rc = computeCounts( m_pStack, pUntilBtree->m_pStack, pui64BlkCount,
			  pui64KeyCount, pbTotalsEstimated, uiAvgBlkFullness);

Exit:

	releaseBlocks( FALSE);
	pUntilBtree->releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:	Function to release the blocks in the stack, and optionally, reset
		the stack
****************************************************************************/
void F_Btree::releaseBlocks(
	FLMBOOL			bResetStack)
{
	FLMUINT		uiLevel;

	// Release any blocks still held in the stack.
	
	for( uiLevel = 0; uiLevel <= m_uiRootLevel; uiLevel++)
	{
		if( m_Stack[ uiLevel].pSCache)
		{
			if( m_Stack[ uiLevel].pSCache->m_uiUseCount)
			{
				ScaReleaseCache( m_Stack[ uiLevel].pSCache, FALSE);
			}
			
			m_Stack[ uiLevel].pSCache = NULL;
			m_Stack[ uiLevel].pBlkHdr = NULL;
		}
		
		if( bResetStack)
		{
			m_Stack[ uiLevel].ui32BlkAddr = 0;
			m_Stack[ uiLevel].uiKeyLen = 0;
			m_Stack[ uiLevel].uiCurOffset = 0;
			m_Stack[ uiLevel].uiLevel = 0;
		}
	}
	
	if( bResetStack)
	{
		m_uiStackLevels = 0;
		m_uiRootLevel = 0;
		m_bStackSetup = FALSE;
		m_pStack = NULL;
	}
}

/***************************************************************************
Desc: Function to create a new block at the current level.  The new block 
		will always be inserted previous to the current block.  All entries
		that sort ahead of the current insertion point will be moved into
		the new block.  If there is room, the new entry will be inserted
		into the current block.  Otherwise, if there is room, the new entry
		will be inserted into the new block.  If there is still not enough
		room, then if possible, it try to store a partial entry in the new
		block. If we still cannot store anything, we will see if we can 
		store a partial entry in the current block.  If that does not work,
		then it will set the remaining amount and return.  Another block
		split will be needed before we store this entry.
****************************************************************************/
RCODE F_Btree::splitBlock(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT					uiChildBlkAddr,
	FLMUINT 					uiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	FLMBOOL *				pbBlockSplit)
{
	RCODE						rc = NE_SFLM_OK;
	F_CachedBlock *		pNewSCache = NULL;
	F_CachedBlock *		pPrevSCache = NULL;
	F_BTREE_BLK_HDR *		pBlkHdr = NULL;
	FLMUINT					uiBlkAddr;
	FLMUINT					uiEntrySize;
	FLMBOOL					bHaveRoom;
	FLMBOOL					bMovedToPrev = FALSE;
	FLMBOOL					bLastEntry;
	FLMUINT					uiMinEntrySize;
	FLMBOOL					bDefragBlk = FALSE;
	FLMBOOL					bSavedReplaceInfo = FALSE;

	// If the current block is a root block, then we will have to introduce
	// a new level into the B-Tree.
	
	if( isRootBlk( m_pStack->pBlkHdr))
	{
		if( RC_BAD( rc = createNewLevel()))
		{
			goto Exit;
		}
	}

	// If the current block is empty we must insert what we can here.
	// This scenario only occurs when we are engaged in a ReplaceByInsert
	// operation. Normal inserts would never result in an empty block.
	// Since we know we are part of a replace operation, we know that the
	// parent of this block only needs the counts updated, not the key.
	
	if( m_pStack->uiLevel == 0 && m_pStack->pBlkHdr->ui16NumKeys == 0)
	{
		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue,
				uiLen, uiFlags, uiChildBlkAddr, uiCounts, ppucRemainingValue,
				puiRemainingLen, FALSE)))
		{
			goto Exit;
		}
		
		*pbBlockSplit = FALSE;
		goto MoveToPrev;
	}

	// Create a new block and insert it as previous to this block.
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock( m_pDb, &pNewSCache)))
	{
		goto Exit;
	}

	*pbBlockSplit = TRUE;

	// Setup the header ...
	
	pBlkHdr = (F_BTREE_BLK_HDR *)pNewSCache->m_pBlkHdr;

	unsetRootBlk( pBlkHdr);
	setBlkLfType( pBlkHdr, m_pLFile->eLfType);

	pBlkHdr->ui16NumKeys = 0;
	pBlkHdr->ui8BlkLevel = (FLMUINT8)m_pStack->uiLevel;
	pBlkHdr->stdBlkHdr.ui8BlkType = m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType;
	pBlkHdr->ui16LogicalFile = m_pStack->pBlkHdr->ui16LogicalFile;
	
	// Check for encrypted block.
	
	if( isEncryptedBlk( (F_BLK_HDR *)m_pStack->pBlkHdr))
	{
		setBlockEncrypted( (F_BLK_HDR *)pBlkHdr);
	}

	pBlkHdr->stdBlkHdr.ui16BlkBytesAvail =
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlkHdr(pBlkHdr));

	pBlkHdr->ui16HeapSize =
		(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlkHdr(pBlkHdr));

	// We are going to make changes to the current block.  The pSCache could
	// have changed since making this call, so we need to update the block
	// header
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk(
		m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}

	m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
	m_pStack->pui16OffsetArray =
							  BtOffsetArray( (FLMBYTE *)m_pStack->pBlkHdr, 0);

	// Get the current previous block if there is one.
	
	uiBlkAddr = m_pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain;

	if( uiBlkAddr)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			uiBlkAddr, NULL, &pPrevSCache)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pPrevSCache)))
		{
			goto Exit;
		}
	}

	// Link the new block between the current and it's previous
	
	pBlkHdr->stdBlkHdr.ui32NextBlkInChain = m_pStack->ui32BlkAddr;
	pBlkHdr->stdBlkHdr.ui32PrevBlkInChain = (FLMUINT32)uiBlkAddr;

	m_pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain =
											pBlkHdr->stdBlkHdr.ui32BlkAddr;

	// There may not be a previous block.
	
	if( pPrevSCache)
	{
		pPrevSCache->m_pBlkHdr->ui32NextBlkInChain =
											pBlkHdr->stdBlkHdr.ui32BlkAddr;

		// Release the old previous block since we no longer need it.
		
		ScaReleaseCache( pPrevSCache, FALSE);
		pPrevSCache = NULL;
	}

	// We will move all entries in the current block up to but NOT including
	// the entry pointed to by uiCurOffset to the new block.
	
	if( m_pStack->uiCurOffset > 0)
	{
		if( RC_BAD( rc = moveToPrev( 0, m_pStack->uiCurOffset - 1, &pNewSCache)))
		{
			goto Exit;
		}
		
		// All entries prior to the old insertion point were moved.
		// Therefore, the new insertion point must be at the beginning.
		
		m_pStack->uiCurOffset = 0;

		// If we emptied the block.  This will require us to update the parent.

		if( m_pStack->pBlkHdr->ui16NumKeys == 0)
		{
			if (RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
			
			bSavedReplaceInfo = TRUE;
		}
	}

	// If the block is now empty, we will store a partial entry in it here.
	// This scenario only occurs when we are engaged in a ReplaceByInsert
	// operation. Normal inserts would never result in an empty block.
	// Since we know we are part of a replace operation, we know that the
	// parent of this block only needs the counts updated, not the key.
	
	if( m_pStack->uiLevel == 0 && m_pStack->pBlkHdr->ui16NumKeys == 0)
	{
		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue, uiLen,
			uiFlags, uiChildBlkAddr, uiCounts, ppucRemainingValue, 
			puiRemainingLen, FALSE)))
		{
			goto Exit;
		}

		goto MoveToPrev;
	}

	// Is there room for the new entry now in the current block?
	
	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiLen, &uiEntrySize,
		&bHaveRoom, &bDefragBlk)))
	{
		goto Exit;
	}

	if( bHaveRoom)
	{
		if( bDefragBlk)
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiLen,
				uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts, uiEntrySize,
				&bLastEntry)))
		{
			goto Exit;
		}

		if( bLastEntry && !bSavedReplaceInfo)
		{
			// Since we just added/replaced an entry to the last position of the
			// current block. we will need to preserve the current stack so that
			// we can finish updating the parentage later. Should only happen as
			// a result of a replace operation where the new entry is larger than
			// the existing one while in the upper levels.
			
			if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
		}

		// If we are keeping counts, we must update those too.
		
		if( m_bCounts && !isRootBlk( m_pStack->pBlkHdr))
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		if( m_pStack->uiLevel == 0)
		{
			*ppucRemainingValue = NULL;
			*puiRemainingLen = 0;
		}

		goto MoveToPrev;
	}

	// Can we store the whole thing in the new block?
	
	if( uiEntrySize <= pBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
	{
		// If this block has a parent block, and the btree is maintaining counts
		// we will want to update the counts on the parent block.
		
		if( m_bCounts && !isRootBlk( m_pStack->pBlkHdr))
		{
			if (RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		// We can release the current block since it is no longer needed.
		
		ScaReleaseCache( m_pStack->pSCache, FALSE);

		m_pStack->pSCache = pNewSCache;
		pNewSCache = NULL;
		m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pBlkHdr;
		m_pStack->ui32BlkAddr = pBlkHdr->stdBlkHdr.ui32BlkAddr;
		m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

		// Setting the uiCurOffset to the actual number of keys will cause the
		// new entry to go in as the last element.
		
		m_pStack->uiCurOffset = m_pStack->pBlkHdr->ui16NumKeys;

		// We don't need to check to see if we need to defragment this block
		// because it is "new".  Anything that just got written to it will
		// be contiguous already.

		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiLen,
			uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts, uiEntrySize,
			&bLastEntry)))
		{
			goto Exit;
		}

		flmAssert( bLastEntry);

		if( m_pStack->uiLevel == 0)
		{
			*ppucRemainingValue = NULL;
			*puiRemainingLen = 0;
		}

		bMovedToPrev = TRUE;
		goto MoveToPrev;
	}

	// Can we store part of the new entry into the new block?
	// Calculate the minimum entry size to store.
	
	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, 1, &uiMinEntrySize,
		&bHaveRoom, &bDefragBlk)))
	{
		goto Exit;
	}

	// bHaveRoom refers to the current block, and we want to put this into
	// the previous  block.
	
	if( uiMinEntrySize <= pBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
	{
		// If this block has a parent block, and the btree is maintaining counts
		// we will want to update the counts on the parent block.
		
		if( !isRootBlk( m_pStack->pBlkHdr))
		{
			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}
		}

		// We can release the current block since it is no longer needed.
		
		ScaReleaseCache( m_pStack->pSCache, FALSE);

		m_pStack->pSCache = pNewSCache;
		pNewSCache = NULL;
		m_pStack->pBlkHdr = pBlkHdr;
		m_pStack->ui32BlkAddr = pBlkHdr->stdBlkHdr.ui32BlkAddr;
		m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

		// Setting the uiCurOffset to the actual number of keys will cause the
		// new entry to go in as the last element.
		
		m_pStack->uiCurOffset = m_pStack->pBlkHdr->ui16NumKeys;

		if( RC_BAD( rc = storePartialEntry( pucKey, uiKeyLen, pucValue,
			uiLen, uiFlags, uiChildBlkAddr, uiCounts, ppucRemainingValue,
			puiRemainingLen, TRUE)))
		{
			goto Exit;
		}

		bMovedToPrev = TRUE;
	}
	else if( uiMinEntrySize <= m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
	{
		// We will store part of the entry in the current block
		
		if( RC_BAD( rc = storePartialEntry(
			pucKey, uiKeyLen, pucValue, uiLen, uiFlags, uiChildBlkAddr, uiCounts,
			ppucRemainingValue, puiRemainingLen, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		// Couldn't store anything, so try again after updating the parents.
		
		*ppucRemainingValue = pucValue;
		*puiRemainingLen = uiLen;
	}

MoveToPrev:

	if( *pbBlockSplit)
	{
		// Release the current entry if it hasn't already been released.
		
		if( !bMovedToPrev && RC_OK( rc))
		{

			// If this block has a parent block, and the btree is maintaining counts
			// we will want to update the counts on the parent block.
			
			if( !isRootBlk( m_pStack->pBlkHdr) && m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}

			ScaReleaseCache( m_pStack->pSCache, FALSE);

			flmAssert( pNewSCache);

			m_pStack->pSCache = pNewSCache;
			pNewSCache = NULL;
			m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pBlkHdr;
			m_pStack->ui32BlkAddr = pBlkHdr->stdBlkHdr.ui32BlkAddr;
			m_pStack->uiCurOffset = m_pStack->pBlkHdr->ui16NumKeys - 1;
			m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);
		}
	}

Exit:

	if( *pbBlockSplit)
	{
		if( m_pDb->m_pDbStats)
		{
			SFLM_LFILE_STATS *		pLFileStats;

			if( (pLFileStats = m_pDb->getLFileStatPtr( m_pLFile)) != NULL)
			{
				pLFileStats->bHaveStats = TRUE;
				pLFileStats->ui64BlockSplits++;
			}
		}
	}

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	if( pNewSCache)
	{
		ScaReleaseCache( pNewSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: Function to create a new level in the Btree.
		This function will ensure that the F_BTSK stack is consistent with
		the way it was configured before the function was called.

		This function will create a new block and copy the current contents
		of the root block into it.  It will then insert a single entry into
		the root block to point to the new child.

		Note that there is a maximum of BH_MAX_LEVELS levels to the Btree.
		Any effort to exceed that level will result in an error.
****************************************************************************/
RCODE F_Btree::createNewLevel( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pNewSCache = NULL;
	FLMBYTE *			pSrcBlk;
	FLMBYTE *			pDstBlk;
	F_BTREE_BLK_HDR *	pBlkHdr;
	FLMUINT				uiCounts = 0;
	FLMBYTE *			pucEntry;
	FLMBYTE *			pucNull = NULL;
	FLMBYTE				ucBuffer[ SFLM_MAX_KEY_SIZE + BTE_NLC_KEY_START];
	FLMUINT				uiMaxNLKey = SFLM_MAX_KEY_SIZE + BTE_NLC_KEY_START;
	FLMUINT				uiEntrySize;
	F_BTSK *				pRootStack;
	FLMUINT				uiFlags;

	// Assert that we are looking at the root block!
	
	flmAssert( isRootBlk( m_pStack->pBlkHdr));

	// Check the root level
	
	if( m_pStack->uiLevel >= BH_MAX_LEVELS - 1)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_FULL);
		goto Exit;
	}

	// Create a new block to copy the contents of the root block into
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock( m_pDb, &pNewSCache)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	// Log that we are about to change the root block
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}

	// Update the stack since the pSCache could have changed
	
	m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)m_pStack->pBlkHdr, 0);

	// Copy the data from the root block to the new block
	
	pSrcBlk = (FLMBYTE *)m_pStack->pui16OffsetArray;
	pBlkHdr = (F_BTREE_BLK_HDR *)pNewSCache->m_pBlkHdr;

	// Check for encryption

	if( isEncryptedBlk( (F_BLK_HDR *)m_pStack->pBlkHdr))
	{
		setBlockEncrypted( (F_BLK_HDR *)pBlkHdr);
	}

	pDstBlk = (FLMBYTE *)BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

	unsetRootBlk( pBlkHdr);
	setBlkLfType( pBlkHdr, m_pLFile->eLfType);

	pBlkHdr->ui16LogicalFile = (FLMUINT16)m_pLFile->uiLfNum;
	pBlkHdr->ui16NumKeys = m_pStack->pBlkHdr->ui16NumKeys;
	pBlkHdr->ui8BlkLevel = m_pStack->pBlkHdr->ui8BlkLevel;
	pBlkHdr->ui16HeapSize = m_pStack->pBlkHdr->ui16HeapSize;

	pBlkHdr->stdBlkHdr.ui8BlkType =
		((F_BLK_HDR *)m_pStack->pBlkHdr)->ui8BlkType;

	pBlkHdr->stdBlkHdr.ui16BlkBytesAvail =
		((F_BLK_HDR *)m_pStack->pBlkHdr)->ui16BlkBytesAvail;

	pBlkHdr->stdBlkHdr.ui32PrevBlkInChain = 0;
	pBlkHdr->stdBlkHdr.ui32NextBlkInChain = 0;

	// Copy the data from the root block to the new block.
	
	f_memcpy( pDstBlk, pSrcBlk, m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr));

	// Empty out the root block data.

#ifdef FLM_DEBUG
	f_memset( BtOffsetArray( (FLMBYTE *)m_pStack->pBlkHdr, 0),
				 0, m_uiBlockSize - sizeofBTreeBlkHdr( m_pStack->pBlkHdr));
#endif

	m_pStack->pBlkHdr->ui16NumKeys = 0;
	m_pStack->pBlkHdr->ui16HeapSize = 
		((F_BLK_HDR *)m_pStack->pBlkHdr)->ui16BlkBytesAvail =
			(FLMUINT16)(m_uiBlockSize - sizeofBTreeBlkHdr( m_pStack->pBlkHdr));

	// Check the root block type to see if we need to change it.  The root
	// block may have been a leaf node.
	
	if( (m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF) ||
		 (m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA))
	{
		// Need to set the block type to either 
		// BT_NON_LEAF or BT_NON_LEAF_COUNTS
		
		if( m_bCounts)
		{
			m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType = BT_NON_LEAF_COUNTS;
		}
		else
		{
			m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType = BT_NON_LEAF;
		}
	}

	// Now add a new entry to the stack.
	
	pRootStack = m_pStack;
	pRootStack++;

	f_memcpy( pRootStack, m_pStack, sizeof( F_BTSK));

	// Now fix the entries in the stack.
	
	pRootStack->uiLevel++;
	pRootStack->pBlkHdr->ui8BlkLevel++;
	pRootStack->uiCurOffset = 0;  // First entry
	pRootStack->pui16OffsetArray = BtOffsetArray( 
												(FLMBYTE *)pRootStack->pBlkHdr, 0);

	m_pStack->pBlkHdr = pBlkHdr;  // Point to new block
	m_pStack->ui32BlkAddr = (FLMUINT32)pNewSCache->m_uiBlkAddress;
	m_pStack->pSCache = pNewSCache;
	pNewSCache = NULL;
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

	// Build a new entry for the root block that will point to the newly created
	// child block. If the root block type is BT_NON_LEAF_COUNTS, then we
	// need to sum the counts from the child block
	
	if( m_bCounts)
	{
		uiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
	}

	// Create and insert a LEM entry to mark the last position in the block.
	
	uiFlags = BTE_FLAG_LAST_ELEMENT | BTE_FLAG_FIRST_ELEMENT;

	if( RC_BAD( rc = buildAndStoreEntry( 
		((F_BLK_HDR *)pRootStack->pBlkHdr)->ui8BlkType,
		uiFlags, pucNull, 0, pucNull, 0, 0, m_pStack->ui32BlkAddr,
		uiCounts, &ucBuffer[ 0], uiMaxNLKey, &uiEntrySize)))
	{
		goto Exit;
	}

	// Copy the entry into the root block.
	
	pucEntry = (FLMBYTE *)pRootStack->pBlkHdr + m_uiBlockSize - uiEntrySize;
	f_memcpy( pucEntry, &ucBuffer[ 0], uiEntrySize);
	bteSetEntryOffset( pRootStack->pui16OffsetArray, 0,
							 (FLMUINT16)(pucEntry - (FLMBYTE *)pRootStack->pBlkHdr));

	pRootStack->pBlkHdr->ui16NumKeys++;

	pRootStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail -= (FLMUINT16)uiEntrySize + 2;

	pRootStack->pBlkHdr->ui16HeapSize -= (FLMUINT16)uiEntrySize + 2;

	m_uiStackLevels++;
	m_uiRootLevel++;

Exit:

	if( pNewSCache)
	{
		ScaReleaseCache( pNewSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to calculate the optimal data length size to store.  This 
		method is called when storing a partial entry, and we need to know
		what the largest data size we c an store is.
****************************************************************************/
RCODE F_Btree::calcOptimalDataLength(
	FLMUINT			uiKeyLen,
	FLMUINT			uiDataLen,
	FLMUINT			uiBytesAvail,
	FLMUINT *		puiNewDataLen)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiFixedAmounts;
	FLMUINT			uiRemainder;

	switch( ((F_BLK_HDR *)m_pStack->pBlkHdr)->ui8BlkType)
	{
		case BT_LEAF:
		case BT_NON_LEAF:
		case BT_NON_LEAF_COUNTS:
		{
			// These blocks do not have any data.
			
			*puiNewDataLen = 0;
			break;
		}

		case BT_LEAF_DATA:
		{
			// These amounts don't change. Note that the overhead includes the
			// Overall Data Length Field, even though it may not be there in
			// the end.
			
			uiFixedAmounts = BTE_LEAF_DATA_OVHD +
								  (uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +
								  uiKeyLen;

			uiRemainder = uiBytesAvail - uiFixedAmounts;

			if (uiRemainder >= (ONE_BYTE_SIZE + 2))
			{
				*puiNewDataLen = uiRemainder - 2;
			}
			else
			{
				*puiNewDataLen = uiRemainder - 1;
			}
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
	}

	if( uiDataLen < *puiNewDataLen)
	{
		*puiNewDataLen = uiDataLen;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This function will count the total number of keys in the block. 
		Typically the value ui16NumKeys will yield this number, however, if
		the block type is BT_NON_LEAF_COUNTS, we also want to include the 
		counts in each entry.
****************************************************************************/
RCODE F_Btree::updateParentCounts(
	F_CachedBlock *	pChildSCache,
	F_CachedBlock **	ppParentSCache,
	FLMUINT				uiParentElm)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiCounts;
	FLMBYTE *			pucCounts;
	F_CachedBlock *	pParentSCache;
	FLMBYTE *			pBlk = (FLMBYTE *)pChildSCache->m_pBlkHdr; 

	flmAssert( getBlkType( pBlk) == BT_NON_LEAF_COUNTS);
	uiCounts = countKeys( pBlk);

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, ppParentSCache)))
	{
		goto Exit;
	}

	pParentSCache = *ppParentSCache;
	pucCounts = BtEntry( (FLMBYTE *)pParentSCache->m_pBlkHdr, uiParentElm);
	pucCounts += 4;
	UD2FBA( (FLMUINT32)uiCounts, pucCounts);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This function will count the total number of keys in the block.
		Typically the value ui16NumKeys will yield this number, however, if 
		the block type is BT_NON_LEAF_COUNTS, we also want to include the 
		counts in each entry.
****************************************************************************/
FLMUINT F_Btree::countKeys(
	FLMBYTE *			pBlk)
{
	FLMUINT				uiTotal = 0;
	FLMUINT				uiIndex;
	FLMBYTE *			pucEntry;
	FLMUINT16 *			puiOffsetArray;

	puiOffsetArray = BtOffsetArray( pBlk, 0);

	if( getBlkType(pBlk) != BT_NON_LEAF_COUNTS)
	{
		uiTotal = ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys;
	}
	else
	{
		for (uiIndex = 0; uiIndex <
			((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys; uiIndex++)
		{
			pucEntry = BtEntry( pBlk, uiIndex);
			uiTotal += FB2UD( &pucEntry[ BTE_NLC_COUNTS]);
		}
	}

	return( uiTotal);
}

/***************************************************************************
Desc: Function to store an entry in a Data-only block.
****************************************************************************/
RCODE F_Btree::storeDataOnlyBlocks(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bSaveKey,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pPrevSCache = NULL;
	const FLMBYTE *	pucLocalData = pucData;
	FLMUINT				uiDataToWrite = uiDataLen;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMBYTE *			pDestPtr = NULL;
	FLMUINT				uiAmtToCopy;

	if( bSaveKey)
	{
		if( !m_pSCache)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_ui32CurBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pSCache)))
		{
			goto Exit;
		}

		// Assert that the current block is empty and has no previous link.
		
		flmAssert( m_pSCache->m_pBlkHdr->ui16BlkBytesAvail ==
			m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr));

		flmAssert( m_pSCache->m_pBlkHdr->ui32PrevBlkInChain == 0);

		pBlkHdr =  m_pSCache->m_pBlkHdr;
		pDestPtr = (FLMBYTE *)pBlkHdr + 
							sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr);

		UW2FBA( (FLMUINT16)uiKeyLen, pDestPtr);
		pDestPtr += sizeof( FLMUINT16);

		f_memcpy( pDestPtr, pucKey, uiKeyLen);
		pDestPtr += uiKeyLen;
		
		m_uiDataRemaining -= (uiKeyLen + sizeof( FLMUINT16));
		pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)m_uiDataRemaining;
	}

	while( uiDataToWrite > 0)
	{
		if( !m_pSCache)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_ui32CurBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pSCache)))
		{
			goto Exit;
		}

		if( !bSaveKey)
		{
			pBlkHdr = m_pSCache->m_pBlkHdr;

			// Now copy as much of the remaining data as we  can into the new block.
			
			pDestPtr = (FLMBYTE *)pBlkHdr + sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr );
			pDestPtr += (m_uiBlockSize - 
								sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr ) -
								m_uiDataRemaining);
		}
		else
		{
			bSaveKey = FALSE;
		}

		uiAmtToCopy = (uiDataToWrite <= m_uiDataRemaining
								? uiDataToWrite
								: m_uiDataRemaining);

		f_memcpy( pDestPtr, pucLocalData, uiAmtToCopy);

		m_uiDataRemaining -= uiAmtToCopy;
		m_uiOADataLength += uiAmtToCopy;
		uiDataToWrite -= uiAmtToCopy;
		pucLocalData += uiAmtToCopy;
		pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)m_uiDataRemaining;

		// Now get the next block (if needed)
		
		if( uiDataToWrite)
		{
			pPrevSCache = m_pSCache;
			m_pSCache = NULL;

			// Now create a new block
			
			if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock( m_pDb, &m_pSCache)))
			{
				goto Exit;
			}

			pBlkHdr = m_pSCache->m_pBlkHdr;
			pBlkHdr->ui8BlkType = BT_DATA_ONLY;
			pBlkHdr->ui32PrevBlkInChain = pPrevSCache->m_pBlkHdr->ui32BlkAddr;
			pBlkHdr->ui32NextBlkInChain = 0;

			if( m_pLFile->uiEncDefNum)
			{
				((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum = (FLMUINT32)m_pLFile->uiEncDefNum;
				setBlockEncrypted( pBlkHdr);
			}

			pBlkHdr->ui16BlkBytesAvail =	
				(FLMUINT16)(m_uiBlockSize - sizeofDOBlkHdr( pBlkHdr));

			pPrevSCache->m_pBlkHdr->ui32NextBlkInChain = pBlkHdr->ui32BlkAddr;

			m_ui32CurBlkAddr = pBlkHdr->ui32BlkAddr;
			m_uiDataRemaining = m_uiBlockSize - sizeofDOBlkHdr( pBlkHdr);

			if( pPrevSCache)
			{
				ScaReleaseCache( pPrevSCache, FALSE);
				pPrevSCache = NULL;
			}
		}
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}
	
	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}
	
	return( rc);
}

/***************************************************************************
Desc: Function to Replace data in data only blocks.
****************************************************************************/
RCODE F_Btree::replaceDataOnlyBlocks(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bSaveKey,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMBOOL				bLast,
	FLMBOOL				bTruncate)
{
	RCODE						rc = NE_SFLM_OK;
	F_CachedBlock *		pPrevSCache = NULL;
	const FLMBYTE *		pucLocalData = pucData;
	FLMUINT					uiDataToWrite = uiDataLen;
	F_BLK_HDR *				pBlkHdr = NULL;
	FLMBYTE *				pDestPtr = NULL;
	FLMUINT					uiAmtToCopy;
	FLMUINT32				ui32NextBlkAddr;

	// Do we need to store the key too?
	
	if( bSaveKey)
	{
		if( !m_pSCache)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_ui32CurBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pSCache)))
		{
			goto Exit;
		}

		// Assert that the current block is empty and has no previous link.
		
		flmAssert( m_pSCache->m_pBlkHdr->ui16BlkBytesAvail ==
			m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr));

		flmAssert( m_pSCache->m_pBlkHdr->ui32PrevBlkInChain == 0);

		pBlkHdr = m_pSCache->m_pBlkHdr;
		pDestPtr = (FLMBYTE *)pBlkHdr + 
							sizeofDOBlkHdr( (F_BLK_HDR *)m_pSCache->m_pBlkHdr );

		UW2FBA( (FLMUINT16)uiKeyLen, pDestPtr);
		pDestPtr += sizeof( FLMUINT16);

		f_memcpy( pDestPtr, pucKey, uiKeyLen);
		pDestPtr += uiKeyLen;
		
		m_uiDataRemaining -= (uiKeyLen + sizeof( FLMUINT16));
		pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)m_uiDataRemaining;
	}

	while( uiDataToWrite > 0)
	{
		if( !m_pSCache)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				m_ui32CurBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pSCache)))
		{
			goto Exit;
		}

		if( !bSaveKey)
		{
			pBlkHdr = m_pSCache->m_pBlkHdr;

			// Now copy as much of the remaining data as we  can into the new block.
			
			pDestPtr = (FLMBYTE *)pBlkHdr + sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr);
			pDestPtr += (m_uiBlockSize - sizeofDOBlkHdr( 
								(F_BLK_HDR *)pBlkHdr ) - m_uiDataRemaining);
		}
		else
		{
			bSaveKey = FALSE;
		}

		uiAmtToCopy = (uiDataToWrite <= m_uiDataRemaining
								? uiDataToWrite
								: m_uiDataRemaining);

		f_memcpy( pDestPtr, pucLocalData, uiAmtToCopy);

		m_uiDataRemaining -= uiAmtToCopy;
		m_uiOADataLength += uiAmtToCopy;
		uiDataToWrite -= uiAmtToCopy;
		pucLocalData += uiAmtToCopy;

		if( bTruncate || (m_uiDataRemaining < pBlkHdr->ui16BlkBytesAvail))
		{
			pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)m_uiDataRemaining;
		}

		// Now get the next block (if needed)
		
		if( uiDataToWrite)
		{
			pPrevSCache = m_pSCache;
			m_pSCache = NULL;
			ui32NextBlkAddr = pPrevSCache->m_pBlkHdr->ui32NextBlkInChain;
			
			if( ui32NextBlkAddr)
			{
				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					ui32NextBlkAddr, NULL, &m_pSCache)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
					m_pDb, &m_pSCache)))
				{
					goto Exit;
				}
				pBlkHdr = m_pSCache->m_pBlkHdr;
			}
			else
			{
				// Now create a new block
				
				if( RC_BAD( rc = m_pDb->m_pDatabase->createBlock(
					m_pDb, &m_pSCache)))
				{
					goto Exit;
				}

				pBlkHdr = m_pSCache->m_pBlkHdr;
				pBlkHdr->ui8BlkType = BT_DATA_ONLY;
				pBlkHdr->ui32PrevBlkInChain = pPrevSCache->m_pBlkHdr->ui32BlkAddr;
				pBlkHdr->ui32NextBlkInChain = 0;

				if( m_pLFile->uiEncDefNum)
				{
					setBlockEncrypted( pBlkHdr);
					((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum = (FLMUINT32)m_pLFile->uiEncDefNum;
				}

				pBlkHdr->ui16BlkBytesAvail =
						(FLMUINT16)(m_uiBlockSize - sizeofDOBlkHdr( pBlkHdr));
			}

			pPrevSCache->m_pBlkHdr->ui32NextBlkInChain = pBlkHdr->ui32BlkAddr;

			m_ui32CurBlkAddr = pBlkHdr->ui32BlkAddr;
			m_uiDataRemaining = m_uiBlockSize - sizeofDOBlkHdr( pBlkHdr);

			if( pPrevSCache)
			{
				ScaReleaseCache( pPrevSCache, FALSE);
				pPrevSCache = NULL;
			}
		}
	}

	// If this was the last pass to store the data, then see if we need to
	// remove any left over blocks.  We will not truncate the data if
	// the bTruncate parameter is not set.
	
	if( bLast && bTruncate)
	{
		flmAssert( m_pSCache);

		ui32NextBlkAddr = m_pSCache->m_pBlkHdr->ui32NextBlkInChain;
		m_pSCache->m_pBlkHdr->ui32NextBlkInChain = 0;
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;

		// If there are any blocks left over, they must be freed.
		
		while( ui32NextBlkAddr)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				ui32NextBlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}

			ui32NextBlkAddr = m_pSCache->m_pBlkHdr->ui32NextBlkInChain;

			rc = m_pDb->m_pDatabase->blockFree( m_pDb, m_pSCache);
			m_pSCache = NULL;
			
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}
	
	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to construct a new leaf entry using the key and value
		information passed in.
****************************************************************************/
RCODE F_Btree::buildAndStoreEntry(
	FLMUINT				uiBlkType,
	FLMUINT				uiFlags,
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	FLMUINT				uiOADataLen,		// If zero, it will not be used.
	FLMUINT				uiChildBlkAddr,
	FLMUINT				uiCounts,
	FLMBYTE *			pucBuffer,
	FLMUINT				uiBufferSize,
	FLMUINT *			puiEntrySize)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucTemp = pucBuffer;

	if( puiEntrySize)
	{
		*puiEntrySize = calcEntrySize( uiBlkType, uiFlags, 
									uiKeyLen, uiDataLen, uiOADataLen);

		if( !(*puiEntrySize) || *puiEntrySize > uiBufferSize)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}

	switch( uiBlkType)
	{
		case BT_LEAF:
		{
			// No Data in this entry, so it is easy to make.

			UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
			pucTemp += 2;
			
			f_memcpy( pucTemp, pucKey, uiKeyLen);
			break;
		}

		case BT_LEAF_DATA:
		{
			// Make sure the correct flags are set...

			if( uiKeyLen > ONE_BYTE_SIZE)
			{
				uiFlags |= BTE_FLAG_KEY_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_KEY_LEN;
			}

			if( uiDataLen > ONE_BYTE_SIZE)
			{
				uiFlags |= BTE_FLAG_DATA_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_DATA_LEN;
			}

			// Only the first element of an entry that spans elements
			// will hold an OADataLen field.

			if( uiOADataLen && (uiFlags & BTE_FLAG_FIRST_ELEMENT))
			{
				uiFlags |= BTE_FLAG_OA_DATA_LEN;
			}
			else
			{
				uiFlags &= ~BTE_FLAG_OA_DATA_LEN;
			}

			// Now start setting the elements of the entry.
			// Flags first.

			*pucTemp = (FLMBYTE)uiFlags;
			pucTemp++;

			// KeyLen

			if( uiFlags & BTE_FLAG_KEY_LEN)
			{
				UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
				pucTemp += 2;
			}
			else
			{
				*pucTemp = (FLMBYTE)uiKeyLen;
				pucTemp++;
			}

			if( uiFlags & BTE_FLAG_DATA_LEN)
			{
				UW2FBA( (FLMUINT16)uiDataLen, pucTemp);
				pucTemp += 2;
			}
			else
			{
				*pucTemp = (FLMBYTE)uiDataLen;
				pucTemp++;
			}

			if( uiFlags & BTE_FLAG_OA_DATA_LEN)
			{
				UD2FBA( (FLMUINT32)uiOADataLen, pucTemp);
				pucTemp += 4;
			}

			// Key

			f_memcpy( pucTemp, pucKey, uiKeyLen);
			pucTemp += uiKeyLen;

			// Data

			f_memcpy( pucTemp, pucData, uiDataLen);
			break;
		}

		case BT_NON_LEAF:
		case BT_NON_LEAF_COUNTS:
		{
			// Child block address - 4 bytes

			pucTemp = pucBuffer;

			flmAssert( uiChildBlkAddr);
			UD2FBA( (FLMUINT32)uiChildBlkAddr, pucTemp);
			pucTemp += 4;

			// Counts - 4 bytes

			if( uiBlkType == BT_NON_LEAF_COUNTS)
			{
				UD2FBA( (FLMUINT32)uiCounts, pucTemp);
				pucTemp += 4;
			}

			// KeyLen field - 2 bytes

			UW2FBA( (FLMUINT16)uiKeyLen, pucTemp);
			pucTemp += 2;

			// Key - variable length (uiKeyLen)

			f_memcpy( pucTemp, pucKey, uiKeyLen);
			break;
		}

		default:
		{
			// Invalid block type

			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to remove an entry from a block. This method will delete the
		entry pointed to by the current Stack. This method does NOT defragment
		the block.  If the entry points to any data only blocks, they will
		also be removed from circulation if the parameter bDeleteDOBlocks is
		set to true.  Otherwise, they will not be freed.  This is so we can
		call this method when we are moving entries between blocks or
		replacing entries etc.
****************************************************************************/
RCODE F_Btree::remove(
	FLMBOOL			bDeleteDOBlocks)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntrySize;
	FLMUINT				uiTmp;
	FLMBYTE *			pucEntry;
	FLMBOOL				bDOBlock;
	F_CachedBlock *	pSCache = NULL;
	FLMUINT				uiBlkAddr;
	FLMBYTE *			pucEndOfHeap;
	F_BTREE_BLK_HDR * pBlkHdr;

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb,
		&m_pStack->pSCache)))
	{
		goto Exit;
	}

	pBlkHdr = m_pStack->pBlkHdr = 
					(F_BTREE_BLK_HDR *)m_pStack->pSCache->getBlockPtr();
					
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

	uiNumKeys = pBlkHdr->ui16NumKeys;

	if( !uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	// Point to the entry...
	
	pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, m_pStack->uiCurOffset);
	uiEntrySize = 	getEntrySize( (FLMBYTE *)pBlkHdr, m_pStack->uiCurOffset);

	pucEndOfHeap = (FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr(pBlkHdr) +
						(uiNumKeys * 2) + pBlkHdr->ui16HeapSize;

	// We are only going to have data only blocks if we are storing data
	// in the btree.
	
	if( m_bData)
	{
		bDOBlock = bteDataBlockFlag( pucEntry);

		// If the data for this entry is in one or more Data Only blocks, then
		// we must delete those blocks first.
		
		if( bDOBlock && bDeleteDOBlocks)
		{
			FLMBYTE	ucDOBlkAddr[ 4];

			// Get the block address of the DO Block.
			
			if( RC_BAD( rc = btGetEntryData( pucEntry, ucDOBlkAddr,
				sizeof( FLMUINT), NULL)))
			{
				goto Exit;
			}

			uiBlkAddr = bteGetBlkAddr( (FLMBYTE *)&ucDOBlkAddr[ 0]);
			while( uiBlkAddr)
			{
				// We need to delete the data only blocks first.
				
				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					uiBlkAddr, NULL, &pSCache)))
				{
					goto Exit;
				}

				// Get the next block address (if any)
				
				uiBlkAddr = pSCache->m_pBlkHdr->ui32NextBlkInChain;

				// Now put the block into the Avail list.
				
				rc = m_pDb->m_pDatabase->blockFree( m_pDb, pSCache);
				pSCache = NULL;
				
				if( RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
	}

	pui16OffsetArray = m_pStack->pui16OffsetArray;

	// Move the offsets around to effectively remove the entry.
	
	for( uiTmp = m_pStack->uiCurOffset; (uiTmp + 1) < uiNumKeys; uiTmp++)
	{
		bteSetEntryOffset( pui16OffsetArray, uiTmp,
								 bteGetEntryOffset( pui16OffsetArray, (uiTmp + 1)));
	}

#ifdef FLM_DEBUG
	// Erase the last offset entry.
	
	bteSetEntryOffset( pui16OffsetArray, uiTmp, 0);
#endif

	pBlkHdr->ui16NumKeys--;
	pBlkHdr->stdBlkHdr.ui16BlkBytesAvail += (FLMUINT16)uiEntrySize;
	pBlkHdr->ui16HeapSize += 2;  // One offset was removed.

	// Was this entry we just removed adjacent to the heap space?  If
	// so then we can increase the heap space.
	
	if( pucEndOfHeap == pucEntry)
	{
		pBlkHdr->ui16HeapSize += (FLMUINT16)actualEntrySize(uiEntrySize);
	}

#ifdef FLM_DEBUG
	// Let's erase whatever was in the entry space.
	
	f_memset( pucEntry, 0, actualEntrySize(uiEntrySize));
#endif

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to remove multiple entries from a block. The entries must be
		contiguous.  If any entries store data in data-only blocks, they will
		be freed and put into the avail list.
****************************************************************************/
RCODE F_Btree::removeRange(
	FLMUINT			uiStartElm,
	FLMUINT			uiEndElm,
	FLMBOOL			bDeleteDOBlocks)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT16 *			pui16OffsetArray;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntrySize;
	FLMBYTE *			pucEntry;
	FLMBOOL				bDOBlock;
	F_CachedBlock *	pSCache = NULL;
	FLMUINT				uiBlkAddr;
	FLMUINT				uiCurOffset;
	FLMUINT				uiCounter;
	FLMBYTE *			pucEndOfHeap;
	FLMBYTE *			pucStartOfHeap;
	F_BTREE_BLK_HDR * pBlkHdr;

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
		m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}

	pBlkHdr = m_pStack->pBlkHdr = 
				(F_BTREE_BLK_HDR *)m_pStack->pSCache->getBlockPtr();
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);
	uiNumKeys = pBlkHdr->ui16NumKeys;
	
	if( !uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	flmAssert( uiEndElm < uiNumKeys);

	// Point to the entry ...
	
	for( uiCurOffset = uiStartElm; uiCurOffset <= uiEndElm; uiCurOffset++)
	{
		pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiCurOffset);
		uiEntrySize = getEntrySize( (FLMBYTE *)pBlkHdr, uiCurOffset);
		pBlkHdr->stdBlkHdr.ui16BlkBytesAvail += (FLMUINT16)uiEntrySize;
		pBlkHdr->ui16NumKeys--;

		bDOBlock = bteDataBlockFlag(pucEntry);

		// If the data for this entry is in a Data Only block, then we must delete
		// those blocks first.
		
		if( bDOBlock && bDeleteDOBlocks)
		{
			FLMBYTE	ucDOBlkAddr[ 4];

			// Get the block address of the DO Block.
			
			if( RC_BAD( rc = btGetEntryData( pucEntry, ucDOBlkAddr, 4, NULL)))
			{
				goto Exit;
			}

			uiBlkAddr = bteGetBlkAddr( (FLMBYTE *)&ucDOBlkAddr[ 0]);
			while( uiBlkAddr)
			{
				// We need to delete the data only blocks first.
				
				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					uiBlkAddr, NULL, &pSCache)))
				{
					goto Exit;
				}

				// Get the next block address (if any)
				
				uiBlkAddr = pSCache->m_pBlkHdr->ui32NextBlkInChain;

				// Now put the block into the Avail list.
				
				rc = m_pDb->m_pDatabase->blockFree( m_pDb, pSCache);
				pSCache = NULL;
				
				if( RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}

		// Now erase the old entry
		
#ifdef FLM_DEBUG
		f_memset( pucEntry, 0, actualEntrySize(uiEntrySize));
#endif
	}

	// Move the offsets around to effectively remove the entries.
	
	pui16OffsetArray = m_pStack->pui16OffsetArray;
	if( uiEndElm < (uiNumKeys - 1))
	{
		// We will need to move the remaining offsets forward
		// to delete the desired range.

		for (uiCurOffset = uiStartElm, uiCounter = 0;
			  uiCounter < (uiNumKeys - (uiEndElm + 1));
			  uiCounter++, uiCurOffset++)
		{
			bteSetEntryOffset( pui16OffsetArray, uiCurOffset,
									 bteGetEntryOffset( pui16OffsetArray,
															  (uiEndElm + uiCounter + 1)));
		}
	}

#ifdef FLM_DEBUG
	// Erase the remaining offsets
	
	while (uiCurOffset < (uiNumKeys - 1))
	{
		bteSetEntryOffset( pui16OffsetArray, uiCurOffset++, 0);
	}
	
#endif

	// We need to determine if we have gained any more heap space.  We start
	// by pointing to the end of the block, them moving forward until we reach
	// the closest entry.
	
	pucEndOfHeap = (FLMBYTE *)pBlkHdr + m_uiBlockSize;
	
	for ( uiCurOffset = 0; uiCurOffset < pBlkHdr->ui16NumKeys; uiCurOffset++)
	{
		pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiCurOffset);
		
		if (pucEntry < pucEndOfHeap)
		{
			pucEndOfHeap = pucEntry;
		}
	}
	
	// Now clean up the heap space.
	
	pucStartOfHeap = (FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr( pBlkHdr) +
						  (pBlkHdr->ui16NumKeys * 2);

	pBlkHdr->ui16HeapSize = (FLMUINT16)(pucEndOfHeap - pucStartOfHeap);

#ifdef FLM_DEBUG
	f_memset( pucStartOfHeap, 0, pBlkHdr->ui16HeapSize);
#endif

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to try to move entries (whole) from the target block to the
		previous block.  The entries may be moved, up to but not including
		the current entry position.  We do not want to change the parentage
		of this block.  We need to use the stack to fix up the parentage of
		the previous block. Entries are moved from the lowest to highest.
****************************************************************************/
RCODE F_Btree::moveEntriesToPrevBlk(
	FLMUINT				uiNewEntrySize,
	F_CachedBlock **	ppPrevSCache,
	FLMBOOL *			pbEntriesWereMoved)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiLocalAvailSpace;
	FLMUINT				uiAvailSpace;
	FLMUINT				uiHeapSize;
	F_CachedBlock *	pPrevSCache = NULL;
	FLMUINT				uiPrevBlkAddr;
	FLMUINT				uiOAEntrySize = 0;
	FLMUINT				uiStart;
	FLMUINT				uiFinish;
	FLMUINT				uiCount;
	FLMUINT				uiOffset;

	// Assume nothing to move.
	
	*pbEntriesWereMoved = FALSE;

	// If we are already at the first entry in the block, there
	// is nothing that we can move since we will always insert ahead of
	// the current position.
	
	if( !m_pStack->uiCurOffset)
	{
		goto Exit;
	}

	// Get the previous block.
	
	if( (uiPrevBlkAddr = 
		m_pStack->pSCache->m_pBlkHdr->ui32PrevBlkInChain) == 0)
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		uiPrevBlkAddr, NULL, &pPrevSCache)))
	{
		goto Exit;
	}

	uiLocalAvailSpace = m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail;
	uiAvailSpace = pPrevSCache->m_pBlkHdr->ui16BlkBytesAvail;
	uiHeapSize = ((F_BTREE_BLK_HDR *)pPrevSCache->m_pBlkHdr)->ui16HeapSize;

	// If we add the available space in this block and the previous block, would
	// it be enough to make room for the new entry?  If so, then we will
	// see if we can make that room by moving ( whole) entries.
	
	if( (uiAvailSpace + uiLocalAvailSpace) < uiNewEntrySize)
	{
		goto Exit;
	}

	uiStart = 0;
	uiFinish = m_pStack->uiCurOffset;

	// Get the size of each entry until we are over the available size limit
	
	for( uiOffset = 0, uiCount = 0 ; uiOffset < uiFinish; uiOffset++)
	{
		FLMUINT		uiLocalEntrySize;

		uiLocalEntrySize = getEntrySize( (FLMBYTE *)m_pStack->pBlkHdr, uiOffset);

		if( (uiLocalEntrySize + uiOAEntrySize) < uiAvailSpace)
		{
			uiOAEntrySize += uiLocalEntrySize;
			uiLocalAvailSpace += uiLocalEntrySize;
			uiCount++;
		}
		else
		{
			break;
		}
	}

	if( !uiCount)
	{
		goto Exit;
	}

	// It looks like we can move at least one entry.
	// Will this give use enough room to store the new entry?
	
	if( uiLocalAvailSpace < uiNewEntrySize)
	{
		// Moving these entries will not benefit us, so don't bother
		
		goto Exit;
	}

	// Do we need to defragment the block first?
	
	if( uiHeapSize < uiOAEntrySize)
	{
		flmAssert( uiHeapSize != uiAvailSpace);
		if( RC_BAD( rc = defragmentBlock( &pPrevSCache)))
		{
			goto Exit;
		}
	}

	// We are going to get some benefit from moving, so let's do it...
	
	if (RC_BAD( rc = moveToPrev( uiStart, uiStart + uiCount - 1, &pPrevSCache)))
	{
		goto Exit;
	}

	// We will need to return this block.
	
	*ppPrevSCache = pPrevSCache;
	pPrevSCache = NULL;

	// Adjust the current offset in the stack so we are still pointing to the
	// same entry.
	
	m_pStack->uiCurOffset -= uiCount;

	// If this block has a parent block, and the btree is maintaining counts
	// we will want to update the counts on the parent block.
	
	if( !isRootBlk( m_pStack->pBlkHdr))
	{
		if( m_bCounts)
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}
	}

	*pbEntriesWereMoved = TRUE;

Exit:

	if (pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: This method will move entries beginning at uiStart, up to and
		including uiFinish from the current block (m_pStack) to pPrevSCache.
		As a part of this operation, both the target block and the source 
		block will be changed.  A call to logPhysBlock will be made before
		each block is changed.  Never move the highest key in the block 
		because we don't want to have to update the parentage of the 
		current block...
****************************************************************************/
RCODE F_Btree::moveToPrev(
	FLMUINT				uiStart,
	FLMUINT				uiFinish,
	F_CachedBlock **	ppPrevSCache)
{
	RCODE							rc = NE_SFLM_OK;
	FLMUINT16 *					pui16DstOffsetA = NULL;
	F_BTREE_BLK_HDR *			pSrcBlkHdr = NULL;
	F_BTREE_BLK_HDR *			pDstBlkHdr = NULL;
	FLMBYTE *					pucSrcEntry;
	FLMBYTE *					pucDstEntry;
	FLMUINT						uiEntrySize;
	FLMUINT						uiIndex;
	F_CachedBlock *			pPrevSCache;
	FLMBOOL						bEntriesCombined = FALSE;

	// Make sure we have logged the block we are changing.
	// Note that the source block will be logged in the removeRange method.

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, ppPrevSCache)))
	{
		goto Exit;
	}

	pPrevSCache = *ppPrevSCache;
	pSrcBlkHdr = m_pStack->pBlkHdr;
	pDstBlkHdr = (F_BTREE_BLK_HDR *)pPrevSCache->m_pBlkHdr;
	pui16DstOffsetA = BtOffsetArray( (FLMBYTE *)pDstBlkHdr, 0);

	pucDstEntry = getBlockEnd( pDstBlkHdr);

	// Beginning at the start, copy each entry over from the source
	// to the destination block.
	
	for( uiIndex = uiStart; uiIndex <= uiFinish; uiIndex++)
	{
		if( RC_BAD( rc = combineEntries( pSrcBlkHdr, uiIndex, pDstBlkHdr,
													(pDstBlkHdr->ui16NumKeys
														? pDstBlkHdr->ui16NumKeys - 1
														: 0),
													&bEntriesCombined, &uiEntrySize)))
		{
			goto Exit;
		}

		if( bEntriesCombined)
		{
			F_BTSK		tmpStack;
			F_BTSK *		pTmpStack;

			tmpStack.pSCache = pPrevSCache;
			tmpStack.pBlkHdr = pDstBlkHdr;
			tmpStack.uiCurOffset = pDstBlkHdr->ui16NumKeys - 1;  // Last entry

			pTmpStack = m_pStack;
			m_pStack = &tmpStack;

			rc = remove( FALSE);
			m_pStack = pTmpStack;
			
			if( RC_BAD( rc))
			{
				goto Exit;
			}

			if( pDstBlkHdr->ui16HeapSize != 
						pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
			{
				if( RC_BAD( rc = defragmentBlock( &pPrevSCache)))
				{
					goto Exit;
				}
			}

			pucDstEntry = getBlockEnd( pDstBlkHdr) - uiEntrySize;
			f_memcpy( pucDstEntry, m_pucTempBlk, uiEntrySize);

			bteSetEntryOffset( pui16DstOffsetA,
									 pDstBlkHdr->ui16NumKeys++,
									 (FLMUINT16)(pucDstEntry - (FLMBYTE *)pDstBlkHdr));

			pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail -=
												((FLMUINT16)uiEntrySize + 2);

			pDstBlkHdr->ui16HeapSize -= ((FLMUINT16)uiEntrySize + 2);
			bEntriesCombined = FALSE;
		}
		else
		{
			pucSrcEntry = BtEntry( (FLMBYTE *)pSrcBlkHdr, uiIndex);
			uiEntrySize = getEntrySize( (FLMBYTE *)pSrcBlkHdr, uiIndex);
			pucDstEntry -= actualEntrySize(uiEntrySize);

			f_memcpy( pucDstEntry, pucSrcEntry, actualEntrySize(uiEntrySize));

			bteSetEntryOffset( pui16DstOffsetA,
									 pDstBlkHdr->ui16NumKeys++,
									 (FLMUINT16)(pucDstEntry - (FLMBYTE *)pDstBlkHdr));

			pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail -= (FLMUINT16)uiEntrySize;
			pDstBlkHdr->ui16HeapSize -= (FLMUINT16)uiEntrySize;
		}
	}

	// Now remove the entries from the Src block.
	
	if( RC_BAD( rc = removeRange( uiStart, uiFinish, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to try to move entries (whole) from the target block to the
		next block.  The entries may be moved up to but not including
		the current entry position depending on how much room is available if
		any.  Entries are moved from the highest to lowest.
****************************************************************************/
RCODE F_Btree::moveEntriesToNextBlk(
	FLMUINT		uiNewEntrySize,
	FLMBOOL *	pbEntriesWereMoved)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiLocalAvailSpace;
	FLMUINT				uiAvailSpace;
	FLMUINT				uiHeapSize;
	F_CachedBlock *	pNextSCache = NULL;
	FLMUINT				uiNextBlkAddr;
	FLMUINT				uiOAEntrySize = 0;
	FLMUINT				uiStart;
	FLMUINT				uiFinish;
	FLMUINT				uiCount;
	FLMUINT				uiOffset;
	F_CachedBlock *	pChildSCache = NULL;
	F_CachedBlock *	pParentSCache = NULL;
	F_BTSK *				pParentStack;
	FLMUINT				uiLevel;
	FLMBOOL				bReleaseChild = FALSE;
	FLMBOOL				bReleaseParent = FALSE;
	FLMBOOL				bCommonParent = FALSE;

	// Assume nothing to move.
	
	*pbEntriesWereMoved = FALSE;

	// Get the next block.
	
	if( (uiNextBlkAddr = m_pStack->pSCache->m_pBlkHdr->ui32NextBlkInChain) == 0)
	{
		goto Exit;
	}

	if( (FLMUINT16)m_pStack->uiCurOffset >= (m_pStack->pBlkHdr->ui16NumKeys - 1))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		uiNextBlkAddr, NULL, &pNextSCache)))
	{
		goto Exit;
	}

	// Our first task is to determine if we can move anything at all.
	// How much free space is there in the next block?
	
	uiLocalAvailSpace = m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail;
	uiAvailSpace = pNextSCache->m_pBlkHdr->ui16BlkBytesAvail;
	uiHeapSize = ((F_BTREE_BLK_HDR *)pNextSCache->m_pBlkHdr)->ui16HeapSize;

	// If we add the available space in this block and the next block, would
	// it be enough to make room for the new entry?  If so, then we will
	// see if we can make that room by moving ( whole) entries.
	
	if( (uiAvailSpace + uiLocalAvailSpace) < uiNewEntrySize)
	{
		goto Exit;
	}

	// Begin at the last entry and work backward.
	
	uiStart = m_pStack->pBlkHdr->ui16NumKeys - 1;
	uiFinish = m_pStack->uiCurOffset;

	// Get the size of each entry (plus 2 for the offset entry) until we are
	// over the available size limit.
	
	for( uiOffset = uiStart, uiCount = 0 ; uiOffset > uiFinish; uiOffset--)
	{
		FLMUINT		uiLocalEntrySize;

		uiLocalEntrySize = getEntrySize( (FLMBYTE *)m_pStack->pBlkHdr,
													uiOffset);

		if( (uiLocalEntrySize + uiOAEntrySize) < uiAvailSpace)
		{
			uiOAEntrySize += uiLocalEntrySize;
			uiLocalAvailSpace += uiLocalEntrySize;
			uiCount++;
		}
		else
		{
			break;
		}
	}

	if( uiCount == 0)
	{
		goto Exit;
	}

	// It looks like we can move at least one entry.
	// Will this give use enough room to store the new entry?
	
	if( uiLocalAvailSpace < uiNewEntrySize)
	{
		goto Exit;
	}

	flmAssert( uiStart > uiFinish);

	// Do we need to defragment the block first before we do the move?
	
	if( uiHeapSize < uiOAEntrySize)
	{
		flmAssert( uiHeapSize != uiAvailSpace);
		if( RC_BAD( rc = defragmentBlock( &pNextSCache)))
		{
			goto Exit;
		}
	}

	// We are going to get some benefit from moving, so let's do it...
	
	if( RC_BAD( rc = moveToNext( uiStart, uiStart - uiCount + 1,
		&pNextSCache)))
	{
		goto Exit;
	}

	// If this block has a parent block, and the btree is maintaining counts
	// we will need to update the counts on the parent blocks.
	
	if( m_bCounts)
	{
		for( uiLevel = m_pStack->uiLevel;
			  uiLevel < m_uiStackLevels - 1;
			  uiLevel++)
		{
			pParentStack = &m_Stack[ uiLevel + 1];

			// If we are at "current" level, then we want to use the pNextSCache
			// block as the child.  Otherwise, we want to use the previous parent
			// block as the child.
			
			if( uiLevel == m_pStack->uiLevel)
			{
				pChildSCache =	 pNextSCache;
				bReleaseChild = TRUE;
				pNextSCache =	 NULL;
			}
			else
			{
				pChildSCache =	  pParentSCache;
				bReleaseChild =  bReleaseParent;
				bReleaseParent = FALSE;
			}

			// Check to see if the parent entry is the last entry in the
			// block.   If it is, then we will need to get the next block.
			// If the parent block is the same for both blocks, then we
			// only need to reference the next entry.  We don't want to release
			// the parent as it is referenced in the stack.
			
			if( bCommonParent ||
				  (pParentStack->uiCurOffset <
							(FLMUINT)(pParentStack->pBlkHdr->ui16NumKeys - 1)))
			{
				pParentSCache = pParentStack->pSCache;
				bReleaseParent = FALSE;

				if (RC_BAD( rc = updateParentCounts( pChildSCache, &pParentSCache,
					(bCommonParent
						? pParentStack->uiCurOffset
						: pParentStack->uiCurOffset + 1))))
				{
					goto Exit;
				}
				
				// The parent has changed, so update the stack.
				
				pParentStack->pBlkHdr =
					(F_BTREE_BLK_HDR *)pParentSCache->m_pBlkHdr;

				pParentStack->pSCache = pParentSCache;
				bCommonParent = TRUE;
			}
			else
			{
				// We need to get the next block at the parent level first.  We
				// release the previous parent if there was one.
				
				uiNextBlkAddr = pParentStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain;

				flmAssert( uiNextBlkAddr);

				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					uiNextBlkAddr, NULL, &pParentSCache)))
				{
					goto Exit;
				}

				bReleaseParent = TRUE;

				if( RC_BAD( rc = updateParentCounts( pChildSCache,
					&pParentSCache, 0)))
				{
					goto Exit;
				}
			}

			if( bReleaseChild)
			{
				ScaReleaseCache( pChildSCache, FALSE);
				pChildSCache = NULL;
				bReleaseChild = FALSE;
			}
		}
	}

	*pbEntriesWereMoved = TRUE;

Exit:

	if( pChildSCache && bReleaseChild)
	{
		ScaReleaseCache( pChildSCache, FALSE);
	}

	if( pParentSCache && bReleaseParent)
	{
		ScaReleaseCache( pParentSCache, FALSE);
	}

	if( pNextSCache)
	{
		ScaReleaseCache( pNextSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: This method will move entries beginning at uiStart, down to and
		including uiFinish from the current block (m_pStack) to pNextSCache.
		As a part of this operation, both the target block and the source
		block will be changed.
****************************************************************************/
RCODE F_Btree::moveToNext(
	FLMUINT				uiStart,
	FLMUINT				uiFinish,
	F_CachedBlock **	ppNextSCache)
{
	RCODE							rc = NE_SFLM_OK;
	FLMUINT16 *					pui16DstOffsetA = NULL;
	F_BTREE_BLK_HDR *			pSrcBlkHdr = NULL;
	F_BTREE_BLK_HDR *			pDstBlkHdr = NULL;
	FLMBYTE *					pucSrcEntry;
	FLMBYTE *					pucDstEntry;
	FLMUINT						uiEntrySize;
	FLMINT						iIndex;
	FLMUINT						uiBytesToCopy;
	FLMUINT						uiNumKeysToAdd;
	F_CachedBlock *			pNextSCache = *ppNextSCache;
	FLMBOOL						bEntriesCombined;
	FLMBYTE *					pucOffsetArray;

	// Make sure we have logged the block we are changing.
	// Note that the source block will be logged in the removeRange method.
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pNextSCache)))
	{
		goto Exit;
	}
	
	// SCache block may have changed.  Need to pass it back.
	
	*ppNextSCache = pNextSCache;
	
	pSrcBlkHdr = m_pStack->pBlkHdr;
	pDstBlkHdr = (F_BTREE_BLK_HDR *)pNextSCache->m_pBlkHdr;

	// We will need to save off the current offset array.  We will do this
	// by copying it into our temporary block.
	
	uiBytesToCopy = pDstBlkHdr->ui16NumKeys * 2;
	if( uiBytesToCopy > m_uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	pui16DstOffsetA = BtOffsetArray((FLMBYTE *)pDstBlkHdr, 0);
	pucOffsetArray = &m_pucBuffer[ m_uiBufferSize] - uiBytesToCopy;

	f_memcpy( pucOffsetArray, (FLMBYTE *)pui16DstOffsetA, uiBytesToCopy);

	// Point to the last entry in the block.
	
	pucDstEntry = getBlockEnd( pDstBlkHdr);

	// Beginning at the start, copy each entry over from the Src to the Dst
	// block.  Note that the uiStart parameter represents a higher position
	// in the block.  In otherwords, we are actually copying from the end or
	// highest position to a lower position in the block.  Therefore we want
	// to make sure the offset array is copied in the same way, otherwise it
	// would reverse the order of the entries.
	
	uiNumKeysToAdd = uiStart - uiFinish + 1;
	pui16DstOffsetA = (FLMUINT16 *)pucOffsetArray;

	for( iIndex = uiStart; iIndex >= (FLMINT)uiFinish; iIndex--)
	{
		if( RC_BAD( rc = combineEntries( pSrcBlkHdr, iIndex, pDstBlkHdr,
			0, &bEntriesCombined, &uiEntrySize)))
		{
			goto Exit;
		}

		if( bEntriesCombined)
		{
			F_BTSK		tmpStack;
			F_BTSK *		pTmpStack;

			tmpStack.pSCache = pNextSCache;
			tmpStack.pBlkHdr = pDstBlkHdr;
			tmpStack.uiCurOffset = 0;  // 1st entry.

			pTmpStack = m_pStack;
			m_pStack = &tmpStack;

			rc = remove( FALSE);
			m_pStack = pTmpStack;
			
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			if( pDstBlkHdr->ui16HeapSize !=
						pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
			{
				if( RC_BAD( rc = defragmentBlock( &pNextSCache)))
				{
					goto Exit;
				}

				// Refresh the saved offset array.
				
				uiBytesToCopy -= 2;
				pucOffsetArray = &m_pucBuffer[ m_uiBufferSize] - uiBytesToCopy;

				f_memcpy( pucOffsetArray,
							(FLMBYTE *)BtOffsetArray( (FLMBYTE *)pDstBlkHdr, 0),
							uiBytesToCopy);
			}

			pucDstEntry = getBlockEnd( pDstBlkHdr) - uiEntrySize;
			f_memcpy( pucDstEntry, m_pucTempBlk, uiEntrySize);

			bteSetEntryOffset( pui16DstOffsetA, 0, 
				(FLMUINT16)(pucDstEntry - (FLMBYTE *)pDstBlkHdr));

			pDstBlkHdr->ui16NumKeys++;

			pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail -=
												((FLMUINT16)uiEntrySize + 2);

			pDstBlkHdr->ui16HeapSize -= ((FLMUINT16)uiEntrySize + 2);

			bEntriesCombined = FALSE;
		}
		else
		{
			pucSrcEntry = BtEntry( (FLMBYTE *)pSrcBlkHdr, iIndex);
			uiEntrySize = getEntrySize( (FLMBYTE *)pSrcBlkHdr, iIndex);

			pucDstEntry -= actualEntrySize(uiEntrySize);

			f_memcpy( pucDstEntry, pucSrcEntry,
						 actualEntrySize(uiEntrySize));

			pui16DstOffsetA--;

			bteSetEntryOffset( pui16DstOffsetA, 0,
									 (FLMUINT16)(pucDstEntry - (FLMBYTE *)pDstBlkHdr));

			pDstBlkHdr->ui16NumKeys++;
			pDstBlkHdr->stdBlkHdr.ui16BlkBytesAvail -= (FLMUINT16)uiEntrySize;
			pDstBlkHdr->ui16HeapSize -= (FLMUINT16)uiEntrySize;
		}
	}

	// Now put the new offset array into the block.
	
	f_memcpy( BtOffsetArray( (FLMBYTE *)pDstBlkHdr, 0),
				 pui16DstOffsetA,
				 &m_pucBuffer[ m_uiBufferSize] - (FLMBYTE *)pui16DstOffsetA);

	// Now remove the entries from the Src block.
	
	if( RC_BAD( rc = removeRange( uiFinish, uiStart, FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to advance to the next entry.  If there are no more entries
		in the block, it will release the current block and get the next in
		the chain. If there are no more entries, i.e. no more blocks in 
		the chain, NE_SFLM_EOF_HIT will be returned.
****************************************************************************/
RCODE F_Btree::advanceToNextElement(
	FLMBOOL				bAdvanceStack)
{
	RCODE							rc = NE_SFLM_OK;
	F_BTREE_BLK_HDR *			pBlkHdr;

	flmAssert( m_pSCache);

	pBlkHdr = (F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr;

	if( m_uiCurOffset + 1 >= pBlkHdr->ui16NumKeys)
	{
		// We are out of entries in this block, so we will release it
		// and get the next block in the chain (if any).
		
		if( RC_BAD( rc = getNextBlock( &m_pSCache)))
		{
			goto Exit;
		}

		m_ui32PrimaryBlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;
		m_uiPrimaryOffset = 0;
		m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
		m_uiCurOffset = 0;

		if( bAdvanceStack)
		{
			if( RC_BAD( rc = moveStackToNext( m_pSCache)))
			{
				goto Exit;
			}
			
			// This block now has two uses.  It will be released twice.
			
			m_pSCache->m_uiUseCount++;
		}
	}
	else
	{
		m_uiPrimaryOffset++;
		m_uiCurOffset++;
		m_pStack->uiCurOffset++;
	}

Exit:

	// We do not want to release the m_pSCache here.  That is to be done by the
	// caller.
	
	return( rc);
}

/***************************************************************************
Desc:	Method to backup the stack to the previous entry.  If there are no
		more entries in the block, it will release the current block and get
		the previous in the chain. If there are no more entries, i.e. no
		more blocks in the chain, NE_SFLM_BOF_HIT will be returned.
****************************************************************************/
RCODE F_Btree::backupToPrevElement(
	FLMBOOL				bBackupStack)
{
	RCODE							rc = NE_SFLM_OK;
	F_BTREE_BLK_HDR *			pBlkHdr;

	flmAssert( m_pSCache);

	pBlkHdr = (F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr;

	if( !m_uiCurOffset)
	{
		// We are out of entries in this block, so we will release it
		// and get the previous block in the chain (if any).
		
		if( RC_BAD( rc = getPrevBlock( &m_pSCache)))
		{
			goto Exit;
		}

		m_ui32PrimaryBlkAddr = m_pSCache->m_pBlkHdr->ui32BlkAddr;

		m_uiPrimaryOffset =
			((F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr)->ui16NumKeys - 1;

		m_ui32CurBlkAddr = m_ui32PrimaryBlkAddr;
		m_uiCurOffset = m_uiPrimaryOffset;

		if( bBackupStack)
		{
			if( RC_BAD( rc = moveStackToPrev( m_pSCache)))
			{
				goto Exit;
			}
			
			// This block now has two uses.  It will be released twice.
			
			m_pSCache->m_uiUseCount++;
		}
	}
	else
	{
		m_uiPrimaryOffset--;
		m_uiCurOffset--;
		m_pStack->uiCurOffset--;
	}

Exit:

	// We do not want to release the m_pSCache here.  That is to be done by the
	// caller.
	
	return( rc);
}

/***************************************************************************
Desc:	Method to extract the key length from a given entry.  The optional
		pucKeyRV is a buffer where we can return the address of the start of
		the actual key.
****************************************************************************/
FLMUINT F_Btree::getEntryKeyLength(
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlockType,
	const FLMBYTE **	ppucKeyRV)
{
	FLMUINT				uiKeyLength;
	FLMBYTE *			pucTmp = NULL;

	// The way we get the key length depends on the type of block we have.

	switch( uiBlockType)
	{
		case BT_LEAF_DATA:
		{
			pucTmp = &pucEntry[ 1];  // skip past the flags
			
			if( bteKeyLenFlag( pucEntry))
			{
				uiKeyLength = FB2UW( pucTmp);
				pucTmp += 2;
			}
			else
			{
				uiKeyLength = *pucTmp;
				pucTmp += 1;
			}

			if( bteDataLenFlag(pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp += 1;
			}

			// Check for the presence of the OverallDataLength field (4 bytes).
			
			if( bteOADataLenFlag( pucEntry))
			{
				pucTmp += 4;
			}

			break;
		}
		
		case BT_LEAF:
		{
			uiKeyLength = FB2UW( pucEntry);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_KEY_START];
			}

			break;
		}
		
		case BT_NON_LEAF:
		{
			uiKeyLength = FB2UW( &pucEntry[ BTE_NL_KEY_LEN]);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_NL_KEY_START];
			}

			break;
		}
		
		case BT_NON_LEAF_COUNTS:
		{
			uiKeyLength = FB2UW( &pucEntry[ BTE_NLC_KEY_LEN]);

			if( ppucKeyRV)
			{
				pucTmp = &pucEntry[ BTE_NLC_KEY_START];
			}

			break;
		}
		
		default:
		{
			flmAssert( 0);
			uiKeyLength = 0;
			pucTmp = NULL;
			break;
		}
	}

	// Do we need to return the key pointer?
	
	if( ppucKeyRV)
	{
		*ppucKeyRV = pucTmp;
	}

	return( uiKeyLength);
}

/***************************************************************************
Desc:	Method to extract the data length from a given entry. The parameter
		pucDataRV is an optional return value that will hold the address
		of the beginning of the data in the entry.  This method 
		** assumes ** the entry is from a BT_LEAF_DATA block.  No other block
		type has any data.
****************************************************************************/
FSTATIC FLMUINT btGetEntryDataLength(
	FLMBYTE *			pucEntry,
	const FLMBYTE **	ppucDataRV,				// Optional
	FLMUINT *			puiOADataLengthRV,	// Optional
	FLMBOOL *			pbDOBlockRV)			// Optional
{
	const FLMBYTE *	pucTmp;
	FLMUINT				uiDataLength;
	FLMUINT				uiKeyLength;

	pucTmp = &pucEntry[ 1];  // skip past the flags
	
	if( bteKeyLenFlag( pucEntry))
	{
		uiKeyLength = FB2UW( pucTmp);
		pucTmp += 2; 
	}
	else
	{
		uiKeyLength = *pucTmp;
		pucTmp += 1;
	}

	if( bteDataLenFlag(pucEntry))
	{
		uiDataLength = FB2UW( pucTmp);
		pucTmp += 2;
	}
	else
	{
		uiDataLength = *pucTmp;
		pucTmp += 1;
	}

	// Check for the presence of the OverallDataLength field (4 bytes).
	
	if( bteOADataLenFlag(pucEntry))
	{
		if( puiOADataLengthRV)
		{
			*puiOADataLengthRV = FB2UD( pucTmp);
		}
		pucTmp += 4;
	}
	else if (puiOADataLengthRV)
	{
		*puiOADataLengthRV = uiDataLength;
	}

	// Are we to return a pointer to the data?
	
	if( ppucDataRV)
	{
		// Advance to the Data since we are currently pointing to the Key.
		
		*ppucDataRV = (FLMBYTE *)(pucTmp + uiKeyLength);
	}

	if( pbDOBlockRV)
	{
		*pbDOBlockRV = bteDataBlockFlag( pucEntry);
	}

	return( uiDataLength);
}

/***************************************************************************
Desc:	Method to extract the data value from a given block. This method
		expects to receive a buffer to copy the data into.  This method does
		not read data across blocks.  The puiLenDataRV is an optional 
		parameter that will hold the actual data size returned.
****************************************************************************/
FSTATIC RCODE btGetEntryData(
	FLMBYTE *		pucEntry,	// Pointer to the entry containing the data
	FLMBYTE *		pucBufferRV,
	FLMUINT			uiBufferSize,
	FLMUINT *		puiLenDataRV)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiDataLength;
	const FLMBYTE *	pucData;

	// Get the data length
	
	uiDataLength = btGetEntryDataLength( pucEntry, &pucData, NULL, NULL);

	if( uiDataLength > uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

#ifdef FLM_DEBUG
	f_memset( pucBufferRV, 0, uiBufferSize);
#endif
	f_memcpy( pucBufferRV, pucData, uiDataLength);

	// Do we need to return the data length?

	if( puiLenDataRV)
	{
		*puiLenDataRV = uiDataLength;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will return the overall size of the entry at uiOffset in
		pBlk.  The size returned includes a two byte allowance for the offset
		entry used by this entry.
****************************************************************************/
FLMUINT F_Btree::getEntrySize(
	FLMBYTE *		pBlk,
	FLMUINT			uiOffset,
	FLMBYTE **		ppucEntry)
{
	FLMBYTE *		pucEntry;
	FLMUINT			uiEntrySize;

	// Point to the entry ...
	
	pucEntry = BtEntry( pBlk, uiOffset);

	if( ppucEntry)
	{
		*ppucEntry = pucEntry;
	}

	// Different block types have different entry formats.
	
	switch( getBlkType( pBlk))
	{
		case BT_LEAF:
		{
			uiEntrySize =  4 + FB2UW( pucEntry);
			break;
		}
		case BT_LEAF_DATA:
		{
			FLMBYTE * pucTmp = &pucEntry[ 1];

			// Stuff we know
			
			uiEntrySize = 3;

			// Get the key length
			
			if( *pucEntry & BTE_FLAG_KEY_LEN)
			{
				uiEntrySize += FB2UW( pucTmp) + 2;
				pucTmp += 2;
			}
			else
			{
				uiEntrySize += (*pucTmp + 1);
				pucTmp++;
			}

			// Get the data length
			
			if( *pucEntry & BTE_FLAG_DATA_LEN)
			{
				// 2 byte data length field
				
				uiEntrySize += (FB2UW( pucTmp) + 2);
			}
			else
			{
				// 1 byte data length field
				
				uiEntrySize += (FLMUINT)*pucTmp + 1;
			}

			// Get the Overall Data length (if present)
			
			if( *pucEntry & BTE_FLAG_OA_DATA_LEN)
			{
				uiEntrySize += 4;
			}
			
			break;
		}
		
		case BT_NON_LEAF:
		{
			uiEntrySize = 8 + FB2UW( &pucEntry[ BTE_NL_KEY_LEN]);
			break;
		}
		
		case BT_NON_LEAF_COUNTS:
		{
			uiEntrySize = 12 + FB2UW( &pucEntry[ BTE_NLC_KEY_LEN]);
			break;
		}
		
		default:
		{
			flmAssert( 0);
			uiEntrySize = 0;
			break;
		}
	}

	return( uiEntrySize);
}

/***************************************************************************
Desc:	Method to search the BTree for a specific entry. Upon a successful
		return from this method, the local stack will be setup and pointing
		to either the desired entry, or if the entry does not exist, it will
		be pointing to the entry that would be immediately after the desired
		entry.  This method therefore can be used both for reads and updates
		where we want to insert a new entry into the BTree.
****************************************************************************/
RCODE F_Btree::findEntry(
	const FLMBYTE *	pucKey,				// In
	FLMUINT 				uiKeyLen,			// In
	FLMUINT				uiMatch,				// In
	FLMUINT *			puiPosition,		// Out
	FLMUINT32 *			pui32BlkAddr,		// In/Out
	FLMUINT *			puiOffsetIndex)	// In/Out
{
	RCODE					rc = NE_SFLM_OK;
	F_BTSK *				pStack = NULL;
	FLMUINT32			ui32BlkAddress;
	F_CachedBlock *	pSCache = NULL;
	FLMBYTE *			pucEntry;
	FLMUINT				uiPrevCounts = 0;
	FLMUINT				uiLevel;

	// Make sure the stack is clean before we start.
	
	btRelease();

	// No input key is needed to get the first or last key.
	
	if( uiMatch == FLM_FIRST || uiMatch == FLM_LAST)
	{
		uiKeyLen = 0;
	}

	if( uiKeyLen > SFLM_MAX_KEY_SIZE)
	{
		rc = RC_SET( NE_SFLM_BTREE_KEY_SIZE);
		goto Exit;
	}

	// Have we been passed a block address to look in?
	
	if( pui32BlkAddr && *pui32BlkAddr)
	{
		if( RC_OK( rc = findInBlock( pucKey, uiKeyLen, uiMatch, puiPosition,
			pui32BlkAddr, puiOffsetIndex)))
		{
			goto Exit;
		}
	}

	// Beginning at the root node, we will scan until we find the first key
	// that is greater than or equal to our target key.  If we don't find any
	// key that is larger than our target key, we will use the last block found.
	
	ui32BlkAddress = (FLMUINT32)m_pLFile->uiRootBlk;

	for( ;;)
	{
		// Get the block - Note that this will place a use on the block.
		// It must be properly released when done.

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32BlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}

		// We are building the stack inverted to make traversing it a bit easier.
		
		uiLevel = ((F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr)->ui8BlkLevel;
		pStack = &m_Stack[ uiLevel];

		m_uiStackLevels++;

		pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
		pStack->ui32BlkAddr = ui32BlkAddress;
		pStack->pSCache = pSCache;
		pSCache = NULL;
		pStack->uiLevel = uiLevel;
		pStack->uiKeyLen = uiKeyLen;
		pStack->pucKeyBuf = pucKey;
		pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pStack->pBlkHdr, 0);

		if( isRootBlk( pStack->pBlkHdr))
		{
			m_uiRootLevel = uiLevel;
		}

		// Search the block for the key.  When we return from this method
		// the pStack will be pointing to the last entry we looked at.
		
		if( RC_BAD( rc = scanBlock( pStack, uiMatch)))
		{
			// It is okay if we couldn't find the key.  Especially if
			// we are still in the upper levels of the B-tree.
			
			if( (rc != NE_SFLM_NOT_FOUND) && (rc != NE_SFLM_EOF_HIT))
			{
				goto Exit;
			}
		}

		// Are we at the end of our search?
		
		if( (pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA) ||
			 (pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF) ||
			 (m_uiStackLevels - 1 >= m_uiSearchLevel))
		{
			if( m_bCounts && puiPosition)
			{
				flmAssert( m_uiSearchLevel >= BH_MAX_LEVELS);
				*puiPosition = uiPrevCounts + pStack->uiCurOffset;
			}

			// If this is a search for the last entry, then we should adjust the
			// uiCurOffset so that it points to a valid entry.
			
			if( uiMatch == FLM_LAST)
			{
				m_pStack = pStack;

				for (;;)
				{
					if( RC_BAD( rc = moveStackToPrev( NULL)))
					{
						goto Exit;
					}
					
					// If we are on the leaf level, we need to make sure we are
					// looking at a first occurrence of an entry.
					
					if( pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA)
					{
						pucEntry = BtEntry(
									(FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

						if( bteFirstElementFlag( pucEntry))
						{
							break;
						}
					}
					else
					{
						break;
					}
				}
			}

			break;
		}
		else
		{
			if( m_bCounts && puiPosition)
			{
				uiPrevCounts += countRangeOfKeys( pStack, 0, pStack->uiCurOffset);
			}

			// Get the Child Block Address
			
			pucEntry = BtEntry(
				(FLMBYTE *)pStack->pSCache->m_pBlkHdr, pStack->uiCurOffset);
				
			ui32BlkAddress = bteGetBlkAddr( pucEntry);
		}
	}

	// Return the block and offset if needed.
	
	if( pui32BlkAddr)
	{
		*pui32BlkAddr = pStack->ui32BlkAddr;
	}

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = pStack->uiCurOffset;
	}

	m_bStackSetup = TRUE;

Exit:

	if( RC_OK( rc) || (rc == NE_SFLM_NOT_FOUND) || (rc == NE_SFLM_EOF_HIT))
	{
		if( pStack)
		{
			m_pStack = pStack;
		}
	}

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method to search for a particular key in a pre-designted
		block offset.  If we don't find it at the given offset, we will do a
		binary search for it.  Note that a uiMatch of FLM_FIRST & FLM_LAST
		will be ignored if we locate the entry by the puiOffsetIndex parameter.  
		Also, this method does not setup the full stack.  Only the level where
		the block address passed in resides.
****************************************************************************/
RCODE F_Btree::findInBlock(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiMatch,
	FLMUINT *				puiPosition,
	FLMUINT32 *				pui32BlkAddr,
	FLMUINT *				puiOffsetIndex)
{
	RCODE					rc = NE_SFLM_OK;
	F_BTSK *				pStack;
	F_CachedBlock *	pSCache = NULL;
	FLMBYTE *			pucEntry;
	const FLMBYTE *	pucBlkKey;
	FLMUINT				uiBlkKeyLen;

	// Get the block - Note that this will place a use on the block.
	// It must be properly released when done.

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		*pui32BlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}

	if( !blkIsBTree( pSCache->getBlockPtr()))
	{
		rc = RC_SET( NE_SFLM_NOT_FOUND);
		goto Exit;
	}
	
	// Verify that the block belongs to the correct collection number
	
	if( ((F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr)->ui16LogicalFile !=
		m_pLFile->uiLfNum)
	{
		rc = RC_SET( NE_SFLM_NOT_FOUND);
		goto Exit;
	}
	
	// Verify that we are looking at the same type of block,
	// i.e. collection vs index.
	
	if( ((F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr)->ui8BTreeFlags & BLK_IS_INDEX &&
		m_pLFile->eLfType != SFLM_LF_INDEX)
	{
		rc = RC_SET( NE_SFLM_NOT_FOUND);
		goto Exit;
	}

	// If the block is not a leaf block, the caller will
	// need to do a full search down the B-Tree

	if( ((F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr)->ui8BlkLevel != 0)
	{
		rc = RC_SET( NE_SFLM_NOT_FOUND);
		goto Exit;
	}

	pStack = &m_Stack[ 0];
	m_uiStackLevels++;

	pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
	pStack->ui32BlkAddr = *pui32BlkAddr;
	pStack->pSCache = pSCache;
	pSCache = NULL;
	pStack->uiLevel = 0;
	pStack->uiKeyLen = uiKeyLen;
	pStack->pucKeyBuf = pucKey;
	pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pStack->pBlkHdr, 0);
	pStack->uiCurOffset = puiOffsetIndex ? *puiOffsetIndex : 0;

	if( isRootBlk( pStack->pBlkHdr))
	{
		m_uiRootLevel = 0;
	}

	// See if the entry we are looking for is at the passed offset
	
	if( puiOffsetIndex)
	{
		if( *puiOffsetIndex < pStack->pBlkHdr->ui16NumKeys)
		{
			pucEntry = BtEntry( (FLMBYTE *)pStack->pBlkHdr, *puiOffsetIndex);

			uiBlkKeyLen = getEntryKeyLength( pucEntry,
								getBlkType( (FLMBYTE *)pStack->pBlkHdr), &pucBlkKey);

			if( uiKeyLen == uiBlkKeyLen)
			{
				if( f_memcmp( pucKey, pucBlkKey, uiKeyLen) == 0)
				{
					goto GotEntry;
				}
			}
		}
	}

	// Search the block for the key.  When we return from this method
	// the pStack will be pointing to the last entry we looked at.

	if( RC_BAD( rc = scanBlock( pStack, uiMatch)))
	{
		goto Exit;
	}

GotEntry:

	if( m_bCounts && puiPosition)
	{
		flmAssert( m_uiSearchLevel >= BH_MAX_LEVELS);

		// VISIT: These counts aren't accurate in this case.
		
		*puiPosition = pStack->uiCurOffset;
	}

	// Verify that we are looking at an entry with the firstElement flag set.
	
	m_pStack = pStack;

	for (;;)
	{
		// If we are on the leaf level, we need to make sure we are
		// looking at a first occurrence of an entry.
		
		if( m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA)
		{
			pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
									  m_pStack->uiCurOffset);

			if( bteFirstElementFlag( pucEntry))
			{
				break;
			}
		}
		else
		{
			break;
		}

		if( RC_BAD( rc = moveStackToPrev( NULL)))
		{
			goto Exit;
		}
	}

	*pui32BlkAddr = m_pStack->ui32BlkAddr;

	if( puiOffsetIndex)
	{
		*puiOffsetIndex = m_pStack->uiCurOffset;
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( RC_BAD( rc))
	{
		btRelease();
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to search through a BTree block to find a specific key.  If 
		that key cannot be found, then the pStack will be positioned right 
		after the last entry in the block.  The search is a binary search that
		is looking for the first key that is >= the target key.  The uiMatch
		parameter further qualifies the search.  The FLM_FIRST & FLM_LAST
		values will ignore the key altogether and just return the first or last
		key respectively.  The FLM_INCL value will return the key if found or the 
		first key following if not found.  The FLM_EXACT will return an 
		NE_SFLM_NOT_FOUND if the key cannot be found.  FLM_EXCL will return
		the first key following the target key.
****************************************************************************/
RCODE F_Btree::scanBlock(
	F_BTSK *			pStack,
	FLMUINT			uiMatch)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiTop;
	FLMUINT				uiMid;
	FLMUINT				uiBottom;
	FLMINT				iResult;
	F_CachedBlock *	pSCache = NULL;
	const FLMBYTE *	pucBlockKey;
	FLMBYTE *			pucEntry;
	FLMUINT				uiBlockKeyLen;

	if( pStack->pBlkHdr->ui16NumKeys == 0)
	{
		rc = RC_SET( NE_SFLM_BOF_HIT);
		goto Exit;
	}

	uiTop = 0;
	uiBottom = (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1);

	if( uiMatch == FLM_FIRST)
	{
		pStack->uiCurOffset = uiTop;
		goto Exit;
	}

	if( uiMatch == FLM_LAST || pStack->uiKeyLen == 0)
	{
		pStack->uiCurOffset = uiBottom;
		goto Exit;
	}

	flmAssert( uiMatch == FLM_INCL || uiMatch == FLM_EXCL || 
				  uiMatch == FLM_EXACT);

	// Test the first entry
	
	pucEntry = (FLMBYTE *)pStack->pBlkHdr +
										bteGetEntryOffset( pStack->pui16OffsetArray,
										uiTop);
											 
	uiBlockKeyLen = getEntryKeyLength( pucEntry,
										((F_BLK_HDR *)pStack->pBlkHdr)->ui8BlkType,
										 &pucBlockKey);

	// Compare the entries ...
	
	if( !uiBlockKeyLen)
	{
		// The LEM entry will always sort last!!

		iResult = 1;
		goto ResultGreater1;
	}
	else
	{
		if( RC_BAD( rc = compareBlkKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
		{
			goto Exit;
		}
	}
	
	if( iResult >= 0)
	{
ResultGreater1:

		if( iResult && uiMatch == FLM_EXACT)
		{
			rc = RC_SET( NE_SFLM_NOT_FOUND);
		}
		
		uiMid = uiTop;
		goto VerifyPosition;
	}

	// If there is more than one entry in the block, we can skip the first
	// one since we have already seen it.
	
	if( uiTop < uiBottom)
	{
		uiTop++;
	}

	// Test the last
	
	pucEntry = (FLMBYTE *)pStack->pBlkHdr +
					bteGetEntryOffset( pStack->pui16OffsetArray,
											 uiBottom);
											 
	uiBlockKeyLen = getEntryKeyLength( pucEntry,
										((F_BLK_HDR *)pStack->pBlkHdr)->ui8BlkType,
										&pucBlockKey);

	if( !uiBlockKeyLen)
	{
		// The LEM entry will always sort last!!

		iResult = 1;
		goto ResultGreater2;
	}
	else
	{
		if( RC_BAD( rc = compareBlkKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
		{
			goto Exit;
		}
	}
	
	if( iResult <= 0)
	{
		if( iResult < 0 && uiMatch != FLM_INCL)
		{
			rc = RC_SET( NE_SFLM_NOT_FOUND);
		}
		
		uiMid = uiBottom;
		goto VerifyPosition;
	}

ResultGreater2:

	for( ;;)
	{

		if( uiTop == uiBottom)
		{
			// We're done - didn't find it.
			
			if( uiMatch == FLM_EXACT)
			{
				rc = RC_SET( NE_SFLM_NOT_FOUND);
			}
			
			uiMid = uiTop;
			break;
		}

		// Get the midpoint
		
		uiMid = (uiTop + uiBottom) / 2;

		pucEntry = (FLMBYTE *)pStack->pBlkHdr +
						bteGetEntryOffset( pStack->pui16OffsetArray,
												 uiMid);
												 
		uiBlockKeyLen = getEntryKeyLength( pucEntry,
								((F_BLK_HDR *)pStack->pBlkHdr)->ui8BlkType,
								&pucBlockKey);

		// Compare the entries

		if( !uiBlockKeyLen)
		{
			// The LEM entry will always sort last!!

			iResult = 1;
			goto ResultGreater;
		}
		else
		{
			if( RC_BAD( rc = compareBlkKeys( pucBlockKey, uiBlockKeyLen,
				pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
			{
				goto Exit;
			}
		}

		if( iResult > 0)
		{
ResultGreater:

			// Midpoint (block key) is > Target key
			
			uiBottom = uiMid;
			continue;
		}

		if( iResult < 0)
		{
			// Midpoint (block key) is < Target key
			// Since we want to find the first key that is >= to the target key,
			// and we have aleady visited the key at uiMid and know that it is <
			// our target key, we can skip it and advance to the key that is one
			// beyond it.
			
			flmAssert( uiMid < uiBottom);
			uiTop = uiMid + 1;
			continue;
		}
		
		break;
	}

VerifyPosition:

	if( uiMatch != FLM_EXCL)
	{
		// Verify that we are looking at the first occurrence of this key.
		
		while( iResult == 0)
		{
			if( uiMid > 0)
			{
				pucEntry = (FLMBYTE *)pStack->pBlkHdr +
								bteGetEntryOffset( pStack->pui16OffsetArray,
														 (uiMid - 1));

				uiBlockKeyLen = getEntryKeyLength( pucEntry,
										((F_BLK_HDR *)pStack->pBlkHdr)->ui8BlkType,
										&pucBlockKey);

				if( !uiBlockKeyLen)
				{
					// The LEM entry will always sort last!!

					iResult = 1;
				}
				else
				{
					if( RC_BAD( rc = compareBlkKeys( pucBlockKey, uiBlockKeyLen,
						pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
					{
						goto Exit;
					}

					if( iResult == 0)
					{
						uiMid--;
					}
				}
			}
			else
			{
				break;
			}
		}
		
		pStack->uiCurOffset = uiMid;
	}
	else if( uiMatch == FLM_EXCL)
	{
		// If we are at the leaf level, then we want to see if
		// this is the last entry in the last block.
		// If it is, then we cannot satisfy the request, otherwise
		// we will position to the next key and return ok.
		
		if( pStack->pBlkHdr->ui8BlkLevel == 0 &&
			 pStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain == 0 &&
			 uiMid == (FLMUINT)pStack->pBlkHdr->ui16NumKeys - 1 &&
			 iResult == 0)
		{
			rc = RC_SET( NE_SFLM_EOF_HIT);
		}
		else if( pStack->pBlkHdr->ui8BlkLevel == 0)
		{
			// Check for the next entry at leaf level			
			
			while( iResult == 0)
			{
				// Are we on the last key?
				
				if( uiMid == (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1))
				{
					if( pStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain == 0)
					{
						rc = RC_SET( NE_SFLM_NOT_FOUND);
					}
					else
					{
						pStack->uiCurOffset = uiMid;
						m_pStack = pStack;

						if( RC_BAD( rc = moveStackToNext( NULL)))
						{
							goto Exit;
						}
						
						uiMid = 0;
					}
				}
				else
				{
					uiMid++;
				}

				pucEntry = (FLMBYTE *)pStack->pBlkHdr +
								bteGetEntryOffset( pStack->pui16OffsetArray,
														 uiMid);

				uiBlockKeyLen = getEntryKeyLength( pucEntry,
											((F_BLK_HDR *)pStack->pBlkHdr)->ui8BlkType,
											&pucBlockKey);

				if( !uiBlockKeyLen)
				{
					// The LEM entry will always sort last!!

					iResult = 1;
				}
				else
				{
					if( RC_BAD( rc = compareBlkKeys( pucBlockKey, uiBlockKeyLen,
						pStack->pucKeyBuf, pStack->uiKeyLen, &iResult)))
					{
						goto Exit;
					}
				}
			}
			
			pStack->uiCurOffset = uiMid;
			if( uiMid == (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1) &&
							pStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain == 0)
			{
				rc = RC_SET( NE_SFLM_EOF_HIT);
			}
		}
		else
		{
			pStack->uiCurOffset = uiMid;
		}
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc: This method will compare two key fields.
		Returned values:  0 - Keys are equal
								1 - Key in Block is > Target key
							  -1 - Key in Block is < Target key
****************************************************************************/
RCODE F_Btree::compareKeys(
	const FLMBYTE *	pucKey1,
	FLMUINT				uiKeyLen1,
	const FLMBYTE *	pucKey2,
	FLMUINT				uiKeyLen2,
	FLMINT *				piCompare)
{
	RCODE		rc = NE_SFLM_OK;
	
	if( !m_pCompare)
	{
		if( (*piCompare = f_memcmp( pucKey1, pucKey2, 
				f_min( uiKeyLen1, uiKeyLen2))) == 0)
		{
			*piCompare = uiKeyLen1 == uiKeyLen2
								? 0
								: uiKeyLen1 < uiKeyLen2 
									? -1
									: 1;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pCompare->compare( pucKey1, uiKeyLen1,
										pucKey2, uiKeyLen2, piCompare)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method for positioning to a specific entry.
****************************************************************************/
RCODE F_Btree::positionToEntry(
	FLMUINT			uiPosition)
{
	RCODE					rc = NE_SFLM_OK;
	F_BTSK *				pStack = NULL;
	FLMUINT32			ui32BlkAddress;
	F_CachedBlock *	pSCache = NULL;
	FLMUINT				uiLevel;
	FLMBYTE *			pucEntry;
	FLMUINT				uiPrevCounts = 0;

	// Make sure the stack is clean before we start.
	
	btRelease();

	// Beginning at the root node.
	
	ui32BlkAddress = (FLMUINT32)m_pLFile->uiRootBlk;

	// Get the block - Note that this will place a use on the block.
	// It must be properly released when done.
	
	while( ui32BlkAddress)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32BlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}

		uiLevel = ((F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr)->ui8BlkLevel;
		pStack = &m_Stack[ uiLevel];

		pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
		pStack->ui32BlkAddr = ui32BlkAddress;
		pStack->pSCache = pSCache;
		pSCache = NULL;
		pStack->uiLevel = uiLevel;
		pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pStack->pBlkHdr, 0);

		m_uiStackLevels++;

		if( RC_BAD( rc = searchBlock( pStack->pBlkHdr, &uiPrevCounts,
			uiPosition, &pStack->uiCurOffset)))
		{
			goto Exit;
		}

		if( (pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF_DATA) ||
			 (pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_LEAF))
		{
			ui32BlkAddress = 0;
		}
		else
		{
			// Get the next child block address
			
			pucEntry = BtEntry( (FLMBYTE *)pStack->pBlkHdr,
									  pStack->uiCurOffset);

			ui32BlkAddress = bteGetBlkAddr( pucEntry);
		}
	}

	m_uiRootLevel = m_uiStackLevels - 1;

Exit:

	if( RC_OK( rc) || (rc == NE_SFLM_NOT_FOUND) || (rc == NE_SFLM_EOF_HIT))
	{
		m_pStack = pStack;
	}

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_Btree::searchBlock(
	F_BTREE_BLK_HDR *		pBlkHdr,
	FLMUINT *				puiPrevCounts,
	FLMUINT					uiPosition,
	FLMUINT *				puiOffset)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiOffset;
	FLMUINT			uiNumKeys;
	FLMUINT			uiCounts;
	FLMBYTE *		pucEntry;

	uiNumKeys = pBlkHdr->ui16NumKeys;

	if( getBlkType( (FLMBYTE *)pBlkHdr) != BT_NON_LEAF_COUNTS)
	{
		flmAssert( uiPosition >= *puiPrevCounts);
		
		uiOffset = uiPosition - *puiPrevCounts;
		*puiPrevCounts = uiPosition;
	}
	else
	{
		for( uiOffset = 0; uiOffset < uiNumKeys; uiOffset++)
		{
			pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, uiOffset);
			pucEntry += 4;
			
			uiCounts = FB2UD( pucEntry);

			if( *puiPrevCounts + uiCounts >= (uiPosition + 1))
			{
				break;
			}
			else
			{
				*puiPrevCounts += uiCounts;
			}
		}
	}

	if( uiOffset >= uiNumKeys)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
	}

	*puiOffset = uiOffset;
	return( rc);
}

/***************************************************************************
Desc:	Method to move all the data in the block into a contiguous space.
****************************************************************************/
RCODE F_Btree::defragmentBlock(
	F_CachedBlock **			ppSCache)
{
	RCODE							rc = NE_SFLM_OK;
	FLMUINT						uiNumKeys;
	FLMBOOL						bSorted;
	FLMBYTE *					pucCurEntry;
	FLMBYTE *					pucPrevEntry;
	FLMBYTE *					pucTempEntry;
	FLMUINT						uiTempToMove;
	FLMUINT						uiIndex;
	FLMUINT						uiAmtToMove;
	FLMUINT						uiFirstHole;
	FLMUINT16					ui16BlkBytesAvail;
	FLMUINT16 *					pui16OffsetArray;
	F_CachedBlock *			pSCache = *ppSCache;
	F_BTREE_BLK_HDR *			pBlk = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
	F_BTREE_BLK_HDR *			pOldBlk = NULL;
	FLMBYTE *					pucHeap;
	FLMBYTE *					pucBlkEnd;
	F_CachedBlock *			pOldSCache = NULL;

	flmAssert( pBlk->stdBlkHdr.ui16BlkBytesAvail != pBlk->ui16HeapSize);

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
		m_pDb, &pSCache, &pOldSCache)))
	{
		goto Exit;
	}

	pBlk = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
	*ppSCache = pSCache;
	uiNumKeys = getBlkEntryCount( (FLMBYTE *)pBlk);
	
	// Determine if the entries are sorted

	pucPrevEntry = (FLMBYTE *)pBlk + m_uiBlockSize;
	bSorted = TRUE;
	uiFirstHole = 0;
	pucHeap = (FLMBYTE *)pBlk + m_uiBlockSize;

	for( uiIndex = 0; uiIndex < uiNumKeys; uiIndex++)
	{
		pucCurEntry = BtEntry( (FLMBYTE *)pBlk, uiIndex);

		if( pucPrevEntry < pucCurEntry)
		{
			bSorted = FALSE;
			break;
		}
		else
		{
			uiAmtToMove = actualEntrySize( 
									getEntrySize( (FLMBYTE *)pBlk, uiIndex));
			pucHeap -= uiAmtToMove;

			if( !uiFirstHole && pucHeap != pucCurEntry)
			{
				uiFirstHole = uiIndex + 1;
			}
		}

		pucPrevEntry = pucCurEntry;
	}
	
	ui16BlkBytesAvail = (FLMUINT16)(m_uiBlockSize - sizeofBTreeBlkHdr( pBlk)) -
							  (FLMUINT16)(uiNumKeys * 2);
	pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlk, 0);
	pucBlkEnd = (FLMBYTE *)pBlk + m_uiBlockSize;

	if( uiFirstHole > 1)
	{
		uiFirstHole--;
		pucHeap = BtEntry( (FLMBYTE *)pBlk, uiFirstHole - 1);
		ui16BlkBytesAvail -= (FLMUINT16)(pucBlkEnd - pucHeap);
	}
	else
	{
		uiFirstHole = 0;
		pucHeap = pucBlkEnd;
	}

	if( !bSorted)
	{
		FLMUINT16 *			pui16OldOffsetArray;

		// If old and new blocks are the same (because of a 
		// prior call to logBlock), we need to save a copy of the block
		// before making changes.

		if( !pOldSCache)
		{
			f_memcpy( m_pucTempDefragBlk, pSCache->m_pBlkHdr, m_uiBlockSize);
			pOldBlk = (F_BTREE_BLK_HDR *)m_pucTempDefragBlk;
		}
		else
		{
			pOldBlk = (F_BTREE_BLK_HDR *)pOldSCache->m_pBlkHdr;
		}

		pui16OldOffsetArray = BtOffsetArray( (FLMBYTE *)pOldBlk, 0);

		// Rebuild the block so that all of the entries are in order

		for( uiIndex = uiFirstHole; uiIndex < uiNumKeys; uiIndex++)
		{
			pucCurEntry = BtEntry( (FLMBYTE *)pOldBlk, uiIndex);
			uiAmtToMove = actualEntrySize( getEntrySize( (FLMBYTE *)pOldBlk, uiIndex));
			pucHeap -= uiAmtToMove;
			bteSetEntryOffset( pui16OffsetArray, uiIndex,
				(FLMUINT16)(pucHeap - (FLMBYTE *)pBlk));
			uiIndex++;

			while( uiIndex < uiNumKeys)
			{
				pucTempEntry = BtEntry( (FLMBYTE *)pOldBlk, uiIndex);
				uiTempToMove = actualEntrySize( getEntrySize( (FLMBYTE *)pOldBlk, uiIndex));

				if ((pucCurEntry - uiTempToMove) != pucTempEntry)
				{
					uiIndex--;
					break;
				}
				else
				{
					pucCurEntry -= uiTempToMove;
					pucHeap -= uiTempToMove;
					uiAmtToMove += uiTempToMove;
					bteSetEntryOffset( pui16OffsetArray, uiIndex, 
							(FLMUINT16)(pucHeap - (FLMBYTE *)pBlk));
					uiIndex++;
				}
			}

			f_memcpy( pucHeap, pucCurEntry, uiAmtToMove);
			ui16BlkBytesAvail -= (FLMUINT16)uiAmtToMove;
		}
	}
	else
	{
		// Work back from the first hole.  Move entries to fill all of the
		// holes in the block.

		for( uiIndex = uiFirstHole; uiIndex < uiNumKeys; uiIndex++)
		{
			pucCurEntry = BtEntry( (FLMBYTE *)pBlk, uiIndex);
			uiAmtToMove = actualEntrySize( getEntrySize( (FLMBYTE *)pBlk, uiIndex));
			pucHeap -= uiAmtToMove;

			if( pucHeap != pucCurEntry)
			{
				// We have a hole.  We don't want to move just one entry
				// if we can avoid it.  We would like to continue searching
				// until we find either the end, or another hole.  Then we
				// can move a larger block of data instead of one entry.

				bteSetEntryOffset( pui16OffsetArray, uiIndex, 
						(FLMUINT16)(pucHeap - (FLMBYTE *)pBlk));
				uiIndex++;

				while( uiIndex < uiNumKeys)
				{
					pucTempEntry = BtEntry( (FLMBYTE *)pBlk, uiIndex);
					uiTempToMove = actualEntrySize( 
											getEntrySize( (FLMBYTE *)pBlk, uiIndex));

					if( (pucCurEntry - uiTempToMove) != pucTempEntry)
					{
						uiIndex--;
						break;
					}
					else
					{
						pucCurEntry -= uiTempToMove;
						pucHeap -= uiTempToMove;
						uiAmtToMove += uiTempToMove;
						bteSetEntryOffset( pui16OffsetArray, 
							uiIndex, (FLMUINT16)(pucHeap - (FLMBYTE *)pBlk));
						uiIndex++;
					}
				}
			}

			// Now move the range we have determined.

			f_memmove( pucHeap, pucCurEntry, uiAmtToMove);
			ui16BlkBytesAvail -= (FLMUINT16)(uiAmtToMove);
		}
	}

	// Set the available space.  If there are no keys in this block, we should
	// set the it to the calculated available space

	if( !uiNumKeys)
	{
		pBlk->stdBlkHdr.ui16BlkBytesAvail = ui16BlkBytesAvail;
	}

	flmAssert( pBlk->stdBlkHdr.ui16BlkBytesAvail == ui16BlkBytesAvail);
	pBlk->ui16HeapSize = ui16BlkBytesAvail;

	// Clean up the heap space.

#ifdef FLM_DEBUG
	f_memset( getBlockEnd( pBlk) - ui16BlkBytesAvail, 0, ui16BlkBytesAvail);
#endif

Exit:

	if( pOldSCache)
	{
		ScaReleaseCache( pOldSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to handle the insertion, deletion and replacment of a single
		entry in a block.
		Assumption:  The find method has already been called to locate the
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_Btree::updateEntry(
	const FLMBYTE *		pucKey,		// In
	FLMUINT					uiKeyLen,	// In
	const FLMBYTE *		pucValue,	// In
	FLMUINT					uiLen,		// In
	F_ELM_UPD_ACTION		eAction,
	FLMBOOL					bTruncate)
{
	RCODE						rc = NE_SFLM_OK;
	const FLMBYTE *		pucRemainingValue = NULL;
	FLMUINT					uiRemainingLen = 0;
	const FLMBYTE *		pucSavKey = pucKey;
	FLMUINT					uiSavKeyLen = uiKeyLen;
	FLMUINT					uiChildBlkAddr = 0;
	FLMUINT					uiCounts = 0;
	FLMUINT					uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;
	FLMBOOL					bMoreToRemove = FALSE;
	FLMBOOL					bDone = FALSE;
	FLMUINT					uiOrigDataLen = uiLen;
	FLMBOOL					bOrigTruncate = bTruncate;

	flmAssert( m_pReplaceInfo == NULL);

	// For each level that needs modifying...
	
	while( !bDone)
	{

		switch( eAction)
		{
			case ELM_INSERT_DO:
			{
				// In this case, the uiLen parameter represents the OADataLength.
				
				uiFlags = BTE_FLAG_DATA_BLOCK |
							 BTE_FLAG_FIRST_ELEMENT |
							 BTE_FLAG_LAST_ELEMENT |
							 BTE_FLAG_OA_DATA_LEN;

				if( RC_BAD( rc = insertEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlkAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction)))
				{
					goto Exit;
				}
				
				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				break;
			}

			case ELM_INSERT:
			{
				// This function will return all info needed to handle the next
				// level up in the Btree (if anything), including setting up
				// the stack.  pucKey & uiKeyLen will be pointing to the key that
				// the upper level needs to insert, replace or delete.
				//
				// It will be pointing to an entry in a lower level block, so that
				// block must not be released until after we are all done.

				if( RC_BAD( rc = insertEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlkAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction)))
				{
					goto Exit;
				}
				
				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				break;
			}
			
			case ELM_REPLACE_DO:
			{
				// In this case, the uiLen parameter represents the OADataLength.
				
				uiFlags = BTE_FLAG_DATA_BLOCK |
							 BTE_FLAG_FIRST_ELEMENT |
							 BTE_FLAG_LAST_ELEMENT |
							 BTE_FLAG_OA_DATA_LEN;

				// Should only get here if we are able to truncate the data.
				
				flmAssert( bTruncate);

				if( RC_BAD( rc = replaceEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlkAddr, &uiCounts,
					&pucRemainingValue, &uiRemainingLen, &eAction)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				bTruncate = TRUE;
				break;
			}
			
			case ELM_REPLACE:
			{
				if( RC_BAD( rc = replaceEntry( &pucKey, &uiKeyLen, pucValue,
					uiLen, uiFlags, &uiChildBlkAddr, &uiCounts, &pucRemainingValue,
					&uiRemainingLen, &eAction, bTruncate)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the Btree.
				
				pucValue = NULL;
				uiLen = 0;
				bTruncate = TRUE;
				break;
			}
			
			case ELM_REMOVE:
			{
				if (RC_BAD( rc = removeEntry( &pucKey, &uiKeyLen, &uiChildBlkAddr,
					&uiCounts, &bMoreToRemove, &eAction)))
				{
					goto Exit;
				}

				// Not needed for upper levels of the B-Tree.
				
				pucValue = NULL;
				uiLen = 0;

				break;
			}
			
			case ELM_DONE:
			{
				if( m_pReplaceInfo)
				{
					// This info structure gets generated when the replaced entry in
					// the upper levels is the last entry in the block and we had to
					// move entries to a previous block to accommodate it.
					// We will therefore need to update the parent block with this
					// new information. We need to take care of this before we check
					// for any additional data to store.
					
					if( RC_BAD( rc = restoreReplaceInfo( &pucKey, &uiKeyLen,
						&uiChildBlkAddr, &uiCounts)))
					{
					  goto Exit;
					}
					
					bTruncate = bOrigTruncate;
					eAction = ELM_REPLACE;
				}
				else if( bMoreToRemove)
				{
					eAction = ELM_REMOVE;
					
					// We need to locate where we should remove the entry.
					
					if( RC_BAD( rc = findEntry( pucSavKey, uiSavKeyLen, FLM_EXACT)))
					{
						goto Exit;
					}

				}
				else if( pucRemainingValue && uiRemainingLen)
				{
					eAction = ELM_INSERT;
					
					// We need to locate where we should insert the new entry.
					
					rc = findEntry( pucSavKey, uiSavKeyLen, FLM_EXCL);

					// We could find this entry.  If we get back anything other than
					// an NE_SFLM_EOF_HIT or NE_SFLM_OK, then there is a problem.
					
					if( rc != NE_SFLM_OK && rc != NE_SFLM_EOF_HIT &&
						 rc != NE_SFLM_NOT_FOUND)
					{
						goto Exit;
					}

					pucValue = pucRemainingValue;
					uiLen = uiRemainingLen;
					pucKey = pucSavKey;
					uiKeyLen = uiSavKeyLen;

					// Make certain that the  BTE_FIRST_ELEMENT flag is NOT set if
					// the first part of the data was stored.
					
					if( uiOrigDataLen != uiLen)
					{
						uiFlags = BTE_FLAG_LAST_ELEMENT;
					}
					else
					{
						uiFlags = BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT;
					}
				}
				else
				{
					bDone = TRUE;
				}

				break;
			}
			
			// Should never get this!
			
			case ELM_BLK_MERGE:
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will coordinate inserting an entry into a block.  If it
		cannot fit it all in, then it may have to break the entry up so that
		it spans more than one block.  It will also setup for the next level
		before returning.
****************************************************************************/
RCODE F_Btree::insertEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlkAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_SFLM_OK;
	const FLMBYTE *		pucDataValue = pucValue;
	FLMUINT					uiDataLen = uiLen;
	FLMUINT					uiOADataLen = 0;
	FLMUINT					uiEntrySize = 0;
	FLMBOOL					bEntriesWereMoved = FALSE;
	FLMBOOL					bHaveRoom;
	FLMBOOL					bLastEntry;
	const FLMBYTE *		pucKey = *ppucKey;
	FLMUINT					uiKeyLen = *puiKeyLen;
	FLMUINT					uiChildBlkAddr = *puiChildBlkAddr;
	FLMUINT					uiCounts = *puiCounts;
	F_CachedBlock *		pPrevSCache = NULL;
	FLMBYTE *				pucEntry;
	FLMBOOL					bDefragBlk = FALSE;
	FLMBOOL					bBlockSplit;

	if( m_pStack->uiLevel == 0)
	{
		// We are only safe to do this when we are working on level 0
		// (leaf level) of the Btree.
		
		*ppucRemainingValue = NULL;
		*puiRemainingLen = 0;
	}

	if( *peAction == ELM_INSERT_DO)
	{
		// Adjust the data entry sizes as the data passed in is the 
		// OA Data Length.
		
		uiOADataLen = uiLen;
		uiDataLen = 4;
	}

	// Process until we are done

StartOver:

	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiDataLen, &uiEntrySize,
			&bHaveRoom, &bDefragBlk)))
	{
		goto Exit;
	}

	// Does the entry fit into the block?
	
	if( bHaveRoom)
	{
		if( bDefragBlk)
		{
			// We will have to defragment the block before we can store the data
			
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		if( (bLastEntry || m_bCounts) && !isRootBlk( m_pStack->pBlkHdr))
		{
			// Are we in here because of the counts only?  If so, then we
			// can update the counts right here, no need to continue.
			
			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Ensure we are updating with the correct key.
				
				pucEntry = BtLastEntry( (FLMBYTE *)m_pStack->pBlkHdr);

				*puiKeyLen = getEntryKeyLength( pucEntry,
									m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, ppucKey);

				*puiChildBlkAddr = m_pStack->ui32BlkAddr;

				// Do we need counts for the next level?
				
				if( m_bCounts)
				{
					*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// Can we move entries around at all to make some room?
	
	if( RC_BAD( rc = moveEntriesToPrevBlk(  uiEntrySize, &pPrevSCache,
			&bEntriesWereMoved)))
	{
		goto Exit;
	}

	if( bEntriesWereMoved)
	{
		// Only defragment the block if the heap size is not big enough.
		
		if( uiEntrySize > m_pStack->pBlkHdr->ui16HeapSize)
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
			{
				goto Exit;
			}
		}
		
		// Store the entry now because we know there is enough room
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		// Ordinarily, this would NEVER be the last element in the
		// block because we need to adjust the stack to take care of the
		// elements we just moved!  There is only one condition where we would
		// insert as the last entry in the block, and that is when this
		// insert is actually a part of a replace operation where the data
		// is too large to fit in the block.  We had to remove the entry, then
		// insert the new one and we are in the upper levels of the
		// btree. (i.e. not at the leaf).
		//
		// VISIT:  Should I assert that we are not at the leaf level 
		// if we get in here?
		
		if( bLastEntry)
		{
			// Since we just added an entry to the last position of the
			// current block.  We will need to preserve the current stack so
			// that we can finish updating the parentage later. Should only
			// happen as a result of a replace operation where the new entry
			// is larger than the existing one while in the upper levels.
			
			if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
			{
				goto Exit;
			}
		}

		// Need to update the counts of the parents if we are maintining
		// counts before we abandon
		
		if( m_bCounts)
		{
			if( RC_BAD( rc = updateCounts()))
			{
				goto Exit;
			}
		}

		// This method will release any blocks no longer referenced
		// in the stack.  Then pull in the previous block information into
		// the stack.
		
		if( RC_BAD( rc = moveStackToPrev( pPrevSCache)))
		{
			goto Exit;
		}

		// If we are maintaining counts, then lets return a count of the
		// current number of keys referenced below this point.
		
		if( m_bCounts)
		{
			*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
		}

		flmAssert( !isRootBlk( m_pStack->pBlkHdr));

		// Return the key to the last entry in the prevous block.
		// Recall that we have changed that stack now so that it
		// is referencing the changed block (pPrevSCache).
		
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);
								  
		*puiKeyLen = getEntryKeyLength( pucEntry,
								pPrevSCache->m_pBlkHdr->ui8BlkType, ppucKey);

		// Return the new child block address
		
		*puiChildBlkAddr = m_pStack->ui32BlkAddr;

		// Set up to fixup the parentage of the previous block on return...
		
		m_pStack++;

		// Return the new action for the parent block.
		
		*peAction = ELM_REPLACE;
		goto Exit;
	}

	// Try moving to the next block...
	
	if( RC_BAD( rc = moveEntriesToNextBlk( uiEntrySize, &bEntriesWereMoved)))
	{
		goto Exit;
	}

	if( bEntriesWereMoved)
	{
		// Only defragment the block if the heap size is not big enough.
		
		if( uiEntrySize > m_pStack->pBlkHdr->ui16HeapSize)
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
			{
				goto Exit;
			}
		}

		// Store the entry now because we know there is enough room
		
		if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucDataValue,
				uiDataLen, uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts,
				uiEntrySize, &bLastEntry)))
		{
			goto Exit;
		}

		// Return the key to the last entry in the current block.
		// Note: If bLastEntry is TRUE, we already know what the key is.
		
		if( !bLastEntry)
		{
			// Get the last key from the block.
			
			pucEntry = BtLastEntry( (FLMBYTE *)m_pStack->pBlkHdr);

			*puiKeyLen = getEntryKeyLength( pucEntry,
									m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, ppucKey);

		}

		flmAssert( !isRootBlk( m_pStack->pBlkHdr));

		// if we are maintaining counts, then lets return a count of the
		// current number of keys referenced below this point.
		
		if( m_bCounts)
		{
			*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
		}

		// Return the new child block address
		
		*puiChildBlkAddr = m_pStack->ui32BlkAddr;

		// Set up to fixup the parentage of the this block on return...
		
		m_pStack++;
		*peAction = ELM_REPLACE;

		goto Exit;
	}

	// Before we incur the expense of a block split, see if we can store this
	// entry in the previous block.  If we can, we will save some space.  This
	// will only happen if we are trying to insert at the first position in
	// this block.  We would only ever get into this block of code once for
	// each level of the btree.

	if( m_pStack->uiCurOffset == 0 && 
		  m_pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
				m_pDb, m_pLFile, m_pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain,
				NULL, &pPrevSCache)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = moveStackToPrev( pPrevSCache)))
		{
			goto Exit;
		}

		// Increment so we point to one past the last entry.

		m_pStack->uiCurOffset++;
		goto StartOver;
	}

	// We will have to split the block to make room for this entry.
	
	if( RC_BAD( rc = splitBlock( *ppucKey, *puiKeyLen, pucDataValue,
			uiDataLen, uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts,
			ppucRemainingValue, puiRemainingLen, &bBlockSplit)))
	{
		goto Exit;
	}

	// Return the new key value.
	
	pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

	*puiKeyLen = getEntryKeyLength( pucEntry,
						((F_BLK_HDR *)m_pStack->pBlkHdr)->ui8BlkType, ppucKey);

	// Return the child block address and the counts (if needed).
	
	*puiChildBlkAddr = m_pStack->ui32BlkAddr;

	// Return the counts if we are maintaining them
	
	if( m_bCounts)
	{
		*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
	}

	// The bBlockSplit boolean will only be FALSE if we were involved in a
	// ReplaceByInsert operation and the call to split resulted in an empty
	// block.  Thus we were able to store the new entry.  In such cases,
	// only the count (if any) need to be updated, not the keys.
	
	if( bBlockSplit)
	{
		*peAction = ELM_INSERT;
		m_pStack++;
	}
	else
	{
		*peAction = ELM_DONE;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to handle the insertion of a single entry into a block.
		Assumption:  The find method has already been called to locate the 
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_Btree::storeEntry(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT					uiChildBlkAddr,
	FLMUINT					uiCounts,
	FLMUINT					uiEntrySize,
	FLMBOOL *				pbLastEntry)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiBlkType = m_pStack->pSCache->m_pBlkHdr->ui8BlkType;
	FLMBYTE *				pucInsertAt;
	FLMUINT16 *				pui16OffsetArray;
	FLMUINT					uiNumKeys;
	FLMUINT					uiTmp;
	F_BTREE_BLK_HDR *		pBlk;

	// Assume this is not the last entry for now.
	// We will change it later if needed.

	*pbLastEntry = FALSE;

	// We can go ahead and insert this entry as it is.  All checking has been
	// made before getting to this point.

	uiEntrySize = calcEntrySize( uiBlkType, uiFlags,
										  uiKeyLen, uiLen, uiOADataLen);

	// Log this block before making any changes to it.  Since the
	// pSCache could change, we must update the block header after the call.

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}

	pBlk = m_pStack->pBlkHdr =	(F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlk, 0);
	uiNumKeys = getBlkEntryCount( (FLMBYTE *)pBlk);
	pucInsertAt = getBlockEnd( pBlk) - uiEntrySize;
	pui16OffsetArray = m_pStack->pui16OffsetArray;

	if( RC_BAD( rc = buildAndStoreEntry( uiBlkType, uiFlags, pucKey, uiKeyLen,
		pucValue, uiLen, uiOADataLen, uiChildBlkAddr, uiCounts,
		pucInsertAt, uiEntrySize, NULL)))
	{
		goto Exit;
	}

	// Now to update the offset in the offset array.  This will move all
	// entries that sort after the new entry down by one position.

	for( uiTmp = uiNumKeys; uiTmp > m_pStack->uiCurOffset; uiTmp--)
	{
		bteSetEntryOffset( pui16OffsetArray, uiTmp,
								 bteGetEntryOffset( pui16OffsetArray, uiTmp - 1));
	}

	bteSetEntryOffset( pui16OffsetArray, m_pStack->uiCurOffset,
							 (FLMUINT16)(pucInsertAt - (FLMBYTE *)pBlk));

	// Update the available space and the number of keys.
	// Account for the new offset entry too.

	m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail -= 
			(FLMUINT16)(uiEntrySize + 2);
	m_pStack->pBlkHdr->ui16HeapSize -= (FLMUINT16)(uiEntrySize + 2);
	m_pStack->pBlkHdr->ui16NumKeys++;

	// Check to see if this was the last entry

	if( m_pStack->uiCurOffset == (FLMUINT)(m_pStack->pBlkHdr->ui16NumKeys - 1))
	{
		*pbLastEntry = TRUE;
	}

	if( !m_pStack->uiLevel && (uiFlags & BTE_FLAG_FIRST_ELEMENT))
	{
		m_ui32PrimaryBlkAddr = m_pStack->ui32BlkAddr;
		m_uiCurOffset = m_pStack->uiCurOffset;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method will coordinate removing an entry from a block. If the
		entry spans more than one block, it will set the flag pbMoreToRemove.
		It will also setup for the next level before returning.
****************************************************************************/
RCODE F_Btree::removeEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	FLMUINT *				puiChildBlkAddr,
	FLMUINT *				puiCounts,
	FLMBOOL *				pbMoreToRemove,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bLastEntry = FALSE;
	FLMBYTE *		pucEntry;
	FLMBOOL			bMergedWithPrev = FALSE;
	FLMBOOL			bMergedWithNext = FALSE;

	if( m_pStack->uiLevel == 0)
	{
		// We are only safe to do this when we are working on level 0
		// (leaf level) of the Btree.
		
		*pbMoreToRemove = FALSE;
	}

	// Check the current entry to see if it spans more than a single block.

	pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
							  m_pStack->uiCurOffset);

	// We only need to worry about data spanning more than one block if it is
	// at level zero (i.e. leaf block) and the lastElement flag is not set.

	if( (m_pStack->uiLevel == 0) && m_bData && !bteLastElementFlag( pucEntry))
	{
		*pbMoreToRemove = TRUE;
	}

	// Find out if we are looking at the last entry in the block.

	if( m_pStack->uiCurOffset == (FLMUINT)(m_pStack->pBlkHdr->ui16NumKeys - 1))
	{
		bLastEntry = TRUE;
	}

	// Now we remove the entry... Will also remove any chained Data Only blocks

	if( RC_BAD( rc = remove( TRUE)))
	{
		goto Exit;
	}

	// If the block is now empty, we will free the block.

	if( !m_pStack->pBlkHdr->ui16NumKeys)
	{
		FLMBOOL			bIsRoot;

		// Test for root block.

		bIsRoot = isRootBlk( m_pStack->pBlkHdr);

		if( RC_BAD( rc = deleteEmptyBlock()))
		{
			goto Exit;
		}

		// Need to remove the parent entry referencing the deleted block.

		if( !bIsRoot)
		{
			*peAction = ELM_REMOVE;
			m_pStack++;
		}
		else
		{
			// If we ever get here, it means we have just deleted the root block.
			// I have put in the possibility, but typically, deleting the Btree
			// is done by calling btDeleteTree.

			*peAction = ELM_DONE;
		}
	}
	else
	{
		if( ((((F_BLK_HDR *)m_pStack->pBlkHdr)->ui16BlkBytesAvail * 100) /
													m_uiBlockSize) >= BT_LOW_WATER_MARK)
		{
			// We will need to check to see if we can merge two blocks into one to
			// conserve space.

			if( RC_BAD( rc = mergeBlocks( bLastEntry, &bMergedWithPrev,
				&bMergedWithNext, peAction)))
			{
				goto Exit;
			}
		}

		// If the entry that we just removed was the last entry in the block and
		// we did not merge any blocks, we will need to prep for an update to the
		// parent with a new key.

		if( bLastEntry && !bMergedWithPrev && !bMergedWithNext)
		{
			if( m_bCounts)
			{
				*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
			}

			// Backup to the new "last" entry (remove() does not adjust the offset
			// in the stack).

			flmAssert( m_pStack->uiCurOffset > 0);

			m_pStack->uiCurOffset--;
			pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
									  m_pStack->uiCurOffset);

			*puiKeyLen = getEntryKeyLength( pucEntry,
								m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, ppucKey);

			*puiChildBlkAddr = m_pStack->ui32BlkAddr;
			*peAction = ELM_REPLACE;
			m_pStack++;
		}
		else
		{
			// Are we tracking counts?

			if( !bMergedWithPrev && !bMergedWithNext)
			{
				if( m_bCounts)
				{
					if( RC_BAD( rc = updateCounts()))
					{
						goto Exit;
					}
				}
				
				*peAction = ELM_DONE;
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to replace an existing entry with a new one.
****************************************************************************/
RCODE F_Btree::replaceEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlkAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction,
	FLMBOOL					bTruncate)
{
	RCODE					rc = NE_SFLM_OK;
	const FLMBYTE *	pucDataValue = pucValue;
	FLMUINT				uiDataLen = uiLen;
	FLMUINT				uiOADataLen = 0;
	FLMBYTE *			pucEntry = NULL;
	FLMUINT32			ui32OrigDOAddr = 0;
	const FLMBYTE *	pucData = NULL;

	if( m_pStack->uiLevel == 0)
	{
		*ppucRemainingValue = NULL;
		*puiRemainingLen = 0;
	}

	if( *peAction == ELM_REPLACE_DO)
	{
		// Adjust the data entry sizes as the data passed in
		// is the OA Data Length.
		
		uiOADataLen = uiLen;
		uiDataLen = 4;
	}

	if( m_pStack->uiLevel == 0 && m_bData)
	{
		if( m_bOrigInDOBlocks)
		{
			flmAssert( bTruncate);

			pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
									  m_pStack->uiCurOffset);

			btGetEntryDataLength( pucEntry, &pucData, NULL, NULL);
			ui32OrigDOAddr = bteGetBlkAddr( pucData);
		}
	}

	// We only have to worry about updating the upper levels of the Btree
	// when we are doing a replacement at a non-leaf level or we are maintaining
	// counts.  Replacements at the leaf level do not require a change in the
	// parent block.  The only exception is when the old entry spanned to
	// another block, but the new one did not.  This results in removing the
	// excess part of the old entry unless we are not truncating the element.
	// Even then, we only update the parent if the excess entry was the only key
	// in the block, i.e. the block became empty as a result of the removal.
	// All of this would have been handled already by the time we return from
	// this call.

	// When bTruncate is FALSE we do not trim back the entry so we don't worry
	// about updating the parentage.
	
	if( RC_BAD( rc = replaceOldEntry( ppucKey, puiKeyLen, pucDataValue,
		uiDataLen, uiFlags, uiOADataLen, puiChildBlkAddr, puiCounts,
		ppucRemainingValue, puiRemainingLen, peAction, bTruncate)))
	{
		goto Exit;
	}

	// Do we need to free the original DO blocks since they are not
	// used in the new entry?

	if( m_bOrigInDOBlocks && !m_bDataOnlyBlock && m_pStack->uiLevel == 0)
	{
		if( RC_BAD( rc = removeDOBlocks( ui32OrigDOAddr)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to handle replacing a single entry in a block.
		ASSUMPTION:  The find method has already been called to locate the
		insertion point, so the stack has already been setup.
****************************************************************************/
RCODE F_Btree::replaceOldEntry(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiOADataLen,
	FLMUINT *				puiChildBlkAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction,
	FLMBOOL					bTruncate)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiOldEntrySize;
	FLMBYTE *			pucEntry = NULL;
	FLMBYTE *			pucData = NULL;
	FLMUINT				uiEntrySize;
	FLMBOOL				bLastEntry = FALSE;
	FLMBOOL				bLastElement = TRUE;
	FLMBOOL				bHaveRoom;
	FLMBOOL				bDefragBlk;
	FLMUINT				uiDataLen = 0;
	FLMUINT				uiOldOADataLen = 0;
	FLMBOOL				bRemoveOADataAllowance = FALSE;

	uiOldEntrySize = actualEntrySize(
								getEntrySize( (FLMBYTE *)m_pStack->pBlkHdr,
												  m_pStack->uiCurOffset, &pucEntry));

	if( m_pStack->uiLevel == 0 && m_bData)
	{
		bLastElement = bteLastElementFlag( pucEntry);

		uiDataLen = btGetEntryDataLength( pucEntry, (const FLMBYTE **)&pucData, 
			&uiOldOADataLen, NULL);

		// Test to see if we need to worry about the bTruncate flag.

		if( uiDataLen == uiOldOADataLen)
		{
			if( uiLen > uiDataLen)
			{
				bTruncate = TRUE;
			}
			else if( uiLen <= uiDataLen && uiOADataLen == 0)
			{
				bRemoveOADataAllowance = TRUE;
			}
		}
		else
		{
			if( uiLen > uiOldOADataLen)
			{
				bTruncate = TRUE;
			}
		}
	}

	// bTruncate has no meaning if we have no data or we are not at the
	// leaf level.

	if( m_pStack->uiLevel != 0 || !m_bData)
	{
		bTruncate = TRUE;
	}

	// The calcNewEntrySize function will tack on 2 bytes for the offset.
	// It also adds an extra 4 bytes for the OADataLen, even though it may
	// not be needed.  We will need to be aware of this here as it may affect
	// our decision as to how we will replace the entry.

	if( RC_BAD( rc = calcNewEntrySize( *puiKeyLen, uiLen, &uiEntrySize,
			&bHaveRoom, &bDefragBlk)))
	{
		goto Exit;
	}

	if( bRemoveOADataAllowance)
	{
		uiEntrySize -= 4;
	}

	// Since this is a replace operation, we don't need to know about the offset
	// as that won't be a factor in what we are doing. 'actualEntrySize' will
	// remove those two bytyes from the size.

	uiEntrySize = actualEntrySize( uiEntrySize);
	if( uiEntrySize <= uiOldEntrySize)
	{
		if( !bTruncate)
		{
			flmAssert( uiLen <= uiDataLen);
			f_memcpy( pucData, pucValue, uiLen);

			if( m_pStack->uiCurOffset == 
					(FLMUINT)(m_pStack->pBlkHdr->ui16NumKeys - 1))
			{
				bLastEntry = TRUE;
			}
		}
		else
		{
			// We can go ahead and replace this entry as it is.  All checking
			// has been made before getting to this point.

			if( RC_BAD( rc = buildAndStoreEntry( 
				m_pStack->pSCache->m_pBlkHdr->ui8BlkType,
				uiFlags, *ppucKey, *puiKeyLen, pucValue, uiLen, uiOADataLen,
				*puiChildBlkAddr, *puiCounts, m_pucTempBlk, m_uiBlockSize,
				&uiEntrySize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = replace( m_pucTempBlk, uiEntrySize, &bLastEntry)))
			{
				goto Exit;
			}
		}

		if( !bLastElement && bTruncate)
		{
			// The element that we replaced actually spans more than one entry.
			// We will have to remove the remaining entries.

			if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
			{
				goto Exit;
			}
		}

		if( (bLastEntry || m_bCounts) && !isRootBlk( m_pStack->pBlkHdr) &&
			  (m_pStack->uiLevel != 0))
		{
			// Are we in here because of the counts only?  If so, then make
			// sure we don't change the key in the parent.

			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Return the key to the last entry in the block.

				pucEntry = BtLastEntry( (FLMBYTE *)m_pStack->pBlkHdr);

				*puiKeyLen = getEntryKeyLength( pucEntry,
									m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, ppucKey);
				*puiChildBlkAddr = m_pStack->ui32BlkAddr;

				// Do we need counts for the next level?

				if( m_bCounts)
				{
					*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// If we do not have a stack setup yet (which can happen if the replace
	// is trying to shortcut to the previously known block address and offset),
	// then at this point, we must build the stack, since it may be required
	// to adjust the upper levels of the btree.

	if( !m_bStackSetup)
	{
		if( RC_BAD( rc = findEntry( *ppucKey, *puiKeyLen, FLM_EXACT)))
		{
			goto Exit;
		}
	}

	// The new entry will not fit into the original entry's space.
	// If we remove the entry in the block, will there be enough room
	// to put it in?

	if( bTruncate &&
		 m_pStack->pSCache->m_pBlkHdr->ui16BlkBytesAvail +
		 uiOldEntrySize >= uiEntrySize)
	{
		// First remove the current entry.  Do not delete any DO blocks chained
		// to this entry.

		if( RC_BAD( rc = remove( FALSE)))
		{
			goto Exit;
		}

		if( (m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail !=
			 m_pStack->pBlkHdr->ui16HeapSize) &&
			 ((uiEntrySize + 2) > m_pStack->pBlkHdr->ui16HeapSize))
		{
			if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
			{
				goto Exit;
			}
		}

		// Now insert the new entry.

		if( RC_BAD( rc = storeEntry( *ppucKey, *puiKeyLen, pucValue, uiLen,
				uiFlags, uiOADataLen, *puiChildBlkAddr, *puiCounts, uiEntrySize,
				&bLastEntry)))
		{
			goto Exit;
		}

		// Check if the original element spanned more than one entry

		if( !bLastElement)
		{
			// The element that we replaced actually spans more than one entry.
			// We will have to remove the remaining entries.

			if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
			{
				goto Exit;
			}
		}

		if( (bLastEntry || m_bCounts) && !isRootBlk( m_pStack->pBlkHdr) &&
			 (m_pStack->uiLevel != 0))
		{
			// Are we in here because of the counts only?

			if( !bLastEntry)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
				
				*peAction = ELM_DONE;
			}
			else
			{
				// Set the key to the last entry in the block.

				pucEntry = BtLastEntry( (FLMBYTE *)m_pStack->pBlkHdr);

				*puiKeyLen = getEntryKeyLength( pucEntry,
									m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, ppucKey);
				*puiChildBlkAddr = m_pStack->ui32BlkAddr;

				// Do we need counts for the next level?

				if( m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType == BT_NON_LEAF_COUNTS)
				{
					*puiCounts = countKeys( (FLMBYTE *)m_pStack->pBlkHdr);
				}
				
				m_pStack++;
				*peAction = ELM_REPLACE;
			}
		}
		else
		{
			*peAction = ELM_DONE;
		}
		
		goto Exit;
	}

	// If the original element does not span multiple entries and we still don't
	// have room for the replacement, then we will remove this entry and insert
	// the replacement.  When the insert happens, it will take care of moving
	// things around or splitting the block as needed to get it in.  If bTruncate
	// is FALSE, and the new entry is larger than the original, we can ignore it.

	if( bLastElement)
	{
		if( RC_BAD( rc = replaceByInsert( ppucKey, puiKeyLen,
			pucValue, uiLen, uiOADataLen, uiFlags, puiChildBlkAddr,
			puiCounts, ppucRemainingValue, puiRemainingLen,
			peAction)))
		{
			goto Exit;
		}

		goto Exit;
	}

	if( bTruncate)
	{
		if( RC_BAD( rc = replaceMultiples( ppucKey, puiKeyLen, pucValue,
			uiLen, uiFlags, puiChildBlkAddr, puiCounts, ppucRemainingValue,
			puiRemainingLen, peAction)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = replaceMultiNoTruncate( ppucKey, puiKeyLen,
			pucValue, uiLen, uiFlags, puiChildBlkAddr, puiCounts,
			ppucRemainingValue, puiRemainingLen, peAction)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This method is called whenever a replacement entry will not fit in
		the block, even if we remove the existing entry.  It ASSUMES that the
		original element does not continue to another entry, either in the
		same block or in another block.
****************************************************************************/
RCODE F_Btree::replaceByInsert(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiDataLen,
	FLMUINT					uiOADataLen,
	FLMUINT					uiFlags,
	FLMUINT *				puiChildBlkAddr,
	FLMUINT *				puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiLen = uiDataLen;

	if( *peAction == ELM_REPLACE_DO)
	{
		uiLen = uiOADataLen;
		*peAction = ELM_INSERT_DO;
	}
	else
	{
		*peAction = ELM_INSERT;
	}

	// At this point, it is clear that this new entry is larger than the
	// old entry.  We will remove the old entry first.  Then we can treat
	// this whole operation as an insert rather than as a replace.

	if( RC_BAD( rc = remove( FALSE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = insertEntry( ppucKey, puiKeyLen, pucDataValue, uiLen,
		uiFlags, puiChildBlkAddr, puiCounts, ppucRemainingValue, puiRemainingLen,
		peAction)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to replace an entry in a block and update the available
		space.  This method expects to receive a buffer with an entry already
		prepared to be written to the block.
****************************************************************************/
RCODE F_Btree::replace(
	FLMBYTE *		pucEntry,
	FLMUINT			uiEntrySize,
	FLMBOOL *		pbLastEntry)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBYTE *				pucReplaceAt;
	FLMUINT					uiNumKeys;
	FLMBYTE *				pBlk;
	FLMUINT					uiOldEntrySize;

	*pbLastEntry = FALSE;

	// Log this block before making any changes to it.  Since the
	// pSCache could change, we must update the block header after the call.

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}
	
	m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
	pBlk = (FLMBYTE *)m_pStack->pBlkHdr;
	m_pStack->pui16OffsetArray = BtOffsetArray( pBlk, 0);

	uiNumKeys = getBlkEntryCount( pBlk);

	uiOldEntrySize = actualEntrySize( getEntrySize( 
														pBlk, m_pStack->uiCurOffset));

	flmAssert( uiOldEntrySize >= uiEntrySize);

	pucReplaceAt = BtEntry( pBlk, m_pStack->uiCurOffset);

	// Let's go ahead and copy the entry into the block now.

	f_memcpy( pucReplaceAt, pucEntry, uiEntrySize);

#ifdef FLM_DEBUG
	// Clean up the empty space (if any)

	if( uiOldEntrySize > uiEntrySize)
	{
		pucReplaceAt += uiEntrySize;
		f_memset( pucReplaceAt, 0, uiOldEntrySize - uiEntrySize);
	}
#endif

	// Update the available space.  It may not have changed at all if the
	// two entries are the same size. The Heap size will not have changed.
	// This is because we write the entry into the same location as the
	// original.  Even though the new entry may be smaller, we start at
	// the same location, possibly leaving a hole in the block.

	m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail +=
									(FLMUINT16)(uiOldEntrySize - uiEntrySize);


	if( m_pStack->uiCurOffset == (FLMUINT)(m_pStack->pBlkHdr->ui16NumKeys - 1))
	{
		*pbLastEntry = TRUE;
	}

	// Preserve the block and offset index in case it is wanted on the way out.

	if( !m_pStack->uiLevel && bteFirstElementFlag( pucEntry))
	{
		m_ui32PrimaryBlkAddr = m_pStack->ui32BlkAddr;
		m_uiCurOffset = m_pStack->uiCurOffset;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to rebuild the stack so that it references the parentage of
		the parameter pSCache block.  The assumption is that we will begin at
		whatever level m_pStack is currently sitting at. Therefore, this
		method can be called for any level in the Btree.
****************************************************************************/
RCODE F_Btree::moveStackToPrev(
	F_CachedBlock *	pSCache)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiBlkAddr;
	F_BTREE_BLK_HDR *	pBlkHdr;
	F_BTSK *				pStack = m_pStack;
	F_CachedBlock *	pPrevSCache = NULL;

	if( pSCache)
	{
		if( pStack->pSCache)
		{
			// Make sure the block we passed in really is the previous
			// block in the chain.

			if( pSCache->m_uiBlkAddress !=
					pStack->pSCache->m_pBlkHdr->ui32PrevBlkInChain)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			// Cannot be the same block.

			if( pSCache == pStack->pSCache)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			// Release the current block.  We don't need to fetch
			// the new block because it was passed in to us.  If
			// we encounter this situation further up the tree,
			// we will have to fetch the block as well.

			ScaReleaseCache( pStack->pSCache, FALSE);
		}

		pStack->pSCache = pSCache;
		pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
		pStack->pBlkHdr = pBlkHdr;
		pStack->ui32BlkAddr = ((F_BLK_HDR *)pBlkHdr)->ui32BlkAddr;
		pStack->uiCurOffset = pBlkHdr->ui16NumKeys - 1;  // Last entry
		pStack->uiLevel =	 pBlkHdr->ui8BlkLevel;
		pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

		// Now walk up the stack until done.

		pStack++;
	}

	for (;;)
	{
		// If we don't have this block in the stack, we must first get it.

		if( pStack->pSCache == NULL)
		{
			// Don't continue if we don't have this level in the stack.

			if( pStack->ui32BlkAddr == 0)
			{
				break;
			}

			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				pStack->ui32BlkAddr, NULL, &pStack->pSCache)))
			{
				goto Exit;
			}
			
			pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pStack->pSCache->m_pBlkHdr;
		}

		// See if we need to go to the previous block.

		if( pStack->uiCurOffset == 0)
		{
			// If this is the root block and we are looking at the first
			// entry in the block, then we have a problem.

			if( !isRootBlk( pStack->pBlkHdr))
			{
				// When the stack is pointing to the first entry, this
				// means that we want the target stack to point to the previous
				// block in the chain.

				uiBlkAddr = pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain;
				flmAssert( uiBlkAddr);

				// Fetch the new block

				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					uiBlkAddr, NULL, &pPrevSCache)))
				{
					goto Exit;
				}

				// Release the old block

				ScaReleaseCache( pStack->pSCache, FALSE);

				pBlkHdr = (F_BTREE_BLK_HDR *)pPrevSCache->m_pBlkHdr;
				pStack->pSCache = pPrevSCache;
				pPrevSCache = NULL;
				pStack->pBlkHdr = pBlkHdr;
				pStack->ui32BlkAddr = ((F_BLK_HDR *)pBlkHdr)->ui32BlkAddr;
				pStack->uiCurOffset = pBlkHdr->ui16NumKeys - 1;  // Last Entry
				pStack->uiLevel = pBlkHdr->ui8BlkLevel;
				pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);
			}
			else
			{
				// We have no previous.

				rc = RC_SET( NE_SFLM_BOF_HIT);
				goto Exit;
			}
		}
		else
		{
			pStack->uiCurOffset--;  // Previous Entry
			break;
		}
		
		pStack++;
	}

Exit:

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to rebuild the stack so that it references the parentage of
		the parameter pSCache block. The assumption is that we will begin at
		whatever level m_pStack is currently sitting at.  Therefore, this 
		method can be called for any level in the Btree.
****************************************************************************/
RCODE F_Btree::moveStackToNext(
	F_CachedBlock *	pSCache,
	FLMBOOL				bReleaseCurrent)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiBlkAddr;
	F_BTREE_BLK_HDR *	pBlkHdr;
	F_BTSK *				pStack = m_pStack;
	F_CachedBlock *	pNextSCache = NULL;

	if( pSCache)
	{
		if( pStack->pSCache)
		{
			// Make sure the block we passed in really is the next in chain.

			if( pSCache->m_uiBlkAddress !=
					pStack->pSCache->m_pBlkHdr->ui32NextBlkInChain)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			// Cannot be the same block.

			if( pSCache == pStack->pSCache)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			// Release the current block.  We don't need to fetch
			// the new block because it was passed in to us.  If
			// we encounter this situation further up the tree,
			// we will have to fetch the block as well.

			if( bReleaseCurrent)
			{
				ScaReleaseCache( pStack->pSCache, FALSE);
			}
		}

		pStack->pSCache = pSCache;
		pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
		pStack->pBlkHdr = pBlkHdr;
		pStack->ui32BlkAddr = ((F_BLK_HDR *)pBlkHdr)->ui32BlkAddr;
		pStack->uiCurOffset = 0;  // First entry
		pStack->uiLevel = pBlkHdr->ui8BlkLevel;
		pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);

		// Now walk up the stack until done.

		pStack++;
	}

	for (;;)
	{
		// If we don't currently have this block, let's get it.

		if( pStack->pSCache == NULL)
		{
			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				pStack->ui32BlkAddr, NULL, &pStack->pSCache)))
			{
				goto Exit;
			}
			
			pStack->pBlkHdr = (F_BTREE_BLK_HDR *)pStack->pSCache->m_pBlkHdr;
		}

		// See if we need to go to the next block.

		if( pStack->uiCurOffset == (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1))
		{
			// If this is the root block and we are looking at the last entry in the
			// block, then we have a problem.

			if( !isRootBlk( pStack->pBlkHdr))
			{
				// When the stack is pointing to the last entry, this
				// means that we want the target stack to point the next block in
				// the chain.

				uiBlkAddr = pStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain;
				flmAssert( uiBlkAddr);

				// Get the next block

				if( RC_BAD( rc = getNextBlock( &pStack->pSCache)))
				{
					goto Exit;
				}

				pBlkHdr = (F_BTREE_BLK_HDR *)pStack->pSCache->m_pBlkHdr;
				pStack->pBlkHdr = pBlkHdr;
				pStack->ui32BlkAddr = ((F_BLK_HDR *)pBlkHdr)->ui32BlkAddr;
				pStack->uiCurOffset = 0;   // First Entry
				pStack->uiLevel = pBlkHdr->ui8BlkLevel;
				pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pBlkHdr, 0);
			}
			else
			{
				// We should never have to attempt to get a previous block
				// on the root.

				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}
		}
		else
		{
			pStack->uiCurOffset++;  // Next Entry
			break;
		}
		
		pStack++;
	}

Exit:

	if( pNextSCache)
	{
		ScaReleaseCache( pNextSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to calculate the actual entry size of a new entry
****************************************************************************/
RCODE F_Btree::calcNewEntrySize(
	FLMUINT			uiKeyLen,
	FLMUINT			uiDataLen,
	FLMUINT *		puiEntrySize,
	FLMBOOL *		pbHaveRoom,
	FLMBOOL *		pbDefragBlk)
{
	RCODE				rc = NE_SFLM_OK;

	// Calculate the entry size.

	switch( m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType)
	{
		case BT_LEAF:
		{
			// This block type is a leaf block, No Data
			
			*puiEntrySize = BTE_LEAF_OVHD + uiKeyLen;
			break;
		}

		case BT_LEAF_DATA:
		{
			// Leaf block with data
			
			*puiEntrySize = BTE_LEAF_DATA_OVHD +
								 (uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +
								 (uiDataLen > ONE_BYTE_SIZE ? 2 : 1) +
								 uiKeyLen + uiDataLen;
			break;
		}

		case BT_NON_LEAF:
		{
			*puiEntrySize = BTE_NON_LEAF_OVHD + uiKeyLen;
			break;
		}

		case BT_NON_LEAF_COUNTS:
		{
			*puiEntrySize = BTE_NON_LEAF_COUNTS_OVHD + uiKeyLen;
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			*puiEntrySize = 0;
			goto Exit;
		}
	}

	// See if we have room in the heap first.  If not, maybe we can make
	// room by defraging the block.

	if( *puiEntrySize <= m_pStack->pBlkHdr->ui16HeapSize)
	{
		*pbDefragBlk = FALSE;
		*pbHaveRoom = TRUE;
	}
	else if( *puiEntrySize <= m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
	{
		// A defrag of the block is required to make room.  We will only defrag
		// if we can recover a minimum of 5% of the total block size.

		if( m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail >= m_uiDefragThreshold)
		{
			*pbHaveRoom = TRUE;
			*pbDefragBlk = TRUE;
		}
		else
		{
			*pbHaveRoom = FALSE;
			*pbDefragBlk = FALSE;
		}
	}
	else
	{
		*pbHaveRoom = FALSE;
		*pbDefragBlk = FALSE;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Function to save the replacement information that we could not store
		on the current go round.  The replace function will check for the
		presence of this structure and deal with it later.
****************************************************************************/
RCODE F_Btree::saveReplaceInfo(
	const FLMBYTE *	pucNewKey,
	FLMUINT				uiNewKeyLen)
{
	RCODE								rc = NE_SFLM_OK;
	BTREE_REPLACE_STRUCT *		pPrev;
	F_BTSK *							pStack = m_pStack;
	const FLMBYTE *				pucParentKey;
	FLMBYTE *						pucEntry;

	if( m_uiReplaceLevels + 1 >= BH_MAX_LEVELS)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	pPrev = m_pReplaceInfo;
	m_pReplaceInfo = &m_pReplaceStruct[ m_uiReplaceLevels++];
	m_pReplaceInfo->pPrev = (void *)pPrev;

	// We should not be at the root level already!

	flmAssert( pStack->uiLevel != m_uiStackLevels - 1);

	m_pReplaceInfo->uiParentLevel = pStack->uiLevel+1;
	m_pReplaceInfo->uiNewKeyLen = uiNewKeyLen;
	m_pReplaceInfo->uiChildBlkAddr = pStack->ui32BlkAddr;
	
	if( m_bCounts)
	{
		m_pReplaceInfo->uiCounts = countKeys( (FLMBYTE *)pStack->pBlkHdr);
	}
	else
	{
		m_pReplaceInfo->uiCounts = 0;
	}

	f_memcpy( &m_pReplaceInfo->pucNewKey[0], pucNewKey, uiNewKeyLen);

	pStack++;
	pucEntry = BtEntry( (FLMBYTE *)pStack->pBlkHdr, pStack->uiCurOffset);

	m_pReplaceInfo->uiParentKeyLen = getEntryKeyLength( pucEntry,
						pStack->pBlkHdr->stdBlkHdr.ui8BlkType, &pucParentKey);

	f_memcpy( &m_pReplaceInfo->pucParentKey[0], pucParentKey, 
				 m_pReplaceInfo->uiParentKeyLen);

	m_pReplaceInfo->uiParentChildBlkAddr = bteGetBlkAddr( pucEntry);

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to restore the stack to a state where we  can finish updating
		the parent with the new key information.
****************************************************************************/
RCODE F_Btree::restoreReplaceInfo(
	const FLMBYTE **	ppucKey,
	FLMUINT *			puiKeyLen,
	FLMUINT *			puiChildBlkAddr,
	FLMUINT *			puiCounts)
{
	RCODE					rc = NE_SFLM_OK;
	RCODE					rcTmp = NE_SFLM_OK;
	FLMUINT				uiLoop;
	FLMBYTE *			pucEntry;
	FLMUINT				uiKeyLen;
	const FLMBYTE *	pucKey;
	FLMUINT				uiSearchLevel = m_uiSearchLevel;
	FLMUINT				uiStackLevels = m_uiStackLevels;

	// We will need to redo our stack from the top down to
	// make sure we are looking at the correct blocks.

	m_uiSearchLevel = m_uiStackLevels - m_pReplaceInfo->uiParentLevel - 1;
	rcTmp = findEntry( m_pReplaceInfo->pucParentKey,
							 m_pReplaceInfo->uiParentKeyLen, FLM_EXACT);

	m_uiSearchLevel = uiSearchLevel;

	if ((rcTmp != NE_SFLM_OK) &&
		 (rcTmp != NE_SFLM_NOT_FOUND) &&
		 (rcTmp != NE_SFLM_EOF_HIT))
	{
		rc = RC_SET( rcTmp);
		goto Exit;
	}

	// Set the stack pointer to the parent level that we want to replace.

	m_pStack = &m_Stack[ m_pReplaceInfo->uiParentLevel];

	// There is always the possibility that the key we are searching for
	// has a duplicate key ahead of it, as a result of a continuation element.
	// We really must replace the entry we were looking at when the information
	// was stored, therefore, we will verify that we have the right entry.

	for( ;;)
	{
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

		uiKeyLen = getEntryKeyLength( pucEntry, 
							m_pStack->pBlkHdr->stdBlkHdr.ui8BlkType, &pucKey);

		if( uiKeyLen != m_pReplaceInfo->uiParentKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		if( f_memcmp( &m_pReplaceInfo->pucParentKey[0], pucKey, uiKeyLen) == 0)
		{
			if( bteGetBlkAddr( pucEntry) != m_pReplaceInfo->uiParentChildBlkAddr)
			{
				// Try moving forward to the next entry ...
				
				if( RC_BAD( rc = moveStackToNext( NULL)))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
	}

	// Now return the other important stuff

	*puiChildBlkAddr = m_pReplaceInfo->uiChildBlkAddr;
	*puiKeyLen = m_pReplaceInfo->uiNewKeyLen;
	*puiCounts = m_pReplaceInfo->uiCounts;

	for( uiLoop = 0; uiLoop < m_uiStackLevels; uiLoop++)
	{
		m_Stack[ uiLoop].uiKeyLen = m_pReplaceInfo->uiNewKeyLen;
	}

	m_uiStackLevels = uiStackLevels;

	// Point to the key

	*ppucKey = &m_pReplaceInfo->pucNewKey[ 0];

	// Free the current ReplaceInfo Buffer

	m_pReplaceInfo = (BTREE_REPLACE_STRUCT *)m_pReplaceInfo->pPrev;
	m_uiReplaceLevels--;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to set the key to be returned to the caller.
****************************************************************************/
FINLINE RCODE F_Btree::setReturnKey(
	FLMBYTE *		pucEntry,
	FLMUINT			uiBlockType,
	FLMBYTE *		pucKey,
	FLMUINT *		puiKeyLen,
	FLMUINT			uiKeyBufSize)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiKeyLen;
	const FLMBYTE *	pucKeyRV;

	uiKeyLen =  getEntryKeyLength( pucEntry, uiBlockType, &pucKeyRV);
	
	if( uiKeyLen == 0)
	{
		// We hit the LEM, hence the EOF error
		
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}

	if( uiKeyLen <= uiKeyBufSize)
	{
		f_memcpy( pucKey, pucKeyRV, uiKeyLen);
		*puiKeyLen = uiKeyLen;
	}
	else
	{
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc: Method to return the data from either the BTREE block or
		the DO block.  It will update the tracking variables too.
		This method assumes that the m_pSCache has already been setup for 
		the 1st go-round.
****************************************************************************/
RCODE F_Btree::extractEntryData(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMBYTE *			pucBuffer,
	FLMUINT				uiBufSiz,
	FLMUINT *			puiDataLen)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucDestPtr = pucBuffer;
	FLMUINT32		ui32BlkAddr = 0;
	FLMBOOL			bNewBlock;
	FLMUINT			uiDataLen = 0;

	flmAssert( m_pSCache);

	if( puiDataLen)
	{
		*puiDataLen = 0;
	}

#ifdef FLM_DEBUG
	if( pucBuffer)
	{
		f_memset( pucBuffer, 0, uiBufSiz);
	}
#endif

	// Is there anything to read?
	
	if( m_uiOADataRemaining == 0)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}

	while( m_uiOADataRemaining && (uiDataLen < uiBufSiz))
	{
		if( m_uiDataRemaining <= (uiBufSiz - uiDataLen))
		{
			// Let's take what we have left in this block first.

			if( pucDestPtr)
			{
				f_memcpy( pucDestPtr, m_pucDataPtr, m_uiDataRemaining);
				pucDestPtr += m_uiDataRemaining;
			}

			uiDataLen += m_uiDataRemaining;
			m_uiOADataRemaining -= m_uiDataRemaining;
			m_uiDataRemaining = 0;
		}
		else
		{
			// Buffer is too small to hold everything in this block.
			
			if( pucDestPtr)
			{
				f_memcpy( pucDestPtr, m_pucDataPtr, uiBufSiz - uiDataLen);
				pucDestPtr += (uiBufSiz - uiDataLen);
			}

			m_pucDataPtr += (uiBufSiz - uiDataLen);
			m_uiOADataRemaining -= (uiBufSiz - uiDataLen);
			m_uiDataRemaining -= (uiBufSiz - uiDataLen);
			uiDataLen += (uiBufSiz - uiDataLen);
		}

		// If there is still more overall data remaining, we need to get the
		// next DO block or standard block and setup to read it too.
		// i.e. More to come, but nothing left in this block.
		
		if( (m_uiOADataRemaining > 0) && (m_uiDataRemaining == 0))
		{
			if (!m_bDataOnlyBlock &&
				 (m_uiCurOffset <
				  (FLMUINT)(((F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr)->ui16NumKeys - 1)))
			{
				m_uiCurOffset++;
				bNewBlock = FALSE;
			}
			else
			{
				// Get the next block address
				
				ui32BlkAddr = m_pSCache->m_pBlkHdr->ui32NextBlkInChain;

				// Release the current block before we get the next one.
				
				ScaReleaseCache( m_pSCache, FALSE);
				m_pSCache = NULL;

				if( ui32BlkAddr == 0)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}

				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
					ui32BlkAddr, NULL, &m_pSCache)))
				{
					goto Exit;
				}

				updateTransInfo( m_pSCache->m_pBlkHdr->ui64TransID,
									  m_pSCache->m_ui64HighTransID);

				m_ui64LastBlkTransId = m_pSCache->m_pBlkHdr->ui64TransID;
				bNewBlock = TRUE;
			}

			// If this is a data only block, then we can get the local data size
			// from the header.

			if( m_bDataOnlyBlock)
			{
				flmAssert( m_pSCache->m_pBlkHdr->ui8BlkType == BT_DATA_ONLY);

				m_pucDataPtr = (FLMBYTE *)m_pSCache->m_pBlkHdr +
										  sizeofDOBlkHdr( m_pSCache->m_pBlkHdr);
										  
				m_uiDataRemaining = m_uiBlockSize -
										  sizeofDOBlkHdr( m_pSCache->m_pBlkHdr) -
										  m_pSCache->m_pBlkHdr->ui16BlkBytesAvail;
										  
				m_uiDataLength = m_uiDataRemaining;
				m_ui32CurBlkAddr = ui32BlkAddr;
			}
			else
			{
				F_BTREE_BLK_HDR *		pBlkHdr;
				FLMBYTE *				pucEntry;

				// In a BTREE block, we MUST ensure that the first entry is a
				// continuation of the previous entry in the previous block.
				
				pBlkHdr = (F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr;

				if( pBlkHdr->ui16NumKeys == 0)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}

				if( bNewBlock)
				{
					m_uiCurOffset = 0;
				}

				// Point to the first entry ...
				
				pucEntry = BtEntry( (FLMBYTE *)pBlkHdr, m_uiCurOffset);

				if( !checkContinuedEntry( pucKey, uiKeyLen, NULL, pucEntry,
					pBlkHdr->stdBlkHdr.ui8BlkType))
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}

				m_uiDataRemaining = btGetEntryDataLength( pucEntry,
					&m_pucDataPtr, NULL, NULL);
					
				m_uiDataLength = m_uiDataRemaining;

				if( bNewBlock)
				{
					m_ui32CurBlkAddr = ui32BlkAddr;
				}
			}
			
			// Update the offset at the begining of the current entry.
			
			m_uiOffsetAtStart = m_uiOADataLength - m_uiOADataRemaining;
		}
	}

Exit:

	if( puiDataLen)
	{
		*puiDataLen = uiDataLen;
	}

	if( m_pSCache)
	{
		// We must release the SCache block
		
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to prepare the Btree state for reading.  Since several APIs do
		the same thing, this has been put into a private method.
****************************************************************************/
RCODE F_Btree::setupReadState(
	F_BLK_HDR *		pBlkHdr,
	FLMBYTE *		pucEntry)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache = NULL;
	const FLMBYTE *	pucData;

	// Is there any data?  Check the block type.
	
	if( pBlkHdr->ui8BlkType == BT_LEAF_DATA)
	{
		// How large is the value for this entry?
		
		m_uiDataLength = btGetEntryDataLength( pucEntry, &pucData,
									&m_uiOADataLength, &m_bDataOnlyBlock);

		m_uiPrimaryDataLen = m_uiDataLength;
	}
	else
	{
		m_uiDataLength = 0;
		m_uiOADataLength = 0;
		m_bDataOnlyBlock = FALSE;
	}

  // Represents the offset at the beginning entry in the first block.  This
  // will change as we move through the blocks.
									
	m_uiOffsetAtStart = 0;
	
	// Watch the transaction id and the transaction count during streaming
	// read operations.  If either changes after an initial read, then
	// we abort the operation.
	
	m_ui64CurrTransID = m_pDb->m_ui64CurrTransID;
	m_uiBlkChangeCnt = m_pDb->m_uiBlkChangeCnt;
	m_ui64LastBlkTransId = pBlkHdr->ui64TransID;
	m_ui64PrimaryBlkTransId = pBlkHdr->ui64TransID;

	// Track the overall length progress
  
	m_uiOADataRemaining = m_uiOADataLength;
	
	// Track the local entry progress
	
	m_uiDataRemaining = m_uiDataLength;

	if( m_bDataOnlyBlock)
	{
		m_ui32DOBlkAddr = bteGetBlkAddr( pucData);
		m_ui32CurBlkAddr = m_ui32DOBlkAddr;

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32DOBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		m_ui64LastBlkTransId = pSCache->m_pBlkHdr->ui64TransID;

		// Local amount of data in this block
		
		m_uiDataRemaining = m_uiBlockSize -
										sizeofDOBlkHdr((F_BLK_HDR *)pSCache->m_pBlkHdr) -
										pSCache->m_pBlkHdr->ui16BlkBytesAvail;

		// Keep the actual local data size for later.
		
		m_uiDataLength = m_uiDataRemaining;

		// Adjust for the key at the beginning of the first block.
		
		if( pSCache->m_pBlkHdr->ui32PrevBlkInChain == 0)
		{
			FLMBYTE *	pucPtr = (FLMBYTE *)pSCache->m_pBlkHdr +
										sizeofDOBlkHdr((F_BLK_HDR *)pSCache->m_pBlkHdr);
			FLMUINT16	ui16KeyLen = FB2UW( pucPtr);

			m_uiDataLength -= (ui16KeyLen + 2);
			m_uiDataRemaining -= (ui16KeyLen + 2);
		}

		// Now release the DO Block.  We will get it again when we need it.
		
		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to remove extra entries after a replace operation.
****************************************************************************/
RCODE F_Btree::removeRemainingEntries(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen)
{
	RCODE						rc = NE_SFLM_OK;
	F_BTREE_BLK_HDR *		pBlkHdr;
	FLMBOOL					bLastElement = FALSE;
	FLMBYTE *				pucEntry;
	FLMBOOL					bFirst = TRUE;

	// We should never get to this function when in the upper levels.

	flmAssert( m_pStack->uiLevel == 0);

	// If we do not have a stack setup yet (which can happen if the replace
	// is trying to shortcut to the previously known block address and offset),
	// then at this point, we must build the stack, since it may be required
	// to adjust the upper levels of the btree.

	if( !m_bStackSetup)
	{
		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
		{
			goto Exit;
		}
	}

	while( !bLastElement)
	{
		// Begin each iteration at the leaf level.
		
		m_pStack = &m_Stack[ 0];

		// Advance the stack to the next entry.
		
		if (bFirst ||
				m_pStack->uiCurOffset >= (FLMUINT)(m_pStack->pBlkHdr->ui16NumKeys))
		{
			if( RC_BAD( rc = moveStackToNext( NULL)))
			{
				goto Exit;
			}
		}

		bFirst = FALSE;
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

		if( !checkContinuedEntry( pucKey, uiKeyLen, &bLastElement,
					pucEntry, getBlkType( (FLMBYTE *)m_pStack->pBlkHdr)))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		// Remove the entry from this block.
		
		if( RC_BAD( rc = remove( FALSE)))
		{
			goto Exit;
		}
		
		pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;

		// Is the block empty now?  If it is, then we will want to remove this
		// block and remove the entry in the parent that points to this block.
		
		if( pBlkHdr->ui16NumKeys == 0)
		{
			for (;;)
			{
				flmAssert( !isRootBlk( m_pStack->pBlkHdr));

				// Remove this block, then update the parent.
				
				if( RC_BAD( rc = deleteEmptyBlock()))
				{
					goto Exit;
				}

				// Now update the parent blocks
				
				m_pStack++;

				if( RC_BAD( rc = remove( FALSE)))
				{
					goto Exit;
				}

				// Update the counts if keeping counts.
				
				if( m_bCounts && !isRootBlk(pBlkHdr))
				{
					if( RC_BAD( rc = updateCounts()))
					{
						goto Exit;
					}
				}

				if( m_pStack->pBlkHdr->ui16NumKeys > 0)
				{
					break;
				}
			}

			// Rebuild the stack to the beginning after a delete block operation.
			
			if( RC_BAD( findEntry( pucKey, uiKeyLen, FLM_EXACT)))
			{
				goto Exit;
			}
			
			bFirst = TRUE;
		}
		else
		{
			// Update the counts if keeping counts.
			
			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to delete an empty block.  The block that will be deleted is
		the current block pointed to by m_pStack.
****************************************************************************/
RCODE F_Btree::deleteEmptyBlock( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT32			ui32PrevBlkAddr;
	FLMUINT32			ui32NextBlkAddr;
	F_CachedBlock *	pSCache = NULL;

	// Get the previous block address so we can back everything up in the stack

	ui32PrevBlkAddr = m_pStack->pBlkHdr->stdBlkHdr.ui32PrevBlkInChain;
	ui32NextBlkAddr = m_pStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain;

	// Free the block

	rc = m_pDb->m_pDatabase->blockFree(m_pDb, m_pStack->pSCache);
	
	m_pStack->pSCache = NULL;
	m_pStack->pBlkHdr = NULL;
	
	if( RC_BAD( rc))
	{
		goto Exit;
	}

	// Update the previous block.

	if( ui32PrevBlkAddr)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32PrevBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pSCache)))
		{
			goto Exit;
		}

		pSCache->m_pBlkHdr->ui32NextBlkInChain = ui32NextBlkAddr;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

	// Update the next block

	if( ui32NextBlkAddr)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32NextBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pSCache)))
		{
			goto Exit;
		}

		pSCache->m_pBlkHdr->ui32PrevBlkInChain = ui32PrevBlkAddr;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	
	return( rc);
}

/***************************************************************************
Desc:	Method to remove (free) all data only blocks that are linked to the
		data only block whose address is passed in (inclusive).
****************************************************************************/
RCODE F_Btree::removeDOBlocks(
	FLMUINT32		ui32BlkAddr)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT32			ui32NextBlkAddr;
	F_CachedBlock *	pSCache = NULL;

	ui32NextBlkAddr = ui32BlkAddr;

	while( ui32NextBlkAddr)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32NextBlkAddr, NULL, &pSCache)))
		{
			goto Exit;
		}

		flmAssert( getBlkType( (FLMBYTE *)pSCache->m_pBlkHdr) == BT_DATA_ONLY);
		ui32NextBlkAddr = pSCache->m_pBlkHdr->ui32NextBlkInChain;

		rc = m_pDb->m_pDatabase->blockFree( m_pDb, pSCache);
		pSCache = NULL;
		
		if( RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method used to replace entries where the original spans multiple
		elements and we are NOT to truncate it.  To do this, we will attempt
		to fill each block until we have stored everything.
****************************************************************************/
RCODE F_Btree::replaceMultiples(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiLen,
	FLMUINT,					//uiFlags,
	FLMUINT *,				//puiChildBlkAddr,
	FLMUINT *,				//puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBOOL					bLastElement = FALSE;
	FLMUINT					uiRemainingData = uiLen;
	const FLMBYTE *		pucRemainingValue = pucDataValue;
	FLMBYTE *				pucEntry = NULL;
	FLMBYTE *				pucData;
	FLMUINT					uiDataLength;
	FLMUINT					uiOADataLength = uiLen;
	FLMUINT					uiOldOADataLength;
	FLMUINT					uiAmtCopied;

	// Must be at the leaf level!

	flmAssert( m_pStack->uiLevel == 0);

	while( uiRemainingData)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
			m_pDb, &m_pStack->pSCache)))
		{
			goto Exit;
		}

		m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
		m_pStack->pui16OffsetArray = BtOffsetArray( 
													(FLMBYTE *)m_pStack->pBlkHdr, 0);

		// Get a pointer to the current entry
		
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

		// Determine the data size for this entry
		
		uiDataLength = btGetEntryDataLength( pucEntry, (const FLMBYTE **)&pucData,
								&uiOldOADataLength, NULL);

		// Now over-write as much of the data as we can
		
		if( uiRemainingData >= uiDataLength)
		{
			f_memcpy( pucData, pucRemainingValue, uiDataLength);

			uiAmtCopied = uiDataLength;
			pucRemainingValue += uiDataLength;
			uiRemainingData -= uiDataLength;
		}
		else
		{
			f_memcpy( pucData, pucRemainingValue, uiRemainingData);
			uiAmtCopied = uiRemainingData;
			pucRemainingValue += uiRemainingData;
			uiRemainingData = 0;
		}

		// Do we need to adjust the data length?
		
		if( uiDataLength > uiAmtCopied)
		{
			FLMBYTE *	pucTmp = pucEntry;
			
			// Skip the flag
			
			pucTmp++;
			
			if( bteKeyLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			if( bteDataLenFlag( pucEntry))
			{
				UW2FBA( (FLMUINT16)uiAmtCopied, pucTmp);
				pucTmp += 2;
			}
			else
			{
				*pucTmp = (FLMBYTE)uiAmtCopied;
				pucTmp++;
			}

			// We need to adjust the free space in the block too.

			m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail +=
									(FLMUINT16)(uiDataLength - uiAmtCopied);


#ifdef FLM_DEBUG
			// Clear the unused portion of the block now.
	
			pucTmp = pucData + uiAmtCopied;
			f_memset( pucTmp, 0, (uiDataLength - uiAmtCopied));
#endif
		}

		// Adjust the OA Data length if needed.  We only need to worry about this
		// on the first element.  No others have it.

		if( bteFirstElementFlag( pucEntry) && uiOADataLength != uiOldOADataLength)
		{
			FLMBYTE *		pucTmp = pucEntry;

			flmAssert( bteOADataLenFlag( pucEntry));

			pucTmp++;
			
			if( bteKeyLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			if( bteDataLenFlag( pucEntry))
			{
				pucTmp += 2;
			}
			else
			{
				pucTmp++;
			}

			UD2FBA( (FLMUINT32)uiOADataLength, pucTmp);
		}

		// If we just updated the last member of this entry so break out.
		
		if( uiRemainingData == 0)
		{
			break;
		}

		// Was this the last element for this entry?
		
		if( bteLastElementFlag(pucEntry))
		{
			FLMBYTE *	pucTmp = pucEntry;

			// Turn off the lastElement flag on this entry.
			
			*pucTmp &= ~BTE_FLAG_LAST_ELEMENT;

			// No more to replace, the rest is going to be new data.
			
			*ppucRemainingValue = pucRemainingValue;
			*puiRemainingLen = uiRemainingData;
			break;
		}

		// Advance to the next entry, this block or the next...
		// The function expects to find the block in m_pSCache, so
		// let's put it there for now.

		if( RC_BAD( rc = moveStackToNext( NULL)))
		{
			goto Exit;
		}

		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
								  m_pStack->uiCurOffset);

		// Make sure we are still looking at the same key etc.
		
		if( !checkContinuedEntry( *ppucKey, *puiKeyLen, &bLastElement,
			pucEntry, getBlkType( (FLMBYTE *)m_pStack->pBlkHdr)))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
	}

	// Are there any more entries to remove?
	
	if( !bteLastElementFlag( pucEntry) && !uiRemainingData)
	{
		*pucEntry |= BTE_FLAG_LAST_ELEMENT;
		
		if( RC_BAD( rc = removeRemainingEntries( *ppucKey, *puiKeyLen)))
		{
			goto Exit;
		}
	}

	*peAction = ELM_DONE;

Exit:

	// Only release the m_pSCache if the use count is greater than 1.  It is
	// pointed to by the stack also.
	
	if( m_pSCache && m_pSCache->m_uiUseCount > 1)
	{
		ScaReleaseCache( m_pSCache, FALSE);
	}
	
	m_pSCache = NULL;
	return( rc);
}

/***************************************************************************
Desc:	Method used to replace entries where the original spans multiple
		elements and we are not to truncate it.  To do this, we will attempt
		to fill each block until we have stored everything.
****************************************************************************/
RCODE F_Btree::replaceMultiNoTruncate(
	const FLMBYTE **		ppucKey,
	FLMUINT *				puiKeyLen,
	const FLMBYTE *		pucDataValue,
	FLMUINT					uiLen,
	FLMUINT,					//uiFlags,
	FLMUINT *,				//puiChildBlkAddr,
	FLMUINT *,				//puiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	F_ELM_UPD_ACTION *	peAction)
{
	RCODE						rc = NE_SFLM_OK;
	FLMBOOL					bLastElement = FALSE;
	FLMUINT					uiRemainingData = uiLen;
	const FLMBYTE *		pucRemainingValue = pucDataValue;
	FLMBYTE *				pucEntry;
	FLMBYTE *				pucData;
	FLMUINT					uiDataLength;

	// Must be at the leaf level
	
	flmAssert( m_pStack->uiLevel == 0);

	while( uiRemainingData)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
			m_pDb, &m_pStack->pSCache)))
		{
			goto Exit;
		}

		m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
		m_pStack->pui16OffsetArray = BtOffsetArray( 
												(FLMBYTE *)m_pStack->pBlkHdr, 0);

		// Get a pointer to the current entry
		
		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

		// Determine the data size for this entry
		
		uiDataLength = btGetEntryDataLength( pucEntry, 
							(const FLMBYTE **)&pucData, NULL, NULL);

		// Now over-write as much of the data as we can.
		
		if( uiRemainingData > uiDataLength)
		{
			f_memcpy( pucData, pucRemainingValue, uiDataLength);
			pucRemainingValue += uiDataLength;
			uiRemainingData -= uiDataLength;
		}
		else
		{
			f_memcpy( pucData, pucRemainingValue, uiRemainingData);
			pucRemainingValue += uiRemainingData;
			uiRemainingData = 0;
		}

		// We just updated the last member of this entry so break out.
		
		if( uiRemainingData == 0)
		{
			break;
		}

		// Was this the last element for this entry?
		
		if( bteLastElementFlag( pucEntry))
		{
			// No more to replace, the rest is going to be new data.
			
			*ppucRemainingValue = pucRemainingValue;
			*puiRemainingLen = uiRemainingData;
			break;
		}

		// Advance to the next entry, this block or the next...
		// The function expects to find the block in m_pSCache, so
		// let's put it there f or now.

		if( RC_BAD( rc = moveStackToNext( NULL)))
		{
			goto Exit;
		}

		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr,
								  m_pStack->uiCurOffset);

		// Make sure we are still looking at the same key etc.
		
		if( !checkContinuedEntry( *ppucKey, *puiKeyLen, &bLastElement,
			pucEntry, getBlkType( (FLMBYTE *)m_pStack->pBlkHdr)))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}
	}

	*peAction = ELM_DONE;

Exit:

	// Only release the m_pSCache if the use count is greater than 1.  It is
	// pointed to by the stack also.

	if( m_pSCache && m_pSCache->m_uiUseCount > 1)
	{
		ScaReleaseCache( m_pSCache, FALSE);
	}
	
	m_pSCache = NULL;
	return( rc);
}

/***************************************************************************
Desc:	Private method to retrieve the next block in the chain relative to
		the block that is passed in.  The block that is passed in is always
		released prior to getting the next block.
****************************************************************************/
FINLINE RCODE F_Btree::getNextBlock(
	F_CachedBlock **	ppSCache)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT32		ui32BlkAddr;

	ui32BlkAddr = (*ppSCache)->m_pBlkHdr->ui32NextBlkInChain;

	ScaReleaseCache( *ppSCache, FALSE);
	*ppSCache = NULL;

	if( ui32BlkAddr == 0)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		ui32BlkAddr, NULL, ppSCache)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to retrieve the previous block in the chain relative to
		the  block that is passed in.  The block that is passed in is always
		released prior to getting the previous block.
****************************************************************************/
FINLINE RCODE F_Btree::getPrevBlock(
	F_CachedBlock **	ppSCache)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT32	ui32BlkAddr;

	ui32BlkAddr = (*ppSCache)->m_pBlkHdr->ui32PrevBlkInChain;

	ScaReleaseCache( *ppSCache, FALSE);
	*ppSCache = NULL;

	if( ui32BlkAddr == 0)
	{
		rc = RC_SET( NE_SFLM_BOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		ui32BlkAddr, NULL, ppSCache)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to verify that the entry we are looking at in the stack
		is a continuation entry.  The key must match the key we pass in and
		the entry must be marked as a continuation, i.e. not the first
		element.
****************************************************************************/
FLMBOOL F_Btree::checkContinuedEntry(
	const FLMBYTE *	pucKey,
	FLMUINT				uiKeyLen,
	FLMBOOL *			pbLastElement,
	FLMBYTE *			pucEntry,
	FLMUINT				uiBlkType)
{
	FLMBOOL				bOk = TRUE;
	FLMUINT				uiBlkKeyLen;
	const FLMBYTE *	pucBlkKey;

	if( pbLastElement)
	{
		*pbLastElement = bteLastElementFlag( pucEntry);
	}

	uiBlkKeyLen = getEntryKeyLength( pucEntry, uiBlkType, &pucBlkKey);

	// Must be the same size key!
	
	if( uiKeyLen != uiBlkKeyLen)
	{
		bOk = FALSE;
		goto Exit;
	}

	// Must be identical!
	
	if( f_memcmp( pucKey, pucBlkKey, uiKeyLen) != 0)
	{
		bOk = FALSE;
		goto Exit;
	}
		
	// Must not be the first element!
	
	if( bteFirstElementFlag( pucEntry))
	{
		bOk = FALSE;
		goto Exit;
	}

Exit:

	return( bOk);
}

/***************************************************************************
Desc:	Private method to assend the tree, updating the counts for a
		particular block.  This method allows us to update the counts quickly
		without the need to continually loop, replacing existing keys with 
		new counts.
****************************************************************************/
RCODE F_Btree::updateCounts( void)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiLevel;

	for( uiLevel = m_pStack->uiLevel;
		  uiLevel < m_uiStackLevels - 1;
		  uiLevel++)
	{
		if( RC_BAD( rc = updateParentCounts( m_Stack[ uiLevel].pSCache,
			&m_Stack[ uiLevel + 1].pSCache, m_Stack[ uiLevel + 1].uiCurOffset)))
		{
			goto Exit;
		}
		
		m_Stack[ uiLevel + 1].pBlkHdr =
			(F_BTREE_BLK_HDR *)m_Stack[ uiLevel + 1].pSCache->m_pBlkHdr;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private method to store part of an entry in a block.  This method will
		determine how much of the data can be stored in the block.  The amount
		that does not get stored will be returned in ppucRemainingValue and
		puiRemainingLen.
****************************************************************************/
RCODE F_Btree::storePartialEntry(
	const FLMBYTE *		pucKey,
	FLMUINT					uiKeyLen,
	const FLMBYTE *		pucValue,
	FLMUINT					uiLen,
	FLMUINT					uiFlags,
	FLMUINT					uiChildBlkAddr,
	FLMUINT 					uiCounts,
	const FLMBYTE **		ppucRemainingValue,
	FLMUINT *				puiRemainingLen,
	FLMBOOL					bNewBlock)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiNewDataLen;
	FLMUINT			uiOADataLen = 0;
	FLMUINT			uiEntrySize;
	FLMBOOL			bHaveRoom;
	FLMBOOL			bDefragBlk;
	FLMBOOL			bLastEntry;

	if( RC_BAD( rc = calcOptimalDataLength( uiKeyLen,
				uiLen, m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail,
				&uiNewDataLen)))
	{
		goto Exit;
	}

	if( uiNewDataLen < uiLen)
	{
		// Turn off the last element flag.
		
		uiFlags &= ~BTE_FLAG_LAST_ELEMENT;
		
		if( uiFlags & BTE_FLAG_FIRST_ELEMENT)
		{
			// Store the overall data length from this point forward.
			
			uiOADataLen = uiLen;
		}
	}

	if( RC_BAD( rc = calcNewEntrySize( uiKeyLen, uiNewDataLen, &uiEntrySize,
		&bHaveRoom, &bDefragBlk)))
	{
		goto Exit;
	}

	// We will defragment the block first if the avail and heap
	// are not the same size.

	if( m_pStack->pBlkHdr->ui16HeapSize !=
						m_pStack->pBlkHdr->stdBlkHdr.ui16BlkBytesAvail)
	{
		if( RC_BAD( rc = defragmentBlock( &m_pStack->pSCache)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = storeEntry( pucKey, uiKeyLen, pucValue, uiNewDataLen,
		uiFlags, uiOADataLen, uiChildBlkAddr, uiCounts, 
		uiEntrySize, &bLastEntry)))
	{
		goto Exit;
	}

	// If this block has a parent block, and the btree is maintaining counts
	// we will want to update the counts on the parent block.

	if( !isRootBlk( m_pStack->pBlkHdr) && m_bCounts && !bNewBlock)
	{
		if( RC_BAD( rc = updateCounts()))
		{
			goto Exit;
		}
	}

	if( uiNewDataLen < uiLen)
	{
		// Save the portion of the data that was not written.
		// It will be written later.
		
		*ppucRemainingValue = pucValue + uiNewDataLen;
		*puiRemainingLen = uiLen - uiNewDataLen;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Private meethod for checking the down links in the btree to make sure
		they are not corrupt.
****************************************************************************/
RCODE F_Btree::checkDownLinks( void)
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pParentSCache = NULL;

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
		m_pDb, m_pLFile, m_pLFile->uiRootBlk, NULL, &pParentSCache)))
	{
		goto Exit;
	}

	if( (pParentSCache->m_pBlkHdr->ui8BlkType == BT_NON_LEAF) ||
		 (pParentSCache->m_pBlkHdr->ui8BlkType == BT_NON_LEAF_COUNTS))
	{
		if( RC_BAD( rc = verifyChildLinks( pParentSCache)))
		{
			goto Exit;
		}
	}

Exit:

	if( pParentSCache)
	{
		ScaReleaseCache( pParentSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method (recursive) that checks the child links in the given
		blocks to ensure they are correct.
****************************************************************************/
RCODE F_Btree::verifyChildLinks(
	F_CachedBlock *		pParentSCache)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiNumKeys;
	F_CachedBlock *		pChildSCache = NULL;
	F_BTREE_BLK_HDR *		pParentBlkHdr;
	F_BTREE_BLK_HDR *		pChildBlkHdr;
	FLMUINT					uiCurOffset;
	FLMBYTE *				pucEntry;
	FLMUINT32				ui32BlkAddr;
	const FLMBYTE *		pucParentKey;
	FLMBYTE *				pucChildEntry;
	const FLMBYTE *		pucChildKey;
	FLMUINT					uiParentKeyLen;
	FLMUINT					uiChildKeyLen;

	pParentBlkHdr = (F_BTREE_BLK_HDR *)pParentSCache->m_pBlkHdr;
	uiNumKeys = pParentBlkHdr->ui16NumKeys;

	for( uiCurOffset = 0; uiCurOffset < uiNumKeys; uiCurOffset++)
	{
		pucEntry = BtEntry( (FLMBYTE *)pParentBlkHdr, uiCurOffset);
		
		// Non-leaf nodes have children.
		
		ui32BlkAddr = bteGetBlkAddr( pucEntry);
		flmAssert( ui32BlkAddr);

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
			m_pDb, m_pLFile, ui32BlkAddr, NULL, &pChildSCache)))
		{
			goto Exit;
		}
		
		pChildBlkHdr = (F_BTREE_BLK_HDR *)pChildSCache->m_pBlkHdr;

		// Get key from the parent entry and compare it to the
		// last key in the child block.
		
		uiParentKeyLen = getEntryKeyLength(
			pucEntry, pParentBlkHdr->stdBlkHdr.ui8BlkType, &pucParentKey);

		// Get the last entry in the child block.
		
		pucChildEntry = BtLastEntry( (FLMBYTE *)pChildBlkHdr);

		uiChildKeyLen = getEntryKeyLength(
			pucChildEntry, pChildBlkHdr->stdBlkHdr.ui8BlkType, &pucChildKey);

		if( uiParentKeyLen != uiChildKeyLen)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		if( f_memcmp( pucParentKey, pucChildKey, uiParentKeyLen) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		if( (pChildBlkHdr->stdBlkHdr.ui8BlkType == BT_NON_LEAF) ||
			 (pChildBlkHdr->stdBlkHdr.ui8BlkType == BT_NON_LEAF_COUNTS))
		{
			if( RC_BAD( rc = verifyChildLinks( pChildSCache)))
			{
				goto Exit;
			}
		}
		
		ScaReleaseCache( pChildSCache, FALSE);
		pChildSCache = NULL;
	}

Exit:

	if( pChildSCache)
	{
		ScaReleaseCache( pChildSCache, FALSE);
	}
	
	return( rc);
}

/***************************************************************************
Desc:	This is a private method that computes the number of entries (keys)
		and the number of blocks between two points in the Btree.
****************************************************************************/
RCODE F_Btree::computeCounts(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT64 *		pui64BlockCount,
	FLMUINT64 *		pui64KeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlkFullness)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT64	ui64TotalKeys = 0;
	FLMUINT64	ui64EstKeyCount = 0;
	FLMUINT64	ui64TotalBlocksBetween = 0;
	FLMUINT64	ui64EstBlocksBetween = 0;
	FLMUINT		uiBlkKeyCount;

	ui64TotalBlocksBetween = 0;
	*pbTotalsEstimated = FALSE;

	// The stack that we are looking at does not hold the blocks
	// we need. We first need to restore the blocks as needed.

	if( RC_BAD( rc = getCacheBlocks( pFromStack, pUntilStack)))
	{
		goto Exit;
	}

	// Are the from and until positions in the same block?

	if( pFromStack->ui32BlkAddr == pUntilStack->ui32BlkAddr)
	{
		rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
			pUntilStack->uiCurOffset, &uiBlkKeyCount, NULL);
		ui64TotalKeys = (FLMUINT64)uiBlkKeyCount;
		goto Exit;
	}

	// Are we maintaining counts on this Btree?  If so, we can just
	// use the counts we have...  The blocks count may still be estimated.

	if( m_bCounts)
	{
		return( getStoredCounts( pFromStack, pUntilStack, pui64BlockCount,
			pui64KeyCount, pbTotalsEstimated, uiAvgBlkFullness));
	}

	// Since we are not keeping counts on this Btree, we will need to
	// count them and possibly estimate them.

	// Gather the counts in the from and until leaf blocks.

	if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
			pFromStack->pBlkHdr->ui16NumKeys - 1, &uiBlkKeyCount, NULL)))
	{
		goto Exit;
	}
	ui64TotalKeys += (FLMUINT64)uiBlkKeyCount;

	if( RC_BAD( rc = blockCounts( pUntilStack, 0,
			pUntilStack->uiCurOffset, &uiBlkKeyCount, NULL)))
	{
		goto Exit;
	}

	ui64TotalKeys += (FLMUINT64)uiBlkKeyCount;

	// Do the obvious check to see if the blocks are neighbors.  If they
	// are, we are done.

	if( pFromStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain ==
							pUntilStack->ui32BlkAddr)
	{
		goto Exit;
	}

	// Estimate the number of elements in the parent block.

	*pbTotalsEstimated = TRUE;

	ui64EstKeyCount = (FLMUINT64)getAvgKeyCount( pFromStack, pUntilStack, uiAvgBlkFullness);
	ui64EstBlocksBetween = 1;

	for (;;)
	{
		FLMUINT		uiBlkElementCount;
		FLMUINT		uiTempBlkElementCount;
		FLMUINT64	ui64EstElementCount;

		// Go up a b-tree level and check out how far apart the elements are.

		pFromStack++;
		pUntilStack++;

		if( RC_BAD( rc = getCacheBlocks( pFromStack, pUntilStack)))
		{
			goto Exit;
		}

		// Share the same block?

		if( pFromStack->ui32BlkAddr == pUntilStack->ui32BlkAddr)
		{
			if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
					pUntilStack->uiCurOffset, NULL, &uiBlkElementCount)))
			{
				goto Exit;
			}

			// Don't count the pFromStack or the pUntilStack current elements.

			uiBlkElementCount -= 2;

			ui64TotalBlocksBetween += ui64EstBlocksBetween *
											(FLMUINT64)(uiBlkElementCount > 0 ? uiBlkElementCount : 1);
			ui64TotalKeys += ui64EstKeyCount *
								(FLMUINT64)(uiBlkElementCount > 0 ? uiBlkElementCount : 1);
			goto Exit;
		}

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
				pFromStack->pBlkHdr->ui16NumKeys - 1, NULL, &uiBlkElementCount)))
		{
			goto Exit;
		}

		// Don't count the first element.

		uiBlkElementCount--;

		if( RC_BAD( rc = blockCounts( pUntilStack, 0,
				pUntilStack->uiCurOffset, NULL, &uiTempBlkElementCount)))
		{
			goto Exit;
		}

		uiBlkElementCount += (uiTempBlkElementCount - 1);

		ui64TotalBlocksBetween += ui64EstBlocksBetween * (FLMUINT64)uiBlkElementCount;
		ui64TotalKeys += ui64EstKeyCount * (FLMUINT64)uiBlkElementCount;

		// Do the obvious check to see if the blocks are neighbors.

		if( (FLMUINT)pFromStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain ==
			 pUntilStack->ui32BlkAddr)
		{
			goto Exit;
		}

		// Recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		ui64EstElementCount = (FLMUINT64)getAvgKeyCount(
										pFromStack, pUntilStack, uiAvgBlkFullness);

		// Adjust the estimated key/ref count to be the counts from a complete
		// (not partial) block starting at this level going to the leaf.

		ui64EstKeyCount *= ui64EstElementCount;
		ui64EstBlocksBetween *= ui64EstElementCount;
	}

Exit:

	if( pui64KeyCount)
	{
		*pui64KeyCount = ui64TotalKeys;
	}

	if( pui64BlockCount)
	{
		*pui64BlockCount = ui64TotalBlocksBetween;
	}

	return( rc);
}

/***************************************************************************
Desc:	Private method to count the number of unique keys between two points.
		The count returned is inclusive of the first and last offsets.
****************************************************************************/
RCODE F_Btree::blockCounts(
	F_BTSK *			pStack,
	FLMUINT			uiFirstOffset,
	FLMUINT			uiLastOffset,
	FLMUINT *		puiKeyCount,
	FLMUINT *		puiElementCount)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiKeyCount;
	FLMUINT			uiElementCount;
	FLMBYTE *		pucBlk;
	FLMBYTE *		pucEntry;

	// Debug checks.

	flmAssert( uiFirstOffset <= uiLastOffset);
	flmAssert( uiLastOffset <= (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1));

	uiKeyCount = uiElementCount = 0;
	pucBlk = (FLMBYTE *)pStack->pBlkHdr;

	// Loop gathering the statistics.

	while( uiFirstOffset <= uiLastOffset)
	{
		uiElementCount++;

		if( puiKeyCount)
		{
			pucEntry = BtEntry( pucBlk, uiFirstOffset);

			// We only have to worry about first key elements when we are at the
			// leaf level and we are keeping data at that level.

			if( pStack->uiLevel == 0 && m_bData)
			{
				if( bteFirstElementFlag( pucEntry))
				{
					uiKeyCount++;
				}
			}
			else
			{
				uiKeyCount++;
			}
		}

		// Next element.

		if( uiFirstOffset == (FLMUINT)(pStack->pBlkHdr->ui16NumKeys - 1))
		{
			break;
		}
		else
		{
			uiFirstOffset++;
		}
	}

	if( puiKeyCount)
	{
		*puiKeyCount = uiKeyCount;
	}

	if( puiElementCount)
	{
		*puiElementCount = uiElementCount;
	}

	return( rc);
}

/***************************************************************************
Desc:	Similar to computeCounts, except we use the stored counts.
****************************************************************************/
RCODE F_Btree::getStoredCounts(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT64 *		pui64BlockCount,
	FLMUINT64 *		pui64KeyCount,
	FLMBOOL *		pbTotalsEstimated,
	FLMUINT			uiAvgBlkFullness)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT64		ui64OmittedKeys;
	FLMUINT64		ui64TotalKeys;
	FLMUINT64		ui64EstBlocksBetween;
	FLMUINT64		ui64TotalBlocksBetween;

	*pbTotalsEstimated = FALSE;
	*pui64BlockCount = 0;
	ui64TotalBlocksBetween = 0;

	// Are these blocks adjacent?

	if( pFromStack->pBlkHdr->stdBlkHdr.ui32NextBlkInChain ==
			pUntilStack->ui32BlkAddr)
	{
		*pui64KeyCount = (pFromStack->pBlkHdr->ui16NumKeys - 
								pFromStack->uiCurOffset) + pUntilStack->uiCurOffset + 1;
		goto Exit;
	}

	*pbTotalsEstimated = TRUE;

	// How many keys are excluded in the From and Until blocks?

	ui64OmittedKeys = (FLMUINT64)countRangeOfKeys(
					pFromStack, 0, pFromStack->uiCurOffset) - 1;

	ui64OmittedKeys += (FLMUINT64)countRangeOfKeys(
					pUntilStack, pUntilStack->uiCurOffset,
					pUntilStack->pBlkHdr->ui16NumKeys - 1) - 1;

	ui64TotalKeys = 0;
	ui64EstBlocksBetween = 1;

	for( ;;)
	{
		FLMUINT		uiBlkElementCount;
		FLMUINT		uiBlkTempElementCount;
		FLMUINT64	ui64EstElementCount;

		// Go up a b-tree level and check out how far apart the elements are.

		pFromStack++;
		pUntilStack++;

		if( RC_BAD( rc = getCacheBlocks( pFromStack, pUntilStack)))
		{
			goto Exit;
		}

		// Share the same block?  We can get the actual key count now.

		if( pFromStack->ui32BlkAddr == pUntilStack->ui32BlkAddr)
		{

			if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
					pUntilStack->uiCurOffset, NULL, &uiBlkElementCount)))
			{
				goto Exit;
			}

			// Don't count the pFromStack current element.

			uiBlkElementCount -= 2;
			ui64TotalBlocksBetween += ui64EstBlocksBetween *
											(FLMUINT64)(uiBlkElementCount > 0 ? uiBlkElementCount : 1);

			// Add one to the last offset to include the last entry in the count.
			
			ui64TotalKeys = (FLMUINT64)countRangeOfKeys(
				pFromStack, pFromStack->uiCurOffset, pUntilStack->uiCurOffset);

			*pui64KeyCount = ui64TotalKeys - ui64OmittedKeys;
			*pui64BlockCount = ui64TotalBlocksBetween;
			goto Exit;
		}

		// How many to exclude from the From & Until blocks.

		if( pFromStack->uiCurOffset)
		{
			ui64OmittedKeys += (FLMUINT64)countRangeOfKeys(
				pFromStack, 0, pFromStack->uiCurOffset - 1);
		}

		ui64OmittedKeys += (FLMUINT64)countRangeOfKeys(
				pUntilStack,  pUntilStack->uiCurOffset + 1,
				pUntilStack->pBlkHdr->ui16NumKeys - 1);

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = blockCounts( pFromStack, pFromStack->uiCurOffset,
				pFromStack->pBlkHdr->ui16NumKeys - 1, NULL, &uiBlkElementCount)))
		{
			goto Exit;
		}

		// Don't count the first element.

		uiBlkElementCount--;

		if( RC_BAD( rc = blockCounts( pUntilStack, 0,
				pUntilStack->uiCurOffset, NULL, &uiBlkTempElementCount)))
		{
			goto Exit;
		}

		uiBlkElementCount += (uiBlkTempElementCount - 1);
		ui64TotalBlocksBetween += ui64EstBlocksBetween * (FLMUINT64)uiBlkElementCount;

		// We are not going to check if these blocks are neighbors here because
		// we want to find the common parent.  That will tell us what the actual
		// counts are at the leaf level.

		// Recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		ui64EstElementCount = (FLMUINT64)getAvgKeyCount( 
										pFromStack, pUntilStack, uiAvgBlkFullness);
		ui64EstBlocksBetween *= ui64EstElementCount;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Retrieve the blocks identified in the two stack entries.  Used in
		computing counts (btComputeCounts etc.)
****************************************************************************/
RCODE F_Btree::getCacheBlocks(
	F_BTSK *			pStack1,
	F_BTSK *			pStack2)
{
	RCODE				rc = NE_SFLM_OK;

	// If these blocks are at the root level, we must ensure that we retrieve
	// the root block.  The root block can potentially change address, so
	// we wil reset it here to be sure.

	if( pStack1->uiLevel == m_uiRootLevel)
	{
		pStack1->ui32BlkAddr = (FLMUINT32)m_pLFile->uiRootBlk;
	}

	if( pStack2->uiLevel == m_uiRootLevel)
	{
		pStack2->ui32BlkAddr = (FLMUINT32)m_pLFile->uiRootBlk;
	}

	if( RC_BAD( m_pDb->m_pDatabase->getBlock(
		m_pDb, m_pLFile, pStack1->ui32BlkAddr, NULL, &pStack1->pSCache)))
	{
		goto Exit;
	}
	
	pStack1->pBlkHdr = (F_BTREE_BLK_HDR *)pStack1->pSCache->m_pBlkHdr;

	if( RC_BAD( m_pDb->m_pDatabase->getBlock(
		m_pDb, m_pLFile, pStack2->ui32BlkAddr, NULL, &pStack2->pSCache)))
	{
		goto Exit;
	}
	
	pStack2->pBlkHdr = (F_BTREE_BLK_HDR *)pStack2->pSCache->m_pBlkHdr;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to tally the counts in a block between (inclusive) the
		uiFromOffset & uiUntilOffset parameters.
****************************************************************************/
FLMUINT F_Btree::countRangeOfKeys(
	F_BTSK *				pFromStack,
	FLMUINT				uiFromOffset,
	FLMUINT				uiUntilOffset)
{
	FLMUINT			uiCount = 0;
	FLMBYTE *		pucBlk;
	FLMUINT			uiLoop = uiFromOffset;
	FLMBYTE *		pucEntry;
	FLMUINT			uiBlkType;

	pucBlk = (FLMBYTE *)pFromStack->pBlkHdr;
	uiBlkType = getBlkType( pucBlk);

	if( uiBlkType == BT_NON_LEAF_COUNTS)
	{
		while( uiLoop < uiUntilOffset)
		{
			pucEntry = BtEntry( pucBlk, uiLoop);
			pucEntry += 4;
			uiCount += FB2UD( pucEntry);
			uiLoop++;
		}
	}
	else
	{
		uiCount = uiUntilOffset;
	}

	return( uiCount);
}

/***************************************************************************
Desc:	Method to estimate the average number of keys, based on the anticipated
		average block usage (passed in) and the actual block usage.
****************************************************************************/
FINLINE FLMUINT F_Btree::getAvgKeyCount(
	F_BTSK *			pFromStack,
	F_BTSK *			pUntilStack,
	FLMUINT			uiAvgBlkFullness)
{
	FLMUINT			uiFromUsed;
	FLMUINT			uiUntilUsed;
	FLMUINT			uiTotalUsed;
	FLMUINT			uiFromKeys;
	FLMUINT			uiUntilKeys;
	FLMUINT			uiTotalKeys;

	uiFromUsed = m_uiBlockSize -
		((F_BLK_HDR *)pFromStack->pBlkHdr)->ui16BlkBytesAvail;
		
	uiUntilUsed = m_uiBlockSize -
		((F_BLK_HDR *)pUntilStack->pBlkHdr)->ui16BlkBytesAvail;

	uiTotalUsed = uiFromUsed + uiUntilUsed;

	uiFromKeys = pFromStack->pBlkHdr->ui16NumKeys;
	uiUntilKeys = pUntilStack->pBlkHdr->ui16NumKeys;
	uiTotalKeys = uiFromKeys + uiUntilKeys;

	return( (uiAvgBlkFullness * uiTotalKeys) / uiTotalUsed);
}

/***************************************************************************
Desc:	Method to test if two blocks can be merged together to make a single
		block.  This is done only after a remove operation and is intended to
		try to consolidate space as much as possible.  If we can consolidate
		two blocks, we will do it, then update the tree.
****************************************************************************/
RCODE F_Btree::mergeBlocks(
	FLMBOOL						bLastEntry,
	FLMBOOL *					pbMergedWithPrev,
	FLMBOOL *					pbMergedWithNext,
	F_ELM_UPD_ACTION *		peAction)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT32			ui32PrevBlkAddr;
	F_CachedBlock *	pPrevSCache = NULL;
	FLMUINT32			ui32NextBlkAddr;
	F_CachedBlock *	pNextSCache = NULL;

	*pbMergedWithPrev = FALSE;
	*pbMergedWithNext = FALSE;

	// Our first check is to see if we can merge the current block with its
	// previous block.

	ui32PrevBlkAddr = m_pStack->pSCache->m_pBlkHdr->ui32PrevBlkInChain;
	if( ui32PrevBlkAddr)
	{
		// Get the block.

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
			m_pDb, m_pLFile, ui32PrevBlkAddr, NULL, &pPrevSCache)))
		{
			goto Exit;
		}

		// Is there room to merge?

		if( (FLMUINT)(pPrevSCache->m_pBlkHdr->ui16BlkBytesAvail +
				m_pStack->pSCache->m_pBlkHdr->ui16BlkBytesAvail) >=
				(FLMUINT)(m_uiBlockSize - sizeofBTreeBlkHdr(
									(F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr)))
		{
			// Looks like we can merge these two.  We will move the content
			// of the previous block into this one.

			if( RC_BAD( rc = merge( &pPrevSCache, &m_pStack->pSCache)))
			{
				goto Exit;
			}

			// Save the changed block header address

			m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;

			// Update the counts for the current block before releasing it.

			if( m_bCounts)
			{
				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}
			}

			if( bLastEntry)
			{
				// Need to save the replace information for the last entry in
				// the block before we move to the previous block.  This will
				// allow us to do the replace later.

				FLMBYTE *			pucEntry;
				const FLMBYTE *	pucKey;
				FLMUINT				uiKeyLen;

				pucEntry = BtEntry(
					(FLMBYTE *)m_pStack->pBlkHdr, m_pStack->pBlkHdr->ui16NumKeys - 1);
					
				uiKeyLen = getEntryKeyLength(
					pucEntry, getBlkType( (FLMBYTE *)m_pStack->pBlkHdr), &pucKey);

				if( RC_BAD( rc = saveReplaceInfo( pucKey, uiKeyLen)))
				{
					goto Exit;
				}
			}

			// Move the stack to the previous entry

			if( RC_BAD( rc = moveStackToPrev( pPrevSCache)))
			{
				goto Exit;
			}
			pPrevSCache = NULL;

			flmAssert( m_pStack->pBlkHdr->ui16NumKeys == 0);

			// Free the empty block.

			if( RC_BAD( rc = deleteEmptyBlock()))
			{
				goto Exit;
			}

			// Now we want to remove the parent entry for the block that was
			// freed.

			m_pStack++;
			*peAction = ELM_REMOVE;
			*pbMergedWithPrev = TRUE;
			
			goto Exit;
		}
		else
		{
			// No room here so release the block.

			ScaReleaseCache( pPrevSCache, FALSE);
			pPrevSCache = NULL;
		}
	}

	// Can we merge with the next block?

	ui32NextBlkAddr = m_pStack->pSCache->m_pBlkHdr->ui32NextBlkInChain;
	if( ui32NextBlkAddr)
	{
		// Get the block.

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
			m_pDb, m_pLFile, ui32NextBlkAddr, NULL, &pNextSCache)))
		{
			goto Exit;
		}

		// Is there room to merge?

		if( (FLMUINT)(pNextSCache->m_pBlkHdr->ui16BlkBytesAvail +
				m_pStack->pSCache->m_pBlkHdr->ui16BlkBytesAvail) >=
				(FLMUINT)(m_uiBlockSize - sizeofBTreeBlkHdr(
								(F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr)))
		{
			// Looks like we can merge these two.

			if( RC_BAD( rc = merge( &m_pStack->pSCache, &pNextSCache)))
			{
				goto Exit;
			}

			// Save the changed block header address.

			m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;

			// Update the counts for the current block and the next block.

			if( m_bCounts)
			{
				pPrevSCache = m_pStack->pSCache;

				// Need to move the stack to the next entry.  Don't let the current
				// block get released because we still need it.

				if( RC_BAD( rc = moveStackToNext( pNextSCache, FALSE)))
				{
					goto Exit;
				}
				pNextSCache = NULL;

				if( RC_BAD( rc = updateCounts()))
				{
					goto Exit;
				}

				// Move back to the original stack again.  It's okay to release the
				// now current block.

				if( RC_BAD( rc = moveStackToPrev( pPrevSCache)))
				{
					goto Exit;
				}
				
				pPrevSCache = NULL;
			}

			flmAssert( m_pStack->pBlkHdr->ui16NumKeys == 0);

			// Free the empty block.

			if( RC_BAD( rc = deleteEmptyBlock()))
			{
				goto Exit;
			}

			// Now we want to remove the parent entry for the block that was freed.

			m_pStack++;
			*peAction = ELM_REMOVE;
			*pbMergedWithNext = TRUE;
			goto Exit;
		}
		else
		{
			// No room here so release the block.

			ScaReleaseCache( pNextSCache, FALSE);
			pNextSCache = NULL;
		}
	}

Exit:

	if( *pbMergedWithPrev || *pbMergedWithNext)
	{
		if( m_pDb->m_pDbStats != NULL)
		{
			SFLM_LFILE_STATS *		pLFileStats;

			if( (pLFileStats = m_pDb->getLFileStatPtr( m_pLFile)) != NULL)
			{
				pLFileStats->bHaveStats = TRUE;
				pLFileStats->ui64BlockCombines++;
			}
		}
	}

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	if( pNextSCache)
	{
		ScaReleaseCache( pNextSCache, FALSE);
	}

	return( rc);
}

/***************************************************************************
Desc:	Method to move the contents of the ppFromSCache block into the
		ppToSCache block.  Note that all merges are a move to next operation.
****************************************************************************/
RCODE F_Btree::merge(
	F_CachedBlock **	ppFromSCache,
	F_CachedBlock **	ppToSCache)
{
	RCODE					rc = NE_SFLM_OK;
	F_BTSK				tempStack;
	F_BTSK *				pStack = NULL;
	F_CachedBlock *	pSCache;
	F_BTREE_BLK_HDR *	pBlkHdr;

	// May need to defragment the blocks first.

	pBlkHdr = (F_BTREE_BLK_HDR *)(*ppToSCache)->m_pBlkHdr;
	if( pBlkHdr->stdBlkHdr.ui16BlkBytesAvail != pBlkHdr->ui16HeapSize)
	{
		if( RC_BAD( rc = defragmentBlock( ppToSCache)))
		{
			goto Exit;
		}
	}

	// Make a temporary stack entry so we can "fool" the moveToNext
	// function into moving the entries for us.

	pSCache = *ppFromSCache;
	tempStack.pBlkHdr = (F_BTREE_BLK_HDR *)pSCache->m_pBlkHdr;
	tempStack.ui32BlkAddr = pSCache->m_pBlkHdr->ui32BlkAddr;
	tempStack.pSCache = pSCache;
	tempStack.uiCurOffset = 0;
	tempStack.uiLevel = m_pStack->uiLevel;
	tempStack.pui16OffsetArray = BtOffsetArray( (FLMBYTE *)pSCache->m_pBlkHdr, 0);

	// Save the current m_pStack.

	pStack = m_pStack;
	m_pStack = &tempStack;

	// Now do the move

	if( RC_BAD( rc = moveToNext( tempStack.pBlkHdr->ui16NumKeys - 1,
		0, ppToSCache)))
	{
		goto Exit;
	}

	// Return the changed block structure

	*ppFromSCache = tempStack.pSCache;

Exit:

	// Must always restore the stack.

	m_pStack = pStack;
	return( rc);
}

/***************************************************************************
Desc:	Method to test if the src and dst entries can be combined into one
		entry.  If they can, then they will be combined and stored in the
		m_pucTempBlk buffer.
****************************************************************************/
RCODE F_Btree::combineEntries(
	F_BTREE_BLK_HDR *		pSrcBlkHdr,
	FLMUINT					uiSrcOffset,
	F_BTREE_BLK_HDR *		pDstBlkHdr,
	FLMUINT					uiDstOffset,
	FLMBOOL *				pbEntriesCombined,
	FLMUINT *				puiEntrySize)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucSrcEntry;
	FLMBYTE *			pucDstEntry;
	FLMUINT				uiSrcKeyLen;
	FLMUINT				uiDstKeyLen;
	const FLMBYTE *	pucSrcKey;
	const FLMBYTE *	pucDstKey;
	FLMUINT				uiFlags = 0;
	FLMBYTE *			pucTmp;
 	FLMUINT				uiSrcOADataLen;
 	FLMUINT				uiDstOADataLen;
	const FLMBYTE *	pucSrcData;
	const FLMBYTE *	pucDstData;
	FLMUINT				uiSrcDataLen;
	FLMUINT				uiDstDataLen;
	FLMUINT				uiEntrySize;

	*pbEntriesCombined = FALSE;
	*puiEntrySize = 0;

	if( pDstBlkHdr->ui16NumKeys == 0)
	{
		goto Exit;
	}

	if( pSrcBlkHdr->ui16NumKeys == 0)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	if( getBlkType( (FLMBYTE *)pSrcBlkHdr) != BT_LEAF_DATA)
	{
		goto Exit;
	}

	pucSrcEntry = BtEntry( (FLMBYTE *)pSrcBlkHdr, uiSrcOffset);
	pucDstEntry = BtEntry( (FLMBYTE *)pDstBlkHdr, uiDstOffset);

	// Do we have the same key?

	uiSrcKeyLen = getEntryKeyLength( pucSrcEntry, BT_LEAF_DATA, &pucSrcKey);
	uiDstKeyLen = getEntryKeyLength( pucDstEntry, BT_LEAF_DATA, &pucDstKey);

	if( uiSrcKeyLen != uiDstKeyLen)
	{
		// Not the same key.
		
		goto Exit;
	}

	if( f_memcmp( pucSrcKey, pucDstKey, uiSrcKeyLen) != 0)
	{
		// Not the same key.
		
		goto Exit;
	}

	// They match, so we can combine them.

	pucTmp = &m_pucTempBlk[ 1];		// Key length position
	uiFlags = (pucDstEntry[0] & (BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT)) |
				 (pucSrcEntry[0] & (BTE_FLAG_FIRST_ELEMENT | BTE_FLAG_LAST_ELEMENT));
	uiEntrySize = 1;

	if( uiSrcKeyLen > ONE_BYTE_SIZE)
	{
		uiFlags |= BTE_FLAG_KEY_LEN;
		UW2FBA( (FLMUINT16)uiSrcKeyLen, pucTmp);
		pucTmp += 2;
		uiEntrySize += 2;
	}
	else
	{
		*pucTmp = (FLMBYTE)uiSrcKeyLen;
		pucTmp++;
		uiEntrySize++;
	}

	uiSrcDataLen = btGetEntryDataLength(
		pucSrcEntry, &pucSrcData, &uiSrcOADataLen, NULL);

	uiDstDataLen = btGetEntryDataLength(
		pucDstEntry, &pucDstData, &uiDstOADataLen, NULL);

	if( (uiSrcDataLen + uiDstDataLen) > ONE_BYTE_SIZE)
	{
		uiFlags |= BTE_FLAG_DATA_LEN;
		UW2FBA( (FLMUINT16)(uiSrcDataLen + uiDstDataLen), pucTmp);
		pucTmp += 2;
		uiEntrySize += 2;
	}
	else
	{
		*pucTmp = (FLMBYTE)(uiSrcDataLen + uiDstDataLen);
		pucTmp++;
		uiEntrySize++;
	}

	// Verify the OA Data length

	if( (*pucSrcEntry & BTE_FLAG_OA_DATA_LEN) &&
			(uiSrcOADataLen > (uiSrcDataLen + uiDstDataLen)))
	{
		uiFlags |= BTE_FLAG_OA_DATA_LEN;
		UD2FBA( (FLMUINT32)uiSrcOADataLen, pucTmp);
		pucTmp += 4;
		uiEntrySize += 4;
	}
	else if( (*pucDstEntry & BTE_FLAG_OA_DATA_LEN) &&
			(uiDstOADataLen > (uiSrcDataLen + uiDstDataLen)))
	{
		uiFlags |= BTE_FLAG_OA_DATA_LEN;
		UD2FBA( (FLMUINT32)uiDstOADataLen, pucTmp);
		pucTmp += 4;
		uiEntrySize += 4;
	}

	f_memcpy( pucTmp, pucSrcKey, uiSrcKeyLen);
	pucTmp += uiSrcKeyLen;
	uiEntrySize += uiSrcKeyLen;

	// Need to put the entry together in the right order.  If the Src block is
	// before the Dst block, then we will put down the Src data first.

	if( pSrcBlkHdr->stdBlkHdr.ui32NextBlkInChain ==
							pDstBlkHdr->stdBlkHdr.ui32BlkAddr)
	{
		f_memcpy( pucTmp, pucSrcData, uiSrcDataLen);
		pucTmp += uiSrcDataLen;
		uiEntrySize += uiSrcDataLen;

		f_memcpy( pucTmp, pucDstData, uiDstDataLen);
		uiEntrySize += uiDstDataLen;
	}
	else
	{
		f_memcpy( pucTmp, pucDstData, uiDstDataLen);
		uiEntrySize += uiDstDataLen;
		pucTmp += uiDstDataLen;
		
		f_memcpy( pucTmp, pucSrcData, uiSrcDataLen);
		uiEntrySize += uiSrcDataLen;
	}

	m_pucTempBlk[ 0] = (FLMBYTE)uiFlags;
	*puiEntrySize = uiEntrySize;
	*pbEntriesCombined = TRUE;

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Method to move a block from one location to another.
****************************************************************************/
RCODE F_Btree::btMoveBlock(
	FLMUINT32			ui32FromBlkAddr,
	FLMUINT32			ui32ToBlkAddr)
{
	RCODE						rc = NE_SFLM_OK;
	FLMUINT					uiType;

	if( !m_bOpened || m_bSetupForRead || m_bSetupForReplace ||
		  (m_bSetupForWrite))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	flmAssert( m_uiSearchLevel >= BH_MAX_LEVELS);

	// Verify the Txn type

	if( m_pDb->m_eTransType != SFLM_UPDATE_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( m_pDb->m_eTransType == SFLM_NO_TRANS
								? NE_SFLM_NO_TRANS_ACTIVE
								: NE_SFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Get the From block and retrieve the last key in the block.  Make note
	// of the level of the block.  We will need this to make sure we get the
	// right block.

	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		ui32FromBlkAddr, NULL, &m_pSCache)))
	{
		goto Exit;
	}

	// Find out if this is a Btree block or a DO block.

	uiType = getBlkType((FLMBYTE *)m_pSCache->m_pBlkHdr);

	if( uiType == BT_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	if( uiType == BT_DATA_ONLY)
	{
		if( RC_BAD( rc = moveDOBlock( ui32FromBlkAddr, ui32ToBlkAddr)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = moveBtreeBlock( ui32FromBlkAddr, ui32ToBlkAddr)))
		{
			goto Exit;
		}
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	return( rc);
}

/***************************************************************************
Desc:	Move a Btree block from one address to another, updating its parent.
****************************************************************************/
RCODE F_Btree::moveBtreeBlock(
	FLMUINT32			ui32FromBlkAddr,
	FLMUINT32			ui32ToBlkAddr)
{
	RCODE						rc = NE_SFLM_OK;
	F_BTREE_BLK_HDR *		pBlkHdr = NULL;
	F_BTREE_BLK_HDR *		pNewBlkHdr = NULL;
	FLMBYTE *				pucEntry;
	const FLMBYTE *		pucKeyRV = NULL;
	FLMBYTE *				pucKey = NULL;
	FLMUINT					uiBlkLevel;
	FLMBYTE *				pucSrc;
	FLMBYTE *				pucDest;
	F_CachedBlock *		pSCache = NULL;
	FLMUINT					uiKeyLen;

	// m_pSCache has already been retrieved.

	flmAssert( m_pSCache);

	pBlkHdr = (F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr;
	uiBlkLevel = pBlkHdr->ui8BlkLevel;

	pucEntry = BtLastEntry( (FLMBYTE *)pBlkHdr);

	uiKeyLen = getEntryKeyLength( pucEntry, getBlkType((FLMBYTE *)pBlkHdr),
											&pucKeyRV);

	if( RC_BAD( rc = f_calloc( uiKeyLen, &pucKey)))
	{
		goto Exit;
	}
	
	f_memcpy( pucKey, pucKeyRV, uiKeyLen);

	// Release the block and search for the key.
	
	ScaReleaseCache( m_pSCache, FALSE);
	m_pSCache = NULL;

	if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
	{
		// We must find it!
		
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	// Verify that we found the right block.
	
	m_pStack = &m_Stack[ uiBlkLevel];

	if( ui32FromBlkAddr != m_pStack->ui32BlkAddr)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}

	m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;
	m_pStack->pui16OffsetArray = BtOffsetArray( (FLMBYTE *)m_pStack->pBlkHdr, 0);
	pBlkHdr = m_pStack->pBlkHdr;

	// Get the new block and verify that it is a free block.
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		ui32ToBlkAddr, NULL, &m_pSCache)))
	{
		goto Exit;
	}

	if( getBlkType( (FLMBYTE *)m_pSCache->m_pBlkHdr) != BT_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	// Update the header of the new block to point to the prev and next
	// blocks etc ...
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &m_pSCache)))
	{
		goto Exit;
	}

	pNewBlkHdr = (F_BTREE_BLK_HDR *)m_pSCache->m_pBlkHdr;

	pNewBlkHdr->stdBlkHdr.ui32PrevBlkInChain =
													pBlkHdr->stdBlkHdr.ui32PrevBlkInChain;
	pNewBlkHdr->stdBlkHdr.ui32NextBlkInChain =
													pBlkHdr->stdBlkHdr.ui32NextBlkInChain;
	pNewBlkHdr->stdBlkHdr.ui16BlkBytesAvail =
													pBlkHdr->stdBlkHdr.ui16BlkBytesAvail;
	pNewBlkHdr->stdBlkHdr.ui8BlkType = pBlkHdr->stdBlkHdr.ui8BlkType;
	pNewBlkHdr->stdBlkHdr.ui8BlkFlags = pBlkHdr->stdBlkHdr.ui8BlkFlags;

	pNewBlkHdr->ui16LogicalFile = pBlkHdr->ui16LogicalFile;
	pNewBlkHdr->ui16NumKeys = pBlkHdr->ui16NumKeys;
	pNewBlkHdr->ui8BlkLevel = pBlkHdr->ui8BlkLevel;
	pNewBlkHdr->ui8BTreeFlags = pBlkHdr->ui8BTreeFlags;
	pNewBlkHdr->ui16HeapSize = pBlkHdr->ui16HeapSize;

	// Get the previous and next blocks and set their next and prev addresses.
	
	if( pBlkHdr->stdBlkHdr.ui32PrevBlkInChain)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			pBlkHdr->stdBlkHdr.ui32PrevBlkInChain, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pSCache)))
		{
			goto Exit;
		}
		
		flmAssert( pSCache->m_pBlkHdr->ui32NextBlkInChain == ui32FromBlkAddr);
		pSCache->m_pBlkHdr->ui32NextBlkInChain = ui32ToBlkAddr;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

	if( pBlkHdr->stdBlkHdr.ui32NextBlkInChain)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			pBlkHdr->stdBlkHdr.ui32NextBlkInChain, NULL, &pSCache)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( m_pDb, &pSCache)))
		{
			goto Exit;
		}
		
		flmAssert( pSCache->m_pBlkHdr->ui32PrevBlkInChain == ui32FromBlkAddr);
		pSCache->m_pBlkHdr->ui32PrevBlkInChain = ui32ToBlkAddr;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;
	}

	// Copy the content of the old block into the new block.
	
	pucSrc = (FLMBYTE *)pBlkHdr + sizeofBTreeBlkHdr( pBlkHdr);
	pucDest = (FLMBYTE *)pNewBlkHdr + sizeofBTreeBlkHdr( pNewBlkHdr);

	f_memcpy( pucDest, pucSrc, m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr));

	if( isRootBlk( pBlkHdr))
	{
		m_pLFile->uiRootBlk = ui32ToBlkAddr;
		rc  = m_pDb->m_pDatabase->lFileWrite( m_pDb, m_pLFile);
		goto Exit;
	}

	// Move up one level to the parent entry.
	
	m_pStack++;
	flmAssert( m_pStack->pSCache);

	// Log that we are making a change to the block.
	
	if( RC_BAD( rc = m_pDb->m_pDatabase->logPhysBlk( 
		m_pDb, &m_pStack->pSCache)))
	{
		goto Exit;
	}
	
	m_pStack->pBlkHdr = (F_BTREE_BLK_HDR *)m_pStack->pSCache->m_pBlkHdr;

	// Update the parent block with a new address for the new block.
	
	pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);
	UD2FBA( ui32ToBlkAddr, pucEntry);

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	f_free( &pucKey);
	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc:	Move a DO block from one address to another, updating its reference
		btree entry.
****************************************************************************/
RCODE F_Btree::moveDOBlock(
	FLMUINT32			ui32FromBlkAddr,
	FLMUINT32			ui32ToBlkAddr)
{
	RCODE						rc = NE_SFLM_OK;
	F_BLK_HDR *				pBlkHdr = NULL;
	F_BLK_HDR *				pNewBlkHdr = NULL;
	FLMBYTE *				pucEntry;
	FLMBYTE *				pucKey = NULL;
	FLMBYTE *				pucSrc;
	FLMBYTE *				pucDest;
	F_CachedBlock *		pSCache = NULL;
	F_CachedBlock *		pPrevSCache = NULL;
	F_CachedBlock *		pNextSCache = NULL;
	FLMUINT					uiKeyLen;
	FLMUINT					uiOADataLen;
	const FLMBYTE *		pucData;
	FLMUINT32				ui32DOBlkAddr;
	FLMUINT					uiDataLen;
	FLMBYTE					ucDataBuffer[ sizeof(FLMUINT32)];
	FLMUINT					uiBlkHdrSize;

	// m_pSCache has already been retrieved.

	flmAssert( m_pSCache);

	// Log that we are changing this block.

	if( RC_BAD( m_pDb->m_pDatabase->logPhysBlk(  m_pDb, &m_pSCache)))
	{
		goto Exit;
	}
	pBlkHdr = m_pSCache->m_pBlkHdr;

	// Get the new block and verify that it is a free block.

	if( RC_BAD( m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
		ui32ToBlkAddr, NULL, &pSCache)))
	{
		goto Exit;
	}

	if( getBlkType( (FLMBYTE *)pSCache->m_pBlkHdr) != BT_FREE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	// Update the header of the new block to point to the prev and next
	// blocks etc..

	if( RC_BAD( m_pDb->m_pDatabase->logPhysBlk(  m_pDb, &pSCache)))
	{
		goto Exit;
	}

	pNewBlkHdr = pSCache->m_pBlkHdr;
	pNewBlkHdr->ui32PrevBlkInChain = pBlkHdr->ui32PrevBlkInChain;
	pNewBlkHdr->ui32NextBlkInChain = pBlkHdr->ui32NextBlkInChain;
	pNewBlkHdr->ui16BlkBytesAvail = pBlkHdr->ui16BlkBytesAvail;
	pNewBlkHdr->ui8BlkType = pBlkHdr->ui8BlkType;
	pNewBlkHdr->ui8BlkFlags = pBlkHdr->ui8BlkFlags;

	// Get the previous and next blocks and set their next and prev addresses.

	if( pBlkHdr->ui32PrevBlkInChain)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			pBlkHdr->ui32PrevBlkInChain, NULL, &pPrevSCache)))
		{
			goto Exit;
		}
		
		if( RC_BAD( m_pDb->m_pDatabase->logPhysBlk(  m_pDb, &pPrevSCache)))
		{
			goto Exit;
		}
		
		flmAssert( pPrevSCache->m_pBlkHdr->ui32NextBlkInChain == ui32FromBlkAddr);
		pPrevSCache->m_pBlkHdr->ui32NextBlkInChain = ui32ToBlkAddr;
		ScaReleaseCache( pPrevSCache, FALSE);
		pPrevSCache = NULL;
	}

	if( pBlkHdr->ui32NextBlkInChain)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			pBlkHdr->ui32NextBlkInChain, NULL, &pNextSCache)))
		{
			goto Exit;
		}
		
		if( RC_BAD( m_pDb->m_pDatabase->logPhysBlk(  m_pDb, &pNextSCache)))
		{
			goto Exit;
		}
		
		flmAssert( pNextSCache->m_pBlkHdr->ui32PrevBlkInChain == ui32FromBlkAddr);
		pNextSCache->m_pBlkHdr->ui32PrevBlkInChain = ui32ToBlkAddr;
		ScaReleaseCache( pNextSCache, FALSE);
		pNextSCache = NULL;
	}

	// Copy the content of the old block into the new block.

	uiBlkHdrSize = sizeofDOBlkHdr((F_BLK_HDR *)pBlkHdr);
	pucSrc = (FLMBYTE *)pBlkHdr + uiBlkHdrSize;
	pucDest = (FLMBYTE *)pNewBlkHdr + uiBlkHdrSize;
	f_memcpy( pucDest, pucSrc, m_uiBlockSize - uiBlkHdrSize);

	// Do we need to update the reference btree entry.

	if( pBlkHdr->ui32PrevBlkInChain == 0)
	{
		// Get the key from the beginning of the block.

		uiKeyLen = FB2UW( pucDest);
		pucKey = pucDest + sizeof( FLMUINT16);

		if( RC_BAD( rc = findEntry( pucKey, uiKeyLen, FLM_EXACT)))
		{
			// We must find it!
			
			RC_UNEXPECTED_ASSERT( rc);
			goto Exit;
		}

		// Verify that we found the right block.

		pucEntry = BtEntry( (FLMBYTE *)m_pStack->pBlkHdr, m_pStack->uiCurOffset);

		if( !bteDataBlockFlag( pucEntry))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		uiDataLen = btGetEntryDataLength( pucEntry, &pucData,
							&uiOADataLen, NULL);

		ui32DOBlkAddr = bteGetBlkAddr( pucData);

		if( ui32DOBlkAddr != ui32FromBlkAddr)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		if( uiDataLen != sizeof( ucDataBuffer))
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			goto Exit;
		}

		// Make the data entry with the new block address

		UD2FBA( ui32ToBlkAddr, ucDataBuffer);

		if( RC_BAD( rc = updateEntry(
			pucKey, uiKeyLen, ucDataBuffer, uiOADataLen, ELM_REPLACE_DO)))
		{
			goto Exit;
		}
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	if( pNextSCache)
	{
		ScaReleaseCache( pNextSCache, FALSE);
	}

	releaseBlocks( TRUE);
	return( rc);
}

/***************************************************************************
Desc: Method to move the read point in an entry to a particular position
		within the entry.  This method will move to a previous or a later
		position.
****************************************************************************/
RCODE F_Btree::btSetReadPosition(
	FLMBYTE *				pucKey,
	FLMUINT					uiKeyLen,
	FLMUINT					uiPosition)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE *			pucEntry;
	F_BLK_HDR *			pBlkHdr = NULL;
	FLMUINT32			ui32BlkAddr;
	FLMBOOL				bLastElement = FALSE;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	// We cannot position to a point beyond the end of the current entry.
	
	if( uiPosition >= m_uiOADataLength)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	// If the transaction Id or the Block Change Count has changed,
	// we must re-sync ourselves.
	
	if( (m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		 (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt))
	{

		// Test to see if we really need to re-synch...
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
										m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				( m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// We must call btLocateEntry so we can re-initialize the read.
			
			if( !m_bFirstRead)
			{
				if( RC_BAD( rc = btLocateEntry(
									pucKey, uiKeyLen, &uiKeyLen, FLM_EXACT)))
				{
					goto Exit;
				}

				// Will need a new version of this block.
				
				ScaReleaseCache( m_pSCache, FALSE);
				m_pSCache = NULL;
			}
			else
			{
				rc = RC_SET(NE_SFLM_BTREE_BAD_STATE);
				goto Exit;
			}
		}
	}

	// The easiest case to handle is when we want to position within the
	// current entry.  We should not have to worry about the data only blocks
	// because the m_uiDataLength and m_uiDataRemaining are being set correctly
	// in setupReadState (via btLocateEntry, btNextEntry, btPrevEntry,
	// btFirstEntry and btLastEntry) which is always called before this method is
	// called.

	if( (uiPosition < (m_uiOffsetAtStart + m_uiDataLength)) &&
			(uiPosition >= m_uiOffsetAtStart))
	{
		m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
		m_uiOADataRemaining = m_uiOADataLength - uiPosition;
		goto Exit;
	}

	// Get the current block.  It is either a DO or a Btree block.
	
	if( m_pSCache == NULL)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}
	}

	// The next case is when the new position is in a *previous* entry, possibly
	// a previous block.
	
	while( uiPosition < m_uiOffsetAtStart)
	{
		pBlkHdr = m_pSCache->m_pBlkHdr;

		// Are we dealing with DataOnly blocks?
		
		if( m_bDataOnlyBlock)
		{
			ui32BlkAddr = pBlkHdr->ui32PrevBlkInChain;
			flmAssert( ui32BlkAddr);

			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;

			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				ui32BlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
			
			m_ui32CurBlkAddr = ui32BlkAddr;
			pBlkHdr = m_pSCache->m_pBlkHdr;

			m_uiDataLength = m_uiBlockSize - pBlkHdr->ui16BlkBytesAvail -
										sizeofDOBlkHdr((F_BLK_HDR *)pBlkHdr);
										
			if( !pBlkHdr->ui32PrevBlkInChain)
			{
				FLMBYTE *		pucPtr = (FLMBYTE *)pBlkHdr + sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr);
				FLMUINT16		ui16KeyLen = FB2UW( pucPtr);

				// We need to adjust for the key in the first block.
				
				m_uiDataLength -= ui16KeyLen;
			}
			
			// Decrement by the size of the current data
			
			m_uiOffsetAtStart -= m_uiDataLength;
		}
		else
		{
			// Backup to the previous element. This may or may not get
			// another block
			
			if( RC_BAD( rc = backupToPrevElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr, m_uiCurOffset);

			// Make sure we are still looking at the same key etc.
			
			if( !checkContinuedEntry(
				pucKey, uiKeyLen, &bLastElement, pucEntry,
				getBlkType( (FLMBYTE *)m_pSCache->m_pBlkHdr)))
			{
				// Should always match at this point!
				
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			m_uiDataLength = btGetEntryDataLength( pucEntry, NULL, NULL, NULL);
			m_uiOffsetAtStart -= m_uiDataLength;
		}
	}

	// Did we find the block?
	
	if( (uiPosition < (m_uiOffsetAtStart + m_uiDataLength)) &&
			(uiPosition >= m_uiOffsetAtStart))
	{
		m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
		m_uiOADataRemaining = m_uiOADataLength - uiPosition;
		goto Exit;
	}

	// Finally, we realize that the new position is beyond the current entry.
	
	while( uiPosition >= (m_uiOffsetAtStart + m_uiDataLength))
	{
		flmAssert( m_uiDataLength + m_uiOffsetAtStart <= m_uiOADataLength);

		// Get the next entry.
		
		pBlkHdr = m_pSCache->m_pBlkHdr;

		// Are we dealing with DataOnly blocks?
		
		if( m_bDataOnlyBlock)
		{
			ui32BlkAddr = pBlkHdr->ui32NextBlkInChain;
			flmAssert( ui32BlkAddr);

			ScaReleaseCache( m_pSCache, FALSE);
			m_pSCache = NULL;

			if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
				ui32BlkAddr, NULL, &m_pSCache)))
			{
				goto Exit;
			}
			
			m_ui32CurBlkAddr = ui32BlkAddr;
			pBlkHdr = m_pSCache->m_pBlkHdr;

			// Increment by the size of the previous data.  Note that in this
			// case, we do not have to be concerned about the key in the first
			// DO block since we will never move forward to it.
			
			m_uiOffsetAtStart += m_uiDataLength;
			m_uiDataLength = m_uiBlockSize - pBlkHdr->ui16BlkBytesAvail -
														sizeofDOBlkHdr((F_BLK_HDR *)pBlkHdr);
		}
		else
		{
			// Advance to the next element. This may or may not get another block.
			// Be sure we do not advance the stack since we do not have one.
			
			if( RC_BAD( rc = advanceToNextElement( FALSE)))
			{
				goto Exit;
			}

			pucEntry = BtEntry( (FLMBYTE *)m_pSCache->m_pBlkHdr, m_uiCurOffset);

			// Make sure we are still looking at the same key etc.
			
			if( !checkContinuedEntry(
				pucKey, uiKeyLen, &bLastElement, pucEntry,
				getBlkType( (FLMBYTE *)m_pSCache->m_pBlkHdr)))
			{
				// Should always match at this point!
				
				rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
				goto Exit;
			}

			// Get the data length of the current entry.
			
			m_uiOffsetAtStart += m_uiDataLength;
			m_uiDataLength = btGetEntryDataLength( pucEntry, NULL, NULL, NULL);
		}
	}

	// Did we find the block?  If we still don't find it, then we
	// have a big problem.
	
	if( (uiPosition >= (m_uiOffsetAtStart + m_uiDataLength)) ||
			(uiPosition < m_uiOffsetAtStart))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

	m_uiDataRemaining = m_uiDataLength - (uiPosition - m_uiOffsetAtStart);
	m_uiOADataRemaining = m_uiOADataLength - uiPosition;
	updateTransInfo( m_pSCache->m_pBlkHdr->ui64TransID,
						  m_pSCache->m_ui64HighTransID);

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_Btree::btGetReadPosition(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT *			puiPosition)
{
	RCODE					rc = NE_SFLM_OK;

	if( !m_bOpened || !m_bSetupForRead)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_BAD_STATE);
		goto Exit;
	}

	flmAssert( puiPosition);

	// If the transaction ID or the block change count has changed,
	// we must re-sync ourselves.
	
	if( !m_bTempDb &&
		 ((m_ui64CurrTransID != m_pDb->m_ui64CurrTransID) ||
		  (m_uiBlkChangeCnt != m_pDb->m_uiBlkChangeCnt)))
	{
		// Test to see if we really need to re-sync ...

		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
										m_ui32CurBlkAddr, NULL, &m_pSCache)))
		{
			goto Exit;
		}

		if( m_pSCache->m_pBlkHdr->ui64TransID != m_ui64LastBlkTransId ||
				(m_pDb->m_eTransType == SFLM_UPDATE_TRANS &&
					m_pDb->m_ui64CurrTransID == m_pSCache->m_pBlkHdr->ui64TransID))
		{
			// We must call btLocateEntry so we can re-initialize the read
			
			if( !m_bFirstRead)
			{
				if( RC_BAD( rc = btLocateEntry(
									pucKey, uiKeyLen, &uiKeyLen, FLM_EXACT)))
				{
					goto Exit;
				}

				// Will need a new version of this block.
				
				ScaReleaseCache( m_pSCache, FALSE);
				m_pSCache = NULL;
			}
			else
			{
				rc = RC_SET(NE_SFLM_BTREE_BAD_STATE);
				goto Exit;
			}
		}
	}

	*puiPosition = m_uiOffsetAtStart + (m_uiDataLength - m_uiDataRemaining);

	if( m_pSCache)
	{
		updateTransInfo( m_pSCache->m_pBlkHdr->ui64TransID,
							  m_pSCache->m_ui64HighTransID);
	}

Exit:

	if( m_pSCache)
	{
		ScaReleaseCache( m_pSCache, FALSE);
		m_pSCache = NULL;
	}

	releaseBlocks( FALSE);
	return( rc);
}

/***************************************************************************
Desc: Performs a consistancy check on the BTree
		NOTE: Must be performed inside of a read transaction!
****************************************************************************/
RCODE F_Btree::btCheck(
	BTREE_ERR_STRUCT *	pErrStruct)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT32			ui32NextBlkAddr = 0;
	FLMUINT32			ui32NextLevelBlkAddr = 0;
	FLMUINT32			ui32ChildBlkAddr = 0;
	FLMUINT32			ui32DOBlkAddr = 0;
	FLMUINT				uiNumKeys;
	const FLMBYTE *	pucPrevKey;
	FLMUINT				uiPrevKeySize;
	const FLMBYTE *	pucCurKey;
	FLMUINT				uiCurKeySize;
	F_CachedBlock *	pCurrentBlk = NULL;
	F_CachedBlock *	pPrevSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMBYTE *			pucEntry = NULL;
	FLMBYTE *			pucPrevEntry = NULL;
	F_CachedBlock *	pChildBlk = NULL;
	FLMUINT16 *			puiOffsetArray;
	BTREE_ERR_STRUCT	localErrStruct;
	FLMINT				iCmpResult;
	FLMUINT				uiOADataLength = 0;

	// Verify the Txn type
	
	if( m_pDb->m_eTransType == SFLM_NO_TRANS && !m_bTempDb)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// Initial setup...
	
	ui32NextLevelBlkAddr = (FLMUINT32)m_pLFile->uiRootBlk;
	f_memset( &localErrStruct, 0, sizeof( localErrStruct));
	localErrStruct.uiBlockSize = m_uiBlockSize;

	// While there's a next level....
	
	while( ui32NextLevelBlkAddr)
	{
		localErrStruct.uiLevels++;
		ui32NextBlkAddr = ui32NextLevelBlkAddr;

		// Update uiNextLevelBlkAddr
		
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32NextBlkAddr, NULL, &pCurrentBlk)))
		{
			localErrStruct.type = GET_BLOCK_FAILED;
			f_sprintf( localErrStruct.szMsg, 
				"Failed to get block at %X", ui32NextBlkAddr);
			goto Exit;
		}
		
		pBlk = (FLMBYTE *)pCurrentBlk->m_pBlkHdr;
		puiOffsetArray = BtOffsetArray( pBlk, 0);
		
		if( (getBlkType( pBlk) == BT_LEAF) || (getBlkType( pBlk) == BT_LEAF_DATA))
		{
			ui32NextLevelBlkAddr = 0;
		}
		else
		{
			pucEntry = BtEntry( pBlk, 0);

			// The child block address is the first part of the entry
			
			ui32NextLevelBlkAddr = bteGetBlkAddr( pucEntry);
		}

		if( pPrevSCache)
		{
			ScaReleaseCache( pPrevSCache, FALSE);
			pPrevSCache = NULL;
		}

		// While there's another block on this level...
		
		while( ui32NextBlkAddr) 
		{
			// This loop assumes that pCurrentBlk and pBlk are already initialized.
			
			localErrStruct.uiBlocksChecked++;
			localErrStruct.uiAvgFreeSpace =
				(localErrStruct.uiAvgFreeSpace * (localErrStruct.uiBlocksChecked - 1) / 
					localErrStruct.uiBlocksChecked) +
				(getBlkAvailSpace(pBlk) / localErrStruct.uiBlocksChecked);
			localErrStruct.ui64FreeSpace += getBlkAvailSpace(pBlk);

			localErrStruct.LevelStats[ localErrStruct.uiLevels - 1].uiDOBlockCnt++;
			localErrStruct.LevelStats[ localErrStruct.uiLevels - 1].uiDOBytesUsed +=
										(m_uiBlockSize - getBlkAvailSpace(pBlk));

			uiNumKeys = ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys;

			// VISIT:  Verify the block header fields
			/*
				ui32PrevBlkInChain =
				ui32BlkCRC =
				ui16BlkBytesAvail < ?
				ui8BlkLevel??
				ui8BlkIsRoot??
			*/

			// Verify that the keys are in order...
			// Make sure that we check the keys between blocks as well.
			
			if( pPrevSCache)
			{
				pucEntry = BtLastEntry( (FLMBYTE *)pPrevSCache->m_pBlkHdr);
				uiPrevKeySize = getEntryKeyLength( pucEntry, getBlkType(
							(FLMBYTE *)pPrevSCache->m_pBlkHdr), &pucPrevKey);
			}
			else
			{
				pucEntry = BtEntry( pBlk, 0);
				uiPrevKeySize = getEntryKeyLength( pucEntry, getBlkType( pBlk),
															  &pucPrevKey);
				if( getBlkType(pBlk) == BT_LEAF_DATA)
				{
					if( bteFirstElementFlag( pucEntry))
					{
						localErrStruct.LevelStats[ 
								localErrStruct.uiLevels - 1].uiFirstKeyCnt++;
					}
				}
				else
				{
					// Everything else is a first key.
					
					localErrStruct.LevelStats[
						localErrStruct.uiLevels - 1].uiFirstKeyCnt++;
				}
			}
			
			for( FLMUINT uiLoop = (pPrevSCache ? 0: 1); 
				  uiLoop < uiNumKeys; uiLoop++)
			{
				pucPrevEntry = pucEntry;
				pucEntry = BtEntry( pBlk, uiLoop);

				if( getBlkType(pBlk) == BT_LEAF_DATA)
				{
					if( bteFirstElementFlag( pucEntry))
					{
						localErrStruct.LevelStats[ 
							localErrStruct.uiLevels - 1].uiFirstKeyCnt++;
					}
				}
				else
				{
					// Everything else is a first key.
					
					localErrStruct.LevelStats[
						localErrStruct.uiLevels - 1].uiFirstKeyCnt++;
				}

				uiCurKeySize = getEntryKeyLength( pucEntry,
										getBlkType( pBlk), &pucCurKey);

				// The last key in the last block of each level is an infinity marker
				// It must have a 0 keylength and if it's a leaf node, a 0 datalength.
				
				if( (uiLoop == uiNumKeys - 1) &&
					(((F_BLK_HDR *)pBlk)->ui32NextBlkInChain == 0))
				{
					// If the key size is not 0, or we're a leaf block, and the
					// data size is not 0 ...
					
					if( (uiCurKeySize != 0) ||
						  (((getBlkType( pBlk) == BT_LEAF_DATA)) &&
							 (btGetEntryDataLength( pucEntry, NULL, NULL, NULL) > 0)))
					{
						localErrStruct.type = INFINITY_MARKER;
						localErrStruct.uiBlkAddr = ((F_BLK_HDR *)pBlk)->ui32BlkAddr;
						f_sprintf( localErrStruct.szMsg, "Invalid Infinity Marker %ul", uiLoop);
						rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
						goto Exit;
					}
				}
				else
				{
					// Do a comparison of the previous and current keys ...
					
					if( RC_BAD( rc = compareKeys( pucPrevKey, uiPrevKeySize,
						pucCurKey, uiCurKeySize, &iCmpResult)))
					{
						goto Exit;
					}
					
					if( iCmpResult > 0)
					{
						localErrStruct.type = KEY_ORDER;
						localErrStruct.uiBlkAddr = ((F_BLK_HDR *)pBlk)->ui32BlkAddr;
						f_sprintf( localErrStruct.szMsg, "Key Number %ul", uiLoop);
						rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
						goto Exit;
					}

					if( getBlkType(pBlk) == BT_LEAF_DATA)
					{
						if( iCmpResult < 0)
						{
							flmAssert( *pucEntry & BTE_FLAG_FIRST_ELEMENT);
						}
						else if( iCmpResult == 0)
						{
							flmAssert( (*pucEntry & BTE_FLAG_FIRST_ELEMENT) == 0);
							flmAssert( (*pucPrevEntry & BTE_FLAG_LAST_ELEMENT) == 0);
						}
					}
				}

				pucPrevKey = pucCurKey;
				uiPrevKeySize = uiCurKeySize;
			}

			localErrStruct.uiNumKeys += uiNumKeys;
			localErrStruct.LevelStats[ 
							localErrStruct.uiLevels - 1].uiKeyCnt += uiNumKeys;

			// If this is a leaf block, check for any pointers to data-only
			// blocks.  Verify the blocks...
			
			if( getBlkType( pBlk) == BT_LEAF || 
				 getBlkType( pBlk) == BT_LEAF_DATA)
			{
				if( getBlkType( pBlk) == BT_LEAF_DATA)
				{
					for( FLMUINT uiLoop = 0; uiLoop < uiNumKeys; uiLoop++)
					{
						pucEntry = BtEntry( pBlk, uiLoop);
						
						if( bteDataBlockFlag( pucEntry))
						{
							FLMBYTE	ucDOBlkAddr[ 4];

							if( RC_BAD( rc = btGetEntryData( pucEntry, 
								&ucDOBlkAddr[ 0], 4, NULL)))
							{
								RC_UNEXPECTED_ASSERT( rc);
								localErrStruct.type = CATASTROPHIC_FAILURE;
								localErrStruct.uiBlkAddr = ((F_BLK_HDR *)pBlk)->ui32BlkAddr;
								f_sprintf( localErrStruct.szMsg, 
										"getEntryData couldn't get the DO blk addr.");
								goto Exit;
							}

							ui32DOBlkAddr = bteGetBlkAddr( (FLMBYTE *)&ucDOBlkAddr[ 0]);

							// Verify that there is an OverallDataLength field

							if( bteOADataLenFlag( pucEntry) == 0)
							{
								localErrStruct.type = MISSING_OVERALL_DATA_LENGTH;
								localErrStruct.uiBlkAddr = ((F_BLK_HDR *)pBlk)->ui32BlkAddr;
								f_sprintf( localErrStruct.szMsg, 
									"OverallDataLength field is missing");
							}
							else
							{
								if( bteKeyLenFlag( pucEntry))
								{
									uiOADataLength = FB2UD( pucEntry + 4);
								}
								else
								{
									uiOADataLength = FB2UD( pucEntry + 3);
								}
							}

							if( RC_BAD( rc = verifyDOBlkChain( ui32DOBlkAddr,
								uiOADataLength , &localErrStruct)))
							{
								goto Exit;
							}
						}
					}
				}
			}
			else
			{
				// This is a non-leaf block, verify that blocks exist for all
				// the child block addresses

				// NOTE: Also need to somehow verify that no two elements have the
				// same child block address...				
					
				for( FLMUINT uiLoop = 0; uiLoop < uiNumKeys; uiLoop++)
				{
					pucEntry = BtEntry( pBlk, uiLoop);
					ui32ChildBlkAddr = bteGetBlkAddr( pucEntry);
					if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
						ui32ChildBlkAddr, NULL, &pChildBlk)))
					{
						localErrStruct.type = GET_BLOCK_FAILED;
						f_sprintf( localErrStruct.szMsg, "Failed to get block at %X", 
							ui32ChildBlkAddr);
						goto Exit;
					}

					ScaReleaseCache( pChildBlk, FALSE);
				}
			}

			// Release the current block and get the next one
			
			ui32NextBlkAddr = ((F_BLK_HDR *)pBlk)->ui32NextBlkInChain;
			
			if( pPrevSCache)
			{
				ScaReleaseCache( pPrevSCache, FALSE);
				pPrevSCache = NULL;
			}
			
			pPrevSCache = pCurrentBlk;
			pCurrentBlk = NULL;
			
			if( ui32NextBlkAddr)
			{
				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
												ui32NextBlkAddr, NULL, &pCurrentBlk)))
				{
					localErrStruct.type = GET_BLOCK_FAILED;
					f_sprintf( localErrStruct.szMsg, 
						"Failed to get block at %X", ui32ChildBlkAddr);
					goto Exit;
				}
				
				pBlk = (FLMBYTE *)pCurrentBlk->m_pBlkHdr;
			}
		}
	}

	if( m_bCounts)
	{
		if( RC_BAD( rc = verifyCounts( &localErrStruct)))
		{
			goto Exit;
		}
	}

Exit:

	if( pPrevSCache)
	{
		ScaReleaseCache( pPrevSCache, FALSE);
	}

	if( pCurrentBlk)
	{
		ScaReleaseCache( pCurrentBlk, FALSE);
	}
	
	f_memcpy( pErrStruct, &localErrStruct, sizeof( localErrStruct));
	return( rc);
}

/***************************************************************************
Desc: Performs an integrity check on a chain of data-only blocks.  Should
		only be called from btCheck().  Note that unlike btCheck(),
		errStruct CANNOT be NULL here.
****************************************************************************/
RCODE F_Btree::verifyDOBlkChain(
	FLMUINT					uiDOAddr,			// Address of first block in chain
	FLMUINT					uiDataLength,		// The length of the entire entry
	BTREE_ERR_STRUCT *	errStruct)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiRunningLength = 0; // A running total of the DataLength fields
														// for all of the blocks in this chain
	F_CachedBlock *	pCurrentBlk = NULL;
	FLMUINT32			ui32NextAddr = (FLMUINT32)uiDOAddr;
	FLMBYTE *			pBlk;
	FLMUINT				uiDataSize;

	while( ui32NextAddr)
	{
		errStruct->LevelStats[ errStruct->uiLevels - 1].uiDOBlockCnt++;
		
		// Get the next block
		
		if( RC_BAD( m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
			ui32NextAddr, NULL, &pCurrentBlk)))
		{
			errStruct->type = GET_BLOCK_FAILED;
			f_sprintf( errStruct->szMsg, "Failed to get block at %X", uiDOAddr);
			goto Exit;
		}
		
		pBlk = (FLMBYTE *)pCurrentBlk->m_pBlkHdr;

		// Verify that it's really a DO Block
		
		if( getBlkType( pBlk) != BT_DATA_ONLY)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
			errStruct->type = NOT_DATA_ONLY_BLOCK;
			goto Exit;
		}

		// Update counts info in errStruct
		
		errStruct->LevelStats[ errStruct->uiLevels - 1].uiDOBytesUsed +=
						m_uiBlockSize - pCurrentBlk->m_pBlkHdr->ui16BlkBytesAvail;
						
		// Update the data length running total
		
		uiDataSize = m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlk) -
										((F_BLK_HDR *)pBlk)->ui16BlkBytesAvail;
										
		if( ((F_BLK_HDR *)pBlk)->ui32PrevBlkInChain == 0)
		{
			FLMBYTE *		pucPtr = pBlk + sizeofDOBlkHdr( (F_BLK_HDR *)pBlk);
			FLMUINT16		ui16KeyLen = FB2UW( pucPtr);
			
			uiDataSize -= (ui16KeyLen + 2);
		}
		
		uiRunningLength += uiDataSize;

		// Update ui32nextAddr
		
		ui32NextAddr = ((F_BLK_HDR *)pBlk)->ui32NextBlkInChain;

		// Release it when we no longer need it.
		
		ScaReleaseCache( pCurrentBlk, FALSE);
		pCurrentBlk = NULL;
	}

	// Check the calculated overall length vs. uiDataLength
	
	if( uiRunningLength != uiDataLength)
	{
		errStruct->type = BAD_DO_BLOCK_LENGTHS;
		rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
		goto Exit;
	}

Exit:

	if( pCurrentBlk)
	{
		ScaReleaseCache( pCurrentBlk, FALSE);
	}

	if( rc == NE_SFLM_BTREE_ERROR)
	{
		f_sprintf( errStruct->szMsg, "Corrupt DO chain starting at %X", uiDOAddr);
	}

	return( NE_SFLM_OK);
}

/***************************************************************************
Desc:	Method to check the counts in a database with counts.
****************************************************************************/
RCODE F_Btree::verifyCounts(
	BTREE_ERR_STRUCT *	pErrStruct)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiNextLevelBlkAddr;
	FLMUINT				uiNextBlkAddr;
	FLMUINT				uiChildBlkAddr;
	F_CachedBlock *	pCurrentBlk = NULL;
	F_CachedBlock *	pChildBlk = NULL;
	FLMBYTE *			pucEntry;
	FLMUINT				uiNumKeys;
	FLMUINT				uiEntryNum;
	FLMUINT				uiParentCounts;
	FLMUINT				uiChildCounts;
	FLMBYTE *			pBlk;
	FLMBOOL				bDone = FALSE;

	flmAssert( m_bCounts);

	// Repeat at each level, starting at the root.
	
	uiNextLevelBlkAddr = m_pLFile->uiRootBlk;

	while( uiNextLevelBlkAddr)
	{
		if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
			m_pDb, m_pLFile, uiNextLevelBlkAddr, NULL, &pCurrentBlk)))
		{
			goto Exit;
		}

		if( pCurrentBlk->m_pBlkHdr->ui8BlkType != BT_NON_LEAF_COUNTS)
		{
			ScaReleaseCache( pCurrentBlk, FALSE);
			pCurrentBlk = NULL;
			break;
		}

		pucEntry = BtEntry( (FLMBYTE *)pCurrentBlk->m_pBlkHdr, 0);
		uiNextLevelBlkAddr = bteGetBlkAddr( pucEntry);

		// For every entry in the block, and for every block on this level,
		// check that the counts match the actual counts in the corresponding
		// child block.
		
		bDone = FALSE;
		while( !bDone)
		{
			uiNumKeys = ((F_BTREE_BLK_HDR *)pCurrentBlk->m_pBlkHdr)->ui16NumKeys;
			pBlk = (FLMBYTE *)pCurrentBlk->m_pBlkHdr;

			// Now check every entry in this block.
			
			for( uiEntryNum = 0; uiEntryNum < uiNumKeys; uiEntryNum++)
			{
				pucEntry = BtEntry( pBlk, uiEntryNum);
				uiChildBlkAddr = bteGetBlkAddr( pucEntry);

				pucEntry += 4;
				uiParentCounts = FB2UD( pucEntry);

				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock( m_pDb, m_pLFile,
											uiChildBlkAddr, NULL, &pChildBlk)))
				{
					goto Exit;
				}

				uiChildCounts = countKeys( (FLMBYTE *)pChildBlk->m_pBlkHdr);

				if( uiChildCounts != uiParentCounts)
				{
					pErrStruct->type = BAD_COUNTS;
					pErrStruct->uiBlkAddr = pChildBlk->m_pBlkHdr->ui32BlkAddr;
					f_sprintf(
						pErrStruct->szMsg,
						"Counts do not match.  Expected %d, got %d",
						uiParentCounts, uiChildCounts);
					rc = RC_SET_AND_ASSERT( NE_SFLM_BTREE_ERROR);
					goto Exit;
				}

				ScaReleaseCache( pChildBlk, FALSE);
				pChildBlk = NULL;
			}

			// Now get the next block at this level.
			
			uiNextBlkAddr = pCurrentBlk->m_pBlkHdr->ui32NextBlkInChain;
			ScaReleaseCache( pCurrentBlk, FALSE);
			pCurrentBlk = NULL;

			if( uiNextBlkAddr == 0)
			{
				bDone = TRUE;
			}
			else
			{
				if( RC_BAD( rc = m_pDb->m_pDatabase->getBlock(
					m_pDb, m_pLFile, uiNextBlkAddr, NULL, &pCurrentBlk)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	if( pCurrentBlk)
	{
		ScaReleaseCache( pCurrentBlk, FALSE);
	}

	if( pChildBlk)
	{
		ScaReleaseCache( pChildBlk, FALSE);
	}
	
	return( rc);
}
