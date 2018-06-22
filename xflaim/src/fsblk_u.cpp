//------------------------------------------------------------------------------
// Desc:	Contains routines for handling blocks in the avail list - putting
//			blocks into the avail list and removing them from the avail list.
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

/***************************************************************************
Desc:	Take current avail block off of free list.  Have db header point
		to next block in the chain.
*****************************************************************************/
RCODE F_Database::blockUseNextAvail(
	F_Db *				pDb,
	F_CachedBlock **	ppSCache	// Returns pointer to cache structure.
	)
{
	RCODE					rc = NE_XFLM_OK;
	F_CachedBlock *	pSCache = NULL;
	F_CachedBlock *	pNextSCache = NULL;
	XFLM_DB_HDR *		pDbHdr;

	pDbHdr = &m_uncommittedDbHdr;

	if (RC_BAD( rc = getBlock( pDb, NULL, pDb->m_uiFirstAvailBlkAddr,
									NULL, &pSCache)))
	{
		goto Exit;
	}

	// A corruption we have seen a couple of times is where a free
	// block points to itself in the free list.  This will hang the machine
	// so this check has been added to verify that the block is a free block.

	if (pSCache->m_pBlkHdr->ui8BlkType != BT_FREE ||
		 isEncryptedBlk( pSCache->m_pBlkHdr))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Log the block because we are changing it!

	if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}

	*ppSCache = pSCache;

	pDb->m_uiFirstAvailBlkAddr = (FLMUINT)pSCache->m_pBlkHdr->ui32NextBlkInChain;
	pDbHdr->ui32FirstAvailBlkAddr = (FLMUINT32)pDb->m_uiFirstAvailBlkAddr;
	pSCache->m_pBlkHdr->ui32NextBlkInChain = 0;

	// Set the next block's previous to zero.

	if (pDb->m_uiFirstAvailBlkAddr)
	{
		if (RC_BAD( rc = getBlock( pDb, NULL, pDb->m_uiFirstAvailBlkAddr,
										NULL, &pNextSCache)))
		{
			goto Exit;
		}

		if (pNextSCache->m_pBlkHdr->ui8BlkType != BT_FREE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		// Log the block because we are changing it

		if (RC_BAD( rc = logPhysBlk( pDb, &pNextSCache)))
		{
			goto Exit;
		}
		pNextSCache->m_pBlkHdr->ui32PrevBlkInChain = 0;
		ScaReleaseCache( pNextSCache, FALSE);
		pNextSCache = NULL;
	}

Exit:

	if (RC_BAD( rc))
	{
		if (pSCache)
		{
			ScaReleaseCache( pSCache, FALSE);
		}
		if (pNextSCache)
		{
			ScaReleaseCache( pNextSCache, FALSE);
		}
	}

	return( rc);
}

/***************************************************************************
Desc:		Put a block into the avail list.
Notes:	This routine assumes that the block pointed to by pSCache has
			been locked into memory.  Regardless of whether or not the block
			is actually freed on disk, its cache will be released.  The
			cached block should NOT be accessed after a call to FSBlockFree.
*****************************************************************************/
RCODE F_Database::blockFree(
	F_Db *				pDb,
	F_CachedBlock *	pSCache	// Pointer to pointer of cache block
										// that is to be freed.  NOTE: Regardless of whether
										// or not the block is actually freed, it will be
										// released.
	)
{
	RCODE					rc = NE_XFLM_OK;
	F_BLK_HDR *			pBlkHdr;
	F_CachedBlock *	pFirstAvailSCache;
	XFLM_DB_HDR *		pDbHdr;

	pDbHdr = &m_uncommittedDbHdr;

	// Log the block before modifying it.

	if (RC_BAD( rc = logPhysBlk( pDb, &pSCache)))
	{
		goto Exit;
	}
	pBlkHdr = pSCache->m_pBlkHdr;

	// Modify header to be an avail block.

	if (isEncryptedBlk( pBlkHdr))
	{
		// If block was previously encrypted, need to zero it
		// out so that clear data will not be written to disk
		// when this avail block is written out.
		
		unsetBlockEncrypted( pBlkHdr);
		f_memset( ((FLMBYTE *)pBlkHdr) + SIZEOF_STD_BLK_HDR, 0,
			m_uiBlockSize - SIZEOF_STD_BLK_HDR);
	}
	pBlkHdr->ui8BlkType = BT_FREE;
	pBlkHdr->ui16BlkBytesAvail = (FLMUINT16)(m_uiBlockSize -
															SIZEOF_STD_BLK_HDR);

	// Modify back chain of first block in avail list, if any,
	// to point back to this block.

	if (pDbHdr->ui32FirstAvailBlkAddr)
	{

		// Read the block at head of avail list.

		if (RC_BAD( rc = getBlock( pDb, NULL,
									(FLMUINT)pDbHdr->ui32FirstAvailBlkAddr,
									NULL, &pFirstAvailSCache)))
		{
			goto Exit;
		}

		// Log the block

		if (RC_OK( rc = logPhysBlk( pDb, &pFirstAvailSCache)))
		{

			// Previous block pointer better be zero at this point.

			flmAssert( pFirstAvailSCache->m_pBlkHdr->ui32PrevBlkInChain == 0);
			pFirstAvailSCache->m_pBlkHdr->ui32PrevBlkInChain = pBlkHdr->ui32BlkAddr;
		}
		ScaReleaseCache( pFirstAvailSCache, FALSE);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// Link block at head of avail list.

	pBlkHdr->ui32PrevBlkInChain = 0;
	pBlkHdr->ui32NextBlkInChain = pDbHdr->ui32FirstAvailBlkAddr;
	pDbHdr->ui32FirstAvailBlkAddr = pBlkHdr->ui32BlkAddr;
	pDb->m_uiFirstAvailBlkAddr = (FLMUINT)pBlkHdr->ui32BlkAddr;

Exit:

	ScaReleaseCache( pSCache, FALSE);
	return( rc);
}
