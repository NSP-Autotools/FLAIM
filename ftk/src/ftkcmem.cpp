//------------------------------------------------------------------------------
// Desc: Block cache allocator
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
typedef struct SLABINFO
{
	void *			pvSlab;
	SLABINFO *		pPrevInGlobal;
	SLABINFO *		pNextInGlobal;
	SLABINFO *		pPrevInBucket;
	SLABINFO *		pNextInBucket;
	SLABINFO *		pPrevSlabWithAvail;
	SLABINFO *		pNextSlabWithAvail;
	FLMUINT8			ui8NextNeverUsed;
	FLMUINT8			ui8AvailBlocks;
	FLMUINT8			ui8FirstAvail;
	FLMUINT8			ui8AllocatedBlocks;
#define F_ALLOC_MAP_BYTES	4
#define F_ALLOC_MAP_BITS	(F_ALLOC_MAP_BYTES * 8)
	FLMBYTE			ucAllocMap[ F_ALLOC_MAP_BYTES];
} SLABINFO;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct AVAILBLOCK
{
	FLMUINT8			ui8NextAvail;
} AVAILBLOCK;

/****************************************************************************
Desc:
****************************************************************************/
class F_BlockAlloc : public IF_BlockAlloc
{
public:

	F_BlockAlloc();

	virtual ~F_BlockAlloc();

	RCODE FTKAPI setup(
		FLMBOOL					bMultiThreaded,
		IF_SlabManager *		pSlabManager,
		IF_Relocator *			pRelocator,
		FLMUINT					uiBlockSize,
		FLM_SLAB_USAGE *		pUsageStats,
		FLMUINT *				puiTotalBytesAllocated);

	RCODE FTKAPI allocBlock(
		void **					ppvBlock);

	void FTKAPI freeBlock(
		void **					ppvBlock);

	void FTKAPI freeUnused( void);

	void FTKAPI freeAll( void);

	void FTKAPI defragmentMemory( void);
	
private:

	void cleanup( void);

	RCODE getCell(
		SLABINFO **				ppSlab,
		void **					ppvCell);

	RCODE getAnotherSlab(
		SLABINFO **				ppSlab);
		
	void freeSlab(
		SLABINFO **				ppSlab);

	void freeCell(
		SLABINFO **				ppSlab,
		void **					ppvCell);
		
	FINLINE FLMUINT getHashBucket(
		void *				pvAlloc)
	{
		return( (((FLMUINT)pvAlloc) & m_uiHashMask) % m_uiBuckets);
	}

	IF_SlabManager *			m_pSlabManager;
	IF_Relocator *				m_pRelocator;
	IF_FixedAlloc *			m_pInfoAllocator;
	SLABINFO *					m_pFirstSlab;
	SLABINFO *					m_pLastSlab;
	SLABINFO *					m_pFirstSlabWithAvail;
	SLABINFO *					m_pLastSlabWithAvail;
	SLABINFO **					m_pHashTable;
	FLMUINT						m_uiBuckets;
	FLMUINT						m_uiHashMask;
	FLMBOOL						m_bAvailListSorted;
	FLMUINT						m_uiSlabSize;
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiBlocksPerSlab;
	FLMUINT						m_uiSlabsWithAvail;
	FLMUINT						m_uiTotalAvailBlocks;
	FLM_SLAB_USAGE *			m_pUsageStats;
	FLMUINT *					m_puiTotalBytesAllocated;
	F_MUTEX						m_hMutex;
	
friend class F_SlabInfoRelocator;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_SlabInfoRelocator : public IF_Relocator
{
public:

	F_SlabInfoRelocator()
	{
		m_pBlockAlloc = NULL;
	}
	
	virtual ~F_SlabInfoRelocator()
	{
	}

	void FTKAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL FTKAPI canRelocate(
		void *	pvOldAlloc);
		
private:

	F_BlockAlloc *		m_pBlockAlloc;
		
friend class F_BlockAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI slabInfoAddrCompareFunc(
	void *					pvBuffer,
	FLMUINT					uiPos1,
	FLMUINT					uiPos2)
{
	SLABINFO *		pSlab1 = (((SLABINFO **)pvBuffer)[ uiPos1]);
	SLABINFO *		pSlab2 = (((SLABINFO **)pvBuffer)[ uiPos2]);

	f_assert( pSlab1 != pSlab2);

	if( pSlab1->pvSlab < pSlab2->pvSlab)
	{
		return( -1);
	}

	return( 1);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI slabInfoAddrSwapFunc(
	void *					pvBuffer,
	FLMUINT					uiPos1,
	FLMUINT					uiPos2)
{
	SLABINFO **		ppSlab1 = &(((SLABINFO **)pvBuffer)[ uiPos1]);
	SLABINFO **		ppSlab2 = &(((SLABINFO **)pvBuffer)[ uiPos2]);
	SLABINFO *		pTmp;

	pTmp = *ppSlab1;
	*ppSlab1 = *ppSlab2;
	*ppSlab2 = pTmp;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_BlockAlloc::F_BlockAlloc()
{
	m_pSlabManager = NULL;
	m_pRelocator = NULL;
	m_pInfoAllocator = NULL;
	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pFirstSlabWithAvail = NULL;
	m_pLastSlabWithAvail = NULL;
	m_pHashTable = NULL;
	m_uiBuckets = 0;
	m_uiHashMask = 0;
	m_bAvailListSorted = FALSE;
	m_uiSlabSize = 0;
	m_uiBlockSize = 0;
	m_uiBlocksPerSlab = 0;
	m_uiSlabsWithAvail = 0;
	m_uiTotalAvailBlocks = 0;
	m_pUsageStats = NULL;
	m_puiTotalBytesAllocated = NULL;
	m_hMutex = F_MUTEX_NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_BlockAlloc::~F_BlockAlloc()
{
	cleanup();
}

/****************************************************************************
Desc:
****************************************************************************/
void F_BlockAlloc::cleanup( void)
{
	freeAll();
	
	if( m_pHashTable)
	{
		f_free( &m_pHashTable);
	}
	
	if( m_pInfoAllocator)
	{
		m_pInfoAllocator->Release();
		m_pInfoAllocator = NULL;
	}
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
		m_pSlabManager = NULL;
	}
	
	if( m_pRelocator)
	{
		m_pRelocator->Release();
		m_pRelocator = NULL;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BlockAlloc::setup(
	FLMBOOL						bMultiThreaded,
	IF_SlabManager *			pSlabManager,
	IF_Relocator *				pRelocator,
	FLMUINT						uiBlockSize,
	FLM_SLAB_USAGE *			pUsageStats,
	FLMUINT *					puiTotalBytesAllocated)
{
	RCODE							rc = NE_FLM_OK;
	F_SlabInfoRelocator *	pSlabInfoRelocator = NULL;

	f_assert( pSlabManager);
	f_assert( pRelocator);
	f_assert( uiBlockSize);
	f_assert( pUsageStats);
	
	if( uiBlockSize != 4096 && uiBlockSize != 8192)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM);
		goto Exit;
	}
	
	if( bMultiThreaded)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}
	
	m_pUsageStats = pUsageStats;
	m_puiTotalBytesAllocated = puiTotalBytesAllocated;
	
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	m_pRelocator = pRelocator;
	m_pRelocator->AddRef();
	
	m_uiBlockSize = uiBlockSize;
	m_uiSlabSize = m_pSlabManager->getSlabSize();
	f_assert( (m_uiSlabSize % 1024) == 0);
	
	m_uiBlocksPerSlab = m_uiSlabSize / m_uiBlockSize;
	f_assert( F_ALLOC_MAP_BITS >= m_uiBlocksPerSlab);
	
	// Set up the SLABINFO allocator
	
	if( RC_BAD( rc = FlmAllocFixedAllocator( &m_pInfoAllocator)))
	{
		goto Exit;
	}
	
	if( (pSlabInfoRelocator = f_new F_SlabInfoRelocator) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	pSlabInfoRelocator->m_pBlockAlloc = this;
	
	if( RC_BAD( rc = m_pInfoAllocator->setup( FALSE, m_pSlabManager, 
		pSlabInfoRelocator, sizeof( SLABINFO), 
		m_pUsageStats, puiTotalBytesAllocated)))
	{
		goto Exit;
	}
	
	// Set up the hash table
	
	m_uiBuckets = m_uiSlabSize - 1;
	if( RC_BAD( rc = f_calloc( sizeof( SLABINFO *) * m_uiBuckets, 
		&m_pHashTable)))
	{
		goto Exit;
	}

	m_uiHashMask = ~(FLMUINT)((m_uiSlabSize - 1));

Exit:

	if( pSlabInfoRelocator)
	{
		pSlabInfoRelocator->Release();
	}
	
	if( RC_BAD( rc))
	{
		cleanup();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_BlockAlloc::allocBlock(
	void **				ppvBlock)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( RC_BAD( rc = getCell( NULL, ppvBlock)))
	{
		goto Exit;
	}

#ifndef FLM_NLM
	f_assert( ((FLMUINT)(*ppvBlock) % m_uiBlockSize) == 0);
#endif

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_BlockAlloc::freeBlock(
	void **				ppvBlock)
{
	SLABINFO *			pSlab = NULL;
	void *				pvBlock = *ppvBlock;
	FLMUINT				uiLoop;
	FLMUINT				uiBucket = 0;
	FLMUINT				uiDelta = m_uiSlabSize - m_uiBlockSize;
	FLMBOOL				bMutexLocked = FALSE;
	
	f_assert( pvBlock);
#ifndef FLM_NLM
	f_assert( ((FLMUINT)pvBlock % m_uiBlockSize) == 0);
#endif
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	for( uiLoop = 0; uiLoop < 3; uiLoop++)
	{
		switch( uiLoop)
		{
			case 0:
			{
				uiBucket = getHashBucket( pvBlock);
				break;
			}
			
			case 1:
			{
				if( ((FLMUINT)pvBlock) > FLM_MAX_UINT - uiDelta)
				{
					break;
				}
				
				uiBucket = getHashBucket( ((FLMBYTE *)pvBlock) + uiDelta);
				break;
			}
				
			case 2:
			{
				if( ((FLMUINT)pvBlock) <= uiDelta)
				{
					break;
				}
				
				uiBucket = getHashBucket( ((FLMBYTE *)pvBlock) - uiDelta);
				break;
			}
		}
		
		pSlab = m_pHashTable[ uiBucket];
		
		while( pSlab)
		{
			if( pvBlock >= pSlab->pvSlab && 
				pvBlock <= ((FLMBYTE *)pSlab->pvSlab + uiDelta))
			{
				goto FoundSlab;
			}
			
			pSlab = pSlab->pNextInBucket;
		}
	}

FoundSlab:
	
	if( !pSlab || !pSlab->pvSlab)
	{
		f_assert( 0);
		goto Exit;
	}
	
	freeCell( &pSlab, ppvBlock);
	
Exit:
	
	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BlockAlloc::getCell(
	SLABINFO **		ppSlab,
	void **			ppvCell)
{
	RCODE				rc = NE_FLM_OK;
	AVAILBLOCK *	pAvailBlock;
	SLABINFO *		pSlab = NULL;
	FLMBYTE *		pCell = NULL;

#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	// If there is a slab that has an avail cell, that one gets priority

	if( (pSlab = m_pFirstSlabWithAvail) != NULL)
	{
		f_assert( pSlab->ui8AvailBlocks <= m_uiTotalAvailBlocks);
		f_assert( m_uiTotalAvailBlocks);
		f_assert( pSlab->ui8AllocatedBlocks < m_uiBlocksPerSlab);
		f_assert( !f_isBitSet( pSlab->ucAllocMap, pSlab->ui8FirstAvail));

		pAvailBlock = (AVAILBLOCK *)(((FLMBYTE *)pSlab->pvSlab) + 
									(pSlab->ui8FirstAvail * m_uiBlockSize));

		pSlab->ui8AllocatedBlocks++;
		pSlab->ui8AvailBlocks--;
		m_uiTotalAvailBlocks--;
		
		f_assert( !f_isBitSet( pSlab->ucAllocMap, pSlab->ui8FirstAvail));
		f_setBit( pSlab->ucAllocMap, pSlab->ui8FirstAvail);
		f_assert( f_isBitSet( pSlab->ucAllocMap, pSlab->ui8FirstAvail));
		
		// A free block holds as its contents the next pointer in the free chain.
		// Free chains do not span slabs.

		pSlab->ui8FirstAvail = pAvailBlock->ui8NextAvail;

		// If there are no other free blocks in this slab, we need to unlink 
		// the slab from the slabs-with-avail-blocks list
		
		if( !pSlab->ui8AvailBlocks)
		{
			// Save a copy of the slab we're going to unlink

			SLABINFO * 		pSlabToUnlink = pSlab;

			f_assert( !pSlabToUnlink->ui8AvailBlocks);
			f_assert( !pSlabToUnlink->pPrevSlabWithAvail);				

			// Update m_pFirstSlabWithAvail to point to the next one
			// and unlink from slabs-with-avail-cells list

			if( (m_pFirstSlabWithAvail =
				pSlabToUnlink->pNextSlabWithAvail) == NULL)
			{
				f_assert( m_pLastSlabWithAvail == pSlabToUnlink);
				m_pLastSlabWithAvail = NULL;
			}
			else
			{
				pSlabToUnlink->pNextSlabWithAvail->pPrevSlabWithAvail =
					pSlabToUnlink->pPrevSlabWithAvail;
				pSlabToUnlink->pNextSlabWithAvail = NULL;
			}

			f_assert( !pSlabToUnlink->pPrevSlabWithAvail);
			f_assert( !pSlabToUnlink->pNextSlabWithAvail);

			// Decrement the slab count

			f_assert( m_uiSlabsWithAvail);
			m_uiSlabsWithAvail--;
		}
		
		pCell = (FLMBYTE *)pAvailBlock;
	}
	else
	{
		if( !m_pFirstSlab ||
			 (m_pFirstSlab->ui8NextNeverUsed == m_uiBlocksPerSlab))
		{
			SLABINFO *		pNewSlab;
			FLMUINT			uiBucket;
			
			if( RC_BAD( rc = getAnotherSlab( &pNewSlab)))
			{
				goto Exit;
			}
			
			f_assert( pNewSlab->pvSlab);

			// Link the slab into the global list
			
			if( m_pFirstSlab)
			{
				pNewSlab->pNextInGlobal = m_pFirstSlab;
				m_pFirstSlab->pPrevInGlobal = pNewSlab;
			}
			else
			{
				m_pLastSlab = pNewSlab;
			}

			m_pFirstSlab = pNewSlab;
			
			// Link the slab to its hash bucket
			
			uiBucket = getHashBucket( pNewSlab->pvSlab);
			if( (pNewSlab->pNextInBucket = m_pHashTable[ uiBucket]) != NULL)
			{
				m_pHashTable[ uiBucket]->pPrevInBucket = pNewSlab;
			}
			
			m_pHashTable[ uiBucket] = pNewSlab;
		}

		pSlab = m_pFirstSlab;
		pSlab->ui8AllocatedBlocks++;
		
		pCell = (((FLMBYTE *)pSlab->pvSlab) + 
									(pSlab->ui8NextNeverUsed * m_uiBlockSize));

		f_assert( !f_isBitSet( pSlab->ucAllocMap, pSlab->ui8NextNeverUsed));
		f_setBit( pSlab->ucAllocMap, pSlab->ui8NextNeverUsed);									
		f_assert( f_isBitSet( pSlab->ucAllocMap, pSlab->ui8NextNeverUsed));
		pSlab->ui8NextNeverUsed++;
	}

	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells++;
	}
	
	if( ppSlab)
	{
		*ppSlab = pSlab;
	}
	
	*ppvCell = pCell;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_BlockAlloc::freeCell(
	SLABINFO **		ppSlab,
	void **			ppvCell)
{
	SLABINFO *		pSlab = *ppSlab;
	FLMBYTE *		pCell = (FLMBYTE *)*ppvCell;
	AVAILBLOCK *	pAvailBlock = (AVAILBLOCK *)pCell;
	FLMUINT			uiBlockNum;
	
#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	// Make sure the cell and slab look sane

	if( !pSlab || !pCell || pCell < pSlab->pvSlab || 
		(pCell + m_uiBlockSize) > ((FLMBYTE *)pSlab->pvSlab + m_uiSlabSize))
	{
		f_assert( 0);
		goto Exit;
	}
	
	// Verify that the cell address is on a block boundary
	
#ifndef FLM_NLM
	f_assert( ((pCell - (FLMBYTE *)pSlab->pvSlab) % m_uiBlockSize) == 0);
#endif
	
	// Determine the block number

	uiBlockNum = (pCell - (FLMBYTE *)pSlab->pvSlab) / m_uiBlockSize;
	
	// Make sure the block is valie
	
	f_assert( uiBlockNum < m_uiBlocksPerSlab);
	f_assert( f_isBitSet( pSlab->ucAllocMap, uiBlockNum));
	
	// Clear the "allocated" bit
	
	f_clearBit( pSlab->ucAllocMap, uiBlockNum);
	
	// Should always be non-null on a free
	
	f_assert( m_pFirstSlab);
	
	// Add the cell to the slab's free list

	pAvailBlock->ui8NextAvail = pSlab->ui8FirstAvail;
	pSlab->ui8FirstAvail = (FLMUINT8)uiBlockNum;
	pSlab->ui8AvailBlocks++;

	f_assert( pSlab->ui8AllocatedBlocks);
	pSlab->ui8AllocatedBlocks--;

	// If there's no chain, make this one the first

	if( !m_pFirstSlabWithAvail)
	{
		f_assert( !pSlab->pNextSlabWithAvail);
		f_assert( !pSlab->pPrevSlabWithAvail);
		f_assert( !m_pLastSlabWithAvail);
		
		m_pFirstSlabWithAvail = pSlab;
		m_pLastSlabWithAvail = pSlab;
		m_uiSlabsWithAvail++;
		m_bAvailListSorted = TRUE;
	}
	else if( pSlab->ui8AvailBlocks == 1)
	{
		// This item is not linked in to the chain, so link it in

		if( m_bAvailListSorted && 
			 pSlab->pvSlab > m_pFirstSlabWithAvail->pvSlab)
		{
			m_bAvailListSorted = FALSE;
		}

		pSlab->pNextSlabWithAvail = m_pFirstSlabWithAvail;
		pSlab->pPrevSlabWithAvail = NULL;
		
		m_pFirstSlabWithAvail->pPrevSlabWithAvail = pSlab;
		m_pFirstSlabWithAvail = pSlab;
		
		m_uiSlabsWithAvail++;
	}

	// Adjust counter, because the block is now available

	m_uiTotalAvailBlocks++;

	// If this slab is now totally avail

	if( pSlab->ui8AvailBlocks == m_uiBlocksPerSlab)
	{
		f_assert( !pSlab->ui8AllocatedBlocks);

		// If we have met our threshold for being able to free a slab

		if( m_uiTotalAvailBlocks >= m_uiBlocksPerSlab)
		{
			freeSlab( &pSlab);
		}
		else if( pSlab != m_pFirstSlabWithAvail)
		{
			// Link the slab to the front of the avail list so that
			// it can be freed quickly at some point in the future
			
			f_assert( pSlab->pPrevSlabWithAvail);
			f_assert( pSlab->pNextSlabWithAvail || pSlab == m_pLastSlabWithAvail);

			pSlab->pPrevSlabWithAvail->pNextSlabWithAvail =
					pSlab->pNextSlabWithAvail;

			if( pSlab->pNextSlabWithAvail)
			{
				pSlab->pNextSlabWithAvail->pPrevSlabWithAvail =
					pSlab->pPrevSlabWithAvail;
			}
			else
			{
				f_assert( m_pLastSlabWithAvail == pSlab);
				m_pLastSlabWithAvail = pSlab->pPrevSlabWithAvail;
			}

			if( m_pFirstSlabWithAvail)
			{
				m_pFirstSlabWithAvail->pPrevSlabWithAvail = pSlab;
			}

			pSlab->pPrevSlabWithAvail = NULL;
			pSlab->pNextSlabWithAvail = m_pFirstSlabWithAvail;
			m_pFirstSlabWithAvail = pSlab;
		}
	}
	
	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells--;
	}

	*ppSlab = pSlab;
	*ppvCell = NULL;
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_BlockAlloc::getAnotherSlab(
	SLABINFO **		ppSlab)
{
	RCODE				rc = NE_FLM_OK;
	SLABINFO *		pSlab = NULL;
	
#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif
			
	if( (pSlab = (SLABINFO *)m_pInfoAllocator->allocCell( 
		NULL, NULL)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	f_memset( pSlab, 0, sizeof( SLABINFO));
	
	if( RC_BAD( rc = m_pSlabManager->allocSlab( &pSlab->pvSlab)))
	{
		m_pInfoAllocator->freeCell( pSlab);
		goto Exit;
	}

	if( m_pUsageStats)
	{
		m_pUsageStats->ui64Slabs++;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		(*m_puiTotalBytesAllocated) += m_uiSlabSize;
	}
	
	*ppSlab = pSlab;

Exit:
	
	return( rc);
}

/****************************************************************************
Desc:	Private internal method to free an unused empty slab back to the OS.
****************************************************************************/
void F_BlockAlloc::freeSlab(
	SLABINFO **			ppSlab)
{
	SLABINFO *			pSlab = *ppSlab;
	FLMUINT				uiLoop;

	f_assert( pSlab);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}

	if( pSlab->ui8AllocatedBlocks)
	{
		// Memory corruption detected!

		f_assert( 0);
		return;
	}
	
	// Make sure all "allocated" bits have been cleared
	
	for( uiLoop = 0; uiLoop < F_ALLOC_MAP_BYTES; uiLoop++)
	{
		if( pSlab->ucAllocMap[ uiLoop])
		{
			f_assert( 0);
			return;
		}
	}

	// Unlink from the global list

	if( pSlab->pNextInGlobal)
	{
		pSlab->pNextInGlobal->pPrevInGlobal = pSlab->pPrevInGlobal;
	}
	else
	{
		f_assert( m_pLastSlab == pSlab);
		m_pLastSlab = pSlab->pPrevInGlobal;
	}

	if( pSlab->pPrevInGlobal)
	{
		pSlab->pPrevInGlobal->pNextInGlobal = pSlab->pNextInGlobal;
	}
	else
	{
		f_assert( m_pFirstSlab == pSlab);
		m_pFirstSlab = pSlab->pNextInGlobal;
	}
	
	// Unlink from the hash bucket
	
	if( pSlab->pNextInBucket)
	{
		pSlab->pNextInBucket->pPrevInBucket = pSlab->pPrevInBucket;
	}

	if( pSlab->pPrevInBucket)
	{
		pSlab->pPrevInBucket->pNextInBucket = pSlab->pNextInBucket;
	}
	else
	{
		f_assert( m_pHashTable[ getHashBucket( pSlab->pvSlab)] == pSlab);
		m_pHashTable[ getHashBucket( pSlab->pvSlab)] = pSlab->pNextInBucket;
	}

	// Unlink from slabs-with-avail-cells list

	if( pSlab->pNextSlabWithAvail)
	{
		pSlab->pNextSlabWithAvail->pPrevSlabWithAvail =
			pSlab->pPrevSlabWithAvail;
	}
	else
	{
		f_assert( m_pLastSlabWithAvail == pSlab);
		m_pLastSlabWithAvail = pSlab->pPrevSlabWithAvail;
	}

	if( pSlab->pPrevSlabWithAvail)
	{
		pSlab->pPrevSlabWithAvail->pNextSlabWithAvail =
			pSlab->pNextSlabWithAvail;
	}
	else
	{
		f_assert( m_pFirstSlabWithAvail == pSlab);
		m_pFirstSlabWithAvail = pSlab->pNextSlabWithAvail;
	}

	f_assert( m_uiSlabsWithAvail);
	m_uiSlabsWithAvail--;
	
	f_assert( m_uiTotalAvailBlocks >= pSlab->ui8AvailBlocks);
	m_uiTotalAvailBlocks -= pSlab->ui8AvailBlocks;
	
	m_pSlabManager->freeSlab( (void **)&pSlab->pvSlab);
	m_pInfoAllocator->freeCell( pSlab);
	
	if( m_pUsageStats)
	{
		f_assert( m_pUsageStats->ui64Slabs);
		m_pUsageStats->ui64Slabs--;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		f_assert( (*m_puiTotalBytesAllocated) >= m_uiSlabSize);
		(*m_puiTotalBytesAllocated) -= m_uiSlabSize;
	}
	
	*ppSlab = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_BlockAlloc::freeAll( void)
{
	SLABINFO *		pNextSlab;
	SLABINFO *		pCurSlab;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	pCurSlab = m_pFirstSlab;

	while( pCurSlab)
	{
		pNextSlab = pCurSlab->pNextInGlobal;
		freeSlab( &pCurSlab);
		pCurSlab = pNextSlab;
	}

	f_assert( !m_uiTotalAvailBlocks);
	f_assert( !m_pFirstSlab);
	f_assert( !m_pLastSlab);
	f_assert( !m_pFirstSlabWithAvail);
	f_assert( !m_pLastSlabWithAvail);
	f_assert( !m_uiSlabsWithAvail);
	f_assert( !m_uiTotalAvailBlocks);

	m_bAvailListSorted = TRUE;
	
	if( m_pHashTable)
	{
		f_memset( m_pHashTable, 0, sizeof( SLABINFO *) * m_uiBuckets);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void FTKAPI F_BlockAlloc::freeUnused( void) // VISIT
{
	SLABINFO *		pSlab;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	if( (pSlab = m_pFirstSlabWithAvail) != NULL && !pSlab->ui8AllocatedBlocks)
	{
		freeSlab( &pSlab);
	}

	if( (pSlab = m_pFirstSlab) != NULL && !pSlab->ui8AllocatedBlocks)
	{
		freeSlab( &pSlab);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_BlockAlloc::defragmentMemory( void)
{
	RCODE				rc = NE_FLM_OK;
	SLABINFO *		pCurSlab;
	SLABINFO *		pPrevSib;
	FLMUINT			uiLoop;
	SLABINFO **		pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	SLABINFO *		smallSortBuf[ SMALL_SORT_BUF_SIZE];
	FLMBOOL			bMutexLocked = FALSE;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( m_uiTotalAvailBlocks < m_uiBlocksPerSlab)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiSlabsWithAvail;

	// Re-sort the slabs in the avail list according to
	// their memory addresses to help reduce logical fragmentation

	if( !m_bAvailListSorted && uiMaxSortEntries > 1)
	{
		if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
		{
			pSortBuf = smallSortBuf;
		}
		else
		{
			if( RC_BAD( rc = f_alloc( uiMaxSortEntries * sizeof( SLABINFO *),
				&pSortBuf)))
			{
				goto Exit;
			}
		}

		pCurSlab = m_pFirstSlabWithAvail;

		while( pCurSlab)
		{
			f_assert( uiSortEntries != uiMaxSortEntries);
			
			pSortBuf[ uiSortEntries++] = pCurSlab;
			pCurSlab = pCurSlab->pNextSlabWithAvail;
		}

		// Quick sort

		f_assert( uiSortEntries);

		f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
			slabInfoAddrCompareFunc, slabInfoAddrSwapFunc);

		// Re-link the items in the list according to the new 
		// sort order

		m_pFirstSlabWithAvail = NULL;
		m_pLastSlabWithAvail = NULL;

		pCurSlab = NULL;
		pPrevSib = NULL;

		for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
		{
			pCurSlab = pSortBuf[ uiLoop];
			
			pCurSlab->pNextSlabWithAvail = NULL;
			pCurSlab->pPrevSlabWithAvail = NULL;

			if( pPrevSib)
			{
				pCurSlab->pPrevSlabWithAvail = pPrevSib;
				pPrevSib->pNextSlabWithAvail = pCurSlab;
			}
			else
			{
				m_pFirstSlabWithAvail = pCurSlab;
			}

			pPrevSib = pCurSlab;
		}

		m_pLastSlabWithAvail = pCurSlab;
		m_bAvailListSorted = TRUE;
	}

	// Process the avail list (which should be sorted unless
	// we are too low on memory)

	pCurSlab = m_pLastSlabWithAvail;

	while( pCurSlab)
	{
		if( m_uiTotalAvailBlocks < m_uiBlocksPerSlab)
		{
			// No need to continue ... we aren't above the
			// free cell threshold

			goto Exit;
		}

		pPrevSib = pCurSlab->pPrevSlabWithAvail;

		if( pCurSlab == m_pFirstSlabWithAvail || !pCurSlab->ui8AvailBlocks)
		{
			// We've either hit the beginning of the avail list or
			// the slab that we are now positioned on has been
			// removed from the avail list.  In either case,
			// we are done.

			break;
		}

		if( pCurSlab->ui8AvailBlocks == m_uiBlocksPerSlab ||
			pCurSlab->ui8NextNeverUsed == pCurSlab->ui8AvailBlocks)
		{
			freeSlab( &pCurSlab);
			pCurSlab = pPrevSib;
			continue;
		}

		for( uiLoop = 0; uiLoop < pCurSlab->ui8NextNeverUsed &&
			pCurSlab != m_pFirstSlabWithAvail &&
			m_uiTotalAvailBlocks >= m_uiBlocksPerSlab; uiLoop++)
		{
			FLMBYTE *	pucBlock;
			
			pucBlock = (FLMBYTE *)(pCurSlab->pvSlab) + (uiLoop * m_uiBlockSize);
			
			if( f_isBitSet( pCurSlab->ucAllocMap, uiLoop))
			{
				FLMBYTE *	pucReloc = NULL;
				SLABINFO *	pRelocSlab;
				
				if( m_pRelocator->canRelocate( pucBlock))
				{
					if( RC_BAD( rc = getCell( &pRelocSlab, (void **)&pucReloc)))
					{
						goto Exit;
					}
					
					f_memcpy( pucReloc, pucBlock, m_uiBlockSize);
					m_pRelocator->relocate( pucBlock, pucReloc);

					freeCell( &pCurSlab, (void **)&pucBlock);
					
					if( !pCurSlab)
					{
						break;
					}
				}
			}
		}

		pCurSlab = pPrevSib;
	}
	
	// Defragment the slab info list
	
	m_pInfoAllocator->defragmentMemory();
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/
FLMBOOL F_SlabInfoRelocator::canRelocate(
	void *		pvAlloc)
{
	F_UNREFERENCED_PARM( pvAlloc);
	return( TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_SlabInfoRelocator::relocate(
	void *			pvOldAlloc,
	void *			pvNewAlloc)
{
	SLABINFO *		pOldSlabInfo = (SLABINFO *)pvOldAlloc;
	SLABINFO *		pNewSlabInfo = (SLABINFO *)pvNewAlloc;
	F_BlockAlloc *	pBlockAlloc = m_pBlockAlloc;
	
	// Fix the global links
	
	if( pOldSlabInfo->pPrevInGlobal)
	{
		f_assert( pOldSlabInfo != pBlockAlloc->m_pFirstSlab); 
		pOldSlabInfo->pPrevInGlobal->pNextInGlobal = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pBlockAlloc->m_pFirstSlab); 
		pBlockAlloc->m_pFirstSlab = pNewSlabInfo;
	}
	
	if( pOldSlabInfo->pNextInGlobal)
	{
		f_assert( pOldSlabInfo != pBlockAlloc->m_pLastSlab);
		pOldSlabInfo->pNextInGlobal->pPrevInGlobal = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pBlockAlloc->m_pLastSlab);
		pBlockAlloc->m_pLastSlab = pNewSlabInfo;
	}
	
	// Fix the hash links

	if( pOldSlabInfo->pPrevInBucket)
	{
		pOldSlabInfo->pPrevInBucket->pNextInBucket = pNewSlabInfo;
	}
	else
	{
		FLMUINT		uiBucket = pBlockAlloc->getHashBucket( pOldSlabInfo->pvSlab);
		
		f_assert( pBlockAlloc->m_pHashTable[ uiBucket] == pOldSlabInfo);
		pBlockAlloc->m_pHashTable[ uiBucket] = pNewSlabInfo; 
	}
	
	if( pOldSlabInfo->pNextInBucket)
	{
		pOldSlabInfo->pNextInBucket->pPrevInBucket = pNewSlabInfo;
	}
	
	// Fix the avail list links
		
	if( pOldSlabInfo->ui8AvailBlocks)
	{
		if( pOldSlabInfo->pPrevSlabWithAvail)
		{
			f_assert( pOldSlabInfo != pBlockAlloc->m_pFirstSlabWithAvail);
			pOldSlabInfo->pPrevSlabWithAvail->pNextSlabWithAvail = pNewSlabInfo;
		}
		else
		{
			f_assert( pOldSlabInfo == pBlockAlloc->m_pFirstSlabWithAvail);
			pBlockAlloc->m_pFirstSlabWithAvail = pNewSlabInfo;
		}
		
		if( pOldSlabInfo->pNextSlabWithAvail)
		{
			f_assert( pOldSlabInfo != pBlockAlloc->m_pLastSlabWithAvail);
			pOldSlabInfo->pNextSlabWithAvail->pPrevSlabWithAvail = pNewSlabInfo;
		}
		else
		{
			f_assert( pOldSlabInfo == pBlockAlloc->m_pLastSlabWithAvail);
			pBlockAlloc->m_pLastSlabWithAvail = pNewSlabInfo;
		}
	}
	else
	{
		f_assert( !pOldSlabInfo->pPrevSlabWithAvail);
		f_assert( !pOldSlabInfo->pNextSlabWithAvail);
	}
	
#ifdef FLM_DEBUG
	f_memset( pOldSlabInfo, 0, sizeof( SLABINFO));
#endif
}

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FTKAPI FlmAllocBlockAllocator(
	IF_BlockAlloc **			ppBlockAllocator)
{
	if( (*ppBlockAllocator = f_new F_BlockAlloc) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
