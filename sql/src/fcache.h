//------------------------------------------------------------------------------
// Desc:	Various classes used to manage cache.
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FCACHE_H
#define FCACHE_H

class F_Rfl;
class F_Db;
class F_DbSystem;
class F_Database;
class F_MultiAlloc;
class F_Row;
class F_CachedBlock;
class F_BlockCacheMgr;
class F_RowCacheMgr;
class F_GlobalCacheMgr;
class F_CacheList;
class F_CachedItem;
class F_Btree;
class F_BTreeIStream;
class F_BTreeInfo;
class F_RowRelocator;
class F_ColumnDataRelocator;
class F_ColumnListRelocator;
class F_BlockRelocator;

#define MIN_HASH_BUCKETS	0x10000			// 65536 buckets - multiple of 2.
#define MAX_HASH_BUCKETS 0x20000000			// roughly 500,000,000 buckets.

FLMUINT caGetBestHashTblSize(				// scache.cpp
	FLMUINT	uiCurrItemCount);
	
FINLINE FLMUINT minItemCount(
	FLMUINT	uiNumHashBuckets)
{
	return( uiNumHashBuckets / 4);
}
		
FINLINE FLMUINT maxItemCount(
	FLMUINT	uiNumHashBuckets)
{
	return( uiNumHashBuckets * 4);
}

FINLINE FLMBOOL shouldRehash(
	FLMUINT	uiItemCount,
	FLMUINT	uiNumBuckets)
{
	return( ((uiItemCount > maxItemCount( uiNumBuckets) &&
			    uiNumBuckets < MAX_HASH_BUCKETS) ||
			   (uiItemCount < minItemCount( uiNumBuckets) &&
			    uiNumBuckets > MIN_HASH_BUCKETS))
				? TRUE
				: FALSE);
}
		
/***************************************************************************
Desc:	See if enough time has passed since we last tried to rehash.  We
		don't want to be continually trying to rehash.
***************************************************************************/
FINLINE FLMBOOL checkHashFailTime(
	FLMUINT *	puiHashFailTime)
{
	if (*puiHashFailTime)
	{
		FLMUINT	uiCurrTime = FLM_GET_TIMER();
		
		if (FLM_ELAPSED_TIME( uiCurrTime, (*puiHashFailTime)) >=
					gv_SFlmSysData.uiRehashAfterFailureBackoffTime)
		{
			*puiHashFailTime = 0;
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}
	else
	{
		return( TRUE);
	}
}

/*****************************************************************************
Desc:	Cached item
******************************************************************************/
class F_CachedItem
{
public:

	F_CachedItem()
	{
		m_pNextInGlobal = NULL;
		m_pPrevInGlobal = NULL;
	}
	
	virtual ~F_CachedItem()
	{
	}

private:

	F_CachedItem *	m_pPrevInGlobal;
	F_CachedItem *	m_pNextInGlobal;

friend class F_CacheList;
friend class F_CachedBlock;
friend class F_Row;
friend class F_GlobalCacheMgr;
friend class F_BlockCacheMgr;
friend class F_RowCacheMgr;
friend class F_Database;
friend class F_Db;
friend class F_RowRelocator;
friend class F_ColumnDataRelocator;
friend class F_ColumnListRelocator;
friend class F_BlockRelocator;
};

/***************************************************************************
Desc:	Object for keeping track of an MRU/LRU list of cached items (rows
		or blocks)
***************************************************************************/
class F_CacheList
{
public:

	F_CacheList()
	{
		m_pMRUItem = NULL;
		m_pLRUItem = NULL;
		m_pLastMRUItem = NULL;
	}
	
	~F_CacheList()
	{
		flmAssert( !m_pMRUItem);
		flmAssert( !m_pLRUItem);
		flmAssert( !m_pLastMRUItem);
	}
	
	// Link a cached item into the global list as the MRU item. This routine
	// assumes that the cache mutex for managing this list
	// has already been locked.
	
	FINLINE void linkGlobalAsMRU(
		F_CachedItem *	pItem)
	{
		if ((pItem->m_pNextInGlobal = m_pMRUItem) != NULL)
		{
			pItem->m_pNextInGlobal->m_pPrevInGlobal = pItem;
		}
		else
		{
			m_pLRUItem = pItem;
			m_pLastMRUItem = pItem;
		}

		pItem->m_pPrevInGlobal = NULL;
		m_pMRUItem = pItem;
		flmAssert( pItem != pItem->m_pPrevInGlobal);
		flmAssert( pItem != pItem->m_pNextInGlobal);
	}
	
	// Link a cached item into the global list as the last MRU item. 
	// This routine assumes that the cache mutex for managing this list
	// has already been locked.
	
	FINLINE void linkGlobalAsLastMRU(
		F_CachedItem *	pItem)
	{
		if( !m_pLastMRUItem)
		{
			flmAssert( !m_pMRUItem);
			linkGlobalAsMRU( pItem);
			return;
		}
		
		flmAssert( m_pLastMRUItem);

		if( m_pLastMRUItem->m_pNextInGlobal)
		{
			m_pLastMRUItem->m_pNextInGlobal->m_pPrevInGlobal = pItem;
			pItem->m_pNextInGlobal = m_pLastMRUItem->m_pNextInGlobal;
		}
		else
		{
			flmAssert( m_pLRUItem == m_pLastMRUItem);
			m_pLRUItem = pItem;
		}

		m_pLastMRUItem->m_pNextInGlobal = pItem;

		pItem->m_pPrevInGlobal = m_pLastMRUItem;
		m_pLastMRUItem = pItem;

		flmAssert( pItem != pItem->m_pPrevInGlobal);
		flmAssert( pItem != pItem->m_pNextInGlobal);
	}
	
	// Link a cached item into the global list as the LRU item. This routine
	// assumes that the cache mutex for managing this list
	// has already been locked.
	
	FINLINE void linkGlobalAsLRU(
		F_CachedItem *	pItem)
	{
		if ((pItem->m_pPrevInGlobal = m_pLRUItem) != NULL)
		{
			pItem->m_pPrevInGlobal->m_pNextInGlobal = pItem;
		}
		else
		{
			flmAssert( !m_pMRUItem);
			flmAssert( !m_pLastMRUItem);
			
			m_pMRUItem = pItem;
			m_pLastMRUItem = pItem;
		}

		pItem->m_pNextInGlobal = NULL;
		m_pLRUItem = pItem;

		flmAssert( pItem != pItem->m_pPrevInGlobal);
		flmAssert( pItem != pItem->m_pNextInGlobal);
	}

	// Unlink a cached item from the global list. This routine
	// assumes that the cache mutex for managing this list
	// has already been locked.
	
	FINLINE void unlinkGlobal(
		F_CachedItem *	pItem)
	{
		if( pItem == m_pLastMRUItem)
		{
			if( m_pLastMRUItem->m_pPrevInGlobal)
			{
				m_pLastMRUItem = m_pLastMRUItem->m_pPrevInGlobal;
			}
			else
			{
				m_pLastMRUItem = m_pLastMRUItem->m_pNextInGlobal;
			}
		}

		if (pItem->m_pNextInGlobal)
		{
			flmAssert( pItem != m_pLRUItem);

			pItem->m_pNextInGlobal->m_pPrevInGlobal = pItem->m_pPrevInGlobal;
		}
		else
		{
			m_pLRUItem = pItem->m_pPrevInGlobal;
		}
	
		if (pItem->m_pPrevInGlobal)
		{
			flmAssert( pItem != m_pMRUItem);

			pItem->m_pPrevInGlobal->m_pNextInGlobal = pItem->m_pNextInGlobal;
		}
		else
		{
			m_pMRUItem = pItem->m_pNextInGlobal;
		}
		
		pItem->m_pNextInGlobal = NULL; 
		pItem->m_pPrevInGlobal = NULL;
	}

	// Moves a cached item one step closer to the MRU slot in the global list.
	// This routine assumes that the cache mutex for managing this list
	// has already been locked.
	
	FINLINE void stepUpInGlobal(
		F_CachedItem *	pItem)
	{
		F_CachedItem *	pPrevItem;
	
		if ((pPrevItem = pItem->m_pPrevInGlobal) != NULL)
		{
			if( pItem == m_pLastMRUItem)
			{
				m_pLastMRUItem = m_pLastMRUItem->m_pPrevInGlobal;
			}
		
			if (pPrevItem->m_pPrevInGlobal)
			{
				pPrevItem->m_pPrevInGlobal->m_pNextInGlobal = pItem;
			}
			else
			{
				m_pMRUItem = pItem;
			}
	
			pItem->m_pPrevInGlobal = pPrevItem->m_pPrevInGlobal;
			pPrevItem->m_pPrevInGlobal = pItem;
			pPrevItem->m_pNextInGlobal = pItem->m_pNextInGlobal;

			if (pItem->m_pNextInGlobal)
			{
				pItem->m_pNextInGlobal->m_pPrevInGlobal = pPrevItem;
			}
			else
			{
				m_pLRUItem = pPrevItem;
			}

			pItem->m_pNextInGlobal = pPrevItem;
		}
	}
	
private:

	F_CachedItem *		m_pMRUItem;
	F_CachedItem *		m_pLRUItem;
	F_CachedItem *		m_pLastMRUItem;
	
friend class F_CachedItem;
friend class F_RowCacheMgr;
friend class F_BlockCacheMgr;
friend class F_CachedBlock;
friend class F_RowRelocator;
friend class F_ColumnDataRelocator;
friend class F_ColumnListRelocator;
friend class F_BlockRelocator;
};
		
/***************************************************************************
Desc:	Global cache manager for FLAIM-SQL.
***************************************************************************/
class F_GlobalCacheMgr : public F_Object
{
public:
	F_GlobalCacheMgr();
	
	~F_GlobalCacheMgr();
	
	RCODE setup( void);
	
	FINLINE void incrTotalBytes(
		FLMUINT	uiIncrAmount)
	{
		m_pSlabManager->incrementTotalBytesAllocated( uiIncrAmount);
	}
	
	FINLINE void decrTotalBytes(
		FLMUINT	uiDecrAmount)
	{
		m_pSlabManager->decrementTotalBytesAllocated( uiDecrAmount);
	}
	
	FINLINE FLMUINT totalBytes( void)
	{
		return( m_pSlabManager->totalBytesAllocated());
	}
	
	FINLINE FLMUINT availSlabs( void)
	{
		return( m_pSlabManager->availSlabs());
	}

	FINLINE FLMUINT allocatedSlabs( void)
	{
		return( m_pSlabManager->getTotalSlabs());
	}

	FINLINE FLMBOOL cacheOverLimit( void)
	{
		if( allocatedSlabs() > m_uiMaxSlabs) 
		{
			return( TRUE);
		}
		
		return( FALSE);
	}
	
	RCODE setCacheLimit(
		FLMUINT		uiMaxCache,
		FLMBOOL		bPreallocateCache);
	
	RCODE setDynamicMemoryLimit(
		FLMUINT		uiCacheAdjustPercent,
		FLMUINT		uiCacheAdjustMin,
		FLMUINT		uiCacheAdjustMax,
		FLMUINT		uiCacheAdjustMinToLeave);
		
	RCODE setHardMemoryLimit(
		FLMUINT		uiPercent,
		FLMBOOL		bPercentOfAvail,
		FLMUINT		uiMin,
		FLMUINT		uiMax,
		FLMUINT		uiMinToLeave,
		FLMBOOL		bPreallocate);
		
	void getCacheInfo(
		SFLM_CACHE_INFO *	pMemInfo);
		
	RCODE adjustCache(
		FLMUINT *	puiCurrTime,
		FLMUINT *	puiLastCacheAdjustTime);

	RCODE clearCache(
		F_Db *		pDb);

	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
	
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}
	
private:

	IF_SlabManager *	m_pSlabManager;
	FLMUINT				m_uiMaxBytes;
	FLMUINT				m_uiMaxSlabs;
	FLMBOOL				m_bCachePreallocated;
	FLMBOOL				m_bDynamicCacheAdjust;
													// Is cache to be dynamically adjusted?
	FLMUINT				m_uiCacheAdjustPercent;
													// Percent of available memory to adjust to.
	FLMUINT				m_uiCacheAdjustMin;
													// Minimum limit to adjust cache to.
	FLMUINT				m_uiCacheAdjustMax;
													// Maximum limit to adjust cache to.
	FLMUINT				m_uiCacheAdjustMinToLeave;
													// Minimum bytes to leave when adjusting cache.
	FLMUINT				m_uiCacheAdjustInterval;
													// Interval for adjusting cache limit.
	FLMUINT				m_uiCacheCleanupInterval;
													// Interval for cleaning up old things out of
													// cache.
	FLMUINT				m_uiUnusedCleanupInterval;
													// Interval for cleaning up unused structures
	F_MUTEX				m_hMutex;			// Mutex to control access to global cache
													// manager object.
	
friend class F_CachedItem;
friend class F_Row;
friend class F_CachedBlock;
friend class F_BlockCacheMgr;
friend class F_RowCacheMgr;
friend class F_Database;
friend class F_DbSystem;
};

/****************************************************************************
Desc:	Class for moving cache blocks in cache.
****************************************************************************/
class F_BlockRelocator : public IF_Relocator
{
public:

	F_BlockRelocator()
	{
	}
	
	virtual ~F_BlockRelocator()
	{
	}

	void SQFAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL SQFAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	This class manages block cache.
****************************************************************************/
class F_BlockCacheMgr : public F_Object
{
public:
	F_BlockCacheMgr();
	
	~F_BlockCacheMgr();
	
	RCODE initCache( void);
	
	void cleanupLRUCache( void);
		
	void cleanupReplaceList( void);
		
	void cleanupFreeCache( void);
	
	void reduceReuseList( void);
	
	RCODE reduceCache(
		F_Db *				pDb);
		
	RCODE rehash( void);
	
	RCODE allocBlock(
		F_Db *				pDb,
		F_CachedBlock **	ppSCache);
		
	// Returns a pointer to the correct entry in the block cache hash table for
	// the given block address

	FINLINE F_CachedBlock ** blockHash(
		FLMUINT	uiSigBitsInBlkSize,
		FLMUINT	uiBlkAddress
		)
	{
		return( &m_ppHashBuckets[ (uiBlkAddress >> 
			uiSigBitsInBlkSize) & m_uiHashMask]);
	}

	FINLINE void defragmentMemory(
		FLMBOOL		bMutexLocked = FALSE)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		}

		m_pBlockAllocator->defragmentMemory();

		if( !bMutexLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		}
	}

private:

	RCODE initHashTbl( void);
	
	F_CacheList			m_MRUList;		// List of all block objects in MRU order.
	F_CachedBlock *	m_pMRUReplace;	// Pointer to the MRU end of the list
												// of cache items with no flags set.
	F_CachedBlock *	m_pLRUReplace;	// Pointer to the LRU end of the list
												// of cache items with no flags set.
	F_CachedBlock *	m_pFirstFree;
												// Pointer to a linked list of cache
												// blocks that need to be freed.
												// Cache blocks in this list are no
												// longer associated with a file and can
												// be freed or re-used as needed.  They
												// are linked using the pNextInFile and
												// pPrevInFile pointers.
	F_CachedBlock *	m_pLastFree;
												// Pointer to a linked list of cache
												// blocks that need to be freed.
												// Cache blocks in this list are no
												// longer associated with a file and can
												// be freed or re-used as needed.  They
												// are linked using the pNextInFile and
												// pPrevInFile pointers.
	SFLM_CACHE_USAGE	m_Usage;			// Contains usage information.
	FLMUINT				m_uiFreeBytes;	// Number of free bytes
	FLMUINT				m_uiFreeCount;	// Number of free blocks
	FLMUINT				m_uiReplaceableCount;
												// Number of blocks whose flags are 0
	FLMUINT				m_uiReplaceableBytes;
												// Number of bytes belonging to blocks whose
												// flags are 0
	FLMBOOL				m_bAutoCalcMaxDirty;
												// Flag indicating we should automatically
												// calculate maximum dirty cache.
	FLMUINT				m_uiMaxDirtyCache;
												// Maximum cache that can be dirty.
	FLMUINT				m_uiLowDirtyCache;
												// When maximum dirty cache is exceeded,
												// threshhold it should be brought back
												// under
	FLMUINT				m_uiTotalUses;	// Total number of uses currently held
												// on blocks in cache.
	FLMUINT				m_uiBlocksUsed;// Total number of blocks in cache that
												// are being used.
	FLMUINT				m_uiPendingReads;
												// Total reads currently pending.
	FLMUINT				m_uiIoWaits;	// Number of times multiple threads
												// were reading the same block from
												// disk at the same time.
	F_CachedBlock **	m_ppHashBuckets;// This is a pointer to a hash table that
												// is used to find cache blocks.  Each
												// element in the table points to a
												// linked list of F_CachedBlock objects that
												// all hash to the same hash bucket.
	FLMUINT				m_uiNumBuckets;// This contains the number of buckets
												// in the hash table.
	FLMUINT				m_uiHashFailTime;
												// Last time we tried to rehash and
												// failed.  Want to wait before we
												// retry again.
	FLMUINT				m_uiHashMask;	// Bits that are significant
												// for the number of hash buckets.
	IF_MultiAlloc *	m_pBlockAllocator;	
												// Fixed size allocators for cache blocks
	F_BlockRelocator	m_blockRelocator;
												// Relocator for cache blocks
	FLMBOOL				m_bReduceInProgress;
#ifdef FLM_DEBUG
	FLMBOOL				m_bDebug;		// Enables checksumming and cache use
												// monitoring.  Only available when
												// debug is compiled in.
#endif

friend class F_CachedBlock;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_Db;
friend class F_DbSystem;
friend class F_BlockRelocator;
};

#ifdef FLM_DEBUG
/****************************************************************************
Struct:	SCACHE_USE	(Cache Block Use)
Desc:	 	This is a debug only structure that is used to keep track of the
			threads that are currently using a block.
****************************************************************************/
typedef struct SCache_Use
{
	SCache_Use *	pNext;			// Pointer to next SCACHE_USE structure in
											// the list.
	FLMUINT			uiThreadId;		// Thread ID of thread using the block.
	FLMUINT			uiUseCount;		// Use count for this particular thread.
} SCACHE_USE;
#endif

// Flags for m_ui16Flags field in F_CachedBlock

#define CA_DIRTY						0x0001
														// This bit indicates that the block is
														// dirty and needs to be flushed to disk.
														// NOTE: For 3.x files, this bit may remain
														// set on prior versions of blocks until the
														// current transaction commits.
#define CA_WRITE_INHIBIT			0x0002
														// Must not write block until use count
														// goes to zero.  NOTE: Can ignore when
														// in the checkpoint thread.
#define CA_READ_PENDING				0x0004
														// This bit indicates that the block is
														// currently being read in from disk.
#define CA_WRITE_TO_LOG				0x0008
														// This bit indicates that this version of
														// the block should be written to the
														// rollback log before being replaced.
														// During an update transaction, the first
														// time a block is updated, FLAIM will
														// create a new version of the block and
														// insert it into cache.  The prior version
														// of the block is marked with this flag
														// to indicate that it needs to be written
														// to the log before it can be replaced.
#define CA_LOG_FOR_CP				0x0010
														// This bit indicates that this version of
														// the block needs to be logged to the
														// physical rollback in order to restore
														// the last checkpoint.  This is only
														// applicable to 3.x files.
#define CA_WAS_DIRTY					0x0020
														// This bit indicates that this version of
														// the block was dirty before the newer
														// version of the block was created.
														// Its dirty state should be restored if
														// the current transaction aborts.  This
														// flag is only used for 3.x files.
#define CA_WRITE_PENDING			0x0040
														// This bit indicates that a block is in
														// the process of being written out to
														// disk.
#define CA_IN_WRITE_PENDING_LIST	0x0080
														// This bit indicates that a block is in
														// the write pending list.
#define CA_FREE						0x0100
														// The block has been linked to the free
														// list (and unlinked from all other lists)
#define CA_IN_FILE_LOG_LIST		0x0200
														// Block is in the list of blocks that may
														//	have one or more versions that need to
														// be logged
#define CA_IN_NEW_LIST				0x0400
														// Dirty block that is beyond the last CP EOF
#define CA_DUMMY_FLAG				0x0800
														// Used to prevent blocks from being linked
														// into the replace list in cases where
														// they will be removed immediately (because
														// a bit is going to being set)
														
/****************************************************************************
Desc:	This is the header structure for a cached data block.
****************************************************************************/
class F_CachedBlock : public F_CachedItem
{
public:
	F_CachedBlock(
		FLMUINT	uiBlockSize);
	
	~F_CachedBlock();
	
	FINLINE FLMUINT memSize( void)
	{
		return( gv_SFlmSysData.pBlockCacheMgr->m_pBlockAllocator->getTrueSize( 
				(FLMBYTE *)this));
	}
	
	FINLINE FLMUINT blkAddress( void)
	{
		return( m_uiBlkAddress);
	}
	
	FINLINE F_BLK_HDR * getBlockPtr( void)
	{
		return( m_pBlkHdr);
	}
	
	FINLINE F_Database * getDatabase( void)
	{
		return( m_pDatabase);
	}
	
	FINLINE FLMUINT16 getModeFlags( void)
	{
		return( m_ui16Flags);
	}
	
	FINLINE FLMUINT getUseCount( void)
	{
		return( m_uiUseCount);
	}
	
	// Gets the prior image block address from the block header.
	// NOTE: This function assumes that the block cache mutex is locked.
	
	FINLINE FLMUINT getPriorImageAddress( void)
	{
		return( (FLMUINT)m_pBlkHdr->ui32PriorBlkImgAddr);
	}

	// Gets the transaction ID from the block header.  NOTE: This function
	// assumes that the block cache mutex is locked.
	
	FINLINE FLMUINT64 getLowTransID( void)
	{
		return( m_pBlkHdr->ui64TransID);
	}

	//	Set the high transaction ID for a cache block.
	// NOTE: This function assumes that the block cache mutex is locked.
	
	FINLINE void setTransID(
		FLMUINT64	ui64NewTransID)
	{
		if (m_ui64HighTransID == FLM_MAX_UINT64 && ui64NewTransID != FLM_MAX_UINT64)
		{
			gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes += memSize();
			gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount++;
		}
		else if (m_ui64HighTransID != FLM_MAX_UINT64 && ui64NewTransID == FLM_MAX_UINT64)
		{
			FLMUINT	uiSize = memSize();
	
			flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
			gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
			flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount);
			gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount--;
		}

		m_ui64HighTransID = ui64NewTransID;
	}
	
	// Determines if a cache block is needed by a read transaction.
	FINLINE FLMBOOL neededByReadTrans( void)
	{
		return( m_pDatabase->neededByReadTrans( getLowTransID(),
						m_ui64HighTransID));
	}

	// Link a cache block into the replace list as the MRU item. This routine
	// assumes that the block cache mutex has already been locked.
	FINLINE void linkToReplaceListAsMRU( void)
	{
		flmAssert( !m_ui16Flags);
	
		if ((m_pNextInReplaceList = 
			gv_SFlmSysData.pBlockCacheMgr->m_pMRUReplace) != NULL)
		{
			m_pNextInReplaceList->m_pPrevInReplaceList = this;
		}
		else
		{
			gv_SFlmSysData.pBlockCacheMgr->m_pLRUReplace = this;
		}
		
		m_pPrevInReplaceList = NULL;
		gv_SFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
		gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableCount++;
		gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes += memSize();
	}

	// Link a cache block into the replace list as the LRU item. This routine
	// assumes that the block cache mutex has already been locked.
	FINLINE void linkToReplaceListAsLRU( void)
	{
		flmAssert( !m_ui16Flags);
	
		if ((m_pPrevInReplaceList = gv_SFlmSysData.pBlockCacheMgr->m_pLRUReplace) != NULL)
		{
			m_pPrevInReplaceList->m_pNextInReplaceList = this;
		}
		else
		{
			gv_SFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
		}

		m_pNextInReplaceList = NULL;
		gv_SFlmSysData.pBlockCacheMgr->m_pLRUReplace = this;
		gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableCount++;
		gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes += memSize();
	}

	// Moves a block one step closer to the MRU slot in the replace list.
	// This routine assumes that the block cache mutex has already been locked.
	FINLINE void stepUpInReplaceList( void)
	{
		F_CachedBlock *		pPrevSCache;
	
		flmAssert( !m_ui16Flags);
	
		if( (pPrevSCache = m_pPrevInReplaceList) != NULL)
		{
			if( pPrevSCache->m_pPrevInReplaceList)
			{
				pPrevSCache->m_pPrevInReplaceList->m_pNextInReplaceList = this;
			}
			else
			{
				gv_SFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
			}
	
			m_pPrevInReplaceList = pPrevSCache->m_pPrevInReplaceList;

			pPrevSCache->m_pPrevInReplaceList = this;
			pPrevSCache->m_pNextInReplaceList = m_pNextInReplaceList;
			
			if( m_pNextInReplaceList)
			{
				m_pNextInReplaceList->m_pPrevInReplaceList = pPrevSCache;
			}
			else
			{
				gv_SFlmSysData.pBlockCacheMgr->m_pLRUReplace = pPrevSCache;
			}

			m_pNextInReplaceList = pPrevSCache;
		}
	}

	// Clears the passed-in flags from the F_CachedBlock object
	// This routine assumes that the block cache mutex is locked.
	FINLINE void clearFlags(
		FLMUINT16	ui16FlagsToClear)
	{
		if( m_ui16Flags)
		{
			if( (m_ui16Flags &= ~ui16FlagsToClear) == 0)
			{
				if( !m_pPrevInGlobal ||
					m_ui64HighTransID == ~((FLMUINT64)0) ||
					neededByReadTrans())
				{
					linkToReplaceListAsMRU();
				}
				else
				{
					linkToReplaceListAsLRU();
				}
			}
		}
	}

	// Sets the passed-in flags on the object
	// This routine assumes that the block cache mutex is locked.
	
	FINLINE void setFlags(
		FLMUINT16	ui16FlagsToSet)
	{
		flmAssert( ui16FlagsToSet);
	
		if( !m_ui16Flags)
		{
			unlinkFromReplaceList();
		}

		m_ui16Flags |= ui16FlagsToSet;
	}

	// Set the dirty flag on a cache block.
	// This routine assumes that the block cache mutex is locked.
	
	FINLINE void setDirtyFlag(
		F_Database *	pDatabase)
	{
		flmAssert( !(m_ui16Flags &
			(CA_DIRTY | CA_WRITE_PENDING | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
		setFlags( CA_DIRTY);
		pDatabase->incrementDirtyCacheCount();
	}
	
	// Unset the dirty flag on a cache block.
	// This routine assumes that the block cache mutex is locked.
	FINLINE void unsetDirtyFlag( void)
	{
		flmAssert( m_ui16Flags & CA_DIRTY);
		flmAssert( m_pDatabase->getDirtyCacheCount());
	
		if (m_ui16Flags & CA_IN_FILE_LOG_LIST)
		{
			unlinkFromLogList();
		}
		else if (m_ui16Flags & CA_IN_NEW_LIST)
		{
			unlinkFromNewList();
		}
	
		clearFlags( CA_DIRTY);
		m_pDatabase->decrementDirtyCacheCount();
	}

	FINLINE FLMUINT getBlkSize( void)
	{
		return( (FLMUINT)m_ui16BlkSize);
	}

#ifdef FLM_DBG_LOG
	void logFlgChange(
		FLMUINT16	ui16OldFlags,
		char			cPlace);
#endif
		
	void linkToLogList( void);
	
	void unlinkFromLogList( void);
	
	void linkToNewList( void);
	
	void unlinkFromNewList( void);

	void linkToDatabase(
		F_Database *	pDatabase);
		
	void unlinkFromDatabase( void);
	
	void unlinkFromTransLogList( void);
	
	// Increment the use count on a cache block for a particular
	// thread.  NOTE: This routine assumes that the block cache mutex
	// is locked.
	FINLINE void useForThread(
	#ifdef FLM_DEBUG
		FLMUINT		uiThreadId)
	#else
		FLMUINT)		// uiThreadId)
	#endif
	{

	#ifdef FLM_DEBUG
		if (m_pUseList ||
			 (gv_SFlmSysData.pBlockCacheMgr->m_bDebug && !m_uiUseCount))
		{
			dbgUseForThread( uiThreadId);
		}
		else
	#endif
		{
			if (!m_uiUseCount)
			{
				gv_SFlmSysData.pBlockCacheMgr->m_uiBlocksUsed++;
			}
			m_uiUseCount++;
			gv_SFlmSysData.pBlockCacheMgr->m_uiTotalUses++;
		}
	}
	
	// Decrement the use count on a cache block for a particular
	// thread.  NOTE: This routine assumes that the block cache mutex
	// is locked.
	FINLINE void releaseForThread( void)
	{
		if (!m_uiUseCount)
		{
			return;
		}
	
	#ifdef FLM_DEBUG
		if (m_pUseList)
		{
			dbgReleaseForThread();
		}
		else
	#endif
		{
	
	#ifdef FLM_DEBUG
	
			// If count is one, it will be decremented to zero.
	
			if (m_uiUseCount == 1)
			{
				m_uiChecksum = computeChecksum();
			}
	#endif
	
			m_uiUseCount--;
			gv_SFlmSysData.pBlockCacheMgr->m_uiTotalUses--;
			if (!m_uiUseCount)
			{
				gv_SFlmSysData.pBlockCacheMgr->m_uiBlocksUsed--;
			}
		}
	}

	// Tests if a block can be freed from cache.
	// NOTE: This routine assumes that the block cache mutex is locked.
	FINLINE FLMBOOL canBeFreed( void)
	{
		if (!m_uiUseCount && !m_ui16Flags)
		{
			F_CachedBlock *	pNewerSCache = m_pPrevInVersionList;
	
			// The following code is attempting to ensure that newer
			// versions of the block have had the prior block address
			// properly transferred to them from an older version of
			// the block.  If not, we cannot remove the current version
			// of the block (pointed to by pSCache), because it is
			// the older version that needs to be logged in order for
			// the prior block address to be properly transferred to
			// the newer version of the block.
	
			// If there is no newer version of the block, we can remove
			// this block, because it means that there was at one point
			// in time a newer version, the prior block address was
			// safely transferred - otherwise, the newer version would
			// still be in cache.
	
			// If there is a newer version of the block, but it is in the
			// process of being read in from disk (CA_READ_PENDING bit is
			// set), we can know that the prior block address has been
			// properly transferred to the block being read in.
			// Explanation: If the CA_READ_PENDING bit it set, the block
			// had to have been written out to disk at some prior time.  The
			// rules for writing out a block to disk are such that it is
			// impossible for a block to be written out without having a
			// pointer to some prior version of the block.  The only
			// exception to this is a newly created block - but in that
			// case, the block does not need to have a prior version pointer
			// - because there are none!
			// This assertion is obvious for a version of a block that is
			// being read from the rollback log - it would be impossible
			// to be reading such a block from the rollback log if it hadn't
			// been part of a version chain!  As for the current version of a
			// block, it cannot be written out and removed from cache without
			// having a pointer to the chain of older versions that may still
			// be needed (by a read transactions, for rollback, or to recover
			// a checkpoint).
	
			// NOTE: Although we know that a block being read in from disk
			// has to already have a prior block address, we cannot just
			// look at the block header, because it is being read in from
			// disk and the prior block address is not yet there.  Usually,
			// it will still be zeroes - making it look as though the block
			// does not have a prior block address when, in fact, it does.
			// Thus, we look at the CA_READ_PENDING bit first.  If that
			// is not set, we can safely look at the prior block address.
	
			// Note also that even if there is a newer block that doesn't
			// have a prior block address, we may still be able to remove
			// the current block (pSCache) if it is not needed by any
			// read transactions.
	
			if (!pNewerSCache ||
				 (pNewerSCache->m_ui16Flags & CA_READ_PENDING) ||
				 pNewerSCache->getPriorImageAddress() != 0 ||
				 !m_pDatabase ||
				 !neededByReadTrans())
			{
				return( TRUE);
			}
		}
		return( FALSE);
	}
	
	void linkToFreeList(
		FLMUINT		uiFreeTime);
		
	void unlinkFromFreeList( void);

	void * operator new(
		FLMSIZET			uiSize,
		FLMUINT			uiBlockSize)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new(
		FLMSIZET			uiSize)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new(
		FLMSIZET			uiSize,
		const char *	pszFile,
		int				iLine)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new[](
		FLMSIZET			uiSize)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new[](
		FLMSIZET			uiSize,
		const char *	pszFile,
		int				iLine)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void operator delete(
		void *			ptr);

#if !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete(
		void *			ptr,
		FLMSIZET			uiSize,
		const char *	pszFileName,
		int				iLine);
#endif

	void operator delete[](
		void *			ptr);

#if !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete(
		void *			ptr,
		FLMUINT			uiBlockSize,
		FLMBOOL			bAllocMutexLocked);
#endif


#if !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete(
		void *			ptr,
		const char *	file,
		int				line);
#endif

#if !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete[](
		void *			ptr,
		const char *	pszFileName,
		int				iLineNum);
#endif
	
private:

	void unlinkFromReplaceList( void);

#ifdef FLM_DEBUG
	void dbgUseForThread(
		FLMUINT	uiThreadId);
		
	void dbgReleaseForThread( void);
	
	FLMUINT computeChecksum( void);
#endif
		
#ifdef SCACHE_LINK_CHECKING
	void verifyCache(
		int	iPlace);
#else
	FINLINE void verifyCache(
		int)
	{
	}
#endif
	
	static void SQFAPI objectAllocInit(
		void *		pvAlloc,
		FLMUINT		uiSize);
		
	// Link a cached block into the global list as the MRU item. This routine
	// assumes that the block cache mutex has already been locked.
	
	FINLINE void linkToGlobalListAsMRU( void)
	{
		if( (m_pBlkHdr->ui8BlkType & BT_FREE) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF_DATA) ||
			 (m_pBlkHdr->ui8BlkType & BT_DATA_ONLY))
		{
			gv_SFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLastMRU(
							(F_CachedItem *)this);
		}
		else
		{
			gv_SFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsMRU(
							(F_CachedItem *)this);
		}
		
		if( !m_ui16Flags)
		{
			linkToReplaceListAsMRU();
		}
	}
	
	// Link a cached block into the global list as the LRU item. This routine
	// assumes that the block cache mutex has already been locked.
	
	FINLINE void linkToGlobalListAsLRU( void)
	{
		if( (m_pBlkHdr->ui8BlkType & BT_FREE) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF_DATA) ||
			 (m_pBlkHdr->ui8BlkType & BT_DATA_ONLY))
		{
			gv_SFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLRU(
						(F_CachedItem *)this);
		}
		else
		{
			gv_SFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLastMRU(
						(F_CachedItem *)this);
		}
					
		if( !m_ui16Flags)
		{
			linkToReplaceListAsLRU();
		}
	}
	
	// Unlink a cache block from the global list. This routine
	// assumes that the block cache mutex has already been locked.
	
	FINLINE void unlinkFromGlobalList( void)
	{
		gv_SFlmSysData.pBlockCacheMgr->m_MRUList.unlinkGlobal(
				(F_CachedItem *)this);
		if( !m_ui16Flags)
		{
			unlinkFromReplaceList();
		}
	}

	// Moves a block one step closer to the MRU slot in the global list.  This
	// routine assumes that the block cache mutex has already been locked.

	FINLINE void stepUpInGlobalList( void)
	{
		gv_SFlmSysData.pBlockCacheMgr->m_MRUList.stepUpInGlobal(
				(F_CachedItem *)this);
		if( !m_ui16Flags)
		{
			stepUpInReplaceList();
		}
	}

	#ifdef SCACHE_LINK_CHECKING
	void checkHashLinks(
		F_CachedBlock **	ppSCacheBucket);
		
	void checkHashUnlinks(
		F_CachedBlock **	ppSCacheBucket);
	#endif
	
	// Link a cache block to its hash bucket.  This routine assumes
	// that the block cache mutex has already been locked.
	FINLINE void linkToHashBucket(
		F_CachedBlock **	ppSCacheBucket)
	{
	#ifdef SCACHE_LINK_CHECKING
		checkHashLinks( ppSCacheBucket);
	#endif
		m_pPrevInHashBucket = NULL;
		if ((m_pNextInHashBucket = *ppSCacheBucket) != NULL)
		{
			m_pNextInHashBucket->m_pPrevInHashBucket = this;
		}
		
		*ppSCacheBucket = this;
	}
	
	// Unlink a cache block from its hash bucket.  This routine assumes
	// that the block cache mutex has already been locked.
	FINLINE void unlinkFromHashBucket(
		F_CachedBlock **	ppSCacheBucket)
	{
	#ifdef SCACHE_LINK_CHECKING
		checkHashUnlinks( ppSCacheBucket);
	#endif
	
		// Make sure it is not in the list of log blocks.
	
		flmAssert( !(m_ui16Flags & CA_WRITE_TO_LOG));

		if (m_pNextInHashBucket)
		{
			m_pNextInHashBucket->m_pPrevInHashBucket = m_pPrevInHashBucket;
		}
	
		if (m_pPrevInHashBucket)
		{
			m_pPrevInHashBucket->m_pNextInHashBucket = m_pNextInHashBucket;
		}
		else
		{
			*ppSCacheBucket = m_pNextInHashBucket;
		}
	
		m_pNextInHashBucket = NULL;
		m_pPrevInHashBucket = NULL;
	}

	void unlinkCache(
		FLMBOOL				bFreeIt,
		RCODE					NotifyRc);
		
	void savePrevBlkAddress( void);
		
	F_CachedBlock *	m_pPrevInDatabase;	// This is a pointer to the previous block
														// in the linked list of blocks that are
														// in the same database.
	F_CachedBlock *	m_pNextInDatabase;	// This is a pointer to the next block in
														// the linked list of blocks that are in
														// the same database.
	F_BLK_HDR *			m_pBlkHdr;				// Pointer to this block's header and data.
	F_Database *		m_pDatabase;			// Pointer to the database this data block
														// belongs to.
	FLMUINT				m_uiBlkAddress;		// Block address.
	F_CachedBlock *	m_pNextInReplaceList;// This is a pointer to the next block in
														// the global linked list of cache blocks
														// that have a flags value of zero.
	F_CachedBlock *	m_pPrevInReplaceList;// This is a pointer to the previous block in
														// the global linked list of cache blocks
														// that have a flags value of zero.
	F_CachedBlock *	m_pPrevInHashBucket;	// This is a pointer to the previous block
														// in the linked list of blocks that are
														// in the same hash bucket.
	F_CachedBlock *	m_pNextInHashBucket;	// This is a pointer to the next block in
														// the linked list of blocks that are in
														// the same hash bucket.
	F_CachedBlock *	m_pPrevInVersionList;// This is a pointer to the previous block
														// in the linked list of blocks that are
														// all just different versions of the
														// same block.  The previous block is a
														// more recent version of the block.
	F_CachedBlock *	m_pNextInVersionList;// This is a pointer to the next block in
														// the linked list of blocks that are all
														// just different versions of the same
														// block.  The next block is an older
														// version of the block.
	FNOTIFY *			m_pNotifyList;			// This is a pointer to a list of threads
														// that want to be notified when a pending
														// I/O is complete.  This pointer is only
														// non-null if the block is currently being
														// read from disk and there are multiple
														// threads all waiting for the block to
														// be read in.
	FLMUINT64			m_ui64HighTransID;	// This indicates the highest known moment
														// in the file's update history when this
														// version of the block was the active
														// block.
														// A block's low transaction ID and high
														// transaction ID indicate a span of
														// transactions where this version of the
														// block was the active version of the
														// block.
	FLMUINT				m_uiUseCount;			// Number of times this block has been
														// retrieved for use by threads.  A use
														// count of zero indicates that no thread
														// is currently using the block.  Note that
														// a block cannot be replaced when its use
														// count is non-zero.
	FLMUINT16			m_ui16Flags;			// This is a set of flags for the block
														// that indicate various things about the
														// block's current state.
	FLMUINT16			m_ui16BlkSize;			// Block size
	FLMBOOL				m_bCanRelocate;		// Can the block be moved in memory

// NOTE: Keep debug items at the END of the structure.

#ifdef FLM_DEBUG
	FLMUINT				m_uiChecksum;		// Checksum for the block and header.
	SCACHE_USE *		m_pUseList;			// This is a pointer to a list of threads
													// that are currently using the block.
#endif
friend class F_BlockCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_DbSystem;
friend class F_Db;
friend class F_Btree;
friend class F_BTreeInfo;
friend class F_BlockRelocator;
};

/****************************************************************************
Desc:	Class for moving rows in cache.
****************************************************************************/
class F_RowRelocator : public IF_Relocator
{
public:

	F_RowRelocator()
	{
	}
	
	virtual ~F_RowRelocator()
	{
	}

	void SQFAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL SQFAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving row data buffers in cache.
****************************************************************************/
class F_ColumnDataRelocator : public IF_Relocator
{
public:

	F_ColumnDataRelocator()
	{
	}
	
	virtual ~F_ColumnDataRelocator()
	{
	}

	void SQFAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL SQFAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving row column lists in cache.
****************************************************************************/
class F_ColumnListRelocator : public IF_Relocator
{
public:

	F_ColumnListRelocator()
	{
	}
	
	virtual ~F_ColumnListRelocator()
	{
	}

	void SQFAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL SQFAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	This class is used to control the row cache.
****************************************************************************/
class F_RowCacheMgr : public F_Object
{
public:
	F_RowCacheMgr();
	
	~F_RowCacheMgr();
	
	RCODE initCache( void);
		
	void cleanupOldCache( void);
	
	void cleanupPurgedCache( void);

	void reduceCache( void);
	
	FINLINE void setDebugMode(
		FLMBOOL	bDebug)
	{
#ifdef FLM_DEBUG
		m_bDebug = bDebug;
#else
		(void)bDebug;
#endif
	}
	
	RCODE allocRow(
		F_Row **		ppRow,
		FLMBOOL				bMutexLocked);
		
	RCODE retrieveRow(
		F_Db *		pDb,
		FLMUINT		uiTableNum,
		FLMUINT64	ui64RowId,
		F_Row **		ppRow);
		
	RCODE createRow(
		F_Db *	pDb,
		FLMUINT	uiTableNum,
		F_Row **	ppRow);
		
	RCODE makeWriteCopy(
		F_Db *	pDb,
		F_Row **	ppRow);

	void removeRow(
		F_Db *	pDb,
		F_Row *	pRow,
		FLMBOOL	bDecrementUseCount,
		FLMBOOL	bMutexLocked = FALSE);
		
	void removeRow(
		F_Db *			pDb,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64RowId);

	FINLINE void defragmentMemory(
		FLMBOOL		bMutexLocked = FALSE)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		}

		m_pRowAllocator->defragmentMemory();
		m_pBufAllocator->defragmentMemory();

		if( !bMutexLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
		}
	}

private:

	// Hash function for hashing to rows in row cache.
	// Assumes that the row cache mutex has already been locked.
	FINLINE F_Row ** rowHash(
		FLMUINT64	ui64RowId)
	{
		return( &m_ppHashBuckets[(FLMUINT)ui64RowId & m_uiHashMask]);
	}

	RCODE rehash( void);
	
	RCODE waitNotify(
		F_Db *			pDb,
		F_Row **	ppRow);
		
	void notifyWaiters(
		FNOTIFY *		pNotify,
		F_Row *	pUseRow,
		RCODE				NotifyRc);
		
	void linkIntoRowCache(
		F_Row *	pNewerRow,
		F_Row *	pOlderRow,
		F_Row *	pRow,
		FLMBOOL			bLinkAsMRU);
		
	void findRow(
		F_Db *			pDb,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64RowId,
		FLMUINT64		ui64VersionNeeded,
		FLMBOOL			bDontPoisonCache,
		FLMUINT *		puiNumLooks,
		F_Row **	ppRow,
		F_Row **	ppNewerRow,
		F_Row **	ppOlderRow);
		
	RCODE readRowFromDisk(
		F_Db *				pDb,
		FLMUINT				uiTableNum,
		FLMUINT64			ui64RowId,
		F_Row *				pRow,
		FLMUINT64 *			pui64LowTransId,
		FLMBOOL *			pbMostCurrent);
		
	RCODE _makeWriteCopy(
		F_Db *			pDb,
		F_Row **			ppRow);
		
	// Private Data
	
	F_CacheList			m_MRUList;			// List of all row objects in MRU order.
	F_Row *				m_pPurgeList;		// List of F_Row objects that
													// should be deleted when the use count
													// goes to zero.
	F_Row *				m_pHeapList;		// List of rows with heap allocations
	F_Row *				m_pOldList;			// List of old versions
	SFLM_CACHE_USAGE	m_Usage;				// Contains maximum, bytes used, etc.
	F_Row **				m_ppHashBuckets;	// Array of hash buckets.
	FLMUINT				m_uiNumBuckets;	// Total number of hash buckets.
													// must be an exponent of 2.
	FLMUINT				m_uiHashFailTime;
													// Last time we tried to rehash and
													// failed.  Want to wait before we
													// retry again.
	FLMUINT				m_uiHashMask;		// Hash mask mask for hashing a
													// row id to a hash bucket.
	FLMUINT				m_uiPendingReads;	// Total reads currently pending.
	FLMUINT				m_uiIoWaits;		// Number of times multiple threads
													// were reading the same row from
													// disk at the same time.
	IF_FixedAlloc *	m_pRowAllocator;	// Fixed size allocator for F_Row
													// objects
	IF_BufferAlloc *	m_pBufAllocator;	// Buffer allocator for buffers in
													// F_Row objects
													
	F_RowRelocator				m_rowRelocator;
	F_ColumnDataRelocator	m_columnDataRelocator;
	F_ColumnListRelocator	m_columnListRelocator;
	FLMBOOL						m_bReduceInProgress;

#ifdef FLM_DEBUG
	FLMBOOL				m_bDebug;			// Debug mode?
#endif
friend class F_Row;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_DbSystem;
friend class F_RowRelocator;
friend class F_ColumnDataRelocator;
friend class F_ColumnListRelocator;
};

// Flags kept in F_Row::m_uiCacheFlags.  Mutex needs to be locked to
// access these.

#define NCA_READING_IN					0x80000000
#define NCA_UNCOMMITTED					0x40000000
#define NCA_LATEST_VER					0x20000000
#define NCA_PURGED						0x10000000
#define NCA_LINKED_TO_DATABASE		0x08000000

#define NCA_COUNTER_BITS		(~(NCA_READING_IN | NCA_UNCOMMITTED | \
											NCA_LATEST_VER | NCA_PURGED | \
											NCA_LINKED_TO_DATABASE))

// Flags kept in F_Row::m_uiFlags.  Mutex does not have to be locked to
// access these.

#define FROW_DIRTY						0x00000010
#define FROW_NEW							0x00000020
#define FROW_HEAP_ALLOC					0x00000040

/*****************************************************************************
Desc:	Header for each column	
******************************************************************************/
typedef struct F_COLUMN_ITEM
{
	FLMUINT	uiDataLen;
	FLMUINT	uiDataOffset;
} F_COLUMN_ITEM;

/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT allocOverhead( void)
{
	// Round sizeof( F_Row *) + 1 to nearest 8 byte boundary.
	
	return( (sizeof( F_Row *) + 9) & (~((FLMUINT)7)));
}

/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMBYTE * getActualPointer(
	void *	pvPtr)
{
	if (pvPtr)
	{
		return( (FLMBYTE *)pvPtr - allocOverhead());
	}
	else
	{
		return( NULL);
	}
}

/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT calcColumnListBufSize(
	FLMUINT	uiColumnCount)
{
	return( uiColumnCount * sizeof( F_COLUMN_ITEM) + allocOverhead()); 
}
	
/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT calcDataBufSize(
	FLMUINT	uiDataSize)
{
	return( uiDataSize + allocOverhead());
}

/*****************************************************************************
Desc:	Cached Row
******************************************************************************/
class F_Row : public F_CachedItem
{
public:

	F_Row();
	
	~F_Row();

	// This method assumes that the row cache mutex has been locked.

	FINLINE FLMBOOL canBeFreed( void)
	{
		return( (!rowInUse() && !readingInRow() && !rowIsDirty()) ? TRUE : FALSE);
	}
	
	// This method assumes that the row cache mutex has been locked.

	FINLINE void freeRow( void)
	{
		if (rowPurged())
		{
			freePurged();
		}
		else
		{
			freeCache( FALSE);
		}
	}

	FINLINE FLMUINT64 getLowTransId( void)
	{
		return( m_ui64LowTransId);
	}
		
	FINLINE FLMUINT64 getHighTransId( void)
	{
		return( m_ui64HighTransId);
	}
		
	RCODE resizeDataBuffer(
		FLMUINT		uiSize,
		FLMBOOL		bMutexAlreadyLocked);
		
	RCODE resizeColumnList(
		FLMUINT	uiColumnCount,
		FLMBOOL	bMutexAlreadyLocked);
		
	RCODE flushRow(
		F_Db *		pDb,
		F_Btree *	pBTree);
		
	RCODE flushRow(
		F_Db *	pDb);
		
	RCODE copyColumnList(
		F_Db *			pDb,
		F_Row *			pSourceRow,
		FLMBOOL			bMutexAlreadyLocked);
		
	RCODE readRow(
		F_Db *			pDb,
		FLMUINT			uiTableNum,
		FLMUINT64		ui64RowId,
		IF_IStream *	pIStream,
		FLMUINT			uiRowDataLength);

	FINLINE void setRowAndDataPtr(
		FLMBYTE *	pucActualAlloc)
	{
		*((F_Row **)(pucActualAlloc)) = this;
		m_pucColumnData = pucActualAlloc + allocOverhead();
	}
	
	FINLINE void setColumnListPtr(
		FLMBYTE *	pucActualAlloc)
	{
		*((F_Row **)(pucActualAlloc)) = this;
		m_pColumns = (F_COLUMN_ITEM *)(pucActualAlloc + allocOverhead());
	}

	FINLINE FLMUINT64 getRowId( void)
	{
		return( m_ui64RowId);
	}

	FINLINE F_Database * getDatabase( void)
	{
		return( m_pDatabase);
	}

	FINLINE FLMUINT getTableNum( void)
	{
		return( m_uiTableNum);
	}

	FINLINE FLMUINT getOffsetIndex( void)
	{
		return( m_uiOffsetIndex);
	}
	
	FINLINE void setOffsetIndex(
		FLMUINT	uiOffsetIndex)
	{
		m_uiOffsetIndex = uiOffsetIndex;
	}
	
	FINLINE FLMUINT32 getBlkAddr( void)
	{
		return( m_ui32BlkAddr);
	}
	
	FINLINE void setBlkAddr(
		FLMUINT32	ui32BlkAddr)
	{
		m_ui32BlkAddr = ui32BlkAddr;
	}
	
	FINLINE FLMBOOL isRightVersion(
		FLMUINT64	ui64TransId)
	{
		return( (ui64TransId >= m_ui64LowTransId &&
					ui64TransId <= m_ui64HighTransId)
				  ? TRUE
				  : FALSE);
	}
	
	// Generally, assumes that the row cache mutex has already been locked.
	// There is one case where it is not locked, but it is not
	// critical that it be locked - inside syncFromDb.

	FINLINE FLMBOOL rowPurged( void)
	{
		return( (m_uiCacheFlags & NCA_PURGED ) ? TRUE : FALSE);
	}
	
#ifdef FLM_DEBUG
	void checkReadFromDisk(
		F_Db *	pDb);
#endif

	void * operator new(
		FLMSIZET			uiSize)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new(
		FLMSIZET			uiSize,
		const char *	pszFile,
		int				iLine)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new[](
		FLMSIZET			uiSize)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void * operator new[](
		FLMSIZET			uiSize,
		const char *	pszFile,
		int				iLine)
	#if !defined( FLM_NLM)
		throw()
	#endif
		;

	void operator delete(
		void *			ptr);

	void operator delete[](
		void *			ptr);

#if defined( FLM_DEBUG) && !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete(
		void *			ptr,
		const char *	file,
		int				line);
#endif

#if defined( FLM_DEBUG) && !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
	void operator delete[](
		void *			ptr,
		const char *	pszFileName,
		int				iLineNum);
#endif
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void incrRowUseCount( void)
	{
		m_uiCacheFlags = (m_uiCacheFlags & (~(NCA_COUNTER_BITS))) |
						 (((m_uiCacheFlags & NCA_COUNTER_BITS) + 1));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void decrRowUseCount( void)
	{
		m_uiCacheFlags = (m_uiCacheFlags & (~(NCA_COUNTER_BITS))) |
						 (((m_uiCacheFlags & NCA_COUNTER_BITS) - 1));
	}
	
	void setRowDirty(
		F_Db *		pDb,
		FLMBOOL		bNew);
	
	void unsetRowDirtyAndNew(
		F_Db *			pDb,
		FLMBOOL			bMutexAlreadyLocked = FALSE);
	
	FINLINE FLMBOOL rowIsDirty( void)
	{
		return( (m_uiFlags & FROW_DIRTY) ? TRUE : FALSE);
	}
		
	FINLINE FLMBOOL rowIsNew( void)
	{
		return( (m_uiFlags & FROW_NEW) ? TRUE : FALSE);
	}

	// Assumes that the row cache mutex has already been locked.
	FINLINE FLMBOOL rowUncommitted( void)
	{
		return( (m_uiCacheFlags & NCA_UNCOMMITTED ) ? TRUE : FALSE);
	}
	
	void freeCache(
		FLMBOOL	bPutInPurgeList);
		
	// Column functions

	FINLINE FLMBYTE * getColumnDataPtr(
		FLMUINT	uiColumnNum)
	{
		F_COLUMN_ITEM *	pColumnItem = getColumn( uiColumnNum);
		return( (FLMBYTE *)(pColumnItem->uiDataLen <= sizeof( FLMUINT)
								  ? (FLMBYTE *)(&pColumnItem->uiDataOffset)
								  : (FLMBYTE *)(m_pucColumnData + pColumnItem->uiDataOffset)));
	}
		
	RCODE	allocColumnDataSpace(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT		uiSizeNeeded,
		FLMBOOL		bMutexAlreadyLocked);
		
	RCODE setNumber64(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT64	ui64Value,
		FLMBOOL		bNeg);
		
	RCODE getNumber64(
		F_Db *			pDb,
		FLMUINT			uiColumnNum,
		FLMUINT64 *		pui64Value,
		FLMBOOL *		pbNeg,
		FLMBOOL *		pbIsNull);

	FINLINE RCODE getUINT32(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT32 *	pui32Num,
		FLMBOOL *	pbIsNull)
	{
		RCODE			rc = NE_SFLM_OK;
		FLMUINT64	ui64Num;
		FLMBOOL		bNeg;
		
		if( RC_OK( rc = getNumber64( pDb, uiColumnNum, &ui64Num,
									&bNeg, pbIsNull)))
		{
			if (!(*pbIsNull))
			{
				rc = convertToUINT32( ui64Num, bNeg, pui32Num);
			}
		}
		return( rc);
	}

	FINLINE RCODE getUINT(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT *	puiNum,
		FLMBOOL *	pbIsNull)
	{
		RCODE			rc = NE_SFLM_OK;
		FLMUINT64	ui64Num;
		FLMBOOL		bNeg;
		
		if( RC_OK( rc = getNumber64( pDb, uiColumnNum, &ui64Num,
									&bNeg, pbIsNull)))
		{
			if (!(*pbIsNull))
			{
				rc = convertToUINT( ui64Num, bNeg, puiNum);
			}
		}
		return( rc);
	}

	FINLINE RCODE getUINT64(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT64 *	pui64Num,
		FLMBOOL *	pbIsNull)
	{
		RCODE			rc = NE_SFLM_OK;
		FLMUINT64	ui64Num;
		FLMBOOL		bNeg;
		
		if( RC_OK( rc = getNumber64( pDb, uiColumnNum, &ui64Num,
									&bNeg, pbIsNull)))
		{
			if (!(*pbIsNull))
			{
				rc = convertToUINT64( ui64Num, bNeg, pui64Num);
			}
		}
		
		return( rc);
	}

	FINLINE RCODE getINT(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMINT *		piNum,
		FLMBOOL *	pbIsNull)
	{
		RCODE			rc = NE_SFLM_OK;
		FLMUINT64	ui64Num;
		FLMBOOL		bNeg;
		
		if (RC_OK( rc = getNumber64( pDb, uiColumnNum, &ui64Num,
									&bNeg, pbIsNull)))
		{
			if (!(*pbIsNull))
			{
				rc = convertToINT( ui64Num, bNeg, piNum);
			}
		}
		
		return( rc);
	}

	FINLINE RCODE getINT64(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMINT64 *	pi64Num,
		FLMBOOL *	pbIsNull)
	{
		RCODE			rc = NE_SFLM_OK;
		FLMUINT64	ui64Num;
		FLMBOOL		bNeg;
		
		if (RC_OK( rc = getNumber64( pDb, uiColumnNum, &ui64Num,
									&bNeg, pbIsNull)))
		{
			if (!(*pbIsNull))
			{
				rc = convertToINT64( ui64Num, bNeg, pi64Num);
			}
		}
		
		return( rc);
	}

	FINLINE RCODE setUINT(
		F_Db *	pDb,
		FLMUINT	uiColumnNum,
		FLMUINT	uiValue)
	{
		return( setNumber64( pDb, uiColumnNum, (FLMUINT64)uiValue, FALSE));
	}

	FINLINE RCODE setUINT64(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		FLMUINT64	ui64Value)
	{
		return( setNumber64( pDb, uiColumnNum, ui64Value, FALSE));
	}

	FINLINE RCODE setINT(
		F_Db *	pDb,
		FLMUINT	uiColumnNum,
		FLMINT	iValue)
	{
		FLMBOOL	bNeg;
		FLMUINT	ui64Value;
		
		if (iValue < 0)
		{
			bNeg = TRUE;
			ui64Value = (FLMUINT64)((FLMINT64)(-iValue));
		}
		else
		{
			bNeg = FALSE;
			ui64Value = (FLMUINT64)iValue;
		}
		return( setNumber64( pDb, uiColumnNum, ui64Value, bNeg));
	}

	FINLINE RCODE setINT64(
		F_Db *	pDb,
		FLMUINT	uiColumnNum,
		FLMINT64	i64Value)
	{
		FLMBOOL	bNeg;
		FLMUINT	ui64Value;
		
		if (i64Value < 0)
		{
			bNeg = TRUE;
			ui64Value = (FLMUINT64)(-i64Value);
		}
		else
		{
			bNeg = FALSE;
			ui64Value = (FLMUINT64)i64Value;
		}
		return( setNumber64( pDb, uiColumnNum, ui64Value, bNeg));
	}

	RCODE setValue(
		F_Db *				pDb,
		F_COLUMN *			pColumn,
		const FLMBYTE *	pucValue,
		FLMUINT				uiValueLen);
	
	RCODE setUTF8(
		F_Db *			pDb,
		FLMUINT			uiColumnNum,
		const char *	pszValue,
		FLMUINT			uiNumBytesInValue,
		FLMUINT			uiNumCharsInValue);
	
	RCODE getUTF8(
		F_Db *		pDb,
		FLMUINT		uiColumnNum,
		char *		pszValueBuffer,
		FLMUINT		uiBufferSize,
		FLMBOOL *	pbIsNull,
		FLMUINT *	puiCharsReturned,
		FLMUINT *	puiBufferBytesUsed);
		
	RCODE setBinary(
		F_Db *			pDb,
		FLMUINT			uiColumnNum,
		const void *	pvValue,
		FLMUINT			uiNumBytesInBuffer);

	RCODE getBinary(
		F_Db *			pDb,
		FLMUINT			uiColumnNum,
		void *			pvBuffer,
		FLMUINT			uiBufferLen,
		FLMUINT *		puiDataLen,
		FLMBOOL *		pbIsNull);
		
	void setToNull(
		F_Db *			pDb,
		FLMUINT			uiColumnNum);
		
	void getDataLen(
		F_Db *			pDb,
		FLMUINT			uiColumnNum,
		FLMUINT *		puiDataLen,
		FLMBOOL *		pbIsNull);
		
	FINLINE F_COLUMN_ITEM * getColumn(
		FLMUINT	uiColumnNum)
	{
		return( (F_COLUMN_ITEM *)((uiColumnNum && uiColumnNum <= m_uiNumColumns &&
			 								m_pColumns [uiColumnNum - 1].uiDataLen)
										  ? &m_pColumns [uiColumnNum - 1]
										  : (F_COLUMN_ITEM *)NULL));
	}

	RCODE getIStream(
		F_Db *				pDb,
		FLMUINT				uiColumnNum,
		FLMBOOL *			pbIsNull,
		F_BufferIStream *	pBufferIStream,
		eDataType *			peDataType,
		FLMUINT *			puiDataLength);
		
	RCODE getTextIStream(
		F_Db *				pDb,
		FLMUINT				uiColumnNum,
		FLMBOOL *			pbIsNull,
		F_BufferIStream *	pBufferIStream,
		FLMUINT *			puiNumChars);

	FINLINE void ReleaseRow( void)
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		decrRowUseCount();
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}
		
private:

	// Assumes that the row cache mutex has already been locked.
	FINLINE FLMBOOL readingInRow( void)
	{
		return( (m_uiCacheFlags & NCA_READING_IN ) ? TRUE : FALSE);
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void setReadingIn( void)
	{
		m_uiCacheFlags |= NCA_READING_IN;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unsetReadingIn( void)
	{
		m_uiCacheFlags &= (~(NCA_READING_IN));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void setUncommitted( void)
	{
		m_uiCacheFlags |= NCA_UNCOMMITTED;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unsetUncommitted( void)
	{
		m_uiCacheFlags &= (~(NCA_UNCOMMITTED));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE FLMBOOL rowIsLatestVer( void)
	{
		return( (m_uiCacheFlags & NCA_LATEST_VER ) ? TRUE : FALSE);
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void setLatestVer( void)
	{
		m_uiCacheFlags |= NCA_LATEST_VER;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unsetLatestVer( void)
	{
		m_uiCacheFlags &= (~(NCA_LATEST_VER));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void setPurged( void)
	{
		m_uiCacheFlags |= NCA_PURGED;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unsetPurged( void)
	{
		m_uiCacheFlags &= (~(NCA_PURGED));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE FLMBOOL rowLinkedToDatabase( void)
	{
		return( (m_uiCacheFlags & NCA_LINKED_TO_DATABASE ) ? TRUE : FALSE);
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void setLinkedToDatabase( void)
	{
		m_uiCacheFlags |= NCA_LINKED_TO_DATABASE;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unsetLinkedToDatabase( void)
	{
		m_uiCacheFlags &= (~(NCA_LINKED_TO_DATABASE));
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE FLMBOOL rowInUse( void)
	{
		return( (m_uiCacheFlags & NCA_COUNTER_BITS ) ? TRUE : FALSE);
	}
	
	// Unlink a row from the global purged list.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromPurged( void)
	{
		if (m_pNextInGlobal)
		{
			m_pNextInGlobal->m_pPrevInGlobal = m_pPrevInGlobal;
		}

		if (m_pPrevInGlobal)
		{
			m_pPrevInGlobal->m_pNextInGlobal = m_pNextInGlobal;
		}
		else
		{
			gv_SFlmSysData.pRowCacheMgr->m_pPurgeList = 
				(F_Row *)m_pNextInGlobal;
		}

		m_pPrevInGlobal = NULL;
		m_pNextInGlobal = NULL;
	}
	
	// Link a row to an F_Database list at the head of the list.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToDatabaseAtHead(
		F_Database *	pDatabase)
	{
		if (!pDatabase->m_pLastDirtyRow || rowIsDirty())
		{
			m_pPrevInDatabase = NULL;
			if ((m_pNextInDatabase = pDatabase->m_pFirstRow) != NULL)
			{
				pDatabase->m_pFirstRow->m_pPrevInDatabase = this;
			}
			else
			{
				pDatabase->m_pLastRow = this;
			}
		
			pDatabase->m_pFirstRow = this;
			if (rowIsDirty() && !pDatabase->m_pLastDirtyRow)
			{
				pDatabase->m_pLastDirtyRow = this;
			}
		}
		else
		{
			// pDatabase->m_pLastDirtyRow is guaranteed to be non-NULL,
			// Hence, m_pPrevInDatabase will be non-NULL.
			// We are also guaranteed that the row is not dirty.
			
			m_pPrevInDatabase = pDatabase->m_pLastDirtyRow;
			m_pNextInDatabase = m_pPrevInDatabase->m_pNextInDatabase;

			m_pPrevInDatabase->m_pNextInDatabase = this;
			if (m_pNextInDatabase)
			{
				m_pNextInDatabase->m_pPrevInDatabase = this;
			}
			else
			{
				pDatabase->m_pLastRow = this;
			}
		}

		m_pDatabase = pDatabase;
		setLinkedToDatabase();
	}
	
	// Link a row to an F_Database list at the end of the list.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToDatabaseAtEnd(
		F_Database *	pDatabase)
	{
		// Row cannot be a dirty row.
		
		flmAssert( !rowIsDirty());
		m_pNextInDatabase = NULL;
		if( (m_pPrevInDatabase = pDatabase->m_pLastRow) != NULL)
		{
			pDatabase->m_pLastRow->m_pNextInDatabase = this;
		}
		else
		{
			pDatabase->m_pFirstRow = this;
		}

		pDatabase->m_pLastRow = this;
		m_pDatabase = pDatabase;
		setLinkedToDatabase();
	}
	
	// Unlink a row from its F_Database list.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromDatabase( void)
	{
		if( rowLinkedToDatabase())
		{
			// If this is the last dirty row, change the database's
			// last dirty pointer to point to the previous row, if any.
			
			if (m_pDatabase->m_pLastDirtyRow == this)
			{
				flmAssert( rowIsDirty());
				m_pDatabase->m_pLastDirtyRow = m_pPrevInDatabase;
			}
			
			// Remove the row from the database's list.
			
			if( m_pNextInDatabase)
			{
				m_pNextInDatabase->m_pPrevInDatabase = m_pPrevInDatabase;
			}
			else
			{
				m_pDatabase->m_pLastRow = m_pPrevInDatabase;
			}

			if( m_pPrevInDatabase)
			{
				m_pPrevInDatabase->m_pNextInDatabase = m_pNextInDatabase;
			}
			else
			{
				m_pDatabase->m_pFirstRow = m_pNextInDatabase;
			}

			m_pPrevInDatabase = NULL;
			m_pNextInDatabase = NULL;
			m_pDatabase = NULL;
			unsetLinkedToDatabase();
		}
	}
	
	// Link a row into its hash bucket.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToHashBucket( void)
	{
		F_Row ** ppHashBucket = gv_SFlmSysData.pRowCacheMgr->rowHash( 
													m_ui64RowId);
		
		flmAssert( m_pNewerVersion == NULL);
	
		m_pPrevInBucket = NULL;

		if ((m_pNextInBucket = *ppHashBucket) != NULL)
		{
			m_pNextInBucket->m_pPrevInBucket = this;
		}

		*ppHashBucket = this;
	}
	
	// Unlink a row from its hash bucket.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromHashBucket( void)
	{
		flmAssert( m_pNewerVersion == NULL);
		
		if (m_pNextInBucket)
		{
			m_pNextInBucket->m_pPrevInBucket = m_pPrevInBucket;
		}

		if (m_pPrevInBucket)
		{
			m_pPrevInBucket->m_pNextInBucket = m_pNextInBucket;
		}
		else
		{
			F_Row ** ppHashBucket =
									gv_SFlmSysData.pRowCacheMgr->rowHash( 
											m_ui64RowId);
			
			*ppHashBucket = m_pNextInBucket;
		}

		m_pPrevInBucket = NULL;
		m_pNextInBucket = NULL;
	}
	
	// Unlink a row from its version list.
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToVerList(
		F_Row *	pNewerVer,
		F_Row *	pOlderVer)
	{
		if( (m_pNewerVersion = pNewerVer) != NULL)
		{
			pNewerVer->m_pOlderVersion = this;
		}

		if ((m_pOlderVersion = pOlderVer) != NULL)
		{
			pOlderVer->m_pNewerVersion = this;
		}
	}
	
	// Unlink a row from its version list.  This routine
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromVerList( void)
	{
		if (m_pNewerVersion)
		{
			m_pNewerVersion->m_pOlderVersion = m_pOlderVersion;
		}

		if (m_pOlderVersion)
		{
			m_pOlderVersion->m_pNewerVersion = m_pNewerVersion;
		}

		m_pNewerVersion = NULL;
		m_pOlderVersion = NULL;
	}

	// Link a row into the heap list
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToHeapList( void)
	{
		flmAssert( !m_pPrevInHeapList);
		flmAssert( (m_uiFlags & FROW_HEAP_ALLOC) == 0);
	
		if( (m_pNextInHeapList = 
			gv_SFlmSysData.pRowCacheMgr->m_pHeapList) != NULL)
		{
			gv_SFlmSysData.pRowCacheMgr->m_pHeapList->m_pPrevInHeapList = this;
		}
		
		gv_SFlmSysData.pRowCacheMgr->m_pHeapList = this;
		m_uiFlags |= FROW_HEAP_ALLOC;
	}
	
	// Unlink a row from the heap list
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromHeapList( void)
	{
		flmAssert( m_uiFlags & FROW_HEAP_ALLOC);
	
		if (m_pNextInHeapList)
		{
			m_pNextInHeapList->m_pPrevInHeapList = m_pPrevInHeapList;
		}
		
		if (m_pPrevInHeapList)
		{
			m_pPrevInHeapList->m_pNextInHeapList = m_pNextInHeapList;
		}
		else
		{
			gv_SFlmSysData.pRowCacheMgr->m_pHeapList = m_pNextInHeapList;
		}
		
		m_pPrevInHeapList = NULL;
		m_pNextInHeapList = NULL;
		m_uiFlags &= ~FROW_HEAP_ALLOC;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void linkToOldList( void)
	{
		flmAssert( !m_pPrevInOldList);
	
		if( (m_pNextInOldList = 
			gv_SFlmSysData.pRowCacheMgr->m_pOldList) != NULL)
		{
			gv_SFlmSysData.pRowCacheMgr->m_pOldList->m_pPrevInOldList = this;
		}
		
		gv_SFlmSysData.pRowCacheMgr->m_pOldList = this;
	}
	
	// Assumes that the row cache mutex has already been locked.
	FINLINE void unlinkFromOldList( void)
	{
		if (m_pNextInOldList)
		{
			m_pNextInOldList->m_pPrevInOldList = m_pPrevInOldList;
		}
		
		if (m_pPrevInOldList)
		{
			m_pPrevInOldList->m_pNextInOldList = m_pNextInOldList;
		}
		else
		{
			gv_SFlmSysData.pRowCacheMgr->m_pOldList = m_pNextInOldList;
		}
		
		m_pPrevInOldList = NULL;
		m_pNextInOldList = NULL;
	}
	
	FINLINE FLMUINT memSize( void)
	{
		FLMUINT	uiSize = gv_SFlmSysData.pRowCacheMgr->m_pRowAllocator->getCellSize();
				
		if (m_pucColumnData)
		{
			uiSize += gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->getTrueSize( 
				calcDataBufSize( m_uiColumnDataBufSize),
				getActualPointer( m_pucColumnData));
		}

		if (m_pColumns)
		{
			uiSize += gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->getTrueSize(
				calcColumnListBufSize( m_uiNumColumns),
				getActualPointer( m_pColumns));
		}

		return( uiSize);
	}
	
	// Assumes that the row cache mutex is locked, because
	// it potentially updates the cache usage statistics.

	FINLINE void setTransID(
		FLMUINT64		ui64NewTransID)
	{
		FLMUINT	uiSize;
		
		if (m_ui64HighTransId == FLM_MAX_UINT64 &&
			 ui64NewTransID != FLM_MAX_UINT64)
		{
			uiSize = memSize();
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes += uiSize;
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount++;
			linkToOldList();
		}
		else if (m_ui64HighTransId != FLM_MAX_UINT64 &&
					ui64NewTransID == FLM_MAX_UINT64)
		{
			uiSize = memSize();
			flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
			flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount);
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount--;
			unlinkFromOldList();
		}

		m_ui64HighTransId = ui64NewTransID;
	}

	void freePurged( void);
		
	void linkToDatabase(
		F_Database *	pDatabase,
		F_Db *			pDb,
		FLMUINT64		ui64LowTransId,
		FLMBOOL			bMostCurrent);
		
	void resetRow( void);
		
	// Things that manage the row's place in row cache.  These are
	// always initialized in the constructor
	
	F_Row *			m_pPrevInBucket;
	F_Row *			m_pNextInBucket;
	F_Row *			m_pPrevInDatabase;
	F_Row *			m_pNextInDatabase;
	F_Row *			m_pOlderVersion;
	F_Row *			m_pNewerVersion;
	F_Row *			m_pPrevInHeapList;
	F_Row *			m_pNextInHeapList;
	F_Row *			m_pPrevInOldList;
	F_Row *			m_pNextInOldList;
	
	FLMUINT64				m_ui64LowTransId;
	FLMUINT64				m_ui64HighTransId;
	FNOTIFY *				m_pNotifyList;
	FLMUINT					m_uiCacheFlags;
	FLMUINT					m_uiStreamUseCount;
	
	// Things we hash on - initialized by caller of constructor
	
	F_Database *			m_pDatabase;
	
	// Items initialized in constructor
	
	FLMUINT					m_uiTableNum;
	FLMUINT64				m_ui64RowId;
	FLMUINT					m_uiFlags;
	F_COLUMN_ITEM *		m_pColumns;
	FLMUINT					m_uiNumColumns;
	FLMBYTE *				m_pucColumnData;
	FLMUINT					m_uiColumnDataBufSize;
	
	// Items initialized by caller of constructor, but not
	// in constructor - for performance reasons - so we don't
	// end up setting them twice.
	
	FLMUINT					m_uiOffsetIndex;
	FLMUINT32				m_ui32BlkAddr;
	
	// Items that m_uiFlags indicates whether they are present

friend class F_RowCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_Db;
friend class F_DbSystem;
friend class F_BTreeIStream;
friend class F_Rfl;
friend class F_RebuildRowIStream;
friend class F_DbRebuild;
friend class F_RowRelocator;
friend class F_ColumnDataRelocator;
friend class F_ColumnListRelocator;	
};

#endif // FCACHE_H
