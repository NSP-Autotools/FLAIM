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
class F_DOMNode;
class F_Db;
class F_DbSystem;
class F_Database;
class F_LocalNodeCache;
class F_CachedNode;
class F_CachedBlock;
class F_BlockCacheMgr;
class F_NodeCacheMgr;
class F_GlobalCacheMgr;
class F_CacheList;
class F_CachedItem;
class F_Btree;
class F_BTreeIStream;
class F_NodeInfo;
class F_BTreeInfo;
class F_DynaBuf;
class F_AttrItem;
class F_NodeRelocator;
class F_NodeDataRelocator;
class F_NodeListRelocator;	
class F_AttrListRelocator;	
class F_AttrItemRelocator;	
class F_AttrBufferRelocator;	
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
					gv_XFlmSysData.uiRehashAfterFailureBackoffTime)
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
class F_CachedItem : public F_Object
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
friend class F_CachedNode;
friend class F_GlobalCacheMgr;
friend class F_BlockCacheMgr;
friend class F_NodeCacheMgr;
friend class F_Database;
friend class F_Db;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
friend class F_BlockRelocator;
};

/***************************************************************************
Desc:	Object for keeping track of an MRU/LRU list of cached items (nodes
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
friend class F_NodeCacheMgr;
friend class F_BlockCacheMgr;
friend class F_CachedBlock;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
friend class F_BlockRelocator;
};
		
/***************************************************************************
Desc:	Global cache manager for XFLAIM.
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
		XFLM_CACHE_INFO *	pMemInfo);
		
	RCODE adjustCache(
		FLMUINT *	puiCurrTime,
		FLMUINT *	puiLastCacheAdjustTime);

	RCODE clearCache(
		IF_Db *		pDb);

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
friend class F_CachedNode;
friend class F_CachedBlock;
friend class F_BlockCacheMgr;
friend class F_NodeCacheMgr;
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

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
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
		FLMUINT	uiBlkAddress)
	{
		return( &m_ppHashBuckets[ (uiBlkAddress >> 
			uiSigBitsInBlkSize) & m_uiHashMask]);
	}

	FINLINE void defragmentMemory(
		FLMBOOL		bMutexLocked = FALSE)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
		}

		m_pBlockAllocator->defragmentMemory();

		if( !bMutexLocked)
		{
			f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
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
	XFLM_CACHE_USAGE	m_Usage;			// Contains usage information.
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
		return( gv_XFlmSysData.pBlockCacheMgr->m_pBlockAllocator->getTrueSize( 
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
			gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes += memSize();
			gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount++;
		}
		else if (m_ui64HighTransID != FLM_MAX_UINT64 && ui64NewTransID == FLM_MAX_UINT64)
		{
			FLMUINT	uiSize = memSize();
	
			flmAssert( gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
			gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
			flmAssert( gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount);
			gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount--;
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
			gv_XFlmSysData.pBlockCacheMgr->m_pMRUReplace) != NULL)
		{
			m_pNextInReplaceList->m_pPrevInReplaceList = this;
		}
		else
		{
			gv_XFlmSysData.pBlockCacheMgr->m_pLRUReplace = this;
		}
		
		m_pPrevInReplaceList = NULL;
		gv_XFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
		gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableCount++;
		gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes += memSize();
	}

	// Link a cache block into the replace list as the LRU item. This routine
	// assumes that the block cache mutex has already been locked.
	FINLINE void linkToReplaceListAsLRU( void)
	{
		flmAssert( !m_ui16Flags);
	
		if ((m_pPrevInReplaceList = gv_XFlmSysData.pBlockCacheMgr->m_pLRUReplace) != NULL)
		{
			m_pPrevInReplaceList->m_pNextInReplaceList = this;
		}
		else
		{
			gv_XFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
		}

		m_pNextInReplaceList = NULL;
		gv_XFlmSysData.pBlockCacheMgr->m_pLRUReplace = this;
		gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableCount++;
		gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes += memSize();
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
				gv_XFlmSysData.pBlockCacheMgr->m_pMRUReplace = this;
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
				gv_XFlmSysData.pBlockCacheMgr->m_pLRUReplace = pPrevSCache;
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
			 (gv_XFlmSysData.pBlockCacheMgr->m_bDebug && !m_uiUseCount))
		{
			dbgUseForThread( uiThreadId);
		}
		else
	#endif
		{
			if (!m_uiUseCount)
			{
				gv_XFlmSysData.pBlockCacheMgr->m_uiBlocksUsed++;
			}
			m_uiUseCount++;
			gv_XFlmSysData.pBlockCacheMgr->m_uiTotalUses++;
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
			gv_XFlmSysData.pBlockCacheMgr->m_uiTotalUses--;
			if (!m_uiUseCount)
			{
				gv_XFlmSysData.pBlockCacheMgr->m_uiBlocksUsed--;
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
			// case, the block does not need to have a prior version pointer -
			// because there are none!
			
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

	void operator delete[](
		void *			ptr);

	void operator delete(
		void *			ptr,
		FLMUINT			uiBlockSize);

	void operator delete(
		void *			ptr,
		const char *	pszFile,
		int				iLine);

	void operator delete[](
		void *			ptr,
		const char *	pszFile,
		int				iLine);

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
	
	// Link a cached block into the global list as the MRU item. This routine
	// assumes that the block cache mutex has already been locked.
	
	FINLINE void linkToGlobalListAsMRU( void)
	{
		if( (m_pBlkHdr->ui8BlkType & BT_FREE) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF) ||
			 (m_pBlkHdr->ui8BlkType & BT_LEAF_DATA) ||
			 (m_pBlkHdr->ui8BlkType & BT_DATA_ONLY))
		{
			gv_XFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLastMRU(
							(F_CachedItem *)this);
		}
		else
		{
			gv_XFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsMRU(
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
			gv_XFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLRU(
						(F_CachedItem *)this);
		}
		else
		{
			gv_XFlmSysData.pBlockCacheMgr->m_MRUList.linkGlobalAsLastMRU(
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
		gv_XFlmSysData.pBlockCacheMgr->m_MRUList.unlinkGlobal(
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
		gv_XFlmSysData.pBlockCacheMgr->m_MRUList.stepUpInGlobal(
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

	static void XFLAPI objectAllocInit(
		void *				pvAlloc,
		FLMUINT				uiSize);

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
	F_NOTIFY_LIST_ITEM *	m_pNotifyList;		// This is a pointer to a list of threads
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
	FLMBOOL				m_bCanRelocate;		// Can the block object be relocated
														// if defragmenting memory?

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
Desc:	Class for moving nodes in cache.
****************************************************************************/
class F_NodeRelocator : public IF_Relocator
{
public:

	F_NodeRelocator()
	{
	}
	
	virtual ~F_NodeRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving node data buffers in cache.
****************************************************************************/
class F_NodeDataRelocator : public IF_Relocator
{
public:

	F_NodeDataRelocator()
	{
	}
	
	virtual ~F_NodeDataRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving node lists in cache.
****************************************************************************/
class F_NodeListRelocator : public IF_Relocator
{
public:

	F_NodeListRelocator()
	{
	}
	
	virtual ~F_NodeListRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving attr lists in cache.
****************************************************************************/
class F_AttrListRelocator : public IF_Relocator
{
public:

	F_AttrListRelocator()
	{
	}
	
	virtual ~F_AttrListRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving attr items in cache.
****************************************************************************/
class F_AttrItemRelocator : public IF_Relocator
{
public:

	F_AttrItemRelocator()
	{
	}
	
	virtual ~F_AttrItemRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Class for moving attr buffers in cache.
****************************************************************************/
class F_AttrBufferRelocator : public IF_Relocator
{
public:

	F_AttrBufferRelocator()
	{
	}
	
	virtual ~F_AttrBufferRelocator()
	{
	}

	void XFLAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL XFLAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Struct:	NODE_CACHE_MGR (XFLAIM Node Cache Manager)
Desc:	 	This structure defines the header information that is used to
			control the FLAIM record cache.  This structure will be embedded
			in the FLMSYSDATA structure.
****************************************************************************/
class F_NodeCacheMgr : public F_Object
{
public:
	F_NodeCacheMgr();
	
	~F_NodeCacheMgr();
	
	void insertDOMNode(
		F_DOMNode *		pNode);

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
	
	RCODE allocNode(
		F_CachedNode **	ppNode,
		FLMBOOL				bMutexLocked);
		
	RCODE allocDOMNode(
		F_DOMNode **		ppDOMNode);
		
	RCODE retrieveNode(
		F_Db *			pDb,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		F_DOMNode **	ppDOMNode);
		
	RCODE createNode(
		F_Db *			pDb,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		F_DOMNode **	ppDOMNode);
		
	RCODE makeWriteCopy(
		F_Db *				pDb,
		F_CachedNode **	ppCachedNode);

	void removeNode(
		F_Db *			pDb,
		F_CachedNode *	pNode,
		FLMBOOL			bDecrementUseCount,
		FLMBOOL			bMutexLocked = FALSE);
		
	void removeNode(
		F_Db *			pDb,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId);

	FINLINE void defragmentMemory(
		FLMBOOL		bMutexLocked = FALSE)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		}

		m_pNodeAllocator->defragmentMemory();
		m_pBufAllocator->defragmentMemory();
		m_pAttrItemAllocator->defragmentMemory();

		if( !bMutexLocked)
		{
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		}
	}

private:

	// Hash function for hashing to nodes in node cache.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE F_CachedNode ** nodeHash(
		FLMUINT64	ui64NodeId)
	{
		return( &m_ppHashBuckets[(FLMUINT)ui64NodeId & m_uiHashMask]);
	}

	RCODE rehash( void);
	
	RCODE waitNotify(
		F_Db *				pDb,
		F_CachedNode **	ppNode);
		
	void notifyWaiters(
		F_NOTIFY_LIST_ITEM *	pNotify,
		F_CachedNode *			pUseNode,
		RCODE						NotifyRc);
		
	void linkIntoNodeCache(
		F_CachedNode *	pNewerNode,
		F_CachedNode *	pOlderNode,
		F_CachedNode *	pNode,
		FLMBOOL			bLinkAsMRU);
		
	void findNode(
		F_Db *				pDb,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT64			ui64VersionNeeded,
		FLMBOOL				bDontPoisonCache,
		FLMUINT *			puiNumLooks,
		F_CachedNode **	ppNode,
		F_CachedNode **	ppNewerNode,
		F_CachedNode **	ppOlderNode);
		
	RCODE readNodeFromDisk(
		F_Db *				pDb,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		F_CachedNode *		pNode,
		FLMUINT64 *			pui64LowTransId,
		FLMBOOL *			pbMostCurrent);
		
	RCODE _makeWriteCopy(
		F_Db *				pDb,
		F_CachedNode **	ppCachedNode);
		
	// Private Data
	
	F_CacheList			m_MRUList;			// List of all node objects in MRU order.
	F_CachedNode *		m_pPurgeList;		// List of F_CachedNode objects that
													// should be deleted when the use count
													// goes to zero.
	F_CachedNode *		m_pHeapList;		// List of nodes with heap allocations
	F_CachedNode *		m_pOldList;			// List of old versions
	XFLM_CACHE_USAGE	m_Usage;				// Contains maximum, bytes used, etc.
	F_CachedNode **	m_ppHashBuckets;	// Array of hash buckets.
	FLMUINT				m_uiNumBuckets;	// Total number of hash buckets.
													// must be an exponent of 2.
	FLMUINT				m_uiHashFailTime;
													// Last time we tried to rehash and
													// failed.  Want to wait before we
													// retry again.
	FLMUINT				m_uiHashMask;		// Hash mask mask for hashing a
													// node id to a hash bucket.
	FLMUINT				m_uiPendingReads;	// Total reads currently pending.
	FLMUINT				m_uiIoWaits;		// Number of times multiple threads
													// were reading the same node from
													// disk at the same time.
	IF_FixedAlloc *	m_pNodeAllocator;	// Fixed size allocator for F_CachedNode
													// objects
	IF_BufferAlloc *	m_pBufAllocator;	// Buffer allocator for buffers in
													// F_CachedNode objects
	IF_FixedAlloc *	m_pAttrItemAllocator;
													// Allocator for attribute list items
	F_NodeRelocator			m_nodeRelocator;
	F_NodeDataRelocator		m_nodeDataRelocator;
	F_NodeListRelocator		m_nodeListRelocator;	
	F_AttrListRelocator		m_attrListRelocator;	
	F_AttrItemRelocator		m_attrItemRelocator;	
	F_AttrBufferRelocator	m_attrBufferRelocator;	
		
	F_DOMNode *			m_pFirstNode;		// List of DOM Nodes in a pool
	FLMBOOL				m_bReduceInProgress;

#ifdef FLM_DEBUG
	FLMBOOL				m_bDebug;			// Debug mode?
#endif
friend class F_CachedNode;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_DbSystem;
friend class F_DOMNode;
friend class F_AttrItem;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
};

// Bits for m_uiCacheFlags field in F_CachedNode

#define NCA_READING_IN					0x80000000
#define NCA_UNCOMMITTED					0x40000000
#define NCA_LATEST_VER					0x20000000
#define NCA_PURGED						0x10000000
#define NCA_LINKED_TO_DATABASE		0x08000000

#define NCA_COUNTER_BITS		(~(NCA_READING_IN | NCA_UNCOMMITTED | \
											NCA_LATEST_VER | NCA_PURGED | \
											NCA_LINKED_TO_DATABASE))

// In-memory node flags.
//
// Flags that are not actually written to disk - only held in memory
// Reserve upper byte for these.  These flags need to be here so that
// we don't have to worry about locking the mutex to access them.

#define FDOM_READ_ONLY					0x00000001
#define FDOM_CANNOT_DELETE				0x00000002
#define FDOM_QUARANTINED				0x00000004
#define FDOM_VALUE_ON_DISK				0x00000008
#define FDOM_SIGNED_QUICK_VAL			0x00000010
#define FDOM_UNSIGNED_QUICK_VAL		0x00000020
#define FDOM_DIRTY						0x00000040
#define FDOM_NEW							0x00000080
#define FDOM_HEAP_ALLOC					0x00000100
#define FDOM_HAVE_CELM_LIST			0x00000200
#define FDOM_NAMESPACE_DECL			0x00000400
#define FDOM_FIXED_SIZE_HEADER		0x00000800

#define FDOM_PERSISTENT_FLAGS			(FDOM_READ_ONLY | \
												 FDOM_CANNOT_DELETE | \
												 FDOM_QUARANTINED | \
												 FDOM_NAMESPACE_DECL)
/*****************************************************************************
Desc:	Node storage flags
******************************************************************************/
#define NSF_HAVE_BASE_ID_BIT				0x0001
#define NSF_HAVE_META_VALUE_BIT			0x0002
#define NSF_HAVE_SIBLINGS_BIT				0x0004
#define NSF_HAVE_CHILDREN_BIT				0x0008
#define NSF_HAVE_ATTR_LIST_BIT			0x0010
#define NSF_HAVE_CELM_LIST_BIT			0x0020
#define NSF_HAVE_DATA_LEN_BIT				0x0040

#define NSF_EXT_HAVE_DCHILD_COUNT_BIT	0x0080
#define NSF_EXT_READ_ONLY_BIT				0x0100
#define NSF_EXT_CANNOT_DELETE_BIT		0x0200
#define NSF_EXT_HAVE_PREFIX_BIT			0x0400
#define NSF_EXT_ENCRYPTED_BIT				0x0800
#define NSF_EXT_NAMESPACE_DECL_BIT		0x1000
#define NSF_EXT_ANNOTATION_BIT			0x2000
#define NSF_EXT_QUARANTINED_BIT			0x4000

/*****************************************************************************
Desc:	Attribute storage flags and masks
******************************************************************************/
#define ASF_PAYLOAD_LEN_MASK				0x0000000F
#define ASF_MAX_EMBEDDED_PAYLOAD_LEN	0x0000000E
#define ASF_HAVE_PAYLOAD_LEN_SEN			0x0000000F

#define ASF_HAVE_PREFIX_BIT				0x00000010
#define ASF_READ_ONLY_BIT					0x00000020
#define ASF_CANNOT_DELETE_BIT				0x00000040
#define ASF_ENCRYPTED_BIT					0x00000080

/*****************************************************************************
Desc:	
******************************************************************************/
typedef struct NODE_ITEM
{
	FLMUINT		uiNameId;
	FLMUINT64	ui64NodeId;
} NODE_ITEM;
	
/*****************************************************************************
Desc:
******************************************************************************/
class F_AttrItem
{
public:

	F_AttrItem()
	{
		m_pCachedNode = NULL;
		m_pucPayload = NULL;
		m_uiPayloadLen = 0;
		m_uiDataType = 0;
		m_uiNameId = 0;
		m_uiFlags = 0;
		m_uiPrefixId = 0;
		m_ui64QuickVal = 0;
		m_uiEncDefId = 0;
		m_uiIVLen = 0;
		m_uiDecryptedDataLen = 0;
	}

	~F_AttrItem();

	RCODE resizePayloadBuffer(
		FLMUINT		uiTotalNeeded,
		FLMBOOL		bMutexAlreadyLocked);
			
	RCODE setupAttribute(
		F_Db *		pDb,
		FLMUINT		uiEncDefId,
		FLMUINT		uiSizeNeeded,
		FLMBOOL		bOkToGenerateIV,
		FLMBOOL		bMutexAlreadyLocked);

	void getAttrSizeNeeded(
		FLMUINT				uiBaseNameId,
		XFLM_NODE_INFO *	pNodeInfo,
		FLMUINT *			puiSaveStorageFlags,
		FLMUINT *			puiSizeNeeded);
		
	FINLINE void copyFrom(
		F_AttrItem *	pSrcAttrItem)
	{
		m_pCachedNode = pSrcAttrItem->m_pCachedNode;
		m_pucPayload = pSrcAttrItem->m_pucPayload;
		m_uiPayloadLen = pSrcAttrItem->m_uiPayloadLen;
		m_uiDataType = pSrcAttrItem->m_uiDataType;
		m_uiNameId = pSrcAttrItem->m_uiNameId;
		m_uiFlags = pSrcAttrItem->m_uiFlags;
		m_uiPrefixId = pSrcAttrItem->m_uiPrefixId;
		m_ui64QuickVal = pSrcAttrItem->m_ui64QuickVal;
		m_uiEncDefId = pSrcAttrItem->m_uiEncDefId;
		m_uiIVLen = pSrcAttrItem->m_uiIVLen;
		m_uiDecryptedDataLen = pSrcAttrItem->m_uiDecryptedDataLen;
		if( m_uiPayloadLen > sizeof( FLMBYTE *))
		{
			m_pucPayload = NULL;
			m_uiPayloadLen = 0;
			m_uiEncDefId = 0;
		}
	}

	FINLINE FLMBYTE * getAttrDataPtr( void)
	{
		if( m_uiPayloadLen <= sizeof( FLMBYTE *))
		{
			return( (FLMBYTE *)&m_pucPayload);
		}

		return( m_pucPayload + m_uiIVLen);
	}

	FINLINE FLMUINT getAttrDataLength( void)
	{
		if( m_uiEncDefId)
		{
			return( m_uiDecryptedDataLen);
		}
		
		return( m_uiPayloadLen - m_uiIVLen);
	}
	
	FINLINE FLMUINT getAttrDataBufferSize( void)
	{
		return( m_uiPayloadLen - m_uiIVLen);
	}
	
	FINLINE FLMBYTE * getAttrIVPtr( void)
	{
		if( m_uiPayloadLen <= sizeof( FLMBYTE *))
		{
			flmAssert( !m_uiIVLen);
			return( NULL);
		}

		return( m_pucPayload);
	}

	FINLINE FLMBYTE * getAttrPayloadPtr( void)
	{
		if( m_uiPayloadLen <= sizeof( FLMBYTE *))
		{
			return( (FLMBYTE *)&m_pucPayload);
		}

		return( m_pucPayload);
	}
	
	FINLINE FLMUINT getAttrPayloadSize( void)
	{
		return( m_uiPayloadLen);
	}

	FINLINE FLMUINT getAttrModeFlags( void)
	{
		return( m_uiFlags & FDOM_PERSISTENT_FLAGS);
	}

	FINLINE FLMUINT getAttrEncDefId( void)
	{
		return( m_uiEncDefId);
	}

	FINLINE FLMUINT memSize( void)
	{
		FLMUINT	uiSize = gv_XFlmSysData.pNodeCacheMgr->m_pAttrItemAllocator->getCellSize();

		f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);
		
		if( m_uiPayloadLen > sizeof( FLMBYTE *))
		{
			uiSize += gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->getTrueSize( 
				m_uiPayloadLen + sizeof( F_AttrItem *),
				m_pucPayload - sizeof( F_AttrItem *));
		}
		return( uiSize);
	}
			
	FINLINE FLMUINT getAttrStorageFlags( void)
	{
		FLMUINT		uiFlags = 0;
		
		if( m_uiPayloadLen <= ASF_MAX_EMBEDDED_PAYLOAD_LEN)
		{
			uiFlags |= (FLMBYTE)m_uiPayloadLen;
		}
		else
		{
			uiFlags |= ASF_HAVE_PAYLOAD_LEN_SEN;
		}
		
		if( m_uiPayloadLen)
		{
			if( m_uiEncDefId)
			{
				uiFlags |= ASF_ENCRYPTED_BIT;
			}
		}
		
		if( m_uiPrefixId)
		{
			uiFlags |= ASF_HAVE_PREFIX_BIT;
		}
		
		if( m_uiFlags & FDOM_READ_ONLY)
		{
			uiFlags |= ASF_READ_ONLY_BIT;
		}
		
		if( m_uiFlags & FDOM_CANNOT_DELETE)
		{
			uiFlags |= ASF_CANNOT_DELETE_BIT;
		}
		
		return( uiFlags);
	}

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

private:

	F_CachedNode *		m_pCachedNode;
	FLMBYTE *			m_pucPayload;
	FLMUINT				m_uiPayloadLen;
	FLMUINT				m_uiDataType;
	FLMUINT				m_uiNameId;
	FLMUINT				m_uiFlags;
	FLMUINT				m_uiPrefixId;
	FLMUINT64			m_ui64QuickVal;
	FLMUINT				m_uiEncDefId;
	FLMUINT				m_uiIVLen;
	FLMUINT				m_uiDecryptedDataLen;

friend class F_DOMNode;
friend class F_Rfl;
friend class F_Db;
friend class F_DbRebuild;
friend class F_CachedNode;
friend class F_NodeCacheMgr;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
friend class F_NodeInfo;
};

/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT allocOverhead( void)
{
	// Round sizeof( F_CachedNode *) + 1 to nearest 8 byte boundary.
	
	return( (sizeof( F_CachedNode *) + 9) & (~((FLMUINT)7)));
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
FINLINE FLMUINT calcNodeListBufSize(
	FLMUINT	uiNodeCount)
{
	return( uiNodeCount * sizeof( NODE_ITEM) + allocOverhead()); 
}
	
/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT calcAttrListBufSize(
	FLMUINT	uiAttrCount)
{
	return( uiAttrCount * sizeof( F_AttrItem *) + allocOverhead()); 
}
	
/*****************************************************************************
Desc:	
******************************************************************************/
FINLINE FLMUINT calcDataBufSize(
	FLMUINT	uiDataSize)
{
	// Leave room for in buffer for a pointer back to the F_CachedNode.
	// We reserve enough so that the actual allocation will start on
	// the next eight byte boundary.
	
	return( uiDataSize + allocOverhead());
}

/*****************************************************************************
Desc:
******************************************************************************/
typedef struct
{
	FLMUINT64				ui64NodeId;
	FLMUINT64				ui64DocumentId;
	FLMUINT64				ui64ParentId;
	FLMUINT64				ui64MetaValue;
	FLMUINT64				ui64FirstChildId;
	FLMUINT64				ui64LastChildId;
	FLMUINT64				ui64PrevSibId;
	FLMUINT64				ui64NextSibId;
	FLMUINT64				ui64AnnotationId;

	eDomNodeType			eNodeType;
	FLMUINT					uiCollection;
	FLMUINT					uiChildElmCount;
	FLMUINT					uiDataLength;
	FLMUINT					uiDataType;
	FLMUINT					uiPrefixId;
	FLMUINT					uiNameId;
	FLMUINT					uiDataChildCount;
	FLMUINT					uiEncDefId;
} F_NODE_INFO;

/*****************************************************************************
Desc:	Cached DOM NODE
******************************************************************************/
class F_CachedNode : public F_CachedItem
{
public:

	F_CachedNode();
	
	~F_CachedNode();

	// This method assumes that the node cache mutex has been locked.

	FINLINE FLMBOOL canBeFreed( void)
	{
		return( (!nodeInUse() && !readingInNode() && !nodeIsDirty()) 
							? TRUE 
							: FALSE);
	}
	
	// This method assumes that the node cache mutex has been locked.

	FINLINE void freeNode( void)
	{
		if (nodePurged())
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
		
	RCODE getIStream(
		F_Db *						pDb,
		F_NodeBufferIStream *	pStackStream,
		IF_PosIStream **			ppIStream,
		FLMUINT *					puiDataType = NULL,
		FLMUINT *					puiDataLength = NULL);
		
	RCODE headerToBuf(
		FLMBOOL				bFixedSizeHeader,
		FLMBYTE *			pucBuf,
		FLMUINT *			puiHeaderSize,
		XFLM_NODE_INFO *	pNodeInfo,
		F_Db *				pDb);
		
	RCODE readNode(
		F_Db *				pDb,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		IF_IStream *		pIStream,
		FLMUINT				uiOverallLength,
		FLMBYTE *			pucIV);
	
	RCODE getRawIStream(
		F_Db *				pDb,
		IF_PosIStream **	ppIStream);
		
	RCODE openPendingInput(
		F_Db *			pDb,
		FLMUINT			uiNewDataType);
		
	RCODE flushPendingInput(
		F_Db *			pDb,
		FLMBOOL			bLast);
		
	RCODE resizeChildElmList(
		FLMUINT	uiChildElmCount,
		FLMBOOL	bMutexAlreadyLocked);
		
	RCODE resizeAttrList(
		FLMUINT	uiAttrCount,
		FLMBOOL	bMutexAlreadyLocked);
		
	RCODE resizeDataBuffer(
		FLMUINT	uiSize,
		FLMBOOL	bMutexAlreadyLocked);

	FINLINE void setNodeAndDataPtr(
		FLMBYTE *	pucActualAlloc)
	{
		*((F_CachedNode **)(pucActualAlloc)) = this;
		m_pucData = pucActualAlloc + allocOverhead();
	}
	
	FINLINE void setNodeListPtr(
		FLMBYTE *	pucActualAlloc)
	{
		*((F_CachedNode **)(pucActualAlloc)) = this;
		m_pNodeList = (NODE_ITEM *)(pucActualAlloc + allocOverhead());
	}
	
	FINLINE void setAttrListPtr(
		FLMBYTE *	pucActualAlloc)
	{
		*((F_CachedNode **)(pucActualAlloc)) = this;
		m_ppAttrList = (F_AttrItem **)(pucActualAlloc + allocOverhead());
	}

	FINLINE FLMBYTE * getDataPtr( void)
	{
		return( m_pucData);
	}
	
	FINLINE FLMUINT getDataBufSize( void)
	{
		return( m_uiDataBufSize);
	}
			
	FINLINE FLMUINT getModeFlags( void)
	{
		return( m_uiFlags);
	}

	FINLINE FLMUINT getPersistentFlags( void)
	{
		return( m_uiFlags & FDOM_PERSISTENT_FLAGS);
	}

	FINLINE void setFlags(
		FLMUINT	uiFlags)
	{
		m_uiFlags |= uiFlags;
	}

	FINLINE void unsetFlags(
		FLMUINT	uiFlags)
	{
		m_uiFlags &= ~uiFlags;
	}

	FINLINE FLMUINT64 getNodeId( void)
	{
		return( m_nodeInfo.ui64NodeId);
	}

	FINLINE F_Database * getDatabase( void)
	{
		return( m_pDatabase);
	}

	FINLINE FLMUINT getCollection( void)
	{
		return( m_nodeInfo.uiCollection);
	}

	FINLINE eDomNodeType getNodeType( void)
	{
		return( m_nodeInfo.eNodeType);
	}
	
	FINLINE void setNodeType(
		eDomNodeType eNodeType)
	{
		m_nodeInfo.eNodeType = eNodeType;
	}
	
	FINLINE FLMUINT getDataType( void)
	{
		return( m_nodeInfo.uiDataType);
	}
	
	FINLINE void setDataType(
		FLMUINT	uiDataType)
	{
		m_nodeInfo.uiDataType = uiDataType;
	}
	
	FINLINE FLMUINT getNameId( void)
	{
		return( m_nodeInfo.uiNameId);
	}
	
	FINLINE void setNameId(
		FLMUINT	uiNameId)
	{
		m_nodeInfo.uiNameId = uiNameId;
	}
	
	FINLINE FLMUINT getPrefixId( void)
	{
		return( m_nodeInfo.uiPrefixId);
	}

	FINLINE void setPrefixId(
		FLMUINT	uiPrefixId)
	{
		m_nodeInfo.uiPrefixId = uiPrefixId;
	}

	FINLINE FLMUINT64 getDocumentId( void)
	{
		return( m_nodeInfo.ui64DocumentId);
	}
	
	FINLINE FLMBOOL isRootNode( void)
	{
		return( m_nodeInfo.ui64NodeId == m_nodeInfo.ui64DocumentId 
									? TRUE 
									: FALSE);
	}

	FINLINE void setDocumentId(
		FLMUINT64	ui64DocumentId)
	{
		m_nodeInfo.ui64DocumentId = ui64DocumentId;
	}

	FINLINE FLMUINT64 getParentId( void)
	{
		return( m_nodeInfo.ui64ParentId);
	}

	FINLINE void setParentId(
		FLMUINT64	ui64ParentId)
	{
		m_nodeInfo.ui64ParentId = ui64ParentId;
	}

	FINLINE FLMUINT64 getNextSibId( void)
	{
		return( m_nodeInfo.ui64NextSibId);
	}
	
	FINLINE void setNextSibId(
		FLMUINT64	ui64NextSibId)
	{
		m_nodeInfo.ui64NextSibId = ui64NextSibId;
	}

	FINLINE FLMUINT64 getPrevSibId( void)
	{
		return( m_nodeInfo.ui64PrevSibId);
	}
	
	FINLINE void setPrevSibId(
		FLMUINT64	ui64PrevSibId)
	{
		m_nodeInfo.ui64PrevSibId = ui64PrevSibId;
	}

	FINLINE void setSibIds(
		FLMUINT64	ui64PrevSibId,
		FLMUINT64	ui64NextSibId)
	{
		m_nodeInfo.ui64NextSibId = ui64NextSibId;
		m_nodeInfo.ui64PrevSibId = ui64PrevSibId;
	}

	FINLINE FLMUINT64 getFirstChildId( void)
	{
		return( m_nodeInfo.ui64FirstChildId);
	}
	
	FINLINE void setFirstChildId(
		FLMUINT64	ui64FirstChildId)
	{
		m_nodeInfo.ui64FirstChildId = ui64FirstChildId;
	}

	FINLINE FLMUINT64 getLastChildId( void)
	{
		return( m_nodeInfo.ui64LastChildId);
	}

	FINLINE void setLastChildId(
		FLMUINT64	ui64LastChildId)
	{
		m_nodeInfo.ui64LastChildId = ui64LastChildId;
	}

	FINLINE void setChildIds(
		FLMUINT64	ui64FirstChildId,
		FLMUINT64	ui64LastChildId)
	{
		m_nodeInfo.ui64FirstChildId = ui64FirstChildId;
		m_nodeInfo.ui64LastChildId = ui64LastChildId;
	}

	FINLINE FLMUINT getDataChildCount( void)
	{
		return( m_nodeInfo.uiDataChildCount);
	}
	
	FINLINE void setDataChildCount(
		FLMUINT		uiDataChildCount)
	{
		flmAssert( m_nodeInfo.ui64FirstChildId);
		m_nodeInfo.uiDataChildCount = uiDataChildCount;
	}
	
	FINLINE FLMUINT64 getAnnotationId( void)
	{
		return( m_nodeInfo.ui64AnnotationId);
	}

	FINLINE void setAnnotationId(
		FLMUINT64	ui64AnnotationId)
	{
		m_nodeInfo.ui64AnnotationId = ui64AnnotationId;
	}
	
	FINLINE FLMBOOL hasAttributes( void)
	{
		return( m_uiAttrCount ? TRUE : FALSE);
	}
	
	FINLINE FLMUINT getChildElmCount( void)
	{
		flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
		return( m_nodeInfo.uiChildElmCount);
	}
	
	FINLINE FLMUINT getChildElmNameId(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
		return( m_pNodeList [ uiChildElmOffset].uiNameId);
	}

	FINLINE FLMUINT64 getChildElmNodeId(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
		return( m_pNodeList [ uiChildElmOffset].ui64NodeId);
	}

	FLMBOOL findChildElm(
		FLMUINT		uiChildElmNameId,
		FLMUINT *	puiInsertPos);
	
	FINLINE RCODE removeChildElm(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
		if (uiChildElmOffset < m_nodeInfo.uiChildElmCount - 1)
		{
			f_memmove( &m_pNodeList [ uiChildElmOffset], 
				&m_pNodeList [ uiChildElmOffset + 1],
				sizeof( NODE_ITEM) * (m_nodeInfo.uiChildElmCount - 
					uiChildElmOffset - 1));
		}
		return( resizeChildElmList( m_nodeInfo.uiChildElmCount - 1, FALSE));
	}
	
	RCODE insertChildElm(
		FLMUINT		uiChildElmOffset,
		FLMUINT		uiChildElmNameId,
		FLMUINT64	ui64ChildElmNodeId);
		
	FINLINE FLMUINT getDataLength( void)
	{
		return( m_nodeInfo.uiDataLength);
	}

	FINLINE void setDataLength(
		FLMUINT	uiDataLength)
	{
		m_nodeInfo.uiDataLength = uiDataLength;
	}

	FINLINE FLMUINT getEncDefId( void)
	{
		return( m_nodeInfo.uiEncDefId);
	}

	FINLINE void setEncDefId(
		FLMUINT	uiEncDefId)
	{
		m_nodeInfo.uiEncDefId = uiEncDefId;
	}
	
	FINLINE FLMBOOL getQuickNumber64(
		FLMUINT64 *	pui64Num,
		FLMBOOL *	pbNeg)
	{
		if (m_uiFlags & FDOM_UNSIGNED_QUICK_VAL)
		{
			*pui64Num = m_numberVal.ui64Val;
			*pbNeg = FALSE;
			return( TRUE);
		}

		if (m_uiFlags & FDOM_SIGNED_QUICK_VAL)
		{
			if( m_numberVal.i64Val < 0)
			{
				*pui64Num = (FLMUINT64)(-m_numberVal.i64Val);
				*pbNeg = TRUE;
			}
			else
			{
				*pui64Num = (FLMUINT64)m_numberVal.i64Val;
				*pbNeg = FALSE;
			}

			return( TRUE);
		}

		*pui64Num = 0;
		*pbNeg = FALSE;

		return( FALSE);
	}
	
	FINLINE FLMINT64 getQuickINT64( void)
	{
		flmAssert( m_uiFlags & FDOM_SIGNED_QUICK_VAL);
		return( m_numberVal.i64Val);
	}

	FINLINE FLMUINT64 getQuickUINT64( void)
	{
		flmAssert( m_uiFlags & FDOM_UNSIGNED_QUICK_VAL);
		return( m_numberVal.ui64Val);
	}

	FINLINE void setUINT64(
		FLMUINT64	ui64Value)
	{
		m_numberVal.ui64Val = ui64Value;
		m_uiFlags &= ~FDOM_SIGNED_QUICK_VAL;
		m_uiFlags |= FDOM_UNSIGNED_QUICK_VAL;
	}

	FINLINE void setINT64(
		FLMINT64	i64Value)
	{
		m_numberVal.i64Val = i64Value;
		m_uiFlags &= ~FDOM_UNSIGNED_QUICK_VAL;
		m_uiFlags |= FDOM_SIGNED_QUICK_VAL;
	}
	
	FINLINE void setMetaValue(
		FLMUINT64		ui64Value)
	{
		m_nodeInfo.ui64MetaValue = ui64Value;
	}

	FINLINE FLMUINT64 getMetaValue( void)
	{
		return( m_nodeInfo.ui64MetaValue); 
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
	
	// Generally, assumes that the node cache mutex has already been locked.
	// There is one case where it is not locked, but it is not
	// critical that it be locked - inside syncFromDb.

	FINLINE FLMBOOL nodePurged( void)
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

	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void incrNodeUseCount( void)
	{
		m_uiCacheFlags = (m_uiCacheFlags & (~(NCA_COUNTER_BITS))) |
						 (((m_uiCacheFlags & NCA_COUNTER_BITS) + 1));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void decrNodeUseCount( void)
	{
		m_uiCacheFlags = (m_uiCacheFlags & (~(NCA_COUNTER_BITS))) |
						 (((m_uiCacheFlags & NCA_COUNTER_BITS) - 1));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void incrStreamUseCount( void)
	{
		m_uiStreamUseCount++;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void decrStreamUseCount( void)
	{
		flmAssert( m_uiStreamUseCount);
		m_uiStreamUseCount--;
	}
	
	FINLINE FLMUINT getStreamUseCount( void)
	{
		return( m_uiStreamUseCount);
	}
	
	void setNodeDirty(
		F_Db *		pDb,
		FLMBOOL		bNew);
	
	void unsetNodeDirtyAndNew(
		F_Db *			pDb,
		FLMBOOL			bMutexAlreadyLocked = FALSE);
	
	FINLINE FLMBOOL nodeIsDirty( void)
	{
		return( (m_uiFlags & FDOM_DIRTY) ? TRUE : FALSE);
	}
		
	FINLINE FLMBOOL nodeIsNew( void)
	{
		return( (m_uiFlags & FDOM_NEW) ? TRUE : FALSE);
	}

	// Assumes that the node cache mutex has already been locked.
	
	FINLINE FLMBOOL nodeUncommitted( void)
	{
		return( (m_uiCacheFlags & NCA_UNCOMMITTED ) ? TRUE : FALSE);
	}
	
	void freeCache(
		FLMBOOL	bPutInPurgeList);
		
	// Attribute functions
		
	RCODE setNumber64(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		FLMUINT64				ui64Value,
		FLMBOOL					bNeg,
		FLMUINT					uiEncDefId);
		
	RCODE getNumber64(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		FLMUINT64 *				pui64Value,
		FLMBOOL *				pbNeg);

	RCODE setUTF8(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		const void *			pvValue,
		FLMUINT					uiNumBytesInValue,
		FLMUINT					uiNumCharsInValue,
		FLMUINT					uiEncDefId);
	
	RCODE setBinary(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		const void *			pvValue,
		FLMUINT					uiNumBytesInBuffer,
		FLMUINT					uiEncDefId);

	RCODE getBinary(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		void *					pvBuffer,
		FLMUINT					uiBufferLen,
		FLMUINT *				puiDataLen);
		
	RCODE setStorageValue(
		F_Db *					pDb,
		FLMUINT					uiAttrNameId,
		const void *			pvValue,
		FLMUINT					uiValueLen,
		FLMUINT					uiEncDefId);

	F_AttrItem * getAttribute(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiInsertPos);

	FINLINE F_AttrItem * getFirstAttribute( void)
	{
		return( m_uiAttrCount ? m_ppAttrList [0] : NULL);
	}

	FINLINE F_AttrItem * getLastAttribute( void)
	{
		return( m_uiAttrCount ? m_ppAttrList [m_uiAttrCount - 1] : NULL);
	}
	
	RCODE getPrevSiblingNode(
		FLMUINT					uiCurrentNameId,
		IF_DOMNode **			ppSib);
		
	RCODE getNextSiblingNode(
		FLMUINT					uiCurrentNameId,
		IF_DOMNode **			ppSib);

	RCODE allocAttribute(
		F_Db *			pDb,
		FLMUINT			uiAttrNameId,
		F_AttrItem *	pSrcAttrItem,
		FLMUINT			uiInsertPos,
		F_AttrItem **	ppAttrItem,
		FLMBOOL			bMutexAlreadyLocked);
	
	RCODE freeAttribute(
		F_AttrItem *	pAttrItem,
		FLMUINT			uiPos);
		
	RCODE createAttribute(
		F_Db *			pDb,
		FLMUINT			uiAttrNameId,
		F_AttrItem **	ppAttrItem);
		
	RCODE removeAttribute(
		F_Db *	pDb,
		FLMUINT	uiAttrNameId);
		
	RCODE exportAttributeList(
		F_Db *					pDb,
		F_DynaBuf *				pDynaBuf,
		XFLM_NODE_INFO *		pNodeInfo);

	RCODE addModeFlags(
		F_Db *	pDb,
		FLMUINT	uiAttrNameId,
		FLMUINT	uiFlags);

	RCODE removeModeFlags(
		F_Db *	pDb,
		FLMUINT	uiAttrNameId,
		FLMUINT	uiFlags);

	RCODE setPrefixId(
		F_Db *	pDb,
		FLMUINT	uiAttrNameId,
		FLMUINT	uiPrefixId);
	
	FINLINE F_AttrItem * getPrevSibling(
		FLMUINT	uiAttrNameId)
	{
		F_AttrItem *	pAttrItem;
		FLMUINT			uiPos;
		
		if ((pAttrItem = getAttribute( uiAttrNameId, &uiPos)) != NULL)
		{
			pAttrItem = uiPos ? m_ppAttrList [uiPos - 1] : NULL;
		}
		
		return( pAttrItem);
	}

	FINLINE F_AttrItem * getNextSibling(
		FLMUINT	uiAttrNameId)
	{
		F_AttrItem *	pAttrItem;
		FLMUINT			uiPos;
		
		if( (pAttrItem = getAttribute( uiAttrNameId, &uiPos)) != NULL)
		{
			pAttrItem = uiPos < m_uiAttrCount - 1 ? m_ppAttrList [uiPos + 1] : NULL;
		}
		
		return( pAttrItem);
	}

	FINLINE FLMUINT getModeFlags(
		FLMUINT	uiAttrNameId)
	{
		F_AttrItem *	pAttrItem;
		
		if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
		{
			return( 0);
		}
		
		return( pAttrItem->getAttrModeFlags());
	}

	FINLINE RCODE getPrefixId(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiPrefixId)
	{
		F_AttrItem *	pAttrItem;
	
		if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
	
		*puiPrefixId = pAttrItem->m_uiPrefixId;
		return( NE_XFLM_OK);
	}

	FINLINE RCODE getDataLength(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiDataLength)
	{
		F_AttrItem *	pAttrItem;
	
		if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
		{
			return( RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
		
		*puiDataLength = pAttrItem->getAttrDataLength();
		return( NE_XFLM_OK);
	}

	FINLINE RCODE getDataType(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiDataType)
	{
		F_AttrItem *	pAttrItem;
	
		if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
	
		*puiDataType = pAttrItem->m_uiDataType;
		return( NE_XFLM_OK);
	}

	FINLINE RCODE getEncDefId(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiEncDefId)
	{
		F_AttrItem *	pAttrItem;
	
		if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
	
		*puiEncDefId = pAttrItem->m_uiEncDefId;
		return( NE_XFLM_OK);
	}

	RCODE getIStream(
		F_Db *						pDb,
		FLMUINT						uiAttrNameId,
		F_NodeBufferIStream *	pStackStream,
		IF_PosIStream **			ppIStream,
		FLMUINT *					puiDataType = NULL,
		FLMUINT *					puiDataLength = NULL);

private:

	// Assumes that the node cache mutex has already been locked.
	
	FINLINE FLMBOOL readingInNode( void)
	{
		return( (m_uiCacheFlags & NCA_READING_IN ) ? TRUE : FALSE);
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void setReadingIn( void)
	{
		m_uiCacheFlags |= NCA_READING_IN;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unsetReadingIn( void)
	{
		m_uiCacheFlags &= (~(NCA_READING_IN));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void setUncommitted( void)
	{
		m_uiCacheFlags |= NCA_UNCOMMITTED;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unsetUncommitted( void)
	{
		m_uiCacheFlags &= (~(NCA_UNCOMMITTED));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE FLMBOOL nodeIsLatestVer( void)
	{
		return( (m_uiCacheFlags & NCA_LATEST_VER ) ? TRUE : FALSE);
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void setLatestVer( void)
	{
		m_uiCacheFlags |= NCA_LATEST_VER;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unsetLatestVer( void)
	{
		m_uiCacheFlags &= (~(NCA_LATEST_VER));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void setPurged( void)
	{
		m_uiCacheFlags |= NCA_PURGED;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unsetPurged( void)
	{
		m_uiCacheFlags &= (~(NCA_PURGED));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE FLMBOOL nodeLinkedToDatabase( void)
	{
		return( (m_uiCacheFlags & NCA_LINKED_TO_DATABASE ) ? TRUE : FALSE);
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void setLinkedToDatabase( void)
	{
		m_uiCacheFlags |= NCA_LINKED_TO_DATABASE;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unsetLinkedToDatabase( void)
	{
		m_uiCacheFlags &= (~(NCA_LINKED_TO_DATABASE));
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE FLMBOOL nodeInUse( void)
	{
		return( (m_uiCacheFlags & NCA_COUNTER_BITS ) ? TRUE : FALSE);
	}
	
	// Unlink a node from the global purged list.
	// Assumes that the node cache mutex has already been locked.
	
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
			gv_XFlmSysData.pNodeCacheMgr->m_pPurgeList = 
				(F_CachedNode *)m_pNextInGlobal;
		}

		m_pPrevInGlobal = NULL;
		m_pNextInGlobal = NULL;
	}
	
	// Link a node to an F_Database list at the head of the list.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToDatabaseAtHead(
		F_Database *	pDatabase)
	{
		if (!pDatabase->m_pLastDirtyNode || nodeIsDirty())
		{
			m_pPrevInDatabase = NULL;
			if ((m_pNextInDatabase = pDatabase->m_pFirstNode) != NULL)
			{
				pDatabase->m_pFirstNode->m_pPrevInDatabase = this;
			}
			else
			{
				pDatabase->m_pLastNode = this;
			}
		
			pDatabase->m_pFirstNode = this;
			if (nodeIsDirty() && !pDatabase->m_pLastDirtyNode)
			{
				pDatabase->m_pLastDirtyNode = this;
			}
		}
		else
		{
			// pDatabase->m_pLastDirtyNode is guaranteed to be non-NULL,
			// Hence, m_pPrevInDatabase will be non-NULL.
			// We are also guaranteed that the node is not dirty.
			
			m_pPrevInDatabase = pDatabase->m_pLastDirtyNode;
			m_pNextInDatabase = m_pPrevInDatabase->m_pNextInDatabase;

			m_pPrevInDatabase->m_pNextInDatabase = this;
			if (m_pNextInDatabase)
			{
				m_pNextInDatabase->m_pPrevInDatabase = this;
			}
			else
			{
				pDatabase->m_pLastNode = this;
			}
		}

		m_pDatabase = pDatabase;
		setLinkedToDatabase();
	}
	
	// Link a node to an F_Database list at the end of the list.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToDatabaseAtEnd(
		F_Database *	pDatabase)
	{
		// Node cannot be a dirty node.
		
		flmAssert( !nodeIsDirty());
		m_pNextInDatabase = NULL;
		if( (m_pPrevInDatabase = pDatabase->m_pLastNode) != NULL)
		{
			pDatabase->m_pLastNode->m_pNextInDatabase = this;
		}
		else
		{
			pDatabase->m_pFirstNode = this;
		}

		pDatabase->m_pLastNode = this;
		m_pDatabase = pDatabase;
		setLinkedToDatabase();
	}
	
	// Unlink a node from its F_Database list.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unlinkFromDatabase( void)
	{
		if( nodeLinkedToDatabase())
		{
			// If this is the last dirty node, change the database's
			// last dirty pointer to point to the previous node, if any.
			
			if (m_pDatabase->m_pLastDirtyNode == this)
			{
				flmAssert( nodeIsDirty());
				m_pDatabase->m_pLastDirtyNode = m_pPrevInDatabase;
			}
			
			// Remove the node from the database's list.
			
			if( m_pNextInDatabase)
			{
				m_pNextInDatabase->m_pPrevInDatabase = m_pPrevInDatabase;
			}
			else
			{
				m_pDatabase->m_pLastNode = m_pPrevInDatabase;
			}

			if( m_pPrevInDatabase)
			{
				m_pPrevInDatabase->m_pNextInDatabase = m_pNextInDatabase;
			}
			else
			{
				m_pDatabase->m_pFirstNode = m_pNextInDatabase;
			}

			m_pPrevInDatabase = NULL;
			m_pNextInDatabase = NULL;
			m_pDatabase = NULL;
			unsetLinkedToDatabase();
		}
	}
	
	// Link a node into its hash bucket.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToHashBucket( void)
	{
		F_CachedNode ** ppHashBucket = gv_XFlmSysData.pNodeCacheMgr->nodeHash( 
													m_nodeInfo.ui64NodeId);
		
		flmAssert( m_pNewerVersion == NULL);
	
		m_pPrevInBucket = NULL;

		if ((m_pNextInBucket = *ppHashBucket) != NULL)
		{
			m_pNextInBucket->m_pPrevInBucket = this;
		}

		*ppHashBucket = this;
	}
	
	// Unlink a node from its hash bucket.
	// Assumes that the node cache mutex has already been locked.
	
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
			F_CachedNode ** ppHashBucket =
									gv_XFlmSysData.pNodeCacheMgr->nodeHash( 
											m_nodeInfo.ui64NodeId);
			
			*ppHashBucket = m_pNextInBucket;
		}

		m_pPrevInBucket = NULL;
		m_pNextInBucket = NULL;
	}
	
	// Unlink a node from its version list.
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToVerList(
		F_CachedNode *	pNewerVer,
		F_CachedNode *	pOlderVer)
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
	
	// Unlink a node from its version list.  This routine
	// Assumes that the node cache mutex has already been locked.
	
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

	// Link a node into the heap list
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToHeapList( void)
	{
		flmAssert( !m_pPrevInHeapList);
		flmAssert( (m_uiFlags & FDOM_HEAP_ALLOC) == 0);
	
		if( (m_pNextInHeapList = 
			gv_XFlmSysData.pNodeCacheMgr->m_pHeapList) != NULL)
		{
			gv_XFlmSysData.pNodeCacheMgr->m_pHeapList->m_pPrevInHeapList = this;
		}
		
		gv_XFlmSysData.pNodeCacheMgr->m_pHeapList = this;
		m_uiFlags |= FDOM_HEAP_ALLOC;
	}
	
	// Unlink a node from the heap list
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void unlinkFromHeapList( void)
	{
		flmAssert( m_uiFlags & FDOM_HEAP_ALLOC);
	
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
			gv_XFlmSysData.pNodeCacheMgr->m_pHeapList = m_pNextInHeapList;
		}
		
		m_pPrevInHeapList = NULL;
		m_pNextInHeapList = NULL;
		m_uiFlags &= ~FDOM_HEAP_ALLOC;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
	FINLINE void linkToOldList( void)
	{
		flmAssert( !m_pPrevInOldList);
	
		if( (m_pNextInOldList = 
			gv_XFlmSysData.pNodeCacheMgr->m_pOldList) != NULL)
		{
			gv_XFlmSysData.pNodeCacheMgr->m_pOldList->m_pPrevInOldList = this;
		}
		
		gv_XFlmSysData.pNodeCacheMgr->m_pOldList = this;
	}
	
	// Assumes that the node cache mutex has already been locked.
	
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
			gv_XFlmSysData.pNodeCacheMgr->m_pOldList = m_pNextInOldList;
		}
		
		m_pPrevInOldList = NULL;
		m_pNextInOldList = NULL;
	}
	
	FINLINE FLMUINT memSize( void)
	{
		FLMUINT	uiSize = gv_XFlmSysData.pNodeCacheMgr->m_pNodeAllocator->getCellSize(); 
				
		f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);
		
		if (m_pucData)
		{
			uiSize += gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->getTrueSize( 
				m_uiDataBufSize, getActualPointer( m_pucData));
		}

		if( m_pNodeList)
		{
			uiSize += gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->getTrueSize(
				calcNodeListBufSize( m_nodeInfo.uiChildElmCount),
				getActualPointer( m_pNodeList));
		}

		if( m_ppAttrList)
		{
			uiSize += gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->getTrueSize(
				calcAttrListBufSize( m_uiAttrCount),
				getActualPointer( m_ppAttrList));
		}

		return( uiSize + m_uiTotalAttrSize);
	}
	
	// Assumes that the node cache mutex is locked, because
	// it potentially updates the cache usage statistics.

	FINLINE void setTransID(
		FLMUINT64		ui64NewTransID)
	{
		FLMUINT	uiSize;
		
		if (m_ui64HighTransId == FLM_MAX_UINT64 &&
			 ui64NewTransID != FLM_MAX_UINT64)
		{
			uiSize = memSize();
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiSize;
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount++;
			linkToOldList();
		}
		else if (m_ui64HighTransId != FLM_MAX_UINT64 &&
					ui64NewTransID == FLM_MAX_UINT64)
		{
			uiSize = memSize();
			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount);
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount--;
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
		
	RCODE importAttributeList(
		F_Db *					pDb,
		IF_IStream *			pIStream,
		FLMBOOL					bMutexAlreadyLocked);
	
	RCODE importAttributeList(
		F_Db *					pDb,
		F_CachedNode *			pSourceNode,
		FLMBOOL					bMutexAlreadyLocked);
		
	void resetNode( void);
		
	typedef union
	{
		FLMUINT64			ui64Val;
		FLMINT64				i64Val;
	} FLMNUMBER;

	// Things that manage the node's place in node cache.  These are
	// always initialized in the constructor
	
	F_CachedNode *			m_pPrevInBucket;
	F_CachedNode *			m_pNextInBucket;
	F_CachedNode *			m_pPrevInDatabase;
	F_CachedNode *			m_pNextInDatabase;
	F_CachedNode *			m_pOlderVersion;
	F_CachedNode *			m_pNewerVersion;
	F_CachedNode *			m_pPrevInHeapList;
	F_CachedNode *			m_pNextInHeapList;
	F_CachedNode *			m_pPrevInOldList;
	F_CachedNode *			m_pNextInOldList;
	
	FLMUINT64				m_ui64LowTransId;
	FLMUINT64				m_ui64HighTransId;
	F_NOTIFY_LIST_ITEM *	m_pNotifyList;
	FLMUINT					m_uiCacheFlags;
	FLMUINT					m_uiStreamUseCount;
	
	// Things we hash on - initialized by caller of constructor
	
	F_Database *			m_pDatabase;
	
	// Items initialized in constructor
	
	F_NODE_INFO				m_nodeInfo;
	FLMUINT					m_uiFlags;
	FLMBYTE *				m_pucData;
	FLMUINT					m_uiDataBufSize;
	NODE_ITEM *				m_pNodeList;	
	F_AttrItem **			m_ppAttrList;
	FLMUINT					m_uiAttrCount;
	FLMUINT					m_uiTotalAttrSize;
	
	// Items initialized by caller of constructor, but not
	// in constructor - for performance reasons - so we don't
	// end up setting them twice.
	
	FLMUINT					m_uiOffsetIndex;
	FLMUINT32				m_ui32BlkAddr;
	
	// Items that m_uiFlags indicates whether they are present

	FLMNUMBER				m_numberVal;				// Valid only if FDOM_SIGNED_QUICK_VAL
																// or FDOM_UNSIGNED_QUICK_VAL is set on m_uiFlags
																
friend class F_NodeCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_Database;
friend class F_Db;
friend class F_DbSystem;
friend class F_BTreeIStream;
friend class F_LocalNodeCache;
friend class F_DOMNode;
friend class F_Rfl;
friend class F_NodeInfo;
friend class F_RebuildNodeIStream;
friend class F_DbRebuild;
friend class F_AttrItem;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
};

RCODE flmReadNodeInfo(
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	IF_IStream *		pIStream,
	FLMUINT				uiOverallLength,
	FLMBOOL				bAssertOnCorruption,
	F_NODE_INFO *		pNodeInfo,
	FLMUINT *			puiStorageFlags,
	FLMBOOL *			pbFixedSizeHeader = NULL);
	
#endif // FCACHE_H
