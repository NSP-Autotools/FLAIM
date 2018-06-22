//------------------------------------------------------------------------------
// Desc:	Block Cache routines
// Tabs:	3
//
// Copyright (c) 1997-2007 Novell, Inc. All Rights Reserved.
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

#define MAX_BLOCKS_TO_SORT			500
#define FLM_MAX_IO_BUFFER_BLOCKS 16

FSTATIC void ScaNotify(
	FNOTIFY *			pNotify,
	F_CachedBlock *	pUseSCache,
	RCODE					NotifyRc);

FSTATIC void SQFAPI scaWriteComplete(
	IF_IOBuffer *		pIOBuffer,
	void *				pvData);
	
#ifdef SCACHE_LINK_CHECKING
FSTATIC void scaVerify(
	int				iPlace);
#else
#define scaVerify(iPlace)
#endif

/***************************************************************************
Desc:	Compare two cache blocks to determine which one has lower address.
*****************************************************************************/
FINLINE FLMINT scaCompare(
	F_CachedBlock *	pSCache1,
	F_CachedBlock *	pSCache2)
{
	if (FSAddrIsAtOrBelow( pSCache1->blkAddress(), pSCache2->blkAddress()))
	{
		flmAssert( pSCache1->blkAddress() != pSCache2->blkAddress());
		return( -1);
	}
	else
	{
		return( 1);
	}
}

/***************************************************************************
Desc:	Compare two cache blocks during a sort to determine which 
		one has lower address.
*****************************************************************************/
FINLINE FLMINT SQFAPI scaSortCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	return( scaCompare( ((F_CachedBlock **)pvBuffer)[ uiPos1], 
							  ((F_CachedBlock **)pvBuffer)[ uiPos2]));
}

/***************************************************************************
Desc:	Swap two entries in cache table during sort.
*****************************************************************************/
FINLINE void SQFAPI scaSortSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	F_CachedBlock **	ppSCacheTbl = (F_CachedBlock **)pvBuffer;
	F_CachedBlock *	pTmpSCache = ppSCacheTbl[ uiPos1];

	ppSCacheTbl[ uiPos1] = ppSCacheTbl[ uiPos2];
	ppSCacheTbl[ uiPos2] = pTmpSCache;
}

/****************************************************************************
Desc:	Link a cache block into the list of F_Database blocks that need one or
		more versions of the block to be logged.  This routine assumes that
		the block cache mutex is locked.
*****************************************************************************/
void F_CachedBlock::linkToLogList( void)
{
	FLMUINT		uiPrevBlkAddress;
	
	flmAssert( m_pDatabase);
	flmAssert( m_ui16Flags & CA_DIRTY);
	flmAssert( !(m_ui16Flags & (CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	flmAssert( !m_pPrevInReplaceList);
	flmAssert( !m_pNextInReplaceList);

	uiPrevBlkAddress = getPriorImageAddress();
	if( uiPrevBlkAddress || !m_pNextInVersionList)
	{
		goto Exit;
	}

	if ((m_pNextInReplaceList = m_pDatabase->m_pFirstInLogList) != NULL)
	{
		m_pNextInReplaceList->m_pPrevInReplaceList = this;
	}
	else
	{
		m_pDatabase->m_pLastInLogList = this;
	}

	setFlags( CA_IN_FILE_LOG_LIST);
	m_pPrevInReplaceList = NULL;
	m_pDatabase->m_pFirstInLogList = this;
	m_pDatabase->m_uiLogListCount++;

Exit:

	return;
}

/****************************************************************************
Desc:	Unlinks a cache block from the F_Database's log list
		NOTE: This function assumes that the block cache mutex is locked.
****************************************************************************/
void F_CachedBlock::unlinkFromLogList( void)
{
	flmAssert( m_pDatabase);
	flmAssert( m_ui16Flags & CA_IN_FILE_LOG_LIST);
	flmAssert( m_pDatabase->m_uiLogListCount);

	if (m_pNextInReplaceList)
	{
		m_pNextInReplaceList->m_pPrevInReplaceList = m_pPrevInReplaceList;
	}
	else
	{
		m_pDatabase->m_pLastInLogList = m_pPrevInReplaceList;
	}

	if (m_pPrevInReplaceList)
	{
		m_pPrevInReplaceList->m_pNextInReplaceList = m_pNextInReplaceList;
	}
	else
	{
		m_pDatabase->m_pFirstInLogList = m_pNextInReplaceList;
	}

	m_pNextInReplaceList = NULL;
	m_pPrevInReplaceList = NULL;
	clearFlags( CA_IN_FILE_LOG_LIST);
	m_pDatabase->m_uiLogListCount--;
}

/****************************************************************************
Desc:	Link a cache block into the list of F_Database blocks that are beyond
		the EOF.  The blocks are linked to the end of the list so that they
		are kept in ascending order.
		NOTE: This function assumes that the block cache mutex is locked.
*****************************************************************************/
void F_CachedBlock::linkToNewList( void)
{
	flmAssert( m_pDatabase);

	flmAssert( m_ui64HighTransID == ~((FLMUINT64)0));
	flmAssert( m_ui16Flags & CA_DIRTY);
	flmAssert( !(m_ui16Flags & (CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	flmAssert( !m_pPrevInReplaceList);
	flmAssert( !m_pNextInReplaceList);

	if ((m_pPrevInReplaceList = m_pDatabase->m_pLastInNewList) != NULL)
	{
		flmAssert( scaCompare( m_pDatabase->m_pLastInNewList, this) < 0);

		m_pPrevInReplaceList->m_pNextInReplaceList = this;
	}
	else
	{
		m_pDatabase->m_pFirstInNewList = this;
	}

	m_pNextInReplaceList = NULL;
	m_pDatabase->m_pLastInNewList = this;
	setFlags( CA_IN_NEW_LIST);
	m_pDatabase->m_uiNewCount++;
}

/****************************************************************************
Desc:	Unlinks a cache block from the F_Database's new block list
		NOTE: This function assumes that the block cache mutex is locked.
****************************************************************************/
void F_CachedBlock::unlinkFromNewList( void)
{
	flmAssert( m_pDatabase);

	flmAssert( m_ui16Flags & CA_IN_NEW_LIST);
	flmAssert( m_pDatabase->m_uiNewCount);

	if (m_pNextInReplaceList)
	{
		m_pNextInReplaceList->m_pPrevInReplaceList = m_pPrevInReplaceList;
	}
	else
	{
		m_pDatabase->m_pLastInNewList = m_pPrevInReplaceList;
	}

	if (m_pPrevInReplaceList)
	{
		m_pPrevInReplaceList->m_pNextInReplaceList = m_pNextInReplaceList;
	}
	else
	{
		m_pDatabase->m_pFirstInNewList = m_pNextInReplaceList;
	}

	m_pNextInReplaceList = NULL;
	m_pPrevInReplaceList = NULL;
	clearFlags( CA_IN_NEW_LIST);
	m_pDatabase->m_uiNewCount--;
}

/****************************************************************************
Desc:	Unlinks a cache block from the replace list
		NOTE: This function assumes that the block cache mutex is locked.
****************************************************************************/
void F_CachedBlock::unlinkFromReplaceList( void)
{
	FLMUINT	uiSize = memSize();

	flmAssert( !m_ui16Flags);

	if( m_pNextInReplaceList)
	{
		m_pNextInReplaceList->m_pPrevInReplaceList = m_pPrevInReplaceList;
	}
	else
	{
		gv_SFlmSysData.pBlockCacheMgr->m_pLRUReplace = m_pPrevInReplaceList;
	}

	if( m_pPrevInReplaceList)
	{
		m_pPrevInReplaceList->m_pNextInReplaceList = m_pNextInReplaceList;
	}
	else
	{
		gv_SFlmSysData.pBlockCacheMgr->m_pMRUReplace = m_pNextInReplaceList;
	}

	m_pNextInReplaceList = NULL;
	m_pPrevInReplaceList = NULL;

	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableCount);
	gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableCount--;
	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes >= uiSize);
	gv_SFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes -= uiSize;
}

/****************************************************************************
Desc:	Check hash links.
		This routine assumes that the block cache mutex has already been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
void F_CachedBlock::checkHashLinks(
	F_CachedBlock **	ppSCacheBucket)
{
	F_CachedBlock *	pBlock;

	if (!m_pDatabase)
	{
		f_breakpoint( 1);
	}

	if (m_pPrevInVersionList)
	{
		f_breakpoint( 2);
	}

	if (m_pNextInVersionList == this)
	{
		f_breakpoint( 3);
	}

	if (m_pPrevInVersionList == this)
	{
		f_breakpoint( 4);
	}

	// Make sure that the block isn't added into the list a second time.

	for (pBlock = *ppSCacheBucket;
			pBlock;
			pBlock = pBlock->m_pNextInHashBucket)
	{
		if (this == pBlock)
		{
			f_breakpoint( 5);
		}
	}

	// Make sure the block is not in the transaction
	// log list.

	for (pBlock = m_pDatabase->getTransLogList();
			pBlock;
			pBlock = pBlock->m_pNextInHashBucket)
	{
		if (this == pBlock)
		{
			f_breakpoint( 6);
		}
	}
}
#endif

/****************************************************************************
Desc:	Unlink a cache block from its hash bucket.  This routine assumes
		that the block cache mutex has already been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
void F_CachedBlock::checkHashUnlinks(
	F_CachedBlock **	ppSCacheBucket)
{
	F_CachedBlock *	pTmpSCache;

	// Make sure the cache is actually in this bucket

	pTmpSCache = *ppSCacheBucket;
	while (pTmpSCache && pTmpSCache != this)
	{
		pTmpSCache = pTmpSCache->m_pNextInHashBucket;
	}

	if (!pTmpSCache)
	{
		f_breakpoint( 333);
	}

	for (pTmpSCache = m_pDatabase->getTransLogList();
			pTmpSCache;
			pTmpSCache = pTmpSCache->m_pNextInHashBucket)
	{
		if (this == pTmpSCache)
		{
			f_breakpoint( 334);
		}
	}
}
#endif

/****************************************************************************
Desc:	Link a cache block to its F_Database structure.  This routine assumes
		that the block cache mutex has already been locked.
****************************************************************************/
void F_CachedBlock::linkToDatabase(
	F_Database *	pDatabase)
{
	flmAssert( !m_pDatabase);

	if (m_ui16Flags & CA_WRITE_PENDING)
	{
		if ((m_pNextInDatabase = pDatabase->m_pPendingWriteList) != NULL)
		{
			m_pNextInDatabase->m_pPrevInDatabase = this;
		}

		pDatabase->m_pPendingWriteList = this;
		setFlags( CA_IN_WRITE_PENDING_LIST);
	}
	else
	{
		F_CachedBlock *	pPrevSCache;
		F_CachedBlock *	pNextSCache;

		// Link at end of dirty blocks.

		if (pDatabase->m_pLastDirtyBlk)
		{
			pPrevSCache = pDatabase->m_pLastDirtyBlk;
			pNextSCache = pPrevSCache->m_pNextInDatabase;
		}
		else
		{
			// No dirty blocks, so link to head of list.

			pPrevSCache = NULL;
			pNextSCache = pDatabase->m_pSCacheList;
		}

		// If the block is dirty, change the last dirty block pointer.

		if (m_ui16Flags & CA_DIRTY)
		{
			pDatabase->m_pLastDirtyBlk = this;
		}

		if ((m_pNextInDatabase = pNextSCache) != NULL)
		{
			pNextSCache->m_pPrevInDatabase = this;
		}
	
		if ((m_pPrevInDatabase = pPrevSCache) != NULL)
		{
			pPrevSCache->m_pNextInDatabase = this;
		}
		else
		{
			pDatabase->m_pSCacheList = this;
		}
	}

	m_pDatabase = pDatabase;
}

/****************************************************************************
Desc:	Unlink a cache block from its F_Database object.  This routine assumes
		that the block cache mutex has already been locked.
****************************************************************************/
void F_CachedBlock::unlinkFromDatabase( void)
{
	flmAssert( m_pDatabase);

	if (m_ui16Flags & CA_IN_WRITE_PENDING_LIST)
	{
		if (m_pPrevInDatabase)
		{
			m_pPrevInDatabase->m_pNextInDatabase = m_pNextInDatabase;
		}
		else
		{
			m_pDatabase->m_pPendingWriteList = m_pNextInDatabase;
		}

		if (m_pNextInDatabase)
		{
			m_pNextInDatabase->m_pPrevInDatabase = m_pPrevInDatabase;
		}

		clearFlags( CA_IN_WRITE_PENDING_LIST);
	}
	else
	{
		if (this == m_pDatabase->m_pLastDirtyBlk)
		{
			m_pDatabase->m_pLastDirtyBlk = m_pDatabase->m_pLastDirtyBlk->m_pPrevInDatabase;
#ifdef FLM_DEBUG

			// If m_pLastDirtyBlk is non-NULL, it had better be pointing
			// to a dirty block.

			if (m_pDatabase->m_pLastDirtyBlk)
			{
				flmAssert( m_pDatabase->m_pLastDirtyBlk->m_ui16Flags & CA_DIRTY);
			}
#endif
		}

		if (m_pNextInDatabase)
		{
			m_pNextInDatabase->m_pPrevInDatabase = m_pPrevInDatabase;
		}

		if (m_pPrevInDatabase)
		{
			m_pPrevInDatabase->m_pNextInDatabase = m_pNextInDatabase;
		}
		else
		{
			m_pDatabase->m_pSCacheList = m_pNextInDatabase;
		}

		m_pNextInDatabase  = NULL;
		m_pPrevInDatabase = NULL;
	}

	m_pDatabase = NULL;
}

/****************************************************************************
Desc:	Link a cache block to the free list.  This routine assumes that the
		block cache mutex is locked.
****************************************************************************/
void F_CachedBlock::linkToFreeList(
	FLMUINT		uiFreeTime)
{
	flmAssert( !m_ui16Flags);
	flmAssert( !m_pDatabase);
	flmAssert( !m_pPrevInReplaceList);
	flmAssert( !m_pNextInReplaceList);

	if (m_ui64HighTransID != ~((FLMUINT64)0))
	{
		// Set the transaction ID to ~((FLMUINT64)0) so that the old version
		// counts will be decremented if this is an old version
		// of the block.  Also, we want the transaction ID to be
		// ~((FLMUINT64)0) so that when the block is re-used in allocBlock()
		// the old version counts won't be decremented again.

		setTransID( ~((FLMUINT64)0));
	}

	if ((m_pNextInDatabase = gv_SFlmSysData.pBlockCacheMgr->m_pFirstFree) != NULL)
	{
		m_pNextInDatabase->m_pPrevInDatabase = this;
	}
	else
	{
		gv_SFlmSysData.pBlockCacheMgr->m_pLastFree = this;
	}

	m_pPrevInDatabase = NULL;
	m_uiBlkAddress = uiFreeTime;
	m_ui16Flags = CA_FREE;
	
	gv_SFlmSysData.pBlockCacheMgr->m_pFirstFree = this;
	gv_SFlmSysData.pBlockCacheMgr->m_uiFreeBytes += memSize();
	gv_SFlmSysData.pBlockCacheMgr->m_uiFreeCount++;
}

/****************************************************************************
Desc:	Unlink a cache block from the free list.  This routine assumes
		that the block cache mutex has already been locked.
****************************************************************************/
void F_CachedBlock::unlinkFromFreeList( void)
{
	FLMUINT	uiSize = memSize();

	flmAssert( !m_uiUseCount);
	flmAssert( m_ui16Flags == CA_FREE);

	if( m_pNextInDatabase)
	{
		m_pNextInDatabase->m_pPrevInDatabase = m_pPrevInDatabase;
	}
	else
	{
		gv_SFlmSysData.pBlockCacheMgr->m_pLastFree = m_pPrevInDatabase;
	}

	if( m_pPrevInDatabase)
	{
		m_pPrevInDatabase->m_pNextInDatabase = m_pNextInDatabase;
	}
	else
	{
		gv_SFlmSysData.pBlockCacheMgr->m_pFirstFree = m_pNextInDatabase;
	}

	m_pNextInDatabase = NULL;
	m_pPrevInDatabase = NULL;
	m_ui16Flags &= ~CA_FREE;
	flmAssert( !m_ui16Flags);

	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_uiFreeBytes >= uiSize);
	gv_SFlmSysData.pBlockCacheMgr->m_uiFreeBytes -= uiSize;
	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_uiFreeCount);
	gv_SFlmSysData.pBlockCacheMgr->m_uiFreeCount--;
}

/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read or write
		to complete.
		NOTE:  This routine assumes that the block cache mutex is already
		locked.
****************************************************************************/
FSTATIC void ScaNotify(
	FNOTIFY *			pNotify,
	F_CachedBlock *	pUseSCache,
	RCODE					NotifyRc)
{
	while (pNotify)
	{
		F_SEM	hSem;

		*(pNotify->pRc) = NotifyRc;
		if (RC_OK( NotifyRc))
		{
			if (pNotify->pvUserData)
			{
				*((F_CachedBlock **)pNotify->pvUserData) = pUseSCache;
			}
			if (pUseSCache)
			{
				pUseSCache->useForThread( pNotify->uiThreadId);
			}
		}
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
}

/****************************************************************************
Desc:	This routine logs information about changes to a cache block's flags
****************************************************************************/
#ifdef FLM_DBG_LOG
void F_CachedBlock::logFlgChange(
	FLMUINT16	ui16OldFlags,
	char			cPlace
	)
{
	char			szBuf [60];
	char *		pszTmp;
	FLMUINT16	ui16NewFlags = m_ui16Flags;

	szBuf [0] = cPlace;
	szBuf [1] = '-';
	f_strcpy( &szBuf[2], "FLG:");
	pszTmp = &szBuf [6];

	if (ui16OldFlags & CA_DIRTY)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_DIRTY))
		{
			*pszTmp++ = '-';
		}
		*pszTmp++ = 'D';
		*pszTmp = 0;
	}
	else if (ui16NewFlags & CA_DIRTY)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+D");
		pszTmp += 2;
	}

	if (ui16OldFlags & CA_WRITE_INHIBIT)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_WRITE_INHIBIT))
		{
			*pszTmp++ = '-';
		}
		f_strcpy( pszTmp, "WI");
		pszTmp += 2;
	}
	else if (ui16NewFlags & CA_WRITE_INHIBIT)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+WI");
		pszTmp += 3;
	}

	if (ui16OldFlags & CA_READ_PENDING)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_READ_PENDING))
		{
			*pszTmp++ = '-';
		}
		f_strcpy( pszTmp, "RD");
		pszTmp += 2;
	}
	else if (ui16NewFlags & CA_READ_PENDING)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+RD");
		pszTmp += 3;
	}

	if (ui16OldFlags & CA_WRITE_TO_LOG)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_WRITE_TO_LOG))
		{
			*pszTmp++ = '-';
		}
		f_strcpy( pszTmp, "WL");
		pszTmp += 2;
	}
	else if (ui16NewFlags & CA_WRITE_TO_LOG)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+WL");
		pszTmp += 3;
	}

	if (ui16OldFlags & CA_LOG_FOR_CP)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_LOG_FOR_CP))
		{
			*pszTmp++ = '-';
		}
		f_strcpy( pszTmp, "CP");
		pszTmp += 2;
	}
	else if (ui16NewFlags & CA_LOG_FOR_CP)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+CP");
		pszTmp += 3;
	}

	if (ui16OldFlags & CA_WAS_DIRTY)
	{
		*pszTmp++ = ' ';
		if (!(ui16NewFlags & CA_WAS_DIRTY))
		{
			*pszTmp++ = '-';
		}
		f_strcpy( pszTmp, "WD");
		pszTmp += 2;
	}
	else if (ui16NewFlags & CA_WAS_DIRTY)
	{
		*pszTmp++ = ' ';
		f_strcpy( pszTmp, "+WD");
		pszTmp += 3;
	}

	if (pszTmp != &szBuf [6])
	{
		flmDbgLogWrite( m_pDatabase, m_uiBlkAddress, 0, getLowTransID(), szBuf);
	}
}
#endif

/****************************************************************************
Desc:	This routine frees the memory for a cache block and decrements the
		necessary counters in the cache manager.
		NOTE:  This routine assumes that the block cache mutex is already locked.
****************************************************************************/
F_CachedBlock::~F_CachedBlock()
{
	FLMUINT	uiSize = memSize();

	if (m_ui64HighTransID != ~((FLMUINT64)0))
	{
		flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
		gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
		flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount);
		gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiOldVerCount--;
	}
	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiByteCount >= uiSize);
	gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiByteCount -= uiSize;
	flmAssert( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCount);
	gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCount--;
	if (shouldRehash( gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCount,
							gv_SFlmSysData.pBlockCacheMgr->m_uiNumBuckets))
	{
		if (checkHashFailTime( &gv_SFlmSysData.pBlockCacheMgr->m_uiHashFailTime))
		{
			(void)gv_SFlmSysData.pBlockCacheMgr->rehash();
		}
	}
}

/****************************************************************************
Desc:	This routine unlinks a cache block from all of its lists and then
		optionally frees it.  NOTE:  This routine assumes that the block cache
		mutex is already locked.
****************************************************************************/
void F_CachedBlock::unlinkCache(
	FLMBOOL			bFreeIt,
	RCODE				NotifyRc)
{
#ifdef FLM_DEBUG
	SCACHE_USE *	pUse;
#endif

	// Cache block better not be dirty and better not need to be written
	// to the log.

#ifdef FLM_DEBUG
	if( RC_OK( NotifyRc))
	{
		flmAssert (!(m_ui16Flags &
					(CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP |
					CA_WAS_DIRTY | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	}
#endif

	unlinkFromGlobalList();

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( m_pDatabase, m_uiBlkAddress, 0, getLowTransID(), "UNLINK");
#endif

	// If cache block has no previous versions linked to it, it
	// is in the hash bucket and needs to be unlinked from it.
	// Otherwise, it only needs to be unlinked from the version list.

	if (m_pDatabase)
	{
		if (!m_pPrevInVersionList)
		{
			F_CachedBlock **	ppSCacheBucket;

			ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
					m_pDatabase->getSigBitsInBlkSize(),
					(FLMUINT)m_uiBlkAddress);
			unlinkFromHashBucket( ppSCacheBucket);
			if (m_pNextInVersionList)
			{
				// Older version better not be needing to be logged

#ifdef FLM_DEBUG
				if( RC_OK( NotifyRc))
				{
					flmAssert( !(m_pNextInVersionList->m_ui16Flags &
									(CA_WRITE_TO_LOG | CA_DIRTY |
									CA_WAS_DIRTY | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
				}
#endif
				m_pNextInVersionList->m_pPrevInVersionList = NULL;
				m_pNextInVersionList->linkToHashBucket( ppSCacheBucket);
				m_pNextInVersionList->verifyCache( 2100);
				m_pNextInVersionList = NULL;
			}
		}
		else
		{
			verifyCache( 2000);
			savePrevBlkAddress();

			m_pPrevInVersionList->m_pNextInVersionList = m_pNextInVersionList;
			m_pPrevInVersionList->verifyCache( 2200);

			if (m_pNextInVersionList)
			{
				// Older version better not be dirty or not yet logged.

#ifdef FLM_DEBUG
				if( RC_OK( NotifyRc))
				{
					flmAssert( !(m_pNextInVersionList->m_ui16Flags &
									(CA_WRITE_TO_LOG | CA_DIRTY | CA_WAS_DIRTY)));
				}
#endif
				m_pNextInVersionList->m_pPrevInVersionList = m_pPrevInVersionList;
				m_pNextInVersionList->verifyCache( 2300);
			}

			m_pNextInVersionList = NULL;
			m_pPrevInVersionList = NULL;
		}
#ifdef SCACHE_LINK_CHECKING

		// Verify that the thing is not in a hash bucket.
		{
			F_CachedBlock **	ppSCacheBucket;
			F_CachedBlock *	pTmpSCache;

			ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
					m_pDatabase->getSigBitsInBlkSize(),
					m_uiBlkAddress);
			pTmpSCache = *ppSCacheBucket;
			while (pTmpSCache && this != pTmpSCache)
			{
				pTmpSCache = pTmpSCache->m_pNextInHashBucket;
			}

			if (pTmpSCache)
			{
				f_breakpoint( 4);
			}
		}
#endif

		unlinkFromDatabase();
	}

	if (bFreeIt)
	{
		
		// Free the notify list associated with the cache block.
		// NOTE: If there is actually a notify list, NotifyRc WILL ALWAYS
		// be something other than NE_SFLM_OK.  If there is a notify list,
		// the notified threads will thus get a non-OK return code
		// in every case.

#ifdef FLM_DEBUG
		if( NotifyRc == NE_SFLM_OK)
		{
			flmAssert( m_pNotifyList == NULL);
		}
#endif

		ScaNotify( m_pNotifyList, NULL, NotifyRc);
		m_pNotifyList = NULL;

#ifdef FLM_DEBUG

		// Free the use list associated with the cache block

		pUse = m_pUseList;
		while (pUse)
		{
			SCACHE_USE *	pTmp;

			pTmp = pUse;
			pUse = pUse->pNext;
			f_free( &pTmp);
		}
#endif

		delete this;
	}
}

/****************************************************************************
Desc:	Unlink all log blocks for a file that were logged during the transaction.
		NOTE: This is only called when a transaction is aborted.
		WHEN A TRANSACTION IS ABORTED, THIS FUNCTION SHOULD BE CALLED BEFORE
		FREEING DIRTY BLOCKS.  OTHERWISE, THE m_pPrevInVersionList POINTER
		WILL BE NULL AND WILL CAUSE AN ABEND WHEN IT IS ACCESSED.
		NOTE: This routine assumes that the block cache mutex has been
		locked.
****************************************************************************/
void F_Database::unlinkTransLogBlocks( void)
{
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pNextSCache;

	pSCache = m_pTransLogList;
	while (pSCache)
	{
#ifdef FLM_DBG_LOG
		FLMUINT16	ui16OldFlags = pSCache->m_ui16Flags;
#endif

		if (pSCache->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
		{
			flmAssert( m_uiLogCacheCount);
			m_uiLogCacheCount--;
		}

		pSCache->clearFlags( CA_WRITE_TO_LOG | CA_LOG_FOR_CP);
		pNextSCache = pSCache->m_pNextInHashBucket;

		if (pSCache->m_ui16Flags & CA_WAS_DIRTY)
		{
			flmAssert( this == pSCache->m_pDatabase);
			pSCache->setDirtyFlag( this);
			pSCache->clearFlags( CA_WAS_DIRTY);

			// Move the block into the dirty blocks.

			pSCache->unlinkFromDatabase();
			pSCache->linkToDatabase( this);
		}

#ifdef FLM_DBG_LOG
		pSCache->logFlgChange( ui16OldFlags, 'A');
#endif

		// Perhaps we don't really need to set these pointers to NULL,
		// but it helps keep things clean.

		pSCache->m_pNextInHashBucket = NULL;
		pSCache->m_pPrevInHashBucket = NULL;
		pSCache = pNextSCache;
	}
	m_pTransLogList = NULL;
}

/****************************************************************************
Desc:	Unlink a cache block from the list of cache blocks that are in the log
		list for the current transaction.
****************************************************************************/
void F_CachedBlock::unlinkFromTransLogList( void)
{

#ifdef SCACHE_LINK_CHECKING

	// Make sure the block is not in a hash bucket

	{
		F_CachedBlock **	ppSCacheBucket;
		F_CachedBlock *	pTmpSCache;

		ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
					m_pDatabase->m_uiSigBitsInBlkSize,
					m_uiBlkAddress);
		pTmpSCache = *ppSCacheBucket;
		while (pTmpSCache && pTmpSCache != this)
		{
			pTmpSCache = pTmpSCache->m_pNextInHashBucket;
		}

		if (pTmpSCache)
		{
			f_breakpoint( 1001);
		}

		// Make sure the block is in the log list.

		pTmpSCache = m_pDatabase->m_pTransLogList;
		while (pTmpSCache && pTmpSCache != this)
		{
			pTmpSCache = pTmpSCache->m_pNextInHashBucket;
		}

		if (!pTmpSCache)
		{
			f_breakpoint( 1002);
		}
	}
#endif

	if (m_pPrevInHashBucket)
	{
		m_pPrevInHashBucket->m_pNextInHashBucket = m_pNextInHashBucket;
	}
	else
	{
		m_pDatabase->m_pTransLogList = m_pNextInHashBucket;
	}

	if (m_pNextInHashBucket)
	{
		m_pNextInHashBucket->m_pPrevInHashBucket = m_pPrevInHashBucket;
	}

	m_pNextInHashBucket = NULL;
	m_pPrevInHashBucket = NULL;
}

/****************************************************************************
Desc:	The block pointed to by pSCache is about to be removed from from the
		version list for a particular block address because it is no longer
		needed.  Before doing that, the previous block address should be
		moved to the next newer version's block header so that it will not be
		lost, but only if the next newer version's block header is not already
		pointing to a prior version of the block.
		This method assumes the block cache mutex is locked.
****************************************************************************/
void F_CachedBlock::savePrevBlkAddress( void)
{
	FLMUINT				uiPrevBlkAddress = getPriorImageAddress();
	F_CachedBlock *	pNewerSCache;
	FLMUINT				uiNewerBlkPrevAddress;

	// NOTE: If a block is being read in from disk, it has to have a
	// previous block address in its header.  Otherwise, it could never
	// have been written out to disk and removed from cache in the first
	// place.  This is obvious for versions being read from the rollback
	// log - it would be impossible to retrieve them from the rollback
	// log if they weren't already part of a version chain!  It is also
	// true for the most current version of a block.  The most current
	// version of a block can never be written out and removed from
	// cache without having a pointer to the chain of older versions that
	// may still be needed by read transactions - or to rollback the
	// transaction - or to recover a checkpoint.

	if ((uiPrevBlkAddress) &&
		 ((pNewerSCache = m_pPrevInVersionList) != NULL) &&
		 (!(pNewerSCache->m_ui16Flags & CA_READ_PENDING)))
	{

		// Only move the older version's previous block address to the
		// newer version, if the newer version doesn't already have a
		// previous block address.  Also need to set the previous
		// transaction ID.
		//
		// NOTE: The newer block may or may not be dirty.  It is OK
		// to change the prior version address in the header of a
		// non-dirty block in this case.  This is because the block
		// may or may not be written out to the roll-back log.  If it
		// is, we want to make sure it has the correct prior version
		// address.  If it isn't ever written out to the log, it
		// will eventually fall out of cache because it is no longer
		// needed.

		uiNewerBlkPrevAddress = pNewerSCache->getPriorImageAddress();
		if (!uiNewerBlkPrevAddress)
		{

			// Need to temporarily use the newer version of the block
			// before changing its prior image block address.

			pNewerSCache->useForThread( 0);
			flmAssert( uiPrevBlkAddress);

			pNewerSCache->m_pBlkHdr->ui32PriorBlkImgAddr =
				(FLMUINT32)uiPrevBlkAddress;

			// Need to remove the newer block from the file log
			// list, since it no longer needs to be logged

			if( pNewerSCache->m_ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				pNewerSCache->unlinkFromLogList();
			}

			pNewerSCache->releaseForThread();
		}
	}
}

/****************************************************************************
Desc:	See if we should force a checkpoint.
****************************************************************************/
FINLINE FLMBOOL scaSeeIfForceCheckpoint(
	FLMUINT		uiCurrTime,
	FLMUINT		uiLastCheckpointTime,
	CP_INFO *	pCPInfo)
{
	if (FLM_ELAPSED_TIME( uiCurrTime, uiLastCheckpointTime) >=
			gv_SFlmSysData.uiMaxCPInterval)
	{
		if (pCPInfo)
		{
			pCPInfo->bForcingCheckpoint = TRUE;
			pCPInfo->eForceCheckpointReason = SFLM_CP_TIME_INTERVAL_REASON;
			pCPInfo->uiForceCheckpointStartTime = (FLMUINT)FLM_GET_TIMER();
		}

		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Allocate the array that keeps track of blocks written or logged.
****************************************************************************/
RCODE F_Database::allocBlocksArray(
	FLMUINT		uiNewSize,
	FLMBOOL		bOneArray)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiOldSize = m_uiBlocksDoneArraySize;

	if (!uiNewSize)
	{
		uiNewSize = uiOldSize + 500;
	}

	// Re-alloc the array

	if (RC_BAD( rc = f_realloc(
							(FLMUINT)(uiNewSize *
									   (sizeof( F_CachedBlock *) +
										 sizeof( F_CachedBlock *))),
							&m_ppBlocksDone)))
	{
		goto Exit;
	}

	// Copy the old stuff into the two new areas of the new array.

	if (uiOldSize && !bOneArray)
	{
		f_memmove( &m_ppBlocksDone [uiNewSize],
			&m_ppBlocksDone [uiOldSize],
			uiOldSize * sizeof( F_CachedBlock *));
	}

	// Set the new array size

	m_uiBlocksDoneArraySize = uiNewSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Write out log blocks to the rollback log for a database.
****************************************************************************/
RCODE F_Database::flushLogBlocks(
	F_SEM						hWaitSem,
	SFLM_DB_STATS *		pDbStats,
	F_SuperFileHdl *		pSFileHdl,
	FLMBOOL					bIsCPThread,
	FLMUINT					uiMaxDirtyCache,
	FLMBOOL *				pbForceCheckpoint,
	FLMBOOL *				pbWroteAll)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiLogEof;
	SFLM_DB_HDR *		pDbHdr;
	F_CachedBlock *	pTmpSCache;
	F_CachedBlock *	pLastBlockToLog;
	F_CachedBlock *	pFirstBlockToLog;
	F_CachedBlock *	pDirtySCache;
	F_CachedBlock *	pSavedSCache = NULL;
	FLMUINT				uiDirtyCacheLeft;
	FLMUINT				uiPrevBlkAddress;
	FLMBOOL				bMutexLocked = TRUE;
	FLMBOOL				bLoggedFirstBlk = FALSE;
	FLMBOOL				bLoggedFirstCPBlk = FALSE;
	FLMUINT				uiCurrTime;
	FLMUINT				uiSaveEOFAddr;
	FLMUINT				uiSaveFirstCPBlkAddr = 0;
	FLMBOOL				bDone = FALSE;
	F_CachedBlock *	pUsedSCache;
	F_CachedBlock *	pNextSCache = NULL;
	F_CachedBlock **	ppUsedBlocks = (F_CachedBlock **)((m_ppBlocksDone)
											? &m_ppBlocksDone [m_uiBlocksDoneArraySize]
											: (F_CachedBlock **)NULL);
	FLMUINT				uiTotalLoggedBlocks = 0;
	FLMBOOL				bForceCheckpoint = *pbForceCheckpoint;
#ifdef FLM_DBG_LOG
	FLMUINT16			ui16OldFlags;
#endif

	m_uiCurrLogWriteOffset = 0;

	// Get the correct log header.  If we are in an update transaction,
	// need to use the uncommitted log header.  Otherwise, use the last
	// committed log header.

	pDbHdr = bIsCPThread
					? &m_lastCommittedDbHdr
					: &m_uncommittedDbHdr;

	uiLogEof = (FLMUINT)pDbHdr->ui32RblEOF;
	
	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	pDirtySCache = m_pFirstInLogList;
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

	flmAssert( m_pCurrLogBuffer == NULL);

	uiDirtyCacheLeft = (m_uiDirtyCacheCount + m_uiLogCacheCount) *
							m_uiBlockSize;

	for (;;)
	{
		if( !pDirtySCache)
		{
			bDone = TRUE;
			goto Write_Log_Blocks;
		}

		flmAssert( pDirtySCache->m_ui16Flags & CA_DIRTY);
		flmAssert( pDirtySCache->m_ui16Flags & CA_IN_FILE_LOG_LIST);

		// See if we should give up our write lock.  Will do so if we
		// are not forcing a checkpoint and we have not exceeded the
		// maximum time since the last checkpoint AND the dirty cache
		// left is below the maximum.

		if (!bForceCheckpoint && bIsCPThread)
		{
			if (scaSeeIfForceCheckpoint( uiCurrTime, m_uiLastCheckpointTime,
													m_pCPInfo))
			{
				bForceCheckpoint = TRUE;
			}
			else
			{
				if (m_pWriteLockObj->getWaiterCount() &&
					 uiDirtyCacheLeft <= uiMaxDirtyCache)
				{
					bDone = TRUE;
					*pbWroteAll = FALSE;
					goto Write_Log_Blocks;
				}
			}
		}

		uiPrevBlkAddress = pDirtySCache->getPriorImageAddress();
		if (uiPrevBlkAddress)
		{
			// We shouldn't find anything in the log list that has
			// already been logged.  However, if we do find something,
			// we will deal with it rather than returning an error.

			flmAssert( 0);
			pTmpSCache = pDirtySCache->m_pNextInReplaceList;
			pDirtySCache->unlinkFromLogList();
			pDirtySCache = pTmpSCache;
			continue;
		}

		// The replace list pointers are used to maintain links
		// between items in the file log list

		pTmpSCache = pDirtySCache->m_pNextInVersionList;
		pLastBlockToLog = NULL;
		pFirstBlockToLog = NULL;

		// Grab the next block in the chain and see if we are done.
		// NOTE: pDirtySCache should not be accessed in the loop
		// below, because it has been changed to point to the
		// next cache block in the log list.  If you need to access
		// the current block, use pSavedSCache.

		pSavedSCache = pDirtySCache;
		if ((pDirtySCache = pDirtySCache->m_pNextInReplaceList) == NULL)
		{
			bDone = TRUE;
		}
#ifdef FLM_DEBUG
		else
		{
			flmAssert( pDirtySCache->m_ui16Flags & CA_DIRTY);
			flmAssert( pDirtySCache->m_ui16Flags & CA_IN_FILE_LOG_LIST);
		}
#endif

		// Traverse down the list of prior versions of the block until
		// we hit one that has a prior version on disk.  Throw out
		// any not marked as CA_WRITE_TO_LOG, CA_LOG_FOR_CP, and
		// not needed by a read transaction.

		while (pTmpSCache)
		{
			pNextSCache = pTmpSCache->m_pNextInVersionList;
			FLMBOOL	bWillLog;

			uiPrevBlkAddress = pTmpSCache->getPriorImageAddress();

			// If we determine that we need to log a block, put a use on the
			// newer version of the block to prevent other threads from verifying
			// their checksums while we are writing the older versions to
			// the log.  This is because lgOutputBlock may modify information
			// in the newer block's header area.

			if (pTmpSCache->m_ui16Flags & CA_READ_PENDING)
			{

				// No need to go further down the list if this block is
				// being read in.  If it is being read in, every older
				// version has a path to it - otherwise, it would never
				// have been written out so that it would need to be
				// read back in.

				break;
			}
			else if (pTmpSCache->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
			{
				bWillLog = TRUE;
			}

			// Even if the block is not needed by a read transaction, if it
			// has a use count, we need to log it so that all blocks between
			// pFirstBlockToLog and pLastBlockToLog are logged.  This is
			// necessary to ensure that previous block addresses carry all
			// the way up the version chain.  Also, the loop that does the
			// actual logging below assumes that the links from pLastBlockToLog
			// to pFirstBlockToLog will NOT be altered - even though the mutex
			// is not locked.  This can only be ensured if every block between
			// the two points is guaranteed to be logged - which also guarantees
			// that it will not be moved out of the list - because of the fact
			// that some sort of logging bit has been set.
			// Note that a block can have a use count even though it is no
			// longer needed by a read transaction because another thread
			// may have temporarily put a use on it while traversing down
			// the chain - or for any number of other reasons.

			else if (pTmpSCache->neededByReadTrans() ||
						pTmpSCache->m_uiUseCount)
			{
				bWillLog = TRUE;
			}
			else
			{
				bWillLog = FALSE;

				// Since the block is no longer needed by a read transaction,
				// and it is not in use, free it

				pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);
			}

			// Add this block to the list of those we will be logging if the
			// bWillLog flag got set above.

			if (bWillLog)
			{
				if (uiTotalLoggedBlocks >= m_uiBlocksDoneArraySize)
				{
					if (RC_BAD( rc = allocBlocksArray( 0, FALSE)))
					{
						goto Exit;
					}
					ppUsedBlocks = &m_ppBlocksDone [m_uiBlocksDoneArraySize];
				}

				pLastBlockToLog = pTmpSCache;
				if (!pFirstBlockToLog)
				{
					pFirstBlockToLog = pLastBlockToLog;
				}

				pTmpSCache->m_pPrevInVersionList->useForThread( 0);
				pTmpSCache->useForThread( 0);
				m_ppBlocksDone [uiTotalLoggedBlocks] = pTmpSCache;
				ppUsedBlocks [uiTotalLoggedBlocks] = pTmpSCache->m_pPrevInVersionList;
				uiTotalLoggedBlocks++;
			}

			// No need to go further down the list if this block has
			// has a previous block address.

			if (uiPrevBlkAddress)
			{
				break;
			}
			pTmpSCache = pNextSCache;
		}

#ifdef FLM_DEBUG
		while (pNextSCache)
		{
			flmAssert( !(pNextSCache->m_ui16Flags &
							 (CA_WRITE_TO_LOG | CA_LOG_FOR_CP)));
			pNextSCache = pNextSCache->m_pNextInVersionList;

		}
#endif

		// If nothing to log for the block, unlink it from the
		// log list.  We check CA_IN_FILE_LOG_LIST again, because
		// savePrevBlkAddress may have been called during an
		// unlink above.  savePrevBlkAddress will remove
		// the dirty cache block from the log list if it determines
		// that there is no need to log prior versions

		if (!pLastBlockToLog)
		{
			if (pSavedSCache->m_ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				pSavedSCache->unlinkFromLogList();
			}
			continue;
		}

		// Don't want the mutex locked while we do the I/O

		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;

		// Write the log blocks to the rollback log.
		// Do all of the blocks from oldest to most current.  Stop when we
		// hit the first log block.

		while (pLastBlockToLog)
		{
			FLMUINT	uiLogPos = uiLogEof;

			if (RC_BAD( rc = lgOutputBlock( pDbStats, pSFileHdl,
											pLastBlockToLog,
											pLastBlockToLog->m_pPrevInVersionList->m_pBlkHdr,
											&uiLogEof)))
			{
				goto Exit;
			}

			if (pLastBlockToLog->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
			{
				flmAssert( uiDirtyCacheLeft >= m_uiBlockSize);
				uiDirtyCacheLeft -= m_uiBlockSize;
			}

			// If we are logging a block for the current update
			// transaction, and this is the first block we have logged,
			// remember the block address where we logged it.

			if ((pLastBlockToLog->m_ui16Flags & CA_WRITE_TO_LOG) &&
				 !m_uiFirstLogBlkAddress)
			{
				// This better not EVER happen in the CP thread.

				flmAssert( !bIsCPThread);
				bLoggedFirstBlk = TRUE;
				m_uiFirstLogBlkAddress = uiLogPos;
			}

			// If we are logging the checkpoint version of the
			// block, and this is the first block we have logged
			// since the last checkpoint, remember its position so
			// that we can write it out to the log header when we
			// complete the checkpoint.

			if ((pLastBlockToLog->m_ui16Flags & CA_LOG_FOR_CP) &&
				 !m_uiFirstLogCPBlkAddress)
			{
				bLoggedFirstCPBlk = TRUE;
				m_uiFirstLogCPBlkAddress = uiLogPos;
			}

			// Break when we hit the first log block.

			if (pLastBlockToLog == pFirstBlockToLog)
			{
				break;
			}

			pLastBlockToLog = pLastBlockToLog->m_pPrevInVersionList;
		}

		// If we have logged some blocks, force the log header to be
		// updated on one of the following conditions:

		// 1. We have logged over 2000 blocks.  We do this to keep
		//		our array of logged blocks from growing too big.
		//	2.	We are done logging.

Write_Log_Blocks:

		if (uiTotalLoggedBlocks &&				// Must be at least one logged block
			 (uiTotalLoggedBlocks >= 2000 || bDone))
		{
			if (bMutexLocked)
			{
				f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
				bMutexLocked = FALSE;
			}

			// Flush the last log buffer, if not already flushed.

			if (m_uiCurrLogWriteOffset)
			{
				if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl)))
				{
					goto Exit;
				}
			}

			// If doing async, wait for pending writes to complete before writing
			// the log header.

			if (RC_BAD( rc = m_pBufferMgr->waitForAllPendingIO()))
			{
				goto Exit;
			}

			// Must wait for all RFL writes before writing out log header.

			if (!bIsCPThread)
			{
				(void)m_pRfl->seeIfRflWritesDone( hWaitSem, TRUE);
			}

			// Save the EOF address so we can restore it if
			// the write fails.

			uiSaveEOFAddr = (FLMUINT)pDbHdr->ui32RblEOF;
			pDbHdr->ui32RblEOF = (FLMUINT32)uiLogEof;

			if (bLoggedFirstCPBlk)
			{
				uiSaveFirstCPBlkAddr = pDbHdr->ui32RblFirstCPBlkAddr;
				pDbHdr->ui32RblFirstCPBlkAddr =
					(FLMUINT32)m_uiFirstLogCPBlkAddress;
			}

			if (RC_BAD( rc = writeDbHdr( pDbStats, pSFileHdl,
									pDbHdr, &m_checkpointDbHdr, FALSE)))
			{

				// If the write of the log header fails,
				// we want to restore the log header to what it was before
				// because we always use the log header from memory instead
				// of reading it from disk.  The one on disk is only
				// current for many fields as of the last checkpoint.

				pDbHdr->ui32RblEOF = (FLMUINT32)uiSaveEOFAddr;
				if (bLoggedFirstCPBlk)
				{
					pDbHdr->ui32RblFirstCPBlkAddr = (FLMUINT32)uiSaveFirstCPBlkAddr;
				}
				goto Exit;
			}

			// Need to update the committed log header when we are operating in
			// an uncommitted transaction so that if the transaction turns out
			// to be empty, we will have the correct values in the committed
			// log header for subsequent transactions or the checkpoint thread
			// itself.

			if (!bIsCPThread)
			{
				m_lastCommittedDbHdr.ui32RblEOF = pDbHdr->ui32RblEOF;

				if (bLoggedFirstCPBlk)
				{
					m_lastCommittedDbHdr.ui32RblFirstCPBlkAddr =
						pDbHdr->ui32RblFirstCPBlkAddr;
				}
			}

			// Once the write is safe, we can reset things to start over.

			bLoggedFirstBlk = FALSE;
			bLoggedFirstCPBlk = FALSE;

			// Clean up the log blocks array - releasing blocks, etc.

			if (m_pCPInfo)
			{
				lockMutex();
				m_pCPInfo->uiLogBlocksWritten += uiTotalLoggedBlocks;
				unlockMutex();
			}
			
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;

			while (uiTotalLoggedBlocks)
			{
				uiTotalLoggedBlocks--;
				pTmpSCache = m_ppBlocksDone [uiTotalLoggedBlocks];
#ifdef FLM_DBG_LOG
				ui16OldFlags = pTmpSCache->m_ui16Flags;
#endif
				pUsedSCache = ppUsedBlocks [uiTotalLoggedBlocks];

				// Newer block should be released, whether we succeeded
				// or not - because it will always have been used.

				pUsedSCache->releaseForThread();
				pTmpSCache->releaseForThread();

				// The current version of the block may have already been removed from
				// the file log list if more than one block in the version chain
				// needed to be logged.  If the block is still in the file log list,
				// it will be removed.  Otherwise, the prior image address better
				// be a non-zero value.

				if( pUsedSCache->m_ui16Flags & CA_IN_FILE_LOG_LIST)
				{
					pUsedSCache->unlinkFromLogList();
				}

				flmAssert( pUsedSCache->getPriorImageAddress());

				// Unlink from list of transaction log blocks

				if (pTmpSCache->m_ui16Flags & CA_WRITE_TO_LOG)
				{
					pTmpSCache->unlinkFromTransLogList();
				}

				// Unset logging flags on logged block.

				if (pTmpSCache->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
				{
					flmAssert( m_uiLogCacheCount);
					m_uiLogCacheCount--;
				}

				pTmpSCache->clearFlags( CA_LOG_FOR_CP | CA_WRITE_TO_LOG | CA_WAS_DIRTY);

#ifdef FLM_DBG_LOG
				pTmpSCache->logFlgChange( ui16OldFlags, 'D');
#endif
				
				if (!pTmpSCache->m_uiUseCount &&
				    !pTmpSCache->m_ui16Flags &&
					 !pTmpSCache->neededByReadTrans())
				{
					flmAssert( pTmpSCache->m_ui64HighTransID != ~((FLMUINT64)0));
					pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);
				}
			}

			uiDirtyCacheLeft =
					(m_uiDirtyCacheCount + m_uiLogCacheCount) *
					m_uiBlockSize;

			// When the current set of log blocks were flushed, they were
			// also unlinked from the file log list.  So, we need to
			// start at the beginning of the log list to pick up
			// where we left off.

			pDirtySCache = m_pFirstInLogList;
		}
		else if (!bDone)
		{
			if (!bMutexLocked)
			{
				f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
				bMutexLocked = TRUE;
			}

			// Need to reset pDirtySCache here because the background cache
			// cleanup thread may have unlinked it from the log list and
			// cleaned up any prior versions if it determined that the blocks
			// were no longer needed.

			if( (pDirtySCache = pSavedSCache->m_pNextInReplaceList) == NULL)
			{
				bDone = TRUE;
				goto Write_Log_Blocks;
			}
		}

		if (bDone)
		{
			break;
		}

		flmAssert( bMutexLocked);
	}

#ifdef FLM_DEBUG
	if( bForceCheckpoint || !bIsCPThread ||
		(!bForceCheckpoint && bIsCPThread && *pbWroteAll))
	{
		flmAssert( !m_uiLogListCount);
		flmAssert( !m_uiLogCacheCount);
	}
#endif

Exit:

	if (RC_BAD( rc))
	{
		// Flush the last log buffer, if not already flushed.

		if (m_uiCurrLogWriteOffset)
		{

			if (bMutexLocked)
			{
				f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
				bMutexLocked = FALSE;
			}

			// Don't care what rc is at this point.  Just calling
			// lgFlushLogBuffer to clear the buffer.

			(void)lgFlushLogBuffer( pDbStats, pSFileHdl);
		}

		// Need to wait for any async writes to complete.

		if (bMutexLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = FALSE;
		}

		// Don't care about rc here, but we don't want to leave
		// this routine until all pending IO is taken care of.

		(void)m_pBufferMgr->waitForAllPendingIO();
		
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}

		// Clean up the log blocks array - releasing blocks, etc.

		while (uiTotalLoggedBlocks)
		{
			uiTotalLoggedBlocks--;
			pTmpSCache = m_ppBlocksDone [uiTotalLoggedBlocks];
			pUsedSCache = ppUsedBlocks [uiTotalLoggedBlocks];

#ifdef FLM_DEBUG

			// If this is the most current version of the block, it
			// should still be in the file log list.

			if( !pUsedSCache->m_pPrevInVersionList)
			{
				flmAssert( pUsedSCache->m_ui16Flags & CA_IN_FILE_LOG_LIST);
			}
#endif

			// Used blocks should be released, whether we succeeded
			// or not.

			pUsedSCache->releaseForThread();
			pTmpSCache->releaseForThread();

			// If we quit before logging the blocks, we don't really
			// want to change anything on the block, but we do want
			// to set the previous block address back to zero on the
			// block that is just newer than this one.

			// Must put a USE on the block so that the memory cache
			// verifying code will not barf when we change the
			// data in the block - checksum is calculated and set when
			// the use count goes from one to zero, and then verified
			// when it goes from zero to one.

			pTmpSCache->m_pPrevInVersionList->useForThread( 0);
			pTmpSCache->m_pPrevInVersionList->m_pBlkHdr->ui32PriorBlkImgAddr = 0;
			pTmpSCache->m_pPrevInVersionList->releaseForThread();
		}

#ifdef SCACHE_LINK_CHECKING

		// If above logic changes where mutex might not be locked at
		// this point, be sure to modify this code to re-lock it.

		flmAssert( bMutexLocked);
		scaVerify( 100);
#endif

		// Things to restore to their original state if we had an error.

		if (bLoggedFirstBlk)
		{
			m_uiFirstLogBlkAddress = 0;
		}
		if (bLoggedFirstCPBlk)
		{
			m_uiFirstLogCPBlkAddress = 0;
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;
	}

	// Better not be any incomplete writes at this point.

	flmAssert( !m_pBufferMgr->isIOPending());
	flmAssert( m_pCurrLogBuffer == NULL);

	*pbForceCheckpoint = bForceCheckpoint;
	return( rc);
}

/****************************************************************************
Desc:	This routine is called whenever a write of a dirty block completes.
****************************************************************************/
/****************************************************************************
Desc:	This routine is called whenever a write of a dirty block completes.
****************************************************************************/
FSTATIC void SQFAPI scaWriteComplete(
	IF_IOBuffer *		pIOBuffer,
	void *				pvData)
{
	RCODE					rc;
	FLMUINT				uiNumBlocks = 0;
	F_CachedBlock *	pSCache = NULL;
	F_Database *		pDatabase;
	SFLM_DB_STATS *	pDbStats = (SFLM_DB_STATS *)pvData;
	FLMUINT				uiMilliPerBlock = 0;
	FLMUINT				uiExtraMilli = 0;
#ifdef FLM_DBG_LOG
	FLMUINT16			ui16OldFlags;
#endif

	f_assert( pIOBuffer->isComplete());

	rc = pIOBuffer->getCompletionCode();
	uiNumBlocks = pIOBuffer->getCallbackDataCount();

	if( pDbStats)
	{
		FLMUINT64	ui64ElapMilli = pIOBuffer->getElapsedTime();

		uiMilliPerBlock = (FLMUINT)(ui64ElapMilli / (FLMUINT64)uiNumBlocks);
		uiExtraMilli = (FLMUINT)(ui64ElapMilli % (FLMUINT64)uiNumBlocks);
	}

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	while( uiNumBlocks)
	{
		uiNumBlocks--;
		pSCache = (F_CachedBlock *)pIOBuffer->getCallbackData( uiNumBlocks);
		pDatabase = pSCache->getDatabase();

		if (pDbStats)
		{
			F_BLK_HDR *				pBlkHdr = pSCache->getBlockPtr();
			SFLM_LFILE_STATS *	pLFileStats;
			SFLM_BLOCKIO_STATS *	pBlockIOStats;

			if (!blkIsBTree( pBlkHdr))
			{
				pLFileStats = NULL;
			}
			else
			{
				if (RC_BAD( flmStatGetLFile( pDbStats,
						(FLMUINT)((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile,
						getBlkLfType( (F_BTREE_BLK_HDR *)pBlkHdr),
						0, &pLFileStats, NULL, NULL)))
				{
					pLFileStats = NULL;
				}
			}
			if ((pBlockIOStats = flmGetBlockIOStatPtr( pDbStats,
											pLFileStats, (FLMBYTE *)pBlkHdr)) != NULL)
			{
				pBlockIOStats->BlockWrites.ui64Count++;
				pBlockIOStats->BlockWrites.ui64TotalBytes +=
											pDatabase->getBlockSize();
				if (uiExtraMilli)
				{
					pBlockIOStats->BlockWrites.ui64ElapMilli +=
						(uiMilliPerBlock + 1);
					uiExtraMilli--;
				}
				else
				{
					pBlockIOStats->BlockWrites.ui64ElapMilli +=
						uiMilliPerBlock;
				}
			}
		}

		pSCache->releaseForThread();
		if (pSCache->getModeFlags() & CA_DIRTY)
		{
			flmAssert( pSCache->getModeFlags() & CA_WRITE_PENDING);
#ifdef FLM_DBG_LOG
			ui16OldFlags = pSCache->getModeFlags();
#endif
			pSCache->clearFlags( CA_WRITE_PENDING);
			if (RC_OK( rc))
			{
				pSCache->unsetDirtyFlag();
			}

#ifdef FLM_DBG_LOG
			pSCache->logFlgChange( ui16OldFlags, 'H');
#endif

			// If there are more dirty blocks after this
			// one, move this one out of the dirty
			// blocks.

			pSCache->unlinkFromDatabase();
			pSCache->linkToDatabase( pDatabase);
		}
		else
		{
			flmAssert( !(pSCache->getModeFlags() & CA_WRITE_PENDING));
		}
	}
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_BlockCacheMgr::cleanupLRUCache( void)
{
	FLMUINT				uiByteThreshold;
	FLMUINT				uiSlabThreshold;
	FLMUINT				uiSlabSize;
	F_CachedBlock *	pPrevSCache;
	F_CachedBlock *	pTmpSCache;
	FLMBOOL				bDefragNeeded = FALSE;
	
	// Remove non-dirty blocks from the LRU end of the cache

	uiSlabThreshold = gv_SFlmSysData.pGlobalCacheMgr->m_uiMaxSlabs >> 1;
	uiSlabSize = gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager->getSlabSize();
	
	// If the cache isn't over its slab threshold, we are done
	
	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
		!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
	{
		goto Exit;
	}
	
	uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
								? m_Usage.uiByteCount - uiSlabSize
								: 0;

	pTmpSCache = (F_CachedBlock *)m_MRUList.m_pLRUItem;
	while( pTmpSCache)
	{
		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pTmpSCache below, in which case we would
		// have lost the next entry.

		pPrevSCache = (F_CachedBlock *)pTmpSCache->m_pPrevInGlobal;

		// Block must not currently be in use, cannot be dirty in any way,
		// cannot be in the process of being read in from disk,
		// and must not be needed by a read transaction.

		if( !pTmpSCache->m_uiUseCount && !pTmpSCache->m_ui16Flags &&
			(!pTmpSCache->m_pDatabase || !pTmpSCache->neededByReadTrans()))
		{
			pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);
			bDefragNeeded = TRUE;
			
			if( m_Usage.uiByteCount <= uiByteThreshold)
			{
				if( pPrevSCache)
				{
					pPrevSCache->useForThread( 0);
				}
				
				gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);
				bDefragNeeded = FALSE;
				
				if( !pPrevSCache)
				{
					break;
				}
	
				pPrevSCache->releaseForThread();

				// We're going to quit when we get under 50 percent for block cache
				// or we aren't over the global limit.  Note that this means we
				// may quit reducing before we get under the global limit.  We
				// don't want to get into a situation where we are starving block
				// cache because node cache is over its limit.
				
				if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
					!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
				{
					goto Exit;
				}

				uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
											? m_Usage.uiByteCount - uiSlabSize
											: 0;

			}
		}

		pTmpSCache = pPrevSCache;
	}
	
	if( bDefragNeeded)
	{
		gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:	Cleanup old blocks in cache that are no longer needed by any
		transaction.  This routine assumes that the block cache mutex has
		been locked.
****************************************************************************/
void F_BlockCacheMgr::cleanupReplaceList( void)
{
	F_CachedBlock *	pTmpSCache;
	F_CachedBlock *	pPrevSCache;

	pTmpSCache = m_pLRUReplace;

	for (;;)
	{
		// Stop when we reach end of list or all old blocks have
		// been freed.

		if (!pTmpSCache || !m_Usage.uiOldVerBytes)
		{
			break;
		}

		// Shouldn't encounter anything with CA_FREE set

		flmAssert( !(pTmpSCache->m_ui16Flags & CA_FREE));

		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pTmpSCache below, in which case we would
		// have lost the next entry.

		pPrevSCache = pTmpSCache->m_pPrevInReplaceList;

		// Block must not currently be in use,
		// Must not be the most current version of a block,
		// Cannot be dirty in any way,
		// Cannot be in the process of being read in from disk,
		// And must not be needed by a read transaction.

		if (!pTmpSCache->m_uiUseCount &&
			 pTmpSCache->m_ui64HighTransID != ~((FLMUINT64)0) &&
			 !pTmpSCache->m_ui16Flags &&
			 (!pTmpSCache->m_pDatabase ||
			  !pTmpSCache->neededByReadTrans()))
		{
			pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);
		}
		pTmpSCache = pPrevSCache;
	}
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_BlockCacheMgr::cleanupFreeCache( void)
{
	F_CachedBlock *	pSCache = m_pLastFree;
	F_CachedBlock *	pPrevSCache;

	while( pSCache)
	{
		pPrevSCache = pSCache->m_pPrevInDatabase;
		if( !pSCache->m_uiUseCount)
		{
			pSCache->unlinkFromFreeList();
			delete pSCache;
		}
		pSCache = pPrevSCache;
	}
}

/****************************************************************************
Desc:	This routine will reduce the number of blocks in the reuse list
		until cache is below its limit.
		NOTE: This routine assumes that the block cache mutex is already locked.
****************************************************************************/
void F_BlockCacheMgr::reduceReuseList( void)
{
	F_CachedBlock *	pTmpSCache;
	F_CachedBlock *	pPrevSCache;
	FLMUINT				uiSlabThreshold;
	FLMUINT				uiSlabSize;
	FLMUINT				uiByteThreshold;

	// Determine if the block limit for block cache been exceeded

	uiSlabThreshold = gv_SFlmSysData.pGlobalCacheMgr->m_uiMaxSlabs >> 1;
	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold) 
	{
		goto Exit;
	}

	// Remove items from cache starting from the LRU

	pTmpSCache = m_pLRUReplace;
	uiSlabSize = gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager->getSlabSize();
	uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
								? m_Usage.uiByteCount - uiSlabSize
								: 0;

	while( pTmpSCache)
	{
		// Need to save the pointer to the previous entry in the list because
		// we may end up freeing pTmpNode below.

		pPrevSCache = pTmpSCache->m_pPrevInReplaceList;

		// See if the item can be freed.

		if( pTmpSCache->canBeFreed())
		{
			pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);

			if( m_Usage.uiByteCount <= uiByteThreshold)
			{
				if( pPrevSCache)
				{
					pPrevSCache->useForThread( 0);
				}

				gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

				if( !pPrevSCache)
				{
					break;
				}

				pPrevSCache->releaseForThread();

				if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold)
				{
					goto Exit;
				}

				uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
											? m_Usage.uiByteCount - uiSlabSize
											: 0;
			}
		}

		pTmpSCache = pPrevSCache;
	}

Exit:

	return;
}

/****************************************************************************
Desc:	Reduce cache to below the cache limit.  NOTE: This routine assumes
		that the block cache mutex is locked.  It may temporarily unlock the mutex
		to write out dirty blocks, but it will always return with the mutex
		still locked.
****************************************************************************/
RCODE F_BlockCacheMgr::reduceCache(
	F_Db *					pDb)
{
	RCODE						rc = NE_SFLM_OK;
	F_Database *			pDatabase = pDb ? pDb->m_pDatabase : NULL;
	FLMBOOL					bForceCheckpoint;
	FLMBOOL					bWroteAll;
	FLMUINT					uiSlabSize;
	FLMUINT					uiSlabThreshold;
	FLMBOOL					bDoingReduce = FALSE;

	// If cache is not full, we are done.

	if( !gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit() || m_bReduceInProgress) 
	{
		goto Exit;
	}
	
	m_bReduceInProgress = TRUE;
	bDoingReduce = TRUE;

	// Determine the cache threshold

	uiSlabThreshold = gv_SFlmSysData.pGlobalCacheMgr->m_uiMaxSlabs >> 1;
	uiSlabSize = gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager->getSlabSize();
	
	// Are we over the threshold?

	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold) 
	{
		goto Exit;
	}

	// Try cleaning up the replace list

	if( m_Usage.uiOldVerBytes)
	{
		cleanupReplaceList();
		gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

		// We're going to quit when we get under 50 percent for block cache
		// or we aren't over the global limit.  Note that this means we
		// may quit reducing before we get under the global limit.  We
		// don't want to get into a situation where we are starving block
		// cache because node cache is over its limit.
		
		if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
			!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
		{
			goto Exit;
		}
	}

	// Clean up the free list

	if( m_uiFreeBytes)
	{
		cleanupFreeCache();

		gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

		// We're going to quit when we get under 50 percent for block cache
		// or we aren't over the global limit.  Note that this means we
		// may quit reducing before we get under the global limit.  We
		// don't want to get into a situation where we are starving block
		// cache because node cache is over its limit.
		
		if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
			!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
		{
			goto Exit;
		}
	}
	
	// Clean up the LRU list
	
	cleanupLRUCache();

	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
		!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
	{
		goto Exit;
	}
		
	// If this isn't an update transaction, there isn't anything else
	// that can be done to reduce cache.

	if( !pDb || 
		(pDb->m_eTransType != SFLM_UPDATE_TRANS && !pDatabase->m_bTempDb))
	{
		goto Exit;
	}

	// Flush log blocks

	if( pDatabase->m_pFirstInLogList)
	{
		bForceCheckpoint = FALSE;
		bWroteAll = TRUE;

		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);

		if( RC_BAD( rc = pDatabase->flushLogBlocks( 
			pDb->m_hWaitSem, pDb->m_pDbStats, pDb->m_pSFileHdl, FALSE, 0,
			&bForceCheckpoint, &bWroteAll)))
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			goto Exit;
		}

		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

		cleanupFreeCache();
		reduceReuseList();

		gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

		// We're going to quit when we get under 50 percent for block cache
		// or we aren't over the global limit.  Note that this means we
		// may quit reducing before we get under the global limit.  We
		// don't want to get into a situation where we are starving block
		// cache because node cache is over its limit.
		
		if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
			!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
		{
			goto Exit;
		}
	}

	// Flush new blocks

	for( ;;)
	{
		FLMUINT		uiNewBlocks = 0;

		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		if( RC_BAD( rc = pDatabase->reduceNewBlocks( 
			pDb->m_pDbStats, pDb->m_pSFileHdl, &uiNewBlocks)))
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			goto Exit;
		}
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

		if( !uiNewBlocks)
		{
			break;
		}

		cleanupFreeCache();
		reduceReuseList();

		gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

		// We're going to quit when we get under 50 percent for block cache
		// or we aren't over the global limit.  Note that this means we
		// may quit reducing before we get under the global limit.  We
		// don't want to get into a situation where we are starving block
		// cache because node cache is over its limit.
		
		if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
			!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
		{
			goto Exit;
		}
	}

	// Flush dirty blocks

	flmAssert( !pDatabase->m_pFirstInLogList);

	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	if( RC_BAD( rc = pDatabase->reduceDirtyCache( 
		pDb->m_pDbStats, pDb->m_pSFileHdl)))
	{
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		goto Exit;
	}

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

	cleanupFreeCache();
	reduceReuseList();

	gv_SFlmSysData.pBlockCacheMgr->defragmentMemory( TRUE);

	// We're going to quit when we get under 50 percent for block cache
	// or we aren't over the global limit.  Note that this means we
	// may quit reducing before we get under the global limit.  We
	// don't want to get into a situation where we are starving block
	// cache because node cache is over its limit.
	
	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
		!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
	{
		goto Exit;
	}

	// Try cleaning up the LRU again
	
	cleanupLRUCache();
	
Exit:

	if( RC_BAD( rc) && pDb)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bDoingReduce)
	{
		m_bReduceInProgress = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:	Constructor for cached block.
****************************************************************************/
F_CachedBlock::F_CachedBlock(
	FLMUINT	uiBlockSize)
{
	m_pPrevInDatabase = NULL;
	m_pNextInDatabase = NULL;
	m_pBlkHdr = (F_BLK_HDR *)((FLMBYTE *)this + sizeof( F_CachedBlock));
	m_pDatabase = NULL;
	m_uiBlkAddress = 0;
	m_pNextInReplaceList = NULL;
	m_pPrevInReplaceList = NULL;
	m_pPrevInHashBucket = NULL;
	m_pNextInHashBucket = NULL;
	m_pPrevInVersionList = NULL;
	m_pNextInVersionList = NULL;
	m_pNotifyList = NULL;
	
	// Need to set high transaction ID to 0xFFFFFFFF.  This indicates that
	// the block is not currently counted in the Usage.uiOldVerBytes tally -
	// seeing as how it was just allocated.
	// DO NOT USE setTransID routine here because that routine
	// will adjust the tally.  The caller of this routine should call
	// setTransID to ensure that the tally is set appropriately.
	// This is the only place in the code where it is legal to set
	// ui64HighTransID without calling setTransID.

	m_ui64HighTransID = ~((FLMUINT64)0);
	m_uiUseCount = 0;
	m_ui16Flags = 0;
	m_ui16BlkSize = (FLMUINT16)uiBlockSize;
	
#ifdef FLM_DEBUG
	m_uiChecksum = 0;
	m_pUseList = NULL;
#endif
}

/****************************************************************************
Desc:	Allocate a cache block.  If we are at the cache limit, unused cache
		blocks will be replaced.  NOTE: This routine assumes that the block
		cache mutex is locked.
****************************************************************************/
RCODE F_BlockCacheMgr::allocBlock(
	F_Db *				pDb,
	F_CachedBlock **	ppSCacheRV)
{
	RCODE					rc = NE_SFLM_OK;
	F_Database *		pDatabase = pDb->getDatabase();
	FLMUINT				uiBlockSize = pDatabase->getBlockSize();
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pTmpSCache;
	F_CachedBlock *	pPrevSCache;

	// Quick check to see if there is a block in the free list that can be
	// re-used.  Start at the MRU end of the list so that if items in the
	// free list are only being used periodically, the items at the LRU end
	// will age out and the size of the list will be reduced.

	pSCache = m_pFirstFree;
	while (pSCache)
	{
		if (!pSCache->m_uiUseCount &&
			pSCache->getBlkSize() == uiBlockSize)
		{
			pSCache->unlinkFromFreeList();
			goto Reuse_Block;
		}
		pSCache = pSCache->m_pNextInDatabase;
	}

	// The intent of this little loop is to be optimistic and hope that
	// there is a block we can cannibalize or free without having to write
	// it.  If not, we will still allocate a new block and allow ourselves
	// to be temporarily over the cache limit.  In this case, the cache size
	// will be reduced only AFTER this new block is safely linked into cache.
	// This is necessary because we don't want two different threads allocating
	// memory for the same block.

	pTmpSCache = m_pLRUReplace;
	while( pTmpSCache && gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit()) 
	{
		// Need to save the pointer to the previous entry in the list because
		// we may end up unlinking it below, in which case we would have lost
		// the previous entry.

		pPrevSCache = pTmpSCache->m_pPrevInReplaceList;

		// See if the cache block can be replaced or freed.

		flmAssert( !pTmpSCache->m_ui16Flags);
		if (pTmpSCache->canBeFreed())
		{
			if (pTmpSCache->getBlkSize() == uiBlockSize)
			{
				pSCache = pTmpSCache;
				flmAssert( !pSCache->m_ui16Flags);
				pTmpSCache->unlinkCache( FALSE, NE_SFLM_OK);

				// We use a goto instead of a break because then
				// we don't have to do the additional test
				// down below.  We already know that pSCache
				// will be non-NULL.

				goto Reuse_Block;
			}
			else
			{
				// NOTE: This call will free the memory pointed to by
				// pTmpSCache.  Hence, pTmpSCache should NOT be used after
				// this point.

				pTmpSCache->unlinkCache( TRUE, NE_SFLM_OK);
			}
		}
		pTmpSCache = pPrevSCache;
	}

	// If we were not able to cannibalize an F_CachedBlock object,
	// allocate one.

	if (pSCache)
	{
Reuse_Block:

		flmAssert( !pSCache->m_pPrevInReplaceList);
		flmAssert( !pSCache->m_pNextInReplaceList);
		flmAssert( !pSCache->m_ui16Flags);
		flmAssert( !pSCache->m_uiUseCount);

		// If block is an old version, need to decrement the
		// Usage.uiOldVerBytes tally.

		if (pSCache->m_ui64HighTransID != ~((FLMUINT64)0))
		{
			FLMUINT	uiSize = pSCache->memSize();
			flmAssert( m_Usage.uiOldVerBytes >= uiSize);
			m_Usage.uiOldVerBytes -= uiSize;
			flmAssert( m_Usage.uiOldVerCount);
			m_Usage.uiOldVerCount--;
		}

		// If we are cannibalizing, be sure to reset certain fields.

		pSCache->m_ui16Flags = 0;
		pSCache->m_uiUseCount = 0;
#ifdef FLM_DEBUG
		pSCache->m_uiChecksum = 0;
#endif

		// Need to set high transaction ID to 0xFFFFFFFF.  This indicates that
		// the block is not currently counted in the Usage.uiOldVerBytes tally -
		// seeing as how it was just allocated.
		// DO NOT USE setTransID routine here because that routine
		// will adjust the tally.  The caller of this routine should call
		// setTransID to ensure that the tally is set appropriately.
		// This is the only place in the code where it is legal to set
		// ui64HighTransID without calling setTransID.

		pSCache->m_ui64HighTransID = ~((FLMUINT64)0);
	}
	else
	{
		if ((pSCache = new( uiBlockSize) F_CachedBlock( uiBlockSize)) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}

		m_Usage.uiCount++;
		m_Usage.uiByteCount += pSCache->memSize();
		
		if (shouldRehash( m_Usage.uiCount, m_uiNumBuckets))
		{
			if (checkHashFailTime( &m_uiHashFailTime))
			{
				if (RC_BAD( rc = rehash()))
				{
					goto Exit;
				}
			}
		}
	}

	*ppSCacheRV = pSCache;

	// Set use count to one so the block cannot be replaced.

	pSCache->useForThread( 0);

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/********************************************************************
Desc:	This converts a block header to native format.
*********************************************************************/
void convertBlkHdr(
	F_BLK_HDR *	pBlkHdr
	)
{

	// This routine should only be called on blocks that are NOT
	// currently in native format.

	flmAssert( blkIsNonNativeFormat( pBlkHdr));

	convert32( &pBlkHdr->ui32BlkAddr);
	convert32( &pBlkHdr->ui32PrevBlkInChain);
	convert32( &pBlkHdr->ui32NextBlkInChain);
	convert32( &pBlkHdr->ui32PriorBlkImgAddr);
	convert64( &pBlkHdr->ui64TransID);
	convert32( &pBlkHdr->ui32BlkCRC);
	convert16( &pBlkHdr->ui16BlkBytesAvail);
	if (blkIsBTree( pBlkHdr))
	{
		convert16( &(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16LogicalFile));
		convert16( &(((F_BTREE_BLK_HDR *)pBlkHdr)->ui16NumKeys));
	}
	blkSetNativeFormat( pBlkHdr);
}

/********************************************************************
Desc:	This converts a logical file header structure
*********************************************************************/
void convertLfHdr(
	F_LF_HDR *	pLfHdr)
{
	convert64( &pLfHdr->ui64NextRowId);
	convert32( &pLfHdr->ui32LfType);
	convert32( &pLfHdr->ui32RootBlkAddr);
	convert32( &pLfHdr->ui32LfNum);
	convert32( &pLfHdr->ui32EncDefNum);
}

/********************************************************************
Desc:	This converts a block header to native format.
*********************************************************************/
void convertBlk(
	FLMUINT		uiBlockSize,
	F_BLK_HDR *	pBlkHdr)
{
	// This routine should only be called on blocks that are NOT
	// currently in native format.

	convertBlkHdr( pBlkHdr);
	if (pBlkHdr->ui8BlkType == BT_LFH_BLK)
	{
		FLMUINT		uiPos = SIZEOF_STD_BLK_HDR;
		FLMUINT		uiEnd = blkGetEnd( uiBlockSize, SIZEOF_STD_BLK_HDR,
										pBlkHdr);
		F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)pBlkHdr +
														SIZEOF_STD_BLK_HDR);

		// Only one block type requires further conversion.

		while (uiPos + sizeof( F_LF_HDR) <= uiEnd)
		{
			convertLfHdr( pLfHdr);
			pLfHdr++;
			uiPos += sizeof( F_LF_HDR);
		}
	}
}

/********************************************************************
Desc:	This routine prepares a block for use after reading it in from
		disk.  It will convert the block to native format if necessary,
		and will also verify the CRC on the block.  We always want to
		convert the block if we can, so even if the CRC is bad or the
		block end is bad, we will attempt to do a convert if it is in
		non-native format.
*********************************************************************/
RCODE flmPrepareBlockForUse(
	FLMUINT			uiBlockSize,
	F_BLK_HDR *		pBlkHdr)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT32		ui32CRC;
	FLMUINT16		ui16BlkBytesAvail;
	FLMUINT			uiBlkEnd;
	FLMBOOL			bBadBlkEnd;

	// Determine if we should convert the block here.
	// Calculation of CRC should be on unconverted block.

	ui16BlkBytesAvail = pBlkHdr->ui16BlkBytesAvail;
	if (blkIsNonNativeFormat( pBlkHdr))
	{
		convert16( &ui16BlkBytesAvail);
	}
	
	if( (FLMUINT)ui16BlkBytesAvail > uiBlockSize - blkHdrSize( pBlkHdr))
	{
		uiBlkEnd = blkHdrSize( pBlkHdr);
		bBadBlkEnd = TRUE;
	}
	else
	{
		uiBlkEnd = (blkIsNewBTree( pBlkHdr)
						? uiBlockSize
						: uiBlockSize - (FLMUINT)ui16BlkBytesAvail);
		bBadBlkEnd = FALSE;
	}

	// CRC must be calculated BEFORE converting the block.

	ui32CRC = calcBlkCRC( pBlkHdr, uiBlkEnd);
	
	if( blkIsNonNativeFormat( pBlkHdr))
	{
		convertBlk( uiBlockSize, pBlkHdr);
	}
	
	if( ui32CRC != pBlkHdr->ui32BlkCRC || bBadBlkEnd)
	{
		rc = RC_SET( NE_SFLM_BLOCK_CRC);
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:	This routine attempts to read a block from disk.  It will
		attempt the specified number of times.
*********************************************************************/
RCODE F_Database::readTheBlock(
	F_Db *				pDb,
	TMP_READ_STATS *	pTmpReadStats,		// READ statistics.
	F_BLK_HDR *			pBlkHdr,				// Pointer to buffer where block is
													// to be read into.
	FLMUINT				uiFilePos,			// File position to be read from.  If
													// file position != block address, we
													// are reading from the log.
	FLMUINT				uiBlkAddress		// Block address that is to be read.
	)
{
	RCODE	  				rc = NE_SFLM_OK;
	FLMUINT				uiBytesRead;
	F_TMSTAMP			StartTime;
	FLMUINT64			ui64ElapMilli;
	SFLM_DB_STATS *	pDbStats = pDb->m_pDbStats;

	flmAssert( this == pDb->m_pDatabase);

	// We should NEVER be attempting to read a block address that is
	// beyond the current logical end of file.

	if (!FSAddrIsBelow( uiBlkAddress, pDb->m_uiLogicalEOF))
	{
		rc = RC_SET( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	// Read the block

	if (pDb->m_uiKilledTime)
	{
		rc = RC_SET( NE_SFLM_OLD_VIEW);
		goto Exit;
	}

	if (pTmpReadStats)
	{
		if (uiFilePos != uiBlkAddress)
		{
			pTmpReadStats->OldViewBlockReads.ui64Count++;
			pTmpReadStats->OldViewBlockReads.ui64TotalBytes +=
				m_uiBlockSize;
		}
		else
		{
			pTmpReadStats->BlockReads.ui64Count++;
			pTmpReadStats->BlockReads.ui64TotalBytes +=
				m_uiBlockSize;
		}
		ui64ElapMilli = 0;
		f_timeGetTimeStamp( &StartTime);
	}

	if (RC_BAD( rc = pDb->m_pSFileHdl->readBlock( uiFilePos,
								 m_uiBlockSize, pBlkHdr, &uiBytesRead)))
	{
		if (pDbStats)
		{
			pDbStats->uiReadErrors++;
		}

		if (rc == NE_FLM_IO_END_OF_FILE)
		{

			// Should only be possible when reading a root block,
			// because the root block address in the LFILE may be
			// a block that was just created by an update
			// transaction.

			flmAssert( pDb->m_uiKilledTime);
			rc = RC_SET( NE_SFLM_OLD_VIEW);
		}
		goto Exit;
	}

	if (pTmpReadStats)
	{
		flmAddElapTime( &StartTime, &ui64ElapMilli);
		if (uiFilePos != uiBlkAddress)
		{
			pTmpReadStats->OldViewBlockReads.ui64ElapMilli += ui64ElapMilli;
		}
		else
		{
			pTmpReadStats->BlockReads.ui64ElapMilli += ui64ElapMilli;
		}
	}

	if (uiBytesRead < m_uiBlockSize)
	{

		// Should only be possible when reading a root block,
		// because the root block address in the LFILE may be
		// a block that was just created by an update
		// transaction.

		flmAssert( pDb->m_uiKilledTime);
		rc = RC_SET( NE_SFLM_OLD_VIEW);
#ifdef FLM_DBG_LOG
		// Must make this call so we can be ensured that the
		// transaction ID in the block header has been
		// converted if need be.
		(void)flmPrepareBlockForUse( m_uiBlockSize, pBlkHdr);
#endif
	}
	else
	{
		rc = flmPrepareBlockForUse( m_uiBlockSize, pBlkHdr);
	}
	
	// Decrypt the block if it was encrypted
	
	if (RC_BAD( rc = decryptBlock( pDb->m_pDict, (FLMBYTE *)pBlkHdr)))
	{
		goto Exit;
	}

#ifdef FLM_DBG_LOG
	if (uiFilePos != uiBlkAddress)
	{
		flmDbgLogWrite( this, uiBlkAddress, uiFilePos,
						(FLMUINT)pBlkHdr->ui64TransID, "LGRD");
	}
	else
	{
		flmDbgLogWrite( this, uiBlkAddress, 0,
						pBlkHdr->ui64TransID, "READ");
	}
#endif

	if (RC_BAD( rc))
	{
		if (pTmpReadStats &&
			 (rc == NE_SFLM_BLOCK_CRC || rc == NE_SFLM_OLD_VIEW))
		{
			if (uiFilePos != uiBlkAddress)
			{
				pTmpReadStats->uiOldViewBlockChkErrs++;
			}
			else
			{
				pTmpReadStats->uiBlockChkErrs++;
			}
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Read a data block into cache.  This routine reads the requested
		version of a block into memory.  It follows links to previous
		versions of the block if necessary in order to do this.
****************************************************************************/
RCODE F_Database::readBlock(
	F_Db *				pDb,
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiFilePos,			// File position where we are to
													// start reading from.
	FLMUINT				uiBlkAddress,		// Address of block that is to
													// be read into cache.
	FLMUINT64			ui64NewerBlkLowTransID,
													// Low transaction ID of the last newer
													// version of the block.
													// NOTE: This has no meaning
													// when uiFilePos == uiBlkAddress.
	F_CachedBlock *	pSCache,				// Cache block to read the data
													// into.
	FLMBOOL *			pbFoundVerRV,		// Returns a flag to the caller to
													// tell it whether it found any
													// versions of the requested block
													// starting at the file position
													// that was passed in.
	FLMBOOL *			pbDiscardRV			// Returns a flag which, if TRUE,
													// tells the caller to discard
													// the block that was just read
													// in and set the high transaction ID
													// on the block that comes just
													// after it - because they are
													// the same version.
	)
{
	RCODE						rc = NE_SFLM_OK;
	F_BLK_HDR *				pBlkHdr = pSCache->m_pBlkHdr;
	F_CachedBlock *		pNextSCache;
	FLMBOOL					bMutexLocked = FALSE;
	SFLM_LFILE_STATS *	pLFileStats;
	SFLM_BLOCKIO_STATS *	pBlockIOStats;
	FLMBOOL					bIncrPriorImageCnt = FALSE;
	FLMBOOL					bIncrOldViewCnt = FALSE;
	TMP_READ_STATS			TmpReadStats;
	TMP_READ_STATS *		pTmpReadStats;
	SFLM_DB_STATS *		pDbStats = pDb->m_pDbStats;

	flmAssert( this == pDb->m_pDatabase);

	*pbFoundVerRV = FALSE;
	*pbDiscardRV = FALSE;

	if (pDbStats)
	{
		f_memset( &TmpReadStats, 0, sizeof( TmpReadStats));
		pTmpReadStats = &TmpReadStats;
	}
	else
	{
		pTmpReadStats = NULL;
	}

	// Read in the block from the database

	// Stay in a loop reading until we get an error or get the block

	for (;;)
	{
		if (pDbStats)
		{
			if (uiFilePos != uiBlkAddress)
			{
				bIncrPriorImageCnt = TRUE;
			}
			bIncrOldViewCnt = FALSE;
		}

		// Read and verify the block.

		if (RC_BAD( rc = readTheBlock( pDb, pTmpReadStats,
								pBlkHdr, uiFilePos, uiBlkAddress)))
		{
			goto Exit;
		}
		pBlkHdr->ui8BlkFlags &= ~(BLK_IS_BEFORE_IMAGE);

		// See if we can use the current version of the block, or if we
		// must go get a previous version.

		// See if we even got the block we thought we wanted.

		if ((FLMUINT)pBlkHdr->ui32BlkAddr != uiBlkAddress)
		{
			if (uiFilePos == uiBlkAddress)
			{
				rc = RC_SET( NE_SFLM_DATA_ERROR);
			}
			else
			{
				// Should only be possible when reading a root block,
				// because the root block address in the LFILE may be
				// a block that was just created by an update
				// transaction.

				flmAssert( pDb->m_uiKilledTime);
				rc = RC_SET( NE_SFLM_OLD_VIEW);
			}

			goto Exit;
		}

		// This flag is set to true to indicate that we found at least one
		// version of the requested block.  NOTE: This flag does NOT mean
		// that we found the specific version requested, only that we
		// found some version starting at the given address.

		*pbFoundVerRV = TRUE;

		// Check to see if the transaction range for the block we just read
		// overlaps the transaction range for next older version of the block
		// in the version list.  If the ranges overlap, the transaction ID on
		// each block had better be the same or we have a corruption.  If the
		// transaction IDs are the same, they are the same version of the block,
		// and we can discard the one we just read.

		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = TRUE;

Get_Next_Block:

		if ((pNextSCache = pSCache->m_pNextInVersionList) != NULL)
		{
			FLMUINT64	ui64TmpTransID1;
			FLMUINT64	ui64TmpTransID2;

			// If next block is still being read in, we must wait for
			// it to complete before looking at its transaction IDs.

			if (pNextSCache->m_ui16Flags & CA_READ_PENDING)
			{
				gv_SFlmSysData.pBlockCacheMgr->m_uiIoWaits++;
				if (RC_BAD( rc = flmWaitNotifyReq( 
					gv_SFlmSysData.hBlockCacheMutex, pDb->m_hWaitSem,
					&pNextSCache->m_pNotifyList, (void *)&pNextSCache)))
				{
					goto Exit;
				}

				// The thread doing the notify "uses" the cache block
				// on behalf of this thread to prevent the cache block
				// from being flushed after it unlocks the mutex.
				// At this point, since we have locked the mutex,
				// we need to release the cache block - because we
				// will put a "use" on it below.

				pNextSCache->releaseForThread();

				// See if we still have the same next block.

				goto Get_Next_Block;
			}

			// Check for overlapping trans ID ranges.  NOTE: At this
			// point, if we have an overlap, we know we have the version
			// of the block we need (see comment above).  Hence, we will
			// either break out of the loop at this point or goto exit
			// and return an error.

			ui64TmpTransID1 = pBlkHdr->ui64TransID;
			if (ui64TmpTransID1 <= pNextSCache->m_ui64HighTransID)
			{
				ui64TmpTransID2 = pNextSCache->getLowTransID();

				// If the low trans IDs on the two blocks are not equal
				// we have a corruption.

				if (ui64TmpTransID1 != ui64TmpTransID2)
				{
					rc = RC_SET( NE_SFLM_DATA_ERROR);
					goto Exit;
				}

				// The blocks are the same, discard one of them.

				*pbDiscardRV = TRUE;

				// Set the high trans ID on the block we are NOT discarding.
				// To find the version of the block we want, we have been
				// reading through a chain of blocks, from newer versions to
				// progressively older versions.  If uiFilePos == uiBlkAddress,
				// we are positioned on the most current version of the block.
				// In this case, the high trans ID for the block should be
				// set to the highest possible value.

				// If uiFilePos != uiBlkAddress, we are positioned on an older
				// version of the block.  The variable ui64NewerBlkLowTransID
				// contains the low transaction ID for a newer version of the
				// block we read just prior to reading this block.

				if (pDb->m_eTransType == SFLM_UPDATE_TRANS || m_bTempDb ||
					 uiFilePos == uiBlkAddress)
				{
					pNextSCache->setTransID( ~((FLMUINT64)0));
				}
				else
				{
					pNextSCache->setTransID( (ui64NewerBlkLowTransID - 1));
				}

				// When discard flag is TRUE, we need to go right to
				// exit, because we don't want to decrypt, do sanity
				// check, etc.  NOTE: mutex is still locked, and
				// we want it to remain locked - see code at Exit.

				goto Exit;
			}
		}

		// See if this version of the block is what we want

		if (pBlkHdr->ui64TransID <= pDb->m_ui64CurrTransID)
		{

			// Set the high trans ID on the block.  If we are in an
			// update transaction, or uiFilePos == uiBlkAddress, we
			// are positioned on the most current version of the block.
			// In this case, the high trans ID for the block should be
			// set to ~((FLMUINT64)0).  Otherwise we are positioned on an older
			// version of the block, and the block's high transaction ID
			// should be set to the newer block's low transaction ID minus one.

			if (pDb->m_eTransType == SFLM_UPDATE_TRANS || m_bTempDb ||
				 uiFilePos == uiBlkAddress)
			{
				pSCache->setTransID( ~((FLMUINT64)0));
			}
			else
			{
				pSCache->setTransID( (ui64NewerBlkLowTransID - 1));
			}
			f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = FALSE;
			break;
		}
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;

		// At this point, we know we are going to have to get a prior
		// version of the block.  In an update transaction, this is
		// indicative of a file corruption.

		if (pDb->m_eTransType != SFLM_READ_TRANS)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
			goto Exit;
		}

		// At this point, we know we are in a read transaction.  Save the
		// block's low trans ID.

		ui64NewerBlkLowTransID = pBlkHdr->ui64TransID;

		// See if there is a prior version of the block and determine whether
		// it's expected trans ID is in the range we need.
		// NOTE: If the prior version address is zero or is the same as our
		// current file position, there is no previous version of the block.

		if ((FLMUINT)pBlkHdr->ui32PriorBlkImgAddr == uiFilePos)
		{
			// Should only be possible when reading a root block,
			// because the root block address in the LFILE may be
			// a block that was just created by an update
			// transaction.

			flmAssert( pDb->m_uiKilledTime);
			rc = RC_SET( NE_SFLM_OLD_VIEW);
			goto Exit;
		}
		uiFilePos = (FLMUINT)pBlkHdr->ui32PriorBlkImgAddr;
		if (!uiFilePos)
		{
			// Should only be possible when reading a root block,
			// because the root block address in the LFILE may be
			// a block that was just created by an update
			// transaction.

			flmAssert( pDb->m_uiKilledTime);
			rc = RC_SET( NE_SFLM_OLD_VIEW);
			goto Exit;
		}
	}

	// Perform a sanity check on the block header.

	if ((FLMUINT)pBlkHdr->ui16BlkBytesAvail >
			m_uiBlockSize - blkHdrSize( pBlkHdr))
	{
		rc = RC_SET( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

Exit:

	// NOTE: When we are discarding the block, we CANNOT unlock
	// the mutex, because we have to take care of it on
	// the outside.  Mutex better be locked if we are discarding.

	if (*pbDiscardRV)
	{
		flmAssert( bMutexLocked);
	}
	else if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	// If we got an old view error, it has to be a corruption, unless we
	// were killed.

	if (rc == NE_SFLM_OLD_VIEW)
	{
		if (!pDb->m_uiKilledTime || pDb->m_eTransType == SFLM_UPDATE_TRANS ||
			 m_bTempDb)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
		}
	}

	// Increment cache fault statistics

	if (pDbStats)
	{
		if ((pLFileStats = pDb->getLFileStatPtr( pLFile)) == NULL)
		{
			pBlockIOStats = flmGetBlockIOStatPtr( pDbStats, NULL, (FLMBYTE *)pBlkHdr);
		}
		else if (RC_BAD( rc))
		{
			// Didn't really get a valid block, assign all statistics
			// gathered to the leaf block statistics.

			pBlockIOStats = &pLFileStats->LeafBlockStats;
		}
		else
		{
			pBlockIOStats = flmGetBlockIOStatPtr( pDbStats,
											pLFileStats, (FLMBYTE *)pBlkHdr);
		}

		if (pBlockIOStats)
		{
			pDbStats->bHaveStats = TRUE;
			if (pLFileStats)
			{
				pLFileStats->bHaveStats = TRUE;
			}

			flmUpdateDiskIOStats( &pBlockIOStats->BlockReads,
										 &TmpReadStats.BlockReads);

			flmUpdateDiskIOStats( &pBlockIOStats->OldViewBlockReads,
										 &TmpReadStats.OldViewBlockReads);

			pBlockIOStats->uiBlockChkErrs +=
					TmpReadStats.uiBlockChkErrs;

			pBlockIOStats->uiOldViewBlockChkErrs +=
					TmpReadStats.uiOldViewBlockChkErrs;
			if (rc == NE_SFLM_OLD_VIEW || bIncrOldViewCnt)
			{
				pBlockIOStats->uiOldViewErrors++;
			}
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Increment the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the block cache mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
void F_CachedBlock::dbgUseForThread(
	FLMUINT		uiThreadId)
{
	SCACHE_USE *	pUse;
	FLMUINT			uiMyThreadId = (FLMUINT)(!uiThreadId
														 ? (FLMUINT)f_threadId()
														 : uiThreadId);

	// If the use count is 0, make sure there are not entries
	// in the use list.

	if (!m_uiUseCount && m_pUseList != NULL)
	{
		return;
	}

	// First increment the overall use count for the block

	m_uiUseCount++;
	if (m_uiUseCount == 1)
	{
		gv_SFlmSysData.pBlockCacheMgr->m_uiBlocksUsed++;
		if (m_uiChecksum)
		{
			flmAssert( m_uiChecksum == computeChecksum());
		}
	}
	gv_SFlmSysData.pBlockCacheMgr->m_uiTotalUses++;

	// Now add the thread's usage record - or increment it if there
	// is already one there for the thread.

	// See if we already have this thread in the use list

	pUse = m_pUseList;
	while (pUse && pUse->uiThreadId != uiMyThreadId)
	{
		pUse = pUse->pNext;
	}

	if (!pUse)
	{
		if (RC_BAD( f_calloc( (FLMUINT)sizeof( SCACHE_USE),
							&pUse)))
		{
			return;
		}

		f_memset( pUse, 0, sizeof( SCACHE_USE));
		pUse->uiThreadId = uiMyThreadId;
		pUse->pNext = m_pUseList;
		m_pUseList = pUse;
	}

	pUse->uiUseCount++;
}
#endif

/****************************************************************************
Desc:	Decrement the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the block cache mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
void F_CachedBlock::dbgReleaseForThread( void)
{
	SCACHE_USE *	pUse;
	SCACHE_USE *	pPrevUse;
	FLMUINT			uiMyThreadId = (FLMUINT)f_threadId();

	// Find the thread's use

	pUse = m_pUseList;
	pPrevUse = NULL;
	while (pUse && pUse->uiThreadId != uiMyThreadId)
	{
		pPrevUse = pUse;
		pUse = pUse->pNext;
	}

	if (!pUse)
	{
		return;
	}

	m_uiUseCount--;
	gv_SFlmSysData.pBlockCacheMgr->m_uiTotalUses--;
	if (!m_uiUseCount)
	{
		m_uiChecksum = computeChecksum();
		gv_SFlmSysData.pBlockCacheMgr->m_uiBlocksUsed--;
		flmAssert( pUse->uiUseCount == 1);
	}

	// Free the use record if its count goes to zero

	pUse->uiUseCount--;
	if (!pUse->uiUseCount)
	{
		if (!pPrevUse)
		{
			m_pUseList = pUse->pNext;
		}
		else
		{
			pPrevUse->pNext = pUse->pNext;
		}
		f_free( &pUse);
	}
}
#endif

/****************************************************************************
Desc:	Read a data block into cache.  This routine takes care of allocating
		a cache block and reading the block from disk into memory.  NOTE:
		This routine assumes that the block cache mutex is locked.  It may
		unlock the block cache mutex long enough to do the read, but the
		mutex will still be locked when it exits.
****************************************************************************/
RCODE F_Database::readIntoCache(
	F_Db *				pDb,
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiBlkAddress,		// Address of block that is to
													// be read into cache.
	F_CachedBlock *	pPrevInVerList,	// Previous block in version list to
													// link the block to.
	F_CachedBlock *	pNextInVerList,	// Next block in version list to link
													// the block to.
	F_CachedBlock **	ppSCacheRV,			// Returns allocated cache block.
	FLMBOOL *			pbGotFromDisk)		// Returns TRUE if block was read
													// from disk
{
	RCODE					rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pTmpSCache;
	FNOTIFY *			pNotify;
	FLMUINT				uiFilePos;
	FLMUINT64			ui64NewerBlkLowTransID = 0;
	FLMBOOL				bFoundVer;
	FLMBOOL				bDiscard;

	flmAssert( this == pDb->m_pDatabase);

	*pbGotFromDisk = FALSE;

	// Lock the prev and next in place by incrementing their use
	// count.  We don't want allocBlock to use them.

	if (pPrevInVerList)
	{
		pPrevInVerList->useForThread( 0);
	}

	if (pNextInVerList)
	{
		pNextInVerList->useForThread( 0);
	}

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	rc = gv_SFlmSysData.pBlockCacheMgr->allocBlock( pDb, &pSCache);

	if (pPrevInVerList)
	{
		pPrevInVerList->releaseForThread();
	}

	if (pNextInVerList)
	{
		pNextInVerList->releaseForThread();
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	pSCache->m_uiBlkAddress = uiBlkAddress;

	// Set the "dummy" flag so that we won't incur the overhead of
	// linking the block into the replace list.  It would be removed
	// from the replace list almost immediately anyway, when the
	// "read pending" flag is set below.

	pSCache->m_ui16Flags |= CA_DUMMY_FLAG;

	// Link block into various lists

	if( pDb->m_uiFlags & FDB_DONT_POISON_CACHE)
	{
		if( !(pDb->m_uiFlags & FDB_BACKGROUND_INDEXING) ||
			(pLFile && pLFile->eLfType != SFLM_LF_INDEX))
		{
			pSCache->linkToGlobalListAsLRU();
		}
		else
		{
			pSCache->linkToGlobalListAsMRU();
		}
	}
	else
	{
		pSCache->linkToGlobalListAsMRU();
	}

	pSCache->linkToDatabase( this);
	if (!pPrevInVerList)
	{
		F_CachedBlock **	ppSCacheBucket;

		ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
												m_uiSigBitsInBlkSize, uiBlkAddress);
		uiFilePos = uiBlkAddress;
		if (pNextInVerList)
		{
			pNextInVerList->unlinkFromHashBucket( ppSCacheBucket);
		}
		pSCache->linkToHashBucket( ppSCacheBucket);
	}
	else
	{
		uiFilePos = pPrevInVerList->getPriorImageAddress();
		ui64NewerBlkLowTransID = pPrevInVerList->getLowTransID();
		pPrevInVerList->m_pNextInVersionList = pSCache;
		pPrevInVerList->verifyCache( 2400);
	}

	if (pNextInVerList)
	{
		pNextInVerList->m_pPrevInVersionList = pSCache;
		pNextInVerList->verifyCache( 2500);
	}

	pSCache->m_pPrevInVersionList = pPrevInVerList;
	pSCache->m_pNextInVersionList = pNextInVerList;
	pSCache->verifyCache( 2600);

	// Set the read-pending flag for this block.  This will force other
	// threads that need to read this block to wait for the I/O to
	// complete.

	pSCache->setFlags( CA_READ_PENDING);
	pSCache->m_ui16Flags &= ~CA_DUMMY_FLAG;
	gv_SFlmSysData.pBlockCacheMgr->m_uiPendingReads++;

	// Unlock the mutex and attempt to read the block into memory

	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);

	rc = readBlock( pDb, pLFile, uiFilePos, uiBlkAddress,
								ui64NewerBlkLowTransID,
								pSCache, &bFoundVer, &bDiscard);

	// NOTE: If the bDiscard flag is TRUE, the mutex will still be
	// locked.  If FALSE, we need to relock it.

	if (!bDiscard)
	{
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	}

	// Get a pointer to the notify list BEFORE discarding the cache
	// block - if we are going to discard - because pSCache can
	// change if we discard.

	pNotify = pSCache->m_pNotifyList;
	pSCache->m_pNotifyList = NULL;

	// Unset the read pending flag and reset the use count to zero.
	// Both of these actions should be done before doing a discard,
	// if a discard is going to be done.

	pSCache->clearFlags( CA_READ_PENDING);
	gv_SFlmSysData.pBlockCacheMgr->m_uiPendingReads--;
	pSCache->releaseForThread();

	// If we had no errors, take care of some other things

	if (RC_OK( rc))
	{
		// The bDiscard flag tells us that we should discard the
		// block that we just read and use the next block in the
		// version list - because they are the same version.

		if (bDiscard)
		{

			// NOTE: We are guaranteed that pSCache->m_pNextInVersionList
			// is non-NULL at this point, because when we set the
			// bDiscard flag to TRUE, it was non-NULL, and we know that
			// the mutex was NOT unlocked in that case.

			pTmpSCache = pSCache->m_pNextInVersionList;
			pSCache->unlinkCache( TRUE, NE_SFLM_OK);
			pSCache = pTmpSCache;
		}
		else
		{
			*pbGotFromDisk = TRUE;
		}
	}

	// Notify all of the waiters of the read result.
	// IMPORTANT NOTE: This should be the LAST thing that is
	// done except for unlink the block below in the case of
	// an error having occurred.

	ScaNotify( pNotify, pSCache, rc);

	// If we had a BAD rc, unlink the block from the lists it is in and
	// free the memory.

	if (RC_BAD( rc))
	{
		pSCache->unlinkCache( TRUE, NE_SFLM_OK);
		goto Exit;
	}

	*ppSCacheRV = pSCache;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine frees all cache blocks that have been modified by
		the update transaction.  This routine is called whenever a
		transaction is to be aborted.
****************************************************************************/
void F_Database::freeModifiedBlocks(
	FLMUINT64	ui64CurrTransId)
{
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pNextSCache;
	FLMBOOL				bFirstPass = TRUE;
	FLMBOOL				bFreedAll;

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

	// Unlink all log blocks and reset their flags so they
	// won't be marked as needing to be written to disk.

	unlinkTransLogBlocks();

Do_Free_Pass:

	pSCache = m_pSCacheList;
	flmAssert( !m_pPendingWriteList);
	bFreedAll = TRUE;
	while (pSCache)
	{

		// If the high transaction ID on the block is one less than this
		// transaction's ID, the block is the most current block.  Therefore,
		// its high transaction ID should be reset to ~((FLMUINT64)0).

		if (pSCache->m_ui64HighTransID == ui64CurrTransId - 1)
		{
			pSCache->setTransID( ~((FLMUINT64)0));

			// Need to link blocks that become the current version again
			// into the file log list if they are dirty.  linkToLogList
			// will check to see if the block has already been logged.  If it has,
			// it won't be linked into the list.
			// NOTE: If the blocks were in the "new" list originally, we don't take
			// the time to put them back into that list because they would have to
			// be inserted in order.  They will still get written out eventually, but
			// they won't be written out by the reduceNewBlocks call.

			if (pSCache->m_ui16Flags & CA_DIRTY)
			{
				pSCache->linkToLogList();
			}
		}
		else if (pSCache->m_ui64HighTransID == ~((FLMUINT64)0) &&
					pSCache->getLowTransID() >= ui64CurrTransId &&
					!(pSCache->m_ui16Flags & CA_READ_PENDING))

		{
			pNextSCache = pSCache->m_pNextInDatabase;

			// Another thread might have a temporary "use" on this
			// block.  Unlock the mutex long enough to allow the
			// other thread(s) to get rid of their "uses".  Then start
			// from the top of the list again.

			if (pSCache->m_uiUseCount)
			{

				// Don't want to unlock the mutex during the first pass
				// because it opens the door to the prior version of one of
				// these modified blocks being removed from cache before we
				// have a chance to reset its ui64HighTransID back to
				// ~((FLMUINT64)0).
				// During the first pass, we want to get through all of the
				// blocks so that the code up above will get exercised for
				// each such block.

				if (bFirstPass)
				{
					bFreedAll = FALSE;
					pSCache = pNextSCache;
					continue;
				}
				else
				{
					f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
					f_sleep( 10);
					f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
					pSCache = m_pSCacheList;
					continue;
				}
			}
			else
			{
#ifdef FLM_DEBUG
				F_CachedBlock *		pResetDirty = NULL;
#endif

				// Unset dirty flag so we don't get an assert in unlinkCache.

				if (pSCache->m_ui16Flags & CA_DIRTY)
				{
#ifdef FLM_DBG_LOG
					FLMUINT16	ui16OldFlags = pSCache->m_ui16Flags;
#endif
					flmAssert( this == pSCache->m_pDatabase);
					pSCache->unsetDirtyFlag();
#ifdef FLM_DBG_LOG
					pSCache->logFlgChange( ui16OldFlags, 'G');
#endif
				}

#ifdef FLM_DEBUG
				// If m_pNextInVersionList is dirty it is because
				// ScaUnlinkTransLogBlocks changed the WAS_DIRTY flag to
				// DIRTY.  If we don't temporarily clear the DIRTY flag,
				// unlinkCache will assert.

				if( pSCache->m_pNextInVersionList &&
					(pSCache->m_pNextInVersionList->m_ui16Flags & CA_DIRTY))
				{
					pResetDirty = pSCache->m_pNextInVersionList;
					pResetDirty->m_ui16Flags &= ~CA_DIRTY;
				}
#endif

				pSCache->unlinkCache( TRUE, NE_SFLM_OK);

#ifdef FLM_DEBUG
				if( pResetDirty)
				{
					pResetDirty->m_ui16Flags |= CA_DIRTY;
				}
#endif
				pSCache = pNextSCache;
				continue;
			}
		}

		pSCache = pSCache->m_pNextInDatabase;
	}

	if (!bFreedAll && bFirstPass)
	{
		bFirstPass = FALSE;
		goto Do_Free_Pass;
	}

	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:	Prepares a block to be written out.  Calculates the checksum and
		converts the block to native format if not currently in native
		format.
****************************************************************************/
RCODE flmPrepareBlockToWrite(
	FLMUINT		uiBlockSize,
	F_BLK_HDR *	pBlkHdr)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiBlkLen;

	if ((FLMUINT)pBlkHdr->ui16BlkBytesAvail >
					uiBlockSize - blkHdrSize( pBlkHdr))
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BLOCK_CRC);
		goto Exit;
	}
	uiBlkLen = (blkIsNewBTree( pBlkHdr)
					? uiBlockSize
					: uiBlockSize - (FLMUINT)pBlkHdr->ui16BlkBytesAvail);

	// Block should already be in native format.

	flmAssert( !blkIsNonNativeFormat( pBlkHdr));

	// Calculate and set the block CRC.

	pBlkHdr->ui32BlkCRC = calcBlkCRC( pBlkHdr, uiBlkLen);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine writes all blocks in the sorted list, or releases them.
		It attempts to write as many as it can that are currently
		contiguous.
		NOTE: This routine assumes that the block cache mutex is NOT locked.
****************************************************************************/
RCODE F_Database::writeSortedBlocks(
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiMaxDirtyCache,
	FLMUINT *			puiDirtyCacheLeft,
	FLMBOOL *			pbForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMUINT				uiNumSortedBlocks,
	FLMBOOL *			pbWroteAll)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiStartBlkAddr = 0;
	FLMUINT				uiLastBlkAddr = 0;
	FLMUINT				uiContiguousBlocks = 0;
	FLMUINT				uiNumSortedBlocksProcessed;
	FLMUINT				uiBlockCount;
	F_CachedBlock *	ppContiguousBlocks[ FLM_MAX_IO_BUFFER_BLOCKS];
	FLMBOOL				bBlockDirty[ FLM_MAX_IO_BUFFER_BLOCKS];
	FLMUINT				uiOffset;
	FLMUINT				uiTmpOffset;
	FLMUINT				uiLoop;
	FLMUINT				uiStartOffset;
	FLMUINT				uiCopyLen;
	FLMBOOL				bForceCheckpoint = *pbForceCheckpoint;
	F_CachedBlock *	pSCache;
	IF_IOBuffer *		pIOBuffer = NULL;
	FLMBYTE *			pucBuffer;

	uiOffset = 0;
	for (;;)
	{

		// Mutex must be locked to test dirty flags.

		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}

		// See how many we have that are contiguous

		uiContiguousBlocks = 0;
		uiNumSortedBlocksProcessed = 0;
		uiStartOffset = uiTmpOffset = uiOffset;
		while (uiTmpOffset < uiNumSortedBlocks)
		{
			pSCache = m_ppBlocksDone [uiTmpOffset];

			// See if this block is still eligible for writing out.
			// If so, mark it as write pending and add to list.

			flmAssert( pSCache->m_ui16Flags & CA_DIRTY);

			// Is it contiguous with last block or the first block?

			if (!uiContiguousBlocks ||
				 (FSGetFileNumber( uiLastBlkAddr) ==
				  FSGetFileNumber( pSCache->m_uiBlkAddress) &&
				  uiLastBlkAddr + m_uiBlockSize == pSCache->m_uiBlkAddress))
			{

				// Block is either first block or contiguous with
				// last block.

Add_Contiguous_Block:
				uiLastBlkAddr = pSCache->m_uiBlkAddress;

				// Set first block address if this is the first one.

				if (!uiContiguousBlocks)
				{
					uiStartBlkAddr = pSCache->m_uiBlkAddress;
				}
				ppContiguousBlocks [uiContiguousBlocks] = pSCache;
				bBlockDirty [uiContiguousBlocks++] = TRUE;
				uiNumSortedBlocksProcessed++;
				if (uiContiguousBlocks == FLM_MAX_IO_BUFFER_BLOCKS)
				{
					break;
				}
				uiTmpOffset++;
			}
			else
			{
				FLMUINT	uiGap;
				FLMUINT	uiSaveContiguousBlocks;
				FLMUINT	uiBlkAddress;

				// Ran into a non-contiguous block.  If we are not forcing
				// a checkpoint, take what we have and write it out.
				// If we are forcing a checkpoint, see if we can fill the
				// gap with other blocks in cache.

				if (!bForceCheckpoint)
				{
					break;
				}

				// See if the gap is worth trying to fill.

				// If blocks are in different files, cannot fill gap.

				if (FSGetFileNumber( uiLastBlkAddr) !=
					 FSGetFileNumber( pSCache->m_uiBlkAddress))
				{
					break;
				}

				// If 32K won't encompass both blocks, not worth it to try
				// and fill the gap.

				uiGap = FSGetFileOffset( pSCache->m_uiBlkAddress) -
							FSGetFileOffset( uiLastBlkAddr) - m_uiBlockSize;
				if (uiGap > 32 * 1024 - (m_uiBlockSize * 2))
				{
					break;
				}

				// If the gap would run us off the maximum blocks to
				// request, don't try to fill it.

				if (uiContiguousBlocks + uiGap / m_uiBlockSize + 1 >
						FLM_MAX_IO_BUFFER_BLOCKS)
				{
					break;
				}

				uiSaveContiguousBlocks = uiContiguousBlocks;
				uiBlkAddress = uiLastBlkAddr + m_uiBlockSize;
				while (uiBlkAddress != pSCache->m_uiBlkAddress)
				{
					F_CachedBlock **	ppSCacheBucket;
					F_CachedBlock *	pTmpSCache;

					ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
													m_uiSigBitsInBlkSize, uiBlkAddress);
					pTmpSCache = *ppSCacheBucket;
					while (pTmpSCache &&
							 (pTmpSCache->m_uiBlkAddress != uiBlkAddress ||
							  pTmpSCache->m_pDatabase != this))
					{
						pTmpSCache = pTmpSCache->m_pNextInHashBucket;
					}
					if (!pTmpSCache ||
						 (pTmpSCache->m_ui16Flags &
							(CA_READ_PENDING | CA_WRITE_PENDING | CA_WRITE_INHIBIT)) ||
						 pTmpSCache->m_ui64HighTransID != ~((FLMUINT64)0))
					{
						break;
					}
					ppContiguousBlocks [uiContiguousBlocks] = pTmpSCache;

					bBlockDirty [uiContiguousBlocks++] =
						(pTmpSCache->m_ui16Flags & CA_DIRTY)
						? TRUE
						: FALSE;

					pTmpSCache->useForThread( 0);
					uiBlkAddress += m_uiBlockSize;
				}

				// If we couldn't fill in the entire gap, we are done.

				if (uiBlkAddress != pSCache->m_uiBlkAddress)
				{

					// Release the blocks we obtained in the above loop.

					while (uiContiguousBlocks > uiSaveContiguousBlocks)
					{
						uiContiguousBlocks--;
						ppContiguousBlocks [uiContiguousBlocks]->releaseForThread();
					}
					break;
				}
				else
				{
					goto Add_Contiguous_Block;
				}
			}
		}

		// At this point, we know how many are contiguous.

		if (!uiContiguousBlocks)
		{
			flmAssert( uiOffset == uiNumSortedBlocks);
			break;
		}

		if (bMutexLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = FALSE;
		}

		// Ask for a buffer of the size needed.

		flmAssert( pIOBuffer == NULL);
		if (RC_BAD( rc = m_pBufferMgr->getBuffer( 
			uiContiguousBlocks * m_uiBlockSize, &pIOBuffer))) 
		{
			goto Exit;
		}
		pIOBuffer->setCompletionCallback( scaWriteComplete, pDbStats);

		// Callback will now take care of everything between
		// uiStartOffset and uiStartOffset + uiNumSortedBlocksProcessed - 1
		// inclusive, as well as any non-dirty blocks that were
		// put in for filler.

		flmAssert( uiNumSortedBlocksProcessed);
		uiOffset = uiStartOffset + uiNumSortedBlocksProcessed;
		uiBlockCount = uiContiguousBlocks;

		// Must set to zero so we don't process ppContiguousBlocks
		// at exit.

		uiContiguousBlocks = 0;

		// Set write pending on all of the blocks before unlocking
		// the mutex.
		// Then unlock the mutex and come out and copy
		// the blocks.

		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}

		for (uiLoop = 0; uiLoop < uiBlockCount; uiLoop++)
		{
			pSCache = ppContiguousBlocks [uiLoop];
			if (bBlockDirty [uiLoop])
			{
				flmAssert( pSCache->m_ui16Flags & CA_DIRTY);
				flmAssert( !(pSCache->m_ui16Flags & CA_WRITE_INHIBIT));
				pSCache->setFlags( CA_WRITE_PENDING);
				flmAssert( *puiDirtyCacheLeft >= m_uiBlockSize);
				(*puiDirtyCacheLeft) -= m_uiBlockSize;
				pSCache->unlinkFromDatabase();
				pSCache->linkToDatabase( this);
			}
			else
			{
				flmAssert( !(pSCache->m_ui16Flags & CA_DIRTY));
			}

			// Set callback data so we will release these and clear
			// the pending flag if we don't do the I/O.

			pIOBuffer->addCallbackData( pSCache);
		}

		if (bMutexLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = FALSE;
		}

		// Copy blocks into the IO buffer.

		pucBuffer = pIOBuffer->getBufferPtr();
		for (uiLoop = 0;
			  uiLoop < uiBlockCount;
			  uiLoop++, pucBuffer += m_uiBlockSize)
		{
			pSCache = ppContiguousBlocks [uiLoop];

			// Copy data from block to the write buffer

			uiCopyLen = blkGetEnd( m_uiBlockSize,
								blkHdrSize( pSCache->m_pBlkHdr),
								pSCache->m_pBlkHdr);
			f_memcpy( pucBuffer, pSCache->m_pBlkHdr, uiCopyLen);

			// Encrypt the block if needed

			if (RC_BAD( rc = encryptBlock( m_pDictList,
													 pucBuffer)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = flmPrepareBlockToWrite( m_uiBlockSize,
										(F_BLK_HDR *)pucBuffer)))
			{
				goto Exit;
			}
		}
		
		pSFileHdl->setMaxAutoExtendSize( m_uiMaxFileSize);
		pSFileHdl->setExtendSize( m_uiFileExtendSize);
	
		rc = pSFileHdl->writeBlock( uiStartBlkAddr, 
			pIOBuffer->getBufferSize(), pIOBuffer);
			
		pIOBuffer->Release();
		pIOBuffer = NULL;
		
		if( RC_BAD( rc))
		{
			if (pDbStats)
			{
				pDbStats->bHaveStats = TRUE;
				pDbStats->uiWriteErrors++;
			}
			
			goto Exit;
		}

		// See if we should give up our write lock.  Will do so if we
		// are not forcing a checkpoint and we have not exceeded the
		// maximum time since the last checkpoint AND we have gotten
		// below the maximum dirty blocks allowed.

		if (!bForceCheckpoint && bIsCPThread)
		{
			FLMUINT	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

			if (scaSeeIfForceCheckpoint( uiCurrTime, m_uiLastCheckpointTime,
													m_pCPInfo))
			{
				bForceCheckpoint = TRUE;
			}
			else
			{
				if (m_pWriteLockObj->getWaiterCount() &&
					 *puiDirtyCacheLeft <= uiMaxDirtyCache)
				{

					// Break out of loop and finish writing whatever
					// we have pending.

					*pbWroteAll = FALSE;
					goto Exit;
				}
			}
		}
	}

Exit:

	// Unuse any blocks that did not get processed.

	while (uiOffset < uiNumSortedBlocks)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}
		m_ppBlocksDone[ uiOffset]->releaseForThread();
		uiOffset++;
	}

	while (uiContiguousBlocks)
	{
		uiContiguousBlocks--;

		// Only release the non-dirty blocks, because dirty blocks
		// will have been taken care of in the loop above.

		if (!bBlockDirty [uiContiguousBlocks])
		{
			if (!bMutexLocked)
			{
				f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
				bMutexLocked = TRUE;
			}

			ppContiguousBlocks [uiContiguousBlocks]->releaseForThread();
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	// If we allocated a write buffer, but did not do a write with it,
	// still need to call notifyComplete so that our callback function
	// will be called and the F_CachedBlock objects will be released.

	if (pIOBuffer)
	{
		flmAssert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}

	*pbForceCheckpoint = bForceCheckpoint;
	return( rc);
}

/****************************************************************************
Desc:	This routine writes all dirty cache blocks to disk.  This routine
		is called when a transaction is committed.
****************************************************************************/
RCODE F_Database::flushDirtyBlocks(
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiMaxDirtyCache,
	FLMBOOL				bForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMBOOL *			pbWroteAll)
{
	RCODE					rc = NE_SFLM_OK;
	RCODE					rc2;
	F_CachedBlock *	pSCache;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiSortedBlocks = 0;
	FLMUINT				uiBlockCount = 0;
	FLMBOOL				bWasForcing;
	FLMBOOL				bWriteInhibited;
	FLMUINT				uiDirtyCacheLeft;
	FLMBOOL				bAllocatedAll = FALSE;

	flmAssert( !m_uiLogCacheCount);

	if (m_pCPInfo)
	{
		lockMutex();
		m_pCPInfo->bWritingDataBlocks = TRUE;
		unlockMutex();
	}

	flmAssert( !m_pPendingWriteList);

	uiDirtyCacheLeft = m_uiDirtyCacheCount * m_uiBlockSize;

	// If we are forcing a checkpoint, pre-allocate an array big enough
	// to hold all of the dirty blocks.  We do this so we won't end up
	// continually re-allocating the array in the loop below.

Force_Checkpoint:

	if (bForceCheckpoint)
	{
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		pSCache = m_pSCacheList;
		uiBlockCount = 0;
		while (pSCache && (pSCache->m_ui16Flags & CA_DIRTY))
		{
			uiBlockCount++;
			pSCache = pSCache->m_pNextInDatabase;
		}
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);

		bAllocatedAll = TRUE;
		if (uiBlockCount > m_uiBlocksDoneArraySize * 2)
		{
			if (RC_BAD( rc = allocBlocksArray(
										(uiBlockCount + 1) / 2, TRUE)))
			{
				if (rc == NE_SFLM_MEM)
				{
					bAllocatedAll = FALSE;
					rc = NE_SFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}
	}

	for (;;)
	{

		flmAssert( !bMutexLocked);
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = TRUE;

		// Create a list of blocks to write out - MAX_BLOCKS_TO_SORT at most.

		pSCache = m_pSCacheList;
		uiSortedBlocks = 0;
		for (;;)
		{
			FLMUINT	uiPrevBlkAddress;

			if (bAllocatedAll)
			{
				if (uiSortedBlocks == uiBlockCount)
				{
#ifdef FLM_DEBUG
					// Better not be any dirty blocks after the last one.

					if (uiSortedBlocks)
					{
						pSCache = m_ppBlocksDone [uiSortedBlocks - 1];
						flmAssert( !pSCache->m_pNextInDatabase ||
									  !(pSCache->m_pNextInDatabase->m_ui16Flags & CA_DIRTY));
					}
#endif
					break;
				}
				flmAssert( pSCache && (pSCache->m_ui16Flags & CA_DIRTY));
			}
			else
			{
				if (!pSCache || !(pSCache->m_ui16Flags & CA_DIRTY) ||
					 uiSortedBlocks == MAX_BLOCKS_TO_SORT)
				{
					break;
				}
			}

			flmAssert( !(pSCache->m_ui16Flags & CA_WRITE_PENDING));
			uiPrevBlkAddress = pSCache->getPriorImageAddress();

			bWriteInhibited = FALSE;
			if (pSCache->m_ui16Flags & CA_WRITE_INHIBIT)
			{
				// When the checkpoint thread is running there is no need to
				// inhibit writes - because it is not possible for an updater
				// to be making any changes at this point.  However,
				// the inhibit writes bit may still be set because the
				// thread that originally did the update transaction never
				// got the use count to go to zero (due to a reader that
				// simultaneously had a use) and hence, the inhibit bit
				// has never been unset.  It is only unset when we see
				// the use count go to zero.

				if (bIsCPThread)
				{
					pSCache->clearFlags( CA_WRITE_INHIBIT);
				}
				else
				{
					bWriteInhibited = TRUE;
				}
			}

			// Skip blocks that are write inhibited or that have
			// not been properly logged yet.

			if (bWriteInhibited ||
				 (!uiPrevBlkAddress && pSCache->m_pNextInVersionList))
			{
				flmAssert( !bForceCheckpoint);
			}
			else
			{
				if (uiSortedBlocks == m_uiBlocksDoneArraySize * 2)
				{
					if (RC_BAD( rc = allocBlocksArray( 0, TRUE)))
					{
						goto Exit;
					}
				}

				// Keep list of blocks to process

				m_ppBlocksDone [uiSortedBlocks++] = pSCache;

				// Must use to keep from going away.

				pSCache->useForThread( 0);
			}
			pSCache = pSCache->m_pNextInDatabase;
		}
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;

		// Sort the list of blocks by block address.

		if (uiSortedBlocks)
		{
			if (uiSortedBlocks > 1)
			{
				f_qsort( m_ppBlocksDone, 
					0, uiSortedBlocks - 1, scaSortCompare, scaSortSwap);
			}
			bWasForcing = bForceCheckpoint;
			rc = writeSortedBlocks( pDbStats, pSFileHdl,
									uiMaxDirtyCache, &uiDirtyCacheLeft,
									&bForceCheckpoint, bIsCPThread,
									uiSortedBlocks, pbWroteAll);
		}
		else
		{
			goto Exit;
		}

		// Set to zero so won't get released at exit.

		uiSortedBlocks = 0;

		if (!bIsCPThread || RC_BAD( rc) || !(*pbWroteAll))
		{
			goto Exit;
		}
		if (bForceCheckpoint)
		{
			if (!bWasForcing)
			{

				// Needs to be the checkpoint thread that does this
				// because all of the log blocks have to have been
				// written out - which the checkpoint thread does
				// before calling this routine.

				flmAssert( bIsCPThread);
				goto Force_Checkpoint;
			}
			else if (bAllocatedAll)
			{
				// We did all of the blocks in one pass, so
				// break out of the loop.

				goto Exit;
			}
		}
	}

Exit:

	// Release any blocks that are still used.

	while (uiSortedBlocks)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}
		uiSortedBlocks--;

		// Release any blocks we didn't process through.

		pSCache = m_ppBlocksDone [uiSortedBlocks];
		pSCache->releaseForThread();
	}

	// Need to finish up any async writes.

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;
	}

	// Wait for writes to complete.

	if (RC_BAD( rc2 = m_pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	flmAssert( !m_pPendingWriteList);

	// Better not be any incomplete writes at this point.

	flmAssert( !m_pBufferMgr->isIOPending());

	// Don't keep around a large block array if we happened to
	// allocate one that is bigger than our normal size.  It may
	// be huge because we were forcing a checkpoint.

	if (m_uiBlocksDoneArraySize > MAX_BLOCKS_TO_SORT)
	{
		f_free( &m_ppBlocksDone);
		m_uiBlocksDoneArraySize = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Database::reduceDirtyCache(
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl)
{
	RCODE					rc = NE_SFLM_OK;
	RCODE					rc2;
	F_CachedBlock *	pSCache;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiDirtyCacheLeft;
	FLMUINT				uiSortedBlocks = 0;
	FLMUINT				uiInhibitCount;
	FLMBOOL				bForceCheckpoint;
	FLMBOOL				bWroteAll;

	flmAssert( !m_uiLogCacheCount);

	flmAssert( !m_pPendingWriteList);

	if( m_uiDirtyCacheCount > m_uiBlocksDoneArraySize * 2)
	{
		if( RC_BAD( rc = allocBlocksArray( 
			(m_uiDirtyCacheCount + 1) / 2, TRUE)))
		{
			goto Exit;
		}
	}

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = TRUE;

	pSCache = m_pSCacheList;
	uiSortedBlocks = 0;
	uiInhibitCount = 0;

	while( pSCache &&
		(pSCache->m_ui16Flags & CA_DIRTY))
	{
		if( (pSCache->m_ui16Flags & CA_WRITE_INHIBIT) != 0)
		{
			uiInhibitCount++;
		}
		else
		{
			flmAssert( uiSortedBlocks < m_uiDirtyCacheCount);
			m_ppBlocksDone[ uiSortedBlocks++] = pSCache;
			pSCache->useForThread( 0);
		}

		pSCache = pSCache->m_pNextInDatabase;
	}

	flmAssert( uiSortedBlocks + uiInhibitCount == m_uiDirtyCacheCount);

	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = FALSE;

	if( !uiSortedBlocks)
	{
		goto Exit;
	}

	if( uiSortedBlocks > 1)
	{
		f_qsort( m_ppBlocksDone, 
			0, uiSortedBlocks - 1, scaSortCompare, scaSortSwap);
	}

	uiDirtyCacheLeft = m_uiDirtyCacheCount * m_uiBlockSize;
	bForceCheckpoint = FALSE;
	bWroteAll = TRUE;

	rc = writeSortedBlocks( pDbStats, pSFileHdl, 0, &uiDirtyCacheLeft,
			&bForceCheckpoint, FALSE, uiSortedBlocks, &bWroteAll);
	
	uiSortedBlocks = 0;

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	while( uiSortedBlocks)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}

		uiSortedBlocks--;

		// Release any blocks we didn't process through.

		pSCache = m_ppBlocksDone[ uiSortedBlocks];
		pSCache->releaseForThread();
	}

	// Need to finish up any async writes.

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;
	}

	// Wait for writes to complete.

	if( RC_BAD( rc2 = m_pBufferMgr->waitForAllPendingIO()))
	{
		if( RC_OK( rc))
		{
			rc = rc2;
		}
	}

	flmAssert( !m_pPendingWriteList);

	// Better not be any incomplete writes at this point.

	flmAssert( !m_pBufferMgr->isIOPending());

	// Don't keep around a large block array if we happened to
	// allocate one that is bigger than our normal size.  It may
	// be huge because we were forcing a checkpoint.

	if( m_uiBlocksDoneArraySize > MAX_BLOCKS_TO_SORT)
	{
		f_free( &m_ppBlocksDone);
		m_uiBlocksDoneArraySize = 0;
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine writes new cache blocks to disk.  The purpose of this
		routine is to allow cache to be reduced as quickly as possible.
		This is best accomplished by flushing blocks that are contiguous
		before resorting to writing out non-contiguous blocks.  The "new"
		block list attempts to accomplish this by keeping an ordered list
		of most of the blocks that have been created since the last checkpoint.
		The list may not contain all of the new blocks if a transaction, which
		modified blocks, was aborted since the last checkpoint completed.
		In this case, the blocks would have been removed from the new list
		when new versions of the blocks were created.  Upon aborting, the
		blocks that were originally in the new list are not put back into the
		list because of the cost associated with finding their correct places
		in the list.  Even though these blocks aren't in the new list anymore,
		they are still marked as being dirty and will written out eventually.
****************************************************************************/
RCODE F_Database::reduceNewBlocks(
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT *			puiBlocksFlushed)
{
	RCODE					rc = NE_SFLM_OK;
	RCODE					rc2;
	F_CachedBlock *	pSCache;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiSortedBlocks = 0;
	FLMUINT				uiDirtyCacheLeft;
	FLMUINT				uiBlocksFlushed = 0;

	flmAssert( !m_uiLogCacheCount);
	if (m_pCPInfo)
	{
		lockMutex();
		m_pCPInfo->bWritingDataBlocks = TRUE;
		unlockMutex();
	}

	flmAssert( !m_pPendingWriteList);
	uiDirtyCacheLeft = m_uiDirtyCacheCount * m_uiBlockSize;

	if (m_uiBlocksDoneArraySize < MAX_BLOCKS_TO_SORT)
	{
		if (RC_BAD( rc = allocBlocksArray( MAX_BLOCKS_TO_SORT, TRUE)))
		{
			// If the array size is non-zero, but we were unable to allocate
			// the size we wanted, we'll just be content to output as many
			// blocks as possible with the existing size of the array

			if( rc == NE_SFLM_MEM && m_uiBlocksDoneArraySize)
			{
				rc = NE_SFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	// Create a list of blocks to write out

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = TRUE;
	pSCache = m_pFirstInNewList;
	uiSortedBlocks = 0;
	for (;;)
	{
		FLMUINT	uiPrevBlkAddress;

		if (!pSCache || uiSortedBlocks == m_uiBlocksDoneArraySize)
		{
			break;
		}

		flmAssert( !(pSCache->m_ui16Flags & CA_WRITE_PENDING));
		flmAssert( pSCache->m_ui16Flags & (CA_DIRTY | CA_IN_NEW_LIST));

		uiPrevBlkAddress = pSCache->getPriorImageAddress();

		// Skip blocks that are write inhibited

		if( pSCache->m_ui16Flags & CA_WRITE_INHIBIT)
		{
			pSCache = pSCache->m_pNextInReplaceList;
			continue;
		}

		// Keep list of blocks to process

		m_ppBlocksDone [uiSortedBlocks++] = pSCache;

		// Must use to keep from going away.

		pSCache->useForThread( 0);
		pSCache = pSCache->m_pNextInReplaceList;
	}

	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = FALSE;

	if (uiSortedBlocks)
	{
		FLMBOOL	bForceCheckpoint = FALSE;
		FLMBOOL	bDummy;

		rc = writeSortedBlocks( pDbStats, pSFileHdl,
								~((FLMUINT)0), &uiDirtyCacheLeft,
								&bForceCheckpoint, FALSE,
								uiSortedBlocks, &bDummy);

		if( RC_OK( rc))
		{
			uiBlocksFlushed += uiSortedBlocks;
		}
	}
	else
	{
		goto Exit;
	}

	// Set to zero so won't get released at exit.

	uiSortedBlocks = 0;

Exit:

	// Release any blocks that are still used.

	while (uiSortedBlocks)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}
		uiSortedBlocks--;

		// Release any blocks we didn't process through.

		pSCache = m_ppBlocksDone [uiSortedBlocks];

#ifdef FLM_DEBUG
		if( RC_OK( rc))
		{
			flmAssert( !(pSCache->m_ui16Flags & CA_IN_NEW_LIST));
		}
#endif

		pSCache->releaseForThread();
	}

	// Need to finish up any async writes.

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
		bMutexLocked = FALSE;
	}

	// Wait for writes to complete.

	if (RC_BAD( rc2 = m_pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	flmAssert( !m_pPendingWriteList);

	// Better not be any incomplete writes at this point.

	flmAssert( !m_pBufferMgr->isIOPending());

	// Don't keep around a large block array if we happened to
	// allocate one that is bigger than our normal size.  It may
	// be huge because we were forcing a checkpoint.

	if (m_uiBlocksDoneArraySize > MAX_BLOCKS_TO_SORT)
	{
		f_free( &m_ppBlocksDone);
		m_uiBlocksDoneArraySize = 0;
	}

	if (puiBlocksFlushed)
	{
		*puiBlocksFlushed = uiBlocksFlushed;
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called to determine if a cache block or cache record
		is still needed.
****************************************************************************/
FLMBOOL F_Database::neededByReadTrans(
	FLMUINT64	ui64LowTransId,
	FLMUINT64	ui64HighTransId)
{
	FLMBOOL	bNeeded = FALSE;
	F_Db *	pReadTrans;
	
	lockMutex();

	// Quick check - so we don't have to traverse all read transactions.

	if (!m_pFirstReadTrans ||
		 ui64HighTransId < m_pFirstReadTrans->m_ui64CurrTransID ||
		 ui64LowTransId > m_pLastReadTrans->m_ui64CurrTransID)
	{
		goto Exit;
	}

	// Traverse all read transactions - this loop assumes that the
	// read transactions are in order of when they started - meaning
	// that the ui64CurrTransID on each will be ascending order.  The
	// loop will quit early once it can detect that the block is
	// too old for all remaining transactions.

	pReadTrans = m_pFirstReadTrans;
	while (pReadTrans)
	{
		if (pReadTrans->m_ui64CurrTransID >= ui64LowTransId &&
			 pReadTrans->m_ui64CurrTransID <= ui64HighTransId)
		{
			bNeeded = TRUE;
			goto Exit;
		}
		else if (pReadTrans->m_ui64CurrTransID > ui64HighTransId)
		{
			// All remaining transaction's transaction IDs will
			// also be greater than the block's high trans ID
			// Therefore, we can quit here.

			goto Exit;
		}
		pReadTrans = pReadTrans->m_pNextReadTrans;
	}

Exit:

	unlockMutex();
	return( bNeeded);
}

/****************************************************************************
Desc:	This routine is called just after a transaction has successfully
		committed.  It will unset the flags on log blocks
		that would cause them to be written to disk.  If the block is no longer
		needed by a read transaction, it will also put the block in the
		LRU list so it will be selected for replacement first.
****************************************************************************/
void F_Database::releaseLogBlocks( void)
{
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pNextSCache;
	
	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

	pSCache = m_pTransLogList;
	while (pSCache)
	{

#ifdef FLM_DBG_LOG
		FLMUINT16	ui16OldFlags = pSCache->m_ui16Flags;
#endif

		// A block in this list should never be dirty.

		flmAssert( !(pSCache->m_ui16Flags & CA_DIRTY));
		if ((pSCache->m_ui16Flags & CA_WRITE_TO_LOG) &&
			!(pSCache->m_ui16Flags & CA_LOG_FOR_CP))
		{
			flmAssert( m_uiLogCacheCount);
			m_uiLogCacheCount--;
		}

		pSCache->clearFlags( CA_WRITE_TO_LOG | CA_WAS_DIRTY);

#ifdef FLM_DBG_LOG
		pSCache->logFlgChange( ui16OldFlags, 'I');
#endif
		pNextSCache = pSCache->m_pNextInHashBucket;

		// Perhaps we don't really need to set these pointers to NULL,
		// but it helps keep things clean.

		pSCache->m_pNextInHashBucket = NULL;
		pSCache->m_pPrevInHashBucket = NULL;
		
		// If the block is no longer needed by a read transaction,
		// and it does not need to be logged for the checkpoint,
		// move it to the free list.

		if ((!pSCache->m_uiUseCount) &&
			 (!pSCache->neededByReadTrans()) &&
			 (!(pSCache->m_ui16Flags & CA_LOG_FOR_CP)))
		{
			F_CachedBlock *	pNewerVer = pSCache->m_pPrevInVersionList;

			if( !pSCache->m_pNextInVersionList && pNewerVer &&
				pNewerVer->m_ui64HighTransID == ~((FLMUINT64)0) &&
				pNewerVer->m_ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				pNewerVer->unlinkFromLogList();
			}

			pSCache->unlinkCache( TRUE, NE_SFLM_OK);
		}

		pSCache = pNextSCache;
	}
	m_pTransLogList = NULL;
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:	Retrieve a data block.  Shared cache is searched first.  If the block
		is not in shared cache, it will be retrieved from disk and put into
		cache.  The use count on the block will be incremented.
****************************************************************************/
RCODE F_Database::getBlock(
	F_Db *				pDb,
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiBlkAddress,		// Address of requested block.
	FLMUINT *			puiNumLooksRV,		// Pointer to FLMUINT where number of
													// cache lookups is to be returned.
													// If pointer is non-NULL it indicates
													// that we only want to find the block
													// if it is in cache.  If it is NOT
													// in cache, do NOT read it in from
													// disk. -- This capability is needed
													// by the FlmDbReduceSize function.
	F_CachedBlock **	ppSCacheRV)			// Returns pointer to cache block.
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT64			ui64BlkVersion;
	FLMUINT				uiNumLooks;
	F_CachedBlock **	ppSCacheBucket;
	F_CachedBlock *	pSBlkVerCache;
	F_CachedBlock *	pSMoreRecentVerCache;
	F_CachedBlock *	pSCache;
	FLMBOOL				bGotFromDisk = FALSE;

	flmAssert( this == pDb->m_pDatabase);
	flmAssert( uiBlkAddress != 0);

	*ppSCacheRV = NULL;

	// We should NEVER be attempting to read a block address that is
	// beyond the current logical end of file.

	if (!FSAddrIsBelow( uiBlkAddress, pDb->m_uiLogicalEOF))
	{
		rc = RC_SET( NE_SFLM_DATA_ERROR);
		goto Exit;
	}

	// Release CPU to prevent CPU hog

	f_yieldCPU();

	// Lock the mutex

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = TRUE;
	pDb->m_uiInactiveTime = 0;

	// Search shared cache for the desired version of the block.
	// First, determine the hash bucket.

	ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
									m_uiSigBitsInBlkSize, uiBlkAddress);

	// Search down the linked list of F_CachedBlock objects off of the bucket
	// looking for the correct cache block.

	pSCache = *ppSCacheBucket;
	uiNumLooks = 1;
	while ((pSCache) &&
			 (pSCache->m_uiBlkAddress != uiBlkAddress ||
			  pSCache->m_pDatabase != this))
	{
		if ((pSCache = pSCache->m_pNextInHashBucket) != NULL)
		{
			uiNumLooks++;
		}
	}

	// If there was no block found with the appropriate file/address we need to
	// create a dummy block and attempt to read it in.

	if (!pSCache)
	{
		if (puiNumLooksRV)
		{
			*puiNumLooksRV = uiNumLooks;
			*ppSCacheRV = NULL;
			goto Exit;
		}
		gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaults++;
		gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaultLooks += uiNumLooks;
		if (RC_BAD( rc = readIntoCache( pDb, pLFile, uiBlkAddress,
										NULL, NULL, &pSCache, &bGotFromDisk)))
		{
			goto Exit;
		}
	}
	else
	{
		// A block with the appropriate file/address was found.  We now
		// need to follow the chain until we find the version of the
		// block that we need.

		pSMoreRecentVerCache = NULL;

		// Save pointer to block that is newest version.

		pSBlkVerCache = pSCache;
		ui64BlkVersion = pDb->m_ui64CurrTransID;

		for (;;)
		{

			// If the block is being read into memory, wait for the read
			// to complete so we can see what it is.

			if (pSCache && (pSCache->m_ui16Flags & CA_READ_PENDING))
			{
				gv_SFlmSysData.pBlockCacheMgr->m_uiIoWaits++;
				if (RC_BAD( rc = flmWaitNotifyReq( 
					gv_SFlmSysData.hBlockCacheMutex, pDb->m_hWaitSem, 
					&pSCache->m_pNotifyList, (void *)&pSCache)))
				{
					goto Exit;
				}

				// The thread doing the notify "uses" the cache block
				// on behalf of this thread to prevent the cache block
				// from being flushed after it unlocks the mutex.
				// At this point, since we have locked the mutex,
				// we need to release the cache block.

				pSCache->releaseForThread();

				// Start over at the top of the list.

				pSBlkVerCache = pSCache;
				while (pSBlkVerCache->m_pPrevInVersionList)
				{
					pSBlkVerCache = pSBlkVerCache->m_pPrevInVersionList;
				}
				pSCache = pSBlkVerCache;
				pSMoreRecentVerCache = NULL;
				continue;
			}
			
			if (!pSCache || ui64BlkVersion > pSCache->m_ui64HighTransID)
			{
				if (puiNumLooksRV)
				{
					*puiNumLooksRV = uiNumLooks;
					*ppSCacheRV = NULL;
					goto Exit;
				}

				// The version of the block we want is not in the list,
				// either because we are at the end of the list (!pSCache),
				// or because the block version we want is higher than
				// the high trans ID on the cache block we are looking
				// at.  See if there is anything on disk that comes after
				// that block.  If not, simply return an OLD_VIEW
				// error.

				if (pSMoreRecentVerCache &&
					 pSMoreRecentVerCache->getPriorImageAddress() == 0)
				{
					// Should only be possible when reading a root block,
					// because the root block address in the LFILE may be
					// a block that was just created by an update
					// transaction.

					flmAssert( pDb->m_uiKilledTime);
					rc = RC_SET( NE_SFLM_OLD_VIEW);
					goto Exit;
				}
			
				gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaults++;
				gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaultLooks += uiNumLooks;
				
				if (pSMoreRecentVerCache)
				{
					if( RC_BAD( rc = readIntoCache( pDb, pLFile, uiBlkAddress,
									pSMoreRecentVerCache,
									pSMoreRecentVerCache->m_pNextInVersionList,
									&pSCache, &bGotFromDisk)))
					{
						goto Exit;
					}
				}
				else
				{
					if( RC_BAD( rc = readIntoCache( pDb, pLFile, uiBlkAddress,
									NULL, pSBlkVerCache, &pSCache, &bGotFromDisk)))
					{
						goto Exit;
					}
				}

				// At this point, if the read was successful, we should
				// have the block we want.

				break;
			}
			else if (ui64BlkVersion >= pSCache->getLowTransID())
			{

				// This is the version of the block that we need.

				gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheHits++;
				gv_SFlmSysData.pBlockCacheMgr->m_Usage.uiCacheHitLooks += uiNumLooks;
				break;
			}
			else
			{
				// If we are in an update transaction, the version of the
				// block we want should ALWAYS be at the top of the list.
				// If not, we have a serious problem!

				flmAssert( pDb->m_eTransType != SFLM_UPDATE_TRANS);

				pSMoreRecentVerCache = pSCache;
				pSCache = pSCache->m_pNextInVersionList;

				if (pSCache)
				{
					uiNumLooks++;
				}
			}
		}
	}

	// Increment the use count on the block.

	pSCache->useForThread( 0);

	// Block was found, make it the MRU block or bump it up in the MRU list,
	// if it is not already at the top.

	if( pDb->m_uiFlags & FDB_DONT_POISON_CACHE)
	{
		if (!(pDb->m_uiFlags & FDB_BACKGROUND_INDEXING) ||
			(pLFile && pLFile->eLfType != SFLM_LF_INDEX))
		{
			if (!bGotFromDisk)
			{
				pSCache->stepUpInGlobalList();
			}

			// If the block was read from disk and FDB_DONT_POISION_CACHE is
			// set, we don't need to do anything because the block is
			// already linked at the LRU position.
		}
		else if (pSCache->m_pPrevInGlobal)
		{
			pSCache->unlinkFromGlobalList();
			pSCache->linkToGlobalListAsMRU();
		}
	}
	else if (pSCache->m_pPrevInGlobal)
	{
		pSCache->unlinkFromGlobalList();
		pSCache->linkToGlobalListAsMRU();
	}

	*ppSCacheRV = pSCache;

Exit:

#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}
		scaVerify( 300);
	}
#endif

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Create a data block.
****************************************************************************/
RCODE F_Database::createBlock(
	F_Db *				pDb,
	F_CachedBlock **	ppSCacheRV)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiBlkAddress;
	F_BLK_HDR *			pBlkHdr;
	F_CachedBlock *	pSCache = NULL;
	F_CachedBlock *	pOldSCache = NULL;
	FLMBOOL				bMutexLocked = FALSE;
	FLMBOOL				bLocalCacheAllocation = FALSE;
	FLMUINT				uiOldLogicalEOF;
	F_CachedBlock **	ppSCacheBucket;
	FLMUINT				uiBlockSize = pDb->m_pDatabase->getBlockSize();

	pDb->m_bHadUpdOper = TRUE;

	// First see if there is a free block in the avail list.

	if (pDb->m_uiFirstAvailBlkAddr)
	{
		rc = blockUseNextAvail( pDb, ppSCacheRV);
		goto Exit;
	}

	// See if we need to free any cache or write dirty cache

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = TRUE;
	if (RC_BAD( rc = gv_SFlmSysData.pBlockCacheMgr->reduceCache( pDb)))
	{
		goto Exit;
	}
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = FALSE;

	// Create a new block at EOF

	uiBlkAddress = pDb->m_uiLogicalEOF;

	// Time for a new block file?

	if (FSGetFileOffset(uiBlkAddress) >= m_uiMaxFileSize)
	{
		FLMUINT	uiFileNumber = FSGetFileNumber( uiBlkAddress) + 1;

		if (uiFileNumber > MAX_DATA_BLOCK_FILE_NUMBER)
		{
			rc = RC_SET( NE_SFLM_DB_FULL);
			goto Exit;
		}

		if (RC_BAD( rc = pDb->m_pSFileHdl->createFile( uiFileNumber)))
		{
			goto Exit;
		}
		uiBlkAddress = FSBlkAddress( uiFileNumber, 0 );
	}

	// Allocate a cache block for this new block.  If we have older
	// versions of this block already in cache, we need to link the
	// new block above the older version.  If reasonable, try to
	// allocate the new cache block without locking the mutex.

	if( !gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())
	{
		if( (pSCache = new( uiBlockSize) F_CachedBlock( uiBlockSize)) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}

		bLocalCacheAllocation = TRUE;
	}

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bMutexLocked = TRUE;

	// Determine the hash bucket the new block should be put into.

	ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
									m_uiSigBitsInBlkSize, uiBlkAddress);

	// Search down the linked list of F_CachedBlock objects off of the bucket
	// looking for an older version of the block.  If there are older
	// versions, we should get rid of them.

	pOldSCache = *ppSCacheBucket;
	while (pOldSCache &&
			 (pOldSCache->m_uiBlkAddress != uiBlkAddress ||
			  pOldSCache->m_pDatabase != this))
	{
		pOldSCache = pOldSCache->m_pNextInHashBucket;
	}

	while (pOldSCache)
	{
		F_CachedBlock *	pNextSCache = pOldSCache->m_pNextInVersionList;

		// Older versions of blocks should not be in use or needed
		// by anyone because the only we we would have an older
		// version of a block beyond the logical EOF is if
		// FlmDbReduceSize had been called.  But it forces a
		// checkpoint that requires any read transactions to be
		// non-active, or killed.

		flmAssert( !pOldSCache->m_ui16Flags);
		flmAssert( !pOldSCache->m_uiUseCount);
		flmAssert( pOldSCache->m_ui64HighTransID == ~((FLMUINT64)0) ||
			!pOldSCache->neededByReadTrans());
		pOldSCache->unlinkCache( TRUE, NE_SFLM_OK);
		pOldSCache = pNextSCache;
	}

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	if( bLocalCacheAllocation)
	{
		F_BlockCacheMgr *		pBlockCacheMgr = gv_SFlmSysData.pBlockCacheMgr;

		// Now that the mutex is locked, update stats and do other work
		// that couldn't be done when the block was allocated.

		pBlockCacheMgr->m_Usage.uiCount++;
		pBlockCacheMgr->m_Usage.uiByteCount += pSCache->memSize();

		// Set use count to one so the block cannot be replaced.

		pSCache->m_bCanRelocate = TRUE;
		pSCache->useForThread( 0);
	}
	else
	{
		if (RC_BAD( rc = gv_SFlmSysData.pBlockCacheMgr->allocBlock( pDb, &pSCache)))
		{
			goto Exit;
		}
	}

	pSCache->m_uiBlkAddress = uiBlkAddress;
	pSCache->setTransID( ~((FLMUINT64)0));

	// Initialize the block data, dirty flag is set so that it will be
	// flushed as needed.

	pBlkHdr = pSCache->m_pBlkHdr;
	f_memset( pBlkHdr, 0, m_uiBlockSize);
	pBlkHdr->ui32BlkAddr = (FLMUINT32)uiBlkAddress;
	pBlkHdr->ui64TransID = pDb->m_ui64CurrTransID;
	pBlkHdr->ui16BlkBytesAvail =
		(FLMUINT16)(m_uiBlockSize - SIZEOF_STD_BLK_HDR);
	blkSetNativeFormat( pBlkHdr);

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( this, pSCache->m_uiBlkAddress, 0,
						 pSCache->getLowTransID(),
						"CREATE");
#endif

	// Link block into the global list

	pSCache->m_ui16Flags |= CA_DUMMY_FLAG;
	pSCache->linkToGlobalListAsMRU();

	// Set the dirty flag

	pSCache->setDirtyFlag( this);
	pSCache->m_ui16Flags &= ~CA_DUMMY_FLAG;

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	pSCache->setFlags( CA_WRITE_INHIBIT);

#ifdef FLM_DBG_LOG
	pSCache->logFlgChange( 0, 'J');
#endif

	// Now that the dirty flag and write inhibit flag
	// have been set, link the block to the file

	pSCache->linkToDatabase( this);
	pSCache->linkToHashBucket( ppSCacheBucket);

	uiOldLogicalEOF = pDb->m_uiLogicalEOF;
	pDb->m_uiLogicalEOF = uiBlkAddress + m_uiBlockSize;

	// Link the block into the "new" list

	pSCache->linkToNewList();

	// Return a pointer to the block

	*ppSCacheRV = pSCache;

Exit:

#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bMutexLocked = TRUE;
		}
		scaVerify( 400);
	}
#endif

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	if (RC_BAD( rc))
	{
		*ppSCacheRV = NULL;
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine releases a cache block by decrementing its use count.
		If the use count goes to zero, the block will be moved to the MRU
		position in the global cache list.
****************************************************************************/
void ScaReleaseCache(
	F_CachedBlock *	pSCache,
	FLMBOOL				bMutexAlreadyLocked)
{
	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	}

	// Can turn off write inhibit when the use count is about to go to zero,
	// because we are guaranteed at that point that nobody is going to still
	// update it.

	if (pSCache->getUseCount() == 1)
	{
		pSCache->clearFlags( CA_WRITE_INHIBIT);
	}

	pSCache->releaseForThread();

	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}
}

/****************************************************************************
Desc:	This routine increments the use count on a cache block
****************************************************************************/
void ScaUseCache(
	F_CachedBlock *	pSCache,
	FLMBOOL				bMutexAlreadyLocked)
{
	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	}

	pSCache->useForThread( 0);

	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}
}

/****************************************************************************
Desc:	Test and set the dirty flag on a block if not already set.  This is
		called on the case where a block didn't need to be logged because
		it had already been logged, but it still needs to have its dirty
		bit set.
****************************************************************************/
void F_Database::setBlkDirty(
	F_CachedBlock *	pSCache)
{
#ifdef FLM_DBG_LOG
	FLMUINT16	ui16OldFlags;
#endif

	// If the dirty flag is already set, we will NOT attempt to set it.

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
#ifdef FLM_DBG_LOG
	ui16OldFlags = pSCache->m_ui16Flags;
#endif

	if (!(pSCache->m_ui16Flags & CA_DIRTY))
	{
		flmAssert( this == pSCache->m_pDatabase);
		pSCache->setDirtyFlag( this);
	}

	// Move the block into the dirty blocks.  Even if block was
	// already dirty, put at the end of the list of dirty blocks.
	// This will make it so that when reduceCache hits a dirty
	// block, it is likely to also be one that will be written
	// out by a call to ScaFlushDirtyBlocks.

	pSCache->unlinkFromDatabase();
	pSCache->linkToDatabase( this);

	// Move the block to the MRU slot in the global list

	if (pSCache->m_pPrevInGlobal)
	{
		pSCache->unlinkFromGlobalList();
		pSCache->linkToGlobalListAsMRU();
	}

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	pSCache->setFlags( CA_WRITE_INHIBIT);
#ifdef FLM_DBG_LOG
	pSCache->logFlgChange( ui16OldFlags, 'O');
#endif
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:	This routine logs a block before it is modified.  In the shared cache
		system, the block is cloned.  The old version of the block is marked
		so that it will be written to the rollback log before the new version
		of the block can be written to disk.
		NOTE: It is assumed that the caller has "used" the block that is passed
		in. This routine will release it once we have made a copy of the block.
****************************************************************************/
RCODE F_Database::logPhysBlk(
	F_Db *				pDb,
	F_CachedBlock **	ppSCacheRV,	// This is a pointer to the pointer of the
											// cache block that is to be logged.
											// If the block has not been logged before
											// during the transaction, a new version
											// of the block will be created and a
											// pointer to that block will be returned.
											// Otherwise, the pointer is unchanged.
	F_CachedBlock **	ppOldSCache)
{
	RCODE 				rc = NE_SFLM_OK;
	F_CachedBlock *	pSCache = *ppSCacheRV;
	F_BLK_HDR *			pBlkHdr = pSCache->m_pBlkHdr;
	F_CachedBlock *	pNewSCache = NULL;
	FLMBOOL				bLockedMutex = FALSE;
	FLMBOOL				bLocalCacheAllocation = FALSE;
	FLMUINT				uiBlockSize = getBlockSize();
	F_CachedBlock **	ppSCacheBucket;
#ifdef FLM_DBG_LOG
	FLMUINT16			ui16OldFlags;
#endif

	flmAssert( this == pDb->m_pDatabase);
	flmAssert( pSCache->m_pPrevInVersionList == NULL);

	if( ppOldSCache)
	{
		*ppOldSCache = NULL;
	}
	
	// Increment the block change count -- this is not an accurate
	// indication of the number of blocks that have actually changed.  The
	// count is used by the cursor code to determine when to re-position in
	// the B-Tree.  The value is only used by cursors operating within
	// an update transaction.

	pDb->m_uiBlkChangeCnt++;

	// See if the block has already been logged since the last transaction.
	// If so, there is no need to log it again.

	if( pBlkHdr->ui64TransID == pDb->m_ui64CurrTransID)
	{
		flmAssert( pDb->m_bHadUpdOper);
		setBlkDirty( pSCache);
		goto Exit;
	}
	pDb->m_bHadUpdOper = TRUE;

	// See if we need to free any cache or write dirty cache

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bLockedMutex = TRUE;
	if (RC_BAD( rc = gv_SFlmSysData.pBlockCacheMgr->reduceCache( pDb)))
	{
		goto Exit;
	}
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	bLockedMutex = FALSE;

	// See if the transaction ID is greater than the last backup
	// transaction ID.  If so, we need to update our block change
	// count.

	if (pBlkHdr->ui64TransID < m_uncommittedDbHdr.ui64LastBackupTransID)
	{
		m_uncommittedDbHdr.ui32BlksChangedSinceBackup++;
	}

	// pDb->m_uiTransEOF contains what the EOF address was at
	// the beginning of the transaction.  There is no need to log the
	// block if it's address is beyond that point because it is a
	// NEW block.

	if (!FSAddrIsBelow( (FLMUINT)pSCache->m_pBlkHdr->ui32BlkAddr,
								pDb->m_uiTransEOF))
	{
		pBlkHdr->ui64TransID = pDb->m_ui64CurrTransID;
		setBlkDirty( pSCache);
		goto Exit;
	}

	// Allocate a cache block for this new block.  If we have older
	// versions of this block already in cache, we need to link the
	// new block above the older version.  Try to allocate the new
	// block outside of the block cache mutex.  If not possible, lock
	// the mutex and allocate.

	if( !gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())
	{
		if( (pNewSCache = new( uiBlockSize) F_CachedBlock( 
			uiBlockSize)) == NULL)
		{
			rc = RC_SET( NE_SFLM_MEM);
			goto Exit;
		}

		bLocalCacheAllocation = TRUE;

		// Copy the old block's data into this one.

		pBlkHdr = pNewSCache->m_pBlkHdr;
		f_memcpy( pBlkHdr, pSCache->m_pBlkHdr, m_uiBlockSize);
	}

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	bLockedMutex = TRUE;

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	if( bLocalCacheAllocation)
	{
		F_BlockCacheMgr *		pBlockCacheMgr = gv_SFlmSysData.pBlockCacheMgr;

		// Now that the mutex is locked, update stats and do other work
		// that couldn't be done when the block was allocated.

		pBlockCacheMgr->m_Usage.uiCount++;
		pBlockCacheMgr->m_Usage.uiByteCount += pNewSCache->memSize();

		// Set use count to one so the block cannot be replaced.

		pNewSCache->m_bCanRelocate = TRUE;
		pNewSCache->useForThread( 0);
	}
	else
	{
		if (RC_BAD( rc = gv_SFlmSysData.pBlockCacheMgr->allocBlock( 
			pDb, &pNewSCache)))
		{
			goto Exit;
		}

		// Copy the old block's data into this one.

		pBlkHdr = pNewSCache->m_pBlkHdr;
		f_memcpy( pBlkHdr, pSCache->m_pBlkHdr, m_uiBlockSize);
	}

#ifdef FLM_DEBUG

	// Make sure the caller isn't logging one that has already been
	// logged.

	if (gv_SFlmSysData.pBlockCacheMgr->m_bDebug)
	{
		F_CachedBlock *	pTmpSCache = m_pTransLogList;

		while (pTmpSCache)
		{
			flmAssert( pTmpSCache != pSCache);
			pTmpSCache = pTmpSCache->m_pNextInHashBucket;
		}
	}
#endif

	pNewSCache->m_uiBlkAddress = pSCache->m_uiBlkAddress;

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( this,
						 (FLMUINT)pNewSCache->m_pBlkHdr->ui32BlkAddr, 0,
						 pDb->m_ui64CurrTransID,
						"NEW-VER");
#endif

	// Link the block to the global list

	pNewSCache->m_ui16Flags |= CA_DUMMY_FLAG;
	pNewSCache->linkToGlobalListAsMRU();

	// Set flags so that appropriate flushing to log and DB will be done.

	pNewSCache->setDirtyFlag( this);
	pNewSCache->m_ui16Flags &= ~CA_DUMMY_FLAG;

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	pNewSCache->setFlags( CA_WRITE_INHIBIT);

	// Previous block address should be zero until we actually log the
	// prior version of the block.

	pBlkHdr->ui32PriorBlkImgAddr = 0;

	// Set the low and high trans IDs on the newly created block.

	pNewSCache->setTransID( ~((FLMUINT64)0));
	pBlkHdr->ui64TransID = pDb->m_ui64CurrTransID;
#ifdef FLM_DBG_LOG
	pNewSCache->logFlgChange( 0, 'L');
#endif

	// Determine the hash bucket the new block should be put into.

	flmAssert( pSCache->m_uiBlkAddress);
	ppSCacheBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
									m_uiSigBitsInBlkSize, pSCache->m_uiBlkAddress);

	// Link new block into various lists.

	pSCache->unlinkFromHashBucket( ppSCacheBucket);
	pSCache->m_pPrevInVersionList = pNewSCache;
	pSCache->verifyCache( 2900);
	pNewSCache->m_pNextInVersionList = pSCache;
	pNewSCache->linkToDatabase( this);
	pNewSCache->linkToHashBucket( ppSCacheBucket);
	pNewSCache->verifyCache( 3000);

	// Set the high trans ID on the old block to be one less than
	// the current trans ID.  Also set the flag indicating that
	// the block needs to be written to the rollback log.

	pSCache->setTransID( (pDb->m_ui64CurrTransID - 1));
#ifdef FLM_DBG_LOG
	ui16OldFlags = pSCache->m_ui16Flags;
#endif

	if (!(pSCache->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP)))
	{
		m_uiLogCacheCount++;
	}

	pSCache->setFlags( CA_WRITE_TO_LOG);

	if (pSCache->getLowTransID() <=
			m_uncommittedDbHdr.ui64RflLastCPTransID)
	{
		pSCache->setFlags( CA_LOG_FOR_CP);
	}

	if (pSCache->m_ui16Flags & CA_DIRTY)
	{
		pSCache->setFlags( CA_WAS_DIRTY);
		flmAssert( this == pSCache->m_pDatabase);
		pSCache->unsetDirtyFlag();

		// No more need to write inhibit - because the old version of the
		// block cannot possibly be changed.

		pSCache->clearFlags( CA_WRITE_INHIBIT);

		// Move the block out of the dirty blocks.

		pSCache->unlinkFromDatabase();
		pSCache->linkToDatabase( this);
	}

#ifdef FLM_DBG_LOG
	pSCache->logFlgChange( ui16OldFlags, 'N');
#endif

	// Put the old block into the list of the transaction's
	// log blocks

	pSCache->m_pPrevInHashBucket = NULL;
	if ((pSCache->m_pNextInHashBucket = m_pTransLogList) != NULL)
	{
		pSCache->m_pNextInHashBucket->m_pPrevInHashBucket = pSCache;
	}
	m_pTransLogList = pSCache;

	// Link the new block to the file log list

	pNewSCache->linkToLogList();

	// If this is an indexing thread, the old version of the
	// block will probably not be needed again so put it at the LRU end
	// of the cache.  The assumption is that a background indexing thread
	// has also set the FDB_DONT_POISON_CACHE flag.

	if (pDb->m_uiFlags & FDB_BACKGROUND_INDEXING)
	{
		pSCache->unlinkFromGlobalList();
		pSCache->linkToGlobalListAsLRU();
	}

	// Release the old block and return a pointer to the new block.

	if( !ppOldSCache)
	{
		ScaReleaseCache( pSCache, bLockedMutex);
	}
	else
	{
		*ppOldSCache = pSCache;
	}
	
	*ppSCacheRV = pNewSCache;

Exit:
#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bLockedMutex)
		{
			f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
			bLockedMutex = TRUE;
		}
		scaVerify( 500);
	}
#endif

	if (bLockedMutex)
	{
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:	Constructor for block cache manager.
****************************************************************************/
F_BlockCacheMgr::F_BlockCacheMgr()
{
	m_pBlockAllocator = NULL;
	m_pMRUReplace = NULL;
	m_pLRUReplace = NULL;
	m_pFirstFree = NULL;
	m_pLastFree = NULL;
	m_ppHashBuckets = NULL;
	m_uiNumBuckets = 0;
	m_uiHashFailTime = 0;
	f_memset( &m_Usage, 0, sizeof( m_Usage));
	m_uiFreeBytes = 0;
	m_uiFreeCount = 0;
	m_uiReplaceableCount = 0;
	m_uiReplaceableBytes = 0;
	m_bAutoCalcMaxDirty = FALSE;
	m_uiMaxDirtyCache = 0;
	m_uiLowDirtyCache = 0;
	m_uiTotalUses = 0;
	m_uiBlocksUsed = 0;
	m_uiPendingReads = 0;
	m_uiIoWaits = 0;
	m_uiHashMask = 0;
	m_bReduceInProgress = FALSE;	
#ifdef FLM_DEBUG
	m_bDebug = FALSE;
#endif
}

/****************************************************************************
Desc:	This routine initializes the hash table for block cache.
****************************************************************************/
RCODE F_BlockCacheMgr::initHashTbl( void)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiAllocSize;

	// Calculate the number of bits needed to represent values in the
	// hash table.

	m_uiNumBuckets = MIN_HASH_BUCKETS;
	m_uiHashMask = (m_uiNumBuckets - 1);
	uiAllocSize = (FLMUINT)sizeof( F_CachedBlock *) * m_uiNumBuckets;
	if (RC_BAD( rc = f_calloc( uiAllocSize, &m_ppHashBuckets)))
	{
		goto Exit;
	}
	gv_SFlmSysData.pGlobalCacheMgr->incrTotalBytes( f_msize( m_ppHashBuckets));

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine initializes the block cache manager.
****************************************************************************/
RCODE F_BlockCacheMgr::initCache( void)
{
	RCODE			rc = NE_SFLM_OK;
#define MAX_NUM_BLOCK_SIZES		16
	FLMUINT		uiBlockSizes[ MAX_NUM_BLOCK_SIZES];
	FLMUINT		uiBlockSize;
	FLMUINT		uiLoop;

	// Allocate memory for the hash table.

	if (RC_BAD( rc = initHashTbl()))
	{
		goto Exit;
	}
	
	// Initialize the cache block allocator

	for( uiLoop = 0, uiBlockSize = SFLM_MIN_BLOCK_SIZE; 
		  uiBlockSize <= SFLM_MAX_BLOCK_SIZE; 
		  uiLoop++, uiBlockSize *= 2)
	{
		if( uiLoop >= MAX_NUM_BLOCK_SIZES)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_MEM);
			goto Exit;
		}

		uiBlockSizes[ uiLoop] = uiBlockSize + sizeof( F_CachedBlock);
	}

	uiBlockSizes[ uiLoop] = 0;

	if( RC_BAD( rc = FlmAllocMultiAllocator( &m_pBlockAllocator)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pBlockAllocator->setup(
		TRUE, gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager, NULL, uiBlockSizes,
		&m_Usage.slabUsage, NULL)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine resizes the hash table for the block cache manager.
		NOTE: This routine assumes that the cache block mutex has been locked.
****************************************************************************/
RCODE F_BlockCacheMgr::rehash( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiNewHashTblSize;
	F_CachedBlock **	ppOldHashTbl;
	FLMUINT				uiOldHashTblSize;
	F_CachedBlock **	ppBucket;
	F_CachedBlock **	ppSCacheBucket;
	FLMUINT				uiLoop;
	F_CachedBlock *	pTmpSCache;
	F_CachedBlock *	pTmpNextSCache;
	FLMUINT				uiOldMemSize;

	uiNewHashTblSize = caGetBestHashTblSize( m_Usage.uiCount);

	// At this point we better have a different hash table size
	// or something is mucked up!

	flmAssert( uiNewHashTblSize != m_uiNumBuckets);

	// Save the old hash table and its size.

	if ((ppOldHashTbl = m_ppHashBuckets) != NULL)
	{
		uiOldMemSize = f_msize( ppOldHashTbl);
	}
	else
	{
		uiOldMemSize = 0;
	}
	uiOldHashTblSize = m_uiNumBuckets;

	// Allocate a new hash table.

	if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( F_CachedBlock *) *
								(FLMUINT)uiNewHashTblSize, &m_ppHashBuckets)))
	{
		m_uiHashFailTime = FLM_GET_TIMER();
		m_ppHashBuckets = ppOldHashTbl;
		goto Exit;
	}

	// Subtract off old size and add in new size.

	gv_SFlmSysData.pGlobalCacheMgr->decrTotalBytes( uiOldMemSize);
	gv_SFlmSysData.pGlobalCacheMgr->incrTotalBytes( f_msize( m_ppHashBuckets));

	m_uiNumBuckets = uiNewHashTblSize;
	m_uiHashMask = uiNewHashTblSize - 1;

	// Relink all of the cache blocks into the new
	// hash table.

	for (uiLoop = 0, ppBucket = ppOldHashTbl;
		  uiLoop < uiOldHashTblSize;
		  uiLoop++, ppBucket++)
	{
		pTmpSCache = *ppBucket;
		while (pTmpSCache)
		{
			pTmpNextSCache = pTmpSCache->m_pNextInHashBucket;
			
			// Should not be anything in a hash bucket that is not
			// associated with a database.
			
			flmAssert( pTmpSCache->m_pDatabase);
			flmAssert( pTmpSCache->m_uiBlkAddress);
			ppSCacheBucket = blockHash(
									pTmpSCache->m_pDatabase->getSigBitsInBlkSize(),
									pTmpSCache->m_uiBlkAddress);

			pTmpSCache->linkToHashBucket( ppSCacheBucket);
			pTmpSCache = pTmpNextSCache;
		}
	}
	
	// Throw away the old hash table.

	f_free( &ppOldHashTbl);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine performs various checks on an individual cache block
		to verify that it is linked into the proper lists, etc.  This routine
		assumes that the cache block mutex has been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
void F_CachedBlock::verifyCache(
	int	iPlace)
{
	F_CachedBLock *		pTmpSCache;
	FLMUINT					uiTmp = getBlkSize();
	FLMUINT					uiSigBitsInBlkSize;
	F_CachedBlock **		ppBucket;

	uiSigBitsInBlkSize = calcSigBits( uiTmp);
	ppBucket = gv_SFlmSysData.pBlockCacheMgr->blockHash(
										uiSigBitsInBlkSize, m_uiBlkAddress);
	pTmpSCache = *ppBucket;
	while (pTmpSCache && pTmpSCache != this)
	{
		pTmpSCache = pTmpSCache->m_pNextInHashBucket;
	}

	if( pTmpSCache)
	{
		if (!m_pDatabase)
		{
			f_breakpoint( iPlace+3);
		}
		
		if (m_pPrevInVersionList)
		{
			f_breakpoint( iPlace+4);
		}

		// Verify that it is not in the log list.

		if (m_ui16Flags & CA_WRITE_TO_LOG)
		{
			f_breakpoint( iPlace+5);
		}
		pTmpSCache = m_pDatabase->getTransLogList();
		while (pTmpSCache && pTmpSCache != this)
		{
			pTmpSCache = pTmpSCache->m_pNextInHashBucket;
		}
		if (pTmpSCache)
		{
			f_breakpoint( iPlace+6);
		}
	}
	else
	{
		if (m_pDatabase && !m_pPrevInVersionList)
		{
			f_breakpoint( iPlace+7);
		}

		// If the block is marked as needing to be logged, verify that
		// it is in the log list.

		if (m_ui16Flags & CA_WRITE_TO_LOG)
		{
			pTmpSCache = m_pDatabase->getTransLogList();
			while (pTmpSCache && pTmpSCache != this)
			{
				pTmpSCache = pTmpSCache->m_pNextInHashBucket;
			}

			if (!pTmpSCache)
			{
				// Not in the log list

				f_breakpoint( iPlace+8);
			}

			// Better also have a newer version.

			if (!m_pPrevInVersionList)
			{
				// Not linked to a prior version.

				f_breakpoint( iPlace+9);
			}
		}
	}

	// Verify that the prev and next pointers do not point to itself.

	if (m_pPrevInVersionList == this)
	{
		f_breakpoint( iPlace+10);
	}

	if (m_pNextInVersionList == this)
	{
		f_breakpoint( iPlace+11);
	}
}
#endif

/****************************************************************************
Desc:	This routine performs various checks on the cache to verify that
		things are linked into the proper lists, etc.  This routine assumes
		that the cache block mutex has been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
FSTATIC void scaVerify(
	int	iPlace)
{
	FLMUINT				uiLoop;
	F_CachedBlock **	ppBucket;
	F_CachedBlock *	pTmpSCache;

	// Verify that everything in buckets has a pFile and does NOT
	// have a m_pPrevInVersionList

	for (uiLoop = 0, ppBucket = gv_SFlmSysData.pBlockCacheMgr->m_ppHashBuckets;
		  uiLoop < gv_SFlmSysData.pBlockCacheMgr->m_uiNumBuckets;
		  uiLoop++, ppBucket++)
	{
		pTmpSCache = *ppBucket;
		while (pTmpSCache)
		{
			if (!pTmpSCache->m_pDatabase)
			{
				f_breakpoint(iPlace+1);
			}
			if (pTmpSCache->m_pPrevInVersionList)
			{
				f_breakpoint(iPlace+2);
			}
			pTmpSCache = pTmpSCache->m_pNextInHashBucket;
		}
	}

	// Traverse the entire list - make sure that everything
	// with a file is hashed and linked properly
	// and everything without a file is NOT hashed.

	pTmpSCache = (F_CachedBlock *)gv_SFlmSysData.pBlockCacheMgr->m_MRUList.m_pMRUItem;
	while (pTmpSCache)
	{
		pTmpSCache->verifyCache( 1000 + iPlace);
		pTmpSCache = pTmpSCache->m_pNextInGlobal;
	}
}
#endif

/****************************************************************************
Desc:	This routine determines what hash table size best fits the current
		item count.  It finds the hash bucket size whose midpoint between
		the minimum and maximum range is closest to the node count.
****************************************************************************/
FLMUINT caGetBestHashTblSize(
	FLMUINT	uiCurrItemCount
	)
{
	FLMUINT	uiNumHashBuckets;
	FLMUINT	uiMaxItemsForNumHashBuckets;
	FLMUINT	uiMinItemsForNumHashBuckets;
	FLMUINT	uiClosestNumHashBuckets = 0;
	FLMUINT	uiDistanceFromMidpoint;
	FLMUINT	uiLowestDistanceFromMidpoint;
	FLMUINT	uiHashTblItemsMidpoint;

	uiLowestDistanceFromMidpoint = 0xFFFFFFFF;
	for (uiNumHashBuckets = MIN_HASH_BUCKETS;
		  uiNumHashBuckets <= MAX_HASH_BUCKETS;
		  uiNumHashBuckets *= 2)
	{

		// Maximum desirable record count for a specific hash table size
		// we have arbitrarily chosen to be four times the number of buckets.
		// Minimum desirable record count we have arbitrarily chosen to be
		// the hash table size divided by fourn.

		uiMaxItemsForNumHashBuckets = maxItemCount( uiNumHashBuckets);
		uiMinItemsForNumHashBuckets = minItemCount( uiNumHashBuckets);

		// Ignore any hash bucket sizes where the current record count
		// is not between the desired minimum and maximum.

		if (uiCurrItemCount >= uiMinItemsForNumHashBuckets &&
			 uiCurrItemCount <= uiMaxItemsForNumHashBuckets)
		{

			// Calculate the midpoint between the minimum and maximum
			// for this particular hash table size.

			uiHashTblItemsMidpoint = (uiMaxItemsForNumHashBuckets -
											 uiMinItemsForNumHashBuckets) / 2;

			// See how far our current record count is from this midpoint.

			uiDistanceFromMidpoint = (FLMUINT)((uiHashTblItemsMidpoint > uiCurrItemCount)
											 ? (uiHashTblItemsMidpoint - uiCurrItemCount)
											 : (uiCurrItemCount - uiHashTblItemsMidpoint));

			// If the distance from the midpoint is closer than our previous
			// lowest distance, save it.

			if (uiDistanceFromMidpoint < uiLowestDistanceFromMidpoint)
			{
				uiClosestNumHashBuckets = uiNumHashBuckets;
				uiLowestDistanceFromMidpoint = uiDistanceFromMidpoint;
			}
		}
	}

	// Take the number of buckets whose middle was closest to the
	// current record count;

	if (uiLowestDistanceFromMidpoint == 0xFFFFFFFF)
	{
		// If we did not fall between any of the minimums or maximums,
		// we are either below the lowest minimum, or higher than the
		// highest maximum.

		uiNumHashBuckets = (FLMUINT)((uiCurrItemCount < minItemCount( MIN_HASH_BUCKETS))
										  ? (FLMUINT)MIN_HASH_BUCKETS
										  : (FLMUINT)MAX_HASH_BUCKETS);

	}
	else
	{
		uiNumHashBuckets = uiClosestNumHashBuckets;
	}
	return( uiNumHashBuckets);
}

/****************************************************************************
Desc:	This routine shuts down the shared cache manager and frees all
		resources allocated by it.
****************************************************************************/
F_BlockCacheMgr::~F_BlockCacheMgr()
{
	
	// Free the hash table

	if (m_ppHashBuckets)
	{
		gv_SFlmSysData.pGlobalCacheMgr->decrTotalBytes( f_msize( m_ppHashBuckets));
		f_free( &m_ppHashBuckets);
	}

	if( m_pBlockAllocator)
	{
		m_pBlockAllocator->Release();
	}

	flmAssert( !m_MRUList.m_pMRUItem && !m_MRUList.m_pLRUItem);
}

/****************************************************************************
Desc:	This routine frees all of the cache associated with an F_Database object.
****************************************************************************/
void F_Database::freeBlockCache( void)
{
	F_CachedBlock *	pSCache;
	F_CachedBlock *	pNextSCache;
	
	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

	// First, unlink as many as can be unlinked.

	pSCache = m_pSCacheList;
	flmAssert( !m_pPendingWriteList);
	while (pSCache)
	{
		f_yieldCPU();
		pNextSCache = pSCache->m_pNextInDatabase;

		if (!pSCache->m_uiUseCount)
		{
			// Turn off all bits that would cause an assert - we don't
			// care at this point, because we are forcing the file to
			// be closed.

			if (pSCache->m_ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
			{
				flmAssert( m_uiLogCacheCount);
				m_uiLogCacheCount--;
			}
			if (pSCache->m_pNextInVersionList &&
				 (pSCache->m_pNextInVersionList->m_ui16Flags &
					(CA_WRITE_TO_LOG | CA_LOG_FOR_CP)))
			{
				flmAssert( m_uiLogCacheCount);
				m_uiLogCacheCount--;
			}

#ifdef FLM_DEBUG
			pSCache->clearFlags(
				CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP | CA_WAS_DIRTY);

			if (pSCache->m_pNextInVersionList)
			{
				pSCache->m_pNextInVersionList->clearFlags(
					CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP | CA_WAS_DIRTY);
			}
#endif

			if (pSCache->m_ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				pSCache->unlinkFromLogList();
			}
			else if( pSCache->m_ui16Flags & CA_IN_NEW_LIST)
			{
				pSCache->unlinkFromNewList();
			}

			pSCache->unlinkCache( TRUE, NE_SFLM_OK);
		}
		else
		{
			// Another thread must have a temporary use on this block
			// because it is traversing cache for some reason.  We
			// don't want to free this block until the use count
			// is zero, so just put it into the free list so that
			// when its use count goes to zero we will either
			// re-use or free it.

			pSCache->unlinkCache( FALSE, NE_SFLM_OK);
			pSCache->linkToFreeList( FLM_GET_TIMER());
		}
		pSCache = pNextSCache;
	}

	// Set the F_Database cache list pointer to NULL.  Even if we didn't free
	// all of the cache blocks right now, we at least unlinked them from
	// the F_Database object.

	m_pSCacheList = NULL;
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
}


/****************************************************************************
Desc:	This routine computes an in-memory checksum on a cache block.  This
		is to guard against corrupt cache blocks being written back to disk.
		NOTE: This routine assumes that the cache block mutex is already locked.
****************************************************************************/
#ifdef FLM_DEBUG
FLMUINT F_CachedBlock::computeChecksum( void)
{
	FLMUINT	uiChecksum = 0;

	if( gv_SFlmSysData.pBlockCacheMgr->m_bDebug)
	{
		FLMUINT		uiBlkSize = getBlkSize();
		FLMBYTE *	pucBlk = (FLMBYTE *)m_pBlkHdr;
		FLMUINT		uiLoop;

		for( uiLoop = 0; uiLoop < uiBlkSize; uiLoop += 4, pucBlk += 4)
		{
			uiChecksum = (uiChecksum ^ (FLMUINT)(*((FLMUINT32 *)(pucBlk))));
		}

		if (!uiChecksum)
		{
			uiChecksum = 1;
		}
	}

	return( uiChecksum);
}
#endif

/****************************************************************************
Desc:	This routine finishes a checkpoint.  At this point we are guaranteed
		to have both the file lock and the write lock, and all dirty blocks
		have been written to the database.  This is the code that writes out
		the log header and truncates the rollback log, roll-forward log, and
		database files as needed.
****************************************************************************/
RCODE F_Database::finishCheckpoint(
	F_SEM					hWaitSem,
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FLMBOOL				bDoTruncate,
	FLMUINT				uiCPFileNum,
	FLMUINT				uiCPOffset,
	FLMUINT				uiCPStartTime,
	FLMUINT				uiTotalToWrite)
{
	RCODE					rc = NE_SFLM_OK;
	SFLM_DB_HDR *		pCommittedDbHdr = &m_lastCommittedDbHdr;
	SFLM_DB_HDR			saveDbHdr;
	FLMUINT				uiNewCPFileNum;
	FLMUINT64			ui64CurrTransID;
	FLMUINT				uiSaveTransOffset;
	FLMUINT				uiSaveCPFileNum;
	FLMBOOL				bTruncateLog = FALSE;
	FLMBOOL				bTruncateRflFile = FALSE;
	FLMUINT				uiTruncateRflSize = 0;
	FLMUINT				uiLogEof;
	FLMUINT				uiHighLogFileNumber;
#ifdef FLM_DBG_LOG
	FLMBOOL				bResetRBL = FALSE;
#endif

	// Update the DB header to indicate that we now
	// have a new checkpoint.

	f_memcpy( &saveDbHdr, pCommittedDbHdr, sizeof( SFLM_DB_HDR));

	// Save some of the values we are going to change.  These values
	// will be needed below.

	ui64CurrTransID = pCommittedDbHdr->ui64CurrTransID;
	uiSaveTransOffset = (FLMUINT)pCommittedDbHdr->ui32RflLastTransOffset;
	uiSaveCPFileNum = (FLMUINT)pCommittedDbHdr->ui32RflLastCPFileNum;

#ifdef FLM_DEBUG
	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

	// If we get to this point, there should be no dirty blocks for
	// the file.

	flmAssert( !m_uiDirtyCacheCount && !m_pPendingWriteList &&
				  (!m_pSCacheList ||
					  !(m_pSCacheList->m_ui16Flags & CA_DIRTY)));
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
#endif

	// Determine if we can reset the physical log.  The log can be reset if
	// there are no blocks in the log that are needed to preserve a read
	// consistent view for a read transaction.  By definition, this will
	// be the case if there are no read transactions that started before
	// the last transaction committed.  Thus, that is what we check.

	// If we exceed a VERY large size, we need to wait for the read
	// transactions to empty out so we can force a truncation of the
	// log.  This is also true if we are truncating the database and
	// changing the logical EOF.

	uiLogEof = (FLMUINT)pCommittedDbHdr->ui32RblEOF;
	uiHighLogFileNumber = FSGetFileNumber( uiLogEof);
	
	// Lock the database mutex while modifying the last committed header.

	lockMutex();
	if (uiHighLogFileNumber > 0 || bDoTruncate ||
		 FSGetFileOffset( uiLogEof) > LOW_VERY_LARGE_LOG_THRESHOLD_SIZE)
	{
		FLMINT						iWaitCnt = 0;
		F_Db *						pFirstDb;
		FLMUINT						ui5MinutesTime;
		FLMUINT						ui30SecTime;
		char							szMsgBuf[ 128];
		IF_LogMessageClient *	pLogMsg = NULL;
		FLMUINT						uiFirstDbInactiveSecs;
		FLMUINT						uiElapTime;
		FLMUINT						uiLastMsgTime = FLM_GET_TIMER();
		FLMBOOL						bMustTruncate = (bDoTruncate ||
											  uiHighLogFileNumber ||
											  FSGetFileOffset( uiLogEof) >=
												HIGH_VERY_LARGE_LOG_THRESHOLD_SIZE)
											 ? TRUE
											 : FALSE;

		ui5MinutesTime = FLM_SECS_TO_TIMER_UNITS( 300);
		ui30SecTime = FLM_SECS_TO_TIMER_UNITS( 30);

		if (m_pCPInfo && bMustTruncate)
		{
			m_pCPInfo->uiStartWaitTruncateTime = FLM_GET_TIMER();
		}

		pFirstDb = m_pFirstReadTrans;
		while ((!m_pCPInfo || !m_pCPInfo->bShuttingDown) && pFirstDb &&
				 pFirstDb->m_ui64CurrTransID < ui64CurrTransID)
		{
			FLMUINT		uiTime;
			FLMUINT		uiFirstDbInactiveTime = 0;
			FLMUINT64	ui64FirstDbCurrTransID = pFirstDb->m_ui64CurrTransID;
			FLMUINT		uiFirstDbThreadId = pFirstDb->m_uiThreadId;
			F_Db *		pTmpDb;

			uiTime = (FLMUINT)FLM_GET_TIMER();

			if( !bMustTruncate)
			{
				pTmpDb = pFirstDb;
				while( pTmpDb && pTmpDb->m_ui64CurrTransID < ui64CurrTransID)
				{
					if (!pTmpDb->m_uiInactiveTime)
					{
						pTmpDb->m_uiInactiveTime = uiTime;
					}
					pTmpDb = pTmpDb->m_pNextReadTrans;
				}
				uiFirstDbInactiveTime = pFirstDb->m_uiInactiveTime;
			}

			// If the read transaction has been inactive for 5 minutes,
			// forcibly kill it unless it has been marked as a "don't kill"
			// transaction.

			if( !(pFirstDb->m_uiFlags & FDB_DONT_KILL_TRANS) &&
				(bMustTruncate || (uiFirstDbInactiveTime &&
				FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime) >= ui5MinutesTime)))
			{
				pFirstDb->m_uiKilledTime = uiTime;
				if ((m_pFirstReadTrans = pFirstDb->m_pNextReadTrans) != NULL)
				{
					m_pFirstReadTrans->m_pPrevReadTrans = NULL;
				}
				else
				{
					m_pLastReadTrans = NULL;
				}
				pFirstDb->m_pPrevReadTrans = NULL;
				
				if ((pFirstDb->m_pNextReadTrans = m_pFirstKilledTrans) != NULL)
				{
					pFirstDb->m_pNextReadTrans->m_pPrevReadTrans = pFirstDb;
				}
				m_pFirstKilledTrans = pFirstDb;

				unlockMutex();

				// Log a message indicating that we have killed the transaction

				if ((pLogMsg = flmBeginLogMessage( SFLM_GENERAL_MESSAGE)) != NULL)
				{
					uiElapTime = FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime);
					uiFirstDbInactiveSecs = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

					f_sprintf( szMsgBuf,
						"Killed transaction %I64u."
						"  Thread: %X."
						"  Inactive time: %u seconds.",
						ui64FirstDbCurrTransID,
						(unsigned)uiFirstDbThreadId,
						(unsigned)uiFirstDbInactiveSecs);

					pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
					pLogMsg->appendString( szMsgBuf);
					flmEndLogMessage( &pLogMsg);
				}

				lockMutex();
				pFirstDb = m_pFirstReadTrans;
				continue;
			}
			else if (!bMustTruncate)
			{
				if (iWaitCnt >= 200)
				{
					break;
				}
			}

			unlockMutex();

			if (!bMustTruncate)
			{
				iWaitCnt++;
				f_sleep( 6);
			}
			else
			{
				// Log a message indicating that we are waiting for the
				// transaction to complete

				if (FLM_ELAPSED_TIME( uiTime, uiLastMsgTime) >= ui30SecTime)
				{
					if ((pLogMsg = flmBeginLogMessage( SFLM_GENERAL_MESSAGE)) != NULL)
					{
						uiElapTime = FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime);
						uiFirstDbInactiveSecs = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

						f_sprintf( szMsgBuf,
							"Waiting for transaction %I64u to complete."
							"  Thread: %X."
							"  Inactive time: %u seconds.",
							ui64FirstDbCurrTransID,
							(unsigned)uiFirstDbThreadId,
							(unsigned)uiFirstDbInactiveSecs);

						pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
						pLogMsg->appendString( szMsgBuf);
						flmEndLogMessage( &pLogMsg);
					}

					uiLastMsgTime = FLM_GET_TIMER();
				}

				f_sleep( 100);
			}
			lockMutex();
			pFirstDb = m_pFirstReadTrans;
		}

		if (bMustTruncate && m_pCPInfo)
		{
			m_pCPInfo->uiStartWaitTruncateTime = 0;
		}
	}

	if (!m_pFirstReadTrans ||
		 m_pFirstReadTrans->m_ui64CurrTransID >= ui64CurrTransID)
	{
		
		// We may want to truncate the log file if it has grown real big.

		if (uiHighLogFileNumber > 0 ||
			 FSGetFileOffset( uiLogEof) > LOG_THRESHOLD_SIZE)
		{
			bTruncateLog = TRUE;
		}
		pCommittedDbHdr->ui32RblEOF = (FLMUINT32)pCommittedDbHdr->ui16BlockSize;
#ifdef FLM_DBG_LOG
		bResetRBL = TRUE;
#endif
	}
	pCommittedDbHdr->ui32RblFirstCPBlkAddr = 0;

	// Set the checkpoint RFL file number and offset to be the same as
	// the last transaction's RFL file number and offset if nothing
	// is passed in.  If a non-zero uiCPFileNum is passed in, it is because
	// we are checkpointing the last transaction that has been recovered
	// by the recovery process.
	// In this case, instead of moving the pointers all the way forward,
	// to the last committed transaction, we simply move them forward to
	// the last recovered transaction.

	if (uiCPFileNum)
	{
		pCommittedDbHdr->ui32RflLastCPFileNum = (FLMUINT32)uiCPFileNum;
		pCommittedDbHdr->ui32RflLastCPOffset = (FLMUINT32)uiCPOffset;
	}
	else
	{
		FLMBOOL	bResetRflFile = FALSE;

		// If the RFL volume is full, and the LOG_AUTO_TURN_OFF_KEEP_RFL
		// flag is TRUE, change the LOG_KEEP_RFL_FILES to FALSE.

		if (m_pRfl->isRflVolumeFull() &&
			 pCommittedDbHdr->ui8RflKeepFiles &&
			 pCommittedDbHdr->ui8RflAutoTurnOffKeep)
		{
			pCommittedDbHdr->ui8RflKeepFiles = 0;
			bResetRflFile = TRUE;
		}

		pCommittedDbHdr->ui32RflLastCPFileNum =
			pCommittedDbHdr->ui32RflCurrFileNum;

		if (!pCommittedDbHdr->ui8RflKeepFiles)
		{
			pCommittedDbHdr->ui32RflLastCPOffset = 512;
			if (bResetRflFile)
			{

				// This will cause the RFL file to be recreated on the
				// next transaction - causing the keep signature to be
				// changed.  Also need to set up to use new serial
				// numbers so restore can't wade into this RFL file and
				// attempt to start restoring from it.

				pCommittedDbHdr->ui32RflLastTransOffset = 0;
				f_createSerialNumber( pCommittedDbHdr->ucLastTransRflSerialNum);
				f_createSerialNumber( pCommittedDbHdr->ucNextRflSerialNum);
			}

			// If LOG_RFL_LAST_TRANS_OFFSET is zero, someone has set this up
			// intentionally to cause the RFL file to be created at the
			// beginning of the next transaction.  We don't want to lose
			// that, so if it is zero, we don't change it.

			else if (pCommittedDbHdr->ui32RflLastTransOffset != 0)
			{
				pCommittedDbHdr->ui32RflLastTransOffset = 512;
			}
			uiTruncateRflSize = (FLMUINT)pCommittedDbHdr->ui32RflMinFileSize;
			if ((uiSaveTransOffset >= RFL_TRUNCATE_SIZE) ||
			    (uiSaveTransOffset >= uiTruncateRflSize))
			{
				bTruncateRflFile = TRUE;
				if (uiTruncateRflSize > RFL_TRUNCATE_SIZE)
				{
					uiTruncateRflSize = RFL_TRUNCATE_SIZE;
				}
				else if (uiTruncateRflSize < 512)
				{
					uiTruncateRflSize = 512;
				}

				// Set to nearest 512 byte boundary

				uiTruncateRflSize &= 0xFFFFFE00;
			}
		}
		else
		{
			FLMUINT	uiLastTransOffset =
						(FLMUINT)pCommittedDbHdr->ui32RflLastTransOffset;

			// If the RFL volume is not OK, and we are not currently positioned
			// at the beginning of an RFL file, we should set things up to roll to
			// the next RFL file.  That way, if they need to change RFL volumes
			// it will be OK, and we can create the new RFL file.

			if (!m_pRfl->seeIfRflVolumeOk() && uiLastTransOffset > 512)
			{
				pCommittedDbHdr->ui32RflLastTransOffset = 0;
				pCommittedDbHdr->ui32RflCurrFileNum++;
				pCommittedDbHdr->ui32RflLastCPOffset = 512;
				pCommittedDbHdr->ui32RflLastCPFileNum =
							pCommittedDbHdr->ui32RflCurrFileNum;
			}
			else
			{
				// If the transaction offset is zero, we want the last CP offset
				// to be 512 - it should never be set to zero.  It is possible
				// for the transaction offset to still be zero at this point if
				// we haven't done a non-empty transaction yet.

				if (!uiLastTransOffset)
				{
					uiLastTransOffset = 512;
				}
				pCommittedDbHdr->ui32RflLastCPOffset =
						(FLMUINT32)uiLastTransOffset;
			}
		}
	}

	// Set the checkpoint Trans ID to be the trans ID of the
	// last committed transaction.

	pCommittedDbHdr->ui64RflLastCPTransID = pCommittedDbHdr->ui64CurrTransID;

	unlockMutex();

	// Write the log header - this will complete the checkpoint.

	if (RC_BAD( rc = writeDbHdr( pDbStats, pSFileHdl,
									pCommittedDbHdr,
									&m_checkpointDbHdr, TRUE)))
	{

		// Restore log header.

		lockMutex();
		f_memcpy( pCommittedDbHdr, &saveDbHdr, sizeof( SFLM_DB_HDR));
		unlockMutex();
		goto Exit;
	}
	else if (bTruncateLog)
	{
		IF_FileHdl *		pCFileHdl;

		if (uiHighLogFileNumber)
		{
			(void)pSFileHdl->truncateFiles(
					FIRST_LOG_BLOCK_FILE_NUMBER,
					uiHighLogFileNumber);
		}

		if (RC_OK( pSFileHdl->getFileHdl( 0, TRUE, &pCFileHdl)))
		{
			(void)pCFileHdl->truncateFile( LOG_THRESHOLD_SIZE);
		}
	}

#ifdef FLM_DBG_LOG
	if (bResetRBL)
	{
		char	szMsg [80];

		if (bTruncateLog)
		{
			f_sprintf( szMsg, "f%u, Reset&TruncRBL, CPTID:%I64u",
				(unsigned)((FLMUINT)this),
				pCommittedDbHdr->ui64RflLastCPTransID);
		}
		else
		{
			f_sprintf( szMsg, "f%u, ResetRBL, CPTID:%I64u",
				(unsigned)((FLMUINT)this),
				pCommittedDbHdr->ui64RflLastCPTransID);
		}
		flmDbgLogMsg( szMsg);
	}
#endif

	// The checkpoint is now complete.  Reset the first checkpoint
	// block address to zero.

	m_uiFirstLogCPBlkAddress = 0;
	m_uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();

	// Save the state of the log header into the checkpointDbHdr buffer.

	f_memcpy( &m_checkpointDbHdr, pCommittedDbHdr, sizeof( SFLM_DB_HDR));

	// See if we need to delete RFL files that are no longer in use.

	uiNewCPFileNum = (FLMUINT)pCommittedDbHdr->ui32RflLastCPFileNum;

	if (!pCommittedDbHdr->ui8RflKeepFiles &&
		 uiSaveCPFileNum != uiNewCPFileNum &&
		 uiNewCPFileNum > 1)
	{
		FLMUINT	uiLastRflFileDeleted =
						(FLMUINT)pCommittedDbHdr->ui32RflLastFileNumDeleted;

		uiLastRflFileDeleted++;
		while (uiLastRflFileDeleted < uiNewCPFileNum)
		{
			char		szLogFilePath [F_PATH_MAX_SIZE];
			RCODE		TempRc;
			FLMUINT	uiFileNameSize = sizeof( szLogFilePath);
			FLMBOOL	bNameTruncated;

			m_pRfl->getFullRflFileName( uiLastRflFileDeleted,
										szLogFilePath, &uiFileNameSize,
										&bNameTruncated);
			if (bNameTruncated)
			{
				break;
			}
			if (RC_BAD( TempRc = gv_SFlmSysData.pFileSystem->deleteFile( 
				szLogFilePath)))
			{
				if (TempRc != NE_FLM_IO_PATH_NOT_FOUND &&
					 TempRc != NE_FLM_IO_INVALID_FILENAME)
				{
					break;
				}
			}
			uiLastRflFileDeleted++;
		}
		uiLastRflFileDeleted--;

		// If we actually deleted a file, update the log header.

		if (uiLastRflFileDeleted !=
				(FLMUINT)pCommittedDbHdr->ui32RflLastFileNumDeleted)
		{
			pCommittedDbHdr->ui32RflLastFileNumDeleted =
				(FLMUINT32)uiLastRflFileDeleted;
			if (RC_BAD( rc = writeDbHdr( pDbStats, pSFileHdl,
									pCommittedDbHdr,
									&m_checkpointDbHdr, TRUE)))
			{
				goto Exit;
			}

			// Save the state of the log header into the checkpointDbHdr buffer
			// and update the last checkpoint time again.

			f_memcpy( &m_checkpointDbHdr, pCommittedDbHdr,
								sizeof( SFLM_DB_HDR));
			m_uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();
		}
	}

	// Truncate the RFL file, if the truncate flag was set above.

	if (bTruncateRflFile)
	{
		(void)m_pRfl->truncate( hWaitSem, uiTruncateRflSize);
	}

	// Truncate the files, if requested to do so - this would be a request of
	// FlmDbReduceSize.

	if (bDoTruncate)
	{
		if (RC_BAD( rc = pSFileHdl->truncateFile(
									(FLMUINT)pCommittedDbHdr->ui32LogicalEOF)))
		{
			goto Exit;
		}
	}

	// Re-enable the RFL volume OK flag - in case it was turned off somewhere.

	m_pRfl->setRflVolumeOk();

	// If we complete a checkpoint successfully, we want to set the
	// pFile->CheckpointRc so that new transactions can come in.
	// NOTE: CheckpointRc should only be set while we still have the
	// lock on the database - which should always be the case at this
	// point.  This routine can only be called if we have obtained both
	// the write lock and the file lock.

	m_CheckpointRc = NE_SFLM_OK;

	// If we were calculating our maximum dirty cache, finish the
	// calculation.

	if (uiCPStartTime)
	{
		FLMUINT	uiCPEndTime = FLM_GET_TIMER();
		FLMUINT	uiCPElapsedTime = FLM_ELAPSED_TIME( uiCPEndTime, uiCPStartTime);
		FLMUINT	uiElapsedMilli;
		FLMUINT	ui15Seconds;
		FLMUINT	uiMaximum;
		FLMUINT	uiLow;

		// Get elapsed time in milliseconds - only calculate a new maximum if
		// we did at least a half second worth of writing.

		uiElapsedMilli = FLM_TIMER_UNITS_TO_MILLI( uiCPElapsedTime);

		if (uiElapsedMilli >= 500)
		{

			// Calculate what could be written in 15 seconds - set maximum
			// to that.  If calculated maximum is zero, we will not change
			// the current maximum.

			ui15Seconds = FLM_SECS_TO_TIMER_UNITS( 15);

			uiMaximum = (FLMUINT)(((FLMUINT64)uiTotalToWrite *
							 (FLMUINT64)ui15Seconds) / (FLMUINT64)uiCPElapsedTime);
			if (uiMaximum)
			{
				// Low is maximum minus what could be written in roughly
				// two seconds.

				uiLow = uiMaximum - (uiMaximum / 7);

				// Only set the maximum if we are still in auto-calculate mode.

				if (gv_SFlmSysData.pBlockCacheMgr->m_bAutoCalcMaxDirty)
				{
					f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);

					// Test flag again after locking the mutex

					if (gv_SFlmSysData.pBlockCacheMgr->m_bAutoCalcMaxDirty)
					{
						gv_SFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache = uiMaximum;
						gv_SFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache = uiLow;
					}
					f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
				}
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine performs a checkpoint.  It will stay in here until
		it either finishes, gets interrupted, or gets an error.  If we are not
		forcing a checkpoint, we periodically check to see if we should switch
		to a forced mode.  We also periodically check to see if another thread
		needs is waiting to obtain the write lock.
****************************************************************************/
RCODE F_Database::doCheckpoint(
	F_SEM					hWaitSem,
	SFLM_DB_STATS *	pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FLMBOOL				bDoTruncate,
	FLMBOOL				bForceCheckpoint,
	eForceCPReason		eForceReason,
	FLMUINT				uiCPFileNum,
	FLMUINT				uiCPOffset)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bWroteAll;
	FLMUINT				uiCPStartTime = 0;
	FLMUINT				uiTotalToWrite;
	FLMUINT				uiMaxDirtyCache;
	F_CachedBlock *	pSCache;
	FLMUINT				uiTimestamp;

	if (m_pCPInfo)
	{
		lockMutex();
		m_pCPInfo->bDoingCheckpoint = TRUE;
		m_pCPInfo->uiStartTime = (FLMUINT)FLM_GET_TIMER();
		m_pCPInfo->bForcingCheckpoint = bForceCheckpoint;
		if (bForceCheckpoint)
		{
			m_pCPInfo->uiForceCheckpointStartTime = m_pCPInfo->uiStartTime;
		}
		m_pCPInfo->eForceCheckpointReason = eForceReason;
		m_pCPInfo->uiDataBlocksWritten =
		m_pCPInfo->uiLogBlocksWritten = 0;
		unlockMutex();
	}

	uiTotalToWrite = (m_uiDirtyCacheCount + m_uiLogCacheCount) *
						m_uiBlockSize;

	f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
	if (bForceCheckpoint)
	{
		if (gv_SFlmSysData.pBlockCacheMgr->m_bAutoCalcMaxDirty)
		{
			uiCPStartTime = FLM_GET_TIMER();
		}
	}

	// If the amount of dirty cache is over our maximum, we must at least bring
	// it down below the low threshhold.  Otherwise, we set uiMaxDirtyCache
	// to the highest possible value - which will not require us to get
	// it below anything - because it is already within limits.

	if (gv_SFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache &&
		 uiTotalToWrite > gv_SFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache)
	{
		uiMaxDirtyCache = gv_SFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache;
	}
	else
	{
		uiMaxDirtyCache = ~((FLMUINT)0);
	}
	f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);

	// Write out log blocks first.

	bWroteAll = TRUE;
	if (RC_BAD( rc = flushLogBlocks( hWaitSem, pDbStats, pSFileHdl,
								TRUE, uiMaxDirtyCache,
								&bForceCheckpoint, &bWroteAll)))
	{
		goto Exit;
	}

	// If we didn't write out all log blocks, we got interrupted.

	if (!bWroteAll)
	{
		flmAssert( !bForceCheckpoint);
		goto Exit;
	}

	// Now write out dirty blocks

	if (RC_BAD( rc = flushDirtyBlocks( pDbStats, pSFileHdl,
								uiMaxDirtyCache,
								bForceCheckpoint, TRUE, &bWroteAll)))
	{
		goto Exit;
	}

	// If we didn't write out all dirty blocks, we got interrupted

	if (!bWroteAll)
	{
		flmAssert( !bForceCheckpoint);
		goto Exit;
	}

	// All dirty blocks and log blocks have been written, so we just
	// need to finish the checkpoint.

	if (RC_BAD( rc = finishCheckpoint( hWaitSem, pDbStats, pSFileHdl,
								bDoTruncate, uiCPFileNum, uiCPOffset,
								uiCPStartTime, uiTotalToWrite)))
	{
		goto Exit;
	}

Exit:

	// If we were attempting to force a checkpoint and it failed,
	// we want to set m_CheckpointRc, because we want to
	// prevent new transactions from starting until this situation
	// is cleared up (see fltrbeg.cpp).  Note that setting
	// m_CheckpointRc to something besides NE_SFLM_OK will cause
	// the checkpoint thread to force checkpoints whenever it is woke
	// up until it succeeds (see flopen.cpp).

	if (RC_BAD( rc) && bForceCheckpoint)
	{
		m_CheckpointRc = rc;
	}

	// Timestamp all of the items in the free list

	if (bForceCheckpoint)
	{
		uiTimestamp = FLM_GET_TIMER();
		
		f_mutexLock( gv_SFlmSysData.hBlockCacheMutex);
		pSCache = gv_SFlmSysData.pBlockCacheMgr->m_pFirstFree;
		while (pSCache)
		{
			pSCache->m_uiBlkAddress = uiTimestamp;
			pSCache = pSCache->m_pNextInDatabase;
		}
		f_mutexUnlock( gv_SFlmSysData.hBlockCacheMutex);
	}

	if (m_pCPInfo)
	{
		lockMutex();
		m_pCPInfo->bDoingCheckpoint = FALSE;
		unlockMutex();
	}

	return( rc);
}

/****************************************************************************
Notes:	This routine assumes the cache block cache mutex is locked
			This is a static method, so there is no "this" pointer to the
			F_BlockCacheMgr object.
****************************************************************************/
FLMBOOL F_BlockRelocator::canRelocate(
	void *		pvAlloc)
{
	return( ((F_CachedBlock *)pvAlloc)->m_uiUseCount ? FALSE : TRUE);
}

/****************************************************************************
Desc:		Fixes up all pointers needed to allow an F_CachedBlock object to be
			moved to a different location in memory
Notes:	This routine assumes the cache block mutex is locked
			This is a static method, so there is no "this" pointer to the
			F_BlockCacheMgr object.
****************************************************************************/
void F_BlockRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	F_CachedBlock *		pOldSCache = (F_CachedBlock *)pvOldAlloc;
	F_CachedBlock *		pNewSCache = (F_CachedBlock *)pvNewAlloc;
	F_CachedBlock **		ppBucket;
	F_BlockCacheMgr *		pBlockCacheMgr = gv_SFlmSysData.pBlockCacheMgr;
	F_Database *			pDatabase = pOldSCache->m_pDatabase;

	flmAssert( !pOldSCache->m_uiUseCount);

	if( pNewSCache->m_pPrevInDatabase)
	{
		pNewSCache->m_pPrevInDatabase->m_pNextInDatabase = pNewSCache;
	}

	if( pNewSCache->m_pNextInDatabase)
	{
		pNewSCache->m_pNextInDatabase->m_pPrevInDatabase = pNewSCache;
	}

	if( pNewSCache->m_pPrevInGlobal)
	{
		pNewSCache->m_pPrevInGlobal->m_pNextInGlobal = pNewSCache;
	}

	if( pNewSCache->m_pNextInGlobal)
	{
		pNewSCache->m_pNextInGlobal->m_pPrevInGlobal = pNewSCache;
	}

	if( pNewSCache->m_pPrevInReplaceList)
	{
		pNewSCache->m_pPrevInReplaceList->m_pNextInReplaceList = pNewSCache;
	}

	if( pNewSCache->m_pNextInReplaceList)
	{
		pNewSCache->m_pNextInReplaceList->m_pPrevInReplaceList = pNewSCache;
	}

	if( pNewSCache->m_pPrevInHashBucket)
	{
		pNewSCache->m_pPrevInHashBucket->m_pNextInHashBucket = pNewSCache;
	}

	if( pNewSCache->m_pNextInHashBucket)
	{
		pNewSCache->m_pNextInHashBucket->m_pPrevInHashBucket = pNewSCache;
	}

	if( pNewSCache->m_pPrevInVersionList)
	{
		pNewSCache->m_pPrevInVersionList->m_pNextInVersionList = pNewSCache;
	}

	if( pNewSCache->m_pNextInVersionList)
	{
		pNewSCache->m_pNextInVersionList->m_pPrevInVersionList = pNewSCache;
	}

	if( pDatabase)
	{
		if( pDatabase->m_pSCacheList == pOldSCache)
		{
			pDatabase->m_pSCacheList = pNewSCache;
		}

		if( pDatabase->m_pLastDirtyBlk == pOldSCache)
		{
			pDatabase->m_pLastDirtyBlk = pNewSCache;
		}

		if( pDatabase->m_pFirstInLogList == pOldSCache)
		{
			pDatabase->m_pFirstInLogList = pNewSCache;
		}

		if( pDatabase->m_pLastInLogList == pOldSCache)
		{
			pDatabase->m_pLastInLogList = pNewSCache;
		}

		if( pDatabase->m_pFirstInNewList == pOldSCache)
		{
			pDatabase->m_pFirstInNewList = pNewSCache;
		}

		if( pDatabase->m_pLastInNewList == pOldSCache)
		{
			pDatabase->m_pLastInNewList = pNewSCache;
		}

		if( pDatabase->m_pTransLogList == pOldSCache)
		{
			pDatabase->m_pTransLogList = pNewSCache;
		}

		ppBucket = pBlockCacheMgr->blockHash( pDatabase->getSigBitsInBlkSize(),
				pOldSCache->m_uiBlkAddress);

		if( *ppBucket == pOldSCache)
		{
			*ppBucket = pNewSCache;
		}

		flmAssert( pDatabase->m_pPendingWriteList != pOldSCache);
	}

	if (pBlockCacheMgr->m_MRUList.m_pMRUItem == (F_CachedItem *)pOldSCache)
	{
		pBlockCacheMgr->m_MRUList.m_pMRUItem = pNewSCache;
	}

	if (pBlockCacheMgr->m_MRUList.m_pLRUItem == (F_CachedItem *)pOldSCache)
	{
		pBlockCacheMgr->m_MRUList.m_pLRUItem = pNewSCache;
	}
	
	if (pBlockCacheMgr->m_MRUList.m_pLastMRUItem == (F_CachedItem *)pOldSCache)
	{
		pBlockCacheMgr->m_MRUList.m_pLastMRUItem = pNewSCache;
	}

	if (pBlockCacheMgr->m_pMRUReplace == pOldSCache)
	{
		pBlockCacheMgr->m_pMRUReplace = pNewSCache;
	}

	if (pBlockCacheMgr->m_pLRUReplace == pOldSCache)
	{
		pBlockCacheMgr->m_pLRUReplace = pNewSCache;
	}

	if (pBlockCacheMgr->m_pFirstFree == pOldSCache)
	{
		pBlockCacheMgr->m_pFirstFree = pNewSCache;
	}

	if (pBlockCacheMgr->m_pLastFree == pOldSCache)
	{
		pBlockCacheMgr->m_pLastFree = pNewSCache;
	}

	pNewSCache->m_pBlkHdr = (F_BLK_HDR *)&pNewSCache[ 1];
}

/****************************************************************************
Desc:	This function will encrypt the block of data passed in. This function
		assumes that the buffer passed in includes the block header.
****************************************************************************/
RCODE F_Database::encryptBlock(
	F_Dict *		pDict,
	FLMBYTE *	pucBuffer
	)
{
	RCODE						rc = NE_SFLM_OK;
	F_INDEX *				pIndex;
	F_TABLE *				pTable;
	FLMUINT					uiLfNum;
	F_BTREE_BLK_HDR *		pBlkHdr = (F_BTREE_BLK_HDR *)pucBuffer;
	F_ENCDEF *				pEncDef = NULL;
	FLMUINT					uiEncDefNum;
	FLMUINT					uiEncLen = m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr);
#ifdef FLM_USE_NICI
	F_CCS *					pCcs = NULL;
#endif

	if (!blkIsBTree( (F_BLK_HDR *)pucBuffer))
	{
		// Nothing to do.  We are only interested in btree blocks.
		goto Exit;
	}
	
	if (!isEncryptedBlk( (F_BLK_HDR *)pBlkHdr))
	{
		goto Exit;
	}

	if (pBlkHdr->stdBlkHdr.ui8BlkType == BT_DATA_ONLY)
	{
		uiEncLen = m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr);
		
		if (!m_bTempDb)
		{
			uiEncDefNum = (FLMUINT)(((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum);
			
			// Need to get the encryption object.
	
			if ((pEncDef = pDict->getEncDef( uiEncDefNum)) == NULL)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
				goto Exit;
			}
		}
	}
	else if (isTableBlk( pBlkHdr))
	{
		if (!m_bTempDb)
		{
			uiLfNum = pBlkHdr->ui16LogicalFile;
	
			// Get the table
			if ((pTable = pDict->getTable( uiLfNum)) == NULL)
			{
				goto Exit;
			}
	
			// The collection may not be encrypted.
			// We can just exit here.
	
			if (!pTable->lfInfo.uiEncDefNum)
			{
				goto Exit;  // NE_SFLM_OK;
			}
	
			// Need to get the encryption object.
			pEncDef = pDict->getEncDef( pTable->lfInfo.uiEncDefNum);
			flmAssert( pEncDef);
		}
	}
	else if (isIndexBlk( pBlkHdr))
	{
		if (!m_bTempDb)
		{
			uiLfNum = pBlkHdr->ui16LogicalFile;
	
			// Get the index.
			if ((pIndex = pDict->getIndex( uiLfNum)) == NULL)
			{
				goto Exit;
			}
		
			// The index may not be encrypted.
			// We can just exit here.
			if (!pIndex->lfInfo.uiEncDefNum)
			{
				goto Exit;  // NE_SFLM_OK;
			}
		
			// Need to get the encryption object.
			pEncDef = pDict->getEncDef( pIndex->lfInfo.uiEncDefNum);
			flmAssert( pEncDef);
		}
	}
	else
	{
		goto Exit;  // NE_SFLM_OK
	}

#ifndef FLM_USE_NICI
	rc = RC_SET( NE_SFLM_ENCRYPTION_UNAVAILABLE);
	goto Exit;
#else

	if (m_bInLimitedMode)
	{
		rc = RC_SET( NE_SFLM_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if (!m_bTempDb)
	{
		flmAssert( pEncDef);
		pCcs = pEncDef->pCcs;
	}
	else
	{
		flmAssert( !pEncDef);
		pCcs = m_pWrappingKey;
	}
		
	flmAssert( pCcs);
	flmAssert( !(uiEncLen % 16));

	// Encrypt the buffer in place.
	
	if (pBlkHdr->stdBlkHdr.ui8BlkType == BT_DATA_ONLY)
	{
		if (RC_BAD( rc = pCcs->encryptToStore( &pucBuffer[ sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)],
															uiEncLen,
															&pucBuffer[ sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)],
															&uiEncLen)))
		{
			goto Exit;
		}
	
		flmAssert( uiEncLen == (m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)));
	
	}
	else
	{
		if (RC_BAD( rc = pCcs->encryptToStore( &pucBuffer[ sizeofBTreeBlkHdr( pBlkHdr)],
															uiEncLen,
															&pucBuffer[ sizeofBTreeBlkHdr( pBlkHdr)],
															&uiEncLen)))
		{
			goto Exit;
		}
	
		flmAssert( uiEncLen == (m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr)));
	
	}
#endif

Exit:

	return rc;
}

/****************************************************************************
Desc:	This function will decrypt the block of data passed in.
****************************************************************************/
RCODE F_Database::decryptBlock(
	F_Dict *		pDict,
	FLMBYTE *	pucBuffer)
{
	RCODE						rc = NE_SFLM_OK;
	F_INDEX *				pIndex;
	F_TABLE *				pTable;
	FLMUINT					uiLfNum;
	F_BTREE_BLK_HDR *		pBlkHdr = (F_BTREE_BLK_HDR *)pucBuffer;
	FLMUINT					uiEncLen;
	F_ENCDEF *				pEncDef = NULL;
#ifdef FLM_USE_NICI
	F_CCS *					pCcs = NULL;
#endif

	if (!blkIsBTree( (F_BLK_HDR *)pucBuffer))
	{
		// Nothing to do.  We are only interested in btree blocks.
		goto Exit;
	}

	if (!isEncryptedBlk( (F_BLK_HDR *)pBlkHdr))
	{
		goto Exit;
	}

	if (pBlkHdr->stdBlkHdr.ui8BlkType == BT_DATA_ONLY)
	{

		uiEncLen = m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr);
		
		if (!m_bTempDb)
		{
			
			// Need to get the encryption object.
	
			if ((pEncDef = pDict->getEncDef(
					(FLMUINT)(((F_ENC_DO_BLK_HDR *)pBlkHdr)->ui32EncDefNum))) == NULL)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_ENCDEF_NUM);
				goto Exit;
			}
		}
	}
	else if (isTableBlk( pBlkHdr))
	{
		uiEncLen = m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr);
		
		if (!m_bTempDb)
		{
			uiLfNum = pBlkHdr->ui16LogicalFile;
	
			// Get the index.
			if ((pTable = pDict->getTable( uiLfNum)) == NULL)
			{
				goto Exit;
			}
	
			// The collection may not be encrypted.
			// We can just exit here.
	
			if (!pTable->lfInfo.uiEncDefNum)
			{
				goto Exit;  // NE_SFLM_OK;
			}
		
			// Need to get the encryption object.
	
			pEncDef = pDict->getEncDef( pTable->lfInfo.uiEncDefNum);
			flmAssert( pEncDef);
		}
	}
	else if (isIndexBlk( pBlkHdr))
	{
		uiEncLen = m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr);
		
		if (!m_bTempDb)
		{
			uiLfNum = pBlkHdr->ui16LogicalFile;
	
			// Get the index.
	
			if ((pIndex = pDict->getIndex( uiLfNum)) == NULL)
			{
				goto Exit;
			}
		
			// The index may not be encrypted.
			// We can just exit here.
	
			if (!pIndex->lfInfo.uiEncDefNum)
			{
				goto Exit;  // NE_SFLM_OK;
			}
		
			// Need to get the encryption object.
	
			pEncDef = pDict->getEncDef( pIndex->lfInfo.uiEncDefNum);
			flmAssert( pEncDef);
		}
	}
	else
	{
		goto Exit;  // NE_SFLM_OK
	}


#ifndef FLM_USE_NICI
	rc = RC_SET( NE_SFLM_ENCRYPTION_UNAVAILABLE);
	goto Exit;
#else

	if (m_bInLimitedMode)
	{
		rc = RC_SET( NE_SFLM_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if (!m_bTempDb)
	{
		flmAssert( pEncDef);
		pCcs = pEncDef->pCcs;
	}
	else
	{
		flmAssert( !pEncDef);
		pCcs = m_pWrappingKey;
	}
		
	flmAssert( pCcs);
	flmAssert( !(uiEncLen % 16));

	if (pBlkHdr->stdBlkHdr.ui8BlkType == BT_DATA_ONLY)
	{
		if (RC_BAD( rc = pCcs->decryptFromStore( 
			&pucBuffer[ sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)], uiEncLen,
			&pucBuffer[ sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)], &uiEncLen)))
		{
			goto Exit;
		}
	
		flmAssert( uiEncLen == 
					 (m_uiBlockSize - sizeofDOBlkHdr( (F_BLK_HDR *)pBlkHdr)));
	}
	else
	{
		if (RC_BAD( rc = pCcs->decryptFromStore( 
			&pucBuffer[ sizeofBTreeBlkHdr( pBlkHdr)], uiEncLen,
			&pucBuffer[ sizeofBTreeBlkHdr( pBlkHdr)], &uiEncLen)))
		{
			goto Exit;
		}
	
		flmAssert( uiEncLen == (m_uiBlockSize - sizeofBTreeBlkHdr( pBlkHdr)));
	}

#endif

Exit:

	return rc;
}

#undef new
#undef delete

/****************************************************************************
Desc:
****************************************************************************/
void SQFAPI F_CachedBlock::objectAllocInit(
	void *		pvAlloc,
	FLMUINT		uiSize)
{
	F_UNREFERENCED_PARM( uiSize);
	
	// Need to make sure that m_bCanRelocate is initialized to zero
	// prior to unlocking the mutex.  This is so the allocator 
	// doesn't see garbage values that may cause it to relocate the object 
	// before the constructor has been called.
	
	((F_CachedBlock *)pvAlloc)->m_bCanRelocate = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_CachedBlock::operator new(
	FLMSIZET			uiSize,
	FLMUINT			uiBlockSize)
#ifndef FLM_NLM	
	throw()
#endif
{
	void *		pvPtr;
	
	flmAssert( uiSize == sizeof( F_CachedBlock));

	if( RC_BAD( gv_SFlmSysData.pBlockCacheMgr->m_pBlockAllocator->allocBuf(
		&gv_SFlmSysData.pBlockCacheMgr->m_blockRelocator,
		uiSize + uiBlockSize, F_CachedBlock::objectAllocInit,
		(FLMBYTE **)&pvPtr)))
	{
		pvPtr = NULL;
	}
	
	flmAssert( !((F_CachedBlock *)pvPtr)->m_bCanRelocate); 
	return( pvPtr);
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_CachedBlock::operator new(
	FLMSIZET)	//uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
	// This new should never be called
	flmAssert( 0);
	return( NULL);
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_CachedBlock::operator new[]( FLMSIZET)
#ifndef FLM_NLM	
	throw()
#endif
{
	flmAssert( 0);
	return( NULL);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void * F_CachedBlock::operator new(
	FLMSIZET,		// uiSize,
	const char *,	// pszFileName,
	int)				// iLineNum)
#ifndef FLM_NLM	
	throw()
#endif
{
	// This new should never be called
	flmAssert( 0);
	return( NULL);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void * F_CachedBlock::operator new[](
	FLMSIZET,		// uiSize,
	const char *,	// pszFileName,
	int)				// iLine)
#ifndef FLM_NLM	
	throw()
#endif
{
	flmAssert( 0);
	return( NULL);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
void F_CachedBlock::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	gv_SFlmSysData.pBlockCacheMgr->m_pBlockAllocator->freeBuf( (FLMBYTE **)&ptr);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CachedBlock::operator delete[](
	void *)		// ptr
{
	flmAssert( 0);
}
