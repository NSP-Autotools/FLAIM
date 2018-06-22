//------------------------------------------------------------------------------
// Desc:	This is the DOM Node cache for XFLAIM
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

#include "flaimsys.h"

#if defined( FLM_NLM) && !defined( __MWERKS__)
// Disable "Warning! W549: col(XX) 'sizeof' operand contains
// compiler generated information"
	#pragma warning 549 9
#endif

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_NodeCacheMgr::F_NodeCacheMgr()
{
	m_pNodeAllocator = NULL;
	m_pBufAllocator = NULL;
	m_pAttrItemAllocator = NULL;
	m_pPurgeList = NULL;
	m_pHeapList = NULL;
	m_pOldList = NULL;
	f_memset( &m_Usage, 0, sizeof( m_Usage));
	m_ppHashBuckets = NULL;
	m_uiNumBuckets = 0;
	m_uiHashFailTime = 0;
	m_uiHashMask = 0;
	m_uiPendingReads = 0;
	m_uiIoWaits = 0;
	m_pFirstNode = NULL;
	m_bReduceInProgress = FALSE;
#ifdef FLM_DEBUG
	m_bDebug = FALSE;
#endif
}
	
/****************************************************************************
Desc:	Constructor for F_CachedNode
****************************************************************************/
F_CachedNode::F_CachedNode()
{
	m_pPrevInBucket = NULL;
	m_pNextInBucket = NULL;
	m_pPrevInDatabase = NULL;
	m_pNextInDatabase = NULL;
	m_pOlderVersion = NULL;
	m_pNewerVersion = NULL;
	m_pPrevInHeapList = NULL;
	m_pNextInHeapList = NULL;
	m_pPrevInOldList = NULL;
	m_pNextInOldList = NULL;
	m_ui64LowTransId = 0;
	
	// Set the high transaction ID to FLM_MAX_UINT64 so that this will NOT
	// be treated as one that had memory assigned to the old version nodes.
	
	m_ui64HighTransId = FLM_MAX_UINT64;
	m_pNotifyList = NULL;
	m_uiCacheFlags = 0;
	m_uiStreamUseCount = 0;
	
	// Items initialized in constructor

	m_uiDataBufSize = 0;	
	m_pucData = NULL;
	m_pNodeList = NULL;
	m_ppAttrList = NULL;
	m_uiAttrCount = 0;
	m_uiTotalAttrSize = 0;
	m_uiFlags = 0;
	
	f_memset( &m_nodeInfo, 0, sizeof( F_NODE_INFO));
}

/****************************************************************************
Desc:	Destructor for F_CachedNode object.
		This routine assumes the global mutex is already locked.
****************************************************************************/
F_CachedNode::~F_CachedNode()
{
	// Don't include attribute size, because it will be subtracted out
	// when we delete the attr items.

	FLMUINT		uiSize = memSize() - m_uiTotalAttrSize;
	FLMBYTE *	pucActualAlloc;
	
	flmAssert( !m_uiStreamUseCount);
	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);

	// If this is an old version, decrement the old version counters.

	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiSize &&
					  gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount--;
		unlinkFromOldList();
	}

	flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiSize &&
				  gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCount);
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiSize;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCount--;

	if( m_uiFlags & FDOM_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}
	
	// Free the m_pucData, if any
	
	if (m_pucData)
	{
		pucActualAlloc = getActualPointer( m_pucData);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
							m_uiDataBufSize, &pucActualAlloc);
		m_pucData = NULL;
	}
	
	if (m_pNodeList)
	{
		pucActualAlloc = getActualPointer( m_pNodeList);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
							calcNodeListBufSize( m_nodeInfo.uiChildElmCount),
							&pucActualAlloc);
		m_pNodeList = NULL;
	}

	if (m_uiAttrCount)
	{
		FLMUINT	uiLoop;

		for (uiLoop = 0; uiLoop < m_uiAttrCount; uiLoop++)
		{
			delete m_ppAttrList [uiLoop];
		}

		flmAssert( !m_uiTotalAttrSize);

		pucActualAlloc = getActualPointer( m_ppAttrList);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
							calcAttrListBufSize( m_uiAttrCount),
							&pucActualAlloc);
		m_ppAttrList = NULL;
		m_uiAttrCount = 0;
	}

	if (shouldRehash( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCount,
							gv_XFlmSysData.pNodeCacheMgr->m_uiNumBuckets))
	{
		if (checkHashFailTime( &gv_XFlmSysData.pNodeCacheMgr->m_uiHashFailTime))
		{
			(void)gv_XFlmSysData.pNodeCacheMgr->rehash();
		}
	}
}

/****************************************************************************
Desc:	This routine frees a purged node from node cache.  This routine assumes
		that the node cache mutex has already been locked.
****************************************************************************/
void F_CachedNode::freePurged( void)
{

	// Unlink the node from the purged list.

	unlinkFromPurged();

	// Free the F_CachedNode object.

	unsetPurged();

	delete this;
}

/****************************************************************************
Desc:	This routine frees a node in the node cache.  This routine assumes
		that the node cache mutex has already been locked.
****************************************************************************/
void F_CachedNode::freeCache(
	FLMBOOL	bPutInPurgeList)
{
	FLMBOOL	bOldVersion;

	bOldVersion = (FLMBOOL)((m_ui64HighTransId != FLM_MAX_UINT64)
									? TRUE
									: FALSE);

	// Unlink the node from its various lists.

	gv_XFlmSysData.pNodeCacheMgr->m_MRUList.unlinkGlobal(
					(F_CachedItem *)this);
	unlinkFromDatabase();
	
	if (!m_pNewerVersion)
	{
		F_CachedNode *	pOlderVersion = m_pOlderVersion;

		unlinkFromHashBucket();

		// If there was an older version, it now needs to be
		// put into the hash bucket.

		if (pOlderVersion)
		{
			unlinkFromVerList();
			pOlderVersion->linkToHashBucket();
		}
	}
	else
	{
		unlinkFromVerList();
	}
	
	if( m_uiFlags & FDOM_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

	// Free the F_CachedNode structure if not putting in purge list.

	if (!bPutInPurgeList)
	{
		delete this;
	}
	else
	{
		if ((m_pNextInGlobal = gv_XFlmSysData.pNodeCacheMgr->m_pPurgeList) != NULL)
		{
			m_pNextInGlobal->m_pPrevInGlobal = this;
		}
		gv_XFlmSysData.pNodeCacheMgr->m_pPurgeList = this;
		
		// Unset the dirty flags - don't want anything in the purge list
		// to be dirty.
		
		m_uiFlags &= ~(FDOM_DIRTY | FDOM_NEW);
		setPurged();
		flmAssert( !m_pPrevInGlobal);
	}
}

/****************************************************************************
Desc:	This routine initializes node cache manager.
****************************************************************************/
RCODE F_NodeCacheMgr::initCache( void)
{
	RCODE		rc = NE_XFLM_OK;

	// Allocate the hash buckets.

	if (RC_BAD( rc = f_calloc(
								(FLMUINT)sizeof( F_CachedNode *) *
								(FLMUINT)MIN_HASH_BUCKETS,
								&m_ppHashBuckets)))
	{
		goto Exit;
	}
	m_uiNumBuckets = MIN_HASH_BUCKETS;
	m_uiHashMask = m_uiNumBuckets - 1;
	gv_XFlmSysData.pGlobalCacheMgr->incrTotalBytes( f_msize( m_ppHashBuckets));

	// Set up the F_CachedNode object allocator

	if( RC_BAD( rc = FlmAllocFixedAllocator( &m_pNodeAllocator)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pNodeAllocator->setup( 
		FALSE, gv_XFlmSysData.pGlobalCacheMgr->m_pSlabManager, &m_nodeRelocator, 
		sizeof( F_CachedNode), &m_Usage.slabUsage, NULL)))
	{
		goto Exit;
	}

	// Set up the buffer allocator for F_CachedNode objects

	if( RC_BAD( rc = FlmAllocBufferAllocator( &m_pBufAllocator)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pBufAllocator->setup(
		FALSE, gv_XFlmSysData.pGlobalCacheMgr->m_pSlabManager, 
		NULL, &m_Usage.slabUsage, NULL)))
	{
		goto Exit;
	}

	// Set up the allocator for attribute items

	if( RC_BAD( rc = FlmAllocFixedAllocator( &m_pAttrItemAllocator)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pAttrItemAllocator->setup(
		FALSE, gv_XFlmSysData.pGlobalCacheMgr->m_pSlabManager,
		&m_attrItemRelocator, sizeof( F_AttrItem), &m_Usage.slabUsage, NULL)))
	{
		goto Exit;
	}

#ifdef FLM_DEBUG
	m_bDebug = TRUE;
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Determine if a node can be moved.
Notes:	This routine assumes the node cache mutex is locked
			This is a static method, so there is no "this" pointer to the
			F_NodeCacheMgr object.
****************************************************************************/
FLMBOOL F_NodeRelocator::canRelocate(
	void *		pvAlloc)
{
	return( ((F_CachedNode *)pvAlloc)->nodeInUse() ? FALSE : TRUE);
}

/****************************************************************************
Desc:		Fixes up all pointers needed to allow an F_CachedNode object to be
			moved to a different location in memory
Notes:	This routine assumes the node cache mutex is locked.
			This is a static method, so there is no "this" pointer to the
			F_NodeCacheMgr object.
****************************************************************************/
void F_NodeRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	F_CachedNode *			pOldNode = (F_CachedNode *)pvOldAlloc;
	F_CachedNode *			pNewNode = (F_CachedNode *)pvNewAlloc;
	F_CachedNode **		ppBucket;
	F_Database *			pDatabase = pOldNode->m_pDatabase;
	F_NodeCacheMgr *		pNodeCacheMgr = gv_XFlmSysData.pNodeCacheMgr;
	FLMBYTE *				pucActualAlloc;

	flmAssert( !pOldNode->nodeInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);

	// Update the F_CachedNode pointer in the data buffer

	if( pNewNode->m_pucData)
	{
		pucActualAlloc = getActualPointer( pNewNode->m_pucData);
		flmAssert( *((F_CachedNode **)(pucActualAlloc)) == pOldNode);
		pNewNode->setNodeAndDataPtr( pucActualAlloc);
	}

	if( pNewNode->m_pNodeList)
	{
		pucActualAlloc = getActualPointer( pNewNode->m_pNodeList);
		flmAssert( *((F_CachedNode **)(pucActualAlloc)) == pOldNode);
		pNewNode->setNodeListPtr( pucActualAlloc);
	}

	if( pNewNode->m_ppAttrList)
	{
		FLMUINT	uiLoop;
		
		pucActualAlloc = getActualPointer( pNewNode->m_ppAttrList);
		flmAssert( *((F_CachedNode **)(pucActualAlloc)) == pOldNode);
		pNewNode->setAttrListPtr( pucActualAlloc);
		
		for (uiLoop = 0; uiLoop < pNewNode->m_uiAttrCount; uiLoop++)
		{
			pNewNode->m_ppAttrList [uiLoop]->m_pCachedNode = pNewNode;
		}
	}

	if (pNewNode->m_pPrevInDatabase)
	{
		pNewNode->m_pPrevInDatabase->m_pNextInDatabase = pNewNode;
	}

	if (pNewNode->m_pNextInDatabase)
	{
		pNewNode->m_pNextInDatabase->m_pPrevInDatabase = pNewNode;
	}

	if (pNewNode->m_pPrevInGlobal)
	{
		pNewNode->m_pPrevInGlobal->m_pNextInGlobal = pNewNode;
	}

	if (pNewNode->m_pNextInGlobal)
	{
		pNewNode->m_pNextInGlobal->m_pPrevInGlobal = pNewNode;
	}

	if (pNewNode->m_pPrevInBucket)
	{
		pNewNode->m_pPrevInBucket->m_pNextInBucket = pNewNode;
	}

	if (pNewNode->m_pNextInBucket)
	{
		pNewNode->m_pNextInBucket->m_pPrevInBucket = pNewNode;
	}

	if (pNewNode->m_pOlderVersion)
	{
		pNewNode->m_pOlderVersion->m_pNewerVersion = pNewNode;
	}

	if (pNewNode->m_pNewerVersion)
	{
		pNewNode->m_pNewerVersion->m_pOlderVersion = pNewNode;
	}
	
	if (pNewNode->m_pPrevInHeapList)
	{
		pNewNode->m_pPrevInHeapList->m_pNextInHeapList = pNewNode;
	}
	
	if (pNewNode->m_pNextInHeapList)
	{
		pNewNode->m_pNextInHeapList->m_pPrevInHeapList = pNewNode;
	}

	if (pNewNode->m_pPrevInOldList)
	{
		pNewNode->m_pPrevInOldList->m_pNextInOldList = pNewNode;
	}
	
	if (pNewNode->m_pNextInOldList)
	{
		pNewNode->m_pNextInOldList->m_pPrevInOldList = pNewNode;
	}
	
	if( pDatabase)
	{
		if (pDatabase->m_pFirstNode == pOldNode)
		{
			pDatabase->m_pFirstNode = pNewNode;
		}

		if( pDatabase->m_pLastNode == pOldNode)
		{
			pDatabase->m_pLastNode = pNewNode;
		}
		
		if( pDatabase->m_pLastDirtyNode == pOldNode)
		{
			pDatabase->m_pLastDirtyNode = pNewNode;
		}
	}

	ppBucket = pNodeCacheMgr->nodeHash( pOldNode->m_nodeInfo.ui64NodeId);
	if( *ppBucket == pOldNode)
	{
		*ppBucket = pNewNode;
	}

	if (pNodeCacheMgr->m_MRUList.m_pMRUItem == (F_CachedItem *)pOldNode)
	{
		pNodeCacheMgr->m_MRUList.m_pMRUItem = pNewNode;
	}

	if (pNodeCacheMgr->m_MRUList.m_pLRUItem == (F_CachedItem *)pOldNode)
	{
		pNodeCacheMgr->m_MRUList.m_pLRUItem = pNewNode;
	}
	
	if (pNodeCacheMgr->m_pHeapList == pOldNode)
	{
		pNodeCacheMgr->m_pHeapList = pNewNode;
	}
	
	if (pNodeCacheMgr->m_pOldList == pOldNode)
	{
		pNodeCacheMgr->m_pOldList = pNewNode;
	}

	if (pNodeCacheMgr->m_pPurgeList == pOldNode)
	{
		pNodeCacheMgr->m_pPurgeList = pNewNode;
	}
}

/****************************************************************************
Desc:	Determine if a data buffer of an F_CachedNode object can be moved.
		This routine assumes that the node cache mutex is locked.
****************************************************************************/
FLMBOOL F_NodeDataRelocator::canRelocate(
	void *	pvAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvAlloc);
	
	if( pNode->nodeInUse())
	{
		return( FALSE);
	}
	else
	{
		flmAssert( getActualPointer( pNode->m_pucData) == (FLMBYTE *)pvAlloc);
		return( TRUE);
	}
}

/****************************************************************************
Desc:	Relocate the data buffer of an F_CachedNode object.  This routine assumes
		that the node cache mutex is locked.
****************************************************************************/
void F_NodeDataRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvOldAlloc);

	flmAssert( !pNode->nodeInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( getActualPointer( pNode->m_pucData) == (FLMBYTE *)pvOldAlloc);
	
	pNode->setNodeAndDataPtr( (FLMBYTE *)pvNewAlloc);
}

/****************************************************************************
Desc:	Determine if a node list of an F_CachedNode object can be moved.
		This routine assumes that the node cache mutex is locked.
****************************************************************************/
FLMBOOL F_NodeListRelocator::canRelocate(
	void *	pvAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvAlloc);
	
	if( pNode->nodeInUse())
	{
		return( FALSE);
	}
	else
	{
		flmAssert( getActualPointer( pNode->m_pNodeList) == (FLMBYTE *)pvAlloc);
		return( TRUE);
	}
}

/****************************************************************************
Desc:	Relocate the node list of an F_CachedNode object.  This routine assumes
		that the node cache mutex is locked.
****************************************************************************/
void F_NodeListRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvOldAlloc);

	flmAssert( !pNode->nodeInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( getActualPointer( pNode->m_pNodeList) == (FLMBYTE *)pvOldAlloc);
	
	pNode->setNodeListPtr( (FLMBYTE *)pvNewAlloc);
}

/****************************************************************************
Desc:	Determine if an attr list of an F_CachedNode object can be moved.
		This routine assumes that the node cache mutex is locked.
****************************************************************************/
FLMBOOL F_AttrListRelocator::canRelocate(
	void *	pvAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvAlloc);
	
	if( pNode->nodeInUse())
	{
		return( FALSE);
	}
	else
	{
		flmAssert( getActualPointer( pNode->m_ppAttrList) == (FLMBYTE *)pvAlloc);
		return( TRUE);
	}
}

/****************************************************************************
Desc:	Relocate the attr list of an F_CachedNode object.  This routine assumes
		that the node cache mutex is locked.
****************************************************************************/
void F_AttrListRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_CachedNode *	pNode = *((F_CachedNode **)pvOldAlloc);

	flmAssert( !pNode->nodeInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( getActualPointer( pNode->m_ppAttrList) == (FLMBYTE *)pvOldAlloc);
	
	pNode->setAttrListPtr( (FLMBYTE *)pvNewAlloc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_AttrItemRelocator::canRelocate(
	void *		pvAlloc)
{
	F_AttrItem *		pAttrItem = (F_AttrItem *)pvAlloc;
	
	if( pAttrItem->m_pCachedNode && !pAttrItem->m_pCachedNode->nodeInUse())
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_AttrItemRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_AttrItem *	pOldAttrItem = (F_AttrItem *)pvOldAlloc;
	F_AttrItem *	pNewAttrItem = (F_AttrItem *)pvNewAlloc;
	F_CachedNode *	pCachedNode = pNewAttrItem->m_pCachedNode;
	FLMUINT			uiPos;

	flmAssert( !pCachedNode->nodeInUse());
	
	// Find the new attr item slot
	
	if (pCachedNode->getAttribute( pNewAttrItem->m_uiNameId, &uiPos) ==
		 pOldAttrItem)
	{
		pCachedNode->m_ppAttrList [uiPos] = pNewAttrItem;
	}
	else
	{
		flmAssert( 0);
	}

	if( pOldAttrItem->m_uiPayloadLen > sizeof( FLMBYTE *))
	{
		*((F_AttrItem **)(pNewAttrItem->m_pucPayload - sizeof( F_AttrItem *))) = 
			pNewAttrItem;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_AttrBufferRelocator::canRelocate(
	void *	pvAlloc)
{
	F_AttrItem *	pAttrItem = *((F_AttrItem **)pvAlloc);

	flmAssert( pAttrItem->m_pucPayload == 
				 (FLMBYTE *)pvAlloc + sizeof( F_AttrItem *));
	
	if( pAttrItem->m_pCachedNode && !pAttrItem->m_pCachedNode->nodeInUse())
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_AttrBufferRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	F_AttrItem *	pAttrItem = *((F_AttrItem **)pvOldAlloc);

	flmAssert( !pAttrItem->m_pCachedNode->nodeInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( pAttrItem->m_pucPayload == 
				 (FLMBYTE *)pvOldAlloc + sizeof( F_AttrItem *));
	
	pAttrItem->m_pucPayload = 
		((FLMBYTE *)pvNewAlloc) + sizeof( F_AttrItem *);
}

/****************************************************************************
Desc:	This routine resizes the hash table for the cache manager.
		NOTE: This routine assumes that the node cache mutex has been locked.
****************************************************************************/
RCODE F_NodeCacheMgr::rehash( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiNewHashTblSize;
	F_CachedNode **	ppOldHashTbl;
	FLMUINT				uiOldHashTblSize;
	F_CachedNode **	ppBucket;
	FLMUINT				uiLoop;
	F_CachedNode *		pTmpNode;
	F_CachedNode *		pTmpNextNode;
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

	if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( F_CachedNode *) *
								(FLMUINT)uiNewHashTblSize, &m_ppHashBuckets)))
	{
		m_uiHashFailTime = FLM_GET_TIMER();
		m_ppHashBuckets = ppOldHashTbl;
		goto Exit;
	}

	// Subtract off old size and add in new size.

	gv_XFlmSysData.pGlobalCacheMgr->decrTotalBytes( uiOldMemSize);
	gv_XFlmSysData.pGlobalCacheMgr->incrTotalBytes( f_msize( m_ppHashBuckets));

	m_uiNumBuckets = uiNewHashTblSize;
	m_uiHashMask = uiNewHashTblSize - 1;

	// Relink all of the nodes into the new hash table.

	for (uiLoop = 0, ppBucket = ppOldHashTbl;
		  uiLoop < uiOldHashTblSize;
		  uiLoop++, ppBucket++)
	{
		pTmpNode = *ppBucket;
		while (pTmpNode)
		{
			pTmpNextNode = pTmpNode->m_pNextInBucket;
			pTmpNode->linkToHashBucket();
			pTmpNode = pTmpNextNode;
		}
	}

	// Throw away the old hash table.

	f_free( &ppOldHashTbl);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine shuts down the node cache manager and frees all
		resources allocated by it.  NOTE: Node cache mutex must be locked
		already, or we must be shutting down so that only one thread is
		calling this routine.
****************************************************************************/
F_NodeCacheMgr::~F_NodeCacheMgr()
{
	F_CachedItem *	pItem;
	F_CachedItem *	pNextItem;
	F_DOMNode *		pTmp;	

	// Free the DOM Node Pool

	while( (pTmp = m_pFirstNode) != NULL)
	{
		m_pFirstNode = m_pFirstNode->m_pNextInPool;
		pTmp->m_refCnt = 0;
		pTmp->m_pNextInPool = NULL;
		pTmp->m_pCachedNode = NULL;
		delete pTmp;
	}

	// Free all of the node cache objects.
	
	pItem = m_MRUList.m_pMRUItem;
	while (pItem)
	{
		pNextItem = pItem->m_pNextInGlobal;
		((F_CachedNode *)pItem)->freeCache( FALSE);
		pItem = pNextItem;
	}
	flmAssert( !m_MRUList.m_pMRUItem && !m_MRUList.m_pLRUItem);

	// Must free those in the purge list too.

	while (m_pPurgeList)
	{
		m_pPurgeList->freePurged();
	}

	// The math better be consistent!

	flmAssert( m_Usage.uiCount == 0);
	flmAssert( m_Usage.uiOldVerCount == 0);
	flmAssert( m_Usage.uiOldVerBytes == 0);

	// Free the hash bucket array

	if (m_ppHashBuckets)
	{
		FLMUINT	uiTotalMemory = f_msize( m_ppHashBuckets);
		
		f_free( &m_ppHashBuckets);
		gv_XFlmSysData.pGlobalCacheMgr->decrTotalBytes( uiTotalMemory);
	}

	// Free the allocators

	if (m_pNodeAllocator)
	{
		m_pNodeAllocator->Release();
	}

	if( m_pBufAllocator)
	{
		m_pBufAllocator->Release();
	}

	if( m_pAttrItemAllocator)
	{
		m_pAttrItemAllocator->Release();
	}
}

/****************************************************************************
Desc: This routine links a notify request into a node's notification list and
		then waits to be notified that the event has occurred.
		NOTE: This routine assumes that the node cache mutex is locked and that
		it is supposed to unlock it.  It will relock the mutex on its way out.
****************************************************************************/
RCODE F_NodeCacheMgr::waitNotify(
	F_Db *				pDb,
	F_CachedNode **	ppNode)
{
	return( f_notifyWait( gv_XFlmSysData.hNodeCacheMutex, 
		pDb->m_hWaitSem, ppNode, &((*ppNode)->m_pNotifyList)));
}

/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read to complete.
		NOTE:  This routine assumes that the node cache mutex is already locked.
****************************************************************************/
void F_NodeCacheMgr::notifyWaiters(
	F_NOTIFY_LIST_ITEM *	pNotify,
	F_CachedNode *			pUseNode,
	RCODE						NotifyRc)
{
	while (pNotify)
	{
		F_SEM	hSem;

		*(pNotify->pRc) = NotifyRc;
		if (RC_OK( NotifyRc))
		{
			*((F_CachedNode **)pNotify->pvData) = pUseNode;
			pUseNode->incrNodeUseCount();
		}
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
}

/****************************************************************************
Desc:	Allocate an F_CachedNode object.
****************************************************************************/
RCODE F_NodeCacheMgr::allocNode(
	F_CachedNode **	ppNode,
	FLMBOOL				bMutexLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bUnlockMutex = FALSE;
	
	if( !bMutexLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		bUnlockMutex = TRUE;
	}

	if ((*ppNode = new F_CachedNode) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	// Increment statistics.
	
	m_Usage.uiCount++;
	m_Usage.uiByteCount += (*ppNode)->memSize();
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
	
Exit:

	if( bUnlockMutex)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Cleanup old nodes in cache that are no longer needed by any
		transaction.  This routine assumes that the node cache mutex
		has been locked.
****************************************************************************/
void F_NodeCacheMgr::cleanupOldCache( void)
{
	F_CachedNode *		pCurNode;
	F_CachedNode *		pNextNode;

	pCurNode = m_pOldList;

	// Stay in the loop until we have freed all old nodes, or
	// we have run through the entire list.

	while( pCurNode)
	{
		flmAssert( pCurNode->m_ui64HighTransId != FLM_MAX_UINT64); 
		
		// Save the pointer to the next entry in the list because
		// we may end up unlinking pCurNode below, in which case we would
		// have lost the next node.

		pNextNode = pCurNode->m_pNextInOldList;
		if (!pCurNode->nodeInUse() &&
			 !pCurNode->readingInNode() &&
			 (!pCurNode->nodeLinkedToDatabase() ||
			  !pCurNode->m_pDatabase->neededByReadTrans( 
			  	pCurNode->m_ui64LowTransId, pCurNode->m_ui64HighTransId)))
		{
			pCurNode->freeNode();
		}
		pCurNode = pNextNode;
	}
}

/****************************************************************************
Desc:	Cleanup nodes that have been purged.  This routine assumes that the
		node cache mutex has been locked.
****************************************************************************/
void F_NodeCacheMgr::cleanupPurgedCache( void)
{
	F_CachedNode *		pCurNode;
	F_CachedNode *		pNextNode;

	pCurNode = m_pPurgeList;

	// Stay in the loop until we have freed all purged nodes, or
	// we have run through the entire list.

	while( pCurNode)
	{
		// Save the pointer to the next entry in the list because
		// we may end up unlinking pCurNode below, in which case we would
		// have lost the next node.

		pNextNode = (F_CachedNode *)pCurNode->m_pNextInGlobal;
		flmAssert( pCurNode->nodePurged());
		
		if (!pCurNode->nodeInUse())
		{
			pCurNode->freePurged();
		}
		pCurNode = pNextNode;
	}
}

/****************************************************************************
Desc:	Reduce node cache to below the cache limit.  NOTE: This routine assumes
		that the node cache mutex is locked upon entering the routine, but
		it may unlock and re-lock the mutex.
****************************************************************************/
void F_NodeCacheMgr::reduceCache( void)
{
	F_CachedNode *		pTmpNode;
	F_CachedNode *		pPrevNode;
	F_CachedNode *		pNextNode;
	FLMUINT				uiSlabSize;
	FLMUINT				uiByteThreshold;
	FLMUINT				uiSlabThreshold;
	FLMBOOL				bDoingReduce = FALSE;

	// Discard items that are allocated on the heap.  These are large
	// allocations that could not be satisfied by the buffer allocator and have
	// the side effect of causing memory fragmentation.

	pTmpNode = m_pHeapList;
	while( pTmpNode)
	{
		// Need to save the pointer to the next entry in the list because
		// we may end up freeing pTmpNode below.

		pNextNode = pTmpNode->m_pNextInHeapList;

		// See if the item can be freed.

		if( pTmpNode->canBeFreed())
		{
			// NOTE: This call will free the memory pointed to by
			// pTmpNode.  Hence, pTmpNode should NOT be used after
			// this point.

			pTmpNode->freeNode();
		}

		pTmpNode = pNextNode;
	}

	// If cache is not full, we are done.

	if( !gv_XFlmSysData.pGlobalCacheMgr->cacheOverLimit() || m_bReduceInProgress) 
	{
		goto Exit;
	}
	
	m_bReduceInProgress = TRUE;
	bDoingReduce = TRUE;

	// Cleanup cache that is no longer needed by anyone

	cleanupOldCache();
	cleanupPurgedCache();
	
	// Determine the cache threshold

	uiSlabThreshold = gv_XFlmSysData.pGlobalCacheMgr->m_uiMaxSlabs >> 1;
	uiSlabSize = gv_XFlmSysData.pGlobalCacheMgr->m_pSlabManager->getSlabSize();
	
	// Are we over the threshold?

	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold) 
	{
		goto Exit;
	}
	
	// Remove items from cache starting from the LRU

	pTmpNode = (F_CachedNode *)m_MRUList.m_pLRUItem;
	uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
								? m_Usage.uiByteCount - uiSlabSize
								: 0;

	while( pTmpNode)
	{
		// Need to save the pointer to the next entry in the list because
		// we may end up freeing pTmpNode below.

		pPrevNode = (F_CachedNode *)pTmpNode->m_pPrevInGlobal;

		// See if the item can be freed.

		if( pTmpNode->canBeFreed())
		{
			pTmpNode->freeNode();
			
			if( m_Usage.uiByteCount <= uiByteThreshold)
			{
				if( pPrevNode)
				{
					pPrevNode->incrNodeUseCount();
				}
				
				gv_XFlmSysData.pNodeCacheMgr->defragmentMemory( TRUE);
				
				if( !pPrevNode)
				{
					break;
				}
				
				pPrevNode->decrNodeUseCount();
	
				// We're going to quit when we get under 50 percent for node cache
				// or we aren't over the global limit.  Note that this means we
				// may quit reducing before we get under the global limit.  We
				// don't want to get into a situation where we are starving node
				// cache because block cache is over its limit.
				
				if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
					!gv_XFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
				{
					goto Exit;
				}

				uiByteThreshold = uiByteThreshold > uiSlabSize 
											? uiByteThreshold - uiSlabSize
											: 0;
			}
		}

		pTmpNode = pPrevNode;
	}

Exit:

	if( bDoingReduce)
	{
		m_bReduceInProgress = FALSE;
	}

	return;
}

/****************************************************************************
Desc:	This routine finds a node in the node cache.  If it cannot
		find the node, it will return the position where the node should
		be inserted.
		NOTE: This routine assumes that the node cache mutex has been locked.
****************************************************************************/
void F_NodeCacheMgr::findNode(
	F_Db *				pDb,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	FLMUINT64			ui64VersionNeeded,
	FLMBOOL				bDontPoisonCache,
	FLMUINT *			puiNumLooks,
	F_CachedNode **	ppNode,
	F_CachedNode **	ppNewerNode,
	F_CachedNode **	ppOlderNode)
{
	F_CachedNode *		pNode;
	FLMUINT				uiNumLooks = 0;
	FLMBOOL				bFound;
	F_CachedNode *		pNewerNode;
	F_CachedNode *		pOlderNode;
	F_Database *		pDatabase = pDb->m_pDatabase;

	// Search down the hash bucket for the matching item.

Start_Find:

	// NOTE: Need to always calculate hash bucket because
	// the hash table may have been changed while we
	// were waiting to be notified below - mutex can
	// be unlocked, but it is guaranteed to be locked
	// here.

	pNode = *(nodeHash( ui64NodeId));
	bFound = FALSE;
	uiNumLooks = 1;
	while (pNode &&
			 (pNode->m_nodeInfo.ui64NodeId != ui64NodeId ||
			  pNode->m_nodeInfo.uiCollection != uiCollection ||
			  pNode->m_pDatabase != pDatabase))
	{
		if ((pNode = pNode->m_pNextInBucket) != NULL)
		{
			uiNumLooks++;
		}
	}

	// If we found the node, see if we have the right version.

	if (!pNode)
	{
		pNewerNode = pOlderNode = NULL;
	}
	else
	{
		pNewerNode = NULL;
		pOlderNode = pNode;
		for (;;)
		{

			// If this one is being read in, we need to wait on it.

			if (pNode->readingInNode())
			{
				// We need to wait for this record to be read in
				// in case it coalesces with other versions, resulting
				// in a version that satisfies our request.

				m_uiIoWaits++;
				if (RC_BAD( waitNotify( pDb, &pNode)))
				{

					// Don't care what error the other thread had reading
					// the thing in from disk - we'll bail out and start
					// our find again.

					goto Start_Find;
				}

				//	The thread doing the notify "uses" the record cache
				// on behalf of this thread to prevent the record
				// from being replaced after it unlocks the mutex.
				// At this point, since we have locked the mutex,
				// we need to release the record - because we
				// will put a "use" on it below.

				pNode->decrNodeUseCount();

				if (pNode->nodePurged())
				{
					if (!pNode->nodeInUse())
					{
						pNode->freePurged();
					}
				}

				// Start over with the find because the list
				// structure has changed.

				goto Start_Find;
			}

			// See if this record version is the one we need.

			if (ui64VersionNeeded < pNode->m_ui64LowTransId)
			{
				pNewerNode = pNode;
				if ((pOlderNode = pNode = pNode->m_pOlderVersion) == NULL)
				{
					break;
				}
				uiNumLooks++;
			}
			else if (ui64VersionNeeded <= pNode->m_ui64HighTransId)
			{

				// Make this the MRU record.

				if (puiNumLooks)
				{
					if (bDontPoisonCache)
					{
						m_MRUList.stepUpInGlobal( (F_CachedItem *)pNode);
					}
					else if (pNode->m_pPrevInGlobal)
					{
						m_MRUList.unlinkGlobal( (F_CachedItem *)pNode);
						m_MRUList.linkGlobalAsMRU( (F_CachedItem *)pNode);
					}
					m_Usage.uiCacheHits++;
					m_Usage.uiCacheHitLooks += uiNumLooks;
				}
				bFound = TRUE;
				break;
			}
			else
			{
				pOlderNode = pNode;
				pNewerNode = pNode->m_pNewerVersion;

				// Set pNode to NULL as an indicator that we did not
				// find the version we needed.

				pNode = NULL;
				break;
			}
		}
	}

	*ppNode = pNode;

	if( ppOlderNode)
	{
		*ppOlderNode = pOlderNode;
	}

	if( ppNewerNode)
	{
		*ppNewerNode = pNewerNode;
	}

	if (puiNumLooks)
	{
		*puiNumLooks = uiNumLooks;
	}
}

/****************************************************************************
Desc:	This routine links a new node into the global list and
		into the correct place in its hash bucket.  This routine assumes that
		the node cache mutex is already locked.
****************************************************************************/
void F_NodeCacheMgr::linkIntoNodeCache(
	F_CachedNode *	pNewerNode,
	F_CachedNode *	pOlderNode,
	F_CachedNode *	pNode,
	FLMBOOL			bLinkAsMRU
	)
{
	if( bLinkAsMRU)
	{
		m_MRUList.linkGlobalAsMRU( (F_CachedItem *)pNode);
	}
	else
	{
		m_MRUList.linkGlobalAsLRU( (F_CachedItem *)pNode);
	}

	if (pNewerNode)
	{
		pNode->linkToVerList( pNewerNode, pOlderNode);
	}
	else
	{
		if (pOlderNode)
		{
			pOlderNode->unlinkFromHashBucket();
		}
		pNode->linkToHashBucket();
		pNode->linkToVerList( NULL, pOlderNode);
	}
}

/****************************************************************************
Desc:	This routine links a new node to its F_Database according to whether
		or not it is an update transaction or a read transaction.
		It coalesces out any unnecessary versions. This routine assumes 
		that the node cache mutex is already locked.
****************************************************************************/
void F_CachedNode::linkToDatabase(
	F_Database *		pDatabase,
	F_Db *				pDb,
	FLMUINT64			ui64LowTransId,
	FLMBOOL				bMostCurrent)
{
	F_CachedNode *	pTmpNode;

	m_ui64LowTransId = ui64LowTransId;

	// Before coalescing, link to F_Database.
	// The following test determines if the node is an
	// uncommitted version generated by the update transaction.
	// If so, we mark it as such, and link it at the head of the
	// F_Database list - so we can get rid of it quickly if we abort
	// the transaction.

	if (pDb->getTransType() == XFLM_UPDATE_TRANS)
	{

		// If we are in an update transaction, there better not
		// be any newer versions in the list and the high
		// transaction ID returned better be FLM_MAX_UINT64.

		flmAssert( m_pNewerVersion == NULL);
		setTransID( FLM_MAX_UINT64);

		// If the low transaction ID is the same as the transaction,
		// we may have modified this node during the transaction.
		// Unfortunately, there is no sure way to tell, so we are
		// forced to assume it may have been modified.  If the
		// transaction aborts, we will get rid if this version out
		// of cache.

		if (ui64LowTransId == pDb->getTransID())
		{
			setUncommitted();
			linkToDatabaseAtHead( pDatabase);
		}
		else
		{
			unsetUncommitted();
			linkToDatabaseAtEnd( pDatabase);
		}
	}
	else
	{
		FLMUINT64	ui64HighTransId;

		// Adjust the high transaction ID to be the same as
		// the transaction ID - we may have gotten a FLM_MAX_UINT64
		// back, but that is possible even if the node is
		// not the most current version.  Besides that, it is
		// possible that in the mean time one or more update
		// transactions have come along and created one or
		// more newer versions of the node.

		if (bMostCurrent)
		{
			// This may be showing up as most current simply because we have
			// a newer node that was dirty - meaning it would not have been
			// written to block cache yet - so our read operation would have
			// read the "most current" version of the block that contains
			// this node - but it isn't really the most current version of
			// the node.

			if (m_pNewerVersion && !m_pNewerVersion->readingInNode())
			{
				ui64HighTransId = m_pNewerVersion->getLowTransId() - 1;
			}
			else
			{
				ui64HighTransId = FLM_MAX_UINT64;
			}
		}
		else
		{
			ui64HighTransId = pDb->getTransID();
		}

		setTransID( ui64HighTransId);

		// For a read transaction, if there is a newer version,
		// it better have a higher "low transaction ID"

#ifdef FLM_DEBUG
		if (m_pNewerVersion && !m_pNewerVersion->readingInNode())
		{
			flmAssert( m_ui64HighTransId < m_pNewerVersion->m_ui64LowTransId);
			if( m_ui64HighTransId >= m_pNewerVersion->m_ui64LowTransId)
			{
				checkReadFromDisk( pDb);
			}
		}
#endif
		unsetUncommitted();
		linkToDatabaseAtEnd( pDatabase);
	}

	// Coalesce any versions that overlap - can only
	// coalesce older versions.  For an updater, there
	// should not be any newer versions.  For a reader, it
	// is impossible to know how high up it can coalesce.
	// The read operation that read the node may have
	// gotten back a FLM_MAX_UINT64 for its high transaction
	// ID - but after that point in time, it is possible
	// that one or more update transactions may have come
	// along and created one or more newer versions that
	// it would be incorrect to coalesce with.
	// In reality, a read transaction has to ignore the
	// FLM_MAX_UINT64 in the high transaction ID anyway
	// because there is no way to know if it is correct.

	// Coalesce older versions.

	for (;;)
	{
		if ((pTmpNode = m_pOlderVersion) == NULL)
		{
			break;
		}

		// Stop if we encounter one that is being read in.

		if (pTmpNode->readingInNode())
		{
			break;
		}

		// If there is no overlap between these two, there is
		// nothing more to coalesce.

		if (m_ui64LowTransId > pTmpNode->m_ui64HighTransId)
		{
			break;
		}

		if (m_ui64HighTransId <= pTmpNode->m_ui64HighTransId)
		{
			// This assert represents the following case,
			// which should not be possible to hit:
			
			// pOlder->m_ui64HighTransId > m_ui64HighTransId.
			//	This cannot be, because if pOlder has a higher
			//	transaction ID, we would have found it up above and
			//	not tried to have read it in.

			flmAssert( 0);
#ifdef FLM_DEBUG
			checkReadFromDisk( pDb);
#endif
		}
		else if (m_ui64LowTransId >= pTmpNode->m_ui64LowTransId)
		{
			m_ui64LowTransId = pTmpNode->m_ui64LowTransId;
			pTmpNode->freeCache(
						(FLMBOOL)((pTmpNode->nodeInUse() ||
									  pTmpNode->readingInNode())
									 ? TRUE
									 : FALSE));
		}
		else
		{
			// This assert represents the following case,
			// which should not be possible to hit:
			
			// m_ui64LowTransId < pOlder->m_ui64LowTransId.
			//	This cannot be, because pOlder has to have been read
			//	in to memory by a transaction whose transaction ID is
			//	less than or equal to our own.  That being the case,
			//	it would be impossible for our transaction to have
			//	found a version of the node that is older than pOlder.

			flmAssert( 0);
#ifdef FLM_DEBUG
			checkReadFromDisk( pDb);
#endif
		}
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::resizeDataBuffer(
	FLMUINT		uiSize,
	FLMBOOL		bMutexAlreadyLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiOldSize;
	FLMUINT		uiNewSize;
	FLMUINT		uiDataBufSize = calcDataBufSize( uiSize);
	FLMBYTE *	pucActualAlloc;
	FLMBOOL		bHeapAlloc = FALSE;
	void *		pvThis = this;
	FLMBOOL		bLockedMutex = FALSE;

	flmAssert( !m_uiDataBufSize || m_pucData);
	flmAssert( !m_uiStreamUseCount);

	if( uiDataBufSize == m_uiDataBufSize)
	{
		goto Exit;
	}

	if( !bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		bLockedMutex = TRUE;
	}

	uiOldSize = memSize();
	
	if (!m_pucData)
	{
		pucActualAlloc = NULL;
		if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->allocBuf(
			&gv_XFlmSysData.pNodeCacheMgr->m_nodeDataRelocator,
			uiDataBufSize, &pvThis, sizeof( void *), 
			&pucActualAlloc, &bHeapAlloc)))
		{
			goto Exit;
		}
	}
	else
	{
		pucActualAlloc = getActualPointer( m_pucData);
		if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->reallocBuf(
			&gv_XFlmSysData.pNodeCacheMgr->m_nodeDataRelocator,
			m_uiDataBufSize, uiDataBufSize, &pvThis, sizeof( void *),
			&pucActualAlloc, &bHeapAlloc)))
		{
			goto Exit;
		}
	}
	
	flmAssert( *((F_CachedNode **)pucActualAlloc) == this);
	setNodeAndDataPtr( pucActualAlloc);
	m_uiDataBufSize = uiDataBufSize;
	uiNewSize = memSize();
	
	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiOldSize);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiOldSize;
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiNewSize;
	}

	flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiOldSize);
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiOldSize;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiNewSize;
	
	if( bHeapAlloc)
	{
		linkToHeapList();
	}
	else if( m_uiFlags & FDOM_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

Exit:

	if( bLockedMutex)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	flmAssert( !m_uiDataBufSize || m_pucData);
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::resizeChildElmList(
	FLMUINT	uiChildElmCount,
	FLMBOOL	bMutexAlreadyLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiOldSize;
	FLMBYTE *	pucActualAlloc;
	FLMBOOL		bHeapAlloc = FALSE;
	void *		pvThis = this;
	
	if( uiChildElmCount == m_nodeInfo.uiChildElmCount)
	{
		goto Exit;
	}

	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	}

	uiOldSize = memSize();
	
	if( !uiChildElmCount)
	{
		// The only thing we better be doing if we pass in a zero, is
		// reducing the number of child elements.  Hence, the current
		// child element count better be non-zero.
		
		flmAssert( m_nodeInfo.uiChildElmCount);
		
		pucActualAlloc = getActualPointer( m_pNodeList);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
							calcNodeListBufSize( m_nodeInfo.uiChildElmCount),
							&pucActualAlloc);
	}
	else
	{
		if( !m_nodeInfo.uiChildElmCount)
		{
			pucActualAlloc = NULL;
			rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->allocBuf(
								&gv_XFlmSysData.pNodeCacheMgr->m_nodeListRelocator,
								calcNodeListBufSize( uiChildElmCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		else
		{
			pucActualAlloc = getActualPointer( m_pNodeList);
			rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->reallocBuf(
								&gv_XFlmSysData.pNodeCacheMgr->m_nodeListRelocator,
								calcNodeListBufSize( m_nodeInfo.uiChildElmCount),
								calcNodeListBufSize( uiChildElmCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		
		flmAssert( *((F_CachedNode **)pucActualAlloc) == this);
	}
	
	if (RC_OK( rc))
	{
		FLMUINT	uiNewSize;
		
		m_nodeInfo.uiChildElmCount = uiChildElmCount;
		if (m_nodeInfo.uiChildElmCount)
		{
			setNodeListPtr( pucActualAlloc);
		}
		else
		{
			m_pNodeList = NULL;
		}

		uiNewSize = memSize();

		if (m_ui64HighTransId != FLM_MAX_UINT64)
		{
			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiOldSize);
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiOldSize;
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiNewSize;
		}

		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiOldSize);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiOldSize;
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiNewSize;
		
		if( bHeapAlloc)
		{
			linkToHeapList();
		}
		else if( m_uiFlags & FDOM_HEAP_ALLOC)
		{
			unlinkFromHeapList();
		}
	}
	
	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::resizeAttrList(
	FLMUINT	uiAttrCount,
	FLMBOOL	bMutexAlreadyLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiOldSize;
	FLMBYTE *	pucActualAlloc;
	FLMBOOL		bHeapAlloc = FALSE;
	void *		pvThis = this;
	
	if( uiAttrCount == m_uiAttrCount)
	{
		goto Exit;
	}

	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	}

	uiOldSize = memSize();
	
	if( !uiAttrCount)
	{
		// The only thing we better be doing if we pass in a zero, is
		// reducing the number of attributes.  Hence, the current
		// attribute count better be non-zero.
		
		flmAssert( m_uiAttrCount);
		pucActualAlloc = getActualPointer( m_ppAttrList);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
							calcAttrListBufSize( m_uiAttrCount),
							&pucActualAlloc);
	}
	else
	{
		if( !m_uiAttrCount)
		{
			pucActualAlloc = NULL;
			rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->allocBuf(
								&gv_XFlmSysData.pNodeCacheMgr->m_attrListRelocator,
								calcAttrListBufSize( uiAttrCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		else
		{
			pucActualAlloc = getActualPointer( m_ppAttrList);
			rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->reallocBuf(
								&gv_XFlmSysData.pNodeCacheMgr->m_attrListRelocator,
								calcAttrListBufSize( m_uiAttrCount),
								calcAttrListBufSize( uiAttrCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		
		flmAssert( *((F_CachedNode **)pucActualAlloc) == this);
	}
	
	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	if (RC_OK( rc))
	{
		FLMUINT	uiNewSize;
		
		m_uiAttrCount = uiAttrCount;
		if (m_uiAttrCount)
		{
			setAttrListPtr( pucActualAlloc);
		}
		else
		{
			m_ppAttrList = NULL;
		}

		uiNewSize = memSize();

		if (m_ui64HighTransId != FLM_MAX_UINT64)
		{
			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiOldSize);
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiOldSize;
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiNewSize;
		}

		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiOldSize);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiOldSize;
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiNewSize;
		
		if( bHeapAlloc)
		{
			linkToHeapList();
		}
		else if( m_uiFlags & FDOM_HEAP_ALLOC)
		{
			unlinkFromHeapList();
		}
	}
	
	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine retrieves a node from disk.
****************************************************************************/
RCODE F_NodeCacheMgr::readNodeFromDisk(
	F_Db *				pDb,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	F_CachedNode *		pNode,
	FLMUINT64 *			pui64LowTransId,
	FLMBOOL *			pbMostCurrent)
{
	RCODE					rc = NE_XFLM_OK;
	F_Btree *			pBTree = NULL;
	FLMBOOL				bCloseIStream = FALSE;
	F_BTreeIStream		btreeIStream;

	if( RC_BAD( rc = pDb->getCachedBTree( uiCollection, &pBTree)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = btreeIStream.openStream( pDb, pBTree, XFLM_EXACT,
								uiCollection, ui64NodeId, 0, 0)))
	{
		goto Exit;
	}
	bCloseIStream = TRUE;
	
	// Read the node from the B-Tree

	if (RC_BAD( rc = pNode->readNode( pDb, uiCollection, 
		ui64NodeId, &btreeIStream, (FLMUINT)btreeIStream.remainingSize(),
		NULL)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}

		goto Exit;
	}
	
	pNode->m_uiOffsetIndex = btreeIStream.getOffsetIndex();
	pNode->m_ui32BlkAddr = btreeIStream.getBlkAddr();

	pBTree->btGetTransInfo( pui64LowTransId, pbMostCurrent);

Exit:

	if( bCloseIStream)
	{
		btreeIStream.closeStream();
	}

	if( pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine retrieves a node from the node cache.
****************************************************************************/
RCODE F_NodeCacheMgr::retrieveNode(
	F_Db *			pDb,
	FLMUINT			uiCollection,			// Collection node is in.
	FLMUINT64		ui64NodeId,				// Node ID
	F_DOMNode **	ppDOMNode)
{
	RCODE							rc = NE_XFLM_OK;
	FLMBOOL						bMutexLocked = FALSE;
	F_Database *				pDatabase = pDb->m_pDatabase;
	F_CachedNode *				pNode;
	F_CachedNode *				pNewerNode;
	F_CachedNode *				pOlderNode;
	FLMUINT64					ui64LowTransId;
	FLMBOOL						bMostCurrent;
	FLMUINT64					ui64CurrTransId;
	F_NOTIFY_LIST_ITEM *		pNotify;
	FLMUINT						uiNumLooks;
	FLMBOOL						bDontPoisonCache = pDb->m_uiFlags & FDB_DONT_POISON_CACHE
														? TRUE 
														: FALSE;

	if (RC_BAD( rc = pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Get the current transaction ID

	flmAssert( pDb->m_eTransType != XFLM_NO_TRANS);
	ui64CurrTransId = pDb->getTransID();
	flmAssert( ui64NodeId != 0);

	// Lock the node cache mutex

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;

	// Reset the DB's inactive time

	pDb->m_uiInactiveTime = 0;

Start_Find:

	findNode( pDb, uiCollection, ui64NodeId,
						ui64CurrTransId, bDontPoisonCache, 
						&uiNumLooks, &pNode,
						&pNewerNode, &pOlderNode);

	if (pNode)
	{
		// Have the DOM Node point to the node we found
		goto Exit1;
	}

	// Did not find the node, fetch from disk
	// Increment the number of faults only if we retrieve the record from disk.

	m_Usage.uiCacheFaults++;
	m_Usage.uiCacheFaultLooks += uiNumLooks;

	// Create a place holder for the object.

	if (RC_BAD( rc = allocNode( &pNode, TRUE)))
	{
		goto Exit;
	}

	pNode->m_nodeInfo.ui64NodeId = ui64NodeId;
	pNode->m_nodeInfo.uiCollection = uiCollection;

	// Set the F_Database so that other threads looking for this node in
	// cache will find it and wait until the read has completed.  If
	// the F_Database is not set, other threads will attempt their own read,
	// because they won't match a NULL F_Database.  The result of not setting
	// the F_Database is that multiple copies of the same version of a particular
	// node could end up in cache.
	
	pNode->m_pDatabase = pDatabase;

	linkIntoNodeCache( pNewerNode, pOlderNode, pNode, !bDontPoisonCache);

	pNode->setReadingIn();
	pNode->incrNodeUseCount();
	pNode->m_pNotifyList = NULL;

	// Unlock mutex before reading in from disk.

	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = FALSE;

	// Read node from disk.

	rc = readNodeFromDisk( pDb, uiCollection, ui64NodeId, pNode,
						&ui64LowTransId, &bMostCurrent);

	// Relock mutex

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;

	// If read was successful, link the node to its place in
	// the F_Database list and coalesce any versions that overlap
	// this one.

	if (RC_OK( rc))
	{
		pNode->linkToDatabase(
				pDb->m_pDatabase, pDb, ui64LowTransId, bMostCurrent);
	}

	pNode->unsetReadingIn();

	// Notify any threads waiting for the read to complete.

	pNotify = pNode->m_pNotifyList;
	pNode->m_pNotifyList = NULL;
	if (pNotify)
	{
		notifyWaiters( pNotify,
				(F_CachedNode *)((RC_BAD( rc))
							  ? (F_CachedNode *)NULL
							  : pNode), rc);
	}
	pNode->decrNodeUseCount();

	// If we did not succeed, free the F_CachedNode structure.

	if (RC_BAD( rc))
	{
		pNode->freeCache( FALSE);
		goto Exit;
	}

	// If this item was purged while we were reading it in,
	// start over with the search.

	if (pNode->nodePurged())
	{
		if (!pNode->nodeInUse())
		{
			pNode->freePurged();
		}

		// Start over with the find - this one has
		// been marked for purging.

		goto Start_Find;
	}

Exit1:

	// Have the DOM Node point to the node we read in from disk

	if( *ppDOMNode == NULL)
	{	
		if( RC_BAD( rc = allocDOMNode( ppDOMNode)))
		{
			goto Exit;
		}
	}

	if ( (*ppDOMNode)->m_pCachedNode)
	{
		(*ppDOMNode)->m_pCachedNode->decrNodeUseCount();
	}
	(*ppDOMNode)->m_pCachedNode = pNode;
	pNode->incrNodeUseCount();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine creates a node into the node cache.  This is ONLY
		called when a new node is being created.
****************************************************************************/
RCODE F_NodeCacheMgr::createNode(
	F_Db *			pDb,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	F_DOMNode **	ppDOMNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_Database *		pDatabase = pDb->m_pDatabase;
	F_COLLECTION *		pCollection;
	F_CachedNode *		pNode = NULL;
	F_CachedNode *		pNewerNode = NULL;
	F_CachedNode *		pOlderNode = NULL;
	FLMBOOL				bMutexLocked = FALSE;
	
	// A zero ui64NodeId means we are to use the next node ID for the
	// collection.

	if( !ui64NodeId)
	{
		if (RC_BAD( rc = pDb->m_pDict->getCollection( uiCollection, &pCollection)))
		{
			goto Exit;
		}

		ui64NodeId = pCollection->ui64NextNodeId;

		// Lock the node cache mutex

		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		bMutexLocked = TRUE;
	}
	else
	{
		// Lock the node cache mutex

		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		bMutexLocked = TRUE;

		// See if we can find the node in cache

		findNode( pDb, uiCollection, ui64NodeId,
							pDb->m_ui64CurrTransID,	TRUE, NULL, &pNode,
							&pNewerNode, &pOlderNode);

		if (pNode)
		{
			// If we found the last committed version, instead of replacing it,
			// we want to change its high transaction ID, and go create a new
			// node to put in cache.

			if (pNode->m_ui64LowTransId < pDb->m_ui64CurrTransID)
			{

				// pOlderNode and pNode should be the same at this point if we
				// found something.  Furthermore, the high transaction ID on what
				// we found better be -1 - most current version.

				flmAssert( pOlderNode == pNode);
				flmAssert( pOlderNode->m_ui64HighTransId == FLM_MAX_UINT64);

				pOlderNode->setTransID( (pDb->m_ui64CurrTransID - 1));

				flmAssert( pOlderNode->m_ui64HighTransId >= 
							  pOlderNode->m_ui64LowTransId);

				pOlderNode->setUncommitted();
				pOlderNode->setLatestVer();
				pOlderNode->unlinkFromDatabase();
				pOlderNode->linkToDatabaseAtHead( pDatabase);
			}
			else
			{
				// Found latest UNCOMMITTED VERSION
				
				pNode = NULL;
				rc = RC_SET_AND_ASSERT( NE_XFLM_EXISTS);
				goto Exit;
			}
		}
	}
	
	// We are positioned to insert the new node.  For an update, it
	// must always be the newest version.

	flmAssert( !pNewerNode);
	
	// Create a new object.

	if (RC_BAD( rc = allocNode( &pNode, bMutexLocked)))
	{
		goto Exit;
	}

	pNode->m_nodeInfo.ui64NodeId = ui64NodeId;
	pNode->m_nodeInfo.uiCollection = uiCollection;
	pNode->m_uiOffsetIndex = 0;
	pNode->m_ui32BlkAddr = 0;
	
	// NOTE: Not everything is initialized in pNode at this point, but
	// no other thread should be accessing it anyway.  The caller of this
	// function must ensure that all of the necessary items get set before
	// releasing the node.

	// Set the F_Database so that other threads looking for this node in
	// cache will find it and wait until the read has completed.  If
	// the F_Database is not set, other threads will attempt their own read,
	// because they won't match a NULL F_Database.  The result of not setting
	// the F_Database is that multiple copies of the same version of a particular
	// node could end up in cache.
	
	pNode->m_pDatabase = pDatabase;

	linkIntoNodeCache( pNewerNode, pOlderNode, pNode, TRUE);

	// Link the node to its place in the F_Database list

	pNode->linkToDatabase( pDatabase, pDb, pDb->m_ui64CurrTransID, TRUE);

	// Have the DOM node point to the node we created

	flmAssert( *ppDOMNode == NULL);
	
	if( RC_BAD( rc = allocDOMNode( ppDOMNode)))
	{
		goto Exit;
	}

	if ( (*ppDOMNode)->m_pCachedNode)
	{
		(*ppDOMNode)->m_pCachedNode->decrNodeUseCount();
	}
	(*ppDOMNode)->m_pCachedNode = pNode;
	pNode->incrNodeUseCount();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	This routine makes a writeable copy of the node pointed to by F_DOMNode.
****************************************************************************/
RCODE F_NodeCacheMgr::_makeWriteCopy(
	F_Db *				pDb,
	F_CachedNode **	ppCachedNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_Database *		pDatabase = pDb->m_pDatabase;
	F_CachedNode *		pNewerNode = NULL;
	F_CachedNode *		pOlderNode = *ppCachedNode;
	FLMBOOL				bMutexLocked = FALSE;
	
	flmAssert( pOlderNode->m_ui64HighTransId == FLM_MAX_UINT64);
	flmAssert( !pOlderNode->m_pNewerVersion);
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;

	// Create a new object.

	if (RC_BAD( rc = allocNode( &pNewerNode, TRUE)))
	{
		goto Exit;
	}

	// If we found the last committed version, instead of replacing it,
	// we want to change its high transaction ID, and go create a new
	// node to put in cache.

	// Although this routine could be written to not do anything if we
	// are already on the uncommitted version of the node, for performance
	// reasons, we would prefer that they make the check on the outside
	// before calling this routine.
	
	flmAssert( pOlderNode->m_ui64LowTransId < pDb->m_ui64CurrTransID);
	pOlderNode->setTransID( pDb->m_ui64CurrTransID - 1);
	flmAssert( pOlderNode->m_ui64HighTransId >= pOlderNode->m_ui64LowTransId);

	pOlderNode->setUncommitted();
	pOlderNode->setLatestVer();
	pOlderNode->unlinkFromDatabase();
	pOlderNode->linkToDatabaseAtHead( pDatabase);

	pNewerNode->m_pDatabase = pDatabase;
	pNewerNode->m_uiFlags = pOlderNode->m_uiFlags;
	pNewerNode->m_uiOffsetIndex = pOlderNode->m_uiOffsetIndex;
	pNewerNode->m_ui32BlkAddr = pOlderNode->m_ui32BlkAddr;
	
	if( pNewerNode->m_uiFlags & FDOM_HEAP_ALLOC)
	{
		pNewerNode->m_uiFlags &= ~FDOM_HEAP_ALLOC;
	}
	
	f_memcpy( &pNewerNode->m_nodeInfo, 
		&pOlderNode->m_nodeInfo, sizeof( F_NODE_INFO));

	if (pNewerNode->m_uiFlags & (FDOM_SIGNED_QUICK_VAL | FDOM_UNSIGNED_QUICK_VAL))
	{
		pNewerNode->m_numberVal = pOlderNode->m_numberVal;
	}
	
	if( pNewerNode->m_uiFlags & FDOM_HAVE_CELM_LIST)
	{
		
		// Need to set to zero, because we really haven't allocated
		// space for it yet.
		
		pNewerNode->m_nodeInfo.uiChildElmCount = 0;
		
		if( pOlderNode->m_nodeInfo.uiChildElmCount)
		{
			if( RC_BAD( rc = pNewerNode->resizeChildElmList( 
				pOlderNode->m_nodeInfo.uiChildElmCount, TRUE)))
			{
				goto Exit;
			}
			
			f_memcpy( pNewerNode->m_pNodeList, pOlderNode->m_pNodeList,
						 sizeof( NODE_ITEM) * pNewerNode->m_nodeInfo.uiChildElmCount);
		}
	}
	else
	{
		flmAssert( !pOlderNode->m_nodeInfo.uiChildElmCount);
	}
	
	if( !(pNewerNode->m_uiFlags & FDOM_VALUE_ON_DISK))
	{
		if( pNewerNode->getDataLength())
		{
			if (RC_BAD( rc = pNewerNode->resizeDataBuffer( 
				pNewerNode->getDataLength(), TRUE)))
			{
				goto Exit;
			}
		
			f_memcpy( pNewerNode->getDataPtr(), pOlderNode->getDataPtr(),
							 pNewerNode->getDataLength());
		}
	}
	else
	{
		flmAssert( pNewerNode->getDataLength());
		flmAssert( !pNewerNode->m_nodeInfo.uiChildElmCount);
	}
	
	if( pOlderNode->m_uiAttrCount)
	{
		if( RC_BAD( rc = pNewerNode->importAttributeList( 
			pDb, pOlderNode, TRUE)))
		{
			goto Exit;
		}
	}
		
	linkIntoNodeCache( NULL, pOlderNode, pNewerNode, TRUE);

	// Link the node to its place in the F_Database list

	pNewerNode->linkToDatabase( pDatabase, pDb, pDb->m_ui64CurrTransID, TRUE);

	// Update the node pointer passed into the routine
	
	if( *ppCachedNode)
	{
		(*ppCachedNode)->decrNodeUseCount();
	}
	
	*ppCachedNode = pNewerNode;
	pNewerNode->incrNodeUseCount();

	// Set pNewerNode to NULL so it won't get freed at Exit

	pNewerNode = NULL;

Exit:

	// A non-NULL pNewerNode means there was an error of some kind where we will
	// need to free up the cached item.

	if (pNewerNode)
	{
		flmAssert( RC_BAD( rc));
		delete pNewerNode;
	}
	
	if( bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called to remove a node from cache.  If this is
		an uncommitted version of the node, it should remove that version
		from cache.  If the last committed version is in
		cache, it should set the high transaction ID on that version to be
		one less than the transaction ID of the update transaction.
****************************************************************************/
void F_NodeCacheMgr::removeNode(
	F_Db *				pDb,
	F_CachedNode *		pNode,
	FLMBOOL				bDecrementUseCount,
	FLMBOOL				bMutexLocked)
{
	F_Database *	pDatabase = pDb->m_pDatabase;

	flmAssert( pNode);

	// Lock the mutex

	if( !bMutexLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	}

	// Decrement the node use count if told to by the caller.

	if (bDecrementUseCount)
	{
		pNode->decrNodeUseCount();
	}

	// Unset the new and dirty flags

	pNode->unsetNodeDirtyAndNew( pDb, TRUE);

	// Determine if pNode is the last committed version
	// or a node that was added by this same transaction.
	// If it is the last committed version, set its high transaction ID.
	// Otherwise, remove the node from cache.

	if (pNode->m_ui64LowTransId < pDb->m_ui64CurrTransID)
	{
		
		// The high transaction ID on pNode better be -1 - most current version.

		flmAssert( pNode->m_ui64HighTransId == FLM_MAX_UINT64);

		pNode->setTransID( (pDb->m_ui64CurrTransID - 1));
		flmAssert( pNode->m_ui64HighTransId >= pNode->m_ui64LowTransId);
		pNode->setUncommitted();
		pNode->setLatestVer();
		pNode->unlinkFromDatabase();
		pNode->linkToDatabaseAtHead( pDatabase);
	}
	else
	{
		pNode->freeCache( pNode->nodeInUse());
	}

	if( !bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
}

/****************************************************************************
Desc:	This routine is called to remove a node from cache.
****************************************************************************/
void F_NodeCacheMgr::removeNode(
	F_Db *				pDb,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId)
{
	F_CachedNode *		pNode;

	// Lock the mutex

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);

	// Find the node in cache

	findNode( pDb, uiCollection, ui64NodeId, 
		pDb->m_ui64CurrTransID, TRUE, NULL, &pNode, NULL, NULL);

	if( pNode)
	{
		removeNode( pDb, pNode, FALSE, TRUE);
	}

	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an F_Database object is going to be removed
		from the shared memory area.  At that point, we also need to get rid
		of all nodes that have been cached for that F_Database.
****************************************************************************/
void F_Database::freeNodeCache( void)
{
	FLMUINT	uiNumFreed = 0;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	while (m_pFirstNode)
	{
		m_pFirstNode->freeCache( m_pFirstNode->nodeInUse());

		// Release the CPU every 100 nodes freed.

		if (++uiNumFreed == 100)
		{
			f_yieldCPU();
			uiNumFreed = 0;
		}
	}
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction aborts.  At that
		point, we need to get rid of any uncommitted versions of nodes in
		the node cache.
****************************************************************************/
void F_Database::freeModifiedNodes(
	F_Db *		pDb,
	FLMUINT64	ui64OlderTransId)
{
	F_CachedNode *		pNode;
	F_CachedNode *		pOlderVersion;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	pNode = m_pFirstNode;
	while (pNode)
	{
		if (pNode->nodeUncommitted())
		{
			if (pNode->nodeIsLatestVer())
			{
				pNode->setTransID( FLM_MAX_UINT64);
				pNode->unsetUncommitted();
				pNode->unsetLatestVer();
				pNode->unlinkFromDatabase();
				pNode->linkToDatabaseAtEnd( this);
			}
			else
			{
				// Save the older version - we may be changing its
				// high transaction ID back to FLM_MAX_UINT64

				pOlderVersion = pNode->m_pOlderVersion;

				// Clear the dirty and new flags
				
				pNode->unsetNodeDirtyAndNew( pDb, TRUE);

				// Free the uncommitted version.

				pNode->freeCache(
						(FLMBOOL)((pNode->nodeInUse() ||
									  pNode->readingInNode())
									 ? TRUE
									 : FALSE));

				// If the older version has a high transaction ID that
				// is exactly one less than our current transaction,
				// it is the most current version.  Hence, we need to
				// change its high transaction ID back to FLM_MAX_UINT64.

				if (pOlderVersion &&
					 pOlderVersion->m_ui64HighTransId == ui64OlderTransId)
				{
					pOlderVersion->setTransID( FLM_MAX_UINT64);
				}
			}
			pNode = m_pFirstNode;
		}
		else
		{
			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction commits.  At that
		point, we need to unset the "uncommitted" flag on any nodes
		currently in node cache for the F_Database object.
****************************************************************************/
void F_Database::commitNodeCache( void)
{
	F_CachedNode *	pNode;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	pNode = m_pFirstNode;
	while (pNode)
	{
		if (pNode->nodeUncommitted())
		{
			pNode->unsetUncommitted();
			pNode->unsetLatestVer();
			pNode = pNode->m_pNextInDatabase;
		}
		else
		{

			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when a collection in the database is deleted.
		All nodes in node cache that are in that collection must be
		removed from cache.
****************************************************************************/
void F_Db::removeCollectionNodes(
	FLMUINT		uiCollection,
	FLMUINT64	ui64TransId)
{
	F_CachedNode *	pNode;
	F_CachedNode *	pNextNode;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	pNode = m_pDatabase->m_pFirstNode;

	// Stay in the loop until we have freed all nodes in the
	// collection

	while (pNode)
	{

		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pNode below, in which case we would
		// have lost the previous entry.

		pNextNode = pNode->m_pNextInDatabase;

		// Only look at nodes in this collection

		if (pNode->m_nodeInfo.uiCollection == uiCollection)
		{
			flmAssert( pNode->m_pDatabase == m_pDatabase);
			
			// Only look at the most current versions.

			if (pNode->m_ui64HighTransId == FLM_MAX_UINT64)
			{

				// Better not be a newer version.

				flmAssert( pNode->m_pNewerVersion == NULL);

				if (pNode->m_ui64LowTransId < ui64TransId)
				{

					// This version was not added or modified by this
					// transaction so it's high transaction ID should simply
					// be set to one less than the current transaction ID.

					pNode->setTransID( ui64TransId - 1);
					flmAssert( pNode->m_ui64HighTransId >= pNode->m_ui64LowTransId);
					pNode->setUncommitted();
					pNode->setLatestVer();
					pNode->unlinkFromDatabase();
					pNode->linkToDatabaseAtHead( m_pDatabase);
				}
				else
				{

					// The node was added or modified in this
					// transaction. Simply remove it from cache.

					pNode->freeCache( pNode->nodeInUse());
				}
			}
			else
			{

				// If not most current version, the node's high transaction
				// ID better already be less than transaction ID.

				flmAssert( pNode->m_ui64HighTransId < ui64TransId);
			}
		}
		pNode = pNextNode;

	}
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_CachedNode::findChildElm(
	FLMUINT		uiChildElmNameId,
	FLMUINT *	puiInsertPos)
{
	FLMBOOL		bFound = FALSE;
	FLMUINT		uiLoop;
	NODE_ITEM *	pChildElmNode;
	FLMUINT		uiTblSize;
	FLMUINT		uiLow;
	FLMUINT		uiMid;
	FLMUINT		uiHigh;
	FLMUINT		uiTblNameId;

	// If the child element count is <= 4, do a sequential search through
	// the array.  Otherwise, do a binary search.
	
	if ((uiTblSize = m_nodeInfo.uiChildElmCount) <= 4)
	{
		for (uiLoop = 0, pChildElmNode = m_pNodeList;
			  uiLoop < m_nodeInfo.uiChildElmCount && 
			  		pChildElmNode->uiNameId < uiChildElmNameId;
			  uiLoop++, pChildElmNode++)
		{
			;
		}
		if (uiLoop < m_nodeInfo.uiChildElmCount)
		{
			*puiInsertPos = uiLoop;
			if (pChildElmNode->uiNameId == uiChildElmNameId)
			{
				bFound = TRUE;
			}
		}
		else
		{
			*puiInsertPos = uiLoop;
		}
	}
	else
	{
		uiHigh = --uiTblSize;
		uiLow = 0;
		for (;;)
		{
			uiMid = (uiLow + uiHigh) / 2;
			uiTblNameId = m_pNodeList [uiMid].uiNameId;
			if (uiTblNameId == uiChildElmNameId)
			{
				// Found Match
	
				*puiInsertPos = uiMid;
				bFound = TRUE;
				goto Exit;
			}
	
			// Check if we are done
	
			if (uiLow >= uiHigh)
			{
				// Done, item not found
	
				*puiInsertPos = (uiChildElmNameId < uiTblNameId)
										 ? uiMid
										 : uiMid + 1;
				goto Exit;
			}
	
			if (uiChildElmNameId < uiTblNameId)
			{
				if (uiMid == 0)
				{
					*puiInsertPos = 0;
					goto Exit;
				}
				uiHigh = uiMid - 1;
			}
			else
			{
				if (uiMid == uiTblSize)
				{
					*puiInsertPos = uiMid + 1;
					goto Exit;
				}
				uiLow = uiMid + 1;
			}
		}
	}
	
Exit:

	return( bFound);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void F_CachedNode::checkReadFromDisk(
	F_Db *	pDb)
{
	FLMUINT64	ui64LowTransId;
	FLMBOOL		bMostCurrent;
	RCODE			rc;
	
	// Need to unlock the node cache mutex before doing the read.

	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);	
	rc = gv_XFlmSysData.pNodeCacheMgr->readNodeFromDisk( pDb,
					m_nodeInfo.uiCollection, m_nodeInfo.ui64NodeId,
					this, &ui64LowTransId, &bMostCurrent);
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);	
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
void F_CachedNode::setNodeDirty(
	F_Db *		pDb,
	FLMBOOL		bNew)
{
	if (!nodeIsDirty())
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		
		// Should already be the uncommitted version.
		
		flmAssert( nodeUncommitted());
		
		// Should NOT be the latest version - those cannot
		// be set to dirty. - latest ver flag is only set
		// for nodes that should be returned to being the
		// latest version of the node if the transaction
		// aborts.
		
		flmAssert( !nodeIsLatestVer());
		
		// Unlink from its database, set the dirty flag,
		// and relink at the head.
		
		unlinkFromDatabase();
		
		if (bNew)
		{
			m_uiFlags |= (FDOM_DIRTY | FDOM_NEW);
		}
		else
		{
			m_uiFlags |= FDOM_DIRTY;
		}
		
		linkToDatabaseAtHead( pDb->m_pDatabase);
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		
		pDb->m_uiDirtyNodeCount++;
	}
	else if (bNew)
	{
		m_uiFlags |= FDOM_NEW;
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
void F_CachedNode::unsetNodeDirtyAndNew(
	F_Db *			pDb,
	FLMBOOL			bMutexAlreadyLocked)
{
	// When outputting a binary or text stream, it is possible that the
	// dirty flag was unset when the last buffer was output
	
	if (nodeIsDirty())
	{
		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		}
			
		// Unlink from its database, unset the dirty flag,
		// and relink at the head.
		
		unlinkFromDatabase();
		if( m_uiFlags & FDOM_DIRTY)
		{
			flmAssert( pDb->m_uiDirtyNodeCount);
			pDb->m_uiDirtyNodeCount--;
		}

		m_uiFlags &= ~(FDOM_DIRTY | FDOM_NEW);
		linkToDatabaseAtHead( pDb->m_pDatabase);
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		}
	}
}
	
/*****************************************************************************
Desc:	Calculate the SEN length of a number and add it to the node info item.
******************************************************************************/
FINLINE void flmAddInfoSenLen(
	XFLM_NODE_INFO_ITEM *	pInfoItem,
	FLMUINT64					ui64Num,
	FLMUINT *					puiTotalOverhead)
{
	FLMUINT	uiSenLen = f_getSENByteCount( ui64Num);

	pInfoItem->ui64Bytes += (FLMUINT64)uiSenLen;
	pInfoItem->ui64Count++;
	(*puiTotalOverhead) += uiSenLen;
}

/*****************************************************************************
Desc:	Node header format
******************************************************************************/
// Header size							1 byte
//
// Node and data type 				1 byte (bits = HDDDNNNN)
//		H = Have data (1 bit)
//		D = Data type (3 bits)
//		N = Node type (4 bits)
//
// Storage Flags						1-5 bytes (typically 1 byte)
//		NSF_HAVE_BASE_ID_BIT
//		NSF_HAVE_META_VALUE_BIT
//		NSF_HAVE_SIBLINGS_BIT
//		NSF_HAVE_CHILDREN_BIT
//		NSF_HAVE_ATTR_LIST_BIT
//		NSF_HAVE_CELM_LIST_BIT
//		NSF_HAVE_DATA_LEN_BIT
//
//		NSF_EXT_HAVE_DCHILD_COUNT_BIT
//		NSF_EXT_READ_ONLY_BIT
//		NSF_EXT_CANNOT_DELETE_BIT
//		NSF_EXT_PREFIX_BIT
//		NSF_EXT_ENCRYPTED_BIT
//		NSF_EXT_ANNOTATION_BIT
//		NSF_EXT_QUARANTINED_BIT
//
// Document ID							0-9 byte SEN
// Base ID								1-9 byte SEN - if NSF_HAVE_BASE_ID_BIT is set
// Parent ID							0-9 byte SEN (offset from base)
// Name ID								0-5 byte SEN
// Prefix ID							0-5 byte SEN
// Meta value							0-9 byte SEN - if NSF_HAVE_META_VALUE_BIT is set
// Prev+Next Siblings				0-9 + 0-9 byte SEN (offset from base) - if NSF_HAVE_SIBLINGS_BIT is set
// First+Last Child ID				0-9 + 0-9 byte SEN (offset from base) - if NSF_HAVE_CHILDREN_BIT is set
// Data node child count			0-5 byte SEN - If NSF_HAVE_CHILDREN_BIT and NSF_EXT_HAVE_DCHILD_COUNT_BIT is set
// Child Element Count				0-5 byte SEN - if NSF_HAVE_CELM_LIST_BIT is set
// Encryption ID						0-5 byte SEN - if NSF_EXT_ENCRYPTED_BIT set
// Annotation node					0-9 byte SEN (offset from base) - if NSF_EXT_ANNOTATION_BIT is set
// Data length							0-5 byte SEN - if NSF_HAVE_DATA_LEN_BIT is set
//
// After the "core" header, one or more of the following may be present:
//
// If HAVE_CELM_LIST_BIT is set and we actually have a non-zero child element count:
// child element list			[0-5 SEN for element ID + 0-9 SEN for node id]...
//
// IV									0 if no encryption, 8 or 16 bytes if encrypting
//	Value								0+ bytes

/*****************************************************************************
Desc:		Creates a variable-sized header for the node and copies it into
			the supplied buffer.  The buffer must be sized to allow at least
			MAX_DOM_HEADER_SIZE bytes.
******************************************************************************/
RCODE F_CachedNode::headerToBuf(
	FLMBOOL				bFixedSizeHeader,
	FLMBYTE *			pucBuf,
	FLMUINT *			puiHeaderStorageSize,
	XFLM_NODE_INFO *	pNodeInfo,
	F_Db *				pDb)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucStart = pucBuf;
	FLMUINT			uiLoop;
	FLMUINT			uiFlagsLen;
	FLMUINT			uiStorageFlags = 0;
	FLMUINT			uiDataChildCount = getDataChildCount();
	FLMUINT			uiEncDefId = 0;
	FLMUINT			uiDataLength = getDataLength();
	FLMUINT			uiNameId = getNameId();
	FLMUINT			uiPrefixId = getPrefixId();
	FLMUINT			uiFlags = m_uiFlags;
	eDomNodeType	eNodeType = getNodeType();
	FLMUINT64		ui64NodeId = m_nodeInfo.ui64NodeId;
	FLMUINT64		ui64BaseId = ui64NodeId;
	FLMUINT64		ui64DocId = getDocumentId();
	FLMUINT64		ui64ParentId = getParentId();
	FLMUINT64		ui64FirstChildId = getFirstChildId();
	FLMUINT64		ui64LastChildId = getLastChildId();
	FLMUINT64		ui64PrevSibId = getPrevSibId();
	FLMUINT64		ui64NextSibId = getNextSibId();
	FLMUINT64		ui64AnnotationId = getAnnotationId();
	FLMUINT64		ui64MetaValue = getMetaValue();
	FLMUINT64		ui64Tmp;
	FLMBYTE			ucTmpSEN[ FLM_MAX_SEN_LEN];
	FLMBYTE *		pucTmpSEN;
	FLMUINT			uiTotalOverhead = 0;

	if( !ui64NodeId)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Storage flags used by fixed and variable size headers

	if( uiFlags & FDOM_READ_ONLY)
	{
		uiStorageFlags |= NSF_EXT_READ_ONLY_BIT;
	}
	
	if( uiFlags & FDOM_CANNOT_DELETE)
	{
		uiStorageFlags |= NSF_EXT_CANNOT_DELETE_BIT;
	}
	
	if( uiFlags & FDOM_QUARANTINED)
	{
		uiStorageFlags |= NSF_EXT_QUARANTINED_BIT;
	}

	if( uiFlags & FDOM_HAVE_CELM_LIST)
	{
		uiStorageFlags |= NSF_HAVE_CELM_LIST_BIT;
	}
	
	if( uiFlags & FDOM_NAMESPACE_DECL)
	{
		uiStorageFlags |= NSF_EXT_NAMESPACE_DECL_BIT;
	}

	if( m_uiAttrCount)
	{
		flmAssert( eNodeType == ELEMENT_NODE);
		uiStorageFlags |= NSF_HAVE_ATTR_LIST_BIT;
	}
	
	if( uiDataLength)
	{
		if( (uiEncDefId = getEncDefId()) != 0)
		{
			uiStorageFlags |= NSF_EXT_ENCRYPTED_BIT;
		}
	}

	// Output the header

	if( bFixedSizeHeader)
	{
		if (pucBuf)
		{
			flmAssert( !pNodeInfo && !pDb && puiHeaderStorageSize);

			// Set the header size bytes

			*pucBuf++ = XFLM_FIXED_SIZE_HEADER_TOKEN;
			
			// Encode the node type
			
			*pucBuf++ = ((((FLMBYTE)m_nodeInfo.uiDataType) & 0x07) << 4) |
							(((FLMBYTE)eNodeType) & 0x0F);
							
			// Document ID

			U642FBA( ui64DocId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// Parent ID

			U642FBA( ui64ParentId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// Name ID

			UD2FBA( (FLMUINT32)uiNameId, pucBuf);
			pucBuf += sizeof( FLMUINT32);
			
			// Prefix ID

			UD2FBA( (FLMUINT32)uiPrefixId, pucBuf);
			pucBuf += sizeof( FLMUINT32);

			// Metavalue

			U642FBA( ui64MetaValue, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// Previous and next siblings

			U642FBA( ui64PrevSibId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			U642FBA( ui64NextSibId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// First and last children

			U642FBA( ui64FirstChildId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			U642FBA( ui64LastChildId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// Data child count

			UD2FBA( (FLMUINT32)uiDataChildCount, pucBuf);
			pucBuf += sizeof( FLMUINT32);

			// Child element count

			UD2FBA( (FLMUINT32)m_nodeInfo.uiChildElmCount, pucBuf);
			pucBuf += sizeof( FLMUINT32);
			
			// Data length 

			UD2FBA( (FLMUINT32)uiDataLength, pucBuf);
			pucBuf += sizeof( FLMUINT32);

			// Encryption definition ID
				
			UD2FBA( (FLMUINT32)uiEncDefId, pucBuf);
			pucBuf += sizeof( FLMUINT32);

			// Annotation ID
			
			U642FBA( ui64AnnotationId, pucBuf);
			pucBuf += sizeof( FLMUINT64);

			// Storage flags

			UD2FBA( (FLMUINT32)uiStorageFlags, pucBuf);
			pucBuf += sizeof( FLMUINT32);

			flmAssert( (FLMUINT)(pucBuf - pucStart) == FIXED_DOM_HEADER_SIZE);
			*puiHeaderStorageSize = FIXED_DOM_HEADER_SIZE;
		}
		else
		{
			flmAssert( pNodeInfo && pDb && !puiHeaderStorageSize);

			pNodeInfo->headerSize.ui64Bytes++;
			pNodeInfo->headerSize.ui64Count++;

			pNodeInfo->nodeAndDataType.ui64Bytes++;
			pNodeInfo->nodeAndDataType.ui64Count++;

			pNodeInfo->documentId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->documentId.ui64Count++;

			pNodeInfo->parentId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->parentId.ui64Count++;

			pNodeInfo->nameId.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->nameId.ui64Count++;

			pNodeInfo->prefixId.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->prefixId.ui64Count++;

			pNodeInfo->metaValue.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->metaValue.ui64Count++;

			pNodeInfo->prevSibId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->prevSibId.ui64Count++;

			pNodeInfo->nextSibId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->nextSibId.ui64Count++;

			pNodeInfo->firstChildId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->firstChildId.ui64Count++;

			pNodeInfo->lastChildId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->lastChildId.ui64Count++;

			pNodeInfo->dataChildCount.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->dataChildCount.ui64Count++;

			pNodeInfo->childElmCount.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->childElmCount.ui64Count++;

			pNodeInfo->unencDataLen.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->unencDataLen.ui64Count++;

			pNodeInfo->encDefId.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->encDefId.ui64Count++;

			pNodeInfo->annotationId.ui64Bytes += sizeof( FLMUINT64);
			pNodeInfo->annotationId.ui64Count++;

			pNodeInfo->flags.ui64Bytes += sizeof( FLMUINT32);
			pNodeInfo->flags.ui64Count++;

			uiTotalOverhead += FIXED_DOM_HEADER_SIZE;
		}
	}
	else
	{

		// Determine the base ID
		
		if( ui64DocId < ui64BaseId)
		{
			flmAssert( ui64DocId);
			ui64BaseId = ui64DocId;
		}
		
		if( ui64ParentId && ui64ParentId < ui64BaseId)
		{
			ui64BaseId = m_nodeInfo.ui64ParentId;
		}

		if( ui64PrevSibId && ui64PrevSibId < ui64BaseId)
		{
			ui64BaseId = ui64PrevSibId;
		}

		if( ui64NextSibId && ui64NextSibId < ui64BaseId)
		{
			ui64BaseId = ui64NextSibId;
		}

		if( ui64FirstChildId && ui64FirstChildId < ui64BaseId)
		{
			flmAssert( ui64LastChildId);
			ui64BaseId = ui64FirstChildId;
		}

		if( ui64LastChildId && ui64LastChildId < ui64BaseId)
		{
			flmAssert( ui64FirstChildId);
			ui64BaseId = ui64LastChildId;
		}
		
		if( ui64AnnotationId && ui64AnnotationId < ui64BaseId)
		{
			ui64BaseId = ui64AnnotationId;
		}
			
		if (pucBuf)
		{
			flmAssert( !pNodeInfo && !pDb && puiHeaderStorageSize);

			// Reserve a byte for the header length

			pucBuf++;
			
			// Encode the node type
			
			*pucBuf++ = ((((FLMBYTE)m_nodeInfo.uiDataType) & 0x07) << 4) |
							(((FLMBYTE)eNodeType) & 0x0F) |
							(uiDataLength ? 0x80 : 0);
							
			// Document ID

			f_encodeSEN( ui64DocId, &pucBuf);
			
			// Encode the base ID if it isn't equal to the document ID

			if( ui64BaseId != ui64DocId)
			{
				uiStorageFlags |= NSF_HAVE_BASE_ID_BIT;
				f_encodeSEN( ui64BaseId, &pucBuf);
			}
			
			// Parent ID

			if( (ui64Tmp = ui64ParentId) == 0)
			{
				ui64Tmp = ui64NodeId;
			}

			f_encodeSEN( ui64Tmp - ui64BaseId, &pucBuf);
			
			// Name ID

			f_encodeSEN( uiNameId, &pucBuf);

			// Prefix ID

			if( uiPrefixId)
			{
				uiStorageFlags |= NSF_EXT_HAVE_PREFIX_BIT;
				f_encodeSEN( uiPrefixId, &pucBuf);
			}
			
			// Metavalue

			if( ui64MetaValue)
			{
				uiStorageFlags |= NSF_HAVE_META_VALUE_BIT;
				f_encodeSEN( ui64MetaValue, &pucBuf);
			}
			
			// Previous and next siblings

			if( ui64PrevSibId || ui64NextSibId)
			{
				if( (ui64Tmp = ui64PrevSibId) == 0)
				{
					ui64Tmp = ui64NodeId;
				}

				f_encodeSEN( ui64Tmp - ui64BaseId, &pucBuf);

				if( (ui64Tmp = ui64NextSibId) == 0)
				{
					ui64Tmp = ui64NodeId;
				}

				f_encodeSEN( ui64Tmp - ui64BaseId, &pucBuf);
				uiStorageFlags |= NSF_HAVE_SIBLINGS_BIT;
			}
			
			// First, last, and data children

			if( ui64FirstChildId)
			{
				flmAssert( ui64LastChildId);
				
				f_encodeSEN( ui64FirstChildId - ui64BaseId, &pucBuf);
				f_encodeSEN( ui64LastChildId - ui64BaseId, &pucBuf);
				uiStorageFlags |= NSF_HAVE_CHILDREN_BIT;
				
				if( uiDataChildCount)
				{
					f_encodeSEN( uiDataChildCount, &pucBuf);
					uiStorageFlags |= NSF_EXT_HAVE_DCHILD_COUNT_BIT;
				}
			}
			
			// Child element count

			if( uiFlags & FDOM_HAVE_CELM_LIST)
			{
				// NOTE: It is legal for m_nodeInfo.uiChildElmCount to be zero.
				// The FDOM_EXT_CHILD_ELM_LIST bit is also used to enforce
				// the fact that all of the child elements must have unique
				// name IDs.
				
				f_encodeSEN( m_nodeInfo.uiChildElmCount, &pucBuf);
			}
			
			// Encryption ID

			if( uiEncDefId)
			{
				uiStorageFlags |= NSF_EXT_ENCRYPTED_BIT;
				f_encodeSEN( uiEncDefId, &pucBuf);
			}
			
			// Annotation ID
			
			if( ui64AnnotationId)
			{
				uiStorageFlags |= NSF_EXT_ANNOTATION_BIT;
				f_encodeSEN( ui64AnnotationId - ui64BaseId, &pucBuf);
			}
			
			// Output the data length if needed
			
			if( uiDataLength && 
				(uiEncDefId || (uiFlags & FDOM_HAVE_CELM_LIST) || m_uiAttrCount))
			{
				uiStorageFlags |= NSF_HAVE_DATA_LEN_BIT;
				f_encodeSEN( uiDataLength, &pucBuf);
			}
			
			// Output the storage flags (inverted SEN)

			uiFlagsLen = f_getSENByteCount( uiStorageFlags);
			if( uiFlagsLen > 1)
			{
				pucTmpSEN = ucTmpSEN;
				f_encodeSEN( uiStorageFlags, &pucTmpSEN);
				
				for( uiLoop = uiFlagsLen; uiLoop > 0; uiLoop--)
				{
					*pucBuf++ = ucTmpSEN[ uiLoop - 1];
				}
			}
			else
			{
				*pucBuf++ = (FLMBYTE)uiStorageFlags;
			}

			flmAssert( (FLMUINT)(pucBuf - pucStart) <= MAX_DOM_HEADER_SIZE);
			
			// Set the header size
			
			*puiHeaderStorageSize = (FLMUINT)(pucBuf - pucStart);
			*pucStart = (FLMBYTE)(*puiHeaderStorageSize);
		}
		else
		{
			flmAssert( pNodeInfo && pDb && !puiHeaderStorageSize);

			pNodeInfo->headerSize.ui64Bytes++;
			pNodeInfo->headerSize.ui64Count++;

			pNodeInfo->nodeAndDataType.ui64Bytes++;
			pNodeInfo->nodeAndDataType.ui64Count++;

			pNodeInfo->documentId.ui64Bytes += f_getSENByteCount( ui64DocId);
			pNodeInfo->documentId.ui64Count++;

			uiTotalOverhead = 3;

			// Document ID

			flmAddInfoSenLen( &pNodeInfo->documentId, ui64DocId, &uiTotalOverhead);
			
			// Encode the base ID if it isn't equal to the document ID

			if (ui64BaseId != ui64DocId)
			{
				uiStorageFlags |= NSF_HAVE_BASE_ID_BIT;
				flmAddInfoSenLen( &pNodeInfo->baseId, ui64BaseId, &uiTotalOverhead);
			}
			
			// Parent ID

			if ((ui64Tmp = ui64ParentId) == 0)
			{
				ui64Tmp = ui64NodeId;
			}
			flmAddInfoSenLen( &pNodeInfo->parentId, ui64Tmp - ui64BaseId,
				&uiTotalOverhead);

			// Name ID

			flmAddInfoSenLen( &pNodeInfo->nameId, uiNameId, &uiTotalOverhead);

			// Prefix ID

			if (uiPrefixId)
			{
				uiStorageFlags |= NSF_EXT_HAVE_PREFIX_BIT;
				flmAddInfoSenLen( &pNodeInfo->prefixId, uiPrefixId, &uiTotalOverhead);
			}

			// Meta Value

			if (ui64MetaValue)
			{
				uiStorageFlags |= NSF_HAVE_META_VALUE_BIT;
				flmAddInfoSenLen( &pNodeInfo->metaValue, ui64MetaValue, &uiTotalOverhead);
			}

			// First/Last sibling

			if (ui64PrevSibId || ui64NextSibId)
			{
				if ((ui64Tmp = ui64PrevSibId) == 0)
				{
					ui64Tmp = ui64NodeId;
				}
				flmAddInfoSenLen( &pNodeInfo->prevSibId, ui64Tmp, &uiTotalOverhead);

				if ((ui64Tmp = ui64NextSibId) == 0)
				{
					ui64Tmp = ui64NodeId;
				}
				flmAddInfoSenLen( &pNodeInfo->nextSibId, ui64Tmp, &uiTotalOverhead);
				uiStorageFlags |= NSF_HAVE_SIBLINGS_BIT;
			}
		
			// First, last, and data children

			if (ui64FirstChildId)
			{
				flmAddInfoSenLen( &pNodeInfo->firstChildId,
					ui64FirstChildId - ui64BaseId, &uiTotalOverhead);
				flmAddInfoSenLen( &pNodeInfo->lastChildId,
					ui64LastChildId - ui64BaseId, &uiTotalOverhead);
				uiStorageFlags |= NSF_HAVE_CHILDREN_BIT;
				if (uiDataChildCount)
				{
					flmAddInfoSenLen( &pNodeInfo->dataChildCount,
						uiDataChildCount, &uiTotalOverhead);
					uiStorageFlags |= NSF_EXT_HAVE_DCHILD_COUNT_BIT;
				}
			}
		
			// Child element count - may be zero, so we should test the flag.

			if (uiFlags & FDOM_HAVE_CELM_LIST)
			{
				flmAddInfoSenLen( &pNodeInfo->childElmCount,
					m_nodeInfo.uiChildElmCount, &uiTotalOverhead);
			}
			
			// Encryption ID

			if (uiEncDefId)
			{
				uiStorageFlags |= NSF_EXT_ENCRYPTED_BIT;
				flmAddInfoSenLen( &pNodeInfo->encDefId, uiEncDefId, &uiTotalOverhead);
			}
			
			// Annotation ID
			
			if( ui64AnnotationId)
			{
				uiStorageFlags |= NSF_EXT_ANNOTATION_BIT;
				flmAddInfoSenLen( &pNodeInfo->annotationId,
					ui64AnnotationId - ui64BaseId, &uiTotalOverhead);
			}
		
			// Data length if needed
			
			if( uiDataLength && 
				(uiEncDefId || (uiFlags & FDOM_HAVE_CELM_LIST) || m_uiAttrCount))
			{
				uiStorageFlags |= NSF_HAVE_DATA_LEN_BIT;
				flmAddInfoSenLen( &pNodeInfo->unencDataLen,
					uiDataLength, &uiTotalOverhead);
			}
		
			flmAddInfoSenLen( &pNodeInfo->flags, uiStorageFlags, &uiTotalOverhead);
		}
	}

	// Account for other overhead.

	if (pNodeInfo)
	{
		if (eNodeType == ELEMENT_NODE)
		{
			NODE_ITEM *		pNodeItem;
			FLMUINT			uiNodeCount = getChildElmCount();
			FLMUINT			uiPrevNameId = 0;
			FLMUINT64		ui64ElmNodeId = getNodeId();

			// Go through the child element list and calculate the length needed
			// to store the name id and node id for each one.
			
			pNodeItem = m_pNodeList;
			for( uiLoop = 0; uiLoop < uiNodeCount; pNodeItem++, uiLoop++)
			{
				flmAssert( pNodeItem->uiNameId > uiPrevNameId);
				flmAssert( pNodeItem->ui64NodeId > ui64ElmNodeId);

				flmAddInfoSenLen( &pNodeInfo->childElmNameId,
					pNodeItem->uiNameId - uiPrevNameId, &uiTotalOverhead);
				flmAddInfoSenLen( &pNodeInfo->childElmNodeId,
					pNodeItem->ui64NodeId - ui64ElmNodeId, &uiTotalOverhead);
				uiPrevNameId = pNodeItem->uiNameId;
			}
			
			// Determine space taken by attributes.
			
			if (m_uiAttrCount)
			{
				if( RC_BAD( rc = exportAttributeList( pDb, NULL, pNodeInfo)))
				{
					goto Exit;
				}
			}
		}

		// Determine space needed to store encryption IV and any
		// encryption padding.

		if (uiEncDefId)
		{
			F_ENCDEF *	pEncDef;
			FLMUINT		uiTmp;
			
			if (RC_BAD( rc = pDb->m_pDict->getEncDef( uiEncDefId, &pEncDef)))
			{
				goto Exit;
			}
			
			uiTmp = pEncDef->pCcs->getIVLen();
			flmAssert( uiTmp == 8 || uiTmp == 16);
			pNodeInfo->encIV.ui64Bytes += (FLMUINT64)uiTmp;
			pNodeInfo->encIV.ui64Count++;
			uiTotalOverhead += uiTmp;
			
			uiTmp = getEncLen( uiDataLength) - uiDataLength;
			if (uiTmp)
			{
				pNodeInfo->encPadding.ui64Bytes += (FLMUINT64)uiTmp;
				pNodeInfo->encPadding.ui64Count++;
				uiTotalOverhead += uiTmp;
			}
		}
												
		pNodeInfo->totalOverhead.ui64Bytes += (FLMUINT64)uiTotalOverhead;
		pNodeInfo->totalOverhead.ui64Count++;
		switch (eNodeType)
		{
			case ELEMENT_NODE:
				pNodeInfo->elementNode.ui64Bytes +=
					(FLMUINT64)uiDataLength + (FLMUINT64)uiTotalOverhead;
				pNodeInfo->elementNode.ui64Count++;
				break;
			case DATA_NODE:
				pNodeInfo->dataNode.ui64Bytes +=
					(FLMUINT64)uiDataLength + (FLMUINT64)uiTotalOverhead;
				pNodeInfo->dataNode.ui64Count++;
				break;
			case COMMENT_NODE:
				pNodeInfo->commentNode.ui64Bytes +=
					(FLMUINT64)uiDataLength + (FLMUINT64)uiTotalOverhead;
				pNodeInfo->commentNode.ui64Count++;
				break;
			default:
				pNodeInfo->otherNode.ui64Bytes +=
					(FLMUINT64)uiDataLength + (FLMUINT64)uiTotalOverhead;
				pNodeInfo->otherNode.ui64Count++;
				break;
		}

		switch (m_nodeInfo.uiDataType)
		{
			case XFLM_NODATA_TYPE:
				pNodeInfo->dataNodata.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataNodata.ui64Count++;
				break;
			case XFLM_TEXT_TYPE:
				pNodeInfo->dataString.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataString.ui64Count++;
				break;
			case XFLM_NUMBER_TYPE:
				pNodeInfo->dataNumeric.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataNumeric.ui64Count++;
				break;
			case XFLM_BINARY_TYPE:
				pNodeInfo->dataBinary.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataBinary.ui64Count++;
				break;
			default:
				flmAssert( 0);
				break;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmReadNodeInfo(
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	IF_IStream *		pIStream,
	FLMUINT				uiOverallLength,
	FLMBOOL				bAssertOnCorruption,
	F_NODE_INFO *		pNodeInfo,
	FLMUINT *			puiStorageFlags,
	FLMBOOL *			pbFixedSizeHeader)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLoop;
	FLMUINT				uiStorageFlags = 0;
	FLMUINT				uiStorageFlagsLen;
	FLMUINT				uiHeaderStorageSize;
	FLMBYTE				ucHeader[ MAX_DOM_HEADER_SIZE];
	const FLMBYTE *	pucHeader;
	const FLMBYTE *	pucHeaderEnd;
	FLMUINT64			ui64BaseId = 0;
	FLMBYTE				ucTmpSEN[ FLM_MAX_SEN_LEN];
	FLMBYTE *			pucTmpSEN;
	FLMBOOL				bEOFValid = TRUE;
	FLMBOOL				bHaveData;
	
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( bAssertOnCorruption);
#endif
	
	// Set the node ID and collection
	
	pNodeInfo->uiCollection = uiCollection;
	pNodeInfo->ui64NodeId = ui64NodeId;
	
	// Read the expected length of the header

	if( RC_BAD( rc = pIStream->read( &ucHeader[ 0], 1)))
	{
		goto Exit;
	}

	bEOFValid = FALSE;

	// A length value of XFLM_FIXED_SIZE_HEADER_TOKEN indicates that we have a 
	// non-compressed header on the node.  This type of header is used when a
	// large text or binary value is streamed into the database.

	if( ucHeader[ 0] == XFLM_FIXED_SIZE_HEADER_TOKEN)
	{
		// Read the rest of the header

		uiHeaderStorageSize = FIXED_DOM_HEADER_SIZE;
		if( RC_BAD( rc = pIStream->read( &ucHeader[ 1], 
			uiHeaderStorageSize - 1, NULL)))
		{
			goto Exit;
		}

		pucHeader = ucHeader;
		pucHeaderEnd = pucHeader + uiHeaderStorageSize;
		
		// Skip past the header size byte

		pucHeader++;

		// Get the node type, data type, and data flag 

		pNodeInfo->eNodeType = (eDomNodeType)((*pucHeader) & 0x0F);
		pNodeInfo->uiDataType = ((*pucHeader) >> 4) & 0x07;
		pucHeader++;

		// Document ID

		pNodeInfo->ui64DocumentId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// Parent ID

		pNodeInfo->ui64ParentId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// Name ID
		
		pNodeInfo->uiNameId = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		// Prefix ID

		pNodeInfo->uiPrefixId = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		// Metavalue

		pNodeInfo->ui64MetaValue = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// Previous and next siblings

		pNodeInfo->ui64PrevSibId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		pNodeInfo->ui64NextSibId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// First and last children

		pNodeInfo->ui64FirstChildId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		pNodeInfo->ui64LastChildId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// Data child count

		pNodeInfo->uiDataChildCount = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		// Child element count

		pNodeInfo->uiChildElmCount = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		if( pNodeInfo->uiChildElmCount &&
			 pNodeInfo->eNodeType != ELEMENT_NODE)
		{
#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}

		// Data length

		pNodeInfo->uiDataLength = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		// Encryption Id

		pNodeInfo->uiEncDefId = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		if( pNodeInfo->uiEncDefId && !pNodeInfo->uiDataLength)
		{
#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}
		
		// Annotation ID

		pNodeInfo->ui64AnnotationId = FB2U64( pucHeader);
		pucHeader += sizeof( FLMUINT64);

		// Storage flags
		
		uiStorageFlags = FB2UD( pucHeader);
		pucHeader += sizeof( FLMUINT32);

		// Set the fixed size header flag

		if( pbFixedSizeHeader)
		{
			*pbFixedSizeHeader = TRUE;
		}
	}
	else
	{
		if( (uiHeaderStorageSize = (FLMUINT)ucHeader[ 0]) > MAX_DOM_HEADER_SIZE)
		{
#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}
		
		// Read the rest of the header
		
		if( RC_BAD( rc = pIStream->read( &ucHeader[ 1], 
			uiHeaderStorageSize - 1, NULL)))
		{
			goto Exit;
		}
		
		pucHeader = ucHeader;
		pucHeaderEnd = pucHeader + uiHeaderStorageSize;
		
		// Get the storage flags
		
		uiStorageFlags = pucHeader[ uiHeaderStorageSize - 1];
		uiStorageFlagsLen = f_getSENLength( (FLMBYTE)uiStorageFlags);

		if( uiStorageFlagsLen > 1)
		{
			pucTmpSEN = ucTmpSEN;
			for( uiLoop = 1; uiLoop <= uiStorageFlagsLen; uiLoop++)
			{
				*pucTmpSEN++ = pucHeader[ uiHeaderStorageSize - uiLoop]; 
			}

			pucTmpSEN = ucTmpSEN;	
			if( RC_BAD( rc = f_decodeSEN( (const FLMBYTE **)&pucTmpSEN, 
				pucTmpSEN + uiStorageFlagsLen, &uiStorageFlags)))
			{
				goto Exit;
			}
		}

		// Skip past the header size byte

		pucHeader++;

		// Get the node type, data type, and data flag 

		pNodeInfo->eNodeType = (eDomNodeType)((*pucHeader) & 0x0F);

		if( pNodeInfo->eNodeType == INVALID_NODE ||
			 pNodeInfo->eNodeType > PROCESSING_INSTRUCTION_NODE)
		{
#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}

		pNodeInfo->uiDataType = ((*pucHeader) >> 4) & 0x07;
		bHaveData = *pucHeader & 0x80 ? TRUE : FALSE;
		pucHeader++;
		
		// Document ID

		if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd,
			&pNodeInfo->ui64DocumentId)))
		{
			goto Exit;
		}
		
		// Base ID

		if( uiStorageFlags & NSF_HAVE_BASE_ID_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, 
				pucHeaderEnd, &ui64BaseId)))
			{
				goto Exit;
			}
		}
		else
		{
			ui64BaseId = pNodeInfo->ui64DocumentId;
		}
		
		// Parent ID

		if( RC_BAD( rc = f_decodeSEN64( &pucHeader, 
			pucHeaderEnd, &pNodeInfo->ui64ParentId)))
		{
			goto Exit;
		}
		
		if( (pNodeInfo->ui64ParentId += ui64BaseId) == ui64NodeId)
		{
			pNodeInfo->ui64ParentId = 0;
		}
		
		// Name ID
		
		if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd,
			&pNodeInfo->uiNameId)))
		{
			goto Exit;
		}
		
		// Prefix ID

		if( uiStorageFlags & NSF_EXT_HAVE_PREFIX_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd,
				&pNodeInfo->uiPrefixId)))
			{
				goto Exit;
			}
		}
		else
		{
			pNodeInfo->uiPrefixId = 0;
		}
		
		// Metavalue

		if( uiStorageFlags & NSF_HAVE_META_VALUE_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd, 
				&pNodeInfo->ui64MetaValue)))
			{
				goto Exit;
			}
		}
		else
		{
			pNodeInfo->ui64MetaValue = 0;
		}
		
		// Previous and next siblings

		if( uiStorageFlags & NSF_HAVE_SIBLINGS_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd,
				&pNodeInfo->ui64PrevSibId)))
			{
				goto Exit;
			}

			if( (pNodeInfo->ui64PrevSibId += ui64BaseId) == ui64NodeId)
			{
				pNodeInfo->ui64PrevSibId = 0;
			}

			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd, 
				&pNodeInfo->ui64NextSibId)))
			{
				goto Exit;
			}

			if( (pNodeInfo->ui64NextSibId += ui64BaseId) == ui64NodeId)
			{
				pNodeInfo->ui64NextSibId = 0;
			}
		}
		else
		{
			pNodeInfo->ui64PrevSibId = 0;
			pNodeInfo->ui64NextSibId = 0;
		}
		
		// First and last children.  Also will read the data child count.

		if( uiStorageFlags & NSF_HAVE_CHILDREN_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd, 
				&pNodeInfo->ui64FirstChildId)))
			{
				goto Exit;
			}
			
			pNodeInfo->ui64FirstChildId += ui64BaseId;

			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd, 
				&pNodeInfo->ui64LastChildId)))
			{
				goto Exit;
			}
			
			pNodeInfo->ui64LastChildId += ui64BaseId;
			
			if( uiStorageFlags & NSF_EXT_HAVE_DCHILD_COUNT_BIT)
			{
				if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd, 
					&pNodeInfo->uiDataChildCount)))
				{
					goto Exit;
				}
			}
			else
			{
				pNodeInfo->uiDataChildCount = 0;
			}
		}
		else
		{
			pNodeInfo->ui64FirstChildId = 0;
			pNodeInfo->ui64LastChildId = 0;
			pNodeInfo->uiDataChildCount = 0;
		}
		
		// Child element count

		if( uiStorageFlags & NSF_HAVE_CELM_LIST_BIT)
		{
			// This bit should only be set for elements.
			
			if( pNodeInfo->eNodeType != ELEMENT_NODE)
			{
#ifdef FLM_DEBUG
				if( bAssertOnCorruption)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				else
#endif
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
				}
			
				goto Exit;
			}
			
			// NOTE: This bit may be set even if there are no children.
			// This bit also serves to indicate that this particular
			// element is a unique child element, and we should keep
			// a list of child elements as they are added and removed.
			// We should also enforce that no children have the same
			// name id.
			
			if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd,
				&pNodeInfo->uiChildElmCount)))
			{
				goto Exit;
			}
			
			// If the count > 0, the NSF_HAVE_CHILDREN_BIT better also be set.
			
			if( pNodeInfo->uiChildElmCount)
			{
				if( !(uiStorageFlags & NSF_HAVE_CHILDREN_BIT))
				{
#ifdef FLM_DEBUG
					if( bAssertOnCorruption)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					else
#endif
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
					}
					
					goto Exit;
				}
			}
		}
		else
		{
			pNodeInfo->uiChildElmCount = 0;
		}
		
		// Encryption ID
	
		if( uiStorageFlags & NSF_EXT_ENCRYPTED_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd,
				&pNodeInfo->uiEncDefId)))
			{
				goto Exit;
			}
		}
		else
		{
			pNodeInfo->uiEncDefId = 0;
		}
		
		// Annotation ID

		if( uiStorageFlags & NSF_EXT_ANNOTATION_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN64( &pucHeader, pucHeaderEnd, 
				&pNodeInfo->ui64AnnotationId)))
			{
				goto Exit;
			}

			pNodeInfo->ui64AnnotationId += ui64BaseId;
		}
		else
		{
			pNodeInfo->ui64AnnotationId = 0;
		}
		
		if( uiStorageFlags & NSF_HAVE_DATA_LEN_BIT)
		{
			if( RC_BAD( rc = f_decodeSEN( &pucHeader, pucHeaderEnd,
				&pNodeInfo->uiDataLength)))
			{
				goto Exit;
			}
		}
		else
		{
			pNodeInfo->uiDataLength = 0;
		}
		
		// Account for storage flags
		
		pucHeader += uiStorageFlagsLen;
		
		// Make sure the header was the expected size
		
		if( pucHeader != pucHeaderEnd)
		{
#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}

		// Set the data length to whatever is remaining if we didn't
		// have a data length in the header
		
		if( bHaveData && !(uiStorageFlags & NSF_HAVE_DATA_LEN_BIT))
		{
			flmAssert( uiOverallLength >= uiHeaderStorageSize);
			pNodeInfo->uiDataLength = uiOverallLength - uiHeaderStorageSize;
		}

		// Set the fixed size header flag

		if( pbFixedSizeHeader)
		{
			*pbFixedSizeHeader = FALSE;
		}
	}
	
	if( pNodeInfo->uiEncDefId && !pNodeInfo->uiDataLength)
	{
#ifdef FLM_DEBUG
		if( bAssertOnCorruption)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
		else
#endif
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
		}
		
		goto Exit;
	}
	
	if( puiStorageFlags)
	{
		*puiStorageFlags = uiStorageFlags;
	}
	
Exit:

	if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BAD_SEN)
	{
		if( !bEOFValid)
		{
			// If one of the calls to read from the stream returned an EOF error,
			// the database is corrupt.

#ifdef FLM_DEBUG
			if( bAssertOnCorruption)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			else
#endif
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::readNode(
	F_Db *				pDb,
	FLMUINT				uiCollection,
	FLMUINT64			ui64NodeId,
	IF_IStream *		pIStream,
	FLMUINT				uiOverallLength,
	FLMBYTE *			pucIV)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiStorageLength;
	FLMUINT				uiStorageFlags;
	FLMUINT				uiIVLen;
	FLMBYTE				ucIV[ 16];
	FLMBOOL				bEOFValid = TRUE;
	FLMBOOL				bFixedSizeHeader;
	
	if( RC_BAD( rc = flmReadNodeInfo( uiCollection, 
		ui64NodeId, pIStream, uiOverallLength, FALSE, &m_nodeInfo, 
		&uiStorageFlags, &bFixedSizeHeader)))
	{
		goto Exit;
	}
	
	bEOFValid = FALSE;
	m_uiFlags = 0;

	if( bFixedSizeHeader)
	{
		m_uiFlags |= FDOM_FIXED_SIZE_HEADER;
	}
	
	// Read the child element list, if any
	
	if( uiStorageFlags & NSF_HAVE_CELM_LIST_BIT)
	{
		if( m_nodeInfo.uiChildElmCount)
		{
			FLMUINT			uiLen;
			FLMUINT			uiLoop;
			NODE_ITEM *		pElmNode;
			FLMUINT			uiPrevNameId = 0;
			FLMUINT64		ui64ElmNodeId = getNodeId();
			FLMUINT			uiChildElmCount = m_nodeInfo.uiChildElmCount;
			
			flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
			
			// Need to set to zero so the resizeChildElmList will work.  This
			// is the actual size allocated.
			
			m_nodeInfo.uiChildElmCount = 0;
						  
			if( RC_BAD( rc = resizeChildElmList( uiChildElmCount, FALSE)))
			{
				goto Exit;
			}
			
			// Read in all of the element name IDs and node IDs.
			
			for( uiLoop = 0, pElmNode = m_pNodeList;
				  uiLoop < m_nodeInfo.uiChildElmCount;
				  uiLoop++, pElmNode++)
			{
				if( RC_BAD( rc = f_readSEN( pIStream, &pElmNode->uiNameId,
					&uiLen)))
				{
					goto Exit;
				}
				
				pElmNode->uiNameId += uiPrevNameId;
				uiPrevNameId = pElmNode->uiNameId;
				
				if( RC_BAD( rc = f_readSEN64( pIStream, 
					&pElmNode->ui64NodeId, &uiLen)))
				{
					goto Exit;
				}
				
				pElmNode->ui64NodeId += ui64ElmNodeId;
			}
		}
		
		m_uiFlags |= FDOM_HAVE_CELM_LIST;
	}
	
	// Read the attribute list
	
	if( uiStorageFlags & NSF_HAVE_ATTR_LIST_BIT)
	{
		if( m_nodeInfo.eNodeType != ELEMENT_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
		
		if( RC_BAD( rc = importAttributeList( pDb, pIStream, FALSE)))
		{
			goto Exit;
		}
	}
	
	// Read the initialization vector if this is an encrypted node

	if( uiStorageFlags & NSF_EXT_ENCRYPTED_BIT)
	{
		F_Dict *		pDict;
		F_ENCDEF *	pEncDef;
		
		if( RC_BAD( rc = pDb->getDictionary( &pDict)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDict->getEncDef( getEncDefId(), &pEncDef)))
		{
			goto Exit;
		}
		
		uiIVLen = pEncDef->pCcs->getIVLen();

		if( uiIVLen != 8 && uiIVLen != 16)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
		
		if( RC_BAD( rc = pIStream->read( ucIV, uiIVLen)))
		{
			goto Exit;
		}

		if( pucIV)
		{
			f_memcpy( pucIV, ucIV, uiIVLen);
		}

		uiStorageLength = getEncLen( m_nodeInfo.uiDataLength);
	}
	else
	{
		uiStorageLength = m_nodeInfo.uiDataLength;
	}

	// Read the data part of the node, if any

	if( uiStorageLength)
	{
		// Data size must have room for data to point back to
		// the node.  Always align on 8 byte boundaries - just to be safe.
		// It is the highest alignment we support.
		
		if( calcDataBufSize( uiStorageLength) <= 
				gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->getMaxCellSize() ||
			m_nodeInfo.eNodeType == ELEMENT_NODE)
		{
			if( RC_BAD( rc = resizeDataBuffer( uiStorageLength, FALSE)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pIStream->read( (char *)getDataPtr(), 
				uiStorageLength)))
			{
				goto Exit;
			}

			// Decrypt the data

			if( uiStorageFlags & NSF_EXT_ENCRYPTED_BIT)
			{
				if( RC_BAD( rc = pDb->decryptData(
					m_nodeInfo.uiEncDefId, ucIV, getDataPtr(),
					uiStorageLength, getDataPtr(), uiStorageLength)))
				{
					goto Exit;
				}
			}
			
			// If the type is a number, cache a 'quick' number.
			
			if( m_nodeInfo.uiDataType == XFLM_NUMBER_TYPE)
			{
				FLMUINT64			ui64Num;
				FLMBOOL				bNeg;
				
				if( RC_BAD( rc = flmStorageNumberToNumber(
					getDataPtr(), getDataLength(), &ui64Num, &bNeg)))
				{
					goto Exit;
				}
				
				if( !bNeg)
				{
					setUINT64( ui64Num);
				}
				else
				{
					setINT64( -(FLMINT64)ui64Num);
				}
			}
		}
		else
		{
			flmAssert( m_nodeInfo.eNodeType != ELEMENT_NODE);
			flmAssert( m_nodeInfo.uiDataType == XFLM_TEXT_TYPE ||
						m_nodeInfo.uiDataType == XFLM_BINARY_TYPE);
			m_uiFlags |= FDOM_VALUE_ON_DISK;
		}
	}
	
	if( uiStorageFlags & NSF_EXT_READ_ONLY_BIT)
	{
		m_uiFlags |= FDOM_READ_ONLY;
	}
	
	if( uiStorageFlags & NSF_EXT_CANNOT_DELETE_BIT)
	{
		m_uiFlags |= FDOM_CANNOT_DELETE;
	}
	
	if( uiStorageFlags & NSF_EXT_QUARANTINED_BIT)
	{
		m_uiFlags |= FDOM_QUARANTINED;
	}
	
	if( uiStorageFlags & NSF_EXT_NAMESPACE_DECL_BIT)
	{
		m_uiFlags |= FDOM_NAMESPACE_DECL;
	}
	
	// Sanity checks

	switch( m_nodeInfo.eNodeType)
	{
		case DOCUMENT_NODE:
		{
			if( !isRootNode() || getFirstChildId() != getLastChildId())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			break;
		}

		case ELEMENT_NODE:
		{
			if( isRootNode() && getParentId())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			break;
		}

		default:
		{
			break;
		}
	}
	
Exit:

	if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BAD_SEN)
	{
		if( !bEOFValid)
		{
			// If one of the calls to read from the stream returned an EOF error,
			// the database is corrupt.

			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::importAttributeList(
	F_Db *					pDb,
	IF_IStream *			pIStream,
	FLMBOOL					bMutexAlreadyLocked)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	FLMUINT			uiAttrCount;
	FLMUINT			uiNameId;
	FLMUINT			uiBaseNameId;
	FLMUINT			uiStorageFlags;
	FLMUINT			uiPayloadLength;
	FLMUINT			uiLoop;
	FLMUINT			uiInsertPos;
	F_AttrElmInfo	defInfo;
	
	flmAssert( !m_uiAttrCount);
	flmAssert( m_nodeInfo.eNodeType == ELEMENT_NODE);
	
	// Determine the number of attributes

	if( RC_BAD( rc = f_readSEN( pIStream, &uiAttrCount)))
	{
		goto Exit;
	}
	
	if( !uiAttrCount)
	{
		goto Exit;
	}

	// Import the attributes
	
	if( RC_BAD( rc = f_readSEN( pIStream, &uiBaseNameId)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiAttrCount; uiLoop++)
	{
		if( RC_BAD( rc = f_readSEN( pIStream, &uiNameId)))
		{
			goto Exit;
		}
		
		uiNameId += uiBaseNameId;
		
		if (getAttribute( uiNameId, &uiInsertPos) != NULL)
		{
			flmAssert( 0);
		}
		if( RC_BAD( rc = allocAttribute( pDb, 
			uiNameId, NULL, uiInsertPos, &pAttrItem, bMutexAlreadyLocked)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = f_readSEN( pIStream, &uiStorageFlags)))
		{
			goto Exit;
		}
		
		if( uiStorageFlags & ASF_READ_ONLY_BIT)
		{
			pAttrItem->m_uiFlags |= FDOM_READ_ONLY;
		}
		
		if( uiStorageFlags & ASF_CANNOT_DELETE_BIT)
		{
			pAttrItem->m_uiFlags |= FDOM_CANNOT_DELETE;
		}

		if( uiStorageFlags & ASF_HAVE_PREFIX_BIT)
		{
			if( RC_BAD( rc = f_readSEN( pIStream, &pAttrItem->m_uiPrefixId)))
			{
				goto Exit;
			}
		}
		
		uiPayloadLength = (uiStorageFlags & ASF_PAYLOAD_LEN_MASK);

		if( uiPayloadLength == ASF_HAVE_PAYLOAD_LEN_SEN)
		{
			if( RC_BAD( rc = f_readSEN( pIStream, &uiPayloadLength)))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = pDb->m_pDict->getAttribute( pDb, uiNameId, &defInfo)))
		{
			goto Exit;
		}

		pAttrItem->m_uiDataType = defInfo.getDataType();
		
		if( uiStorageFlags & ASF_ENCRYPTED_BIT)
		{
			F_ENCDEF *	pEncDef;
			
			if( RC_BAD( rc = f_readSEN( pIStream, &pAttrItem->m_uiEncDefId)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = f_readSEN( pIStream, 
				&pAttrItem->m_uiDecryptedDataLen)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pDb->m_pDict->getEncDef( 
				pAttrItem->m_uiEncDefId, &pEncDef)))
			{
				goto Exit;
			}
			
			pAttrItem->m_uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( pAttrItem->m_uiIVLen == 8 || pAttrItem->m_uiIVLen == 16);
		}
		
		if( uiPayloadLength)
		{
			if( RC_BAD( rc = pAttrItem->resizePayloadBuffer( 
				uiPayloadLength, bMutexAlreadyLocked)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pIStream->read( pAttrItem->getAttrPayloadPtr(),
				uiPayloadLength)))
			{
				goto Exit;
			}
		}
		
		pAttrItem->m_uiPayloadLen = uiPayloadLength;
		
		if( pAttrItem->m_uiDataType == XFLM_NUMBER_TYPE && 
			!pAttrItem->m_uiEncDefId)
		{
			FLMBOOL		bNeg;

			if( RC_BAD( rc = flmStorageNumberToNumber( 
				pAttrItem->getAttrDataPtr(), pAttrItem->getAttrDataLength(), 
				&pAttrItem->m_ui64QuickVal, &bNeg)))
			{
				goto Exit;
			}

			if( !bNeg)
			{
				pAttrItem->m_uiFlags |= FDOM_UNSIGNED_QUICK_VAL;
			}
			else
			{
				pAttrItem->m_uiFlags |= FDOM_SIGNED_QUICK_VAL;
			}
		}
	}
	
Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::importAttributeList(
	F_Db *			pDb,
	F_CachedNode *	pSourceNode,
	FLMBOOL			bMutexAlreadyLocked)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pNewItem;
	F_AttrItem *	pSourceItem;
	FLMUINT			uiLoop;
	
	flmAssert( !m_uiAttrCount);
	if( RC_BAD( rc = resizeAttrList( pSourceNode->m_uiAttrCount,
								bMutexAlreadyLocked)))
	{
		goto Exit;
	}

	for (uiLoop = 0; uiLoop < pSourceNode->m_uiAttrCount; uiLoop++)
	{
		pSourceItem = pSourceNode->m_ppAttrList [uiLoop];
	
		if( RC_BAD( rc = allocAttribute( pDb, 
			pSourceItem->m_uiNameId, pSourceItem, uiLoop, &pNewItem,
			bMutexAlreadyLocked)))
		{
			goto Exit;
		}

		if( pSourceItem->m_uiPayloadLen > sizeof( FLMBYTE *))
		{
			if( RC_BAD( rc = pNewItem->setupAttribute( 
				pDb, pSourceItem->m_uiEncDefId, 
				pSourceItem->getAttrDataLength(), FALSE,
				bMutexAlreadyLocked)))
			{
				goto Exit;
			}

			flmAssert( pSourceItem->getAttrPayloadSize() == 
						  pNewItem->getAttrPayloadSize());
			
			f_memcpy( pNewItem->getAttrPayloadPtr(), 
				pSourceItem->getAttrPayloadPtr(), pSourceItem->m_uiPayloadLen);
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_AttrItem::getAttrSizeNeeded(
	FLMUINT					uiBaseNameId,
	XFLM_NODE_INFO *		pNodeInfo,
	FLMUINT *				puiSaveStorageFlags,
	FLMUINT *				puiSizeNeeded)
{
	FLMUINT	uiNameSize;
	FLMUINT	uiFlagsSize;
	FLMUINT	uiPrefixSize;
	FLMUINT	uiPayloadLength;
	FLMUINT	uiPayloadLenSize;
	FLMUINT	uiEncIdSize;
	FLMUINT	uiUnencLenSize;
	FLMUINT	uiOverhead = 0;
	FLMUINT	uiStorageFlags;

	uiNameSize = f_getSENByteCount( m_uiNameId - uiBaseNameId);
	uiOverhead += uiNameSize;
							
	uiStorageFlags = getAttrStorageFlags();
	if (puiSaveStorageFlags)
	{
		*puiSaveStorageFlags = uiStorageFlags;
	}

	uiFlagsSize = f_getSENByteCount( uiStorageFlags);
	uiOverhead += uiFlagsSize;

	if( m_uiPrefixId)
	{
		uiPrefixSize = f_getSENByteCount( m_uiPrefixId);
		uiOverhead += uiPrefixSize;
	}
	else
	{
		uiPrefixSize = 0;
	}

	uiPayloadLength = m_uiPayloadLen;
	(*puiSizeNeeded) += uiPayloadLength;
	
	if( uiPayloadLength > ASF_MAX_EMBEDDED_PAYLOAD_LEN)
	{
		uiPayloadLenSize = f_getSENByteCount( uiPayloadLength);
		uiOverhead += uiPayloadLenSize;
	}
	else
	{
		uiPayloadLenSize = 0;
	}

	if( m_uiEncDefId)
	{
		flmAssert( uiPayloadLength);
		
		uiEncIdSize = f_getSENByteCount( m_uiEncDefId);
		uiOverhead += uiEncIdSize;
		uiUnencLenSize = f_getSENByteCount( m_uiDecryptedDataLen);
		uiOverhead += uiUnencLenSize;
	}
	else
	{
		uiEncIdSize = 0;
		uiUnencLenSize = 0;
	}

	(*puiSizeNeeded) += uiOverhead;

	if (pNodeInfo)
	{
		FLMUINT	uiDataLength;

		pNodeInfo->nameId.ui64Bytes += (FLMUINT64)uiNameSize;
		pNodeInfo->nameId.ui64Count++;

		pNodeInfo->attrFlags.ui64Bytes += (FLMUINT64)uiFlagsSize;
		pNodeInfo->attrFlags.ui64Count++;

		if (uiPrefixSize)
		{
			pNodeInfo->prefixId.ui64Bytes += (FLMUINT64)uiPrefixSize;
			pNodeInfo->prefixId.ui64Count++;
		}

		if (uiPayloadLenSize)
		{
			pNodeInfo->attrPayloadLen.ui64Bytes += (FLMUINT64)uiPayloadLenSize;
			pNodeInfo->attrPayloadLen.ui64Count++;
		}

		uiDataLength = getAttrDataLength();
		if (m_uiEncDefId)
		{
			FLMUINT	uiEncPadding;

			pNodeInfo->encDefId.ui64Bytes += (FLMUINT64)uiEncIdSize;
			pNodeInfo->encDefId.ui64Count++;
			pNodeInfo->unencDataLen.ui64Bytes += (FLMUINT64)uiUnencLenSize;
			pNodeInfo->unencDataLen.ui64Count++;
			pNodeInfo->encIV.ui64Bytes += (FLMUINT64)m_uiIVLen;
			pNodeInfo->encIV.ui64Count++;
			uiOverhead += m_uiIVLen;
			flmAssert( m_uiPayloadLen >= m_uiIVLen - uiDataLength);
			uiEncPadding = m_uiPayloadLen - m_uiIVLen - uiDataLength;
			if (uiEncPadding)
			{
				pNodeInfo->encPadding.ui64Bytes += (FLMUINT64)uiEncPadding;
				pNodeInfo->encPadding.ui64Count++;
				uiOverhead += uiEncPadding;
			}
		}

		pNodeInfo->totalOverhead.ui64Bytes += (FLMUINT64)uiOverhead;
		pNodeInfo->totalOverhead.ui64Count++;

		pNodeInfo->attributeNode.ui64Bytes += (FLMUINT64)(uiOverhead + uiDataLength);
		pNodeInfo->attributeNode.ui64Count++;
		switch (m_uiDataType)
		{
			case XFLM_NODATA_TYPE:
				pNodeInfo->dataNodata.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataNodata.ui64Count++;
				break;
			case XFLM_TEXT_TYPE:
				pNodeInfo->dataString.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataString.ui64Count++;
				break;
			case XFLM_NUMBER_TYPE:
				pNodeInfo->dataNumeric.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataNumeric.ui64Count++;
				break;
			case XFLM_BINARY_TYPE:
				pNodeInfo->dataBinary.ui64Bytes += (FLMUINT64)uiDataLength;
				pNodeInfo->dataBinary.ui64Count++;
				break;
			default:
				flmAssert( 0);
				break;
		}
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::exportAttributeList(
	F_Db *					pDb,
	F_DynaBuf *				pDynaBuf,
	XFLM_NODE_INFO *		pNodeInfo)
{
	RCODE						rc = NE_XFLM_OK;
	F_AttrItem *			pAttrItem;
	FLMBYTE *				pucStart;
	FLMBYTE *				pucBuf;
	FLMBYTE *				pucEnd;
	FLMUINT					uiPayloadLength;
	FLMUINT					uiLoop;
	FLMUINT					uiSizeNeeded;
	FLMUINT					uiStorageFlags;
	FLMUINT					uiBaseNameId = m_ppAttrList [0]->m_uiNameId;
#define MAX_STORAGE_FLAGS	32
	FLMUINT					storageFlagsList[ MAX_STORAGE_FLAGS];
	
	// Logging should be done by the caller
	
#ifdef FLM_DEBUG
	if (!pNodeInfo)
	{
		flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	}
#endif

	// Determine the size of the node buffer

	uiSizeNeeded = 0;
	for (uiLoop = 0; uiLoop < m_uiAttrCount; uiLoop++)
	{
		pAttrItem = m_ppAttrList [uiLoop];
		
		// If uiAttributeCount < MAX_STORAGE_FLAGS, we pass in a pointer
		// to that slot in the storageFlagsList array to save the
		// storage flags, so we don't have to recalculate them in the
		// 2nd pass - up to the first MAX_STORAGE_FLAGS storage flags
		// will be saved.
		
		if (uiLoop < MAX_STORAGE_FLAGS)
		{
			pAttrItem->getAttrSizeNeeded( uiBaseNameId, pNodeInfo,
									&storageFlagsList [uiLoop],
									&uiSizeNeeded);
		}
		else
		{
			pAttrItem->getAttrSizeNeeded( uiBaseNameId, pNodeInfo, NULL,
									&uiSizeNeeded);
		}
	}
	
	flmAssert( m_uiAttrCount);
	if (pNodeInfo)
	{
		flmAssert( !pDynaBuf);
		pNodeInfo->attrCount.ui64Bytes += f_getSENByteCount( m_uiAttrCount);
		pNodeInfo->attrCount.ui64Count++;
		pNodeInfo->attrBaseId.ui64Bytes += f_getSENByteCount( uiBaseNameId);
		pNodeInfo->attrBaseId.ui64Count++;
	}
	else
	{
		uiSizeNeeded += f_getSENByteCount( m_uiAttrCount);
		uiSizeNeeded += f_getSENByteCount( uiBaseNameId);
		flmAssert( pDynaBuf);

		if( RC_BAD( rc = pDynaBuf->allocSpace( uiSizeNeeded, (void **)&pucBuf)))
		{
			goto Exit;
		}

		pucStart = pucBuf;
		pucEnd = pucStart + uiSizeNeeded;

		if( RC_BAD( rc = f_encodeSEN( m_uiAttrCount, &pucBuf, pucEnd)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = f_encodeSEN( uiBaseNameId, &pucBuf, pucEnd)))
		{
			goto Exit;
		}
		
		// Once we have written out attribute count, we can reset it to zero.

		for (uiLoop = 0; uiLoop < m_uiAttrCount; uiLoop++)
		{
			pAttrItem = m_ppAttrList [uiLoop];
			if( RC_BAD( rc = f_encodeSEN( pAttrItem->m_uiNameId - uiBaseNameId,
				&pucBuf, pucEnd)))
			{
				goto Exit;
			}
			
			// If we saved the storage flags in the first pass, get them
			// from storageFlagsList, otherwise recalculate them.
			
			uiStorageFlags = (uiLoop < MAX_STORAGE_FLAGS)
									? storageFlagsList [uiLoop]
									: pAttrItem->getAttrStorageFlags();

			if( RC_BAD( rc = f_encodeSEN( uiStorageFlags, &pucBuf, pucEnd)))
			{
				goto Exit;
			}
			
			if( pAttrItem->m_uiPrefixId)
			{
				if( RC_BAD( rc = f_encodeSEN( pAttrItem->m_uiPrefixId,
					&pucBuf, pucEnd)))
				{
					goto Exit;
				}
			}

			uiPayloadLength = pAttrItem->m_uiPayloadLen;

			if( uiPayloadLength > ASF_MAX_EMBEDDED_PAYLOAD_LEN)
			{
				if( RC_BAD( rc = f_encodeSEN( uiPayloadLength, &pucBuf, pucEnd)))
				{
					goto Exit;
				}
			}
			
			if( pAttrItem->m_uiEncDefId)
			{
				flmAssert( uiPayloadLength);
				
				if( RC_BAD( rc = f_encodeSEN( pAttrItem->m_uiEncDefId,
					&pucBuf, pucEnd)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = f_encodeSEN( pAttrItem->m_uiDecryptedDataLen,
					&pucBuf, pucEnd)))
				{
					goto Exit;
				}
			}

			f_memcpy( pucBuf, pAttrItem->getAttrPayloadPtr(), uiPayloadLength);
			pucBuf += uiPayloadLength;
		}
		
		flmAssert( pucBuf == pucEnd);
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CachedNode::resetNode( void)
{
	FLMBYTE *	pucActualAlloc;

	// Should not count attribute size here, because it will be subtracted
	// when the attributes themselves are deleted.

	FLMUINT		uiSize = memSize() - m_uiTotalAttrSize;

	flmAssert( !m_pPrevInBucket);
	flmAssert( !m_pNextInBucket);
	flmAssert( !m_pOlderVersion);
	flmAssert( !m_pNewerVersion);
	flmAssert( !m_pPrevInOldList);
	flmAssert( !m_pNextInOldList);
	flmAssert( !m_pNotifyList);
	flmAssert( !m_uiStreamUseCount);
	flmAssert( !nodeInUse());

	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);

	// If this is an old version, decrement the old version counters.

	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiSize &&
					  gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerCount);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
	}

	flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiSize &&
				  gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCount);
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiSize;

	if( m_uiFlags & FDOM_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

	if( m_pucData || m_pNodeList || m_ppAttrList)
	{
		f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);

		if( m_pucData)
		{
			pucActualAlloc = getActualPointer( m_pucData);
			gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
								m_uiDataBufSize, &pucActualAlloc);
			m_pucData = NULL;
			m_uiDataBufSize = 0;
		}
		
		if( m_pNodeList)
		{
			pucActualAlloc = getActualPointer( m_pNodeList);
			gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
								calcNodeListBufSize( m_nodeInfo.uiChildElmCount),
								&pucActualAlloc);
			m_pNodeList = NULL;
		}
	
		if( m_ppAttrList)
		{
			FLMUINT	uiLoop;

			for (uiLoop = 0; uiLoop < m_uiAttrCount; uiLoop++)
			{
				delete m_ppAttrList [uiLoop];
			}
			pucActualAlloc = getActualPointer( m_ppAttrList);
			gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf(
								calcAttrListBufSize( m_uiAttrCount), &pucActualAlloc);
			m_ppAttrList = NULL;
			m_uiAttrCount = 0;
		}
	}

	m_ui64LowTransId = 0;
	m_ui64HighTransId = FLM_MAX_UINT64;
	m_uiCacheFlags = 0;
	m_pDatabase = NULL;
	m_uiFlags = 0;
	m_uiOffsetIndex = 0;
	m_ui32BlkAddr = 0;
	f_memset( &m_nodeInfo, 0, sizeof( F_NODE_INFO));

	uiSize = memSize();
	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiSize;
	}
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiSize;
}
	
#undef new
#undef delete

/****************************************************************************
Desc:
****************************************************************************/
void * F_CachedNode::operator new(
	FLMSIZET			uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( uiSize);
#endif
	flmAssert( uiSize == sizeof( F_CachedNode));
	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);

	return( gv_XFlmSysData.pNodeCacheMgr->m_pNodeAllocator->allocCell(
				&gv_XFlmSysData.pNodeCacheMgr->m_nodeRelocator, NULL, 0));
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_CachedNode::operator new[]( FLMSIZET)
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
void * F_CachedNode::operator new(
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
void * F_CachedNode::operator new[](
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
void F_CachedNode::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);
	gv_XFlmSysData.pNodeCacheMgr->m_pNodeAllocator->freeCell( (FLMBYTE *)ptr);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CachedNode::operator delete[](
	void *)		// ptr)
{
	flmAssert( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
F_AttrItem::~F_AttrItem()
{
	FLMUINT	uiSize = memSize();

	if (m_pCachedNode)
	{
		flmAssert( m_pCachedNode->m_uiTotalAttrSize >= uiSize);
		m_pCachedNode->m_uiTotalAttrSize -= uiSize;
		if (m_pCachedNode->m_ui64HighTransId != FLM_MAX_UINT64)
		{
			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
		}
		flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiSize);
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiSize;
	}
	
	if( m_uiPayloadLen > sizeof( FLMBYTE *))
	{
		f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);
		
		m_pucPayload -= sizeof( F_AttrItem *);
		gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf( 
			m_uiPayloadLen + sizeof( F_AttrItem *),
			&m_pucPayload);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_AttrItem::operator new(
	FLMSIZET			uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( uiSize);
#endif
	flmAssert( uiSize == sizeof( F_AttrItem));
	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);

	return( gv_XFlmSysData.pNodeCacheMgr->m_pAttrItemAllocator->allocCell(
				&gv_XFlmSysData.pNodeCacheMgr->m_attrItemRelocator, NULL, 0));
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_AttrItem::operator new[]( FLMSIZET)
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
void * F_AttrItem::operator new(
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
void * F_AttrItem::operator new[](
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
void F_AttrItem::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	f_assertMutexLocked( gv_XFlmSysData.hNodeCacheMutex);
	gv_XFlmSysData.pNodeCacheMgr->m_pAttrItemAllocator->freeCell( (FLMBYTE *)ptr);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_AttrItem::operator delete[](
	void *)		// ptr)
{
	flmAssert( 0);
}
