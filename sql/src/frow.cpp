//------------------------------------------------------------------------------
// Desc:	This is the row cache for FLAIM-SQL
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

#include "flaimsys.h"

#if defined( FLM_NLM) && !defined( __MWERKS__)
// Disable "Warning! W549: col(XX) 'sizeof' operand contains
// compiler generated information"
	#pragma warning 549 9
#endif

FSTATIC RCODE getStorageAsNumber(
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	eDataType			eDataType,
	FLMUINT64 *			pui64Number,
	FLMBOOL *			pbNeg);
	
/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_RowCacheMgr::F_RowCacheMgr()
{
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
	m_bReduceInProgress = FALSE;
#ifdef FLM_DEBUG
	m_bDebug = FALSE;
#endif
}
	
/****************************************************************************
Desc:	Constructor for F_Row
****************************************************************************/
F_Row::F_Row()
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
	// be treated as one that had memory assigned to the old version rows.
	
	m_ui64HighTransId = FLM_MAX_UINT64;
	m_pNotifyList = NULL;
	m_uiCacheFlags = 0;
	m_uiStreamUseCount = 0;
	
	// Items initialized in constructor

	m_uiColumnDataBufSize = 0;	
	m_pucColumnData = NULL;
	m_pColumns = NULL;
	m_uiFlags = 0;
	
	m_uiTableNum = 0;
	m_ui64RowId = 0;
}

/****************************************************************************
Desc:	Destructor for F_Row object.
		This routine assumes the global mutex is already locked.
****************************************************************************/
F_Row::~F_Row()
{
	FLMUINT		uiSize = memSize();
	FLMBYTE *	pucActualAlloc;
	
	flmAssert( !m_uiStreamUseCount);
	f_assertMutexLocked( gv_SFlmSysData.hRowCacheMutex);

	// If this is an old version, decrement the old version counters.

	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes >= uiSize &&
					  gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount);
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount--;
		unlinkFromOldList();
	}

	flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount >= uiSize &&
				  gv_SFlmSysData.pRowCacheMgr->m_Usage.uiCount);
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount -= uiSize;
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiCount--;

	if( m_uiFlags & FROW_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}
	
	// Free the column data, if any
	
	if (m_pucColumnData)
	{
		pucActualAlloc = getActualPointer( m_pucColumnData);
		gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->freeBuf(
							calcDataBufSize(m_uiColumnDataBufSize),
							&pucActualAlloc);
		m_pucColumnData = NULL;
	}
	
	// Free the column item list
	
	if (m_pColumns)
	{
		pucActualAlloc = getActualPointer( m_pColumns);
		gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->freeBuf(
							calcColumnListBufSize( m_uiNumColumns),
							&pucActualAlloc);
		m_pColumns = NULL;
		m_uiNumColumns = 0;
	}

	if (shouldRehash( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiCount,
							gv_SFlmSysData.pRowCacheMgr->m_uiNumBuckets))
	{
		if (checkHashFailTime( &gv_SFlmSysData.pRowCacheMgr->m_uiHashFailTime))
		{
			(void)gv_SFlmSysData.pRowCacheMgr->rehash();
		}
	}
}

/****************************************************************************
Desc:	This routine frees a purged row from row cache.  This routine assumes
		that the row cache mutex has already been locked.
****************************************************************************/
void F_Row::freePurged( void)
{

	// Unlink the row from the purged list.

	unlinkFromPurged();

	// Free the F_Row object.

	unsetPurged();

	delete this;
}

/****************************************************************************
Desc:	This routine frees a row in the row cache.  This routine assumes
		that the row cache mutex has already been locked.
****************************************************************************/
void F_Row::freeCache(
	FLMBOOL	bPutInPurgeList)
{
	FLMBOOL	bOldVersion;

	bOldVersion = (FLMBOOL)((m_ui64HighTransId != FLM_MAX_UINT64)
									? TRUE
									: FALSE);

	// Unlink the row from its various lists.

	gv_SFlmSysData.pRowCacheMgr->m_MRUList.unlinkGlobal(
					(F_CachedItem *)this);
	unlinkFromDatabase();
	
	if (!m_pNewerVersion)
	{
		F_Row *	pOlderVersion = m_pOlderVersion;

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
	
	if( m_uiFlags & FROW_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

	// Free the F_Row structure if not putting in purge list.

	if (!bPutInPurgeList)
	{
		delete this;
	}
	else
	{
		if ((m_pNextInGlobal = gv_SFlmSysData.pRowCacheMgr->m_pPurgeList) != NULL)
		{
			m_pNextInGlobal->m_pPrevInGlobal = this;
		}
		gv_SFlmSysData.pRowCacheMgr->m_pPurgeList = this;
		
		// Unset the dirty flags - don't want anything in the purge list
		// to be dirty.
		
		m_uiFlags &= ~(FROW_DIRTY | FROW_NEW);
		setPurged();
		flmAssert( !m_pPrevInGlobal);
	}
}

/****************************************************************************
Desc:	This routine initializes row cache manager.
****************************************************************************/
RCODE F_RowCacheMgr::initCache( void)
{
	RCODE		rc = NE_SFLM_OK;

	// Allocate the hash buckets.

	if (RC_BAD( rc = f_calloc(
								(FLMUINT)sizeof( F_Row *) *
								(FLMUINT)MIN_HASH_BUCKETS,
								&m_ppHashBuckets)))
	{
		goto Exit;
	}
	m_uiNumBuckets = MIN_HASH_BUCKETS;
	m_uiHashMask = m_uiNumBuckets - 1;
	gv_SFlmSysData.pGlobalCacheMgr->incrTotalBytes( f_msize( m_ppHashBuckets));

	if( RC_BAD( rc = FlmAllocFixedAllocator( &m_pRowAllocator)))
	{
		goto Exit;
	}

	// Set up the F_Row object allocator

	if (RC_BAD( rc = m_pRowAllocator->setup(
		TRUE, gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager,
		&m_rowRelocator, sizeof( F_Row), &m_Usage.slabUsage, NULL)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = FlmAllocBufferAllocator( &m_pBufAllocator)))
	{
		goto Exit;
	}
	
	// Set up the buffer allocator for F_Row objects

	if (RC_BAD( rc = m_pBufAllocator->setup(
		TRUE, gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager, NULL,
		&m_Usage.slabUsage, NULL)))
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
Desc:		Determine if a row can be moved.
Notes:	This routine assumes the row cache mutex is locked
			This is a static method, so there is no "this" pointer to the
			F_RowCacheMgr object.
****************************************************************************/
FLMBOOL F_RowRelocator::canRelocate(
	void *		pvAlloc)
{
	return( ((F_Row *)pvAlloc)->rowInUse() ? FALSE : TRUE);
}

/****************************************************************************
Desc:		Fixes up all pointers needed to allow an F_Row object to be
			moved to a different location in memory
Notes:	This routine assumes the row cache mutex is locked.
			This is a static method, so there is no "this" pointer to the
			F_RowCacheMgr object.
****************************************************************************/
void F_RowRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	F_Row *				pOldRow = (F_Row *)pvOldAlloc;
	F_Row *				pNewRow = (F_Row *)pvNewAlloc;
	F_Row **				ppBucket;
	F_Database *		pDatabase = pOldRow->m_pDatabase;
	F_RowCacheMgr *	pRowCacheMgr = gv_SFlmSysData.pRowCacheMgr;
	FLMBYTE *			pucActualAlloc;

	flmAssert( !pOldRow->rowInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);

	// Update the F_Row pointer in the data buffer

	if (pNewRow->m_pucColumnData)
	{
		pucActualAlloc = getActualPointer( pNewRow->m_pucColumnData);
		flmAssert( *((F_Row **)(pucActualAlloc)) == pOldRow);
		pNewRow->setRowAndDataPtr( pucActualAlloc);
	}

	if (pNewRow->m_pColumns)
	{
		pucActualAlloc = getActualPointer( pNewRow->m_pColumns);
		flmAssert( *((F_Row **)(pucActualAlloc)) == pOldRow);
		pNewRow->setColumnListPtr( pucActualAlloc);
	}

	if (pNewRow->m_pPrevInDatabase)
	{
		pNewRow->m_pPrevInDatabase->m_pNextInDatabase = pNewRow;
	}

	if (pNewRow->m_pNextInDatabase)
	{
		pNewRow->m_pNextInDatabase->m_pPrevInDatabase = pNewRow;
	}

	if (pNewRow->m_pPrevInGlobal)
	{
		pNewRow->m_pPrevInGlobal->m_pNextInGlobal = pNewRow;
	}

	if (pNewRow->m_pNextInGlobal)
	{
		pNewRow->m_pNextInGlobal->m_pPrevInGlobal = pNewRow;
	}

	if (pNewRow->m_pPrevInBucket)
	{
		pNewRow->m_pPrevInBucket->m_pNextInBucket = pNewRow;
	}

	if (pNewRow->m_pNextInBucket)
	{
		pNewRow->m_pNextInBucket->m_pPrevInBucket = pNewRow;
	}

	if (pNewRow->m_pOlderVersion)
	{
		pNewRow->m_pOlderVersion->m_pNewerVersion = pNewRow;
	}

	if (pNewRow->m_pNewerVersion)
	{
		pNewRow->m_pNewerVersion->m_pOlderVersion = pNewRow;
	}
	
	if (pNewRow->m_pPrevInHeapList)
	{
		pNewRow->m_pPrevInHeapList->m_pNextInHeapList = pNewRow;
	}
	
	if (pNewRow->m_pNextInHeapList)
	{
		pNewRow->m_pNextInHeapList->m_pPrevInHeapList = pNewRow;
	}

	if (pNewRow->m_pPrevInOldList)
	{
		pNewRow->m_pPrevInOldList->m_pNextInOldList = pNewRow;
	}
	
	if (pNewRow->m_pNextInOldList)
	{
		pNewRow->m_pNextInOldList->m_pPrevInOldList = pNewRow;
	}
	
	if( pDatabase)
	{
		if (pDatabase->m_pFirstRow == pOldRow)
		{
			pDatabase->m_pFirstRow = pNewRow;
		}

		if( pDatabase->m_pLastRow == pOldRow)
		{
			pDatabase->m_pLastRow = pNewRow;
		}
		
		if( pDatabase->m_pLastDirtyRow == pOldRow)
		{
			pDatabase->m_pLastDirtyRow = pNewRow;
		}
	}

	ppBucket = pRowCacheMgr->rowHash( pOldRow->m_ui64RowId);
	if( *ppBucket == pOldRow)
	{
		*ppBucket = pNewRow;
	}

	if (pRowCacheMgr->m_MRUList.m_pMRUItem == (F_CachedItem *)pOldRow)
	{
		pRowCacheMgr->m_MRUList.m_pMRUItem = pNewRow;
	}

	if (pRowCacheMgr->m_MRUList.m_pLRUItem == (F_CachedItem *)pOldRow)
	{
		pRowCacheMgr->m_MRUList.m_pLRUItem = pNewRow;
	}
	
	if (pRowCacheMgr->m_pHeapList == pOldRow)
	{
		pRowCacheMgr->m_pHeapList = pNewRow;
	}
	
	if (pRowCacheMgr->m_pOldList == pOldRow)
	{
		pRowCacheMgr->m_pOldList = pNewRow;
	}

	if (pRowCacheMgr->m_pPurgeList == pOldRow)
	{
		pRowCacheMgr->m_pPurgeList = pNewRow;
	}
}

/****************************************************************************
Desc:	Determine if a data buffer of an F_Row object can be moved.
		This routine assumes that the row cache mutex is locked.
****************************************************************************/
FLMBOOL F_ColumnDataRelocator::canRelocate(
	void *	pvAlloc)
{
	F_Row *	pRow = *((F_Row **)pvAlloc);
	
	if( pRow->rowInUse())
	{
		return( FALSE);
	}
	else
	{
		flmAssert( getActualPointer( pRow->m_pucColumnData) == (FLMBYTE *)pvAlloc);
		return( TRUE);
	}
}

/****************************************************************************
Desc:	Relocate the data buffer of an F_Row object.  This routine assumes
		that the row cache mutex is locked.
****************************************************************************/
void F_ColumnDataRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_Row *	pRow = *((F_Row **)pvOldAlloc);

	flmAssert( !pRow->rowInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( getActualPointer( pRow->m_pucColumnData) == (FLMBYTE *)pvOldAlloc);
	
	pRow->setRowAndDataPtr( (FLMBYTE *)pvNewAlloc);
}

/****************************************************************************
Desc:	Determine if an column list of an F_Row object can be moved.
		This routine assumes that the row cache mutex is locked.
****************************************************************************/
FLMBOOL F_ColumnListRelocator::canRelocate(
	void *	pvAlloc)
{
	F_Row *	pRow = *((F_Row **)pvAlloc);
	
	if( pRow->rowInUse())
	{
		return( FALSE);
	}
	else
	{
		flmAssert( getActualPointer( pRow->m_pColumns) == (FLMBYTE *)pvAlloc);
		return( TRUE);
	}
}

/****************************************************************************
Desc:	Relocate the column list of an F_Row object.  This routine assumes
		that the row cache mutex is locked.
****************************************************************************/
void F_ColumnListRelocator::relocate(
	void *	pvOldAlloc,
	void *	pvNewAlloc)
{
	F_Row *	pRow = *((F_Row **)pvOldAlloc);

	flmAssert( !pRow->rowInUse());
	flmAssert( pvNewAlloc < pvOldAlloc);
	flmAssert( getActualPointer( pRow->m_pColumns) == (FLMBYTE *)pvOldAlloc);
	
	pRow->setColumnListPtr( (FLMBYTE *)pvNewAlloc);
}

/****************************************************************************
Desc:	This routine resizes the hash table for the cache manager.
		NOTE: This routine assumes that the row cache mutex has been locked.
****************************************************************************/
RCODE F_RowCacheMgr::rehash( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiNewHashTblSize;
	F_Row **		ppOldHashTbl;
	FLMUINT		uiOldHashTblSize;
	F_Row **		ppBucket;
	FLMUINT		uiLoop;
	F_Row *		pTmpRow;
	F_Row *		pTmpNextRow;
	FLMUINT		uiOldMemSize;

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

	if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( F_Row *) *
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

	// Relink all of the rows into the new hash table.

	for (uiLoop = 0, ppBucket = ppOldHashTbl;
		  uiLoop < uiOldHashTblSize;
		  uiLoop++, ppBucket++)
	{
		pTmpRow = *ppBucket;
		while (pTmpRow)
		{
			pTmpNextRow = pTmpRow->m_pNextInBucket;
			pTmpRow->linkToHashBucket();
			pTmpRow = pTmpNextRow;
		}
	}

	// Throw away the old hash table.

	f_free( &ppOldHashTbl);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine shuts down the row cache manager and frees all
		resources allocated by it.  NOTE: Row cache mutex must be locked
		already, or we must be shutting down so that only one thread is
		calling this routine.
****************************************************************************/
F_RowCacheMgr::~F_RowCacheMgr()
{
	F_CachedItem *	pItem;
	F_CachedItem *	pNextItem;

	// Free all of the row cache objects.
	
	pItem = m_MRUList.m_pMRUItem;
	while (pItem)
	{
		pNextItem = pItem->m_pNextInGlobal;
		((F_Row *)pItem)->freeCache( FALSE);
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
		gv_SFlmSysData.pGlobalCacheMgr->decrTotalBytes( uiTotalMemory);
	}
	if (m_pRowAllocator)
	{
		m_pRowAllocator->Release();
	}
	if (m_pBufAllocator)
	{
		m_pBufAllocator->Release();
	}
}

/****************************************************************************
Desc: This routine links a notify request into a row's notification list and
		then waits to be notified that the event has occurred.
		NOTE: This routine assumes that the row cache mutex is locked and that
		it is supposed to unlock it.  It will relock the mutex on its way out.
****************************************************************************/
RCODE F_RowCacheMgr::waitNotify(
	F_Db *	pDb,
	F_Row **	ppRow)
{
	return( flmWaitNotifyReq( gv_SFlmSysData.hRowCacheMutex, 
		pDb->m_hWaitSem, &((*ppRow)->m_pNotifyList), ppRow));
}

/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read to complete.
		NOTE:  This routine assumes that the row cache mutex is already locked.
****************************************************************************/
void F_RowCacheMgr::notifyWaiters(
	FNOTIFY *	pNotify,
	F_Row *		pUseRow,
	RCODE			NotifyRc)
{
	while (pNotify)
	{
		F_SEM	hSem;

		*(pNotify->pRc) = NotifyRc;
		if (RC_OK( NotifyRc))
		{
			*((F_Row **)pNotify->pvUserData) = pUseRow;
			pUseRow->incrRowUseCount();
		}
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
}

/****************************************************************************
Desc:	Allocate an F_Row object.
****************************************************************************/
RCODE F_RowCacheMgr::allocRow(
	F_Row **	ppRow,
	FLMBOOL	bMutexLocked)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bUnlockMutex = FALSE;
	
	if( !bMutexLocked)
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		bUnlockMutex = TRUE;
	}

	if ((*ppRow = new F_Row) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	
	// Increment statistics.
	
	m_Usage.uiCount++;
	m_Usage.uiByteCount += (*ppRow)->memSize();
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
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Cleanup old rows in cache that are no longer needed by any
		transaction.  This routine assumes that the row cache mutex
		has been locked.
****************************************************************************/
void F_RowCacheMgr::cleanupOldCache( void)
{
	F_Row *		pCurRow;
	F_Row *		pNextRow;

	pCurRow = m_pOldList;

	// Stay in the loop until we have freed all old rows, or
	// we have run through the entire list.

	while( pCurRow)
	{
		flmAssert( pCurRow->m_ui64HighTransId != FLM_MAX_UINT64); 
		
		// Save the pointer to the next entry in the list because
		// we may end up unlinking pCurRow below, in which case we would
		// have lost the next row.

		pNextRow = pCurRow->m_pNextInOldList;
		if (!pCurRow->rowInUse() &&
			 !pCurRow->readingInRow() &&
			 (!pCurRow->rowLinkedToDatabase() ||
			  !pCurRow->m_pDatabase->neededByReadTrans( 
			  	pCurRow->m_ui64LowTransId, pCurRow->m_ui64HighTransId)))
		{
			pCurRow->freeRow();
		}
		pCurRow = pNextRow;
	}
}

/****************************************************************************
Desc:	Cleanup rows that have been purged.  This routine assumes that the
		row cache mutex has been locked.
****************************************************************************/
void F_RowCacheMgr::cleanupPurgedCache( void)
{
	F_Row *		pCurRow;
	F_Row *		pNextRow;

	pCurRow = m_pPurgeList;

	// Stay in the loop until we have freed all purged rows, or
	// we have run through the entire list.

	while( pCurRow)
	{
		// Save the pointer to the next entry in the list because
		// we may end up unlinking pCurRow below, in which case we would
		// have lost the next row.

		pNextRow = (F_Row *)pCurRow->m_pNextInGlobal;
		flmAssert( pCurRow->rowPurged());
		
		if (!pCurRow->rowInUse())
		{
			pCurRow->freePurged();
		}
		pCurRow = pNextRow;
	}
}

/****************************************************************************
Desc:	Reduce row cache to below the cache limit.  NOTE: This routine assumes
		that the row cache mutex is locked upon entering the routine, but
		it may unlock and re-lock the mutex.
****************************************************************************/
void F_RowCacheMgr::reduceCache( void)
{
	F_Row *		pTmpRow;
	F_Row *		pPrevRow;
	F_Row *		pNextRow;
	FLMUINT		uiSlabSize;
	FLMUINT		uiByteThreshold;
	FLMUINT		uiSlabThreshold;
	FLMBOOL		bDoingReduce = FALSE;

	// Discard items that are allocated on the heap.  These are large
	// allocations that could not be satisfied by the buffer allocator and have
	// the side effect of causing memory fragmentation.

	pTmpRow = m_pHeapList;
	while( pTmpRow)
	{
		// Need to save the pointer to the next entry in the list because
		// we may end up freeing pTmpRow below.

		pNextRow = pTmpRow->m_pNextInHeapList;

		// See if the item can be freed.

		if( pTmpRow->canBeFreed())
		{
			// NOTE: This call will free the memory pointed to by
			// pTmpRow.  Hence, pTmpRow should NOT be used after
			// this point.

			pTmpRow->freeRow();
		}

		pTmpRow = pNextRow;
	}

	// If cache is not full, we are done.

	if( !gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit() || m_bReduceInProgress) 
	{
		goto Exit;
	}
	
	m_bReduceInProgress = TRUE;
	bDoingReduce = TRUE;

	// Cleanup cache that is no longer needed by anyone

	cleanupOldCache();
	cleanupPurgedCache();
	
	// Determine the cache threshold

	uiSlabThreshold = gv_SFlmSysData.pGlobalCacheMgr->m_uiMaxSlabs >> 1;
	uiSlabSize = gv_SFlmSysData.pGlobalCacheMgr->m_pSlabManager->getSlabSize();
	
	// Are we over the threshold?

	if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold) 
	{
		goto Exit;
	}
	
	// Remove items from cache starting from the LRU

	pTmpRow = (F_Row *)m_MRUList.m_pLRUItem;
	uiByteThreshold = m_Usage.uiByteCount > uiSlabSize
								? m_Usage.uiByteCount - uiSlabSize
								: 0;

	while( pTmpRow)
	{
		// Need to save the pointer to the next entry in the list because
		// we may end up freeing pTmpRow below.

		pPrevRow = (F_Row *)pTmpRow->m_pPrevInGlobal;

		// See if the item can be freed.

		if( pTmpRow->canBeFreed())
		{
			pTmpRow->freeRow();
			
			if( m_Usage.uiByteCount <= uiByteThreshold)
			{
				if( pPrevRow)
				{
					pPrevRow->incrRowUseCount();
				}
				
				gv_SFlmSysData.pRowCacheMgr->defragmentMemory( TRUE);
				
				if( !pPrevRow)
				{
					break;
				}
				
				pPrevRow->decrRowUseCount();
	
				// We're going to quit when we get under 50 percent for row cache
				// or we aren't over the global limit.  Note that this means we
				// may quit reducing before we get under the global limit.  We
				// don't want to get into a situation where we are starving row
				// cache because block cache is over its limit.
				
				if( m_Usage.slabUsage.ui64Slabs <= uiSlabThreshold ||
					!gv_SFlmSysData.pGlobalCacheMgr->cacheOverLimit())					
				{
					goto Exit;
				}

				uiByteThreshold = uiByteThreshold > uiSlabSize 
											? uiByteThreshold - uiSlabSize
											: 0;
			}
		}

		pTmpRow = pPrevRow;
	}

Exit:

	if( bDoingReduce)
	{
		m_bReduceInProgress = FALSE;
	}

	return;
}

/****************************************************************************
Desc:	This routine finds a row in the row cache.  If it cannot
		find the row, it will return the position where the row should
		be inserted.
		NOTE: This routine assumes that the row cache mutex has been locked.
****************************************************************************/
void F_RowCacheMgr::findRow(
	F_Db *		pDb,
	FLMUINT		uiTableNum,
	FLMUINT64	ui64RowId,
	FLMUINT64	ui64VersionNeeded,
	FLMBOOL		bDontPoisonCache,
	FLMUINT *	puiNumLooks,
	F_Row **		ppRow,
	F_Row **		ppNewerRow,
	F_Row **		ppOlderRow)
{
	F_Row *			pRow;
	FLMUINT			uiNumLooks = 0;
	FLMBOOL			bFound;
	F_Row *			pNewerRow;
	F_Row *			pOlderRow;
	F_Database *	pDatabase = pDb->m_pDatabase;

	// Search down the hash bucket for the matching item.

Start_Find:

	// NOTE: Need to always calculate hash bucket because
	// the hash table may have been changed while we
	// were waiting to be notified below - mutex can
	// be unlocked, but it is guaranteed to be locked
	// here.

	pRow = *(rowHash( ui64RowId));
	bFound = FALSE;
	uiNumLooks = 1;
	while (pRow &&
			 (pRow->m_ui64RowId != ui64RowId ||
			  pRow->m_uiTableNum != uiTableNum ||
			  pRow->m_pDatabase != pDatabase))
	{
		if ((pRow = pRow->m_pNextInBucket) != NULL)
		{
			uiNumLooks++;
		}
	}

	// If we found the row, see if we have the right version.

	if (!pRow)
	{
		pNewerRow = pOlderRow = NULL;
	}
	else
	{
		pNewerRow = NULL;
		pOlderRow = pRow;
		for (;;)
		{

			// If this one is being read in, we need to wait on it.

			if (pRow->readingInRow())
			{
				// We need to wait for this record to be read in
				// in case it coalesces with other versions, resulting
				// in a version that satisfies our request.

				m_uiIoWaits++;
				if (RC_BAD( waitNotify( pDb, &pRow)))
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

				pRow->decrRowUseCount();

				if (pRow->rowPurged())
				{
					if (!pRow->rowInUse())
					{
						pRow->freePurged();
					}
				}

				// Start over with the find because the list
				// structure has changed.

				goto Start_Find;
			}

			// See if this record version is the one we need.

			if (ui64VersionNeeded < pRow->m_ui64LowTransId)
			{
				pNewerRow = pRow;
				if ((pOlderRow = pRow = pRow->m_pOlderVersion) == NULL)
				{
					break;
				}
				uiNumLooks++;
			}
			else if (ui64VersionNeeded <= pRow->m_ui64HighTransId)
			{

				// Make this the MRU record.

				if (puiNumLooks)
				{
					if (bDontPoisonCache)
					{
						m_MRUList.stepUpInGlobal( (F_CachedItem *)pRow);
					}
					else if (pRow->m_pPrevInGlobal)
					{
						m_MRUList.unlinkGlobal( (F_CachedItem *)pRow);
						m_MRUList.linkGlobalAsMRU( (F_CachedItem *)pRow);
					}
					m_Usage.uiCacheHits++;
					m_Usage.uiCacheHitLooks += uiNumLooks;
				}
				bFound = TRUE;
				break;
			}
			else
			{
				pOlderRow = pRow;
				pNewerRow = pRow->m_pNewerVersion;

				// Set pRow to NULL as an indicator that we did not
				// find the version we needed.

				pRow = NULL;
				break;
			}
		}
	}

	*ppRow = pRow;

	if( ppOlderRow)
	{
		*ppOlderRow = pOlderRow;
	}

	if( ppNewerRow)
	{
		*ppNewerRow = pNewerRow;
	}

	if (puiNumLooks)
	{
		*puiNumLooks = uiNumLooks;
	}
}

/****************************************************************************
Desc:	This routine links a new row into the global list and
		into the correct place in its hash bucket.  This routine assumes that
		the row cache mutex is already locked.
****************************************************************************/
void F_RowCacheMgr::linkIntoRowCache(
	F_Row *	pNewerRow,
	F_Row *	pOlderRow,
	F_Row *	pRow,
	FLMBOOL	bLinkAsMRU
	)
{
	if( bLinkAsMRU)
	{
		m_MRUList.linkGlobalAsMRU( (F_CachedItem *)pRow);
	}
	else
	{
		m_MRUList.linkGlobalAsLRU( (F_CachedItem *)pRow);
	}

	if (pNewerRow)
	{
		pRow->linkToVerList( pNewerRow, pOlderRow);
	}
	else
	{
		if (pOlderRow)
		{
			pOlderRow->unlinkFromHashBucket();
		}
		pRow->linkToHashBucket();
		pRow->linkToVerList( NULL, pOlderRow);
	}
}

/****************************************************************************
Desc:	This routine links a new row to its F_Database according to whether
		or not it is an update transaction or a read transaction.
		It coalesces out any unnecessary versions. This routine assumes 
		that the row cache mutex is already locked.
****************************************************************************/
void F_Row::linkToDatabase(
	F_Database *		pDatabase,
	F_Db *				pDb,
	FLMUINT64			ui64LowTransId,
	FLMBOOL				bMostCurrent)
{
	F_Row *	pTmpRow;

	m_ui64LowTransId = ui64LowTransId;

	// Before coalescing, link to F_Database.
	// The following test determines if the row is an
	// uncommitted version generated by the update transaction.
	// If so, we mark it as such, and link it at the head of the
	// F_Database list - so we can get rid of it quickly if we abort
	// the transaction.

	if (pDb->getTransType() == SFLM_UPDATE_TRANS)
	{

		// If we are in an update transaction, there better not
		// be any newer versions in the list and the high
		// transaction ID returned better be FLM_MAX_UINT64.

		flmAssert( m_pNewerVersion == NULL);
		setTransID( FLM_MAX_UINT64);

		// If the low transaction ID is the same as the transaction,
		// we may have modified this row during the transaction.
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
		// back, but that is possible even if the row is
		// not the most current version.  Besides that, it is
		// possible that in the mean time one or more update
		// transactions have come along and created one or
		// more newer versions of the row.

		if (bMostCurrent)
		{
			// This may be showing up as most current simply because we have
			// a newer row that was dirty - meaning it would not have been
			// written to block cache yet - so our read operation would have
			// read the "most current" version of the block that contains
			// this row - but it isn't really the most current version of
			// the row.

			if (m_pNewerVersion && !m_pNewerVersion->readingInRow())
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
		if (m_pNewerVersion && !m_pNewerVersion->readingInRow())
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
	// The read operation that read the row may have
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
		if ((pTmpRow = m_pOlderVersion) == NULL)
		{
			break;
		}

		// Stop if we encounter one that is being read in.

		if (pTmpRow->readingInRow())
		{
			break;
		}

		// If there is no overlap between these two, there is
		// nothing more to coalesce.

		if (m_ui64LowTransId > pTmpRow->m_ui64HighTransId)
		{
			break;
		}

		if (m_ui64HighTransId <= pTmpRow->m_ui64HighTransId)
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
		else if (m_ui64LowTransId >= pTmpRow->m_ui64LowTransId)
		{
			m_ui64LowTransId = pTmpRow->m_ui64LowTransId;
			pTmpRow->freeCache(
						(FLMBOOL)((pTmpRow->rowInUse() ||
									  pTmpRow->readingInRow())
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
			//	found a version of the row that is older than pOlder.

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
RCODE F_Row::resizeDataBuffer(
	FLMUINT		uiSize,
	FLMBOOL		bMutexAlreadyLocked)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiOldSize;
	FLMUINT		uiNewSize;
	FLMUINT		uiDataBufSize = calcDataBufSize( uiSize);
	FLMBYTE *	pucActualAlloc;
	FLMBOOL		bHeapAlloc = FALSE;
	void *		pvThis = this;
	FLMBOOL		bLockedMutex = FALSE;

	flmAssert( !m_uiColumnDataBufSize || m_pucColumnData);

	if( uiDataBufSize == calcDataBufSize(m_uiColumnDataBufSize))
	{
		goto Exit;
	}

	if( !bMutexAlreadyLocked)
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		bLockedMutex = TRUE;
	}

	uiOldSize = memSize();
	
	if (!m_pucColumnData)
	{
		pucActualAlloc = NULL;
		if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->allocBuf(
			&gv_SFlmSysData.pRowCacheMgr->m_columnDataRelocator,
			uiDataBufSize, &pvThis, sizeof( void *), 
			&pucActualAlloc, &bHeapAlloc)))
		{
			goto Exit;
		}
	}
	else
	{
		pucActualAlloc = getActualPointer( m_pucColumnData);
		if( RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->reallocBuf(
			&gv_SFlmSysData.pRowCacheMgr->m_columnDataRelocator,
			calcDataBufSize(m_uiColumnDataBufSize),
			uiDataBufSize, &pvThis, sizeof( void *),
			&pucActualAlloc, &bHeapAlloc)))
		{
			goto Exit;
		}
	}
	
	flmAssert( *((F_Row **)pucActualAlloc) == this);
	setRowAndDataPtr( pucActualAlloc);

	m_uiColumnDataBufSize = uiSize;
	uiNewSize = memSize();
	
	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes >= uiOldSize);
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes -= uiOldSize;
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes += uiNewSize;
	}

	flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount >= uiOldSize);
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount -= uiOldSize;
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount += uiNewSize;
	
	if( bHeapAlloc)
	{
		linkToHeapList();
	}
	else if( m_uiFlags & FROW_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

Exit:

	if( bLockedMutex)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	flmAssert( !m_uiColumnDataBufSize || m_pucColumnData);
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::resizeColumnList(
	FLMUINT	uiColumnCount,
	FLMBOOL	bMutexAlreadyLocked)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiOldSize;
	FLMBYTE *	pucActualAlloc;
	FLMBOOL		bHeapAlloc = FALSE;
	void *		pvThis = this;
	
	if( uiColumnCount == m_uiNumColumns)
	{
		goto Exit;
	}

	if( !bMutexAlreadyLocked)
	{
		flmAssert( rowInUse());
	}

	uiOldSize = memSize();
	
	if( !uiColumnCount)
	{
		// The only thing we better be doing if we pass in a zero, is
		// reducing the number of columns.  Hence, the current
		// column count better be non-zero.
		
		flmAssert( m_uiNumColumns);
		pucActualAlloc = getActualPointer( m_pColumns);
		gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->freeBuf(
							calcColumnListBufSize( m_uiNumColumns),
							&pucActualAlloc);
	}
	else
	{
		if( !m_uiNumColumns)
		{
			pucActualAlloc = NULL;
			rc = gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->allocBuf(
								&gv_SFlmSysData.pRowCacheMgr->m_columnListRelocator,
								calcColumnListBufSize( uiColumnCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		else
		{
			pucActualAlloc = getActualPointer( m_pColumns);
			rc = gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->reallocBuf(
								&gv_SFlmSysData.pRowCacheMgr->m_columnListRelocator,
								calcColumnListBufSize( m_uiNumColumns),
								calcColumnListBufSize( uiColumnCount),
								&pvThis, sizeof( void *), &pucActualAlloc, &bHeapAlloc);
		}
		
		flmAssert( *((F_Row **)pucActualAlloc) == this);
	}
	
	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	}
	
	if (RC_OK( rc))
	{
		FLMUINT	uiNewSize;
		
		m_uiNumColumns = uiColumnCount;
		if (m_uiNumColumns)
		{
			setColumnListPtr( pucActualAlloc);
		}
		else
		{
			m_pColumns = NULL;
		}

		uiNewSize = memSize();

		if (m_ui64HighTransId != FLM_MAX_UINT64)
		{
			flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes >= uiOldSize);
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes -= uiOldSize;
			gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes += uiNewSize;
		}

		flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount >= uiOldSize);
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount -= uiOldSize;
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount += uiNewSize;
		
		if( bHeapAlloc)
		{
			linkToHeapList();
		}
		else if( m_uiFlags & FROW_HEAP_ALLOC)
		{
			unlinkFromHeapList();
		}
	}
	
	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::getCachedBTree(
	FLMUINT		uiTableNum,
	F_Btree **	ppBTree)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable = m_pDict->getTable( uiTableNum);

	if (m_pCachedBTree)
	{
		flmAssert( m_pCachedBTree->getRefCount() == 1);
		m_pCachedBTree->btClose();
	}
	else
	{
		// Reserve a B-Tree from the pool

		if( RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &m_pCachedBTree)))
		{
			goto Exit;
		}
	}

	// Set up the btree object

	if( RC_BAD( rc = m_pCachedBTree->btOpen( this,
		&pTable->lfInfo, FALSE, TRUE)))
	{
		goto Exit;
	}

	m_pCachedBTree->AddRef();
	*ppBTree = m_pCachedBTree;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
F_BTreeIStreamPool::~F_BTreeIStreamPool()
{
	F_BTreeIStream *	pTmpBTreeIStream;

	while ((pTmpBTreeIStream = m_pFirstBTreeIStream) != NULL)
	{
		m_pFirstBTreeIStream = m_pFirstBTreeIStream->m_pNextInPool;
		pTmpBTreeIStream->m_refCnt = 0;
		pTmpBTreeIStream->m_pNextInPool = NULL;
		delete pTmpBTreeIStream;
	}

	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStreamPool::setup( void)
{
	RCODE		rc = NE_SFLM_OK;

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStreamPool::allocBTreeIStream(
	F_BTreeIStream **	ppBTreeIStream)
{
	RCODE	rc = NE_SFLM_OK;

	if( m_pFirstBTreeIStream)
	{
		f_mutexLock( m_hMutex);
		if( !m_pFirstBTreeIStream)
		{
			f_mutexUnlock( m_hMutex);
		}
		else
		{
			f_resetStackInfo( m_pFirstBTreeIStream);
			*ppBTreeIStream = m_pFirstBTreeIStream;
			m_pFirstBTreeIStream = m_pFirstBTreeIStream->m_pNextInPool;
			(*ppBTreeIStream)->m_pNextInPool = NULL;
	
			f_mutexUnlock( m_hMutex);
			goto Exit;
		}
	}

	if( (*ppBTreeIStream = f_new F_BTreeIStream) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_BTreeIStreamPool::insertBTreeIStream(
	F_BTreeIStream *	pBTreeIStream)
{
	flmAssert( pBTreeIStream->m_refCnt == 1);

	pBTreeIStream->reset();
	f_mutexLock( m_hMutex);
	pBTreeIStream->m_pNextInPool = m_pFirstBTreeIStream;
	m_pFirstBTreeIStream = pBTreeIStream;
	f_mutexUnlock( m_hMutex);
}

/*****************************************************************************
Desc:
******************************************************************************/
FLMINT F_BTreeIStream::Release( void)
{
	FLMATOMIC	refCnt = --m_refCnt;
	
	if (m_refCnt == 0)
	{
		closeStream();
		if( gv_SFlmSysData.pBTreeIStreamPool)
		{
			m_refCnt = 1;
			gv_SFlmSysData.pBTreeIStreamPool->insertBTreeIStream( this);
			return( 0);
		}
		else
		{
			delete this;
		}
	}
	
	return( refCnt);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::openStream(
	F_Db *		pDb,
	FLMUINT		uiTableNum,
	FLMUINT64	ui64RowId,
	FLMUINT32	ui32BlkAddr,
	FLMUINT		uiOffsetIndex)
{
	RCODE			rc = NE_SFLM_OK;
	F_Dict *		pDict = pDb->m_pDict;
	F_Btree *	pBTree = NULL;
	F_TABLE *	pTable = pDict->getTable( uiTableNum);
	
	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pBTree)))
	{
		goto Exit;
	}

	// Set up the btree object

	if (RC_BAD( rc = pBTree->btOpen( pDb, &pTable->lfInfo, FALSE, TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = openStream( pDb, pBTree, FLM_EXACT, uiTableNum, ui64RowId,
		ui32BlkAddr, uiOffsetIndex)))
	{
		goto Exit;
	}

	pBTree = NULL;
	m_bReleaseBTree = TRUE;

Exit:

	if (pBTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pBTree);
	}

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::openStream(
	F_Db *			pDb,
	F_Btree *		pBTree,
	FLMUINT			uiFlags,
	FLMUINT			uiTableNum,
	FLMUINT64 		ui64RowId,
	FLMUINT32		ui32BlkAddr,
	FLMUINT			uiOffsetIndex)
{
	RCODE		rc = NE_SFLM_OK;

	flmAssert( !m_pBTree);

	m_pDb = pDb;
	m_uiTableNum = uiTableNum;
	m_pBTree = pBTree;

	// Save the key and key length

	m_uiKeyLength = sizeof( m_ucKey);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64RowId, &m_uiKeyLength,
									m_ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	m_ui32BlkAddr = ui32BlkAddr;
	m_uiOffsetIndex = uiOffsetIndex;

	if( RC_BAD( rc = m_pBTree->btLocateEntry(
		m_ucKey, sizeof( m_ucKey), &m_uiKeyLength, uiFlags,
		NULL, &m_uiStreamSize, &m_ui32BlkAddr, &m_uiOffsetIndex)))
	{
		if( rc == NE_SFLM_NOT_FOUND)
		{
			rc = RC_SET( NE_SFLM_ROW_NOT_FOUND);
		}
		goto Exit;
	}

	if( uiFlags == FLM_EXACT)
	{
		m_ui64RowId = ui64RowId;
	}
	else
	{
		if( RC_BAD( rc = flmCollation2Number( m_uiKeyLength, m_ucKey,
			&m_ui64RowId, NULL, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::positionTo(
	FLMUINT64		ui64Position)
{
	RCODE				rc = NE_SFLM_OK;

	if( ui64Position >= m_uiBufferStartOffset &&
		 ui64Position <= m_uiBufferStartOffset + m_uiBufferBytes)
	{
		m_uiBufferOffset = (FLMUINT)(ui64Position - m_uiBufferStartOffset);
	}
	else
	{
		if( RC_BAD( rc = m_pBTree->btSetReadPosition( m_ucKey,
								m_uiKeyLength, (FLMUINT)ui64Position)))
		{
			goto Exit;
		}

		m_uiBufferStartOffset = (FLMUINT)ui64Position;
		m_uiBufferOffset = 0;
		m_uiBufferBytes = 0;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBYTE *		pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT			uiBufBytesAvail;
	FLMUINT			uiTmp;
	FLMUINT			uiOffset = 0;

	flmAssert( m_pBTree);
	
	while( uiBytesToRead)
	{
		if( (uiBufBytesAvail = m_uiBufferBytes - m_uiBufferOffset) != 0)
		{
			uiTmp = f_min( uiBufBytesAvail, uiBytesToRead);
			if( pucBuffer)
			{
				f_memcpy( &pucBuffer[ uiOffset], 
					&m_pucBuffer[ m_uiBufferOffset], uiTmp);
			}
			m_uiBufferOffset += uiTmp;
			uiOffset += uiTmp;
			if( (uiBytesToRead -= uiTmp) == 0)
			{
				break;
			}
		}
		else
		{
			m_uiBufferStartOffset += m_uiBufferBytes;
			m_uiBufferOffset = 0;

			if( RC_BAD( rc = m_pBTree->btGetEntry(
				m_ucKey, m_uiKeyLength, m_uiKeyLength,
				m_pucBuffer, m_uiBufferSize, &m_uiBufferBytes)))
			{
				if( rc == NE_SFLM_EOF_HIT)
				{
					if( !m_uiBufferBytes)
					{
						goto Exit;
					}
					rc = NE_SFLM_OK;
					continue;
				}
				goto Exit;
			}
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiOffset;
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::flushRow(
	F_Db *				pDb,
	F_Btree *			pBTree)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	F_ENCDEF *			pEncDef;
	F_COLUMN_ITEM *	pColItem;
	FLMUINT				uiIVLen;
	FLMUINT				uiEncLen;
	FLMUINT				uiEncOutputLen;
	FLMUINT				uiColumnDataLen;
	FLMUINT				uiColumnLenLen;
	FLMBYTE *			pucOutputColLengths;
	FLMBYTE *			pucOutputColData;
	FLMBYTE *			pucColumnData;
	FLMBYTE *			pucIV;
	FLMBYTE				ucKeyBuf[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				uiKeyLen;
	F_DynaBuf			dynaBuf( m_pDatabase->m_pucUpdBuffer, m_pDatabase->m_uiUpdBufferSize);
	FLMUINT32			ui32BlkAddr;
	FLMUINT				uiOffsetIndex;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMUINT				uiLoop;

	// Row should be dirty

	flmAssert( rowIsDirty());

	// Transaction IDs should match

	if (getLowTransId() != pDb->getTransID())
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// First pass through columns is to calculate the total length that will
	// be needed for column lengths and data.
	
	uiColumnDataLen = 0;
	uiColumnLenLen = 0;
	pTable = pDb->m_pDict->getTable( m_uiTableNum);
	for (uiLoop = 0, pColItem = m_pColumns, pColumn = pTable->pColumns;
		  uiLoop < m_uiNumColumns;
		  uiLoop++, pColItem++, pColumn++)
	{
		uiColumnLenLen += f_getSENByteCount( pColItem->uiDataLen);
		if (!pColumn->uiEncDefNum || !pColItem->uiDataLen)
		{
			uiColumnDataLen += pColItem->uiDataLen;
		}
		else
		{
			pEncDef = pDb->m_pDict->getEncDef( pColumn->uiEncDefNum);
			uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( uiIVLen == 8 || uiIVLen == 16);
			
			uiEncLen = getEncLen( pColItem->uiDataLen);
			uiColumnDataLen += (f_getSENByteCount( uiEncLen) + uiIVLen + uiEncLen);
		}
	}
	
	// Allocate the space needed to output all column data.
		
	if (RC_BAD( rc = dynaBuf.allocSpace( uiColumnLenLen + uiColumnDataLen,
										(void **)&pucOutputColLengths)))
	{
		goto Exit;
	}
	pucOutputColData = pucOutputColLengths + uiColumnLenLen;
	
	// Second pass is to output both column lengths and column data.
	
	for (uiLoop = 0, pColItem = m_pColumns, pColumn = pTable->pColumns;
		  uiLoop < m_uiNumColumns;
		  uiLoop++, pColItem++, pColumn++)
	{
		f_encodeSEN( pColItem->uiDataLen, &pucOutputColLengths);
		if (!pColItem->uiDataLen)
		{
			continue;
		}
		pucColumnData = getColumnDataPtr( uiLoop + 1);
		if (!pColumn->uiEncDefNum)
		{
			f_memcpy( pucOutputColData, pucColumnData, pColItem->uiDataLen);
		}
		else
		{
			pEncDef = pDb->m_pDict->getEncDef( pColumn->uiEncDefNum);
			
			uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( uiIVLen == 8 || uiIVLen == 16);
			
			// Output the encryption data length
			
			uiEncLen = getEncLen( pColItem->uiDataLen);
			f_encodeSEN( uiEncLen, &pucOutputColData);
			
			// Output the IV
			
			pucIV = pucOutputColData;
			if( RC_BAD( rc = pEncDef->pCcs->generateIV( uiIVLen, pucOutputColData)))
			{
				goto Exit;
			}
			pucOutputColData += uiIVLen;
			
			// Output the data and encrypt it.
			
			f_memcpy( pucOutputColData, pucColumnData, pColItem->uiDataLen);
			if (RC_BAD( rc = pDb->encryptData( pColumn->uiEncDefNum, pucIV, 
				pucOutputColData, uiEncLen, pColItem->uiDataLen, &uiEncOutputLen)))
			{
				goto Exit;
			}
			flmAssert( uiEncOutputLen == uiEncLen);
			
			pucOutputColData += uiEncLen;
		}
	}
	
	uiKeyLen = sizeof( ucKeyBuf);
	if( RC_BAD( rc = flmNumber64ToStorage( m_ui64RowId, &uiKeyLen, ucKeyBuf,
									FALSE, TRUE)))
	{
		goto Exit;
	}
	
	ui32BlkAddr = m_ui32BlkAddr;
	uiOffsetIndex = m_uiOffsetIndex;
	
	bMustAbortOnError = TRUE;
	if (rowIsNew())
	{
		if (RC_BAD( rc = pBTree->btInsertEntry(
						ucKeyBuf, uiKeyLen,
						dynaBuf.getBufferPtr(), dynaBuf.getDataLength(),
						TRUE, TRUE, &ui32BlkAddr, &uiOffsetIndex)))
		{
			if( rc == NE_SFLM_NOT_UNIQUE)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_EXISTS);
			}
			
			goto Exit;
		}
	}
	else
	{
		
		// Replace the row on disk.

		if( RC_BAD( rc = pBTree->btReplaceEntry(
						ucKeyBuf, uiKeyLen,
						dynaBuf.getBufferPtr(), dynaBuf.getDataLength(),
						TRUE, TRUE, TRUE, &ui32BlkAddr,
						&uiOffsetIndex)))
		{
			goto Exit;
		}
	}
	
	m_ui32BlkAddr = ui32BlkAddr;
	m_uiOffsetIndex = uiOffsetIndex;
	
	// Clear the dirty flag and the new flag.

	unsetRowDirtyAndNew( pDb);

Exit:

	if (RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::flushRow(
	F_Db *	pDb)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBOOL		bMutexLocked = FALSE;
	F_Btree *	pBTree = NULL;

	if (!rowIsDirty())
	{
		goto Exit;
	}

	// Get a B-Tree object

	if( RC_BAD( rc = pDb->getCachedBTree( m_uiTableNum, &pBTree)))
	{
		goto Exit;
	}
	
	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;

	incrRowUseCount();
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = FALSE;

	rc = flushRow( pDb, pBTree);

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;

	decrRowUseCount();

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	if (pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::flushDirtyRows( void)
{
	RCODE			rc = NE_SFLM_OK;
	F_Row *		pRow;
	F_Btree *	pBtree = NULL;
	FLMUINT		uiTableNum = 0;
	
	if( !m_uiDirtyRowCount)
	{
		goto Exit;
	}

	// All of the dirty nodes should be at the front of the list.

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	while ((pRow = m_pDatabase->m_pFirstRow) != NULL)
	{
		if( !pRow->rowIsDirty())
		{
			break;
		}
		
		// Flushing the node should remove it from the front of the list.
		// Need to increment the use count on the node to prevent it from
		// being moved while we are flushing it to disk.
		
		pRow->incrRowUseCount();
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);

		if (uiTableNum != pRow->m_uiTableNum)
		{
			if( pBtree)
			{
				pBtree->Release();
			}

			uiTableNum = pRow->m_uiTableNum;
			if( RC_BAD( rc = getCachedBTree( uiTableNum, &pBtree)))
			{
				goto Exit;
			}
		}

		rc = pRow->flushRow( this, pBtree);

		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		pRow->decrRowUseCount();

		if( rc == NE_SFLM_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		}

		if( RC_BAD( rc))
		{
			f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
			goto Exit;
		}
	}

	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	flmAssert( !m_uiDirtyRowCount);

Exit:

	if (pBtree)
	{
		pBtree->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::readRow(
	F_Db *			pDb,
	FLMUINT			uiTableNum,
	FLMUINT64,		// ui64RowId,
	IF_IStream *	pIStream,
	FLMUINT			uiRowDataLength)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiBytesRead;
	F_TABLE *			pTable = pDb->m_pDict->getTable( uiTableNum);
	F_COLUMN *			pColumn;
	F_COLUMN_ITEM *	pColItem;
	FLMUINT				uiLoop;
	FLMBOOL				bMutexLocked = FALSE;
	FLMBYTE *			pucColumnData;
	FLMBYTE *			pucColumnDataEnd;
	FLMUINT				uiEncDataLen = 0;
	F_ENCDEF *			pEncDef;
	FLMUINT				uiIVLen;
	FLMUINT				uiTotalColumnDataLen;
	FLMBOOL				bMustMoveOneColumnAtATime;
	
	// Allocate a buffer for the column data and a column list.
	
	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;
	if (RC_BAD( rc = resizeDataBuffer( uiRowDataLength, TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = resizeColumnList( pTable->uiNumColumns, TRUE)))
	{
		goto Exit;
	}
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = FALSE;
	
	// Read the column data into memory.
	
	if (RC_BAD( rc = pIStream->read( m_pucColumnData, uiRowDataLength,
										&uiBytesRead)))
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		}
		goto Exit;
	}
	else if (uiBytesRead < uiRowDataLength)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_DATA_ERROR);
		goto Exit;
	}
	
	// Parse through all of the columns - some may be missing.
	// All of the lengths precede the actual data.  Sum them up to determine
	// how big of a buffer we actually need in the end.
	
	uiTotalColumnDataLen = 0;
	pucColumnData = m_pucColumnData;
	pucColumnDataEnd = m_pucColumnData + uiRowDataLength;
	bMustMoveOneColumnAtATime = FALSE;
	for (uiLoop = 0, pColumn = pTable->pColumns, pColItem = m_pColumns;
		  uiLoop < pTable->uiNumColumns;
		  uiLoop++, pColumn++, pColItem++)
	{
		// Get the data length for the column
		
		if( RC_BAD( rc = f_decodeSEN( (const FLMBYTE **)&pucColumnData,
												(const FLMBYTE *)pucColumnDataEnd,
												&pColItem->uiDataLen)))
		{
			goto Exit;
		}
		if (pColumn->uiEncDefNum)
		{
			bMustMoveOneColumnAtATime = TRUE;
		}
		if (pColItem->uiDataLen > sizeof( FLMUINT))
		{
			pColItem->uiDataOffset = uiTotalColumnDataLen;
			uiTotalColumnDataLen += pColItem->uiDataLen;
		}
		else if (pColItem->uiDataLen)
		{
			bMustMoveOneColumnAtATime = TRUE;
		}
	}
	
	// Now move all of the data into place.  If there were no encrypted
	// columns and no columns whose size was <= sizeof( FLMUINT) we can do
	// it all with a single memmove.
	
	if (!bMustMoveOneColumnAtATime)
	{
		flmAssert( pucColumnData + uiTotalColumnDataLen == pucColumnDataEnd);
		f_memmove( m_pucColumnData, pucColumnData, uiTotalColumnDataLen);
	}
	else
	{
		
		// Must move data a column at a time.
		
		for (uiLoop = 0, pColumn = pTable->pColumns, pColItem = m_pColumns;
			  uiLoop < pTable->uiNumColumns;
			  uiLoop++, pColumn++, pColItem++)
		{
			
			// If there is no data for the column, skip it.
			
			if (!pColItem->uiDataLen)
			{
				continue;
				
			}
			
			// See if column is encrypted.  If so, decrypt it in place before
			// moving it to where it needs to go.

			if (pColumn->uiEncDefNum)
			{
				FLMBYTE *	pucIV;
				
				pEncDef = pDb->m_pDict->getEncDef( pColumn->uiEncDefNum);
				flmAssert( pEncDef);
				
				// Get the encrypted data length and the IV
				
				if (RC_BAD( rc = f_decodeSEN( (const FLMBYTE **)&pucColumnData,
									(const FLMBYTE *)pucColumnDataEnd, &uiEncDataLen)))
				{
					goto Exit;
				}
				
				uiIVLen = pEncDef->pCcs->getIVLen();
				flmAssert( uiIVLen == 8 || uiIVLen == 16);
				
				pucIV = pucColumnData;
				pucColumnData += uiIVLen;
				
				// Decrypt the data in place, then move it to where it needs to go.
				// We do NOT decrypt it directly to the destination, because the
				// destination may overlap with the buffer we are decrypting.
				// The decryptData routine does not deal with overlapping buffers
				// except the case where the source and destination buffers are
				// exactly the same (at least that is the assumption).
				
				if (RC_BAD( rc = pDb->decryptData( pColumn->uiEncDefNum, pucIV,
					pucColumnData, uiEncDataLen, pucColumnData, uiEncDataLen)))
				{
					goto Exit;
				}
			}
			if (pColItem->uiDataLen <= sizeof( FLMUINT))
			{
				f_memcpy( &pColItem->uiDataOffset, pucColumnData,
								pColItem->uiDataLen);
			}
			else
			{
				f_memmove( m_pucColumnData + pColItem->uiDataOffset,
							pucColumnData, pColItem->uiDataLen);
			}
				
			// Move pointer past the encrypted data

			if (pColumn->uiEncDefNum)
			{
				pucColumnData += uiEncDataLen;
			}
			else
			{
				pucColumnData += pColItem->uiDataLen;
			}
		}
	}
	
	// We better have ended precisely on the end of the buffer.
	
	flmAssert( pucColumnData == pucColumnDataEnd);	

	// Resize the buffer down.

	if (RC_BAD( rc = resizeDataBuffer( uiTotalColumnDataLen, bMutexLocked)))
	{
		goto Exit;
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine retrieves a row from disk.
****************************************************************************/
RCODE F_RowCacheMgr::readRowFromDisk(
	F_Db *			pDb,
	FLMUINT			uiTableNum,
	FLMUINT64		ui64RowId,
	F_Row *			pRow,
	FLMUINT64 *		pui64LowTransId,
	FLMBOOL *		pbMostCurrent)
{
	RCODE					rc = NE_SFLM_OK;
	F_Btree *			pBTree = NULL;
	FLMBOOL				bCloseIStream = FALSE;
	F_BTreeIStream		btreeIStream;

	if( RC_BAD( rc = pDb->getCachedBTree( uiTableNum, &pBTree)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = btreeIStream.openStream( pDb, pBTree, FLM_EXACT,
								uiTableNum, ui64RowId, 0, 0)))
	{
		goto Exit;
	}
	bCloseIStream = TRUE;
	
	// Read the row from the B-Tree

	if (RC_BAD( rc = pRow->readRow( pDb, uiTableNum, 
		ui64RowId, &btreeIStream, (FLMUINT)btreeIStream.remainingSize())))
	{
		if( rc == NE_SFLM_EOF_HIT)
		{
			rc = RC_SET( NE_SFLM_DATA_ERROR);
		}

		goto Exit;
	}
	
	pRow->m_uiOffsetIndex = btreeIStream.getOffsetIndex();
	pRow->m_ui32BlkAddr = btreeIStream.getBlkAddr();

	pBTree->btGetTransInfo( pui64LowTransId, pbMostCurrent);

Exit:

	if (bCloseIStream)
	{
		btreeIStream.closeStream();
	}

	if (pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine retrieves a row from the row cache.
****************************************************************************/
RCODE F_RowCacheMgr::retrieveRow(
	F_Db *		pDb,
	FLMUINT		uiTableNum,
	FLMUINT64	ui64RowId,
	F_Row **		ppRow)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bMutexLocked = FALSE;
	F_Database *	pDatabase = pDb->m_pDatabase;
	F_Row *			pRow;
	F_Row *			pNewerRow;
	F_Row *			pOlderRow;
	FLMUINT64		ui64LowTransId;
	FLMBOOL			bMostCurrent;
	FLMUINT64		ui64CurrTransId;
	FNOTIFY *		pNotify;
	FLMUINT			uiNumLooks;
	FLMBOOL			bDontPoisonCache = pDb->m_uiFlags & FDB_DONT_POISON_CACHE
													? TRUE 
													: FALSE;

	if (RC_BAD( rc = pDb->checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Get the current transaction ID

	flmAssert( pDb->m_eTransType != SFLM_NO_TRANS);
	ui64CurrTransId = pDb->getTransID();
	flmAssert( ui64RowId != 0);

	// Lock the row cache mutex

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;

	// Reset the DB's inactive time

	pDb->m_uiInactiveTime = 0;

Start_Find:

	findRow( pDb, uiTableNum, ui64RowId,
						ui64CurrTransId, bDontPoisonCache, 
						&uiNumLooks, &pRow,
						&pNewerRow, &pOlderRow);

	if (pRow)
	{
		// Have pRow point to the row we found
		goto Exit1;
	}

	// Did not find the row, fetch from disk
	// Increment the number of faults only if we retrieve the record from disk.

	m_Usage.uiCacheFaults++;
	m_Usage.uiCacheFaultLooks += uiNumLooks;

	// Create a place holder for the object.

	if (RC_BAD( rc = allocRow( &pRow, TRUE)))
	{
		goto Exit;
	}

	pRow->m_ui64RowId = ui64RowId;
	pRow->m_uiTableNum = uiTableNum;

	// Set the F_Database so that other threads looking for this row in
	// cache will find it and wait until the read has completed.  If
	// the F_Database is not set, other threads will attempt their own read,
	// because they won't match a NULL F_Database.  The result of not setting
	// the F_Database is that multiple copies of the same version of a particular
	// row could end up in cache.
	
	pRow->m_pDatabase = pDatabase;

	linkIntoRowCache( pNewerRow, pOlderRow, pRow, !bDontPoisonCache);

	pRow->setReadingIn();
	pRow->incrRowUseCount();
	pRow->m_pNotifyList = NULL;

	// Unlock mutex before reading in from disk.

	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = FALSE;

	// Read row from disk.

	rc = readRowFromDisk( pDb, uiTableNum, ui64RowId, pRow,
						&ui64LowTransId, &bMostCurrent);

	// Relock mutex

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;

	// If read was successful, link the row to its place in
	// the F_Database list and coalesce any versions that overlap
	// this one.

	if (RC_OK( rc))
	{
		pRow->linkToDatabase(
				pDb->m_pDatabase, pDb, ui64LowTransId, bMostCurrent);
	}

	pRow->unsetReadingIn();

	// Notify any threads waiting for the read to complete.

	pNotify = pRow->m_pNotifyList;
	pRow->m_pNotifyList = NULL;
	if (pNotify)
	{
		notifyWaiters( pNotify,
				(F_Row *)((RC_BAD( rc))
							  ? (F_Row *)NULL
							  : pRow), rc);
	}
	pRow->decrRowUseCount();

	// If we did not succeed, free the F_Row structure.

	if (RC_BAD( rc))
	{
		pRow->freeCache( FALSE);
		goto Exit;
	}

	// If this item was purged while we were reading it in,
	// start over with the search.

	if (pRow->rowPurged())
	{
		if (!pRow->rowInUse())
		{
			pRow->freePurged();
		}

		// Start over with the find - this one has
		// been marked for purging.

		goto Start_Find;
	}

Exit1:

	// Have *ppRow point to the row we read in from disk

	if (*ppRow)
	{
		(*ppRow)->decrRowUseCount();
	}
	*ppRow = pRow;
	pRow->incrRowUseCount();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine creates a row into the row cache.  This is ONLY
		called when a new row is being created.
****************************************************************************/
RCODE F_RowCacheMgr::createRow(
	F_Db *		pDb,
	FLMUINT		uiTableNum,
	F_Row **		ppRow)
{
	RCODE				rc = NE_SFLM_OK;
	F_Database *	pDatabase = pDb->m_pDatabase;
	F_Row *			pRow = NULL;
	F_Row *			pNewerRow = NULL;
	F_Row *			pOlderRow = NULL;
	FLMBOOL			bMutexLocked = FALSE;
	F_TABLE *		pTable = pDb->m_pDict->getTable( uiTableNum);
		
	flmAssert( pTable);

	// Lock the row cache mutex

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;
	
	// We are positioned to insert the new row.  For an update, it
	// must always be the newest version.

	flmAssert( !pNewerRow);
	
	// Create a new object.

	if (RC_BAD( rc = allocRow( &pRow, bMutexLocked)))
	{
		goto Exit;
	}

	pRow->m_ui64RowId = pTable->lfInfo.ui64NextRowId;
	pTable->lfInfo.ui64NextRowId++;
	pTable->lfInfo.bNeedToWriteOut = TRUE;
	pRow->m_uiTableNum = uiTableNum;
	pRow->m_uiOffsetIndex = 0;
	pRow->m_ui32BlkAddr = 0;
	
	// NOTE: Not everything is initialized in pRow at this point, but
	// no other thread should be accessing it anyway.  The caller of this
	// function must ensure that all of the necessary items get set before
	// releasing the row.

	// Set the F_Database so that other threads looking for this row in
	// cache will find it and wait until the read has completed.  If
	// the F_Database is not set, other threads will attempt their own read,
	// because they won't match a NULL F_Database.  The result of not setting
	// the F_Database is that multiple copies of the same version of a particular
	// row could end up in cache.
	
	pRow->m_pDatabase = pDatabase;

	linkIntoRowCache( pNewerRow, pOlderRow, pRow, TRUE);

	// Link the row to its place in the F_Database list

	pRow->linkToDatabase( pDatabase, pDb, pDb->m_ui64CurrTransID, TRUE);

	flmAssert( *ppRow == NULL);
	*ppRow = pRow;
	pRow->incrRowUseCount();

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	This routine makes a writeable copy of the row pointed to by F_Row.
****************************************************************************/
RCODE F_RowCacheMgr::_makeWriteCopy(
	F_Db *		pDb,
	F_Row **		ppRow)
{
	RCODE				rc = NE_SFLM_OK;
	F_Database *	pDatabase = pDb->m_pDatabase;
	F_Row *			pNewerRow = NULL;
	F_Row *			pOlderRow = *ppRow;
	FLMBOOL			bMutexLocked = FALSE;
	
	flmAssert( pOlderRow->m_ui64HighTransId == FLM_MAX_UINT64);
	flmAssert( !pOlderRow->m_pNewerVersion);
	
	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	bMutexLocked = TRUE;

	// Create a new object.

	if (RC_BAD( rc = allocRow( &pNewerRow, TRUE)))
	{
		goto Exit;
	}

	// If we found the last committed version, instead of replacing it,
	// we want to change its high transaction ID, and go create a new
	// row to put in cache.

	// Although this routine could be written to not do anything if we
	// are already on the uncommitted version of the row, for performance
	// reasons, we would prefer that they make the check on the outside
	// before calling this routine.
	
	flmAssert( pOlderRow->m_ui64LowTransId < pDb->m_ui64CurrTransID);
	pOlderRow->setTransID( pDb->m_ui64CurrTransID - 1);
	flmAssert( pOlderRow->m_ui64HighTransId >= pOlderRow->m_ui64LowTransId);

	pOlderRow->setUncommitted();
	pOlderRow->setLatestVer();
	pOlderRow->unlinkFromDatabase();
	pOlderRow->linkToDatabaseAtHead( pDatabase);

	pNewerRow->m_pDatabase = pDatabase;
	pNewerRow->m_uiFlags = pOlderRow->m_uiFlags;
	pNewerRow->m_uiOffsetIndex = pOlderRow->m_uiOffsetIndex;
	pNewerRow->m_ui32BlkAddr = pOlderRow->m_ui32BlkAddr;
	
	if( pNewerRow->m_uiFlags & FROW_HEAP_ALLOC)
	{
		pNewerRow->m_uiFlags &= ~FROW_HEAP_ALLOC;
	}
	
	pNewerRow->m_uiTableNum = pOlderRow->m_uiTableNum;
	pNewerRow->m_ui64RowId = pOlderRow->m_ui64RowId;

	if (pNewerRow->m_uiColumnDataBufSize)
	{
		if (RC_BAD( rc = pNewerRow->resizeDataBuffer( 
			pNewerRow->m_uiColumnDataBufSize, TRUE)))
		{
			goto Exit;
		}
	
		f_memcpy( pNewerRow->m_pucColumnData, pOlderRow->m_pucColumnData,
						 pNewerRow->m_uiColumnDataBufSize);
	}
	
	if (pOlderRow->m_uiNumColumns)
	{
		if (RC_BAD( rc = pNewerRow->copyColumnList( pDb, pOlderRow, TRUE)))
		{
			goto Exit;
		}
	}
		
	linkIntoRowCache( NULL, pOlderRow, pNewerRow, TRUE);

	// Link the row to its place in the F_Database list

	pNewerRow->linkToDatabase( pDatabase, pDb, pDb->m_ui64CurrTransID, TRUE);

	// Update the row pointer passed into the routine
	
	if( *ppRow)
	{
		(*ppRow)->decrRowUseCount();
	}
	
	*ppRow = pNewerRow;
	pNewerRow->incrRowUseCount();

	// Set pNewerRow to NULL so it won't get freed at Exit

	pNewerRow = NULL;

Exit:

	// A non-NULL pNewerRow means there was an error of some kind where we will
	// need to free up the cached item.

	if (pNewerRow)
	{
		flmAssert( RC_BAD( rc));
		delete pNewerRow;
	}
	
	if( bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE	F_Row::allocColumnDataSpace(
	F_Db *		pDb,
	FLMUINT		uiColumnNum,
	FLMUINT		uiSizeNeeded,
	FLMBOOL		bMutexAlreadyLocked)
{
	RCODE					rc = NE_SFLM_OK;
	F_COLUMN_ITEM *	pColumnItem;
	F_COLUMN_ITEM *	pTmpColumnItem;
	FLMUINT				uiLoop;
	FLMUINT				uiOldLength;
	FLMUINT				uiOldOffset;
	FLMUINT				uiOldColumnDataBufSize;
	FLMUINT				uiChangeSize;
	FLMBYTE *			pucNextColumnDataStart;
	FLMUINT				uiSizeAfterCurrColumn;
	
	// If we have not yet allocated the column list array, allocate it now.
	
	if (!m_pColumns)
	{
		if (RC_BAD( rc = resizeColumnList(
								pDb->m_pDict->getTable( m_uiTableNum)->uiNumColumns,
								bMutexAlreadyLocked)))
		{
			goto Exit;
		}
	}
	
	pColumnItem = &m_pColumns [uiColumnNum - 1];
	uiOldLength = pColumnItem->uiDataLen;
	uiOldOffset = pColumnItem->uiDataOffset;
	uiOldColumnDataBufSize = m_uiColumnDataBufSize;
	if (uiOldLength < uiSizeNeeded)
	{
		// Only need to increase the buffer size if the size needed is
		// greater than sizeof( FLMUINT).  Otherwise, both the old data
		// and the new data would have fit in the data offset.  Hence,
		// there would be no need to change anything.
		
		if (uiSizeNeeded > sizeof( FLMUINT))
		{
			if (uiOldLength <= sizeof( FLMUINT))
			{
				uiChangeSize = uiSizeNeeded;
			}
			else
			{
				uiChangeSize = uiSizeNeeded - uiOldLength;
			}
			
			// Increase the size of the buffer.
			
			if (RC_BAD( rc = resizeDataBuffer( m_uiColumnDataBufSize + uiChangeSize,
											bMutexAlreadyLocked)))
			{
				goto Exit;
			}
			
			// If the old data was not stored in the buffer, the column's offset
			// can simply be set to the old end of the buffer.
			
			if (uiOldLength <= sizeof( FLMUINT))
			{
				pColumnItem->uiDataOffset = uiOldColumnDataBufSize;
			}
			else
			{
				
				// Move all of the data up for all columns that came after this
				// column.  The following check is to see if this is the last
				// piece of data in the buffer.  If it is the last one, no need
				// to move anything down.
				
				if (uiOldOffset + uiOldLength < uiOldColumnDataBufSize)
				{
					pucNextColumnDataStart = m_pucColumnData + uiOldOffset + uiOldLength;
					uiSizeAfterCurrColumn = uiOldColumnDataBufSize - uiOldOffset - uiOldLength;
											
					f_memmove( pucNextColumnDataStart + uiChangeSize, pucNextColumnDataStart,
								  uiSizeAfterCurrColumn);
								  
					// Increment all of the data offsets after this current
					// column - except for those that are not stored in the buffer.
					
					for (uiLoop = uiColumnNum, pTmpColumnItem = pColumnItem + 1;
						  uiLoop < m_uiNumColumns;
						  uiLoop++, pTmpColumnItem++)
					{
						
						// Only increment the offset if the data is not stored
						// in the offset.
						
						if (pTmpColumnItem->uiDataLen > sizeof( FLMUINT))
						{
							pTmpColumnItem->uiDataOffset += uiChangeSize;
						}
					}
				}
			}
		}
		pColumnItem->uiDataLen = uiSizeNeeded;
	}
	else if (uiOldLength > uiSizeNeeded)
	{
		
		// If the old length is less than or equal to sizeof( FLMUINT), then
		// there is no need to change the size of the buffer, because the
		// new data will fit where the old data did - inside the data offset.
		
		if (uiOldLength > sizeof( FLMUINT))
		{
			if (uiSizeNeeded <= sizeof( FLMUINT))
			{
				uiChangeSize = uiOldLength;
			}
			else
			{
				uiChangeSize = uiOldLength - uiSizeNeeded;
			}
			
			// Scoot down all of the column data in the buffer that comes after
			// the current column data, if any.
			// BUFFER MUST NOT BE RESIZED UNTIL AFTER THIS IS DONE!
			
			if (uiOldOffset + uiOldLength < uiOldColumnDataBufSize)
			{
				pucNextColumnDataStart = m_pucColumnData + uiOldOffset + uiOldLength;
				uiSizeAfterCurrColumn = uiOldColumnDataBufSize - uiOldOffset - uiOldLength;
																
				f_memmove( pucNextColumnDataStart - uiChangeSize, pucNextColumnDataStart,
									uiSizeAfterCurrColumn);
									
				// Decrement all of the data offsets after this current
				// column - except for those that are not stored in the buffer.
				
				for (uiLoop = uiColumnNum, pTmpColumnItem = pColumnItem + 1;
					  uiLoop < m_uiNumColumns;
					  uiLoop++, pTmpColumnItem++)
				{
					
					// Only increment the offset if the data is not stored
					// in the offset.
					
					if (pTmpColumnItem->uiDataLen > sizeof( FLMUINT))
					{
						pTmpColumnItem->uiDataOffset -= uiChangeSize;
					}
				}
			}
			
			// Need to scoot everything down in the column data buffer.
			
			if (RC_BAD( rc = resizeDataBuffer( m_uiColumnDataBufSize - uiChangeSize,
											bMutexAlreadyLocked)))
			{
				goto Exit;
			}
		}
		pColumnItem->uiDataLen = uiSizeNeeded;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::setNumber64(
	F_Db *		pDb,
	FLMUINT		uiColumnNum,
	FLMUINT64	ui64Value,
	FLMBOOL		bNeg)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable = pDb->m_pDict->getTable( m_uiTableNum);
	F_COLUMN *	pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum);
	FLMUINT		uiValLen;
	FLMBYTE		ucNumBuf [40];
	
	switch (pColumn->eDataTyp)
	{
		case SFLM_NUMBER_TYPE:
		{
			if (!bNeg && ui64Value <= 0x7F)
			{
				if (RC_BAD( rc = allocColumnDataSpace( pDb, uiColumnNum, 1, FALSE)))
				{
					goto Exit;
				}
				*(getColumnDataPtr( uiColumnNum)) = (FLMBYTE)ui64Value;
				goto Exit;
			}
			else
			{
				uiValLen = sizeof( ucNumBuf);
				if( RC_BAD( rc = flmNumber64ToStorage( 
					ui64Value, &uiValLen, ucNumBuf, bNeg, FALSE)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = allocColumnDataSpace( pDb, uiColumnNum, uiValLen, FALSE)))
				{
					goto Exit;
				}
				f_memcpy( getColumnDataPtr( uiColumnNum), ucNumBuf, uiValLen);
			}
			
			break;
		}
		
		case SFLM_STRING_TYPE:
		{
			FLMBYTE *	pucSen;
			FLMUINT		uiSenLen;
			
			if( bNeg)
			{
				ucNumBuf[ 1] = '-';
				f_ui64toa( ui64Value, (char *)&ucNumBuf[ 2]);
			}
			else
			{
				f_ui64toa( ui64Value, (char *)&ucNumBuf[ 1]);
			}
			
			uiValLen = f_strlen( (const char *)&ucNumBuf[ 1]);
			pucSen = &ucNumBuf [0];
			uiSenLen = f_encodeSEN( uiValLen, &pucSen, (FLMUINT)0);
			flmAssert( uiSenLen == 1);
			uiValLen += uiSenLen + 1;
			if (RC_BAD( rc = allocColumnDataSpace( pDb, uiColumnNum, uiValLen, FALSE)))
			{
				goto Exit;
			}
			f_memcpy( getColumnDataPtr( uiColumnNum), ucNumBuf, uiValLen);
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_DATA_TYPE);
			goto Exit;
		}
	}
	setRowDirty( pDb, FALSE);
	
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
FSTATIC RCODE getStorageAsNumber(
	const FLMBYTE *	pucData,
	FLMUINT				uiDataLen,
	eDataType			eDataType,
	FLMUINT64 *			pui64Number,
	FLMBOOL *			pbNeg)
{
	RCODE	rc = NE_SFLM_OK;
	
	*pbNeg = FALSE;
	*pui64Number = 0;

	switch (eDataType)
	{
		case SFLM_NUMBER_TYPE:
			if (RC_BAD( rc = flmStorageNumberToNumber( pucData, uiDataLen,
										pui64Number, pbNeg)))
			{
				goto Exit;
			}
			break;

		case SFLM_STRING_TYPE :
		{
			const FLMBYTE *	pucDataEnd = pucData + uiDataLen;
			FLMBYTE				ucChar;
			FLMBOOL				bHex = FALSE;
			FLMUINT				uiIncrAmount = 0;
			FLMBOOL				bFirstChar = TRUE;

			// Skip the character count

			if( RC_BAD( rc = f_decodeSEN64( &pucData, pucDataEnd, NULL)))
			{
				goto Exit;
			}

			while (pucData < pucDataEnd)
			{
				ucChar = *pucData;
				if (ucChar >= '0' && ucChar <= '9')
				{
					uiIncrAmount = (FLMUINT)(ucChar - '0');
				}
				else if (ucChar >= 'A' && ucChar <= 'F')
				{
					if (bHex)
					{
						uiIncrAmount = (FLMUINT)(ucChar - 'A' + 10);
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if (ucChar >= 'a' && ucChar <= 'f')
				{
					if (bHex)
					{
						uiIncrAmount = (FLMUINT)(ucChar - 'a' + 10);
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if (ucChar == 'X' || ucChar == 'x')
				{
					if (*pui64Number == 0 && !bHex)
					{
						bHex = TRUE;
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if (ucChar == '-' && bFirstChar)
				{
					*pbNeg = TRUE;
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_BAD_DIGIT);
					goto Exit;
				}

				if (!bHex)
				{
					if (*pui64Number > (~(FLMUINT64)0) / 10)
					{
						rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_NUM_OVERFLOW);
						goto Exit;
					}
	
					(*pui64Number) *= (FLMUINT64)10;
				}
				else
				{
					if (*pui64Number > (~(FLMUINT64)0) / 16)
					{
						rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_NUM_OVERFLOW);
						goto Exit;
					}
	
					(*pui64Number) *= (FLMUINT64)16;
				}
	
				if (*pui64Number > (~(FLMUINT64)0) - (FLMUINT64)uiIncrAmount)
				{
					rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				(*pui64Number) += (FLMUINT64)uiIncrAmount;
				pucData++;
				bFirstChar = FALSE;
			}

			break;
		}

		default :
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_CONV_BAD_DIGIT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::getNumber64(
	F_Db *					pDb,
	FLMUINT					uiColumnNum,
	FLMUINT64 *				pui64Value,
	FLMBOOL *				pbNeg,
	FLMBOOL *				pbIsNull)
{
	RCODE					rc = NE_SFLM_OK;
	F_COLUMN_ITEM *	pColumnItem;
	
	*pbIsNull = FALSE;
	if ((pColumnItem = getColumn( uiColumnNum)) == NULL)
	{
		*pbIsNull = TRUE;
		goto Exit;
	}
	else
	{
		F_TABLE *	pTable = pDb->m_pDict->getTable( m_uiTableNum);
		F_COLUMN *	pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum);
		
		if (RC_BAD( rc = getStorageAsNumber( getColumnDataPtr( uiColumnNum),
										pColumnItem->uiDataLen,
										pColumn->eDataTyp, pui64Value, pbNeg)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::setBinary(
	F_Db *				pDb,
	FLMUINT				uiColumnNum,
	const void *		pvValue,
	FLMUINT				uiValueLen)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable;
	F_COLUMN *	pColumn;
	
	if (RC_BAD( rc = allocColumnDataSpace( pDb, uiColumnNum, uiValueLen, FALSE)))
	{
		goto Exit;
	}
	if( uiValueLen)
	{
		f_memcpy( getColumnDataPtr( uiColumnNum), pvValue, uiValueLen);
	}
	setRowDirty( pDb, FALSE);
	
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
RCODE F_Row::getBinary(
	F_Db *,				// pDb,
	FLMUINT				uiColumnNum,
	void *				pvBuffer,
	FLMUINT				uiBufferLen,
	FLMUINT *			puiDataLen,
	FLMBOOL *			pbIsNull)
{
	RCODE					rc = NE_SFLM_OK;
	F_COLUMN_ITEM *	pColumnItem;
	FLMUINT				uiCopySize;
	
	*pbIsNull = FALSE;
	if ((pColumnItem = getColumn( uiColumnNum)) == NULL)
	{
		*pbIsNull = TRUE;
		goto Exit;
	}
	
	// If a NULL buffer is passed in, just return the
	// data length

	if (!pvBuffer)
	{
		if (puiDataLen)
		{
			*puiDataLen = pColumnItem->uiDataLen;
		}
		goto Exit;
	}
	
	uiCopySize = f_min( uiBufferLen, pColumnItem->uiDataLen);
	if( uiCopySize)
	{
		f_memcpy( pvBuffer, getColumnDataPtr( uiColumnNum), uiCopySize);
	}
		
	if (puiDataLen)
	{
		*puiDataLen = uiCopySize;
	}
	
	// If we didn't return all of the data, return an overflow error.
	
	if (uiCopySize < pColumnItem->uiDataLen)
	{
		rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_Row::setToNull(
	F_Db *		pDb,
	FLMUINT		uiColumnNum)
{
	F_COLUMN_ITEM *	pColumnItem;
	
	// If column is already NULL, no need to do anything.
	
	if ((pColumnItem = getColumn( uiColumnNum)) != NULL)
	{
		pColumnItem->uiDataLen = 0;
		setRowDirty( pDb, FALSE);
	}
}

/*****************************************************************************
Desc:
******************************************************************************/	
void F_Row::getDataLen(
	F_Db *,				// pDb,
	FLMUINT				uiColumnNum,
	FLMUINT *			puiDataLen,
	FLMBOOL *			pbIsNull)
{
	F_COLUMN_ITEM *	pColumnItem;
	
	*pbIsNull = FALSE;
	if ((pColumnItem = getColumn( uiColumnNum)) == NULL)
	{
		*pbIsNull = TRUE;
		if (puiDataLen)
		{
			*puiDataLen = 0;
		}
	}
	else
	{
		if (puiDataLen)
		{
			*puiDataLen = pColumnItem->uiDataLen;
		}
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::setValue(
	F_Db *					pDb,
	F_COLUMN *				pColumn,
	const FLMBYTE *		pucValue,
	FLMUINT					uiValueLen)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Make sure this is not a read-only column
	
	if (pColumn->uiFlags & COL_READ_ONLY)
	{
		rc = RC_SET( NE_SFLM_COLUMN_IS_READ_ONLY);
		goto Exit;
	}

	// If a length of zero is specified, we have a NULL value.  Make sure
	// that is legal.
	
	if (!uiValueLen &&
		 !(pColumn->uiFlags & COL_NULL_ALLOWED))
	{
		rc = RC_SET( NE_SFLM_NULL_NOT_ALLOWED_IN_COLUMN);
		goto Exit;
	}

		
	if (RC_BAD( rc = allocColumnDataSpace( pDb, pColumn->uiColumnNum, uiValueLen, FALSE)))
	{
		goto Exit;
	}
	if (uiValueLen)
	{
		FLMBYTE *	pucTmp = getColumnDataPtr( pColumn->uiColumnNum);
		
		f_memcpy( pucTmp, pucValue, uiValueLen);
	}
	setRowDirty( pDb, FALSE);

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
RCODE F_Row::setUTF8(
	F_Db *					pDb,
	FLMUINT					uiColumnNum,
	const char *			pszValue,
	FLMUINT					uiNumBytesInValue,
	FLMUINT					uiNumCharsInValue)
{
	RCODE						rc = NE_SFLM_OK;
	F_TABLE *				pTable = pDb->m_pDict->getTable( m_uiTableNum);
	F_COLUMN *				pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum);
	FLMUINT					uiValLen = 0;
	FLMUINT					uiSenLen;
	FLMBYTE *				pucValue = (FLMBYTE *)pszValue;
	FLMBOOL					bNullTerminate = FALSE;

	if (pColumn->eDataTyp != SFLM_STRING_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	if (pucValue && uiNumBytesInValue)
	{
		if (pucValue[ uiNumBytesInValue - 1] != 0)
		{
			bNullTerminate = TRUE;
		}
		uiSenLen = f_getSENByteCount( uiNumCharsInValue);
		uiValLen = uiNumBytesInValue + uiSenLen + (bNullTerminate ? 1 : 0);
	}
	else
	{
		uiSenLen = 0;
		uiValLen = 0;
	}

	if (RC_BAD( rc = allocColumnDataSpace( pDb, uiColumnNum, uiValLen, FALSE)))
	{
		goto Exit;
	}
	if (uiValLen)
	{
		FLMBYTE *	pucTmp = getColumnDataPtr( uiColumnNum);
		
		f_encodeSENKnownLength( uiNumCharsInValue, uiSenLen, &pucTmp);
		f_memcpy( pucTmp, pucValue, uiNumBytesInValue);
		
		if (bNullTerminate)
		{
			pucTmp[ uiNumBytesInValue] = 0;
		}
	}
	setRowDirty( pDb, FALSE);

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
RCODE F_Row::getIStream(
	F_Db *				pDb,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	eDataType *			peDataType,
	FLMUINT *			puiDataLength)
{
	RCODE					rc = NE_SFLM_OK;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	F_COLUMN_ITEM *	pColumnItem;
	
	*pbIsNull = FALSE;
	if ((pColumnItem = getColumn( uiColumnNum)) == NULL)
	{
		*pbIsNull = TRUE;
		goto Exit;
	}
	if (puiDataLength)
	{
		*puiDataLength = pColumnItem->uiDataLen;
	}
	if (peDataType)
	{
		pTable = pDb->m_pDict->getTable( m_uiTableNum);
		pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum);
		*peDataType = pColumn->eDataTyp;
	}
	if (RC_BAD( rc = pBufferIStream->openStream( 
		(const char *)getColumnDataPtr( uiColumnNum), 
		pColumnItem->uiDataLen, NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
	
/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::getTextIStream(
	F_Db *				pDb,
	FLMUINT				uiColumnNum,
	FLMBOOL *			pbIsNull,
	F_BufferIStream *	pBufferIStream,
	FLMUINT *			puiNumChars)
{
	RCODE			rc = NE_SFLM_OK;
	eDataType	eDataTyp;
	FLMUINT		uiDataLength;

	*puiNumChars = 0;

	if( RC_BAD( rc = getIStream( pDb, uiColumnNum, pbIsNull,
								pBufferIStream, &eDataTyp, &uiDataLength)))
	{
		goto Exit;
	}

	if (eDataTyp != SFLM_STRING_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_DATA_TYPE);
		goto Exit;
	}
	
	if (*pbIsNull)
	{
		goto Exit;
	}

	// Skip the leading SEN so that the stream is positioned to
	// read raw utf8.

	if (pBufferIStream->remainingSize())
	{
		if (RC_BAD( rc = f_readSEN( pBufferIStream, puiNumChars)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::getUTF8(
	F_Db *		pDb,
	FLMUINT		uiColumnNum,
	char *		pszValueBuffer,
	FLMUINT		uiBufferSize,
	FLMBOOL *	pbIsNull,
	FLMUINT *	puiCharsReturned,
	FLMUINT *	puiBufferBytesUsed)
{
	RCODE						rc = NE_SFLM_OK;
	F_TABLE *				pTable;
	F_COLUMN *				pColumn;
	F_COLUMN_ITEM *		pColumnItem;
	const FLMBYTE *		pucStart;
	const FLMBYTE *		pucEnd;
	FLMUINT					uiCharCount;
	FLMUINT					uiStrByteLen;
	
	*pbIsNull = FALSE;
	if ((pColumnItem = getColumn( uiColumnNum)) == NULL)
	{
		*pbIsNull = TRUE;
		goto Exit;
	}
	pTable = pDb->m_pDict->getTable( m_uiTableNum);
	pColumn = pDb->m_pDict->getColumn( pTable, uiColumnNum);
	
	pucStart = getColumnDataPtr( uiColumnNum);
	pucEnd = pucStart + pColumnItem->uiDataLen;

	if( RC_BAD( rc = f_decodeSEN( &pucStart, pucEnd, &uiCharCount)))
	{
		goto Exit;
	}

	uiStrByteLen = (FLMUINT)(pucEnd - pucStart);
	
	// If buffer has room, we can simply memcpy the entire string.
	
	if (uiBufferSize >= uiStrByteLen || !pszValueBuffer)
	{
		if (pszValueBuffer)
		{
			f_memcpy( pszValueBuffer, pucStart, uiStrByteLen);
		}
		if (puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = uiStrByteLen;
		}
		if (puiCharsReturned)
		{
			*puiCharsReturned = uiCharCount;
		}
	}
	else
	{
		
		// Get as many characters as will fit in the buffer.  We will return
		// an overflow error.
		
		uiStrByteLen = 0;
		uiCharCount = 0;
		while (pucStart < pucEnd)
		{
			if (!(*pucStart))
			{
				pucEnd = pucStart;
				break;
			}
			if (!uiBufferSize)
			{
				break;
			}
			
			*pszValueBuffer++ = (char)(*pucStart);
			if (*pucStart <= 0x7F)
			{
				*pszValueBuffer++ = (char)(*pucStart);
				pucStart++;
				uiBufferSize--;
				uiStrByteLen++;
				uiCharCount++;
			}
			else if ((pucStart [1] >> 6) != 0x02)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_UTF8);
				goto Exit;
			}
			else if ((pucStart [0] >> 5) == 0x06)
			{
				if (uiBufferSize < 2)
				{
					break;
				}
				pszValueBuffer [0] = pucStart [0];
				pszValueBuffer [1] = pucStart [1];
				pszValueBuffer += 2;
				pucStart += 2;
				uiBufferSize -= 2;
				uiStrByteLen += 2;
				uiCharCount++;
			}
			else if ((pucStart [0] >> 4) != 0x0E || (pucStart [2] >> 6) != 0x02)
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_BAD_UTF8);
				goto Exit;
			}
			else
			{
				if (uiBufferSize < 3)
				{
					break;
				}
				pszValueBuffer [0] = pucStart [0];
				pszValueBuffer [1] = pucStart [1];
				pszValueBuffer [2] = pucStart [2];
				pszValueBuffer += 3;
				pucStart += 3;
				uiBufferSize -= 3;
				uiStrByteLen += 3;
				uiCharCount++;
			}
		}
		if (puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = uiStrByteLen;
		}
		if (puiCharsReturned)
		{
			*puiCharsReturned = uiCharCount;
		}
		
		// If we didn't get all of the characters, return an overflow error.
		
		if (pucStart < pucEnd)
		{
			rc = RC_SET( NE_SFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine is called to remove a row from cache.  If this is
		an uncommitted version of the row, it should remove that version
		from cache.  If the last committed version is in
		cache, it should set the high transaction ID on that version to be
		one less than the transaction ID of the update transaction.
****************************************************************************/
void F_RowCacheMgr::removeRow(
	F_Db *		pDb,
	F_Row *		pRow,
	FLMBOOL		bDecrementUseCount,
	FLMBOOL		bMutexLocked)
{
	F_Database *	pDatabase = pDb->m_pDatabase;

	flmAssert( pRow);

	// Lock the mutex

	if( !bMutexLocked)
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	}

	// Decrement the row use count if told to by the caller.

	if (bDecrementUseCount)
	{
		pRow->decrRowUseCount();
	}

	// Unset the new and dirty flags

	pRow->unsetRowDirtyAndNew( pDb, TRUE);

	// Determine if pRow is the last committed version
	// or a row that was added by this same transaction.
	// If it is the last committed version, set its high transaction ID.
	// Otherwise, remove the row from cache.

	if (pRow->m_ui64LowTransId < pDb->m_ui64CurrTransID)
	{
		
		// The high transaction ID on pRow better be -1 - most current version.

		flmAssert( pRow->m_ui64HighTransId == FLM_MAX_UINT64);

		pRow->setTransID( (pDb->m_ui64CurrTransID - 1));
		flmAssert( pRow->m_ui64HighTransId >= pRow->m_ui64LowTransId);
		pRow->setUncommitted();
		pRow->setLatestVer();
		pRow->unlinkFromDatabase();
		pRow->linkToDatabaseAtHead( pDatabase);
	}
	else
	{
		pRow->freeCache( pRow->rowInUse());
	}

	if( !bMutexLocked)
	{
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
	}
}

/****************************************************************************
Desc:	This routine is called to remove a row from cache.
****************************************************************************/
void F_RowCacheMgr::removeRow(
	F_Db *				pDb,
	FLMUINT				uiTableNum,
	FLMUINT64			ui64RowId)
{
	F_Row *	pRow;

	// Lock the mutex

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);

	// Find the row in cache

	findRow( pDb, uiTableNum, ui64RowId, 
		pDb->m_ui64CurrTransID, TRUE, NULL, &pRow, NULL, NULL);

	if( pRow)
	{
		removeRow( pDb, pRow, FALSE, TRUE);
	}

	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an F_Database object is going to be removed
		from the shared memory area.  At that point, we also need to get rid
		of all rows that have been cached for that F_Database.
****************************************************************************/
void F_Database::freeRowCache( void)
{
	FLMUINT	uiNumFreed = 0;

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	while (m_pFirstRow)
	{
		m_pFirstRow->freeCache( m_pFirstRow->rowInUse());

		// Release the CPU every 100 rows freed.

		if (++uiNumFreed == 100)
		{
			f_yieldCPU();
			uiNumFreed = 0;
		}
	}
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction aborts.  At that
		point, we need to get rid of any uncommitted versions of rows in
		the row cache.
****************************************************************************/
void F_Database::freeModifiedRows(
	F_Db *		pDb,
	FLMUINT64	ui64OlderTransId)
{
	F_Row *		pRow;
	F_Row *		pOlderVersion;

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	pRow = m_pFirstRow;
	while (pRow)
	{
		if (pRow->rowUncommitted())
		{
			if (pRow->rowIsLatestVer())
			{
				pRow->setTransID( FLM_MAX_UINT64);
				pRow->unsetUncommitted();
				pRow->unsetLatestVer();
				pRow->unlinkFromDatabase();
				pRow->linkToDatabaseAtEnd( this);
			}
			else
			{
				// Save the older version - we may be changing its
				// high transaction ID back to FLM_MAX_UINT64

				pOlderVersion = pRow->m_pOlderVersion;

				// Clear the dirty and new flags
				
				pRow->unsetRowDirtyAndNew( pDb, TRUE);

				// Free the uncommitted version.

				pRow->freeCache(
						(FLMBOOL)((pRow->rowInUse() ||
									  pRow->readingInRow())
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
			pRow = m_pFirstRow;
		}
		else
		{
			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction commits.  At that
		point, we need to unset the "uncommitted" flag on any rows
		currently in row cache for the F_Database object.
****************************************************************************/
void F_Database::commitRowCache( void)
{
	F_Row *	pRow;

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	pRow = m_pFirstRow;
	while (pRow)
	{
		if (pRow->rowUncommitted())
		{
			pRow->unsetUncommitted();
			pRow->unsetLatestVer();
			pRow = pRow->m_pNextInDatabase;
		}
		else
		{

			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
}

/****************************************************************************
Desc:	This routine is called when a table in the database is deleted.
		All rows in row cache that are in that table must be removed from cache.
****************************************************************************/
void F_Db::removeTableRows(
	FLMUINT		uiTableNum,
	FLMUINT64	ui64TransId)
{
	F_Row *	pRow;
	F_Row *	pNextRow;

	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
	pRow = m_pDatabase->m_pFirstRow;

	// Stay in the loop until we have freed all rows in the
	// collection

	while (pRow)
	{

		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pRow below, in which case we would
		// have lost the previous entry.

		pNextRow = pRow->m_pNextInDatabase;

		// Only look at rows in this collection

		if (pRow->m_uiTableNum == uiTableNum)
		{
			flmAssert( pRow->m_pDatabase == m_pDatabase);
			
			// Only look at the most current versions.

			if (pRow->m_ui64HighTransId == FLM_MAX_UINT64)
			{

				// Better not be a newer version.

				flmAssert( pRow->m_pNewerVersion == NULL);

				if (pRow->m_ui64LowTransId < ui64TransId)
				{

					// This version was not added or modified by this
					// transaction so it's high transaction ID should simply
					// be set to one less than the current transaction ID.

					pRow->setTransID( ui64TransId - 1);
					flmAssert( pRow->m_ui64HighTransId >= pRow->m_ui64LowTransId);
					pRow->setUncommitted();
					pRow->setLatestVer();
					pRow->unlinkFromDatabase();
					pRow->linkToDatabaseAtHead( m_pDatabase);
				}
				else
				{

					// The row was added or modified in this
					// transaction. Simply remove it from cache.

					pRow->freeCache( pRow->rowInUse());
				}
			}
			else
			{

				// If not most current version, the row's high transaction
				// ID better already be less than transaction ID.

				flmAssert( pRow->m_ui64HighTransId < ui64TransId);
			}
		}
		pRow = pNextRow;

	}
	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void F_Row::checkReadFromDisk(
	F_Db *	pDb)
{
	FLMUINT64	ui64LowTransId;
	FLMBOOL		bMostCurrent;
	RCODE			rc;
	
	// Need to unlock the row cache mutex before doing the read.

	f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);	
	rc = gv_SFlmSysData.pRowCacheMgr->readRowFromDisk( pDb,
					m_uiTableNum, m_ui64RowId,
					this, &ui64LowTransId, &bMostCurrent);
	
	f_mutexLock( gv_SFlmSysData.hRowCacheMutex);	
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
void F_Row::setRowDirty(
	F_Db *		pDb,
	FLMBOOL		bNew)
{
	if (!rowIsDirty())
	{
		f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		
		// Should already be the uncommitted version.
		
		flmAssert( rowUncommitted());
		
		// Should NOT be the latest version - those cannot
		// be set to dirty. - latest ver flag is only set
		// for rows that should be returned to being the
		// latest version of the row if the transaction
		// aborts.
		
		flmAssert( !rowIsLatestVer());
		
		// Unlink from its database, set the dirty flag,
		// and relink at the head.
		
		unlinkFromDatabase();
		
		if (bNew)
		{
			m_uiFlags |= (FROW_DIRTY | FROW_NEW);
		}
		else
		{
			m_uiFlags |= FROW_DIRTY;
		}
		
		linkToDatabaseAtHead( pDb->m_pDatabase);
		f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
		
		pDb->m_uiDirtyRowCount++;
	}
	else if (bNew)
	{
		m_uiFlags |= FROW_NEW;
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
void F_Row::unsetRowDirtyAndNew(
	F_Db *			pDb,
	FLMBOOL			bMutexAlreadyLocked)
{
	// When outputting a binary or text stream, it is possible that the
	// dirty flag was unset when the last buffer was output
	
	if (rowIsDirty())
	{
		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( gv_SFlmSysData.hRowCacheMutex);
		}
			
		// Unlink from its database, unset the dirty flag,
		// and relink at the head.
		
		unlinkFromDatabase();
		if( m_uiFlags & FROW_DIRTY)
		{
			flmAssert( pDb->m_uiDirtyRowCount);
			pDb->m_uiDirtyRowCount--;
		}

		m_uiFlags &= ~(FROW_DIRTY | FROW_NEW);
		linkToDatabaseAtHead( pDb->m_pDatabase);
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( gv_SFlmSysData.hRowCacheMutex);
		}
	}
}
	
/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Row::copyColumnList(
	F_Db *	pDb,
	F_Row *	pSourceRow,
	FLMBOOL	bMutexAlreadyLocked)
{
	RCODE	rc = NE_SFLM_OK;
	
	flmAssert( !m_uiNumColumns);
	if( RC_BAD( rc = resizeColumnList( pSourceRow->m_uiNumColumns,
								bMutexAlreadyLocked)))
	{
		goto Exit;
	}
	if (m_uiNumColumns)
	{
		f_memcpy( m_pColumns, pSourceRow->m_pColumns,
						sizeof( F_COLUMN_ITEM) * m_uiNumColumns);
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
void F_Row::resetRow( void)
{
	FLMBYTE *	pucActualAlloc;
	FLMUINT		uiSize = memSize();

	flmAssert( !m_pPrevInBucket);
	flmAssert( !m_pNextInBucket);
	flmAssert( !m_pOlderVersion);
	flmAssert( !m_pNewerVersion);
	flmAssert( !m_pPrevInOldList);
	flmAssert( !m_pNextInOldList);
	flmAssert( !m_pNotifyList);
	flmAssert( !m_uiStreamUseCount);
	flmAssert( !rowInUse());

	f_assertMutexLocked( gv_SFlmSysData.hRowCacheMutex);

	// If this is an old version, decrement the old version counters.

	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes >= uiSize &&
					  gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerCount);
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
	}

	flmAssert( gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount >= uiSize &&
				  gv_SFlmSysData.pRowCacheMgr->m_Usage.uiCount);
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount -= uiSize;

	if( m_uiFlags & FROW_HEAP_ALLOC)
	{
		unlinkFromHeapList();
	}

	if (m_pucColumnData || m_pColumns)
	{
		f_assertMutexLocked( gv_SFlmSysData.hRowCacheMutex);

		if (m_pucColumnData)
		{
			pucActualAlloc = getActualPointer( m_pucColumnData);
			gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->freeBuf(
								calcDataBufSize(m_uiColumnDataBufSize),
								&pucActualAlloc);
			m_pucColumnData = NULL;
			m_uiColumnDataBufSize = 0;
		}
		
		if (m_pColumns)
		{
			pucActualAlloc = getActualPointer( m_pColumns);
			gv_SFlmSysData.pRowCacheMgr->m_pBufAllocator->freeBuf(
								calcColumnListBufSize( m_uiNumColumns), &pucActualAlloc);
			m_pColumns = NULL;
			m_uiNumColumns = 0;
		}
	}

	m_ui64LowTransId = 0;
	m_ui64HighTransId = FLM_MAX_UINT64;
	m_uiCacheFlags = 0;
	m_pDatabase = NULL;
	m_uiFlags = 0;
	m_uiOffsetIndex = 0;
	m_ui32BlkAddr = 0;
	m_uiTableNum = 0;
	m_ui64RowId = 0;

	uiSize = memSize();
	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		gv_SFlmSysData.pRowCacheMgr->m_Usage.uiOldVerBytes += uiSize;
	}
	gv_SFlmSysData.pRowCacheMgr->m_Usage.uiByteCount += uiSize;
}
	
#undef new
#undef delete

/****************************************************************************
Desc:
****************************************************************************/
void * F_Row::operator new(
	FLMSIZET			uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
	void *		pvCell;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( uiSize);
#endif
	flmAssert( uiSize == sizeof( F_Row));
	f_assertMutexLocked( gv_SFlmSysData.hRowCacheMutex);

	pvCell = gv_SFlmSysData.pRowCacheMgr->m_pRowAllocator->allocCell(
				&gv_SFlmSysData.pRowCacheMgr->m_rowRelocator, NULL);

	return( pvCell);
}

/****************************************************************************
Desc:
****************************************************************************/
void * F_Row::operator new[]( FLMSIZET)
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
void * F_Row::operator new(
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
void * F_Row::operator new[](
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
void F_Row::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	gv_SFlmSysData.pRowCacheMgr->m_pRowAllocator->freeCell( (FLMBYTE *)ptr);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Row::operator delete[](
	void *		// ptr)
	)
{
	flmAssert( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
void F_Row::operator delete( 
	void *			ptr,
	const char *,	// pszFileName
	int				// iLineNum
	)
{
	if( !ptr)
	{
		return;
	}

	gv_SFlmSysData.pRowCacheMgr->m_pRowAllocator->freeCell( (FLMBYTE *)ptr);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__) && !defined( FLM_SOLARIS)
void F_Row::operator delete[](
	void *,			// ptr,
	const char *,	// pszFileName
	int				// iLineNum
	)
{
	flmAssert( 0);
}
#endif

