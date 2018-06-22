//-------------------------------------------------------------------------
// Desc:	Record caching
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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
Desc:
****************************************************************************/
class F_RCacheRelocator : public IF_Relocator
{
public:

	F_RCacheRelocator()
	{
	}
	
	virtual ~F_RCacheRelocator()
	{
	}

	void FLMAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL FLMAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:
****************************************************************************/
class F_RecRelocator : public IF_Relocator
{
public:

	F_RecRelocator()
	{
	}
	
	virtual ~F_RecRelocator()
	{
	}

	void FLMAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL FLMAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:
****************************************************************************/
class F_RecBufferRelocator : public IF_Relocator
{
public:

	F_RecBufferRelocator()
	{
	}
	
	virtual ~F_RecBufferRelocator()
	{
	}

	void FLMAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL FLMAPI canRelocate(
		void *	pvOldAlloc);
};

/****************************************************************************
Desc:	Extended record object for accessing private members of FlmRecord
****************************************************************************/
struct FlmRecordExt
{
	static FINLINE void clearCached(
		FlmRecord *			pRec)
	{
		pRec->clearCached();
	}

	static FINLINE void setCached(
		FlmRecord *			pRec)
	{
		pRec->setCached();
	}

	static FINLINE void setReadOnly(
		FlmRecord *			pRec)
	{
		pRec->setReadOnly();
	}

	static FINLINE FLMINT Release(
		FlmRecord *			pRec,
		FLMBOOL				bMutexLocked)
	{
		return( pRec->Release( bMutexLocked));
	}

	static FINLINE void setOldVersion(
		FlmRecord *			pRec)
	{
		pRec->setOldVersion();
	}

	static FINLINE void clearOldVersion(
		FlmRecord *			pRec)
	{
		pRec->clearOldVersion();
	}
	
	static FINLINE FLMUINT getFlags(
		FlmRecord *			pRec)
	{
		return( pRec->m_uiFlags);
	}
};

// Functions for calculating minimum and maximum record counts for a
// given hash table size.

#define FLM_RCA_MIN_REC_CNT(uiHashTblSz) \
	((uiHashTblSz) / 4)
	
#define FLM_RCA_MAX_REC_CNT(uiHashTblSz) \
	((uiHashTblSz) * 4)

// Hash function for hashing to records in record cache.

#define FLM_RCA_HASH( uiDrn) \
	(RCACHE **)(&(gv_FlmSysData.RCacheMgr.ppHashBuckets[(uiDrn) & \
						(gv_FlmSysData.RCacheMgr.uiHashMask)]))

// Local functions

FSTATIC void flmRcaFreePurged(
	RCACHE *			pRCache);

FSTATIC void flmRcaFreeCache(
	RCACHE *			pRCache,
	FLMBOOL			bPutInPurgeList);

FSTATIC FLMUINT flmRcaGetBestHashTblSize(
	FLMUINT			uiCurrRecCount);

FSTATIC RCODE flmRcaRehash( void);

FSTATIC RCODE flmRcaSetMemLimit(
	FLMUINT			uiMaxCacheBytes);

FSTATIC void flmRcaNotify(
	F_NOTIFY_LIST_ITEM *		pNotify,
	RCACHE *						pUseRCache,
	RCODE							NotifyRc);

FSTATIC RCODE flmRcaAllocCacheStruct(
	RCACHE **		ppRCache);

FSTATIC void flmRcaFreeCacheStruct(
	RCACHE  **		ppRCache);

FSTATIC void flmRcaSetRecord(
	RCACHE *			pRCache,
	FlmRecord *		pNewRecord);

FSTATIC void flmRcaLinkIntoRCache(
	RCACHE *			pNewerRCache,
	RCACHE *			pOlderRCache,
	RCACHE *			pRCache,
	FLMBOOL			bLinkAsMRU);

FSTATIC void flmRcaLinkToFFILE(
	RCACHE *			pRCache,
	FFILE *			pFile,
	FDB *				pDb,
	FLMUINT			uiLowTransId,
	FLMBOOL			bMostCurrent);

#ifdef FLM_DEBUG
FSTATIC RCODE flmRcaCheck(
	FDB *				pDb,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn);
#endif

/****************************************************************************
Desc:	This inline assumes that the global mutex is locked, because
		it potentially updates the cache usage statistics.
****************************************************************************/
FINLINE void flmRcaSetTransID(
	RCACHE *		pRCache,
	FLMUINT		uiNewTransID)
{
	if (pRCache->uiHighTransId == 0xFFFFFFFF &&
		 uiNewTransID != 0xFFFFFFFF)
	{
		FLMUINT	uiSize = (FLMUINT)((pRCache->pRecord)
								 ? pRCache->pRecord->getTotalMemory()
								 : (FLMUINT)0) + sizeof( RCACHE);
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes += uiSize;
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount++;

		if( pRCache->pRecord)
		{
			FlmRecordExt::setOldVersion( pRCache->pRecord);
		}
	}
	else if (pRCache->uiHighTransId != 0xFFFFFFFF &&
				uiNewTransID == 0xFFFFFFFF)
	{
		FLMUINT	uiSize = (FLMUINT)((pRCache->pRecord)
								 ? pRCache->pRecord->getTotalMemory()
								 : (FLMUINT)0) + sizeof( RCACHE);
		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes >= uiSize);
		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount);
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes -= uiSize;
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount--;
		if( pRCache->pRecord)
		{
			FlmRecordExt::clearOldVersion( pRCache->pRecord);
		}
	}
	pRCache->uiHighTransId = uiNewTransID;
}

/****************************************************************************
Desc:	This routine links a record into the global list as the MRU record.
		This routine assumes that the record cache mutex has already
		been locked.
****************************************************************************/
FINLINE void flmRcaLinkToGlobalAsMRU(
	RCACHE *			pRCache)
{
	pRCache->pPrevInGlobal = NULL;
	if ((pRCache->pNextInGlobal = gv_FlmSysData.RCacheMgr.pMRURecord) != NULL)
	{
		gv_FlmSysData.RCacheMgr.pMRURecord->pPrevInGlobal = pRCache;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pLRURecord = pRCache;
	}
	gv_FlmSysData.RCacheMgr.pMRURecord = pRCache;
}

/****************************************************************************
Desc:	This routine links a record into the global list as the LRU record.
		This routine assumes that the record cache mutex has already
		been locked.
****************************************************************************/
FINLINE void flmRcaLinkToGlobalAsLRU(
	RCACHE *			pRCache)
{
	pRCache->pNextInGlobal = NULL;
	if ((pRCache->pPrevInGlobal = gv_FlmSysData.RCacheMgr.pLRURecord) != NULL)
	{
		gv_FlmSysData.RCacheMgr.pLRURecord->pNextInGlobal = pRCache;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pMRURecord = pRCache;
	}
	gv_FlmSysData.RCacheMgr.pLRURecord = pRCache;
}

/****************************************************************************
Desc:	Moves a record one step closer to the MRU slot in the global list.
		This routine assumes that the record cache mutex has already
		been locked.
****************************************************************************/
FINLINE void flmRcaStepUpInGlobalList(
	RCACHE *			pRCache)
{
	RCACHE *		pPrevRCache;

	if( (pPrevRCache = pRCache->pPrevInGlobal) != NULL)
	{
		if( pPrevRCache->pPrevInGlobal)
		{
			pPrevRCache->pPrevInGlobal->pNextInGlobal = pRCache;
		}
		else
		{
			gv_FlmSysData.RCacheMgr.pMRURecord = pRCache;
		}

		pRCache->pPrevInGlobal = pPrevRCache->pPrevInGlobal;
		pPrevRCache->pPrevInGlobal = pRCache;
		pPrevRCache->pNextInGlobal = pRCache->pNextInGlobal;

		if( pRCache->pNextInGlobal)
		{
			pRCache->pNextInGlobal->pPrevInGlobal = pPrevRCache;
		}
		else
		{
			gv_FlmSysData.RCacheMgr.pLRURecord = pPrevRCache;
		}
		pRCache->pNextInGlobal = pPrevRCache;
	}
}

/****************************************************************************
Desc:	This routine unlinks a record from the global list  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaUnlinkFromGlobal(
	RCACHE *			pRCache)
{
	if (pRCache->pNextInGlobal)
	{
		pRCache->pNextInGlobal->pPrevInGlobal = pRCache->pPrevInGlobal;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pLRURecord = pRCache->pPrevInGlobal;
	}
	if (pRCache->pPrevInGlobal)
	{
		pRCache->pPrevInGlobal->pNextInGlobal = pRCache->pNextInGlobal;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pMRURecord = pRCache->pNextInGlobal;
	}
	pRCache->pPrevInGlobal = pRCache->pNextInGlobal = NULL;
}

/****************************************************************************
Desc:	This routine unlinks a record from the global purged list  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaUnlinkFromPurged(
	RCACHE *			pRCache)
{
	if (pRCache->pNextInGlobal)
	{
		pRCache->pNextInGlobal->pPrevInGlobal = pRCache->pPrevInGlobal;
	}
	
	if (pRCache->pPrevInGlobal)
	{
		pRCache->pPrevInGlobal->pNextInGlobal = pRCache->pNextInGlobal;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pPurgeList = pRCache->pNextInGlobal;
	}
	
	pRCache->pPrevInGlobal = NULL; 
	pRCache->pNextInGlobal = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void flmRcaLinkToHeapList(
	RCACHE *			pRCache)
{
	flmAssert( !pRCache->pPrevInHeapList);
	flmAssert( !pRCache->pNextInHeapList);
	flmAssert( !RCA_IS_IN_HEAP_LIST( pRCache->uiFlags));
	flmAssert( FlmRecordExt::getFlags( pRCache->pRecord) & RCA_HEAP_BUFFER);
	
	if( (pRCache->pNextInHeapList = gv_FlmSysData.RCacheMgr.pHeapList) != NULL)
	{
		pRCache->pNextInHeapList->pPrevInHeapList = pRCache;
	}
	gv_FlmSysData.RCacheMgr.pHeapList = pRCache;
	RCA_SET_IN_HEAP_LIST( pRCache->uiFlags);
}
		
/****************************************************************************
Desc:
****************************************************************************/
FINLINE void flmRcaUnlinkFromHeapList(
	RCACHE *			pRCache)
{
	flmAssert( RCA_IS_IN_HEAP_LIST( pRCache->uiFlags));
	
	if( pRCache->pNextInHeapList)
	{
		pRCache->pNextInHeapList->pPrevInHeapList = pRCache->pPrevInHeapList;
	}
	
	if( pRCache->pPrevInHeapList)
	{
		pRCache->pPrevInHeapList->pNextInHeapList = pRCache->pNextInHeapList;
	}
	else
	{
		gv_FlmSysData.RCacheMgr.pHeapList = pRCache->pNextInHeapList;
	}
	
	pRCache->pPrevInHeapList = NULL; 
	pRCache->pNextInHeapList = NULL;
	
	RCA_UNSET_IN_HEAP_LIST( pRCache->uiFlags);
}

/****************************************************************************
Desc:	This routine links a record to an FFILE list at the head of the list.
		This routine assumes that the record cache mutex has already been
		locked.
****************************************************************************/
FINLINE void flmRcaLinkToFileAtHead(
	RCACHE *		pRCache,
	FFILE *		pFile)
{
	pRCache->pPrevInFile = NULL;
	if ((pRCache->pNextInFile = pFile->pFirstRecord) != NULL)
	{
		pFile->pFirstRecord->pPrevInFile = pRCache;
	}
	else
	{
		pFile->pLastRecord = pRCache;
	}

	pFile->pFirstRecord = pRCache;
	pRCache->pFile = pFile;
	RCA_SET_LINKED_TO_FILE( pRCache->uiFlags);
}

/****************************************************************************
Desc:	This routine links a record to an FFILE list at the end of the list.
		This routine assumes that the record cache mutex has already been
		locked.
****************************************************************************/
FINLINE void flmRcaLinkToFileAtEnd(
	RCACHE *		pRCache,
	FFILE *		pFile)
{
	pRCache->pNextInFile = NULL;
	if( (pRCache->pPrevInFile = pFile->pLastRecord) != NULL)
	{
		pFile->pLastRecord->pNextInFile = pRCache;
	}
	else
	{
		pFile->pFirstRecord = pRCache;
	}
	pFile->pLastRecord = pRCache;
	pRCache->pFile = pFile;
	RCA_SET_LINKED_TO_FILE( pRCache->uiFlags);
}

/****************************************************************************
Desc:	This routine unlinks a record from its FFILE list.  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaUnlinkFromFile(
	RCACHE *		pRCache)
{
	if( RCA_IS_LINKED_TO_FILE( pRCache->uiFlags))
	{
		if( pRCache->pNextInFile)
		{
			pRCache->pNextInFile->pPrevInFile = pRCache->pPrevInFile;
		}
		else
		{
			pRCache->pFile->pLastRecord = pRCache->pPrevInFile;
		}
		
		if( pRCache->pPrevInFile)
		{
			pRCache->pPrevInFile->pNextInFile = pRCache->pNextInFile;
		}
		else
		{
			pRCache->pFile->pFirstRecord = pRCache->pNextInFile;
		}
		
		pRCache->pPrevInFile = pRCache->pNextInFile = NULL;
		RCA_UNSET_LINKED_TO_FILE( pRCache->uiFlags);
	}
}

/****************************************************************************
Desc:	This routine links a record into its hash bucket.  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaLinkToHashBucket(
	RCACHE *			pRCache)
{
	RCACHE **	ppHashBucket = FLM_RCA_HASH( pRCache->uiDrn);
	
	flmAssert( pRCache->pNewerVersion == NULL);

	pRCache->pPrevInBucket = NULL;
	if( (pRCache->pNextInBucket = *ppHashBucket) != NULL)
	{
		pRCache->pNextInBucket->pPrevInBucket = pRCache;
	}
	*ppHashBucket = pRCache;
}

/****************************************************************************
Desc:	This routine unlinks a record from its hash bucket.  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaUnlinkFromHashBucket(
	RCACHE *			pRCache)
{
	flmAssert( pRCache->pNewerVersion == NULL);
	if (pRCache->pNextInBucket)
	{
		pRCache->pNextInBucket->pPrevInBucket = pRCache->pPrevInBucket;
	}
	
	if (pRCache->pPrevInBucket)
	{
		pRCache->pPrevInBucket->pNextInBucket = pRCache->pNextInBucket;
	}
	else
	{
		RCACHE ** ppHashBucket = FLM_RCA_HASH( pRCache->uiDrn);
		*ppHashBucket = pRCache->pNextInBucket;
	}
	pRCache->pPrevInBucket = pRCache->pNextInBucket = NULL;
}

/****************************************************************************
Desc:	This routine unlinks a record from its version list.  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaLinkToVerList(
	RCACHE *			pRCache,
	RCACHE *			pNewerVer,
	RCACHE *			pOlderVer)
{
	if( (pRCache->pNewerVersion = pNewerVer) != NULL)
	{
		pNewerVer->pOlderVersion = pRCache;
	}
	
	if ((pRCache->pOlderVersion = pOlderVer) != NULL)
	{
		pOlderVer->pNewerVersion = pRCache;
	}
}

/****************************************************************************
Desc:	This routine unlinks a record from its version list.  This routine
		assumes that the record cache mutex has already been locked.
****************************************************************************/
FINLINE void flmRcaUnlinkFromVerList(
	RCACHE *			pRCache)
{
	if (pRCache->pNewerVersion)
	{
		pRCache->pNewerVersion->pOlderVersion = pRCache->pOlderVersion;
	}
	
	if (pRCache->pOlderVersion)
	{
		pRCache->pOlderVersion->pNewerVersion = pRCache->pNewerVersion;
	}
	pRCache->pNewerVersion = pRCache->pOlderVersion = NULL;
}

/****************************************************************************
Desc:	This routine frees a purged from record cache.  This routine assumes
		that the record cache mutex has already been locked.
****************************************************************************/
FSTATIC void flmRcaFreePurged(
	RCACHE *			pRCache)
{
	FLMUINT			uiTotalMemory;
	FLMUINT			uiFreeMemory;

	// Release the record data object we are pointing to.

	if (pRCache->pRecord)
	{
		if( RCA_IS_IN_HEAP_LIST( pRCache->uiFlags))
		{
			flmRcaUnlinkFromHeapList( pRCache);
		}
	
		uiTotalMemory = pRCache->pRecord->getTotalMemory();
		uiFreeMemory = pRCache->pRecord->getFreeMemory();
		flmAssert( uiTotalMemory >= uiFreeMemory);
		FlmRecordExt::clearCached( pRCache->pRecord);
		FlmRecordExt::Release( pRCache->pRecord, TRUE);
		pRCache->pRecord = NULL;
	}
	else
	{
		uiTotalMemory = 0;
		uiFreeMemory = 0;
	}

	// If this is an old version, decrement the old version counters.

	if (pRCache->uiHighTransId != 0xFFFFFFFF)
	{
		FLMUINT	uiTotalOldMemory = uiTotalMemory + sizeof( RCACHE);

		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes >=
							uiTotalOldMemory);
		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount);
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes -= uiTotalOldMemory;
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount--;
	}

	// Unlink the RCACHE from the purged list.

	flmRcaUnlinkFromPurged( pRCache);

	// Free the RCACHE structure.

	RCA_UNSET_PURGED( pRCache->uiFlags);
	flmRcaFreeCacheStruct( &pRCache);
}

/****************************************************************************
Desc:	This routine frees a record in the record cache.  This routine assumes
		that the record cache mutex has already been locked.
****************************************************************************/
FSTATIC void flmRcaFreeCache(
	RCACHE *			pRCache,
	FLMBOOL			bPutInPurgeList)
{
	FLMUINT	uiTotalMemory;
	FLMUINT	uiFreeMemory;
	FLMBOOL	bOldVersion;
#ifdef FLM_DBG_LOG
	char		szTmpBuf[ 80];
#endif

	// Release the record data object we are pointing to.

	if (pRCache->pRecord && !bPutInPurgeList)
	{
		if( RCA_IS_IN_HEAP_LIST( pRCache->uiFlags))
		{
			flmRcaUnlinkFromHeapList( pRCache);
		}
		
		uiTotalMemory = pRCache->pRecord->getTotalMemory();
		uiFreeMemory = pRCache->pRecord->getFreeMemory();
		flmAssert( uiTotalMemory >= uiFreeMemory);
		FlmRecordExt::clearCached( pRCache->pRecord);
		FlmRecordExt::Release( pRCache->pRecord, TRUE);
		pRCache->pRecord = NULL;
	}
	else
	{
		uiTotalMemory = 0;
		uiFreeMemory = 0;
	}
	bOldVersion = (FLMBOOL)((pRCache->uiHighTransId != 0xFFFFFFFF)
									? TRUE
									: FALSE);

#ifdef FLM_DBG_LOG
	f_sprintf( szTmpBuf, "RCD:H%X",
		(unsigned)pRCache->uiHighTransId);

	flmDbgLogWrite( pRCache->pFile ? pRCache->pFile->uiFFileId : 0, pRCache->uiContainer,
		pRCache->uiDrn, pRCache->uiLowTransId, szTmpBuf);
#endif

	// Unlink the RCACHE from its various lists.

	flmRcaUnlinkFromGlobal( pRCache);
	flmRcaUnlinkFromFile( pRCache);
	if (!pRCache->pNewerVersion)
	{
		RCACHE *		pOlderVersion = pRCache->pOlderVersion;

		flmRcaUnlinkFromHashBucket( pRCache);

		// If there was an older version, it now needs to be
		// put into the hash bucket.

		if (pOlderVersion)
		{
			flmRcaUnlinkFromVerList( pRCache);
			flmRcaLinkToHashBucket( pOlderVersion);
		}
	}
	else
	{
		flmRcaUnlinkFromVerList( pRCache);
	}

	// Free the RCACHE structure if not putting in purge list.

	if (!bPutInPurgeList)
	{
		// If this was an older version, decrement the old version counters.

		if (bOldVersion)
		{
			FLMUINT	uiTotalOldMemory = uiTotalMemory + sizeof( RCACHE);

			flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes >=
								uiTotalOldMemory);
			flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount);
			gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount--;
			gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes -=
								uiTotalOldMemory;
		}
		flmRcaFreeCacheStruct( &pRCache);
	}
	else
	{
		if ((pRCache->pNextInGlobal = gv_FlmSysData.RCacheMgr.pPurgeList) != NULL)
		{
			pRCache->pNextInGlobal->pPrevInGlobal = pRCache;
		}
		gv_FlmSysData.RCacheMgr.pPurgeList = pRCache;
		RCA_SET_PURGED( pRCache->uiFlags);
	}
}

/****************************************************************************
Desc:	This routine initializes record cache manager.
****************************************************************************/
RCODE flmRcaInit(
	FLMUINT		uiMaxRecordCacheBytes)
{
	RCODE							rc = FERR_OK;
	F_RCacheRelocator *		pRCacheRelocator = NULL;
	F_RecRelocator *			pRecRelocator = NULL;
	F_RecBufferRelocator *	pRecBufferRelocator = NULL;
	

	f_memset( &gv_FlmSysData.RCacheMgr, 0, sizeof( RCACHE_MGR));
	gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes = uiMaxRecordCacheBytes;
	gv_FlmSysData.RCacheMgr.hMutex = F_MUTEX_NULL;

	// Allocate the hash buckets.

	if (RC_BAD( rc = f_calloc(
								(FLMUINT)sizeof( RCACHE *) *
								(FLMUINT)MIN_RCACHE_BUCKETS,
								&gv_FlmSysData.RCacheMgr.ppHashBuckets)))
	{
		goto Exit;
	}
	gv_FlmSysData.RCacheMgr.uiNumBuckets = MIN_RCACHE_BUCKETS;
	gv_FlmSysData.RCacheMgr.uiHashMask =
		gv_FlmSysData.RCacheMgr.uiNumBuckets - 1;
	gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated +=
			(sizeof( RCACHE *) * gv_FlmSysData.RCacheMgr.uiNumBuckets);

	// Allocate the mutex for controlling access to the
	// record cache.

	if (RC_BAD( rc = f_mutexCreate( &gv_FlmSysData.RCacheMgr.hMutex)))
	{
		goto Exit;
	}
	
	// Set up the RCACHE struct allocator
	
	if( RC_BAD( rc = FlmAllocFixedAllocator( 
		&gv_FlmSysData.RCacheMgr.pRCacheAlloc)))
	{
		goto Exit;
	}
	
	if( (pRCacheRelocator = f_new F_RCacheRelocator) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRCacheAlloc->setup(
		FALSE, gv_FlmSysData.pSlabManager, pRCacheRelocator, sizeof( RCACHE), 
		&gv_FlmSysData.RCacheMgr.Usage.SlabUsage, 
		&gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated)))
	{
		goto Exit;
	}
	
	// Set up the record object allocator
	
	if( RC_BAD( rc = FlmAllocFixedAllocator(
		&gv_FlmSysData.RCacheMgr.pRecAlloc)))
	{
		goto Exit;
	}
	
	if( (pRecRelocator = f_new F_RecRelocator) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecAlloc->setup(
		TRUE, gv_FlmSysData.pSlabManager, pRecRelocator, sizeof( FlmRecord),
		&gv_FlmSysData.RCacheMgr.Usage.SlabUsage,
		&gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated)))
	{
		goto Exit;
	}

	// Set up the record buffer allocator
	
	if( RC_BAD( rc = FlmAllocBufferAllocator(
		&gv_FlmSysData.RCacheMgr.pRecBufAlloc)))
	{
		goto Exit;
	}

	if( (pRecBufferRelocator = f_new F_RecBufferRelocator) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->setup( 
		TRUE, gv_FlmSysData.pSlabManager, pRecBufferRelocator,
		&gv_FlmSysData.RCacheMgr.Usage.SlabUsage,
		&gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated))) 
	{
		goto Exit;
	}

#ifdef FLM_DEBUG
	gv_FlmSysData.RCacheMgr.bDebug = TRUE;
#endif

Exit:

	if( pRCacheRelocator)
	{
		pRCacheRelocator->Release();
	}

	if( pRecRelocator)
	{
		pRecRelocator->Release();
	}
	
	if( pRecBufferRelocator)
	{
		pRecBufferRelocator->Release();
	}
	
	if (RC_BAD( rc))
	{
		flmRcaExit();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	This routine determines what hash table size best fits the current
		record count.  It finds the hash bucket size whose midpoint between
		the minimum and maximum range is closest to the record count.
****************************************************************************/
FSTATIC FLMUINT flmRcaGetBestHashTblSize(
	FLMUINT		uiCurrRecCount)
{
	FLMUINT		uiHashTblSize;
	FLMUINT		uiMaxRecsForHashTblSize;
	FLMUINT		uiMinRecsForHashTblSize;
	FLMUINT		uiClosestHashTblSize = 0;
	FLMUINT		uiDistanceFromMidpoint;
	FLMUINT		uiLowestDistanceFromMidpoint;
	FLMUINT		uiHashTblRecsMidpoint;

	uiLowestDistanceFromMidpoint = 0xFFFFFFFF;
	for (uiHashTblSize = MIN_RCACHE_BUCKETS;
		  uiHashTblSize <= MAX_RCACHE_BUCKETS;
		  uiHashTblSize *= 2)
	{

		// Maximum desirable record count for a specific hash table size
		// we have arbitrarily chosen to be four times the number of buckets.
		// Minimum desirable record count we have arbitrarily chosen to be
		// the hash table size divided by four.

		uiMaxRecsForHashTblSize = FLM_RCA_MAX_REC_CNT( uiHashTblSize);
		uiMinRecsForHashTblSize = FLM_RCA_MIN_REC_CNT( uiHashTblSize);

		// Ignore any hash bucket sizes where the current record count
		// is not between the desired minimum and maximum.

		if (uiCurrRecCount >= uiMinRecsForHashTblSize &&
			 uiCurrRecCount <= uiMaxRecsForHashTblSize)
		{

			// Calculate the midpoint between the minimum and maximum
			// for this particular hash table size.

			uiHashTblRecsMidpoint = (uiMaxRecsForHashTblSize -
											 uiMinRecsForHashTblSize) / 2;

			// See how far our current record count is from this midpoint.

			uiDistanceFromMidpoint = (FLMUINT)((uiHashTblRecsMidpoint > uiCurrRecCount)
											 ? (uiHashTblRecsMidpoint - uiCurrRecCount)
											 : (uiCurrRecCount - uiHashTblRecsMidpoint));

			// If the distance from the midpoint is closer than our previous
			// lowest distance, save it.

			if (uiDistanceFromMidpoint < uiLowestDistanceFromMidpoint)
			{
				uiClosestHashTblSize = uiHashTblSize;
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

		uiHashTblSize = 
			(FLMUINT)((uiCurrRecCount < FLM_RCA_MIN_REC_CNT( MIN_RCACHE_BUCKETS))
									  ? (FLMUINT)MIN_RCACHE_BUCKETS
									  : (FLMUINT)MAX_RCACHE_BUCKETS);

	}
	else
	{
		uiHashTblSize = uiClosestHashTblSize;
	}
	return( uiHashTblSize);
}

/****************************************************************************
Desc:	This routine resizes the hash table for the record cache manager.
		NOTE: This routine assumes that the record cache mutex has been locked.
****************************************************************************/
FSTATIC RCODE flmRcaRehash( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNewHashTblSize;
	RCACHE **	ppOldHashTbl;
	FLMUINT		uiOldHashTblSize;
	RCACHE **	ppBucket;
	FLMUINT		uiLoop;
	RCACHE *		pTmpRCache;
	RCACHE *		pTmpNextRCache;

	uiNewHashTblSize = flmRcaGetBestHashTblSize(
									gv_FlmSysData.RCacheMgr.Usage.uiCount);

	// At this point we better have a different hash table size
	// or something is mucked up!

	flmAssert( uiNewHashTblSize !=
						gv_FlmSysData.RCacheMgr.uiNumBuckets);

	// Save the old hash table and its size.

	ppOldHashTbl = gv_FlmSysData.RCacheMgr.ppHashBuckets;
	uiOldHashTblSize = gv_FlmSysData.RCacheMgr.uiNumBuckets;

	// Allocate a new hash table.

	if (RC_BAD( rc = f_calloc(
								(FLMUINT)sizeof( RCACHE *) *
								(FLMUINT)uiNewHashTblSize,
								&gv_FlmSysData.RCacheMgr.ppHashBuckets)))
	{
		gv_FlmSysData.RCacheMgr.ppHashBuckets = ppOldHashTbl;
		goto Exit;
	}

	gv_FlmSysData.RCacheMgr.uiNumBuckets = uiNewHashTblSize;
	gv_FlmSysData.RCacheMgr.uiHashMask = uiNewHashTblSize - 1;

	// Relink all of the records into the new
	// hash table.

	for (uiLoop = 0, ppBucket = ppOldHashTbl;
		  uiLoop < uiOldHashTblSize;
		  uiLoop++, ppBucket++)
	{
		pTmpRCache = *ppBucket;
		while (pTmpRCache)
		{
			pTmpNextRCache = pTmpRCache->pNextInBucket;
			flmRcaLinkToHashBucket( pTmpRCache);
			pTmpRCache = pTmpNextRCache;
		}
	}

	// Throw away the old hash table.

	f_free( &ppOldHashTbl);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine changes the cache size for the record cache manager.
		If necessary, it will resize the hash table. NOTE: This routine
		assumes that the record cache mutex has been locked.
****************************************************************************/
FSTATIC RCODE flmRcaSetMemLimit(
	FLMUINT		uiMaxCacheBytes)
{
	RCODE			rc = FERR_OK;

	// If we are shrinking the maximum cache, clean up and
	// defragment cache first

	gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes = uiMaxCacheBytes;
	if (gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated >
				uiMaxCacheBytes)
	{
		flmRcaCleanupCache( ~((FLMUINT)0), TRUE);
	}

	// If the current record count is below the minimum records for the
	// number of buckets or is greater than the maximum records for the
	// number of buckets, we want to resize the hash table.

	if ((gv_FlmSysData.RCacheMgr.Usage.uiCount >
			FLM_RCA_MAX_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets < MAX_RCACHE_BUCKETS) ||
		 (gv_FlmSysData.RCacheMgr.Usage.uiCount <
			FLM_RCA_MIN_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets > MIN_RCACHE_BUCKETS))
	{
		if (RC_BAD( rc = flmRcaRehash()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine configures the record cache manager.  NOTE: This routine
		assumes that the record cache mutex has been locked.
****************************************************************************/
RCODE flmRcaConfig(
	FLMUINT		uiType,
	void *		Value1,
	void *		Value2)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( Value2);

	switch (uiType)
	{
		case FLM_CACHE_LIMIT:
			rc = flmRcaSetMemLimit( (FLMUINT)Value1);
			break;
		case FLM_SCACHE_DEBUG:
#ifdef FLM_DEBUG
			gv_FlmSysData.RCacheMgr.bDebug = (FLMBOOL)(Value1 ? 
															(FLMBOOL)TRUE : (FLMBOOL)FALSE);
#endif
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine shuts down the record cache manager and frees all
		resources allocated by it.
****************************************************************************/
void flmRcaExit( void)
{
	FLMUINT		uiCount;
	
	if (gv_FlmSysData.RCacheMgr.hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	// Free all of the record cache objects.

	uiCount = 0;
	while (gv_FlmSysData.RCacheMgr.pMRURecord)
	{
		if( (++uiCount & 0xFF) == 0)
		{
			f_yieldCPU();
		}
		
		flmRcaFreeCache( gv_FlmSysData.RCacheMgr.pMRURecord, FALSE);
	}

	// Must free those in the purge list too.

	uiCount = 0;
	while (gv_FlmSysData.RCacheMgr.pPurgeList)
	{
		if( (++uiCount & 0xFF) == 0)
		{
			f_yieldCPU();
		}
		flmRcaFreePurged( gv_FlmSysData.RCacheMgr.pPurgeList);
	}

	// The math better be consistent!

	flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiCount == 0);
	flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount == 0);
	flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes == 0);

	// Free the hash bucket array

	if (gv_FlmSysData.RCacheMgr.ppHashBuckets)
	{
		f_free( &gv_FlmSysData.RCacheMgr.ppHashBuckets);
	}

	// Free the mutex that controls access to record cache.
	// NOTE: This should be done last so that the mutex is not
	// unlocked until the very end.

	if (gv_FlmSysData.RCacheMgr.hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexDestroy( &gv_FlmSysData.RCacheMgr.hMutex);
	}

	// Free the allocators

	if( gv_FlmSysData.RCacheMgr.pRecBufAlloc)
	{
		gv_FlmSysData.RCacheMgr.pRecBufAlloc->Release();
		gv_FlmSysData.RCacheMgr.pRecBufAlloc = NULL;
	}

	if( gv_FlmSysData.RCacheMgr.pRecAlloc)
	{
		gv_FlmSysData.RCacheMgr.pRecAlloc->Release();
		gv_FlmSysData.RCacheMgr.pRecAlloc = NULL;
	}
	
	if( gv_FlmSysData.RCacheMgr.pRCacheAlloc)
	{
		gv_FlmSysData.RCacheMgr.pRCacheAlloc->Release();
		gv_FlmSysData.RCacheMgr.pRCacheAlloc = NULL;
	}

	// Zero the entire structure out, just for good measure.

	f_memset( &gv_FlmSysData.RCacheMgr, 0, sizeof( RCACHE_MGR));
}

/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read to complete.
		NOTE:  This routine assumes that the record cache mutex is already
		locked.
****************************************************************************/
FSTATIC void flmRcaNotify(
	F_NOTIFY_LIST_ITEM *	pNotify,
	RCACHE *					pUseRCache,
	RCODE						NotifyRc)
{
	while (pNotify)
	{
		F_SEM	hSem;

		*(pNotify->pRc) = NotifyRc;
		if (RC_OK( NotifyRc))
		{
			RCA_INCR_USE_COUNT( pUseRCache->uiFlags);
		}
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
}

/****************************************************************************
Desc:	Allocate a new record cache structure.  This routine assumes that the
		record cache mutex has already been locked.
****************************************************************************/
FSTATIC RCODE flmRcaAllocCacheStruct(
	RCACHE **		ppRCache)
{
	RCODE				rc = FERR_OK;
	
	f_assertMutexLocked( gv_FlmSysData.RCacheMgr.hMutex);
	
	if( (*ppRCache = 
		(RCACHE *)gv_FlmSysData.RCacheMgr.pRCacheAlloc->allocCell( 
		NULL, NULL)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	f_memset( *ppRCache, 0, sizeof( RCACHE));
	
	// Increment the total records cached

	gv_FlmSysData.RCacheMgr.Usage.uiCount++;

	// Set the high transaction ID to 0xFFFFFFFF so that this will NOT
	// be treated as one that had memory assigned to the old records.

	(*ppRCache)->uiHighTransId = 0xFFFFFFFF;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Free a record cache structure.  This routine assumes that the record
		cache mutex has already been locked.
****************************************************************************/
FSTATIC void flmRcaFreeCacheStruct(
	RCACHE **		ppRCache)
{
	flmAssert( !RCA_IS_IN_HEAP_LIST( (*ppRCache)->uiFlags));
	f_assertMutexLocked( gv_FlmSysData.RCacheMgr.hMutex);
	
	gv_FlmSysData.RCacheMgr.pRCacheAlloc->freeCell( *ppRCache);
	*ppRCache = NULL;	
	
	flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiCount > 0);
	gv_FlmSysData.RCacheMgr.Usage.uiCount--;
}

/****************************************************************************
Desc:	Cleanup old records in cache that are no longer needed by any
		transaction.
****************************************************************************/
void flmRcaCleanupCache(
	FLMUINT			uiMaxLockTime,
	FLMBOOL			bMutexesLocked)
{
	RCACHE *			pTmpRCache;
	RCACHE *			pPrevRCache;
	RCACHE *			pNextRCache;
	FLMUINT			uiRecordsExamined = 0;
	FLMUINT			uiLastTimePaused = FLM_GET_TIMER();
	FLMUINT			uiCurrTime;
	FLMBOOL			bUnlockMutexes = FALSE;

	if( !bMutexesLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		bUnlockMutexes = TRUE;
	}
	
	// Try to free everything in the heap list
	
	pTmpRCache = gv_FlmSysData.RCacheMgr.pHeapList;
	
	while( pTmpRCache)
	{
		uiRecordsExamined++;

		// Save the pointer to the next entry in the list because
		// we may end up unlinking pTmpRCache below

		pNextRCache = pTmpRCache->pNextInHeapList;

		// Determine if the item can be freed
		
		flmAssert( RCA_IS_IN_HEAP_LIST( pTmpRCache->uiFlags));
		
		if( !RCA_IS_IN_USE( pTmpRCache->uiFlags) &&
			 !RCA_IS_READING_IN( pTmpRCache->uiFlags))
		{
			flmRcaFreeCache( pTmpRCache, FALSE);
		}

		pTmpRCache = pNextRCache;
	}

	// Now, free any old versions that are no longer needed

	flmRcaReduceCache( TRUE);
	pTmpRCache = gv_FlmSysData.RCacheMgr.pLRURecord;

	// Stay in the loop until we have freed all old records, or
	// we have run through the entire list.

	for( ;;)
	{
		if( !pTmpRCache || !gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes)
		{
			break;
		}

		// After each 200 records examined, see if our maximum
		// time has elapsed for examining without a pause.

		if( uiRecordsExamined >= 200)
		{
			uiRecordsExamined = 0;
			uiCurrTime = FLM_GET_TIMER();
			
			if( FLM_ELAPSED_TIME( uiCurrTime, uiLastTimePaused) >= uiMaxLockTime)
			{
				// IMPORTANT! Don't stop and pause on one that is
				// being read in.

				while( pTmpRCache && RCA_IS_READING_IN( pTmpRCache->uiFlags))
				{
					pTmpRCache = pTmpRCache->pPrevInGlobal;
				}

				if( !pTmpRCache)
				{
					break;
				}

				if( bUnlockMutexes)
				{
					// Increment the use count so that this item will not
					// go away while we are paused.

					RCA_INCR_USE_COUNT( pTmpRCache->uiFlags);
					f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
					f_mutexUnlock( gv_FlmSysData.hShareMutex);

					// Shortest possible pause - to allow other threads
					// to do work.

					f_yieldCPU();

					// Relock the mutexes

					uiLastTimePaused = FLM_GET_TIMER();
					f_mutexLock( gv_FlmSysData.hShareMutex);
					f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

					// Decrement use count that was added above.

					RCA_DECR_USE_COUNT( pTmpRCache->uiFlags);
				}

				// If the item was purged while we were paused,
				// finish the job and then start again at the
				// top of the list.

				if( RCA_IS_PURGED( pTmpRCache->uiFlags))
				{
					if( !RCA_IS_IN_USE( pTmpRCache->uiFlags))
					{
						flmRcaFreePurged( pTmpRCache);
					}
					
					pTmpRCache = gv_FlmSysData.RCacheMgr.pLRURecord;
					continue;
				}
			}
		}

		uiRecordsExamined++;

		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pTmpRCache below, in which case we would
		// have lost the previous entry.

		pPrevRCache = pTmpRCache->pPrevInGlobal;

		// Block must not currently be in use,
		// Must not be the most current version of a block,
		// Cannot be dirty in any way,
		// Cannot be in the process of being read in from disk,
		// And must not be needed by a read transaction.

		if( !RCA_IS_IN_USE( pTmpRCache->uiFlags) &&
			 !RCA_IS_READING_IN( pTmpRCache->uiFlags) &&
			 (RCA_IS_IN_HEAP_LIST( pTmpRCache->uiFlags) ||
				(pTmpRCache->uiHighTransId != 0xFFFFFFFF &&
					!flmNeededByReadTrans( pTmpRCache->pFile,
						pTmpRCache->uiLowTransId,
						pTmpRCache->uiHighTransId))))
		{
			flmRcaFreeCache( pTmpRCache, FALSE);
		}

		pTmpRCache = pPrevRCache;
	}

	// Defragment memory

	gv_FlmSysData.RCacheMgr.pRCacheAlloc->defragmentMemory();
	gv_FlmSysData.RCacheMgr.pRecAlloc->defragmentMemory();
	gv_FlmSysData.RCacheMgr.pRecBufAlloc->defragmentMemory();

	// Unlock the mutexes

	if( bUnlockMutexes)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	This routine reduces record cache down to the limit expected.
****************************************************************************/
void flmRcaReduceCache(
	FLMBOOL		bMutexAlreadyLocked)
{
	RCACHE *		pRCache;
	RCACHE *		pPrevRCache;

	// Make sure the mutex is locked.

	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	pRCache = gv_FlmSysData.RCacheMgr.pLRURecord;

	// Free things until we get down below our memory limit.

	while( gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated >
		gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes)
	{
		if( !pRCache)
		{
			break;
		}

		// If the total of block and record cache is below the global
		// cache maximum, there is no need to reduce record cache.

		if( (gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated + 
			gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated) <=
			gv_FlmSysData.uiMaxCache)
		{
			break;
		}

		pPrevRCache = pRCache->pPrevInGlobal;
		if (!(RCA_IS_IN_USE( pRCache->uiFlags)) &&
			 !(RCA_IS_READING_IN( pRCache->uiFlags)))
		{
			flmRcaFreeCache( pRCache, FALSE);
		}
		pRCache = pPrevRCache;
	}

	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}
}

/****************************************************************************
Desc:	This routine finds a record in the record cache.  If it cannot
		find the record, it will return the position where the record should
		be inserted.  NOTE: This routine assumes that the record cache mutex
		has been locked.
****************************************************************************/
void flmRcaFindRec(
	FFILE *			pFile,
	F_SEM				hWaitSem,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn,
	FLMUINT			uiVersionNeeded,
	FLMBOOL			bDontPoisonCache,
	FLMUINT *		puiNumLooks,
	RCACHE **		ppRCache,
	RCACHE **		ppNewerRCache,
	RCACHE **		ppOlderRCache)
{
	RCACHE *			pRCache;
	FLMUINT			uiNumLooks = 0;
	FLMBOOL			bFound;
	RCACHE *			pNewerRCache;
	RCACHE *			pOlderRCache;

	// Search down the hash bucket for the matching item.

Start_Find:

	// NOTE: Need to always calculate hash bucket because
	// the hash table may have been changed while we
	// were waiting to be notified below - mutex can
	// be unlocked, but it is guaranteed to be locked
	// here.

	pRCache = *(FLM_RCA_HASH( uiDrn));
	bFound = FALSE;
	uiNumLooks = 1;
	while ((pRCache) &&
			 ((pRCache->uiDrn != uiDrn) ||
			  (pRCache->uiContainer != uiContainer) ||
			  (pRCache->pFile != pFile)))
	{
		if ((pRCache = pRCache->pNextInBucket) != NULL)
		{
			uiNumLooks++;
		}
	}

	// If we found the record, see if we have the right version.

	if (!pRCache)
	{
		pNewerRCache = pOlderRCache = NULL;
	}
	else
	{
		pNewerRCache = NULL;
		pOlderRCache = pRCache;
		
		for (;;)
		{
			// If this one is being read in, we need to wait on it.

			if (RCA_IS_READING_IN( pRCache->uiFlags))
			{
				// We need to wait for this record to be read in
				// in case it coalesces with other versions, resulting
				// in a version that satisfies our request.

				gv_FlmSysData.RCacheMgr.uiIoWaits++;
				if( RC_BAD( f_notifyWait( gv_FlmSysData.RCacheMgr.hMutex, 
					hWaitSem, NULL, &pRCache->pNotifyList)))
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

				RCA_DECR_USE_COUNT( pRCache->uiFlags);

				if (RCA_IS_PURGED( pRCache->uiFlags))
				{
					if (!RCA_IS_IN_USE( pRCache->uiFlags))
					{
						flmRcaFreePurged( pRCache);
					}
				}

				// Start over with the find because the list
				// structure has changed.

				goto Start_Find;
			}

			// See if this record version is the one we need.

			if (uiVersionNeeded < pRCache->uiLowTransId)
			{
				pNewerRCache = pRCache;
				if ((pOlderRCache = pRCache = pRCache->pOlderVersion) == NULL)
				{
					break;
				}
				uiNumLooks++;
			}
			else if (uiVersionNeeded <= pRCache->uiHighTransId)
			{
				// Make this the MRU record.

				if (puiNumLooks)
				{
					if (bDontPoisonCache)
					{
						flmRcaStepUpInGlobalList( pRCache);
					}
					else if (pRCache->pPrevInGlobal)
					{
						flmRcaUnlinkFromGlobal( pRCache);
						flmRcaLinkToGlobalAsMRU( pRCache);
					}
					
					gv_FlmSysData.RCacheMgr.Usage.uiCacheHits++;
					gv_FlmSysData.RCacheMgr.Usage.uiCacheHitLooks += uiNumLooks;
				}
				
				bFound = TRUE;
				break;
			}
			else
			{
				pOlderRCache = pRCache;
				pNewerRCache = pRCache->pNewerVersion;

				// Set pRCache to NULL as an indicator that we did not
				// find the version we needed.

				pRCache = NULL;
				break;
			}
		}
	}

	*ppRCache = pRCache;
	*ppOlderRCache = pOlderRCache;
	*ppNewerRCache = pNewerRCache;

	if (puiNumLooks)
	{
		*puiNumLooks = uiNumLooks;
	}
}

/****************************************************************************
Desc:	This routine replaces the FlmRecord that a record is pointing to
		with a new one.  NOTE: This routine assumes that the record cache
		mutex is already locked.
****************************************************************************/
FSTATIC void flmRcaSetRecord(
	RCACHE *				pRCache,
	FlmRecord *			pNewRecord)
{
	FLMUINT		uiTotalMemory = 0;
	FLMUINT		uiFreeMemory;
	FlmRecord *	pOldRecord;

	// Release the cache's pointer to the old record data

	if ((pOldRecord = pRCache->pRecord) != NULL)
	{
		if( RCA_IS_IN_HEAP_LIST( pRCache->uiFlags))
		{
			flmRcaUnlinkFromHeapList( pRCache);
		}
		
		uiTotalMemory = pOldRecord->getTotalMemory();
		uiFreeMemory = pOldRecord->getFreeMemory();
		flmAssert( uiTotalMemory >= uiFreeMemory);
		FlmRecordExt::clearCached( pOldRecord);
		FlmRecordExt::Release( pOldRecord, TRUE);
		pRCache->pRecord = NULL;
	}

	if (pRCache->uiHighTransId != 0xFFFFFFFF)
	{
		FLMUINT	uiTotalOldMemory = uiTotalMemory + sizeof( RCACHE);

		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes >=
			uiTotalOldMemory);
		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount);
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes -=
			uiTotalOldMemory;
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount--;
	}

	// Point to the new record data.

	flmAssert( pNewRecord->getID() == pRCache->uiDrn);
	flmAssert( pNewRecord->getContainerID() == pRCache->uiContainer);
	pRCache->pRecord = pNewRecord;
	flmAssert( !pNewRecord->isCached());
	FlmRecordExt::setCached( pNewRecord);
	FlmRecordExt::setReadOnly( pNewRecord);
	pNewRecord->AddRef();
	
	if( FlmRecordExt::getFlags( pNewRecord) & RCA_HEAP_BUFFER)
	{
		flmRcaLinkToHeapList( pRCache);
	}

	uiTotalMemory = pNewRecord->getTotalMemory();

	if (pRCache->uiHighTransId != 0xFFFFFFFF)
	{
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes +=
			(uiTotalMemory + sizeof( RCACHE));
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerCount++;
	}
	uiFreeMemory = pNewRecord->getFreeMemory();
	flmAssert( uiTotalMemory >= uiFreeMemory);
}

/****************************************************************************
Desc:	This routine links a new RCACHE structure into the global list and
		into the correct place in its hash bucket.  This routine assumes that
		the record cache mutex is already locked.
****************************************************************************/
FSTATIC void flmRcaLinkIntoRCache(
	RCACHE *			pNewerRCache,
	RCACHE *			pOlderRCache,
	RCACHE *			pRCache,
	FLMBOOL			bLinkAsMRU)
{
	if( bLinkAsMRU)
	{
		flmRcaLinkToGlobalAsMRU( pRCache);
	}
	else
	{
		flmRcaLinkToGlobalAsLRU( pRCache);
	}

	if (pNewerRCache)
	{
		flmRcaLinkToVerList( pRCache, pNewerRCache, pOlderRCache);
	}
	else
	{
		RCACHE *		pNull = NULL;

		if (pOlderRCache)
		{
			flmRcaUnlinkFromHashBucket( pOlderRCache);
		}
		flmRcaLinkToHashBucket( pRCache);
		flmRcaLinkToVerList( pRCache, pNull, pOlderRCache);
	}
}

/****************************************************************************
Desc:	This routine links a new record to its FFILE according to whether
		or not it is an update transaction or a read transaction.
		It coalesces out any unnecessary versions. This routine assumes 
		that the record cache mutex is already locked.
****************************************************************************/
FSTATIC void flmRcaLinkToFFILE(
	RCACHE *			pRCache,
	FFILE *			pFile,
	FDB *				pDb,
	FLMUINT			uiLowTransId,
	FLMBOOL			bMostCurrent)
{
	RCACHE *		pTmpRCache;
#ifdef FLM_DBG_LOG
	char			szTmpBuf[ 80];
#endif


	pRCache->uiLowTransId = uiLowTransId;

	// Before coalescing, link to FFILE.
	// The following test determines if the record is an
	// uncommitted version generated by the update transaction.
	// If so, we mark it as such, and link it at the head of the
	// FFILE list - so we can get rid of it quickly if we abort
	// the transaction.

	if (flmGetDbTransType( pDb) == FLM_UPDATE_TRANS)
	{

		// If we are in an update transaction, there better not
		// be any newer versions in the list and the high
		// transaction ID returned better be 0xFFFFFFFF.

		flmAssert( pRCache->pNewerVersion == NULL);
		flmRcaSetTransID( pRCache, 0xFFFFFFFF);

		// If the low transaction ID is the same as the transaction,
		// we may have modified this record during the transaction.
		// Unfortunately, there is no sure way to tell, so we are
		// forced to assume it may have been modified.  If the
		// transaction aborts, we will get rid if this version out
		// of cache.

		if (uiLowTransId == pDb->LogHdr.uiCurrTransID)
		{
			RCA_SET_UNCOMMITTED( pRCache->uiFlags);
			flmRcaLinkToFileAtHead( pRCache, pFile);
		}
		else
		{
			RCA_UNSET_UNCOMMITTED( pRCache->uiFlags);
			flmRcaLinkToFileAtEnd( pRCache, pFile);
		}
#ifdef FLM_DBG_LOG
		f_sprintf( szTmpBuf, "RCI:L%X,H%X",
			(unsigned)pRCache->uiLowTransId,
			(unsigned)pRCache->uiHighTransId);

		flmDbgLogWrite( pFile->uiFFileId, pRCache->uiContainer, pRCache->uiDrn, 
			pDb->LogHdr.uiCurrTransID, szTmpBuf);
#endif
	}
	else
	{
		// Adjust the high transaction ID to be the same as
		// the transaction ID - we may have gotten a 0xFFFFFFF
		// back, but that is possible even if the record is
		// not the most current version.  Besides that, it is
		// possible that in the mean time one or more update
		// transactions have come along and created one or
		// more newer versions of the record.

		FLMUINT uiHighTransId = (FLMUINT)((bMostCurrent)
													  ? (FLMUINT)0xFFFFFFFF
													  : pDb->LogHdr.uiCurrTransID);

		flmRcaSetTransID( pRCache, uiHighTransId);

		// For a read transaction, if there is a newer version,
		// it better have a higher "low transaction ID"

#ifdef FLM_DBG_LOG
		f_sprintf( szTmpBuf, "RCA:L%X,H%X",
			(unsigned)pRCache->uiLowTransId,
			(unsigned)pRCache->uiHighTransId);

		flmDbgLogWrite( pFile->uiFFileId, pRCache->uiContainer, pRCache->uiDrn, 
			pDb->LogHdr.uiCurrTransID, szTmpBuf);
#endif

#ifdef FLM_DEBUG
		if (pRCache->pNewerVersion &&
			 !RCA_IS_READING_IN( pRCache->pNewerVersion->uiFlags))
		{
			flmAssert( pRCache->uiHighTransId <
						  pRCache->pNewerVersion->uiLowTransId);
			if( pRCache->uiHighTransId >=
				pRCache->pNewerVersion->uiLowTransId)
			{
				flmRcaCheck( pDb, pRCache->uiContainer, pRCache->uiDrn);
			}
		}
#endif
		RCA_UNSET_UNCOMMITTED( pRCache->uiFlags);
		flmRcaLinkToFileAtEnd( pRCache, pFile);
	}

	// Coalesce any versions that overlap - can only
	// coalesce older versions.  For an updater, there
	// should not be any newer versions.  For a reader, it
	// is impossible to know how high up it can coalesce.
	// The read operation that read the record may have
	// gotten back a 0xFFFFFFFF for its high transaction
	// ID - but after that point in time, it is possible
	// that one or more update transactions may have come
	// along and created one or more newer versions that
	// it would be incorrect to coalesce with.
	// In reality, a read transaction has to ignore the
	// 0xFFFFFFFF in the high transaction ID anyway
	// because there is no way to know if it is correct.

	// Coalesce older versions.

	for (;;)
	{
		if ((pTmpRCache = pRCache->pOlderVersion) == NULL)
		{
			break;
		}

		// Stop if we encounter one that is being read in.

		if (RCA_IS_READING_IN( pTmpRCache->uiFlags))
		{
			break;
		}

		// If there is no overlap between these two, there is
		// nothing more to coalesce.

		if (pRCache->uiLowTransId > pTmpRCache->uiHighTransId)
		{
			break;
		}

		if (pRCache->uiHighTransId <= pTmpRCache->uiHighTransId)
		{
			// This assert represents the following case,
			// which should not be possible to hit:
			
			// pOlder->uiHighTransId > pRCache->uiHighTransId.
			//	This cannot be, because if pOlder has a higher
			//	transaction ID, we would have found it up above and
			//	not tried to have read it in.

			flmAssert( 0);
#ifdef FLM_DEBUG
			flmRcaCheck( pDb, pRCache->uiContainer, pRCache->uiDrn);
#endif
		}
		else if (pRCache->uiLowTransId >= pTmpRCache->uiLowTransId)
		{
			pRCache->uiLowTransId = pTmpRCache->uiLowTransId;
			flmRcaFreeCache( pTmpRCache,
						(FLMBOOL)((RCA_IS_IN_USE( pTmpRCache->uiFlags) ||
									  RCA_IS_READING_IN( pTmpRCache->uiFlags))
									 ? TRUE
									 : FALSE));
		}
		else
		{
			// This assert represents the following case,
			// which should not be possible to hit:
			
			// pRCache->uiLowTransId < pOlder->uiLowTransId.
			//	This cannot be, because pOlder has to have been read
			//	in to memory by a transaction whose transaction ID is
			//	less than or equal to our own.  That being the case,
			//	it would be impossible for our transaction to have
			//	found a version of the record that is older than pOlder.

			flmAssert( 0);
#ifdef FLM_DEBUG
			flmRcaCheck( pDb, pRCache->uiContainer, pRCache->uiDrn);
#endif
		}
	}
}

/****************************************************************************
Desc:	This routine retrieves a record from the record cache.
****************************************************************************/
RCODE flmRcaRetrieveRec(
	FDB *						pDb,
	FLMBOOL *				pbTransStarted,
	FLMUINT					uiContainer,			// Container record is in.
	FLMUINT					uiDrn,					// DRN of record.
	FLMBOOL					bOkToGetFromDisk,		// If not in cache, OK to get from disk?
	BTSK *					pStack,					// Use stack to retrieve, if NON-NULL.
	LFILE *					pLFile,					// LFILE to use, if retrieving with stack
	FlmRecord **			ppRecord)
{
	RCODE						rc = FERR_OK;
	FLMBOOL					bRCacheMutexLocked = FALSE;
	FFILE *					pFile = pDb->pFile;
	RCACHE *					pRCache;
	RCACHE *					pNewerRCache;
	RCACHE *					pOlderRCache;
	FlmRecord *				pRecord = NULL;
	FLMBOOL					bGotFromDisk = FALSE;
	FlmRecord *				pNewRecord = NULL;
	FlmRecord *				pOldRecord = NULL;
	FLMUINT					uiLowTransId;
	FLMBOOL					bMostCurrent;
	FLMUINT					uiCurrTransId;
	F_NOTIFY_LIST_ITEM *	pNotify;
	FLMUINT					uiNumLooks;
	FLMBOOL					bInitializedFdb = FALSE;
	FLMBOOL					bDontPoisonCache = pDb->uiFlags & FDB_DONT_POISON_CACHE
														? TRUE 
														: FALSE;

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// Get the current transaction ID

	if( pDb->uiTransType != FLM_NO_TRANS)
	{
		uiCurrTransId = pDb->LogHdr.uiCurrTransID;
	}
	else
	{
		flmAssert( pbTransStarted != NULL);

		f_mutexLock( gv_FlmSysData.hShareMutex);

		// Get the last committed transaction ID.

		uiCurrTransId = (FLMUINT)FB2UD(
								&pFile->ucLastCommittedLogHdr[ LOG_CURR_TRANS_ID]);

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	flmAssert( uiDrn != 0);

	// Lock the mutex

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	bRCacheMutexLocked = TRUE;

	// Reset the FDB's inactive time

	pDb->uiInactiveTime = 0;

	// See if we should resize the hash table.

	if ((gv_FlmSysData.RCacheMgr.Usage.uiCount >
			FLM_RCA_MAX_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets < MAX_RCACHE_BUCKETS) ||
		 (gv_FlmSysData.RCacheMgr.Usage.uiCount <
			FLM_RCA_MIN_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets > MIN_RCACHE_BUCKETS))
	{
		if (RC_BAD( rc = flmRcaRehash()))
		{
			goto Exit;
		}
	}

Start_Find:

	flmRcaFindRec( pFile, pDb->hWaitSem, uiContainer, uiDrn, 
		uiCurrTransId, bDontPoisonCache, &uiNumLooks, &pRCache, 
		&pNewerRCache, &pOlderRCache);

	if (pRCache)
	{
		if( ppRecord)
		{
			goto Found_Record;
		}
		goto Exit;
	}

	// Did not find the record, fetch from disk, if OK to do so.

	if (!bOkToGetFromDisk || !ppRecord)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Code to handle case where we are not in a transaction.
	// If we are already in a transaction, we will do the
	// call to fdbInit below - AFTER allocating the RCACHE, etc.

	if( pbTransStarted && pDb->uiTransType == FLM_NO_TRANS)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		bRCacheMutexLocked = FALSE;

		if ( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
				FDB_TRANS_GOING_OK, 0, pbTransStarted)))
		{
			fdbExit( pDb);
			goto Exit;
		}
		bInitializedFdb = TRUE;

		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		bRCacheMutexLocked = TRUE;

		uiCurrTransId = pDb->LogHdr.uiCurrTransID;
		goto Start_Find;
	}

	// Increment the number of faults only if we retrieve the record from disk.

	gv_FlmSysData.RCacheMgr.Usage.uiCacheFaults++;
	gv_FlmSysData.RCacheMgr.Usage.uiCacheFaultLooks += uiNumLooks;

	// Create a place holder for the object.

	if (RC_BAD( rc = flmRcaAllocCacheStruct( &pRCache)))
	{
		goto Exit;
	}
	pRCache->uiDrn = uiDrn;
	pRCache->uiContainer = uiContainer;

	// Set the FFILE so that other threads looking for this record in
	// cache will find it and wait until the read has completed.  If
	// the FFILE is not set, other threads will attempt their own read,
	// because they won't match a NULL FFILE.  The result of not setting
	// the FFILE is that multiple copies of the same version of a particular
	// record could end up in cache.
	
	pRCache->pFile = pFile;

	flmRcaLinkIntoRCache( pNewerRCache, pOlderRCache, 
		pRCache, !bDontPoisonCache);

	RCA_SET_READING_IN( pRCache->uiFlags);
	RCA_INCR_USE_COUNT( pRCache->uiFlags);
	pRCache->pNotifyList = NULL;

	// Unlock mutex before reading in from disk.

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	bRCacheMutexLocked = FALSE;

	if( pbTransStarted && !bInitializedFdb)
	{
		if ( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
				FDB_TRANS_GOING_OK, 0, pbTransStarted)))
		{
			fdbExit( pDb);
			goto Notify_Waiters;
		}
		bInitializedFdb = TRUE;
	}

	// Read record from disk.

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( pFile->uiFFileId, uiContainer,
		uiDrn, pDb->LogHdr.uiCurrTransID, "RRD");
#endif

	if (pStack)
	{
		rc = FSReadElement( pDb, &pDb->TempPool, pLFile, 
									uiDrn, pStack, TRUE, ppRecord, 
									&uiLowTransId, &bMostCurrent);
	}
	else if ((pLFile) ||
				(RC_OK( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile))))
	{
		rc = FSReadRecord( pDb, pLFile, uiDrn, ppRecord, 
									&uiLowTransId, &bMostCurrent);
	}

Notify_Waiters:

	// Relock mutex

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	bRCacheMutexLocked = TRUE;

	// If read was successful, link the record to its place in
	// the FFILE list and coalesce any versions that overlap
	// this one.

	if (RC_OK( rc))
	{
		flmRcaLinkToFFILE( pRCache,
				pDb->pFile, pDb, uiLowTransId, bMostCurrent);
	}

	RCA_UNSET_READING_IN( pRCache->uiFlags);

	// Notify any threads waiting for the read to complete.

	pNotify = pRCache->pNotifyList;
	pRCache->pNotifyList = NULL;
	if (pNotify)
	{
		flmRcaNotify( pNotify,
				(RCACHE *)((RC_BAD( rc))
							  ? NULL
							  : pRCache), rc);
	}
	RCA_DECR_USE_COUNT( pRCache->uiFlags);

	// If we did not succeed, free the RCACHE structure.

	if (RC_BAD( rc))
	{
		flmRcaFreeCache( pRCache, FALSE);
		goto Exit;
	}

	// If this item was purged while we were reading it in,
	// start over with the search.

	if (RCA_IS_PURGED( pRCache->uiFlags))
	{
		if (!RCA_IS_IN_USE( pRCache->uiFlags))
		{
			flmRcaFreePurged( pRCache);
		}

		// Start over with the find - this one has
		// been marked for purging.

		pNewRecord = NULL;
		pOldRecord = NULL;
		bGotFromDisk = FALSE;
		goto Start_Find;
	}

	// When reading from disk, no need to verify that we
	// are getting the app implementation - because we
	// always will.

	bGotFromDisk = TRUE;
	pRecord = *ppRecord;
	flmRcaSetRecord( pRCache, pRecord);
	
Found_Record:

	if (!bGotFromDisk)
	{
		if( (pRecord = *ppRecord) != pRCache->pRecord)
		{
			if (*ppRecord)
			{
				FlmRecordExt::Release( *ppRecord, bRCacheMutexLocked);
			}

			pRecord = *ppRecord = pRCache->pRecord;
			pRecord->AddRef();
		}
	}

	// Clean up cache, if necessary.

	if (gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated >
			 gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes)
	{
		flmRcaReduceCache( bRCacheMutexLocked);
	}

Exit:

	if (pNewRecord)
	{
		FlmRecordExt::Release( pNewRecord, bRCacheMutexLocked);
	}

	if (bRCacheMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	if (bInitializedFdb)
	{
		fdbExit(pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine inserts a record into the record cache.  This is ONLY
		called by FlmRecordModify or FlmRecordAdd.  In the case of a modify,
		this should replace any uncommitted version of the record that may
		have been put there by a prior call to FlmRecordModify.
****************************************************************************/
RCODE flmRcaInsertRec(
	FDB *				pDb,
	LFILE *			pLFile,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiContainer = pLFile->uiLfNum;
	FLMBOOL			bMutexLocked = FALSE;
	RCACHE *			pRCache;
	RCACHE *			pNewerRCache;
	RCACHE *			pOlderRCache;
	FLMBOOL			bDontPoisonCache = pDb->uiFlags & FDB_DONT_POISON_CACHE
														? TRUE
														: FALSE;

	if (pLFile->bMakeFieldIdTable && !pRecord->fieldIdTableEnabled())
	{
		
		// NOTE: createFieldIdTable will call sortFieldIdTable().
		
		if (RC_BAD( rc = pRecord->createFieldIdTable( TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		pRecord->sortFieldIdTable();
		if (getFieldIdTableItemCount( pRecord->getFieldIdTbl()) !=
			 getFieldIdTableArraySize( pRecord->getFieldIdTbl()))
		{
			if (RC_BAD( rc = pRecord->truncateFieldIdTable()))
			{
				goto Exit;
			}
		}
	}
	flmAssert( uiDrn != 0);
	
	// Lock the mutex

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	bMutexLocked = TRUE;

	if ((gv_FlmSysData.RCacheMgr.Usage.uiCount >
			FLM_RCA_MAX_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets < MAX_RCACHE_BUCKETS) ||
		 (gv_FlmSysData.RCacheMgr.Usage.uiCount <
			FLM_RCA_MIN_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets > MIN_RCACHE_BUCKETS))
	{
		if (RC_BAD( rc = flmRcaRehash()))
		{
			goto Exit;
		}
	}

	// See if we can find the record in cache

	flmRcaFindRec( pFile, pDb->hWaitSem, uiContainer, uiDrn,
						pDb->LogHdr.uiCurrTransID,	bDontPoisonCache, NULL, &pRCache,
						&pNewerRCache, &pOlderRCache);

	if (pRCache)
	{

		// If we found the last committed version, instead of replacing it,
		// we want to change its high transaction ID, and go create a new
		// record to put in cache.

		if (pRCache->uiLowTransId < pDb->LogHdr.uiCurrTransID)
		{

			// pOlderRCache and pRCache should be the same at this point if we
			// found something.  Furthermore, the high transaction ID on what
			// we found better be -1 - most current version.

			flmAssert( pOlderRCache == pRCache);
			flmAssert( pOlderRCache->uiHighTransId == 0xFFFFFFFF);

			flmRcaSetTransID( pOlderRCache, (pDb->LogHdr.uiCurrTransID - 1));

			flmAssert( pOlderRCache->uiHighTransId >= pOlderRCache->uiLowTransId);

			RCA_SET_UNCOMMITTED( pOlderRCache->uiFlags);
			RCA_SET_LATEST_VER( pOlderRCache->uiFlags);
			flmRcaUnlinkFromFile( pOlderRCache);
			flmRcaLinkToFileAtHead( pOlderRCache, pFile);
		}
		else
		{
			// Found latest UNCOMMITTED VERSION - replace it.

			if (RC_BAD( rc = pRecord->compressMemory()))
			{
				goto Exit;
			}

			// Replace the old record data with the new record data.

			flmRcaSetRecord( pRCache, pRecord);

			// Make sure we set the "uncommitted" flag and move the record
			// to the head of the FFILE's list.

			if (!RCA_IS_UNCOMMITTED( pRCache->uiFlags))
			{
				RCA_SET_UNCOMMITTED( pRCache->uiFlags);
				flmRcaUnlinkFromFile( pRCache);
				flmRcaLinkToFileAtHead( pRCache, pFile);
			}

			// Will not have already been put at MRU if bDonPoisonCache is TRUE.

			if (pRCache->pPrevInGlobal)
			{
				flmRcaUnlinkFromGlobal( pRCache);
				flmRcaLinkToGlobalAsMRU( pRCache);
			}
			
			goto Exit;
		}
	}

	// We are positioned to insert the new record.  For an update, it
	// must always be the newest version.

	flmAssert( !pNewerRCache);

	if (RC_BAD( rc = pRecord->compressMemory()))
	{
		goto Exit;
	}

	// Allocate a new RCACHE structure.

	if (RC_BAD( rc = flmRcaAllocCacheStruct( &pRCache)))
	{
		goto Exit;
	}

	// Set the DRN and container for the structure.

	pRCache->uiDrn = uiDrn;
	pRCache->uiContainer = uiContainer;
	pRCache->pFile = pFile;

	// Link into the global list and hash bucket.

	flmRcaLinkIntoRCache( pNewerRCache, pOlderRCache, pRCache, TRUE);

	// Link to its FFILE and coalesce out duplicates.
	// NOTE: This routine automatically puts an uncommitted version
	// at the head of the FFILE's list and sets the uncommitted flag.

	flmRcaLinkToFFILE( pRCache, pFile, pDb, pDb->LogHdr.uiCurrTransID, TRUE);

	// Set the record data pointer for the RCACHE structure.  This will also
	// release the old data pointer and set the read only flag and the mutex
	// for the record data. Also updates memory usage fields in RCacheMgr.

	flmRcaSetRecord( pRCache, pRecord);

	// Clean up cache, if necessary.

	flmRcaReduceCache( bMutexLocked);

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called by FlmRecordDelete to remove a record from
		cache.  If there is an uncommitted version of the record, it should
		remove that version from cache.  If the last committed version is in
		cache, it should set the high transaction ID on that version to be
		one less than the transaction ID of the update transaction that is
		doing the FlmRecordDelete call.
****************************************************************************/
RCODE flmRcaRemoveRec(
	FDB *			pDb,
	FLMUINT		uiContainer,
	FLMUINT		uiDrn)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bMutexLocked = FALSE;
	RCACHE *		pRCache;
	RCACHE *		pNewerRCache;
	RCACHE *		pOlderRCache;
	FFILE *		pFile = pDb->pFile;

	flmAssert( uiDrn != 0);

	// Lock the semaphore

	bMutexLocked = TRUE;
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

	if ((gv_FlmSysData.RCacheMgr.Usage.uiCount >
			FLM_RCA_MAX_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets < MAX_RCACHE_BUCKETS) ||
		 (gv_FlmSysData.RCacheMgr.Usage.uiCount <
			FLM_RCA_MIN_REC_CNT( gv_FlmSysData.RCacheMgr.uiNumBuckets) &&
		  gv_FlmSysData.RCacheMgr.uiNumBuckets > MIN_RCACHE_BUCKETS))
	{
		if (RC_BAD( rc = flmRcaRehash()))
		{
			goto Exit;
		}
	}

	// See if we can find the record in cache

	flmRcaFindRec( pFile, pDb->hWaitSem, uiContainer, uiDrn,
						pDb->LogHdr.uiCurrTransID, FALSE, NULL, &pRCache,
						&pNewerRCache, &pOlderRCache);

	if (pRCache)
	{
		// FlmRecordDelete is calling this routine, so we determine if we found
		// the last committed version or a record that was added by this same
		// transaction.  If we found the last committed version, set its high
		// transaction ID.  Otherwise, remove the record from cache.

		if (pRCache->uiLowTransId < pDb->LogHdr.uiCurrTransID)
		{

			// pOlderRCache and pRCache should be the same at this point if we
			// found something.  Furthermore, the high transaction ID on what
			// we found better be -1 - most current version.

			flmAssert( pOlderRCache == pRCache);
			flmAssert( pOlderRCache->uiHighTransId == 0xFFFFFFFF);

			flmRcaSetTransID( pOlderRCache, (pDb->LogHdr.uiCurrTransID - 1));
			flmAssert( pOlderRCache->uiHighTransId >= pOlderRCache->uiLowTransId);
			RCA_SET_UNCOMMITTED( pOlderRCache->uiFlags);
			RCA_SET_LATEST_VER( pOlderRCache->uiFlags);
			flmRcaUnlinkFromFile( pOlderRCache);
			flmRcaLinkToFileAtHead( pOlderRCache, pFile);
		}
		else
		{
			flmRcaFreeCache( pRCache,
						(FLMBOOL)(RCA_IS_IN_USE( pRCache->uiFlags)
									 ? TRUE
									 : FALSE));
		}
	}

	// Take the opportunity to clean up cache.

	flmRcaReduceCache( bMutexLocked);

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called when an FFILE structure is going to be removed
		from the shared memory area.  At that point, we also need to get rid
		of all records that have been cached for that FFILE.
****************************************************************************/
void flmRcaFreeFileRecs(
	FFILE *			pFile)
{
	FLMUINT	uiNumFreed = 0;

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	while (pFile->pFirstRecord)
	{
		flmRcaFreeCache( pFile->pFirstRecord,
						RCA_IS_IN_USE( pFile->pFirstRecord->uiFlags)
								? TRUE
								: FALSE);

		// Release the CPU every 100 records freed.

		if (uiNumFreed < 100)
		{
			uiNumFreed++;
		}
		else
		{
			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_yieldCPU();
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
			uiNumFreed = 0;
		}
	}
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction aborts.  At that
		point, we need to get rid of any uncommitted versions of records in
		the record cache.
****************************************************************************/
void flmRcaAbortTrans(
	FDB *			pDb)
{
	FFILE *		pFile = pDb->pFile;
	RCACHE *		pRCache;
	RCACHE *		pOlderVersion;
	FLMUINT		uiOlderTransId = pDb->LogHdr.uiCurrTransID - 1;

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	
	pRCache = pFile->pFirstRecord;
	while (pRCache)
	{
		if (RCA_IS_UNCOMMITTED( pRCache->uiFlags))
		{
			if (RCA_IS_LATEST_VER( pRCache->uiFlags))
			{
				flmRcaSetTransID( pRCache, 0xFFFFFFFF);
				RCA_UNSET_UNCOMMITTED( pRCache->uiFlags);
				RCA_UNSET_LATEST_VER( pRCache->uiFlags);
				flmRcaUnlinkFromFile( pRCache);
				flmRcaLinkToFileAtEnd( pRCache, pFile);
			}
			else
			{
				// Save the older version - we may be changing its
				// high transaction ID back to 0xFFFFFFFF

				pOlderVersion = pRCache->pOlderVersion;

				// Free the uncommitted version.

				flmRcaFreeCache( pRCache,
						(FLMBOOL)((RCA_IS_IN_USE( pRCache->uiFlags) ||
									  RCA_IS_READING_IN( pRCache->uiFlags))
									 ? TRUE
									 : FALSE));

				// If the older version has a high transaction ID that
				// is exactly one less than our current transaction,
				// it is the most current version.  Hence, we need to
				// change its high transaction ID back to 0xFFFFFFFF.

				if ((pOlderVersion) &&
					 (pOlderVersion->uiHighTransId == uiOlderTransId))
				{
					flmRcaSetTransID( pOlderVersion, 0xFFFFFFFF);
				}
			}
			pRCache = pFile->pFirstRecord;
		}
		else
		{
			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
}

/****************************************************************************
Desc:	This routine is called when an update transaction commits.  At that
		point, we need to unset the "uncommitted" flag on any records
		currently in record cache for the FFILE.
****************************************************************************/
void flmRcaCommitTrans(
	FDB *			pDb)
{
	RCACHE *		pRCache;

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	
	pRCache = pDb->pFile->pFirstRecord;
	while (pRCache)
	{
		if (RCA_IS_UNCOMMITTED( pRCache->uiFlags))
		{
			RCA_UNSET_UNCOMMITTED( pRCache->uiFlags);
			RCA_UNSET_LATEST_VER( pRCache->uiFlags);
			pRCache = pRCache->pNextInFile;
		}
		else
		{

			// We can stop when we hit a committed version, because
			// uncommitted versions are always linked in together at
			// the head of the list.

			break;
		}
	}
	
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
}

/****************************************************************************
Desc:	This routine is called when a container in the database is deleted.
		All records in record cache that are in that container must be
		removed from cache.
****************************************************************************/
void flmRcaRemoveContainerRecs(
	FDB *			pDb,
	FLMUINT		uiContainer)
{
	FFILE *		pFile = pDb->pFile;
	RCACHE *		pRCache;
	RCACHE *		pPrevRCache;
	FLMUINT		uiTransId = pDb->LogHdr.uiCurrTransID;

	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	pRCache = gv_FlmSysData.RCacheMgr.pLRURecord;

	// Stay in the loop until we have freed all old records, or
	// we have run through the entire list.

	while (pRCache)
	{
		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pRCache below, in which case we would
		// have lost the previous entry.

		pPrevRCache = pRCache->pPrevInGlobal;

		// Only look at records in this container and this database.

		if ((pRCache->uiContainer == uiContainer)  &&
			 (pRCache->pFile == pFile))
		{

			// Only look at the most current versions.

			if (pRCache->uiHighTransId == 0xFFFFFFFF)
			{

				// Better not be a newer version.

				flmAssert( pRCache->pNewerVersion == NULL);

				if (pRCache->uiLowTransId < uiTransId)
				{

					// This version was not added or modified by this
					// transaction so it's high transaction ID should simply
					// be set to one less than the current transaction ID.

					flmRcaSetTransID( pRCache, uiTransId - 1);
					flmAssert( pRCache->uiHighTransId >= pRCache->uiLowTransId);
					RCA_SET_UNCOMMITTED( pRCache->uiFlags);
					RCA_SET_LATEST_VER( pRCache->uiFlags);
					flmRcaUnlinkFromFile( pRCache);
					flmRcaLinkToFileAtHead( pRCache, pFile);
				}
				else
				{

					// The record was added or modified in this
					// transaction. Simply remove it from cache.

					flmRcaFreeCache( pRCache,
								(FLMBOOL)(RCA_IS_IN_USE( pRCache->uiFlags)
											 ? TRUE
											 : FALSE));
				}
			}
			else
			{

				// If not most current version, the record's high transaction
				// ID better already be less than transaction ID.

				flmAssert( pRCache->uiHighTransId < uiTransId);
			}
		}
		pRCache = pPrevRCache;

	}
	
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC RCODE flmRcaCheck(
	FDB *				pDb,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn)
{
	LFILE *			pLFile;
	FlmRecord *		pRecord = NULL;
	FLMUINT			uiLowTransId;
	FLMBOOL			bMostCurrent;
	RCODE				rc;

	if( RC_OK( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
		rc = FSReadRecord( pDb, pLFile, uiDrn, &pRecord,
								&uiLowTransId, &bMostCurrent);
	}

	if( pRecord)
	{
		pRecord->Release();
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
FLMBOOL F_RecRelocator::canRelocate(
	void *			pvAlloc)
{
	FlmRecord *		pRec = (FlmRecord *)pvAlloc;

	// DO NOT call getRefCount() or isCached() methods because the
	// constructors may not have been called yet - meaning that the v-table
	// pointers may not yet be set up.  We are relying on
	// FlmRecord::objectAllocInit to initialize m_refCnt and m_uiFlags.
	// objecAllocInit is called while the slab allocator mutex is locked.

	if( pRec->m_refCnt == 1 && (pRec->m_uiFlags & RCA_CACHED))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_RecRelocator::relocate(
	void *			pvOldAlloc,
	void *			pvNewAlloc)
{
	FlmRecord *		pOldRec = (FlmRecord *)pvOldAlloc;
	FlmRecord *		pNewRec = (FlmRecord *)pvNewAlloc;
	RCACHE *			pRCache;
	RCACHE *			pVersion;

	flmAssert( pOldRec->m_refCnt == 1);
	flmAssert( pOldRec->m_uiFlags & RCA_CACHED);
	flmAssert( pvNewAlloc < pvOldAlloc);

	// Update the record pointer in the data buffer

	if( pNewRec->m_pucBuffer)
	{
		flmAssert( *((FlmRecord **)pOldRec->m_pucBuffer) == pOldRec);
		*((FlmRecord **)pNewRec->m_pucBuffer) = pNewRec;
	}
	
	if( pNewRec->m_pucFieldIdTable)
	{
		flmAssert( *((FlmRecord **)pOldRec->m_pucFieldIdTable) == pOldRec);
		*((FlmRecord **)pNewRec->m_pucFieldIdTable) = pNewRec;
	}

	// Find the record

	pRCache = *(FLM_RCA_HASH( pNewRec->m_uiRecordID));

	while( pRCache)
	{
		if( pRCache->uiDrn == pNewRec->m_uiRecordID)
		{
			pVersion = pRCache;

			while( pVersion)
			{
				if( pVersion->pRecord == pOldRec)
				{
					pVersion->pRecord = pNewRec;
					goto Done;
				}

				pVersion = pVersion->pOlderVersion;
			}
		}

		pRCache = pRCache->pNextInBucket;
	}

Done:

	flmAssert( pRCache);
}

/****************************************************************************
Desc:	
****************************************************************************/
FLMBOOL F_RecBufferRelocator::canRelocate(
	void *			pvAlloc)
{
	FlmRecord *		pRec = *((FlmRecord **)pvAlloc);

	flmAssert( pRec->m_pucBuffer == pvAlloc || 
				  pRec->m_pucFieldIdTable == pvAlloc);

	// DO NOT call getRefCount() or isCached() methods because the
	// constructors may not have been called yet - meaning that the v-table
	// pointers may not yet be set up.  We are relying on
	// FlmRecord::objectAllocInit to initialize m_refCnt and m_uiFlags.
	// objecAllocInit is called while the slab allocator mutex is locked.

	if( pRec->m_refCnt == 1 && (pRec->m_uiFlags & RCA_CACHED))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_RecBufferRelocator::relocate(
	void *			pvOldAlloc,
	void *			pvNewAlloc)
{
	FlmRecord *		pRec = *((FlmRecord **)pvOldAlloc);

	flmAssert( pRec->m_refCnt == 1);
	flmAssert( pRec->m_uiFlags & RCA_CACHED);
	flmAssert( pvNewAlloc < pvOldAlloc);

	// Update the buffer pointer in the record

	if (pRec->m_pucBuffer == pvOldAlloc)
	{
		pRec->m_pucBuffer = (FLMBYTE *)pvNewAlloc;
	}
	else if (pRec->m_pucFieldIdTable == pvOldAlloc)
	{
		pRec->m_pucFieldIdTable = (FLMBYTE *)pvNewAlloc;
	}
	else
	{
		flmAssert( 0);
	}
}

/****************************************************************************
Desc:		
Notes:	This routine assumes the rcache mutex is locked
****************************************************************************/
FLMBOOL F_RCacheRelocator::canRelocate(
	void *		pvAlloc)
{
	RCACHE *		pRCache = (RCACHE *)pvAlloc;

	if( RCA_IS_IN_USE( pRCache->uiFlags) ||
		 RCA_IS_READING_IN( pRCache->uiFlags))
	{
		return( FALSE);
	}

	return( TRUE);
}

/****************************************************************************
Desc:		Fixes up all pointers needed to allow an RCACHE struct to be
			moved to a different location in memory
Notes:	This routine assumes the rcache mutex is locked
****************************************************************************/
void F_RCacheRelocator::relocate(
	void *			pvOldAlloc,
	void *			pvNewAlloc)
{
	RCACHE *			pOldRCache = (RCACHE *)pvOldAlloc;
	RCACHE *			pNewRCache = (RCACHE *)pvNewAlloc;
	RCACHE **		ppBucket;
	RCACHE_MGR *	pRCacheMgr = &gv_FlmSysData.RCacheMgr;
	FFILE *			pFile = pOldRCache->pFile;

	flmAssert( !RCA_IS_IN_USE( pOldRCache->uiFlags));
	flmAssert( !RCA_IS_READING_IN( pOldRCache->uiFlags));
	flmAssert( !pOldRCache->pNotifyList);

	if( pNewRCache->pPrevInFile)
	{
		pNewRCache->pPrevInFile->pNextInFile = pNewRCache;
	}

	if( pNewRCache->pNextInFile)
	{
		pNewRCache->pNextInFile->pPrevInFile = pNewRCache;
	}

	if( pNewRCache->pPrevInGlobal)
	{
		pNewRCache->pPrevInGlobal->pNextInGlobal = pNewRCache;
	}

	if( pNewRCache->pNextInGlobal)
	{
		pNewRCache->pNextInGlobal->pPrevInGlobal = pNewRCache;
	}

	if( pNewRCache->pPrevInBucket)
	{
		pNewRCache->pPrevInBucket->pNextInBucket = pNewRCache;
	}

	if( pNewRCache->pNextInBucket)
	{
		pNewRCache->pNextInBucket->pPrevInBucket = pNewRCache;
	}

	if( pNewRCache->pOlderVersion)
	{
		pNewRCache->pOlderVersion->pNewerVersion = pNewRCache;
	}

	if( pNewRCache->pNewerVersion)
	{
		pNewRCache->pNewerVersion->pOlderVersion = pNewRCache;
	}
	
	if( pNewRCache->pPrevInHeapList)
	{
		pNewRCache->pPrevInHeapList->pNextInHeapList = pNewRCache;
	}
	
	if( pNewRCache->pNextInHeapList)
	{
		pNewRCache->pNextInHeapList->pPrevInHeapList = pNewRCache;
	}
	
	ppBucket = FLM_RCA_HASH( pOldRCache->uiDrn);
	
	if( *ppBucket == pOldRCache)
	{
		*ppBucket = pNewRCache;
	}
	
	if( pRCacheMgr->pHeapList == pOldRCache)
	{
		pRCacheMgr->pHeapList = pNewRCache;
	}
	
	if( pRCacheMgr->pPurgeList == pOldRCache)
	{
		pRCacheMgr->pPurgeList = pNewRCache;
	}
	
	if( pRCacheMgr->pMRURecord == pOldRCache)
	{
		pRCacheMgr->pMRURecord = pNewRCache;
	}
	
	if( pRCacheMgr->pLRURecord == pOldRCache)
	{
		pRCacheMgr->pLRURecord = pNewRCache;
	}

	if( pFile)
	{
		if( pFile->pFirstRecord == pOldRCache)
		{
			pFile->pFirstRecord = pNewRCache;
		}
		
		if( pFile->pLastRecord == pOldRCache)
		{
			pFile->pLastRecord = pNewRCache;
		}
	}

#ifdef FLM_DEBUG
	f_memset( pOldRCache, 0, sizeof( RCACHE));
#endif
}
