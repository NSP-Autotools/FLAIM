//-------------------------------------------------------------------------
// Desc:	Block cache.
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

#ifdef FLM_DEBUG
	extern RCODE	gv_CriticalFSError;
#endif

#define MAX_BLOCKS_TO_SORT			500
#define FLM_MAX_IO_BUFFER_BLOCKS 16

typedef struct TMP_READ_STATS
{
	DISKIO_STAT		BlockReads;						// Statistics on block reads
	DISKIO_STAT		OldViewBlockReads;			// Statistics on old view
															// block reads
	FLMUINT			uiBlockChkErrs;				// Number of times we had
															// check errors reading
															// blocks
	FLMUINT			uiOldViewBlockChkErrs;		// Number of times we had
															// check errors reading an
															// old view of a block
} TMP_READ_STATS;

FLMUINT ScaGetBlkSize(
	SCACHE **		pSCache);

#define ScaGetBlkSize( pSCache) \
	(FLMUINT)((pSCache)->ui16BlkSize)

FSTATIC void ScaUnlinkFromGlobalList(
	SCACHE *			pSCache);

#ifdef SCACHE_LINK_CHECKING
FSTATIC void ScaLinkToHashBucket(
	SCACHE *			pSCache,
	SCACHE **		ppSCacheBucket);

FSTATIC void ScaUnlinkFromHashBucket(
	SCACHE *			pSCache,
	SCACHE **		ppSCacheBucket);
#endif

FSTATIC void ScaLinkToFile(
	SCACHE *			pSCache,
	FFILE *			pFile);

FSTATIC void ScaUnlinkFromFile(
	SCACHE *			pSCache);

FSTATIC void ScaLinkToFreeList(
	SCACHE *			pSCache,
	FLMUINT			uiFreeTime);

FSTATIC void ScaUnlinkFromFreeList(
	SCACHE *			pSCache);

#ifdef FLM_DEBUG
FSTATIC void ScaDebugMsg(
	const char *	pszMsg,
	SCACHE *			pSCache,
	SCACHE_USE *	pUse);

FSTATIC void _ScaDbgUseForThread(
	SCACHE *			pSCache,
	FLMUINT *		puiThreadId);

FSTATIC void _ScaDbgReleaseForThread(
	SCACHE *			pSCache);
#endif

FSTATIC void ScaNotify(
	F_NOTIFY_LIST_ITEM *	pNotify,
	SCACHE *					pUseSCache,
	RCODE						NotifyRc);

FSTATIC void ScaFree(
	SCACHE *			pSCache);

FSTATIC void ScaUnlinkCache(
	SCACHE *			pSCache,
	FLMBOOL			bFreeIt,
	RCODE				NotifyRc);

FSTATIC void ScaUnlinkTransLogBlocks(
	FFILE *			pFile);

FSTATIC void ScaUnlinkFromTransLogList(
	SCACHE *			pSCache,
	FFILE *			pFile);

FSTATIC void ScaSavePrevBlkAddress(
	SCACHE *			pSCache);

FSTATIC RCODE ScaAllocBlocksArray(
	FFILE *			pFile,
	FLMUINT			uiNewSize,
	FLMBOOL			bOneArray);

FSTATIC RCODE ScaFlushLogBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bIsCPThread,
	FLMUINT				uiMaxDirtyCache,
	FLMBOOL *			pbForceCheckpoint,
	FLMBOOL *			pbWroteAll);

FSTATIC void FLMAPI scaWriteComplete(
	IF_IOBuffer *		pIOBuffer,
	void *				pvData);

FSTATIC RCODE ScaReduceCache(
	FDB *				pDb);

FSTATIC RCODE scaAllocCacheBlock(
	FLMUINT			uiBlockSize,
	SCACHE **		ppSCache);

FSTATIC RCODE ScaAllocCache(
	FDB *				pDb,
	SCACHE **		ppSCacheRV);

FSTATIC RCODE ScaReadTheBlock(
	FDB *	  				pDb,
	LFILE *				pLFile,
	TMP_READ_STATS *	pTmpReadStats,
	FLMBYTE *			pucBlk,
	FLMUINT				uiFilePos,
	FLMUINT				uiBlkAddress);

FSTATIC RCODE ScaBlkSanityCheck(
	FDB *				pDb,
	FFILE *			pFile,
	LFILE *			pLFile,
	FLMBYTE *		pucBlk,
	FLMUINT			uiBlkAddress,
	FLMBOOL			bCheckFullBlkAddr,
	FLMUINT			uiSanityLevel);

FSTATIC RCODE ScaReadBlock(
	FDB *				pDb,
	FLMUINT			uiBlkType,
	LFILE *			pLFile,
	FLMUINT			uiFilePos,
	FLMUINT			uiBlkAddress,
	FLMUINT			uiNewerBlkLowTransID,
	FLMUINT			uiExpectedLowTransID,
	SCACHE *			pSCache,
	FLMBOOL *		pbFoundVerRV,
	FLMBOOL *		pbDiscardRV);

FSTATIC RCODE scaFinishCheckpoint(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bTruncateRollBackLog,
	FLMUINT				uiCPFileNum,
	FLMUINT				uiCPOffset,
	FLMUINT				uiCPStartTime,
	FLMUINT				uiTotalToWrite);

FSTATIC RCODE ScaReadIntoCache(
	FDB *					pDb,
	FLMUINT				uiBlkType,
	LFILE *				pLFile,
	FLMUINT				uiBlkAddress,
	SCACHE *				pPrevInVerList,
	SCACHE *				pNextInVerList,
	SCACHE **			ppSCacheRV,
	FLMBOOL *			pbGotFromDisk);

FSTATIC void scaSort(
	SCACHE **		ppSCacheTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds);

FSTATIC RCODE scaWriteSortedBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT				uiMaxDirtyCache,
	FLMUINT *			puiDirtyCacheLeft,
	FLMBOOL *			pbForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMUINT				uiNumSortedBlocks,
	FLMBOOL *			pbWroteAll);

FSTATIC RCODE ScaFlushDirtyBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT				uiMaxDirtyCache,
	FLMBOOL				bForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMBOOL *			pbWroteAll);

FSTATIC RCODE ScaReduceNewBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT *			puiBlocksFlushed);

FSTATIC void scaSetBlkDirty(
	FFILE *	pFile,
	SCACHE *	pSCache);

FSTATIC FLMUINT ScaNumHashBuckets(
	FLMUINT	uiMaxSharedCache
	);

FSTATIC RCODE ScaInitHashTbl(
	FLMUINT			uiNumBuckets
	);

#ifdef SCACHE_LINK_CHECKING
	FSTATIC void scaVerifyCache(
		SCACHE *			pSCache,
		int				iPlace);

	FSTATIC void scaVerify(
		int				iPlace);
#else

	#define scaVerifyCache(pSCache,iPlace)
	#define scaVerify(iPlace)

#endif

FSTATIC void scaReduceFreeCache(
	FLMBOOL			bFreeAll);

FSTATIC void scaReduceReuseList( void);

#ifdef FLM_DEBUG
FSTATIC FLMUINT ScaComputeChecksum(
	SCACHE *			pSCache);

FSTATIC void ScaVerifyChecksum(
	SCACHE *			pSCache);
#endif

FLMBOOL ScaNeededByReadTrans(
	FFILE *			pFile,
	SCACHE *			pSCache);

#define ScaNeededByReadTrans(pFile,pSCache) \
	flmNeededByReadTrans( (pFile), scaGetLowTransID( pSCache), \
						(pSCache)->uiHighTransID)

FSTATIC void ScaLinkToFileLogList(
	SCACHE *			pSCache);

FSTATIC void ScaUnlinkFromReplaceList(
	SCACHE *			pSCache);

FSTATIC void ScaUnlinkFromFileLogList(
	SCACHE *			pSCache);

FSTATIC void ScaLinkToNewList(
	SCACHE *			pSCache);

FSTATIC void ScaUnlinkFromNewList(
	SCACHE *			pSCache);

#ifdef FLM_DEBUG

	FSTATIC void ScaDbgUseForThread(
		SCACHE *			pSCache,
		FLMUINT *		puiThreadId);

	FSTATIC void ScaDbgReleaseForThread(
		SCACHE *			pSCache);

	#define ScaUseForThread			ScaDbgUseForThread
	#define ScaReleaseForThread	ScaDbgReleaseForThread

#else

	#define ScaUseForThread			ScaNonDbgUseForThread
	#define ScaReleaseForThread	ScaNonDbgReleaseForThread

#endif

/****************************************************************************
Desc:
****************************************************************************/
class F_SCacheRelocator : public IF_Relocator
{
public:

	F_SCacheRelocator()
	{
	}
	
	virtual ~F_SCacheRelocator()
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
class F_BlockRelocator : public IF_Relocator
{
public:

	F_BlockRelocator( FLMUINT uiBlockSize)
	{
		m_uiSigBitsInBlkSize = flmGetSigBits( uiBlockSize);
	}
	
	virtual ~F_BlockRelocator()
	{
	}

	SCACHE * getSCachePtr(
		void *		pvAlloc);
	
	void FLMAPI relocate(
		void *		pvOldAlloc,
		void *		pvNewAlloc);

	FLMBOOL FLMAPI canRelocate(
		void *		pvOldAlloc);
		
	FLMUINT			m_uiSigBitsInBlkSize;
};

/***************************************************************************
Desc:
*****************************************************************************/
FINLINE FLMUINT SCA_MEM_SIZE(
	SCACHE *		pSCache)
{
	return( sizeof( SCACHE) + pSCache->ui16BlkSize);
}

/***************************************************************************
Desc:	Compare two cache blocks to determine which one has lower address.
*****************************************************************************/
FINLINE FLMINT scaCompare(
	SCACHE *		pSCache1,
	SCACHE *		pSCache2
	)
{
	if (FSAddrIsAtOrBelow( pSCache1->uiBlkAddress, pSCache2->uiBlkAddress))
	{
		flmAssert( pSCache1->uiBlkAddress != pSCache2->uiBlkAddress);
		return( -1);
	}
	else
	{
		return( 1);
	}
}

/****************************************************************************
Desc:	Gets the prior image block address from the block header.
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FINLINE FLMUINT scaGetPriorImageAddress(
	SCACHE *		pSCache)
{
	return( (FLMUINT)FB2UD( &pSCache->pucBlk [BH_PREV_BLK_ADDR]));
}

/****************************************************************************
Desc:	Gets the prior image transaction ID from the block header.
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FINLINE FLMUINT scaGetPriorImageTransID(
	SCACHE *		pSCache)
{
	return( (FLMUINT)FB2UD( &pSCache->pucBlk [BH_PREV_TRANS_ID]));
}

/****************************************************************************
Desc:	Link a cache block into the replace list as the MRU item. This routine
		assumes that the global mutex has already been locked.
*****************************************************************************/
FINLINE void ScaLinkToReplaceListAsMRU(
	SCACHE *		pSCache)
{
	flmAssert( !pSCache->ui16Flags);

	if( (pSCache->pNextInReplaceList = 
		gv_FlmSysData.SCacheMgr.pMRUReplace) != NULL)
	{
		pSCache->pNextInReplaceList->pPrevInReplaceList = pSCache;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLRUReplace = pSCache;
	}
	pSCache->pPrevInReplaceList = NULL;
	gv_FlmSysData.SCacheMgr.pMRUReplace = pSCache;
	gv_FlmSysData.SCacheMgr.uiReplaceableCount++;
	gv_FlmSysData.SCacheMgr.uiReplaceableBytes += SCA_MEM_SIZE( pSCache);
}

/****************************************************************************
Desc:	Link a cache block into the replace list as the LRU item. This routine
		assumes that the global mutex has already been locked.
****************************************************************************/
FINLINE void ScaLinkToReplaceListAsLRU(
	SCACHE *		pSCache)
{
	flmAssert( !pSCache->ui16Flags);

	if ((pSCache->pPrevInReplaceList = 
		gv_FlmSysData.SCacheMgr.pLRUReplace) != NULL)
	{
		pSCache->pPrevInReplaceList->pNextInReplaceList = pSCache;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pMRUReplace = pSCache;
	}
	pSCache->pNextInReplaceList = NULL;
	gv_FlmSysData.SCacheMgr.pLRUReplace = pSCache;
	gv_FlmSysData.SCacheMgr.uiReplaceableCount++;
	gv_FlmSysData.SCacheMgr.uiReplaceableBytes += SCA_MEM_SIZE( pSCache);
}

/****************************************************************************
Desc:	Moves a block one step closer to the MRU slot in the replace list.
		This routine assumes that the global mutex has already been locked.
****************************************************************************/
FINLINE void ScaStepUpInReplaceList(
	SCACHE *		pSCache)
{
	SCACHE *		pPrevSCache;

	flmAssert( !pSCache->ui16Flags);

	if( (pPrevSCache = pSCache->pPrevInReplaceList) != NULL)
	{
		if( pPrevSCache->pPrevInReplaceList)
		{
			pPrevSCache->pPrevInReplaceList->pNextInReplaceList = pSCache;
		}
		else
		{
			gv_FlmSysData.SCacheMgr.pMRUReplace = pSCache;
		}

		pSCache->pPrevInReplaceList = pPrevSCache->pPrevInReplaceList;
		pPrevSCache->pPrevInReplaceList = pSCache;
		pPrevSCache->pNextInReplaceList = pSCache->pNextInReplaceList;

		if( pSCache->pNextInReplaceList)
		{
			pSCache->pNextInReplaceList->pPrevInReplaceList = pPrevSCache;
		}
		else
		{
			gv_FlmSysData.SCacheMgr.pLRUReplace = pPrevSCache;
		}
		pSCache->pNextInReplaceList = pPrevSCache;
	}
}

/****************************************************************************
Desc:	Clears the passed-in flags from the SCACHE struct
		This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE void scaClearFlags(
	SCACHE *		pSCache,
	FLMUINT16	ui16FlagsToClear)
{
	if( pSCache->ui16Flags)
	{
		if( (pSCache->ui16Flags &= ~ui16FlagsToClear) == 0)
		{
			if( !pSCache->pPrevInGlobalList ||
				pSCache->uiHighTransID == 0xFFFFFFFF ||
				ScaNeededByReadTrans( pSCache->pFile, pSCache))
			{
				ScaLinkToReplaceListAsMRU( pSCache);
			}
			else
			{
				ScaLinkToReplaceListAsLRU( pSCache);
			}
		}
	}
}

/****************************************************************************
Desc:	Sets the passed-in flags on the SCACHE
		This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE void scaSetFlags(
	SCACHE *		pSCache,
	FLMUINT16	ui16FlagsToSet
	)
{
	flmAssert( ui16FlagsToSet);

	if( !pSCache->ui16Flags)
	{
		ScaUnlinkFromReplaceList( pSCache);
	}
	pSCache->ui16Flags |= ui16FlagsToSet;
}

/****************************************************************************
Desc:	Link a cache block into the list of FFILE blocks that need one or
		more versions of the block to be logged.  This routine assumes that
		the global mutex is locked.
*****************************************************************************/
FSTATIC void ScaLinkToFileLogList(
	SCACHE *		pSCache)
{
	FFILE *		pFile = pSCache->pFile;
	FLMUINT		uiPrevBlkAddress;

	flmAssert( pSCache->ui16Flags & CA_DIRTY);
	flmAssert( !(pSCache->ui16Flags & (CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	flmAssert( !pSCache->pPrevInReplaceList);
	flmAssert( !pSCache->pNextInReplaceList);

	uiPrevBlkAddress = scaGetPriorImageAddress( pSCache);
	if( (uiPrevBlkAddress && uiPrevBlkAddress != BT_END) ||
		!pSCache->pNextInVersionList)
	{
		goto Exit;
	}

	if( (pSCache->pNextInReplaceList = pFile->pFirstInLogList) != NULL)
	{
		pSCache->pNextInReplaceList->pPrevInReplaceList = pSCache;
	}
	else
	{
		pFile->pLastInLogList = pSCache;
	}

	scaSetFlags( pSCache, CA_IN_FILE_LOG_LIST);
	pSCache->pPrevInReplaceList = NULL;
	pFile->pFirstInLogList = pSCache;
	pFile->uiLogListCount++;

Exit:

	return;
}

/****************************************************************************
Desc:	Unlinks a cache block from the FFILE's log list
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromFileLogList(
	SCACHE *		pSCache)
{
	FFILE *		pFile = pSCache->pFile;

	flmAssert( pSCache->ui16Flags & CA_IN_FILE_LOG_LIST);
	flmAssert( pFile->uiLogListCount);

	if( pSCache->pNextInReplaceList)
	{
		pSCache->pNextInReplaceList->pPrevInReplaceList =
			pSCache->pPrevInReplaceList;
	}
	else
	{
		pFile->pLastInLogList = pSCache->pPrevInReplaceList;
	}

	if( pSCache->pPrevInReplaceList)
	{
		pSCache->pPrevInReplaceList->pNextInReplaceList =
			pSCache->pNextInReplaceList;
	}
	else
	{
		pFile->pFirstInLogList = pSCache->pNextInReplaceList;
	}

	pSCache->pNextInReplaceList = NULL;
	pSCache->pPrevInReplaceList = NULL;

	scaClearFlags( pSCache, CA_IN_FILE_LOG_LIST);
	pFile->uiLogListCount--;
}

/****************************************************************************
Desc:	Link a cache block into the list of FFILE blocks that are beyond the
		EOF.  The blocks are linked to the end of the list so that they
		are kept in ascending order.
		NOTE: This function assumes that the global mutex is locked.
*****************************************************************************/
FSTATIC void ScaLinkToNewList(
	SCACHE *		pSCache)
{
	FFILE *		pFile = pSCache->pFile;

	flmAssert( pSCache->uiHighTransID == 0xFFFFFFFF);
	flmAssert( pSCache->ui16Flags & CA_DIRTY);
	flmAssert( !(pSCache->ui16Flags & (CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	flmAssert( !pSCache->pPrevInReplaceList);
	flmAssert( !pSCache->pNextInReplaceList);

	if ((pSCache->pPrevInReplaceList = pFile->pLastInNewList) != NULL)
	{
		flmAssert( scaCompare( pFile->pLastInNewList, pSCache) < 0);
		pSCache->pPrevInReplaceList->pNextInReplaceList = pSCache;
	}
	else
	{
		pFile->pFirstInNewList = pSCache;
	}
	pSCache->pNextInReplaceList = NULL;
	pFile->pLastInNewList = pSCache;
	scaSetFlags( pSCache, CA_IN_NEW_LIST);
	pFile->uiNewCount++;
}

/****************************************************************************
Desc:	Unlinks a cache block from the FFILE's new block list
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromNewList(
	SCACHE *		pSCache)
{
	FFILE *		pFile = pSCache->pFile;

	flmAssert( pSCache->ui16Flags & CA_IN_NEW_LIST);
	flmAssert( pFile->uiNewCount);

	if( pSCache->pNextInReplaceList)
	{
		pSCache->pNextInReplaceList->pPrevInReplaceList =
			pSCache->pPrevInReplaceList;
	}
	else
	{
		pFile->pLastInNewList = pSCache->pPrevInReplaceList;
	}

	if( pSCache->pPrevInReplaceList)
	{
		pSCache->pPrevInReplaceList->pNextInReplaceList =
			pSCache->pNextInReplaceList;
	}
	else
	{
		pFile->pFirstInNewList = pSCache->pNextInReplaceList;
	}

	pSCache->pNextInReplaceList = NULL;
	pSCache->pPrevInReplaceList = NULL;

	scaClearFlags( pSCache, CA_IN_NEW_LIST);
	pFile->uiNewCount--;
}

/****************************************************************************
Desc:	Set the high transaction ID for a cache block.
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FINLINE void scaSetTransID(
	SCACHE *	pSCache,
	FLMUINT	uiNewTransID)
{
	if (pSCache->uiHighTransID == 0xFFFFFFFF && uiNewTransID != 0xFFFFFFFF)
	{
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes += SCA_MEM_SIZE( pSCache);
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount++;
	}
	else if (pSCache->uiHighTransID != 0xFFFFFFFF && uiNewTransID == 0xFFFFFFFF)
	{
		FLMUINT	uiSize = SCA_MEM_SIZE( pSCache);

		flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes >= uiSize);
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes -= uiSize;
		flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount);
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount--;
	}
	pSCache->uiHighTransID = uiNewTransID;
}

/****************************************************************************
Desc:	Set the dirty flag on a cache block.
		This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE void scaSetDirtyFlag(
	SCACHE *		pSCache,
	FFILE *		pFile)
{
	flmAssert( !(pSCache->ui16Flags & 
		(CA_DIRTY | CA_WRITE_PENDING | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	scaSetFlags( pSCache, CA_DIRTY);
	pFile->uiDirtyCacheCount++;
}

/****************************************************************************
Desc:	Unset the dirty flag on a cache block.
		This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE void scaUnsetDirtyFlag(
	SCACHE *		pSCache,
	FFILE *		pFile)
{
	flmAssert( pSCache->ui16Flags & CA_DIRTY);
	flmAssert( pFile->uiDirtyCacheCount);

	if( pSCache->ui16Flags & CA_IN_FILE_LOG_LIST)
	{
		ScaUnlinkFromFileLogList( pSCache);
	}
	else if( pSCache->ui16Flags & CA_IN_NEW_LIST)
	{
		ScaUnlinkFromNewList( pSCache);
	}

	scaClearFlags( pSCache, CA_DIRTY);
	pFile->uiDirtyCacheCount--;
}

/****************************************************************************
Desc:	Non-debug mode - use a cache block.  NOTE: This function assumes
		that the global mutext is locked.
****************************************************************************/
FINLINE void ScaNonDbgUseForThread(
	SCACHE *		pSCache,
	FLMUINT *	// puiThreadId
	)
{
	if (!pSCache->uiUseCount)
	{
		gv_FlmSysData.SCacheMgr.uiBlocksUsed++;
	}
	pSCache->uiUseCount++;
	gv_FlmSysData.SCacheMgr.uiTotalUses++;
}

/****************************************************************************
Desc:	Non-debug mode - release a cache block.  NOTE: This function assumes
		that the global mutext is locked.
****************************************************************************/
FINLINE void ScaNonDbgReleaseForThread(
	SCACHE *		pSCache)
{
	if (pSCache->uiUseCount)
	{
		pSCache->uiUseCount--;
		gv_FlmSysData.SCacheMgr.uiTotalUses--;
		if (!pSCache->uiUseCount)
		{
			gv_FlmSysData.SCacheMgr.uiBlocksUsed--;
		}
	}
}

/****************************************************************************
Desc:	Returns TRUE if the cache is over the cache limit
		This routine assumes that the global mutex is locked
****************************************************************************/
FINLINE FLMBOOL scaIsCacheOverLimit( void)
{
	if( gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated >
		gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes)
	{
		if( (gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +
			gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated) >
			gv_FlmSysData.uiMaxCache)
		{
			return( TRUE);
		}
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Unlinks a cache block from the replace list
		NOTE: This function assumes that the global mutex is locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromReplaceList(
	SCACHE *			pSCache)
{
	FLMUINT	uiSize = SCA_MEM_SIZE( pSCache);

	flmAssert( !pSCache->ui16Flags);

	if( pSCache->pNextInReplaceList)
	{
		pSCache->pNextInReplaceList->pPrevInReplaceList =
			pSCache->pPrevInReplaceList;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLRUReplace = pSCache->pPrevInReplaceList;
	}

	if( pSCache->pPrevInReplaceList)
	{
		pSCache->pPrevInReplaceList->pNextInReplaceList =
			pSCache->pNextInReplaceList;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pMRUReplace = pSCache->pNextInReplaceList;
	}

	pSCache->pNextInReplaceList = NULL;
	pSCache->pPrevInReplaceList = NULL;

	flmAssert( gv_FlmSysData.SCacheMgr.uiReplaceableCount);
	gv_FlmSysData.SCacheMgr.uiReplaceableCount--;
	flmAssert( gv_FlmSysData.SCacheMgr.uiReplaceableBytes >= uiSize);
	gv_FlmSysData.SCacheMgr.uiReplaceableBytes -= uiSize;
}

/****************************************************************************
Desc:	Link a cache block into the global list as the MRU item. This routine
		assumes that the global mutex has already been locked.
*****************************************************************************/
FINLINE void ScaLinkToGlobalListAsMRU(
	SCACHE *			pSCache)
{
	if ((pSCache->pNextInGlobalList = gv_FlmSysData.SCacheMgr.pMRUCache) != NULL)
	{
		pSCache->pNextInGlobalList->pPrevInGlobalList = pSCache;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLRUCache = pSCache;
	}
	pSCache->pPrevInGlobalList = NULL;
	gv_FlmSysData.SCacheMgr.pMRUCache = pSCache;

	if( !pSCache->ui16Flags)
	{
		ScaLinkToReplaceListAsMRU( pSCache);
	}
}

/****************************************************************************
Desc:	Link a cache block into the global list as the LRU item. This routine
		assumes that the global mutex has already been locked.
****************************************************************************/
FINLINE void ScaLinkToGlobalListAsLRU(
	SCACHE *			pSCache)
{
	if ((pSCache->pPrevInGlobalList = 
		gv_FlmSysData.SCacheMgr.pLRUCache) != NULL)
	{
		pSCache->pPrevInGlobalList->pNextInGlobalList = pSCache;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pMRUCache = pSCache;
	}
	pSCache->pNextInGlobalList = NULL;
	gv_FlmSysData.SCacheMgr.pLRUCache = pSCache;

	if( !pSCache->ui16Flags)
	{
		ScaLinkToReplaceListAsLRU( pSCache);
	}
}

/****************************************************************************
Desc:	Unlink a cache block from the global list. This routine
		assumes that the global mutex has already been locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromGlobalList(
	SCACHE *			pSCache)
{
	if (pSCache->pNextInGlobalList)
	{
		pSCache->pNextInGlobalList->pPrevInGlobalList =
			pSCache->pPrevInGlobalList;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLRUCache = pSCache->pPrevInGlobalList;
	}

	if (pSCache->pPrevInGlobalList)
	{
		pSCache->pPrevInGlobalList->pNextInGlobalList =
			pSCache->pNextInGlobalList;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pMRUCache = pSCache->pNextInGlobalList;
	}
	pSCache->pNextInGlobalList = pSCache->pPrevInGlobalList = (SCACHE *)NULL;

	if( !pSCache->ui16Flags)
	{
		ScaUnlinkFromReplaceList( pSCache);
	}
}

/****************************************************************************
Desc:	Moves a block one step closer to the MRU slot in the global list.  This
		routine assumes that the global mutex has already been locked.
****************************************************************************/
FINLINE void ScaStepUpInGlobalList(
	SCACHE *			pSCache)
{
	SCACHE *		pPrevSCache;

	if( (pPrevSCache = pSCache->pPrevInGlobalList) != NULL)
	{
		if( pPrevSCache->pPrevInGlobalList)
		{
			pPrevSCache->pPrevInGlobalList->pNextInGlobalList = pSCache;
		}
		else
		{
			gv_FlmSysData.SCacheMgr.pMRUCache = pSCache;
		}

		pSCache->pPrevInGlobalList = pPrevSCache->pPrevInGlobalList;
		pPrevSCache->pPrevInGlobalList = pSCache;
		pPrevSCache->pNextInGlobalList = pSCache->pNextInGlobalList;

		if( pSCache->pNextInGlobalList)
		{
			pSCache->pNextInGlobalList->pPrevInGlobalList = pPrevSCache;
		}
		else
		{
			gv_FlmSysData.SCacheMgr.pLRUCache = pPrevSCache;
		}
		pSCache->pNextInGlobalList = pPrevSCache;
	}

	if( !pSCache->ui16Flags)
	{
		ScaStepUpInReplaceList( pSCache);
	}
}

/****************************************************************************
Desc:	Link a cache block to its hash bucket.  This routine assumes
		that the global mutex has already been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
FSTATIC void ScaLinkToHashBucket(
#else
FINLINE void ScaLinkToHashBucket(
#endif
	SCACHE *		pSCache,				// SCACHE structure to be linked
	SCACHE **	ppSCacheBucket)	// Hash bucket
{
#ifdef SCACHE_LINK_CHECKING
	SCACHE *	pBlock;

	if (!pSCache->pFile)
	{
		f_breakpoint(1);
	}

	if (pSCache->pPrevInVersionList)
	{
		f_breakpoint(2);
	}

	if (pSCache->pNextInVersionList == pSCache)
	{
		f_breakpoint( 3);
	}

	if (pSCache->pPrevInVersionList == pSCache)
	{
		f_breakpoint( 4);
	}

	// Make sure that the block isn't added into the list a second time.

	for (pBlock = *ppSCacheBucket;
			pBlock;
			pBlock = pBlock->pNextInHashBucket)
	{
		if (pSCache == pBlock)
		{
			f_breakpoint(5);
		}
	}

	// Make sure the block is not in the transaction
	// log list.

	for (pBlock = pSCache->pFile->pTransLogList;
			pBlock;
			pBlock = pBlock->pNextInHashBucket)
	{
		if (pSCache == pBlock)
		{
			f_breakpoint(6);
		}
	}
#endif

	pSCache->pPrevInHashBucket = NULL;
	if ((pSCache->pNextInHashBucket = *ppSCacheBucket) != NULL)
	{
		pSCache->pNextInHashBucket->pPrevInHashBucket = pSCache;
	}
	*ppSCacheBucket = pSCache;
}

/****************************************************************************
Desc:	Unlink a cache block from its hash bucket.  This routine assumes
		that the global mutex has already been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
FSTATIC void ScaUnlinkFromHashBucket(
#else
FINLINE void ScaUnlinkFromHashBucket(
#endif
	SCACHE *		pSCache,				// SCACHE structure to be unlinked
	SCACHE **	ppSCacheBucket)	// Hash bucket
{
#ifdef SCACHE_LINK_CHECKING

	SCACHE *	pTmpSCache;

	// Make sure the cache is actually in this bucket

	pTmpSCache = *ppSCacheBucket;
	while (pTmpSCache && pTmpSCache != pSCache)
	{
		pTmpSCache = pTmpSCache->pNextInHashBucket;
	}

	if (!pTmpSCache)
	{
		f_breakpoint( 333);
	}

	for (pTmpSCache = pSCache->pFile->pTransLogList;
			pTmpSCache;
			pTmpSCache = pTmpSCache->pNextInHashBucket)
	{
		if (pSCache == pTmpSCache)
		{
			f_breakpoint(334);
		}
	}
#endif

	// Make sure it is not in the list of log blocks.

	flmAssert( !(pSCache->ui16Flags & CA_WRITE_TO_LOG));
	if (pSCache->pNextInHashBucket)
	{
		pSCache->pNextInHashBucket->pPrevInHashBucket =
			pSCache->pPrevInHashBucket;
	}

	if (pSCache->pPrevInHashBucket)
	{
		pSCache->pPrevInHashBucket->pNextInHashBucket =
			pSCache->pNextInHashBucket;
	}
	else
	{
		*ppSCacheBucket = pSCache->pNextInHashBucket;
	}

	pSCache->pNextInHashBucket = NULL;
	pSCache->pPrevInHashBucket = NULL;
}

/****************************************************************************
Desc:	Link a cache block to its FFILE structure.  This routine assumes
		that the global mutex has already been locked.
****************************************************************************/
FSTATIC void ScaLinkToFile(
	SCACHE *		pSCache,		// SCACHE structure to be linked
	FFILE *		pFile			// FFILE structure the SCACHE structure is
									// to be linked into.
	)
{
	if (pSCache->ui16Flags & CA_WRITE_PENDING)
	{
		if ((pSCache->pNextInFile = pFile->pPendingWriteList) != NULL)
		{
			pSCache->pNextInFile->pPrevInFile = pSCache;
		}
		pFile->pPendingWriteList = pSCache;
		scaSetFlags( pSCache, CA_IN_WRITE_PENDING_LIST);
	}
	else
	{
		SCACHE *	pPrevSCache;
		SCACHE *	pNextSCache;

		// Link at end of dirty blocks.

		if (pFile->pLastDirtyBlk)
		{
			pPrevSCache = pFile->pLastDirtyBlk;
			pNextSCache = pPrevSCache->pNextInFile;
		}
		else
		{

			// No dirty blocks, so link to head of list.

			pPrevSCache = NULL;
			pNextSCache = pFile->pSCacheList;
		}

		// If the block is dirty, change the last dirty block pointer.

		if (pSCache->ui16Flags & CA_DIRTY)
		{
			pFile->pLastDirtyBlk = pSCache;
		}

		if ((pSCache->pNextInFile = pNextSCache) != NULL)
		{
			pNextSCache->pPrevInFile = pSCache;
		}

		if ((pSCache->pPrevInFile = pPrevSCache) != NULL)
		{
			pPrevSCache->pNextInFile = pSCache;
		}
		else
		{
			pFile->pSCacheList = pSCache;
		}
	}
	pSCache->pFile = pFile;
}

/****************************************************************************
Desc:	Unlink a cache block from its FFILE structure.  This routine assumes
		that the global mutex has already been locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromFile(
	SCACHE *		pSCache)
{
	FFILE *	pFile = pSCache->pFile;

	if (pFile)
	{
		if (pSCache->ui16Flags & CA_IN_WRITE_PENDING_LIST)
		{
			if (pSCache->pPrevInFile)
			{
				pSCache->pPrevInFile->pNextInFile = pSCache->pNextInFile;
			}
			else
			{
				pFile->pPendingWriteList = pSCache->pNextInFile;
			}
			if (pSCache->pNextInFile)
			{
				pSCache->pNextInFile->pPrevInFile = pSCache->pPrevInFile;
			}

			scaClearFlags( pSCache, CA_IN_WRITE_PENDING_LIST);
		}
		else
		{
			if (pSCache == pFile->pLastDirtyBlk)
			{
				pFile->pLastDirtyBlk = pFile->pLastDirtyBlk->pPrevInFile;
#ifdef FLM_DEBUG

				// If pLastDirtyBlk is non-NULL, it had better be pointing
				// to a dirty block.

				if (pFile->pLastDirtyBlk)
				{
					flmAssert( pFile->pLastDirtyBlk->ui16Flags & CA_DIRTY);
				}
#endif
			}
			if (pSCache->pNextInFile)
			{
				pSCache->pNextInFile->pPrevInFile = pSCache->pPrevInFile;
			}
			if (pSCache->pPrevInFile)
			{
				pSCache->pPrevInFile->pNextInFile = pSCache->pNextInFile;
			}
			else
			{
				pFile->pSCacheList = pSCache->pNextInFile;
			}
			pSCache->pNextInFile = pSCache->pPrevInFile = (SCACHE *)NULL;
		}
		pSCache->pFile = NULL;
	}
}

/****************************************************************************
Desc:	Link a cache block to the free list
****************************************************************************/
FSTATIC void ScaLinkToFreeList(
	SCACHE *		pSCache,
	FLMUINT		uiFreeTime)
{
	flmAssert( !pSCache->ui16Flags);
	flmAssert( !pSCache->pFile);
	flmAssert( !pSCache->pPrevInReplaceList);
	flmAssert( !pSCache->pNextInReplaceList);

	if( pSCache->uiHighTransID != 0xFFFFFFFF)
	{
		// Set the transaction ID to -1 so that the old version
		// counts will be decremented if this is an old version
		// of the block.  Also, we want the transaction ID to be
		// -1 so that when the block is re-used in ScaAllocCache()
		// the old version counts won't be decremented again.

		scaSetTransID( pSCache, 0xFFFFFFFF);
	}

	if( (pSCache->pNextInFile = gv_FlmSysData.SCacheMgr.pFirstFree) != NULL)
	{
		pSCache->pNextInFile->pPrevInFile = pSCache;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLastFree = pSCache;
	}

	pSCache->pPrevInFile = NULL;
	pSCache->uiBlkAddress = uiFreeTime;
	pSCache->ui16Flags = CA_FREE;
	gv_FlmSysData.SCacheMgr.pFirstFree = pSCache;
	gv_FlmSysData.SCacheMgr.uiFreeBytes += SCA_MEM_SIZE( pSCache);
	gv_FlmSysData.SCacheMgr.uiFreeCount++;
}

/****************************************************************************
Desc:	Unlink a cache block from the free list.  This routine assumes
		that the global mutex has already been locked.
****************************************************************************/
FSTATIC void ScaUnlinkFromFreeList(
	SCACHE *		pSCache)
{
	FLMUINT	uiSize = SCA_MEM_SIZE( pSCache);

	flmAssert( !pSCache->uiUseCount);
	flmAssert( pSCache->ui16Flags == CA_FREE);

	if( pSCache->pNextInFile)
	{
		pSCache->pNextInFile->pPrevInFile = pSCache->pPrevInFile;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pLastFree = pSCache->pPrevInFile;
	}

	if( pSCache->pPrevInFile)
	{
		pSCache->pPrevInFile->pNextInFile = pSCache->pNextInFile;
	}
	else
	{
		gv_FlmSysData.SCacheMgr.pFirstFree = pSCache->pNextInFile;
	}

	pSCache->pNextInFile = NULL;
	pSCache->pPrevInFile = NULL;
	pSCache->ui16Flags = 0;

	flmAssert( gv_FlmSysData.SCacheMgr.uiFreeBytes >= uiSize);
	gv_FlmSysData.SCacheMgr.uiFreeBytes -= uiSize;
	flmAssert( gv_FlmSysData.SCacheMgr.uiFreeCount);
	gv_FlmSysData.SCacheMgr.uiFreeCount--;
}

/****************************************************************************
Desc:	This routine prints out a debug message for shared cache problems
		that have come up.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void ScaDebugMsg(
	const char *	pszMsg,		// Message to be displayed
	SCACHE *			pSCache,		// Shared cache block - may be NULL
	SCACHE_USE *	pUse)			// Shared cache use structure - may be NULL
{
#ifndef SCACHE_DEBUG
	F_UNREFERENCED_PARM( pszMsg);
	F_UNREFERENCED_PARM( pSCache);
	F_UNREFERENCED_PARM( pUse);
	flmAssert( 0);
#else
	char		szMsg [100];
	FLMUINT	uiChar;

	f_sprintf( szMsg, "SHARED CACHE MESSAGE FROM THREAD: %u\r\n",
			(unsigned)f_threadId());
	WpsStrOut( (WFSTR)szMsg);
	WpsStrOut( (WFSTR)pszMsg);
	WpsStrOut( (WFSTR)"\r\n");
	if (pSCache)
	{
		WpsStrOut( (WFSTR)szMsg);
		f_sprintf( szMsg, "Cache Block: 0x%08X\r\n", (unsigned)pSCache->uiBlkAddress);
		WpsStrOut( (WFSTR)szMsg);
		f_sprintf( szMsg, "Total Use Count: %u\r\n", (unsigned)pSCache->uiUseCount);
		WpsStrOut( (WFSTR)szMsg);
	}
	if (pUse)
	{
		f_sprintf( szMsg, "USE Thread ID: %u\r\n", (unsigned)pUse->uiThreadId);
		WpsStrOut( (WFSTR)szMsg);
		f_sprintf( szMsg, "Thread Use Count: %u\r\n", (unsigned)pUse->uiUseCount);
		WpsStrOut( (WFSTR)szMsg);
	}
	WpsStrOut( (WFSTR)"Press 'D' to debug, any other character to continue\r\n");
	uiChar = (FLMUINT)WpkIncar();
	WpsStrOut( (WFSTR)"\r\n\r\n");
	if ((uiChar == 'd') || (uiChar == 'D'))
	{
		flmAssert( 0);
	}
#endif
}
#endif
	
/****************************************************************************
Desc:	This routine notifies threads waiting for a pending read or write
		to complete.
		NOTE:  This routine assumes that the global mutex is already
		locked.
****************************************************************************/
FSTATIC void ScaNotify(
	F_NOTIFY_LIST_ITEM *		pNotify,
	SCACHE *						pUseSCache,
	RCODE							NotifyRc)
{
	while( pNotify)
	{
		F_SEM	hSem;

		*(pNotify->pRc) = NotifyRc;
		
		if( RC_OK( NotifyRc))
		{
			if( pNotify->pvData)
			{
				*((SCACHE **)pNotify->pvData) = pUseSCache;
			}
			
			if (pUseSCache)
			{
				ScaUseForThread( pUseSCache, &(pNotify->uiThreadId));
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
FSTATIC void scaLogFlgChange(
	SCACHE *		pSCache,
	FLMUINT16	ui16OldFlags,
	char			cPlace)
{
	char			szBuf [60];
	char *		pszTmp;
	FLMUINT16	ui16NewFlags = pSCache->ui16Flags;

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
		flmDbgLogWrite( pSCache->pFile ? pSCache->pFile->uiFFileId : 0,
			pSCache->uiBlkAddress, 0, scaGetLowTransID( pSCache), szBuf);
	}
}
#endif

/****************************************************************************
Desc:	This routine frees the memory for a cache block and decrements the
		necessary counters in the cache manager.
		NOTE:  This routine assumes that the global mutex is already locked.
****************************************************************************/
FSTATIC void ScaFree(
	SCACHE *			pSCache)
{
	FLMUINT	uiSize = SCA_MEM_SIZE( pSCache);
	
	f_assertMutexLocked( gv_FlmSysData.hShareMutex);

	if (pSCache->uiHighTransID != 0xFFFFFFFF)
	{
		flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes >= uiSize);
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes -= uiSize;
		flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount);
		gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount--;
	}

	gv_FlmSysData.SCacheMgr.Usage.uiCount--;
	gv_FlmSysData.SCacheMgr.pBlockAllocators[ 
		pSCache->ui16BlkSize == 4096 ? 0 : 1]->freeBlock( 
		(void **)&pSCache->pucBlk);
	gv_FlmSysData.SCacheMgr.pSCacheAllocator->freeCell( pSCache);
	pSCache = NULL;
}

/****************************************************************************
Desc:	This routine unlinks a cache block from all of its lists and then
		optionally frees it.  NOTE:  This routine assumes that the global
		mutex is already locked.
****************************************************************************/
FSTATIC void ScaUnlinkCache(
	SCACHE *				pSCache,
	FLMBOOL				bFreeIt,
	RCODE					NotifyRc)
{
#ifdef FLM_DEBUG
	SCACHE_USE *	pUse;
#endif

	// Cache block better not be dirty and better not need to be written
	// to the log.

#ifdef FLM_DEBUG
	if( RC_OK( NotifyRc))
	{
		flmAssert (!(pSCache->ui16Flags &
					(CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP | 
					CA_WAS_DIRTY | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
	}
#endif

	ScaUnlinkFromGlobalList( pSCache);

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( pSCache->pFile ? pSCache->pFile->uiFFileId : 0,
							pSCache->uiBlkAddress, 0,
							scaGetLowTransID( pSCache),
							"UNLINK");
#endif

	// If cache block has no previous versions linked to it, it
	// is in the hash bucket and needs to be unlinked from it.
	// Otherwise, it only needs to be unlinked from the version list.

	if (pSCache->pFile)
	{
		if (!pSCache->pPrevInVersionList)
		{
			SCACHE **	ppSCacheBucket;

			ppSCacheBucket = ScaHash(
					pSCache->pFile->FileHdr.uiSigBitsInBlkSize,
					pSCache->uiBlkAddress);
			ScaUnlinkFromHashBucket( pSCache, ppSCacheBucket);
			if (pSCache->pNextInVersionList)
			{

				// Older version better not be needing to be logged

#ifdef FLM_DEBUG
				if( RC_OK( NotifyRc))
				{
					flmAssert( !(pSCache->pNextInVersionList->ui16Flags &
								(CA_WRITE_TO_LOG | CA_LOG_FOR_CP | CA_WAS_DIRTY)));
				}
#endif
				pSCache->pNextInVersionList->pPrevInVersionList = NULL;
				ScaLinkToHashBucket( pSCache->pNextInVersionList, ppSCacheBucket);
				scaVerifyCache( pSCache->pNextInVersionList, 2100);
				pSCache->pNextInVersionList = NULL;
			}
		}
		else
		{
			scaVerifyCache( pSCache, 2000);
			ScaSavePrevBlkAddress( pSCache);
			pSCache->pPrevInVersionList->pNextInVersionList =
				pSCache->pNextInVersionList;
			scaVerifyCache( pSCache->pPrevInVersionList, 2200);
			if (pSCache->pNextInVersionList)
			{

				// Older version better not be dirty or not yet logged.

#ifdef FLM_DEBUG
				if( RC_OK( NotifyRc))
				{
					flmAssert( !(pSCache->pNextInVersionList->ui16Flags &
									(CA_WRITE_TO_LOG | CA_DIRTY | 
									CA_WAS_DIRTY | CA_IN_FILE_LOG_LIST | CA_IN_NEW_LIST)));
				}
#endif
				pSCache->pNextInVersionList->pPrevInVersionList =
					pSCache->pPrevInVersionList;
				scaVerifyCache( pSCache->pNextInVersionList, 2300);
			}
			pSCache->pNextInVersionList = pSCache->pPrevInVersionList = NULL;
		}
#ifdef SCACHE_LINK_CHECKING

		// Verify that the thing is not in a hash bucket.
		{
			SCACHE **	ppSCacheBucket;
			SCACHE *		pTmpSCache;

			ppSCacheBucket = ScaHash(
					pSCache->pFile->FileHdr.uiSigBitsInBlkSize,
					pSCache->uiBlkAddress);
			pTmpSCache = *ppSCacheBucket;
			while ((pTmpSCache) && (pSCache != pTmpSCache))
			{
				pTmpSCache = pTmpSCache->pNextInHashBucket;
			}

			if (pTmpSCache)
			{
				f_breakpoint(4);
			}
		}
#endif

		ScaUnlinkFromFile( pSCache);
	}

	if (bFreeIt)
	{
		// Free the notify list associated with the cache block.
		// NOTE: If there is actually a notify list, NotifyRc WILL ALWAYS 
		// be something other than FERR_OK.  If there is a notify list, 
		// the notified threads will thus get a non-OK return code
		// in every case.

#ifdef FLM_DEBUG
		if( NotifyRc == FERR_OK)
		{
			flmAssert( pSCache->pNotifyList == NULL);
		}
#endif

		ScaNotify( pSCache->pNotifyList, NULL, NotifyRc);
		pSCache->pNotifyList = NULL;

#ifdef FLM_DEBUG
		if (pSCache->uiUseCount)
		{
			ScaDebugMsg( "Releasing cache that is in use",
					pSCache, pSCache->pUseList);
		}

		// Free the use list associated with the cache block

		pUse = pSCache->pUseList;
		while (pUse)
		{
			SCACHE_USE *	pTmp;

			ScaDebugMsg( "Releasing cache that is in use", pSCache, pUse);
			pTmp = pUse;
			pUse = pUse->pNext;

			f_free( &pTmp);
		}
#endif
		ScaFree( pSCache);
	}
}

/****************************************************************************
Desc:	Unlink all log blocks for a file that were logged during the transaction.
		NOTE: This is only called when a transaction is aborted.
		WHEN A TRANSACTION IS ABORTED, THIS FUNCTION SHOULD BE CALLED BEFORE
		FREEING DIRTY BLOCKS.  OTHERWISE, THE pPrevInVersionList POINTER
		WILL BE NULL AND WILL CAUSE AN ABEND WHEN IT IS ACCESSED.
		NOTE: This routine assumes that the global mutex has been
		locked.
****************************************************************************/
FSTATIC void ScaUnlinkTransLogBlocks(
	FFILE *			pFile)
{
	SCACHE *	pSCache;
	SCACHE *	pNextSCache;

	pSCache = pFile->pTransLogList;
	while (pSCache)
	{
#ifdef FLM_DBG_LOG
		FLMUINT16	ui16OldFlags = pSCache->ui16Flags;
#endif

		if (pSCache->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
		{
			flmAssert( pFile->uiLogCacheCount);
			pFile->uiLogCacheCount--;
		}

		scaClearFlags( pSCache, CA_WRITE_TO_LOG | CA_LOG_FOR_CP);
		pNextSCache = pSCache->pNextInHashBucket;

		if (pSCache->ui16Flags & CA_WAS_DIRTY)
		{
			scaSetDirtyFlag( pSCache, pFile);
			scaClearFlags( pSCache, CA_WAS_DIRTY);

			// Move the block into the dirty blocks.

			ScaUnlinkFromFile( pSCache);
			ScaLinkToFile( pSCache, pFile);
		}

#ifdef FLM_DBG_LOG
		scaLogFlgChange( pSCache, ui16OldFlags, 'A');
#endif

		// Perhaps we don't really need to set these pointers to NULL,
		// but it helps keep things clean.

		pSCache->pNextInHashBucket =
		pSCache->pPrevInHashBucket = (SCACHE *)NULL;
		pSCache = pNextSCache;
	}

	pFile->pTransLogList = NULL;
}

/****************************************************************************
Desc:	Unlink a cache block from the list of cache blocks that are in the log
		list for the current transaction.
****************************************************************************/
FSTATIC void ScaUnlinkFromTransLogList(
	SCACHE *			pSCache,
	FFILE *			pFile)
{

#ifdef SCACHE_LINK_CHECKING

	// Make sure the block is not in a hash bucket

	{
		SCACHE **	ppSCacheBucket;
		SCACHE *		pTmpSCache;

		ppSCacheBucket = ScaHash(
					pFile->FileHdr.uiSigBitsInBlkSize,
					pSCache->uiBlkAddress);
		pTmpSCache = *ppSCacheBucket;
		while (pTmpSCache && pTmpSCache != pSCache)
			pTmpSCache = pTmpSCache->pNextInHashBucket;
		if (pTmpSCache)
		{
			f_breakpoint( 1001);
		}

		// Make sure the block is in the log list.

		pTmpSCache = pFile->pTransLogList;
		while (pTmpSCache && pTmpSCache != pSCache)
			pTmpSCache = pTmpSCache->pNextInHashBucket;
		if (!pTmpSCache)
		{
			f_breakpoint(1002);
		}
	}
#endif

	if (pSCache->pPrevInHashBucket)
	{
		pSCache->pPrevInHashBucket->pNextInHashBucket = pSCache->pNextInHashBucket;
	}
	else
	{
		pFile->pTransLogList = pSCache->pNextInHashBucket;
	}

	if (pSCache->pNextInHashBucket)
	{
		pSCache->pNextInHashBucket->pPrevInHashBucket = pSCache->pPrevInHashBucket;
	}

	pSCache->pNextInHashBucket =
	pSCache->pPrevInHashBucket = NULL;
}

/****************************************************************************
Desc:	The block pointed to by pSCache is about to be removed from from the
		version list for a particular block address because it is no longer
		needed.  Before doing that, the previous block address should be
		moved to the next newer version's block header so that it will not be
		lost, but only if the next newer version's block header is not already
		pointing to a prior version of the block.
****************************************************************************/
FSTATIC void ScaSavePrevBlkAddress(
	SCACHE *		pSCache)
{
	FLMUINT		uiPrevBlkAddress = scaGetPriorImageAddress( pSCache);
	SCACHE *		pNewerSCache;
	FLMUINT		uiNewerBlkPrevAddress;
	FLMUINT		uiPrevTransID;
	FLMBYTE *	pucTmp;

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
		 (uiPrevBlkAddress != BT_END) &&
		 ((pNewerSCache = pSCache->pPrevInVersionList) != NULL) &&
		 (!(pNewerSCache->ui16Flags & CA_READ_PENDING)))
	{

		// Only move the older version's previous block address to the
		// newer version, if the newer version doesn't already have a
		// previous block address.  Also need to set the previous
		// transaction ID.

		uiNewerBlkPrevAddress = scaGetPriorImageAddress( pNewerSCache);
		if ((!uiNewerBlkPrevAddress) || (uiNewerBlkPrevAddress == BT_END))
		{
			// Need to temporarily use the newer version of the block
			// before changing its previous block address.
			// NOTE: The newer block may or may not be dirty.  It is OK
			// to change the prior version address in the header of a
			// non-dirty block in this case.  This is because the block
			// may or may not be written out to the roll-back log.  If it
			// is, we want to make sure it has the correct prior version
			// address.  If it isn't ever written out to the log, it
			// will eventually fall out of cache because it is no longer
			// needed.

			ScaUseForThread( pNewerSCache, NULL);

			pucTmp = &pNewerSCache->pucBlk [BH_PREV_BLK_ADDR];
			UD2FBA( (FLMUINT32)uiPrevBlkAddress, pucTmp);

			pucTmp = &pNewerSCache->pucBlk [BH_PREV_TRANS_ID];
			uiPrevTransID = scaGetPriorImageTransID( pSCache);
			UD2FBA( (FLMUINT32)uiPrevTransID, pucTmp);

			// Need to remove the newer block from the file log
			// list, since it no longer needs to be logged

			if( pNewerSCache->ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				ScaUnlinkFromFileLogList( pNewerSCache);
			}

			ScaReleaseForThread( pNewerSCache);
		}
	}
}

/****************************************************************************
Desc:	See if we should force a checkpoint.
****************************************************************************/
FINLINE FLMBOOL scaSeeIfForceCheckpoint(
	FLMUINT		uiCurrTime,
	FFILE *		pFile,
	CP_INFO *	pCPInfo)
{
	if( FLM_ELAPSED_TIME( uiCurrTime, pFile->uiLastCheckpointTime) >=
			gv_FlmSysData.uiMaxCPInterval)
	{
		if (pCPInfo)
		{
			pCPInfo->bForcingCheckpoint = TRUE;
			pCPInfo->iForceCheckpointReason = CP_TIME_INTERVAL_REASON;
			pCPInfo->uiForceCheckpointStartTime = (FLMUINT)FLM_GET_TIMER();
		}

		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Allocate the array that keeps track of blocks written or logged.
****************************************************************************/
FSTATIC RCODE ScaAllocBlocksArray(
	FFILE *		pFile,
	FLMUINT		uiNewSize,
	FLMBOOL		bOneArray)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiOldSize = pFile->uiBlocksDoneArraySize;

	if (!uiNewSize)
	{
		uiNewSize = uiOldSize + 500;
	}

	// Re-alloc the array

	if (RC_BAD( rc = f_realloc( 
							(FLMUINT)(uiNewSize *
										(sizeof( SCACHE *) + sizeof( SCACHE *))),
							&pFile->ppBlocksDone)))
	{
		goto Exit;
	}

	// Copy the old stuff into the two new areas of the new array.

	if (uiOldSize && !bOneArray)
	{
		f_memmove( &pFile->ppBlocksDone [uiNewSize], 
			&pFile->ppBlocksDone [uiOldSize],
			uiOldSize * sizeof( SCACHE *));
	}

	// Set the new array size

	pFile->uiBlocksDoneArraySize = uiNewSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Write out log blocks to the rollback log for a database.
****************************************************************************/
FSTATIC RCODE ScaFlushLogBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bIsCPThread,
	FLMUINT				uiMaxDirtyCache,
	FLMBOOL *			pbForceCheckpoint,
	FLMBOOL *			pbWroteAll)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiLogEof;
	FLMBYTE *			pucLogHdr;
	CP_INFO *			pCPInfo = pFile->pCPInfo;
	FLMUINT				uiBlockSize = pFile->FileHdr.uiBlockSize;
	SCACHE *				pTmpSCache;
	SCACHE *				pLastBlockToLog;
	SCACHE *				pFirstBlockToLog;
	SCACHE *				pDirtySCache;
	SCACHE *				pSavedSCache = NULL;
	FLMUINT				uiDirtyCacheLeft;
	FLMUINT				uiPrevBlkAddress;
	FLMBOOL				bMutexLocked = TRUE;
	FLMBOOL				bLoggedFirstBlk = FALSE;
	FLMBOOL				bLoggedFirstCPBlk = FALSE;
	FLMUINT				uiCurrTime;
	FLMUINT				uiSaveEOFAddr;
	FLMUINT				uiSaveFirstCPBlkAddr = 0;
	FLMBOOL				bDone = FALSE;
	SCACHE *				pUsedSCache;
	SCACHE *				pNextSCache = NULL;
	FLMUINT				uiBlocksDoneArraySize = pFile->uiBlocksDoneArraySize;
	SCACHE **			ppBlocksDone = pFile->ppBlocksDone;
	SCACHE **			ppUsedBlocks = (SCACHE **)((ppBlocksDone)
											? &ppBlocksDone [uiBlocksDoneArraySize]
											: (SCACHE **)NULL);
	FLMUINT				uiTotalLoggedBlocks = 0;
	FLMBOOL				bForceCheckpoint = *pbForceCheckpoint;
	IF_LockObject *	pWriteLockObj = pFile->pWriteLockObj;
#ifdef FLM_DBG_LOG
	FLMUINT16			ui16OldFlags;
#endif

	pFile->uiCurrLogWriteOffset = 0;

	// Get the correct log header.  If we are in an update transaction,
	// need to use the uncommitted log header.  Otherwise, use the last
	// committed log header.

	pucLogHdr = bIsCPThread
					? &pFile->ucLastCommittedLogHdr [0]
					: &pFile->ucUncommittedLogHdr [0];

	f_mutexLock( gv_FlmSysData.hShareMutex);

	uiLogEof = (FLMUINT)FB2UD( &pucLogHdr [LOG_ROLLBACK_EOF]);
	pDirtySCache = pFile->pFirstInLogList;
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

	flmAssert( pFile->pCurrLogBuffer == NULL);

	uiDirtyCacheLeft = (pFile->uiDirtyCacheCount + pFile->uiLogCacheCount) * 
							uiBlockSize;

	for (;;)
	{
		if (!pDirtySCache)
		{
			bDone = TRUE;
			goto Write_Log_Blocks;
		}

		flmAssert( pDirtySCache->ui16Flags & CA_DIRTY);
		flmAssert( pDirtySCache->ui16Flags & CA_IN_FILE_LOG_LIST);
	 
		// See if we should give up our write lock.  Will do so if we
		// are not forcing a checkpoint and we have not exceeded the
		// maximum time since the last checkpoint AND the dirty cache
		// left is below the maximum.

		if (!bForceCheckpoint && bIsCPThread)
		{
			if (scaSeeIfForceCheckpoint( uiCurrTime, pFile, pCPInfo))
			{
				bForceCheckpoint = TRUE;
			}
			else
			{
				if (pWriteLockObj->getWaiterCount() &&
					 uiDirtyCacheLeft <= uiMaxDirtyCache)
				{
					bDone = TRUE;
					*pbWroteAll = FALSE;
					goto Write_Log_Blocks;
				}
			}
		}

		uiPrevBlkAddress = scaGetPriorImageAddress( pDirtySCache);
		if (uiPrevBlkAddress && uiPrevBlkAddress != BT_END)
		{
			// We shouldn't find anything in the log list that has
			// already been logged.  However, if we do find something,
			// we will deal with it rather than returning an error.

			flmAssert( 0);
			pTmpSCache = pDirtySCache->pNextInReplaceList;
			ScaUnlinkFromFileLogList( pDirtySCache);
			pDirtySCache = pTmpSCache;
			continue;
		}

		// The replace list pointers are used to maintain links
		// between items in the file log list

		pTmpSCache = pDirtySCache->pNextInVersionList;
		pLastBlockToLog = NULL;
		pFirstBlockToLog = NULL;

		// Grab the next block in the chain and see if we are done.
		// NOTE: pDirtySCache should not be accessed in the loop
		// below, because it has been changed to point to the
		// next cache block in the log list.  If you need to access
		// the current block, use pSavedSCache.

		pSavedSCache = pDirtySCache;
		if ((pDirtySCache = pDirtySCache->pNextInReplaceList) == NULL)
		{
			bDone = TRUE;
		}
#ifdef FLM_DEBUG
		else
		{
			flmAssert( pDirtySCache->ui16Flags & CA_DIRTY);
			flmAssert( pDirtySCache->ui16Flags & CA_IN_FILE_LOG_LIST);
		}
#endif

		// Traverse down the list of prior versions of the block until
		// we hit one that has a prior version on disk.  Throw out
		// any not marked as CA_WRITE_TO_LOG, CA_LOG_FOR_CP, and
		// not needed by a read transaction.

		while (pTmpSCache)
		{
			pNextSCache = pTmpSCache->pNextInVersionList;
			FLMBOOL	bWillLog;

			uiPrevBlkAddress = scaGetPriorImageAddress( pTmpSCache);

			// If we determine that we need to log a block, put a use on the
			// newer version of the block to prevent other threads from verifying
			// their checksums while we are writing the older versions to
			// the log.  This is because lgOutputBlock may modify information
			// in the newer block's header area.

			if (pTmpSCache->ui16Flags & CA_READ_PENDING)
			{

				// No need to go further down the list if this block is
				// being read in.  If it is being read in, every older
				// version has a path to it - otherwise, it would never
				// have been written out so that it would need to be
				// read back in.

				break;
			}
			else if (pTmpSCache->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
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

			else if (ScaNeededByReadTrans( pFile, pTmpSCache) ||
						pTmpSCache->uiUseCount)
			{
				bWillLog = TRUE;
			}
			else
			{
				bWillLog = FALSE;

				// Since the block is no longer needed by a read transaction,
				// and it is not in use, free it

				ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
			}

			// Add this block to the list of those we will be logging if the
			// bWillLog flag got set above.

			if (bWillLog)
			{
				if (uiTotalLoggedBlocks >= uiBlocksDoneArraySize)
				{
					if (RC_BAD( rc = ScaAllocBlocksArray( pFile, 0, FALSE)))
					{
						goto Exit;
					}
					ppBlocksDone = pFile->ppBlocksDone;
					uiBlocksDoneArraySize = pFile->uiBlocksDoneArraySize;
					ppUsedBlocks = &ppBlocksDone [uiBlocksDoneArraySize];
				}

				pLastBlockToLog = pTmpSCache;
				if (!pFirstBlockToLog)
				{
					pFirstBlockToLog = pLastBlockToLog;
				}

				ScaUseForThread( pTmpSCache->pPrevInVersionList, NULL);
				ScaUseForThread( pTmpSCache, NULL);
				ppBlocksDone [uiTotalLoggedBlocks] = pTmpSCache;
				ppUsedBlocks [uiTotalLoggedBlocks] = pTmpSCache->pPrevInVersionList;
				uiTotalLoggedBlocks++;
			}

			// No need to go further down the list if this block has
			// has a previous block address.

			if (uiPrevBlkAddress && uiPrevBlkAddress != BT_END)
			{
				break;
			}
			pTmpSCache = pNextSCache;
		}

#ifdef FLM_DEBUG
		while (pNextSCache)
		{
			flmAssert( !(pNextSCache->ui16Flags &
							 (CA_WRITE_TO_LOG | CA_LOG_FOR_CP)));
			pNextSCache = pNextSCache->pNextInVersionList;

		}
#endif

		// If nothing to log for the block, unlink it from the
		// log list.  We check CA_IN_FILE_LOG_LIST again, because
		// ScaSavePrevBlkAddress may have been called during an
		// unlink above.  ScaSavePrevBlkAddress will remove 
		// the dirty cache block from the log list if it determines
		// that there is no need to log prior versions

		if( !pLastBlockToLog)
		{
			if( pSavedSCache->ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				ScaUnlinkFromFileLogList( pSavedSCache);
			}
			continue;
		}

		// Don't want the mutex locked while we do the I/O

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Write the log blocks to the rollback log.
		// Do all of the blocks from oldest to most current.  Stop when we
		// hit the first log block.

		while( pLastBlockToLog)
		{
			FLMUINT	uiLogPos = uiLogEof;

			if( RC_BAD( rc = lgOutputBlock( pDbStats, pSFileHdl,
											pFile, pLastBlockToLog,
											pLastBlockToLog->pPrevInVersionList->pucBlk,
											&uiLogEof)))
			{
				goto Exit;
			}

			if( pLastBlockToLog->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
			{
				flmAssert( uiDirtyCacheLeft >= uiBlockSize);
				uiDirtyCacheLeft -= uiBlockSize;
			}

			// If we are logging a block for the current update
			// transaction, and this is the first block we have logged,
			// remember the block address where we logged it.

			if( (pLastBlockToLog->ui16Flags & CA_WRITE_TO_LOG) &&
				 !pFile->uiFirstLogBlkAddress)
			{
				// This better not EVER happen in the CP thread.

				flmAssert( !bIsCPThread);
				bLoggedFirstBlk = TRUE;
				pFile->uiFirstLogBlkAddress = uiLogPos;
			}

			// If we are logging the checkpoint version of the
			// block, and this is the first block we have logged
			// since the last checkpoint, remember its position so
			// that we can write it out to the log header when we
			// complete the checkpoint.

			if( (pLastBlockToLog->ui16Flags & CA_LOG_FOR_CP) &&
				 !pFile->uiFirstLogCPBlkAddress)
			{
				bLoggedFirstCPBlk = TRUE;
				pFile->uiFirstLogCPBlkAddress = uiLogPos;
			}

			// Break when we hit the first log block.

			if( pLastBlockToLog == pFirstBlockToLog)
			{
				break;
			}

			pLastBlockToLog = pLastBlockToLog->pPrevInVersionList;
		}

		// If we have logged some blocks, force the log header to be
		// updated on one of the following conditions:

		// 1. We have logged over 2000 blocks.  We do this to keep
		//		our array of logged blocks from growing too big.
		//	2.	We are done logging.

Write_Log_Blocks:

		if( uiTotalLoggedBlocks && (uiTotalLoggedBlocks >= 2000 || bDone))
		{
			if (bMutexLocked)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				bMutexLocked = FALSE;
			}

			// Flush the last log buffer, if not already flushed.

			if (pFile->uiCurrLogWriteOffset)
			{
				if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl, pFile)))
				{
					goto Exit;
				}
			}

			// If doing async, wait for pending writes to complete before writing
			// the log header.

			if( RC_BAD( rc = pFile->pBufferMgr->waitForAllPendingIO()))
			{
				goto Exit;
			}

			// Must wait for all RFL writes before writing out log header.

			if( !bIsCPThread)
			{
				(void)pFile->pRfl->seeIfRflWritesDone( TRUE);
			}

			// Save the EOF address so we can restore it if
			// the write fails.

			uiSaveEOFAddr = (FLMUINT)FB2UD( &pucLogHdr [LOG_ROLLBACK_EOF]);
			UD2FBA( (FLMUINT32)uiLogEof, &pucLogHdr [LOG_ROLLBACK_EOF]);

			if( bLoggedFirstCPBlk)
			{
				uiSaveFirstCPBlkAddr =
					(FLMUINT)FB2UD( &pucLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]);
				UD2FBA( (FLMUINT32)pFile->uiFirstLogCPBlkAddress,
								&pucLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]);
			}

			if( RC_BAD( rc = flmWriteLogHdr( pDbStats, pSFileHdl, pFile,
									pucLogHdr, pFile->ucCheckpointLogHdr, FALSE)))
			{
				// If the write of the log header fails,
				// we want to restore the log header to what it was before
				// because we always use the log header from memory instead
				// of reading it from disk.  The one on disk is only
				// current for many fields as of the last checkpoint.

				UD2FBA( (FLMUINT32)uiSaveEOFAddr, &pucLogHdr [LOG_ROLLBACK_EOF]);
				
				if( bLoggedFirstCPBlk)
				{
					UD2FBA( (FLMUINT32)uiSaveFirstCPBlkAddr,
									&pucLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]);
				}
				
				goto Exit;
			}

			// Need to update the committed log header when we are operating in
			// an uncommitted transaction so that if the transaction turns out
			// to be empty, we will have the correct values in the committed
			// log header for subsequent transactions or the checkpoint thread
			// itself.

			if( !bIsCPThread)
			{
				f_memcpy( &pFile->ucLastCommittedLogHdr [LOG_ROLLBACK_EOF],
					&pucLogHdr [LOG_ROLLBACK_EOF], 4);

				if( bLoggedFirstCPBlk)
				{
					f_memcpy(
						&pFile->ucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR],
						&pucLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR], 4);
				}
			}

			// Once the write is safe, we can reset things to start over.

			bLoggedFirstBlk = FALSE;
			bLoggedFirstCPBlk = FALSE;

			// Clean up the log blocks array - releasing blocks, etc.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
			
			if (pCPInfo)
			{
				pCPInfo->uiLogBlocksWritten += uiTotalLoggedBlocks;
			}

			while (uiTotalLoggedBlocks)
			{
				uiTotalLoggedBlocks--;
				pTmpSCache = ppBlocksDone [uiTotalLoggedBlocks];
#ifdef FLM_DBG_LOG
				ui16OldFlags = pTmpSCache->ui16Flags;
#endif
				pUsedSCache = ppUsedBlocks [uiTotalLoggedBlocks];

				// Newer block should be released, whether we succeeded
				// or not - because it will always have been used.

				ScaReleaseForThread( pUsedSCache);
				ScaReleaseForThread( pTmpSCache);

				// The current version of the block may have already been removed from
				// the file log list if more than one block in the version chain
				// needed to be logged.  If the block is still in the file log list,
				// it will be removed.  Otherwise, the prior image address better
				// be a non-zero value.

				if( pUsedSCache->ui16Flags & CA_IN_FILE_LOG_LIST)
				{
					ScaUnlinkFromFileLogList( pUsedSCache);
				}

#ifdef FLM_DEBUG
				{
					FLMUINT uiTmpPriorAddr = scaGetPriorImageAddress( pUsedSCache);
					flmAssert( uiTmpPriorAddr != 0 && uiTmpPriorAddr != BT_END);
				}
#endif

				// Unlink from list of transaction log blocks

				if( pTmpSCache->ui16Flags & CA_WRITE_TO_LOG)
				{
					ScaUnlinkFromTransLogList( pTmpSCache, pFile);
				}

				// Unset logging flags on logged block.

				if( pTmpSCache->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
				{
					flmAssert( pFile->uiLogCacheCount);
					pFile->uiLogCacheCount--;
				}

				scaClearFlags( pTmpSCache, 
					CA_LOG_FOR_CP | CA_WRITE_TO_LOG | CA_WAS_DIRTY);

#ifdef FLM_DBG_LOG
				scaLogFlgChange( pTmpSCache, ui16OldFlags, 'D');
#endif
				if( !pTmpSCache->uiUseCount &&
				    !pTmpSCache->ui16Flags &&
					 !ScaNeededByReadTrans( pTmpSCache->pFile, pTmpSCache))
				{
					flmAssert( pTmpSCache->uiHighTransID != 0xFFFFFFFF);
					ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
				}
			}

			uiDirtyCacheLeft = 
					(pFile->uiDirtyCacheCount + pFile->uiLogCacheCount) * 
					uiBlockSize;

			// When the current set of log blocks were flushed, they were
			// also unlinked from the file log list.  So, we need to
			// start at the beginning of the log list to pick up
			// where we left off.

			pDirtySCache = pFile->pFirstInLogList;
		}
		else if( !bDone)
		{
			if( !bMutexLocked)
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);
				bMutexLocked = TRUE;
			}

			// Need to reset pDirtySCache here because the background cache
			// cleanup thread may have unlinked it from the log list and
			// cleaned up any prior versions if it determined that the blocks
			// were no longer needed.

			if( (pDirtySCache = pSavedSCache->pNextInReplaceList) == NULL)
			{
				bDone = TRUE;
				goto Write_Log_Blocks;
			}
		}

		if( bDone)
		{
			break;
		}

		flmAssert( bMutexLocked);
	}

#ifdef FLM_DEBUG
	if( bForceCheckpoint || !bIsCPThread ||
		(!bForceCheckpoint && bIsCPThread && *pbWroteAll))
	{
		flmAssert( !pFile->uiLogListCount);
		flmAssert( !pFile->uiLogCacheCount);
	}
#endif

Exit:

	if( RC_BAD( rc))
	{
		// Flush the last log buffer, if not already flushed.

		if( pFile->uiCurrLogWriteOffset)
		{

			if( bMutexLocked)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				bMutexLocked = FALSE;
			}

			// Don't care what rc is at this point.  Just calling
			// lgFlushLogBuffer to clear the buffer.

			(void)lgFlushLogBuffer( pDbStats, pSFileHdl, pFile);
		}

		// Need to wait for any async writes to complete.

		if( bMutexLocked)
		{
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			bMutexLocked = FALSE;
		}

		// Don't care about rc here, but we don't want to leave
		// this routine until all pending IO is taken care of.

		(void)pFile->pBufferMgr->waitForAllPendingIO();

		if( !bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		// Clean up the log blocks array - releasing blocks, etc.

		while( uiTotalLoggedBlocks)
		{
			FLMBYTE *	pucTmp;

			uiTotalLoggedBlocks--;
			pTmpSCache = ppBlocksDone [uiTotalLoggedBlocks];
			pUsedSCache = ppUsedBlocks [uiTotalLoggedBlocks];

#ifdef FLM_DEBUG

			// If this is the most current version of the block, it
			// should still be in the file log list.

			if( !pUsedSCache->pPrevInVersionList)
			{
				flmAssert( pUsedSCache->ui16Flags & CA_IN_FILE_LOG_LIST);
			}
#endif

			// Used blocks should be released, whether we succeeded
			// or not.

			ScaReleaseForThread( pUsedSCache);
			ScaReleaseForThread( pTmpSCache);

			// If we quit before logging the blocks, we don't really
			// want to change anything on the block, but we do want
			// to set the previous block address back to zero on the
			// block that is just newer than this one.

			pucTmp = pTmpSCache->pPrevInVersionList->pucBlk;

			// Must put a USE on the block so that the memory cache
			// verifying code will not barf when we change the
			// data in the block - checksum is calculated and set when
			// the use count goes from one to zero, and then verified
			// when it goes from zero to one.

			ScaUseForThread( pTmpSCache->pPrevInVersionList, NULL);
			UD2FBA( 0, &pucTmp [BH_PREV_BLK_ADDR]);
			ScaReleaseForThread( pTmpSCache->pPrevInVersionList);
		}

#ifdef SCACHE_LINK_CHECKING

		// If above logic changes where mutex might not be locked at
		// this point, be sure to modify this code to re-lock it.

		flmAssert( bMutexLocked);
		scaVerify( 100);
#endif

		// Things to restore to their original state if we had an error.

		if( bLoggedFirstBlk)
		{
			pFile->uiFirstLogBlkAddress = 0;
		}
		
		if( bLoggedFirstCPBlk)
		{
			pFile->uiFirstLogCPBlkAddress = 0;
		}
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	// Better not be any incomplete writes at this point.

	flmAssert( !pFile->pBufferMgr->isIOPending());
	flmAssert( pFile->pCurrLogBuffer == NULL);

	*pbForceCheckpoint = bForceCheckpoint;
	return( rc);
}

/****************************************************************************
Desc:	This routine is called whenever a write of a dirty block completes.
****************************************************************************/
FSTATIC void FLMAPI scaWriteComplete(
	IF_IOBuffer *		pIOBuffer,
	void *				pvData)
{
	RCODE					rc;
	FLMUINT				uiNumBlocks = 0;
	SCACHE *				pSCache = NULL;
	FFILE *				pFile;
	DB_STATS *			pDbStats = (DB_STATS *)pvData;
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

	f_mutexLock( gv_FlmSysData.hShareMutex);
	while( uiNumBlocks)
	{
		uiNumBlocks--;
		pSCache = (SCACHE *)pIOBuffer->getCallbackData( uiNumBlocks);
		pFile = pSCache->pFile;

		if( pDbStats)
		{
			FLMBYTE *			pucBlk = pSCache->pucBlk;
			FLMUINT				uiLFileNum;
			LFILE_STATS *		pLFileStats;
			BLOCKIO_STATS *	pBlockIOStats;
			FLMUINT				uiBlkType;
			FLMUINT				uiLfType;

			if( (uiLFileNum = (FLMUINT)FB2UW( &pucBlk [BH_LOG_FILE_NUM])) == 0)
			{
				pLFileStats = NULL;
			}
			else
			{
				uiLfType = 0xFF;

				if( uiLFileNum == FLM_DICT_INDEX)
				{
					uiLfType = LF_INDEX;
				}
				else if( uiLFileNum == FLM_DATA_CONTAINER ||
							uiLFileNum == FLM_DICT_CONTAINER ||
							uiLFileNum == FLM_TRACKER_CONTAINER)
				{
					uiLfType = LF_CONTAINER;
				}

				if( RC_BAD( flmStatGetLFile( pDbStats, uiLFileNum,
									uiLfType, 0, &pLFileStats, NULL, NULL)))
				{
					pLFileStats = NULL;
				}
			}

			if( pLFileStats)
			{
				uiBlkType = BHT_LEAF;
			}
			else
			{
				uiBlkType = (FLMUINT)(BH_GET_TYPE( pucBlk));
			}

			if( (pBlockIOStats = flmGetBlockIOStatPtr( pDbStats,
							pLFileStats, pucBlk, uiBlkType)) != NULL)
			{
				pBlockIOStats->BlockWrites.ui64Count++;
				pBlockIOStats->BlockWrites.ui64TotalBytes +=
						pFile->FileHdr.uiBlockSize;

				if( uiExtraMilli)
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

		ScaReleaseForThread( pSCache);
		
		if( pSCache->ui16Flags & CA_DIRTY)
		{
			flmAssert( pSCache->ui16Flags & CA_WRITE_PENDING);
#ifdef FLM_DBG_LOG
			ui16OldFlags = pSCache->ui16Flags;
#endif
			scaClearFlags( pSCache, CA_WRITE_PENDING);
			
			if( RC_OK( rc))
			{
				scaUnsetDirtyFlag( pSCache, pFile);
			}

#ifdef FLM_DBG_LOG
			scaLogFlgChange( pSCache, ui16OldFlags, 'H');
#endif

			// If there are more dirty blocks after this
			// one, move this one out of the dirty
			// blocks.

			ScaUnlinkFromFile( pSCache);
			ScaLinkToFile( pSCache, pFile);
		}
		else
		{
			flmAssert( !(pSCache->ui16Flags & CA_WRITE_PENDING));
		}
	}
	
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	Cleanup old blocks in cache that are no longer needed by any
		transaction.
****************************************************************************/
void ScaCleanupCache(
	FLMUINT			uiMaxLockTime)
{
	SCACHE *			pTmpSCache;
	SCACHE *			pPrevSCache;
	FLMUINT			uiBlocksExamined = 0;
	FLMUINT			uiLastTimePaused = FLM_GET_TIMER();
	FLMUINT			uiCurrTime;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUReplace;

	for (;;)
	{
		// Stop when we reach end of list or all old blocks have
		// been freed.

		if( !pTmpSCache ||
			 !gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes)
		{
			break;
		}

		// Shouldn't encounter anything with CA_FREE set

		flmAssert( !(pTmpSCache->ui16Flags & CA_FREE));

		// After each 200 blocks examined, see if our maximum
		// time has elapsed for examining without a pause.

		if (uiBlocksExamined >= 200)
		{
			uiBlocksExamined = 0;
			uiCurrTime = FLM_GET_TIMER();

			if( FLM_ELAPSED_TIME( uiCurrTime, uiLastTimePaused) >= uiMaxLockTime)
			{
				// Increment the use count so that this block will not
				// go away while we are paused.

				ScaUseForThread( pTmpSCache, NULL);
				f_mutexUnlock( gv_FlmSysData.hShareMutex);

				// Shortest possible pause - to allow other threads
				// to do work.

				f_yieldCPU();

				// Relock mutex.

				uiLastTimePaused = FLM_GET_TIMER();
				f_mutexLock( gv_FlmSysData.hShareMutex);

				// Decrement use count that was added on up above.

				ScaReleaseForThread( pTmpSCache);

				// If the block was freed while we had the mutex unlocked,
				// it is no longer linked into the global or replace lists.
				// We need to re-sync.

				if( (pTmpSCache->ui16Flags & CA_FREE))
				{
					pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUReplace;
					continue;
				}
			}
		}
		uiBlocksExamined++;

		// Save the pointer to the previous entry in the list because
		// we may end up unlinking pTmpSCache below, in which case we would
		// have lost the next entry.

		pPrevSCache = pTmpSCache->pPrevInReplaceList;

		// Block must not currently be in use,
		// Must not be the most current version of a block,
		// Cannot be dirty in any way,
		// Cannot be in the process of being read in from disk,
		// And must not be needed by a read transaction.

		if (!pTmpSCache->uiUseCount &&
			 pTmpSCache->uiHighTransID != 0xFFFFFFFF &&
			 !pTmpSCache->ui16Flags &&
			 (!pTmpSCache->pFile ||
			  !ScaNeededByReadTrans( pTmpSCache->pFile, pTmpSCache)))
		{
			ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
		}
		pTmpSCache = pPrevSCache;
	}

	// Defrag cache memory

	gv_FlmSysData.SCacheMgr.pBlockAllocators[ 0]->defragmentMemory();
	gv_FlmSysData.SCacheMgr.pBlockAllocators[ 1]->defragmentMemory();
	gv_FlmSysData.SCacheMgr.pSCacheAllocator->defragmentMemory();

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	Tests if a block can be freed from cache.
		NOTE: This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE FLMBOOL scaCanBeFreed(
	SCACHE *		pSCache,
	FLMBOOL		bCheckIfNeededByReader = TRUE)
{
	if( !pSCache->uiUseCount && !pSCache->ui16Flags)
	{
		SCACHE *	pNewerSCache = pSCache->pPrevInVersionList;

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
			 (pNewerSCache->ui16Flags & CA_READ_PENDING) ||
			 (scaGetPriorImageAddress( pNewerSCache) != 0 &&
			  scaGetPriorImageAddress( pNewerSCache) != BT_END) ||
			 !pSCache->pFile ||
			 (!bCheckIfNeededByReader || 
				!ScaNeededByReadTrans( pSCache->pFile, pSCache)))
		{
			return( TRUE);
		}
	}
	return( FALSE);
}

/****************************************************************************
Desc:	Reduce cache to below the cache limit.  NOTE: This routine assumes
		that the global mutex is locked.  It may temporarily unlock the mutex
		to write out dirty blocks, but it will always return with the mutex
		still locked.
****************************************************************************/
FSTATIC RCODE ScaReduceCache(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pTmpSCache;
	SCACHE *		pPrevSCache = NULL;
	FFILE *		pFile = pDb ? pDb->pFile : NULL;
	FLMBOOL		bForceCheckpoint;
	FLMBOOL		bDummy;
	FLMUINT		uiBlocksFlushed;

	// If cache is not full, we are done.

	if( !scaIsCacheOverLimit())
	{
		goto Exit;
	}

	if( gv_FlmSysData.SCacheMgr.uiFreeBytes)
	{
		scaReduceFreeCache( FALSE);

		// If cache is not full, we are done.

		if( !scaIsCacheOverLimit())
		{
			goto Exit;
		}
	}

	// If we have a lot of blocks that need to be logged, cache is full, and
	// we are in an update transaction, let's write the log blocks out first
	// before re-using blocks in the LRU replace list.  This helps to minimize
	// poisoning of the cache due to lots of prior versions that may not be
	// needed once this transaction commits.  Also, since log writes are
	// sequential, it is much more efficient to write out log blocks to
	// reduce cache than it is to write out dirty blocks.

	if( pDb && pDb->uiTransType == FLM_UPDATE_TRANS &&
		pFile->uiLogCacheCount * pFile->FileHdr.uiBlockSize >= 
			(gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated >> 2))
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bForceCheckpoint = FALSE;
		rc = ScaFlushLogBlocks( pDb->pDbStats,
						pDb->pSFileHdl, pFile, FALSE,
						~((FLMUINT)0), &bForceCheckpoint, &bDummy);
		f_mutexLock( gv_FlmSysData.hShareMutex);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// If cache is still full, try to get rid of items in the replace list

	scaReduceReuseList();

	// If cache is not full, we are done.

	if( !scaIsCacheOverLimit())
	{
		goto Exit;
	}

	// If we're still over the cache limit and this is an update transaction,
	// try writing out dirty blocks.

	if( pDb && pDb->uiTransType == FLM_UPDATE_TRANS)
	{
		// Flush out log blocks.

		if( pFile->pFirstInLogList)
		{
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			bForceCheckpoint = FALSE;
			rc = ScaFlushLogBlocks( pDb->pDbStats,
							pDb->pSFileHdl, pFile, FALSE,
							~((FLMUINT)0), &bForceCheckpoint, &bDummy);
			f_mutexLock( gv_FlmSysData.hShareMutex);

			if( RC_BAD( rc))
			{
				goto Exit;
			}

			scaReduceFreeCache( FALSE);
			scaReduceReuseList();

			if( !scaIsCacheOverLimit())
			{
				goto Exit;
			}
		}

		// Flush new blocks (if any)

		while( pFile->uiNewCount)
		{
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			rc = ScaReduceNewBlocks( pDb->pDbStats,
										pDb->pSFileHdl, pFile, &uiBlocksFlushed);
			f_mutexLock( gv_FlmSysData.hShareMutex);

			if( RC_BAD( rc))
			{
				goto Exit;
			}

			if( !uiBlocksFlushed)
			{
				// uiNewCount may not be zero when returning from ScaReduceNewBlocks
				// even if nothing was flushed.  This can happen when blocks in the
				// new list are marked with the CA_WRITE_INHIBIT flag.

				break;
			}

			if( scaIsCacheOverLimit())
			{
				scaReduceFreeCache( FALSE);
			}

			if( scaIsCacheOverLimit())
			{
				scaReduceReuseList();
			}

			if( !scaIsCacheOverLimit())
			{
				goto Exit;
			}
		}

		pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUCache;
		while( pTmpSCache && scaIsCacheOverLimit())
		{
			// Need to save the pointer to the previous entry in the list because
			// we may end up unlinking pTmpSCache below, in which case we would
			// have lost the previous entry.

			pPrevSCache = pTmpSCache->pPrevInGlobalList;

			// See if the cache block can be freed.

			if (scaCanBeFreed( pTmpSCache))
			{

				// NOTE: This call will free the memory pointed to by
				// pTmpSCache.  Hence, pTmpSCache should NOT be used after
				// this point.

				ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
			}
			else if( (pTmpSCache->ui16Flags & CA_DIRTY) &&
				pFile == pTmpSCache->pFile &&
				!(pTmpSCache->ui16Flags & CA_WRITE_INHIBIT))
			{
				ScaUseForThread( pTmpSCache, NULL);
				f_mutexUnlock( gv_FlmSysData.hShareMutex);

				flmAssert( !pFile->uiLogCacheCount);

				// This may not write out the dirty block we are looking at,
				// but that is OK, eventually it will.  It is more than
				// likely that it will, because the older dirty blocks are
				// at the front of the dirty list.

				rc = ScaFlushDirtyBlocks( pDb->pDbStats,
											pDb->pSFileHdl, pFile,
											~((FLMUINT)0), FALSE,
											FALSE, &bDummy);

				f_mutexLock( gv_FlmSysData.hShareMutex);
				ScaReleaseForThread( pTmpSCache);

				if (RC_BAD( rc))
				{
					goto Exit;
				}

				// Stay on this block until we get it written out.

				pPrevSCache = pTmpSCache;
			}
			else if( pTmpSCache->ui16Flags & (CA_LOG_FOR_CP | CA_WRITE_TO_LOG) &&
				(pFile == pTmpSCache->pFile))
			{
				flmAssert( 0);
				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}
			pTmpSCache = pPrevSCache;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Allocates memory for a cache block from the operating system.
			This routine assumes that the global mutex is locked.
****************************************************************************/
FSTATIC RCODE scaAllocCacheBlock(
	FLMUINT			uiBlockSize,
	SCACHE **		ppSCache)
{
	RCODE			rc = FERR_OK;
	SCACHE *		pSCache;

	f_assertMutexLocked( gv_FlmSysData.hShareMutex);
	
	if( (pSCache = (SCACHE *)gv_FlmSysData.SCacheMgr.pSCacheAllocator->allocCell( 
		NULL, NULL)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	f_memset( pSCache, 0, sizeof( SCACHE));
	
	if( RC_BAD( rc = gv_FlmSysData.SCacheMgr.pBlockAllocators[ 
		uiBlockSize == 4096 ? 0 : 1]->allocBlock( (void **)&pSCache->pucBlk)))
	{
		gv_FlmSysData.SCacheMgr.pSCacheAllocator->freeCell( pSCache);
		goto Exit;
	}
	
	// Set the block size.

	pSCache->ui16BlkSize = (FLMUINT16)uiBlockSize;

	// Need to set high transaction ID to 0xFFFFFFFF.  This indicates that
	// the block is not currently counted in the Usage.uiOldVerBytes tally -
	// seeing as how it was just allocated.
	// DO NOT USE scaSetTransID routine here because that routine
	// will adjust the tally.  The caller of this routine should call
	// scaSetTransID to ensure that the tally is set appropriately.
	// This is the only place in the code where it is legal to set
	// uiHighTransID without calling scaSetTransID.

	pSCache->uiHighTransID = 0xFFFFFFFF;
	*ppSCache = pSCache;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocate a cache block.  If we are at the cache limit, unused cache
		blocks will be replaced.  NOTE: This routine assumes that the global
		mutex is locked.
****************************************************************************/
FSTATIC RCODE ScaAllocCache(
	FDB *				pDb,
	SCACHE **		ppSCacheRV)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiBlockSize = pFile->FileHdr.uiBlockSize;
	SCACHE *			pSCache;
	SCACHE *			pTmpSCache;
	SCACHE *			pPrevSCache;

	// Quick check to see if there is a block in the free list that can be
	// re-used.  Start at the MRU end of the list so that if items in the
	// free list are only being used periodically, the items at the LRU end
	// will age out and the size of the list will be reduced.

	pSCache = gv_FlmSysData.SCacheMgr.pFirstFree;
	while( pSCache)
	{
		if( !pSCache->uiUseCount && 
			ScaGetBlkSize( pSCache) == uiBlockSize)
		{
			ScaUnlinkFromFreeList( pSCache);
			goto Reuse_Block;
		}
		pSCache = pSCache->pNextInFile;
	}

	// The intent of this little loop is to be optimistic and hope that
	// there is a block we can cannibalize or free without having to write
	// it.  If not, we will still allocate a new block and allow ourselves
	// to be temporarily over the cache limit.  In this case, the cache size
	// will be reduced only AFTER this new block is safely linked into cache.
	// This is necessary because we don't want two different threads allocating
	// memory for the same block.

	pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUReplace;
	while( pTmpSCache && scaIsCacheOverLimit())
	{
		// Need to save the pointer to the previous entry in the list because
		// we may end up unlinking it below, in which case we would have lost
		// the previous entry.

		pPrevSCache = pTmpSCache->pPrevInReplaceList;

		// See if the cache block can be replaced or freed.

		flmAssert( !pTmpSCache->ui16Flags);
		if( scaCanBeFreed( pTmpSCache))
		{
			if( ScaGetBlkSize( pTmpSCache) == uiBlockSize)
			{
				pSCache = pTmpSCache;
				flmAssert( !pSCache->ui16Flags);
				ScaUnlinkCache( pTmpSCache, FALSE, FERR_OK);

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

				ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
			}
		}
		pTmpSCache = pPrevSCache;
	}

	// If we were not able to cannibalize an SCACHE structure,
	// allocate one.

	if (pSCache)
	{
Reuse_Block:

		flmAssert( !pSCache->pPrevInReplaceList);
		flmAssert( !pSCache->pNextInReplaceList);
		flmAssert( !pSCache->ui16Flags);
		flmAssert( !pSCache->uiUseCount);

		// If block is an old version, need to decrement the
		// Usage.uiOldVerBytes tally.

		if (pSCache->uiHighTransID != 0xFFFFFFFF)
		{
			FLMUINT	uiSize = SCA_MEM_SIZE( pSCache);
			flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes >= uiSize);
			gv_FlmSysData.SCacheMgr.Usage.uiOldVerBytes -= uiSize;
			flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount);
			gv_FlmSysData.SCacheMgr.Usage.uiOldVerCount--;
		}

		// If we are cannibalizing, be sure to reset certain fields.

		pSCache->ui16Flags = 0;
		pSCache->uiUseCount = 0;
#ifdef FLM_DEBUG
		pSCache->uiChecksum = 0;
#endif

		// Need to set high transaction ID to 0xFFFFFFFF.  This indicates that
		// the block is not currently counted in the Usage.uiOldVerBytes tally -
		// seeing as how it was just allocated.
		// DO NOT USE scaSetTransID routine here because that routine
		// will adjust the tally.  The caller of this routine should call
		// scaSetTransID to ensure that the tally is set appropriately.
		// This is the only place in the code where it is legal to set
		// uiHighTransID without calling scaSetTransID.

		pSCache->uiHighTransID = 0xFFFFFFFF;
	}
	else
	{
		if( RC_BAD( rc = scaAllocCacheBlock( uiBlockSize, &pSCache)))
		{
			goto Exit;
		}

		gv_FlmSysData.SCacheMgr.Usage.uiCount++;
	}

	*ppSCacheRV = pSCache;

	// Set use count to one so the block cannot be replaced.  This also
	// unprotects the block so it can be accessed and modified while in
	// memory.

	ScaUseForThread( pSCache, NULL);

Exit:

	return( rc);
}

/********************************************************************
Desc:	This routine attempts to read a block from disk.  It will
		attempt the specified number of times.
*********************************************************************/
FSTATIC RCODE ScaReadTheBlock(
	FDB *	  				pDb,
	LFILE *				pLFile,
	TMP_READ_STATS *	pTmpReadStats,
	FLMBYTE *			pucBlk,
	FLMUINT				uiFilePos,
	FLMUINT				uiBlkAddress)
{
	RCODE	  				rc = FERR_OK;
	FLMUINT				uiBytesRead;
	FFILE *				pFile = pDb->pFile;
	FLMUINT				uiBlkSize = pFile->FileHdr.uiBlockSize;
	DB_STATS *			pDbStats = pDb->pDbStats;
	F_TMSTAMP			StartTime;
	FLMUINT64			ui64ElapMilli;

	// We should NEVER be attempting to read a block address that is
	// beyond the current logical end of file.

	if (!FSAddrIsBelow( uiBlkAddress, pDb->LogHdr.uiLogicalEOF))
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// Read the block

	if (pDb->uiKilledTime)
	{
		rc = RC_SET( FERR_OLD_VIEW);
		goto Exit;
	}

	if (pTmpReadStats)
	{
		if (uiFilePos != uiBlkAddress)
		{
			pTmpReadStats->OldViewBlockReads.ui64Count++;
			pTmpReadStats->OldViewBlockReads.ui64TotalBytes += uiBlkSize;
		}
		else
		{
			pTmpReadStats->BlockReads.ui64Count++;
			pTmpReadStats->BlockReads.ui64TotalBytes += uiBlkSize;
		}
		ui64ElapMilli = 0;
		f_timeGetTimeStamp( &StartTime);
	}
	
	if( RC_BAD( rc = pDb->pSFileHdl->readBlock( uiFilePos,
		uiBlkSize, pucBlk, &uiBytesRead)))
	{
		if (pDbStats)
		{
			pDbStats->uiReadErrors++;
		}

		if (rc == FERR_IO_END_OF_FILE)
		{
			flmAssert( pDb->uiKilledTime);
			rc = RC_SET( FERR_OLD_VIEW);
		}
		goto Exit;
	}

#ifdef FLM_DBG_LOG
	if (uiFilePos != uiBlkAddress)
	{
		flmDbgLogWrite( pFile->uiFFileId,
			uiBlkAddress, uiFilePos,
						FB2UD( &pucBlk [BH_TRANS_ID]),
						"LGRD");
	}
	else
	{
		flmDbgLogWrite( pFile->uiFFileId, uiBlkAddress, 0,
						FB2UD( &pucBlk [BH_TRANS_ID]),
						"READ");
	}
#endif

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

	if (uiBytesRead < uiBlkSize)
	{
		if (pTmpReadStats)
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

		// Should only be possible when reading a root block,
		// because the root block address in the LFILE may be
		// a block that was just created by an update
		// transaction.

		flmAssert( pDb->uiKilledTime);
		rc = RC_SET( FERR_OLD_VIEW);
		goto Exit;
	}

	// Verify the block checksum BEFORE decrypting or using any data.

	if( RC_BAD( rc = BlkCheckSum( pucBlk, CHECKSUM_CHECK, 
		uiBlkAddress, uiBlkSize)))
	{
		if (pTmpReadStats)
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
	
	// If this is an index block it may be encrypted, we
	// need to decrypt it before we can use it.
	// The function ScaDecryptBlock will check if the index
	// is encrypted first.  If not, it will return.
	
	if (pLFile && pLFile->uiLfType == LF_INDEX)
	{
		if (RC_BAD( rc = ScaDecryptBlock( pDb->pFile, pucBlk)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine performs a sanity check on a block that has just been
		read in from disk.
*****************************************************************************/
FSTATIC RCODE ScaBlkSanityCheck(
	FDB *			pDb,						// Can be NULL
	FFILE *		pFile,
	LFILE *		pLFile,					// Pointer to logical file structure.
												// NULL if no logical file.
	FLMBYTE *	pucBlk,					// Pointer to block to be checked.
	FLMUINT		uiBlkAddress,			// Block address.
	FLMBOOL		bCheckFullBlkAddr,
	FLMUINT		uiSanityLevel)			// Level of checking to be done.
{
	RCODE					rc = FERR_OK;
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	FLMBYTE				ucKeyBuffer [MAX_KEY_SIZ];
	LF_HDR				LogicalFile;
	LF_STATS				LfStats;
	FLMBOOL				bIsIndex = FALSE;
	FLMBOOL				bMinimalBasicCheck = FALSE;
	eCorruptionType	eElmCorruptionCode;

	if (!pDb)
	{
		uiSanityLevel = FLM_BASIC_CHECK;
		bMinimalBasicCheck = TRUE;
		pLFile = NULL;
	}
	else
	{

		// If the block is involved in an update transaction, reduce
		// the sanity check level.  This is necessary because the block
		// may have been flushed to disk before it was "completely" sane.
		// This would be a common occurrence during block splits.

		if( flmGetDbTransType( pDb) == FLM_UPDATE_TRANS &&
			((FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]) ==
			pDb->LogHdr.uiCurrTransID) &&
			(uiSanityLevel >= FLM_BASIC_CHECK))
		{
			uiSanityLevel = FLM_BASIC_CHECK;
			bMinimalBasicCheck = TRUE;
		}
	}

	// Set up a STATE_INFO structure for doing the sanity check.  If there
	// is no logical file, it is very easy, because all we really check
	// is the block header.

	if (!pLFile)
	{

		// NOTE: pDb may be NULL.

		(void)flmInitReadState( &StateInfo, &bStateInitialized,
										pFile->FileHdr.uiVersionNum,
										pDb, NULL, 0xFF, (FLMUINT)BH_GET_TYPE( pucBlk),
										ucKeyBuffer);
	}
	else
	{
		f_memset( &LogicalFile, 0, sizeof( LF_HDR));
		f_memset( &LfStats, 0, sizeof( LF_STATS));
		LogicalFile.pLfStats = &LfStats;
		LogicalFile.pLFile = pLFile;
		
		if (pLFile->uiLfType == LF_INDEX)
		{
			bIsIndex = TRUE;
		}
		
		if (bIsIndex)
		{
			if (RC_BAD( fdictGetIndex(
					pDb->pDict, pDb->pFile->bInLimitedMode,
					pLFile->uiLfNum, NULL, &LogicalFile.pIxd, TRUE)))
			{
				uiSanityLevel = FLM_BASIC_CHECK;
				LogicalFile.pIxd = NULL;
				LogicalFile.pIfd = NULL;
			}
			else
			{
				LogicalFile.pIfd = LogicalFile.pIxd->pFirstIfd;
			}
			LfStats.ui64FldRefCount = 0;
		}
		(void) flmInitReadState( &StateInfo, &bStateInitialized,
								pDb->pFile->FileHdr.uiVersionNum,
								pDb, &LogicalFile, 0xFF,
								BH_GET_TYPE(pucBlk),
								ucKeyBuffer);
	}
	StateInfo.pBlk = pucBlk;
	StateInfo.uiBlkAddress = uiBlkAddress;

	if( flmVerifyBlockHeader( &StateInfo, NULL, pFile->FileHdr.uiBlockSize,
		0L, 0L, FALSE, bCheckFullBlkAddr) != FLM_NO_CORRUPTION)
	{
		goto Error_Exit;
	}

	// If it is not a block in a logical file, we are done.
	// NOTE: If pDb is NULL, we will also go no further than
	// here - pLFile will have been set to NULL up above.

	if (!pLFile)
	{
		goto Exit;
	}

	// Read through the elements in the block

	while (StateInfo.uiElmOffset < StateInfo.uiEndOfBlock)
	{
		if (uiSanityLevel == FLM_BASIC_CHECK)
		{
			StateInfo.pElm = &StateInfo.pBlk [StateInfo.uiElmOffset];
			if (StateInfo.uiBlkType == BHT_LEAF)
			{
				if (StateInfo.uiElmOffset + BBE_KEY > StateInfo.uiEndOfBlock)
				{
					goto Error_Exit;
				}
				StateInfo.uiElmLen = (FLMUINT)(BBE_LEN( StateInfo.pElm));

				// Get the element key length and previous key count (PKC)

				StateInfo.uiElmKeyLen = (FLMUINT)(BBE_GET_KL( StateInfo.pElm));
				StateInfo.uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( StateInfo.pElm));
			}
			else if (StateInfo.uiBlkType == BHT_NON_LEAF_DATA)
			{
				if (StateInfo.uiElmOffset + StateInfo.uiElmOvhd >
							StateInfo.uiEndOfBlock)
				{
					goto Error_Exit;
				}
				StateInfo.uiElmLen = BNE_DATA_OVHD;
				StateInfo.pElmKey = StateInfo.pElm;
				StateInfo.uiElmKeyLen = 4;
				StateInfo.uiElmPKCLen = 0;
			}
			else
			{
				if (StateInfo.uiElmOffset + StateInfo.uiElmOvhd > 
					 StateInfo.uiEndOfBlock)
				{
					goto Error_Exit;
				}

				StateInfo.uiElmLen = (FLMUINT) BBE_GET_KL(StateInfo.pElm) +
							StateInfo.uiElmOvhd + 
							(BNE_IS_DOMAIN(StateInfo.pElm) ? BNE_DOMAIN_LEN : 0);

				// Get the element key length and previous key count (PKC)

				StateInfo.uiElmKeyLen = (FLMUINT)(BBE_GET_KL( StateInfo.pElm));
				StateInfo.uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( StateInfo.pElm));
			}

			// Make sure the element doesn't go beyond the end of the block

			if( StateInfo.uiElmOffset + StateInfo.uiElmLen > 
				 StateInfo.uiEndOfBlock)
			{
				goto Error_Exit;
			}

			if( !bMinimalBasicCheck)
			{
			
				// Verify the first/last flags if it is a leaf element

				if( StateInfo.uiBlkType == BHT_LEAF)
				{
					FLMUINT	uiFirstFlag = (FLMUINT)(BBE_IS_FIRST( StateInfo.pElm));
					FLMUINT	uiPrevLastFlag = StateInfo.uiElmLastFlag;

					// Verify the first element flag

					StateInfo.uiElmLastFlag = (FLMUINT)(BBE_IS_LAST( StateInfo.pElm));
					if (uiPrevLastFlag != 0xFF)
					{
						if ((uiPrevLastFlag) && (!uiFirstFlag))
						{
							goto Error_Exit;
						}
						else if ((!uiPrevLastFlag) && (uiFirstFlag))
						{
							goto Error_Exit;
						}
					}
				}

				// If we are on the last element, verify that we are indeed.
				// If we are, set the current key length to zero.

				if( (StateInfo.uiElmLen == StateInfo.uiElmOvhd) &&
					 (StateInfo.uiElmLen + StateInfo.uiElmOffset ==
					   StateInfo.uiEndOfBlock) &&
					 (StateInfo.uiNextBlkAddr == BT_END))
				{
					StateInfo.uiCurKeyLen = 0;
				}

				// If the length in a leaf element is BBE_LEM_LEN and
				// it is not the last element, we have an error.

				else if ((StateInfo.uiBlkType == BHT_LEAF) &&
							(StateInfo.uiElmLen == BBE_LEM_LEN))
				{
					goto Error_Exit;
				}

				// If this is the last element in the block, and this is the
				// last block in the chain, this had better be the LEM.

				else if ((StateInfo.uiElmOffset + StateInfo.uiElmLen ==
							 StateInfo.uiEndOfBlock) &&
							(StateInfo.uiNextBlkAddr == BT_END))
				{
					goto Error_Exit;
				}

				// Verify four things with respect to the key length:
				//
				// 1. Total key length <= MAX_KEY_SIZ
				// 2. Total key length == 4 if it is a container
				// 3. The first element does not have a non-zero PKC length.
				// 4. The PKC length is not longer than the total length of
				//	   the previous key.

				else if ((StateInfo.uiElmKeyLen + StateInfo.uiElmPKCLen > MAX_KEY_SIZ) ||
					 ((!bIsIndex) &&
					  (StateInfo.uiElmKeyLen + StateInfo.uiElmPKCLen != 4)) ||
					 ((!StateInfo.uiCurKeyLen) && (StateInfo.uiElmPKCLen)) ||
					  ((StateInfo.uiCurKeyLen) &&
						(StateInfo.uiElmPKCLen > StateInfo.uiCurKeyLen)))
				{
					goto Error_Exit;
				}
				else
				{
					// Save the current key length

					StateInfo.uiCurKeyLen =
						StateInfo.uiElmPKCLen + StateInfo.uiElmKeyLen;
				}
			}
		}
		else
		{
			if (flmVerifyElement( &StateInfo, FLM_CHK_FIELDS) != FLM_NO_CORRUPTION)
			{
				goto Error_Exit;
			}

			if ((uiSanityLevel > FLM_INTERMEDIATE_CHECK) &&
				 (StateInfo.uiBlkType == BHT_LEAF) &&
				 (StateInfo.uiCurKeyLen))
			{
				if (bIsIndex)
				{
					if( RC_BAD( rc = flmVerifyIXRefs( &StateInfo, NULL, 0,
						&eElmCorruptionCode)) || eElmCorruptionCode != FLM_NO_CORRUPTION)
					{
						goto Error_Exit;
					}
				}
				else if (StateInfo.uiElmDrn != DRN_LAST_MARKER)
				{
					// Parse through the fields in the element

					for (;;)
					{
						if (flmVerifyElmFOP( &StateInfo) != FLM_NO_CORRUPTION)
						{
							goto Error_Exit;
						}

						// Verify the field if it is entirely contained in the
						// element.

						if ((StateInfo.uiFOPDataLen == StateInfo.uiFieldLen) &&
							 (StateInfo.uiFOPDataLen > 0) &&
							 (StateInfo.uiFOPType != FLM_FOP_CONT_DATA))
						{
							if (StateInfo.uiFOPType != FLM_FOP_REC_INFO)
							{
								if (flmVerifyField( &StateInfo, StateInfo.pFOPData, 
									StateInfo.uiFOPDataLen,
									StateInfo.uiFieldType) != FLM_NO_CORRUPTION)
								{
									goto Error_Exit;
								}
							}
						}

						// See if we have reached the end of the element - or quit
						// if the element record offset is not changing.

						if (StateInfo.uiElmRecOffset >= StateInfo.uiElmRecLen)
						{
							break;
						}
					}
				}
			}
		}
		
		StateInfo.uiElmOffset += StateInfo.uiElmLen;
	}

	// Must end right on the end of the block

	if (StateInfo.uiElmOffset != StateInfo.uiEndOfBlock)
	{
		goto Error_Exit;
	}

Exit:

	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
	}
	
	return( rc);

Error_Exit:

	rc = RC_SET( FERR_DATA_ERROR);
	goto Exit;
}

/****************************************************************************
Desc:	Read a data block into cache.  This routine reads the requested
		version of a block into memory.  It follows links to previous
		versions of the block if necessary in order to do this.
****************************************************************************/
FSTATIC RCODE ScaReadBlock(
	FDB *					pDb,
	FLMUINT				uiBlkType,			// Type of block we are attempting
													// to read - used only for stats.
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiFilePos,			// File position where we are to
													// start reading from.
	FLMUINT				uiBlkAddress,		// Address of block that is to
													// be read into cache.
	FLMUINT				uiNewerBlkLowTransID,
													// Low transaction ID of the last newer
													// version of the block.
													// NOTE: This has no meaning
													// when uiFilePos == uiBlkAddress.
	FLMUINT				uiExpectedLowTransID,
													// Expected low trans ID for the
													// block we are going to read, if
													// we are starting our read from
													// a newer block.
													// NOTE: This value has no meaning
													// when uiFilePos == uiBlkAddress.
	SCACHE *				pSCache,				// Cache block to read the data
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
	RCODE					rc = FERR_OK;
	FFILE *				pFile = pDb->pFile;
	FLMBYTE *			pucBlk = pSCache->pucBlk;
	FLMUINT				uiBlkSize = pFile->FileHdr.uiBlockSize;
	SCACHE *				pNextSCache;
	FLMBOOL				bMutexLocked = FALSE;
	DB_STATS *			pDbStats = pDb->pDbStats;
	LFILE_STATS *		pLFileStats;
	BLOCKIO_STATS *	pBlockIOStats;
	FLMBOOL				bIncrPriorImageCnt = FALSE;
	FLMBOOL				bIncrOldViewCnt = FALSE;
	TMP_READ_STATS		TmpReadStats;
	TMP_READ_STATS *	pTmpReadStats;

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

	// Read in the block from the database.  Stay in a loop reading until 
	// we get an error or get the block

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

		if (RC_BAD( rc = ScaReadTheBlock( pDb, pLFile, pTmpReadStats, pucBlk, 
				uiFilePos, uiBlkAddress)))
		{
			goto Exit;
		}
		BH_UNSET_BI( pucBlk);

		// See if we can use the current version of the block, or if we
		// must go get a previous version.

		// See if we even got the block we thought we wanted.

		if (GET_BH_ADDR( pucBlk) != uiBlkAddress)
		{
			if (uiFilePos == uiBlkAddress)
			{
				rc = RC_SET( FERR_DATA_ERROR);
			}
			else
			{
				// Should only be possible when reading a root block,
				// because the root block address in the LFILE may be
				// a block that was just created by an update
				// transaction.

				flmAssert( pDb->uiKilledTime);
				rc = RC_SET( FERR_OLD_VIEW);
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

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;

Get_Next_Block:

		if ((pNextSCache = pSCache->pNextInVersionList) != NULL)
		{
			FLMUINT	uiTmpTransID1;
			FLMUINT	uiTmpTransID2;

			// If next block is still being read in, we must wait for
			// it to complete before looking at its transaction IDs.

			if (pNextSCache->ui16Flags & CA_READ_PENDING)
			{
				gv_FlmSysData.SCacheMgr.uiIoWaits++;
				if (RC_BAD( rc = f_notifyWait( gv_FlmSysData.hShareMutex, 
					pDb->hWaitSem, (void *)&pNextSCache, &pNextSCache->pNotifyList)))
				{
					goto Exit;
				}

				// The thread doing the notify "uses" the cache block
				// on behalf of this thread to prevent the cache block
				// from being flushed after it unlocks the mutex.
				// At this point, since we have locked the mutex,
				// we need to release the cache block - because we
				// will put a "use" on it below.

				ScaReleaseForThread( pNextSCache);

				// See if we still have the same next block.

				goto Get_Next_Block;
			}

			// Check for overlapping trans ID ranges.  NOTE: At this
			// point, if we have an overlap, we know we have the version
			// of the block we need (see comment above).  Hence, we will
			// either break out of the loop at this point or goto exit
			// and return an error.

			uiTmpTransID1 = (FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]);
			if (uiTmpTransID1 <= pNextSCache->uiHighTransID)
			{
				uiTmpTransID2 = scaGetLowTransID( pNextSCache);

				// If the low trans IDs on the two blocks are not equal
				// we have a corruption.

				if (uiTmpTransID1 != uiTmpTransID2)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					goto Exit;
				}

				// The blocks are the same, discard one of them.

				*pbDiscardRV = TRUE;

				// Set the high trans ID on the block we are NOT discarding

				if( flmGetDbTransType( pDb) == FLM_UPDATE_TRANS)
				{
					scaSetTransID( pNextSCache, 0xFFFFFFFF);
				}
				else
				{
					// To find the version of the block we want, we have been
					// reading through a chain of blocks, from newer versions to
					// progressively older versions.  If uiFilePos == uiBlkAddress,
					// we are positioned on the most current version of the block.
					// In this case, the high trans ID for the block should be
					// set to 0xFFFFFFFF.
					//
					// If uiFilePos != uiBlkAddress, we are positioned on an older
					// version of the block.  The variable uiNewerBlkLowTransID contains
					// the low transaction ID for a newer version of the block we read
					// just prior to reading this block.

					if (uiFilePos == uiBlkAddress)
					{
						scaSetTransID( pNextSCache, 0xFFFFFFFF);
					}
					else
					{
						scaSetTransID( pNextSCache, (uiNewerBlkLowTransID - 1));
					}
				}

				// When discard flag is TRUE, we need to go right to
				// exit, because we don't want to decrypt, do sanity
				// check, etc.  NOTE: mutex is still locked, and
				// we want it to remain locked - see code at Exit.

				goto Exit;
			}
		}

		// See if this version of the block is what we want

		if ((FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]) <= pDb->LogHdr.uiCurrTransID)
		{

			// Set the high trans ID on the block

			if( flmGetDbTransType( pDb) == FLM_UPDATE_TRANS)
			{
				scaSetTransID( pSCache, 0xFFFFFFFF);
			}
			else
			{
				// To find the version of the block we want, we have been
				// reading through a chain of blocks, from newer versions to
				// progressively older versions.  If uiFilePos == uiBlkAddress,
				// we are positioned on the most current version of the block.
				// In this case, the high trans ID for the block should be
				// set to 0xFFFFFFFF.
				// 
				// If uiFilePos != uiBlkAddress, we are positioned on an older
				// version of the block.  The variable uiNewerBlkLowTransID contains
				// the low transaction ID for a newer version of the block we read
				// just prior to reading this block.  The variable uiExpectedLowTransID
				// contains the newer block's expectation of what the block's
				// low transaction ID should be (from BH_PREV_TRANS_ID).  Normally, we
				// would set the block's high transaction ID to the newer block's
				// low transaction ID minus one.  However, if the block's low trans ID
				// does not match what was expected, we err on the side of safety
				// and set the block's high trans ID equal to its low trans ID.

				if (uiFilePos == uiBlkAddress)
				{
					scaSetTransID( pSCache, 0xFFFFFFFF);
				}
				else
				{
					if (scaGetLowTransID( pSCache) == uiExpectedLowTransID)
					{
						scaSetTransID( pSCache, (uiNewerBlkLowTransID - 1));
					}
					else
					{
						flmAssert( 0);	// Normally, this should not happen
						scaSetTransID( pSCache,
									scaGetLowTransID( pSCache));
					}
				}
			}
			
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			bMutexLocked = FALSE;
			break;
		}

		flmAssert( bMutexLocked);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// At this point, we know we are going to have to get a prior
		// version of the block.  In an update transaction, this is
		// indicative of a file corruption.

		if( flmGetDbTransType( pDb) != FLM_READ_TRANS)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		// At this point, we know we are in a read transaction.  Save the
		// block's low trans ID as well as the expected low trans ID for
		// the previous version of the block.

		uiExpectedLowTransID = (FLMUINT)FB2UD( &pucBlk [BH_PREV_TRANS_ID]);
		uiNewerBlkLowTransID = (FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]);

		// See if there is a prior version of the block and determine whether
		// it's expected trans ID is in the range we need.
		// NOTE: If the prior version address is zero or is the same as our
		// current file position, there is no previous version of the block.

		if ((FLMUINT)FB2UD( &pucBlk [BH_PREV_BLK_ADDR]) == uiFilePos)
		{
			// Should only be possible when reading a root block,
			// because the root block address in the LFILE may be
			// a block that was just created by an update
			// transaction.

			flmAssert( pDb->uiKilledTime);
			rc = RC_SET( FERR_OLD_VIEW);
			goto Exit;
		}

		uiFilePos = (FLMUINT)FB2UD( &pucBlk [BH_PREV_BLK_ADDR]);
		if (!uiFilePos)
		{
			// Should only be possible when reading a root block,
			// because the root block address in the LFILE may be
			// a block that was just created by an update
			// transaction.

			flmAssert( pDb->uiKilledTime);
			rc = RC_SET( FERR_OLD_VIEW);
			goto Exit;
		}
	}

	// Perform sanity check on entire block

	if (gv_FlmSysData.bCheckCache)
	{
		if (RC_BAD( rc = ScaBlkSanityCheck( pDb, pFile, pLFile,
			pucBlk, uiBlkAddress, TRUE, FLM_EXTENSIVE_CHECK)))
		{
			goto Exit;
		}
	}
	else 
	{
		// Perform a sanity check on the block header if there was no
		// checksum.

		if ((!pucBlk [BH_CHECKSUM_HIGH]) && (!pucBlk [BH_CHECKSUM_LOW]))
		{
			FLMUINT	uiEndOfBlock = (FLMUINT)FB2UW( &pucBlk [BH_ELM_END]);

			if ((FB2UD( &pucBlk [BH_NEXT_BLK]) == 0) ||
			 	 ((BH_GET_TYPE( pucBlk) != BHT_FREE) &&
			  	  ((uiEndOfBlock < BH_OVHD) || (uiEndOfBlock > uiBlkSize))))
			{
				rc = RC_SET( FERR_DATA_ERROR);
				goto Exit;
			}
		}
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
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// If we got an old view error, it has to be a corruption, unless we
	// were killed.

	if (rc == FERR_OLD_VIEW)
	{
		if (!pDb->uiKilledTime ||
				flmGetDbTransType( pDb) == FLM_UPDATE_TRANS)
		{
			rc = RC_SET( FERR_DATA_ERROR);
		}
	}

	// Increment cache fault statistics

	if (pDbStats)
	{
		if ((pLFileStats = fdbGetLFileStatPtr( pDb, pLFile)) == NULL)
		{
			pBlockIOStats = flmGetBlockIOStatPtr( pDbStats, NULL, pucBlk,
											uiBlkType);
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
											pLFileStats, pucBlk, uiBlkType);
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

			if ((rc == FERR_OLD_VIEW) || (bIncrOldViewCnt))
			{
				pBlockIOStats->uiOldViewErrors++;
			}
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Increment the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the global mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void _ScaDbgUseForThread(
	SCACHE *			pSCache,			// Cache whose use count is to be decremented.
	FLMUINT *		puiThreadId)	// Pointer to thread ID requesting use.  If NULL,
											// call f_threadId() to get it.
{
	SCACHE_USE *	pUse;
	FLMUINT			uiMyThreadId = (FLMUINT)((puiThreadId == NULL)
														 ? (FLMUINT)f_threadId()
														 : *puiThreadId);

	// If the use count is 0, make sure there are not entries
	// in the use list.

	if( !pSCache->uiUseCount && pSCache->pUseList != NULL)
	{
			ScaDebugMsg( "Non-empty use list", pSCache, NULL);
			return;
	}
	
	// First increment the overall use count for the block

	pSCache->uiUseCount++;
	if (pSCache->uiUseCount == 1)
	{
		gv_FlmSysData.SCacheMgr.uiBlocksUsed++;
		ScaVerifyChecksum( pSCache);
	}
	gv_FlmSysData.SCacheMgr.uiTotalUses++;

	// Now add the thread's usage record - or increment it if there
	// is already one there for the thread.

	// See if we already have this thread in the use list

	pUse = pSCache->pUseList;
	while ((pUse) && (pUse->uiThreadId != uiMyThreadId))
	{
		pUse = pUse->pNext;
	}

	if (!pUse)
	{
		if (RC_BAD( f_alloc( (FLMUINT)sizeof( SCACHE_USE), &pUse)))
		{
			ScaDebugMsg( "Could not allocate SCACHE_USE structure",
						pSCache, NULL);
			return;
		}

		f_memset( pUse, 0, sizeof( SCACHE_USE));
		pUse->uiThreadId = uiMyThreadId;
		pUse->pNext = pSCache->pUseList;
		pSCache->pUseList = pUse;
	}

	pUse->uiUseCount++;
	if (pUse->uiUseCount > 20)
	{
		ScaDebugMsg( "High use count for thread on cache block",
				pSCache, pUse);
	}
}
#endif

/****************************************************************************
Desc:	Increment the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the global mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void ScaDbgUseForThread(
	SCACHE *			pSCache,			// Cache whose use count is to be decremented.
	FLMUINT *		puiThreadId)	//	Pointer to thread ID requesting use.  If NULL,
											// call f_threadId() to get it.
{
	if ((pSCache->pUseList) ||
		 (gv_FlmSysData.SCacheMgr.bDebug && !pSCache->uiUseCount))
	{
		_ScaDbgUseForThread( pSCache, puiThreadId);
	}
	else
	{
		ScaNonDbgUseForThread( pSCache, puiThreadId);
	}
}
#endif

/****************************************************************************
Desc:	Decrement the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the global mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void _ScaDbgReleaseForThread(
	SCACHE *			pSCache)
{
	SCACHE_USE *	pUse;
	SCACHE_USE *	pPrevUse;
	FLMUINT			uiMyThreadId = (FLMUINT)f_threadId();

	// Find the thread's use

	pUse = pSCache->pUseList;
	pPrevUse = NULL;
	while ((pUse) && (pUse->uiThreadId != uiMyThreadId))
	{
		pPrevUse = pUse;
		pUse = pUse->pNext;
	}

	if (!pUse)
	{
		ScaDebugMsg( "Attempt to release cache that is not in use for thread",
						pSCache, NULL);
		return;
	}

	pSCache->uiUseCount--;
	gv_FlmSysData.SCacheMgr.uiTotalUses--;
	if (!pSCache->uiUseCount)
	{
		pSCache->uiChecksum = ScaComputeChecksum( pSCache);
		gv_FlmSysData.SCacheMgr.uiBlocksUsed--;
		flmAssert( pUse->uiUseCount == 1);
	}

	// Free the use record if its count goes to zero

	pUse->uiUseCount--;
	if (!pUse->uiUseCount)
	{
		if (!pPrevUse)
		{
			pSCache->pUseList = pUse->pNext;
		}
		else
		{
			pPrevUse->pNext = pUse->pNext;
		}

		f_free( &pUse);
	}

	// If the use count is 0, make sure there are not entries
	// in the use list.

	if( !pSCache->uiUseCount && pSCache->pUseList != NULL)
	{
		ScaDebugMsg( "Non-empty use list", pSCache, NULL);
		return;
	}
}
#endif

/****************************************************************************
Desc:	Decrement the use count on a cache block for a particular
		thread.  NOTE: This routine assumes that the global mutex
		is locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void ScaDbgReleaseForThread(
	SCACHE *			pSCache)
{
	if (!pSCache->uiUseCount)
	{
		ScaDebugMsg( "Attempt to release cache that is not in use",
						pSCache, NULL);
		return;
	}

	if (pSCache->pUseList)
	{
		_ScaDbgReleaseForThread( pSCache);
	}
	else
	{

		// If count is one, it will be decremented to zero.

		if (pSCache->uiUseCount == 1)
		{
			pSCache->uiChecksum = ScaComputeChecksum( pSCache);
		}

		// Must do the release afterwards so that we can protect the block
		// if need be.

		ScaNonDbgReleaseForThread( pSCache);
	}
}
#endif

/****************************************************************************
Desc:	Read a data block into cache.  This routine takes care of allocating
		a cache block and reading the block from disk into memory.  NOTE:
		This routine assumes that the global mutex is locked.  It may
		unlock the global mutex long enough to do the read, but the
		mutex will still be locked when it exits.
****************************************************************************/
FSTATIC RCODE ScaReadIntoCache(
	FDB *					pDb,
	FLMUINT				uiBlkType,			// Type of block we are attempting
													// to read - used only for stats.
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiBlkAddress,		// Address of block that is to
													// be read into cache.
	SCACHE *				pPrevInVerList,	// Previous block in version list to
													// link the block to.
	SCACHE *				pNextInVerList,	// Next block in version list to link
													// the block to.
	SCACHE **			ppSCacheRV,			// Returns allocated cache block.
	FLMBOOL *			pbGotFromDisk)		// Returns TRUE if block was read
													// from disk
{
	RCODE							rc = FERR_OK;
	SCACHE *						pSCache;
	SCACHE *						pTmpSCache;
	F_NOTIFY_LIST_ITEM *		pNotify;
	FLMUINT						uiFilePos;
	FLMUINT						uiNewerBlkLowTransID = 0;
	FLMUINT						uiExpectedLowTransID = 0;
	FLMBOOL						bFoundVer;
	FLMBOOL						bDiscard;
	FLMUINT						uiSavePrevLowTransID;
	FLMUINT						uiSavePrevHighTransID;

	*pbGotFromDisk = FALSE;

	// Lock the prev and next in place by incrementing their use
	// count.  We don't want ScaAllocCache to use them.

	if (pPrevInVerList)
	{
		uiSavePrevLowTransID = scaGetLowTransID( pPrevInVerList);
		uiSavePrevHighTransID = pPrevInVerList->uiHighTransID;
		ScaUseForThread( pPrevInVerList, NULL);
	}

	if (pNextInVerList)
	{
		ScaUseForThread( pNextInVerList, NULL);
	}

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	rc = ScaAllocCache( pDb, &pSCache);
	
	if (pPrevInVerList)
	{
		ScaReleaseForThread( pPrevInVerList);
	}

	if (pNextInVerList)
	{
		ScaReleaseForThread( pNextInVerList);
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	pSCache->uiBlkAddress = uiBlkAddress;

	// Set the "dummy" flag so that we won't incur the overhead of
	// linking the block into the replace list.  It would be removed
	// from the replace list almost immediately anyway, when the
	// "read pending" flag is set below.

	pSCache->ui16Flags |= CA_DUMMY_FLAG;

	// Link block into various lists

	if( pDb->uiFlags & FDB_DONT_POISON_CACHE)
	{
		if( !(pDb->uiFlags & FDB_BACKGROUND_INDEXING) || 
			(pLFile && pLFile->uiLfType != LF_INDEX))
		{
			ScaLinkToGlobalListAsLRU( pSCache);
		}
		else
		{
			ScaLinkToGlobalListAsMRU( pSCache);
		}
	}
	else
	{
		ScaLinkToGlobalListAsMRU( pSCache);
	}

	ScaLinkToFile( pSCache, pDb->pFile);
	
	if (!pPrevInVerList)
	{
		SCACHE **	ppSCacheBucket;

		ppSCacheBucket = ScaHash( pDb->pFile->FileHdr.uiSigBitsInBlkSize,
					uiBlkAddress);
		uiFilePos = uiBlkAddress;
		if (pNextInVerList)
		{
			ScaUnlinkFromHashBucket( pNextInVerList, ppSCacheBucket);
		}
		ScaLinkToHashBucket( pSCache, ppSCacheBucket);
	}
	else
	{
		uiFilePos = scaGetPriorImageAddress( pPrevInVerList);
		uiNewerBlkLowTransID = scaGetLowTransID( pPrevInVerList);
		uiExpectedLowTransID = scaGetPriorImageTransID( pPrevInVerList);
		pPrevInVerList->pNextInVersionList = pSCache;
		scaVerifyCache( pPrevInVerList, 2400);
	}

	if (pNextInVerList)
	{
		pNextInVerList->pPrevInVersionList = pSCache;
		scaVerifyCache( pNextInVerList, 2500);
	}

	pSCache->pPrevInVersionList = pPrevInVerList;
	pSCache->pNextInVersionList = pNextInVerList;
	scaVerifyCache( pSCache, 2600);

	// Set the read-pending flag for this block.  This will force other
	// threads that need to read this block to wait for the I/O to
	// complete.
	
	scaSetFlags( pSCache, CA_READ_PENDING);
	pSCache->ui16Flags &= ~CA_DUMMY_FLAG;
	gv_FlmSysData.SCacheMgr.uiPendingReads++;

	// See if we need to free any cache

	if (RC_BAD( rc = ScaReduceCache( pDb)))
	{
		scaClearFlags( pSCache, CA_READ_PENDING);
		gv_FlmSysData.SCacheMgr.uiPendingReads--;
		ScaReleaseForThread( pSCache);
		ScaUnlinkCache( pSCache, TRUE, rc);
		goto Exit;
	}

	// Unlock the mutex and attempt to read the block into memory

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	rc = ScaReadBlock( pDb, uiBlkType,
								pLFile, uiFilePos, uiBlkAddress,
								uiNewerBlkLowTransID, uiExpectedLowTransID,
								pSCache, &bFoundVer, &bDiscard);

	// NOTE: If the bDiscard flag is TRUE, the mutex will still be
	// locked.  If FALSE, we need to relock it.

	if (!bDiscard)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}

	// Get a pointer to the notify list BEFORE discarding the cache
	// block - if we are going to discard - because pSCache can
	// change if we discard.

	pNotify = pSCache->pNotifyList;
	pSCache->pNotifyList = NULL;

	// Unset the read pending flag and reset the use count to zero.
	// Both of these actions should be done before doing a discard,
	// if a discard is going to be done.

	scaClearFlags( pSCache, CA_READ_PENDING);
	gv_FlmSysData.SCacheMgr.uiPendingReads--;
	ScaReleaseForThread( pSCache);

	// If we had no errors, take care of some other things

	if (RC_OK( rc))
	{
		// The bDiscard flag tells us that we should discard the
		// block that we just read and use the next block in the
		// version list - because they are the same version.

		if (bDiscard)
		{

			// NOTE: We are guaranteed that pSCache->pNextInVersionList
			// is non-NULL at this point, because when we set the
			// bDiscard flag to TRUE, it was non-NULL, and we know that
			// the mutex was NOT unlocked in that case.

			pTmpSCache = pSCache->pNextInVersionList;
			ScaUnlinkCache( pSCache, TRUE, FERR_OK);
			pSCache = pTmpSCache;
		}
		else
		{
			*pbGotFromDisk = TRUE;
		}

		// If there is an older version of the block, and it's low trans ID
		// is equal to this newer version's previous trans ID, adjust the older
		// version's high trans ID to be one less than the newer version's
		// low trans ID, because the two versions are adjacent versions in
		// sequence of time.

		if ((pSCache->pNextInVersionList) &&
		 	(scaGetPriorImageTransID( pSCache) ==
		  	 scaGetLowTransID( pSCache->pNextInVersionList)))
		{
			scaSetTransID( pSCache->pNextInVersionList,
				(scaGetLowTransID( pSCache) - 1));
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
		ScaUnlinkCache( pSCache, TRUE, FERR_OK);
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
void ScaFreeModifiedBlocks(
	FDB *				pDb)
{
	FFILE *			pFile = pDb->pFile;
	SCACHE *			pSCache;
	SCACHE *			pNextSCache;
	FLMBOOL			bFirstPass = TRUE;
	FLMBOOL			bFreedAll;

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Unlink all log blocks and reset their flags so they
	// won't be marked as needing to be written to disk.

	ScaUnlinkTransLogBlocks( pFile);

Do_Free_Pass:

	pSCache = pFile->pSCacheList;
	flmAssert( !pFile->pPendingWriteList);
	bFreedAll = TRUE;
	while (pSCache)
	{

		// If the high transaction ID on the block is one less than this
		// transaction's ID, the block is the most current block.  Therefore,
		// its high transaction ID should be reset to 0xFFFFFFFF.

		if (pSCache->uiHighTransID == pFile->uiUpdateTransID - 1)
		{
			scaSetTransID( pSCache, 0xFFFFFFFF);

			// Need to link blocks that become the current version again
			// into the file log list if they are dirty.  ScaLinkToFileLogList
			// will check to see if the block has already been logged.  If it has,
			// it won't be linked into the list.
			// NOTE: If the blocks were in the "new" list originally, we don't take
			// the time to put them back into that list because they would have to
			// be inserted in order.  They will still get written out eventually, but
			// they won't be written out by the ScaReduceNewBlocks call.

			if( pSCache->ui16Flags & CA_DIRTY)
			{
				ScaLinkToFileLogList( pSCache);
			}
		}
		else if ((pSCache->uiHighTransID == 0xFFFFFFFF) &&
					(scaGetLowTransID( pSCache) >= pFile->uiUpdateTransID) &&
					(!(pSCache->ui16Flags & CA_READ_PENDING)))

		{
			pNextSCache = pSCache->pNextInFile;

			// Another thread might have a temporary "use" on this
			// block.  Unlock the mutex long enough to allow the
			// other thread(s) to get rid of their "uses".  Then start
			// from the top of the list again.

			if (pSCache->uiUseCount)
			{

				// Don't want to unlock the mutex during the first pass
				// because it opens the door to the prior version of one of
				// these modified blocks being removed from cache before we
				// have a chance to reset its uiHighTransID back to 0xFFFFFFFF.
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
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					f_sleep( 10);
					f_mutexLock( gv_FlmSysData.hShareMutex);
					pSCache = pFile->pSCacheList;
					continue;
				}
			}
			else
			{

				// Unset dirty flag so we don't get an assert in ScaUnlinkCache.

				if (pSCache->ui16Flags & CA_DIRTY)
				{
#ifdef FLM_DBG_LOG
					FLMUINT16	ui16OldFlags = pSCache->ui16Flags;
#endif
					scaUnsetDirtyFlag( pSCache, pFile);
#ifdef FLM_DBG_LOG
					scaLogFlgChange( pSCache, ui16OldFlags, 'G');
#endif
				}

				ScaUnlinkCache( pSCache, TRUE, FERR_OK);
				pSCache = pNextSCache;
				continue;
			}
		}

		pSCache = pSCache->pNextInFile;
	}

	if (!bFreedAll && bFirstPass)
	{
		bFirstPass = FALSE;
		goto Do_Free_Pass;
	}

	// Set the update trans ID back to zero

	pFile->uiUpdateTransID = 0;
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/***************************************************************************
Desc:	Swap two entries in cache table during sort.
*****************************************************************************/
FINLINE void scaSwap(
	SCACHE **	ppSCacheTbl,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	SCACHE *	pTmpSCache = ppSCacheTbl [uiPos1];
	ppSCacheTbl [uiPos1] = ppSCacheTbl [uiPos2];
	ppSCacheTbl [uiPos2] = pTmpSCache;
}

/***************************************************************************
Desc:	Sort an array of SCACHE pointers by their block address.
****************************************************************************/
FSTATIC void scaSort(
	SCACHE **		ppSCacheTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	SCACHE *			pCurSCache;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurSCache = ppSCacheTbl [uiMIDPos ];
	for (;;)
	{
		while (uiLBPos == uiMIDPos ||
					((iCompare = 
						scaCompare( ppSCacheTbl [uiLBPos], pCurSCache)) < 0))
		{
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while( uiUBPos == uiMIDPos ||
			(((iCompare = scaCompare( pCurSCache, ppSCacheTbl [uiUBPos])) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if( uiLBPos < uiUBPos)
		{

			// Exchange [uiLBPos] with [uiUBPos].

			scaSwap( ppSCacheTbl, uiLBPos, uiUBPos);
			uiLBPos++;
			uiUBPos--;
		}
		else
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		scaSwap( ppSCacheTbl, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{

		// Exchange [uUBPos] with [uiMIDPos]

		scaSwap( ppSCacheTbl, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos)
							? uiMIDPos - uiLowerBounds
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds)
							? uiUpperBounds - uiMIDPos
							: 0;

	if( uiLeftItems < uiRightItems )
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems )
		{
			scaSort( ppSCacheTbl, uiLowerBounds, uiMIDPos - 1);
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			scaSort( ppSCacheTbl, uiMIDPos + 1, uiUpperBounds);
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/****************************************************************************
Desc:	This routine writes all blocks in the sorted list, or releases them.
		It attempts to write as many as it can that are currently
		contiguous.
		NOTE: This routine assumes that the global mutex is NOT locked.
****************************************************************************/
FSTATIC RCODE scaWriteSortedBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT				uiMaxDirtyCache,
	FLMUINT *			puiDirtyCacheLeft,
	FLMBOOL *			pbForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMUINT				uiNumSortedBlocks,
	FLMBOOL *			pbWroteAll)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiStartBlkAddr = 0;
	FLMUINT				uiLastBlkAddr = 0;
	FLMUINT				uiContiguousBlocks = 0;
	FLMUINT				uiNumSortedBlocksProcessed;
	FLMUINT				uiBlockCount;
	FLMUINT				uiBlockSize = pFile->FileHdr.uiBlockSize;
	CP_INFO *			pCPInfo = pFile->pCPInfo;
	SCACHE *				ppContiguousBlocks[ FLM_MAX_IO_BUFFER_BLOCKS];
	FLMBOOL				bBlockDirty[ FLM_MAX_IO_BUFFER_BLOCKS];
	FLMUINT				uiOffset = 0;
	FLMUINT				uiTmpOffset;
	FLMUINT				uiLoop;
	FLMUINT				uiStartOffset;
	FLMUINT				uiCopyLen;
	FLMBOOL				bForceCheckpoint = *pbForceCheckpoint;
	SCACHE *				pSCache;
	IF_IOBuffer *		pIOBuffer = NULL;
	FLMBYTE *			pucBuffer;
	
	// Extend the database to its new size
	
	if( bForceCheckpoint && pSFileHdl->canDoDirectIO() && uiNumSortedBlocks > 1)
	{
		if( RC_BAD( rc = pSFileHdl->allocateBlocks(
			pFile->ppBlocksDone[ 0]->uiBlkAddress,
			pFile->ppBlocksDone[ uiNumSortedBlocks - 1]->uiBlkAddress)))
		{
			goto Exit;
		}
	}

	for (;;)
	{

		// Mutex must be locked to test dirty flags.

		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		// See how many we have that are contiguous

		uiContiguousBlocks = 0;
		uiNumSortedBlocksProcessed = 0;
		uiStartOffset = uiTmpOffset = uiOffset;
		while (uiTmpOffset < uiNumSortedBlocks)
		{
			pSCache = pFile->ppBlocksDone [uiTmpOffset];

			// See if this block is still eligible for writing out.
			// If so, mark it as write pending and add to list.

			flmAssert( pSCache->ui16Flags & CA_DIRTY);

			// Is it contiguous with last block or the first block?

			if (!uiContiguousBlocks ||
				 (FSGetFileNumber( uiLastBlkAddr) ==
				  FSGetFileNumber( pSCache->uiBlkAddress) &&
				  uiLastBlkAddr + uiBlockSize == pSCache->uiBlkAddress))
			{

				// Block is either first block or contiguous with
				// last block.

Add_Contiguous_Block:
				uiLastBlkAddr = pSCache->uiBlkAddress;

				// Set first block address if this is the first one.

				if (!uiContiguousBlocks)
				{
					uiStartBlkAddr = pSCache->uiBlkAddress;
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
					 FSGetFileNumber( pSCache->uiBlkAddress))
				{
					break;
				}

				// If 32K won't encompass both blocks, not worth it to try
				// and fill the gap.

				uiGap = FSGetFileOffset( pSCache->uiBlkAddress) -
							FSGetFileOffset( uiLastBlkAddr) - uiBlockSize;
							
				if (uiGap > 32 * 1024 - (uiBlockSize * 2))
				{
					break;
				}

				// If the gap would run us off the maximum blocks to
				// request, don't try to fill it.

				if (uiContiguousBlocks + uiGap / uiBlockSize + 1 >
						FLM_MAX_IO_BUFFER_BLOCKS)
				{
					break;
				}

				uiSaveContiguousBlocks = uiContiguousBlocks;
				uiBlkAddress = uiLastBlkAddr + uiBlockSize;
				while (uiBlkAddress != pSCache->uiBlkAddress)
				{
					SCACHE **	ppSCacheBucket;
					SCACHE *		pTmpSCache;

					ppSCacheBucket = ScaHash( pFile->FileHdr.uiSigBitsInBlkSize,
							uiBlkAddress);
					pTmpSCache = *ppSCacheBucket;
					
					while (pTmpSCache &&
							 (pTmpSCache->uiBlkAddress != uiBlkAddress ||
							  pTmpSCache->pFile != pFile))
					{
						pTmpSCache = pTmpSCache->pNextInHashBucket;
					}
					
					if (!pTmpSCache ||
						 (pTmpSCache->ui16Flags &
							(CA_READ_PENDING | CA_WRITE_PENDING | CA_WRITE_INHIBIT)) ||
						 pTmpSCache->uiHighTransID != 0xFFFFFFFF)
					{
						break;
					}
					ppContiguousBlocks [uiContiguousBlocks] = pTmpSCache;

					bBlockDirty [uiContiguousBlocks++] =
						(pTmpSCache->ui16Flags & CA_DIRTY)
						? TRUE
						: FALSE;

					ScaUseForThread( pTmpSCache, NULL);
					uiBlkAddress += uiBlockSize;
				}

				// If we couldn't fill in the entire gap, we are done.

				if (uiBlkAddress != pSCache->uiBlkAddress)
				{

					// Release the blocks we obtained in the above loop.

					while (uiContiguousBlocks > uiSaveContiguousBlocks)
					{
						uiContiguousBlocks--;
						ScaReleaseForThread(
							ppContiguousBlocks [uiContiguousBlocks]);
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
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			bMutexLocked = FALSE;
		}

		// Ask for a buffer of the size needed.

		flmAssert( !pIOBuffer);
		
		if( RC_BAD( rc = pFile->pBufferMgr->getBuffer(
						uiContiguousBlocks * uiBlockSize, &pIOBuffer)))
		{
			goto Exit;
		}
		
		f_assert( pIOBuffer->getRefCount() == 2);
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

		// Re-lock the mutex

		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		// Set write pending on all of the blocks before unlocking
		// the mutex.  Then unlock the mutex and come out and copy
		// the blocks.

		for (uiLoop = 0; uiLoop < uiBlockCount; uiLoop++)
		{
			pSCache = ppContiguousBlocks [uiLoop];
			if (bBlockDirty [uiLoop])
			{
				flmAssert( pSCache->ui16Flags & CA_DIRTY);
				flmAssert( !(pSCache->ui16Flags & CA_WRITE_INHIBIT));
				scaSetFlags( pSCache, CA_WRITE_PENDING);
				flmAssert( *puiDirtyCacheLeft >= uiBlockSize);
				(*puiDirtyCacheLeft) -= uiBlockSize;
				ScaUnlinkFromFile( pSCache);
				ScaLinkToFile( pSCache, pFile);
			}
			else
			{
				flmAssert( !(pSCache->ui16Flags & CA_DIRTY));
			}

			// Set callback data so we will release these and clear
			// the pending flag if we don't do the I/O.

			pIOBuffer->addCallbackData( pSCache);
		}

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Copy blocks into the buffer.

		pucBuffer = pIOBuffer->getBufferPtr();
		
		for( uiLoop = 0;
			  uiLoop < uiBlockCount;
			  uiLoop++, pucBuffer += uiBlockSize)
		{
			pSCache = ppContiguousBlocks[ uiLoop];

			// Copy data from block to the write buffer

			uiCopyLen = getEncryptSize( pSCache->pucBlk);
			flmAssert( uiCopyLen >= BH_OVHD && uiCopyLen <= uiBlockSize);
			f_memcpy( pucBuffer, pSCache->pucBlk, uiCopyLen);

			// If this is an encrypted block, see that it gets encrypted.
			
			if (BH_GET_TYPE( pSCache->pucBlk) != BHT_FREE && 
				pSCache->pucBlk[ BH_ENCRYPTED])
			{
				// Encrypt the block? Will check the IXD
				
				if (RC_BAD( rc = ScaEncryptBlock( pSCache->pFile,
						pucBuffer, uiCopyLen, uiBlockSize)))
				{
					goto Exit;
				}
			}


			// Calculate the block checksum.

			if (RC_BAD( BlkCheckSum( pucBuffer, CHECKSUM_SET,
								pSCache->uiBlkAddress, uiBlockSize)))
			{

				// If the block checksum routine failed to set the checksum,
				// it must have encountered a problem when sanity checking.

				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}
		}
		
		pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
		pSFileHdl->setExtendSize( pFile->uiFileExtendSize);

		rc = pSFileHdl->writeBlock( uiStartBlkAddr, 
					pIOBuffer->getBufferSize(), pIOBuffer);
		
		pIOBuffer->Release();
		pIOBuffer = NULL;
		
		if( RC_BAD( rc))
		{
			if( pDbStats)
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

			if (scaSeeIfForceCheckpoint( uiCurrTime, pFile, pCPInfo))
			{
				bForceCheckpoint = TRUE;
			}
			else
			{
				if (pFile->pWriteLockObj->getWaiterCount() &&
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
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		ScaReleaseForThread( pFile->ppBlocksDone[ uiOffset]);
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
				f_mutexLock( gv_FlmSysData.hShareMutex);
				bMutexLocked = TRUE;
			}

			ScaReleaseForThread( ppContiguousBlocks [uiContiguousBlocks]);
		}
	}

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// If we allocated a write buffer, but did not do a write with it,
	// still need to call notifyComplete so that our callback function
	// will be called and the SCACHE structures will be released.

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
FSTATIC RCODE ScaFlushDirtyBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT				uiMaxDirtyCache,
	FLMBOOL				bForceCheckpoint,
	FLMBOOL				bIsCPThread,
	FLMBOOL *			pbWroteAll)
{
	RCODE				rc = FERR_OK;
	RCODE				rc2;
	SCACHE *			pSCache;
	FLMBOOL			bMutexLocked = FALSE;
	FLMUINT			uiSortedBlocks = 0;
	FLMUINT			uiBlockCount = 0;
	FLMBOOL			bWasForcing;
	FLMBOOL			bWriteInhibited;
	FLMUINT			uiDirtyCacheLeft;
	FLMBOOL			bAllocatedAll = FALSE;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	flmAssert( !pFile->uiLogCacheCount);

	if (pFile->pCPInfo)
	{
		pFile->pCPInfo->bWritingDataBlocks = TRUE;
	}

	flmAssert( !pFile->pPendingWriteList);

	uiDirtyCacheLeft = pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;

	// If we are forcing a checkpoint, pre-allocate an array big enough
	// to hold all of the dirty blocks.  We do this so we won't end up
	// continually re-allocating the array in the loop below.

Force_Checkpoint:

	if (bForceCheckpoint)
	{
		pSCache = pFile->pSCacheList;
		uiBlockCount = 0;
		while (pSCache && (pSCache->ui16Flags & CA_DIRTY))
		{
			uiBlockCount++;
			pSCache = pSCache->pNextInFile;
		}

		bAllocatedAll = TRUE;
		if (uiBlockCount > pFile->uiBlocksDoneArraySize * 2)
		{
			if (RC_BAD( rc = ScaAllocBlocksArray( pFile,
										(uiBlockCount + 1) / 2, TRUE)))
			{
				if (rc == FERR_MEM)
				{
					bAllocatedAll = FALSE;
					rc = FERR_OK;
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
		// Mutex better be locked at this point - unless doing a checkpoint

		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		// Create a list of blocks to write out - MAX_BLOCKS_TO_SORT at most.

		pSCache = pFile->pSCacheList;
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
						pSCache = pFile->ppBlocksDone [uiSortedBlocks - 1];
						flmAssert( !pSCache->pNextInFile ||
									  !(pSCache->pNextInFile->ui16Flags & CA_DIRTY));
					}
#endif
					break;
				}
				flmAssert( pSCache && (pSCache->ui16Flags & CA_DIRTY));
			}
			else
			{
				if (!pSCache || !(pSCache->ui16Flags & CA_DIRTY) ||
					 uiSortedBlocks == MAX_BLOCKS_TO_SORT)
				{
					break;
				}
			}

			flmAssert( !(pSCache->ui16Flags & CA_WRITE_PENDING));
			uiPrevBlkAddress = scaGetPriorImageAddress( pSCache);

			bWriteInhibited = FALSE;
			if (pSCache->ui16Flags & CA_WRITE_INHIBIT)
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
					scaClearFlags( pSCache, CA_WRITE_INHIBIT);
				}
				else
				{
					bWriteInhibited = TRUE;
				}
			}

			// Skip blocks that are write inhibited or that have
			// not been properly logged yet.

			if (bWriteInhibited ||
				 ((!uiPrevBlkAddress || uiPrevBlkAddress == BT_END) &&
							pSCache->pNextInVersionList))
			{
				flmAssert( !bForceCheckpoint);
			}
			else
			{
				if (uiSortedBlocks == pFile->uiBlocksDoneArraySize * 2)
				{
					if (RC_BAD( rc = ScaAllocBlocksArray( pFile, 0, TRUE)))
					{
						goto Exit;
					}
				}

				// Keep list of blocks to process

				pFile->ppBlocksDone [uiSortedBlocks++] = pSCache;

				// Must use to keep from going away.

				ScaUseForThread( pSCache, NULL);
			}
			pSCache = pSCache->pNextInFile;
		}
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Sort the list of blocks by block address.

		if (uiSortedBlocks)
		{
			if (uiSortedBlocks > 1)
			{
				scaSort( pFile->ppBlocksDone, 0, uiSortedBlocks - 1);
			}
			bWasForcing = bForceCheckpoint;
			rc = scaWriteSortedBlocks( pDbStats, pSFileHdl,
									pFile, uiMaxDirtyCache, &uiDirtyCacheLeft,
									&bForceCheckpoint, bIsCPThread,
									uiSortedBlocks, pbWroteAll);
		}
		else
		{
			goto Exit;
		}

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;

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
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		uiSortedBlocks--;

		// Release any blocks we didn't process through.

		pSCache = pFile->ppBlocksDone [uiSortedBlocks];
		ScaReleaseForThread( pSCache);
	}

	// Need to finish up any async writes.

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	// Wait for writes to complete.

	if (RC_BAD( rc2 = pFile->pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	flmAssert( !pFile->pPendingWriteList);

	// Better not be any incomplete writes at this point.

	flmAssert( !pFile->pBufferMgr->isIOPending());

	// Don't keep around a large block array if we happened to
	// allocate one that is bigger than our normal size.  It may
	// be huge because we were forcing a checkpoint.

	if (pFile->uiBlocksDoneArraySize > MAX_BLOCKS_TO_SORT)
	{
		f_free( &pFile->ppBlocksDone);
		pFile->uiBlocksDoneArraySize = 0;
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
FSTATIC RCODE ScaReduceNewBlocks(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMUINT *			puiBlocksFlushed)
{
	RCODE				rc = FERR_OK;
	RCODE				rc2;
	SCACHE *			pSCache;
	FLMBOOL			bMutexLocked = FALSE;
	FLMUINT			uiSortedBlocks = 0;
	FLMUINT			uiDirtyCacheLeft;
	FLMUINT			uiBlocksFlushed = 0;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	flmAssert( !pFile->uiLogCacheCount);

	if( !pFile->uiNewCount)
	{
		goto Exit;
	}

	if (pFile->pCPInfo)
	{
		pFile->pCPInfo->bWritingDataBlocks = TRUE;
	}

	flmAssert( !pFile->pPendingWriteList);
	uiDirtyCacheLeft = pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;

	if( pFile->uiBlocksDoneArraySize < MAX_BLOCKS_TO_SORT)
	{
		if (RC_BAD( rc = ScaAllocBlocksArray( pFile, MAX_BLOCKS_TO_SORT, TRUE)))
		{
			// If the array size is non-zero, but we were unable to allocate
			// the size we wanted, we'll just be content to output as many
			// blocks as possible with the existing size of the array

			if( rc == FERR_MEM && pFile->uiBlocksDoneArraySize)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	// Create a list of blocks to write out

	pSCache = pFile->pFirstInNewList;
	uiSortedBlocks = 0;
	for (;;)
	{
		FLMUINT	uiPrevBlkAddress;

		if (!pSCache || uiSortedBlocks == pFile->uiBlocksDoneArraySize)
		{
			break;
		}

		flmAssert( !(pSCache->ui16Flags & CA_WRITE_PENDING));
		flmAssert( pSCache->ui16Flags & (CA_DIRTY | CA_IN_NEW_LIST));

		uiPrevBlkAddress = scaGetPriorImageAddress( pSCache);

		// Skip blocks that are write inhibited

		if( pSCache->ui16Flags & CA_WRITE_INHIBIT)
		{
			pSCache = pSCache->pNextInReplaceList;
			continue;
		}

		// Keep list of blocks to process

		pFile->ppBlocksDone [uiSortedBlocks++] = pSCache;

		// Must use to keep from going away.

		ScaUseForThread( pSCache, NULL);
		pSCache = pSCache->pNextInReplaceList;
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	if (uiSortedBlocks)
	{
		FLMBOOL	bForceCheckpoint = FALSE;
		FLMBOOL	bDummy;

		rc = scaWriteSortedBlocks( pDbStats, pSFileHdl,
								pFile, ~((FLMUINT)0), &uiDirtyCacheLeft,
								&bForceCheckpoint, FALSE, uiSortedBlocks, &bDummy);

		if( RC_OK( rc))
		{
			uiBlocksFlushed += uiSortedBlocks;
		}
	}
	else
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Set to zero so won't get released at exit.

	uiSortedBlocks = 0;

Exit:

	// Release any blocks that are still used.

	while (uiSortedBlocks)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		uiSortedBlocks--;

		// Release any blocks we didn't process through.

		pSCache = pFile->ppBlocksDone [uiSortedBlocks];

#ifdef FLM_DEBUG
		if( RC_OK( rc))
		{
			flmAssert( !(pSCache->ui16Flags & CA_IN_NEW_LIST));
		}
#endif

		ScaReleaseForThread( pSCache);
	}

	// Need to finish up any async writes.

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	// Wait for writes to complete.

	if (RC_BAD( rc2 = pFile->pBufferMgr->waitForAllPendingIO()))
	{
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	flmAssert( !pFile->pPendingWriteList);

	// Better not be any incomplete writes at this point.

	flmAssert( !pFile->pBufferMgr->isIOPending());

	// Don't keep around a large block array if we happened to
	// allocate one that is bigger than our normal size.  It may
	// be huge because we were forcing a checkpoint.

	if (pFile->uiBlocksDoneArraySize > MAX_BLOCKS_TO_SORT)
	{
		f_free( &pFile->ppBlocksDone);
		pFile->uiBlocksDoneArraySize = 0;
	}

	if( puiBlocksFlushed)
	{
		*puiBlocksFlushed = uiBlocksFlushed;
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine is called to determine if a cache block or cache record
		is still needed.  This call assumes that the global mutex on
		FlmSysData is locked.
****************************************************************************/
FLMBOOL flmNeededByReadTrans(
	FFILE *	pFile,
	FLMUINT	uiLowTransId,
	FLMUINT	uiHighTransId)
{
	FLMBOOL	bNeeded = FALSE;
	FDB *		pReadTrans;

	// Quick check - so we don't have to traverse all read transactions.

	if ((!pFile->pFirstReadTrans) ||
		 (uiHighTransId < pFile->pFirstReadTrans->LogHdr.uiCurrTransID) ||
		 (uiLowTransId > pFile->pLastReadTrans->LogHdr.uiCurrTransID))
	{
		goto Exit;
	}

	// Traverse all read transactions - this loop assumes that the
	// read transactions are in order of when they started - meaning
	// that the uiCurrTransID on each will be ascending order.  The
	// loop will quit early once it can detect that the block is
	// too old for all remaining transactions.

	pReadTrans = pFile->pFirstReadTrans;
	while (pReadTrans)
	{
		if ((pReadTrans->LogHdr.uiCurrTransID >= uiLowTransId) &&
			 (pReadTrans->LogHdr.uiCurrTransID <= uiHighTransId))
		{
			bNeeded = TRUE;
			goto Exit;
		}
		else if (pReadTrans->LogHdr.uiCurrTransID > uiHighTransId)
		{

			// All remaining transaction's transaction IDs will
			// also be greater than the block's high trans ID
			// Therefore, we can quit here.

			goto Exit;
		}
		pReadTrans = pReadTrans->pNextReadTrans;
	}

Exit:

	return( bNeeded);
}

/****************************************************************************
Desc:	This routine is called just after a transaction has successfully
		committed.  It will unset the flags on log blocks
		that would cause them to be written to disk.  If the block is no longer
		needed by a read transaction, it will also put the block in the
		LRU list so it will be selected for replacement first.
		This routine assumes that the global mutex is locked.
****************************************************************************/
void ScaReleaseLogBlocks(
	FFILE *	pFile)
{
	SCACHE *	pSCache;
	SCACHE *	pNextSCache;

	pSCache = pFile->pTransLogList;
	while (pSCache)
	{

#ifdef FLM_DBG_LOG
		FLMUINT16	ui16OldFlags = pSCache->ui16Flags;
#endif

		// A block in this list should never be dirty.

		flmAssert( !(pSCache->ui16Flags & CA_DIRTY));
		if ((pSCache->ui16Flags & CA_WRITE_TO_LOG) &&
			!(pSCache->ui16Flags & CA_LOG_FOR_CP))
		{
			flmAssert( pFile->uiLogCacheCount);
			pFile->uiLogCacheCount--;
		}
		scaClearFlags( pSCache, CA_WRITE_TO_LOG | CA_WAS_DIRTY);

#ifdef FLM_DBG_LOG
		scaLogFlgChange( pSCache, ui16OldFlags, 'I');
#endif
		pNextSCache = pSCache->pNextInHashBucket;

		// Perhaps we don't really need to set these pointers to NULL,
		// but it helps keep things clean.

		pSCache->pNextInHashBucket =
		pSCache->pPrevInHashBucket = (SCACHE *)NULL;

		// If the block is no longer needed by a read transaction,
		// and it does not need to be logged for the checkpoint,
		// move it to the free list.

		if ((!pSCache->uiUseCount) &&
			 (!ScaNeededByReadTrans( pFile, pSCache)) &&
			 (!(pSCache->ui16Flags & CA_LOG_FOR_CP)))
		{
			SCACHE *		pNewerVer = pSCache->pPrevInVersionList;

			if( !pSCache->pNextInVersionList && pNewerVer && 
				pNewerVer->uiHighTransID == 0xFFFFFFFF &&
				pNewerVer->ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				ScaUnlinkFromFileLogList( pNewerVer);
			}

			ScaUnlinkCache( pSCache, TRUE, FERR_OK);
		}

		pSCache = pNextSCache;
	}
	pFile->pTransLogList = NULL;
}

/****************************************************************************
Desc:	Retrieve a data block.  Shared cache is searched first.  If the block
		is not in shared cache, it will be retrieved from disk and put into
		cache.  The use count on the block will be incremented.
****************************************************************************/
RCODE ScaGetBlock(
	FDB *					pDb,
	LFILE *				pLFile,				// Pointer to logical file structure
													// We are retrieving the block for.
													// NULL if there is no logical file.
	FLMUINT				uiBlkType,			// Type of block we are attempting
													// to read - used only for stats.
	FLMUINT				uiBlkAddress,		// Address of requested block.
	FLMUINT *			puiNumLooksRV,		// Pointer to FLMUINT where number of
													// cache lookups is to be returned.
													// If pointer is non-NULL it indicates
													// that we only want to find the block
													// if it is in cache.  If it is NOT
													// in cache, do NOT read it in from
													// disk. -- This capability is needed
													// by the FlmDbReduceSize function.
	SCACHE **			ppSCacheRV)			// Returns pointer to cache block.
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiBlkVersion;
	FLMUINT				uiNumLooks;
	SCACHE **			ppSCacheBucket;
	SCACHE *				pSBlkVerCache;
	SCACHE *				pSMoreRecentVerCache;
	SCACHE *				pSCache;
	FFILE *				pFile = pDb->pFile;
	FLMBOOL				bGotFromDisk = FALSE;

	*ppSCacheRV = NULL;

	if( !uiBlkAddress)
	{
		rc = RC_SET_AND_ASSERT( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// We should NEVER be attempting to read a block address that is
	// beyond the current logical end of file.

	if( !FSAddrIsBelow( uiBlkAddress, pDb->LogHdr.uiLogicalEOF))
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	// Lock the mutex

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	pDb->uiInactiveTime = 0;

	// Search shared cache for the desired version of the block.
	// First, determine the hash bucket.

	ppSCacheBucket = ScaHash( pFile->FileHdr.uiSigBitsInBlkSize,
								uiBlkAddress);

	// Search down the linked list of SCACHE structures off of the bucket
	// looking for the correct cache block.
	
	pSCache = *ppSCacheBucket;
	uiNumLooks = 1;
	while ((pSCache) &&
			 ((pSCache->uiBlkAddress != uiBlkAddress) ||
			  (pSCache->pFile != pFile)))
	{
		if ((pSCache = pSCache->pNextInHashBucket) != NULL)
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
		gv_FlmSysData.SCacheMgr.Usage.uiCacheFaults++;
		gv_FlmSysData.SCacheMgr.Usage.uiCacheFaultLooks += uiNumLooks;
		if (RC_BAD( rc = ScaReadIntoCache( pDb, uiBlkType, pLFile,
			uiBlkAddress, NULL, NULL, &pSCache, &bGotFromDisk)))
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
		uiBlkVersion = pDb->LogHdr.uiCurrTransID;

		for( ;;)
		{

			// If the block is being read into memory, wait for the read
			// to complete so we can see what it is.

			if( pSCache && (pSCache->ui16Flags & CA_READ_PENDING))
			{
				gv_FlmSysData.SCacheMgr.uiIoWaits++;
				if( RC_BAD( rc = f_notifyWait( gv_FlmSysData.hShareMutex,
					pDb->hWaitSem, (void *)&pSCache, &pSCache->pNotifyList)))
				{
					goto Exit;
				}

				// The thread doing the notify "uses" the cache block
				// on behalf of this thread to prevent the cache block
				// from being flushed after it unlocks the mutex.
				// At this point, since we have locked the mutex,
				// we need to release the cache block.

				ScaReleaseForThread( pSCache);

				// Start over at the top of the list.

				pSBlkVerCache = pSCache;
				while (pSBlkVerCache->pPrevInVersionList)
				{
					pSBlkVerCache = pSBlkVerCache->pPrevInVersionList;
				}
				pSCache = pSBlkVerCache;
				pSMoreRecentVerCache = NULL;
				continue;
			}
			
			if( !pSCache || uiBlkVersion > pSCache->uiHighTransID)
			{
				if( puiNumLooksRV)
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
					 scaGetPriorImageAddress( pSMoreRecentVerCache) == 0)
				{
					// Should only be possible when reading a root block,
					// because the root block address in the LFILE may be
					// a block that was just created by an update
					// transaction.

					flmAssert( pDb->uiKilledTime);
					rc = RC_SET( FERR_OLD_VIEW);
					goto Exit;
				}

				gv_FlmSysData.SCacheMgr.Usage.uiCacheFaults++;
				gv_FlmSysData.SCacheMgr.Usage.uiCacheFaultLooks += uiNumLooks;

				if (pSMoreRecentVerCache)
				{
					if( RC_BAD( rc = ScaReadIntoCache( pDb, uiBlkType, pLFile,
							uiBlkAddress, pSMoreRecentVerCache,
							pSMoreRecentVerCache->pNextInVersionList,
							&pSCache, &bGotFromDisk)))
					{
						goto Exit;
					}
				}
				else
				{
					if( RC_BAD( rc = ScaReadIntoCache( pDb, uiBlkType, pLFile,
							uiBlkAddress, NULL, pSBlkVerCache, 
							&pSCache, &bGotFromDisk)))
					{
						goto Exit;
					}
				}

				// At this point, if the read was successful, we should
				// have the block we want.

				break;
			}
			else if (uiBlkVersion >= scaGetLowTransID( pSCache))
			{

				// This is the version of the block that we need.

				gv_FlmSysData.SCacheMgr.Usage.uiCacheHits++;
				gv_FlmSysData.SCacheMgr.Usage.uiCacheHitLooks += uiNumLooks;

				if (gv_FlmSysData.bCheckCache)
				{

					// Need to do a use so that the block will be unprotected.

					if( RC_BAD( rc = ScaBlkSanityCheck( pDb, pFile, pLFile,
														pSCache->pucBlk, uiBlkAddress,
														TRUE, FLM_EXTENSIVE_CHECK)))
					{
						goto Exit;
					}
				}

				break;
			}
			else
			{
				// If we are in an update transaction, the version of the
				// block we want should ALWAYS be at the top of the list.
				// If not, we have a serious problem!

				flmAssert( flmGetDbTransType( pDb) != FLM_UPDATE_TRANS);

				pSMoreRecentVerCache = pSCache;
				pSCache = pSCache->pNextInVersionList;

				if (pSCache)
				{
					uiNumLooks++;
				}
			}
		}
	}

	// Increment the use count on the block.

	ScaUseForThread( pSCache, NULL);

	// Block was found, make it the MRU block or bump it up in the MRU list, 
	// if it is not already at the top.

	if( pDb->uiFlags & FDB_DONT_POISON_CACHE)
	{
		if( !(pDb->uiFlags & FDB_BACKGROUND_INDEXING) || 
			(pLFile && pLFile->uiLfType != LF_INDEX))
		{
			if( !bGotFromDisk)
			{
				ScaStepUpInGlobalList( pSCache);
			}

			// If the block was read from disk and FDB_DONT_POISION_CACHE is
			// set, we don't need to do anything because the block is 
			// already linked at the LRU position.
		}
		else if (pSCache->pPrevInGlobalList)
		{
			ScaUnlinkFromGlobalList( pSCache);
			ScaLinkToGlobalListAsMRU( pSCache);
		}
	}
	else if (pSCache->pPrevInGlobalList)
	{
		ScaUnlinkFromGlobalList( pSCache);
		ScaLinkToGlobalListAsMRU( pSCache);
	}

	*ppSCacheRV = pSCache;

Exit:

#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		scaVerify( 300);
	}
#endif

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Create a data block.
****************************************************************************/
RCODE ScaCreateBlock(
	FDB *				pDb,
	LFILE *			pLFile,
	SCACHE **		ppSCacheRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBlkAddress;
	FLMBYTE *		pucBlkBuf;
	SCACHE *			pSCache;
	SCACHE *			pOldSCache;
	FFILE *			pFile = pDb->pFile;
	FLMBOOL			bMutexLocked = FALSE;
	FLMUINT			uiOldLogicalEOF;
	SCACHE **		ppSCacheBucket;
	FLMUINT			uiBlockSize = pFile->FileHdr.uiBlockSize;

	pDb->bHadUpdOper = TRUE;

	// First see if there is a free block in the avail list

	if( pDb->LogHdr.uiFirstAvailBlkAddr != BT_END)
	{
		rc = FSBlockUseNextAvail( pDb, pLFile, ppSCacheRV);
		goto Exit;
	}

	// Determine where the next block ought to be -- we don't want to
	// overwrite any log file segments that may be out there.  Version 2.x.

	uiBlkAddress = pDb->LogHdr.uiLogicalEOF;

	// Time for a new block file?
	
	if( FSGetFileOffset( uiBlkAddress) >= pFile->uiMaxFileSize)
	{
		FLMUINT	uiFileNumber = FSGetFileNumber( uiBlkAddress) + 1;

		if( uiFileNumber > 
				MAX_DATA_BLOCK_FILE_NUMBER( pFile->FileHdr.uiVersionNum))
		{
			rc = RC_SET( FERR_DB_FULL);
			goto Exit;
		}

		if( RC_BAD( rc = pDb->pSFileHdl->createFile( uiFileNumber)))
		{
			goto Exit;
		}
		
		uiBlkAddress = FSBlkAddress( uiFileNumber, 0);
	}

	// Allocate a cache block for this new block.  If we have older
	// versions of this block already in cache, we need to link the
	// new block above the older version.

	// Lock the mutex

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// Determine the hash bucket the new block should be put into

	ppSCacheBucket = ScaHash( pFile->FileHdr.uiSigBitsInBlkSize,
								uiBlkAddress);

	// Search down the linked list of SCACHE structures off of the bucket
	// looking for an older version of the block.  If there are older
	// versions, we should get rid of them.
	
	pOldSCache = *ppSCacheBucket;
	while ((pOldSCache) &&
			 ((pOldSCache->uiBlkAddress != uiBlkAddress) ||
			  (pOldSCache->pFile != pFile)))
	{
		pOldSCache = pOldSCache->pNextInHashBucket;
	}

	while( pOldSCache)
	{
		SCACHE *	pNextSCache = pOldSCache->pNextInVersionList;

		// Older versions of blocks should not be in use or needed
		// by anyone because the only we we would have an older
		// version of a block beyond the logical EOF is if
		// FlmDbReduceSize had been called.  But it forces a
		// checkpoint that requires any read transactions to be
		// non-active, or killed.

		flmAssert( !pOldSCache->ui16Flags);
		flmAssert( !pOldSCache->uiUseCount);
		flmAssert( pOldSCache->uiHighTransID == 0xFFFFFFFF ||
			!ScaNeededByReadTrans( pFile, pOldSCache));

		ScaUnlinkCache( pOldSCache, TRUE, FERR_OK);
		pOldSCache = pNextSCache;
	}

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	if (RC_BAD( rc = ScaAllocCache( pDb, &pSCache)))
	{
		goto Exit;
	}

	pSCache->uiBlkAddress = uiBlkAddress;
	scaSetTransID( pSCache, 0xFFFFFFFF);

	// Initialize the block data, dirty flag is set so that it will be
	// flushed as needed.

	pucBlkBuf = pSCache->pucBlk;
	f_memset( pucBlkBuf, 0, uiBlockSize);
	SET_BH_ADDR( pucBlkBuf, uiBlkAddress );
	UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
		&pucBlkBuf[ BH_TRANS_ID]);
	UW2FBA( BH_OVHD, &pucBlkBuf [BH_ELM_END]);

	// If this is an index block, check to see if it is encrypted.

	if (pLFile && pLFile->pIxd && pLFile->pIxd->uiEncId)
	{
		pucBlkBuf[ BH_ENCRYPTED] = 1;
	}

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( pFile->uiFFileId, pSCache->uiBlkAddress, 0,
						 scaGetLowTransID( pSCache),
						"CREATE");
#endif

	// Link block into the global list

	pSCache->ui16Flags |= CA_DUMMY_FLAG;
	ScaLinkToGlobalListAsMRU( pSCache);

	// Set the dirty flag

	scaSetDirtyFlag( pSCache, pFile);
	pSCache->ui16Flags &= ~CA_DUMMY_FLAG;

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	scaSetFlags( pSCache, CA_WRITE_INHIBIT);

#ifdef FLM_DBG_LOG
	scaLogFlgChange( pSCache, 0, 'J');
#endif

	// Now that the dirty flag and write inhibit flag
	// have been set, link the block to the file

	ScaLinkToFile( pSCache, pFile);
	ScaLinkToHashBucket( pSCache, ppSCacheBucket);

	uiOldLogicalEOF = pDb->LogHdr.uiLogicalEOF;
	pDb->LogHdr.uiLogicalEOF = uiBlkAddress + uiBlockSize;

	// See if we need to free any cache

	if (RC_BAD( rc = ScaReduceCache( pDb)))
	{
#ifdef FLM_DBG_LOG
		FLMUINT16	ui16OldFlags = pSCache->ui16Flags;
#endif
		scaUnsetDirtyFlag( pSCache, pFile);
		scaClearFlags( pSCache, CA_WRITE_INHIBIT);
#ifdef FLM_DBG_LOG
		scaLogFlgChange( pSCache, ui16OldFlags, 'K');
#endif
		ScaReleaseForThread( pSCache);
		ScaUnlinkCache( pSCache, TRUE, FERR_OK);
		pDb->LogHdr.uiLogicalEOF = uiOldLogicalEOF;
		goto Exit;
	}

	// Link the block into the "new" list

	ScaLinkToNewList( pSCache);

	// Return a pointer to the block

	*ppSCacheRV = pSCache;

Exit:

#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		scaVerify( 400);
	}
#endif

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (RC_BAD( rc))
	{
		*ppSCacheRV = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine holds a cache block by incrementing its use count.
****************************************************************************/
void ScaHoldCache(
	SCACHE *		pSCache)
{
	f_mutexLock( gv_FlmSysData.hShareMutex);
	ScaUseForThread( pSCache, NULL);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	This routine releases a cache block by decrementing its use count.
		If the use count goes to zero, the block will be moved to the MRU
		position in the global cache list.
****************************************************************************/
void ScaReleaseCache(
	SCACHE *		pSCache,
	FLMBOOL		bMutexAlreadyLocked)
{
	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}

	ScaReleaseForThread( pSCache);

	// Can turn off write inhibit when use count reaches zero, because
	// we are guaranteed at that point that nobody is going to still
	// update it.

	if (!pSCache->uiUseCount)
	{
		scaClearFlags( pSCache, CA_WRITE_INHIBIT);
	}

	if (!bMutexAlreadyLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	Test and set the dirty flag on a block if not already set.  This is
		called on the case where a block didn't need to be logged because
		it had already been logged, but it still needs to have its dirty
		bit set.
****************************************************************************/
FSTATIC void scaSetBlkDirty(
	FFILE *	pFile,
	SCACHE *	pSCache)
{
#ifdef FLM_DBG_LOG
	FLMUINT16	ui16OldFlags;
#endif

	// If the dirty flag is already set, we will NOT attempt to set it.

	f_mutexLock( gv_FlmSysData.hShareMutex);
#ifdef FLM_DBG_LOG
	ui16OldFlags = pSCache->ui16Flags;
#endif

	if (!(pSCache->ui16Flags & CA_DIRTY))
	{
		scaSetDirtyFlag( pSCache, pFile);
	}

	// Move the block into the dirty blocks.  Even if block was
	// already dirty, put at the end of the list of dirty blocks.
	// This will make it so that when scaReduceCache hits a dirty
	// block, it is likely to also be one that will be written
	// out by a call to ScaFlushDirtyBlocks.

	ScaUnlinkFromFile( pSCache);
	ScaLinkToFile( pSCache, pFile);

	// Move the block to the MRU slot in the global list

	if( pSCache->pPrevInGlobalList)
	{
		ScaUnlinkFromGlobalList( pSCache);
		ScaLinkToGlobalListAsMRU( pSCache);
	}

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	scaSetFlags( pSCache, CA_WRITE_INHIBIT);
#ifdef FLM_DBG_LOG
	scaLogFlgChange( pSCache, ui16OldFlags, 'O');
#endif
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	This routine logs a block before it is modified.  In the shared cache
		system, the block is cloned.  The old version of the block is marked
		so that it will be written to the rollback log before the new version
		of the block can be written to disk.
		NOTE: It is assumed that the caller has "used" the block that is passed
		in.  This means it will be unprotected, and we can access it.  This
		routine will release it once we have made a copy of the block.
****************************************************************************/
RCODE ScaLogPhysBlk(
	FDB *			pDb,
	SCACHE **	ppSCacheRV)	// This is a pointer to the pointer of the
									// cache block that is to be logged.
									// If the block has not been logged before
									// during the transaction, a new version
									// of the block will be created and a
									// pointer to that block will be returned.
									// Otherwise, the pointer is unchanged.
{
	RCODE 				rc = FERR_OK;
	FFILE *				pFile = pDb->pFile;
	SCACHE *				pSCache = *ppSCacheRV;
	FLMBYTE *			pucBlkBuf = pSCache->pucBlk;
	SCACHE *				pNewSCache;
	FLMBOOL				bMutexLocked = FALSE;
	SCACHE **			ppSCacheBucket;
	FLMUINT				uiDbVersion = pDb->pFile->FileHdr.uiVersionNum;
	FLMUINT				uiCopyLen;
#ifdef FLM_DBG_LOG
	FLMUINT16			ui16OldFlags;
#endif

	flmAssert( pSCache->pPrevInVersionList == NULL);

	// Increment the block change count -- this is not an accurate
	// indication of the number of blocks that have actually changed.  The
	// count is used by the cursor code to determine when to re-position in
	// the B-Tree.  The value is only used by cursors operating within
	// an update transaction.
	
	pDb->uiBlkChangeCnt++;

	// See if the block has already been logged since the last transaction.
	// If so, there is no need to log it again.

	if ((FLMUINT)FB2UD( &pucBlkBuf [BH_TRANS_ID]) == pDb->LogHdr.uiCurrTransID)
	{
		flmAssert( pDb->bHadUpdOper);
		scaSetBlkDirty( pFile, pSCache);
		goto Exit;
	}
	pDb->bHadUpdOper = TRUE;

	if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		// See if the transaction ID is greater than the last backup
		// transaction ID.  If so, we need to update our block change
		// count.

		if( (FLMUINT)FB2UD( &pucBlkBuf [BH_TRANS_ID]) <
			 (FLMUINT)FB2UD( &pFile->ucUncommittedLogHdr [LOG_LAST_BACKUP_TRANS_ID]))
		{
			flmIncrUint(
				&pFile->ucUncommittedLogHdr [LOG_BLK_CHG_SINCE_BACKUP], 1);
		}
	}

	// pDb->uiTransEOF contains what the EOF address was at
	// the beginning of the transaction.  There is no need to log the
	// block if it's address is beyond that point because it is a
	// NEW block.

	if (!FSAddrIsBelow( pSCache->uiBlkAddress, pDb->uiTransEOF))
	{
		UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
					&pucBlkBuf [BH_TRANS_ID]);
		scaSetBlkDirty( pFile, pSCache);
		goto Exit;
	}

	// Allocate a cache block for this new block.  If we have older
	// versions of this block already in cache, we need to link the
	// new block above the older version.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// See if we need to free any cache.

	if (RC_BAD( rc = ScaReduceCache( pDb)))
	{
		goto Exit;
	}

	// Allocate a cache block - either a new one or by replacing
	// an existing one.

	if (RC_BAD( rc = ScaAllocCache( pDb, &pNewSCache)))
	{
		goto Exit;
	}

#ifdef FLM_DEBUG

	// Make sure the caller isn't logging one that has already been
	// logged.

	if (gv_FlmSysData.SCacheMgr.bDebug)
	{
		SCACHE *	pTmpSCache = pFile->pTransLogList;

		while (pTmpSCache)
		{
			flmAssert( pTmpSCache != pSCache);
			pTmpSCache = pTmpSCache->pNextInHashBucket;
		}
	}
#endif

	pNewSCache->uiBlkAddress = pSCache->uiBlkAddress;

#ifdef FLM_DBG_LOG
	flmDbgLogWrite( pFile->uiFFileId, pNewSCache->uiBlkAddress, 0,
						 pDb->LogHdr.uiCurrTransID,
						"NEW-VER");
#endif

	// Link the block to the global list

	pNewSCache->ui16Flags |= CA_DUMMY_FLAG;
	ScaLinkToGlobalListAsMRU( pNewSCache);

	// Set flags so that appropriate flushing to log and DB will be done.

	scaSetDirtyFlag( pNewSCache, pFile);
	pNewSCache->ui16Flags &= ~CA_DUMMY_FLAG;

	// Set write inhibit bit so we will not unset the dirty bit
	// until the use count goes to zero.

	scaSetFlags( pNewSCache, CA_WRITE_INHIBIT);

	// Copy the old block's data into this one.  Only need to copy
	// the amount of data actually in the block as indicated by
	// BH_BLK_END in the block header.

	pucBlkBuf = pNewSCache->pucBlk;
	uiCopyLen = getEncryptSize( pSCache->pucBlk);
	f_memcpy( pucBlkBuf, pSCache->pucBlk, uiCopyLen);

	// Previous block address should be zero until we actually log the
	// prior version of the block.

	UD2FBA( 0, &pucBlkBuf [BH_PREV_BLK_ADDR]);

	// Set the low and high trans IDs on the newly created block.

	scaSetTransID( pNewSCache, 0xFFFFFFFF);
	UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
		&pucBlkBuf[ BH_TRANS_ID]);
#ifdef FLM_DBG_LOG
	scaLogFlgChange( pNewSCache, 0, 'L');
#endif

	// Set the previous trans ID on the newly created block.

	f_memcpy( &pucBlkBuf [BH_PREV_TRANS_ID], &pSCache->pucBlk [BH_TRANS_ID], 4);

	// Determine the hash bucket the new block should be put into.

	ppSCacheBucket = ScaHash( pFile->FileHdr.uiSigBitsInBlkSize,
								pSCache->uiBlkAddress);

	// Link new block into various lists.

	ScaUnlinkFromHashBucket( pSCache, ppSCacheBucket);
	pSCache->pPrevInVersionList = pNewSCache;
	scaVerifyCache( pSCache, 2900);
	pNewSCache->pNextInVersionList = pSCache;
	ScaLinkToFile( pNewSCache, pFile);
	ScaLinkToHashBucket( pNewSCache, ppSCacheBucket);
	scaVerifyCache( pNewSCache, 3000);

	// Set the high trans ID on the old block to be one less than
	// the current trans ID.  Also set the flag indicating that
	// the block needs to be written to the rollback log.

	scaSetTransID( pSCache, (pDb->LogHdr.uiCurrTransID - 1));
#ifdef FLM_DBG_LOG
	ui16OldFlags = pSCache->ui16Flags;
#endif

	if (!(pSCache->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP)))
	{
		pFile->uiLogCacheCount++;
	}

	scaSetFlags( pSCache, CA_WRITE_TO_LOG);

	if (scaGetLowTransID( pSCache) <=
		 (FLMUINT)FB2UD( &pFile->ucUncommittedLogHdr [LOG_LAST_CP_TRANS_ID]))
	{
		scaSetFlags( pSCache, CA_LOG_FOR_CP);
	}

	if (pSCache->ui16Flags & CA_DIRTY)
	{
		scaSetFlags( pSCache, CA_WAS_DIRTY);
		scaUnsetDirtyFlag( pSCache, pFile);

		// No more need to write inhibit - because the old version of the
		// block cannot possibly be changed.

		scaClearFlags( pSCache, CA_WRITE_INHIBIT);

		// Move the block out of the dirty blocks.

		ScaUnlinkFromFile( pSCache);
		ScaLinkToFile( pSCache, pFile);
	}

#ifdef FLM_DBG_LOG
	scaLogFlgChange( pSCache, ui16OldFlags, 'N');
#endif

	// Put the old block into the list of the transaction's
	// log blocks

	pSCache->pPrevInHashBucket = NULL;
	if ((pSCache->pNextInHashBucket = pFile->pTransLogList) != NULL)
	{
		pSCache->pNextInHashBucket->pPrevInHashBucket = pSCache;
	}
	pFile->pTransLogList = pSCache;

	// Link the new block to the file log list

	ScaLinkToFileLogList( pNewSCache);

	// If this is an indexing thread, the old version of the
	// block will probably not be needed again so put it at the LRU end
	// of the cache.  The assumption is that a background indexing thread
	// has also set the FDB_DONT_POISON_CACHE flag.

	if( pDb->uiFlags & FDB_BACKGROUND_INDEXING)
	{
		ScaUnlinkFromGlobalList( pSCache);
		ScaLinkToGlobalListAsLRU( pSCache);
	}

	// Release the old block and return a pointer to the new block.

	ScaReleaseCache( pSCache, bMutexLocked);
	*ppSCacheRV = pNewSCache;

Exit:
#ifdef SCACHE_LINK_CHECKING
	if (RC_BAD( rc))
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		scaVerify( 500);
	}
#endif

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine calculates the number of hash buckets to use based on
		the maximum amount of cache we want utilized.
****************************************************************************/
FSTATIC FLMUINT ScaNumHashBuckets(
	FLMUINT	uiMaxSharedCache)
{
	FLMUINT	uiNumBuckets;
	FLMUINT	uiCeiling;

	// Calculate a hash table size that will allow 10 2K blocks
	// per bucket.  Maximum hash table size is 65536 * 8.

	uiNumBuckets = uiMaxSharedCache / (10 * 2048);

	// If we would overflow 65536 * 2 buckets, calculate the
	// number of buckets based on a block size of 4096.  This
	// will have the effect of causing us to use fewer buckets
	// with more blocks in them for larger cache sizes.

	if (uiNumBuckets > 0x20000)
	{
		uiNumBuckets = uiMaxSharedCache / (10 * 4096);

		// Don't want the new calculation to bump us below
		// 65536 * 2.

		if (uiNumBuckets < 0x20000)
		{
			uiNumBuckets = 0x20000;
			goto Exit;
		}
	}

	// Round buckets to nearest exponent of 2 (2^n) - up to
	// 65536 * 8 maximum.

	uiCeiling = 1024;
	for (;;)
	{
		if ((uiNumBuckets <= uiCeiling) || (uiCeiling == 0x80000))
		{
			uiNumBuckets = uiCeiling;
			break;
		}
		uiCeiling <<= 1;
	}

Exit:

	return( uiNumBuckets);
}

/****************************************************************************
Desc:	This routine initializes the hash table for shared cache.
****************************************************************************/
FSTATIC RCODE ScaInitHashTbl(
	FLMUINT	uiNumBuckets)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiAllocSize;

	// Calculate the number of bits needed to represent values in the
	// hash table.

	gv_FlmSysData.SCacheMgr.uiHashTblSize = uiNumBuckets;
	gv_FlmSysData.SCacheMgr.uiHashTblBits = (uiNumBuckets - 1);
	uiAllocSize = (FLMUINT)sizeof( SCACHE *) * uiNumBuckets;

	if (RC_BAD( rc = f_alloc(
						uiAllocSize,
						&gv_FlmSysData.SCacheMgr.ppHashTbl)))
	{
		goto Exit;
	}
	
	f_memset( gv_FlmSysData.SCacheMgr.ppHashTbl, 0, uiAllocSize);
	
	gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +=
		f_msize( gv_FlmSysData.SCacheMgr.ppHashTbl);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine initializes shared cache manager.
****************************************************************************/
RCODE ScaInit(
	FLMUINT					uiMaxSharedCache)
{
	RCODE						rc = FERR_OK;
	FLMUINT					uiLoop;
	FLMUINT					uiBlockSize;
	F_SCacheRelocator *	pSCacheRelocator = NULL;
	F_BlockRelocator *	pBlockRelocator = NULL;

	f_memset( &gv_FlmSysData.SCacheMgr, 0, sizeof( SCACHE_MGR));
	gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes = uiMaxSharedCache;

	// Allocate memory for the hash table.

	if (RC_BAD( rc = ScaInitHashTbl( ScaNumHashBuckets( uiMaxSharedCache))))
	{
		goto Exit;
	}
	
	// Allocate the SCACHE re-locator object

	if( (pSCacheRelocator = f_new F_SCacheRelocator) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	// Initialize the SCACHE allocator
	
	if( RC_BAD( rc = FlmAllocFixedAllocator(
		&gv_FlmSysData.SCacheMgr.pSCacheAllocator)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_FlmSysData.SCacheMgr.pSCacheAllocator->setup( 
		FALSE, gv_FlmSysData.pSlabManager, pSCacheRelocator, sizeof( SCACHE),
		&gv_FlmSysData.SCacheMgr.Usage.SlabUsage,
		&gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated)))
	{
		goto Exit;
	}
	
	// Initialize the cache block allocators

	for( uiLoop = 0, uiBlockSize = 4096; 
		  uiLoop < 2; 
		  uiLoop++, uiBlockSize *= 2)
	{
		if( RC_BAD( rc = FlmAllocBlockAllocator(
			&gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop])))
		{
			goto Exit;
		}
		
		if( (pBlockRelocator = f_new F_BlockRelocator( uiBlockSize)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop]->setup( 
			FALSE, gv_FlmSysData.pSlabManager, pBlockRelocator, uiBlockSize,
			&gv_FlmSysData.SCacheMgr.Usage.SlabUsage,
			&gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated)))
		{
			goto Exit;
		}
		
		pBlockRelocator->Release();
		pBlockRelocator = NULL;
	}

Exit:

	if( pSCacheRelocator)
	{
		pSCacheRelocator->Release();
	}
	
	if( pBlockRelocator)
	{
		pBlockRelocator->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine performs various checks on an individual cache block
		to verify that it is linked into the proper lists, etc.  This routine
		assumes that the mutex has been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
FSTATIC void scaVerifyCache(
	SCACHE *			pSCache,
	int				iPlace)
{
	SCACHE *		pTmpSCache;
	FLMUINT		uiSigBitsInBlkSize;
	SCACHE **	ppBucket;

	uiSigBitsInBlkSize = flmGetSigBits( ScaGetBlkSize( pSCache));
	ppBucket = ScaHash( uiSigBitsInBlkSize, pSCache->uiBlkAddress);
	pTmpSCache = *ppBucket;
	while( (pTmpSCache) && (pTmpSCache != pSCache))
	{
		pTmpSCache = pTmpSCache->pNextInHashBucket;
	}

	if( pTmpSCache)
	{
		if (!pSCache->pFile)
		{
			f_breakpoint( iPlace+3);
		}
		if (pSCache->pPrevInVersionList)
		{
			f_breakpoint( iPlace+4);
		}

		// Verify that it is not in the log list.

		if (pSCache->ui16Flags & CA_WRITE_TO_LOG)
		{
			f_breakpoint( iPlace+5);
		}

		pTmpSCache = pSCache->pFile->pTransLogList;

		while (pTmpSCache && pTmpSCache != pSCache)
			pTmpSCache = pTmpSCache->pNextInHashBucket;
		if (pTmpSCache)
		{
			f_breakpoint( iPlace+6);
		}
	}
	else
	{
		if (pSCache->pFile && !pSCache->pPrevInVersionList)
		{
			f_breakpoint( iPlace+7);
		}

		// If the block is marked as needing to be logged, verify that
		// it is in the log list.

		if (pSCache->ui16Flags & CA_WRITE_TO_LOG)
		{
			pTmpSCache = pSCache->pFile->pTransLogList;
			while (pTmpSCache && pTmpSCache != pSCache)
			{
				pTmpSCache = pTmpSCache->pNextInHashBucket;
			}

			if (!pTmpSCache)
			{
				// Not in the log list

				f_breakpoint( iPlace+8);
			}

			// Better also have a newer version.

			if (!pSCache->pPrevInVersionList)
			{
				// Not linked to a prior version.

				f_breakpoint( iPlace+9);
			}
		}
	}

	// Verify that the prev and next pointers do not point to itself.

	if (pSCache->pPrevInVersionList == pSCache)
	{
		f_breakpoint( iPlace+10);
	}

	if (pSCache->pNextInVersionList == pSCache)
	{
		f_breakpoint( iPlace+11);
	}
}
#endif

/****************************************************************************
Desc:	This routine performs various checks on the cache to verify that
		things are linked into the proper lists, etc.  This routine assumes
		that the mutex has been locked.
****************************************************************************/
#ifdef SCACHE_LINK_CHECKING
FSTATIC void scaVerify(
	int			iPlace)
{
	FLMUINT		uiLoop;
	SCACHE **	ppBucket;
	SCACHE *		pTmpSCache;

	// Verify that everything in buckets has a pFile and does NOT
	// have a pPrevInVersionList

	for (uiLoop = 0, ppBucket = gv_FlmSysData.SCacheMgr.ppHashTbl;
		  uiLoop < gv_FlmSysData.SCacheMgr.uiHashTblSize;
		  uiLoop++, ppBucket++)
	{
		pTmpSCache = *ppBucket;
		while (pTmpSCache)
		{
			if (!pTmpSCache->pFile)
			{
				f_breakpoint(iPlace+1);
			}
			if (pTmpSCache->pPrevInVersionList)
			{
				f_breakpoint(iPlace+2);
			}
			pTmpSCache = pTmpSCache->pNextInHashBucket;
		}
	}

	// Traverse the entire list - make sure that everything
	// with a file is hashed and linked properly
	// and everything without a file is NOT hashed.

	pTmpSCache = gv_FlmSysData.SCacheMgr.pMRUCache;
	while (pTmpSCache)
	{
		scaVerifyCache( pTmpSCache, 1000 + iPlace);
		pTmpSCache = pTmpSCache->pNextInGlobalList;
	}
}
#endif

/****************************************************************************
Desc:	This routine configures the shared cache manager.  NOTE: This routine
		assumes that the global mutex has been locked if necessary.
****************************************************************************/
RCODE ScaConfig(
	FLMUINT			uiType,				// Type of item being configured.
	void *			Value1,				// Data used in conjunction with
												// uiType to do configuration.
	void *			Value2)				// Data used in conjunction with
												// uiType to do configuration.
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiMaxSharedCache;
	FLMUINT	uiSaveMax;
	FLMUINT	uiNumBuckets;

	F_UNREFERENCED_PARM( Value2);

	switch (uiType)
	{
		case FLM_CACHE_LIMIT:
			uiMaxSharedCache = (FLMUINT)Value1;

			// Change the cache limit and call
			// ScaReduceCache to shrink cache below it.

			uiSaveMax = gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes;
			gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes = uiMaxSharedCache;
			(void)ScaReduceCache( NULL);

			// Allocate and populate a new hash table if we changed
			// the size sufficiently to warrant a new hash table
			// size.

			if ((uiNumBuckets = ScaNumHashBuckets( uiMaxSharedCache)) !=
					gv_FlmSysData.SCacheMgr.uiHashTblSize)
			{
				SCACHE **			ppOldHashTbl =
											gv_FlmSysData.SCacheMgr.ppHashTbl;
				SCACHE **			ppBucket;
				FLMUINT				uiOldHashTblSize =
											gv_FlmSysData.SCacheMgr.uiHashTblSize;
				FLMUINT				uiOldHashTblBits =
											gv_FlmSysData.SCacheMgr.uiHashTblBits;
				FLMUINT				uiLoop;
				SCACHE **			ppSCacheBucket;
				SCACHE *				pTmpSCache;
				SCACHE *				pTmpNextSCache;

				scaVerify( 700);

				// Allocate a new hash table.

				gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated -=
											f_msize( ppOldHashTbl);

				if (RC_BAD( ScaInitHashTbl(
									ScaNumHashBuckets( uiMaxSharedCache))))
				{
					gv_FlmSysData.SCacheMgr.ppHashTbl = ppOldHashTbl;
					gv_FlmSysData.SCacheMgr.uiHashTblSize = uiOldHashTblSize;
					gv_FlmSysData.SCacheMgr.uiHashTblBits = uiOldHashTblBits;
					gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +=
						f_msize( ppOldHashTbl);
					gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes = uiSaveMax;
				}
				else
				{
					// Relink all of the cache blocks into the new
					// hash table.

					for (uiLoop = 0, ppBucket = ppOldHashTbl;
						  uiLoop < uiOldHashTblSize;
						  uiLoop++, ppBucket++)
					{
						pTmpSCache = *ppBucket;
						while (pTmpSCache)
						{
							FLMUINT	uiSigBitsInBlkSize;
							
							pTmpNextSCache = pTmpSCache->pNextInHashBucket;

							if (pTmpSCache->pFile)
							{
								uiSigBitsInBlkSize = 
									pTmpSCache->pFile->FileHdr.uiSigBitsInBlkSize;
							}
							else
							{
								// The cache block is not associated with a file
								// Should never happen.

								flmAssert( 0);

								uiSigBitsInBlkSize = 
									flmGetSigBits( ScaGetBlkSize( pTmpSCache));
							}

							ppSCacheBucket = ScaHash( uiSigBitsInBlkSize,
														pTmpSCache->uiBlkAddress);

							ScaLinkToHashBucket( pTmpSCache, ppSCacheBucket);
							pTmpSCache = pTmpNextSCache;
						}
					}

					// Throw away the old hash table.

					f_free( &ppOldHashTbl);
				}

				// Reverify after having made the changes.

				scaVerify( 800);
			}
			break;
		case FLM_SCACHE_DEBUG:
#ifdef FLM_DEBUG
			gv_FlmSysData.SCacheMgr.bDebug =
				(FLMBOOL)(Value1
							 ? (FLMBOOL)TRUE
							 : (FLMBOOL)FALSE);
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
Desc:	This routine shuts down the shared cache manager and frees all
		resources allocated by it.
****************************************************************************/
void ScaExit( void)
{
	FLMUINT		uiLoop;

	flmAssert( !gv_FlmSysData.SCacheMgr.pMRUCache);
	flmAssert( !gv_FlmSysData.SCacheMgr.pLRUCache);

	// All databases should be closed at this point.  This means
	// that all cache memory has been put into the free list.
	// By freeing this list, all of the cache blocks will be
	// released.

	scaReduceFreeCache( TRUE);
	
	// Free all of the cache structs and the allocator
	
	if( gv_FlmSysData.SCacheMgr.pSCacheAllocator)
	{
		gv_FlmSysData.SCacheMgr.pSCacheAllocator->freeAll();
		gv_FlmSysData.SCacheMgr.pSCacheAllocator->Release();
		gv_FlmSysData.SCacheMgr.pSCacheAllocator = NULL;
	}

	// Free all of the cache blocks and allocators

	for( uiLoop = 0; uiLoop < 2; uiLoop++)
	{
		if( gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop])
		{
			gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop]->freeAll();
			gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop]->Release();
			gv_FlmSysData.SCacheMgr.pBlockAllocators[ uiLoop] = NULL;
		}
	}

	// Free the hash table

	f_free( &gv_FlmSysData.SCacheMgr.ppHashTbl);

	// Zero the entire structure out just for good measure

	f_memset( &gv_FlmSysData.SCacheMgr, 0, sizeof( SCACHE_MGR));
}

/****************************************************************************
Desc:	This routine will reduce the amount of free cache blocks until
		the cache is below its limit
		NOTE: This routine assumes that the global mutex is already locked.
****************************************************************************/
FSTATIC void scaReduceFreeCache(
	FLMBOOL		bFreeAll)
{
	SCACHE *		pSCache = gv_FlmSysData.SCacheMgr.pLastFree;
	SCACHE *		pPrevSCache;

	while( pSCache && (bFreeAll || scaIsCacheOverLimit()))
	{
		pPrevSCache = pSCache->pPrevInFile;
		if( !pSCache->uiUseCount)
		{
			ScaUnlinkFromFreeList( pSCache);
			ScaFree( pSCache);
		}
		pSCache = pPrevSCache;
	}
}

/****************************************************************************
Desc:	This routine will reduce the number of blocks in the reuse list
		until cache is below its limit.
		NOTE: This routine assumes that the global mutex is already locked.
****************************************************************************/
FSTATIC void scaReduceReuseList( void)
{
	SCACHE *		pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUReplace;
	SCACHE *		pPrevSCache;

	while( pTmpSCache && scaIsCacheOverLimit())
	{
		// Need to save the pointer to the previous entry in the list because
		// we may end up unlinking pTmpSCache below, in which case we would
		// have lost the previous entry.

		pPrevSCache = pTmpSCache->pPrevInReplaceList;

		// See if the cache block can be freed.

		flmAssert( !pTmpSCache->ui16Flags);
		if( scaCanBeFreed( pTmpSCache))
		{
			// NOTE: This call will free the memory pointed to by
			// pTmpSCache.  Hence, pTmpSCache should NOT be used after
			// this point.

			ScaUnlinkCache( pTmpSCache, TRUE, FERR_OK);
		}

		pTmpSCache = pPrevSCache;
	}
}

/****************************************************************************
Desc:	This routine frees all of the cache associated with an FFILE
		structure.  NOTE: This routine assumes that the global mutex
		is already locked.
****************************************************************************/
void ScaFreeFileCache(
	FFILE *			pFile)
{
	SCACHE *	pSCache = pFile->pSCacheList;
	SCACHE *	pNextSCache;

	// First, unlink as many as can be unlinked.

	flmAssert( !pFile->pPendingWriteList);
	while (pSCache)
	{
		f_yieldCPU();
		pNextSCache = pSCache->pNextInFile;

		if (!pSCache->uiUseCount)
		{
			// Turn off all bits that would cause an assert - we don't
			// care at this point, because we are forcing the file to
			// be closed.

			if (pSCache->ui16Flags & (CA_WRITE_TO_LOG | CA_LOG_FOR_CP))
			{
				flmAssert( pFile->uiLogCacheCount);
				pFile->uiLogCacheCount--;
			}
			if (pSCache->pNextInVersionList &&
				 (pSCache->pNextInVersionList->ui16Flags &
					(CA_WRITE_TO_LOG | CA_LOG_FOR_CP)))
			{
				flmAssert( pFile->uiLogCacheCount);
				pFile->uiLogCacheCount--;
			}

#ifdef FLM_DEBUG
			scaClearFlags( pSCache,
				CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP | CA_WAS_DIRTY);

			if (pSCache->pNextInVersionList)
			{
				scaClearFlags( pSCache->pNextInVersionList,
					CA_DIRTY | CA_WRITE_TO_LOG | CA_LOG_FOR_CP | CA_WAS_DIRTY);
			}
#endif

			if( pSCache->ui16Flags & CA_IN_FILE_LOG_LIST)
			{
				ScaUnlinkFromFileLogList( pSCache);
			}
			else if( pSCache->ui16Flags & CA_IN_NEW_LIST)
			{
				ScaUnlinkFromNewList( pSCache);
			}

			ScaUnlinkCache( pSCache, TRUE, FERR_OK);
		}
		else
		{
			// Another thread must have a temporary use on this block
			// because it is traversing cache for some reason.  We
			// don't want to free this block until the use count
			// is zero, so just put it into the free list so that
			// when its use count goes to zero we will either
			// re-use or free it.

			ScaUnlinkCache( pSCache, FALSE, FERR_OK);
			ScaLinkToFreeList( pSCache, FLM_GET_TIMER());
		}
		pSCache = pNextSCache;
	}

	// Set the pFile's cache list pointer to NULL.  Even if we didn't free
	// all of the cache blocks right now, we at least unlinked them from
	// the file.

	pFile->pSCacheList = NULL;
}


/****************************************************************************
Desc:	This routine computes an in-memory checksum on a cache block.  This
		is to guard against corrupt cache blocks being written back to disk.
		NOTE: This routine assumes that the global mutex is already locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC FLMUINT ScaComputeChecksum(
	SCACHE *			pSCache)
{
	FLMUINT	uiChecksum = 0;

	if( gv_FlmSysData.SCacheMgr.bDebug)
	{
		FLMUINT		uiBlkSize = ScaGetBlkSize( pSCache);
		FLMBYTE *	pucBlk = pSCache->pucBlk;
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
Desc:	This routine verifies a cache block checksum.  If the computed checksum
		does not match the block's checksum, the application will signal a
		breakpoint.
		NOTE: This routine assumes that the global mutex is already locked.
****************************************************************************/
#ifdef FLM_DEBUG
FSTATIC void ScaVerifyChecksum(
	SCACHE *			pSCache)
{
	// Checksum will be zero if it has not yet been computed.
	// In that case, we do NOT want to verify it.

	if (pSCache->uiChecksum)
	{
		flmAssert( pSCache->uiChecksum ==
					  ScaComputeChecksum( pSCache));
	}
}
#endif

/****************************************************************************
Desc:	This routine finishes a checkpoint.  At this point we are guaranteed
		to have both the file lock and the write lock, and all dirty blocks
		have been written to the database.  This is the code that writes out
		the log header and truncates the rollback log, roll-forward log, and
		database files as needed.
****************************************************************************/
FSTATIC RCODE scaFinishCheckpoint(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bTruncateRollBackLog,
	FLMUINT				uiCPFileNum,
	FLMUINT				uiCPOffset,
	FLMUINT				uiCPStartTime,
	FLMUINT				uiTotalToWrite)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	FLMBYTE			ucSaveLogHdr [LOG_HEADER_SIZE];
	FLMUINT			uiNewCPFileNum;
	FLMUINT			uiCurrTransID;
	FLMUINT			uiSaveTransOffset;
	FLMUINT			uiSaveCPFileNum;
	FLMBOOL			bTruncateLog = FALSE;
	FLMBOOL			bTruncateRflFile = FALSE;
	FLMUINT			uiRflTruncateSize = 0;
	FLMUINT			uiLogEof;
	FLMUINT			uiHighLogFileNumber;
#ifdef FLM_DBG_LOG
	FLMBOOL			bResetRBL = FALSE;
#endif

	// Update the log header to indicate that we now
	// have a new checkpoint.

	f_memcpy( ucSaveLogHdr, pucCommittedLogHdr, LOG_HEADER_SIZE);

	// Save some of the values we are going to change.  These values
	// will be needed below.

	uiCurrTransID =
		(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_CURR_TRANS_ID]);

	uiSaveTransOffset =
		(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);

	uiSaveCPFileNum =
		(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]);

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// If we get to this point, there should be no dirty blocks for
	// the file.

	flmAssert( !pFile->uiDirtyCacheCount && !pFile->pPendingWriteList &&
				  (!pFile->pSCacheList ||
					  !(pFile->pSCacheList->ui16Flags & CA_DIRTY)));

	// Determine if we can reset the physical log.  The log can be reset if
	// there are no blocks in the log that are needed to preserve a read
	// consistent view for a read transaction.  By definition, this will
	// be the case if there are no read transactions that started before
	// the last transaction committed.  Thus, that is what we check.

	// If we exceed a VERY large size, we need to wait for the read
	// transactions to empty out so we can force a truncation of the
	// log.  This is also true if we are truncating the database and
	// changing the logical EOF.

	uiLogEof = (FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_ROLLBACK_EOF]);
	uiHighLogFileNumber = FSGetFileNumber( uiLogEof);

	if( uiHighLogFileNumber > 0 || bTruncateRollBackLog ||
		 FSGetFileOffset( uiLogEof) > LOW_VERY_LARGE_LOG_THRESHOLD_SIZE)
	{
		CP_INFO *		pCPInfo = pFile->pCPInfo;
		FLMINT			iWaitCnt = 0;
		FDB *				pFirstDb;
		FLMUINT			ui5MinutesTime;
		FLMUINT			ui30SecTime;
		FLMUINT			uiFirstDbInactiveSecs;
		FLMUINT			uiElapTime;
		FLMUINT			uiLastMsgTime = FLM_GET_TIMER();
		FLMBOOL			bMustTruncate = (bTruncateRollBackLog ||
											  uiHighLogFileNumber ||
											  FSGetFileOffset( uiLogEof) >=
												HIGH_VERY_LARGE_LOG_THRESHOLD_SIZE)
											 ? TRUE
											 : FALSE;

		if( pCPInfo && bMustTruncate)
		{
			pCPInfo->uiStartWaitTruncateTime = FLM_GET_TIMER();
		}

		ui5MinutesTime = FLM_SECS_TO_TIMER_UNITS( 300);
		ui30SecTime = FLM_SECS_TO_TIMER_UNITS( 30);

		pFirstDb = pFile->pFirstReadTrans;
		while ((!pCPInfo || !pCPInfo->bShuttingDown) && pFirstDb &&
				 pFirstDb->LogHdr.uiCurrTransID < uiCurrTransID)
		{
			FLMUINT	uiTime;
			FLMUINT	uiFirstDbInactiveTime = 0;
			FLMUINT	uiFirstDbCurrTransID = pFirstDb->LogHdr.uiCurrTransID;
			FLMUINT	uiFirstDbThreadId = pFirstDb->uiThreadId;
			FDB *		pTmpDb;

			uiTime = (FLMUINT)FLM_GET_TIMER();

			// Lock the RCache mutex and get / set each FDB's
			// inactive time.  We must do this inside of the
			// RCache mutex to prevent the "kill" criteria below
			// from being triggered by a transition of the timeout
			// value from non-zero to zero due to activity in
			// the record cache (where the shared mutex may
			// not be locked).

			if( !bMustTruncate)
			{
				f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
				pTmpDb = pFirstDb;
				while( pTmpDb && pTmpDb->LogHdr.uiCurrTransID < uiCurrTransID)
				{
					if (!pTmpDb->uiInactiveTime)
					{
						pTmpDb->uiInactiveTime = uiTime;
					}
					pTmpDb = pTmpDb->pNextReadTrans;
				}
				uiFirstDbInactiveTime = pFirstDb->uiInactiveTime;
				f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			}

			// If the read transaction has been inactive for 5 minutes,
			// forcibly kill it unless it has been marked as a "don't kill"
			// transaction.

			if( !(pFirstDb->uiFlags & FDB_DONT_KILL_TRANS) &&
				(bMustTruncate || (uiFirstDbInactiveTime && 
				FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime) >= ui5MinutesTime)))
			{
				pFirstDb->uiKilledTime = uiTime;
				if ((pFile->pFirstReadTrans = pFirstDb->pNextReadTrans) != NULL)
				{
					pFile->pFirstReadTrans->pPrevReadTrans = NULL;
				}
				else
				{
					pFile->pLastReadTrans = NULL;
				}

				pFirstDb->pPrevReadTrans = NULL;

				if( (pFirstDb->pNextReadTrans = pFile->pFirstKilledTrans) != NULL)
				{
					pFirstDb->pNextReadTrans->pPrevReadTrans = pFirstDb;
				}
				pFile->pFirstKilledTrans = pFirstDb;

				f_mutexUnlock( gv_FlmSysData.hShareMutex);

				// Log a message indicating that we have killed the transaction

				uiElapTime = FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime);
				uiFirstDbInactiveSecs = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

				flmLogMessage( F_DEBUG_MESSAGE, FLM_YELLOW, FLM_BLACK,
					"Killed transaction 0x%08X."
					"  Thread: 0x%08X."
					"  Inactive time: %u seconds.",
					(unsigned)uiFirstDbCurrTransID,
					(unsigned)uiFirstDbThreadId,
					(unsigned)uiFirstDbInactiveSecs);

				f_mutexLock( gv_FlmSysData.hShareMutex);
				pFirstDb = pFile->pFirstReadTrans;
				continue;
			}
			else if( !bMustTruncate)
			{
				if( iWaitCnt >= 200)
				{
					break;
				}
			}

			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			if( !bMustTruncate)
			{
				iWaitCnt++;
				f_sleep( 6);
			}
			else
			{
				// Log a message indicating that we are waiting for the
				// transaction to complete

				if( FLM_ELAPSED_TIME( uiTime, uiLastMsgTime) >= ui30SecTime)
				{
					uiElapTime = FLM_ELAPSED_TIME( uiTime, uiFirstDbInactiveTime);
					uiFirstDbInactiveSecs = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

					flmLogMessage( F_DEBUG_MESSAGE, FLM_YELLOW, FLM_BLACK,
						"Waiting for transaction 0x%08X to complete."
						"  Thread: 0x%08X."
						"  Inactive time: %u seconds.",
						(unsigned)uiFirstDbCurrTransID,
						(unsigned)uiFirstDbThreadId,
						(unsigned)uiFirstDbInactiveSecs);

					uiLastMsgTime = FLM_GET_TIMER();
				}

				f_sleep( 100);
			}

			f_mutexLock( gv_FlmSysData.hShareMutex);
			pFirstDb = pFile->pFirstReadTrans;
		}

		if( bMustTruncate && pCPInfo)
		{
			pCPInfo->uiStartWaitTruncateTime = 0;
		}
	}

	if( !pFile->pFirstReadTrans ||
		 pFile->pFirstReadTrans->LogHdr.uiCurrTransID >= uiCurrTransID)
	{
		// We may want to truncate the log file if it has grown really big.
	
		if( uiHighLogFileNumber > 0 ||
			 FSGetFileOffset( uiLogEof) > pFile->uiRblFootprintSize)
		{
			bTruncateLog = TRUE;
		}
		
		UD2FBA( (FLMUINT32)pFile->FileHdr.uiBlockSize,
						&pucCommittedLogHdr[ LOG_ROLLBACK_EOF]);
#ifdef FLM_DBG_LOG
		bResetRBL = TRUE;
#endif
	}
	
	UD2FBA( 0, &pucCommittedLogHdr[ LOG_PL_FIRST_CP_BLOCK_ADDR]);

	// Set the checkpoint RFL file number and offset to be the same as
	// the last transaction's RFL file number and offset if nothing
	// is passed in.  If a non-zero uiCPFileNum is passed in, it is because
	// we are checkpointing the last transaction that has been recovered
	// by the recovery process.
	//
	// In this case, instead of moving the pointers all the way forward,
	// to the last committed transaction, we simply move them forward to
	// the last recovered transaction.

	if (uiCPFileNum)
	{
		UD2FBA( (FLMUINT32)uiCPFileNum,
			&pucCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]);
		UD2FBA( (FLMUINT32)uiCPOffset,
			&pucCommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]);
	}
	else
	{
		FLMBOOL	bResetRflFile = FALSE;

		// If the RFL volume is full, and the LOG_AUTO_TURN_OFF_KEEP_RFL
		// flag is TRUE, change the LOG_KEEP_RFL_FILES to FALSE.

		if (pFile->pRfl->isRflVolumeFull() &&
			 pucCommittedLogHdr [LOG_KEEP_RFL_FILES] &&
			 pucCommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL])
		{
			pucCommittedLogHdr [LOG_KEEP_RFL_FILES] = 0;
			bResetRflFile = TRUE;
		}

		f_memcpy( &pucCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM],
					 &pucCommittedLogHdr [LOG_RFL_FILE_NUM], 4);

		if( !pucCommittedLogHdr [LOG_KEEP_RFL_FILES])
		{
			UD2FBA( 512, &pucCommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]);
			if( bResetRflFile)
			{
				// This will cause the RFL file to be recreated on the
				// next transaction - causing the keep signature to be
				// changed.  Also need to set up to use new serial
				// numbers so restore can't wade into this RFL file and
				// attempt to start restoring from it.

				UD2FBA( 0, &pucCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);
				f_createSerialNumber(
						&pucCommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM]);
				f_createSerialNumber(
						&pucCommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM]);
			}
			else if (FB2UD( &pucCommittedLogHdr[ LOG_RFL_LAST_TRANS_OFFSET]) != 0)
			{
				// If LOG_RFL_LAST_TRANS_OFFSET is zero, someone has set this up
				// intentionally to cause the RFL file to be created at the
				// beginning of the next transaction.  We don't want to lose
				// that, so if it is zero, we don't change it.

				UD2FBA( 512, &pucCommittedLogHdr[ LOG_RFL_LAST_TRANS_OFFSET]);
			}
			
			if( uiSaveTransOffset >= pFile->uiRflFootprintSize)
			{
				bTruncateRflFile = TRUE;
				uiRflTruncateSize = pFile->uiRflFootprintSize;
			}
		}
		else
		{
			FLMUINT	uiRflFileNum;
			FLMUINT	uiLastTransOffset =
							FB2UD( &pucCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);

			// If the RFL volume is not OK, and we are not currently positioned
			// at the beginning of an RFL file, we should set things up to roll to
			// the next RFL file.  That way, if they need to change RFL volumes
			// it will be OK, and we can create the new RFL file.

			if (!pFile->pRfl->seeIfRflVolumeOk() && uiLastTransOffset > 512)
			{
				uiRflFileNum = FB2UD( &pucCommittedLogHdr [LOG_RFL_FILE_NUM]) + 1;
				
				UD2FBA( 0, &pucCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET]);
				UD2FBA( uiRflFileNum, &pucCommittedLogHdr [LOG_RFL_FILE_NUM]);
				UD2FBA( 512, &pucCommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]);
				UD2FBA( uiRflFileNum, &pucCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]);
			}
			else
			{
				// If the transaction offset is zero, we want the last CP offset
				// to be 512 - it should never be set to zero.  It is possible
				// for the transaction offset to still be zero at this point if
				// we haven't done a non-empty transaction yet.

				if( !uiLastTransOffset)
				{
					uiLastTransOffset = 512;
				}

				UD2FBA( uiLastTransOffset,
					&pucCommittedLogHdr [LOG_RFL_LAST_CP_OFFSET]);
			}
		}
	}

	// Set the checkpoint Trans ID to be the trans ID of the
	// last committed transaction.

	f_memcpy( &pucCommittedLogHdr [LOG_LAST_CP_TRANS_ID],
				 &pucCommittedLogHdr [LOG_CURR_TRANS_ID], 4);

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	// Write the log header - this will complete the checkpoint.

	if( RC_BAD( rc = flmWriteLogHdr( pDbStats, pSFileHdl, pFile,
									pucCommittedLogHdr,
									pFile->ucCheckpointLogHdr, TRUE)))
	{

		// Restore log header.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_memcpy( pucCommittedLogHdr, ucSaveLogHdr, LOG_HEADER_SIZE);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		goto Exit;
	}
	else if( bTruncateLog)
	{
		FLMUINT		uiRblFootprintSize = pFile->uiRblFootprintSize;
		
		if( uiHighLogFileNumber)
		{
			(void)pSFileHdl->truncateFiles( 
				FIRST_LOG_BLOCK_FILE_NUMBER( pFile->FileHdr.uiVersionNum), 
				uiHighLogFileNumber);
		}
		
		if( uiRblFootprintSize < pFile->FileHdr.uiBlockSize)
		{
			flmAssert( 0);
			uiRblFootprintSize = pFile->FileHdr.uiBlockSize;
		}
		
		(void)pSFileHdl->truncateFile( 0, uiRblFootprintSize);
	}

#ifdef FLM_DBG_LOG
	if (bResetRBL)
	{
		char	szMsg [80];

		if (bTruncateLog)
		{
			f_sprintf( szMsg, "f%u, Reset&TruncRBL, CPTID:%u",
				(unsigned)pFile->uiFFileId,
				(unsigned)FB2UD( &pucCommittedLogHdr [LOG_LAST_CP_TRANS_ID]));
		}
		else
		{
			f_sprintf( szMsg, "f%u, ResetRBL, CPTID:%u",
				(unsigned)pFile->uiFFileId,
				(unsigned)FB2UD( &pucCommittedLogHdr [LOG_LAST_CP_TRANS_ID]));
		}
		
		flmDbgLogMsg( szMsg);
	}
#endif

	// The checkpoint is now complete.  Reset the first checkpoint
	// block address to zero.

	pFile->uiFirstLogCPBlkAddress = 0;
	pFile->uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();

	// Save the state of the log header into the ucCheckpointLogHdr buffer.

	f_memcpy( pFile->ucCheckpointLogHdr, pucCommittedLogHdr, LOG_HEADER_SIZE);

	// See if we need to delete RFL files that are no longer in use.

	uiNewCPFileNum =
		(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_RFL_LAST_CP_FILE_NUM]);

	if( !pucCommittedLogHdr [LOG_KEEP_RFL_FILES] &&
		 uiSaveCPFileNum != uiNewCPFileNum && uiNewCPFileNum > 1)
	{
		FLMUINT	uiLastRflFileDeleted =
						(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_LAST_RFL_FILE_DELETED]);

		uiLastRflFileDeleted++;
		
		while( uiLastRflFileDeleted < uiNewCPFileNum)
		{
			char		szLogFilePath [F_PATH_MAX_SIZE];
			RCODE		TempRc;

			if( RC_BAD( pFile->pRfl->getFullRflFileName( 
				uiLastRflFileDeleted, szLogFilePath)))
			{
				break;
			}

			if( RC_BAD( TempRc = gv_FlmSysData.pFileSystem->deleteFile( 
				szLogFilePath)))
			{
				if( TempRc != FERR_IO_PATH_NOT_FOUND &&
					 TempRc != FERR_IO_INVALID_PATH)
				{
					break;
				}
			}
			uiLastRflFileDeleted++;
		}
		uiLastRflFileDeleted--;

		// If we actually deleted a file, update the log header.

		if( uiLastRflFileDeleted !=
				(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_LAST_RFL_FILE_DELETED]))
		{
			UD2FBA( (FLMUINT32)uiLastRflFileDeleted, 
				&pucCommittedLogHdr [LOG_LAST_RFL_FILE_DELETED]);

			if( RC_BAD( rc = flmWriteLogHdr( pDbStats, pSFileHdl, pFile,
				pucCommittedLogHdr, pFile->ucCheckpointLogHdr, TRUE)))
			{
				goto Exit;
			}

			// Save the state of the log header into the ucCheckpointLogHdr buffer
			// and update the last checkpoint time again.

			f_memcpy( pFile->ucCheckpointLogHdr, pucCommittedLogHdr, LOG_HEADER_SIZE);
			pFile->uiLastCheckpointTime = (FLMUINT)FLM_GET_TIMER();
		}
	}

	// Truncate the RFL file, if the truncate flag was set above.

	if( bTruncateRflFile)
	{
		(void)pFile->pRfl->truncate( uiRflTruncateSize);
	}

	// Truncate the files, if requested to do so - this would be a request of
	// FlmDbReduceSize.

	if( bTruncateRollBackLog)
	{
		if( RC_BAD( rc = pSFileHdl->truncateFile(
							(FLMUINT)FB2UD( &pucCommittedLogHdr [LOG_LOGICAL_EOF]))))
		{
			goto Exit;
		}
	}
	
	// Flush everything to disk
	
	if( RC_BAD( rc = pSFileHdl->flush()))
	{
		goto Exit;
	}

	// Re-enable the RFL volume OK flag - in case it was turned off somewhere.

	pFile->pRfl->setRflVolumeOk();

	// If we complete a checkpoint successfully, we want to set the
	// pFile->CheckpointRc so that new transactions can come in.
	// NOTE: CheckpointRc should only be set while we still have the
	// lock on the database - which should always be the case at this
	// point.  This routine can only be called if we have obtained both
	// the write lock and the file lock.

	pFile->CheckpointRc = FERR_OK;

	// If we were calculating our maximum dirty cache, finish the
	// calculation.

	if( uiCPStartTime)
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

		if( uiElapsedMilli >= 500)
		{

			// Calculate what could be written in 15 seconds - set maximum
			// to that.  If calculated maximum is zero, we will not change
			// the current maximum.

			ui15Seconds = FLM_SECS_TO_TIMER_UNITS( 15);

			uiMaximum = (FLMUINT)(((FLMUINT64)uiTotalToWrite *
							 (FLMUINT64)ui15Seconds) / (FLMUINT64)uiCPElapsedTime);

			if( uiMaximum)
			{
				// Low is maximum minus what could be written in roughly
				// two seconds.

				uiLow = uiMaximum - (uiMaximum / 7);

				// Only set the maximum if we are still in auto-calculate mode.

				if( gv_FlmSysData.SCacheMgr.bAutoCalcMaxDirty)
				{
					f_mutexLock( gv_FlmSysData.hShareMutex);

					// Test flag again after locking the mutex

					if( gv_FlmSysData.SCacheMgr.bAutoCalcMaxDirty)
					{
						gv_FlmSysData.SCacheMgr.uiMaxDirtyCache = uiMaximum;
						gv_FlmSysData.SCacheMgr.uiLowDirtyCache = uiLow;
					}

					f_mutexUnlock( gv_FlmSysData.hShareMutex);
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
RCODE ScaDoCheckpoint(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bTruncateRollBackLog,
	FLMBOOL				bForceCheckpoint,
	FLMINT				iForceReason,
	FLMUINT				uiCPFileNum,
	FLMUINT				uiCPOffset)
{
	RCODE					rc = FERR_OK;
	CP_INFO *			pCPInfo = pFile->pCPInfo;
	FLMBOOL				bWroteAll;
	FLMUINT				uiCPStartTime = 0;
	FLMUINT				uiTotalToWrite;
	FLMUINT				uiMaxDirtyCache;
	SCACHE *				pSCache;
	FLMUINT				uiTimestamp;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	
	if (pCPInfo)
	{
		pCPInfo->bDoingCheckpoint = TRUE;
		pCPInfo->uiStartTime = (FLMUINT)FLM_GET_TIMER();
		pCPInfo->bForcingCheckpoint = bForceCheckpoint;
		
		if( bForceCheckpoint)
		{
			pCPInfo->uiForceCheckpointStartTime = pCPInfo->uiStartTime;
		}
		
		pCPInfo->iForceCheckpointReason = iForceReason;
		pCPInfo->uiDataBlocksWritten =
		pCPInfo->uiLogBlocksWritten = 0;
	}

	uiTotalToWrite = (pFile->uiDirtyCacheCount + pFile->uiLogCacheCount) *
						pFile->FileHdr.uiBlockSize;

	if( bForceCheckpoint)
	{
		if (gv_FlmSysData.SCacheMgr.bAutoCalcMaxDirty)
		{
			uiCPStartTime = FLM_GET_TIMER();
		}
	}

	// If the amount of dirty cache is over our maximum, we must at least bring
	// it down below the low threshhold.  Otherwise, we set uiMaxDirtyCache
	// to the highest possible value - which will not require us to get
	// it below anything - because it is already within limits.

	if( gv_FlmSysData.SCacheMgr.uiMaxDirtyCache &&
		 uiTotalToWrite > gv_FlmSysData.SCacheMgr.uiMaxDirtyCache)
	{
		uiMaxDirtyCache = gv_FlmSysData.SCacheMgr.uiLowDirtyCache;
	}
	else
	{
		uiMaxDirtyCache = ~((FLMUINT)0);
	}
	
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	
	// Write out log blocks first.

	bWroteAll = TRUE;
	if( RC_BAD( rc = ScaFlushLogBlocks( pDbStats, pSFileHdl, pFile,
			TRUE, uiMaxDirtyCache, &bForceCheckpoint, &bWroteAll)))
	{
		goto Exit;
	}

	// If we didn't write out all log blocks, we got interrupted.

	if( !bWroteAll)
	{
		flmAssert( !bForceCheckpoint);
		goto Exit;
	}

	// Now write out dirty blocks

	if( RC_BAD( rc = ScaFlushDirtyBlocks( pDbStats, pSFileHdl, pFile,
			uiMaxDirtyCache, bForceCheckpoint, TRUE, &bWroteAll)))
	{
		goto Exit;
	}

	// If we didn't write out all dirty blocks, we got interrupted

	if( !bWroteAll)
	{
		flmAssert( !bForceCheckpoint);
		goto Exit;
	}

	// All dirty blocks and log blocks have been written, so we just
	// need to finish the checkpoint.

	if( RC_BAD( rc = scaFinishCheckpoint( pDbStats, pSFileHdl, pFile,
		bTruncateRollBackLog, uiCPFileNum, uiCPOffset, uiCPStartTime, uiTotalToWrite)))
	{
		goto Exit;
	}

Exit:

	// If we were attempting to force a checkpoint and it failed,
	// we want to set pFile->CheckpointRc, because we want to
	// prevent new transactions from starting until this situation
	// is cleared up (see fltrbeg.cpp).  Note that setting
	// pFile->CheckpointRc to something besides FERR_OK will cause
	// the checkpoint thread to force checkpoints whenever it is woke
	// up until it succeeds (see flopen.cpp).

	if( RC_BAD( rc) && bForceCheckpoint)
	{
		pFile->CheckpointRc = rc;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Timestamp all of the items in the free list

	if( bForceCheckpoint)
	{
		uiTimestamp = FLM_GET_TIMER();
		pSCache = gv_FlmSysData.SCacheMgr.pFirstFree;
		
		while( pSCache)
		{
			pSCache->uiBlkAddress = uiTimestamp;
			pSCache = pSCache->pNextInFile;
		}
	}

	if( pCPInfo)
	{
		pCPInfo->bDoingCheckpoint = FALSE;
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
FLMBOOL F_SCacheRelocator::canRelocate(
	void *		pvAlloc)
{
	SCACHE *		pSCache = (SCACHE *)pvAlloc;

	if( pSCache->uiUseCount)
	{
		return( FALSE);
	}

	return( TRUE);
}

/****************************************************************************
Desc:		Fixes up all pointers needed to allow an SCACHE struct to be
			moved to a different location in memory
****************************************************************************/
void F_SCacheRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	SCACHE *			pOldSCache = (SCACHE *)pvOldAlloc;
	SCACHE *			pNewSCache = (SCACHE *)pvNewAlloc;
	SCACHE **		ppBucket;
	SCACHE_MGR *	pSCacheMgr = &gv_FlmSysData.SCacheMgr;
	FFILE *			pFile = pOldSCache->pFile;

	flmAssert( !pOldSCache->uiUseCount);

	if( pNewSCache->pPrevInFile)
	{
		pNewSCache->pPrevInFile->pNextInFile = pNewSCache;
	}

	if( pNewSCache->pNextInFile)
	{
		pNewSCache->pNextInFile->pPrevInFile = pNewSCache;
	}

	if( pNewSCache->pPrevInGlobalList)
	{
		pNewSCache->pPrevInGlobalList->pNextInGlobalList = pNewSCache;
	}

	if( pNewSCache->pNextInGlobalList)
	{
		pNewSCache->pNextInGlobalList->pPrevInGlobalList = pNewSCache;
	}

	if( pNewSCache->pPrevInReplaceList)
	{
		pNewSCache->pPrevInReplaceList->pNextInReplaceList = pNewSCache;
	}

	if( pNewSCache->pNextInReplaceList)
	{
		pNewSCache->pNextInReplaceList->pPrevInReplaceList = pNewSCache;
	}

	if( pNewSCache->pPrevInHashBucket)
	{
		pNewSCache->pPrevInHashBucket->pNextInHashBucket = pNewSCache;
	}

	if( pNewSCache->pNextInHashBucket)
	{
		pNewSCache->pNextInHashBucket->pPrevInHashBucket = pNewSCache;
	}

	if( pNewSCache->pPrevInVersionList)
	{
		pNewSCache->pPrevInVersionList->pNextInVersionList = pNewSCache;
	}

	if( pNewSCache->pNextInVersionList)
	{
		pNewSCache->pNextInVersionList->pPrevInVersionList = pNewSCache;
	}

	if( pFile)
	{
		if( pFile->pSCacheList == pOldSCache)
		{
			pFile->pSCacheList = pNewSCache;
		}

		if( pFile->pLastDirtyBlk == pOldSCache)
		{
			pFile->pLastDirtyBlk = pNewSCache;
		}

		if( pFile->pFirstInLogList == pOldSCache)
		{
			pFile->pFirstInLogList = pNewSCache;
		}

		if( pFile->pLastInLogList == pOldSCache)
		{
			pFile->pLastInLogList = pNewSCache;
		}

		if( pFile->pFirstInNewList == pOldSCache)
		{
			pFile->pFirstInNewList = pNewSCache;
		}

		if( pFile->pLastInNewList == pOldSCache)
		{
			pFile->pLastInNewList = pNewSCache;
		}

		if( pFile->pTransLogList == pOldSCache)
		{
			pFile->pTransLogList = pNewSCache;
		}

		ppBucket = ScaHash(
				pFile->FileHdr.uiSigBitsInBlkSize,
				pOldSCache->uiBlkAddress);

		if( *ppBucket == pOldSCache)
		{
			*ppBucket = pNewSCache;
		}

		flmAssert( pFile->pPendingWriteList != pOldSCache);
	}

	if( pSCacheMgr->pMRUCache == pOldSCache)
	{
		pSCacheMgr->pMRUCache = pNewSCache;
	}

	if( pSCacheMgr->pLRUCache == pOldSCache)
	{
		pSCacheMgr->pLRUCache = pNewSCache;
	}

	if( pSCacheMgr->pMRUReplace == pOldSCache)
	{
		pSCacheMgr->pMRUReplace = pNewSCache;
	}

	if( pSCacheMgr->pLRUReplace == pOldSCache)
	{
		pSCacheMgr->pLRUReplace = pNewSCache;
	}

	if( pSCacheMgr->pFirstFree == pOldSCache)
	{
		pSCacheMgr->pFirstFree = pNewSCache;
	}

	if( pSCacheMgr->pLastFree == pOldSCache)
	{
		pSCacheMgr->pLastFree = pNewSCache;
	}

#ifdef FLM_DEBUG
	f_memset( pOldSCache, 0, sizeof( SCACHE));
#endif
}

/****************************************************************************
Desc:		
****************************************************************************/
SCACHE * F_BlockRelocator::getSCachePtr(
	void *		pvAlloc)
{
	FLMBYTE *	pucBlock = (FLMBYTE *)pvAlloc;
	FLMUINT		uiBlockAddr;
	SCACHE *		pSCache;
	SCACHE **	ppSCacheBucket;
	
	// Determine the block address and find the block in cache
	
	uiBlockAddr = GET_BH_ADDR( pucBlock);
	ppSCacheBucket = ScaHash( m_uiSigBitsInBlkSize, uiBlockAddr);

	// Search down the linked list of SCACHE structures off of the bucket
	// looking for the correct cache block.
	
	pSCache = *ppSCacheBucket;
	while( pSCache)
	{
		if( pSCache->uiBlkAddress == uiBlockAddr)
		{
			if( pSCache->pucBlk == pucBlock)
			{
				break;
			}
			else
			{
				SCACHE *		pTmpSCache;

				// Search the version list

				pTmpSCache = pSCache->pPrevInVersionList;
				while( pTmpSCache)
				{
					if( pTmpSCache->pucBlk == pucBlock)
					{
						pSCache = pTmpSCache;
						break;
					}

					pTmpSCache = pTmpSCache->pPrevInVersionList;
				}

				pTmpSCache = pSCache->pNextInVersionList;
				while( pTmpSCache)
				{
					if( pTmpSCache->pucBlk == pucBlock)
					{
						pSCache = pTmpSCache;
						break;
					}

					pTmpSCache = pTmpSCache->pNextInVersionList;
				}
			}
		}
		
		pSCache = pSCache->pNextInHashBucket;
	}
	
	return( pSCache);
}

/****************************************************************************
Desc:		
****************************************************************************/
FLMBOOL F_BlockRelocator::canRelocate(
	void *		pvAlloc)
{
	SCACHE *		pSCache = getSCachePtr( pvAlloc);
	
	if( !pSCache || pSCache->uiUseCount)
	{
		return( FALSE);
	}

	return( TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_BlockRelocator::relocate(
	void *		pvOldAlloc,
	void *		pvNewAlloc)
{
	SCACHE *		pSCache = getSCachePtr( pvOldAlloc);
	
	flmAssert( pSCache);
	flmAssert( pSCache->pucBlk == pvOldAlloc);
	
	pSCache->pucBlk = (FLMBYTE *)pvNewAlloc;
}

/****************************************************************************
Desc:	This function will encrypt the block of data passed in.  It will
		also fill the rest of the block with random data.  This function assumes
		that the buffer passed in includes the block header.
****************************************************************************/
RCODE ScaEncryptBlock(
	FFILE *		pFile,
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufLen,
	FLMUINT		uiBlockSize
	)
{
	RCODE				rc = FERR_OK;
	IXD *				pIxd;
	FLMUINT			uiIxNum;
#ifdef FLM_USE_NICI
	F_CCS *			pCcs;
#endif
	FLMUINT			uiEncLen = uiBufLen - BH_OVHD;

	if (uiEncLen == 0)
	{
		goto Exit;
	}

	uiIxNum = FB2UW( &pucBuffer[ BH_LOG_FILE_NUM]);

	// Get the index
	
	if (RC_BAD( rc = fdictGetIndex( pFile->pDictList, pFile->bInLimitedMode,
			uiIxNum, NULL, &pIxd, TRUE)))
	{
		// Not an index
		
		if (rc == FERR_BAD_IX)
		{
			rc = FERR_OK;
		}
		
		goto Exit;
	}

	// The index may not be encrypted.  We can just exit here.
	
	if (!pIxd || !pIxd->uiEncId)
	{
		flmAssert( pucBuffer[ BH_ENCRYPTED] == 0);
		pucBuffer[ BH_ENCRYPTED] = 0;
		goto Exit;
	}

	flmAssert(pucBuffer[ BH_ENCRYPTED]);
	pucBuffer[ BH_ENCRYPTED] = 1;

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
	F_UNREFERENCED_PARM( uiBlockSize);
	goto Exit;
#else

	if (pFile->bInLimitedMode)
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	// Need to get the encryption object
	
	pCcs = (F_CCS *)pFile->pDictList->pIttTbl[ pIxd->uiEncId].pvItem;

	flmAssert( pCcs);
	flmAssert( !(uiEncLen % 16));

	// Encrypt the buffer in place
	
	if (RC_BAD( rc = pCcs->encryptToStore( &pucBuffer[ BH_OVHD],
		uiEncLen, &pucBuffer[ BH_OVHD], &uiEncLen)))
	{
		goto Exit;
	}

	flmAssert( uiEncLen == uiBufLen - BH_OVHD);

	// Fill the rest of the buffer with random data.
	
	if (uiBufLen < uiBlockSize)
	{
		FLMUINT		uiContext;

		if (CCS_CreateContext(0, &uiContext) != 0)
		{
			rc = RC_SET( FERR_NICI_CONTEXT);
			goto Exit;
		}

		if (CCS_GetRandom( uiContext, &pucBuffer[uiBufLen],
			uiBlockSize - uiBufLen) != 0)
		{
			rc = RC_SET( FERR_NICI_BAD_RANDOM);
			goto Exit;
		}

		CCS_DestroyContext(uiContext);
	}

#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This function will decrypt the block of data passed in.
****************************************************************************/
RCODE ScaDecryptBlock(
	FFILE *		pFile,
	FLMBYTE *	pucBuffer)
{
	RCODE				rc = FERR_OK;
	IXD *				pIxd;
	FLMUINT			uiIxNum;
#ifdef FLM_USE_NICI
	F_CCS *			pCcs;
#endif
	FLMUINT			uiEncLen;
	FLMUINT			uiBufLen;

	uiBufLen = getEncryptSize( pucBuffer);
	uiEncLen = uiBufLen - BH_OVHD;

	if (!uiEncLen)
	{
		goto Exit;
	}

	uiIxNum = FB2UW( &pucBuffer[ BH_LOG_FILE_NUM]);

	// Get the index
	
	if (RC_BAD( rc = fdictGetIndex( pFile->pDictList, pFile->bInLimitedMode,
		uiIxNum, NULL, &pIxd, TRUE)))
	{
		// Not an index
		
		if (rc == FERR_BAD_IX)
		{
			rc = FERR_OK;
		}
		
		goto Exit;
	}

	// The index may not be encrypted.  We can just exit here.
	
	if (!pIxd || !pIxd->uiEncId)
	{
		if (pucBuffer[ BH_ENCRYPTED])
		{
			flmAssert(0);
		}
		
		pucBuffer[ BH_ENCRYPTED] = 0;
		goto Exit;
	}

	if (!pucBuffer[ BH_ENCRYPTED])
	{
		// Block was not encrypted on disk so don't decrypt it. Setting the 
		// BH_ENCRYPTED bit here will ensure we encrypt it next time we write it
		// out.
		
		flmAssert(0);
		pucBuffer[ BH_ENCRYPTED] = 1;
		goto Exit;
	}

#ifndef FLM_USE_NICI
	rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
	goto Exit;
#else

	if (pFile->bInLimitedMode)
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	// Need to get the encryption object.
	
	pCcs = (F_CCS *)pFile->pDictList->pIttTbl[ pIxd->uiEncId].pvItem;

	flmAssert( pCcs);

	if (RC_BAD( rc = pCcs->decryptFromStore( &pucBuffer[ BH_OVHD],
		uiEncLen, &pucBuffer[ BH_OVHD], &uiEncLen)))
	{
		goto Exit;
	}

	flmAssert( uiEncLen == uiBufLen - BH_OVHD);

#endif

Exit:

	return( rc);
}
