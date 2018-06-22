//-------------------------------------------------------------------------
// Desc:	Initialization and shutdown and system data.
// Tabs:	3
//
// Copyright (c) 1995-2007 Novell, Inc. All Rights Reserved.
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

#define ALLOCATE_SYS_DATA 1

#include "flaimsys.h"

#ifdef FLM_32BIT
	#if defined( FLM_LINUX)
	
	// With mmap'd memory on Linux, you're effectively limited to about ~2 GB.
	// Userspace only gets ~3GB of useable address space anyway, and then you
	// have all of the thread stacks too, which you can't have 
	// overlapping the heap.
	
		#define FLM_MAX_CACHE_SIZE		(1500 * 1024 * 1024)
	#else
		#define FLM_MAX_CACHE_SIZE		(2000 * 1024 * 1024)
	#endif
#else
	#define FLM_MAX_CACHE_SIZE			(~((FLMUINT)0))
#endif

#define DEFAULT_OPEN_THRESHOLD		100		// 100 file handles to cache
#define DEFAULT_MAX_AVAIL_TIME		900		// 15 minutes

FLMATOMIC			gv_flmSysSpinLock = 0;
FLMUINT				gv_uiFlmSysStartupCount = 0;

FSTATIC RCODE flmGetCacheBytes(
	FLMUINT		uiPercent,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiMinToLeave,
	FLMBOOL		bCalcOnAvailMem,
	FLMUINT		uiBytesCurrentlyInUse,
	FLMUINT *	puiCacheBytes);

FSTATIC void flmLockSysData( void);

FSTATIC void flmUnlockSysData( void);

FSTATIC RCODE flmSetCacheLimits(
	FLMUINT		uiNewTotalCacheSize,
	FLMBOOL		bForceLimit,
	FLMBOOL		bPreallocateCache);

FSTATIC void flmFreeEvent(
	FEVENT *			pEvent,
	F_MUTEX			hMutex,
	FEVENT **		ppEventListRV);

FSTATIC RCODE flmCloseDbFile(
	const char *	pszDbFileName,
	const char *	pszDataDir);

FSTATIC void flmShutdownDbThreads(
	FFILE *			pFile);

FSTATIC void flmCleanup( void);

FSTATIC void flmUnlinkFileFromBucket(
	FFILE *			pFile);

RCODE FLMAPI flmSystemMonitor(
	IF_Thread *		pThread);

FSTATIC RCODE flmRegisterHttpCallback(
	FLM_MODULE_HANDLE	hModule,
	const char * 		pszUrlString);

FSTATIC RCODE flmDeregisterHttpCallback( void);

/****************************************************************************
Desc: Sets the path for all temporary files that come into use within a
		FLAIM share structure.  The share mutex should be locked when
		settting when called from FlmConfig().
****************************************************************************/
FINLINE RCODE flmSetTmpDir(
	const char *	pszTmpDir)
{
	RCODE          rc;

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->doesFileExist( pszTmpDir)))
	{
		goto Exit;
	}

	f_strcpy( gv_FlmSysData.szTempDir, pszTmpDir);
	gv_FlmSysData.bTempDirSet = TRUE;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine frees all of the local dictionaries in a list of
		local dictionaries.
****************************************************************************/
FINLINE void flmFreeDictList(
	FDICT **		ppDictRV)
{
	FDICT *		pTmp;
	FDICT *		pDict = *ppDictRV;

	while (pDict)
	{
		pTmp = pDict;
		pDict = pDict->pNext;
		flmFreeDict( pTmp);
	}
	
	*ppDictRV = NULL;
}

/****************************************************************************
Desc: This routine determines the number of cache bytes to use for caching
		based on a percentage of available physical memory or a percentage
		of physical memory (depending on bCalcOnAvailMem flag).
		uiBytesCurrentlyInUse indicates how many bytes are currently allocated
		by FLAIM - so it can factor that in if the calculation is to be based
		on the available memory.
		Lower limit is 1 megabyte.
****************************************************************************/
FSTATIC RCODE flmGetCacheBytes(
	FLMUINT		uiPercent,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiMinToLeave,
	FLMBOOL		bCalcOnAvailMem,
	FLMUINT		uiBytesCurrentlyInUse,
	FLMUINT *	puiCacheBytes)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiMem = 0;
	FLMUINT64		ui64TotalPhysMem;
	FLMUINT64		ui64AvailPhysMem;
	
	if( RC_BAD( rc = f_getMemoryInfo( &ui64TotalPhysMem, &ui64AvailPhysMem)))
	{
		goto Exit;
	}
	
	if( ui64TotalPhysMem > FLM_MAX_UINT)
	{
		ui64TotalPhysMem = FLM_MAX_UINT;
	}
	
	if( ui64AvailPhysMem > ui64TotalPhysMem)
	{
		ui64AvailPhysMem = ui64TotalPhysMem;
	}
	
	uiMem = (FLMUINT)((bCalcOnAvailMem)
							? (FLMUINT)ui64AvailPhysMem
							: (FLMUINT)ui64TotalPhysMem);
	
	// If we are basing the calculation on available physical memory,
	// take in to account what has already been allocated.

	if (bCalcOnAvailMem)
	{
		if (uiMem > FLM_MAX_UINT - uiBytesCurrentlyInUse)
		{
			uiMem = FLM_MAX_UINT;
		}
		else
		{
			uiMem += uiBytesCurrentlyInUse;
		}
	}

	// If uiMax is zero, use uiMinToLeave to calculate the maximum.

	if (!uiMax)
	{
		if (!uiMinToLeave)
		{
			uiMax = uiMem;
		}
		else if (uiMinToLeave < uiMem)
		{
			uiMax = uiMem - uiMinToLeave;
		}
		else
		{
			uiMax = 0;
		}
	}

	// Calculate memory as a percentage of memory.

	uiMem = (FLMUINT)((uiMem > FLM_MAX_UINT / 100)
							? (FLMUINT)(uiMem / 100) * uiPercent
							: (FLMUINT)(uiMem * uiPercent) / 100);

	// Don't go above the maximum.

	if (uiMem > uiMax)
	{
		uiMem = uiMax;
	}

	// Don't go below the minimum.

	if (uiMem < uiMin)
	{
		uiMem = uiMin;
	}
	
Exit:

	*puiCacheBytes = uiMem;
	return( rc);
}

/***************************************************************************
Desc:	Lock the system data structure for access - called only by startup
		and shutdown.  NOTE: On platforms that do not support atomic exchange
		this is less than perfect - won't handle tight race conditions.
***************************************************************************/
FSTATIC void flmLockSysData( void)
{
	while( f_atomicExchange( &gv_flmSysSpinLock, 1) == 1)
	{
		f_sleep( 10);
	}
}

/***************************************************************************
Desc:	Unlock the system data structure for access - called only by startup
		and shutdown.
***************************************************************************/
FSTATIC void flmUnlockSysData( void)
{
	f_atomicExchange( &gv_flmSysSpinLock, 0);
}

/****************************************************************************
Desc : Startup FLAIM.
Notes: This routine may be called multiple times.  However, if that is done
		 FlmShutdown() should be called for each time this is called
		 successfully.  This routine does not handle race conditions on
		 platforms that do not support atomic increment.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStartup( void)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCacheBytes;
#ifdef FLM_USE_NICI
	int				iHandle;
#endif

	flmLockSysData();

	// See if FLAIM has already been started.  If so,
	// we are done.

	if( ++gv_uiFlmSysStartupCount > 1)
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = ftkStartup()))
	{
		goto Exit;
	}

	// The memset needs to be first.

	f_memset( &gv_FlmSysData, 0, sizeof( FLMSYSDATA));

	gv_FlmSysData.uiMaxFileSize = f_getMaxFileSize();

	flmAssert( gv_FlmSysData.uiMaxFileSize);

	// Initialize memory tracking variables - should be done before
	// call to f_memoryInit().

	gv_FlmSysData.hShareMutex = F_MUTEX_NULL;
	gv_FlmSysData.uiMaxStratifyIterations = DEFAULT_MAX_STRATIFY_ITERATIONS;
	gv_FlmSysData.uiMaxStratifyTime = DEFAULT_MAX_STRATIFY_TIME;

	// Initialize the event categories to have no mutex.
	
	gv_FlmSysData.UpdateEvents.hMutex = F_MUTEX_NULL;
	gv_FlmSysData.LockEvents.hMutex = F_MUTEX_NULL;
	gv_FlmSysData.SizeEvents.hMutex = F_MUTEX_NULL;
	
	// Set the default file open flags
	
	gv_FlmSysData.uiFileOpenFlags = 
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT;

	gv_FlmSysData.uiFileCreateFlags = 
		gv_FlmSysData.uiFileOpenFlags | FLM_IO_EXCL | FLM_IO_CREATE_DIR;
		
#ifdef FLM_DBG_LOG
	flmDbgLogInit();
#endif

	gv_FlmSysData.uiMaxUnusedTime = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_MAX_UNUSED_TIME);
		
	gv_FlmSysData.uiMaxCPInterval = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_MAX_CP_INTERVAL);
		
	gv_FlmSysData.uiMaxTransTime = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_MAX_TRANS_SECS);
		
	gv_FlmSysData.uiMaxTransInactiveTime = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_MAX_TRANS_INACTIVE_SECS);

	if( f_canGetMemoryInfo())
	{
		gv_FlmSysData.bDynamicCacheAdjust = TRUE;
		gv_FlmSysData.uiCacheAdjustPercent = DEFAULT_CACHE_ADJUST_PERCENT;
		gv_FlmSysData.uiCacheAdjustMin = DEFAULT_CACHE_ADJUST_MIN;
		gv_FlmSysData.uiCacheAdjustMax = DEFAULT_CACHE_ADJUST_MAX;
		gv_FlmSysData.uiCacheAdjustMinToLeave = DEFAULT_CACHE_ADJUST_MIN_TO_LEAVE;
		
		gv_FlmSysData.uiCacheAdjustInterval = 
			FLM_SECS_TO_TIMER_UNITS( DEFAULT_CACHE_ADJUST_INTERVAL);
			
		if( RC_BAD( rc = flmGetCacheBytes( gv_FlmSysData.uiCacheAdjustPercent,
			gv_FlmSysData.uiCacheAdjustMin, gv_FlmSysData.uiCacheAdjustMax,
			gv_FlmSysData.uiCacheAdjustMinToLeave, TRUE, 0, &uiCacheBytes)))
		{
			goto Exit;
		}
	}
	else
	{
		gv_FlmSysData.bDynamicCacheAdjust = FALSE;
		gv_FlmSysData.uiCacheAdjustInterval = 0;
		uiCacheBytes = DEFAULT_CACHE_ADJUST_MIN;
	}

	if( uiCacheBytes > FLM_MAX_CACHE_SIZE)
	{
		uiCacheBytes = FLM_MAX_CACHE_SIZE;
	}

	gv_FlmSysData.uiBlockCachePercentage = DEFAULT_BLOCK_CACHE_PERCENTAGE;

	gv_FlmSysData.uiCacheCleanupInterval = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_CACHE_CLEANUP_INTERVAL);
	gv_FlmSysData.uiUnusedCleanupInterval = 
		FLM_SECS_TO_TIMER_UNITS( DEFAULT_UNUSED_CLEANUP_INTERVAL);
			
	// Get a pointer to the thread manager
	
	if( RC_BAD( rc = FlmGetThreadMgr( &gv_FlmSysData.pThreadMgr)))
	{
		goto Exit;
	}
	
	// Allocate thread group IDs
	
	gv_uiBackIxThrdGroup = gv_FlmSysData.pThreadMgr->allocGroupId();
	gv_uiCPThrdGrp = gv_FlmSysData.pThreadMgr->allocGroupId();
	gv_uiDbThrdGrp = gv_FlmSysData.pThreadMgr->allocGroupId();

	// Initialize the slab manager
	
	if( RC_BAD( rc = FlmAllocSlabManager( &gv_FlmSysData.pSlabManager)))
	{
		goto Exit;
	}
	
	if( !gv_FlmSysData.bDynamicCacheAdjust)
	{
		if( RC_BAD( rc = gv_FlmSysData.pSlabManager->setup( uiCacheBytes)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = gv_FlmSysData.pSlabManager->setup( 0)))
		{
			goto Exit;
		}
	}
	
	// Divide cache bytes evenly between block and record cache.

	gv_FlmSysData.uiMaxCache = uiCacheBytes;
	uiCacheBytes >>= 1;
	
	if (RC_BAD( rc = ScaInit( uiCacheBytes)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmRcaInit( uiCacheBytes)))
	{
		goto Exit;
	}

	// Create the mutex for controlling access to the structure

	if (RC_BAD( rc = f_mutexCreate( &gv_FlmSysData.hShareMutex)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &gv_FlmSysData.hQueryMutex)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &gv_FlmSysData.HttpConfigParms.hMutex)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &gv_FlmSysData.hHttpSessionMutex)))
	{
		goto Exit;
	}

	// Initialize a statistics structure

	if (RC_BAD( rc = flmStatInit( &gv_FlmSysData.Stats, TRUE)))
	{
		goto Exit;
	}
	gv_FlmSysData.bStatsInitialized = TRUE;

	// Allocate memory for the file name hash table

	if (RC_BAD(rc = f_allocHashTable( FILE_HASH_ENTRIES,
								&gv_FlmSysData.pFileHashTbl)))
	{
		goto Exit;
	}

	gv_FlmSysData.uiNextFFileId = 1;

	// Get the file system object
	
	if( RC_BAD( rc = FlmGetFileSystem( &gv_FlmSysData.pFileSystem)))
	{
		goto Exit;
	}
	
	// Allocate a file handle cache
	
	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->allocFileHandleCache( 
		DEFAULT_OPEN_THRESHOLD, DEFAULT_MAX_UNUSED_TIME,		
		&gv_FlmSysData.pFileHdlCache)))
	{
		goto Exit;
	}
	
	// Set up the session manager

	if( (gv_FlmSysData.pSessionMgr = f_new F_SessionMgr) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.pSessionMgr->setupSessionMgr()))
	{
		goto Exit;
	}

	// Set up mutexes for the event table.

	if (RC_BAD( rc = f_mutexCreate(
							&gv_FlmSysData.UpdateEvents.hMutex)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = f_mutexCreate(
							&gv_FlmSysData.LockEvents.hMutex)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = f_mutexCreate(
							&gv_FlmSysData.SizeEvents.hMutex)))
	{
		goto Exit;
	}
	
	// Start the monitor thread - ALWAYS DO LAST.  EVERYTHING MUST BE
	// SETUP PROPERLY BEFORE STARTING THIS THREAD.

	if (RC_BAD( rc = f_threadCreate( &gv_FlmSysData.pMonitorThrd,
		flmSystemMonitor, "FLAIM System Monitor")))
	{
		goto Exit;
	}
	
#ifdef FLM_USE_NICI
	iHandle  = (int)f_getpid();

	// Initialize NICI
	
	if (CCS_Init(&iHandle))
	{
		// Failure.
		rc = RC_SET( FERR_NICI_INIT_FAILED);
		goto Exit;
	}
#endif

Exit:

	// If not successful, free up any resources that were allocated.

	if (RC_BAD( rc))
	{
		flmCleanup();
	}
	
	flmUnlockSysData();
	return( rc);
}

/****************************************************************************
Desc: This routine sets the limits for record cache and block cache - dividing
		the total cache between the two caches.  It uses the same ratio
		currently in force.
****************************************************************************/
FSTATIC RCODE flmSetCacheLimits(
	FLMUINT		uiNewTotalCacheSize,
	FLMBOOL		bForceLimit,
	FLMBOOL		bPreallocateCache)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNewBlockCacheSize;
	FLMBOOL		bResizeAfterConfig = FALSE;
	
	if( !bForceLimit && uiNewTotalCacheSize > FLM_MAX_CACHE_SIZE)
	{
		uiNewTotalCacheSize = FLM_MAX_CACHE_SIZE;
	}

	if( bPreallocateCache)
	{
		if( gv_FlmSysData.bDynamicCacheAdjust)
		{
			// Can't pre-allocate and dynamically adjust.

			goto DONT_PREALLOCATE;
		}

		if( RC_BAD( rc = gv_FlmSysData.pSlabManager->resize( 
			uiNewTotalCacheSize, TRUE, &uiNewTotalCacheSize)))
		{
			// Log a message indicating that we couldn't pre-allocate
			// the cache

			flmLogMessage( F_DEBUG_MESSAGE, FLM_YELLOW, FLM_BLACK,
				"WARNING: Couldn't pre-allocate cache.");

			goto DONT_PREALLOCATE;				
		}
		
		gv_FlmSysData.bCachePreallocated = TRUE;
	}
	else
	{
DONT_PREALLOCATE:

		bResizeAfterConfig = TRUE;
		gv_FlmSysData.bCachePreallocated = FALSE;
	}
	
	if( gv_FlmSysData.uiBlockCachePercentage == 100)
	{
		uiNewBlockCacheSize = uiNewTotalCacheSize;
	}
	else
	{
		uiNewBlockCacheSize = (FLMUINT)((uiNewTotalCacheSize / 100) * 
			gv_FlmSysData.uiBlockCachePercentage);
	}

	if (RC_OK( rc = ScaConfig( FLM_CACHE_LIMIT,
							(void *)uiNewBlockCacheSize, (void *)0)))
	{
		rc = flmRcaConfig( FLM_CACHE_LIMIT,
							(void *)(uiNewTotalCacheSize - uiNewBlockCacheSize),
							(void *)0);
	}

	if( bResizeAfterConfig)
	{
		(void)gv_FlmSysData.pSlabManager->resize( uiNewTotalCacheSize, FALSE);
	}
	
	gv_FlmSysData.uiMaxCache = uiNewTotalCacheSize;
	return( rc);
}

/****************************************************************************
Desc : Configures how memory will be dynamically regulated.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmSetDynamicMemoryLimit(
	FLMUINT			uiCacheAdjustPercent,
	FLMUINT			uiCacheAdjustMin,
	FLMUINT			uiCacheAdjustMax,
	FLMUINT			uiCacheAdjustMinToLeave)
{
	if( !f_canGetMemoryInfo())
	{
		return( RC_SET( FERR_NOT_IMPLEMENTED));
	}
	else
	{
		RCODE		rc = FERR_OK;
		FLMUINT	uiCacheBytes;
	
		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		gv_FlmSysData.bDynamicCacheAdjust = TRUE;
		flmAssert( uiCacheAdjustPercent > 0 &&
					  uiCacheAdjustPercent <= 100);
		gv_FlmSysData.uiCacheAdjustPercent = uiCacheAdjustPercent;
		gv_FlmSysData.uiCacheAdjustMin = uiCacheAdjustMin;
		gv_FlmSysData.uiCacheAdjustMax = uiCacheAdjustMax;
		gv_FlmSysData.uiCacheAdjustMinToLeave = uiCacheAdjustMinToLeave;
		
		if( RC_OK( rc = flmGetCacheBytes( gv_FlmSysData.uiCacheAdjustPercent,
			gv_FlmSysData.uiCacheAdjustMin, gv_FlmSysData.uiCacheAdjustMax,
			gv_FlmSysData.uiCacheAdjustMinToLeave, TRUE,
			gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +
				gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated,
			&uiCacheBytes)))
		{
			rc = flmSetCacheLimits( uiCacheBytes, FALSE, FALSE);
		}
		
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		return( rc);
	}
}

/****************************************************************************
Desc : Sets a hard memory limit for cache.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmSetHardMemoryLimit(
	FLMUINT		uiPercent,
	FLMBOOL		bPercentOfAvail,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiMinToLeave,
	FLMBOOL		bPreallocate)
{
	RCODE			rc = FERR_OK;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	
	gv_FlmSysData.bDynamicCacheAdjust = FALSE;
	
	if (uiPercent)
	{
		if( !f_canGetMemoryInfo())
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
		}
		else
		{
			FLMUINT	uiCacheBytes;
	
			if( RC_OK( rc = flmGetCacheBytes( 
				uiPercent, uiMin, uiMax, uiMinToLeave,
				bPercentOfAvail, gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +
					gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated, 
					&uiCacheBytes)))
			{
				rc = flmSetCacheLimits( uiCacheBytes, FALSE, bPreallocate);
			}
		}
	}
	else
	{
		rc = flmSetCacheLimits( uiMax, TRUE, bPreallocate);
	}
	
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	
	return( rc);
}

/****************************************************************************
Desc : Returns information about memory usage.
****************************************************************************/
FLMEXP void FLMAPI FlmGetMemoryInfo(
	FLM_MEM_INFO *	pMemInfo)
{
	f_memset( pMemInfo, 0, sizeof( FLM_MEM_INFO));
	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	pMemInfo->bDynamicCacheAdjust = gv_FlmSysData.bDynamicCacheAdjust;
	pMemInfo->uiCacheAdjustPercent = gv_FlmSysData.uiCacheAdjustPercent;
	pMemInfo->uiCacheAdjustMin = gv_FlmSysData.uiCacheAdjustMin;
	pMemInfo->uiCacheAdjustMax = gv_FlmSysData.uiCacheAdjustMax;
	pMemInfo->uiCacheAdjustMinToLeave = gv_FlmSysData.uiCacheAdjustMinToLeave;

	// Return record cache information.

	f_memcpy( &pMemInfo->RecordCache, &gv_FlmSysData.RCacheMgr.Usage,
					sizeof( FLM_CACHE_USAGE));

	// Return block cache information.

	f_memcpy( &pMemInfo->BlockCache, &gv_FlmSysData.SCacheMgr.Usage,
					sizeof( FLM_CACHE_USAGE));

	pMemInfo->uiFreeBytes = gv_FlmSysData.SCacheMgr.uiFreeBytes;
	pMemInfo->uiFreeCount = gv_FlmSysData.SCacheMgr.uiFreeCount;
	pMemInfo->uiReplaceableCount = gv_FlmSysData.SCacheMgr.uiReplaceableCount;
	pMemInfo->uiReplaceableBytes = gv_FlmSysData.SCacheMgr.uiReplaceableBytes;
	
	if( gv_FlmSysData.pFileHashTbl)
	{
		FLMUINT			uiLoop;
		FFILE *			pFile;

		for( uiLoop = 0; uiLoop < FILE_HASH_ENTRIES; uiLoop++)
		{
			if( (pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[ 
				uiLoop].pFirstInBucket) != NULL)
			{
				while( pFile)
				{
					if( pFile->uiDirtyCacheCount)
					{
						pMemInfo->uiDirtyBytes += 
							pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;
						pMemInfo->uiDirtyCount += pFile->uiDirtyCacheCount;
					}

					if( pFile->uiNewCount)
					{
						pMemInfo->uiNewBytes += 
							pFile->uiNewCount * pFile->FileHdr.uiBlockSize;
						pMemInfo->uiNewCount += pFile->uiNewCount;
					}

					if( pFile->uiLogCacheCount)
					{
						pMemInfo->uiLogBytes += 
							pFile->uiLogCacheCount * pFile->FileHdr.uiBlockSize;
						pMemInfo->uiLogCount += pFile->uiLogCacheCount;
					}

					pFile = pFile->pNext;
				}
			}
		}
	}

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc : Close a database file - all background threads are closed too.
****************************************************************************/
FSTATIC RCODE flmCloseDbFile(
	const char *	pszDbFileName,
	const char *	pszDataDir)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bMutexLocked = FALSE;
	FFILE *		pFile;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

Retry:

	// Look up the file using flmFindFile to see if we have the
	// file open.  May unlock and re-lock the global mutex.

	if (RC_BAD( rc = flmFindFile( pszDbFileName, pszDataDir, &pFile)))
	{
		goto Exit;
	}

	// If we did not find the file, we are OK.

	if (!pFile)
	{
		goto Exit;
	}

	// If the file is in the process of being opened by another
	// thread, wait for the open to complete.

	if( pFile->uiFlags & DBF_BEING_OPENED)
	{
		if( RC_BAD( rc = f_notifyWait( 
			gv_FlmSysData.hShareMutex, F_SEM_NULL, NULL, &pFile->pOpenNotifies)))
		{
			// If f_notifyWait returns a bad RC, assume that the other thread
			// will unlock and free the pFile structure.  This routine should
			// only unlock the pFile if an error occurs at some other point.
			// See flmVerifyFileUse.

			goto Exit;
		}

		goto Retry;
	}

	if( pFile->uiUseCount)
	{
		// Increment the use count temporarily so that the FFILE won't be
		// moved to the NU list because of db close calls made by
		// releaseFileResources.

		pFile->uiUseCount++;

		// Must unlock the mutex prior to calling releaseFileResources because
		// it may call API-level routines (such as FlmDbClose) that do not
		// expect the mutex to be locked.

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// First, all session resources are released that are
		// associated with the specified database

		if( gv_FlmSysData.pSessionMgr)
		{
			gv_FlmSysData.pSessionMgr->releaseFileResources( pFile);
		}

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;

		// Decrement the temporary use count that was added above.
		// By now, the FFILE may no longer be in use.  If so, we need to
		// link it to the NU list.

		if (!(--pFile->uiUseCount))
		{
			// If the "must close" flag is set, it indicates that
			// the FFILE is being forced to close.  Put the FFILE in
			// the NU list, but specify that it should be quickly
			// timed-out.  We link the file to the NU list even
			// though we are going to unlink it below because
			// flmLinkFileToNUList closes the RFL file(s).

			flmLinkFileToNUList( pFile, pFile->bMustClose);
		}
	}

	// If the FFILE is in the NU list, unlink it

	flmUnlinkFileFromNUList( pFile);

	// If we have non-background threads accessing the database,
	// we cannot close the file.

	if (pFile->uiUseCount > pFile->uiInternalUseCount)
	{
		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

	// Close the file.

	flmFreeFile( pFile);

	// Unlock the mutex

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
	return( rc);
}

/****************************************************************************
Desc : Register our http callback function
****************************************************************************/
FSTATIC RCODE flmRegisterHttpCallback(
	FLM_MODULE_HANDLE	hModule,
	const char * 		pszUrlString)
{
	RCODE			rc = FERR_OK;
	char *		pszTemp;
	FLMUINT		uiRegFlags;

	if (gv_FlmSysData.HttpConfigParms.bRegistered)
	{
		rc = RC_SET( FERR_HTTP_REGISTER_FAILURE);
		goto Exit;
	}

	// Need to save the Url string for later use...

	if( RC_BAD( rc = f_alloc( f_strlen( pszUrlString) + 1, &pszTemp)))
	{
		goto Exit;
	}

	f_strcpy( pszTemp, pszUrlString);
	
	// Set the flags that tell the server what kind of authentication
	// we want:
	// HR_STK_BOTH = Allow both http and https
	// HR_AUTH_USERSA = Allow any user in the NDS tree or the SAdmin user
	// HR_REALM_NDS = Use the NDS realm for authentication
	
	uiRegFlags = HR_STK_BOTH | HR_AUTH_USERSA | HR_REALM_NDS;

	if (gv_FlmSysData.HttpConfigParms.fnReg)
	{
		if( gv_FlmSysData.HttpConfigParms.fnReg( hModule, pszUrlString,
			(FLMUINT32)uiRegFlags, flmHttpCallback, NULL, NULL) != 0)
		{
			rc = RC_SET( FERR_HTTP_REGISTER_FAILURE);
			goto Exit;
		}
	}
	else
	{
		flmAssert( 0);
		rc = RC_SET( FERR_NO_HTTP_STACK);
		goto Exit;
	}

	// Save the URL string in gv_FlmSysData
	
	gv_FlmSysData.HttpConfigParms.pszURLString = pszTemp;
	gv_FlmSysData.HttpConfigParms.uiURLStringLen = f_strlen( pszTemp);
	pszTemp = NULL;

	gv_FlmSysData.HttpConfigParms.bRegistered = TRUE;

Exit:

	if (RC_BAD( rc))
	{
		if (pszTemp)
		{
			f_free( &pszTemp);
			pszTemp = NULL;
		}
	}

	return( rc);
}

/****************************************************************************
Desc : Deregister our http callback function
****************************************************************************/
FSTATIC RCODE flmDeregisterHttpCallback( void)
{
	RCODE		rc = FERR_OK;
	if (!gv_FlmSysData.HttpConfigParms.bRegistered)
	{
		rc = RC_SET( FERR_HTTP_DEREG_FAILURE);
		goto Exit;
	}

	if ( !gv_FlmSysData.HttpConfigParms.fnDereg ||
			!gv_FlmSysData.HttpConfigParms.pszURLString )
	{
		flmAssert( 0);
		rc = RC_SET( FERR_NO_HTTP_STACK);
		goto Exit;
	}

	if ( gv_FlmSysData.HttpConfigParms.fnDereg( 
				gv_FlmSysData.HttpConfigParms.pszURLString,
				flmHttpCallback) != 0)
	{
		rc = RC_SET( FERR_HTTP_DEREG_FAILURE);
		goto Exit;
	}
	
	if( gv_FlmSysData.HttpConfigParms.pszURLString)
	{
		f_free( &gv_FlmSysData.HttpConfigParms.pszURLString);
		gv_FlmSysData.HttpConfigParms.pszURLString = NULL;
	}

	// Now, tell the callback function to delete the webpage factory and cleanup
	// the allocated memory etc...
	f_mutexLock( gv_FlmSysData.HttpConfigParms.hMutex);
	while (gv_FlmSysData.HttpConfigParms.uiUseCount)
	{
		f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);
		f_sleep( 10);
		f_mutexLock( gv_FlmSysData.HttpConfigParms.hMutex);
	}
	flmHttpCallback( NULL, NULL);
	f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);


	gv_FlmSysData.HttpConfigParms.bRegistered = FALSE;
Exit:
	return( rc);
}

/****************************************************************************
Desc: Configures share attributes.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmConfig(
  	eFlmConfigTypes 	eConfigType,
	void *				Value1,
	void *		   	Value2)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiValue;
	FLMUINT				uiSave;
	FLMUINT				uiCurrTime;
	FLMUINT				uiSaveMax;

	switch( eConfigType)
	{
		case FLM_OPEN_THRESHOLD:
		{
			rc = gv_FlmSysData.pFileHdlCache->setOpenThreshold( (FLMUINT)Value1);
			break;
		}
		
		case FLM_DIRECT_IO_STATE:
		{
			// Direct I/O state can only be changed when there are no open
			// databases.  We also only allow DIO to be disabled on Unix
			// platforms.

#ifdef FLM_UNIX			
			f_mutexLock( gv_FlmSysData.hShareMutex);
			
			if( !gv_FlmSysData.uiOpenFFiles)
			{
				if( (FLMBOOL)((FLMUINT)Value1))
				{
					gv_FlmSysData.uiFileOpenFlags = 
						FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT;
				}
				else
				{
					gv_FlmSysData.uiFileOpenFlags = 
						FLM_IO_RDWR | FLM_IO_SH_DENYNONE;
				}
				
				gv_FlmSysData.uiFileCreateFlags = 
					gv_FlmSysData.uiFileOpenFlags | FLM_IO_EXCL | FLM_IO_CREATE_DIR;
			}
			
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
#endif
			break;
		}
	
		case FLM_CACHE_LIMIT:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
			gv_FlmSysData.bDynamicCacheAdjust = FALSE;
			rc = flmSetCacheLimits( (FLMUINT)Value1, TRUE, (FLMBOOL)(FLMUINT)Value2);
			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FLM_BLOCK_CACHE_PERCENTAGE:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

			if( (gv_FlmSysData.uiBlockCachePercentage = (FLMUINT)Value1) > 100)
			{
				gv_FlmSysData.uiBlockCachePercentage = 100;
			}

			rc = flmSetCacheLimits( 
				gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes + 
				gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes,
				gv_FlmSysData.bDynamicCacheAdjust ? FALSE : TRUE,
				gv_FlmSysData.bCachePreallocated);

			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FLM_SCACHE_DEBUG:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
			if (RC_OK( rc = ScaConfig( FLM_SCACHE_DEBUG, Value1, Value2)))
			{
				rc = flmRcaConfig( FLM_SCACHE_DEBUG, Value1, Value2);
			}
			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FLM_CLOSE_UNUSED_FILES:
		{
	
			// Timeout inactive sessions
	
			if( gv_FlmSysData.pSessionMgr)
			{
				gv_FlmSysData.pSessionMgr->timeoutInactiveSessions( 
					(FLMUINT)Value1, TRUE);
			}
	
			// Convert seconds to timer units
	
			uiValue = FLM_SECS_TO_TIMER_UNITS( (FLMUINT) Value1);
	
			// Free any other unused structures that have not been used for the
			// specified amount of time.
	
			uiCurrTime = (FLMUINT)FLM_GET_TIMER();
			f_mutexLock( gv_FlmSysData.hShareMutex);
	
			// Temporarily set the maximum unused seconds in the FLMSYSDATA structure
			// to the value that was passed in to Value1.  Restore it after
			// calling flmCheckNUStructs.
	
			uiSave = gv_FlmSysData.uiMaxUnusedTime;
			gv_FlmSysData.uiMaxUnusedTime = uiValue;
	
			// May unlock and re-lock the global mutex.
			flmCheckNUStructs( uiCurrTime);
			gv_FlmSysData.uiMaxUnusedTime = uiSave;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
			
		case FLM_CLOSE_ALL_FILES:
		{
			break;
		}
	
		case FLM_START_STATS:
		{
			(void)flmStatStart( &gv_FlmSysData.Stats);
	
			// Start query statistics, if they have not
			// already been started.
	
			f_mutexLock( gv_FlmSysData.hQueryMutex);
			if (!gv_FlmSysData.uiMaxQueries)
			{
				gv_FlmSysData.uiMaxQueries = 20;
				gv_FlmSysData.bNeedToUnsetMaxQueries = TRUE;
			}
			f_mutexUnlock( gv_FlmSysData.hQueryMutex);
			break;
		}
	
		case FLM_STOP_STATS:
		{
			(void)flmStatStop( &gv_FlmSysData.Stats);
	
			// Stop query statistics, if they were
			// started by a call to FLM_START_STATS.
	
			f_mutexLock( gv_FlmSysData.hQueryMutex);
			if (gv_FlmSysData.bNeedToUnsetMaxQueries)
			{
				gv_FlmSysData.uiMaxQueries = 0;
				flmFreeSavedQueries( TRUE);
				// NOTE: flmFreeSavedQueries unlocks the mutex.
			}
			else
			{
				f_mutexUnlock( gv_FlmSysData.hQueryMutex);
			}
			
			break;
		}
	
		case FLM_RESET_STATS:
		{
	
			// Lock the record cache manager's mutex
	
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	
			// Reset the record cache statistics
			
			gv_FlmSysData.RCacheMgr.uiIoWaits = 0;
			gv_FlmSysData.RCacheMgr.Usage.uiCacheHits = 0;
			gv_FlmSysData.RCacheMgr.Usage.uiCacheHitLooks = 0;
			gv_FlmSysData.RCacheMgr.Usage.uiCacheFaults = 0;
			gv_FlmSysData.RCacheMgr.Usage.uiCacheFaultLooks = 0;
	
			// Reset the block cache statistics.
	
			gv_FlmSysData.SCacheMgr.uiIoWaits = 0;
			gv_FlmSysData.SCacheMgr.Usage.uiCacheHits = 0;
			gv_FlmSysData.SCacheMgr.Usage.uiCacheHitLooks = 0;
			gv_FlmSysData.SCacheMgr.Usage.uiCacheFaults = 0;
			gv_FlmSysData.SCacheMgr.Usage.uiCacheFaultLooks = 0;
	
			// Unlock the cache manager's mutex
	
			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
	
			(void)flmStatReset( &gv_FlmSysData.Stats, FALSE, TRUE);
	
			f_mutexLock( gv_FlmSysData.hQueryMutex);
			uiSaveMax = gv_FlmSysData.uiMaxQueries;
			gv_FlmSysData.uiMaxQueries = 0;
			flmFreeSavedQueries( TRUE);
			// NOTE: flmFreeSavedQueries unlocks the mutex.
	
			// Restore the old maximum
	
			if (uiSaveMax)
			{
	
				// flmFreeSavedQueries unlocks the mutex, so we
				// must relock it to restore the old maximum.
	
				f_mutexLock( gv_FlmSysData.hQueryMutex);
				gv_FlmSysData.uiMaxQueries = uiSaveMax;
				f_mutexUnlock( gv_FlmSysData.hQueryMutex);
			}
			
			break;
		}
	
		case FLM_TMPDIR:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			rc = flmSetTmpDir( (const char *)Value1);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
			
		case FLM_MAX_CP_INTERVAL:
		{
			gv_FlmSysData.uiMaxCPInterval = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT) Value1);
			break;
		}
	
		case FLM_BLOB_EXT:
		{
			const char *	pszTmp = (const char *)Value1;

			if (!pszTmp)
			{
				gv_FlmSysData.ucBlobExt [0] = 0;
			}
			else
			{
				int	iCnt;

				// Don't save any more than 63 characters.

				for (iCnt = 0;
					  ((iCnt < 63) && (*pszTmp));
					  iCnt++, pszTmp++)
				{
					gv_FlmSysData.ucBlobExt [iCnt] = *pszTmp;
				}
				gv_FlmSysData.ucBlobExt [iCnt] = 0;
			}
			
			break;
		}
	
		case FLM_MAX_TRANS_SECS:
		{
			gv_FlmSysData.uiMaxTransTime = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT) Value1);
			break;
		}
	
		case FLM_MAX_TRANS_INACTIVE_SECS:
		{
			gv_FlmSysData.uiMaxTransInactiveTime = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT) Value1);
			break;
		}
	
		case FLM_CACHE_ADJUST_INTERVAL:
		{
			gv_FlmSysData.uiCacheAdjustInterval = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT)Value1);
			break;
		}
	
		case FLM_CACHE_CLEANUP_INTERVAL:
		{
			gv_FlmSysData.uiCacheCleanupInterval = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT)Value1);
			break;
		}
	
		case FLM_UNUSED_CLEANUP_INTERVAL:
		{
			gv_FlmSysData.uiUnusedCleanupInterval = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT)Value1);
			break;
		}
	
		case FLM_MAX_UNUSED_TIME:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			gv_FlmSysData.uiMaxUnusedTime = 
				FLM_SECS_TO_TIMER_UNITS( (FLMUINT)Value1);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}
	
		case FLM_CACHE_CHECK:
			gv_FlmSysData.bCheckCache = (FLMBOOL)((Value1 != 0)
															  ? (FLMBOOL)TRUE
															  : (FLMBOOL)FALSE);
			break;
	
		case FLM_CLOSE_FILE:
			rc = flmCloseDbFile( (const char *)Value1, (const char *)Value2);
			break;
	
		case FLM_LOGGER:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( !gv_FlmSysData.pLogger && Value1)
			{
				gv_FlmSysData.pLogger = (IF_LoggerClient *)Value1;
				gv_FlmSysData.pLogger->AddRef();
				f_setLoggerClient( gv_FlmSysData.pLogger);
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_ASSIGN_HTTP_SYMS:
			if( gv_FlmSysData.HttpConfigParms.fnReg ||
				 gv_FlmSysData.HttpConfigParms.fnDereg ||
				 gv_FlmSysData.HttpConfigParms.fnReqPath ||
				 gv_FlmSysData.HttpConfigParms.fnReqQuery ||
				 gv_FlmSysData.HttpConfigParms.fnReqHdrValue ||
				 gv_FlmSysData.HttpConfigParms.fnSetHdrValue ||
				 gv_FlmSysData.HttpConfigParms.fnPrintf ||
				 gv_FlmSysData.HttpConfigParms.fnEmit ||
				 gv_FlmSysData.HttpConfigParms.fnSetNoCache ||
				 gv_FlmSysData.HttpConfigParms.fnSendHeader ||
				 gv_FlmSysData.HttpConfigParms.fnSetIOMode ||
				 gv_FlmSysData.HttpConfigParms.fnSendBuffer ||
				 gv_FlmSysData.HttpConfigParms.fnAcquireSession ||
				 gv_FlmSysData.HttpConfigParms.fnReleaseSession ||
				 gv_FlmSysData.HttpConfigParms.fnAcquireUser ||
				 gv_FlmSysData.HttpConfigParms.fnReleaseUser ||
				 gv_FlmSysData.HttpConfigParms.fnSetSessionValue ||
				 gv_FlmSysData.HttpConfigParms.fnGetSessionValue ||
				 gv_FlmSysData.HttpConfigParms.fnGetGblValue ||
				 gv_FlmSysData.HttpConfigParms.fnSetGblValue ||
				 gv_FlmSysData.HttpConfigParms.fnRecvBuffer)
			{
				rc = RC_SET( FERR_HTTP_SYMS_EXIST);
				goto Exit;
			}
			else
			{
				gv_FlmSysData.HttpConfigParms.fnReg = ((HTTPCONFIGPARAMS *)Value1)->fnReg;
				gv_FlmSysData.HttpConfigParms.fnDereg = ((HTTPCONFIGPARAMS *)Value1)->fnDereg;
				gv_FlmSysData.HttpConfigParms.fnReqPath = ((HTTPCONFIGPARAMS *)Value1)->fnReqPath;
				gv_FlmSysData.HttpConfigParms.fnReqQuery = ((HTTPCONFIGPARAMS *)Value1)->fnReqQuery;
				gv_FlmSysData.HttpConfigParms.fnReqHdrValue = ((HTTPCONFIGPARAMS *)Value1)->fnReqHdrValue;
				gv_FlmSysData.HttpConfigParms.fnSetHdrValue = ((HTTPCONFIGPARAMS *)Value1)->fnSetHdrValue;
				gv_FlmSysData.HttpConfigParms.fnPrintf = ((HTTPCONFIGPARAMS *)Value1)->fnPrintf;
				gv_FlmSysData.HttpConfigParms.fnEmit = ((HTTPCONFIGPARAMS *)Value1)->fnEmit;
				gv_FlmSysData.HttpConfigParms.fnSetNoCache = ((HTTPCONFIGPARAMS *)Value1)->fnSetNoCache;
				gv_FlmSysData.HttpConfigParms.fnSendHeader = ((HTTPCONFIGPARAMS *)Value1)->fnSendHeader;
				gv_FlmSysData.HttpConfigParms.fnSetIOMode = ((HTTPCONFIGPARAMS *)Value1)->fnSetIOMode;
				gv_FlmSysData.HttpConfigParms.fnSendBuffer = ((HTTPCONFIGPARAMS *)Value1)->fnSendBuffer;
				gv_FlmSysData.HttpConfigParms.fnAcquireSession = ((HTTPCONFIGPARAMS *)Value1)->fnAcquireSession;
				gv_FlmSysData.HttpConfigParms.fnReleaseSession = ((HTTPCONFIGPARAMS *)Value1)->fnReleaseSession;
				gv_FlmSysData.HttpConfigParms.fnAcquireUser = ((HTTPCONFIGPARAMS *)Value1)->fnAcquireUser;	
				gv_FlmSysData.HttpConfigParms.fnReleaseUser = ((HTTPCONFIGPARAMS *)Value1)->fnReleaseUser;	
				gv_FlmSysData.HttpConfigParms.fnSetSessionValue = ((HTTPCONFIGPARAMS *)Value1)->fnSetSessionValue;
				gv_FlmSysData.HttpConfigParms.fnGetSessionValue = ((HTTPCONFIGPARAMS *)Value1)->fnGetSessionValue;
				gv_FlmSysData.HttpConfigParms.fnGetGblValue = ((HTTPCONFIGPARAMS *)Value1)->fnGetGblValue;
				gv_FlmSysData.HttpConfigParms.fnSetGblValue = ((HTTPCONFIGPARAMS *)Value1)->fnSetGblValue;
				gv_FlmSysData.HttpConfigParms.fnRecvBuffer = ((HTTPCONFIGPARAMS *)Value1)->fnRecvBuffer;
			}
			break;
	
		case FLM_REGISTER_HTTP_URL:
			// Value1: FLM_MODULE_HANDLE
			// Value2: Url string
			if ((Value1 == NULL) || (Value2 == NULL))
			{
				rc = RC_SET( FERR_INVALID_PARM);
				goto Exit;
			}
	
			rc = flmRegisterHttpCallback((FLM_MODULE_HANDLE)Value1, (char *)Value2);
			break;
	
		case FLM_DEREGISTER_HTTP_URL:
			rc = flmDeregisterHttpCallback();		
			break;
		
		case FLM_UNASSIGN_HTTP_SYMS:
			gv_FlmSysData.HttpConfigParms.fnReg = NULL;
			gv_FlmSysData.HttpConfigParms.fnDereg = NULL;
			gv_FlmSysData.HttpConfigParms.fnReqPath = NULL;
			gv_FlmSysData.HttpConfigParms.fnReqQuery = NULL;
			gv_FlmSysData.HttpConfigParms.fnReqHdrValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnSetHdrValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnPrintf = NULL;
			gv_FlmSysData.HttpConfigParms.fnEmit = NULL;
			gv_FlmSysData.HttpConfigParms.fnSetNoCache = NULL;
			gv_FlmSysData.HttpConfigParms.fnSendHeader = NULL;
			gv_FlmSysData.HttpConfigParms.fnSetIOMode = NULL;
			gv_FlmSysData.HttpConfigParms.fnSendBuffer = NULL;
			gv_FlmSysData.HttpConfigParms.fnAcquireSession = NULL;
			gv_FlmSysData.HttpConfigParms.fnReleaseSession = NULL;
			gv_FlmSysData.HttpConfigParms.fnAcquireUser = NULL;
			gv_FlmSysData.HttpConfigParms.fnReleaseUser = NULL;
			gv_FlmSysData.HttpConfigParms.fnSetSessionValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnGetSessionValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnGetGblValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnSetGblValue = NULL;
			gv_FlmSysData.HttpConfigParms.fnRecvBuffer = NULL;
			break;
	
		case FLM_KILL_DB_HANDLES:
		{
			FFILE *		pTmpFile;
	
			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( Value1)
			{
				// Look up the file using flmFindFile to see if we have the
				// file open.  May unlock and re-lock the global mutex.
	
				if( RC_OK( flmFindFile( (const char *)Value1, 
					(const char *)Value2, &pTmpFile)) && pTmpFile)
				{
					flmSetMustCloseFlags( pTmpFile, FERR_OK, TRUE);
				}
			}
			else
			{
				if( gv_FlmSysData.pFileHashTbl)
				{
					FLMUINT		uiLoop;
	
					for( uiLoop = 0; uiLoop < FILE_HASH_ENTRIES; uiLoop++)
					{
						pTmpFile = 
							(FFILE *)gv_FlmSysData.pFileHashTbl[ uiLoop].pFirstInBucket;
	
						while( pTmpFile)
						{
							flmSetMustCloseFlags( pTmpFile, FERR_OK, TRUE);
							pTmpFile = pTmpFile->pNext;
						}
					}
				}
			}
	
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
	
			// Kill all sessions
	
			if( gv_FlmSysData.pSessionMgr)
			{
				gv_FlmSysData.pSessionMgr->shutdownSessions();
			}
	
			break;
		}
		
		case FLM_QUERY_MAX:
			f_mutexLock( gv_FlmSysData.hQueryMutex);
			gv_FlmSysData.uiMaxQueries = (FLMUINT)Value1;
			gv_FlmSysData.bNeedToUnsetMaxQueries = FALSE;
			flmFreeSavedQueries( TRUE);
	
			// flmFreeSavedQueries unlocks the mutex.
	
			break;
	
		case FLM_MAX_DIRTY_CACHE:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			if (!Value1)
			{
				gv_FlmSysData.SCacheMgr.bAutoCalcMaxDirty = TRUE;
				gv_FlmSysData.SCacheMgr.uiMaxDirtyCache = 0;
				gv_FlmSysData.SCacheMgr.uiLowDirtyCache = 0;
			}
			else
			{
				gv_FlmSysData.SCacheMgr.bAutoCalcMaxDirty = FALSE;
				gv_FlmSysData.SCacheMgr.uiMaxDirtyCache = (FLMUINT)Value1;
	
				// Low threshhold must be no higher than maximum!
	
				if ((gv_FlmSysData.SCacheMgr.uiLowDirtyCache =
						(FLMUINT)Value2) > gv_FlmSysData.SCacheMgr.uiMaxDirtyCache)
				{
					gv_FlmSysData.SCacheMgr.uiLowDirtyCache =
						gv_FlmSysData.SCacheMgr.uiMaxDirtyCache;
				}
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_QUERY_STRATIFY_LIMITS:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			gv_FlmSysData.uiMaxStratifyIterations = (FLMUINT)Value1;
			gv_FlmSysData.uiMaxStratifyTime = (FLMUINT)Value2;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc : Gets configured shared attributes.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmGetConfig(
	eFlmConfigTypes	eConfigType,
	void *				Value1)
{
	RCODE		rc = FERR_OK;

	switch( eConfigType)
	{
		case FLM_CACHE_LIMIT:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
			*((FLMUINT *)Value1) = gv_FlmSysData.SCacheMgr.Usage.uiMaxBytes +
											 gv_FlmSysData.RCacheMgr.Usage.uiMaxBytes;
			f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_BLOCK_CACHE_PERCENTAGE:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			*((FLMUINT *)Value1) = gv_FlmSysData.uiBlockCachePercentage;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_SCACHE_DEBUG:
	#ifdef FLM_DEBUG
			*((FLMBOOL *)Value1) = gv_FlmSysData.SCacheMgr.bDebug;
	#else
			*((FLMBOOL *)Value1) = FALSE;
	#endif
			break;
	
		case FLM_OPEN_FILES:
		{
			*((FLMUINT *)Value1) = f_getOpenFileCount();
			break;
		}
	
		case FLM_OPEN_THRESHOLD:
			*((FLMUINT *)Value1) = gv_FlmSysData.pFileHdlCache->getOpenThreshold();
			break;
	
		case FLM_DIRECT_IO_STATE:
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			
			if( gv_FlmSysData.uiFileOpenFlags & FLM_IO_DIRECT)
			{
				*((FLMBOOL *)Value1) = TRUE;
			}
			else
			{
				*((FLMBOOL *)Value1) = FALSE;
			}
			
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}
		
		case FLM_TMPDIR:
			f_mutexLock( gv_FlmSysData.hShareMutex);
	
			if( !gv_FlmSysData.bTempDirSet )
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND );
			
				// Set the output to nulls on failure.
	
				*((char *)Value1) = 0;
			}
			else
			{
				f_strcpy( (char *)Value1, gv_FlmSysData.szTempDir);
			}
		
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_MAX_CP_INTERVAL:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiMaxCPInterval);
			break;
	
		case FLM_BLOB_EXT:
			f_strcpy( (char *)Value1, (const char *)gv_FlmSysData.ucBlobExt);
			break;
	
		case FLM_MAX_TRANS_SECS:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiMaxTransTime);
			break;
	
		case FLM_MAX_TRANS_INACTIVE_SECS:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiMaxTransInactiveTime);
			break;
	
		case FLM_CACHE_ADJUST_INTERVAL:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiCacheAdjustInterval);
			break;
	
		case FLM_CACHE_CLEANUP_INTERVAL:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiCacheCleanupInterval);
			break;
	
		case FLM_UNUSED_CLEANUP_INTERVAL:
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiUnusedCleanupInterval);
			break;
	
		case FLM_MAX_UNUSED_TIME:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			*((FLMUINT *)Value1) = 
				FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiMaxUnusedTime);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_CACHE_CHECK:
			*((FLMBOOL *)Value1) = gv_FlmSysData.bCheckCache;
			break;
	
		case FLM_QUERY_MAX:
			f_mutexLock( gv_FlmSysData.hQueryMutex);
			*((FLMUINT *)Value1) = gv_FlmSysData.uiMaxQueries;
			f_mutexUnlock( gv_FlmSysData.hQueryMutex);
			break;
	
		case FLM_MAX_DIRTY_CACHE:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			*((FLMUINT *)Value1) = gv_FlmSysData.SCacheMgr.uiMaxDirtyCache;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		case FLM_DYNA_CACHE_SUPPORTED:
			if( f_canGetMemoryInfo())
			{
				*((FLMBOOL *)Value1) = TRUE;
			}
			else
			{
				*((FLMBOOL *)Value1) = FALSE;
			}
			break;
	
		case FLM_QUERY_STRATIFY_LIMITS:
			f_mutexLock( gv_FlmSysData.hShareMutex);
			if (Value1)
			{
				*((FLMUINT *)Value1) = gv_FlmSysData.uiMaxStratifyIterations;
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
	
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

//Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI FlmGetThreadInfo(
	F_Pool *				pPool,
	F_THREAD_INFO **	ppThreadInfo,
	FLMUINT *			puiNumThreads,
	const char *		pszUrl)
{
	RCODE					rc = FERR_OK;
	CS_CONTEXT * 		pCSContext = NULL;

	if( pszUrl)
	{
		// flmGetCSConnection may return a NULL CS_CONTEXT if the
		// URL references a local resource.

		if( RC_BAD( rc = flmGetCSConnection( pszUrl, &pCSContext)))
		{
			goto Exit;
		}
	}

	if( pCSContext)
	{
		NODE *				pTree;
		FCL_WIRE				Wire;

		Wire.setContext( pCSContext);

		// Send a request get statistics

		if (RC_BAD( Wire.sendOp( FCS_OPCLASS_GLOBAL, 
			FCS_OP_GLOBAL_GET_THREAD_INFO)))
		{
			goto Exit;
		}

		if (RC_BAD( Wire.sendTerminate()))
		{
			goto Exit;
		}

		// Read the response.

		if (RC_BAD( Wire.read()))
		{
			goto Exit;
		}

		if (RC_BAD( Wire.getRCode()))
		{
			goto Exit;
		}

		if( RC_BAD( Wire.getHTD( pPool, &pTree)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = fcsExtractThreadInfo( pTree, pPool, 
			ppThreadInfo, puiNumThreads)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = gv_FlmSysData.pThreadMgr->getThreadInfo( pPool,
			ppThreadInfo, puiNumThreads)))
		{
			goto Exit;
		}
	}

Exit:

	if( pCSContext)
	{
		flmCloseCSConnection( &pCSContext);
	}

	return( rc);
}

/****************************************************************************
Desc: Returns the temporary directory (path) that is part of the share
		structure.  We have to pass in the FDB structure in order
		to know if we should lock a mutex in order to access the
		path structure.
****************************************************************************/
RCODE flmGetTmpDir(
	char *	pszOutputTmpDir)
{
	RCODE		rc = FERR_OK;

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Return an error if the temp directory has not been set.
	
	if( !gv_FlmSysData.bTempDirSet)
	{
		rc = RC_SET( FERR_IO_PATH_NOT_FOUND );
		
		// Set the output to nulls on failure.
		*pszOutputTmpDir = 0;
	}
	else
	{
		f_strcpy( pszOutputTmpDir, gv_FlmSysData.szTempDir);
	}
	
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	return( rc);
}

/****************************************************************************
Desc: This shuts down the background threads
Note:	This routine assumes that the global mutex is locked.  The mutex will
		be unlocked internally, but will always be locked on exit.
****************************************************************************/
FSTATIC void flmShutdownDbThreads(
	FFILE *		pFile)
{
	RCODE					rc = FERR_OK;
	F_BKGND_IX	*		pBackgroundIx;
	FDB *					pDb;
	IF_Thread *			pThread;
	FLMUINT				uiThreadId;
	FLMUINT				uiThreadCount;
	FLMBOOL				bMutexLocked = TRUE;

	// Shut down the tracker thread

	if( pFile->pMaintThrd)
	{
		pFile->pMaintThrd->setShutdownFlag();
		f_semSignal( pFile->hMaintSem);

		// Unlock the global mutex

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		pFile->pMaintThrd->stopThread();

		// Re-lock the mutex

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;

		pFile->pMaintThrd->Release();
		pFile->pMaintThrd = NULL;

		f_semDestroy( &pFile->hMaintSem);
	}

	// Signal all background threads to shut down that are
	// associated with this FFILE

	for( ;;)
	{
		uiThreadCount = 0;

		// Shut down all background indexing threads.

		uiThreadId = 0;
		for( ;;)
		{
			if( RC_BAD( rc = gv_FlmSysData.pThreadMgr->getNextGroupThread( 
				&pThread, gv_uiBackIxThrdGroup, &uiThreadId)))
			{
				if( rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;
					break;
				}
				else
				{
					flmAssert( 0);
				}
			}
			else
			{
				pBackgroundIx = (F_BKGND_IX *)pThread->getParm1();
				if( pBackgroundIx && pBackgroundIx->pFile == pFile)
				{
					// Set the thread's terminate flag.
					
					uiThreadCount++;
					pThread->setShutdownFlag();
				}

				pThread->Release();
				pThread = NULL;
			}
		}

		// Shut down all threads in the database thread group

		uiThreadId = 0;
		for( ;;)
		{
			if( RC_BAD( rc = gv_FlmSysData.pThreadMgr->getNextGroupThread( 
				&pThread, gv_uiDbThrdGrp, &uiThreadId)))
			{
				if( rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;
					break;
				}
				else
				{
					flmAssert( 0);
				}
			}
			else
			{
				pDb = (FDB *)pThread->getParm2();
				if (pDb && pDb->pFile == pFile)
				{

					// Set the thread's terminate flag.
					
					uiThreadCount++;
					pThread->setShutdownFlag();
				}

				pThread->Release();
				pThread = NULL;
			}
		}

		if( !uiThreadCount)
		{
			break;
		}

		// Unlock the global mutex

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Give the threads a chance to terminate

		f_sleep( 50);

		// Re-lock the mutex and see if any threads are still active

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;
	}

	// Re-lock the mutex

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:		This routine frees all of the structures associated with a file.
			Whoever called this routine has already determined that it is safe
			to do so.
Notes:	The global mutex is assumed to be locked when entering the
			routine.  It may be unlocked and re-locked before the routine
			exits, however.
****************************************************************************/
void flmFreeFile(
	FFILE *  		pFile)
{
	F_NOTIFY_LIST_ITEM *		pCloseNotifies;

	// See if another thread is in the process of closing
	// this FFILE.  It is possible for this to happen, since
	// the monitor thread may have selected this FFILE to be
	// closed because it has been unused for a period of time.
	// At the same time, a foreground thread could have called
	// FlmConfig to close all unused FFILEs.  Since flmFreeFile
	// may unlock and re-lock the mutex, there is a small window
	// of opportunity for both threads to try to free the same
	// FFILE.

	if( pFile->uiFlags & DBF_BEING_CLOSED)
	{
		return;
	}

	// Set the DBF_BEING_CLOSED flag

	pFile->uiFlags |= DBF_BEING_CLOSED;

	// Shut down all background threads before shutting down the CP thread.

	flmShutdownDbThreads( pFile);

	// At this point, the use count better be zero

	flmAssert( pFile->uiUseCount == 0);

	// Unlock the mutex

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	// Shutdown the checkpoint thread

	if( pFile->pCPThrd)
	{
		pFile->pCPThrd->stopThread();
		pFile->pCPThrd->Release();
		pFile->pCPThrd = NULL;
	}
	
	// Shutdown the monitor thread
	
	if( pFile->pMonitorThrd)
	{
		pFile->pMonitorThrd->stopThread();
		pFile->pMonitorThrd->Release();
		pFile->pMonitorThrd = NULL;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Unlink all of the DICTs that are connected to the file.

	while( pFile->pDictList)
	{
		flmUnlinkDict( pFile->pDictList);
	}

	// Take the file out of its name hash bucket, if any.

	if (pFile->uiBucket != 0xFFFF)
	{
		flmUnlinkFileFromBucket( pFile);
	}

	// Unlink the file from the not-used list

	flmUnlinkFileFromNUList( pFile);

	// Free the RFL data, if any.

	if (pFile->pRfl)
	{
		pFile->pRfl->Release();
		pFile->pRfl = NULL;
	}

	// We shouldn't have any open notifies at this point

	flmAssert( pFile->pOpenNotifies == NULL);

	// Save pCloseNotifies -- we will notify any waiters once the
	// FFILE has been freed.

	pCloseNotifies = pFile->pCloseNotifies;

	// Free any dictionary usage structures associated with the file.

	flmFreeDictList( &pFile->pDictList);

	// Free any shared cache associated with the file.

	ScaFreeFileCache( pFile);

	// Free any record cache associated with the file.

	flmRcaFreeFileRecs( pFile);

	// Release the lock objects.

	if( pFile->pWriteLockObj)
	{
		pFile->pWriteLockObj->Release();
		pFile->pWriteLockObj = NULL;
	}

	if( pFile->pFileLockObj)
	{
		pFile->pFileLockObj->Release();
		pFile->pFileLockObj = NULL;
	}

	// Close and delete the lock file.

	if( pFile->pLockFileHdl)
	{
		pFile->pLockFileHdl->Release();
		pFile->pLockFileHdl = NULL;
	}

	// Free the write buffer managers.

	if( pFile->pBufferMgr)
	{
		pFile->pBufferMgr->Release();
		pFile->pBufferMgr = NULL;
	}

	// Free the log header write buffer

	if( pFile->pucLogHdrIOBuf)
	{
		f_freeAlignedBuffer( (void **)&pFile->pucLogHdrIOBuf);
	}

	pFile->krefPool.poolFree();
	
	if( pFile->ppBlocksDone)
	{
		f_free( &pFile->ppBlocksDone);
		pFile->uiBlocksDoneArraySize = 0;
	}

	// Free the maintenance thread's semaphore

	if( pFile->hMaintSem != F_SEM_NULL)
	{
		f_semDestroy( &pFile->hMaintSem);
	}

	// Free the database wrapping key

	if( pFile->pDbWrappingKey)
	{
		pFile->pDbWrappingKey->Release();
		pFile->pDbWrappingKey = NULL;
	}
	
	// Free the password
	
	if( pFile->pszDbPassword)
	{
		f_free( &pFile->pszDbPassword);
	}
	
	// Free the FFILE

	f_free( &pFile);

	// Notify waiters that the FFILE is gone

	while( pCloseNotifies)
	{
		F_SEM		hSem;

		*(pCloseNotifies->pRc) = FERR_OK;
		hSem = pCloseNotifies->hSem;
		pCloseNotifies = pCloseNotifies->pNext;
		f_semSignal( hSem);
	}

	// Global mutex is still locked at this point
}

/****************************************************************************
Desc: This routine frees a registered event.
****************************************************************************/
FSTATIC void flmFreeEvent(
	FEVENT *			pEvent,
	F_MUTEX			hMutex,
	FEVENT **		ppEventListRV)
{
	f_mutexLock( hMutex);
	if (pEvent->pPrev)
	{
		pEvent->pPrev->pNext = pEvent->pNext;
	}
	else
	{
		*ppEventListRV = pEvent->pNext;
	}
	if (pEvent->pNext)
	{
		pEvent->pNext->pPrev = pEvent->pPrev;
	}
	f_mutexUnlock( hMutex);
	f_free( &pEvent);
}

/************************************************************************
Desc : Cleans up - assumes that the spin lock has already been
		 obtained.  This allows it to be called directly from
		 FlmStartup on error conditions.
************************************************************************/
FSTATIC void flmCleanup( void)
{
	FLMUINT		uiCnt;

	// NOTE: We are checking and decrementing a global variable here.
	// However, on platforms that properly support atomic exchange,
	// we are OK, because the caller has obtained a spin lock before
	// calling this routine, so we are guaranteed to be the only thread
	// executing this code at this point.  On platforms that don't
	// support atomic exchange, our spin lock will be less reliable for
	// really tight race conditions.  But in reality, nobody should be
	// calling FlmStartup and FlmShutdown in race conditions like that
	// anyway.  We are only doing the spin lock stuff to try and be
	// nice about it if they are.

	// This check allows FlmShutdown to be called before calling
	// FlmStartup, or even if FlmStartup fails.

	if (!gv_uiFlmSysStartupCount)
	{
		return;
	}

	// If we decrement the count and it doesn't go to zero, we are not
	// ready to do cleanup yet.

	if (--gv_uiFlmSysStartupCount > 0)
	{
		return;
	}

	// Deregister the Http Callback function if it has been registered.
	// Note: Usually, this is all handled at the SMI level (in
	// SMDIBHandle::exit).  This code is here for the cases where we're
	// part of Flint or some other utility that doesn't necessarily use
	// SMI...
	
	if (gv_FlmSysData.HttpConfigParms.fnDereg)
	{
		FlmConfig( FLM_DEREGISTER_HTTP_URL, NULL, NULL);
	}

	// Free any queries that have been saved in the query list.

	if (gv_FlmSysData.hQueryMutex != F_MUTEX_NULL)
	{

		// Setting uiMaxQueries to zero will cause flmFreeSavedQueries
		// to free the entire list.  Also, embedded queries will not be
		// added back into the list when uiMaxQueries is zero.

		gv_FlmSysData.uiMaxQueries = 0;
		flmFreeSavedQueries( FALSE);
	}

	// Shut down the monitor thread, if there is one.

	f_threadDestroy( &gv_FlmSysData.pMonitorThrd);

	// Shut down the session manager

	if( gv_FlmSysData.pSessionMgr)
	{
		gv_FlmSysData.pSessionMgr->Release();
		gv_FlmSysData.pSessionMgr = NULL;
	}

	// Destroy the session mutex

	if( gv_FlmSysData.hHttpSessionMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_FlmSysData.hHttpSessionMutex);
	}

	gv_FlmSysData.pMrnuFile = NULL;
	gv_FlmSysData.pLrnuFile = NULL;

	// Free all of the files and associated structures

	if (gv_FlmSysData.pFileHashTbl)
	{
		F_BUCKET *   pFileHashTbl;

		// flmFreeFile expects the global mutex to be locked
		// IMPORTANT NOTE: pFileHashTbl is ALWAYS allocated
		// AFTER the mutex is allocated, so we are guaranteed
		// to have a mutex if pFileHashTbl is non-NULL.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		for (uiCnt = 0, pFileHashTbl = gv_FlmSysData.pFileHashTbl;
			  uiCnt < FILE_HASH_ENTRIES;
			  uiCnt++, pFileHashTbl++)
		{
			FFILE *		pFile = (FFILE *)pFileHashTbl->pFirstInBucket;
			FFILE *		pTmpFile;

			while( pFile)
			{
				pTmpFile = pFile;
				pFile = pFile->pNext;
				flmFreeFile( pTmpFile);
			}
			pFileHashTbl->pFirstInBucket = NULL;
		}

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		f_free( &gv_FlmSysData.pFileHashTbl);
	}

	// Free the statistics.

	if (gv_FlmSysData.bStatsInitialized)
	{
		FlmFreeStats( &gv_FlmSysData.Stats);
		gv_FlmSysData.bStatsInitialized = FALSE;
	}

	// Free the resources of the shared cache manager.

	ScaExit();

	// Free the resources of the record cache manager.

	flmRcaExit();

	// Free the mutexes last of all.

	if (gv_FlmSysData.hQueryMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_FlmSysData.hQueryMutex);
	}

	if (gv_FlmSysData.hShareMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_FlmSysData.hShareMutex);
	}

	if (gv_FlmSysData.HttpConfigParms.hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_FlmSysData.HttpConfigParms.hMutex);
	}

	// Free up callbacks that have been registered for events.

	if (gv_FlmSysData.UpdateEvents.hMutex != F_MUTEX_NULL)
	{
		while (gv_FlmSysData.UpdateEvents.pEventCBList)
		{
			flmFreeEvent(
				gv_FlmSysData.UpdateEvents.pEventCBList,
				gv_FlmSysData.UpdateEvents.hMutex,
				&gv_FlmSysData.UpdateEvents.pEventCBList);
		}
		f_mutexDestroy( &gv_FlmSysData.UpdateEvents.hMutex);
	}

	if (gv_FlmSysData.LockEvents.hMutex != F_MUTEX_NULL)
	{
		while (gv_FlmSysData.LockEvents.pEventCBList)
		{
			flmFreeEvent(
				gv_FlmSysData.LockEvents.pEventCBList,
				gv_FlmSysData.LockEvents.hMutex,
				&gv_FlmSysData.LockEvents.pEventCBList);
		}
		f_mutexDestroy( &gv_FlmSysData.LockEvents.hMutex);
	}

	if (gv_FlmSysData.SizeEvents.hMutex != F_MUTEX_NULL)
	{
		while (gv_FlmSysData.SizeEvents.pEventCBList)
		{
			flmFreeEvent(
				gv_FlmSysData.SizeEvents.pEventCBList,
				gv_FlmSysData.SizeEvents.hMutex,
				&gv_FlmSysData.SizeEvents.pEventCBList);
		}
		f_mutexDestroy( &gv_FlmSysData.SizeEvents.hMutex);
	}
	
	// Free (release) FLAIM's File Shared File System Object.

	if( gv_FlmSysData.pFileSystem)
	{
		gv_FlmSysData.pFileSystem->Release();
		gv_FlmSysData.pFileSystem = NULL;
	}
	
#ifdef FLM_DBG_LOG
	flmDbgLogExit();
#endif

	// Release the logger (if any)

	if( gv_FlmSysData.pLogger)
	{
		gv_FlmSysData.pLogger->Release();
		gv_FlmSysData.pLogger = NULL;
	}

	// Release the thread manager

	if( gv_FlmSysData.pThreadMgr)
	{
		gv_FlmSysData.pThreadMgr->Release();
		gv_FlmSysData.pThreadMgr = NULL;
	}
	
	// Release the file handle cache
	
	if( gv_FlmSysData.pFileHdlCache)
	{
		gv_FlmSysData.pFileHdlCache->Release();
		gv_FlmSysData.pFileHdlCache = NULL;
	}
	
	// Release the slab manager
	
	if( gv_FlmSysData.pSlabManager)
	{
		gv_FlmSysData.pSlabManager->Release();
		gv_FlmSysData.pSlabManager = NULL;
	}

	// Shutdown NICI

#ifdef FLM_USE_NICI
	CCS_Shutdown();
#endif

	// Shut down the toolkit

	ftkShutdown();
}

/****************************************************************************
Desc:		Shuts down FLAIM.
Notes:	Allows itself to be called multiple times and even before FlmStartup
		 	is called, or even if FlmStartup fails.  Warning: May not handle
			race conditions very well on platforms that do not support atomic
			exchange.
****************************************************************************/
FLMEXP void FLMAPI FlmShutdown( void)
{
	flmLockSysData();
	flmCleanup();
	flmUnlockSysData();
}

/****************************************************************************
Desc: This routine links an FFILE structure to its name hash bucket.
		NOTE: This function assumes that the global mutex has been
		locked.
****************************************************************************/
RCODE flmLinkFileToBucket(
	FFILE *     pFile)
{
	RCODE			rc = FERR_OK;
	FFILE *		pTmpFile;
	F_BUCKET *	pBucket;
	FLMUINT		uiBucket;
	char			szDbPathStr[ F_PATH_MAX_SIZE];

	pBucket = gv_FlmSysData.pFileHashTbl;

	// Normalize the path to a string before hashing on it.

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathToStorageString( 
		pFile->pszDbPath, szDbPathStr)))
	{
		goto Exit;
	}

	uiBucket = f_strHashBucket( szDbPathStr, pBucket, FILE_HASH_ENTRIES);
	pBucket = &pBucket [uiBucket];
	
	if (pBucket->pFirstInBucket)
	{
		pTmpFile = (FFILE *)pBucket->pFirstInBucket;
		pTmpFile->pPrev = pFile;
	}
	
	pFile->uiBucket = uiBucket;
	pFile->pPrev = (FFILE *)NULL;
	pFile->pNext = (FFILE *)pBucket->pFirstInBucket;
	pBucket->pFirstInBucket = pFile;
	gv_FlmSysData.uiOpenFFiles++;
	
Exit:

	return( rc);
}


/****************************************************************************
Desc: This routine unlinks an FFILE structure from its name hash bucket.
		NOTE: This function assumes that the global mutex has been
		locked.
****************************************************************************/
FSTATIC void flmUnlinkFileFromBucket(
	FFILE *  	pFile)
{
	if (pFile->uiBucket != 0xFFFF)
	{
		if (pFile->pPrev)
		{
			pFile->pPrev->pNext = pFile->pNext;
		}
		else
		{
			gv_FlmSysData.pFileHashTbl [pFile->uiBucket].pFirstInBucket = pFile->pNext;
		}
		if (pFile->pNext)
		{
			pFile->pNext->pPrev = pFile->pPrev;
		}
		pFile->uiBucket = 0xFFFF;
	}
	
	flmAssert( gv_FlmSysData.uiOpenFFiles);
	gv_FlmSysData.uiOpenFFiles--;
}

/****************************************************************************
Desc: This routine links an FFILE structure to the unused list.
		NOTE: This function assumes that the global mutex has been
		locked.
****************************************************************************/
void flmLinkFileToNUList(
	FFILE *  pFile,
	FLMBOOL	bQuickTimeout)
{

	if( !bQuickTimeout)
	{
		pFile->pPrevNUFile = NULL;
		if ((pFile->pNextNUFile = gv_FlmSysData.pMrnuFile) == NULL)
		{
			gv_FlmSysData.pLrnuFile = pFile;
		}
		else
		{
			pFile->pNextNUFile->pPrevNUFile = pFile;
		}

		gv_FlmSysData.pMrnuFile = pFile;
		pFile->uiZeroUseCountTime = (FLMUINT)FLM_GET_TIMER();
	}
	else
	{
		pFile->pNextNUFile = NULL;
		if ((pFile->pPrevNUFile = gv_FlmSysData.pLrnuFile) == NULL)
		{
			gv_FlmSysData.pMrnuFile = pFile;
		}
		else
		{
			pFile->pPrevNUFile->pNextNUFile = pFile;
		}

		gv_FlmSysData.pLrnuFile = pFile;
		pFile->uiZeroUseCountTime = 0;
	}

	pFile->uiFlags |= DBF_IN_NU_LIST;

	if (pFile->pRfl)
	{
		pFile->pRfl->closeFile();
	}
}

/****************************************************************************
Desc: This routine unlinks an FFILE structure from the unused list.
		NOTE: This function assumes that the global mutex has been
		locked.
****************************************************************************/
void flmUnlinkFileFromNUList(
	FFILE *  pFile       /* File to be unlinked from unused list. */
	)
{
	if (pFile->uiFlags & DBF_IN_NU_LIST)
	{
		if (!pFile->pPrevNUFile)
		{
			gv_FlmSysData.pMrnuFile = pFile->pNextNUFile;
		}
		else
		{
			pFile->pPrevNUFile->pNextNUFile = pFile->pNextNUFile;
		}
		if (!pFile->pNextNUFile)
		{
			gv_FlmSysData.pLrnuFile = pFile->pPrevNUFile;
		}
		else
		{
			pFile->pNextNUFile->pPrevNUFile = pFile->pPrevNUFile;
		}
		pFile->pPrevNUFile = pFile->pNextNUFile = (FFILE *)NULL;
		pFile->uiFlags &= ~(DBF_IN_NU_LIST);
	}
}

/****************************************************************************
Desc: This routine checks unused structures to see if any have been unused
		longer than the maximum unused time.  If so, it frees them up.
Note: This routine assumes that the calling routine has locked the global
		mutex prior to calling this routine.  The mutex may be unlocked and
		re-locked by one of the called routines.
****************************************************************************/
void flmCheckNUStructs(
	FLMUINT		uiCurrTime)
{
	FFILE *     pFile;

	if (!uiCurrTime)
	{
		uiCurrTime = FLM_GET_TIMER();
	}

	// Look for unused FFILEs

	pFile = gv_FlmSysData.pLrnuFile;
	for (;;)
	{
		// Break out of the loop as soon as we discover an unused FFILE
		// structure which has not been unused the maximum number of seconds.

		if (pFile &&
			 (FLM_ELAPSED_TIME( uiCurrTime, pFile->uiZeroUseCountTime) >=
					gv_FlmSysData.uiMaxUnusedTime || !pFile->uiZeroUseCountTime))
		{

			// Remove the FFILE from memory.

			flmFreeFile( pFile);

			// flmFreeFile may have unlocked (and re-locked the global mutex,
			// so we need to start at the beginning of the list again.

			// Need to unlock the mutex here in case another thread is in
			// the process of closing the last FFILE.  If we hang on to
			// the mutex, it will never be able to get back in and finish
			// the job.

			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			f_yieldCPU();
			f_mutexLock( gv_FlmSysData.hShareMutex);
			pFile = gv_FlmSysData.pLrnuFile;
		}
		else
		{
			break;
		}
	}
	
	gv_FlmSysData.pFileHdlCache->closeUnusedFiles( 
		FLM_TIMER_UNITS_TO_SECS( gv_FlmSysData.uiMaxUnusedTime));
}

/****************************************************************************
Desc: This routine unlinks an FDICT structure from its FFILE structure and
		then frees the FDICT structure.
		NOTE: This routine assumes that the global mutex is locked.
****************************************************************************/
void flmUnlinkDict(
	FDICT * 		pDict)
{
	// Now unlink the local dictionary from its file - if it is connected
	// to one.

	if (pDict->pFile)
	{
		if (pDict->pPrev)
		{
			pDict->pPrev->pNext = pDict->pNext;
		}
		else
		{
			pDict->pFile->pDictList = pDict->pNext;
		}
		if (pDict->pNext)
		{
			pDict->pNext->pPrev = pDict->pPrev;
		}
	}

	/* Finally, free the local dictionary and its associated tables. */

	flmFreeDict( pDict);
}

/****************************************************************************
Desc: This routine links an FDB structure to an FFILE structure.
		NOTE: This routine assumes that the global mutex has been
		locked.
****************************************************************************/
RCODE flmLinkFdbToFile(
	FDB *       pDb,
	FFILE *		pFile)
{
	RCODE			rc = FERR_OK;

	// If the use count on the file used to be zero, unlink it from the
	// unused list.

	flmAssert( !pDb->pFile);
	
	pDb->pPrevForFile = NULL;
	if ((pDb->pNextForFile = pFile->pFirstDb) != NULL)
	{
		pFile->pFirstDb->pPrevForFile = pDb;
	}
	
	pFile->pFirstDb = pDb;
	pDb->pFile = pFile;
	
	if (++pFile->uiUseCount == 1)
	{
		flmUnlinkFileFromNUList( pFile);
	}
	
	if (pDb->uiFlags & FDB_INTERNAL_OPEN)
	{
		pFile->uiInternalUseCount++;
	}
	
	return( rc);
}

/****************************************************************************
Desc: This routine unlinks an FDB structure from its FFILE structure.
		NOTE: This routine assumes that the global mutex has been
		locked.
****************************************************************************/
void flmUnlinkFdbFromFile(
	FDB *    pDb)
{
	FFILE *	pFile;

	if ((pFile = pDb->pFile) != NULL)
	{

		// Unlink the FDB from the FFILE.

		if (pDb->pNextForFile)
		{
			pDb->pNextForFile->pPrevForFile = pDb->pPrevForFile;
		}
		if (pDb->pPrevForFile)
		{
			pDb->pPrevForFile->pNextForFile = pDb->pNextForFile;
		}
		else
		{
			pFile->pFirstDb = pDb->pNextForFile;
		}
		pDb->pNextForFile = pDb->pPrevForFile = NULL;
		pDb->pFile = NULL;

		// Decrement use counts in the FFILE.

		if (pDb->uiFlags & FDB_INTERNAL_OPEN)
		{
			flmAssert( pFile->uiInternalUseCount);
			pFile->uiInternalUseCount--;
		}

		// If the use count goes to zero on the file, put the file
		// into the unused list.

		flmAssert( pFile->uiUseCount);
		if (!(--pFile->uiUseCount))
		{
			// If the "must close" flag is set, it indicates that
			// the FFILE is being forced to close.  Put the FFILE in
			// the NU list, but specify that it should be quickly
			// timed-out.

			flmLinkFileToNUList( pFile, pFile->bMustClose);
		}
	}
}

/****************************************************************************
Desc: This routine functions as a thread.  It monitors open files and
		frees up files which have been closed longer than the maximum
		close time.
****************************************************************************/
RCODE FLMAPI flmSystemMonitor(
	IF_Thread *		pThread)
{
	FLMUINT			uiLastUnusedCleanupTime = 0;
	FLMUINT			uiLastRCacheCleanupTime = 0;
	FLMUINT			uiLastSCacheCleanupTime = 0;
	FLMUINT			uiCurrTime;
	FLMUINT			uiMaxLockTime;
	FLMUINT			uiLastCacheAdjustTime = 0;

	uiMaxLockTime = FLM_MILLI_TO_TIMER_UNITS( 100);

	for (;;)
	{

		// See if we should shut down

		if( pThread->getShutdownFlag())
		{
			break;
		}

		uiCurrTime = FLM_GET_TIMER();

		// Check the not used stuff and lock timeouts.

		if ( FLM_ELAPSED_TIME( uiCurrTime, uiLastUnusedCleanupTime) >=
					gv_FlmSysData.uiUnusedCleanupInterval ||
				(gv_FlmSysData.pLrnuFile && 
					!gv_FlmSysData.pLrnuFile->uiZeroUseCountTime))
		{
			// See if any unused structures have bee unused longer than the
			// maximum unused time.  Free them if they have.
			// May unlock and re-lock the global mutex.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			flmCheckNUStructs( 0);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Reset the timer

			uiCurrTime = uiLastUnusedCleanupTime = FLM_GET_TIMER();
		}

		// Check the adjusting cache limit

		if( f_canGetMemoryInfo())
		{
			if ((gv_FlmSysData.bDynamicCacheAdjust) &&
				 (FLM_ELAPSED_TIME( uiCurrTime, uiLastCacheAdjustTime) >=
						gv_FlmSysData.uiCacheAdjustInterval))
			{
				FLMUINT	uiCacheBytes;
	
				f_mutexLock( gv_FlmSysData.hShareMutex);
				f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	
				// Make sure the dynamic adjust flag is still set.
	
				if ((gv_FlmSysData.bDynamicCacheAdjust) &&
					 (FLM_ELAPSED_TIME( uiCurrTime, uiLastCacheAdjustTime) >=
						gv_FlmSysData.uiCacheAdjustInterval))
				{
					if( RC_OK( flmGetCacheBytes( 
						gv_FlmSysData.uiCacheAdjustPercent,
						gv_FlmSysData.uiCacheAdjustMin,
						gv_FlmSysData.uiCacheAdjustMax,
						gv_FlmSysData.uiCacheAdjustMinToLeave, TRUE,
						gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +
						gv_FlmSysData.RCacheMgr.Usage.uiTotalBytesAllocated, 
						&uiCacheBytes)))
					{
						flmSetCacheLimits( uiCacheBytes, FALSE, FALSE);
					}
				}
				
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
				uiCurrTime = uiLastCacheAdjustTime = FLM_GET_TIMER();
			}
		}

		// See if block cache should be cleaned up

		if ((gv_FlmSysData.uiCacheCleanupInterval) &&
			(FLM_ELAPSED_TIME( uiCurrTime, uiLastSCacheCleanupTime) >=
				gv_FlmSysData.uiCacheCleanupInterval))
		{
			ScaCleanupCache( uiMaxLockTime);
			uiCurrTime = uiLastSCacheCleanupTime = FLM_GET_TIMER();
		}

		// See if record cache should be cleaned up

		if( (gv_FlmSysData.uiCacheCleanupInterval) &&
			 (FLM_ELAPSED_TIME( uiCurrTime, uiLastRCacheCleanupTime) >=
					gv_FlmSysData.uiCacheCleanupInterval))
		{
			flmRcaCleanupCache( uiMaxLockTime, FALSE);
			uiCurrTime = uiLastRCacheCleanupTime = FLM_GET_TIMER();
		}

		// Cleanup old sessions

		if( gv_FlmSysData.pSessionMgr)
		{
			gv_FlmSysData.pSessionMgr->timeoutInactiveSessions( 
				MAX_SESSION_INACTIVE_SECS, FALSE);
		}

		pThread->sleep( 1000);
	}

	return( FERR_OK);
}

/****************************************************************************
Desc : Registers a callback function to receive events.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmRegisterForEvent(
	FEventCategory	eCategory,
	FEVENT_CB		fnEventCB,
	void *			pvAppData,
	HFEVENT *		phEventRV)
{
	RCODE		rc = FERR_OK;
	FEVENT *	pEvent;

	*phEventRV = HFEVENT_NULL;

	// Allocate an event structure

	if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( FEVENT)), &pEvent)))
	{
		goto Exit;
	}
	*phEventRV = (HFEVENT)pEvent;

	// Initialize the structure members and linkt to the
	// list of events off of the event category.

	pEvent->eCategory = eCategory;
	pEvent->fnEventCB = fnEventCB;
	pEvent->pvAppData = pvAppData;

	// Mutex should be locked to link into list.
	
	switch( eCategory)
	{
		case F_EVENT_UPDATES:
		{
			f_mutexLock( gv_FlmSysData.UpdateEvents.hMutex);
			if ((pEvent->pNext =
					gv_FlmSysData.UpdateEvents.pEventCBList) != NULL)
			{
				pEvent->pNext->pPrev = pEvent;
			}
			
			gv_FlmSysData.UpdateEvents.pEventCBList = pEvent;
			f_mutexUnlock( gv_FlmSysData.UpdateEvents.hMutex);
			break;
		}
		
		case F_EVENT_LOCKS:
		{
			f_mutexLock( gv_FlmSysData.LockEvents.hMutex);
			if ((pEvent->pNext =
					gv_FlmSysData.LockEvents.pEventCBList) != NULL)
			{
				pEvent->pNext->pPrev = pEvent;
			}
			
			gv_FlmSysData.LockEvents.pEventCBList = pEvent;
			f_mutexUnlock( gv_FlmSysData.LockEvents.hMutex);
			break;
		}
		
		case F_EVENT_SIZE:
		{
			f_mutexLock( gv_FlmSysData.SizeEvents.hMutex);
			if ((pEvent->pNext =
					gv_FlmSysData.SizeEvents.pEventCBList) != NULL)
			{
				pEvent->pNext->pPrev = pEvent;
			}
			
			gv_FlmSysData.SizeEvents.pEventCBList = pEvent;
			f_mutexUnlock( gv_FlmSysData.SizeEvents.hMutex);
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc : Deregisters a callback function that was registered to receive events.
****************************************************************************/
FLMEXP void FLMAPI FlmDeregisterForEvent(
	HFEVENT *	phEventRV)
{
	if (phEventRV && *phEventRV != HFEVENT_NULL)
	{
		FEVENT *	pEvent = (FEVENT *)(*phEventRV);

		switch( pEvent->eCategory)
		{
			case F_EVENT_UPDATES:
			{
				flmFreeEvent( pEvent,
						gv_FlmSysData.UpdateEvents.hMutex,
						&gv_FlmSysData.UpdateEvents.pEventCBList);
				break;
			}
			
			case F_EVENT_LOCKS:
			{
				flmFreeEvent( pEvent,
						gv_FlmSysData.LockEvents.hMutex,
						&gv_FlmSysData.LockEvents.pEventCBList);
				break;
			}
			
			case F_EVENT_SIZE:
			{
				flmFreeEvent( pEvent,
						gv_FlmSysData.SizeEvents.hMutex,
						&gv_FlmSysData.SizeEvents.pEventCBList);
				break;
			}
			
			default:
			{
				flmAssert( 0);
			}
		}
	
		*phEventRV = HFEVENT_NULL;
	}
}

/****************************************************************************
Desc: This routine does an event callback.  Note that the mutex is
		locked during the callback.
****************************************************************************/
void flmDoEventCallback(
	FEventCategory	eCategory,
	FEventType		eEventType,
	void *			pvEventData1,
	void *			pvEventData2)
{
	FEVENT *	pEvent;

	switch( eCategory)
	{
		case F_EVENT_UPDATES:
		{
			f_mutexLock( gv_FlmSysData.UpdateEvents.hMutex);
			pEvent = gv_FlmSysData.UpdateEvents.pEventCBList;
			while (pEvent)
			{
				(*pEvent->fnEventCB)( eEventType, pEvent->pvAppData,
												pvEventData1,
												pvEventData2);
				pEvent = pEvent->pNext;
			}
			f_mutexUnlock( gv_FlmSysData.UpdateEvents.hMutex);
			break;
		}
		
		case F_EVENT_LOCKS:
		{
			f_mutexLock( gv_FlmSysData.LockEvents.hMutex);
			pEvent = gv_FlmSysData.LockEvents.pEventCBList;
			while (pEvent)
			{
				(*pEvent->fnEventCB)( eEventType, pEvent->pvAppData,
												pvEventData1,
												pvEventData2);
				pEvent = pEvent->pNext;
			}
			f_mutexUnlock( gv_FlmSysData.LockEvents.hMutex);
			break;
		}
		
		case F_EVENT_SIZE:
		{
			f_mutexLock( gv_FlmSysData.SizeEvents.hMutex);
			pEvent = gv_FlmSysData.SizeEvents.pEventCBList;
			while (pEvent)
			{
				(*pEvent->fnEventCB)( eEventType, pEvent->pvAppData,
												pvEventData1,
												pvEventData2);
				pEvent = pEvent->pNext;
			}
			f_mutexUnlock( gv_FlmSysData.SizeEvents.hMutex);
			break;
		}
		
		default:
		{
			flmAssert( 0);
		}
	}
}

/****************************************************************************
Desc: This routine sets the "must close" flags on the FFILE and its FDBs
****************************************************************************/
void flmSetMustCloseFlags(
	FFILE *		pFile,
	RCODE			rcMustClose,
	FLMBOOL		bMutexLocked)
{
	FDB *				pTmpDb;

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}

	if( !pFile->bMustClose)
	{
		pFile->bMustClose = TRUE;
		pFile->rcMustClose = rcMustClose;
		pTmpDb = pFile->pFirstDb;
		while( pTmpDb)
		{
			pTmpDb->bMustClose = TRUE;
			pTmpDb = pTmpDb->pNextForFile;
		}

		// Log a message indicating why the "must close" flag has been
		// set.  Calling flmCheckFFileState with the bMustClose flag
		// already set to TRUE will cause a message to be logged.

		(void)flmCheckFFileState( pFile);
	}

	if( !bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_Session::F_Session()
{
	m_pSessionMgr = NULL;
	m_uiThreadId = 0;
	m_uiThreadLockCount = 0;
	m_hMutex = F_MUTEX_NULL;
	m_pNotifyList = NULL;
	m_pPrev = NULL;
	m_pNext = NULL;
	m_pNameTable = NULL;
	m_uiNameTableFFileId = 0;
	m_uiDictSeqNum = 0;
	m_uiLastUsed = FLM_GET_TIMER();
	m_uiNextToken = FLM_GET_TIMER();
	m_pDbTable = NULL;
	f_memset( m_ucKey, 0, sizeof( m_ucKey));
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_Session::~F_Session()
{
	flmAssert( !m_pPrev);
	flmAssert( !m_pNext);
	flmAssert( !getRefCount());
	flmAssert( !m_uiThreadLockCount);

	// Wake up any waiters

	signalLockWaiters( FERR_FAILURE, FALSE);

	// Free the session mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

	// Clean up any database objects

	if( m_pDbTable)
	{
		m_pDbTable->Release();
	}

	// Free the name table

	if( m_pNameTable)
	{
		m_pNameTable->Release();
	}
}

/****************************************************************************
Desc:	Signals the next thread waiting to acquire the lock on the session
		if rc == FERR_OK.  Otherwise, signals all waiting threads.
****************************************************************************/
void F_Session::signalLockWaiters(
	RCODE				rc,
	FLMBOOL			bMutexLocked)
{
	F_SEM				hSem;

	if( m_pNotifyList)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( m_hMutex);
		}

		while( m_pNotifyList)
		{
			*(m_pNotifyList->pRc) = rc;
			hSem = m_pNotifyList->hSem;
			m_pNotifyList = m_pNotifyList->pNext;
			f_semSignal( hSem);
			if( RC_OK( rc))
			{
				break;
			}
		}

		if( !bMutexLocked)
		{
			f_mutexUnlock( m_hMutex);
		}
	}
}

/****************************************************************************
Desc:	Configures a session for use.
****************************************************************************/
RCODE F_Session::setupSession(
	F_SessionMgr *			pSessionMgr)
{
	RCODE		rc = FERR_OK;

	flmAssert( m_hMutex == F_MUTEX_NULL);
	flmAssert( pSessionMgr);

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	if( (m_pDbTable = f_new F_HashTable) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbTable->setupHashTable( FALSE, 16, 0))) 
	{
		goto Exit;
	}

	m_pSessionMgr = pSessionMgr;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Adds a database handle to the session's list of handles.  Once added
		to the session, the calling code should not close the handle directly.
****************************************************************************/
RCODE F_Session::addDbHandle(
	HFDB				hDb,
	char *			pucKey)
{
	RCODE				rc = FERR_OK;
	F_SessionDb *	pSessionDb = NULL;
	const void *	pvKey;
	FLMUINT			uiKeyLen;

	if( (pSessionDb = f_new F_SessionDb) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pSessionDb->setupSessionDb( this, hDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDbTable->addObject( pSessionDb)))
	{
		goto Exit;
	}

	if( pucKey)
	{
		pvKey = pSessionDb->getKey();
		uiKeyLen = pSessionDb->getKeyLength();
		
		flmAssert( uiKeyLen == F_SESSION_DB_KEY_LEN);
		f_memcpy( pucKey, (FLMBYTE *)pvKey, uiKeyLen);
	}

	pSessionDb->Release();
	pSessionDb = NULL;

Exit:

	if( pSessionDb)
	{
		pSessionDb->m_hDb = HFDB_NULL;
		pSessionDb->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Closes the specific database identified by the passed-in key
****************************************************************************/
void F_Session::closeDb(
	const char *		pucKey)
{
	(void)m_pDbTable->getObject( (void *)pucKey, F_SESSION_DB_KEY_LEN, 
		NULL, TRUE);
}

/****************************************************************************
Desc:	Gets a specific database handle from the session given the passed-in
		key
****************************************************************************/
RCODE F_Session::getDbHandle(
	const char *		pucKey,
	HFDB *				phDb)
{
	RCODE					rc = FERR_OK;
	F_SessionDb *		pSessionDb = NULL;
	F_HashObject *		pObject;

	*phDb = HFDB_NULL;

	if( RC_BAD( rc = m_pDbTable->getObject( (void *)pucKey, F_SESSION_DB_KEY_LEN, 
		&pObject, FALSE)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			rc = RC_SET( FERR_BAD_HDL);
		}

		goto Exit;
	}

	pSessionDb = (F_SessionDb *)pObject;
	*phDb = pSessionDb->m_hDb;

Exit:

	if( pSessionDb)
	{
		pSessionDb->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the next database handle in the global list of handles managed
		by the session.
****************************************************************************/
RCODE F_Session::getNextDb(
	F_SessionDb **		ppSessionDb)
{
	F_HashObject *		pObject = *ppSessionDb;
	RCODE					rc = FERR_OK;

	if( RC_BAD( rc = m_pDbTable->getNextObjectInGlobal( &pObject)))
	{
		goto Exit;
	}

	*ppSessionDb = (F_SessionDb *)pObject;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Releases any session resources associated with the specified FFILE
****************************************************************************/
void F_Session::releaseFileResources(
	FFILE *	pFile)
{
	F_HashObject *		pObject;
	F_HashObject *		pNextObject;
	F_SessionDb *		pSessionDb;

	// Close all database handles with the specified FFILE

	pObject = NULL;
	if( RC_BAD( m_pDbTable->getNextObjectInGlobal( &pObject)))
	{
		goto Exit;
	}

	while( pObject)
	{
		if( (pNextObject = pObject->getNextInGlobal()) != NULL)
		{
			pNextObject->AddRef();
		}

		if( pObject->getObjectType() == HASH_DB_OBJ)
		{
			pSessionDb = (F_SessionDb *)pObject;

			if( ((FDB *)pSessionDb->getDbHandle())->pFile == pFile)
			{
				closeDb( (const char *)pSessionDb->getKey());
			}
		}
		pObject->Release();
		pObject = pNextObject;
	}

Exit:

	return;
}

/****************************************************************************
Desc:	Returns a pointer to a name table generated based on the supplied
		database handle
****************************************************************************/
RCODE F_Session::getNameTable(
	HFDB					hDb,
	F_NameTable **		ppNameTable)
{
	FLMUINT		uiSeq;
	FLMUINT		uiFFileId;
	RCODE			rc = FERR_OK;

	if( !m_pNameTable)
	{
		if( (m_pNameTable = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	if( RC_BAD( rc = FlmDbGetConfig( hDb, 
		FDB_GET_DICT_SEQ_NUM, (void *)&uiSeq)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbGetConfig( hDb,
		FDB_GET_FFILE_ID, (void *)&uiFFileId)))
	{
		goto Exit;
	}

	// If the database handle does not reference the same
	// database or dictionary as the last time the name table
	// was refreshed, we need to re-populate the table.

	if( uiSeq != m_uiDictSeqNum || 
		m_uiNameTableFFileId != uiFFileId)
	{
		if( RC_BAD( rc = m_pNameTable->setupFromDb( hDb)))
		{
			goto Exit;
		}

		m_uiDictSeqNum = uiSeq;
		m_uiNameTableFFileId = uiFFileId;
	}

	*ppNameTable = m_pNameTable;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns a pointer to a name table generated based on the supplied
		FFILE.
****************************************************************************/
RCODE F_Session::getNameTable(
	FFILE *				pFile,
	F_NameTable **		ppNameTable)
{
	FDB *			pDb = NULL;
	RCODE			rc = FERR_OK;

	// Temporarily open the database

	if( RC_BAD( rc = flmOpenFile( pFile, NULL, NULL, NULL, 0,
											TRUE, NULL, NULL,
											pFile->pszDbPassword, &pDb)))
	{
		goto Exit;
	}

	// Get the name table

	if( RC_BAD( rc = getNameTable( (HFDB)pDb, ppNameTable)))
	{
		goto Exit;
	}

Exit:

	if( pDb)
	{
		(void) FlmDbClose( (HFDB *)&pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns a unique token generated by the session manager.  Tokens are
		used to help make handles passed to clients unique across server
		executions
****************************************************************************/
FLMUINT F_Session::getNextToken( void)
{
	return( m_pSessionMgr->getNextToken());
}

/****************************************************************************
Desc:	Locks a session for use by a thread.  If bWait is TRUE, the thread
		will be put into a queue until the lock can be granted.  This routine
		can be called multiple times by the same thread, as long as
		unlockSession is called a corresponding number of times.
****************************************************************************/
RCODE F_Session::lockSession(
	FLMBOOL			bWait)
{
	RCODE		rc = FERR_OK;

	flmAssert( getRefCount());
	f_mutexLock( m_hMutex);

	if( m_uiThreadId && m_uiThreadId != f_threadId())
	{
		if( !bWait)
		{
			rc = RC_SET( FERR_TIMEOUT);
			goto Exit;
		}

		if( RC_BAD( rc = f_notifyWait( m_hMutex, F_SEM_NULL, NULL, 
			&m_pNotifyList)))
		{
			goto Exit;
		}
	}

	m_uiThreadId = f_threadId();
	m_uiThreadLockCount++;

Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}

/****************************************************************************
Desc:	Releases a thread's lock on a session.
****************************************************************************/
void F_Session::unlockSession( void)
{
	F_SEM				hSem;

	flmAssert( getRefCount());
	f_mutexLock( m_hMutex);
	
	if( m_uiThreadId != f_threadId())
	{
		flmAssert( 0);
	}
	else
	{
		if( --m_uiThreadLockCount == 0)
		{
			m_uiThreadId = 0;

			if( m_pNotifyList)
			{
				*(m_pNotifyList->pRc) = FERR_OK;
				hSem = m_pNotifyList->hSem;
				m_pNotifyList = m_pNotifyList->pNext;
				f_semSignal( hSem);
			}
		}

		m_uiLastUsed = FLM_GET_TIMER();
	}
	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:	Gets the session object's hash key
****************************************************************************/
const void * FLMAPI F_Session::getKey( void)
{
	return( m_ucKey);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_Session::getKeyLength( void)
{
	return( sizeof( m_ucKey));
}
	
/****************************************************************************
Desc:	Adds a reference to the session object.  The mutex is locked prior
		to incrementing the count since multiple threads are allowed to
		acquire a pointer to the object.  However, they shouldn't use the
		object w/o first locking it via a call to lockSession.
****************************************************************************/
FLMINT F_Session::AddRef( void)
{
	FLMINT		iRefCnt;

	f_mutexLock( m_hMutex);
	flmAssert( getRefCount());
	iRefCnt = ++m_refCnt;
	f_mutexUnlock( m_hMutex);

	return( iRefCnt);
}

/****************************************************************************
Desc:	Decrements the objects use count
****************************************************************************/
FLMINT F_Session::Release( void)
{
	FLMINT				iRefCnt;

	flmAssert( getRefCount());

	f_mutexLock( m_hMutex);
	if( (iRefCnt = --m_refCnt) == 0)
	{
		f_mutexUnlock( m_hMutex);
		delete this;
		return( iRefCnt);
	}
	
	f_mutexUnlock( m_hMutex);
	return( iRefCnt);
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_SessionMgr::~F_SessionMgr()
{
	if( m_pSessionTable)
	{
		shutdownSessions();
		m_pSessionTable->Release();
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:	Releases all resources (database handles, etc.) in all sessions
		if they are tied to the specified FFILE
****************************************************************************/
void F_SessionMgr::releaseFileResources(
	FFILE *		pFile)
{
	F_Session *			pSession;
	F_HashObject *		pObject;
	FLMBOOL				bMutexLocked = FALSE;

	if( m_hMutex == F_MUTEX_NULL)
	{
		goto Exit;
	}
	
	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	pObject = NULL;
	if( RC_BAD( m_pSessionTable->getNextObjectInGlobal( &pObject)))
	{
		goto Exit;
	}

	while( pObject)
	{
		flmAssert( pObject->getObjectType() == HASH_SESSION_OBJ);

		pSession = (F_Session *)pObject;
		if( (pObject = pObject->getNextInGlobal()) != NULL)
		{
			pObject->AddRef();
		}

		if( RC_OK( pSession->lockSession()))
		{
			pSession->releaseFileResources( pFile);
			pSession->unlockSession();
		}
		pSession->Release();
	}

Exit:
	if (bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return;
}

/****************************************************************************
Desc:	Shuts down all sessions being managed
****************************************************************************/
void F_SessionMgr::shutdownSessions()
{
	F_Session *			pSession;
	F_HashObject *		pObject;
	FLMBOOL				bMutexLocked = FALSE;

	if( m_hMutex == F_MUTEX_NULL)
	{
		goto Exit;
	}
	
	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	pObject = NULL;
	if( RC_BAD( m_pSessionTable->getNextObjectInGlobal( &pObject)))
	{
		goto Exit;
	}

	while( pObject)
	{
		flmAssert( pObject->getObjectType() == HASH_SESSION_OBJ);

		pSession = (F_Session *)pObject;
		if( (pObject = pObject->getNextInGlobal()) != NULL)
		{
			pObject->AddRef();
		}

		if( RC_OK( pSession->lockSession()))
		{
			m_pSessionTable->removeObject( pSession);
			pSession->signalLockWaiters( FERR_FAILURE, FALSE);
			pSession->unlockSession();
		}
		pSession->Release();
	}

Exit:
	if (bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return;
}

/****************************************************************************
Desc:	Configures the session manager prior to use
****************************************************************************/
RCODE F_SessionMgr::setupSessionMgr( void)
{
	RCODE		rc = FERR_OK;

	flmAssert( m_hMutex == F_MUTEX_NULL);

	// Create the mutex

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	// Create the session object table

	if( (m_pSessionTable = f_new F_HashTable) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pSessionTable->setupHashTable( FALSE, 128, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Gets a session given a session key
****************************************************************************/
RCODE F_SessionMgr::getSession(
	const char *	pszKey,
	F_Session **	ppSession)
{
	RCODE				rc = FERR_OK;
	F_Session *		pSession = NULL;
	F_HashObject *	pObject;
	FLMBOOL			bMutexLocked = FALSE;

	*ppSession = NULL;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( RC_BAD( rc = m_pSessionTable->getObject( (void *)pszKey,
		F_SESSION_KEY_LEN, &pObject, FALSE)))
	{
		goto Exit;
	}

	// NOTE: getObject() does an addRef for the caller.

	pSession = (F_Session *)pObject;

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

	if( RC_BAD( rc = pSession->lockSession()))
	{
		pSession->Release();
		goto Exit;
	}

	*ppSession = pSession;

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Unlocks and releases a session being used by a thread
****************************************************************************/
void F_SessionMgr::releaseSession(
	F_Session **		ppSession)
{
	(*ppSession)->unlockSession();
	(*ppSession)->Release();
	*ppSession = NULL;
}

/****************************************************************************
Desc:	Creates a new session
****************************************************************************/
RCODE F_SessionMgr::createSession(
	F_Session **	ppSession)
{
	F_Session *		pNewSession = NULL;
	FLMBOOL			bMutexLocked = FALSE;
	RCODE				rc = FERR_OK;

	if( (pNewSession = f_new F_Session) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pNewSession->setupSession( this)))
	{
		goto Exit;
	}

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	// Session ID

	f_sprintf( (char *)&pNewSession->m_ucKey[ 0], "%0*X", 
		(int)(sizeof( FLMUINT) * 2),
		(unsigned)m_uiNextId++);

	// Token

	f_sprintf( (char *)&pNewSession->m_ucKey[ sizeof( FLMUINT) * 2], "%0*X",
		(int)(sizeof( FLMUINT) * 2),
		(unsigned)m_uiNextToken++);

	pNewSession->m_ucKey[ sizeof( pNewSession->m_ucKey) - 1] = 0;

	// Add the session to the table

	if( RC_BAD( rc = m_pSessionTable->addObject( 
		pNewSession)))
	{
		goto Exit;
	}

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

	if( RC_BAD( rc = pNewSession->lockSession()))
	{
		pNewSession->Release();
		goto Exit;
	}

	*ppSession = pNewSession;
	pNewSession = NULL;

Exit:

	if( pNewSession)
	{
		pNewSession->Release();
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Kills any unused sessions that have been inactive for the specified
		number of seconds
****************************************************************************/
void F_SessionMgr::timeoutInactiveSessions(
	FLMUINT			uiInactiveSecs,
	FLMBOOL			bWaitForLocks)
{
	F_Session *		pSession;
	F_HashObject *	pObject;
	FLMUINT			uiCurrTime;
	FLMUINT			uiElapTime;
	FLMUINT			uiElapSecs;
	FLMBOOL			bMutexLocked = FALSE;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	pObject = NULL;
	if( RC_BAD( m_pSessionTable->getNextObjectInGlobal( &pObject)))
	{
		goto Exit;
	}

	while( pObject)
	{
		flmAssert( pObject->getObjectType() == HASH_SESSION_OBJ);

		pSession = (F_Session *)pObject;
		if( (pObject = pObject->getNextInGlobal()) != NULL)
		{
			pObject->AddRef();
		}

		if( RC_OK( pSession->lockSession( bWaitForLocks)))
		{
			uiCurrTime = FLM_GET_TIMER();
			uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, pSession->m_uiLastUsed);
			uiElapSecs = FLM_TIMER_UNITS_TO_SECS( uiElapTime);

			if( !uiInactiveSecs || uiElapSecs >= uiInactiveSecs)
			{
				m_pSessionTable->removeObject( pSession);
				pSession->signalLockWaiters( FERR_FAILURE, FALSE);
			}
			pSession->unlockSession();
		}
		pSession->Release();
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}
/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_SessionDb::F_SessionDb()
{
	m_hDb = HFDB_NULL;
	f_memset( m_ucKey, 0, sizeof( m_ucKey));
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_SessionDb::~F_SessionDb()
{
	if( m_hDb != HFDB_NULL)
	{
		FlmDbClose( &m_hDb);
	}
}

/****************************************************************************
Desc:	Configures a database object prior to being used
****************************************************************************/
RCODE F_SessionDb::setupSessionDb(
	F_Session *		pSession,
	HFDB				hDb)
{
	flmAssert( hDb != HFDB_NULL);
	flmAssert( m_hDb == HFDB_NULL);
	
	m_pSession = pSession;
	m_hDb = hDb;

	// Handle

	f_sprintf( (char *)&m_ucKey[ 0], "%0*X", 
		(int)(sizeof( FLMUINT) * 2),
		(unsigned)((FLMUINT)hDb));

	// Token

	f_sprintf( (char *)&m_ucKey[ sizeof( FLMUINT) * 2], "%0*X", 
		(int)(sizeof( FLMUINT) * 2),
		(unsigned)m_pSession->getNextToken());

	m_ucKey[ sizeof( m_ucKey) - 1] = 0;
	return( FERR_OK);
}

/****************************************************************************
Desc:	Returns the key and key length of a database object
****************************************************************************/
const void * FLMAPI F_SessionDb::getKey( void)
{
	return( m_ucKey);
}

/****************************************************************************
Desc:	Returns the key and key length of a database object
****************************************************************************/
FLMUINT FLMAPI F_SessionDb::getKeyLength( void)
{
	return( sizeof( m_ucKey));
}

/****************************************************************************
Desc:	Deletes (releases) and F_CCS objected referenced in the ITT table.
*****************************************************************************/
void flmDeleteCCSRefs(
	FDICT *		pDict)
{
	FLMUINT		uiLoop;
	F_CCS *		pCcs = NULL;
	ITT *			pItt;

	if (pDict && pDict->pIttTbl)
	{
		for ( pItt = pDict->pIttTbl, uiLoop = 0;
				uiLoop < pDict->uiIttCnt; pItt++, uiLoop++)
		{
			if (ITT_IS_ENCDEF(pItt))
			{
				pCcs = (F_CCS *)pItt->pvItem;
				pItt->pvItem = NULL;
				if (pCcs)
				{
					pCcs->Release();
					pCcs = NULL;
				}
			}
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileClient::F_SuperFileClient()
{
	m_pszCFileName = NULL;
	m_pszDataFileBaseName = NULL;
	m_uiExtOffset = 0;
	m_uiDataExtOffset = 0;
	m_uiDbVersion = 0;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileClient::~F_SuperFileClient()
{
	if( m_pszCFileName)
	{
		f_free( &m_pszCFileName);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SuperFileClient::setup(
	const char *	pszCFileName,
	const char *	pszDataDir,
	FLMUINT			uiDbVersion)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiNameLen;
	FLMUINT			uiDataNameLen;
	char				szDir[ F_PATH_MAX_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];

	if( !pszCFileName && *pszCFileName == 0)
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	uiNameLen = f_strlen( pszCFileName);
	if (pszDataDir && *pszDataDir)
	{
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
			pszCFileName, szDir, szBaseName)))
		{
			goto Exit;
		}
		f_strcpy( szDir, pszDataDir);
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
			szDir, szBaseName)))
		{
			goto Exit;
		}
		uiDataNameLen = f_strlen( szDir);

		if (RC_BAD( rc = f_alloc( (uiNameLen + 1) + (uiDataNameLen + 1),
									&m_pszCFileName)))
		{
			goto Exit;
		}

		f_memcpy( m_pszCFileName, pszCFileName, uiNameLen + 1);
		m_pszDataFileBaseName = m_pszCFileName + uiNameLen + 1;
		flmGetDbBasePath( m_pszDataFileBaseName, szDir, &m_uiDataExtOffset);
		m_uiExtOffset = uiNameLen - (uiDataNameLen - m_uiDataExtOffset);
	}
	else
	{
		if (RC_BAD( rc = f_alloc( (uiNameLen + 1) * 2, &m_pszCFileName)))
		{
			goto Exit;
		}

		f_memcpy( m_pszCFileName, pszCFileName, uiNameLen + 1);
		m_pszDataFileBaseName = m_pszCFileName + uiNameLen + 1;
		flmGetDbBasePath( m_pszDataFileBaseName, 
			m_pszCFileName, &m_uiDataExtOffset);
		m_uiExtOffset = m_uiDataExtOffset;
	}

	m_uiDbVersion = uiDbVersion;

Exit:

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_SuperFileClient::getFileNumber(
	FLMUINT					uiBlockAddr)
{
	return( FSGetFileNumber( uiBlockAddr));
}
		
/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_SuperFileClient::getFileOffset(
	FLMUINT					uiBlockAddr)
{
	return( FSGetFileOffset( uiBlockAddr));
}
		
/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_SuperFileClient::getBlockAddress(
	FLMUINT					uiFileNumber,
	FLMUINT					uiFileOffset)
{
	return( FSBlkAddress( uiFileNumber, uiFileOffset));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT64 FLMAPI F_SuperFileClient::getMaxFileSize( void)
{
	if( m_uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		return( gv_FlmSysData.uiMaxFileSize);
	}
	
	return( MAX_FILE_SIZE_VER40);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_SuperFileClient::getFilePath(
	FLMUINT			uiFileNumber,
	char *			pszPath)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiExtOffset;

	if (!uiFileNumber)
	{
		f_strcpy( pszPath, m_pszCFileName);
		goto Exit;
	}

	if ((m_uiDbVersion >= FLM_FILE_FORMAT_VER_4_3 &&
		  uiFileNumber <= MAX_DATA_FILE_NUM_VER43) ||
		 uiFileNumber <= MAX_DATA_FILE_NUM_VER40)
	{
		f_memcpy( pszPath, m_pszDataFileBaseName, m_uiDataExtOffset);
		uiExtOffset = m_uiDataExtOffset;
	}
	else
	{
		f_memcpy( pszPath, m_pszCFileName, m_uiExtOffset);
		uiExtOffset = m_uiExtOffset;
	}

	// Modify the file's extension.

	bldSuperFileExtension( m_uiDbVersion, 
		uiFileNumber, &pszPath[ uiExtOffset]);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Generates a file name given a super file number.
			Adds ".xx" to pFileExtension.  Use lower case characters.
Notes:	This is a base 24 alphanumeric value where 
			{ a, b, c, d, e, f, i, l, o, r, u, v } values are removed.
****************************************************************************/
void F_SuperFileClient::bldSuperFileExtension(
	FLMUINT		uiDbVersion,
	FLMUINT		uiFileNum,
	char *		pszFileExtension)
{
	FLMBYTE		ucLetter;
	
	flmAssert( uiDbVersion);

	if (uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
	{
		if (uiFileNum <= MAX_DATA_FILE_NUM_VER43 - 1536)
		{
			// No additional letter - File numbers 1 to 511
			// This is just like pre-4.3 numbering.
			
			ucLetter = 0;
		}
		else if (uiFileNum <= MAX_DATA_FILE_NUM_VER43 - 1024)
		{
			// File numbers 512 to 1023
			
			ucLetter = 'r';
		}
		else if (uiFileNum <= MAX_DATA_FILE_NUM_VER43 - 512)
		{
			// File numbers 1024 to 1535
			
			ucLetter = 's';
		}
		else if (uiFileNum <= MAX_DATA_FILE_NUM_VER43)
		{
			// File numbers 1536 to 2047
			
			ucLetter = 't';
		}
		else if (uiFileNum <= MAX_LOG_FILE_NUM_VER43 - 1536)
		{
			// File numbers 2048 to 2559
			
			ucLetter = 'v';
		}
		else if (uiFileNum <= MAX_LOG_FILE_NUM_VER43 - 1024)
		{
			// File numbers 2560 to 3071
			
			ucLetter = 'w';
		}
		else if (uiFileNum <= MAX_LOG_FILE_NUM_VER43 - 512)
		{
			// File numbers 3072 to 3583
			
			ucLetter = 'x';
		}
		else
		{
			flmAssert( uiFileNum <= MAX_LOG_FILE_NUM_VER43);

			// File numbers 3584 to 4095
			
			ucLetter = 'z';
		}
	}
	else
	{
		if (uiFileNum <= MAX_DATA_FILE_NUM_VER40)
		{
			// No additional letter - File numbers 1 to 511
			// This is just like pre-4.3 numbering.
			
			ucLetter = 0;
		}
		else
		{
			flmAssert( uiFileNum <= MAX_LOG_FILE_NUM_VER40);

			// File numbers 512 to 1023
			
			ucLetter = 'x';
		}
	}

	*pszFileExtension++ = '.';
	*pszFileExtension++ = f_getBase24DigitChar( (FLMBYTE)((uiFileNum & 511) / 24));
	*pszFileExtension++ = f_getBase24DigitChar( (FLMBYTE)((uiFileNum & 511) % 24));
	*pszFileExtension++ = ucLetter;
	*pszFileExtension   = 0;
}
