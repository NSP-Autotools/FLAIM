//------------------------------------------------------------------------------
// Desc:
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

#include "xflaim_DbSystem.h"
#include "xflaim_CREATEOPTS.h"
#include "xflaim_SlabUsage.h"
#include "xflaim_CacheUsage.h"
#include "xflaim_CacheInfo.h"
#include "xflaim_Stats.h"
#include "xflaim_DbStats.h"
#include "xflaim_RTransStats.h"
#include "xflaim_UTransStats.h"
#include "xflaim_LFileStats.h"
#include "xflaim_BlockIOStats.h"
#include "xflaim_DiskIOStat.h"
#include "xflaim_CountTimeStat.h"
#include "xflaim_LockStats.h"
#include "flaimsys.h"
#include "jniftk.h"
#include "jnirestore.h"
#include "jnistatus.h"

#define THIS_DBSYS() ((IF_DbSystem *)(FLMUINT)lThis)
	
// Field IDs for the CREATEOPTS class.

static jfieldID	fid_CREATEOPTS_iBlockSize = NULL;
static jfieldID	fid_CREATEOPTS_iVersionNum = NULL;
static jfieldID	fid_CREATEOPTS_iMinRflFileSize = NULL;
static jfieldID	fid_CREATEOPTS_iMaxRflFileSize = NULL;
static jfieldID	fid_CREATEOPTS_bKeepRflFiles = NULL;
static jfieldID	fid_CREATEOPTS_bLogAbortedTransToRfl = NULL;
static jfieldID	fid_CREATEOPTS_iDefaultLanguage = NULL;

// field IDs for the SlabUsage class.

static jfieldID	fid_SlabUsage_lSlabs;
static jfieldID	fid_SlabUsage_lSlabBytes;
static jfieldID	fid_SlabUsage_lAllocatedCells;
static jfieldID	fid_SlabUsage_lFreeCells;

// field IDs for the CacheUsage class.

static jfieldID	fid_CacheUsage_iByteCount = NULL;
static jfieldID	fid_CacheUsage_iCount = NULL;
static jfieldID	fid_CacheUsage_iOldVerCount = NULL;
static jfieldID	fid_CacheUsage_iOldVerBytes = NULL;
static jfieldID	fid_CacheUsage_iCacheHits = NULL;
static jfieldID	fid_CacheUsage_iCacheHitLooks = NULL;
static jfieldID	fid_CacheUsage_iCacheFaults = NULL;
static jfieldID	fid_CacheUsage_iCacheFaultLooks = NULL;
static jfieldID	fid_CacheUsage_slabUsage = NULL;

// field IDs for the CacheInfo class.

static jfieldID	fid_CacheInfo_iMaxBytes = NULL;
static jfieldID	fid_CacheInfo_iTotalBytesAllocated = NULL;
static jfieldID	fid_CacheInfo_bDynamicCacheAdjust = NULL;
static jfieldID	fid_CacheInfo_iCacheAdjustPercent = NULL;
static jfieldID	fid_CacheInfo_iCacheAdjustMin = NULL;
static jfieldID	fid_CacheInfo_iCacheAdjustMax = NULL;
static jfieldID	fid_CacheInfo_iCacheAdjustMinToLeave = NULL;
static jfieldID	fid_CacheInfo_iDirtyCount = NULL;
static jfieldID	fid_CacheInfo_iDirtyBytes = NULL;
static jfieldID	fid_CacheInfo_iNewCount = NULL;
static jfieldID	fid_CacheInfo_iNewBytes = NULL;
static jfieldID	fid_CacheInfo_iLogCount = NULL;
static jfieldID	fid_CacheInfo_iLogBytes = NULL;
static jfieldID	fid_CacheInfo_iFreeCount = NULL;
static jfieldID	fid_CacheInfo_iFreeBytes = NULL;
static jfieldID	fid_CacheInfo_iReplaceableCount = NULL;
static jfieldID	fid_CacheInfo_iReplaceableBytes = NULL;
static jfieldID	fid_CacheInfo_bPreallocatedCache = NULL;
static jfieldID	fid_CacheInfo_BlockCache = NULL;
static jfieldID	fid_CacheInfo_NodeCache = NULL;

// field IDs for the Stats class.

static jfieldID	fid_Stats_dbStats = NULL;
static jfieldID	fid_Stats_iStartTime = NULL;
static jfieldID	fid_Stats_iStopTime = NULL;

// field IDs for the DbStats class.

static jfieldID	fid_DbStats_sDbName = NULL;
static jfieldID	fid_DbStats_readTransStats = NULL;
static jfieldID	fid_DbStats_updateTransStats = NULL;
static jfieldID	fid_DbStats_lfileStats = NULL;
static jfieldID	fid_DbStats_lfhBlockStats = NULL;
static jfieldID	fid_DbStats_availBlockStats = NULL;
static jfieldID	fid_DbStats_dbHdrWrites = NULL;
static jfieldID	fid_DbStats_logBlockWrites = NULL;
static jfieldID	fid_DbStats_logBlockRestores = NULL;
static jfieldID	fid_DbStats_logBlockReads = NULL;
static jfieldID	fid_DbStats_iLogBlockChkErrs = NULL;
static jfieldID	fid_DbStats_iReadErrors = NULL;
static jfieldID	fid_DbStats_iWriteErrors = NULL;
static jfieldID	fid_DbStats_lockStats = NULL;
	
// field IDs for the RTransStats class

static jfieldID	fid_RTransStats_committedTrans = NULL;
static jfieldID	fid_RTransStats_abortedTrans = NULL;
	
// field IDs for the UTransStats class

static jfieldID	fid_UTransStats_committedTrans = NULL;
static jfieldID	fid_UTransStats_groupCompletes = NULL;
static jfieldID	fid_UTransStats_lGroupFinished = NULL;
static jfieldID	fid_UTransStats_abortedTrans = NULL;

// field IDs for the LFileStats class

static jfieldID	fid_LFileStats_rootBlockStats = NULL;
static jfieldID	fid_LFileStats_middleBlockStats = NULL;
static jfieldID	fid_LFileStats_leafBlockStats = NULL;
static jfieldID	fid_LFileStats_lBlockSplits = NULL;
static jfieldID	fid_LFileStats_lBlockCombines = NULL;
static jfieldID	fid_LFileStats_iLFileNum = NULL;
static jfieldID	fid_LFileStats_bIsIndex = NULL;
	
// field IDs for the BlockIOStats class

static jfieldID	fid_BlockIOStats_blockReads = NULL;
static jfieldID	fid_BlockIOStats_oldViewBlockReads = NULL;
static jfieldID	fid_BlockIOStats_iBlockChkErrs = NULL;
static jfieldID	fid_BlockIOStats_iOldViewBlockChkErrs = NULL;
static jfieldID	fid_BlockIOStats_iOldViewErrors = NULL;
static jfieldID	fid_BlockIOStats_blockWrites = NULL;
	
// field IDs for the DiskIOStat class

static jfieldID	fid_DiskIOStat_lCount = NULL;
static jfieldID	fid_DiskIOStat_lTotalBytes = NULL;
static jfieldID	fid_DiskIOStat_lElapMilli = NULL;

// field IDs for the LockStats class

static jfieldID	fid_LockStats_noLocks = NULL;
static jfieldID	fid_LockStats_waitingForLock = NULL;
static jfieldID	fid_LockStats_heldLock = NULL;

// field IDs for the CountTimeStat class

static jfieldID	fid_CountTimeStat_lCount = NULL;
static jfieldID	fid_CountTimeStat_lElapMilli = NULL;
	
FSTATIC void getCreateOpts(
	JNIEnv *					pEnv,
	jobject					createOpts,
	XFLM_CREATE_OPTS *	pCreateOpts);
	
FSTATIC jobject NewCountTimeStat(
	JNIEnv *					pEnv,
	F_COUNT_TIME_STAT *	pCountTimeStat,
	jclass					jCountTimeStatClass);
	
FSTATIC jobject NewRTransStats(
	JNIEnv *					pEnv,
	XFLM_RTRANS_STATS *	pRTransStats,
	jclass					jRTransStatsClass,
	jclass					jCountTimeStatClass);
	
FSTATIC jobject NewUTransStats(
	JNIEnv *					pEnv,
	XFLM_UTRANS_STATS *	pUTransStats,
	jclass					jUTransStatsClass,
	jclass					jCountTimeStatClass);
	
FSTATIC jobject NewLFileStats(
	JNIEnv *					pEnv,
	XFLM_LFILE_STATS *	pLFileStats,
	jclass					jLFileStatsClass,
	jclass					jBlockIOStatsClass,
	jclass					jDiskIOStatClass);
	
FSTATIC jobject NewDiskIOStat(
	JNIEnv *					pEnv,
	XFLM_DISKIO_STAT *	pDiskIOStat,
	jclass					jDiskIOStatClass);
	
FSTATIC jobject NewBlockIOStats(
	JNIEnv *					pEnv,
	XFLM_BLOCKIO_STATS *	pBlockIOStats,
	jclass					jBlockIOStatsClass,
	jclass					jDiskIOStatClass);
	
FSTATIC jobject NewLockStats(
	JNIEnv *			pEnv,
	F_LOCK_STATS *	pLockStats,
	jclass			jLockStatsClass,
	jclass			jCountTimeStatClass);
	
FSTATIC jobject NewDbStats(
	JNIEnv *				pEnv,
	XFLM_DB_STATS *	pDbStat,
	jclass				jDbStatsClass,
	jclass				jRTransStatsClass,
	jclass				jUTransStatsClass,
	jclass				jLFileStatsClass,
	jclass				jBlockIOStatsClass,
	jclass				jDiskIOStatClass,
	jclass				jCountTimeStatClass,
	jclass				jLockStatsClass);
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Stats_initIDs(
	JNIEnv *	pEnv,
	jclass	jStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_Stats_dbStats = pEnv->GetFieldID( jStatsClass,
								"dbStats", "[Lxflaim/DbStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_Stats_iStartTime = pEnv->GetFieldID( jStatsClass,
								"iStartTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_Stats_iStopTime = pEnv->GetFieldID( jStatsClass,
								"iStopTime", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jDbStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_DbStats_sDbName = pEnv->GetFieldID( jDbStatsClass,
								"sDbName", "Ljava/lang/String;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_readTransStats = pEnv->GetFieldID( jDbStatsClass,
								"readTransStats", "Lxflaim/RTransStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_updateTransStats = pEnv->GetFieldID( jDbStatsClass,
								"updateTransStats", "Lxflaim/UTransStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_lfileStats = pEnv->GetFieldID( jDbStatsClass,
								"lfileStats", "[Lxflaim/LFileStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_lfhBlockStats = pEnv->GetFieldID( jDbStatsClass,
								"lfhBlockStats", "Lxflaim/BlockIOStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_availBlockStats = pEnv->GetFieldID( jDbStatsClass,
								"availBlockStats", "Lxflaim/BlockIOStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_dbHdrWrites = pEnv->GetFieldID( jDbStatsClass,
								"dbHdrWrites", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_logBlockWrites = pEnv->GetFieldID( jDbStatsClass,
								"logBlockWrites", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_logBlockRestores = pEnv->GetFieldID( jDbStatsClass,
								"logBlockRestores", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_logBlockReads = pEnv->GetFieldID( jDbStatsClass,
								"logBlockReads", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_iLogBlockChkErrs = pEnv->GetFieldID( jDbStatsClass,
								"iLogBlockChkErrs", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_iReadErrors = pEnv->GetFieldID( jDbStatsClass,
								"iReadErrors", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_iWriteErrors = pEnv->GetFieldID( jDbStatsClass,
								"iWriteErrors", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DbStats_lockStats = pEnv->GetFieldID( jDbStatsClass,
								"lockStats", "Lxflaim/LockStats;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_RTransStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jRTransStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_RTransStats_committedTrans = pEnv->GetFieldID( jRTransStatsClass,
								"committedTrans", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_RTransStats_abortedTrans = pEnv->GetFieldID( jRTransStatsClass,
								"abortedTrans", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_UTransStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jUTransStatsClass)
{

	// Get the field IDs for the fields in the class.
	
	if ((fid_UTransStats_committedTrans = pEnv->GetFieldID( jUTransStatsClass,
								"committedTrans", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_UTransStats_groupCompletes = pEnv->GetFieldID( jUTransStatsClass,
								"groupCompletes", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_UTransStats_lGroupFinished = pEnv->GetFieldID( jUTransStatsClass,
								"lGroupFinished", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_UTransStats_abortedTrans = pEnv->GetFieldID( jUTransStatsClass,
								"abortedTrans", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_LFileStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jLFileStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_LFileStats_rootBlockStats = pEnv->GetFieldID( jLFileStatsClass,
								"rootBlockStats", "Lxflaim/BlockIOStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_middleBlockStats = pEnv->GetFieldID( jLFileStatsClass,
								"middleBlockStats", "Lxflaim/BlockIOStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_leafBlockStats = pEnv->GetFieldID( jLFileStatsClass,
								"leafBlockStats", "Lxflaim/BlockIOStats;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_lBlockSplits = pEnv->GetFieldID( jLFileStatsClass,
								"lBlockSplits", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_lBlockCombines = pEnv->GetFieldID( jLFileStatsClass,
								"lBlockCombines", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_iLFileNum = pEnv->GetFieldID( jLFileStatsClass,
								"iLFileNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LFileStats_bIsIndex = pEnv->GetFieldID( jLFileStatsClass,
								"bIsIndex", "Z")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_BlockIOStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jBlockIOStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_BlockIOStats_blockReads = pEnv->GetFieldID( jBlockIOStatsClass,
								"blockReads", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_BlockIOStats_oldViewBlockReads = pEnv->GetFieldID( jBlockIOStatsClass,
								"oldViewBlockReads", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_BlockIOStats_iBlockChkErrs = pEnv->GetFieldID( jBlockIOStatsClass,
								"iBlockChkErrs", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_BlockIOStats_iOldViewBlockChkErrs = pEnv->GetFieldID( jBlockIOStatsClass,
								"iOldViewBlockChkErrs", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_BlockIOStats_iOldViewErrors = pEnv->GetFieldID( jBlockIOStatsClass,
								"iOldViewErrors", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_BlockIOStats_blockWrites = pEnv->GetFieldID( jBlockIOStatsClass,
								"blockWrites", "Lxflaim/DiskIOStat;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DiskIOStat_initIDs(
	JNIEnv *	pEnv,
	jclass	jDiskIOStatClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_DiskIOStat_lCount = pEnv->GetFieldID( jDiskIOStatClass,
								"lCount", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DiskIOStat_lTotalBytes = pEnv->GetFieldID( jDiskIOStatClass,
								"lTotalBytes", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_DiskIOStat_lElapMilli = pEnv->GetFieldID( jDiskIOStatClass,
								"lElapMilli", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_LockStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jLockStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_LockStats_noLocks = pEnv->GetFieldID( jLockStatsClass,
								"noLocks", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LockStats_waitingForLock = pEnv->GetFieldID( jLockStatsClass,
								"waitingForLock", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LockStats_heldLock = pEnv->GetFieldID( jLockStatsClass,
								"heldLock", "Lxflaim/CountTimeStat;")) == NULL)
	{
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CountTimeStat_initIDs(
	JNIEnv *	pEnv,
	jclass	jCountTimeStatClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_CountTimeStat_lCount = pEnv->GetFieldID( jCountTimeStatClass,
								"lCount", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CountTimeStat_lElapMilli = pEnv->GetFieldID( jCountTimeStatClass,
								"lElapMilli", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_SlabUsage_initIDs(
	JNIEnv *	pEnv,
	jclass	jSlabUsageClass)
{

	// Get the field IDs for the fields in the class.
	
	if ((fid_SlabUsage_lSlabs = pEnv->GetFieldID( jSlabUsageClass,
								"lSlabs", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_SlabUsage_lSlabBytes = pEnv->GetFieldID( jSlabUsageClass,
								"lSlabBytes", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_SlabUsage_lAllocatedCells = pEnv->GetFieldID( jSlabUsageClass,
								"lAllocatedCells", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_SlabUsage_lFreeCells = pEnv->GetFieldID( jSlabUsageClass,
								"lFreeCells", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CacheUsage_initIDs(
	JNIEnv *	pEnv,
	jclass	jCacheUsageClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_CacheUsage_iByteCount = pEnv->GetFieldID( jCacheUsageClass,
								"iByteCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iCount = pEnv->GetFieldID( jCacheUsageClass,
								"iCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iOldVerCount = pEnv->GetFieldID( jCacheUsageClass,
								"iOldVerCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iOldVerBytes = pEnv->GetFieldID( jCacheUsageClass,
								"iOldVerBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iCacheHits = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheHits", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iCacheHitLooks = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheHitLooks", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iCacheFaults = pEnv->GetFieldID( jCacheUsageClass,
							"iCacheFaults", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_iCacheFaultLooks = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheFaultLooks", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheUsage_slabUsage = pEnv->GetFieldID( jCacheUsageClass,
								"slabUsage", "Lxflaim/SlabUsage;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CacheInfo_initIDs(
	JNIEnv *	pEnv,
	jclass	jCacheInfoClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_CacheInfo_iMaxBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iMaxBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iTotalBytesAllocated = pEnv->GetFieldID( jCacheInfoClass,
								"iTotalBytesAllocated", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_bDynamicCacheAdjust = pEnv->GetFieldID( jCacheInfoClass,
								"bDynamicCacheAdjust", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iCacheAdjustPercent = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustPercent", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iCacheAdjustMin = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustMin", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iCacheAdjustMax = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustMax", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iCacheAdjustMinToLeave = pEnv->GetFieldID( jCacheInfoClass,
							"iCacheAdjustMinToLeave", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iDirtyCount = pEnv->GetFieldID( jCacheInfoClass,
								"iDirtyCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iDirtyBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iDirtyBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iNewCount = pEnv->GetFieldID( jCacheInfoClass,
								"iNewCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iNewBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iNewBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iLogCount = pEnv->GetFieldID( jCacheInfoClass,
								"iLogCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iLogBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iLogBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iFreeCount = pEnv->GetFieldID( jCacheInfoClass,
								"iFreeCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iFreeBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iFreeBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iReplaceableCount = pEnv->GetFieldID( jCacheInfoClass,
								"iReplaceableCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_iReplaceableBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iReplaceableBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_bPreallocatedCache = pEnv->GetFieldID( jCacheInfoClass,
								"bPreallocatedCache", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_BlockCache = pEnv->GetFieldID( jCacheInfoClass,
								"BlockCache", "Lxflaim/CacheUsage;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CacheInfo_NodeCache = pEnv->GetFieldID( jCacheInfoClass,
								"NodeCache", "Lxflaim/CacheUsage;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CREATEOPTS_initIDs(
	JNIEnv *	pEnv,
	jclass	jCREATEOPTSClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_CREATEOPTS_iBlockSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iBlockSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_iVersionNum = pEnv->GetFieldID( jCREATEOPTSClass,
								"iVersionNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_iMinRflFileSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iMinRflFileSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_iMaxRflFileSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iMaxRflFileSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_bKeepRflFiles = pEnv->GetFieldID( jCREATEOPTSClass,
								"bKeepRflFiles", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_bLogAbortedTransToRfl = pEnv->GetFieldID( jCREATEOPTSClass,
							"bLogAbortedTransToRfl", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CREATEOPTS_iDefaultLanguage = pEnv->GetFieldID( jCREATEOPTSClass,
								"iDefaultLanguage", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1createDbSystem(
	JNIEnv *				pEnv,
	jobject)				// obj)
{
	IF_DbSystem * 		pDbSystem = NULL;
	
	if( RC_BAD( FlmAllocDbSystem( &pDbSystem)))
	{
		ThrowError( NE_XFLM_MEM, pEnv);
	}
	
	return( (jlong)((FLMUINT)pDbSystem));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1release(
	JNIEnv *,				// pEnv,
	jobject,					// obj,
	jlong						lThis)
{
	IF_DbSystem *	pDbSystem = THIS_DBSYS();
	
	if (pDbSystem)
	{
		pDbSystem->Release();
	}
}

/****************************************************************************
Desc:	Get create options from the CREATEOPTS Java object.
****************************************************************************/
FSTATIC void getCreateOpts(
	JNIEnv *					pEnv,
	jobject					createOpts,
	XFLM_CREATE_OPTS *	pCreateOpts)
{
	pCreateOpts->ui32BlockSize = (FLMUINT32)pEnv->GetIntField( createOpts,
			fid_CREATEOPTS_iBlockSize); 
	pCreateOpts->ui32VersionNum = (FLMUINT32)pEnv->GetIntField( createOpts,
			fid_CREATEOPTS_iVersionNum);
	pCreateOpts->ui32MinRflFileSize = (FLMUINT32)pEnv->GetIntField( createOpts,
			fid_CREATEOPTS_iMinRflFileSize); 
	pCreateOpts->ui32MaxRflFileSize = (FLMUINT32)pEnv->GetIntField( createOpts,
			fid_CREATEOPTS_iMaxRflFileSize); 
	pCreateOpts->bKeepRflFiles = (FLMBOOL)(pEnv->GetBooleanField( createOpts,
			fid_CREATEOPTS_bKeepRflFiles) ? TRUE : FALSE); 
	pCreateOpts->bLogAbortedTransToRfl = (FLMBOOL)(pEnv->GetBooleanField( createOpts,
			fid_CREATEOPTS_bLogAbortedTransToRfl) ? TRUE : FALSE); 
	pCreateOpts->ui32DefaultLanguage = (FLMUINT32)pEnv->GetIntField( createOpts,
			fid_CREATEOPTS_iDefaultLanguage);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbCreate(
	JNIEnv *					pEnv,
	jobject,					// obj,
	jlong						lThis,
	jstring					sDbPath,
	jstring					sDataDir,
	jstring					sRflDir,
	jstring					sDictFileName,
	jstring					sDictBuf,
	jobject					createOpts)
{
	RCODE						rc = NE_XFLM_OK;
	F_Db *					pDb = NULL;
	XFLM_CREATE_OPTS		Opts;
	XFLM_CREATE_OPTS *	pOpts;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucDictFileName [F_PATH_MAX_SIZE];
	F_DynaBuf				dictFileNameBuf( ucDictFileName, sizeof( ucDictFileName));
	FLMBYTE					ucDictBuf [100];
	F_DynaBuf				dictBufBuf( ucDictBuf, sizeof( ucDictBuf));
	
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictFileName, &dictFileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictBuf, &dictBufBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (!createOpts)
	{
		pOpts = NULL;
	}
	else
	{
		getCreateOpts( pEnv, createOpts, &Opts);
		pOpts = &Opts;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCreate(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							dictFileNameBuf.getDataLength() > 1
							? (const char *)dictFileNameBuf.getBufferPtr()
							: (const char *)NULL,
							dictBufBuf.getDataLength() > 1
							? (const char *)dictBufBuf.getBufferPtr()
							: (const char *)NULL,
							pOpts, (IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

 Exit:

  	return( (jlong)((FLMUINT)pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbOpen(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jboolean			bAllowLimited)
{
	RCODE 			rc = NE_XFLM_OK;
	F_Db * 			pDb = NULL;
	FLMBYTE			ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf		dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE			ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf		dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE			ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf		rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE			ucPassword [100];
	F_DynaBuf		passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
 	if (RC_BAD( rc = THIS_DBSYS()->dbOpen(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							passwordBuf.getDataLength() > 1
							? (const char *)passwordBuf.getBufferPtr()
							: (const char *)NULL,
							bAllowLimited ? TRUE : FALSE,
							(IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
 	
Exit:

	return( (jlong)(FLMUINT)pDb);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRemove(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jboolean			bRemove)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf		dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE			ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf		dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE			ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf		rflDirBuf( ucRflDir, sizeof( ucRflDir));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbRemove(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							bRemove ? TRUE : FALSE)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRestore(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sBackupPath,
	jstring			sPassword,
	jobject			RestoreClient,
	jobject			RestoreStatus)
{
	RCODE						rc = NE_XFLM_OK;
	JavaVM *					pJvm = NULL;
	JNIRestoreClient *	pRestoreClient = NULL;
	JNIRestoreStatus *	pRestoreStatus = NULL;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucBackupPath [F_PATH_MAX_SIZE];
	F_DynaBuf				backupPathBuf( ucBackupPath, sizeof( ucBackupPath));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucPassword [100];
	F_DynaBuf				passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBackupPath, &backupPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	pEnv->GetJavaVM( &pJvm);

	flmAssert( RestoreClient);
	if ((pRestoreClient = f_new JNIRestoreClient( RestoreClient, pJvm)) == NULL)
	{
		ThrowError( NE_XFLM_MEM, pEnv);
		goto Exit;
	}
	
	if (RestoreStatus != NULL)
	{
		if ((pRestoreStatus = f_new JNIRestoreStatus( RestoreStatus, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}		
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbRestore(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		backupPathBuf.getDataLength() > 1
		? (const char *)backupPathBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		passwordBuf.getDataLength() > 1
		? (const char *)passwordBuf.getBufferPtr()
		: (const char *)NULL,
		pRestoreClient, pRestoreStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pRestoreClient)
	{
		pRestoreClient->Release();
	}
	
	if (pRestoreStatus)
	{
		pRestoreStatus->Release();
	}
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbCheck(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jint				iFlags,
	jobject			checkStatus)
{
	RCODE					rc = NE_XFLM_OK;
	JNICheckStatus *	pStatus = NULL;
	F_DbInfo *			pDbInfo = NULL;
	FLMBYTE				ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf			dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE				ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE				ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE				ucPassword [100];
	F_DynaBuf			passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (checkStatus != NULL)
	{
		JavaVM *		pJvm = NULL;
		
		pEnv->GetJavaVM( &pJvm);
		
		if ((pStatus = f_new JNICheckStatus( checkStatus, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}		
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCheck(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		passwordBuf.getDataLength() > 1
		? (const char *)passwordBuf.getBufferPtr()
		: (const char *)NULL,
		(FLMUINT)iFlags, (IF_DbInfo **)&pDbInfo, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	
	return( (jlong)((FLMUINT)pDbInfo));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbCopy(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sSrcDbName,
	jstring			sSrcDataDir,
	jstring			sSrcRflDir,
	jstring			sDestDbName,
	jstring			sDestDataDir,
	jstring			sDestRflDir,
	jobject			copyStatus)
{
	RCODE					rc = NE_XFLM_OK;
	JavaVM *				pJvm;
	JNICopyStatus *	pStatus = NULL;
	FLMBYTE				ucSrcDbName [F_PATH_MAX_SIZE];
	F_DynaBuf			srcDbNameBuf( ucSrcDbName, sizeof( ucSrcDbName));
	FLMBYTE				ucSrcDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			srcDataDirBuf( ucSrcDataDir, sizeof( ucSrcDataDir));
	FLMBYTE				ucSrcRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			srcRflDirBuf( ucSrcRflDir, sizeof( ucSrcRflDir));
	FLMBYTE				ucDestDbName [F_PATH_MAX_SIZE];
	F_DynaBuf			destDbNameBuf( ucDestDbName, sizeof( ucDestDbName));
	FLMBYTE				ucDestDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			destDataDirBuf( ucDestDataDir, sizeof( ucDestDataDir));
	FLMBYTE				ucDestRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			destRflDirBuf( ucDestRflDir, sizeof( ucDestRflDir));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sSrcDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcDbName, &srcDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcDataDir, &srcDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcRflDir, &srcRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sDestDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDbName, &destDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDataDir, &destDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestRflDir, &destRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (copyStatus)
	{
		pEnv->GetJavaVM( &pJvm);
		if ( (pStatus = f_new JNICopyStatus( copyStatus, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCopy(
		(const char *)srcDbNameBuf.getBufferPtr(),
		srcDataDirBuf.getDataLength() > 1
		? (const char *)srcDataDirBuf.getBufferPtr()
		: (const char *)NULL,
		srcRflDirBuf.getDataLength() > 1
		? (const char *)srcRflDirBuf.getBufferPtr()
		: (const char *)NULL,
		(const char *)destDbNameBuf.getBufferPtr(),
		destDataDirBuf.getDataLength() > 1
		? (const char *)destDataDirBuf.getBufferPtr()
		: (const char *)NULL,
		destRflDirBuf.getDataLength() > 1
		? (const char *)destRflDirBuf.getBufferPtr()
		: (const char *)NULL,
		pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}

Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRename(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sNewDbName,
	jboolean			bOverwriteDestOk,
	jobject			renameStatus)
{
	RCODE						rc = NE_XFLM_OK;
	JavaVM *					pJvm;
	JNIRenameStatus *		pStatus = NULL;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucNewDbName [F_PATH_MAX_SIZE];
	F_DynaBuf				newDbNameBuf( ucNewDbName, sizeof( ucNewDbName));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sNewDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sNewDbName, &newDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (renameStatus != NULL)
	{
		pEnv->GetJavaVM( &pJvm);
		if ((pStatus = f_new JNIRenameStatus( renameStatus, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;	
		}
	}

	if (RC_BAD(rc = THIS_DBSYS()->dbRename(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		(const char *)newDbNameBuf.getBufferPtr(),
		bOverwriteDestOk ? TRUE : FALSE, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	return;
}

/****************************************************************************
Desc: Rebuild status callback
****************************************************************************/
class JavaDbRebuildStatus : public IF_DbRebuildStatus
{
public:

	JavaDbRebuildStatus(
		JNIEnv *		pEnv,
		jobject		jDbRebuildStatusObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jDbRebuildStatusObject = pEnv->NewGlobalRef( jDbRebuildStatusObject);
		m_jReportRebuildMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDbRebuildStatusObject),
													"reportRebuild",
													"(IZJJJJJ)I");
		m_jReportRebuildErrMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDbRebuildStatusObject),
													"reportRebuildErr",
													"(IIIIIIIIJ)I");
	}
	
	virtual ~JavaDbRebuildStatus()
	{
		if (m_jDbRebuildStatusObject)
		{
			m_pEnv->DeleteGlobalRef( m_jDbRebuildStatusObject);
		}
	}
			
	RCODE XFLAPI reportRebuild(
		XFLM_REBUILD_INFO *	pRebuild)
	{
		
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setIndexingStatusObject method was called.  It will not
		// work to set the index status object in one thread, but then do
		// the index operation in another thread.
		
		return( (RCODE)m_pEnv->CallIntMethod( m_jDbRebuildStatusObject,
									m_jReportRebuildMethodId,
									(jint)pRebuild->i32DoingFlag,
									(jboolean)(pRebuild->bStartFlag ? JNI_TRUE : JNI_FALSE),
									(jlong)pRebuild->ui64FileSize,
									(jlong)pRebuild->ui64BytesExamined,
									(jlong)pRebuild->ui64TotNodes,
									(jlong)pRebuild->ui64NodesRecov,
									(jlong)pRebuild->ui64DiscardedDocs));
	}
	
	RCODE XFLAPI reportRebuildErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo)
	{
		return( (RCODE)m_pEnv->CallIntMethod( m_jDbRebuildStatusObject,
									m_jReportRebuildErrMethodId,
									(jint)pCorruptInfo->i32ErrCode,
									(jint)pCorruptInfo->ui32ErrLocale,
									(jint)pCorruptInfo->ui32ErrLfNumber,
									(jint)pCorruptInfo->ui32ErrLfType,
									(jint)pCorruptInfo->ui32ErrBTreeLevel,
									(jint)pCorruptInfo->ui32ErrBlkAddress,
									(jint)pCorruptInfo->ui32ErrParentBlkAddress,
									(jint)pCorruptInfo->ui32ErrElmOffset,
									(jlong)pCorruptInfo->ui64ErrNodeId));
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jDbRebuildStatusObject;
	jmethodID	m_jReportRebuildMethodId;
	jmethodID	m_jReportRebuildErrMethodId;
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRebuild(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jstring			sSourceDbPath,
	jstring			sSourceDataDir,
	jstring			sDestDbPath,
	jstring			sDestDataDir,
	jstring			sDestRflDir,
	jstring			sDictPath,
	jstring			sPassword,
	jobject			createOpts,
	jobject			jDbRebuildStatusObj)
{
	RCODE							rc = NE_XFLM_OK;
	JavaDbRebuildStatus *	pDbRebuildStatusObj = NULL;
	IF_DbSystem *				pDbSystem = THIS_DBSYS();
	XFLM_CREATE_OPTS			createOptions;
	XFLM_CREATE_OPTS *		pCreateOptions;
	FLMUINT64					ui64TotNodes;
	FLMUINT64					ui64NodesRecov;
	FLMUINT64					ui64DiscardedDocs;
	FLMBYTE						ucSourceDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf					sourceDbPathBuf( ucSourceDbPath, sizeof( ucSourceDbPath));
	FLMBYTE						ucSourceDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf					sourceDataDirBuf( ucSourceDataDir, sizeof( ucSourceDataDir));
	FLMBYTE						ucDestDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf					destDbPathBuf( ucDestDbPath, sizeof( ucDestDbPath));
	FLMBYTE						ucDestDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf					destDataDirBuf( ucDestDataDir, sizeof( ucDestDataDir));
	FLMBYTE						ucDestRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf					destRflDirBuf( ucDestRflDir, sizeof( ucDestRflDir));
	FLMBYTE						ucDictPath [F_PATH_MAX_SIZE];
	F_DynaBuf					dictPathBuf( ucDictPath, sizeof( ucDictPath));
	FLMBYTE						ucPassword [100];
	F_DynaBuf					passwordBuf( ucPassword, sizeof( ucPassword));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sSourceDbPath, &sourceDbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSourceDataDir, &sourceDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDbPath, &destDbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDataDir, &destDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestRflDir, &destRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictPath, &dictPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Setup callback object, if one was passed in
	
	if (jDbRebuildStatusObj)
	{
		if ((pDbRebuildStatusObj = f_new JavaDbRebuildStatus( pEnv,
													jDbRebuildStatusObj)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	// Set up the create options.
	
	if (!createOpts)
	{
		pCreateOptions = NULL;
	}
	else
	{
		getCreateOpts( pEnv, createOpts, &createOptions);
		pCreateOptions = &createOptions;
	}
	
	// Call the rebuild function.
	
	if (RC_BAD( rc = pDbSystem->dbRebuild(
				(const char *)sourceDbPathBuf.getBufferPtr(),
				sourceDataDirBuf.getDataLength() > 1
				? (const char *)sourceDataDirBuf.getBufferPtr()
				: (const char *)NULL,
				(const char *)destDbPathBuf.getBufferPtr(),
				destDataDirBuf.getDataLength() > 1
				? (const char *)destDataDirBuf.getBufferPtr()
				: (const char *)NULL,
				destRflDirBuf.getDataLength() > 1
				? (const char *)destRflDirBuf.getBufferPtr()
				: (const char *)NULL,
				dictPathBuf.getDataLength() > 1
				? (const char *)dictPathBuf.getBufferPtr()
				: (const char *)NULL,
				passwordBuf.getDataLength() > 1
				? (const char *)passwordBuf.getBufferPtr()
				: (const char *)NULL,
				pCreateOptions,
				&ui64TotNodes,
				&ui64NodesRecov,
				&ui64DiscardedDocs,
				pDbRebuildStatusObj)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pDbRebuildStatusObj)
	{
		pDbRebuildStatusObj->Release();
	}

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong,			// lThis,
	jstring			sBuffer)
{
	RCODE					rc = NE_XFLM_OK;
	const char *		pszBuffer = NULL;
	FLMUINT				uiStrCharCount;
	F_BufferIStream *	pIStream = NULL;
	char *				pszAllocBuffer = NULL;
	
	// Get a pointer to the characters in the string.
	
	flmAssert( sBuffer);
	pszBuffer = pEnv->GetStringUTFChars( sBuffer, NULL);
	uiStrCharCount = (FLMUINT)pEnv->GetStringUTFLength( sBuffer);
	flmAssert( uiStrCharCount);
	
	// Create the buffer stream object.
	
	if ((pIStream = f_new F_BufferIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Call the openStream method so that it will allocate a buffer
	// internally.  Add one to the size so that we allocate space for
	// a null terminating byte - because uiStrCharCount does NOT include
	// the null terminating byte.  Buffer pointer is returned in pucBuffer.
	
	if( RC_BAD( rc = pIStream->openStream( NULL, uiStrCharCount + 1, &pszAllocBuffer)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Copy the data from the passed in string into pucBuffer, including the NULL.
	
	f_memcpy( pszAllocBuffer, pszBuffer, uiStrCharCount);
	
	// NULL terminate the allocated buffer.
	
	pszAllocBuffer [uiStrCharCount] = 0;
	
Exit:

	if (pszBuffer)
	{
		pEnv->ReleaseStringUTFChars( sBuffer, pszBuffer);
	}

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openFileIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPath)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE				ucPath [F_PATH_MAX_SIZE];
	F_DynaBuf			pathBuf( ucPath, sizeof( ucPath));
	IF_PosIStream *	pIStream = NULL;
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sPath, &pathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openFileIStream(
								(const char *)pathBuf.getBufferPtr(), &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pIStream);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openMultiFileIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openMultiFileIStream(
											(const char *)directoryBuf.getBufferPtr(),
											(const char *)baseNameBuf.getBufferPtr(),
											&pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferedIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jint				iBufferSize)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBufferedIStream( pInputStream,
												(FLMUINT)iBufferSize, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openUncompressingIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openUncompressingIStream( pInputStream, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBase64Encoder(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jboolean			bInsertLineBreaks)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBase64Encoder( pInputStream,
								bInsertLineBreaks ? TRUE : FALSE, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBase64Decoder(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBase64Decoder( pInputStream, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openFileOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sFileName,
	jboolean			bTruncateIfFileExists)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	FLMBYTE				ucFileName [F_PATH_MAX_SIZE];
	F_DynaBuf			fileNameBuf( ucFileName, sizeof( ucFileName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sFileName, &fileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openFileOStream(
								(const char *)fileNameBuf.getBufferPtr(),
								bTruncateIfFileExists ? TRUE : FALSE, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openMultiFileOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName,
	jint				iMaxFileSize,
	jboolean			bOkToOverwrite)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openMultiFileOStream(
								(const char *)directoryBuf.getBufferPtr(),
								(const char *)baseNameBuf.getBufferPtr(),
								(FLMUINT)iMaxFileSize,
								bOkToOverwrite ? TRUE : FALSE, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1removeMultiFileStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->removeMultiFileStream(
								(const char *)directoryBuf.getBufferPtr(),
								(const char *)baseNameBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferedOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lInputOStream,
	jint				iBufferSize)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)lInputOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBufferedOStream(
								pInputOStream, (FLMUINT)iBufferSize, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openCompressingOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lInputOStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)lInputOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openCompressingOStream( pInputOStream, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1writeToOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jlong				lOStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = (IF_IStream *)((FLMUINT)lIStream);
	IF_OStream *		pOStream = (IF_OStream *)((FLMUINT)lOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->writeToOStream( pIStream, pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1createJDataVector(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	ifpDataVector = NULL;
	
	if (RC_BAD( rc = THIS_DBSYS()->createIFDataVector( &ifpDataVector)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)ifpDataVector);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1updateIniFile(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jstring			sParamName,
	jstring			sValue)
{
	RCODE							rc = NE_XFLM_OK;
	IF_DbSystem *				pDbSystem = THIS_DBSYS();
	FLMBYTE						ucParamName [80];
	F_DynaBuf					paramNameBuf( ucParamName, sizeof( ucParamName));
	FLMBYTE						ucValue [80];
	F_DynaBuf					valueBuf( ucValue, sizeof( ucValue));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sParamName, &paramNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sValue, &valueBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Call the rebuild function.
	
	if (RC_BAD( rc = pDbSystem->updateIniFile(
				(const char *)paramNameBuf.getBufferPtr(),
				(const char *)valueBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbDup(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jlong				lDbToDup)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DbSystem *	pDbSystem = THIS_DBSYS();
	IF_Db *			pDbToDup = (IF_Db *)((FLMUINT)lDbToDup);
	IF_Db *			pDb = NULL;

	if (!pDbToDup)
	{
		rc = RC_SET( NE_XFLM_INVALID_PARM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = pDbSystem->dbDup( pDbToDup, &pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setDynamicMemoryLimit(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCacheAdjustPercent,
	jint				iCacheAdjustMin,
	jint				iCacheAdjustMax,
	jint				iCacheAdjustMinToLeave)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_DBSYS()->setDynamicMemoryLimit(
								(FLMUINT)iCacheAdjustPercent,
								(FLMUINT)iCacheAdjustMin,
								(FLMUINT)iCacheAdjustMax,
								(FLMUINT)iCacheAdjustMinToLeave)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setHardMemoryLimit(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iPercent,
	jboolean			bPercentOfAvail,
	jint				iMin,
	jint				iMax,
	jint				iMinToLeave,
	jboolean			bPreallocate)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_DBSYS()->setHardMemoryLimit(
								(FLMUINT)iPercent,
								bPercentOfAvail? TRUE : FALSE,
								(FLMUINT)iMin,
								(FLMUINT)iMax,
								(FLMUINT)iMinToLeave,
								bPreallocate ? TRUE : FALSE)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1getDynamicCacheSupported(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_DBSYS()->getDynamicCacheSupported() ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_DbSystem__1getCacheInfo(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	XFLM_CACHE_INFO	cacheInfo;
	jclass				jCacheInfoClass = NULL;
	jclass				jCacheUsageClass = NULL;
	jclass				jSlabUsageClass = NULL;
	jobject				jCacheInfo = NULL;
	jobject				jBlockCacheUsage = NULL;
	jobject				jBlockSlabUsage = NULL;
	jobject				jNodeCacheUsage = NULL;
	jobject				jNodeSlabUsage = NULL;
	
	THIS_DBSYS()->getCacheInfo( &cacheInfo);
	
	// Find the CacheInfo, CacheUsage, and SlabUsage classes

	if ((jCacheInfoClass = pEnv->FindClass( "xflaim/CacheInfo")) == NULL)
	{
		goto Exit;
	}
	if ((jCacheUsageClass = pEnv->FindClass( "xflaim/CacheUsage")) == NULL)
	{
		goto Exit;
	}
	if ((jSlabUsageClass = pEnv->FindClass( "xflaim/SlabUsage")) == NULL)
	{
		goto Exit;
	}

	// Allocate a cache info object and the needed cache usage and slab usage
	// objects.
	
	if ((jCacheInfo = pEnv->AllocObject( jCacheInfoClass)) == NULL)
	{
		goto Exit;
	}
	if ((jBlockCacheUsage = pEnv->AllocObject( jCacheUsageClass)) == NULL)
	{
		goto Exit;
	}
	if ((jNodeCacheUsage = pEnv->AllocObject( jCacheUsageClass)) == NULL)
	{
		goto Exit;
	}
	if ((jBlockSlabUsage = pEnv->AllocObject( jSlabUsageClass)) == NULL)
	{
		goto Exit;
	}
	if ((jNodeSlabUsage = pEnv->AllocObject( jSlabUsageClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the Block slab usage fields.
	
	pEnv->SetLongField( jBlockSlabUsage, fid_SlabUsage_lSlabs,
						(jlong)cacheInfo.BlockCache.slabUsage.ui64Slabs);
	pEnv->SetLongField( jBlockSlabUsage, fid_SlabUsage_lSlabBytes,
						(jlong)cacheInfo.BlockCache.slabUsage.ui64SlabBytes);
	pEnv->SetLongField( jBlockSlabUsage, fid_SlabUsage_lAllocatedCells,
						(jlong)cacheInfo.BlockCache.slabUsage.ui64AllocatedCells);
	pEnv->SetLongField( jBlockSlabUsage, fid_SlabUsage_lFreeCells,
						(jlong)cacheInfo.BlockCache.slabUsage.ui64FreeCells);
	
	// Set the Node slab usage fields.
	
	pEnv->SetLongField( jNodeSlabUsage, fid_SlabUsage_lSlabs,
						(jlong)cacheInfo.NodeCache.slabUsage.ui64Slabs);
	pEnv->SetLongField( jNodeSlabUsage, fid_SlabUsage_lSlabBytes,
						(jlong)cacheInfo.NodeCache.slabUsage.ui64SlabBytes);
	pEnv->SetLongField( jNodeSlabUsage, fid_SlabUsage_lAllocatedCells,
						(jlong)cacheInfo.NodeCache.slabUsage.ui64AllocatedCells);
	pEnv->SetLongField( jNodeSlabUsage, fid_SlabUsage_lFreeCells,
						(jlong)cacheInfo.NodeCache.slabUsage.ui64FreeCells);
						
	// Set the Block cache usage fields
	
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iByteCount,
						(jint)cacheInfo.BlockCache.uiByteCount);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iCount,
						(jint)cacheInfo.BlockCache.uiCount);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iOldVerCount,
						(jint)cacheInfo.BlockCache.uiOldVerCount);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iOldVerBytes,
						(jint)cacheInfo.BlockCache.uiOldVerBytes);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iCacheHits,
						(jint)cacheInfo.BlockCache.uiCacheHits);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iCacheHitLooks,
						(jint)cacheInfo.BlockCache.uiCacheHitLooks);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iCacheFaults,
						(jint)cacheInfo.BlockCache.uiCacheFaults);
	pEnv->SetIntField( jBlockCacheUsage, fid_CacheUsage_iCacheFaultLooks,
						(jint)cacheInfo.BlockCache.uiCacheFaultLooks);
	pEnv->SetObjectField( jBlockCacheUsage, fid_CacheUsage_slabUsage,
						jBlockSlabUsage);
						
	// Set the Node cache usage fields
	
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iByteCount,
						(jint)cacheInfo.NodeCache.uiByteCount);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iCount,
						(jint)cacheInfo.NodeCache.uiCount);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iOldVerCount,
						(jint)cacheInfo.NodeCache.uiOldVerCount);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iOldVerBytes,
						(jint)cacheInfo.NodeCache.uiOldVerBytes);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iCacheHits,
						(jint)cacheInfo.NodeCache.uiCacheHits);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iCacheHitLooks,
						(jint)cacheInfo.NodeCache.uiCacheHitLooks);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iCacheFaults,
						(jint)cacheInfo.NodeCache.uiCacheFaults);
	pEnv->SetIntField( jNodeCacheUsage, fid_CacheUsage_iCacheFaultLooks,
						(jint)cacheInfo.NodeCache.uiCacheFaultLooks);
	pEnv->SetObjectField( jNodeCacheUsage, fid_CacheUsage_slabUsage,
						jNodeSlabUsage);
						
	// Set the cache info fields.
	
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iMaxBytes,
						(jint)cacheInfo.uiMaxBytes);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iTotalBytesAllocated,
						(jint)cacheInfo.uiTotalBytesAllocated);
	pEnv->SetBooleanField( jCacheInfo, fid_CacheInfo_bDynamicCacheAdjust,
						(jboolean)(cacheInfo.bDynamicCacheAdjust ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iCacheAdjustPercent,
						(jint)cacheInfo.uiCacheAdjustPercent);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iCacheAdjustMin,
						(jint)cacheInfo.uiCacheAdjustMin);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iCacheAdjustMax,
						(jint)cacheInfo.uiCacheAdjustMax);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iCacheAdjustMinToLeave,
						(jint)cacheInfo.uiCacheAdjustMinToLeave);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iDirtyCount,
						(jint)cacheInfo.uiDirtyCount);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iDirtyBytes,
						(jint)cacheInfo.uiDirtyBytes);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iNewCount,
						(jint)cacheInfo.uiNewCount);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iNewBytes,
						(jint)cacheInfo.uiNewBytes);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iLogCount,
						(jint)cacheInfo.uiLogCount);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iLogBytes,
						(jint)cacheInfo.uiLogBytes);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iFreeCount,
						(jint)cacheInfo.uiFreeCount);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iFreeBytes,
						(jint)cacheInfo.uiFreeBytes);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iReplaceableCount,
						(jint)cacheInfo.uiReplaceableCount);
	pEnv->SetIntField( jCacheInfo, fid_CacheInfo_iReplaceableBytes,
						(jint)cacheInfo.uiReplaceableBytes);
	pEnv->SetBooleanField( jCacheInfo, fid_CacheInfo_bPreallocatedCache,
						(jboolean)(cacheInfo.bPreallocatedCache ? JNI_TRUE : JNI_FALSE));
	pEnv->SetObjectField( jCacheInfo, fid_CacheInfo_BlockCache, jBlockCacheUsage);
	pEnv->SetObjectField( jCacheInfo, fid_CacheInfo_NodeCache, jNodeCacheUsage);
	
Exit:

	return( jCacheInfo);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1enableCacheDebug(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bDebug)
{
	THIS_DBSYS()->enableCacheDebug( bDebug ? TRUE : FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1cacheDebugEnabled(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_DBSYS()->cacheDebugEnabled() ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1closeUnusedFiles(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_DBSYS()->closeUnusedFiles( (FLMUINT)iSeconds)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1startStats(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_DBSYS()->startStats();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1stopStats(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_DBSYS()->stopStats();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1resetStats(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_DBSYS()->resetStats();
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewCountTimeStat(
	JNIEnv *					pEnv,
	F_COUNT_TIME_STAT *	pCountTimeStat,
	jclass					jCountTimeStatClass)
{
	jobject	jCountTimeStat;
	
	if ((jCountTimeStat = pEnv->AllocObject( jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetLongField( jCountTimeStat, fid_CountTimeStat_lCount,
						(jlong)pCountTimeStat->ui64Count);
	pEnv->SetLongField( jCountTimeStat, fid_CountTimeStat_lElapMilli,
						(jlong)pCountTimeStat->ui64ElapMilli);
	
Exit:

	return( jCountTimeStat);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewRTransStats(
	JNIEnv *					pEnv,
	XFLM_RTRANS_STATS *	pRTransStats,
	jclass					jRTransStatsClass,
	jclass					jCountTimeStatClass)
{
	jobject	jRTransStats = NULL;
	jobject	jCommittedTrans;
	jobject	jAbortedTrans;
	
	// Allocate the sub-objects that part of the read transaction
	// statistics object.
	
	if ((jCommittedTrans = NewCountTimeStat( pEnv, &pRTransStats->CommittedTrans,
										jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jAbortedTrans = NewCountTimeStat( pEnv, &pRTransStats->AbortedTrans,
										jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	
	// Allocate and populate a read transaction statistics object.
	
	if ((jRTransStats = pEnv->AllocObject( jRTransStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jRTransStats, fid_RTransStats_committedTrans,
						jCommittedTrans);
	pEnv->SetObjectField( jRTransStats, fid_RTransStats_abortedTrans,
						jAbortedTrans);
	
Exit:

	return( jRTransStats);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewUTransStats(
	JNIEnv *					pEnv,
	XFLM_UTRANS_STATS *	pUTransStats,
	jclass					jUTransStatsClass,
	jclass					jCountTimeStatClass)
{
	jobject	jUTransStats = NULL;
	jobject	jCommittedTrans;
	jobject	jGroupCompletes;
	jobject	jAbortedTrans;
	
	// Allocate the sub-objects that are part of the update transaction
	// statistics object.
	
	if ((jCommittedTrans = NewCountTimeStat( pEnv, &pUTransStats->CommittedTrans,
										jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jGroupCompletes = NewCountTimeStat( pEnv, &pUTransStats->GroupCompletes,
										jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jAbortedTrans = NewCountTimeStat( pEnv, &pUTransStats->AbortedTrans,
										jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	
	// Allocate and populate an update transaction statistics object.
	
	if ((jUTransStats = pEnv->AllocObject( jUTransStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jUTransStats, fid_UTransStats_committedTrans,
						jCommittedTrans);
	pEnv->SetObjectField( jUTransStats, fid_UTransStats_groupCompletes,
						jGroupCompletes);
	pEnv->SetLongField( jUTransStats, fid_UTransStats_lGroupFinished,
						(jlong)pUTransStats->ui64GroupFinished);
	pEnv->SetObjectField( jUTransStats, fid_UTransStats_abortedTrans,
						jAbortedTrans);
	
Exit:

	return( jUTransStats);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewLFileStats(
	JNIEnv *					pEnv,
	XFLM_LFILE_STATS *	pLFileStats,
	jclass					jLFileStatsClass,
	jclass					jBlockIOStatsClass,
	jclass					jDiskIOStatClass)
{
	jobject	jLFileStats = NULL;
	jobject	jRootBlockStats;
	jobject	jMiddleBlockStats;
	jobject	jLeafBlockStats;
	
	// Allocate the sub-objects that are part of the LFile statistics object.
	
	if ((jRootBlockStats = NewBlockIOStats( pEnv, &pLFileStats->RootBlockStats,
										jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jMiddleBlockStats = NewBlockIOStats( pEnv, &pLFileStats->MiddleBlockStats,
										jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLeafBlockStats = NewBlockIOStats( pEnv, &pLFileStats->LeafBlockStats,
										jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}

	// Allocate and populate the LFile statistics object
	
	if ((jLFileStats = pEnv->AllocObject( jLFileStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jLFileStats, fid_LFileStats_rootBlockStats,
				jRootBlockStats);
	pEnv->SetObjectField( jLFileStats, fid_LFileStats_middleBlockStats,
				jMiddleBlockStats);
	pEnv->SetObjectField( jLFileStats, fid_LFileStats_leafBlockStats,
				jLeafBlockStats);
	pEnv->SetLongField( jLFileStats, fid_LFileStats_lBlockSplits,
				(jlong)pLFileStats->ui64BlockSplits);
	pEnv->SetLongField( jLFileStats, fid_LFileStats_lBlockCombines,
				(jlong)pLFileStats->ui64BlockCombines);
	pEnv->SetIntField( jLFileStats, fid_LFileStats_iLFileNum,
				(jint)pLFileStats->uiLFileNum);
	pEnv->SetBooleanField( jLFileStats, fid_LFileStats_bIsIndex,
				(jboolean)(pLFileStats->eLfType == XFLM_LF_INDEX
							  ? JNI_TRUE
							  : JNI_FALSE));

Exit:

	return( jLFileStats);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewDiskIOStat(
	JNIEnv *					pEnv,
	XFLM_DISKIO_STAT *	pDiskIOStat,
	jclass					jDiskIOStatClass)
{
	jobject	jDiskIOStat;
	
	// Allocate and populate the fields of a disk I/O statistics object.
	
	if ((jDiskIOStat = pEnv->AllocObject( jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetLongField( jDiskIOStat, fid_DiskIOStat_lCount,
			(jlong)pDiskIOStat->ui64Count);
	pEnv->SetLongField( jDiskIOStat, fid_DiskIOStat_lTotalBytes,
			(jlong)pDiskIOStat->ui64TotalBytes);
	pEnv->SetLongField( jDiskIOStat, fid_DiskIOStat_lElapMilli,
			(jlong)pDiskIOStat->ui64ElapMilli);
	
Exit:

	return( jDiskIOStat);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewBlockIOStats(
	JNIEnv *					pEnv,
	XFLM_BLOCKIO_STATS *	pBlockIOStats,
	jclass					jBlockIOStatsClass,
	jclass					jDiskIOStatClass)
{
	jobject	jBlockIOStats = NULL;
	jobject	jBlockReads;
	jobject	jOldViewBlockReads;
	jobject	jBlockWrites;
	
	// Allocate the sub-objects that are part of the block I/O statistics object.
	
	if ((jBlockReads = NewDiskIOStat( pEnv, &pBlockIOStats->BlockReads,
								jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jOldViewBlockReads = NewDiskIOStat( pEnv, &pBlockIOStats->OldViewBlockReads,
								jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jBlockWrites = NewDiskIOStat( pEnv, &pBlockIOStats->BlockWrites,
								jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}

	// Allocate and populate the block I/O statistics object.
	
	if ((jBlockIOStats = pEnv->AllocObject( jBlockIOStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jBlockIOStats, fid_BlockIOStats_blockReads,
		jBlockReads);
	pEnv->SetObjectField( jBlockIOStats, fid_BlockIOStats_oldViewBlockReads,
		jOldViewBlockReads);
	pEnv->SetIntField( jBlockIOStats, fid_BlockIOStats_iBlockChkErrs,
		(jint)pBlockIOStats->ui32BlockChkErrs);
	pEnv->SetIntField( jBlockIOStats, fid_BlockIOStats_iOldViewBlockChkErrs,
		(jint)pBlockIOStats->ui32OldViewBlockChkErrs);
	pEnv->SetIntField( jBlockIOStats, fid_BlockIOStats_iOldViewErrors,
		(jint)pBlockIOStats->ui32OldViewErrors);
	pEnv->SetObjectField( jBlockIOStats, fid_BlockIOStats_blockWrites,
		jBlockWrites);
	
Exit:

	return( jBlockIOStats);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewLockStats(
	JNIEnv *			pEnv,
	F_LOCK_STATS *	pLockStats,
	jclass			jLockStatsClass,
	jclass			jCountTimeStatClass)
{
	jobject	jLockStats = NULL;
	jobject	jNoLocks;
	jobject	jWaitingForLock;
	jobject	jHeldLock;
	
	// Allocate the sub-objects that are part of the lock statisitcs object.

	if ((jNoLocks = NewCountTimeStat( pEnv, &pLockStats->NoLocks,
							jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jWaitingForLock = NewCountTimeStat( pEnv, &pLockStats->WaitingForLock,
							jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jHeldLock = NewCountTimeStat( pEnv, &pLockStats->HeldLock,
							jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}

	// Allocate and populate the lock statistics object.
	
	if ((jLockStats = pEnv->AllocObject( jLockStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jLockStats, fid_LockStats_noLocks,
			jNoLocks);
	pEnv->SetObjectField( jLockStats, fid_LockStats_waitingForLock,
			jWaitingForLock);
	pEnv->SetObjectField( jLockStats, fid_LockStats_heldLock,
			jHeldLock);
	
Exit:

	return( jLockStats);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewDbStats(
	JNIEnv *				pEnv,
	XFLM_DB_STATS *	pDbStats,
	jclass				jDbStatsClass,
	jclass				jRTransStatsClass,
	jclass				jUTransStatsClass,
	jclass				jLFileStatsClass,
	jclass				jBlockIOStatsClass,
	jclass				jDiskIOStatClass,
	jclass				jCountTimeStatClass,
	jclass				jLockStatsClass)
{
	jobject					jDbStats = NULL;
	jstring					jDbName;
	jobject					jRTransStats;
	jobject					jUTransStats;
	jobjectArray			jLFileStatsArray;
	jobject					jLfhBlockStats;
	jobject					jAvailBlockStats;
	jobject					jDbHdrWrites;
	jobject					jLogBlockWrites;
	jobject					jLogBlockRestores;
	jobject					jLogBlockReads;
	jobject					jLockStats;
	
	// Allocate all of the sub-objects that are part of the database statistics
	// object.
	
	if ((jDbName = pEnv->NewStringUTF( pDbStats->pszDbName)) == NULL)
	{
		goto Exit;
	}
	if ((jRTransStats = NewRTransStats( pEnv, &pDbStats->ReadTransStats,
									jRTransStatsClass, jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jUTransStats = NewUTransStats( pEnv, &pDbStats->UpdateTransStats,
									jUTransStatsClass, jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLfhBlockStats = NewBlockIOStats( pEnv, &pDbStats->LFHBlockStats,
									jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jAvailBlockStats = NewBlockIOStats( pEnv, &pDbStats->AvailBlockStats,
									jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jDbHdrWrites = NewDiskIOStat( pEnv, &pDbStats->DbHdrWrites,
									jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLogBlockWrites = NewDiskIOStat( pEnv, &pDbStats->LogBlockWrites,
									jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLogBlockRestores = NewDiskIOStat( pEnv, &pDbStats->LogBlockRestores,
									jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLogBlockReads = NewDiskIOStat( pEnv, &pDbStats->LogBlockReads,
									jDiskIOStatClass)) == NULL)
	{
		goto Exit;
	}
	if ((jLockStats = NewLockStats( pEnv, &pDbStats->LockStats,
									jLockStatsClass, jCountTimeStatClass)) == NULL)
	{
		goto Exit;
	}
	
	// Get a logical file array.
	
	if (!pDbStats->uiNumLFileStats)
	{
		jLFileStatsArray = NULL;
	}
	else
	{
		jobject					jLFileStats;
		FLMUINT					uiLoop;
		XFLM_LFILE_STATS *	pLFileStats;
		
		if ((jLFileStatsArray = pEnv->NewObjectArray( (jsize)pDbStats->uiNumLFileStats,
							jLFileStatsClass, NULL)) == NULL)
		{
			goto Exit;
		}
		
		// Populate the LFileStats array
		
		for (uiLoop = 0, pLFileStats = pDbStats->pLFileStats;
			  uiLoop < pDbStats->uiNumLFileStats;
			  uiLoop++, pLFileStats++)
		{
			
			// Get an LFile statistics object.
			
			if ((jLFileStats = NewLFileStats( pEnv, pLFileStats, jLFileStatsClass,
						jBlockIOStatsClass, jDiskIOStatClass)) == NULL)
			{
				goto Exit;
			}
			
			// Put the LFile statistics object into the array of
			// LFile statistics objects.
			
			pEnv->SetObjectArrayElement( jLFileStatsArray, (jsize)uiLoop,
								jLFileStats);
		}
	}
	
	// Allocate and populate the database statistics object
	
	if ((jDbStats = pEnv->AllocObject( jDbStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jDbStats, fid_DbStats_sDbName,
		jDbName);
	pEnv->SetObjectField( jDbStats, fid_DbStats_readTransStats,
		jRTransStats);
	pEnv->SetObjectField( jDbStats, fid_DbStats_updateTransStats,
		jUTransStats);
	pEnv->SetObjectField( jDbStats, fid_DbStats_lfileStats,
		jLFileStatsArray);
	pEnv->SetObjectField( jDbStats, fid_DbStats_lfhBlockStats,
		jLfhBlockStats);
	pEnv->SetObjectField( jDbStats, fid_DbStats_availBlockStats,
		jAvailBlockStats);
	pEnv->SetObjectField( jDbStats, fid_DbStats_dbHdrWrites,
		jDbHdrWrites);
	pEnv->SetObjectField( jDbStats, fid_DbStats_logBlockWrites,
		jLogBlockWrites);
	pEnv->SetObjectField( jDbStats, fid_DbStats_logBlockRestores,
		jLogBlockRestores);
	pEnv->SetObjectField( jDbStats, fid_DbStats_logBlockReads,
		jLogBlockReads);
	pEnv->SetIntField( jDbStats, fid_DbStats_iLogBlockChkErrs,
		(jint)pDbStats->uiLogBlockChkErrs);
	pEnv->SetIntField( jDbStats, fid_DbStats_iReadErrors,
		(jint)pDbStats->uiReadErrors);
	pEnv->SetIntField( jDbStats, fid_DbStats_iWriteErrors,
		(jint)pDbStats->uiWriteErrors);
	pEnv->SetObjectField( jDbStats, fid_DbStats_lockStats,
		jLockStats);
			
Exit:

	return( jDbStats);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_DbSystem__1getStats(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_STATS			stats;
	jclass				jStatsClass;
	jclass				jDbStatsClass;
	jclass				jRTransStatsClass;
	jclass				jUTransStatsClass;
	jclass				jLFileStatsClass;
	jclass				jBlockIOStatsClass;
	jclass				jDiskIOStatClass;
	jclass				jCountTimeStatClass;
	jclass				jLockStatsClass;
	jobject				jStats = NULL;
	jobjectArray		jDbStatsArray;
	
	// Get the statistics.
	
	f_memset( &stats, 0, sizeof( XFLM_STATS));
	if (RC_BAD( rc = THIS_DBSYS()->getStats( &stats)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Find all of the needed classes that will be used to populate the
	// statistics object.

	if ((jStatsClass = pEnv->FindClass( "xflaim/Stats")) == NULL)
	{
		goto Exit;
	}
	if ((jDbStatsClass = pEnv->FindClass( "xflaim/DbStats")) == NULL)
	{
		goto Exit;
	}
	if ((jRTransStatsClass = pEnv->FindClass( "xflaim/RTransStats")) == NULL)
	{
		goto Exit;
	}
	if ((jUTransStatsClass = pEnv->FindClass( "xflaim/UTransStats")) == NULL)
	{
		goto Exit;
	}
	if ((jLFileStatsClass = pEnv->FindClass( "xflaim/LFileStats")) == NULL)
	{
		goto Exit;
	}
	if ((jBlockIOStatsClass = pEnv->FindClass( "xflaim/BlockIOStats")) == NULL)
	{
		goto Exit;
	}
	if ((jDiskIOStatClass = pEnv->FindClass( "xflaim/DiskIOStat")) == NULL)
	{
		goto Exit;
	}
	if ((jCountTimeStatClass = pEnv->FindClass( "xflaim/CountTimeStat")) == NULL)
	{
		goto Exit;
	}
	if ((jLockStatsClass = pEnv->FindClass( "xflaim/LockStats")) == NULL)
	{
		goto Exit;
	}

	// Allocate and populate an array of database statistics objects.
	
	if (!stats.uiNumDbStats)
	{
		jDbStatsArray = NULL;
	}
	else
	{
		jobject				jDbStats;
		FLMUINT				uiLoop;
		XFLM_DB_STATS *	pDbStats;
		
		if ((jDbStatsArray = pEnv->NewObjectArray( (jsize)stats.uiNumDbStats,
							jDbStatsClass, NULL)) == NULL)
		{
			goto Exit;
		}
		
		// Populate the DbStats array
		
		for (uiLoop = 0, pDbStats = stats.pDbStats;
			  uiLoop < stats.uiNumDbStats;
			  uiLoop++, pDbStats++)
		{
			
			// Allocate a database statistics object.
			
			if ((jDbStats = NewDbStats( pEnv, pDbStats, jDbStatsClass, jRTransStatsClass,
					jUTransStatsClass, jLFileStatsClass, jBlockIOStatsClass,
					jDiskIOStatClass, jCountTimeStatClass, jLockStatsClass)) == NULL)
			{
				goto Exit;
			}
			
			// Put the database statistics object into the array of
			// database statistics objects.
			
			pEnv->SetObjectArrayElement( jDbStatsArray, (jsize)uiLoop, jDbStats);
		}
	}
	
	// Allocate and populate the statistics object
	
	if ((jStats = pEnv->AllocObject( jStatsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetObjectField( jStats, fid_Stats_dbStats, jDbStatsArray);
	pEnv->SetIntField( jStats, fid_Stats_iStartTime, (jint)stats.uiStartTime);
	pEnv->SetIntField( jStats, fid_Stats_iStopTime, (jint)stats.uiStopTime);
	
Exit:

	THIS_DBSYS()->freeStats( &stats);
	return( jStats);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setTempDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPath)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucPath [F_PATH_MAX_SIZE];
	F_DynaBuf		pathBuf( ucPath, sizeof( ucPath));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sPath, &pathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->setTempDir( (const char *)pathBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DbSystem__1getTempDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE		rc = NE_XFLM_OK;
	char		szPath [F_PATH_MAX_SIZE];
	jstring	jPath = NULL;
	
	if (RC_BAD( rc = THIS_DBSYS()->getTempDir( szPath)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	jPath = pEnv->NewStringUTF( szPath);
	
Exit:

	return( jPath);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setCheckpointInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	THIS_DBSYS()->setCheckpointInterval( (FLMUINT)iSeconds);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getCheckpointInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getCheckpointInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setCacheAdjustInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	THIS_DBSYS()->setCacheAdjustInterval( (FLMUINT)iSeconds);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getCacheAdjustInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getCacheAdjustInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setCacheCleanupInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	THIS_DBSYS()->setCacheCleanupInterval( (FLMUINT)iSeconds);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getCacheCleanupInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getCacheCleanupInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setUnusedCleanupInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	THIS_DBSYS()->setUnusedCleanupInterval( (FLMUINT)iSeconds);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getUnusedCleanupInterval(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getUnusedCleanupInterval());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setMaxUnusedTime(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iSeconds)
{
	THIS_DBSYS()->setMaxUnusedTime( (FLMUINT)iSeconds);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getMaxUnusedTime(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getMaxUnusedTime());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1deactivateOpenDb(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbFileName,
	jstring			sDataDir)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucDbFileName [F_PATH_MAX_SIZE];
	F_DynaBuf		dbFileNameBuf( ucDbFileName, sizeof( ucDbFileName));
	FLMBYTE			ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf		dataDirBuf( ucDataDir, sizeof( ucDataDir));
	
	// Get the strings.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDbFileName, &dbFileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	THIS_DBSYS()->deactivateOpenDb( (const char *)dbFileNameBuf.getBufferPtr(),
					(const char *)(dataDirBuf.getDataLength() > 1
										? (const char *)dataDirBuf.getBufferPtr()
										: (const char *)NULL));

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setQuerySaveMax(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iMaxToSave)
{
	THIS_DBSYS()->setQuerySaveMax( (FLMUINT)iMaxToSave);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getQuerySaveMax(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jint)THIS_DBSYS()->getQuerySaveMax());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setDirtyCacheLimits(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iMaxDirty,
	jint				iLowDirty)
{
	THIS_DBSYS()->setDirtyCacheLimits( (FLMUINT)iMaxDirty, (FLMUINT)iLowDirty);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getMaxDirtyCacheLimit(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	FLMUINT	uiMaxDirty;
	FLMUINT	uiLowDirty;
	
	THIS_DBSYS()->getDirtyCacheLimits( &uiMaxDirty, &uiLowDirty);
	return( (jint)uiMaxDirty);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1getLowDirtyCacheLimit(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	FLMUINT	uiMaxDirty;
	FLMUINT	uiLowDirty;
	
	THIS_DBSYS()->getDirtyCacheLimits( &uiMaxDirty, &uiLowDirty);
	return( (jint)uiLowDirty);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DbSystem__1compareStrings(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sLeftString,
	jboolean			bLeftWild,
	jstring			sRightString,
	jboolean			bRightWild,
	jint				iCompareRules,
	jint				iLanguage)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucLeftString [100];
	F_DynaBuf		leftStringBuf( ucLeftString, sizeof( ucLeftString));
	FLMBYTE			ucRightString [100];
	F_DynaBuf		rightStringBuf( ucRightString, sizeof( ucRightString));
	FLMINT			iResult = 0;
	
	// Get the strings.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sLeftString, &leftStringBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRightString, &rightStringBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->compareUTF8Strings(
			(const FLMBYTE *)leftStringBuf.getBufferPtr(),
			leftStringBuf.getDataLength() - 1,
			bLeftWild ? TRUE : FALSE,
			(const FLMBYTE *)rightStringBuf.getBufferPtr(),
			rightStringBuf.getDataLength() - 1,
			bRightWild ? TRUE : FALSE,
			(FLMUINT)iCompareRules, (FLMUINT)iLanguage, &iResult)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)iResult);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1hasSubStr(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sString,
	jstring			sSubString,
	jint				iCompareRules,
	jint				iLanguage)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucString [100];
	F_DynaBuf		stringBuf( ucString, sizeof( ucString));
	FLMBYTE			ucSubString [100];
	F_DynaBuf		subStringBuf( ucSubString, sizeof( ucSubString));
	FLMBOOL			bExists = FALSE;
	
	// Get the strings.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sString, &stringBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSubString, &subStringBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->utf8IsSubStr(
			(const FLMBYTE *)stringBuf.getBufferPtr(),
			(const FLMBYTE *)subStringBuf.getBufferPtr(),
			(FLMUINT)iCompareRules, (FLMUINT)iLanguage, &bExists)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jboolean)(bExists ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1uniIsUpper(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jchar				uzChar)
{
	return( THIS_DBSYS()->uniIsUpper( (FLMUNICODE)uzChar) ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1uniIsLower(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jchar				uzChar)
{
	return( THIS_DBSYS()->uniIsLower( (FLMUNICODE)uzChar) ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1uniIsAlpha(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jchar				uzChar)
{
	return( THIS_DBSYS()->uniIsAlpha( (FLMUNICODE)uzChar) ? JNI_TRUE : JNI_FALSE);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1uniIsDecimalDigit(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jchar				uzChar)
{
	return( THIS_DBSYS()->uniIsDecimalDigit( (FLMUNICODE)uzChar) ? JNI_TRUE : JNI_FALSE);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jchar JNICALL Java_xflaim_DbSystem__1uniToLower(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jchar				uzChar)
{
	return( (jchar)THIS_DBSYS()->uniToLower( (FLMUNICODE)uzChar));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1waitToClose(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbFileName)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucDbFileName [F_PATH_MAX_SIZE];
	F_DynaBuf		dbFileNameBuf( ucDbFileName, sizeof( ucDbFileName));
	
	// Get the strings.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDbFileName, &dbFileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = THIS_DBSYS()->waitToClose(
						(const char *)dbFileNameBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1clearCache(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbWithUpdateTrans)
{
	RCODE		rc = NE_XFLM_OK;
	IF_Db *	pDbWithUpdateTrans = (IF_Db *)((FLMUINT)lDbWithUpdateTrans);
	
	if (RC_BAD( rc = THIS_DBSYS()->clearCache( pDbWithUpdateTrans)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

