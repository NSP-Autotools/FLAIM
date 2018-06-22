//------------------------------------------------------------------------------
// Desc:	This include file contains the structure definitions and prototypes
//			needed to capture statistics.
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

#ifndef FLMSTAT_H
#define FLMSTAT_H

#define INIT_DB_STAT_ARRAY_SIZE			5
#define DB_STAT_ARRAY_INCR_SIZE			5
#define INIT_LFILE_STAT_ARRAY_SIZE		5
#define LFILE_STAT_ARRAY_INCR_SIZE		5

/**************************************************************************
				Various function prototypes.
**************************************************************************/

RCODE	flmStatGetDb(								// Source: flmstat.cpp
	XFLM_STATS *			pFlmStats,
	F_Database *			pDatabase,
	FLMUINT					uiLowStart,
	XFLM_DB_STATS **		ppDbStatsRV,
	FLMUINT *				puiDbAllocSeqRV,
	FLMUINT *				puiDbTblPosRV);

RCODE	flmStatGetLFile(							// Source: flmstat.cpp
	XFLM_DB_STATS *		pDbStats,
	FLMUINT					uiLFileNum,
	eLFileType				eLfType,
	FLMUINT					uiLowStart,
	XFLM_LFILE_STATS **	ppLFileStatsRV,
	FLMUINT *				puiLFileAllocSeqRV,
	FLMUINT *				puiLFileTblPosRV);

void flmStatReset(								// Source: flmstat.cpp
	XFLM_STATS *			pStats,
	FLMBOOL					bFree);

FINLINE void flmStatStart(
	XFLM_STATS *			pStats)
{
	pStats->bCollectingStats = TRUE;
	flmStatReset( pStats, TRUE);
}

FINLINE void flmStatStop(
	XFLM_STATS *	pStats)
{
	if (pStats->bCollectingStats)
	{
		pStats->bCollectingStats = FALSE;
		f_timeGetSeconds( &pStats->uiStopTime);
	}
}

FINLINE void flmStatFree(
	XFLM_STATS *	pStats)
{
	pStats->bCollectingStats = FALSE;
	flmStatReset( pStats, TRUE);
}

void flmUpdateBlockIOStats(					// Source: flmstat.cpp
	XFLM_BLOCKIO_STATS *	pDest,
	XFLM_BLOCKIO_STATS *	pSrc);

RCODE	flmStatUpdate(								// Source: flmstat.cpp
	XFLM_STATS *	pSrcStats);

void flmFreeSavedQueries(						// Source: flmstat.cpp
	FLMBOOL	bMutexAlreadyLocked);

void flmSaveQuery(								// Source: flmstat.cpp
	F_Query *		pQuery);

RCODE	flmStatCopy(								// Source: flmstat.cpp
	XFLM_STATS *	pDestStats,
	XFLM_STATS *	pSrcStats);

XFLM_BLOCKIO_STATS * flmGetBlockIOStatPtr(// Source: flmstat.cpp
	XFLM_DB_STATS *		pDbStats,
	XFLM_LFILE_STATS *	pLFileStats,
	FLMBYTE *				pucBlk);

void flmAddElapTime(								// Source: flmstat.cpp
	F_TMSTAMP  *	pStartTime,
	FLMUINT64 *		pui64ElapMilli);

/****************************************************************************
									Inline Functions
****************************************************************************/

	/*
	Desc:	This routine updates statistics from one DISKIO_STAT structure into
			another.
	*/
	FINLINE void flmUpdateDiskIOStats(
		XFLM_DISKIO_STAT *	pDest,
		XFLM_DISKIO_STAT *	pSrc)
	{
		pDest->ui64Count += pSrc->ui64Count;
		pDest->ui64TotalBytes += pSrc->ui64TotalBytes;
		pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
	}

	FINLINE void flmUpdateCountTimeStats(
		F_COUNT_TIME_STAT *	pDest,
		F_COUNT_TIME_STAT *	pSrc)
	{
		pDest->ui64Count += pSrc->ui64Count;
		pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
	}

#endif	// ifdef FLMSTAT_H
