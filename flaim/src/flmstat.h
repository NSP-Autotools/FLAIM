//-------------------------------------------------------------------------
// Desc:	Routines for updating statistics - for monitoring - definitions.
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

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/**************************************************************************
				Various function prototypes.
**************************************************************************/

RCODE	flmStatGetDb(
	FLM_STATS *			pFlmStats,
	void *				pFile,
	FLMUINT				uiLowStart,
	DB_STATS **			ppDbStatsRV,
	FLMUINT *			puiDbAllocSeqRV,
	FLMUINT *			puiDbTblPosRV);

RCODE	flmStatGetLFile(
	DB_STATS *			pDbStats,
	FLMUINT				uiLFileNum,
	FLMUINT				uiLfType,
	FLMUINT				uiLowStart,
	LFILE_STATS **		ppLFileStatsRV,
	FLMUINT *			puiLFileAllocSeqRV,
	FLMUINT *			puiLFileTblPosRV);

void flmStatReset(
	FLM_STATS *			pStats,
	FLMBOOL				bSemAlreadyLocked,
	FLMBOOL				bFree);

void flmStatStart(
	FLM_STATS *			pStats);

void flmStatStop(
	FLM_STATS *			pStats);

RCODE flmStatInit(
	FLM_STATS *			pStats,
	FLMBOOL				bEnableSharing);

void flmUpdateBlockIOStats(
	BLOCKIO_STATS *	pDest,
	BLOCKIO_STATS *	pSrc);

RCODE	flmStatUpdate(
	FLM_STATS *			pDestStats,
	FLM_STATS *			pSrcStats);

void flmFreeSavedQueries(
	FLMBOOL				bMutexAlreadyLocked);

void flmSaveQuery(
	HFCURSOR				hCursor);

BLOCKIO_STATS * flmGetBlockIOStatPtr(
	DB_STATS *			pDbStats,
	LFILE_STATS *		pLFileStats,
	FLMBYTE *			pBlk,
	FLMUINT				uiBlkType);

void flmAddElapTime(
	F_TMSTAMP  *		pStartTime,
	FLMUINT64 *			pui64ElapMilli);

/****************************************************************************
Desc:	This routine updates statistics from one DISKIO_STAT structure
		into another.
****************************************************************************/
FINLINE void flmUpdateDiskIOStats(
	DISKIO_STAT *		pDest,
	DISKIO_STAT *		pSrc)
{
	pDest->ui64Count += pSrc->ui64Count;
	pDest->ui64TotalBytes += pSrc->ui64TotalBytes;
	pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void flmUpdateCountTimeStats(
	F_COUNT_TIME_STAT *	pDest,
	F_COUNT_TIME_STAT *	pSrc)
{
	pDest->ui64Count += pSrc->ui64Count;
	pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
}

#include "fpackoff.h"

#endif
