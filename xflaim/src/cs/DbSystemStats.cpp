//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DbSystemStats class
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

#include "xflaim.h"

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystemStats_freeStats(
	IF_DbSystem *	pDbSystem,
	XFLM_STATS *	pStats)
{
	if (pStats)
	{
		pDbSystem->freeStats( pStats);
		f_free( &pStats);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DbSystemStats_getGeneralStats(
	XFLM_STATS *	pStats,
	FLMUINT32 *		pui32NumDatabases,
	FLMUINT32 *		pui32StartTime,
	FLMUINT32 *		pui32StopTime)
{
	*pui32NumDatabases = (FLMUINT32)pStats->uiNumDbStats;
	*pui32StartTime = (FLMUINT32)pStats->uiStartTime;
	*pui32StopTime = (FLMUINT32)pStats->uiStopTime;
}

// IMPORTANT NOTE: This structure needs to stay in sync with the
// corresponding class in the C# code.
typedef struct
{
	char *						pszDbName;
	FLMUINT32					ui32NumLFiles;
	XFLM_RTRANS_STATS			ReadTransStats;
	XFLM_UTRANS_STATS			UpdateTransStats;
	XFLM_BLOCKIO_STATS		LFHBlockStats;
	XFLM_BLOCKIO_STATS		AvailBlockStats;
	XFLM_DISKIO_STAT			DbHdrWrites;
	XFLM_DISKIO_STAT			LogBlockWrites;
	XFLM_DISKIO_STAT			LogBlockRestores;
	XFLM_DISKIO_STAT			LogBlockReads;
	FLMUINT32					ui32LogBlockChkErrs;
	FLMUINT32					ui32ReadErrors;
	FLMUINT32					ui32WriteErrors;
	F_LOCK_STATS				LockStats;
} CS_XFLM_DB_STATS;

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystemStats_getDbStats(
	XFLM_STATS *			pStats,
	FLMUINT32				ui32DatabaseNum,
	CS_XFLM_DB_STATS *	pCSDbStats)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_DB_STATS *	pDbStat;

	if ((FLMUINT)ui32DatabaseNum > pStats->uiNumDbStats - 1)
	{
		rc = RC_SET( NE_XFLM_INVALID_PARM);
		goto Exit;
	}
	pDbStat = &pStats->pDbStats [ui32DatabaseNum];
	pCSDbStats->pszDbName = pDbStat->pszDbName;
	pCSDbStats->ui32NumLFiles = (FLMUINT32)pDbStat->uiNumLFileStats;
	f_memcpy( &pCSDbStats->ReadTransStats, &pDbStat->ReadTransStats, sizeof( XFLM_RTRANS_STATS));
	f_memcpy( &pCSDbStats->UpdateTransStats, &pDbStat->UpdateTransStats, sizeof( XFLM_UTRANS_STATS));
	f_memcpy( &pCSDbStats->LFHBlockStats, &pDbStat->LFHBlockStats, sizeof( XFLM_BLOCKIO_STATS));
	f_memcpy( &pCSDbStats->AvailBlockStats, &pDbStat->AvailBlockStats, sizeof( XFLM_BLOCKIO_STATS));
	f_memcpy( &pCSDbStats->DbHdrWrites, &pDbStat->DbHdrWrites, sizeof( XFLM_DISKIO_STAT));
	f_memcpy( &pCSDbStats->LogBlockWrites, &pDbStat->LogBlockWrites, sizeof( XFLM_DISKIO_STAT));
	f_memcpy( &pCSDbStats->LogBlockRestores, &pDbStat->LogBlockRestores, sizeof( XFLM_DISKIO_STAT));
	f_memcpy( &pCSDbStats->LogBlockReads, &pDbStat->LogBlockReads, sizeof( XFLM_DISKIO_STAT));
	pCSDbStats->ui32LogBlockChkErrs = (FLMUINT32)pDbStat->uiLogBlockChkErrs;
	pCSDbStats->ui32ReadErrors = (FLMUINT32)pDbStat->uiReadErrors;
	pCSDbStats->ui32WriteErrors = (FLMUINT32)pDbStat->uiWriteErrors;
	f_memcpy( &pCSDbStats->LockStats, &pDbStat->LockStats, sizeof( F_LOCK_STATS));

Exit:
	return( rc);
}

// IMPORTANT NOTE: This structure needs to stay in sync with the
// corresponding class in the C# code.
typedef struct
{
	XFLM_BLOCKIO_STATS	RootBlockStats;
	XFLM_BLOCKIO_STATS	MiddleBlockStats;
	XFLM_BLOCKIO_STATS	LeafBlockStats;
	FLMUINT64				ui64BlockSplits;
	FLMUINT64				ui64BlockCombines;
	FLMUINT32				ui32LFileNum;
	FLMINT32					i32LfType;
} CS_XFLM_LFILE_STATS;

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DbSystemStats_getLFileStats(
	XFLM_STATS *				pStats,
	FLMUINT32					ui32DatabaseNum,
	FLMUINT32					ui32LFileNum,
	CS_XFLM_LFILE_STATS *	pCSLFileStats)
{
	RCODE						rc = NE_XFLM_OK;
	XFLM_DB_STATS *		pDbStat;
	XFLM_LFILE_STATS *	pLFileStat;

	if (ui32DatabaseNum > pStats->uiNumDbStats - 1)
	{
		rc = RC_SET( NE_XFLM_INVALID_PARM);
		goto Exit;
	}
	pDbStat = &pStats->pDbStats [ui32DatabaseNum];
	if ((FLMUINT)ui32LFileNum > pDbStat->uiNumLFileStats)
	{
		rc = RC_SET( NE_XFLM_INVALID_PARM);
		goto Exit;
	}
	pLFileStat = &pDbStat->pLFileStats [ui32LFileNum];
	f_memcpy( &pCSLFileStats->RootBlockStats, &pLFileStat->RootBlockStats, sizeof( XFLM_BLOCKIO_STATS));
	f_memcpy( &pCSLFileStats->MiddleBlockStats, &pLFileStat->MiddleBlockStats, sizeof( XFLM_BLOCKIO_STATS));
	f_memcpy( &pCSLFileStats->LeafBlockStats, &pLFileStat->LeafBlockStats, sizeof( XFLM_BLOCKIO_STATS));
	pCSLFileStats->ui64BlockSplits = pLFileStat->ui64BlockSplits;
	pCSLFileStats->ui64BlockCombines = pLFileStat->ui64BlockCombines;
	pCSLFileStats->ui32LFileNum = (FLMUINT32)pLFileStat->uiLFileNum;
	pCSLFileStats->i32LfType = (FLMINT32)pLFileStat->eLfType;

Exit:

	return( rc);
}
