//------------------------------------------------------------------------------
// Desc:	This file contains routines for updating FLAIM statistics.
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

FSTATIC RCODE flmStatGetDbByName(
	XFLM_STATS *			pFlmStats,
	char *					pszDbName,
	FLMUINT					uiLowStart,
	XFLM_DB_STATS **		ppDbStatsRV,
	FLMUINT *				puiDBAllocSeqRV,
	FLMUINT *				puiDbTblPosRV);

FSTATIC void flmUpdateLFileStats(
	XFLM_LFILE_STATS *	pDest,
	XFLM_LFILE_STATS *	pSrc);

FSTATIC RCODE flmUpdateDbStats(
	XFLM_DB_STATS *		pDest,
	XFLM_DB_STATS *		pSrc);

FSTATIC FLMUINT flmDaysInMonth(
	FLMUINT					uiYear,
	FLMUINT					uiMonth);

FSTATIC void flmAdjustTime(
	F_TMSTAMP *				pTime,
	FLMINT					iStartPoint);

/****************************************************************************
Desc:	This routine returns a pointer to a particular database's
		statistics block.
****************************************************************************/
FSTATIC RCODE flmStatGetDbByName(
	XFLM_STATS *			pFlmStats,
	char *					pszDbName,
	FLMUINT					uiLowStart,
	XFLM_DB_STATS **		ppDbStatsRV,
	FLMUINT *				puiDBAllocSeqRV,
	FLMUINT *				puiDbTblPosRV)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiTblSize;
	XFLM_DB_STATS *	pDbStatTbl;
	FLMUINT				uiLow;
	FLMUINT				uiMid = 0;
	FLMUINT				uiHigh;
	FLMINT				iCmp = 0;
	FLMUINT				uiNewSize;
	FLMUINT				uiElement;
	char *				pszTmpDbName = NULL;

	// If there is a database array, search it first.

	if (((pDbStatTbl = pFlmStats->pDbStats) != NULL) &&
		 ((uiTblSize = pFlmStats->uiNumDbStats) != 0))
	{
		for (uiHigh = --uiTblSize, uiLow = uiLowStart ; ; )		// Optimize: reduce uiTblSize
		{
			uiMid = (uiLow + uiHigh) >> 1;		// (uiLow + uiHigh) / 2

			// See if we found a match.

#ifdef FLM_UNIX
			if ((iCmp = f_strcmp( pszDbName,
								pDbStatTbl [uiMid].pszDbName)) == 0)
#else
			if ((iCmp = f_stricmp( pszDbName,
								pDbStatTbl [uiMid].pszDbName)) == 0)
#endif
			{

				// Found match.

				*ppDbStatsRV = &pDbStatTbl [uiMid];
				if (puiDBAllocSeqRV)
				{
					*puiDBAllocSeqRV = pFlmStats->uiDBAllocSeq;
				}
				if (puiDbTblPosRV)
				{
					*puiDbTblPosRV = uiMid;
				}
				goto Exit;
			}

			// Check if we are done - where uiLow equals uiHigh.

			if (uiLow >= uiHigh)
			{

				// Item not found.

				break;
			}

			if (iCmp < 0)
			{
				if (uiMid == uiLowStart)
				{
					break;					// Way too high?
				}
				uiHigh = uiMid - 1;		// Too high
			}
			else
			{
				if (uiMid == uiTblSize)
				{
					break;				// Done - Hit the top
				}
				uiLow = uiMid + 1;		// Too low
			}
		}
	}

	// If the array is full, or was never allocated, allocate a new one.

	if (pFlmStats->uiDbStatArraySize <= pFlmStats->uiNumDbStats)
	{
		if (!pFlmStats->pDbStats)
		{
			uiNewSize = INIT_DB_STAT_ARRAY_SIZE;
		}
		else
		{
			uiNewSize = pFlmStats->uiDbStatArraySize +
							DB_STAT_ARRAY_INCR_SIZE;
		}
		if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( XFLM_DB_STATS) * uiNewSize),
							&pDbStatTbl)))
		{
			goto Exit;
		}

		// Save whatever was in the old table, if any.

		if (pFlmStats->pDbStats && pFlmStats->uiNumDbStats)
		{
			f_memcpy( pDbStatTbl, pFlmStats->pDbStats,
					(FLMINT)(sizeof( XFLM_DB_STATS) * pFlmStats->uiNumDbStats));
		}
		if (pFlmStats->pDbStats)
		{
			f_free( &pFlmStats->pDbStats);
		}

		pFlmStats->uiDBAllocSeq++;
		pFlmStats->pDbStats = pDbStatTbl;
		pFlmStats->uiDbStatArraySize = uiNewSize;
	}

	// Allocate space for the database name

	if (RC_BAD( rc = f_alloc( f_strlen( pszDbName) + 1, &pszTmpDbName)))
	{
		goto Exit;
	}

	// Insert the item into the array.

	if (iCmp != 0)
	{
		uiElement = pFlmStats->uiNumDbStats;

		// If our new database number is greater than database number of the
		// element pointed to by uiMid, increment uiMid so that the new
		// database number will be inserted after it instead of before it.

		if (iCmp > 0)
		{
			uiMid++;
		}

		// Move everything up in the array, including the slot pointed to
		// by uiMid.

		while (uiElement > uiMid)
		{
			f_memcpy( &pDbStatTbl [uiElement], &pDbStatTbl [uiElement - 1],
						sizeof( XFLM_DB_STATS));
			uiElement--;
		}
		f_memset( &pDbStatTbl [uiMid], 0, sizeof( XFLM_DB_STATS));
	}
	pDbStatTbl [uiMid].pszDbName = pszTmpDbName;
	pszTmpDbName = NULL;
	f_strcpy( pDbStatTbl [uiMid].pszDbName, pszDbName);
	pFlmStats->uiNumDbStats++;
	*ppDbStatsRV = &pDbStatTbl [uiMid];
	if (puiDBAllocSeqRV)
	{
		*puiDBAllocSeqRV = pFlmStats->uiDBAllocSeq;
	}
	if (puiDbTblPosRV)
	{
		*puiDbTblPosRV = uiMid;
	}
Exit:
	if (pszTmpDbName)
	{
		f_free( &pszTmpDbName);
	}
	return( rc);
}

/****************************************************************************
Desc:	This routine returns a pointer to a particular database's
		statistics block.
****************************************************************************/
RCODE	flmStatGetDb(
	XFLM_STATS *		pFlmStats,
	F_Database *		pDatabase,
	FLMUINT				uiLowStart,
	XFLM_DB_STATS **	ppDbStatsRV,
	FLMUINT *			puiDBAllocSeqRV,
	FLMUINT *			puiDbTblPosRV)
{
	if (!pFlmStats)
	{
		*ppDbStatsRV = NULL;
		if (puiDBAllocSeqRV)
		{
			*puiDBAllocSeqRV = 0;
		}
		if (puiDbTblPosRV)
		{
			*puiDbTblPosRV = 0;
		}
		return( NE_XFLM_OK);
	}

	return( flmStatGetDbByName( pFlmStats, pDatabase->getDbNamePtr(),
						uiLowStart, ppDbStatsRV, puiDBAllocSeqRV,
						puiDbTblPosRV));
}

/****************************************************************************
Desc:	This routine returns a pointer to a particular logical file in a
		particular database's statistics block.
****************************************************************************/
RCODE	flmStatGetLFile(
	XFLM_DB_STATS *		pDbStats,
	FLMUINT					uiLFileNum,
	eLFileType				eLfType,
	FLMUINT					uiLowStart,
	XFLM_LFILE_STATS **	ppLFileStatsRV,
	FLMUINT *				puiLFileAllocSeqRV,
	FLMUINT *				puiLFileTblPosRV)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiTblSize;
	XFLM_LFILE_STATS *	pLFileStatTbl;
	XFLM_LFILE_STATS *	pLFileCurrStat;
	FLMUINT					uiLow;
	FLMUINT					uiMid = 0;
	FLMUINT					uiHigh;
	FLMINT					iCmp = 0;
	FLMUINT					uiNewSize;
	FLMUINT					uiElement;

	if (!pDbStats)
	{
		*ppLFileStatsRV = NULL;
		if (puiLFileAllocSeqRV)
		{
			*puiLFileAllocSeqRV = 0;
		}
		if (puiLFileTblPosRV)
		{
			*puiLFileTblPosRV = 0;
		}
		goto Exit;
	}

	// If there is a database array, search it first.

	if (((pLFileStatTbl = pDbStats->pLFileStats) != NULL) &&
		 ((uiTblSize = pDbStats->uiNumLFileStats) != 0))
	{
		for (uiHigh = --uiTblSize, uiLow = uiLowStart ; ; )		// Optimize: reduce uiTblSize
		{
			uiMid = (uiLow + uiHigh) >> 1;		// (uiLow + uiHigh) / 2

			// See if we found a match.

			pLFileCurrStat = &pLFileStatTbl [uiMid];
			if (uiLFileNum < pLFileCurrStat->uiLFileNum)
			{
				iCmp = -1;
			}
			else if (uiLFileNum > pLFileCurrStat->uiLFileNum)
			{
				iCmp = 1;
			}
			else if (eLfType < pLFileCurrStat->eLfType)
			{
				iCmp = -1;
			}
			else if (eLfType > pLFileCurrStat->eLfType)
			{
				iCmp = 1;
			}
			else
			{

				// Found match.

				*ppLFileStatsRV = pLFileCurrStat;
				if (puiLFileAllocSeqRV)
				{
					*puiLFileAllocSeqRV = pDbStats->uiLFileAllocSeq;
				}
				if (puiLFileTblPosRV)
				{
					*puiLFileTblPosRV = uiMid;
				}
				goto Exit;
			}

			// Check if we are done - where uiLow equals uiHigh.

			if (uiLow >= uiHigh)
			{

				// Item not found.

				break;
			}

			if (iCmp < 0)
			{
				if (uiMid == uiLowStart)
				{
					break;				// Way too high?
				}
				uiHigh = uiMid - 1;		// Too high
			}
			else
			{
				if (uiMid == uiTblSize)
				{
					break;				// Done - Hit the top
				}
				uiLow = uiMid + 1;		// Too low
			}
		}
	}

	// If the array is full, or was never allocated, allocate a new one.

	if (pDbStats->uiLFileStatArraySize <= pDbStats->uiNumLFileStats)
	{
		if (!pDbStats->pLFileStats)
		{
			uiNewSize = INIT_LFILE_STAT_ARRAY_SIZE;
		}
		else
		{
			uiNewSize = pDbStats->uiLFileStatArraySize +
							LFILE_STAT_ARRAY_INCR_SIZE;
		}
		if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( XFLM_LFILE_STATS) * uiNewSize),
							&pLFileStatTbl)))
		{
			goto Exit;
		}

		// Save whatever was in the old table, if any.

		if ((pDbStats->pLFileStats) &&
			 (pDbStats->uiNumLFileStats))
		{
			f_memcpy( pLFileStatTbl, pDbStats->pLFileStats,
					(FLMUINT)(sizeof( XFLM_LFILE_STATS) *
							 pDbStats->uiNumLFileStats));
		}
		if (pDbStats->pLFileStats)
		{
			f_free( &pDbStats->pLFileStats);
		}

		pDbStats->uiLFileAllocSeq++;
		pDbStats->pLFileStats = pLFileStatTbl;
		pDbStats->uiLFileStatArraySize = uiNewSize;
	}

	// Insert the item into the array.

	if (iCmp != 0)
	{
		uiElement = pDbStats->uiNumLFileStats;

		// If our new database number is greater than database number of the
		// element pointed to by uiMid, increment uiMid so that the new
		// database number will be inserted after it instead of before it.

		if (iCmp > 0)
		{
			uiMid++;
		}

		// Move everything up in the array, including the slot pointed to
		// by uiMid.

		while (uiElement > uiMid)
		{
			f_memcpy( &pLFileStatTbl [uiElement], &pLFileStatTbl [uiElement - 1],
						sizeof( XFLM_LFILE_STATS));
			uiElement--;
		}
		f_memset( &pLFileStatTbl [uiMid], 0, sizeof( XFLM_LFILE_STATS));
	}
	pLFileStatTbl [uiMid].uiLFileNum = uiLFileNum;
	pLFileStatTbl [uiMid].eLfType = eLfType;
	pDbStats->uiNumLFileStats++;
	*ppLFileStatsRV = &pLFileStatTbl [uiMid];
	if (puiLFileAllocSeqRV)
	{
		*puiLFileAllocSeqRV = pDbStats->uiLFileAllocSeq;
	}
	if (puiLFileTblPosRV)
	{
		*puiLFileTblPosRV = uiMid;
	}
Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine returns a pointer to a particular LFILE's
		statistics block.
****************************************************************************/
XFLM_LFILE_STATS * F_Db::getLFileStatPtr(
	LFILE *	pLFile)
{
	if (!pLFile)
	{
		return( (XFLM_LFILE_STATS *)NULL);
	}

	if ((!m_pLFileStats) ||
		 (m_uiLFileAllocSeq !=
			 m_pDbStats->uiLFileAllocSeq) ||
		 (m_pLFileStats->uiLFileNum != pLFile->uiLfNum))
	{
		if (RC_BAD( flmStatGetLFile( m_pDbStats, pLFile->uiLfNum,
									pLFile->eLfType, 0, &m_pLFileStats,
									&m_uiLFileAllocSeq, NULL)))
		{
			m_pLFileStats = NULL;
			m_uiLFileAllocSeq = 0;
		}
	}
	return( m_pLFileStats);
}

/****************************************************************************
Desc:	This routine resets the statistics in a FLM_STAT structure.
****************************************************************************/
void flmStatReset(
	XFLM_STATS *	pStats,
	FLMBOOL			bFree)
{
	FLMUINT					uiDb;
	XFLM_DB_STATS	*		pDbStats;
	FLMUINT					uiLFile;
	XFLM_LFILE_STATS *	pLFile;

	if ((pDbStats = pStats->pDbStats) != NULL)
	{
		for (uiDb = 0; uiDb < pStats->uiNumDbStats; uiDb++, pDbStats++)
		{
			if ((pLFile = pDbStats->pLFileStats) != NULL)
			{
				if (bFree)
				{
					f_free( &pDbStats->pLFileStats);
				}
				else
				{
					for (uiLFile = 0;
						  uiLFile < pDbStats->uiNumLFileStats;
						  uiLFile++, pLFile++)
					{
						FLMUINT		uiSaveLFileNum = pLFile->uiLFileNum;
						eLFileType	eSaveLfType = pLFile->eLfType;

						f_memset( pLFile, 0, sizeof( XFLM_LFILE_STATS));
						pLFile->uiLFileNum = uiSaveLFileNum;
						pLFile->eLfType = eSaveLfType;
					}
				}
			}
			if (!bFree)
			{
				char *					pszSaveDbName;
				FLMUINT					uiSaveLFileAllocSeq = pDbStats->uiLFileAllocSeq;
				XFLM_LFILE_STATS *	pSaveLFileStats = pDbStats->pLFileStats;
				FLMUINT					uiSaveLFileStatArraySize =
												pDbStats->uiLFileStatArraySize;
				FLMUINT					uiSaveNumLFileStats = pDbStats->uiNumLFileStats;

				pszSaveDbName = pDbStats->pszDbName;
				f_memset( pDbStats, 0, sizeof( XFLM_DB_STATS));
				pDbStats->pszDbName = pszSaveDbName;
				pDbStats->uiLFileAllocSeq = uiSaveLFileAllocSeq;
				pDbStats->pLFileStats = pSaveLFileStats;
				pDbStats->uiLFileStatArraySize = uiSaveLFileStatArraySize;
				pDbStats->uiNumLFileStats = uiSaveNumLFileStats;
			}
			else
			{
				f_free( &pDbStats->pszDbName);
			}
		}
		if (bFree)
		{
			f_free( &pStats->pDbStats);
		}
	}
	if ((bFree) || (!pDbStats))
	{
		pStats->pDbStats = NULL;
		pStats->uiDbStatArraySize = 0;
		pStats->uiNumDbStats = 0;
	}
	pStats->uiStartTime = 0;
	pStats->uiStopTime = 0;

	if (pStats->bCollectingStats)
	{
		f_timeGetSeconds( &pStats->uiStartTime);
	}
}

/****************************************************************************
Desc:	This routine updates statistics from one XFLM_RTRANS_STATS structure
		into another.
****************************************************************************/
FINLINE void flmUpdateRTransStats(
	XFLM_RTRANS_STATS *	pDest,
	XFLM_RTRANS_STATS *	pSrc)
{
	flmUpdateCountTimeStats( &pDest->CommittedTrans, &pSrc->CommittedTrans);
	flmUpdateCountTimeStats( &pDest->AbortedTrans, &pSrc->AbortedTrans);
}

/****************************************************************************
Desc:	This routine updates statistics from one XFLM_UTRANS_STATS structure
		into another.
****************************************************************************/
FINLINE void flmUpdateUTransStats(
	XFLM_UTRANS_STATS *	pDest,
	XFLM_UTRANS_STATS *	pSrc)
{
	flmUpdateCountTimeStats( &pDest->CommittedTrans, &pSrc->CommittedTrans);
	flmUpdateCountTimeStats( &pDest->GroupCompletes, &pSrc->GroupCompletes);
	pDest->ui64GroupFinished += pSrc->ui64GroupFinished;
	flmUpdateCountTimeStats( &pDest->AbortedTrans, &pSrc->AbortedTrans);
}

/****************************************************************************
Desc:	This routine updates statistics from one XFLM_BLOCKIO_STATS structure
		into another.
****************************************************************************/
void flmUpdateBlockIOStats(
	XFLM_BLOCKIO_STATS *	pDest,
	XFLM_BLOCKIO_STATS *	pSrc)
{
	flmUpdateDiskIOStats( &pDest->BlockReads, &pSrc->BlockReads);
	flmUpdateDiskIOStats( &pDest->OldViewBlockReads,
									&pSrc->OldViewBlockReads);
	pDest->ui32BlockChkErrs += pSrc->ui32BlockChkErrs;
	pDest->ui32OldViewBlockChkErrs += pSrc->ui32OldViewBlockChkErrs;
	pDest->ui32OldViewErrors += pSrc->ui32OldViewErrors;
	flmUpdateDiskIOStats( &pDest->BlockWrites, &pSrc->BlockWrites);
}

/****************************************************************************
Desc:	This routine updates statistics from one XFLM_LFILE_STATS structure
		into another.
****************************************************************************/
FSTATIC void flmUpdateLFileStats(
	XFLM_LFILE_STATS *	pDest,
	XFLM_LFILE_STATS *	pSrc)
{
	pDest->bHaveStats = TRUE;
	flmUpdateBlockIOStats( &pDest->RootBlockStats, &pSrc->RootBlockStats);
	flmUpdateBlockIOStats( &pDest->MiddleBlockStats, &pSrc->MiddleBlockStats);
	flmUpdateBlockIOStats( &pDest->LeafBlockStats, &pSrc->LeafBlockStats);
	pDest->ui64BlockSplits += pSrc->ui64BlockSplits;
	pDest->ui64BlockCombines += pSrc->ui64BlockCombines;
}

/****************************************************************************
Desc:	This routine updates statistics from one XFLM_DB_STATS structure into
		another.
****************************************************************************/
FSTATIC RCODE flmUpdateDbStats(
	XFLM_DB_STATS *	pDestDb,
	XFLM_DB_STATS *	pSrcDb)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiSrcLFile;
	FLMUINT					uiDestLFile;
	XFLM_LFILE_STATS *	pDestLFile;
	XFLM_LFILE_STATS *	pSrcLFile;
	FLMUINT					uiLowLFileStart;
	FLMUINT					uiSaveNumLFiles;

	flmUpdateRTransStats( &pDestDb->ReadTransStats,
								 &pSrcDb->ReadTransStats);
	flmUpdateUTransStats( &pDestDb->UpdateTransStats,
								 &pSrcDb->UpdateTransStats);
	pDestDb->bHaveStats = TRUE;
	flmUpdateBlockIOStats( &pDestDb->LFHBlockStats,
								&pSrcDb->LFHBlockStats);
	flmUpdateBlockIOStats( &pDestDb->AvailBlockStats,
								&pSrcDb->AvailBlockStats);
	flmUpdateDiskIOStats( &pDestDb->DbHdrWrites,
									 &pSrcDb->DbHdrWrites);
	flmUpdateDiskIOStats( &pDestDb->LogBlockWrites,
									 &pSrcDb->LogBlockWrites);
	flmUpdateDiskIOStats( &pDestDb->LogBlockRestores,
									 &pSrcDb->LogBlockRestores);
	flmUpdateDiskIOStats( &pDestDb->LogBlockReads,
									 &pSrcDb->LogBlockReads);
	pDestDb->uiLogBlockChkErrs += pSrcDb->uiLogBlockChkErrs;
	pDestDb->uiReadErrors += pSrcDb->uiReadErrors;
	pDestDb->uiWriteErrors += pSrcDb->uiWriteErrors;
	flmUpdateCountTimeStats( &pDestDb->LockStats.NoLocks, 
									 &pSrcDb->LockStats.NoLocks);
	flmUpdateCountTimeStats( &pDestDb->LockStats.WaitingForLock,
									 &pSrcDb->LockStats.WaitingForLock);
	flmUpdateCountTimeStats( &pDestDb->LockStats.HeldLock,
									 &pSrcDb->LockStats.HeldLock);

	// Go through the LFILE statistics.

	for (uiDestLFile = 0, uiSrcLFile = 0, uiLowLFileStart = 0,
			pSrcLFile = pSrcDb->pLFileStats;
			uiSrcLFile < pSrcDb->uiNumLFileStats;
			uiSrcLFile++, pSrcLFile++)
	{
		if (!pSrcLFile->bHaveStats)
		{
			continue;
		}

		// Find or add the store in the destination store array.

		uiSaveNumLFiles = pDestDb->uiNumLFileStats;
		if (RC_BAD( rc = flmStatGetLFile( pDestDb, pSrcLFile->uiLFileNum,
									pSrcLFile->eLfType,
									uiLowLFileStart, &pDestLFile, NULL,
									&uiLowLFileStart)))
		{
			goto Exit;
		}

		if (uiLowLFileStart < pDestDb->uiNumLFileStats - 1)
		{
			uiLowLFileStart++;
		}

		// If we created the LFILE, all we have to do is copy the
		// LFILE statistics.  It will be quicker.

		if (uiSaveNumLFiles != pDestDb->uiNumLFileStats)
		{
			f_memcpy( pDestLFile, pSrcLFile, sizeof( XFLM_LFILE_STATS));
		}
		else
		{

			// LFILE was already present, need to go through and
			// update the statistics.

			flmUpdateLFileStats( pDestLFile, pSrcLFile);
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine updates statistics from one FLM_STAT structure into
		the global statistics.
****************************************************************************/
RCODE	flmStatUpdate(
	XFLM_STATS *	pSrcStats)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiSrcDb;
	FLMUINT				uiDestDb;
	XFLM_DB_STATS *	pDestDb;
	XFLM_DB_STATS *	pSrcDb;
	FLMUINT				uiLowDbStart;

	// Do not update the statistics if the source statistics were started
	// at an earlier time that the destination statistics start time.

	if (!gv_XFlmSysData.Stats.bCollectingStats ||
		 pSrcStats->uiStartTime < gv_XFlmSysData.Stats.uiStartTime)
	{
		return( NE_XFLM_OK);
	}

	f_mutexLock( gv_XFlmSysData.hStatsMutex);

	// Go through each of the source's databases

	for ( uiDestDb = 0, uiSrcDb = 0, uiLowDbStart = 0,
			pSrcDb = pSrcStats->pDbStats;
			uiSrcDb < pSrcStats->uiNumDbStats;
			uiSrcDb++, pSrcDb++)
	{
		if (!pSrcDb->bHaveStats)
		{
			continue;
		}

		// Find or add the store in the destination store array.

		if (RC_BAD( rc = flmStatGetDbByName( &gv_XFlmSysData.Stats,
									pSrcDb->pszDbName,
									uiLowDbStart, &pDestDb, NULL,
									&uiLowDbStart)))
		{
			goto Exit;
		}

		if (uiLowDbStart < gv_XFlmSysData.Stats.uiNumDbStats - 1)
		{
			uiLowDbStart++;
		}

		if (RC_BAD( rc = flmUpdateDbStats( pDestDb, pSrcDb)))
		{
			goto Exit;
		}
	}

Exit:

	f_mutexUnlock( gv_XFlmSysData.hStatsMutex);

	if (RC_OK( rc))
	{
		flmStatReset( pSrcStats, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine frees however many queries that are over the limit of
		the number we can save.
Note:	This routine will ALWAYS unlock the query mutex on leaving.  It may
		be locked when entering.
****************************************************************************/
void flmFreeSavedQueries(
	FLMBOOL	bMutexAlreadyLocked)
{
	QUERY_HDR *	pQueriesToFree = NULL;

	// Must determine the queries to free inside the mutex lock and then free
	// them outside the mutex lock, because freeing a query may cause an
	// embedded query to be put into the list again, which will cause the
	// mutex to be locked again.

	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hQueryMutex);
	}
	while (gv_XFlmSysData.uiQueryCnt > gv_XFlmSysData.uiMaxQueries)
	{
		gv_XFlmSysData.pOldestQuery = gv_XFlmSysData.pOldestQuery->pPrev;
		gv_XFlmSysData.uiQueryCnt--;
	}

	// Whatever is found after pOldestQuery should be freed.  Unlink
	// those from the list and point to them with pQueriesToFree.

	if (!gv_XFlmSysData.pOldestQuery)
	{
		pQueriesToFree = gv_XFlmSysData.pNewestQuery;
		gv_XFlmSysData.pNewestQuery = NULL;
	}
	else if (gv_XFlmSysData.pOldestQuery->pNext)
	{
		pQueriesToFree = gv_XFlmSysData.pOldestQuery->pNext;
		pQueriesToFree->pPrev = NULL;
		gv_XFlmSysData.pOldestQuery->pNext = NULL;
	}
	f_mutexUnlock( gv_XFlmSysData.hQueryMutex);

	// Now clean up each of the queries in the pQueriesToFree list.
	// This can be done outside the mutex lock because we are now
	// dealing with a completely local list that no other thread can
	// see.  Also, the mutex must NOT be locked at this point because
	// FlmCursorCleanup may free an embedded query, which will want
	// to lock the mutex again.

	while (pQueriesToFree)
	{
		QUERY_HDR *	pQueryHdrToFree = pQueriesToFree;

		pQueriesToFree = pQueriesToFree->pNext;
// VISIT
		flmAssert( 0);
//		pQueryHdrToFree->pCursor->cleanup();
		f_free( &pQueryHdrToFree);
	}
}

/****************************************************************************
Desc:	This routine saves a query so it can be analyzed later.
****************************************************************************/
void flmSaveQuery(
	F_Query *	pQuery)
{
	QUERY_HDR *	pQueryHdr = NULL;
	FLMBOOL		bNeedToCleanup = TRUE;
	FLMBOOL		bMutexLocked = FALSE;

	// Allocate memory for the new query

	if (RC_BAD( f_calloc( sizeof( QUERY_HDR), &pQueryHdr)))
	{
		goto Exit;
	}

	pQueryHdr->pQuery = pQuery;

	f_mutexLock( gv_XFlmSysData.hQueryMutex);
	bMutexLocked = TRUE;

	// uiMaxQueries was originally checked outside of the mutex lock.
	// Make sure it is still non-zero.

	if (gv_XFlmSysData.uiMaxQueries)
	{

		// Link query to head of list.

		bNeedToCleanup = FALSE;
		if ((pQueryHdr->pNext = gv_XFlmSysData.pNewestQuery) != NULL)
		{
			pQueryHdr->pNext->pPrev = pQueryHdr;
		}
		else
		{
			gv_XFlmSysData.pOldestQuery = pQueryHdr;
		}
		gv_XFlmSysData.pNewestQuery = pQueryHdr;

		if (++gv_XFlmSysData.uiQueryCnt > gv_XFlmSysData.uiMaxQueries)
		{
			flmFreeSavedQueries( TRUE);

			// flmFreeSavedQueries will always unlock the mutex.

			bMutexLocked = FALSE;
		}
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hQueryMutex);
	}

	// Must clean up the query if we didn't get it into the list for
	// some reason.

	if (bNeedToCleanup)
	{
		if (pQueryHdr)
		{
			f_free( &pQueryHdr);
		}
// VISIT
		flmAssert( 0);
//		pCursor->cleanup();
	}
}

/****************************************************************************
Desc:	This routine copies statistics from one FLM_STAT structure into
		another.  This is used to retrieve statistics.
****************************************************************************/
RCODE	flmStatCopy(
	XFLM_STATS *		pDestStats,
	XFLM_STATS *		pSrcStats)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiDb;
	XFLM_DB_STATS *		pDestDb;
	XFLM_DB_STATS *		pSrcDb;
	FLMUINT					uiCount;
	FLMUINT					uiLoop;
	XFLM_DB_STATS *		pDbStats;
	XFLM_LFILE_STATS *	pLFile;

	f_memcpy( pDestStats, pSrcStats, sizeof( XFLM_STATS));

	// Zero out the database array.  We need to do this
	// so that if we get an error, we can correctly release memory for
	// the destination structure.

	pDestStats->uiNumDbStats = 0;
	pDestStats->uiDbStatArraySize = 0;
	pDestStats->pDbStats = NULL;
	uiCount = 0;
	if (pSrcStats->uiNumDbStats)
	{
		for (uiLoop = 0, pDbStats = pSrcStats->pDbStats;
			  uiLoop < pSrcStats->uiNumDbStats;
			  uiLoop++, pDbStats++)
		{
			if (pDbStats->bHaveStats)
			{
				uiCount++;
			}
		}
	}
	if (uiCount)
	{
		if (RC_BAD( rc = f_calloc( (FLMUINT)sizeof( XFLM_DB_STATS) * uiCount,
										&pDestStats->pDbStats)))
		{
			goto Exit;
		}
		for (uiLoop = 0, uiCount = 0, pDbStats = pSrcStats->pDbStats;
			  uiLoop < pSrcStats->uiNumDbStats;
			  uiLoop++, pDbStats++)
		{
			if (pDbStats->bHaveStats)
			{
				pDestDb = &pDestStats->pDbStats [uiCount];
				f_memcpy( pDestDb, pDbStats, sizeof( XFLM_DB_STATS));

				// Allocate space for the database name

				if (RC_BAD( rc = f_alloc( f_strlen( pDbStats->pszDbName) + 1,
										   &pDestDb->pszDbName)))
				{
					goto Exit;
				}
				f_strcpy( pDestDb->pszDbName, pDbStats->pszDbName);

				// Zero out each store's LFILE statistics.  We need to do this
				// so that if we get an error, we can correctly release memory for
				// the destination structure.

				pDestDb->uiNumLFileStats = 0;
				pDestDb->uiLFileStatArraySize = 0;
				pDestDb->pLFileStats = NULL;
				uiCount++;
			}
		}
		pDestStats->uiDbStatArraySize =
		pDestStats->uiNumDbStats = uiCount;
	}

	for (uiDb = pSrcStats->uiNumDbStats,
			pDestDb = pDestStats->pDbStats,
			pSrcDb = pSrcStats->pDbStats;
		  uiDb;
		  uiDb--, pSrcDb++)
	{
		if (!pSrcDb->bHaveStats)
		{
			continue;
		}

		pDestDb->uiNumLFileStats = 0;
		pDestDb->uiLFileStatArraySize = 0;
		pDestDb->pLFileStats = NULL;
		uiCount = 0;
		for (uiLoop = 0, pLFile = pSrcDb->pLFileStats;
			  uiLoop < pSrcDb->uiNumLFileStats;
			  uiLoop++, pLFile++)
		{
			if (pLFile->bHaveStats)
			{
				uiCount++;
			}
		}
		if (uiCount)
		{
			if (RC_BAD( rc = f_calloc(
										(FLMUINT)sizeof( XFLM_LFILE_STATS) * uiCount,
										&pDestDb->pLFileStats)))
			{
				goto Exit;
			}
			uiCount = 0;
			for (uiLoop = 0, pLFile = pSrcDb->pLFileStats;
			  	uiLoop < pSrcDb->uiNumLFileStats;
			  	uiLoop++, pLFile++)
			{
				if (pLFile->bHaveStats)
				{
					f_memcpy( &pDestDb->pLFileStats [uiCount],
								pLFile, sizeof( XFLM_LFILE_STATS));
					uiCount++;
				}
			}
			pDestDb->uiLFileStatArraySize =
			pDestDb->uiNumLFileStats = uiCount;
		}
		pDestDb++;
	}

Exit:

	if (RC_BAD( rc))
	{
		flmStatFree( pDestStats);
	}
	return( rc);
}

/****************************************************************************
Desc:	This routine locates the appropriate XFLM_BLOCKIO_STATS structure for
		a given block of data.  NULL is returned if an appropriate one cannot
		be found.
VISIT: uiBlkType passed in is a guess.  Remove this parm and start using
		 pBlk.
****************************************************************************/
XFLM_BLOCKIO_STATS * flmGetBlockIOStatPtr(
	XFLM_DB_STATS *		pDbStats,
	XFLM_LFILE_STATS *	pLFileStats,
	FLMBYTE *				pucBlk)
{
	F_BLK_HDR *	pBlkHdr = (F_BLK_HDR *)pucBlk;

	if (pBlkHdr->ui8BlkType == BT_FREE)
	{
		pDbStats->bHaveStats = TRUE;
		return( &pDbStats->AvailBlockStats);
	}
	else if (pBlkHdr->ui8BlkType == BT_LFH_BLK)
	{
		pDbStats->bHaveStats = TRUE;
		return( &pDbStats->LFHBlockStats);
	}
	else if (pLFileStats)
	{
		pLFileStats->bHaveStats =
		pDbStats->bHaveStats = TRUE;

		// VISIT: Consider a one level tree.
		// Is it more important to count root stats over leaf stats?
		// What about the Data Only Blocks?

		// Consider invalid type.

		if (pBlkHdr->ui8BlkType != BT_LEAF &&
			 pBlkHdr->ui8BlkType != BT_NON_LEAF &&
			 pBlkHdr->ui8BlkType != BT_NON_LEAF_COUNTS &&
			 pBlkHdr->ui8BlkType != BT_LEAF_DATA)
		{
			return( &pLFileStats->LeafBlockStats);
		}
		if (pBlkHdr->ui32NextBlkInChain == 0 &&
			 pBlkHdr->ui32PrevBlkInChain == 0)
		{
			return( &pLFileStats->RootBlockStats);
		}
		else if ((pBlkHdr->ui8BlkType != BT_LEAF) &&
			      (pBlkHdr->ui8BlkType != BT_LEAF_DATA))
		{
			return( &pLFileStats->MiddleBlockStats);
		}
		else
		{
			return( &pLFileStats->LeafBlockStats);
		}
	}
	else
	{
		return( (XFLM_BLOCKIO_STATS *)NULL);
	}
}

/********************************************************************
Desc: Determine if a given year is a leap year.
*********************************************************************/
FINLINE FLMUINT flmLeapYear(
	FLMUINT	uiYear
	)
{
	if (uiYear % 4 != 0)
	{
		return( 0);
	}
	if (uiYear % 100 != 0)
	{
		return( 1);
	}
	if (uiYear % 400 != 0)
	{
		return( 0);
	}
	return( 1);
}

/********************************************************************
Desc: Calculate days in a given month of a given year.
*********************************************************************/
FSTATIC FLMUINT flmDaysInMonth(
	FLMUINT	uiYear,
	FLMUINT	uiMonth
	)
{
	switch (uiMonth + 1)
	{
		case 4:
		case 6:
		case 9:
		case 11:
			return( 30);
		case 2:
			return( 28 + flmLeapYear( uiYear));
		default:
			return( 31);
	}
}

/********************************************************************
Desc: Adjust the time.
*********************************************************************/
FSTATIC void flmAdjustTime(
	F_TMSTAMP *	pTime,
	FLMINT		iStartPoint
	)
{
	switch (iStartPoint)
	{
		case 1:
			goto Adj_1;
		case 2:
			goto Adj_2;
		case 3:
			goto Adj_3;
		case 4:
			goto Adj_4;
		case 5:
			goto Adj_5;
		case 6:
			goto Adj_6;
	}
Adj_1:
	if (pTime->hundredth >= 100)
	{
		pTime->second++;
		pTime->hundredth = 0;
	}
Adj_2:
	if (pTime->second == 60)
	{
		pTime->minute++;
		pTime->second = 0;
	}
Adj_3:
	if (pTime->minute == 60)
	{
		pTime->hour++;
		pTime->minute = 0;
	}
Adj_4:
	if (pTime->hour == 24)
	{
		pTime->day++;
		pTime->hour = 0;
	}
Adj_5:
	if ((FLMUINT)pTime->day > flmDaysInMonth( pTime->year, pTime->month))
	{
		pTime->month++;
		pTime->day = 1;
	}
Adj_6:
	if (pTime->month > 11)
	{
		pTime->year++;
		pTime->month = 1;
	}
}

/********************************************************************
Desc: Calculate the elapsed time, including milliseconds.
*********************************************************************/
void flmAddElapTime(
	F_TMSTAMP *	pStartTime,
	FLMUINT64 *	pui64ElapMilli
	)
{
	F_TMSTAMP	StartTime;
	F_TMSTAMP	EndTime;
	FLMUINT		uiSec = 0;
	FLMUINT		uiHundredth = 0;

	f_timeGetTimeStamp( &EndTime);
	f_memcpy( &StartTime, pStartTime, sizeof( F_TMSTAMP));

	if (StartTime.year < EndTime.year)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			flmAdjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			flmAdjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			flmAdjustTime( &StartTime, 5);
		}
		if (StartTime.day > 1)
		{
			uiSec += (FLMUINT)(flmDaysInMonth( StartTime.year, StartTime.month) -
									StartTime.day + 1) * (FLMUINT)86400;
			StartTime.day = 1;
			StartTime.month++;
			flmAdjustTime( &StartTime, 6);
		}
		if (StartTime.month > 1)
		{
			while (StartTime.month <= 11)
			{
				uiSec += (FLMUINT)((FLMUINT)flmDaysInMonth( StartTime.year,
										StartTime.month) * (FLMUINT)86400);
				StartTime.month++;
			}
			StartTime.year++;
		}
		while (StartTime.year < EndTime.year)
		{
			uiSec += (FLMUINT)((FLMUINT)(365 + flmLeapYear( StartTime.year)) *
							(FLMUINT)86400);
			StartTime.year++;
		}
	}

	if (StartTime.month < EndTime.month)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			flmAdjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			flmAdjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			flmAdjustTime( &StartTime, 5);
		}
		if (StartTime.day > 1)
		{
			uiSec += (FLMUINT)(flmDaysInMonth( StartTime.year, StartTime.month) -
									StartTime.day + 1) * (FLMUINT)86400;
			StartTime.day = 1;
			StartTime.month++;
			flmAdjustTime( &StartTime, 6);
		}
		while (StartTime.month < EndTime.month)
		{
			uiSec += (FLMUINT)((FLMUINT)flmDaysInMonth( StartTime.year,
									StartTime.month) * (FLMUINT)86400);
			StartTime.month++;
		}
	}

	if (StartTime.day < EndTime.day)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			flmAdjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			flmAdjustTime( &StartTime, 4);
		}
		if (StartTime.hour)
		{
			uiSec += (FLMUINT)((24 - StartTime.hour) * 3600);
			StartTime.hour = 0;
			StartTime.day++;
			flmAdjustTime( &StartTime, 5);
		}
		uiSec += (FLMUINT)(EndTime.day - StartTime.day) * (FLMUINT)86400;
		StartTime.day = 1;
		StartTime.month++;
		flmAdjustTime( &StartTime, 6);
	}

	if (StartTime.hour < EndTime.hour)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			flmAdjustTime( &StartTime, 3);
		}
		if (StartTime.minute)
		{
			uiSec += (FLMUINT)((60 - StartTime.minute) * 60);
			StartTime.minute = 0;
			StartTime.hour++;
			flmAdjustTime( &StartTime, 4);
		}
		uiSec += (FLMUINT)((EndTime.hour - StartTime.hour) * 3600);
		StartTime.hour = 0;
		StartTime.day++;
		flmAdjustTime( &StartTime, 5);
	}

	if (StartTime.minute < EndTime.minute)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		if (StartTime.second)
		{
			uiSec += (FLMUINT)(60 - StartTime.second);
			StartTime.second = 0;
			StartTime.minute++;
			flmAdjustTime( &StartTime, 3);
		}
		uiSec += (FLMUINT)((EndTime.minute - StartTime.minute) * 60);
		StartTime.minute = 0;
		StartTime.hour++;
		flmAdjustTime( &StartTime, 4);
	}

	if (StartTime.second < EndTime.second)
	{
		if (StartTime.hundredth)
		{
			uiHundredth += (FLMUINT)(100 - StartTime.hundredth);
			StartTime.hundredth = 0;
			StartTime.second++;
			flmAdjustTime( &StartTime, 2);
		}
		uiSec += (FLMUINT)(EndTime.second - StartTime.second);
		StartTime.second = 0;
		StartTime.minute++;
		flmAdjustTime( &StartTime, 3);
	}

	if (StartTime.hundredth < EndTime.hundredth)
	{
		uiHundredth += (FLMUINT)(EndTime.hundredth - StartTime.hundredth);
	}
	if (uiSec)
	{
		(*pui64ElapMilli) += (FLMUINT64)((uiHundredth * 10 + uiSec * 1000));
	}
	else
	{
		(*pui64ElapMilli) += (FLMUINT64)(uiHundredth * 10);
	}
}
