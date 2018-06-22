//-------------------------------------------------------------------------
// Desc:	Routines for updating statistics - for monitoring.
// Tabs:	3
//
// Copyright (c) 1997-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#define INIT_DB_STAT_ARRAY_SIZE			5
#define DB_STAT_ARRAY_INCR_SIZE			5
#define INIT_LFILE_STAT_ARRAY_SIZE		5
#define LFILE_STAT_ARRAY_INCR_SIZE		5

FSTATIC RCODE flmStatGetDbByName(
	FLM_STATS *				pFlmStats,
	const char *			pszDbName,
	FLMUINT					uiLowStart,
	DB_STATS **				ppDbStatsRV,
	FLMUINT *				puiDBAllocSeqRV,
	FLMUINT *				puiDbTblPosRV);

FSTATIC void flmUpdateLFileStats(
	LFILE_STATS *	pDest,
	LFILE_STATS *	pSrc);

FSTATIC RCODE flmUpdateDbStats(
	DB_STATS *		pDest,
	DB_STATS *		pSrc);

FSTATIC RCODE flmStatCopy(
	FLM_STATS *	pDestStats,
	FLM_STATS *	pSrcStats);

FSTATIC FLMUINT flmDaysInMonth(
	FLMUINT	uiYear,
	FLMUINT	uiMonth);

FSTATIC void flmAdjustTime(
	F_TMSTAMP *	pTime,
	FLMINT		iStartPoint);

/****************************************************************************
Desc:	This routine returns a pointer to a particular database's
		statistics block.
****************************************************************************/
FSTATIC RCODE flmStatGetDbByName(
	FLM_STATS *				pFlmStats,
	const char *			pszDbName,
	FLMUINT					uiLowStart,
	DB_STATS **				ppDbStatsRV,
	FLMUINT *				puiDBAllocSeqRV,
	FLMUINT *				puiDbTblPosRV)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTblSize;
	DB_STATS *		pDbStatTbl;
	FLMUINT			uiLow;
	FLMUINT			uiMid = 0;
	FLMUINT			uiHigh;
	FLMINT			iCmp = 0;
	FLMUINT			uiNewSize;
	FLMUINT			uiElement;
	char *			pszTmpDbName = NULL;

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
		if (RC_BAD( rc = f_calloc(
						(FLMUINT)(sizeof( DB_STATS) * uiNewSize),
							&pDbStatTbl)))
		{
			goto Exit;
		}

		// Save whatever was in the old table, if any.

		if (pFlmStats->pDbStats && pFlmStats->uiNumDbStats)
		{
			f_memcpy( pDbStatTbl, pFlmStats->pDbStats,
					(FLMINT)(sizeof( DB_STATS) * pFlmStats->uiNumDbStats));
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
						sizeof( DB_STATS));
			uiElement--;
		}
		f_memset( &pDbStatTbl [uiMid], 0, sizeof( DB_STATS));
	}
	
	f_strcpy( pszTmpDbName, pszDbName);
	pDbStatTbl[ uiMid].pszDbName = pszTmpDbName;
	pszTmpDbName = NULL;
	
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
	FLM_STATS *		pFlmStats,
	void *			pFile,
	FLMUINT			uiLowStart,
	DB_STATS **		ppDbStatsRV,
	FLMUINT *		puiDBAllocSeqRV,
	FLMUINT *		puiDbTblPosRV)
{
	if( !pFlmStats)
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
		
		return( FERR_OK);
	}

	return( flmStatGetDbByName( pFlmStats, ((FFILE *)pFile)->pszDbPath,
					uiLowStart, ppDbStatsRV, puiDBAllocSeqRV,
					puiDbTblPosRV));
}

/****************************************************************************
Desc:	This routine returns a pointer to a particular logical file in a
		particular database's statistics block.
****************************************************************************/
RCODE	flmStatGetLFile(
	DB_STATS *			pDbStats,
	FLMUINT				uiLFileNum,
	FLMUINT				uiLfType,
	FLMUINT				uiLowStart,
	LFILE_STATS **		ppLFileStatsRV,
	FLMUINT *			puiLFileAllocSeqRV,
	FLMUINT *			puiLFileTblPosRV
	)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTblSize;
	LFILE_STATS *	pLFileStatTbl;
	LFILE_STATS *	pLFileCurrStat;
	FLMUINT			uiLow;
	FLMUINT			uiMid = 0;
	FLMUINT			uiHigh;
	FLMINT			iCmp = 0;
	FLMUINT			uiNewSize;
	FLMUINT			uiElement;

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
			else
			{

				// Found match.

				*ppLFileStatsRV = pLFileCurrStat;
				if (uiLfType != 0xFF)
				{
					pLFileCurrStat->uiFlags &= (~(LFILE_TYPE_UNKNOWN));
					if (uiLfType == LF_INDEX)
					{
						pLFileCurrStat->uiFlags |= LFILE_IS_INDEX;
					}
					else
					{
						pLFileCurrStat->uiFlags &= (~(LFILE_IS_INDEX));
					}
				}
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
		if (RC_BAD( rc = f_calloc( 
			(FLMUINT)(sizeof( LFILE_STATS) * uiNewSize), &pLFileStatTbl)))
		{
			goto Exit;
		}

		// Save whatever was in the old table, if any.

		if ((pDbStats->pLFileStats) &&
			 (pDbStats->uiNumLFileStats))
		{
			f_memcpy( pLFileStatTbl, pDbStats->pLFileStats,
					(FLMUINT)(sizeof( LFILE_STATS) *
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
						sizeof( LFILE_STATS));
			uiElement--;
		}
		f_memset( &pLFileStatTbl [uiMid], 0, sizeof( LFILE_STATS));
	}
	pLFileStatTbl [uiMid].uiLFileNum = uiLFileNum;
	if (uiLfType == LF_INDEX)
	{
		pLFileStatTbl [uiMid].uiFlags |= LFILE_IS_INDEX;
		pLFileStatTbl [uiMid].uiFlags &= (~(LFILE_TYPE_UNKNOWN));
	}
	else if (uiLfType == 0xFF)
	{
		pLFileStatTbl [uiMid].uiFlags |= LFILE_TYPE_UNKNOWN;
	}
	else
	{
		pLFileStatTbl [uiMid].uiFlags &=
			(~(LFILE_IS_INDEX | LFILE_TYPE_UNKNOWN));
	}
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
		statistics block.  It uses the pointer in the OPC if it
		is up-to-date.
****************************************************************************/
LFILE_STATS * fdbGetLFileStatPtr(
	FDB *		pDb,
	LFILE *	pLFile
	)
{
	if (!pLFile)
	{
		return( (LFILE_STATS *)NULL);
	}

	if ((!pDb->pLFileStats) ||
		 (pDb->uiLFileAllocSeq !=
			 pDb->pDbStats->uiLFileAllocSeq) ||
		 (pDb->pLFileStats->uiLFileNum != pLFile->uiLfNum))
	{
		if (RC_BAD( flmStatGetLFile( pDb->pDbStats, pLFile->uiLfNum,
									pLFile->uiLfType, 0, &pDb->pLFileStats,
									&pDb->uiLFileAllocSeq, NULL)))
		{
			pDb->pLFileStats = NULL;
			pDb->uiLFileAllocSeq = 0;
		}
	}
	return( pDb->pLFileStats);
}

/****************************************************************************
Desc:	This routine resets the statistics in a FLM_STAT structure.
****************************************************************************/
void flmStatReset(
	FLM_STATS *	pStats,
	FLMBOOL		bMutexAlreadyLocked,
	FLMBOOL		bFree
	)
{
	FLMUINT			uiDb;
	DB_STATS	*		pDbStats;
	FLMUINT			uiLFile;
	LFILE_STATS *	pLFile;

	// If the structure has a mutex, lock it, if not already locked.

	if (!bMutexAlreadyLocked && pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pStats->hMutex);
	}

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
						FLMUINT	uiSaveLFileNum = pLFile->uiLFileNum;
						FLMUINT	uiSaveFlags = pLFile->uiFlags;

						f_memset( pLFile, 0, sizeof( LFILE_STATS));
						pLFile->uiLFileNum = uiSaveLFileNum;
						pLFile->uiFlags = uiSaveFlags;
					}
				}
			}
			if (!bFree)
			{
				const char *	pszSaveDbName;
				FLMUINT			uiSaveLFileAllocSeq = pDbStats->uiLFileAllocSeq;
				LFILE_STATS *	pSaveLFileStats = pDbStats->pLFileStats;
				FLMUINT			uiSaveLFileStatArraySize =
										pDbStats->uiLFileStatArraySize;
				FLMUINT			uiSaveNumLFileStats = pDbStats->uiNumLFileStats;

				pszSaveDbName = pDbStats->pszDbName;
				f_memset( pDbStats, 0, sizeof( DB_STATS));
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

	// Unlock the mutex, if we locked it in this routine.

	if (!bMutexAlreadyLocked && pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pStats->hMutex);
	}
}

/****************************************************************************
Desc:	This routine starts collecting statistics in a FLM_STAT structure.
****************************************************************************/
void flmStatStart(
	FLM_STATS *	pStats
	)
{
	// If the structure has a mutex, lock it.

	if( pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pStats->hMutex);
	}

	pStats->bCollectingStats = TRUE;
	flmStatReset( pStats, TRUE, TRUE);

	// If the structure has a mutex, unlock it.

	if( pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pStats->hMutex);
	}
}

/****************************************************************************
Desc:	This routine stops collecting statistics in a FLM_STAT structure.
****************************************************************************/
void flmStatStop(
	FLM_STATS *	pStats
	)
{
	// If the structure has a mutex, lock it.

	if (pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pStats->hMutex);
	}

	if (pStats->bCollectingStats)
	{
		pStats->bCollectingStats = FALSE;
		f_timeGetSeconds( &pStats->uiStopTime);
	}

	// If the structure has a mutex, unlock it.

	if (pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pStats->hMutex);
	}
}

/****************************************************************************
Desc:	This routine initializes a FLM_STAT structure.
****************************************************************************/
RCODE flmStatInit(
	FLM_STATS *	pStats,
	FLMBOOL		bEnableSharing
	)
{
	RCODE	rc = FERR_OK;

	f_memset( pStats, 0, sizeof( FLM_STATS));
	if (!bEnableSharing)
	{
		pStats->hMutex = F_MUTEX_NULL;
	}
	else
	{
		if( RC_BAD( rc = f_mutexCreate( &pStats->hMutex)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine frees the memory associated with a FLM_STAT structure.
****************************************************************************/
FLMEXP void FLMAPI FlmFreeStats(
	FLM_STATS *	pStats
	)
{
	// If the structure has a mutex, lock it.

	if (pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pStats->hMutex);
	}
	pStats->bCollectingStats = FALSE;
	flmStatReset( pStats, TRUE, TRUE);

	// Unlock and free the mutex

	if (pStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pStats->hMutex);
		f_mutexDestroy( &pStats->hMutex);
	}
}

/****************************************************************************
Desc:	This routine updates statistics from one RTRANS_STATS structure into
		another.
****************************************************************************/
FINLINE void flmUpdateRTransStats(
	RTRANS_STATS *	pDest,
	RTRANS_STATS *	pSrc)
{
	flmUpdateCountTimeStats( &pDest->CommittedTrans, &pSrc->CommittedTrans);
	flmUpdateCountTimeStats( &pDest->AbortedTrans, &pSrc->AbortedTrans);
	flmUpdateCountTimeStats( &pDest->InvisibleTrans, &pSrc->InvisibleTrans);
}

/****************************************************************************
Desc:	This routine updates statistics from one UTRANS_STATS structure into
		another.
****************************************************************************/
FINLINE void flmUpdateUTransStats(
	UTRANS_STATS *	pDest,
	UTRANS_STATS *	pSrc)
{
	flmUpdateCountTimeStats( &pDest->CommittedTrans, &pSrc->CommittedTrans);
	flmUpdateCountTimeStats( &pDest->GroupCompletes, &pSrc->GroupCompletes);
	pDest->ui64GroupFinished += pSrc->ui64GroupFinished;
	flmUpdateCountTimeStats( &pDest->AbortedTrans, &pSrc->AbortedTrans);
}

/****************************************************************************
Desc:	This routine updates statistics from one BLOCKIO_STATS structure into
		another.
****************************************************************************/
void flmUpdateBlockIOStats(
	BLOCKIO_STATS *	pDest,
	BLOCKIO_STATS *	pSrc
	)
{
	flmUpdateDiskIOStats( &pDest->BlockReads, &pSrc->BlockReads);
	flmUpdateDiskIOStats( &pDest->OldViewBlockReads,
									&pSrc->OldViewBlockReads);
	pDest->uiBlockChkErrs += pSrc->uiBlockChkErrs;
	pDest->uiOldViewBlockChkErrs += pSrc->uiOldViewBlockChkErrs;
	pDest->uiOldViewErrors += pSrc->uiOldViewErrors;
	flmUpdateDiskIOStats( &pDest->BlockWrites, &pSrc->BlockWrites);
}

/****************************************************************************
Desc:	This routine updates statistics from one LFILE_STATS structure into
		another.
****************************************************************************/
FSTATIC void flmUpdateLFileStats(
	LFILE_STATS *	pDest,
	LFILE_STATS *	pSrc
	)
{

	// Set uiFlags in case the number of levels has changed.

	pDest->uiFlags = pSrc->uiFlags;
	pDest->bHaveStats = TRUE;
	flmUpdateBlockIOStats( &pDest->RootBlockStats, &pSrc->RootBlockStats);
	flmUpdateBlockIOStats( &pDest->MiddleBlockStats, &pSrc->MiddleBlockStats);
	flmUpdateBlockIOStats( &pDest->LeafBlockStats, &pSrc->LeafBlockStats);
	pDest->ui64BlockSplits += pSrc->ui64BlockSplits;
	pDest->ui64BlockCombines += pSrc->ui64BlockCombines;
}

/****************************************************************************
Desc:	This routine updates statistics from one DB_STATS structure into
		another.
****************************************************************************/
FSTATIC RCODE flmUpdateDbStats(
	DB_STATS *	pDestDb,
	DB_STATS *	pSrcDb
	)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiSrcLFile;
	FLMUINT			uiDestLFile;
	LFILE_STATS *	pDestLFile;
	LFILE_STATS *	pSrcLFile;
	FLMUINT			uiLowLFileStart;
	FLMUINT			uiSaveNumLFiles;

	flmUpdateRTransStats( &pDestDb->ReadTransStats,
								 &pSrcDb->ReadTransStats);
								 
	flmUpdateUTransStats( &pDestDb->UpdateTransStats,
								 &pSrcDb->UpdateTransStats);
								 
	pDestDb->bHaveStats = TRUE;
	pDestDb->ui64NumCursors += pSrcDb->ui64NumCursors;
	pDestDb->ui64NumCursorReads += pSrcDb->ui64NumCursorReads;
	
	flmUpdateCountTimeStats( &pDestDb->RecordAdds,
									 &pSrcDb->RecordAdds);
									 
	flmUpdateCountTimeStats( &pDestDb->RecordDeletes,
									 &pSrcDb->RecordDeletes);
									 
	flmUpdateCountTimeStats( &pDestDb->RecordModifies,
									 &pSrcDb->RecordModifies);
									 
	pDestDb->ui64NumRecordReads += pSrcDb->ui64NumRecordReads;
	
	flmUpdateBlockIOStats( &pDestDb->LFHBlockStats,
								  &pSrcDb->LFHBlockStats);
								  
	flmUpdateBlockIOStats( &pDestDb->AvailBlockStats,
								  &pSrcDb->AvailBlockStats);
								  
	flmUpdateDiskIOStats( &pDestDb->LogHdrWrites,
								 &pSrcDb->LogHdrWrites);
								 
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
			continue;

		// Find or add the store in the destination store array.

		uiSaveNumLFiles = pDestDb->uiNumLFileStats;
		if (RC_BAD( rc = flmStatGetLFile( pDestDb, pSrcLFile->uiLFileNum,
									(FLMUINT)((pSrcLFile->uiFlags & LFILE_IS_INDEX)
												? (FLMUINT)LF_INDEX
												: (FLMUINT)LF_CONTAINER),
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
			f_memcpy( pDestLFile, pSrcLFile, sizeof( LFILE_STATS));
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
		another.  The source statistics are reset after the
		update.
****************************************************************************/
RCODE	flmStatUpdate(
	FLM_STATS *	pDestStats,
	FLM_STATS *	pSrcStats)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiSrcDb;
	FLMUINT			uiDestDb;
	DB_STATS *		pDestDb;
	DB_STATS *		pSrcDb;
	FLMUINT			uiLowDbStart;

	// Do not update the statistics if the source statistics were started
	// at an earlier time that the destination statistics start time.

	if (!pDestStats->bCollectingStats ||
		 pSrcStats->uiStartTime < pDestStats->uiStartTime)
	{
		return( FERR_OK);
	}

	// If the destination structure has a mutex, lock it.

	if (pDestStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pDestStats->hMutex);
	}

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

		if (RC_BAD( rc = flmStatGetDbByName( pDestStats,
									pSrcDb->pszDbName,
									uiLowDbStart, &pDestDb, NULL,
									&uiLowDbStart)))
		{
			goto Exit;
		}

		if (uiLowDbStart < pDestStats->uiNumDbStats - 1)
		{
			uiLowDbStart++;
		}

		if (RC_BAD( rc = flmUpdateDbStats( pDestDb, pSrcDb)))
		{
			goto Exit;
		}
	}

Exit:

	// Unlock the destination's mutex, if there is one.

	if (pDestStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pDestStats->hMutex);
	}

	// Only clear the source statistics AFTER unlocking the destination
	// statistic's semaphore. This is just an attempt to be slightly more
	// efficient.  It could be done before unlock the destination semaphore
	// and still be correct.  But there is no sense in keeping the destination
	// semaphore locked.

	if (RC_OK( rc))
	{
		flmStatReset( pSrcStats, TRUE, FALSE);
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
	FLMBOOL	bMutexAlreadyLocked
	)
{
	QUERY_HDR *	pQueriesToFree = NULL;

	// Must determine the queries to free inside the mutex lock and then free
	// them outside the mutex lock, because freeing a query may cause an
	// embedded query to be put into the list again, which will cause the
	// mutex to be locked again.

	if (!bMutexAlreadyLocked)
	{
		f_mutexLock( gv_FlmSysData.hQueryMutex);
	}
	while (gv_FlmSysData.uiQueryCnt > gv_FlmSysData.uiMaxQueries)
	{
		gv_FlmSysData.pOldestQuery = gv_FlmSysData.pOldestQuery->pPrev;
		gv_FlmSysData.uiQueryCnt--;
	}

	// Whatever is found after pOldestQuery should be freed.  Unlink
	// those from the list and point to them with pQueriesToFree.

	if (!gv_FlmSysData.pOldestQuery)
	{
		pQueriesToFree = gv_FlmSysData.pNewestQuery;
		gv_FlmSysData.pNewestQuery = NULL;
	}
	else if (gv_FlmSysData.pOldestQuery->pNext)
	{
		pQueriesToFree = gv_FlmSysData.pOldestQuery->pNext;
		pQueriesToFree->pPrev = NULL;
		gv_FlmSysData.pOldestQuery->pNext = NULL;
	}
	f_mutexUnlock( gv_FlmSysData.hQueryMutex);

	// Now clean up each of the queries in the pQueriesToFree list.
	// This can be done outside the mutex lock because we are now
	// dealing with a completely local list that no other thread can
	// see.  Also, the mutex must NOT be locked at this point because
	// flmCurFree may free an embedded query, which will want
	// to lock the mutex again.

	while (pQueriesToFree)
	{
		QUERY_HDR *	pQueryHdrToFree = pQueriesToFree;

		pQueriesToFree = pQueriesToFree->pNext;
		flmCurFree( (CURSOR *)pQueryHdrToFree->hCursor, FALSE);
		f_free( &pQueryHdrToFree);
	}
}

/****************************************************************************
Desc:	This routine saves a query so it can be analyzed later.
****************************************************************************/
void flmSaveQuery(
	HFCURSOR		hCursor
	)
{
	QUERY_HDR *	pQueryHdr = NULL;
	FLMBOOL		bNeedToCleanup = TRUE;
	FLMBOOL		bMutexLocked = FALSE;

	// Allocate memory for the new query

	if (RC_BAD( f_calloc( sizeof( QUERY_HDR), &pQueryHdr)))
	{
		goto Exit;
	}

	pQueryHdr->hCursor = hCursor;

	f_mutexLock( gv_FlmSysData.hQueryMutex);
	bMutexLocked = TRUE;

	// uiMaxQueries was originally checked outside of the mutex lock.
	// Make sure it is still non-zero.

	if (gv_FlmSysData.uiMaxQueries)
	{

		// Link query to head of list.

		bNeedToCleanup = FALSE;
		if ((pQueryHdr->pNext = gv_FlmSysData.pNewestQuery) != NULL)
		{
			pQueryHdr->pNext->pPrev = pQueryHdr;
		}
		else
		{
			gv_FlmSysData.pOldestQuery = pQueryHdr;
		}
		gv_FlmSysData.pNewestQuery = pQueryHdr;

		if (++gv_FlmSysData.uiQueryCnt > gv_FlmSysData.uiMaxQueries)
		{
			flmFreeSavedQueries( TRUE);

			// flmFreeSavedQueries will always unlock the mutex.

			bMutexLocked = FALSE;
		}
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hQueryMutex);
	}

	// Must clean up the query if we didn't get it into the list for
	// some reason.

	if (bNeedToCleanup)
	{
		if (pQueryHdr)
		{
			f_free( &pQueryHdr);
		}
		flmCurFree( (CURSOR *)hCursor, FALSE);
	}
}

/****************************************************************************
Desc:	This routine copies statistics from one FLM_STAT structure into
		another.  This is used to retrieve statistics.
****************************************************************************/
FSTATIC RCODE flmStatCopy(
	FLM_STATS *	pDestStats,
	FLM_STATS *	pSrcStats
	)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiDb;
	DB_STATS *		pDestDb;
	DB_STATS *		pSrcDb;
	FLMUINT			uiCount;
	FLMUINT			uiLoop;
	DB_STATS *		pDbStats;
	LFILE_STATS *	pLFile;

	flmStatInit( pDestStats, FALSE);

	// If the source structure has a mutex, lock it.

	if (pSrcStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( pSrcStats->hMutex);
	}

	f_memcpy( pDestStats, pSrcStats, sizeof( FLM_STATS));

	// Zero out the database array.  We need to do this
	// so that if we get an error, we can correctly release memory for
	// the destination structure.

	pDestStats->hMutex = F_MUTEX_NULL;
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
		if (RC_BAD( rc = f_calloc( 
			(FLMUINT)sizeof( DB_STATS) * uiCount, &pDestStats->pDbStats)))
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
				f_memcpy( pDestDb, pDbStats, sizeof( DB_STATS));

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
				(FLMUINT)sizeof( LFILE_STATS) * uiCount, &pDestDb->pLFileStats)))
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
								pLFile, sizeof( LFILE_STATS));
					uiCount++;
				}
			}
			pDestDb->uiLFileStatArraySize =
			pDestDb->uiNumLFileStats = uiCount;
		}
		pDestDb++;
	}

Exit:

	// Unlock the source's mutex, if there is one.

	if (pSrcStats->hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( pSrcStats->hMutex);
	}

	if (RC_BAD( rc))
	{
		FlmFreeStats( pDestStats);
	}
	return( rc);
}

/****************************************************************************
Desc : Returns statistics that have been collected for a share.
Notes: The statistics returned will be the statistics for ALL
		 databases associated with the share structure.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmGetStats(
	FLM_STATS *	pFlmStats)
{
	RCODE			rc = FERR_OK;

	// Get the statistics
	
	if( RC_BAD( rc = flmStatCopy( pFlmStats, &gv_FlmSysData.Stats)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine locates the appropriate BLOCKIO_STATS structure for a
		given block of data.  NULL is returned if an appropriate one cannot
		be found.
VISIT: uiBlkType passed in is a guess.  Remove this parm and start using
		 pBlk.
****************************************************************************/
BLOCKIO_STATS * flmGetBlockIOStatPtr(
	DB_STATS *		pDbStats,
	LFILE_STATS *	pLFileStats,
	FLMBYTE *		pBlk,
	FLMUINT			uiBlkType)
{
	if (uiBlkType == BHT_FREE)
	{
		pDbStats->bHaveStats = TRUE;
		return( &pDbStats->AvailBlockStats);
	}
	else if (uiBlkType == BHT_LFH_BLK)
	{
		pDbStats->bHaveStats = TRUE;
		return( &pDbStats->LFHBlockStats);
	}
	else if (pLFileStats)
	{
		pLFileStats->bHaveStats =
		pDbStats->bHaveStats = TRUE;
		
		// Consider invalid type.

		if ((BH_GET_TYPE( pBlk) != BHT_LEAF) &&
			 (BH_GET_TYPE( pBlk) != BHT_NON_LEAF) &&
			 (BH_GET_TYPE( pBlk) != BHT_NON_LEAF_DATA) &&
			 (BH_GET_TYPE( pBlk) != BHT_NON_LEAF_COUNTS))
		{
			return( &pLFileStats->LeafBlockStats);
		}
		if ((FB2UD( &(pBlk [BH_NEXT_BLK])) == BT_END) &&
	 		 (FB2UD( &(pBlk [BH_PREV_BLK])) == BT_END))
		{
			return( &pLFileStats->RootBlockStats);
		}
		else if (BH_GET_TYPE( pBlk) != BHT_LEAF)
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
		return( (BLOCKIO_STATS *)NULL);
	}
}

/********************************************************************
Desc: Determine if a given year is a leap year.
*********************************************************************/
FINLINE FLMUINT flmLeapYear(
	FLMUINT		uiYear)
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
		*(pui64ElapMilli) += (FLMUINT64)(uiHundredth * 10 + uiSec * 1000);
	}
	else
	{
		*(pui64ElapMilli) += (FLMUINT64)(uiHundredth * 10);
	}
}
