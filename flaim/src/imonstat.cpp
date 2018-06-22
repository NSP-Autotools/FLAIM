//-------------------------------------------------------------------------
// Desc:	Class for displaying various statistics in HTML on a web page.
// Tabs:	3
//
// Copyright (c) 2002-2007 Novell, Inc. All Rights Reserved.
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

#define STAT_DISPLAY_ORDER		"StatDisplayOrder"
#define CACHE_STAT_STR			"Cache"
#define OPERATIONS_STAT_STR	"Operations"
#define LOCKS_STAT_STR			"Locks"
#define DISK_STAT_STR			"Disk"
#define CHECK_POINT_STR			"CPThread"
#define DEFAULT_STAT_ORDER		CACHE_STAT_STR ";" \
										OPERATIONS_STAT_STR ";" \
										LOCKS_STAT_STR ";" \
										DISK_STAT_STR ";" \
										CHECK_POINT_STR ";"
#define SAVED_STATS				"SavedStats"
#define STAT_FOCUS				"StatFocus"

// Stat types

#define CACHE_STATS			1
#define OPERATION_STATS		2
#define LOCK_STATS			3
#define DISK_STATS			4
#define CHECK_POINT_STATS	5

#define MAX_STAT_TYPES		5

/****************************************************************************
Desc:	Gather statistics from a BLOCKIO_STATS structure.
****************************************************************************/
void F_StatsPage::gatherBlockIOStats(
	STAT_GATHER *		pStatGather,
	DISKIO_STAT *		pReadStat,
	DISKIO_STAT *		pWriteStat,
	BLOCKIO_STATS *	pBlockIOStats
	)
{

	// Gather the read statistics.

	flmUpdateDiskIOStats( &pStatGather->IOReads,
		&pBlockIOStats->BlockReads);
	flmUpdateDiskIOStats( &pStatGather->IOReads,
		&pBlockIOStats->OldViewBlockReads);
	flmUpdateDiskIOStats( &pStatGather->IORollbackBlockReads,
		&pBlockIOStats->OldViewBlockReads);
	flmUpdateDiskIOStats( pReadStat, &pBlockIOStats->BlockReads);

	// Gather the check errors

	pStatGather->uiCheckErrors +=
		(pBlockIOStats->uiBlockChkErrs +
		 pBlockIOStats->uiOldViewBlockChkErrs);

	// Gather the write statistics

	flmUpdateDiskIOStats( &pStatGather->IOWrites,
		&pBlockIOStats->BlockWrites);
	flmUpdateDiskIOStats( pWriteStat, &pBlockIOStats->BlockWrites);
}

/****************************************************************************
Desc:	Gather statistics for an LFILE.
****************************************************************************/
void F_StatsPage::gatherLFileStats(
	STAT_GATHER *	pStatGather,
	LFILE_STATS *	pLFileStats
	)
{
	pStatGather->uiNumLFileStats++;

	pStatGather->ui64BlockSplits += pLFileStats->ui64BlockSplits;
	pStatGather->ui64BlockCombines += pLFileStats->ui64BlockCombines;

	// Gather root block statistics

	gatherBlockIOStats( pStatGather, &pStatGather->IORootBlockReads,
		&pStatGather->IORootBlockWrites, &pLFileStats->RootBlockStats);

	// Gather non-leaf block statistics

	gatherBlockIOStats( pStatGather, &pStatGather->IONonLeafBlockReads,
		&pStatGather->IONonLeafBlockWrites, &pLFileStats->MiddleBlockStats);

	// Gather leaf block statistics

	gatherBlockIOStats( pStatGather, &pStatGather->IOLeafBlockReads,
		&pStatGather->IOLeafBlockWrites, &pLFileStats->LeafBlockStats);
}

/****************************************************************************
Desc:	Gather statistics for a particular database.
****************************************************************************/
void F_StatsPage::gatherDbStats(
	STAT_GATHER *	pStatGather,
	DB_STATS *		pDbStats
	)
{
	FLMUINT	uiLoop;

	pStatGather->uiNumDbStats++;

	flmUpdateCountTimeStats( &pStatGather->CommittedUpdTrans,
		&pDbStats->UpdateTransStats.CommittedTrans);
	flmUpdateCountTimeStats( &pStatGather->GroupCompletes,
		&pDbStats->UpdateTransStats.GroupCompletes);
	pStatGather->ui64GroupFinished +=
		pDbStats->UpdateTransStats.ui64GroupFinished;
	flmUpdateCountTimeStats( &pStatGather->AbortedUpdTrans,
		&pDbStats->UpdateTransStats.AbortedTrans);
	flmUpdateCountTimeStats( &pStatGather->CommittedReadTrans,
		&pDbStats->ReadTransStats.CommittedTrans);
	flmUpdateCountTimeStats( &pStatGather->AbortedReadTrans,
		&pDbStats->ReadTransStats.AbortedTrans);
	pStatGather->Reads.ui64Count += pDbStats->ui64NumRecordReads;
	flmUpdateCountTimeStats( &pStatGather->Adds,
		&pDbStats->RecordAdds);
	flmUpdateCountTimeStats( &pStatGather->Modifies,
		&pDbStats->RecordModifies);
	flmUpdateCountTimeStats( &pStatGather->Deletes,
		&pDbStats->RecordDeletes);
	pStatGather->Queries.ui64Count += pDbStats->ui64NumCursors;
	pStatGather->QueryReads.ui64Count += pDbStats->ui64NumCursorReads;

	if ((m_pFocusBlock == NULL) ||
		 (m_pFocusBlock->uiLFileNum == 0))
	{
		// Gather the avail block statistics

		gatherBlockIOStats( pStatGather, &pStatGather->IOAvailBlockReads,
			&pStatGather->IOAvailBlockWrites, &pDbStats->AvailBlockStats);

		// Gather the LFH block statistics

		gatherBlockIOStats( pStatGather, &pStatGather->IOLFHBlockReads,
			&pStatGather->IOLFHBlockWrites, &pDbStats->LFHBlockStats);

		// Gather log block reads

		flmUpdateDiskIOStats( &pStatGather->IOReads,
			&pDbStats->LogBlockReads);
		flmUpdateDiskIOStats( &pStatGather->IORollbackBlockReads,
			&pDbStats->LogBlockReads);

		// Gather log header writes

		flmUpdateDiskIOStats( &pStatGather->IOWrites,
			&pDbStats->LogHdrWrites);
		flmUpdateDiskIOStats( &pStatGather->IOLogHdrWrites,
			&pDbStats->LogHdrWrites);

		// Gather roll-back log writes

		flmUpdateDiskIOStats( &pStatGather->IOWrites,
			&pDbStats->LogBlockWrites);
		flmUpdateDiskIOStats( &pStatGather->IORollBackLogWrites,
			&pDbStats->LogBlockWrites);

		// Gather rolled-back block writes

		flmUpdateDiskIOStats( &pStatGather->IOWrites,
			&pDbStats->LogBlockRestores);
		flmUpdateDiskIOStats( &pStatGather->IORolledbackBlockWrites,
			&pDbStats->LogBlockRestores);

		// Gather I/O error statistics

		pStatGather->uiReadErrors += pDbStats->uiReadErrors;
		pStatGather->uiWriteErrors += pDbStats->uiWriteErrors;
		pStatGather->uiCheckErrors +=
			(pDbStats->AvailBlockStats.uiBlockChkErrs +
			 pDbStats->AvailBlockStats.uiOldViewBlockChkErrs +
			 pDbStats->LFHBlockStats.uiBlockChkErrs +
			 pDbStats->LFHBlockStats.uiOldViewBlockChkErrs +
			 pDbStats->uiLogBlockChkErrs);
	}
	// Gather lock statistics

	flmUpdateCountTimeStats( &pStatGather->LockStats.NoLocks,
		&pDbStats->LockStats.NoLocks);
		
	flmUpdateCountTimeStats( &pStatGather->LockStats.WaitingForLock,
		&pDbStats->LockStats.WaitingForLock);
		
	flmUpdateCountTimeStats( &pStatGather->LockStats.HeldLock,
		&pDbStats->LockStats.HeldLock);

	for (uiLoop = 0; uiLoop < pDbStats->uiNumLFileStats; uiLoop++)
	{
		if ((m_pFocusBlock == NULL) || 
			 (m_pFocusBlock->uiLFileNum == 0) || 
			 (m_pFocusBlock->uiLFileNum == pDbStats->pLFileStats [uiLoop].uiLFileNum))
		{
			gatherLFileStats( pStatGather, &pDbStats->pLFileStats [uiLoop]);
		}
	}
}

/****************************************************************************
Desc:	Gather statistics for all databases in the FLM_STATS structure.
****************************************************************************/
void F_StatsPage::gatherStats(
	STAT_GATHER *	pStatGather
	)
{
	FLMUINT	uiLoop;

	f_memset( pStatGather, 0, sizeof( STAT_GATHER));

	f_mutexLock( gv_FlmSysData.Stats.hMutex);

	pStatGather->bCollectingStats = gv_FlmSysData.Stats.bCollectingStats;
	if (gv_FlmSysData.Stats.uiStartTime)
	{
		pStatGather->uiStartTime = gv_FlmSysData.Stats.uiStartTime;
		pStatGather->uiStopTime = gv_FlmSysData.Stats.uiStopTime;

		for (uiLoop = 0; uiLoop < gv_FlmSysData.Stats.uiNumDbStats; uiLoop++)
		{
			if ((m_pFocusBlock == NULL) || 
				 (f_strcmp(m_pFocusBlock->szFileName,
							  gv_FlmSysData.Stats.pDbStats [uiLoop].pszDbName) == 0))
			{
				gatherDbStats( pStatGather,
				&gv_FlmSysData.Stats.pDbStats [uiLoop]);
			}
		}
	}

	f_mutexUnlock( gv_FlmSysData.Stats.hMutex);

	// Get the cache statistics.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	f_memcpy( &pStatGather->RecordCache,
		&gv_FlmSysData.RCacheMgr.Usage,
		sizeof( pStatGather->RecordCache));
	f_memcpy( &pStatGather->BlockCache,
		&gv_FlmSysData.SCacheMgr.Usage,
		sizeof( pStatGather->BlockCache));

	pStatGather->uiFreeCount = gv_FlmSysData.SCacheMgr.uiFreeCount;
	pStatGather->uiFreeBytes = gv_FlmSysData.SCacheMgr.uiFreeBytes;
	pStatGather->uiReplaceableCount = gv_FlmSysData.SCacheMgr.uiReplaceableCount;
	pStatGather->uiReplaceableBytes = gv_FlmSysData.SCacheMgr.uiReplaceableBytes;

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	for (uiLoop = 0; uiLoop < FILE_HASH_ENTRIES; uiLoop++)
	{
		FFILE *	pFile = (FFILE *)gv_FlmSysData.pFileHashTbl [uiLoop].pFirstInBucket;

		while (pFile)
		{
			if (pFile->uiDirtyCacheCount)
			{
				pStatGather->uiDirtyBytes +=
					pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;
				pStatGather->uiDirtyBlocks +=
					pFile->uiDirtyCacheCount;
			}

			if (pFile->uiLogCacheCount)
			{
				pStatGather->uiLogBytes +=
					pFile->uiLogCacheCount * pFile->FileHdr.uiBlockSize;
				pStatGather->uiLogBlocks +=
					pFile->uiLogCacheCount;
			}

			gatherCPStats( pStatGather, pFile);
			gatherLockStats( pStatGather, pFile);
			pFile = pFile->pNext;
		}
	}
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
}

/****************************************************************************
Desc:	Prints the web page for system configuration parameters.
****************************************************************************/
RCODE F_StatsPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	void *				pvSession = NULL;
	STAT_GATHER *		pStatGather = NULL;
	STAT_GATHER *		pOldStatGather = NULL;
	char					szStatOrder [50];
	char					szAction [5];
	char					szStatNewOrder [50];
	char *				pszStat;
	char *				ppszStatOrders [MAX_STAT_TYPES];
	FLMUINT				uiLoop;
	FLMUINT				uiAction;
	FLMBOOL				bRefresh;
	FLMBOOL				bFocus;
	char *				pszTemp = NULL;
	char *				pszHeading = NULL;
	char					szCfgAction [50];
	eFlmConfigTypes	eConfigType;
	FLMUINT				uiStatOrders [MAX_STAT_TYPES];

	// Are we attempting to change the focus?
	bFocus = DetectParameter( uiNumParams, ppszParams, "Focus");
	if (bFocus)
	{
		// Prepare the page to request the user input on what to focus on.
		displayFocus( uiNumParams, ppszParams);
		goto Exit;

	}

	if( RC_BAD( rc = f_alloc( 100, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 250, &pszHeading)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if (RC_OK( ExtractParameter( uiNumParams, ppszParams, 
					"CfgAction", sizeof( szCfgAction), szCfgAction)))
	{
		FLMUINT		uiValue1 = 0;
		FLMUINT		uiValue2 = 0;

		eConfigType = (eFlmConfigTypes)f_atoi( szCfgAction);

		// Call FlmConfig to perform this action.
		if (RC_BAD( rc = FlmConfig( eConfigType, (void *)uiValue1,
								(void *)uiValue2)))
		{
			printErrorPage( FERR_FAILURE, TRUE, (char *)"Failed to perform configuration update");
			goto Exit;
		}

	}

	// Get the session, then try to retrieve the StatOrder.  If it isn't there, then
	// we can set it to the default.
	szStatOrder[0] = 0;

	if (gv_FlmSysData.HttpConfigParms.fnAcquireSession)
	{
		char		szFocus[100];

		if ((pvSession = fnAcquireSession()) != NULL)
		{
			FLMUINT	uiSize = sizeof( szStatOrder);

			if (fnGetSessionValue( pvSession,
								  STAT_DISPLAY_ORDER,
								  (void *)szStatOrder,
								  (FLMSIZET *)&uiSize) != 0)
			{
				// If we could not get display order, then set the default.
				f_strcpy( szStatOrder, DEFAULT_STAT_ORDER);
			}

			// Find out if we are focusing our stats collection.
			uiSize = sizeof( szFocus) - 1;
			if (fnGetSessionValue( pvSession,
								  STAT_FOCUS,
								  (void *)szFocus,
								  (FLMSIZET *)&uiSize) == 0)
			{
				szFocus[ uiSize] = '\0';
				if (RC_BAD( setFocus( szFocus)))
				{
					printErrorPage( FERR_MEM, TRUE, "Failed to establish focus criteria");
					goto Exit;
				}
			}
		}
	}


	// See if they changed the display order.

	uiAction = (FLMUINT)(-1);

	
	if (RC_OK( ExtractParameter( uiNumParams, ppszParams, 
									"Action", sizeof( szAction), szAction)))
	{
		uiAction = (FLMUINT)f_atoi( szAction);
	}

	uiLoop = 0;
	pszStat = &szStatOrder [0];
	while (*pszStat && uiLoop < MAX_STAT_TYPES)
	{

		ppszStatOrders [uiLoop] = pszStat;
		uiLoop++;

		while (*pszStat && *pszStat != ';')
		{
			pszStat++;
		}
		if (*pszStat)
		{
			*pszStat = 0;
			pszStat++;
		}
	}

	// See if they changed the order.

	if (uiAction != (FLMUINT)(-1) && uiAction < MAX_STAT_TYPES * 2)
	{

		// Odd numbers mean shift to top (move slot 0 to 0, 1 to 0, 2 to 0, 3 to 0)
		// Even number mean shift up (move slot 0 to 3, 1 to 0, 2 to 1, 3 to 2)

		if (uiAction & 1)
		{
			uiAction /= 2;

			// Trade places with the guy that is in the top position
			// No need to move the top one.

			if (uiAction)
			{
				pszStat = ppszStatOrders [0];
				ppszStatOrders [0] = ppszStatOrders [uiAction];
				ppszStatOrders [uiAction] = pszStat;
			}

		}
		else
		{
			uiAction /= 2;

			pszStat = ppszStatOrders [uiAction];

			// Trade places with the guy that is lower.

			if (!uiAction)
			{
				ppszStatOrders [uiAction] = ppszStatOrders [MAX_STAT_TYPES - 1];
				ppszStatOrders [MAX_STAT_TYPES - 1] = pszStat;
			}
			else
			{
				ppszStatOrders [uiAction] = ppszStatOrders [uiAction - 1];
				ppszStatOrders [uiAction - 1] = pszStat;
			}
		}

		// Output the new order.

		pszStat = &szStatNewOrder [0];
		for (uiLoop = 0; uiLoop < MAX_STAT_TYPES; uiLoop++)
		{
			f_strcpy( pszStat, ppszStatOrders [uiLoop]);
			while (*pszStat)
			{
				pszStat++;
			}
			*pszStat++ = ';';
		}
		*pszStat = 0;
		(void)fnSetSessionValue( pvSession,
									  STAT_DISPLAY_ORDER,
									  szStatNewOrder,
									  (FLMSIZET)(f_strlen( szStatNewOrder) + 1));
	}

	if (RC_BAD( rc = f_calloc( sizeof( STAT_GATHER), &pStatGather)))
	{
		printErrorPage( rc, TRUE, "ERROR ALLOCATING MEMORY: ");
		goto Exit;
	}

	if (RC_BAD( rc = f_calloc( sizeof( STAT_GATHER), &pOldStatGather)))
	{
		printErrorPage( rc, TRUE, "ERROR ALLOCATING MEMORY: ");
		goto Exit;
	}

	// Collect the current statistics...
	gatherStats( pStatGather);

	// See if we have any stats from a previous run stored in the session.
	if (pvSession)
	{
		FLMUINT	uiSize = sizeof( STAT_GATHER);

		if (fnGetSessionValue( pvSession,
							  SAVED_STATS,
							  (void *)pOldStatGather,
							  (FLMSIZET *)&uiSize) != 0)
		{
			// If we could not get any former stats, then just copy the current one.
			f_memcpy( pOldStatGather, pStatGather, sizeof( STAT_GATHER));
		}

		// Now save the new stats for the next go round...
		if (fnSetSessionValue( pvSession,
							  SAVED_STATS,
							  (void *)pStatGather,
							  sizeof( STAT_GATHER)) != 0)
		{
			printErrorPage( rc, TRUE, "ERROR Saving Gathered Statistics in current Session: ");
			goto Exit;
		}

		
	}

	// Output the web page.
	stdHdr();
	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	bRefresh = DetectParameter( uiNumParams, ppszParams, "Refresh");
	if (bRefresh)
	{
		// Send back the page with a refresh command in the header
		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s/Stats?Refresh\">"
			"<TITLE>System Statistics</TITLE>\n",
			m_pszURLString);

		printStyle();
		popupFrame();
	
		fnPrintf( m_pHRequest, "</HEAD>\n");
		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
			"<A HREF=%s/Stats>Stop Auto-refresh</A>",
			m_pszURLString);
	
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>System Statistics</TITLE>\n");
		printStyle();
		popupFrame();
		fnPrintf( m_pHRequest, "</HEAD>\n");
		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
			"<A HREF=%s/Stats?Refresh>Start Auto-refresh (5 sec.)</A>",
			m_pszURLString);
	}


	// Format the main heading title
	formatStatsHeading( pStatGather, pszHeading);

	// Begin the table
	fnPrintf( m_pHRequest, "<table border=0 cellpadding=2 cellspacing=0 width=100%%>\n");
	fnPrintf( m_pHRequest, "<tr class=\"mediumtext\">\n");
	fnPrintf( m_pHRequest, "<td colspan=4 class=\"tablehead1\">\n");
	fnPrintf( m_pHRequest, (char *)pszHeading);
	fnPrintf( m_pHRequest, "</td></tr>\n");

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=%s/Stats%s>Refresh</A>, ",
									m_pszURLString,
									(bRefresh ? "?Refresh" : ""));
	fnPrintf( m_pHRequest, "%s, ", pszTemp);
	if (!pStatGather->uiStartTime || gv_FlmSysData.Stats.uiStopTime)
	{
		fnPrintf( m_pHRequest, "<A HREF=%s/Stats?CfgAction=%d%s>Begin Statistics</A>, ",
										m_pszURLString,
										FLM_START_STATS,
										(bRefresh ? "&Refresh" : ""));
	}
	if (pStatGather->uiStartTime && !gv_FlmSysData.Stats.uiStopTime)
	{
		fnPrintf( m_pHRequest, "<A HREF=%s/Stats?CfgAction=%d%s>End Statistics</A>, ",
										m_pszURLString,
										FLM_STOP_STATS,
										(bRefresh ? "&Refresh" : ""));
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/Stats?CfgAction=%d%s>Reset Statistics</A>, ",
									m_pszURLString,
									FLM_RESET_STATS,
									(bRefresh ? "&Refresh" : ""));
	fnPrintf( m_pHRequest, "<A HREF=\"javascript:openPopup(\'%s/Stats?Focus%s\')\">Set Focus</A>",
									m_pszURLString,
									(bRefresh ? "&Refresh" : ""));

	printColumnHeadingClose();
	printTableRowEnd();

	printTableRowStart( TRUE);
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "Stats Order:&nbsp;&nbsp;");
	for (uiLoop = 0; uiLoop < MAX_STAT_TYPES; uiLoop++)
	{
		FLMUINT		uiStat;
		char *	pszStatOrder = ppszStatOrders [uiLoop];

		if (f_stricmp( pszStatOrder, CACHE_STAT_STR) == 0)
		{
			uiStat = CACHE_STATS;
		}
		else if (f_stricmp( pszStatOrder, OPERATIONS_STAT_STR) == 0)
		{
			uiStat = OPERATION_STATS;
		}
		else if (f_stricmp( pszStatOrder, LOCKS_STAT_STR) == 0)
		{
			uiStat = LOCK_STATS;
		}
		else if (f_stricmp( pszStatOrder, CHECK_POINT_STR) == 0)
		{
			uiStat = CHECK_POINT_STATS;
		}
		else
		{
			uiStat = DISK_STATS;
		}
		uiStatOrders [uiLoop] = uiStat;


		fnPrintf( m_pHRequest, "%s: ", pszStatOrder);
		fnPrintf( m_pHRequest,
			"<a href=%s/Stats?Action=%d%s>Top</a>, ",
			m_pszURLString,
			(uiLoop * 2) + 1, (bRefresh ? "&Refresh" : ""));
		fnPrintf( m_pHRequest,
			"<a href=%s/Stats?Action=%d%s>Up</a>&nbsp;&nbsp;\n",
			m_pszURLString,
			uiLoop * 2, (bRefresh ? "&Refresh" : ""));

	}
	printColumnHeadingClose();
	printTableRowEnd();
	printTableEnd();

	displayStats( pStatGather, pOldStatGather, (FLMUINT *)uiStatOrders);
	printDocEnd();

Exit:

	fnEmit();

	if (pStatGather)
	{
		freeCPInfoHeaders( pStatGather);
		freeLockUsers( pStatGather);
		f_free( &pStatGather);
	}
	if (pOldStatGather)
	{
		f_free( &pOldStatGather);
	}
	if (pvSession)
	{
		fnReleaseSession( pvSession);
	}
	if (pszTemp)
	{
		f_free( &pszTemp);
	}
	if (pszHeading)
	{
		f_free( &pszHeading);
	}

	return( rc);
}

/****************************************************************************
Desc:	Formats the main title heading string so that it displays the start/stop
		/elapsed times for the statistics.
****************************************************************************/
void F_StatsPage::formatStatsHeading(
	STAT_GATHER *		pStatGather,
	const char *		pszHeading)
{
	char 					szBuffer[ 30];
	FLMUINT64			ui64ElapTime;
	FLMUINT				uiCurrTime;

	flmAssert( pStatGather);
	flmAssert( pszHeading);


	f_sprintf( (char *)pszHeading, "Statistics:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;");

	// Are we collecting stats?
	if (pStatGather->uiStartTime)
	{
		printDate( gv_FlmSysData.Stats.uiStartTime, szBuffer);
		f_strcat( (char *)pszHeading, (char *)szBuffer);
		f_strcat( (char *)pszHeading, "&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;");

		// Check for a stop time.
		if (gv_FlmSysData.Stats.uiStopTime)
		{
			printDate( gv_FlmSysData.Stats.uiStopTime, szBuffer);
			f_strcat( (char *)pszHeading, (char *)szBuffer);
			ui64ElapTime = (FLMUINT64)(gv_FlmSysData.Stats.uiStopTime - 
												gv_FlmSysData.Stats.uiStartTime);
		}
		else
		{
			f_strcat( (char *)pszHeading, "Present");
			f_timeGetSeconds( &uiCurrTime);
			ui64ElapTime = (FLMUINT64)(uiCurrTime - gv_FlmSysData.Stats.uiStartTime);
		}

		f_strcat( (char *)pszHeading, "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Elapsed:&nbsp");
		printElapTime( ui64ElapTime, szBuffer, JUSTIFY_LEFT, FALSE);
		f_strcat( (char *)pszHeading, (char *)szBuffer);

	}
	else
	{
		f_strcat( (char *)pszHeading, "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Not collecting");
	}

}


/****************************************************************************
Desc:	Outputs statistics for a particular IO category.
****************************************************************************/
void F_StatsPage::printIORow(
	FLMBOOL			bHighlight,
	const char *	pszIOCategory,
	DISKIO_STAT *	pIOStat,
	DISKIO_STAT *	pOldIOStat)
{
	char				szTemp[30];

	printTableRowStart( bHighlight);

	printTableDataStart( TRUE, JUSTIFY_LEFT);
	fnPrintf( m_pHRequest, "%s", pszIOCategory);
	printTableDataEnd();

	printCommaNum( pIOStat->ui64Count,
						JUSTIFY_RIGHT,
						(pIOStat->ui64Count != pOldIOStat->ui64Count ? TRUE : FALSE));
	
	printCommaNum( pIOStat->ui64TotalBytes,
						JUSTIFY_RIGHT,
						(pIOStat->ui64TotalBytes != pOldIOStat->ui64TotalBytes ? TRUE : FALSE));
	
	printElapTime( pIOStat->ui64ElapMilli, szTemp);
	printTableDataStart( TRUE, JUSTIFY_RIGHT);
	fnPrintf( m_pHRequest, "%s%s%s",
		(pIOStat->ui64ElapMilli != pOldIOStat->ui64ElapMilli ? "<font color=red>" : ""),
		szTemp,
		(pIOStat->ui64ElapMilli != pOldIOStat->ui64ElapMilli ? "</font>" : ""));
	printTableDataEnd();

	if (pIOStat->ui64Count)
	{
		printElapTime( pIOStat->ui64ElapMilli / pIOStat->ui64Count, szTemp);
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%s%s%s",
			(pOldIOStat->ui64Count ? (pIOStat->ui64ElapMilli / pIOStat->ui64Count !=
			pOldIOStat->ui64ElapMilli / pOldIOStat->ui64Count ? "<font color=red>" : "") : "<font color=red>") ,
			szTemp,
			(pOldIOStat->ui64Count ? (pIOStat->ui64ElapMilli / pIOStat->ui64Count !=
			pOldIOStat->ui64ElapMilli / pOldIOStat->ui64Count ? "</font>" : "") : "</font>"));
		printTableDataEnd();
	}
	else
	{
		printElapTime( 0);
	}

	printTableRowEnd();
}

/****************************************************************************
Desc:	Outputs count/time statistics for a particular category.
****************************************************************************/
void F_StatsPage::printCountTimeRow(
	FLMBOOL					bHighlight,
	const char *			pszCategory,
	F_COUNT_TIME_STAT *	pStat,
	F_COUNT_TIME_STAT *	pOldStat,
	FLMBOOL					bPrintCountOnly)
{
	char					szTemp[ 30];

	printTableRowStart( bHighlight);

	printTableDataStart( TRUE, JUSTIFY_LEFT);
	fnPrintf( m_pHRequest, "%s", pszCategory);
	printTableDataEnd();

	printCommaNum( pStat->ui64Count, JUSTIFY_RIGHT,
		(pStat->ui64Count != pOldStat->ui64Count ? TRUE : FALSE));
	
	if (bPrintCountOnly)
	{
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "N/A");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "N/A");
		printTableDataEnd();
	}
	else
	{
		printElapTime( pStat->ui64ElapMilli, szTemp);
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%s%s%s",
			(pStat->ui64ElapMilli != pOldStat->ui64ElapMilli ? "<font color=red>" : ""),
			szTemp,
			(pStat->ui64ElapMilli != pOldStat->ui64ElapMilli ? "</font>" : ""));
		printTableDataEnd();

		if (pStat->ui64Count)
		{
			printElapTime( pStat->ui64ElapMilli / pStat->ui64Count, szTemp);
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "%s%s%s",
				(pOldStat->ui64Count ? 
					(pStat->ui64ElapMilli / pStat->ui64Count != 
					 pOldStat->ui64ElapMilli / pOldStat->ui64Count ?
						"<font color=red>" : "") :
				 "<font color=red>"),
				szTemp,
				(pOldStat->ui64Count ? 
					(pStat->ui64ElapMilli / pStat->ui64Count != 
					 pOldStat->ui64ElapMilli / pOldStat->ui64Count ?
						"</font>" : "") :
				 "</font>"));
			printTableDataEnd();
		}
		else
		{
			printElapTime( 0);
		}
	}

	printTableRowEnd();
}

/****************************************************************************
Desc:	Outputs statistics for a particular Lock category.
****************************************************************************/
void F_StatsPage::printCacheStatRow(
	FLMBOOL				bHighlight,
	const char *		pszCategory,
	FLMUINT				uiBlockCacheValue,
	FLMUINT				uiRecordCacheValue,
	FLMBOOL				bRecordCacheValueApplicable,
	FLMBOOL				bBChangedValue,
	FLMBOOL				bRChangedValue)
{
	printTableRowStart( bHighlight);

	printTableDataStart( TRUE, JUSTIFY_LEFT);
	fnPrintf( m_pHRequest, "%s", pszCategory);
	printTableDataEnd();

	printCommaNum( uiBlockCacheValue, JUSTIFY_RIGHT, bBChangedValue);

	if (bRecordCacheValueApplicable)
	{
		printCommaNum( uiRecordCacheValue, JUSTIFY_RIGHT, bRChangedValue);
	}
	else
	{
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "N/A");
		printTableDataEnd();
	}
	printTableRowEnd();
}

/****************************************************************************
Desc:	Outputs statistics page
****************************************************************************/
void F_StatsPage::displayStats(
	STAT_GATHER *	pStatGather,
	STAT_GATHER *	pOldStatGather,
	FLMUINT *		puiStatOrders
	)
{
	FLMUINT	uiLoop;


	for (uiLoop = 0; uiLoop < MAX_STAT_TYPES; uiLoop++)
	{
		switch (puiStatOrders [uiLoop])
		{
			case CACHE_STATS:
				printCacheStats( pStatGather, pOldStatGather);
				break;

			case OPERATION_STATS:
				printOperationStats( pStatGather, pOldStatGather);
				break;

			case LOCK_STATS:
				printLockStats( pStatGather, pOldStatGather);
				break;

			case DISK_STATS:
				printDiskStats( pStatGather, pOldStatGather);
				break;
			case CHECK_POINT_STATS:
				printCPStats( pStatGather);
				break;
			default:
				break;
		}
	}

	fnPrintf( m_pHRequest, "<br>\n");
}

/****************************************************************************
Desc:	Deletes all LOCK_USER_HEADER structures linked from the pStatGather structure.
****************************************************************************/
void F_StatsPage::freeLockUsers(
		STAT_GATHER *		pStatGather
		)
{
	LOCK_USER_HEADER_p		pTmp;

	while (pStatGather->pLockUsers)
	{
		pTmp = pStatGather->pLockUsers;
		pStatGather->pLockUsers = pStatGather->pLockUsers->pNext;

		if (pTmp->pDbLockUser)
		{
			f_free( &pTmp->pDbLockUser);
		}
		if (pTmp->pTxLockUser)
		{
			f_free( &pTmp->pTxLockUser);
		}

		f_free( &pTmp);
	}
}

/****************************************************************************
Desc:	Gets the LOCK_USER information for the specified pFile.
****************************************************************************/
void F_StatsPage::gatherLockStats(
	STAT_GATHER *			pStatGather,
	FFILE *					pFile)
{
	LOCK_USER_HEADER_p	pTmp;
	RCODE						rc;

	flmAssert( pStatGather);
	flmAssert( pFile);

	// Allocate a new LOCK_USER_HEADER and link it into the list.

	if( RC_BAD( rc = f_alloc( sizeof( LOCK_USER_HEADER), &pTmp)))
	{
		goto Exit;
	}

	pTmp->pNext = pStatGather->pLockUsers;
	pStatGather->pLockUsers = pTmp;

	// Save the file name.
	if (pFile->pszDbPath)
	{
		f_strcpy( (char *)pTmp->szFileName, pFile->pszDbPath);
	}
	else
	{
		f_sprintf( (char *)pTmp->szFileName, "Unknown Db Name");
	}

	// Now let's see if we can get the Lock User Info for the
	// two locks - Write locks (tx) and Wait locks (db).
	if (pFile->pFileLockObj)
	{
		if (RC_BAD( rc = pFile->pFileLockObj->getLockQueue( 
			&pTmp->pDbLockUser)))
		{
			pTmp->pDbLockUser = NULL;
		}
	}
	else
	{
		pTmp->pDbLockUser = NULL;
	}

	if (pFile->pWriteLockObj)
	{
		if (RC_BAD( rc = pFile->pWriteLockObj->getLockQueue( 
			&pTmp->pTxLockUser)))
		{
			pTmp->pTxLockUser = NULL;
		}
	}
	else
	{
		pTmp->pTxLockUser = NULL;
	}


Exit:

	return;
}

/****************************************************************************
Desc:	Gets the CHECKPOINT_INFO information for the specified pFile.
****************************************************************************/
void F_StatsPage::gatherCPStats(
	STAT_GATHER *			pStatGather,
	FFILE *					pFile)
{
	CP_INFO_HEADER_p		pTmp;
	RCODE						rc = FERR_OK;

	flmAssert( pStatGather);
	flmAssert( pFile);

	// Allocate a new CP_INFO_HEADER and link it into the list.
	if( RC_BAD( rc = f_alloc( sizeof( CP_INFO_HEADER), &pTmp)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 
		sizeof( CHECKPOINT_INFO), &pTmp->pCheckpointInfo)))
	{
		goto Exit;
	}
	
	// Save the file name.
	if (pFile->pszDbPath)
	{
		f_strcpy( (char *)pTmp->szFileName, pFile->pszDbPath);
	}
	else
	{
		f_sprintf( (char *)pTmp->szFileName, "Unknown Db Name");
	}

	pTmp->pNext = pStatGather->pCPHeader;
	pStatGather->pCPHeader = pTmp;

	flmGetCPInfo( pFile, pTmp->pCheckpointInfo);

Exit:

	if (RC_BAD(rc) && pTmp)
	{
		f_free( &pTmp);
	}

	return;
}

/****************************************************************************
Desc:	Deletes all CP_INFO_HEADER structures linked from the pStatGather structure.
****************************************************************************/
void F_StatsPage::freeCPInfoHeaders(
		STAT_GATHER *		pStatGather
		)
{
	CP_INFO_HEADER_p		pTmp;

	while (pStatGather->pCPHeader)
	{
		pTmp = pStatGather->pCPHeader;
		pStatGather->pCPHeader = pStatGather->pCPHeader->pNext;

		if (pTmp->pCheckpointInfo)
		{
			f_free( &pTmp->pCheckpointInfo);
		}

		f_free( &pTmp);
	}
}


/****************************************************************************
Desc:	Prints out the Cache Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::printCacheStats(
	STAT_GATHER *		pStatGather,
	STAT_GATHER *		pOldStatGather)
{
	fnPrintf( m_pHRequest, "<br>\n");

	// Cache table.

	printTableStart( "Cache", 3, 50);

	// Cache table column headers

	printTableRowStart();
	printColumnHeading( "Stat Type", JUSTIFY_LEFT);
	printColumnHeading( "Block Cache", JUSTIFY_RIGHT);
	printColumnHeading( "Record Cache", JUSTIFY_RIGHT);
	printTableRowEnd();

	printCacheStatRow( TRUE, "Current Limit (Bytes)",
		pStatGather->BlockCache.uiMaxBytes,
		pStatGather->RecordCache.uiMaxBytes, TRUE,
		(pStatGather->BlockCache.uiMaxBytes !=
		pOldStatGather->BlockCache.uiMaxBytes ? TRUE : FALSE),
		(pStatGather->RecordCache.uiMaxBytes !=
		pOldStatGather->RecordCache.uiMaxBytes ? TRUE : FALSE));

	printCacheStatRow( FALSE, "Total Items Cached",
		pStatGather->BlockCache.uiCount,
		pStatGather->RecordCache.uiCount, TRUE,
		(pStatGather->BlockCache.uiCount !=
		pOldStatGather->BlockCache.uiCount ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCount !=
		pOldStatGather->RecordCache.uiCount ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Total Bytes Cached",
		pStatGather->BlockCache.uiTotalBytesAllocated,
		pStatGather->RecordCache.uiTotalBytesAllocated, TRUE,
		(pStatGather->BlockCache.uiTotalBytesAllocated !=
		pOldStatGather->BlockCache.uiTotalBytesAllocated ? TRUE : FALSE),
		(pStatGather->RecordCache.uiTotalBytesAllocated !=
		pOldStatGather->RecordCache.uiTotalBytesAllocated ? TRUE : FALSE));
	
	printCacheStatRow( FALSE, "Old Items Cached",
		pStatGather->BlockCache.uiOldVerCount,
		pStatGather->RecordCache.uiOldVerCount, TRUE,
		(pStatGather->BlockCache.uiOldVerCount !=
		pOldStatGather->BlockCache.uiOldVerCount ? TRUE : FALSE),
		(pStatGather->RecordCache.uiOldVerCount !=
		pOldStatGather->RecordCache.uiOldVerCount ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Old Bytes Cached",
		pStatGather->BlockCache.uiOldVerBytes,
		pStatGather->RecordCache.uiOldVerBytes, TRUE,
		(pStatGather->BlockCache.uiOldVerBytes !=
		pOldStatGather->BlockCache.uiOldVerBytes ? TRUE : FALSE),
		(pStatGather->RecordCache.uiOldVerBytes !=
		pOldStatGather->RecordCache.uiOldVerBytes ? TRUE : FALSE));
	
	printCacheStatRow( FALSE, "Hits",
		pStatGather->BlockCache.uiCacheHits,
		pStatGather->RecordCache.uiCacheHits, TRUE,
		(pStatGather->BlockCache.uiCacheHits !=
		pOldStatGather->BlockCache.uiCacheHits ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheHits !=
		pOldStatGather->RecordCache.uiCacheHits ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Hit Looks",
		pStatGather->BlockCache.uiCacheHitLooks,
		pStatGather->RecordCache.uiCacheHitLooks, TRUE,
		(pStatGather->BlockCache.uiCacheHitLooks !=
		pOldStatGather->BlockCache.uiCacheHitLooks ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheHitLooks !=
		pOldStatGather->RecordCache.uiCacheHitLooks ? TRUE : FALSE));
	
	printCacheStatRow( FALSE, "Looks per Hit",
		(pStatGather->BlockCache.uiCacheHits
		 ? pStatGather->BlockCache.uiCacheHitLooks /
		 pStatGather->BlockCache.uiCacheHits
		 : (FLMUINT)0),
		(pStatGather->RecordCache.uiCacheHits
		 ? pStatGather->RecordCache.uiCacheHitLooks /
			pStatGather->RecordCache.uiCacheHits
		 : (FLMUINT)0), TRUE,
		(pStatGather->BlockCache.uiCacheHits !=
		pOldStatGather->BlockCache.uiCacheHits ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheHits !=
		pOldStatGather->RecordCache.uiCacheHits ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Faults",
		pStatGather->BlockCache.uiCacheFaults,
		pStatGather->RecordCache.uiCacheFaults, TRUE,
		(pStatGather->BlockCache.uiCacheFaults !=
		pOldStatGather->BlockCache.uiCacheFaults ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheFaults !=
		pOldStatGather->RecordCache.uiCacheFaults ? TRUE : FALSE));
	
	printCacheStatRow( FALSE, "Fault Looks",
		pStatGather->BlockCache.uiCacheFaultLooks,
		pStatGather->RecordCache.uiCacheFaultLooks, TRUE,
		(pStatGather->BlockCache.uiCacheFaultLooks !=
		pOldStatGather->BlockCache.uiCacheFaultLooks ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheFaultLooks !=
		pOldStatGather->RecordCache.uiCacheFaultLooks ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Looks Per Fault",
		(pStatGather->BlockCache.uiCacheFaults
		 ? pStatGather->BlockCache.uiCacheFaultLooks /
			pStatGather->BlockCache.uiCacheFaults
		 : (FLMUINT)0),
		(pStatGather->RecordCache.uiCacheFaults
		 ? pStatGather->RecordCache.uiCacheFaultLooks /
			pStatGather->RecordCache.uiCacheFaults
		 : (FLMUINT)0), TRUE,
		(pStatGather->BlockCache.uiCacheFaults !=
		pOldStatGather->BlockCache.uiCacheFaults ? TRUE : FALSE),
		(pStatGather->RecordCache.uiCacheFaults !=
		pOldStatGather->RecordCache.uiCacheFaults ? TRUE : FALSE));

	printCacheStatRow( FALSE, "Dirty Blocks",
		pStatGather->uiDirtyBlocks, 0, FALSE,
		(pStatGather->uiDirtyBlocks !=
		pOldStatGather->uiDirtyBlocks ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Dirty Bytes",
		pStatGather->uiDirtyBytes, 0, FALSE,
		(pStatGather->uiDirtyBytes !=
		pOldStatGather->uiDirtyBytes ? TRUE : FALSE));
	
	printCacheStatRow( FALSE, "Log Blocks",
		pStatGather->uiLogBlocks, 0, FALSE,
		(pStatGather->uiLogBlocks !=
		pOldStatGather->uiLogBlocks ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Log Bytes",
		pStatGather->uiLogBytes, 0, FALSE,
		(pStatGather->uiLogBytes !=
		pOldStatGather->uiLogBytes ? TRUE : FALSE));

	printCacheStatRow( FALSE, "Free Blocks",
		pStatGather->uiFreeCount, 0, FALSE,
		(pStatGather->uiFreeCount !=
		pOldStatGather->uiFreeCount ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Free Bytes",
		pStatGather->uiFreeBytes, 0, FALSE,
		(pStatGather->uiFreeBytes !=
		pOldStatGather->uiFreeBytes ? TRUE : FALSE));

	printCacheStatRow( FALSE, "Replaceable Blocks",
		pStatGather->uiReplaceableCount, 0, FALSE,
		(pStatGather->uiReplaceableCount !=
		pOldStatGather->uiReplaceableCount ? TRUE : FALSE));
	
	printCacheStatRow( TRUE, "Replaceable Bytes",
		pStatGather->uiReplaceableBytes, 0, FALSE,
		(pStatGather->uiReplaceableBytes !=
		pOldStatGather->uiReplaceableBytes ? TRUE : FALSE));

	printTableEnd();
}


/****************************************************************************
Desc:	Prints out the Operation Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::printOperationStats(
	STAT_GATHER *		pStatGather,
	STAT_GATHER *		pOldStatGather)
{
	F_COUNT_TIME_STAT	Stat;
	F_COUNT_TIME_STAT	OldStat;
	FLMBOOL				bHighlight;

	if (pStatGather->uiStartTime)
	{
		fnPrintf( m_pHRequest, "<br>\n");

		// Database operations table

		printTableStart( "Database Operations", 4, 75);

		// Operations table column headers

		printTableRowStart();
		printColumnHeading( "Operation", JUSTIFY_LEFT);
		printColumnHeading( "Count", JUSTIFY_RIGHT);
		printColumnHeading( "Total Seconds", JUSTIFY_RIGHT);
		printColumnHeading( "Avg Seconds", JUSTIFY_RIGHT);
		printTableRowEnd();

		// Transaction rows

		bHighlight = FALSE;
		printCountTimeRow( bHighlight = ~bHighlight, "Committed Update Trans",
			&pStatGather->CommittedUpdTrans,
			&pOldStatGather->CommittedUpdTrans);
		printCountTimeRow( bHighlight = ~bHighlight, "Aborted Update Trans",
			&pStatGather->AbortedUpdTrans,
			&pOldStatGather->AbortedUpdTrans);
		printCountTimeRow( bHighlight = ~bHighlight, "Group Finishes",
			&pStatGather->GroupCompletes,
			&pOldStatGather->GroupCompletes);

		Stat.ui64Count = pStatGather->ui64GroupFinished;
		OldStat.ui64Count = pOldStatGather->ui64GroupFinished;
		printCountTimeRow( bHighlight = ~bHighlight, "Total Finished",
			&Stat, &OldStat, TRUE);

		if (pStatGather->GroupCompletes.ui64Count)
		{
			Stat.ui64Count = pStatGather->ui64GroupFinished /
				pStatGather->GroupCompletes.ui64Count;
		}
		else
		{
			Stat.ui64Count = 0;
		}
		if (pOldStatGather->GroupCompletes.ui64Count)
		{
			OldStat.ui64Count = pOldStatGather->ui64GroupFinished /
				pOldStatGather->GroupCompletes.ui64Count;
		}
		else
		{
			OldStat.ui64Count = 0;
		}
		printCountTimeRow( bHighlight = ~bHighlight, "Average Per Group",
			&Stat, &OldStat, TRUE);

		printCountTimeRow( bHighlight = ~bHighlight, "Committed Read Trans",
			&pStatGather->CommittedReadTrans,
			&pOldStatGather->CommittedReadTrans);
		printCountTimeRow( bHighlight = ~bHighlight, "Aborted Read Trans",
			&pStatGather->AbortedReadTrans,
			&pOldStatGather->AbortedReadTrans);
		printCountTimeRow( bHighlight = ~bHighlight, "Reads",
			&pStatGather->Reads,
			&pOldStatGather->Reads);
		printCountTimeRow( bHighlight = ~bHighlight, "Adds",
			&pStatGather->Adds,
			&pOldStatGather->Adds);
		printCountTimeRow( bHighlight = ~bHighlight, "Modifies",
			&pStatGather->Modifies,
			&pOldStatGather->Modifies);
		printCountTimeRow( bHighlight = ~bHighlight, "Deletes",
			&pStatGather->Deletes,
			&pOldStatGather->Deletes);
		printCountTimeRow( bHighlight = ~bHighlight, "Queries",
			&pStatGather->Queries,
			&pOldStatGather->Queries, TRUE);
		printCountTimeRow( bHighlight = ~bHighlight, "Query Reads",
			&pStatGather->QueryReads,
			&pOldStatGather->QueryReads, TRUE);

		Stat.ui64Count = pStatGather->ui64BlockSplits;
		OldStat.ui64Count = pOldStatGather->ui64BlockSplits;
		printCountTimeRow( bHighlight = ~bHighlight, "Block Splits",
			&Stat, &OldStat, TRUE);

		Stat.ui64Count = pStatGather->ui64BlockCombines;
		OldStat.ui64Count = pOldStatGather->ui64BlockCombines;
		printCountTimeRow( bHighlight = ~bHighlight, "Block Combines",
			&Stat, &OldStat, TRUE);

		printTableEnd();
	}
}

/****************************************************************************
Desc:	Prints out the Lock Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::printLockStats(
	STAT_GATHER *		pStatGather,
	STAT_GATHER *		pOldStatGather)
{
	FLMUINT				uiLen;
	
	if (pStatGather->uiStartTime)
	{
		LOCK_USER_HEADER_p		pLckHdr;
		fnPrintf( m_pHRequest, "<br>\n");

		// Lock table.

		printTableStart( "Locks", 4, 75);

		// Locks table column headers

		printTableRowStart();
		printColumnHeading( "Stat Type", JUSTIFY_LEFT);
		printColumnHeading( "Count", JUSTIFY_RIGHT);
		printColumnHeading( "Total Seconds", JUSTIFY_RIGHT);
		printColumnHeading( "Avg Seconds", JUSTIFY_RIGHT);
		printTableRowEnd();

		printCountTimeRow( TRUE, "Time No Locks Held",
			&pStatGather->LockStats.NoLocks,
			&pOldStatGather->LockStats.NoLocks);
			
		printCountTimeRow( FALSE, "Time Waiting for Locks",
			&pStatGather->LockStats.WaitingForLock,
			&pOldStatGather->LockStats.WaitingForLock);
			
		printCountTimeRow( TRUE, "Time Locks Held",
			&pStatGather->LockStats.HeldLock,
			&pOldStatGather->LockStats.HeldLock);

		printTableEnd();

		// Display the Lock Queue for each of the files open...
		pLckHdr = pStatGather->pLockUsers;

		while (pLckHdr)
		{
			char				szTitle[ 128];
			FLMBOOL			bHighlight = FALSE;
			FLMBOOL			bLocked = TRUE;
			F_LOCK_USER *	pLckUsr;
			FLMUINT			uiTxWaiters;
			FLMUINT			uiDbWaiters;

			uiTxWaiters = 0;
			pLckUsr = pLckHdr->pTxLockUser;
			while (pLckUsr && pLckUsr->uiThreadId)
			{
				uiTxWaiters++;
				pLckUsr++;
			}

			if( uiTxWaiters)
			{
				uiTxWaiters--;
			}

			uiDbWaiters = 0;
			pLckUsr = pLckHdr->pDbLockUser;
			while (pLckUsr && pLckUsr->uiThreadId)
			{
				uiDbWaiters++;
				pLckUsr++;
			}

			if( uiDbWaiters)
			{
				uiDbWaiters--;
			}

			fnPrintf( m_pHRequest, "<br>\n");

			// Start the new table
			f_sprintf( (char *)szTitle, 
				"Lock Queue - %s, TX Waiters: %u, DB Waiters: %u", pLckHdr->szFileName,
				(unsigned)uiTxWaiters, (unsigned)uiDbWaiters);

			printTableStart( (char *)szTitle, 4, 75);

			printTableRowStart( bHighlight = ~bHighlight);
			printColumnHeading( "Thread Id", JUSTIFY_LEFT);
			printColumnHeading( "Name", JUSTIFY_RIGHT);
			printColumnHeading( "Status", JUSTIFY_RIGHT);
			printColumnHeading( "Time", JUSTIFY_RIGHT);
			printTableRowEnd();

			// Display the body...
			pLckUsr = pLckHdr->pTxLockUser;

			while (pLckUsr && pLckUsr->uiThreadId)
			{
				char		szThreadName[ 50];

				printTableRowStart( bHighlight = ~bHighlight);
				printTableDataStart( TRUE, JUSTIFY_LEFT);
				fnPrintf( m_pHRequest, "%u", (unsigned)pLckUsr->uiThreadId);
				printTableDataEnd();

				printTableDataStart( TRUE, JUSTIFY_RIGHT);
				uiLen = sizeof( szThreadName);
				gv_FlmSysData.pThreadMgr->getThreadName( 
					pLckUsr->uiThreadId, szThreadName, &uiLen);
				fnPrintf( m_pHRequest, "%s", szThreadName);
				printTableDataEnd();
				// Status
				printTableDataStart( TRUE, JUSTIFY_RIGHT);
				fnPrintf( m_pHRequest, "%s (Tx)", bLocked ? "Locked" : "Waiting");
				bLocked = FALSE;
				printTableDataEnd();

				// Time in status
				printElapTime( (FLMUINT64)pLckUsr->uiTime, NULL, JUSTIFY_RIGHT, TRUE);

				printTableRowEnd();

				// Next entry...
				pLckUsr++;

			}

			// Display the Db info.
			pLckUsr = pLckHdr->pDbLockUser;
			bLocked = TRUE;

			while (pLckUsr && pLckUsr->uiThreadId)
			{
				char		szThreadName[ 50];

				printTableRowStart( bHighlight = ~bHighlight);
				printTableDataStart( TRUE, JUSTIFY_LEFT);
				fnPrintf( m_pHRequest, "%u", (unsigned)pLckUsr->uiThreadId);
				printTableDataEnd();

				printTableDataStart( TRUE, JUSTIFY_RIGHT);
				uiLen = sizeof( szThreadName);
				gv_FlmSysData.pThreadMgr->getThreadName( pLckUsr->uiThreadId, 
					szThreadName, &uiLen);
				fnPrintf( m_pHRequest, "%s", szThreadName);
				printTableDataEnd();
				// Status
				printTableDataStart( TRUE, JUSTIFY_RIGHT);
				fnPrintf( m_pHRequest, "%s (Db)", bLocked ? "Locked" : "Waiting");
				bLocked = FALSE;
				printTableDataEnd();

				// Time in status
				printElapTime( (FLMUINT64)pLckUsr->uiTime, NULL, JUSTIFY_RIGHT, TRUE);

				printTableRowEnd();

				// Next entry...
				pLckUsr++;

			}
			
			
			printTableEnd();

			pLckHdr = pLckHdr->pNext;
		}
	}
}



/****************************************************************************
Desc:	Prints out the Disk Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::printDiskStats(
	STAT_GATHER *		pStatGather,
	STAT_GATHER *		pOldStatGather)
{
	char			szTemp[ 100];
	FLMBOOL		bHighlight = FALSE;

	if (pStatGather->uiStartTime)
	{
		fnPrintf( m_pHRequest, "<br>\n");

		// Disk IO table.
		f_sprintf( (char *)szTemp, "Disk IO");

		if (m_pFocusBlock)
		{
			f_strcat( (char *)szTemp, " - focus enabled on ");
			f_strcat( (char *)szTemp, (char *)m_pFocusBlock->szFileName);
			if (m_pFocusBlock->uiLFileNum > 0)
			{
				char		szLFNum[ 20];

				f_strcat( (char *)szTemp, " on logical file ");
				f_sprintf( (char *)szLFNum, "%lu", m_pFocusBlock->uiLFileNum);
				f_strcat( (char *)szTemp, (char *)szLFNum);
			}
		}

		printTableStart( (char *)szTemp, 5, 100);

		// Database operations table column headers

		printTableRowStart();
		printColumnHeading( "IO CATEGORY", JUSTIFY_LEFT);
		printColumnHeading( "Count", JUSTIFY_RIGHT);
		printColumnHeading( "Total Bytes", JUSTIFY_RIGHT);
		printColumnHeading( "Total Seconds", JUSTIFY_RIGHT);
		printColumnHeading( "Avg Seconds", JUSTIFY_RIGHT);
		printTableRowEnd();

		printIORow( bHighlight = !bHighlight, "<strong>READS</strong>",
			&pStatGather->IOReads,
			&pOldStatGather->IOReads);
		printIORow( bHighlight = !bHighlight, "Root Blocks", 
			&pStatGather->IORootBlockReads, 
			&pOldStatGather->IORootBlockReads);
		printIORow( bHighlight = !bHighlight, "Non-Leaf Blocks",
			&pStatGather->IONonLeafBlockReads,
			&pOldStatGather->IONonLeafBlockReads);
		printIORow( bHighlight = !bHighlight, "Leaf Blocks",
			&pStatGather->IOLeafBlockReads,
			&pOldStatGather->IOLeafBlockReads);

		if ((m_pFocusBlock == NULL) ||
			 (m_pFocusBlock->uiLFileNum == 0))
		{
			printIORow( bHighlight = !bHighlight, "Avail Blocks",
				&pStatGather->IOAvailBlockReads,
				&pOldStatGather->IOAvailBlockReads);
			printIORow( bHighlight = !bHighlight, "LFH Blocks",
				&pStatGather->IOLFHBlockReads,
				&pOldStatGather->IOLFHBlockReads);
			printIORow( bHighlight = !bHighlight, "Prior Image Blocks",
				&pStatGather->IORollbackBlockReads,
				&pOldStatGather->IORollbackBlockReads);

			printTableRowStart( bHighlight = !bHighlight); 
			printTableDataStart( TRUE, JUSTIFY_LEFT);
			fnPrintf( m_pHRequest, "Read Errors");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "%s%u%s",
				(pStatGather->uiReadErrors != pOldStatGather->uiReadErrors ? "<font color=red>" : ""),
				(unsigned)pStatGather->uiReadErrors,
				(pStatGather->uiReadErrors != pOldStatGather->uiReadErrors ? "</font>" : ""));
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableRowEnd();

			printTableRowStart( bHighlight = !bHighlight);
			printTableDataStart( TRUE, JUSTIFY_LEFT);
			fnPrintf( m_pHRequest, "Check Errors");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "%s%u%s",
				(pStatGather->uiCheckErrors != pOldStatGather->uiCheckErrors ? "<font color=red>" : ""),
				(unsigned)pStatGather->uiCheckErrors,
				(pStatGather->uiCheckErrors != pOldStatGather->uiCheckErrors ? "</font>" : ""));
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableRowEnd();
		}

		printIORow( bHighlight = !bHighlight, "<strong>WRITES</strong>",
			&pStatGather->IOWrites,
			&pOldStatGather->IOWrites);
		printIORow( bHighlight = !bHighlight, "Root Blocks",
			&pStatGather->IORootBlockWrites,
			&pOldStatGather->IORootBlockWrites);
		printIORow( bHighlight = !bHighlight, "Non-Leaf Blocks",
			&pStatGather->IONonLeafBlockWrites,
			&pOldStatGather->IONonLeafBlockWrites);
		printIORow( bHighlight = !bHighlight, "Leaf Blocks",
			&pStatGather->IOLeafBlockWrites,
			&pOldStatGather->IOLeafBlockWrites);

		if ((m_pFocusBlock == NULL) ||
			 (m_pFocusBlock->uiLFileNum == 0))
		{
			printIORow( bHighlight = !bHighlight, "Avail Blocks",
				&pStatGather->IOAvailBlockWrites,
				&pOldStatGather->IOAvailBlockWrites);
			printIORow( bHighlight = !bHighlight, "LFH Blocks",
				&pStatGather->IOLFHBlockWrites,
				&pOldStatGather->IOLFHBlockWrites);
			printIORow( bHighlight = !bHighlight, "Rollback Log Blocks",
				&pStatGather->IORollBackLogWrites,
				&pOldStatGather->IORollBackLogWrites);
			printIORow( bHighlight = !bHighlight, "Log Header",
				&pStatGather->IOLogHdrWrites,
				&pOldStatGather->IOLogHdrWrites);
			printIORow( bHighlight = !bHighlight, "Undo Blocks",
				&pStatGather->IORolledbackBlockWrites,
				&pOldStatGather->IORolledbackBlockWrites);

			printTableRowStart( bHighlight = !bHighlight); 
			printTableDataStart( TRUE, JUSTIFY_LEFT);
			fnPrintf( m_pHRequest, "Write Errors");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "%s%u%s",
				(pStatGather->uiWriteErrors != pOldStatGather->uiWriteErrors ? "<font color=red>" : ""),
				(unsigned)pStatGather->uiWriteErrors,
				(pStatGather->uiWriteErrors != pOldStatGather->uiWriteErrors ? "</font>" : ""));
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableRowEnd();
		}

		printTableEnd();
	}
}

/****************************************************************************
Desc:	Prints out the Checkpoint Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::printCPStats(
	STAT_GATHER *		pStatGather)
{
	CP_INFO_HEADER_p	pCPHdr;

	fnPrintf( m_pHRequest, "<br>\n");

	// Checkpoint Thread table.
	pCPHdr = pStatGather->pCPHeader;

	while (pCPHdr)
	{
		char						szTitle[ 50];
		CHECKPOINT_INFO *		pCPInfo;
		FLMBOOL					bHighlight = FALSE;

		f_sprintf( (char *)szTitle, "Checkpoint Thread - %s", pCPHdr->szFileName);

		printTableStart( (char *)szTitle, 2, 50);
		printTableRowStart();
		printColumnHeading( "Stat Type", JUSTIFY_LEFT);
		printColumnHeading( "Value", JUSTIFY_RIGHT);
		printTableRowEnd();

		pCPInfo = pCPHdr->pCheckpointInfo;

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "State");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%s", pCPInfo->bRunning ? "Yes" : "No");
		printTableDataEnd();
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Running Time");
		printTableDataEnd();
		printElapTime( (FLMUINT64)pCPInfo->uiRunningTime, NULL, JUSTIFY_RIGHT, TRUE);
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Forcing Checkpoint");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%s", pCPInfo->bForcingCheckpoint ? "Yes" : "No");
		printTableDataEnd();
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Forced Checkpoint Running Time");
		printTableDataEnd();
		printElapTime( (FLMUINT64)pCPInfo->uiForceCheckpointRunningTime, NULL, JUSTIFY_RIGHT, TRUE);
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Forced Checkpoint Reason");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		switch( pCPInfo->iForceCheckpointReason)
		{
			case CP_TIME_INTERVAL_REASON:
				fnPrintf( m_pHRequest, "Time interval");
				break;
			case CP_SHUTTING_DOWN_REASON:
				fnPrintf( m_pHRequest, "Shutting down");
				break;
			case CP_RFL_VOLUME_PROBLEM:
				fnPrintf( m_pHRequest, "RFL volume problem");
				break;
			default:
				fnPrintf( m_pHRequest, "Unknown");
				break;
		}
		printTableDataEnd();
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Waiting for Read Trans Time");
		printTableDataEnd();
		printElapTime( (FLMUINT64)pCPInfo->uiWaitTruncateTime, NULL, JUSTIFY_RIGHT, TRUE);
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Writing Data Blocks");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%s", pCPInfo->bWritingDataBlocks ? "Yes" : "No");
		printTableDataEnd();
		printTableRowEnd();

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Log Blocks Written");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%u", pCPInfo->uiLogBlocksWritten);
		printTableDataEnd();
		printTableRowEnd();
		
		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Data Blocks Written");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%u", pCPInfo->uiDataBlocksWritten);
		printTableDataEnd();
		printTableRowEnd();

		if (pCPInfo->uiDirtyCacheBytes && pCPInfo->uiBlockSize)
		{
			printTableRowStart( bHighlight = !bHighlight);
			printTableDataStart( TRUE, JUSTIFY_LEFT);
			fnPrintf( m_pHRequest, "Dirty Cache Blocks");
			printTableDataEnd();
			printTableDataStart( TRUE, JUSTIFY_RIGHT);
			fnPrintf( m_pHRequest, "%u", pCPInfo->uiDirtyCacheBytes / pCPInfo->uiBlockSize);
			printTableDataEnd();
			printTableRowEnd();
		}

		printTableRowStart( bHighlight = !bHighlight);
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		fnPrintf( m_pHRequest, "Block Size");
		printTableDataEnd();
		printTableDataStart( TRUE, JUSTIFY_RIGHT);
		fnPrintf( m_pHRequest, "%u", pCPInfo->uiBlockSize);
		printTableDataEnd();
		printTableRowEnd();

		printTableEnd();

		pCPHdr = pCPHdr->pNext;

	}
}

/****************************************************************************
Desc:	Prints out the Checkpoint Stats stored in pStatGather.
****************************************************************************/
void F_StatsPage::displayFocus(
	FLMUINT			uiNumParams,
	const char **	ppszParams)
{
	FLMBOOL			bFocusAll;
	FLMBOOL			bFocusLFile;
	FLMBOOL			bFocusDb;
	void *			pvSession = NULL;
	char				szTmpFocus[ 1] = { 0 };
	FLMUINT			uiLoop;

	bFocusAll = DetectParameter( uiNumParams, ppszParams, "All");
	bFocusLFile = DetectParameter( uiNumParams, ppszParams, "LFile");
	bFocusDb = DetectParameter( uiNumParams, ppszParams, "Db");

	if (gv_FlmSysData.HttpConfigParms.fnAcquireSession)
	{
		if ((pvSession = fnAcquireSession()) == NULL)
		{
			printErrorPage( FERR_FAILURE, TRUE, "Could not obtain session handle");
			goto Exit;
		}
	}
	
	if (!bFocusLFile & !bFocusDb & !bFocusAll)
	{

		printDocStart( "Focus");


		fnPrintf( m_pHRequest, "<form name=\"focusAll\" method=\"get\" action=%s/Stats>\n",
			m_pszURLString);
		fnPrintf( m_pHRequest, "<input type=hidden name=\"Focus\" value=\"\">\n");
		fnPrintf( m_pHRequest, "<input type=hidden name=\"All\" value=\"\">\n");

		printTableStart( "All Databases", 1, 100);
		printTableEnd();
		printButton( "Submit", BT_Submit);
		fnPrintf( m_pHRequest, "</form>\n");

		// We need to collect a list of databases to present.

		f_mutexLock( gv_FlmSysData.Stats.hMutex);

		for (uiLoop = 0; uiLoop < gv_FlmSysData.Stats.uiNumDbStats; uiLoop++)
		{
			FLMBOOL			bHighlight = FALSE;

			fnPrintf( m_pHRequest, "<form name=\"focus%d\" method=\"get\" action=%s/Stats>\n",
				uiLoop, m_pszURLString);
			fnPrintf( m_pHRequest, "<input type=hidden name=\"Focus\" value=\"\">\n");
			fnPrintf( m_pHRequest, "<input type=hidden name=\"Db\" value=\"%s\">\n",
				gv_FlmSysData.Stats.pDbStats[ uiLoop].pszDbName); 
			// Start a new table...
			printTableStart(	(char *)gv_FlmSysData.Stats.pDbStats[ uiLoop].pszDbName, 3, 100);
			printTableRowStart();
			printColumnHeading( "Select",JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 1, 1);
			printColumnHeading( "Logical File Type", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 1, 1);
			printColumnHeading( "Logical File Number", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 1, 1);
			printTableRowEnd();
			printTableRowStart( bHighlight = !bHighlight);
			printTableDataStart();
			fnPrintf( m_pHRequest, "<input name=\"LFile\" value=\"0\" checked type=\"radio\">\n");
			printTableDataEnd();
			printTableDataStart( );
			fnPrintf( m_pHRequest, "All Logical files\n");
			printTableDataEnd();
			printTableDataStart();
			fnPrintf( m_pHRequest, "N/A");
			printTableDataEnd();
			printTableRowEnd();

			// Now for each file, present the LFiles...
	 		for (int iLoop = 0; iLoop < (int)gv_FlmSysData.Stats.pDbStats[ uiLoop].uiNumLFileStats;
							iLoop++)
			{
				printTableRowStart( bHighlight = !bHighlight);
				printTableDataStart();
				fnPrintf( m_pHRequest, "<input name=\"LFile\" value=\"%u\" type=\"radio\">",
					gv_FlmSysData.Stats.pDbStats[ uiLoop].pLFileStats[iLoop].uiLFileNum);
				printTableDataEnd();
				printTableDataStart();
				fnPrintf( m_pHRequest, "%s",
					(gv_FlmSysData.Stats.pDbStats[ uiLoop].pLFileStats[iLoop].uiFlags & 
								 LFILE_IS_INDEX ? "Index" : 
					(gv_FlmSysData.Stats.pDbStats[ uiLoop].pLFileStats[iLoop].uiFlags &
								LFILE_TYPE_UNKNOWN ? "Unknown" : "Container")));
				printTableDataEnd();
				printTableDataStart();
				fnPrintf( m_pHRequest, "%u",
						gv_FlmSysData.Stats.pDbStats[ uiLoop].pLFileStats[iLoop].uiLFileNum);
				printTableDataEnd();
				printTableRowEnd();
			}
			printTableEnd();
			printButton( "Submit", BT_Submit);
			fnPrintf( m_pHRequest, "</form>\n");
		}

		f_mutexUnlock( gv_FlmSysData.Stats.hMutex);
		printDocEnd();
		goto Exit;
	}

	if (bFocusAll)
	{
		// A request to set the focus to all indicates that we are currently
		// focusing on something other than All, so we will need to delete
		// the current focus setting.  Setting the value to NULL will delete
		// the existing entry.  If we find an entry in m_pFocusBlock
		// (which we shouldn't) we will delete it.
		
		if (m_pFocusBlock)
		{
			flmAssert( 0);
			f_free( &m_pFocusBlock);
		}

		if (fnSetSessionValue( pvSession,
				STAT_FOCUS, (void *)szTmpFocus, 0) != 0)
		{
			flmAssert( 0);
			printErrorPage( FERR_MEM, TRUE, "Could not process request due to a memory allocation failure");
			goto Exit;
		}
	}
	else
	{
		// We assume we have bDb set since bLFile cannot be set without it.
		// Retrieve the Db value.
		
		char			szDb[ 101];
		char			szLFile[ 21];
		char			szTemp[ 123];

		if (RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
										"Db", sizeof( szDb), szDb)))
		{
			printErrorPage( FERR_INVALID_PARM, TRUE, "Parameter Db not present.  Could not process this request.");
			goto Exit;
		}

		if (bFocusLFile)
		{
			if (RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
											"LFile", sizeof( szLFile), szLFile)))
			{
				printErrorPage( FERR_INVALID_PARM, TRUE, "Parameter Db not present.  Could not process this request.");
				goto Exit;
			}
		}
		
		fcsDecodeHttpString( szDb);
		f_sprintf( (char *)szTemp, "%.100s;%.20s", szDb, (char *)szLFile);

		// Now save this in the current session...
		if (fnSetSessionValue( pvSession,
							  STAT_FOCUS,
							  (void *)szTemp,
							  (FLMSIZET)f_strlen(szTemp)) != 0)
		{
			flmAssert( 0);
			goto Exit;
		}
	}

	// Getting to this point indicates success.  We will return a confirmation page.
	printDocStart( "Focus - Confirmation");
	fnPrintf( m_pHRequest, "<script>this.close()</script>\n",
					m_pszURLString);
	printDocEnd();

Exit:

	if (pvSession)
	{
		fnReleaseSession( pvSession);
	}

}

/****************************************************************************
Desc:	Prints out the Checkpoint Stats stored in pStatGather.
****************************************************************************/
RCODE F_StatsPage::setFocus(
	char *		pszFocus)
{
	RCODE			rc = FERR_OK;
	char *		pTmp;

	if (m_pFocusBlock)
	{
		flmAssert( 0);
		f_free( &m_pFocusBlock);
	}

	if (f_strlen( pszFocus) == 0)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 
		sizeof( FOCUS_BLOCK), &m_pFocusBlock)))
	{
		goto Exit;
	}

	pTmp = pszFocus;

	m_pFocusBlock->uiLFileNum = 0;

	while (*pTmp != ';' && *pTmp != '\0')
	{
		pTmp++;
	}
	
	*pTmp = 0;
	f_strcpy( m_pFocusBlock->szFileName, pszFocus);
	pTmp++;

	if( *pTmp != '\0')
	{
		m_pFocusBlock->uiLFileNum = f_atoud( pTmp);
	}

Exit:

	return( rc);
}
