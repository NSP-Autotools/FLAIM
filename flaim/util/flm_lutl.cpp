//-------------------------------------------------------------------------
// Desc:	Utility routines for presenting selection and statistics lists.
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

#include "flm_lutl.h"

extern FLMBOOL		gv_bShutdown;

FSTATIC RCODE ixDisplayHook(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList);

FSTATIC void flstIndexUpdateEventHook( 
	FEventType	eEventType,
	void *		pvAppData,
	void *		pvEventData1,
	void *		pvEventData2);

#define MAX_VALS_TO_SAVE	10

typedef struct IxDisplayInfo
{
	char				szName [20];
	FINDEX_STATUS	IndexStatus;
	FLMUINT			uiSaveRecsProcessed [MAX_VALS_TO_SAVE];
	FLMUINT			uiRecSaveTime [MAX_VALS_TO_SAVE];
	FLMUINT			uiOldestSaved;
	FLMUINT			uiIndexingRate;
	FLMBOOL			bShowTime;
	F_MUTEX			hScreenMutex;
	FTX_SCREEN *	pScreen;
	FTX_WINDOW *	pLogWin;
	FFILE *			pFile;
} IX_DISPLAY_INFO;

FSTATIC RCODE ixDisplayHook(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList
	)
{
	eColorType			uiBack = FLM_CYAN;
	eColorType			uiFore = FLM_WHITE;
	IX_DISPLAY_INFO *	pDispInfo = (IX_DISPLAY_INFO *)pvData;
	char					szTmpBuf [100];
	const char * 		pszState;
	FLMUINT				uiGMT;


	F_UNREFERENCED_PARM( uiKey);
	F_UNREFERENCED_PARM( uiDataLen);

	flmAssert( uiDataLen == sizeof( IX_DISPLAY_INFO));
	f_timeGetSeconds( &uiGMT );

	if( pDispInfo->IndexStatus.bSuspended)
	{
		pszState = "Susp";
	}
	else if( pDispInfo->IndexStatus.uiLastRecordIdIndexed == RECID_UNDEFINED)
	{
		pszState = "Onln";
	}
	else
	{
		pszState = "Offln";
	}

	if( pDispInfo->IndexStatus.uiLastRecordIdIndexed != RECID_UNDEFINED)
	{
		f_sprintf( szTmpBuf, "%.5u %-15.15s %-5.5s %-10u %-10u %-10u %-10u %-10u",
			(unsigned)uiKey, pDispInfo->szName, pszState,
			(unsigned)pDispInfo->IndexStatus.uiLastRecordIdIndexed,
			(unsigned)pDispInfo->uiIndexingRate,
			(unsigned)pDispInfo->IndexStatus.uiKeysProcessed,
			(unsigned)pDispInfo->IndexStatus.uiRecordsProcessed,
			(unsigned)(pDispInfo->bShowTime 
								? pDispInfo->IndexStatus.uiStartTime 
										? uiGMT - pDispInfo->IndexStatus.uiStartTime 
										: 0 
								: pDispInfo->IndexStatus.uiTransactions));

	}
	else
	{
		f_sprintf( szTmpBuf, 
			"%.5u %-15.15s %-5.5s                                             %-10u", 
			(unsigned)uiKey, pDispInfo->szName, pszState,
			(unsigned)(pDispInfo->bShowTime 
								? pDispInfo->IndexStatus.uiStartTime 
										? uiGMT - pDispInfo->IndexStatus.uiStartTime 
										: 0 
								: pDispInfo->IndexStatus.uiTransactions));
	}

	FTXWinSetCursorPos( pWin, 0, uiRow);
	FTXWinClearToEOL( pWin);
	FTXWinPrintf( pWin, "%s", szTmpBuf);
	if( bSelected && pDynamicList->getShowHorizontalSelector())
	{
		FTXWinPaintRow( pWin, &uiBack, &uiFore, uiRow);
	}
	return( FERR_OK);
}

/****************************************************************************
Desc:	Event callback
*****************************************************************************/
FSTATIC void flstIndexUpdateEventHook( 
	FEventType	eEventType,
	void *		pvAppData,
	void *		pvEventData1,
	void *		pvEventData2)
{
	FDB *		pDb = NULL;
	RCODE		rc = FERR_OK;

	if( eEventType == F_EVENT_INDEXING_COMPLETE)
	{
		IX_DISPLAY_INFO *		pDispInfo = (IX_DISPLAY_INFO *)pvAppData;
		FLMUINT					uiIndexNum = (FLMUINT)pvEventData1;
		FINDEX_STATUS			ixStatus;
		FLMUINT					uiGMT;

		if( !pvEventData2)
		{
			if( RC_BAD( rc = flmOpenFile( pDispInfo->pFile,
				NULL, NULL, NULL, 0, TRUE, NULL, NULL, 
				pDispInfo->pFile->pszDbPassword, &pDb)))
			{
				goto Exit;
			}

			FlmIndexStatus( (HFDB)pDb, uiIndexNum, &ixStatus);

			f_timeGetSeconds( &uiGMT );

			f_mutexLock( pDispInfo->hScreenMutex);
			FTXWinPrintf( pDispInfo->pLogWin, 
				"Index %u came on-line.  Elapsed time = %u second(s)\n", 
				uiIndexNum, uiGMT - ixStatus.uiStartTime);
			f_mutexUnlock( pDispInfo->hScreenMutex);
		}
	}

Exit:

	if( pDb)
	{
		(void)FlmDbClose( (HFDB *) &pDb);
	}
}

/****************************************************************************
Desc:	Thread that displays the current status of all indexes in a database
Note:	The caller must open the database and pass a handle to the thread.
		The handle will be closed when the thread exits.
*****************************************************************************/
RCODE FLMAPI flstIndexManagerThread(
	IF_Thread *		pThread)
{
	F_DynamicList *		pList = f_new F_DynamicList;
	FTX_WINDOW *			pTitleWin;
	FTX_WINDOW *			pListWin;
	FTX_WINDOW *			pHeaderWin;
	FTX_WINDOW *			pMsgWin;
	char						szName[ 100];
	FLMUINT					uiIterations = 0;
	FLMUINT					uiScreenCols;
	FLMUINT					uiScreenRows;
	FLMUINT					uiDrn;
	FLMUINT					uiBufLen;
	FLMUINT					uiUpdateInterval;
	FLMUINT					uiLastUpdateTime;
	IX_DISPLAY_INFO		IxDispInfo;
	FlmRecord *				pRec = NULL;
	DLIST_NODE *			pTmpNd;
	FLMUINT					uiKey;
	FLMBOOL					bShowOnline = TRUE;
	HFDB						hDb = (HFDB)pThread->getParm1();
	IX_DISPLAY_INFO *		pDispInfo;
	FLMUINT					uiOneSec;
	FLMBOOL					bScreenLocked = FALSE;
	HFEVENT					hEvent = HFEVENT_NULL;

#define FIMT_TITLE_HEIGHT		1
#define FIMT_HEADER_HEIGHT		4
#define FIMT_LOG_HEIGHT			10

	f_memset( &IxDispInfo, 0, sizeof( IX_DISPLAY_INFO));
	IxDispInfo.hScreenMutex = F_MUTEX_NULL;
	IxDispInfo.pFile = ((FDB *)hDb)->pFile;
	IxDispInfo.bShowTime = TRUE;

	if( RC_BAD( f_mutexCreate( &IxDispInfo.hScreenMutex)))
	{
		goto Exit;
	}

	if( RC_BAD( FTXScreenInit( "Index Manager", &IxDispInfo.pScreen)))
	{
		goto Exit;
	}

	FTXScreenGetSize( IxDispInfo.pScreen, &uiScreenCols, &uiScreenRows);
	FTXScreenDisplay( IxDispInfo.pScreen);

	if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, 0, 
		FIMT_TITLE_HEIGHT, &pTitleWin)))
	{
		goto Exit;
	}

	FTXWinSetBackFore( pTitleWin, FLM_RED, FLM_WHITE);
	FTXWinClear( pTitleWin);
	FTXWinPrintStr( pTitleWin, "FLAIM Index Manager");
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, 
		uiScreenCols, FIMT_HEADER_HEIGHT, &pHeaderWin)))
	{
		goto Exit;
	}

	FTXWinMove( pHeaderWin, 0, FIMT_TITLE_HEIGHT);
	FTXWinSetBackFore( pHeaderWin, FLM_BLUE, FLM_WHITE);
	FTXWinClear( pHeaderWin);
	FTXWinSetCursorType( pHeaderWin, FLM_CURSOR_INVISIBLE);
	FTXWinSetScroll( pHeaderWin, FALSE);
	FTXWinSetLineWrap( pHeaderWin, FALSE);
	FTXWinOpen( pHeaderWin);

	if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, uiScreenCols,
		uiScreenRows - FIMT_TITLE_HEIGHT - FIMT_HEADER_HEIGHT - FIMT_LOG_HEIGHT,
		&pListWin)))
	{
		goto Exit;
	}
	FTXWinMove( pListWin, 0, FIMT_TITLE_HEIGHT + FIMT_HEADER_HEIGHT);
	FTXWinOpen( pListWin);
	pList->setup( pListWin);

	if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, uiScreenCols, FIMT_LOG_HEIGHT,
		&IxDispInfo.pLogWin)))
	{
		goto Exit;
	}

	FTXWinDrawBorder( IxDispInfo.pLogWin);
	FTXWinMove( IxDispInfo.pLogWin, 0, uiScreenRows - FIMT_LOG_HEIGHT);
	FTXWinSetBackFore( IxDispInfo.pLogWin, FLM_BLUE, FLM_WHITE);
	FTXWinClear( IxDispInfo.pLogWin);
	FTXWinSetCursorType( IxDispInfo.pLogWin, FLM_CURSOR_INVISIBLE);
	FTXWinSetScroll( IxDispInfo.pLogWin, TRUE);
	FTXWinSetLineWrap( IxDispInfo.pLogWin, FALSE);
	FTXWinOpen( IxDispInfo.pLogWin);

	FlmRegisterForEvent( F_EVENT_UPDATES, flstIndexUpdateEventHook,
		&IxDispInfo, &hEvent);

	FTXWinSetFocus( pListWin);
	uiIterations = 0;
	uiUpdateInterval = FLM_SECS_TO_TIMER_UNITS( 1);
	uiOneSec = FLM_SECS_TO_TIMER_UNITS( 1);
	uiLastUpdateTime = 0;
	while( !gv_bShutdown)
	{
		FLMUINT	uiCurrTime = FLM_GET_TIMER();

		if( bScreenLocked)
		{
			f_mutexUnlock( IxDispInfo.hScreenMutex);
			bScreenLocked = FALSE;
		}

		if( FLM_ELAPSED_TIME( uiCurrTime, uiLastUpdateTime) >= uiUpdateInterval)
		{
Update_Screen:

			if( !bScreenLocked)
			{
				f_mutexLock( IxDispInfo.hScreenMutex);
				bScreenLocked = TRUE;
			}

			FTXWinSetCursorPos( pHeaderWin, 0, 1);
			if( IxDispInfo.bShowTime)
			{
				FTXWinPrintf( pHeaderWin, "Index Index           State Last       Rate       Keys       Records    Time");
			}
			else
			{
				FTXWinPrintf( pHeaderWin, "Index Index           State Last       Rate       Keys       Records    Trans");
			}
			FTXWinClearToEOL( pHeaderWin);
			FTXWinPrintf( pHeaderWin, "\n");
			FTXWinPrintf( pHeaderWin, "Num.  Name                  DRN");

			FlmDbTransBegin( hDb, FLM_READ_TRANS, 0);

			pTmpNd = pList->getFirst();
			uiDrn = 0;
			for( ;;)
			{
				if( RC_BAD( FlmIndexGetNext( hDb, &uiDrn)))
				{
					break;
				}

				// Remove all invalid entries

				while( pTmpNd && pTmpNd->uiKey < uiDrn)
				{
					uiKey = pTmpNd->uiKey;
					pTmpNd = pTmpNd->pNext;
					pList->remove( uiKey);
				}

				FlmIndexStatus( hDb, uiDrn, &IxDispInfo.IndexStatus);

				if( !bShowOnline && 
					!IxDispInfo.IndexStatus.bSuspended &&
					IxDispInfo.IndexStatus.uiLastRecordIdIndexed == RECID_UNDEFINED)
				{
					if( pTmpNd && pTmpNd->uiKey == uiDrn)
					{
						uiKey = pTmpNd->uiKey;
						pTmpNd = pTmpNd->pNext;
						pList->remove( uiKey);
					}
					continue;
				}

				if( pTmpNd && pTmpNd->uiKey == uiDrn)
				{
					FLMUINT	uiOldest;
					FLMUINT	uiElapsed;

					pDispInfo = (IX_DISPLAY_INFO *)pTmpNd->pvData;
					f_strcpy( IxDispInfo.szName, pDispInfo->szName);

					// Copy the saved information.

					f_memcpy( &IxDispInfo.uiSaveRecsProcessed [0],
								&pDispInfo->uiSaveRecsProcessed [0],
								sizeof( FLMUINT) * MAX_VALS_TO_SAVE);
					f_memcpy( &IxDispInfo.uiRecSaveTime [0],
								&pDispInfo->uiRecSaveTime [0],
								sizeof( FLMUINT) * MAX_VALS_TO_SAVE);
					uiOldest = IxDispInfo.uiOldestSaved = pDispInfo->uiOldestSaved;

					// Recalculate the indexing rate.

					uiCurrTime = FLM_GET_TIMER();
					uiElapsed = (uiCurrTime - IxDispInfo.uiRecSaveTime [uiOldest]) /
															uiOneSec;
					if (uiElapsed && IxDispInfo.IndexStatus.uiRecordsProcessed)
					{
						if( IxDispInfo.uiSaveRecsProcessed[ uiOldest] < 
							IxDispInfo.IndexStatus.uiRecordsProcessed)
						{
							IxDispInfo.uiIndexingRate =
										// Records processed in time period
										(IxDispInfo.IndexStatus.uiRecordsProcessed -
										 IxDispInfo.uiSaveRecsProcessed [uiOldest]) / uiElapsed;
						}
						else
						{
							IxDispInfo.uiIndexingRate = 0;
						}
					}
					else
					{
						IxDispInfo.uiIndexingRate = 0;
					}

					// Overwrite the oldest with the current data.

					IxDispInfo.uiRecSaveTime [uiOldest] = uiCurrTime;
					IxDispInfo.uiSaveRecsProcessed [uiOldest] =
							IxDispInfo.IndexStatus.uiRecordsProcessed;

					// Move oldest pointer for next update.

					if (++IxDispInfo.uiOldestSaved == MAX_VALS_TO_SAVE)
					{
						IxDispInfo.uiOldestSaved = 0;
					}
				}
				else
				{
					FLMUINT	uiLoop;

					uiCurrTime = FLM_GET_TIMER();
					IxDispInfo.uiIndexingRate = 0;
					for (uiLoop = 0; uiLoop < MAX_VALS_TO_SAVE; uiLoop++)
					{
						IxDispInfo.uiSaveRecsProcessed [uiLoop] =
								IxDispInfo.IndexStatus.uiRecordsProcessed;
						IxDispInfo.uiRecSaveTime [uiLoop] = uiCurrTime;
					}
					IxDispInfo.uiOldestSaved = 0;

					// Only retrieve the index name if we don't already have it.

					if( RC_BAD( FlmRecordRetrieve( hDb, FLM_DICT_CONTAINER,
						uiDrn, FO_EXACT, &pRec, &uiDrn)))
					{
						break;
					}

					flmAssert( pRec->getFieldID( pRec->root()) == FLM_INDEX_TAG);
					uiBufLen = sizeof( szName);
					pRec->getNative( pRec->root(), szName, &uiBufLen);
					if (uiBufLen >= sizeof( IxDispInfo.szName) - 1)
					{
						uiBufLen = sizeof( IxDispInfo.szName) - 1;
						szName [uiBufLen] = 0;
					}
					f_memcpy( IxDispInfo.szName, szName, uiBufLen);
					IxDispInfo.szName [uiBufLen] = 0;
				}

				pList->update( uiDrn, ixDisplayHook, &IxDispInfo, sizeof( IxDispInfo));
				pList->refresh();

				if( pTmpNd && pTmpNd->uiKey == uiDrn)
				{
					pTmpNd = pTmpNd->pNext;
				}
			}
			FlmDbTransAbort( hDb);
			uiLastUpdateTime = FLM_GET_TIMER();
			pList->refresh();
		}

		if( !bScreenLocked)
		{
			f_mutexLock( IxDispInfo.hScreenMutex);
			bScreenLocked = TRUE;
		}

		if( RC_OK( FTXWinTestKB( pListWin)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pListWin, &uiChar);
			f_mutexUnlock( IxDispInfo.hScreenMutex);
			bScreenLocked = FALSE;

			switch( uiChar)
			{
				case 'O':
				case 'o':
				{
					bShowOnline = !bShowOnline;
					goto Update_Screen;
				}

				case '+':
				case 'r':
				{
					if( (pTmpNd = pList->getCurrent()) != NULL)
					{
						FlmIndexResume( hDb, pTmpNd->uiKey);
						goto Update_Screen;
					}
					break;
				}

				case 's':
				{
					if( (pTmpNd = pList->getCurrent()) != NULL)
					{
						FlmIndexSuspend( hDb, pTmpNd->uiKey);
						goto Update_Screen;
					}
					break;
				}

				case FKB_ALT_S:
				case 'S':
				{
					f_mutexLock( IxDispInfo.hScreenMutex);
					FTXMessageWindow( IxDispInfo.pScreen, FLM_RED, FLM_WHITE,
								"Suspending all indexes ....",
								NULL, &pMsgWin);

					f_mutexUnlock( IxDispInfo.hScreenMutex);

					if (RC_OK( FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, 15)))
					{
						uiDrn = 0;
						for( ;;)
						{
							if( RC_BAD( FlmIndexGetNext( hDb, &uiDrn)))
							{
								break;
							}
							FlmIndexSuspend( hDb, uiDrn);
						}
						FlmDbTransCommit( hDb);
					}

					if( pMsgWin)
					{
						f_mutexLock( IxDispInfo.hScreenMutex);
						FTXWinFree( &pMsgWin);
						f_mutexUnlock( IxDispInfo.hScreenMutex);
					}
					goto Update_Screen;
				}

				case 'R':
				case FKB_ALT_R:
				{
					f_mutexLock( IxDispInfo.hScreenMutex);
					FTXMessageWindow( IxDispInfo.pScreen, FLM_RED, FLM_WHITE,
						"Resuming all indexes                                ",
						NULL,
						&pMsgWin);
					f_mutexUnlock( IxDispInfo.hScreenMutex);

					if (RC_OK( FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, 15)))
					{
						uiDrn = 0;
						for( ;;)
						{
							if( RC_BAD( FlmIndexGetNext( hDb, &uiDrn)))
							{
								break;
							}

							FlmIndexResume( hDb, uiDrn);
						}
						FlmDbTransCommit( hDb);
					}
					if( pMsgWin)
					{
						f_mutexLock( IxDispInfo.hScreenMutex);
						FTXWinFree( &pMsgWin);
						f_mutexUnlock( IxDispInfo.hScreenMutex);
					}
					goto Update_Screen;
				}

				case 'T':
				case 't':
				{
					IxDispInfo.bShowTime = !IxDispInfo.bShowTime;
					goto Update_Screen;
				}

				case '?':
				{
					FTX_WINDOW *		pHelpWin = NULL;
					FTX_WINDOW *		pHelpTitle = NULL;
					F_DynamicList *	pHelpList = NULL;
					FLMUINT				uiItem = 0;
					char					szTmpBuf [100];

					f_mutexLock( IxDispInfo.hScreenMutex);
					bScreenLocked = TRUE;

					if( (pHelpList = f_new F_DynamicList) == NULL)
					{
						goto Help_Exit;
					}

					if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, uiScreenCols,
						1, &pHelpTitle)))
					{
						goto Help_Exit;
					}

					FTXWinSetBackFore( pHelpTitle, FLM_RED, FLM_WHITE);
					FTXWinClear( pHelpTitle);
					FTXWinSetCursorType( pHelpTitle, FLM_CURSOR_INVISIBLE);
					FTXWinSetScroll( pHelpTitle, FALSE);
					FTXWinSetLineWrap( pHelpTitle, FALSE);
					FTXWinPrintf( pHelpTitle, "FLAIM Index Manager - Help");
					FTXWinOpen( pHelpTitle);

					if( RC_BAD( FTXWinInit( IxDispInfo.pScreen, uiScreenCols,
						uiScreenRows - 1, &pHelpWin)))
					{
						goto Help_Exit;
					}
					FTXWinDrawBorder( pHelpWin);
					FTXWinOpen( pHelpWin);
					pHelpList->setup( pHelpWin);

					f_sprintf( szTmpBuf, "R, ALT_R  Resume all indexes");
					pHelpList->update( ++uiItem, NULL, szTmpBuf, sizeof( szTmpBuf));

					f_sprintf( szTmpBuf, "S, ALT_S  Suspend all indexes");
					pHelpList->update( ++uiItem, NULL, szTmpBuf, sizeof( szTmpBuf));

					f_sprintf( szTmpBuf, "o, O      Toggle display of on-line indexes");
					pHelpList->update( ++uiItem, NULL, szTmpBuf, sizeof( szTmpBuf));

					f_sprintf( szTmpBuf, "+, r      Resume selected index with auto on-line option");
					pHelpList->update( ++uiItem, NULL, szTmpBuf, sizeof( szTmpBuf));

					f_sprintf( szTmpBuf, "s         Suspend selected index");
					pHelpList->update( ++uiItem, NULL, szTmpBuf, sizeof( szTmpBuf));

					pHelpList->refresh();
					pHelpWin = pHelpList->getListWin();

					f_mutexUnlock( IxDispInfo.hScreenMutex);
					bScreenLocked = FALSE;

					while( !gv_bShutdown)
					{
						f_mutexLock( IxDispInfo.hScreenMutex);
						bScreenLocked = TRUE;

						if( RC_OK( FTXWinTestKB( pHelpWin)))
						{
							FLMUINT		uiTmpChar;
							FTXWinInputChar( pHelpWin, &uiTmpChar);
							if( uiTmpChar == FKB_ESCAPE)
							{
								break;
							}
							pHelpList->defaultKeyAction( uiTmpChar);
						}

						f_mutexUnlock( IxDispInfo.hScreenMutex);
						bScreenLocked = FALSE;
						f_sleep( 10);
					}

Help_Exit:
					if( !bScreenLocked)
					{
						f_mutexLock( IxDispInfo.hScreenMutex);
						bScreenLocked = TRUE;
					}

					if( pHelpList)
					{
						pHelpList->Release();
					}

					if( pHelpTitle)
					{
						FTXWinFree( &pHelpTitle);
					}

					f_mutexUnlock( IxDispInfo.hScreenMutex);
					bScreenLocked = FALSE;
					break;
				}

				case FKB_ESCAPE:
				{
					goto Exit;
				}

				default:
				{
					f_mutexLock( IxDispInfo.hScreenMutex);
					pList->defaultKeyAction( uiChar);
					f_mutexUnlock( IxDispInfo.hScreenMutex);
					break;
				}
			}
			f_mutexLock( IxDispInfo.hScreenMutex);
			pList->refresh();
			f_mutexUnlock( IxDispInfo.hScreenMutex);
		}

		uiIterations++;

		if( pThread->getShutdownFlag())
		{
			break;
		}

		f_sleep( 1);
	}

Exit:

	if( pList)
	{
		pList->Release();
	}

	if( hEvent != HFEVENT_NULL)
	{
		FlmDeregisterForEvent( &hEvent);
	}

	if( IxDispInfo.pScreen)
	{
		FTXScreenFree( &IxDispInfo.pScreen);
	}

	if( IxDispInfo.hScreenMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &IxDispInfo.hScreenMutex);
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( FERR_OK);
}

/****************************************************************************
Desc:	Thread that displays the current status of a database's cache
Note:	The caller must pass a valid share handle to the thread on startup.
*****************************************************************************/
RCODE FLMAPI flstMemoryManagerThread(
	IF_Thread *		pThread)
{
	F_DynamicList *	pList = f_new F_DynamicList;
	FTX_SCREEN *		pScreen;
	FTX_WINDOW *		pTitleWin;
	FTX_WINDOW *		pListWin;
	FTX_WINDOW *		pHeaderWin;
	char					szTmpBuf[ 80];
	FLMUINT				uiLoop;
	FLMUINT				uiIteration = 0;
	FLMUINT				uiScreenCols;
	FLMUINT				uiScreenRows;
	FLM_MEM_INFO		CacheInfo;
	CS_CONTEXT *		pCSContext = NULL;
	FCL_WIRE				Wire;
	NODE *				pTree;
	F_Pool				pool;

#define FMMT_TITLE_HEIGHT 1
#define FMMT_HEADER_HEIGHT 3

	if( RC_BAD( FTXScreenInit( "Memory Manager", &pScreen)))
	{
		goto Exit;
	}

	FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows);
	FTXScreenDisplay( pScreen);

	if( RC_BAD( FTXWinInit( pScreen, 0, FMMT_TITLE_HEIGHT, &pTitleWin)))
	{
		goto Exit;
	}

	FTXWinPaintBackground( pTitleWin, FLM_RED);
	FTXWinPrintStr( pTitleWin, "FLAIM Memory Manager");
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if( RC_BAD( FTXWinInit( pScreen, uiScreenCols, FMMT_HEADER_HEIGHT,
		&pHeaderWin)))
	{
		goto Exit;
	}

	FTXWinSetBackFore( pHeaderWin, FLM_BLUE, FLM_WHITE);
	FTXWinClear( pHeaderWin);
	FTXWinPrintf( pHeaderWin, "\n                                         Record          Block     Both");
	FTXWinSetCursorType( pHeaderWin, FLM_CURSOR_INVISIBLE);
	FTXWinSetScroll( pHeaderWin, FALSE);
	FTXWinSetLineWrap( pHeaderWin, FALSE);
	FTXWinMove( pHeaderWin, 0, FMMT_TITLE_HEIGHT);
	FTXWinOpen( pHeaderWin);

	if( RC_BAD( FTXWinInit( pScreen, uiScreenCols,
		uiScreenRows - FMMT_TITLE_HEIGHT - FMMT_HEADER_HEIGHT,
		&pListWin)))
	{
		goto Exit;
	}
	FTXWinMove( pListWin, 0, FMMT_TITLE_HEIGHT + FMMT_HEADER_HEIGHT);
	FTXWinOpen( pListWin);
	pList->setup( pListWin);

	pool.poolInit( 1024);
	uiLoop = 0;
	while( !gv_bShutdown)
	{
		if( !(uiIteration % 100))
		{
			FLMUINT				uiKey = 0;
			FLM_CACHE_USAGE *	pRecCacheUse = &CacheInfo.RecordCache;
			FLM_CACHE_USAGE *	pBlkCacheUse = &CacheInfo.BlockCache;

			if( pCSContext)
			{
				Wire.setContext( pCSContext);

				/* Send a request get statistics */

				if (RC_BAD( Wire.sendOp( FCS_OPCLASS_GLOBAL, 
					FCS_OP_GLOBAL_MEM_INFO_GET)))
				{
					goto Exit;
				}

				if (RC_BAD( Wire.sendTerminate()))
				{
					goto Transmission_Error;
				}

				/* Read the response. */

				if (RC_BAD( Wire.read()))
				{
					goto Transmission_Error;
				}

				if (RC_BAD( Wire.getRCode()))
				{
					goto Exit;
				}

				pool.poolReset( NULL);
				if( RC_BAD( Wire.getHTD( &pool, &pTree)))
				{
					goto Exit;
				}

				if( RC_BAD( fcsExtractMemInfo( pTree, &CacheInfo)))
				{
					goto Exit;
				}
			}
			else
			{
				FlmGetMemoryInfo( &CacheInfo);
			}

			f_sprintf( szTmpBuf,
				"  Max Bytes ........................ %10u     %10u",
				(unsigned)pRecCacheUse->uiMaxBytes,
				(unsigned)pBlkCacheUse->uiMaxBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Count ............................ %10u     %10u",
				(unsigned)pRecCacheUse->uiCount,
				(unsigned)pBlkCacheUse->uiCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Total Bytes Allocated ............ %10u     %10u",
				(unsigned)pRecCacheUse->uiTotalBytesAllocated,
				(unsigned)pBlkCacheUse->uiTotalBytesAllocated);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Hits ....................... %10u     %10u",
				(unsigned)pRecCacheUse->uiCacheHits,
				(unsigned)pBlkCacheUse->uiCacheHits);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Hit Looks .................. %10u     %10u",
				(unsigned)pRecCacheUse->uiCacheHitLooks,
				(unsigned)pBlkCacheUse->uiCacheHitLooks);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Faults ..................... %10u     %10u",
				(unsigned)pRecCacheUse->uiCacheFaults,
				(unsigned)pBlkCacheUse->uiCacheFaults);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Fault Looks ................ %10u     %10u",
				(unsigned)pRecCacheUse->uiCacheFaultLooks,
				(unsigned)pBlkCacheUse->uiCacheFaultLooks);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dirty Count ......................        N/A     %10u",
				(unsigned)CacheInfo.uiDirtyCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dirty Bytes ......................        N/A     %10u",
				(unsigned)CacheInfo.uiDirtyBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  New Count ........................        N/A     %10u",
				(unsigned)CacheInfo.uiNewCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  New Bytes ........................        N/A     %10u",
				(unsigned)CacheInfo.uiNewBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Log Count ........................        N/A     %10u",
				(unsigned)CacheInfo.uiLogCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Log Bytes ........................        N/A     %10u",
				(unsigned)CacheInfo.uiLogBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Old Version Count ................ %10u     %10u",
				(unsigned)pRecCacheUse->uiOldVerCount,
				(unsigned)pBlkCacheUse->uiOldVerCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Old Version Bytes ................ %10u     %10u",
				(unsigned)pRecCacheUse->uiOldVerBytes,
				(unsigned)pBlkCacheUse->uiOldVerBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Free Count .......................        N/A     %10u",
				(unsigned)CacheInfo.uiFreeCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Free Bytes .......................        N/A     %10u",
				(unsigned)CacheInfo.uiFreeBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Replaceable Count ................        N/A     %10u",
				(unsigned)CacheInfo.uiReplaceableCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Replaceable Bytes ................        N/A     %10u",
				(unsigned)CacheInfo.uiReplaceableBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dynamic Cache Adjust .............                               %s",
				(CacheInfo.bDynamicCacheAdjust ? "YES" : "NO"));
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Percentage ..........                               %u",
				(unsigned)CacheInfo.uiCacheAdjustPercent);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Min .................                               %u",
				(unsigned)CacheInfo.uiCacheAdjustMin);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Min To Leave ........                               %u",
				(unsigned)CacheInfo.uiCacheAdjustMinToLeave);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));
			pList->refresh();
		}
	
		if( RC_OK( FTXWinTestKB( pListWin)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pListWin, &uiChar);
			switch( uiChar)
			{
				case 'R':
				case 'r':
					FlmConfig( FLM_RESET_STATS, (void *)0, (void *)0);
					break;
				case FKB_UP:
					pList->cursorUp();
					break;
				case FKB_DOWN:
					pList->cursorDown();
					break;
				case FKB_PGUP:
					pList->pageUp();
					break;
				case FKB_PGDN:
					pList->pageDown();
					break;
				case FKB_HOME:
					pList->home();
					break;
				case FKB_END:
					pList->end();
					break;
				case FKB_ESCAPE:
					goto Exit;
			}
			pList->refresh();
		}

		if( pThread->getShutdownFlag())
		{
			break;
		}

		f_sleep( 10);
		uiIteration++;
	}

Exit:

	if( pList)
	{
		pList->Release();
	}

	if( pCSContext)
	{
		flmCloseCSConnection( &pCSContext);
	}

	pool.poolFree();

	if( pScreen)
	{
		FTXScreenFree( &pScreen);
	}

	return( FERR_OK);

Transmission_Error:

	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:	Thread that displays the current status of a database's tracker thread
*****************************************************************************/
RCODE FLMAPI flstTrackerMonitorThread(
	IF_Thread *		pThread)
{
	F_DynamicList *	pList = f_new F_DynamicList;
	FTX_SCREEN *		pScreen;
	FTX_WINDOW *		pTitleWin;
	FTX_WINDOW *		pListWin;
	FTX_WINDOW *		pHeaderWin;
	char					szTmpBuf[ 80];
	FLMUINT				uiScreenCols;
	FLMUINT				uiScreenRows;
	HFDB					hDb = (HFDB)pThread->getParm1();
	FMAINT_STATUS		maintStatus;

#define FTMT_TITLE_HEIGHT 1
#define FTMT_HEADER_HEIGHT 3

	if( RC_BAD( FTXScreenInit( "Tracker Monitor", &pScreen)))
	{
		goto Exit;
	}

	FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows);
	FTXScreenDisplay( pScreen);

	if( RC_BAD( FTXWinInit( pScreen, 0, FTMT_TITLE_HEIGHT, &pTitleWin)))
	{
		goto Exit;
	}

	FTXWinPaintBackground( pTitleWin, FLM_RED);
	FTXWinPrintStr( pTitleWin, "FLAIM Tracker Monitor");
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if( RC_BAD( FTXWinInit( pScreen, uiScreenCols, FTMT_HEADER_HEIGHT,
		&pHeaderWin)))
	{
		goto Exit;
	}

	FTXWinSetBackFore( pHeaderWin, FLM_BLUE, FLM_WHITE);
	FTXWinClear( pHeaderWin);
	FTXWinPrintf( pHeaderWin, "\nDescription");
	FTXWinSetCursorType( pHeaderWin, FLM_CURSOR_INVISIBLE);
	FTXWinSetScroll( pHeaderWin, FALSE);
	FTXWinSetLineWrap( pHeaderWin, FALSE);
	FTXWinMove( pHeaderWin, 0, FTMT_TITLE_HEIGHT);
	FTXWinOpen( pHeaderWin);

	if( RC_BAD( FTXWinInit( pScreen, uiScreenCols,
		uiScreenRows - FTMT_TITLE_HEIGHT - FTMT_HEADER_HEIGHT,
		&pListWin)))
	{
		goto Exit;
	}
	FTXWinMove( pListWin, 0, FTMT_TITLE_HEIGHT + FTMT_HEADER_HEIGHT);
	FTXWinOpen( pListWin);
	pList->setup( pListWin);

	while( !gv_bShutdown)
	{
		FLMUINT	uiKey = 0;

		FlmMaintenanceStatus( hDb, &maintStatus);

		switch( maintStatus.eDoing)
		{
			case FLM_MAINT_IDLE:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Idle");
				break;
			}

			case FLM_MAINT_LOOKING_FOR_WORK:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Looking for Work");
				break;
			}

			case FLM_MAINT_WAITING_FOR_LOCK:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Waiting for Lock");
				break;
			}

			case FLM_MAINT_ENDING_TRANS:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Ending Transaction");
				break;
			}

			case FLM_MAINT_TERMINATED:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Terminated");
				break;
			}

			case FLM_MAINT_FREEING_BLOCKS:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Freeing Blocks");
				break;
			}

			default:
			{
				f_sprintf( szTmpBuf,
					"  Status ........................... Unknown");
				break;
			}
		}

		pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

		f_sprintf( szTmpBuf,
			"  Blocks Freed ..................... %I64u",
			(FLMUINT64)maintStatus.ui64BlocksFreed);
		pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

		if( RC_OK( FTXWinTestKB( pListWin)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pListWin, &uiChar);
			switch( uiChar)
			{
				case FKB_UP:
					pList->cursorUp();
					break;
				case FKB_DOWN:
					pList->cursorDown();
					break;
				case FKB_PGUP:
					pList->pageUp();
					break;
				case FKB_PGDN:
					pList->pageDown();
					break;
				case FKB_HOME:
					pList->home();
					break;
				case FKB_END:
					pList->end();
					break;
				case FKB_ESCAPE:
					goto Exit;
			}

			pList->refresh();
		}

		if( pThread->getShutdownFlag())
		{
			break;
		}

		pList->refresh();
		f_sleep( 100);
	}

Exit:

	if( pList)
	{
		pList->Release();
	}

	if( pScreen)
	{
		FTXScreenFree( &pScreen);
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( FERR_OK);
}
