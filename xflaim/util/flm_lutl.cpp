//------------------------------------------------------------------------------
// Desc: Utility routines for presenting selection and statistics lists
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

#include "flaimsys.h"
#include "flm_lutl.h"
#include "flm_dlst.h"

extern FLMBOOL		gv_bShutdown;

FSTATIC RCODE ixDisplayHook(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList);

#define MAX_VALS_TO_SAVE	10

typedef struct IxDisplayInfo
{
	char					szName [256];
	XFLM_INDEX_STATUS	IndexStatus;
	FLMUINT64			ui64SaveDocsProcessed [MAX_VALS_TO_SAVE];
	FLMUINT				uiDocSaveTime [MAX_VALS_TO_SAVE];
	FLMUINT				uiOldestSaved;
	FLMUINT				uiIndexingRate;
	FLMBOOL				bShowTime;
	F_MUTEX				hScreenMutex;
	FTX_SCREEN *		pScreen;
	FTX_WINDOW *		pLogWin;
	F_Db *				pDb;
} IX_DISPLAY_INFO;

FSTATIC RCODE ixDisplayHook(
	FTX_WINDOW *		pWin,
	FLMBOOL				bSelected,
	FLMUINT				uiRow,
	FLMUINT				uiKey,
	void *				pvData,
	FLMUINT				uiDataLen,
	F_DynamicList*		pDynamicList)
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

	if( pDispInfo->IndexStatus.eState == XFLM_INDEX_SUSPENDED)
	{
		pszState = "Susp";
	}
	else if( pDispInfo->IndexStatus.eState == XFLM_INDEX_ONLINE)
	{
		pszState = "Onln";
	}
	else
	{
		pszState = "Offln";
	}

	pDispInfo->szName [15] = 0;
	if( pDispInfo->IndexStatus.ui64LastDocumentIndexed != (FLMUINT64)~0)
	{
		f_sprintf( szTmpBuf, "%5u %-15s %-5s %-10u %-10u %-10u %-10u %-10u",
			(unsigned)uiKey, pDispInfo->szName, pszState,
			(unsigned)pDispInfo->IndexStatus.ui64LastDocumentIndexed,
			(unsigned)pDispInfo->uiIndexingRate,
			(unsigned)pDispInfo->IndexStatus.ui64KeysProcessed,
			(unsigned)pDispInfo->IndexStatus.ui64DocumentsProcessed,
			(unsigned)(pDispInfo->bShowTime
								? pDispInfo->IndexStatus.ui32StartTime
										? uiGMT - pDispInfo->IndexStatus.ui32StartTime
										: 0
								: pDispInfo->IndexStatus.ui64Transactions));

	}
	else
	{
		f_sprintf( szTmpBuf,
			"%5u %-15s %-5s                                             %-10u",
			(unsigned)uiKey, pDispInfo->szName, pszState,
			(unsigned)(pDispInfo->bShowTime
								? pDispInfo->IndexStatus.ui32StartTime
										? uiGMT - pDispInfo->IndexStatus.ui32StartTime
										: 0
								: pDispInfo->IndexStatus.ui64Transactions));
	}

	FTXWinSetCursorPos( pWin, 0, uiRow);
	FTXWinClearToEOL( pWin);
	FTXWinPrintf( pWin, "%s", szTmpBuf);
	if( bSelected && pDynamicList->getShowHorizontalSelector())
	{
		FTXWinPaintRow( pWin, &uiBack, &uiFore, uiRow);
	}
	return( NE_XFLM_OK);
}

/***************************************************************************
Desc:	Event catching object.
***************************************************************************/
class IX_Event : public IF_EventClient
{
public:

	IX_Event()
	{
		m_pDispInfo = NULL;
	}

	virtual ~IX_Event()
	{
	}

	void XFLAPI catchEvent(
		eEventType	eEvent,
		IF_Db *		pDb,
		FLMUINT		uiThreadId,
		FLMUINT64	ui64TransID,
		FLMUINT		uiIndexOrCollection,
		FLMUINT64	ui64NodeId,
		RCODE			rc);

	FINLINE void setDispInfo(
		IX_DISPLAY_INFO *	pDispInfo
		)
	{
		m_pDispInfo = pDispInfo;
	}

private:

	IX_DISPLAY_INFO *	m_pDispInfo;
};

/****************************************************************************
Desc:	Event callback
*****************************************************************************/
void  IX_Event::catchEvent(
	eEventType	eEvent,
	IF_Db *		pDb,
	FLMUINT,		// uiThreadId,
	FLMUINT64,	// ui64TransID,
	FLMUINT		uiIndexOrCollection,
	FLMUINT64	ui64NodeId,
	RCODE			// rc
	)
{
	XFLM_INDEX_STATUS	ixStatus;
	FLMUINT				uiGMT;

	if (eEvent == XFLM_EVENT_INDEXING_PROGRESS && !ui64NodeId && pDb)
	{
		if (RC_OK( ((F_Db *)pDb)->indexStatus( uiIndexOrCollection, &ixStatus)))
		{
			f_timeGetSeconds( &uiGMT );

			f_mutexLock( m_pDispInfo->hScreenMutex);
			FTXWinPrintf( m_pDispInfo->pLogWin,
				"Index %u came on-line.  Elapsed time = %u second(s)\n",
				uiIndexOrCollection, uiGMT - (FLMUINT)ixStatus.ui32StartTime);
			f_mutexUnlock( m_pDispInfo->hScreenMutex);
		}
	}
}

/****************************************************************************
Name:	flstIndexManagerThread
Desc:	Thread that displays the current status of all indexes in a database
Note:	The caller must open the database and pass a handle to the thread.
		The handle will be closed when the thread exits.
*****************************************************************************/
RCODE FTKAPI flstIndexManagerThread(
	IF_Thread *		pThread)
{
	RCODE						rc = NE_XFLM_OK;
	F_DynamicList *		pList = f_new F_DynamicList;
	FTX_WINDOW *			pTitleWin;
	FTX_WINDOW *			pListWin;
	FTX_WINDOW *			pHeaderWin;
	FTX_WINDOW *			pMsgWin;
	FLMUINT					uiIterations = 0;
	FLMUINT					uiScreenCols;
	FLMUINT					uiScreenRows;
	FLMUINT					uiIndex;
	FLMUINT					uiUpdateInterval;
	FLMUINT					uiLastUpdateTime;
	IX_DISPLAY_INFO		IxDispInfo;
	IX_DISPLAY_INFO *		pDispInfo;
	DLIST_NODE *			pTmpNd;
	FLMUINT					uiKey;
	FLMBOOL					bShowOnline = TRUE;
	F_Db *					pDb = (F_Db *)pThread->getParm1();
	FLMUINT					uiOneSec;
	FLMBOOL					bScreenLocked = FALSE;
	IX_Event					event;
	FLMBOOL					bRegisteredForEvent = FALSE;
	IF_DbSystem *			pDbSystem = NULL;

	event.setDispInfo( &IxDispInfo);

#define FIMT_TITLE_HEIGHT		1
#define FIMT_HEADER_HEIGHT		4
#define FIMT_LOG_HEIGHT			10

	f_memset( &IxDispInfo, 0, sizeof( IX_DISPLAY_INFO));
	IxDispInfo.hScreenMutex = F_MUTEX_NULL;
	IxDispInfo.pDb = (F_Db *)pDb;
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
		uiScreenCols, FIMT_HEADER_HEIGHT,
		&pHeaderWin)))
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
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDbSystem->registerForEvent(
		XFLM_EVENT_UPDATES, &event)))
	{
		goto Exit;
	}
	bRegisteredForEvent = TRUE;

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
				FTXWinPrintf( pHeaderWin, "Index Index           State Last       Rate       Keys       Documents  Time");
			}
			else
			{
				FTXWinPrintf( pHeaderWin, "Index Index           State Last       Rate       Keys       Documents  Trans");
			}
			FTXWinClearToEOL( pHeaderWin);
			FTXWinPrintf( pHeaderWin, "\n");
			FTXWinPrintf( pHeaderWin, "Num.  Name                  DOC");

			if (RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS)))
			{
				goto Exit;
			}

			pTmpNd = pList->getFirst();
			uiIndex = 0;
			for( ;;)
			{
				if( RC_BAD( pDb->indexGetNext( &uiIndex)))
				{
					break;
				}

				// Remove all invalid entries

				while( pTmpNd && pTmpNd->uiKey < uiIndex)
				{
					uiKey = pTmpNd->uiKey;
					pTmpNd = pTmpNd->pNext;
					pList->remove( uiKey);
				}

				if (RC_BAD( rc = pDb->indexStatus( uiIndex, &IxDispInfo.IndexStatus)))
				{
					goto Exit;
				}

				if( !bShowOnline &&
					IxDispInfo.IndexStatus.eState == XFLM_INDEX_ONLINE)
				{
					if( pTmpNd && pTmpNd->uiKey == uiIndex)
					{
						uiKey = pTmpNd->uiKey;
						pTmpNd = pTmpNd->pNext;
						pList->remove( uiKey);
					}
					continue;
				}

				if( pTmpNd && pTmpNd->uiKey == uiIndex)
				{
					FLMUINT	uiOldest;
					FLMUINT	uiElapsed;

					pDispInfo = (IX_DISPLAY_INFO *)pTmpNd->pvData;
					f_strcpy( IxDispInfo.szName, pDispInfo->szName);

					// Copy the saved information.

					f_memcpy( &IxDispInfo.ui64SaveDocsProcessed [0],
								&pDispInfo->ui64SaveDocsProcessed [0],
								sizeof( FLMUINT) * MAX_VALS_TO_SAVE);
					f_memcpy( &IxDispInfo.uiDocSaveTime [0],
								&pDispInfo->uiDocSaveTime [0],
								sizeof( FLMUINT) * MAX_VALS_TO_SAVE);
					uiOldest = IxDispInfo.uiOldestSaved = pDispInfo->uiOldestSaved;

					// Recalculate the indexing rate.

					uiCurrTime = FLM_GET_TIMER();
					uiElapsed = (uiCurrTime - IxDispInfo.uiDocSaveTime [uiOldest]) /
															uiOneSec;
					if (uiElapsed && IxDispInfo.IndexStatus.ui64DocumentsProcessed)
					{
						if( IxDispInfo.ui64SaveDocsProcessed[ uiOldest] <
							IxDispInfo.IndexStatus.ui64DocumentsProcessed)
						{
							IxDispInfo.uiIndexingRate =
										// Records processed in time period
										(FLMUINT)((IxDispInfo.IndexStatus.ui64DocumentsProcessed -
										 IxDispInfo.ui64SaveDocsProcessed [uiOldest]) / uiElapsed);
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

					IxDispInfo.uiDocSaveTime [uiOldest] = uiCurrTime;
					IxDispInfo.ui64SaveDocsProcessed [uiOldest] =
							IxDispInfo.IndexStatus.ui64DocumentsProcessed;

					// Move oldest pointer for next update.

					if (++IxDispInfo.uiOldestSaved == MAX_VALS_TO_SAVE)
					{
						IxDispInfo.uiOldestSaved = 0;
					}
				}
				else
				{
					FLMUINT			uiLoop;
					FLMUINT			uiBufLen;
					F_DataVector	srchKey;

					uiCurrTime = FLM_GET_TIMER();
					IxDispInfo.uiIndexingRate = 0;
					for (uiLoop = 0; uiLoop < MAX_VALS_TO_SAVE; uiLoop++)
					{
						IxDispInfo.ui64SaveDocsProcessed [uiLoop] =
								IxDispInfo.IndexStatus.ui64DocumentsProcessed;
						IxDispInfo.uiDocSaveTime [uiLoop] = uiCurrTime;
					}
					IxDispInfo.uiOldestSaved = 0;

					// Retrieve index name

					if (RC_BAD( srchKey.setUINT( 0, ELM_INDEX_TAG)))
					{
						break;
					}
					if (RC_BAD( srchKey.setUINT( 1, uiIndex)))
					{
						break;
					}

					if (RC_BAD( rc = pDb->keyRetrieve( XFLM_DICT_NUMBER_INDEX,
											&srchKey, XFLM_EXACT, &srchKey)))
					{
						if (rc != NE_XFLM_NOT_FOUND)
						{
							break;
						}
					}
					else
					{
						F_DOMNode *	pNode = NULL;

						if (RC_BAD( rc = pDb->getNode( XFLM_DICT_COLLECTION,
													srchKey.getDocumentID(), &pNode)))
						{
							if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
							{
								break;
							}
						}
						else
						{
							uiBufLen = sizeof( IxDispInfo.szName);
							rc = pNode->getAttributeValueUTF8( pDb,
									ATTR_NAME_TAG, (FLMBYTE *)IxDispInfo.szName, uiBufLen);
							pNode->Release();
							if (rc != NE_XFLM_OK &&
								 rc != NE_XFLM_DOM_NODE_NOT_FOUND &&
								 rc != NE_XFLM_CONV_DEST_OVERFLOW)
							{
								break;
							}
						}
					}
				}

				pList->update( uiIndex, ixDisplayHook, &IxDispInfo, sizeof( IxDispInfo));
				pList->refresh();

				if( pTmpNd && pTmpNd->uiKey == uiIndex)
				{
					pTmpNd = pTmpNd->pNext;
				}
			}
			pDb->transAbort();
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
						if (RC_BAD( rc = pDb->indexResume( pTmpNd->uiKey)))
						{
							goto Exit;
						}
						goto Update_Screen;
					}
					break;
				}

				case 's':
				{
					if( (pTmpNd = pList->getCurrent()) != NULL)
					{
						if (RC_BAD( rc = pDb->indexSuspend( pTmpNd->uiKey)))
						{
							goto Exit;
						}
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

					if (RC_OK( pDb->transBegin( XFLM_UPDATE_TRANS)))
					{
						uiIndex = 0;
						for( ;;)
						{
							if( RC_BAD( pDb->indexGetNext( &uiIndex)))
							{
								break;
							}
							if (RC_BAD( pDb->indexSuspend( uiIndex)))
							{
								break;
							}
						}
						if (RC_BAD( pDb->transCommit()))
						{
							(void)pDb->transAbort();
						}
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

					if (RC_OK( pDb->transBegin( XFLM_UPDATE_TRANS)))
					{
						uiIndex = 0;
						for( ;;)
						{
							if( RC_BAD( pDb->indexGetNext( &uiIndex)))
							{
								break;
							}

							if (RC_BAD( pDb->indexResume( uiIndex)))
							{
								break;
							}
						}
						if (RC_BAD( pDb->transCommit()))
						{
							(void)pDb->transAbort();
							break;
						}
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

	if (bRegisteredForEvent)
	{
		pDbSystem->deregisterForEvent( XFLM_EVENT_UPDATES, &event);
	}

	if( IxDispInfo.pScreen)
	{
		FTXScreenFree( &IxDispInfo.pScreen);
	}

	if( IxDispInfo.hScreenMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &IxDispInfo.hScreenMutex);
	}
	
	if( pDb != NULL)
	{
		pDb->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( rc);
}


/****************************************************************************
Name:	flstMemoryManagerThread
Desc:	Thread that displays the current status of a database's cache
Note:	The caller must pass a valid share handle to the thread on startup.
*****************************************************************************/
RCODE FTKAPI flstMemoryManagerThread(
	IF_Thread *		pThread)
{
	RCODE					rc = NE_XFLM_OK;
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
	XFLM_CACHE_INFO	CacheInfo;
	IF_DbSystem *		pDbSystem = NULL;

#define FMMT_TITLE_HEIGHT 1
#define FMMT_HEADER_HEIGHT 3

	if( RC_BAD( FTXScreenInit( "XFlaim Memory Manager", &pScreen)))
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
	FTXWinPrintStr( pTitleWin, "XFlaim Memory Manager");
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if( RC_BAD( FTXWinInit( pScreen, uiScreenCols, FMMT_HEADER_HEIGHT,
		&pHeaderWin)))
	{
		goto Exit;
	}

	FTXWinSetBackFore( pHeaderWin, FLM_BLUE, FLM_WHITE);
	FTXWinClear( pHeaderWin);
	FTXWinPrintf( pHeaderWin,
	"\n                                    Block Cache     Node Cache");
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
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	uiLoop = 0;
	while( !gv_bShutdown)
	{
		if( !(uiIteration % 100))
		{
			FLMUINT					uiKey = 0;
			XFLM_CACHE_USAGE *	pBlkCacheUse = &CacheInfo.BlockCache;
			XFLM_CACHE_USAGE *	pNodeCacheUse = &CacheInfo.NodeCache;

			pDbSystem->getCacheInfo( &CacheInfo);
			f_sprintf( szTmpBuf,
				"  Maximum Cache Bytes............... %10u",
				(unsigned)CacheInfo.uiMaxBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Total Bytes Allocated ............ %10u",
				(unsigned)CacheInfo.uiTotalBytesAllocated);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Total Bytes....................... %10u     %10u",
				(unsigned)pBlkCacheUse->uiByteCount,
				(unsigned)pNodeCacheUse->uiByteCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Count ............................ %10u     %10u",
				(unsigned)pBlkCacheUse->uiCount,
				(unsigned)pNodeCacheUse->uiCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Hits ....................... %10u     %10u",
				(unsigned)pBlkCacheUse->uiCacheHits,
				(unsigned)pNodeCacheUse->uiCacheHits);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Hit Looks .................. %10u     %10u",
				(unsigned)pBlkCacheUse->uiCacheHitLooks,
				(unsigned)pNodeCacheUse->uiCacheHitLooks);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Faults ..................... %10u     %10u",
				(unsigned)pBlkCacheUse->uiCacheFaults,
				(unsigned)pNodeCacheUse->uiCacheFaults);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Fault Looks ................ %10u     %10u",
				(unsigned)pBlkCacheUse->uiCacheFaultLooks,
				(unsigned)pNodeCacheUse->uiCacheFaultLooks);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dirty Count ...................... %10u",
				(unsigned)CacheInfo.uiDirtyCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dirty Bytes ...................... %10u",
				(unsigned)CacheInfo.uiDirtyBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  New Count ........................ %10u",
				(unsigned)CacheInfo.uiNewCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  New Bytes ........................ %10u",
				(unsigned)CacheInfo.uiNewBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Log Count ........................ %10u",
				(unsigned)CacheInfo.uiLogCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Log Bytes ........................ %10u",
				(unsigned)CacheInfo.uiLogBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Old Version Count ................ %10u     %10u",
				(unsigned)pBlkCacheUse->uiOldVerCount,
				(unsigned)pNodeCacheUse->uiOldVerCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Old Version Bytes ................ %10u     %10u",
				(unsigned)pBlkCacheUse->uiOldVerBytes,
				(unsigned)pNodeCacheUse->uiOldVerBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Free Count ....................... %10u",
				(unsigned)CacheInfo.uiFreeCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Free Bytes ....................... %10u",
				(unsigned)CacheInfo.uiFreeBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Replaceable Count ................ %10u",
				(unsigned)CacheInfo.uiReplaceableCount);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Replaceable Bytes ................ %10u",
				(unsigned)CacheInfo.uiReplaceableBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Dynamic Cache Adjust .............        %s",
				(char *)(CacheInfo.bDynamicCacheAdjust ? "YES" : "NO"));
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Percentage .......... %10u",
				(unsigned)CacheInfo.uiCacheAdjustPercent);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Min ................. %10u",
				(unsigned)CacheInfo.uiCacheAdjustMin);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Cache Adjust Min To Leave ........ %10u",
				(unsigned)CacheInfo.uiCacheAdjustMinToLeave);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Slabs ............................ %10u     %10u",
				(unsigned)pBlkCacheUse->slabUsage.ui64Slabs,
				(unsigned)pNodeCacheUse->slabUsage.ui64Slabs);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Slab Bytes ....................... %10u     %10u",
				(unsigned)pBlkCacheUse->slabUsage.ui64SlabBytes,
				(unsigned)pNodeCacheUse->slabUsage.ui64SlabBytes);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Allocated Cells .................. %10u     %10u",
				(unsigned)pBlkCacheUse->slabUsage.ui64AllocatedCells,
				(unsigned)pNodeCacheUse->slabUsage.ui64AllocatedCells);
			pList->update( uiKey++, NULL, szTmpBuf, sizeof( szTmpBuf));

			f_sprintf( szTmpBuf,
				"  Free Cells ....................... %10u     %10u",
				(unsigned)pBlkCacheUse->slabUsage.ui64FreeCells,
				(unsigned)pNodeCacheUse->slabUsage.ui64FreeCells);
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
					pDbSystem->resetStats();
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

	if( pScreen)
	{
		FTXScreenFree( &pScreen);
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( rc);
}
