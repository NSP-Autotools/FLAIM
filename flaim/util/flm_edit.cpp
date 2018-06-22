//-------------------------------------------------------------------------
// Desc:	GEDCOM editor methods.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

#include "flm_dlst.h"
#include "flm_lutl.h"
#include "flm_edit.h"

RCODE f_KeyEditorKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE f_RecEditorFileKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE f_RecEditorFileEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

typedef struct AvgTimeTag
{
	FLMUINT	uiHours;
	FLMUINT	uiMinutes;
	FLMUINT	uiSeconds;
	FLMUINT	uiMilliSeconds;
	FLMUINT	uiAvgTime;
	FLMUINT	uiAvgBytesPerSec;
} AVG_TIME;

/****************************************************************************
Desc:
*****************************************************************************/
F_RecEditor::F_RecEditor( void)
{
	m_scratchPool.poolInit( 512);
	m_treePool.poolInit( 4096);
	m_pEditWindow = NULL;
	m_pEditStatusWin = NULL;
	m_uiEditCanvasRows = 0;
	m_bSetupCalled = FALSE;
	m_pNameTable = NULL;
	m_pucTmpBuf = NULL;
	m_pNameList = NULL;
	m_pLinkHook = (F_RECEDIT_LINK_HOOK)f_RecEditorDefaultLinkHook;
	m_LinkData = 0;
	m_pDisplayHook = (F_RECEDIT_DISP_HOOK)f_RecEditorDefaultDispHook;
	m_DisplayData = 0;
	m_pKeyHook = NULL;
	m_pFileSystem = NULL;
	m_KeyData = 0;
	m_bOwnNameTable = TRUE;
	m_bMonochrome = FALSE;
	m_hDefaultDb = HFDB_NULL;
	reset();
}


/****************************************************************************
Desc:	Destructor
*****************************************************************************/
F_RecEditor::~F_RecEditor( void)
{
	reset();
	if( m_pucTmpBuf)
	{
		f_free( &m_pucTmpBuf);
	}

	if( m_pFileSystem)
	{
		m_pFileSystem->Release();
	}

	m_scratchPool.poolFree();
	m_treePool.poolFree();
}


/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::Setup(
	FTX_SCREEN *		pScreen)
{
	RCODE		rc = FERR_OK;

	flmAssert( pScreen != NULL);

 	if( !m_pucTmpBuf)
	{
		if( RC_BAD( rc = f_alloc( F_RECEDIT_BUF_SIZE, &m_pucTmpBuf)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = FlmGetFileSystem( &m_pFileSystem)))
	{
		goto Exit;
	}

	m_pScreen = pScreen;
	m_bSetupCalled = TRUE;

Exit:

	if( RC_BAD( rc))
	{
		if( m_pucTmpBuf)
		{
			f_free( &m_pucTmpBuf);
			m_pucTmpBuf = NULL;
		}
	}

	return( rc);
}


/****************************************************************************
Desc:	
*****************************************************************************/
void F_RecEditor::reset( void)
{
	m_pTree = NULL;
	m_pCurNd = NULL;
	m_pScrFirstNd = NULL;
	m_hDefaultDb = HFDB_NULL;
	m_uiDefaultCont = FLM_DATA_CONTAINER;
	m_pucTitle[ 0] = '\0';
	m_bReadOnly = FALSE;
	m_pbShutdown = NULL;
	m_pParent = NULL;
	m_uiCurRow = 0;
	m_uiEditCanvasRows = 0;
	m_pHelpHook = NULL;
	m_pEventHook = NULL;
	m_uiLastKey = 0;
	m_pucAdHocQuery[ 0] = 0;
	m_uiULX = 0;
	m_uiULY = 0;
	m_uiLRX = 0;
	m_uiLRY = 0;

	if( m_pNameTable && m_bOwnNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	if( m_pNameList)
	{
		m_pNameList->Release();
		m_pNameList = NULL;
	}

	m_scratchPool.poolReset();
	m_treePool.poolReset();
}


/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::setTree(
	NODE *		pTree,
	NODE **		ppNewNd)
{
	DBE_REC_INFO	recInfo;
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	m_treePool.poolReset();
	if( pTree != NULL)
	{
		if( RC_BAD( rc = copyBuffer( &m_treePool, pTree, &m_pTree)))
		{
			goto Exit;
		}
	}
	else
	{
		m_pTree = NULL;
	}

	m_pCurNd = m_pTree;
	m_pScrFirstNd = m_pTree;
	m_uiCurRow = 0;

	if( ppNewNd)
	{
		*ppNewNd = m_pTree;
	}

	// VISIT: Need to loop through each record in the tree

	if( m_pEventHook)
	{
		f_memset( &recInfo, 0, sizeof( DBE_REC_INFO));
		recInfo.uiContainer = 0;
		recInfo.uiDrn = 0;
		recInfo.pRec = m_pTree;

		if( RC_BAD( rc = m_pEventHook( this, F_RECEDIT_EVENT_RECINSERT,
			(void *)(&recInfo), m_EventData)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::appendTree(
	NODE *		pTree,
	NODE **		ppNewRoot)
{
	NODE *	pNewTree;
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	// VISIT: May want this to call the event hook

	if( ppNewRoot)
	{
		*ppNewRoot = NULL;
	}

	if( !m_pTree)
	{
		if( RC_BAD( rc = setTree( pTree)))
		{
			goto Exit;
		}
		pNewTree = m_pTree;
	}
	else
	{
		if( (pNewTree = GedCopy( &m_treePool, GED_FOREST, pTree)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		GedSibGraft( m_pTree, pNewTree, GED_LAST);
	}

	if( ppNewRoot)
	{
		*ppNewRoot = pNewTree;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::insertRecord(
	NODE *		pRecord,
	NODE **		ppNewRoot,
	NODE *		pStartNd)
{
	NODE *		pNewTree;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( ppNewRoot)
	{
		*ppNewRoot = NULL;
	}

	if( (pNewTree = GedCopy( &m_treePool, GED_TREE, pRecord)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = _insertRecord( pNewTree, pStartNd)))
	{
		goto Exit;
	}

	if( ppNewRoot)
	{
		*ppNewRoot = pNewTree;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	pRecord must be allocated in the treePool
*****************************************************************************/
RCODE F_RecEditor::_insertRecord(
	NODE *		pRecord,
	NODE *		pStartNd)
{
	NODE *			pTmpNd;
	NODE *			pPriorRec;
	FLMUINT			uiInsCont;
	FLMUINT			uiInsDrn;
	FLMUINT			uiCont;
	FLMUINT			uiDrn;
	DBE_REC_INFO	recInfo;
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( GedGetRecSource( pRecord, NULL, &uiInsCont, &uiInsDrn)))
	{
		uiInsCont = 0;
		uiInsDrn = 0;
	}

	if( m_pEventHook)
	{
		f_memset( &recInfo, 0, sizeof( DBE_REC_INFO));
		recInfo.uiContainer = uiInsCont;
		recInfo.uiDrn = uiInsDrn;
		recInfo.pRec = pRecord;

		if( RC_BAD( rc = m_pEventHook( this, F_RECEDIT_EVENT_RECINSERT,
			(void *)(&recInfo), m_EventData)))
		{
			goto Exit;
		}
	}

	if( !m_pTree)
	{
		m_pTree = pRecord;
		m_pCurNd = m_pTree;
	}
	else
	{
		if( pStartNd)
		{
			pTmpNd = pStartNd;
		}
		else
		{
			pTmpNd = m_pTree;
		}

		pPriorRec = NULL;
		while( pTmpNd)
		{
			uiCont = 0;
			uiDrn = 0;
			if( RC_OK( rc = GedGetRecSource( pTmpNd, NULL,
				&uiCont, &uiDrn)) || rc == FERR_NOT_FOUND)
			{
				rc = FERR_OK;
				if( uiCont > uiInsCont ||
					(uiCont == uiInsCont && uiDrn >= uiInsDrn))
				{
					break;
				}
			}
			else
			{
				goto Exit;
			}

			pPriorRec = pTmpNd;
			pTmpNd = getNextRecord( pTmpNd);
		}

		if( pPriorRec)
		{
			GedSibGraft( pPriorRec, pRecord, 0);
		}
		else
		{
			GedSibGraft( pRecord, m_pTree, GED_LAST);
			m_pTree = pRecord;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::setTitle( 
	const char * 	pucTitle)
{
	RCODE				rc = FERR_OK;
	eColorType		back;
	eColorType		fore;

	flmAssert( m_bSetupCalled == TRUE);

	m_pucTitle[ 0] = ' ';
	f_strncpy( &m_pucTitle[ 1], pucTitle, F_RECEDIT_MAX_TITLE_SIZE - 2);
	f_strcat( m_pucTitle, " ");
	m_pucTitle[ F_RECEDIT_MAX_TITLE_SIZE] = '\0';

	if (m_pEditWindow)
	{
		back = m_bMonochrome ? FLM_BLACK : FLM_BLUE;
		fore = FLM_WHITE;
		
		FTXWinSetTitle( m_pEditWindow, m_pucTitle, back, fore);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::setDefaultSource(
	HFDB			hDb,
	FLMUINT		uiContainer)
{
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	m_hDefaultDb = hDb;
	m_uiDefaultCont = uiContainer;

	if( RC_BAD( rc = refreshNameTable()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::interactiveEdit(
	FLMUINT			uiULX,
	FLMUINT			uiULY,
	FLMUINT			uiLRX,
	FLMUINT			uiLRY,
	FLMBOOL			bBorder,
	FLMBOOL			bStatus)
{
	NODE *			pTmpNd;
	NODE *			pCopyNd = NULL;
	FLMBOOL			bRefreshEditWindow = FALSE;
	FLMBOOL			bRefreshStatusWindow = FALSE;
	FLMUINT			uiNumRows = 0;
	FLMUINT			uiNumCols = 0;
	FLMUINT			uiMaxRow = 0;
	FLMUINT			uiStartCol = 0;
	FLMUINT			uiTransType = FLM_NO_TRANS;
	FLMUINT			uiLoop;
	char				pucSearchBuf[ 256];
	FLMUINT			uiCurFlags;
	char				pucAction[ 2];
	FLMUINT			uiTermChar;
	FLMUINT			uiHelpKey = 0;
	FLMBOOL			bDoneEditing = FALSE;
	FLMBOOL			bStartedTrans = FALSE;
	F_Pool			copyPool;
	RCODE				rc = FERR_OK;
	RCODE				tmpRc;
	eColorType		fore;
	eColorType		back;
	IF_Thread *		pIxManagerThrd = NULL;
	IF_Thread *		pMemManagerThrd = NULL;
	IF_Thread *		pTrackerMonitorThrd = NULL;
	char				szDbPath [F_PATH_MAX_SIZE];
	HFDB				hTmpDb = HFDB_NULL;

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_pScreen != NULL);

	copyPool.poolInit( 512);

	m_uiCurRow = 0;
	m_pScrFirstNd = NULL;
	m_uiLastKey = 0;

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !uiLRX && !uiLRY)
	{
		if( RC_BAD( rc = FTXScreenGetSize( m_pScreen,
			&uiNumCols, &uiNumRows)))
		{
			goto Exit;
		}

		uiNumRows -= uiULY;
		uiNumCols -= uiULX;

	}
	else
	{
		uiNumRows = (uiLRY - uiULY) + 1;
		uiNumCols = (uiLRX - uiULX) + 1;
	}

	uiStartCol = uiULX;

	if( bStatus)
	{
		uiNumRows -= 1; // Add 1 to account for the status bar
	}

	m_uiULX = uiULX;
	m_uiULY = uiULY;
	m_uiLRX = uiLRX;
	m_uiLRY = uiLRY;

	if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumCols,
		uiNumRows, &m_pEditWindow)))
	{
		goto Exit;
	}

	FTXWinMove( m_pEditWindow, uiStartCol, uiULY);
	FTXWinSetScroll( m_pEditWindow, FALSE);
	FTXWinSetLineWrap( m_pEditWindow, FALSE);
	FTXWinSetCursorType( m_pEditWindow, FLM_CURSOR_INVISIBLE);

	back = m_bMonochrome ? FLM_BLACK : FLM_BLUE;
	fore = FLM_WHITE;
	FTXWinSetBackFore( m_pEditWindow, back, fore);
	FTXWinClear( m_pEditWindow);

	if( bBorder)
	{
		FTXWinDrawBorder( m_pEditWindow);
	}

	FTXWinSetTitle( m_pEditWindow, m_pucTitle, back, fore);

	if( bStatus)
	{
		if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumCols, 1,
			&m_pEditStatusWin)))
		{
			goto Exit;
		}

		FTXWinMove( m_pEditStatusWin, uiULX, uiULY + uiNumRows);
		FTXWinSetScroll( m_pEditStatusWin, FALSE);
		FTXWinSetCursorType( m_pEditStatusWin, FLM_CURSOR_INVISIBLE);
		FTXWinSetBackFore( m_pEditStatusWin,
			m_bMonochrome ? FLM_BLACK : FLM_GREEN, FLM_WHITE);
		FTXWinClear( m_pEditStatusWin);
		FTXWinOpen( m_pEditStatusWin);
	}
	
	FTXWinOpen( m_pEditWindow);
	FTXWinGetCanvasSize( m_pEditWindow, &uiNumCols, &uiNumRows);

	m_uiEditCanvasRows = uiNumRows;
	uiMaxRow = uiNumRows - 1;

	if (!m_pScrFirstNd)
	{
		m_pScrFirstNd = getRootNode( m_pCurNd);
	}
	
	bRefreshEditWindow = TRUE;
	bRefreshStatusWindow = TRUE;
	pucSearchBuf[ 0] = '\0';

	if( m_hDefaultDb != HFDB_NULL)
	{
		FlmDbGetTransType( m_hDefaultDb, &uiTransType);
	}

	/*
	Call the callback to indicate that the interactive
	editor has been invoked
	*/

	if( m_pEventHook)
	{
		 m_pEventHook( this, F_RECEDIT_EVENT_IEDIT,
			0, m_EventData);
	}

	while( !bDoneEditing && !isExiting())
	{
		if( bRefreshEditWindow)
		{
			/*
			Refresh the edit window
			*/

			if( m_pParent)
			{
				m_bMonochrome = m_pParent->isMonochrome();
			}

			FTXWinPaintBackground( m_pEditWindow, 
				m_bMonochrome ? FLM_BLACK : FLM_BLUE);

			if( m_pEventHook)
			{
				if( RC_BAD( rc = m_pEventHook( this, F_RECEDIT_EVENT_REFRESH,
					0, m_EventData)))
				{
					goto Exit;
				}
			}

			refreshEditWindow( &m_pScrFirstNd, m_pCurNd, &m_uiCurRow);
			FTXWinSetCursorPos( m_pEditWindow, 0, m_uiCurRow);
			bRefreshEditWindow = FALSE;
			bRefreshStatusWindow = TRUE;
		}

		if( m_pEditStatusWin && bRefreshStatusWindow)
		{
			FTXWinSetBackFore( m_pEditStatusWin,
				m_bMonochrome ? FLM_LIGHTGRAY : FLM_GREEN,
				m_bMonochrome ? FLM_BLACK : FLM_WHITE);

			getControlFlags( m_pCurNd, &uiCurFlags);
			if( !(uiCurFlags & F_RECEDIT_FLAG_LIST_ITEM))
			{
				FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
				FTXWinPrintf( m_pEditStatusWin, "? = Help");
				
				if( m_pCurNd)
				{
					FLMUINT	uiTag = GedTagNum( m_pCurNd);

					if( uiTag)
					{
						FTXWinPrintf( m_pEditStatusWin, " | Field: %-5u (0x%4.4X)",
							(unsigned)GedTagNum( m_pCurNd),
							(unsigned)GedTagNum( m_pCurNd));

						FTXWinPrintf( m_pEditStatusWin, " | Type: ");

						switch( GedValType( m_pCurNd))
						{
							case FLM_CONTEXT_TYPE:
							{
								FTXWinPrintf( m_pEditStatusWin, "CONTEXT");
								break;
							}

							case FLM_NUMBER_TYPE:
							{
								FTXWinPrintf( m_pEditStatusWin, "NUMBER ");
								break;
							}

							case FLM_TEXT_TYPE:
							{
								FTXWinPrintf( m_pEditStatusWin, "TEXT   ");
								break;
							}

							case FLM_BINARY_TYPE:
							{
								FTXWinPrintf( m_pEditStatusWin, "BINARY ");
								break;
							}

							case FLM_BLOB_TYPE:
							{
								FTXWinPrintf( m_pEditStatusWin, "BLOB   ");
								break;
							}

							default:
							{
								FTXWinPrintf( m_pEditStatusWin, "(%5.5u)",
									(unsigned)GedValType( m_pCurNd));
								break;
							}
						}
					}

					if( isRecordModified( m_pCurNd))
					{
						FTXWinPrintf( m_pEditStatusWin, " | MOD");
					}
				}

				switch( uiTransType)
				{
					case FLM_UPDATE_TRANS:
					{
						FTXWinPrintf( m_pEditStatusWin, " | UTRANS");
						break;
					}
					case FLM_READ_TRANS:
					{
						FTXWinPrintf( m_pEditStatusWin, " | RTRANS");
						break;
					}
				}
			}
			
			FTXWinClearToEOL( m_pEditStatusWin);
			bRefreshStatusWindow = FALSE;
		}

		if( uiHelpKey || RC_OK( FTXWinTestKB( m_pEditWindow)))
		{
			FLMUINT	uiChar;

			bRefreshEditWindow = TRUE;
			
			if( uiHelpKey)
			{
				uiChar = uiHelpKey;
				uiHelpKey = 0;
			}
			else
			{
				FTXWinInputChar( m_pEditWindow, &uiChar);
			}

			if( uiChar)
			{
				if( m_pKeyHook)
				{
					m_pKeyHook( this, m_pCurNd, uiChar, m_KeyData, &uiChar);
				}
				if (uiChar != FKB_TAB)
				{
					m_uiLastKey = uiChar;
				}
			}

			if( uiChar == FKB_TAB)
			{
				// Grab the last keystroke that was passed to the editor.
				// This is needed in environments where the ALT and
				// control keys are not available and must be selected from
				// the command list.  Rather than requiring the user to
				// re-select the command from the list each time, he/she can
				// simply press the tab key to repeat the last keystroke.

				uiChar = m_uiLastKey;
			}

			getControlFlags( m_pCurNd, &uiCurFlags);
			switch( uiChar)
			{
				case 0:
				{
					/*
					Key handled by the keyboard callback.  Refresh the
					edit window so that any changes made to the buffer by
					the callback will be displayed.
					*/

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Move field cursor to the next field
				*/

				case FKB_DOWN:
				{
					if( (pTmpNd = getNextNode( m_pCurNd)) != NULL)
					{
						if( m_uiCurRow < uiMaxRow)
						{
							refreshRow( m_uiCurRow, m_pCurNd, FALSE);
							m_uiCurRow++;
							refreshRow( m_uiCurRow, pTmpNd, TRUE);
							bRefreshStatusWindow = TRUE;
						}
						else
						{
							bRefreshEditWindow = TRUE;
						}
						m_pCurNd = pTmpNd;
					}
					break;
				}

				/*
				Move field cursor to the prior field
				*/

				case FKB_UP:
				{
					if( (pTmpNd = getPrevNode( m_pCurNd)) != NULL)
					{
						if( m_uiCurRow > 0)
						{
							refreshRow( m_uiCurRow, m_pCurNd, FALSE);
							m_uiCurRow--;
							refreshRow( m_uiCurRow, pTmpNd, TRUE);
							bRefreshStatusWindow = TRUE;
						}
						else
						{
							bRefreshEditWindow = TRUE;
						}
						m_pCurNd = pTmpNd;
					}
					break;
				}

				/*
				Page up
				*/

				case FKB_PGUP:
				{
					for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
					{
						if( (pTmpNd = getPrevNode( m_pScrFirstNd)) != NULL)
						{
							m_pScrFirstNd = pTmpNd;
						}

						if( (pTmpNd = getPrevNode( m_pCurNd)) != NULL)
						{
							m_pCurNd = pTmpNd;
						}
						else
						{
							m_uiCurRow = 0;
							break;
						}
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Jump to the root of the next record
				*/
				
				case '>':
				case FKB_CTRL_DOWN:
				{
					NODE *		pRootNd = getRootNode( m_pCurNd);

					if( pRootNd)
					{
						if( (pTmpNd = GedSibNext( pRootNd)) != NULL)
						{
							m_pCurNd = pTmpNd;
							setCurrentAtTop();
						}
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Jump to the root of the current or prior record
				*/

				case '<':
				case FKB_CTRL_UP:
				{
					NODE *		pRootNd = getRootNode( m_pCurNd);

					if( pRootNd)
					{
						if( m_pCurNd == pRootNd)
						{
							if( (pTmpNd = GedSibPrev( pRootNd)) != NULL)
							{
								m_pCurNd = pTmpNd;
							}
						}
						else
						{
							m_pCurNd = pRootNd;
						}
						setCurrentAtTop();
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Page down
				*/

				case FKB_PGDN:
				{
					for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
					{
						if( (pTmpNd = getNextNode( m_pCurNd)) != NULL)
						{
							m_pCurNd = pTmpNd;
						}
						else
						{
							m_uiCurRow += uiLoop;
							if( m_uiCurRow >= uiNumRows)
							{
								m_uiCurRow = uiNumRows - 1;
							}
							break;
						}
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Go to the top of the buffer
				*/

				case FKB_HOME:
				case FKB_GOTO:
				{
					m_pCurNd = m_pTree;
					m_uiCurRow = 0;
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Jump to the end of the buffer
				*/

				case FKB_END:
				case FKB_CTRL_END:
				{
					m_uiCurRow = uiMaxRow;
					for( ;;)
					{
						if( (pTmpNd = getNextNode( m_pCurNd)) != NULL)
						{
							m_pCurNd = pTmpNd;
						}
						else
						{
							break;
						}
					}

					m_pScrFirstNd = m_pCurNd;
					setCurrentAtBottom();
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Delete the current record from the database
				*/

				case FKB_DELETE:
				{
					NODE *		pRootNd;
					char			pucResponse[ 2];

					if( !m_pCurNd)
					{
						break;
					}
					
					if( canDeleteNode( m_pCurNd))
					{
						pRootNd = getRootNode( m_pCurNd);
						if( pRootNd == m_pCurNd && GedGetRecId( pRootNd))
						{
							*pucResponse = '\0';
							requestInput(
								"Delete Record from the Database? (Y/N)",
								pucResponse, 2, &uiTermChar);
					
							if( uiTermChar == FKB_ESCAPE)
							{
								goto Delete_Exit;
							}
							
							if( *pucResponse == 'y' || *pucResponse == 'Y')
							{
								if( m_pEditStatusWin)
								{
									FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
									FTXWinPrintf( m_pEditStatusWin,
										"Deleting record from database ...");
									FTXWinClearToEOL( m_pEditStatusWin);
								}

								if( RC_BAD( tmpRc = deleteRecordFromDb( m_pCurNd)))
								{
									if( m_pEditStatusWin)
									{
										FTXWinClearLine( m_pEditStatusWin, 0, 0);
									}
									displayMessage(
										"Unable to delete record", tmpRc,
										NULL, FLM_RED, FLM_WHITE);
									goto Delete_Exit;
								}
								pruneTree( m_pCurNd);
							}
						}
						else
						{
							pruneTree( m_pCurNd);
						}
					}
					else
					{
						displayMessage(
							"Deletion not allowed",
							RC_SET( FERR_ACCESS_DENIED),
							NULL, FLM_RED, FLM_WHITE);
					}
Delete_Exit:
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Delete records from the database
				*/
				
				case FKB_ALT_D:
				{
					FLMUINT		uiContainer;
					char			pucResponse[ 16];
					FLMUINT		uiDrn;

					pucAction[ 0] = '\0';
					requestInput(
						"Delete By (r = record #, q = query)",
						pucAction, sizeof( pucAction), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}

					if( *pucAction == 'r' || *pucAction == 'R')
					{
						*pucResponse = '\0';
						requestInput(
							"[DELETE] Record Number",
							pucResponse, sizeof( pucResponse), &uiTermChar);

						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}

						if( pucResponse[ 0])
						{
							if( RC_BAD( tmpRc = getNumber( pucResponse, &uiDrn, NULL)))
							{
								displayMessage(
									"Invalid record number", tmpRc,
									NULL, FLM_RED, FLM_WHITE);
								break;
							}
						}
						else
						{
							uiDrn = 0;
						}

						if( RC_BAD( tmpRc = selectContainer( &uiContainer, &uiTermChar)))
						{
							displayMessage(
								"Error getting container", tmpRc,
								NULL, FLM_RED, FLM_WHITE);
						}
						
						if( uiTermChar != FKB_ENTER)
						{
							break;
						}

						if( m_pEditStatusWin)
						{
							FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
							FTXWinPrintf( m_pEditStatusWin,
								"Deleting record from database ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = deleteRecordFromDb( m_hDefaultDb,
							uiContainer, uiDrn)))
						{
							if( m_pEditStatusWin)
							{
								FTXWinClearLine( m_pEditStatusWin, 0, 0);
							}
							displayMessage( "Delete failed", tmpRc,
								NULL, FLM_RED, FLM_WHITE);
						}
						else
						{
							if( (pTmpNd = findRecord( uiContainer, uiDrn, NULL)) != NULL)
							{
								pruneTree( pTmpNd);
							}
						}
					}
					else if( *pucAction == 'q' || *pucAction == 'Q')
					{
						if( RC_BAD( tmpRc = adHocQuery( FALSE, TRUE)))
						{
							displayMessage( "Query Failure",
								tmpRc, NULL, FLM_RED, FLM_WHITE);
						}
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Modify the current record in the database
				*/

				case FKB_ALT_M:
				{
					char			pucResponse[ 2];
					FLMBOOL		bModifyInBackground = FALSE;
					FLMBOOL		bCreateSuspended = FALSE;
					FLMUINT		uiContainer;

					*pucResponse = '\0';
					requestInput(
						"Update Record in the Database? (Y/N)",
						pucResponse, 2, &uiTermChar);
			
					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( *pucResponse != 'Y' && *pucResponse != 'y')
					{
						break;
					}

					if( RC_OK( GedGetRecSource(
						m_pCurNd, NULL, &uiContainer, NULL)))
					{
						if (uiContainer == FLM_DICT_CONTAINER &&
							 GedTagNum( m_pCurNd) == FLM_INDEX_TAG)
						{
							f_strcpy( pucResponse, "Y");
							requestInput(
								"Modify index in background? (Y/N)",
								pucResponse, 2, &uiTermChar);
							if( uiTermChar == FKB_ESCAPE)
							{
								break;
							}
							
							bModifyInBackground = (*pucResponse == 'Y' || *pucResponse == 'y')
													 ? TRUE
													 : FALSE;

							if( bModifyInBackground)
							{
								f_strcpy( pucResponse, "Y");
								requestInput(
									"Start the indexing thread? (Y/N)",
									pucResponse, 2, &uiTermChar);
								if( uiTermChar == FKB_ESCAPE)
								{
									break;
								}
								
								bCreateSuspended = (*pucResponse == 'Y' || *pucResponse == 'y')
														 ? FALSE
														 : TRUE;
							}
						}
					}

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						FTXWinPrintf( m_pEditStatusWin, "Updating database ...");
						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( RC_BAD( tmpRc = modifyRecordInDb( m_pCurNd, 
						bModifyInBackground, bCreateSuspended)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Modify failed", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}
					else
					{
						clearRecordModified( m_pCurNd);
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Add the current record to the database
				*/
				
				case FKB_ALT_A:
				{
					FLMUINT		uiContainer;
					char			pucResponse[ 16];
					FLMUINT		uiDrn;
					FLMBOOL		bAddInBackground = FALSE;
					FLMBOOL		bCreateSuspended = FALSE;
					
					*pucResponse = '\0';
					requestInput(
						"[ADD] Record Number",
						pucResponse, sizeof( pucResponse), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}

					if( pucResponse[ 0])
					{
						if( RC_BAD( tmpRc = getNumber( pucResponse, &uiDrn, NULL)))
						{
							displayMessage(
								"Invalid record number", tmpRc,
								NULL, FLM_RED, FLM_WHITE);
							break;
						}
					}
					else
					{
						uiDrn = 0;
					}

					if( RC_BAD( tmpRc = selectContainer( &uiContainer, &uiTermChar)))
					{
						displayMessage(
							"Error getting container", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}
					
					if( uiTermChar != FKB_ENTER)
					{
						break;
					}

					if (uiContainer == FLM_DICT_CONTAINER &&
						 GedTagNum( m_pCurNd) == FLM_INDEX_TAG)
					{
						f_strcpy( pucResponse, "Y");
						requestInput(
							"Add index in background? (Y/N)",
							pucResponse, 2, &uiTermChar);
						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}
						
						bAddInBackground = (*pucResponse == 'Y' || *pucResponse == 'y')
												 ? TRUE
												 : FALSE;

						if( bAddInBackground)
						{
							f_strcpy( pucResponse, "Y");
							requestInput(
								"Start the indexing thread? (Y/N)",
								pucResponse, 2, &uiTermChar);
							if( uiTermChar == FKB_ESCAPE)
							{
								break;
							}
							
							bCreateSuspended = (*pucResponse == 'Y' || *pucResponse == 'y')
													 ? FALSE
													 : TRUE;
						}
					}
					else
					{
						bAddInBackground = FALSE;
						bCreateSuspended = FALSE;
					}

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						FTXWinPrintf( m_pEditStatusWin,
							"Adding record to database ...");
						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( RC_BAD( tmpRc = addRecordToDb( m_pCurNd,
						uiContainer, bAddInBackground, bCreateSuspended, &uiDrn)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}

						displayMessage( "Add failed", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}
					else
					{
						clearRecordModified( m_pCurNd);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Index operations
				*/

				case FKB_ALT_I:
				{
					if( m_hDefaultDb == HFDB_NULL)
					{
						break;
					}

					if( RC_BAD( tmpRc = indexList()))
					{
						displayMessage( "Index List Operation Failed", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Retrieve records
				*/

				case FKB_ALT_R:
				{
					FLMUINT		uiContainer;
					char			pucResponse[ 32];
					FLMUINT		uiSrcLen;
					FLMUINT		uiFirstDrn;
					FLMUINT		uiLastDrn;
					
					*pucResponse = '\0';
					requestInput(
						"[READ] Starting Record Number",
						pucResponse, sizeof( pucResponse), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( (uiSrcLen = (FLMUINT)f_strlen( pucResponse)) == 0)
					{
						uiFirstDrn = 0;
					}
					else
					{
						if( RC_BAD( tmpRc = getNumber( pucResponse, &uiFirstDrn, NULL)))
						{
							displayMessage( "Invalid record number", tmpRc,
								NULL, FLM_RED, FLM_WHITE);
							break;
						}
					}

					requestInput(
						"[READ] Ending Record Number",
						pucResponse, sizeof( pucResponse), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( (uiSrcLen = (FLMUINT)f_strlen( pucResponse)) == 0)
					{
						uiLastDrn = 0xFFFFFFFF;
					}
					else
					{
						if( RC_BAD( tmpRc = getNumber( pucResponse, &uiLastDrn, NULL)))
						{
							displayMessage( "Invalid record number", tmpRc,
								NULL, FLM_RED, FLM_WHITE);
							break;
						}
					}

					if( RC_BAD( tmpRc = selectContainer( &uiContainer, &uiTermChar)))
					{
						displayMessage( "Error getting container", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}
					
					if( uiTermChar != FKB_ENTER)
					{
						break;
					}

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);

						if( uiFirstDrn == uiLastDrn)
						{

							FTXWinPrintf( m_pEditStatusWin,
								"Retrieving record from the database ...");
						}
						else if( uiFirstDrn == 0 && uiLastDrn == 0xFFFFFFFF)
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Retrieving all records ...");
						}
						else
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Retrieving records %u - %u from the database ...",
								(unsigned)uiFirstDrn, (unsigned)uiLastDrn);
						}

						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( RC_BAD( tmpRc = retrieveRecordsFromDb( uiContainer,
						uiFirstDrn, uiLastDrn)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Unable to retrieve record", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Search
				*/

				case FKB_ALT_F3:
				{
					*pucSearchBuf = '\0';
					requestInput(
						"[SEARCH]", pucSearchBuf, sizeof( pucSearchBuf),
						&uiTermChar);

					f_strupr( (char *)pucSearchBuf);
					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					/*
					No break.  Fall through to FKB_F3 case.
					*/
				}

				/*
				Search forward
				*/

				case FKB_F3:
				{
					FLMBOOL		bFoundMatch = FALSE;
					FLMBOOL		bTagSearch = FALSE;
					FLMUINT		uiTagNum = 0;
					FLMUINT		uiTmp;

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						FTXWinPrintf( m_pEditStatusWin, "Searching ...");
						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( pucSearchBuf[ 0] == '#')
					{
						if( RC_OK( tmpRc = getNumber( &(pucSearchBuf[ 1]), &uiTmp, NULL)))
						{
							bTagSearch = TRUE;
							uiTagNum = uiTmp;
						}
					}

					pTmpNd = getNextNode( m_pCurNd);
					while( pTmpNd)
					{
						if( bTagSearch)
						{
							if( GedTagNum( pTmpNd) == uiTagNum)
							{
								m_pCurNd = pTmpNd;
								bFoundMatch = TRUE;
								break;
							}
						}
						else
						{
							if( RC_OK( getDisplayValue( pTmpNd, F_RECEDIT_DEFAULT_TYPE,
								m_pucTmpBuf, F_RECEDIT_BUF_SIZE)))
							{
								f_strupr( (char *)m_pucTmpBuf);
								if( f_strstr( m_pucTmpBuf, pucSearchBuf) != 0)
								{
									m_pCurNd = pTmpNd;
									bFoundMatch = TRUE;
									break;
								}
							}
						}
						pTmpNd = getNextNode( pTmpNd);
					}

					if( m_pEditStatusWin)
					{
						FTXWinClearLine( m_pEditStatusWin, 0, 0);
					}
					if( !bFoundMatch)
					{
						displayMessage( "No matches found",
							RC_SET( FERR_EOF_HIT), NULL, FLM_RED, FLM_WHITE);
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Search backward
				*/

				case FKB_SF3:
				{
					FLMBOOL		bFoundMatch = FALSE;
					FLMBOOL		bTagSearch = FALSE;
					FLMUINT		uiTagNum = 0;
					FLMUINT		uiTmp;

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						FTXWinPrintf( m_pEditStatusWin, "Searching ...");
						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( pucSearchBuf[ 0] == '#')
					{
						if( RC_OK( tmpRc = getNumber( &(pucSearchBuf[ 1]), &uiTmp, NULL)))
						{
							bTagSearch = TRUE;
							uiTagNum = uiTmp;
						}
					}

					pTmpNd = getPrevNode( m_pCurNd);
					while( pTmpNd)
					{
						if( bTagSearch)
						{
							if( GedTagNum( pTmpNd) == uiTagNum)
							{
								m_pCurNd = pTmpNd;
								bFoundMatch = TRUE;
								break;
							}
						}
						else
						{
							if( RC_OK( getDisplayValue( pTmpNd, F_RECEDIT_DEFAULT_TYPE,
								m_pucTmpBuf, F_RECEDIT_BUF_SIZE)))
							{
								f_strupr( (char *)m_pucTmpBuf);
								if( f_strstr( m_pucTmpBuf, pucSearchBuf) != 0)
								{
									m_pCurNd = pTmpNd;
									bFoundMatch = TRUE;
									break;
								}
							}
						}
						pTmpNd = getPrevNode( pTmpNd);
					}

					if( m_pEditStatusWin)
					{
						FTXWinClearLine( m_pEditStatusWin, 0, 0);
					}

					if( !bFoundMatch)
					{
						displayMessage( "No matches found",
							RC_SET( FERR_BOF_HIT), NULL, FLM_RED, FLM_WHITE);
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Follow a record pointer
				*/

				case FKB_RIGHT:
				case FKB_LEFT:
				case FKB_CTRL_RIGHT:
				case FKB_CTRL_LEFT:
				{
					if( !m_pCurNd)
					{
						break;
					}

					if( RC_BAD( tmpRc = followLink( m_pCurNd, uiChar)))
					{
						displayMessage( "Unable to follow link", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Insert a new field
				*/

				case FKB_INSERT:
				{
					NODE *		pNewNd = NULL;
					char			pucLocation[ 2];
					FLMBOOL		bChild = FALSE;
					FLMBOOL		bSibling = FALSE;
					FLMBOOL		bRoot = FALSE;

					if( m_pCurNd)
					{
						if( !canEditRecord( m_pCurNd))
						{
							displayMessage( "This record cannot be edited",
								RC_SET( FERR_ACCESS_DENIED), NULL, FLM_RED, FLM_WHITE);
							break;
						}

						pucLocation[ 0] = '\0';
						requestInput(
							"Location (c = child, s = sibling, r = root)",
							pucLocation, sizeof( pucLocation), &uiTermChar);

						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}

						if( *pucLocation == 'c' || *pucLocation == 'C')
						{
							bChild = TRUE;
						}
						else if( *pucLocation == 's' || *pucLocation == 'S')
						{
							bSibling = TRUE;
						}
						else if( *pucLocation == 'r' || *pucLocation == 'R')
						{
							bRoot = TRUE;
						}
						else
						{
							displayMessage( "Invalid Request",
								RC_SET( FERR_FAILURE), NULL, FLM_RED, FLM_WHITE);
							break;
						}
					}
					else
					{
						bRoot = TRUE;
					}
					if (m_hDefaultDb == HFDB_NULL)
					{
						openNewDb();
						if (m_hDefaultDb == HFDB_NULL)
							break;
					}

					if( RC_BAD( tmpRc = createNewField( bRoot, &pNewNd)))
					{
						displayMessage( "Unable to create new field", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
						break;

					}

					if( pNewNd)
					{
						if( m_pCurNd)
						{
							if( bChild)
							{
								GedChildGraft( m_pCurNd, pNewNd, GED_FIRST);
								m_pCurNd = pNewNd;
								(void)markRecordModified( m_pCurNd);
							}
							else if( bSibling)
							{
								GedSibGraft( m_pCurNd, pNewNd, 0);
								m_pCurNd = pNewNd;
								(void)markRecordModified( m_pCurNd);
							}
							else if( bRoot)
							{
								_insertRecord( pNewNd);
								m_pScrFirstNd = NULL;
								m_pCurNd = pNewNd;
								(void)markRecordModified( m_pCurNd);
							}
						}
						else
						{
							m_pTree = pNewNd;
							m_pCurNd = pNewNd;
							m_pScrFirstNd = pNewNd;
							(void)markRecordModified( m_pCurNd);
						}

						bRefreshEditWindow = TRUE;
						break;
					}
					break;
				}

				/*
				Edit the current field
				*/

				case FKB_ENTER:
				{
					if( !m_pCurNd)
					{
						break;
					}

					if( uiCurFlags & F_RECEDIT_FLAG_LIST_ITEM)
					{
						setControlFlags( m_pCurNd,
							uiCurFlags | F_RECEDIT_FLAG_SELECTED);
						bDoneEditing = TRUE;
					}
					else if( !canEditRecord( m_pCurNd))
					{
						displayMessage( "This record cannot be edited",
							RC_SET( FERR_ACCESS_DENIED), NULL, FLM_RED, FLM_WHITE);
					}
					else if( !canEditNode( m_pCurNd))
					{
						displayMessage( "The field cannot be edited",
							RC_SET( FERR_ACCESS_DENIED), NULL, FLM_RED, FLM_WHITE);
					}
					else if( RC_BAD( tmpRc = editNode( m_uiCurRow, m_pCurNd)))
					{
						displayMessage( "The field could not be edited", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Transaction operations
				*/

				case FKB_ALT_T:
				{
					if( m_hDefaultDb == HFDB_NULL)
					{
						break;
					}

					if( uiTransType == FLM_NO_TRANS)
					{
						pucAction[ 0] = '\0';
						requestInput(
							"Begin Transaction (type: r = read, u = update)",
							pucAction, sizeof( pucAction), &uiTermChar);

						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}

						if( m_pEditStatusWin)
						{
							FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						}

						if( *pucAction == 'r' || *pucAction == 'R')
						{
							if( m_pEditStatusWin)
							{
								FTXWinPrintf( m_pEditStatusWin, "Starting read transaction ...");
								FTXWinClearToEOL( m_pEditStatusWin);
							}

							if( RC_OK( tmpRc = FlmDbTransBegin( m_hDefaultDb,
								FLM_READ_TRANS, 0)))
							{
								uiTransType = FLM_READ_TRANS;
								bStartedTrans = TRUE;
							}
						}
						else if( *pucAction == 'u' || *pucAction == 'U')
						{
							if( m_pEditStatusWin)
							{
								FTXWinPrintf( m_pEditStatusWin, "Starting update transaction ...");
								FTXWinClearToEOL( m_pEditStatusWin);
							}

							if( RC_OK( tmpRc = FlmDbTransBegin( m_hDefaultDb,
								FLM_UPDATE_TRANS, 15)))
							{
								uiTransType = FLM_UPDATE_TRANS;
								bStartedTrans = TRUE;
							}
						}
						else
						{
							displayMessage( "Invalid Request",
								RC_SET( FERR_FAILURE), NULL, FLM_RED, FLM_WHITE);
							break;
						}

						if( RC_BAD( tmpRc))
						{
							if( m_pEditStatusWin)
							{
								FTXWinClearLine( m_pEditStatusWin, 0, 0);
							}

							displayMessage( "Unable to begin transaction",
								RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
						}
					}
					else
					{
						pucAction[ 0] = '\0';
						requestInput(
							"End Transaction (a = abort, c = commit)",
							pucAction, sizeof( pucAction), &uiTermChar);

						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}

						if( m_pEditStatusWin)
						{
							FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
						}

						if( *pucAction == 'c' || *pucAction == 'C')
						{
							if( m_pEditStatusWin)
							{
								FTXWinPrintf( m_pEditStatusWin, "Committing transaction ...");
								FTXWinClearToEOL( m_pEditStatusWin);
							}

							if( RC_OK( tmpRc = FlmDbTransCommit( m_hDefaultDb)))
							{
								uiTransType = FLM_NO_TRANS;
								bStartedTrans = FALSE;
							}
						}
						else if( *pucAction == 'a' || *pucAction == 'A')
						{
							if( m_pEditStatusWin)
							{
								FTXWinPrintf( m_pEditStatusWin, "Aborting transaction ...");
								FTXWinClearToEOL( m_pEditStatusWin);
							}

							if( RC_OK( tmpRc = FlmDbTransAbort( m_hDefaultDb)))
							{
								uiTransType = FLM_NO_TRANS;
								bStartedTrans = FALSE;
							}
						}
						else
						{
							displayMessage( "Invalid Request",
								RC_SET( FERR_FAILURE), NULL, FLM_RED, FLM_WHITE);
							break;
						}

						if( RC_BAD( tmpRc))
						{
							if( m_pEditStatusWin)
							{
								FTXWinClearLine( m_pEditStatusWin, 0, 0);
							}

							displayMessage( "Unable to end transaction",
								RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
						}
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Synchronize the current record with the version of the
				record in the database (go get the record from
				the database)
				*/

				case FKB_ALT_S:
				{
					FLMUINT	uiTmpCont;
					FLMUINT	uiTmpDrn;
					NODE *	pTmpRoot;

					if( (pTmpRoot = getRootNode( m_pCurNd)) != NULL)
					{
						if( RC_OK( GedGetRecSource( pTmpRoot, NULL,
							&uiTmpCont, &uiTmpDrn)))
						{
							if( isRecordModified( pTmpRoot))
							{
								char			pucResponse[ 2];

								*pucResponse = '\0';
								requestInput(
									"Syncronizing this record will discard modifications.  OK (Y/N)",
									pucResponse, 2, &uiTermChar);
						
								if( uiTermChar == FKB_ESCAPE)
								{
									goto Sync_Exit;
								}
								
								if( *pucResponse != 'y' && *pucResponse != 'Y')
								{
									goto Sync_Exit;
								}
							}

							if( RC_BAD( tmpRc = retrieveRecordsFromDb( uiTmpCont,
								uiTmpDrn, uiTmpDrn)))
							{
								if( tmpRc == FERR_EOF_HIT || tmpRc == FERR_NOT_FOUND)
								{
									(void)pruneTree( pTmpRoot);
								}
								displayMessage( "Unable to synchronize record",
									RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
							}
							bRefreshEditWindow = TRUE;
						}
					}
Sync_Exit:
					break;
				}

				/*
				Clear all records from the current editor buffer.
				NOTE: This will discard all changes
				*/

				case FKB_ALT_C:
				{
					char			pucResponse[ 2];

					*pucResponse = '\0';
					requestInput(
						"Clear buffer and discard modifications? (Y/N)",
						pucResponse, 2, &uiTermChar);
					
					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( *pucResponse == 'y' || *pucResponse == 'Y')
					{
						setTree( NULL);
						bRefreshEditWindow = TRUE;
					}
					break;
				}

				/*
				Global administration options (including statistics gathering)
				*/

				case '#':
				{
					pucAction[ 0] = '\0';
					requestInput(
						"Statistics (b = begin, e = end, r = reset)",
						pucAction, sizeof( pucAction), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
					}

					if( *pucAction == 'b' || *pucAction == 'B')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Starting statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig(
							F_RECEDIT_CONFIG_STATS_START)))
						{
							displayMessage( "Error Starting Statistics",
								tmpRc, NULL, FLM_RED, FLM_WHITE);
						}
					}
					else if( *pucAction == 'e' || *pucAction == 'E')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Stopping statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig(
							F_RECEDIT_CONFIG_STATS_STOP)))
						{
							displayMessage( "Error Stopping Statistics",
								tmpRc, NULL, FLM_RED, FLM_WHITE);
						}
					}
					else if( *pucAction == 'r' || *pucAction == 'R')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Resetting statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig(
							F_RECEDIT_CONFIG_STATS_RESET)))
						{
							displayMessage( "Error Resetting Statistics",
								tmpRc, NULL, FLM_RED, FLM_WHITE);
						}
					}
					else
					{
						displayMessage( "Invalid Request",
							RC_SET( FERR_FAILURE), NULL, FLM_RED, FLM_WHITE);
					}
					bRefreshStatusWindow = TRUE;
					break;
				}

				case '+':
				case '-':
				{
					FLMBOOL	bDidIt;

					expandNode( m_pCurNd, &bDidIt);
					if (!bDidIt)
					{
						collapseNode( m_pCurNd, &bDidIt);
					}
					if (bDidIt)
					{
						bRefreshEditWindow = TRUE;
					}
					break;
				}

				case '?':
				{
					showHelp( &uiHelpKey);
					break;
				}

				/*
				"Find" records based on ad hoc criteria
				*/

				case FKB_ALT_F:
				{
					if( RC_BAD( tmpRc = adHocQuery()))
					{
						displayMessage( "Query Failure",
							tmpRc, NULL, FLM_RED, FLM_WHITE);
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				case FKB_ALT_F10:
				{
					if( m_bMonochrome)
					{
						m_bMonochrome = FALSE;
					}
					else
					{
						m_bMonochrome = TRUE;
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				case FKB_F1:
				{
					char	szSelectedPath [F_PATH_MAX_SIZE];
					fileManager( NULL, 0, NULL, szSelectedPath, NULL);
					bRefreshEditWindow = FALSE;
					break;
				}

				case FKB_CTRL_C:
				case FKB_CTRL_X:
				{
					if( m_pCurNd && !isSystemNode( m_pCurNd))
					{
						if( uiChar == FKB_CTRL_X && !canDeleteNode( m_pCurNd))
						{
							displayMessage(
								"Deletion not allowed",
								RC_SET( FERR_ACCESS_DENIED),
								NULL, FLM_RED, FLM_WHITE);
						}
						else
						{
							pCopyNd = NULL;
							copyPool.poolReset();

							if( RC_BAD( copyCleanTree( &copyPool, m_pCurNd, &pCopyNd)))
							{
								pCopyNd = NULL;
								break;
							}
							
							if( uiChar == FKB_CTRL_X)
							{
								pruneTree( m_pCurNd);
							}
						}
					}

					break;
				}

				case FKB_CTRL_V:
				{
					if( pCopyNd)
					{
						if( m_pCurNd && !isSystemNode( m_pCurNd))
						{
							if( (pTmpNd = GedCopy( &m_treePool, GED_TREE, pCopyNd)) == NULL)
							{
								rc = RC_SET( FERR_MEM);
								goto Exit;
							}
							GedSibGraft( m_pCurNd, pTmpNd, 0);
							m_pCurNd = GedSibNext( m_pCurNd);
							(void)markRecordModified( m_pCurNd);
						}
						else if( !m_pCurNd)
						{
							setTree( pCopyNd);
							(void)markRecordModified( m_pCurNd);
						}
					}
					break;
				}

				case FKB_F8: // Index Manager
				{
					if( m_hDefaultDb == HFDB_NULL)
					{
						break;
					}

					f_threadDestroy( &pIxManagerThrd);

					if( IsInCSMode( m_hDefaultDb))
					{
						f_strcpy( szDbPath, ((FDB *)m_hDefaultDb)->pCSContext->pucUrl);
					}
					else
					{
						if( RC_BAD( FlmDbGetConfig( m_hDefaultDb, 
							FDB_GET_PATH, (void *)(&szDbPath [0]))))
						{
							break;
						}
					}

					if (RC_OK( FlmDbOpen( szDbPath, NULL, NULL,
						0, NULL, &hTmpDb)))
					{
						f_threadCreate( &pIxManagerThrd, flstIndexManagerThread, 
							"index_manager", 0, 0, (void *)hTmpDb);
					}
					break;
				}

				case FKB_F9: // Memory Manager
				{
					f_threadDestroy( &pMemManagerThrd);
					f_threadCreate( &pMemManagerThrd,
						flstMemoryManagerThread, "memory_manager");
					break;
				}

				case FKB_F10: // Tracker Monitor
				{
					if( m_hDefaultDb == HFDB_NULL)
					{
						break;
					}

					f_threadDestroy( &pTrackerMonitorThrd);

					if( RC_BAD( FlmDbGetConfig( m_hDefaultDb, 
						FDB_GET_PATH, (void *)(&szDbPath [0]))))
					{
						break;
					}

					if( RC_OK( FlmDbOpen( szDbPath, NULL, NULL,
						0, NULL, &hTmpDb)))
					{
						f_threadCreate( &pTrackerMonitorThrd,
							flstTrackerMonitorThread, "tracker_monitor",
							0, 0, (void *)hTmpDb);
					}

					break;
				}

				case FKB_ESCAPE:
				case FKB_ALT_Q:
				{
					// VISIT: See if any of the records in the buffer have
					// been modified and ask the user if the changes should
					// be discarded.  Also, need to add a batch update option
					// to allow all modified records to be added to the database
					// without forcing the user to visit each individual record.

					bDoneEditing = TRUE;
					break;
				}
				
				default:
				{
					/*
					Unrecognized key ... ignore.
					*/

					break;
				}
			}
		}
		else
		{
			f_sleep( 1);
		}
	}

Exit:

	if( bStartedTrans)
	{
		// Abort any active transactions
		FlmDbTransAbort( m_hDefaultDb);
	}

	f_threadDestroy( &pIxManagerThrd);
	f_threadDestroy( &pMemManagerThrd);
	f_threadDestroy( &pTrackerMonitorThrd);

	if( m_pEditWindow)
	{
		(void) FTXWinFree( &m_pEditWindow);
	}

	if( m_pEditStatusWin)
	{
		(void) FTXWinFree( &m_pEditStatusWin);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::refreshEditWindow(
	NODE **				ppFirstNd,
	NODE * 				pCursorNd,
	FLMUINT *			puiCurRow)
{
	FLMUINT		uiLoop;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;
	NODE *		pFirstNd;
	NODE *		pTmpNd;
	FLMUINT		uiCurRow;
	FLMBOOL		bCurrentVisible = FALSE;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( pCursorNd == NULL)
	{
		*ppFirstNd = NULL;
		*puiCurRow = 0;
	}

	FTXWinGetCanvasSize( m_pEditWindow, &uiNumCols, &uiNumRows);

	// VISIT: May want to check the current source
	// against the current record's source and
	// refresh the source and name table based on
	// the source if it is different than the
	// current source.  If the current record does
	// not have source information, do not reset
	// the current source or refresh the name table.

	/*
	See if the current node is already being displayed.
	*/

	uiCurRow = 0;
	pFirstNd = *ppFirstNd;
	pTmpNd = pFirstNd;
	for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
	{
		if( pCursorNd == pTmpNd)
		{
			uiCurRow = uiLoop;
			bCurrentVisible = TRUE;
			break;
		}

		if( (pTmpNd = getNextNode( pTmpNd)) == NULL)
		{
			break;
		}
	}

	/*
	If the current node is not displayed, scroll the screen
	so that the node is visible.
	*/

	if( !bCurrentVisible)
	{
		uiCurRow = *puiCurRow;
		pFirstNd = pCursorNd;
		while( uiCurRow && pFirstNd)
		{
			pTmpNd = getPrevNode( pFirstNd);
			if( pTmpNd)
			{
				pFirstNd = pTmpNd;
			}
			uiCurRow--;
		}
	}

	*ppFirstNd = pFirstNd;
	*puiCurRow = uiCurRow;

	/*
	Turn display refresh off temporarily
	*/

	FTXSetRefreshState( TRUE);

	/*
	Refresh all rows of the edit window.  All rows beyond the end
	of the tree are cleared.
	*/

	pTmpNd = *ppFirstNd;
	for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
	{
		if( pTmpNd && pTmpNd == pCursorNd)
		{
			refreshRow( uiLoop, pTmpNd, TRUE);
			*puiCurRow = uiLoop;
		}
		else
		{
			refreshRow( uiLoop, pTmpNd, FALSE);
		}

		if( pTmpNd)
		{
			pTmpNd = getNextNode( pTmpNd);
		}
	}

	FTXSetRefreshState( FALSE);
	return( rc);
}
  
/****************************************************************************
Desc: Default line display format routine (for GEDCOM)
*****************************************************************************/
RCODE f_RecEditorDefaultDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals)
{
	FLMUINT		uiFlags = 0;
	FLMUINT		uiDrn;
	FLMUINT		uiCont;
	FLMUINT		uiCol = 0;
	char			pucValBuf[ 80];
	NODE *		pSystemNd;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	if( !pNd)
	{
		goto Exit;
	}

	pRecEditor->getControlFlags( pNd, &uiFlags);
	if( !pRecEditor->isSystemNode( pNd))
	{
		/*
		Output the level
		*/

		if( !(uiFlags & F_RECEDIT_FLAG_HIDE_LEVEL))
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"%u", (unsigned)GedNodeLevel( pNd));
			pDispVals[ *puiNumVals].uiCol = (GedNodeLevel( pNd) * 2);
			pDispVals[ *puiNumVals].foreground = FLM_WHITE;
			pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
			uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + (GedNodeLevel( pNd) * 2) + 1;
			(*puiNumVals)++;
		}
		else
		{
			uiCol += GedNodeLevel( pNd) * 2;
		}

		/*
		Output the tag
		*/

		if( !(uiFlags & F_RECEDIT_FLAG_HIDE_TAG))
		{
			if( RC_BAD( pRecEditor->getDictionaryName(
				GedTagNum( pNd), pDispVals[ *puiNumVals].pucString)))
			{
				f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
					"%u", (unsigned)GedTagNum( pNd));
			}

			pDispVals[ *puiNumVals].uiCol = uiCol;
#ifdef FLM_WIN
			pDispVals[ *puiNumVals].foreground = pRecEditor->isMonochrome() ? FLM_WHITE : FLM_LIGHTGREEN;
#else
			pDispVals[ *puiNumVals].foreground = pRecEditor->isMonochrome() ? FLM_LIGHTGRAY : FLM_GREEN;
#endif
			pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
			uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + 1;
			(*puiNumVals)++;
		}

		/*
		Output the record source
		*/

		if( !GedNodeLevel( pNd) && !(uiFlags & F_RECEDIT_FLAG_HIDE_SOURCE) &&
			RC_OK( GedGetRecSource( pNd, NULL, &uiCont, &uiDrn)))
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"@%u@ (0x%4.4X)", (unsigned)uiDrn, (unsigned)uiDrn);
			pDispVals[ *puiNumVals].uiCol = uiCol;
			pDispVals[ *puiNumVals].foreground = FLM_WHITE;
			pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
			uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + 1;
			(*puiNumVals)++;
		}

		/*
		Output the display value
		*/

		if( RC_BAD( rc = pRecEditor->getDisplayValue( pNd,
			F_RECEDIT_DEFAULT_TYPE, pDispVals[ *puiNumVals].pucString,
			sizeof( pDispVals[ *puiNumVals].pucString))))
		{
			goto Exit;
		}

		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].foreground = pRecEditor->isMonochrome() ? FLM_WHITE : FLM_YELLOW;
		pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + 1;
		(*puiNumVals)++;
		
		/*
		Output the encryption definiton
		*/
		if (pNd->ui32EncId)
		{
			
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
					"[%u]", (unsigned)pNd->ui32EncId);
			pDispVals[ *puiNumVals].uiCol = uiCol;
			pDispVals[ *puiNumVals].foreground = pRecEditor->isMonochrome() ? FLM_WHITE : FLM_WHITE;
			pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_RED;
			uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + 1;
			(*puiNumVals)++;
		}
		
	}
	else
	{
		/*
		Get the tree pointed to by the system node.
		*/

		if( RC_OK( rc = pRecEditor->getSystemNode( pNd, 0, 1, &pSystemNd)))
		{
			/*
			Display the node.
			*/

			switch( GedTagNum( pSystemNd))
			{
				case F_RECEDIT_COMMENT_FIELD:
				{	
					pRecEditor->getDisplayValue( pSystemNd, F_RECEDIT_DEFAULT_TYPE,
						pucValBuf, sizeof( pucValBuf));
					f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
						"# %s", pucValBuf);
					pDispVals[ *puiNumVals].uiCol += (GedNodeLevel( pNd) * 2);
					pDispVals[ *puiNumVals].foreground = pRecEditor->isMonochrome() ? FLM_WHITE : FLM_LIGHTGRAY;
					pDispVals[ *puiNumVals].background = pRecEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
					uiCol += f_strlen( pDispVals[ *puiNumVals].pucString) + 1;
					(*puiNumVals)++;
					break;
				}
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::refreshRow(
	FLMUINT				uiRow,
	NODE *				pNd,
	FLMBOOL				bSelected)
{
	FLMUINT				uiNumCols;
	FLMUINT				uiNumRows;
	FLMUINT				uiNumVals;
	FLMUINT				uiLoop;
	DBE_DISP_COLUMN	dispVals[ 16];
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	FTXWinGetCanvasSize( m_pEditWindow, &uiNumCols, &uiNumRows);
	FTXWinSetCursorPos( m_pEditWindow, 0, uiRow);
	FTXWinClearLine( m_pEditWindow, 0, uiRow);

	f_memset( dispVals, 0, sizeof( dispVals));
	uiNumVals = 0;

	// Call the display formatter

	if( m_pDisplayHook)
	{
		if( RC_BAD( rc = m_pDisplayHook( this, pNd,
			m_DisplayData, dispVals, &uiNumVals)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_RecEditorDefaultDispHook( this, pNd, 0,
			dispVals, &uiNumVals)))
		{
			goto Exit;
		}
	}

	for( uiLoop = 0; uiLoop < uiNumVals; uiLoop++)
	{
		FTXWinSetCursorPos( m_pEditWindow, dispVals[ uiLoop].uiCol, uiRow);
		FTXWinCPrintf( m_pEditWindow, dispVals[ uiLoop].background,
			dispVals[ uiLoop].foreground, "%s", dispVals[ uiLoop].pucString);
	}

	if( bSelected)
	{	
		eColorType background = m_bMonochrome ? FLM_LIGHTGRAY : FLM_CYAN;
		eColorType foreground = m_bMonochrome ? FLM_BLACK : FLM_WHITE;
		FTXWinPaintRow( m_pEditWindow, &background, &foreground, uiRow);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Repositions the cursor (and current node) to display at the top of
		the editor window
*****************************************************************************/
RCODE F_RecEditor::setCurrentAtTop( void)
{
	m_uiCurRow = 0;
	m_pScrFirstNd = m_pCurNd;
	return( FERR_OK);
}

/****************************************************************************
Desc: Repositions the cursor (and current node) to display at the bottom of
		the editor window
*****************************************************************************/
RCODE F_RecEditor::setCurrentAtBottom( void)
{
	NODE *		pTmpNd;
	FLMUINT		uiNumRows;
	FLMUINT		uiRowsRemaining;
	RCODE			rc = FERR_OK;

	flmAssert( m_pEditWindow != NULL);

	FTXWinGetCanvasSize( m_pEditWindow, NULL, &uiNumRows);
	uiNumRows--;

	pTmpNd = m_pCurNd;
	uiRowsRemaining = uiNumRows;
	while( uiRowsRemaining > 0)
	{
		if( (pTmpNd = getPrevNode( pTmpNd)) != NULL)
		{
			m_pScrFirstNd = pTmpNd;
		}
		else
		{
			break;
		}
		uiRowsRemaining--;
	}

	m_uiCurRow = uiNumRows - uiRowsRemaining;
	return( rc);
}

/****************************************************************************
Desc: Edits a node's value
*****************************************************************************/
RCODE F_RecEditor::editNode(
	FLMUINT		uiNdRow,
	NODE *		pNd)
{
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows;
	FLMUINT			uiNumWinCols;
	FLMUINT			uiValType = GedValType( pNd);
	FLMBOOL			bModified = FALSE;
	FTX_WINDOW *	pWindow = NULL;
	RCODE				rc = FERR_OK;
	eColorType		back;
	eColorType		fore;

	flmAssert( m_bSetupCalled == TRUE);

	FTXScreenGetSize( m_pScreen, &uiNumCols, &uiNumRows);
	uiNumWinCols = uiNumCols - 2;

	if( uiValType == FLM_BINARY_TYPE)
	{
		uiNumWinRows = uiNumRows / 2;
		if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumWinCols,
			uiNumWinRows, &pWindow)))
		{
			goto Exit;
		}

		FTXWinMove( pWindow, (FLMUINT)((uiNumCols - uiNumWinCols) / 2),
			(FLMUINT)((uiNumRows - uiNumWinRows) / 2));
	}
	else
	{
		uiNumWinRows = 3;
		if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumWinCols,
			uiNumWinRows, &pWindow)))
		{
			goto Exit;
		}

		FTXWinMove( pWindow, (FLMUINT)((uiNumCols - uiNumWinCols) / 2), uiNdRow);
	}

	FTXWinSetScroll( pWindow, FALSE);

	back = m_bMonochrome ? FLM_BLACK : FLM_GREEN;
	fore = FLM_WHITE;
	
	FTXWinSetBackFore( pWindow, back, fore);
	FTXWinDrawBorder( pWindow);

	switch( GedValType( pNd))
	{
		case FLM_TEXT_TYPE:
		{
			FTXWinSetTitle( pWindow, " TEXT ", back, fore);

			if( RC_BAD( rc = editTextNode( pWindow, pNd, &bModified)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_NUMBER_TYPE:
		{
			FTXWinSetTitle( pWindow, " NUMBER ", back, fore);

			if( RC_BAD( rc = editNumberNode( pWindow, pNd, &bModified)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_CONTEXT_TYPE:
		{
			FTXWinSetTitle( pWindow, " CONTEXT ", back, fore);

			if( RC_BAD( rc = editContextNode( pWindow, pNd, &bModified)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_BINARY_TYPE:
		{
			FTXWinSetTitle( pWindow, " BINARY ", back, fore);

			if( RC_BAD( rc = editBinaryNode( pWindow, pNd, &bModified)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			displayMessage( "This field cannot be edited",
				rc, NULL, FLM_RED, FLM_WHITE);
			break;
		}
	}

	if( bModified)
	{
		FLMUINT	uiFlags;

		(void)getControlFlags( pNd, &uiFlags);
		if( !(uiFlags & F_RECEDIT_FLAG_FLDMOD))
		{
			(void)setControlFlags( pNd, uiFlags | F_RECEDIT_FLAG_FLDMOD);
			(void)markRecordModified( pNd);
		}
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}
	
/****************************************************************************
Desc: Edit text node as text
*****************************************************************************/
RCODE F_RecEditor::editTextNode(
	FTX_WINDOW *		pWindow,
	NODE *				pNd,
	FLMBOOL *			pbModified)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTextSize;
	FLMUINT			uiTermChar;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUNICODE *	puzUniStr;
#define F_RECEDIT_MAX_UNI_CHARS		5000
	void *			pvMark = NULL;
	F_Pool			tmpPool;

	flmAssert( m_bSetupCalled == TRUE);

	tmpPool.poolInit( 512);
	if( RC_BAD( rc = tmpPool.poolAlloc( 
		F_RECEDIT_MAX_UNI_CHARS * sizeof( FLMUNICODE),
		(void **)&puzUniStr)))
	{
		goto Exit;
	}
	
	pvMark = tmpPool.poolMark();

	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);

	// VISIT: Allow in-line editing (eliminate the need for a separate window)

	uiTextSize = F_RECEDIT_MAX_UNI_CHARS;
	if( RC_BAD( rc = GedGetUNICODE( pNd, puzUniStr, &uiTextSize)))
	{
		goto Exit;
	}

	for( ;;)
	{
		FTXWinSetCursorPos( pWindow, 0, 0);
		FTXWinClearLine( pWindow, 0, 0);

		if( RC_BAD( rc = UCToAsciiUCMix( puzUniStr,
			m_pucTmpBuf, F_RECEDIT_BUF_SIZE)))
		{
			goto Exit;
		}

		FTXWinOpen( pWindow);

		if( RC_OK( FTXLineEdit( pWindow, m_pucTmpBuf, F_RECEDIT_BUF_SIZE, 
			uiNumCols, &uiTextSize, &uiTermChar)))
		{
			if( uiTermChar == FKB_ESCAPE)
			{
				break;
			}
		
			if( uiTermChar == FKB_ENTER)
			{
				if( *m_pucTmpBuf == 0)
				{
					pNd->ui32Length = 0;
					pNd->value = 0;
				}
				else
				{
					if( RC_BAD( rc = asciiUCMixToUC( m_pucTmpBuf, 
						puzUniStr, F_RECEDIT_MAX_UNI_CHARS)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = GedPutUNICODE( &m_treePool, pNd, puzUniStr)))
					{
						goto Exit;
					}
				}
				*pbModified = TRUE;
				break;
			}
		}
		else
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Edits a numeric node as using the text editor
*****************************************************************************/
RCODE F_RecEditor::editNumberNode(
	FTX_WINDOW *		pWindow,
	NODE *				pNd,
	FLMBOOL *			pbModified)
{
	FLMUINT		uiTextSize;
	FLMUINT		uiTermChar;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;
	FLMUINT		uiVal;
	FLMINT		iVal;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);

	uiTextSize = F_RECEDIT_BUF_SIZE;
	if( RC_BAD( rc = GedGetNATIVE( pNd, m_pucTmpBuf, &uiTextSize)))
	{
		goto Exit;
	}

	FTXWinOpen( pWindow);

	for( ;;)
	{
		FTXWinSetCursorPos( pWindow, 0, 0);
		FTXWinClearLine( pWindow, 0, 0);
		if( RC_OK( FTXLineEdit( pWindow, m_pucTmpBuf, F_RECEDIT_BUF_SIZE, 
			uiNumCols, &uiTextSize, &uiTermChar)))
		{
			if( uiTermChar == FKB_ESCAPE)
			{
				break;
			}
		
			if( uiTermChar == FKB_ENTER)
			{
				if( RC_BAD( rc = getNumber( m_pucTmpBuf, &uiVal, &iVal)))
				{
					goto Exit;
				}

				if( iVal)
				{
					if( RC_BAD( rc = GedPutINT( &m_treePool, pNd, iVal)))
					{
						goto Exit;
					}
				}
				else
				{
					if( RC_BAD( rc = GedPutUINT( &m_treePool, pNd, uiVal)))
					{
						goto Exit;
					}
				}

				*pbModified = TRUE;
				break;
			}
		}
		else
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Edits a context node as using the text editor
*****************************************************************************/
RCODE F_RecEditor::editContextNode(
	FTX_WINDOW *		pWindow,
	NODE *				pNd,
	FLMBOOL *			pbModified)
{
	FLMUINT		uiTextSize;
	FLMUINT		uiTermChar;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;
	FLMUINT		uiRecPtr;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);

	uiTextSize = F_RECEDIT_BUF_SIZE;
	if( RC_BAD( rc = GedGetRecPtr( pNd, &uiRecPtr)) ||
		(uiRecPtr == 0xFFFFFFFF && GedValLen( pNd) == 0))
	{
		m_pucTmpBuf[ 0] = '\0';
	}
	else
	{
		f_sprintf( (char *)m_pucTmpBuf, "%u", (unsigned)uiRecPtr);
	}
	
	FTXWinOpen( pWindow);

	for( ;;)
	{
		FTXWinSetCursorPos( pWindow, 0, 0);
		FTXWinClearLine( pWindow, 0, 0);
		if( RC_OK( FTXLineEdit( pWindow, m_pucTmpBuf, F_RECEDIT_BUF_SIZE, 
			uiNumCols, &uiTextSize, &uiTermChar)))
		{
			if( uiTermChar == FKB_ESCAPE)
			{
				break;
			}
		
			if( uiTermChar == FKB_ENTER)
			{
				if( *m_pucTmpBuf)
				{
					if( RC_BAD( rc = getNumber( m_pucTmpBuf, &uiRecPtr, NULL)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = GedPutRecPtr( &m_treePool, pNd, uiRecPtr)))
					{
						goto Exit;
					}
				}
				else
				{
					pNd->ui32Length = 0;
					pNd->value = 0;
				}

				*pbModified = TRUE;
				break;
			}
		}
		else
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Edits a binary node
*****************************************************************************/
RCODE F_RecEditor::editBinaryNode(
	FTX_WINDOW *		pWindow,
	NODE *				pNd,
	FLMBOOL *			pbModified)
{
	FLMUINT			uiLoop;
	FLMUINT			uiTermChar;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMBYTE *		pucTmpPtr;
	FLMBYTE *		pucCurPtr;
	FLMBYTE *		pucMaxPtr = NULL;
	FLMBYTE *		pucMinPtr = NULL;
	FLMBYTE *		pucScrPtr = NULL;
	FLMBYTE *		pucRowPtr = NULL;
	eColorType		winBackColor;
	eColorType		winForeColor;
	FLMUINT			uiItemsPerRow;
	FLMUINT			uiValRow = 0;
	FLMUINT			uiValCol = 0;
	void *			pPoolMark = m_scratchPool.poolMark();
	FLMBOOL			bRefreshWindow = TRUE;
	FLMBOOL			bDoneEditing = FALSE;
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( pbModified)
	{
		*pbModified = FALSE;
	}

	if( GedValLen( pNd) == 0)
	{
		displayMessage(
			"This field is empty",
			FERR_OK, NULL, FLM_GREEN, FLM_WHITE);
		goto Exit;
	}

	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);
	FTXWinGetBackFore( pWindow, &winBackColor, &winForeColor);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_INVISIBLE);
	FTXWinSetCursorPos( pWindow, 0, 0);
	FTXWinClear( pWindow);
	FTXWinOpen( pWindow);
	
	if( RC_BAD( rc = m_scratchPool.poolAlloc( GedValLen( pNd), 
		(void **)&pucMinPtr)))
	{
		goto Exit;
	}

	f_memcpy( pucMinPtr, GedValPtr( pNd), GedValLen( pNd));
	pucCurPtr = pucMinPtr;
	pucMaxPtr = pucMinPtr + GedValLen( pNd);
	pucScrPtr = pucMinPtr;
	uiItemsPerRow = (((uiNumCols - 2) / 4) / 8) * 8;
	while( !bDoneEditing && !isExiting())
	{
		if( bRefreshWindow)
		{
			FLMUINT		uiTmpRow = 0;
			FLMUINT		uiItemCount = 0;
			FLMUINT		uiMaxRow = uiNumRows - 1;
			eColorType	backColor;
			eColorType	foreColor;
#define	FEDIT_BINROW_OVHD		6
			FLMUINT		uiOverhead = FEDIT_BINROW_OVHD;

			if( (pucScrPtr + (uiItemsPerRow * uiNumRows)) <= pucCurPtr)
			{
				pucScrPtr += uiItemsPerRow;
			}
			else if( pucScrPtr > pucCurPtr)
			{
				pucScrPtr -= uiItemsPerRow;
			}
			
			pucTmpPtr = pucScrPtr;
			pucRowPtr = pucScrPtr;
			for( ;;)
			{
				if( uiItemCount && (uiItemCount % 8) == 0)
				{
					uiOverhead += 2;
				}

				if( pucTmpPtr == pucCurPtr)
				{
					if( m_bMonochrome)
					{
						backColor = FLM_LIGHTGRAY;
						foreColor = FLM_BLACK;
					}
					else
					{
						backColor = FLM_RED;
						foreColor = FLM_WHITE;
					}
					uiValRow = uiTmpRow;
					uiValCol = (FLMUINT)(uiItemCount * 3) + uiOverhead;
				}
				else
				{
					backColor = winBackColor;
					foreColor = winForeColor;
				}

				if( !uiItemCount && (pucTmpPtr < pucMaxPtr))
				{
					FTXWinSetCursorPos( pWindow, 0, uiTmpRow);
					FTXWinCPrintf( pWindow, m_bMonochrome ? FLM_BLACK : FLM_GREEN,
						m_bMonochrome ? FLM_WHITE : FLM_RED,
						"%4.4X", (unsigned)(pucTmpPtr - pucMinPtr));
				}

				FTXWinSetCursorPos( pWindow,
					(FLMUINT)(uiItemCount * 3) + uiOverhead, uiTmpRow);

				if( pucTmpPtr < pucMaxPtr)
				{
					FTXWinCPrintf( pWindow, backColor, foreColor,
						"%2.2X", (unsigned)(*pucTmpPtr));
				}
				else
				{
					FTXWinCPrintf( pWindow, backColor, foreColor,
						"  ", (unsigned)(*pucTmpPtr));
				}

				FTXWinSetCursorPos( pWindow,
					(FLMUINT)(uiItemCount + (uiItemsPerRow * 3) +
						(2 * (uiItemsPerRow / 8)) + FEDIT_BINROW_OVHD) + 
						(uiItemCount / 8), uiTmpRow);

				if( pucTmpPtr < pucMaxPtr)
				{
					if( *pucTmpPtr >= 32 && *pucTmpPtr <= 126)
					{
						FTXWinCPrintf( pWindow, backColor, foreColor,
							"%c", (char)(*pucTmpPtr));
					}
					else
					{
						FTXWinCPrintf( pWindow, backColor, foreColor, ".");
					}

					if( (pucRowPtr + uiItemsPerRow) <= pucCurPtr)
					{
						pucRowPtr = pucScrPtr + (uiItemsPerRow * uiTmpRow);
					}
				}
				else
				{
					FTXWinCPrintf( pWindow, backColor, foreColor, " ");
				}

				uiItemCount++;
				if( uiItemCount >= uiItemsPerRow)
				{
					FTXWinSetCursorPos( pWindow,
						(FLMUINT)(uiItemsPerRow * 3) + uiOverhead, uiTmpRow);
					FTXWinCPrintf( pWindow, winBackColor, winForeColor, "|");

					uiOverhead = FEDIT_BINROW_OVHD;
					uiTmpRow++;
					if( uiTmpRow > uiMaxRow)
					{
						break;
					}
					uiItemCount = 0;
				}

				if( pucTmpPtr < pucMaxPtr)
				{
					pucTmpPtr++;
				}
			}

			FTXWinSetCursorPos( pWindow, uiValCol, uiValRow);
			bRefreshWindow = FALSE;
		}

		if( RC_OK( FTXWinTestKB( pWindow)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pWindow, &uiChar);
			switch( uiChar)
			{
				case FKB_RIGHT:
				{
					if( (pucCurPtr + 1) < pucMaxPtr)
					{
						pucCurPtr++;
						bRefreshWindow = TRUE;
					}
					break;
				}

				case FKB_LEFT:
				{
					if( pucCurPtr > pucMinPtr)
					{
						pucCurPtr--;
						bRefreshWindow = TRUE;
					}
					break;
				}

				case FKB_DOWN:
				{
					if( (pucCurPtr + uiItemsPerRow) < pucMaxPtr)
					{
						pucCurPtr += uiItemsPerRow;
					}
					else
					{
						if( ((pucCurPtr + uiItemsPerRow) >= pucMaxPtr) &&
							(pucRowPtr + uiItemsPerRow) < pucMaxPtr)
						{
							pucCurPtr = pucMaxPtr - 1;
						}
					}
					bRefreshWindow = TRUE;
					break;
				}

				case FKB_UP:
				{
					if( (pucMinPtr + uiItemsPerRow) <= pucCurPtr)
					{
						pucCurPtr -= uiItemsPerRow;
						bRefreshWindow = TRUE;
					}
					break;
				}

				case FKB_HOME:
				{
					pucCurPtr = pucMinPtr;
					pucScrPtr = pucMinPtr;
					bRefreshWindow = TRUE;
					break;
				}

				case '+':
				{
					(*pucCurPtr)++;
					bRefreshWindow = TRUE;
					break;
				}

				case '-':
				{
					(*pucCurPtr)--;
					bRefreshWindow = TRUE;
					break;
				}

				case FKB_ENTER:
				{
					FLMUINT		uiTextSize;
					FLMBYTE		pucHexBuf[ 16];

					f_sprintf( (char *)pucHexBuf, "%2.2X", (unsigned)(*pucCurPtr));
					FTXWinSetCursorPos( pWindow, uiValCol, uiValRow);
					FTXWinSetCursorType( pWindow, FLM_CURSOR_UNDERLINE);

					if( RC_OK( FTXLineEdit( pWindow, (char *)pucHexBuf, 3, 3,
						&uiTextSize, &uiTermChar)))
					{
						if( uiTermChar == FKB_ENTER)
						{
							FLMBYTE		ucVal = 0;

							for( uiLoop = 0; uiLoop < 2; uiLoop++)
							{
								if( pucHexBuf[ uiLoop] >= '0' &&
									pucHexBuf[ uiLoop] <= '9')
								{
									ucVal |= (FLMBYTE)((pucHexBuf[ uiLoop] - '0') <<
										((1 - uiLoop) * 4));
								}
								else if( pucHexBuf[ uiLoop] >= 'a' &&
									pucHexBuf[ uiLoop] <= 'f')
								{
									ucVal |= (FLMBYTE)(((pucHexBuf[ uiLoop] - 'a') + 10) <<
										((1 - uiLoop) * 4));
								}
								else if( pucHexBuf[ uiLoop] >= 'A' &&
									pucHexBuf[ uiLoop] <= 'F')
								{
									ucVal |= (FLMBYTE)(((pucHexBuf[ uiLoop] - 'A') + 10) <<
										((1 - uiLoop) * 4));
								}
								else
								{
									ucVal = *pucCurPtr;
									break;
								}
							}
							*pucCurPtr = ucVal;
						}
					}
					else
					{
						rc = RC_SET( FERR_FAILURE);
						goto Exit;
					}
					FTXWinSetCursorType( pWindow, FLM_CURSOR_INVISIBLE);
					bRefreshWindow = TRUE;
					break;
				}

				case FKB_ESCAPE:
				{
					char	pucResponse[ 2];

					if( f_memcmp( GedValPtr( pNd), pucMinPtr, GedValLen( pNd)) != 0)
					{
						*pucResponse = '\0';
						requestInput(
							"Update field value (Y/N)",
							pucResponse, 2, &uiTermChar);
				
						if( uiTermChar == FKB_ESCAPE)
						{
							break;
						}
						
						if( *pucResponse == 'y' || *pucResponse == 'Y')
						{
							f_memcpy( GedValPtr( pNd), pucMinPtr, GedValLen( pNd));
							if( pbModified)
							{
								*pbModified = TRUE;
							}
						}
					}
					bDoneEditing = TRUE;
					break;
				}
			}
		}
		f_sleep( 1);
	}

Exit:

	m_scratchPool.poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::displayMessage(
	const char *		pucMessage,
	RCODE					rcOfMessage,
	FLMUINT *			puiTermChar,
	eColorType			background,
	eColorType			foreground)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	FTXDisplayMessage( m_pScreen, m_bMonochrome ? FLM_LIGHTGRAY : background,
		m_bMonochrome ? FLM_BLACK : foreground,
		pucMessage, FlmErrorString( rcOfMessage), puiTermChar);

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::openNewDb( void)
{
	RCODE		rc = FERR_OK;
	char		szResponse [100];
	FLMUINT	uiChar;

	szResponse [0] = 0;
	m_hDefaultDb = HFDB_NULL;
	for (;;)
	{
		if (RC_BAD( rc = requestInput( "Enter name of DB to open",
									szResponse, sizeof( szResponse), &uiChar)))
		{
			goto Exit;
		}

		if (uiChar == FKB_ESCAPE)
		{
			break;
		}
		if (RC_BAD( rc = FlmDbOpen( szResponse, NULL, NULL, // VISIT
			0, NULL, &m_hDefaultDb)))
		{
			displayMessage( "Unable to open database", rc,
					NULL, FLM_RED, FLM_WHITE);
			m_hDefaultDb = HFDB_NULL;
			continue;
		}
		break;
	}
	if (m_hDefaultDb != HFDB_NULL)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::requestInput(
	const char *		pucMessage,
	char *				pucResponse,
	FLMUINT				uiMaxRespLen,
	FLMUINT *			puiTermChar)
{
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 3;
	FLMUINT			uiNumWinCols;
	FTX_WINDOW *	pWindow = NULL;
	IF_FileHdl *	pFileHdl = NULL;
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	FTXScreenGetSize( m_pScreen, &uiNumCols, &uiNumRows);
	uiNumWinCols = uiNumCols - 8;

	if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumWinCols,
		uiNumWinRows, &pWindow)))
	{
		goto Exit;
	}

	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_UNDERLINE);
	FTXWinSetBackFore( pWindow, m_bMonochrome ? FLM_BLACK : FLM_CYAN, FLM_WHITE);
	FTXWinClear( pWindow);
	FTXWinDrawBorder( pWindow);
	FTXWinMove( pWindow, (uiNumCols - uiNumWinCols) / 2,
		(uiNumRows - uiNumWinRows) / 2);
	FTXWinOpen( pWindow);

	for( ;;)
	{
		FTXWinClear( pWindow);
		FTXWinPrintf( pWindow, "%s: ", pucMessage);

		if( RC_BAD( rc = FTXLineEdit( pWindow, pucResponse, 
			uiMaxRespLen, uiMaxRespLen, NULL, puiTermChar)))
		{
			goto Exit;
		}

		if( *puiTermChar == FKB_F1)
		{
			FLMUINT		uiBytesRead;
			char *		pucTmp;

			if( RC_BAD( rc = m_pFileSystem->openFile( pucResponse, FLM_IO_RDONLY,
				&pFileHdl)))
			{
				displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
				continue;
			}

			if( RC_BAD( rc = pFileHdl->read( 0, uiMaxRespLen,
				pucResponse, &uiBytesRead)))
			{
				if( rc == FERR_IO_END_OF_FILE)
				{
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}

			pFileHdl->Release();
			pFileHdl = NULL;
			pucResponse[ uiBytesRead] = '\0';

			if( (pucTmp = f_strchr( 
					(const char *)pucResponse, '\r')) != NULL)
			{
				*pucTmp = '\0';
			}

			if( (pucTmp = f_strchr( 
					(const char *)pucResponse, '\n')) != NULL)
			{
				*pucTmp = '\0';
			}
		}
		else
		{
			break;
		}
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::createSystemNode(
	NODE *		pCurNd,
	FLMUINT		uiTagNum,
	NODE **		ppSystemNd)
{
	NODE *		pTmpNd;
	NODE *		pInfoNd;
	FLMUINT		uiNdAddr;
	FLMUINT		uiSizeofUint = sizeof( FLMUINT);
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( (pTmpNd = GedNodeMake( &m_treePool, 0, &rc)) != NULL)
	{
		if( (pInfoNd = GedNodeMake( &m_treePool, uiTagNum, &rc)) != NULL)
		{
			uiNdAddr = (FLMUINT)pInfoNd;

			if( uiSizeofUint == 4)
			{
				if( RC_BAD( rc = GedPutUINT( &m_treePool, pTmpNd, uiNdAddr)))
				{
					goto Exit;
				}
				GedChildGraft( pCurNd, pTmpNd, GED_FIRST);
			}
			else if( uiSizeofUint == 8)
			{
				if( RC_BAD( rc = GedPutBINARY( &m_treePool, pTmpNd,
					(void *)&uiNdAddr, 8)))
				{
					goto Exit;
				} 

				GedChildGraft( pCurNd, pTmpNd, GED_FIRST);
			}
			else
			{
				flmAssert( 0);
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		goto Exit;
	}

	if( ppSystemNd)
	{
		*ppSystemNd = pInfoNd;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::getSystemNode(
	NODE *		pCurNd,
	FLMUINT		uiTagNum,
	FLMUINT		uiNth,
	NODE **		ppSystemNd)
{
	NODE *		pTmpNd = pCurNd;
	NODE *		pInfoNd = NULL;
	FLMUINT32	ui32NdAddr;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !uiNth)
	{
		goto Exit;
	}

	while( pTmpNd)
	{
		if( (GedNodeLevel( pTmpNd) == GedNodeLevel( pCurNd) + 1 ||
				GedNodeLevel( pTmpNd) == GedNodeLevel( pCurNd)) &&
			GedTagNum( pTmpNd) == F_RECEDIT_SYSTEM_FIELD)
		{
#if defined( FLM_UNIX) || defined( FLM_64BIT)
			if( sizeof( FLMUINT) == 4)
#endif
			{
				if( RC_BAD( rc = GedGetUINT32( pTmpNd, &ui32NdAddr)))
				{
					goto Exit;
				}

				pInfoNd = (NODE *)((FLMUINT)ui32NdAddr);
			}
#if defined( FLM_UNIX) || defined( FLM_64BIT)
			else if( sizeof( FLMUINT) == 8)
			{
				FLMUINT		uiLen = 8;
				if( RC_BAD( rc = GedGetBINARY( pTmpNd, (void *)&pInfoNd, &uiLen)))
				{
					goto Exit;
				}
			}
#endif

			if( !uiTagNum || GedTagNum( pInfoNd) == uiTagNum)
			{
				--uiNth;
				if( !uiNth)
				{
					break;
				}
			}
			pInfoNd = NULL;
		}
		pTmpNd = pTmpNd->next;
		pInfoNd = NULL;

		if( pTmpNd && GedNodeLevel( pTmpNd) <= GedNodeLevel( pCurNd))
		{
			break;
		}
	}

	if( !pInfoNd)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if( ppSystemNd)
	{
		*ppSystemNd = pInfoNd;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::getControlNode(
	NODE *		pCurNd,
	FLMBOOL		bCreate,
	NODE **		ppControlNd)
{
	RCODE			rc = FERR_OK;
	NODE *		pControlNd = NULL;
	
	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = getSystemNode( pCurNd,
		F_RECEDIT_CONTROL_INFO_FIELD, 1, &pControlNd)))
	{
		if( rc == FERR_NOT_FOUND && bCreate)
		{
			/*
			Create the control node.
			*/
			
			if( RC_BAD( rc = createSystemNode( pCurNd,
				F_RECEDIT_CONTROL_INFO_FIELD, &pControlNd)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

	if( ppControlNd)
	{
		*ppControlNd = pControlNd;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::getControlFlags(
	NODE *		pCurNd,
	FLMUINT *	puiFlags)
{
	RCODE			rc = FERR_OK;
	NODE *		pControlNd = NULL;
	NODE *		pTmpNd;
	FLMUINT32	ui32Tmp;

	flmAssert( m_bSetupCalled == TRUE);

	*puiFlags = 0;
	
	if( RC_BAD( rc = getControlNode( pCurNd, FALSE, &pControlNd)))
	{
		goto Exit;
	}

	if( (pTmpNd = GedFind( GED_TREE, pControlNd,
		F_RECEDIT_FLAGS_FIELD, 1)) != NULL)
	{
		if( RC_BAD( rc = GedGetUINT32( pTmpNd, &ui32Tmp)))
		{
			goto Exit;
		}
		*puiFlags = ui32Tmp;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::setControlFlags(
	NODE *		pCurNd,
	FLMUINT 		uiFlags)
{
	RCODE			rc = FERR_OK;
	NODE *		pControlNd = NULL;
	NODE *		pTmpNd;
	
	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = getControlNode( pCurNd, TRUE, &pControlNd)))
	{
		goto Exit;
	}

	if( (pTmpNd = GedFind( GED_TREE, pControlNd,
		F_RECEDIT_FLAGS_FIELD, 1)) == NULL)
	{
		if( (pTmpNd = GedNodeMake( &m_treePool,
			F_RECEDIT_FLAGS_FIELD, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		GedChildGraft( pControlNd, pTmpNd, GED_FIRST);
	}

	if( RC_BAD( rc = GedPutUINT( &m_treePool, pTmpNd, uiFlags)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::addAltView(
	NODE *		pCurNd,
	FLMUINT 		uiViewType)
{
	RCODE			rc = FERR_OK;
	NODE *		pViewNd = NULL;
	NODE *		pTmpNd;
	
	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = createSystemNode( pCurNd,
		F_RECEDIT_VAL_VIEW_FIELD, &pViewNd)))
	{
		goto Exit;
	}

	if( (pTmpNd = GedNodeMake( &m_treePool,
		F_RECEDIT_VIEWTYPE_FIELD, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutUINT( &m_treePool,
		pTmpNd, uiViewType)))
	{
		goto Exit;
	}

	GedChildGraft( pViewNd, pTmpNd, GED_FIRST);

	if( (pTmpNd = GedNodeMake( &m_treePool,
		F_RECEDIT_REFNODE_FIELD, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// VISIT
	if( RC_BAD( rc = GedPutUINT( &m_treePool,
		pTmpNd, (FLMUINT)pCurNd)))
	{
		goto Exit;
	}

	GedChildGraft( pViewNd, pTmpNd, GED_FIRST);

Exit:

	return( rc);
}

/****************************************************************************
Desc: Adds a comment line subordinate to the current node
*****************************************************************************/
RCODE F_RecEditor::addComment(
	NODE *			pCurNd,
	FLMBOOL			bVisible,
	const char *	pucFormat, ...)
{
	char			pucBuffer[ 512];
	NODE *		pCommentNd = NULL;
	NODE *		pTmpNd = NULL;
	RCODE			rc = FERR_OK;
	f_va_list	args;
	
	flmAssert( m_bSetupCalled == TRUE);

	f_va_start( args, pucFormat);
	f_vsprintf( pucBuffer, pucFormat, &args);
	f_va_end( args);

	if( RC_BAD( rc = createSystemNode( pCurNd,
		F_RECEDIT_COMMENT_FIELD, &pCommentNd)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( &m_treePool, pCommentNd, pucBuffer)))
	{
		goto Exit;
	}

	if( bVisible)
	{
		if( (pTmpNd = GedNodeMake( &m_treePool,
			F_RECEDIT_VISIBLE_FIELD, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		GedChildGraft( pCommentNd, pTmpNd, GED_FIRST);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Adds an application-specific annotation.  Annotations are
		ignored by the editor, but can be used by the application to
		change the way values are displayed, etc.
*****************************************************************************/
RCODE F_RecEditor::addAnnotation(
	NODE *			pCurNd,
	const char *	pucFormat, ...)
{
	char			pucBuffer[ 512];
	NODE *		pAnnoNd = NULL;
	RCODE			rc = FERR_OK;
	f_va_list	args;
	
	flmAssert( m_bSetupCalled == TRUE);

	f_va_start( args, pucFormat);
	f_vsprintf( (char *)pucBuffer, pucFormat, &args);
	f_va_end( args);

	if( RC_BAD( rc = createSystemNode( pCurNd,
		F_RECEDIT_VALANNO_FIELD, &pAnnoNd)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( &m_treePool, pAnnoNd, pucBuffer)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::setLinkDestination(
	NODE *			pCurNd,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn)
{
	NODE *		pLinkNd = NULL;
	NODE *		pTmpNd = NULL;
	RCODE			rc = FERR_OK;
	
	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = createSystemNode( pCurNd,
		F_RECEDIT_LINK_DEST_FIELD, &pLinkNd)))
	{
		goto Exit;
	}

	if( (pTmpNd = GedNodeMake( &m_treePool,
		FLM_CONTAINER_TAG, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutUINT( &m_treePool, pTmpNd, uiContainer)))
	{
		goto Exit;
	}

	GedChildGraft( pLinkNd, pTmpNd, GED_LAST);

	if( (pTmpNd = GedNodeMake( &m_treePool,
		FLM_PARENT_TAG, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutUINT( &m_treePool, pTmpNd, uiDrn)))
	{
		goto Exit;
	}

	GedChildGraft( pLinkNd, pTmpNd, GED_LAST);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::getLinkDestination(
	NODE *			pCurNd,
	FLMUINT *		puiContainer,
	FLMUINT *		puiDrn)
{
	NODE *		pLinkNd = NULL;
	NODE *		pTmpNd = NULL;
	FLMUINT32	ui32Tmp;
	FLMUINT		uiDestContainer = 0;
	FLMUINT		uiDestDrn = 0;
	RCODE			rc = FERR_OK;
	
	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = getSystemNode( pCurNd,
		F_RECEDIT_LINK_DEST_FIELD, 1, &pLinkNd)))
	{
		goto Exit;
	}

	if( (pTmpNd = GedFind( 1, pLinkNd, FLM_CONTAINER_TAG, 1)) != NULL)
	{
		if( RC_BAD( rc = GedGetUINT32( pTmpNd, &ui32Tmp)))
		{
			goto Exit;
		}
		uiDestContainer = ui32Tmp;
	}
	else if( puiContainer)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if( (pTmpNd = GedFind( 1, pLinkNd, FLM_PARENT_TAG, 1)) != NULL)
	{
		if( RC_BAD( rc = GedGetUINT32( pTmpNd, &ui32Tmp)))
		{
			goto Exit;
		}
		uiDestDrn = ui32Tmp;
	}
	else if( puiDrn)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	if( puiContainer)
	{
		*puiContainer = uiDestContainer;
	}

	if( puiDrn)
	{
		*puiDrn = uiDestDrn;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
NODE * F_RecEditor::getPrevNode(
	NODE *		pCurNd,
	FLMBOOL		bUseCallback)
{
	NODE *				pPrevNd = NULL;
	DBE_NODE_INFO		nodeInfo;

	flmAssert( m_bSetupCalled == TRUE);

	for( ;;)
	{
		if( pCurNd)
		{
			pPrevNd = pCurNd->prior;
		}

		if( bUseCallback && m_pEventHook)
		{
			f_memset( &nodeInfo, 0, sizeof( DBE_NODE_INFO));
			nodeInfo.pCurNd = pCurNd;

			if( RC_OK( m_pEventHook( this, F_RECEDIT_EVENT_GETPREVNODE,
				(void *)(&nodeInfo), m_EventData)))
			{
				if( nodeInfo.bUseNd)
				{
					pPrevNd = nodeInfo.pNd;
				}
			}
		}

		if( !pPrevNd)
		{
			break;
		}

		if( isNodeVisible( pPrevNd))
		{
			break;
		}
		pCurNd = pPrevNd;
	}

	return( pPrevNd);
}

/****************************************************************************
Desc:
*****************************************************************************/
NODE * F_RecEditor::getNextNode(
	NODE *		pCurNd,
	FLMBOOL		bUseCallback)
{
	NODE *				pNextNd = NULL;
	DBE_NODE_INFO		nodeInfo;

	flmAssert( m_bSetupCalled == TRUE);

	for( ;;)
	{
		if( pCurNd)
		{
			pNextNd = pCurNd->next;
		}

		if( bUseCallback && m_pEventHook)
		{
			f_memset( &nodeInfo, 0, sizeof( DBE_NODE_INFO));
			nodeInfo.pCurNd = pCurNd;

			if( RC_OK( m_pEventHook( this, F_RECEDIT_EVENT_GETNEXTNODE,
				(void *)(&nodeInfo), m_EventData)))
			{
				if( nodeInfo.bUseNd)
				{
					pNextNd = nodeInfo.pNd;
				}
			}
		}

		if( !pNextNd)
		{
			break;
		}

		if( isNodeVisible( pNextNd))
		{
			break;
		}
		pCurNd = pNextNd;
	}

	return( pNextNd);
}

/****************************************************************************
Desc: Returns a record's root node (level 0)
*****************************************************************************/
NODE * F_RecEditor::getRootNode(
	NODE *		pCurNd)
{
	NODE *		pRootNd = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		goto Exit;
	}

	pRootNd = pCurNd;
	while( pRootNd && GedNodeLevel( pRootNd) != 0)
	{
		pRootNd = getPrevNode( pRootNd);
	}

Exit:

	return( pRootNd);
}

/****************************************************************************
Desc: Returns a node's first non-system child node
*****************************************************************************/
NODE * F_RecEditor::getChildNode(
	NODE *		pCurNd)
{
	NODE *		pChildNd = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		goto Exit;
	}

	pChildNd = getNextNode( pCurNd);
	if( pChildNd && (GedNodeLevel( pChildNd) != GedNodeLevel( pCurNd) + 1))
	{
		pChildNd = NULL;
	}

Exit:

	return( pChildNd);
}

/****************************************************************************
Desc:
*****************************************************************************/
NODE * F_RecEditor::getNextRecord(
	NODE *		pCurNd)
{
	NODE *		pNextNd = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		goto Exit;
	}

	pNextNd = getNextNode( pCurNd);
	while( pNextNd)
	{
		if( GedNodeLevel( pNextNd) == 0)
		{
			break;
		}
		pNextNd = getNextNode( pNextNd);
	}

Exit:

	return( pNextNd);
}

/****************************************************************************
Desc:
*****************************************************************************/
NODE * F_RecEditor::getPrevRecord(
	NODE *		pCurNd)
{
	NODE *		pPrevNd = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		goto Exit;
	}

	pPrevNd = getPrevNode( getRootNode( pCurNd));
	while( pPrevNd)
	{
		if( GedNodeLevel( pPrevNd) == 0)
		{
			break;
		}
		pPrevNd = getPrevNode( pPrevNd);
	}

Exit:

	return( pPrevNd);
}

/****************************************************************************
Desc: 
*****************************************************************************/
NODE * F_RecEditor::findRecord(
	FLMUINT		uiContainer,
	FLMUINT		uiDrn,
	NODE *		pStartNd)
{
	FLMUINT		uiSourceCont;
	FLMUINT		uiSourceDrn;
	NODE *		pTmpNd;
	NODE *		pCurNd = m_pTree;
	FLMBOOL		bForward = TRUE;

	flmAssert( m_bSetupCalled == TRUE);

	if( pStartNd)
	{
		pCurNd = getRootNode( pStartNd);
	}
	else if( m_pCurNd)
	{
		pTmpNd = getRootNode( m_pCurNd);
		if( RC_OK( GedGetRecSource( pTmpNd, NULL,
			&uiSourceCont, &uiSourceDrn)))
		{
			pCurNd = pTmpNd;
			if( uiSourceCont > uiContainer ||
				(uiSourceCont == uiContainer && uiSourceDrn > uiDrn))
			{
				pCurNd = getPrevRecord( pCurNd);
				bForward = FALSE;
			}

		}
	}

	while( pCurNd)
	{
		if( RC_OK( GedGetRecSource( pCurNd, NULL,
			&uiSourceCont, &uiSourceDrn)))
		{
			if( uiSourceCont == uiContainer && uiSourceDrn == uiDrn)
			{
				break;
			}

			if( bForward)
			{
				if( uiSourceCont > uiContainer ||
					(uiSourceCont == uiContainer && uiSourceDrn > uiDrn))
				{
					pCurNd = NULL;
					break;
				}
			}
			else
			{
				if( uiSourceCont < uiContainer ||
					(uiSourceCont == uiContainer && uiSourceDrn < uiDrn))
				{
					pCurNd = NULL;
					break;
				}
			}
		}

		if( bForward)
		{
			pCurNd = getNextRecord( pCurNd);
		}
		else
		{
			pCurNd = getPrevRecord( pCurNd);
		}

		/*
		Release CPU to prevent CPU hog
		*/

		f_yieldCPU();
	}

	return( pCurNd);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::isNodeVisible(
	NODE *		pCurNd)
{
	NODE *		pInfoNd;
	NODE *		pInvNode;
	FLMUINT		uiInvCnt;
	FLMUINT32	ui32NdAddr;
	FLMBOOL		bVisible = FALSE;

	flmAssert( m_bSetupCalled == TRUE);

	if( isSystemNode( pCurNd))
	{
#if defined( FLM_UNIX) || defined( FLM_64BIT)
		if( sizeof( FLMUINT) == 4)
#endif
		{
			if( RC_BAD( GedGetUINT32( pCurNd, &ui32NdAddr)))
			{
				goto Exit;
			}
			
			pInfoNd = (NODE *)((FLMUINT)ui32NdAddr);
		}
#if defined( FLM_UNIX) || defined( FLM_64BIT)
		else if( sizeof( FLMUINT) == 8)
		{
			FLMUINT		uiLen = 8;
		
			if( RC_BAD( GedGetBINARY( pCurNd, (void *)&pInfoNd, &uiLen)))
			{
				goto Exit;
			}
		}
#endif

		if( GedFind( GED_TREE, pInfoNd, F_RECEDIT_VISIBLE_FIELD, 1))
		{
			bVisible = TRUE;
		}
	}
	else
	{
		bVisible = TRUE;
		if (RC_OK( getSystemNode( pCurNd,
							F_RECEDIT_INVISIBLE_CNT_FIELD, 1,
							&pInvNode)))
		{
			if (RC_OK( GedGetUINT( pInvNode, &uiInvCnt)))
			{
				if (uiInvCnt)
				{
					bVisible = FALSE;
				}
			}
		}
	}

Exit:

	return( bVisible);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::isSystemNode(
	NODE *		pCurNd)
{
	FLMBOOL		bSystemNd = FALSE;

	flmAssert( m_bSetupCalled == TRUE);

	if( GedTagNum( pCurNd) == F_RECEDIT_SYSTEM_FIELD) 
	{
		bSystemNd = TRUE;
	}

	return( bSystemNd);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::markRecordModified(
	NODE *		pCurNd)
{
	NODE *	pRootNd = NULL;
	FLMUINT	uiFlags = 0;
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( (pRootNd = getRootNode( pCurNd)) != NULL)
	{
		(void)getControlFlags( pRootNd, &uiFlags);
		if( !(uiFlags & F_RECEDIT_FLAG_RECMOD))
		{
			(void)setControlFlags( pRootNd, uiFlags | F_RECEDIT_FLAG_RECMOD);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::isRecordModified(
	NODE *		pCurNd)
{
	NODE *	pRootNd = NULL;
	FLMUINT	uiFlags = 0;
	FLMBOOL	bModified = FALSE;

	flmAssert( m_bSetupCalled == TRUE);

	if( (pRootNd = getRootNode( pCurNd)) != NULL)
	{
		(void)getControlFlags( pRootNd, &uiFlags);
		if( (uiFlags & F_RECEDIT_FLAG_RECMOD))
		{
			bModified = TRUE;
		}
	}

	return( bModified);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::clearRecordModified(
	NODE *		pCurNd)
{
	NODE *	pRootNd = NULL;
	NODE *	pTmpNd = NULL;
	FLMUINT	uiFlags = 0;
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Clear the "record modified" flag
	*/
	
	if( (pRootNd = getRootNode( pCurNd)) != NULL)
	{
		(void)getControlFlags( pRootNd, &uiFlags);
		if( (uiFlags & F_RECEDIT_FLAG_RECMOD))
		{
			uiFlags &= ~F_RECEDIT_FLAG_RECMOD;
			if( RC_BAD( rc = setControlFlags( pRootNd, uiFlags)))
			{
				goto Exit;
			}
		}
	}

	/*
	Clear the field "new" and "modified" flags
	*/

	pTmpNd = pRootNd;
	do
	{
		if( !isSystemNode( pTmpNd))
		{
			(void)getControlFlags( pTmpNd, &uiFlags);
			if( (uiFlags & (F_RECEDIT_FLAG_FLDMOD | F_RECEDIT_FLAG_NEWFLD)))
			{
				uiFlags &= ~(F_RECEDIT_FLAG_FLDMOD | F_RECEDIT_FLAG_NEWFLD);
				if( RC_BAD( rc = setControlFlags( pTmpNd, uiFlags)))
				{
					goto Exit;
				}
			}
		}
		pTmpNd = pTmpNd->next;
	}
	while( pTmpNd && GedNodeLevel( pTmpNd) > 0);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::clearSelections( void)
{
	NODE *	pTmpNd = NULL;
	FLMUINT	uiFlags = 0;
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Clear the "selected" flags
	*/

	pTmpNd = m_pTree;
	while( pTmpNd)
	{
		if( !isSystemNode( pTmpNd))
		{
			(void)getControlFlags( pTmpNd, &uiFlags);
			if( (uiFlags & F_RECEDIT_FLAG_SELECTED))
			{
				uiFlags &= ~F_RECEDIT_FLAG_SELECTED;
				if( RC_BAD( rc = setControlFlags( pTmpNd, uiFlags)))
				{
					goto Exit;
				}
			}
		}
		pTmpNd = pTmpNd->next;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_RecEditor::setCurrentNode(
	NODE *				pCurNd)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	// VISIT: Add sanity check to make sure the new current node
	// is contained in the current forest

	m_pCurNd = pCurNd;

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_RecEditor::setFirstNode(
	NODE *				pNd)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	// VISIT: Add sanity check to make sure the new current node
	// is contained in the current forest

	m_pScrFirstNd = pNd;

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::getDisplayValue(
	NODE *				pNd,
	FLMUINT				uiConvType,
	char *				pucBuf,
	FLMUINT				uiBufSize)
{
	RCODE				rc = FERR_OK;
	DBE_VAL_INFO	valInfo;
	char *			pucTmp;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	This is a stupid check, but keep it for now.
	*/

	if( uiBufSize <= 32 || !pNd || !pucBuf)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	*pucBuf = '\0';
	switch( GedValType( pNd))
	{
		case FLM_TEXT_TYPE:
		{
			FLMUINT	uiTextSize = uiBufSize;
			if( RC_BAD( rc = GedGetNATIVE( pNd, pucBuf, &uiTextSize)) && 
				 rc != FERR_CONV_DEST_OVERFLOW)
			{
				f_sprintf( (char *)pucBuf, "<TEXT - UNKNOWN>");
			}
			else
			{
				rc = FERR_OK;
				pucTmp = pucBuf;
				while( *pucTmp)
				{
					if( *pucTmp < 32 || *pucTmp > 126)
					{
						*pucTmp = '?';
					}
					pucTmp++;
				}
			}
			break;
		}

		case FLM_BLOB_TYPE:
		{
			f_sprintf( (char *)pucBuf, "<BLOB>");
			break;
		}

		case FLM_CONTEXT_TYPE:
		{
			FLMUINT32		ui32Val;

			if( GedValLen( pNd) && RC_OK( GedGetUINT32( pNd, &ui32Val)))
			{
				if( ui32Val != 0xFFFFFFFF || GedValLen( pNd) != 0)
				{
					f_sprintf( (char *)pucBuf, "@%u@ (0x%4.4X)",
						(unsigned)ui32Val, (unsigned)ui32Val);
				}
			}
			break;
		}

		case FLM_NUMBER_TYPE:
		{
			FLMUINT32	ui32Val;
			FLMINT32		i32Val;

			if( RC_BAD( GedGetUINT32( pNd, &ui32Val)))
			{
				if( RC_OK( GedGetINT32( pNd, &i32Val)))
				{
					f_sprintf( (char *)pucBuf, "%d", (int)i32Val);
				}
			}
			else
			{
				f_sprintf( (char *)pucBuf, "%u (0x%4.4X)",
					(unsigned)ui32Val, (unsigned)ui32Val);
			}
			break;
		}

		case FLM_BINARY_TYPE:
		{
			FLMBYTE *	pucBin;
			FLMUINT		uiLoop;
			FLMUINT		uiBinLen;
			FLMBYTE *	pucBufPtr = (FLMBYTE *)pucBuf;

			pucBin = (FLMBYTE *)GedValPtr( pNd);
			uiBinLen = 32;
			if( uiBinLen > GedValLen( pNd))
			{
				uiBinLen = GedValLen( pNd);
			}

			if( GedTagNum( pNd) == FLM_REFS_TAG)
			{
				for( uiLoop = 0; uiLoop < uiBinLen; uiLoop += sizeof( FLMUINT))
				{
					if( *((FLMUINT *)&(pucBin[ uiLoop])))
					{
						f_sprintf( (char *)pucBufPtr, "%u ",
							(unsigned)(*((FLMUINT *)&(pucBin[ uiLoop]))));
						pucBufPtr += f_strlen( (const char *)pucBufPtr);
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				switch( uiConvType)
				{
					case F_RECEDIT_TEXT_TYPE:
					{
						for( uiLoop = 0; uiLoop < uiBinLen; uiLoop++)
						{
							if( pucBin[ uiLoop] >= 32 && pucBin[ uiLoop] <= 126)
							{
								*pucBufPtr = pucBin[ uiLoop];
							}
							else
							{
								*pucBufPtr = '.';
							}
							pucBufPtr++;
						}
						break;
					}

					default:
					{
						for( uiLoop = 0; uiLoop < uiBinLen; uiLoop++)
						{
							f_sprintf( (char *)pucBufPtr, "%2.2X ", 
								(unsigned)pucBin[ uiLoop]);
							pucBufPtr += 3;
						}
						break;
					}
				}
			}
			*pucBufPtr = '\0';
			break;
		}

		default:
		{
			f_sprintf( (char *)pucBuf, "<UNSUPPORTED VALUE TYPE>");
			break;
		}
	}

	f_memset( &valInfo, 0, sizeof( DBE_VAL_INFO));
	if( m_pEventHook)
	{
		valInfo.pNd = pNd;
		valInfo.pucBuf = pucBuf;
		valInfo.uiBufLen = uiBufSize;
		valInfo.uiConvType = uiConvType;

		if( RC_BAD( rc = m_pEventHook( this, F_RECEDIT_EVENT_GETDISPVAL,
			(void *)(&valInfo), m_EventData)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::canEditRecord(
	NODE *		pCurNd)
{
	FLMBOOL	bCanEdit = TRUE;
	NODE *	pRootNd;

	// VISIT: Check read-only flag

	flmAssert( m_bSetupCalled == TRUE);

	if( m_bReadOnly || !pCurNd ||
		((pRootNd = getRootNode( pCurNd)) == NULL))
	{
		bCanEdit = FALSE;
		goto Exit;
	}

Exit:

	return( bCanEdit);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::canEditNode(
	NODE *		pCurNd)
{
	FLMBOOL		bCanEdit = TRUE;
	NODE *		pRootNd = NULL;
	FLMUINT		uiFlags;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd || isSystemNode( pCurNd))
	{
		bCanEdit = FALSE;
		goto Exit;
	}

	(void)getControlFlags( pCurNd, &uiFlags);
	if( uiFlags & F_RECEDIT_FLAG_READ_ONLY)
	{
		bCanEdit = FALSE;
		goto Exit;
	}

	/*
	If this is a new (uncommitted) node, allow it to be
	edited.
	*/

	if( uiFlags & F_RECEDIT_FLAG_NEWFLD)
	{
		bCanEdit = TRUE;
		goto Exit;
	}

	pRootNd = getRootNode( pCurNd);
	switch( GedTagNum( pCurNd))
	{
		case FLM_KEY_TAG:
		case FLM_REFS_TAG:
		{
			bCanEdit = FALSE;
			break;
		}
		case FLM_TYPE_TAG:
		{
			if( GedTagNum( pRootNd) == FLM_FIELD_TAG)
			{
				bCanEdit = FALSE;
			}
			break;
		}
	}
	
Exit:

	return( bCanEdit);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::canDeleteRecord(
	NODE *		pCurNd)
{
	FLMUINT	uiFlags;
	FLMBOOL	bCanDelete = TRUE;
	NODE *	pRootNd;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd ||
		((pRootNd = getRootNode( pCurNd)) == NULL))
	{
		bCanDelete = FALSE;
		goto Exit;
	}

	(void)getControlFlags( pRootNd, &uiFlags);
	if( uiFlags & (F_RECEDIT_FLAG_READ_ONLY | F_RECEDIT_FLAG_NO_DELETE))
	{
		bCanDelete = FALSE;
		goto Exit;
	}

	switch( GedTagNum( pRootNd))
	{
		case FLM_KEY_TAG:
		case FLM_FIELD_TAG:
		{
			bCanDelete = FALSE;
			break;
		}
	}
	
Exit:

	return( bCanDelete);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_RecEditor::canDeleteNode(
	NODE *		pCurNd)
{
	FLMBOOL		bCanDelete = TRUE;
	NODE *		pRootNd = NULL;
	FLMUINT		uiFlags;

	if( !pCurNd || isSystemNode( pCurNd))
	{
		bCanDelete = FALSE;
		goto Exit;
	}

	(void)getControlFlags( pCurNd, &uiFlags);
	if( uiFlags & (F_RECEDIT_FLAG_READ_ONLY | F_RECEDIT_FLAG_NO_DELETE))
	{
		bCanDelete = FALSE;
		goto Exit;
	}

	if( (pRootNd = getRootNode( pCurNd)) == pCurNd)
	{
		bCanDelete = canDeleteRecord( pCurNd);
		goto Exit;
	}

	if( !bCanDelete)
	{
		goto Exit;
	}

	switch( GedTagNum( pCurNd))
	{
		case FLM_REFS_TAG:
		{
			bCanDelete = FALSE;
			break;
		}
		case FLM_TYPE_TAG:
		{
			if( GedTagNum( pRootNd) == FLM_FIELD_TAG)
			{
				bCanDelete = FALSE;
			}
			break;
		}
	}
	
Exit:

	return( bCanDelete);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::deleteRecordFromDb(
	NODE *		pCurNd)
{
	HFDB			hSourceDb = HFDB_NULL;
	FLMUINT		uiSourceCont = FLM_DATA_CONTAINER;
	FLMUINT		uiSourceDrn;
	NODE *		pRootNd = NULL;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRootNd = getRootNode( pCurNd);

	if( RC_BAD( rc = GedGetRecSource(
		pRootNd, &hSourceDb, &uiSourceCont, &uiSourceDrn)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = deleteRecordFromDb( hSourceDb, 
		uiSourceCont, uiSourceDrn)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::deleteRecordFromDb(
	HFDB			hSourceDb,
	FLMUINT		uiSourceCont,
	FLMUINT		uiSourceDrn)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( hSourceDb == HFDB_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( rc = FlmRecordDelete( hSourceDb,
		uiSourceCont, uiSourceDrn, FLM_AUTO_TRANS | 10)))
	{
		goto Exit;
	}

	if( uiSourceCont == FLM_DICT_CONTAINER)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Removes a tree from the buffer and marks the record modified
*****************************************************************************/
RCODE F_RecEditor::pruneTree(
	NODE * 			pCurNd)
{
	NODE *		pTmpNd = NULL;
	FLMBOOL		bClippedHasCurr = FALSE;
	RCODE			rc = FERR_OK;

	if( m_pCurNd == pCurNd)
	{
		// The cursor is positioned on a node that will be
		// clipped
		bClippedHasCurr = TRUE;
	}

	pTmpNd = getNextNode( pCurNd);
	while( pTmpNd && GedNodeLevel( pTmpNd) > GedNodeLevel( pCurNd))
	{
		pTmpNd = getNextNode( pTmpNd);
		if( m_pCurNd == pTmpNd)
		{
			// The cursor is positioned on a node that will be
			// clipped
			bClippedHasCurr = TRUE;
		}
	}

	if( !pTmpNd)
	{
		pTmpNd = getPrevNode( pCurNd);
	}

	(void)markRecordModified( pCurNd);
	GedClip( GED_TREE, pCurNd);

	if( m_pTree == pCurNd)
	{
		m_pTree = pTmpNd;
		if( !m_pTree)
		{
			setTree( NULL);
		}
	}

	if( bClippedHasCurr)
	{
		m_pCurNd = pTmpNd;
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::modifyRecordInDb(
	NODE *		pCurNd,
	FLMBOOL		bAddInBackground,
	FLMBOOL		bCreateSuspended)
{
	HFDB					hSourceDb = HFDB_NULL;
	FLMUINT				uiSourceCont = FLM_DATA_CONTAINER;
	FLMUINT				uiSourceDrn;
	FLMUINT				uiFlags;
	NODE *				pRootNd = NULL;
	NODE *				pCleanRootNd;
	FlmRecord *			pTmpRecord = NULL;
	void *				pPoolMark = m_scratchPool.poolMark();
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRootNd = getRootNode( pCurNd);

	if( RC_BAD( rc = GedGetRecSource(
		pRootNd, &hSourceDb, &uiSourceCont, &uiSourceDrn)))
	{
		goto Exit;
	}

	if( hSourceDb == HFDB_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( rc = copyCleanRecord( &m_scratchPool,
		pRootNd, &pCleanRootNd)))
	{
		goto Exit;
	}

	if( (pTmpRecord = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pTmpRecord->importRecord( pCleanRootNd)))
	{
		goto Exit;
	}

	uiFlags = FLM_AUTO_TRANS | 10;
	if (bAddInBackground)
	{
		uiFlags |= FLM_DO_IN_BACKGROUND;
	}

	if( bCreateSuspended)
	{
		uiFlags |= FLM_SUSPENDED;
	}

	if( RC_BAD( rc = FlmRecordModify( hSourceDb,
		uiSourceCont, uiSourceDrn, pTmpRecord, uiFlags)))
	{
		goto Exit;
	}

	if( uiSourceCont == FLM_DICT_CONTAINER)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

Exit:

	if( pPoolMark)
	{
		m_scratchPool.poolReset( pPoolMark);
	}
	
	if (pTmpRecord)
	{
		pTmpRecord->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::addRecordToDb(
	NODE *		pCurNd,
	FLMUINT		uiContainer,
	FLMBOOL		bAddInBackground,
	FLMBOOL		bCreateSuspended,
	FLMUINT *	puiDrn)
{
	NODE *				pRootNd = NULL;
	NODE *				pNewRootNd = NULL;
	NODE *				pCleanRootNd;
	void *				pPoolMark = m_scratchPool.poolMark();
	FlmRecord *			pTmpRecord = NULL;
	RCODE					rc = FERR_OK;
	FLMUINT				uiFlags;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pCurNd)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRootNd = getRootNode( pCurNd);

	if( RC_BAD( rc = copyCleanRecord( &m_scratchPool,
		pRootNd, &pCleanRootNd)))
	{
		goto Exit;
	}

	if( (pTmpRecord = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pTmpRecord->importRecord( pCleanRootNd)))
	{
		goto Exit;
	}

	uiFlags = FLM_AUTO_TRANS | 10;
	if (bAddInBackground)
	{
		uiFlags |= FLM_DO_IN_BACKGROUND;
	}

	if( bCreateSuspended)
	{
		uiFlags |= FLM_SUSPENDED;
	}

	if( RC_BAD( rc = FlmRecordAdd( m_hDefaultDb,
		uiContainer, puiDrn, pTmpRecord, uiFlags)))
	{
		goto Exit;
	}

	if( !(GedValType( pRootNd) & HAS_REC_SOURCE))
	{
		if( m_pTree == pRootNd)
		{
			m_pTree = getNextRecord( pRootNd);
		}

		/*
		Set the record's source information.
		*/

		if( RC_BAD( rc = gedCreateSourceNode( &m_treePool, GedTagNum( pRootNd),
			m_hDefaultDb, uiContainer, *puiDrn, &pNewRootNd)))
		{
			goto Exit;
		}

		if( (pNewRootNd->next = pRootNd->next) != NULL)
		{
			pNewRootNd->next->prior = pNewRootNd;
		}

		if( (pNewRootNd->prior = pRootNd->prior) != NULL)
		{
			pNewRootNd->prior->next = pNewRootNd;
		}

		pNewRootNd->ui32Length = GedValLen( pRootNd);
		pNewRootNd->value = pRootNd->value;
		GedClip( GED_TREE, pNewRootNd);
	}
	else
	{
		/*
		Reset the source information or copy the record if it was re-added to
		the database at a different DRN.
		*/

		if( GedGetRecId( pRootNd))
		{
			if( (pNewRootNd = GedCopy( &m_treePool,
				GED_TREE, pRootNd)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
		else
		{
			if( m_pTree == pRootNd)
			{
				m_pTree = getNextNode( pRootNd);
			}

			pNewRootNd = pRootNd;
			GedClip( GED_TREE, pNewRootNd);
		}
		
		gedSetRecSource( pNewRootNd, m_hDefaultDb, uiContainer, *puiDrn);

	}

	if( RC_BAD( rc = _insertRecord( pNewRootNd)))
	{
		goto Exit;
	}

	m_pCurNd = pNewRootNd;

	if( uiContainer == FLM_DICT_CONTAINER)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

Exit:

	if( pPoolMark)
	{
		m_scratchPool.poolReset( pPoolMark);
	}

	if (pTmpRecord)
	{
		pTmpRecord->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_RecEditor::retrieveRecordsFromDb(
	FLMUINT		uiContainer,
	FLMUINT		uiFirstDrn,
	FLMUINT		uiLastDrn)
{
	NODE *				pGedRec = NULL;
	NODE *				pRec = NULL;
	NODE *				pFirstRec = NULL;
	NODE *				pSearchStart = NULL;
	FlmRecord *			pTmpRec = NULL;
	void *				pvMark = m_scratchPool.poolMark();
	FTX_WINDOW *		pWindow = NULL;
	FLMBOOL				bDone = FALSE;
	FLMUINT				uiRecCount = 0;
	FLMUINT				uiRecId = 0;
	FLMUINT32			ui32Tmp;
	HFCURSOR				hCursor = HFCURSOR_NULL;
	RCODE					rc = FERR_OK;

	/*
	Initialize the cursor
	*/

	if( RC_BAD( rc = FlmCursorInit( m_hDefaultDb, 
		uiContainer, &hCursor)))
	{
		goto Exit;
	}

	/*
	Create a status window
	*/

	if( m_pScreen && (uiLastDrn - uiFirstDrn > 10))
	{
		if( RC_BAD( rc = createStatusWindow(
			" Record Retrieval Status (Press ESC to Interrupt) ",
			FLM_GREEN, FLM_WHITE, NULL, NULL, &pWindow)))
		{
			goto Exit;
		}

		FTXWinOpen( pWindow);
	}

	/*
	Setup the criteria
	*/

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FLM_RECID_FIELD, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_GE_OP)))
	{
		goto Exit;
	}

	ui32Tmp = (FLMUINT32)uiFirstDrn;
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, 
		FLM_UINT32_VAL, &ui32Tmp, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_AND_OP)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FLM_RECID_FIELD, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_LE_OP)))
	{
		goto Exit;
	}

	ui32Tmp = (FLMUINT32)uiLastDrn;
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, 
		FLM_UINT32_VAL, &ui32Tmp, 0)))
	{
		goto Exit;
	}

	while( !isExiting() && !bDone)
	{
		/*
		Update the display
		*/

		if( pWindow)
		{
			FTXWinSetCursorPos( pWindow, 0, 1);
			FTXWinPrintf( pWindow, "Number Retrieved : %u", (unsigned)uiRecCount);
			FTXWinClearToEOL( pWindow);
			FTXWinSetCursorPos( pWindow, 0, 2);
			FTXWinPrintf( pWindow, "Last Record ID   : %u", (unsigned)uiRecId);
			FTXWinClearToEOL( pWindow);

			// Test for the escape key

			if( RC_OK( FTXWinTestKB( pWindow)))
			{
				FLMUINT	uiChar;
				FTXWinInputChar( pWindow, &uiChar);
				if( uiChar == FKB_ESCAPE)
				{
					break;
				}
			}
		}

		/*
		Retrieve records
		*/

		if( RC_BAD( rc = FlmCursorNext( hCursor, &pTmpRec)))
		{
			if( rc != FERR_NOT_FOUND && rc != FERR_EOF_HIT &&
				rc != FERR_BOF_HIT)
			{
				goto Exit;
			}
			else
			{
				rc = FERR_OK;
				bDone = TRUE;
			}
		}

		if( pTmpRec)
		{
			uiRecCount++;
			uiRecId = pTmpRec->getID();
			
			if( RC_BAD( rc = pTmpRec->exportRecord( m_hDefaultDb, 
				&m_scratchPool, &pGedRec)))
			{
				goto Exit;
			}

			if( (pRec = findRecord( uiContainer, uiRecId, pSearchStart)) == NULL)
			{
				if( RC_BAD( rc = insertRecord( pGedRec, &pRec, pSearchStart)))
				{
					goto Exit;
				}
				pSearchStart = pRec;
			}
			else
			{
				if( !areRecordsEqual( pGedRec, pRec))
				{
					if( RC_BAD( rc = pruneTree( pRec)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = insertRecord( pGedRec,
						&pRec, pSearchStart)))
					{
						goto Exit;
					}
				}
				else
				{
					clearRecordModified( m_pCurNd);
				}

				pSearchStart = pRec;
			}

			if( pFirstRec == NULL)
			{
				if( RC_BAD( rc = setCurrentNode( pRec)))
				{
					goto Exit;
				}
				pFirstRec = pRec;
			}
		}

		m_scratchPool.poolReset( pvMark);
		f_yieldCPU();
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	if( pTmpRec)
	{
		pTmpRec->Release();
	}

	if( hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	m_scratchPool.poolReset( pvMark);

	return( rc);
}

/****************************************************************************
Desc:	Poses an ad hoc query.
Note: This routine will clear the contents of the current editor buffer
		without warning (even if there are modifications).
*****************************************************************************/
RCODE F_RecEditor::adHocQuery(
	FLMBOOL			bRetrieve,
	FLMBOOL			bPurge)
{
	HFCURSOR				hCursor = HFCURSOR_NULL;
	FLMUINT				uiTermChar;
	FlmRecord *			pRecord = NULL;
	NODE *				pGedRec;
	F_Pool *				pPool = &m_scratchPool;
	void *				pPoolMark = NULL;
	FLMUINT				uiContainer;
	FLMUINT				uiIndex;
	FLMUINT				uiRecCount = 0;
	FLMUINT				uiRecId = 0;
	FLMUINT				uiNumDeleted = 0;
	FLMUINT				uiDispOffset;
	FLMUINT				uiErrCount = 0;
	FTX_WINDOW *		pWindow = NULL;
	FLMBOOL				bReopenEditor = FALSE;
	RCODE					lastError = FERR_OK;
	RCODE					rc = FERR_OK;

	pPoolMark = pPool->poolMark();

	if( !bPurge)
	{
		requestInput( "Find",
			m_pucAdHocQuery, sizeof( m_pucAdHocQuery), &uiTermChar);
	}
	else
	{
		requestInput( "Find+Delete",
			m_pucAdHocQuery, sizeof( m_pucAdHocQuery), &uiTermChar);
	}

	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	/*
	Close the editor windows to prevent "flicker"
	*/

	if( m_pEditWindow && m_pEditStatusWin)
	{
		FTXWinClose( m_pEditWindow);
		FTXWinClose( m_pEditStatusWin);
		bReopenEditor = TRUE;
	}

	/*
	Select a container
	*/

	if( RC_BAD( rc = selectContainer( &uiContainer, &uiTermChar)))
	{
		goto Exit;
	}

	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	/*
	Select an index
	*/

	if( RC_BAD( rc = selectIndex( uiContainer, F_RECEDIT_ISEL_NOIX,
		&uiIndex, NULL, &uiTermChar)))
	{
		goto Exit;
	}

	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorInit( m_hDefaultDb, uiContainer, &hCursor)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorSetMode( hCursor, 
		FLM_COMP_WILD | FLM_COMP_CASE_INSENSITIVE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmParseQuery( hCursor, m_pNameTable, m_pucAdHocQuery)))
	{
		goto Exit;
	}

	/*
	Set the index
	*/

	if( uiIndex && 
		!(uiIndex == FLM_SELECT_INDEX && *m_pucAdHocQuery == '\0')) // Bug in some versions of FLAIM will cause
																						// FERR_SYNTAX to be returned is FLM_SELECT_INDEX
																						// is specified w/o criteria.
	{
		if( RC_BAD( rc = FlmCursorConfig( hCursor,
													 FCURSOR_SET_FLM_IX,
													 (void *) uiIndex, 0)))
		{
			goto Exit;
		}
	}

	/*
	Create a status window
	*/

	if( RC_BAD( rc = createStatusWindow(
		" Query Status (Press ESC to Interrupt) ",
		FLM_GREEN, FLM_WHITE, NULL, NULL, &pWindow)))
	{
		goto Exit;
	}

	FTXWinOpen( pWindow);

	/*
	Retrieve the records in the result set
	*/

	for( ;;)
	{
		/*
		Test for the escape key
		*/

		if( RC_OK( FTXWinTestKB( pWindow)))
		{
			FLMUINT	uiChar;
			FTXWinInputChar( pWindow, &uiChar);
			if( uiChar == FKB_ESCAPE)
			{
				break;
			}
		}

		/*
		Update the display
		*/

		uiDispOffset = 0;
		FTXWinSetCursorPos( pWindow, 0, ++uiDispOffset);
		FTXWinPrintf( pWindow, "Last Record ID   : %u", (unsigned)uiRecId);
		FTXWinClearToEOL( pWindow);

		if( bRetrieve)
		{
			FTXWinSetCursorPos( pWindow, 0, ++uiDispOffset);
			FTXWinPrintf( pWindow, "Number Retrieved : %u", 
				(unsigned)uiRecCount);
			FTXWinClearToEOL( pWindow);
		}

		if( bPurge)
		{
			FTXWinSetCursorPos( pWindow, 0, ++uiDispOffset);
			FTXWinPrintf( pWindow, "Number Deleted   : %u",
				(unsigned)uiNumDeleted);
			FTXWinClearToEOL( pWindow);
		}

		if( RC_BAD( uiErrCount))
		{
			FTXWinSetCursorPos( pWindow, 0, ++uiDispOffset);
			FTXWinPrintf( pWindow, "Error Count      : %u",
				(unsigned)uiErrCount);
			FTXWinClearToEOL( pWindow);
		}

		if( RC_BAD( lastError))
		{
			FTXWinSetCursorPos( pWindow, 0, ++uiDispOffset);
			FTXWinPrintf( pWindow, "Last Error       : %s", FlmErrorString( lastError));
			FTXWinClearToEOL( pWindow);
		}

		/*
		Get the next record
		*/

		if( bRetrieve)
		{
			if( RC_BAD( rc = FlmCursorNext( hCursor, (FlmRecord **)&pRecord)))
			{
				if( rc == FERR_EOF_HIT && uiRecCount > 0)
				{
					/*
					If no records were retrieved, return the FERR_EOF_HIT error
					*/

					rc = FERR_OK;
				}
				break;
			}
			uiRecId = pRecord->getID();
		}
		else
		{
			if( RC_BAD( rc = FlmCursorNextDRN( hCursor, &uiRecId)))
			{
				if( rc == FERR_EOF_HIT && uiRecCount > 0)
				{
					/*
					If no records were retrieved, return the FERR_EOF_HIT error
					*/

					rc = FERR_OK;
				}
				break;
			}
		}

		// Delete the records.  Note that an explicit transaction is
		// required.

		if( bPurge)
		{
			if( RC_BAD( rc = FlmRecordDelete( m_hDefaultDb, uiContainer, uiRecId, 0)))
			{
				lastError = rc;
				uiErrCount++;
				rc = FERR_OK;
			}
			else
			{
				// Remove the record from the buffer
				pruneTree( findRecord( uiContainer, uiRecId, NULL));
				uiNumDeleted++;
			}
		}

		if( bRetrieve)
		{
			/*
			Clear the buffer on the first successful read
			*/

			if( !uiRecCount)
			{
				if( RC_BAD( rc = setTree( NULL)))
				{
					goto Exit;
				}
			}

			/*
			Insert the record into the buffer
			*/

			if( RC_BAD( rc = pRecord->exportRecord( m_hDefaultDb,	
				&m_scratchPool, &pGedRec)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = insertRecord( pGedRec, NULL)))
			{
				goto Exit;
			}
		}

		pPool->poolReset( pPoolMark);
		uiRecCount++;
		f_yieldCPU();
#ifdef FLM_WIN
		f_sleep( 0);
#endif
	}

	/*
	Position at the top of the buffer
	*/

	m_pCurNd = m_pTree;
	m_uiCurRow = 0;

Exit:

	if( pRecord)
	{
		pRecord->Release();
		pRecord = NULL;
	}

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	if( hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if( bReopenEditor)
	{
		FTXWinOpen( m_pEditStatusWin);
		FTXWinOpen( m_pEditWindow);
	}

	pPool->poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::expandNode(
	NODE *		pNode,
	FLMBOOL *	pbExpanded)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiFlags;
	NODE *	pTmpNode;
	NODE *	pInvNode;
	FLMUINT	uiInvCnt;

	*pbExpanded = FALSE;
	if ((RC_OK( getControlFlags( pNode, &uiFlags))) &&
		 (uiFlags & F_RECEDIT_FLAG_COLLAPSED))
	{
		uiFlags &= ~F_RECEDIT_FLAG_COLLAPSED;
		if (RC_BAD( rc = setControlFlags( pNode, uiFlags)))
		{
			goto Exit;
		}

		// Go through the list of nodes until we hit the next sibling,
		// decrementing the invisible count field as we go.

		pTmpNode = pNode->next;
		while ((pTmpNode) &&
				 (GedNodeLevel( pTmpNode) > GedNodeLevel( pNode)))
		{
			if( !isSystemNode( pTmpNode))
			{
				if (RC_OK( getSystemNode( pTmpNode,
									F_RECEDIT_INVISIBLE_CNT_FIELD, 1,
									&pInvNode)))
				{
					if (RC_BAD( GedGetUINT( pInvNode, &uiInvCnt)))
						goto Exit;
					if (uiInvCnt)
					{
						uiInvCnt--;
					}
					if (RC_BAD( rc = GedPutUINT( &m_treePool, pInvNode, uiInvCnt)))
					{
						goto Exit;
					}
					*pbExpanded = TRUE;
				}
			}
			pTmpNode = pTmpNode->next;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_RecEditor::collapseNode(
	NODE *		pNode,
	FLMBOOL *	pbCollapsed)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiFlags;
	NODE *	pTmpNode;
	NODE *	pInvNode;
	FLMUINT	uiInvCnt;

	*pbCollapsed = FALSE;
	if ((RC_OK( getControlFlags( pNode, &uiFlags))) &&
		 (!(uiFlags & F_RECEDIT_FLAG_COLLAPSED)))
	{
		uiFlags |= F_RECEDIT_FLAG_COLLAPSED;
		if (RC_BAD( rc = setControlFlags( pNode, uiFlags)))
		{
			goto Exit;
		}

		// Go through the list of nodes until we hit the next sibling,
		// incrementing the invisible count field as we go.

		pTmpNode = pNode->next;
		while ((pTmpNode) &&
				 (GedNodeLevel( pTmpNode) > GedNodeLevel( pNode)))
		{
			if( !isSystemNode( pTmpNode))
			{
				if (RC_OK( getSystemNode( pTmpNode,
									F_RECEDIT_INVISIBLE_CNT_FIELD, 1,
									&pInvNode)))
				{
					if (RC_BAD( GedGetUINT( pInvNode, &uiInvCnt)))
						goto Exit;
					uiInvCnt++;
				}
				else
				{
					if (RC_BAD( rc = createSystemNode( pTmpNode,
											F_RECEDIT_INVISIBLE_CNT_FIELD,
											&pInvNode)))
					{
						goto Exit;
					}
					uiInvCnt = 1;
				}
				if (RC_BAD( rc = GedPutUINT( &m_treePool, pInvNode, uiInvCnt)))
				{
					goto Exit;
				}
				*pbCollapsed = TRUE;
			}
			pTmpNode = pTmpNode->next;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Creates a copy of a record w/o any system nodes
*****************************************************************************/
RCODE F_RecEditor::copyCleanRecord(
	F_Pool *		pPool,
	NODE *		pRecNd,
	NODE **		ppCopiedRec)
{
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pPool || !pRecNd || !ppCopiedRec)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = copyCleanTree( pPool, getRootNode( pRecNd), ppCopiedRec)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Creates a copy of a subtree w/o any system nodes
*****************************************************************************/
RCODE F_RecEditor::copyCleanTree(
	F_Pool *		pPool,
	NODE *		pTreeNd,
	NODE **		ppCopiedTree)
{
	NODE *	pNewNd;
	NODE *	pTmpNd;
	NODE *	pCurNd;
	RCODE		rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pPool || !pTreeNd || !ppCopiedTree)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( ( pNewNd = GedCopy( pPool, GED_TREE, pTreeNd)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pCurNd = pNewNd;
	while( pCurNd)
	{
		if( isSystemNode( pCurNd))
		{
			pTmpNd = pCurNd;
			pCurNd = GedParent( pCurNd);
			(void)GedClip( GED_TREE, pTmpNd);
		}
		else
		{
			pCurNd = pCurNd->next;
		}
	}

	*ppCopiedTree = pNewNd;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Creates a copy of the current editor buffer starting at a
		specified node
*****************************************************************************/
RCODE F_RecEditor::copyBuffer(
	F_Pool *		pPool,
	NODE *		pStartNd,
	NODE **		ppNewTree)
{
	NODE *		pCurNd;
	NODE *		pInfoNd;
	NODE *		pNewTree = NULL;
	FLMUINT32	ui32NdAddr;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( !pPool || !pStartNd || !ppNewTree)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( ( pNewTree = GedCopy( pPool, GED_FOREST, pStartNd)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Copy the system nodes and fix up memory pointers
	*/

	pCurNd = pNewTree;
	while( pCurNd)
	{
		if( !GedTagNum( pCurNd))
		{
			if( RC_BAD( rc = GedGetUINT32( pCurNd, &ui32NdAddr)))
			{
				goto Exit;
			}

			if( ( pInfoNd = GedCopy( pPool, GED_FOREST,
				(NODE *)((FLMUINT)ui32NdAddr))) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			// VISIT
			if( RC_BAD( rc = GedPutUINT( pPool, pCurNd,
				(FLMUINT)pInfoNd)))
			{
				goto Exit;
			}
		}
		pCurNd = pCurNd->next;
	}

Exit:

	*ppNewTree = pNewTree;
	return( rc);
}

/****************************************************************************
Desc:	Allows the user to interactively select a field
*****************************************************************************/
RCODE F_RecEditor::createNewField(
	FLMBOOL			bAllocSource,
	NODE **			ppNewField)
{
	NODE *		pTmpNd = NULL;
	NODE *		pNewNd = NULL;
	FLMUINT		uiFlags;
	FLMUINT		uiFldType;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_pNameList != NULL);

	*ppNewField = NULL;

	// VISIT: Check for read-only flags
	if( RC_BAD( rc = m_pNameList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpNd = m_pNameList->getTree()) == NULL)
	{
		goto Exit;
	}

	/*
	Make sure the root node is not a system node!
	*/

	if( m_pNameList->isSystemNode( pTmpNd))
	{
		pTmpNd = m_pNameList->getNextNode( pTmpNd);
	}

	while( pTmpNd)
	{
		m_pNameList->getControlFlags( pTmpNd, &uiFlags);
		if( uiFlags & F_RECEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_RECEDIT_FLAG_SELECTED;
			m_pNameList->setControlFlags( pTmpNd, uiFlags);
			if( bAllocSource)
			{
				if( RC_BAD( rc = gedCreateSourceNode( &m_treePool,
					GedTagNum( pTmpNd), m_hDefaultDb, 
					m_uiDefaultCont, 0, &pNewNd)))
				{
					goto Exit;
				}
			}
			else
			{
				if( (pNewNd = GedNodeMake( &m_treePool,
					GedTagNum( pTmpNd), &rc)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}

			/*
			Set the node's type
			*/

			if( RC_BAD( rc = getFieldType( GedTagNum( pTmpNd), &uiFldType)))
			{
				goto Exit;
			}

			GedValTypeSet( pNewNd, uiFldType);

			/*
			Mark the new node modified
			*/

			(void)setControlFlags( pNewNd,
				(F_RECEDIT_FLAG_FLDMOD | F_RECEDIT_FLAG_NEWFLD));
			break;
		}
		pTmpNd = m_pNameList->getNextNode( pTmpNd);
	}

	*ppNewField = pNewNd;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Gets a field's tag number given its name
*****************************************************************************/
RCODE F_RecEditor::getFieldType(
	FLMUINT			uiFieldNum,
	FLMUINT *		puiFieldType)
{
	FlmRecord *		pTmpRec = NULL;
	void *			pvField;
	FLMUINT			puiPath[ 3];
	char				pucTmpBuf[ 64];
	FLMUINT			uiBufLen;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = FlmRecordRetrieve( m_hDefaultDb,
		FLM_DICT_CONTAINER, uiFieldNum, FO_EXACT, &pTmpRec, NULL)))
	{
		if( rc == FERR_NOT_FOUND)
		{
			*puiFieldType = FLM_TEXT_TYPE;
			rc = FERR_OK;
		}
		goto Exit;
	}

	puiPath[ 0] = FLM_FIELD_TAG;
	puiPath[ 1] = FLM_TYPE_TAG;
	puiPath[ 2] = 0;

	if( (pvField = pTmpRec->find( pTmpRec->root(), puiPath)) == NULL)
	{
		*puiFieldType = FLM_TEXT_TYPE;
		goto Exit;
	}

	uiBufLen = sizeof( pucTmpBuf);
	if( RC_BAD( rc = pTmpRec->getNative( pvField, pucTmpBuf, &uiBufLen)))
	{
		goto Exit;
	}
	
	if( f_strnicmp( pucTmpBuf, "cont", 4) == 0)
	{
		*puiFieldType = FLM_CONTEXT_TYPE;
	}
	else if( f_strnicmp( pucTmpBuf, "numb", 4) == 0)
	{
		*puiFieldType = FLM_NUMBER_TYPE;
	}
	else if( f_strnicmp( pucTmpBuf, "bina", 4) == 0)
	{
		*puiFieldType = FLM_BINARY_TYPE;
	}
	else if( f_strnicmp( pucTmpBuf, "text", 4) == 0)
	{
		*puiFieldType = FLM_TEXT_TYPE;
	}
	else if( f_strnicmp( pucTmpBuf, "blob", 4) == 0)
	{
		*puiFieldType = FLM_BLOB_TYPE;
	}
	else
	{
		*puiFieldType = FLM_TEXT_TYPE;
	}

Exit:

	if( pTmpRec)
	{
		pTmpRec->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Gets a field's tag number given its name
*****************************************************************************/
RCODE F_RecEditor::getFieldNumber(
	const char *	pucFieldName,
	FLMUINT *		puiFieldNum)
{
	RCODE		rc = FERR_OK;

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !m_pNameTable->getFromTagName( NULL, pucFieldName, puiFieldNum))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Gets a dictionary item's name given its number
*****************************************************************************/
RCODE F_RecEditor::getDictionaryName(
	FLMUINT		uiNum,
	char *		pucName)
{
	RCODE		rc = FERR_OK;

	if (m_hDefaultDb == HFDB_NULL && !m_pNameTable)
	{
		FLMBOOL	bSave;

		bSave = FTXRefreshDisabled();
		FTXSetRefreshState( FALSE);
		openNewDb();
		FTXSetRefreshState( bSave);
	}

	*pucName = 0;
	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !m_pNameTable->getFromTagNum( uiNum, NULL, pucName, 128))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Gets a container's number given its name
*****************************************************************************/
RCODE F_RecEditor::getContainerNumber(
	const char *	pucContainerName,
	FLMUINT *		puiContainerNum)
{
	RCODE		rc = FERR_OK;

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !m_pNameTable->getFromTagTypeAndName( NULL, pucContainerName, 
		FLM_CONTAINER_TAG, puiContainerNum))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Gets an index's number given its name
*****************************************************************************/
RCODE F_RecEditor::getIndexNumber(
	const char *	pucIndexName,
	FLMUINT *		puiIndexNum)
{
	RCODE		rc = FERR_OK;

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !m_pNameTable->getFromTagTypeAndName( NULL, pucIndexName, 
		FLM_INDEX_TAG, puiIndexNum))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Updates the name table from the database
*****************************************************************************/
RCODE F_RecEditor::refreshNameTable( void)
{
	NODE *					pRootNd = NULL;
	NODE *					pTmpNd = NULL;
	NODE *					pPriorNd = NULL;
	FLMUINT					uiFlags;
	F_Pool *					pScratchPool = &m_scratchPool;
	void *					pPoolMark = m_scratchPool.poolMark();
	DBE_NAME_TABLE_INFO	nametableInfo;
	FLMUNICODE				uzItemName[ 128];
	FLMUINT					uiId;
	FLMUINT					uiType;
	FLMUINT					uiSubType;
	FLMUINT					uiNextPos;
	RCODE						rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Initialize the name table.
	*/

	if( m_pNameTable && m_bOwnNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	if( !m_pNameList)
	{
		if( (m_pNameList = f_new F_RecEditor) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pNameList->Setup( m_pScreen)))
		{
			goto Exit;
		}
	}
	else
	{
		m_pNameList->reset();
	}

	m_pNameList->setParent( this);
	m_pNameList->setReadOnly( TRUE);
	m_pNameList->setShutdown( m_pbShutdown);
	m_pNameList->setTitle( "Fields - Select One");
	m_pNameList->setKeyHook( f_RecEditorSelectionKeyHook, 0);

	/*
	Call the callback to build the name table
	*/

	f_memset( &nametableInfo, 0, sizeof( DBE_NAME_TABLE_INFO));
	if( m_pEventHook)
	{
		nametableInfo.pNameTable = m_pNameTable;

		if( RC_BAD( rc = m_pEventHook( this, F_RECEDIT_EVENT_NAME_TABLE,
			(void *)(&nametableInfo), m_EventData)))
		{
			goto Exit;
		}
	}

	/*
	Try the default initialization
	*/

	if( !nametableInfo.bInitialized)
	{
		if( m_bOwnNameTable)
		{
			if( (m_pNameTable = f_new F_NameTable) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( RC_BAD( rc = m_pNameTable->setupFromDb( m_hDefaultDb)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		m_pNameTable = nametableInfo.pNameTable;
		m_bOwnNameTable = FALSE;
	}

	// Build the field selection list

	uiNextPos = 0;
	while( m_pNameTable->getNextTagNameOrder( &uiNextPos, uzItemName, 
		NULL, sizeof( uzItemName), 
		&uiId, &uiType, &uiSubType))
	{
		if( uiType == FLM_FIELD_TAG)
		{
			if( (pTmpNd = GedNodeMake( pScratchPool, uiId, &rc)) == NULL)
			{
				goto Exit;
			}

			if( RC_BAD( rc = GedPutUNICODE( pScratchPool, pTmpNd,
				uzItemName)))
			{
				goto Exit;
			}

			if( !pRootNd)
			{
				pRootNd = pTmpNd;
				pPriorNd = pRootNd;
			}
			else
			{
				GedSibGraft( pPriorNd, pTmpNd, GED_LAST);
				pPriorNd = pTmpNd;
			}
		}
	}

	// Pass the list to the editor

	m_pNameList->setTree( pRootNd);

	/*
	Call getTree() and then call setControlFlags for each node.  It is
	important to use the tree returned from getTree() rather than setting
	the flags in the loops above where the nodes are created since
	different pools are used.
	*/

	pTmpNd = m_pNameList->getTree();
	uiFlags = (F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
		F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY);
	while( pTmpNd)
	{
		m_pNameList->setControlFlags( pTmpNd, uiFlags);
		pTmpNd = m_pNameList->getNextNode( pTmpNd);
	}

Exit:

	m_scratchPool.poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Allows the user to interactively select a container
*****************************************************************************/
RCODE F_RecEditor::selectContainer(
	FLMUINT *	puiContainer,
	FLMUINT *	puiTermChar)
{
	NODE *				pRootNd = NULL;
	NODE *				pTmpNd = NULL;
	FLMUINT				uiFlags;
	FLMUNICODE			uzItemName[ 128];
	FLMUINT				uiId;
	FLMUINT				uiType;
	FLMUINT				uiSubType;
	FLMUINT				uiNextPos;
	F_Pool *				pScratchPool = &m_scratchPool;
	void *				pPoolMark = m_scratchPool.poolMark();
	F_RecEditor *		pContainerList = NULL;
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiContainer)
	{
		*puiContainer = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	/*
	Initialize the name table.
	*/

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pContainerList = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pContainerList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pContainerList->setParent( this);
	pContainerList->setReadOnly( TRUE);
	pContainerList->setShutdown( m_pbShutdown);
	pContainerList->setTitle( "Containers - Select One");
	pContainerList->setKeyHook( f_RecEditorSelectionKeyHook, 0);

	if( m_hDefaultDb == HFDB_NULL)
	{
		goto Exit;
	}

	if( (pTmpNd = GedNodeMake( pScratchPool,
		FLM_DATA_CONTAINER, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( pScratchPool, pTmpNd, "Default Data")))
	{
		goto Exit;
	}

	pRootNd = pTmpNd;

	if( (pTmpNd = GedNodeMake( pScratchPool,
		FLM_DICT_CONTAINER, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( pScratchPool, pTmpNd, "Local Dictionary")))
	{
		goto Exit;
	}

	GedSibGraft( pRootNd, pTmpNd, GED_LAST);

	if( (pTmpNd = GedNodeMake( pScratchPool,
		FLM_TRACKER_CONTAINER, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( pScratchPool, pTmpNd, "Tracker")))
	{
		goto Exit;
	}

	GedSibGraft( pRootNd, pTmpNd, GED_LAST);

	uiNextPos = 0;
	while( m_pNameTable->getNextTagNameOrder( &uiNextPos, uzItemName, 
		NULL, sizeof( uzItemName), 
		&uiId, &uiType, &uiSubType))
	{
		if( uiType == FLM_CONTAINER_TAG)
		{
			if( (pTmpNd = GedNodeMake( pScratchPool, uiId, &rc)) == NULL)
			{
				goto Exit;
			}

			if( RC_BAD( rc = GedPutUNICODE( pScratchPool, pTmpNd, uzItemName)))
			{
				goto Exit;
			}

			GedSibGraft( pRootNd, pTmpNd, GED_LAST);
		}
	}

	/*
	Pass the list to the editor
	*/

	pContainerList->setTree( pRootNd);

	/*
	Call getTree() and then call setControlFlags for each node.  It is
	important to use the tree returned from getTree() rather than setting
	the flags in the loops above where the nodes are created since
	different pools are used.
	*/

	pTmpNd = pContainerList->getTree();
	uiFlags = (F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
		F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY);
	while( pTmpNd)
	{
		pContainerList->setControlFlags( pTmpNd, uiFlags);
		pTmpNd = pContainerList->getNextNode( pTmpNd);
	}

	if( RC_BAD( rc = pContainerList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpNd = pContainerList->getTree()) == NULL)
	{
		goto Exit;
	}

	/*
	Make sure the root node is not a system node!
	*/

	if( pContainerList->isSystemNode( pTmpNd))
	{
		pTmpNd = pContainerList->getNextNode( pTmpNd);
	}

	while( pTmpNd)
	{
		pContainerList->getControlFlags( pTmpNd, &uiFlags);
		if( uiFlags & F_RECEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_RECEDIT_FLAG_SELECTED;
			pContainerList->setControlFlags( pTmpNd, uiFlags);
			if( puiContainer)
			{
				*puiContainer = GedTagNum( pTmpNd);
			}
			break;
		}
		pTmpNd = pContainerList->getNextNode( pTmpNd);
	}

	if( puiTermChar)
	{
		*puiTermChar = pContainerList->getLastKey();
	}

Exit:

	if( pContainerList)
	{
		pContainerList->Release();
		pContainerList = NULL;
	}

	m_scratchPool.poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Allows the user to interactively select an index
*****************************************************************************/
RCODE F_RecEditor::selectIndex(
	FLMUINT			uiContainer,
	FLMUINT			uiFlags,
	FLMUINT *		puiIndex,
	FLMUINT *		puiContainer,
	FLMUINT *		puiTermChar)
{
	NODE *				pRootNd = NULL;
	NODE *				pTmpNd = NULL;
	NODE *				pGedRec;
	FLMUINT				uiDispFlags;
	FLMUINT				uiFoundContainer;
	F_Pool *				pPool = &m_scratchPool;
	void *				pPoolMark = m_scratchPool.poolMark();
	FlmRecord *			pDictRec = NULL;
	F_RecEditor *		pIndexList = NULL;
	HFCURSOR				hCursor = HFCURSOR_NULL;
	RCODE					rc = FERR_OK;
	void *				pvFld;
	char					szBuf [80];
	FLMUINT				uiLen;

	flmAssert( m_bSetupCalled == TRUE);

	*puiIndex = 0;

	if (puiContainer)
	{
		*puiContainer = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if( (pIndexList = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pIndexList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pIndexList->setParent( this);
	pIndexList->setReadOnly( TRUE);
	pIndexList->setShutdown( m_pbShutdown);
	pIndexList->setTitle( "Indexes - Select One");
	pIndexList->setKeyHook( f_RecEditorSelectionKeyHook, 0);

	if( m_hDefaultDb == HFDB_NULL)
	{
		goto Exit;
	}

	/*
	Cannot use tag 0 (system info tag) for no index.  Use 0xFFFF and
	translate to 0 on exit
	*/

	if( uiFlags & F_RECEDIT_ISEL_NOIX)
	{
		if( (pTmpNd = GedNodeMake( pPool, FLM_SELECT_INDEX, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = GedPutNATIVE( pPool, pTmpNd,
			"Allow DB to select the index")))
		{
			goto Exit;
		}

		pRootNd = pTmpNd;

		if( (pTmpNd = GedNodeMake( pPool, 0xFFFF, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = GedPutNATIVE( pPool, pTmpNd, "No Index")))
		{
			goto Exit;
		}

		GedSibGraft( pRootNd, pTmpNd, GED_LAST);
	}

	/*
	Set up a query to get indexes for the specified container
	*/

	if( RC_BAD( rc = FlmCursorInit( m_hDefaultDb,
		FLM_DICT_CONTAINER, &hCursor)))
	{
		goto Exit;
	}

	// Eliminate any record that doesn't have an index tag.

	if (RC_BAD( rc = FlmCursorAddField( hCursor, FLM_INDEX_TAG, 0)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = FlmCursorNext( hCursor, &pDictRec)))
		{
			if( rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
			break;
		}

		// See if it matches the container we are looking for

		if (pDictRec->getFieldID( pDictRec->root()) != FLM_INDEX_TAG)
		{
			continue;
		}
		if (uiContainer)
		{
			if ((pvFld = pDictRec->find( pDictRec->root(),
									FLM_CONTAINER_TAG)) == NULL)
			{
				uiFoundContainer = FLM_DATA_CONTAINER;
			}
			else
			{
				uiLen = sizeof( szBuf);
				if (RC_BAD( rc = pDictRec->getNative( pvFld, szBuf, &uiLen)))
				{
					goto Exit;
				}
				if (f_stricmp( szBuf, "ALL") == 0 ||
					 f_stricmp( szBuf, "*") == 0)
				{
					uiFoundContainer = 0;
				}
				else
				{
					if (RC_BAD( rc = pDictRec->getUINT( pvFld, &uiFoundContainer)))
					{
						goto Exit;
					}
					if (uiFoundContainer == 0)
					{
						uiFoundContainer = FLM_DATA_CONTAINER;
					}
				}
			}
			if (uiFoundContainer && uiContainer != uiFoundContainer)
			{
				continue;
			}
		}

		if( RC_BAD( rc = pDictRec->exportRecord( m_hDefaultDb, pPool, &pGedRec)))
		{
			goto Exit;
		}

		if( (pTmpNd = GedNodeCopy( pPool, pGedRec, NULL, NULL)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		flmAssert( pDictRec->getID() != 0);
		GedTagNumSet( pTmpNd, pDictRec->getID());

		if( pRootNd)
		{
			GedSibGraft( pRootNd, pTmpNd, GED_LAST);
		}
		else
		{
			pRootNd = pTmpNd;
		}
	}

	/*
	Pass the list to the editor
	*/

	pIndexList->setTree( pRootNd);

	/*
	Call getTree() and then call setControlFlags for each node.  It is
	important to use the tree returned from getTree() rather than setting
	the flags in the loops above where the nodes are created since
	different pools are used.
	*/

	pTmpNd = pIndexList->getTree();
	uiDispFlags = (F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
		F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY |
		F_RECEDIT_FLAG_HIDE_SOURCE);

	while( pTmpNd)
	{
		pIndexList->setControlFlags( pTmpNd, uiDispFlags);
		pTmpNd = pIndexList->getNextNode( pTmpNd);
	}

	if( RC_BAD( rc = pIndexList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpNd = pIndexList->getTree()) == NULL)
	{
		goto Exit;
	}

	/*
	Make sure the root node is not a system node!
	*/

	if( pIndexList->isSystemNode( pTmpNd))
	{
		pTmpNd = pIndexList->getNextNode( pTmpNd);
	}

	while( pTmpNd)
	{
		pIndexList->getControlFlags( pTmpNd, &uiDispFlags);
		if( uiDispFlags & F_RECEDIT_FLAG_SELECTED)
		{
			uiDispFlags &= ~F_RECEDIT_FLAG_SELECTED;
			pIndexList->setControlFlags( pTmpNd, uiDispFlags);
			*puiIndex = GedTagNum( pTmpNd);
			break;
		}
		pTmpNd = pIndexList->getNextNode( pTmpNd);
	}

	if( puiTermChar)
	{
		*puiTermChar = pIndexList->getLastKey();
	}

	if( pIndexList->getLastKey() == FKB_ESCAPE)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if( *puiIndex == 0xFFFF)
	{
		/*
		Translate 0xFFFF (no index) to 0
		*/
		*puiIndex = 0;
	}
	else if (puiContainer)
	{

		// Read the index record and get the container number

		if( RC_BAD( rc = FlmRecordRetrieve( m_hDefaultDb,
								FLM_DICT_CONTAINER, *puiIndex,
								FO_EXACT, &pDictRec, NULL)))
		{
			goto Exit;
		}
		if ((pvFld = pDictRec->find( pDictRec->root(),
								FLM_CONTAINER_TAG)) == NULL)
		{
			*puiContainer = FLM_DATA_CONTAINER;
		}
		else
		{
			uiLen = sizeof( szBuf);
			if (RC_BAD( rc = pDictRec->getNative( pvFld, szBuf, &uiLen)))
			{
				goto Exit;
			}
			if (f_stricmp( szBuf, "ALL") == 0 ||
				 f_stricmp( szBuf, "*") == 0)
			{
				*puiContainer = 0;
			}
			else
			{
				if (RC_BAD( rc = pDictRec->getUINT( pvFld, puiContainer)))
				{
					goto Exit;
				}
				if (*puiContainer == 0)
				{
					*puiContainer = FLM_DATA_CONTAINER;
				}
			}
		}
	}

Exit:

	if( pDictRec)
	{
		pDictRec->Release();
	}

	if( hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if( pIndexList)
	{
		pIndexList->Release();
		pIndexList = NULL;
	}

	pPool->poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Allows listing of index keys (and references) by having the user
		interactively build from and until keys.
*****************************************************************************/
RCODE F_RecEditor::indexList( void)
{
	F_RecEditor	*		pKeyEditor = NULL;
	FTX_WINDOW *		pStatusWindow = NULL;
	NODE *				pCurNd;
	NODE *				pFromKey = NULL;
	NODE *				pUntilKey = NULL;
	NODE *				pGedRec;
	FLMUINT				uiTermChar;
	FLMUINT				uiIndex;
	FLMUINT				uiIxContainer;
	FLMBOOL				bResetTree = TRUE;
	F_Pool				tmpPool;
	FlmRecord *			pSrchKey = NULL;
	FlmRecord *			pSaveSrchKey;
	FLMUINT				uiSrchDrn;
	FLMUINT				uiSrchFlag;
	FlmRecord *			pFoundKey = NULL;
	FLMUINT				uiFoundDrn;
	FlmRecord *			pTmpRec = NULL;
	FlmRecord *			pFromKeyRec = NULL;
	FlmRecord *			pUntilKeyRec = NULL;
	RCODE					rc = FERR_OK;
	NODE *				pTmpNd;
	void *				pvFld;
	FLMUINT				uiKeyContainer;
	FLMBOOL				bNewKey;
	FLMBYTE *			pucUntilKeyBuf = NULL;
	FLMBYTE *			pucFoundKeyBuf;
	FLMUINT				uiUntilKeyLen;
	FLMUINT				uiFoundKeyLen;
	FLMUINT				uiKeyCount;
	FLMUINT				uiRefCount;

	flmAssert( m_bSetupCalled == TRUE);

	tmpPool.poolInit( 512);

	if( RC_BAD( rc = selectIndex( 0, 0, &uiIndex, &uiIxContainer, &uiTermChar)))
	{
		goto Exit;
	}

	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	// Initialize the key editor

	if( (pKeyEditor = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pKeyEditor->Setup( m_pScreen)))
	{
		goto Exit;
	}

	// Configure the editor

	// VISIT: Need to add configuration option to allow the
	// editor to use its parent's name table

	pKeyEditor->setParent( this);
	pKeyEditor->setShutdown( m_pbShutdown);
	pKeyEditor->setTitle( "Index List");
	pKeyEditor->setDefaultSource( m_hDefaultDb, m_uiDefaultCont);
	pKeyEditor->setKeyHook( f_KeyEditorKeyHook, 0);

	// Get the first key in the index

	if (RC_BAD( rc = FlmKeyRetrieve( m_hDefaultDb, uiIndex, 0, NULL, 0, FO_FIRST,
								&pTmpRec, NULL)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pTmpRec->exportRecord( HFDB_NULL, &tmpPool, &pGedRec)))
	{
		goto Exit;
	}

	// If the index is on ALL containers, need to add a dummy node with
	// the container number from this key, so it will show up on the
	// screen.

	if (!uiIxContainer)
	{
		if ((pTmpNd = GedNodeCreate( &tmpPool, FLM_CONTAINER_TAG,
								0, &rc)) == NULL)
		{
			goto Exit;
		}
		if (RC_BAD( rc = GedPutUINT( &tmpPool, pTmpNd,
									pTmpRec->getContainerID())))
		{
			goto Exit;
		}
		GedChildGraft( pGedRec, pTmpNd, GED_LAST);
	}
	if( RC_BAD( rc = pKeyEditor->appendTree( pGedRec, &pCurNd)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pKeyEditor->addComment( 
		pCurNd, TRUE, "This is the first key in the index")))
	{
		goto Exit;
	}

	tmpPool.poolReset();
	pTmpRec->Release();
	pTmpRec = NULL;

	// Get the last key in the index

	if (RC_BAD( rc = FlmKeyRetrieve( m_hDefaultDb, uiIndex, 0, NULL, 0, FO_LAST,
								&pTmpRec, NULL)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pTmpRec->exportRecord( HFDB_NULL, &tmpPool, &pGedRec)))
	{
		goto Exit;
	}

	// If the index is on ALL containers, need to add a dummy node with
	// the container number from this key, so it will show up on the
	// screen.

	if (!uiIxContainer)
	{
		if ((pTmpNd = GedNodeCreate( &tmpPool, FLM_CONTAINER_TAG,
								0, &rc)) == NULL)
		{
			goto Exit;
		}
		if (RC_BAD( rc = GedPutUINT( &tmpPool, pTmpNd,
									pTmpRec->getContainerID())))
		{
			goto Exit;
		}
		GedChildGraft( pGedRec, pTmpNd, GED_LAST);
	}
	if( RC_BAD( rc = pKeyEditor->appendTree( pGedRec, &pCurNd)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pKeyEditor->addComment( 
		pCurNd, TRUE, "This is the last key in the index")))
	{
		goto Exit;
	}

	tmpPool.poolReset();
	pTmpRec->Release();
	pTmpRec = NULL;

	// Show the keys and allow them to be edited

ix_list_retry:

	tmpPool.poolReset();

	if( RC_BAD( rc = pKeyEditor->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( pKeyEditor->getLastKey() == FKB_ESCAPE)
	{
		goto Exit;
	}

	// Copy "clean" versions of the edited keys into a temp pool

	pKeyEditor->copyCleanRecord( &tmpPool, pKeyEditor->getTree(), &pFromKey);
	pKeyEditor->copyCleanRecord( &tmpPool,
				GedSibNext( pKeyEditor->getTree()), &pUntilKey);

	// Create a status window

	if( RC_BAD( rc = createStatusWindow(
		" Key Retrieval Status (Press ESC to Interrupt) ",
		FLM_GREEN, FLM_WHITE, NULL, NULL, &pStatusWindow)))
	{
		goto Exit;
	}
	FTXWinOpen( pStatusWindow);

	// Get the FROM Key

	if( (pFromKeyRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pFromKeyRec->importRecord( pFromKey)))
	{
		goto Exit;
	}

	// If this index is on all containers, need to set the container ID
	// in the key.

	if (!uiIxContainer)
	{
		if ((pvFld = pFromKeyRec->find( pFromKeyRec->root(),
										FLM_CONTAINER_TAG)) == NULL)
		{
			uiKeyContainer = 0;
		}
		else if (RC_BAD( rc = pFromKeyRec->getUINT( pvFld, &uiKeyContainer)))
		{
			goto Exit;
		}
		pFromKeyRec->setContainerID( uiKeyContainer);
	}

	// Get the UNTIL key

	if( (pUntilKeyRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pUntilKeyRec->importRecord( pUntilKey)))
	{
		goto Exit;
	}

	// If this index is on all containers, need to set the container ID
	// in the key.

	if (!uiIxContainer)
	{
		if ((pvFld = pUntilKeyRec->find( pUntilKeyRec->root(),
										FLM_CONTAINER_TAG)) == NULL)
		{
			uiKeyContainer = 0;
		}
		else if (RC_BAD( rc = pUntilKeyRec->getUINT( pvFld, &uiKeyContainer)))
		{
			goto Exit;
		}
		pUntilKeyRec->setContainerID( uiKeyContainer);
	}

	// Set up the starting search key and DRN

	if ((pSrchKey = pFromKeyRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	uiSrchDrn = 0;
	uiSrchFlag = FO_INCL;
	bNewKey = TRUE;
	uiKeyCount = 0;
	uiRefCount = 0;

	// Allocate key buffers for the until key and the found key so we
	// can do comparisons.

	if( RC_BAD( rc = f_alloc( MAX_KEY_SIZ * 2, &pucUntilKeyBuf)))
	{
		goto Exit;
	}

	pucFoundKeyBuf = &pucUntilKeyBuf [MAX_KEY_SIZ];

	// Get the collated until key.

	if (RC_BAD( rc = FlmKeyBuild( m_hDefaultDb, uiIndex,
										pUntilKeyRec->getContainerID(),
										pUntilKeyRec, 0,
										pucUntilKeyBuf, &uiUntilKeyLen)))
	{
		goto Exit;
	}

	// Read the keys

	while( !isExiting())
	{

		// Update the display

		FTXWinSetCursorPos( pStatusWindow, 0, 1);
		FTXWinPrintf( pStatusWindow, "Keys Retrieved : %u",
			(unsigned)uiKeyCount);
		FTXWinClearToEOL( pStatusWindow);
		FTXWinSetCursorPos( pStatusWindow, 0, 2);
		FTXWinPrintf( pStatusWindow, "References Retrieved : %u",
			(unsigned)uiRefCount);
		FTXWinClearToEOL( pStatusWindow);

		// Test for the escape key

		if( RC_OK( FTXWinTestKB( pStatusWindow)))
		{
			FLMUINT	uiChar;
			FTXWinInputChar( pStatusWindow, &uiChar);
			if( uiChar == FKB_ESCAPE)
			{
				break;
			}
		}

		if (RC_BAD( rc = FlmKeyRetrieve( m_hDefaultDb, uiIndex,
								pSrchKey->getContainerID(), pSrchKey,
								uiSrchDrn, uiSrchFlag, &pFoundKey, &uiFoundDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				if (bNewKey)
				{
					break;
				}
				uiSrchFlag = FO_EXCL;
				bNewKey = TRUE;
				rc = FERR_OK;
				continue;
			}
		}
		if (bNewKey)
		{
			FLMINT	iCmp;
			FLMUINT	uiCmpLen;

			// See if we have gone past the until key.

			if (RC_BAD( rc = FlmKeyBuild( m_hDefaultDb, uiIndex,
												pFoundKey->getContainerID(),
												pFoundKey, 0,
												pucFoundKeyBuf, &uiFoundKeyLen)))
			{
				goto Exit;
			}
			if ((uiCmpLen = uiUntilKeyLen) > uiFoundKeyLen)
			{
				uiCmpLen = uiFoundKeyLen;
			}
			iCmp = f_memcmp( pucFoundKeyBuf, pucUntilKeyBuf, uiCmpLen);
			if ((iCmp > 0) ||
				 (iCmp == 0 && uiFoundKeyLen > uiUntilKeyLen))
			{
				break;
			}
			uiKeyCount++;

			bNewKey = FALSE;
			uiSrchFlag = FO_EXCL | FO_KEY_EXACT;
		}
		uiRefCount++;

		// Display the key.

		pFoundKey->setID( uiFoundDrn);
		if( RC_BAD( rc = pFoundKey->exportRecord( HFDB_NULL, &tmpPool, &pGedRec)))
		{
			goto Exit;
		}

		// If the index is on ALL containers, need to add a dummy node with
		// the container number from this key, so it will show up on the
		// screen.

		if (!uiIxContainer)
		{
			if ((pTmpNd = GedNodeCreate( &tmpPool, FLM_CONTAINER_TAG,
									0, &rc)) == NULL)
			{
				goto Exit;
			}
			if (RC_BAD( rc = GedPutUINT( &tmpPool, pTmpNd,
										pFoundKey->getContainerID())))
			{
				goto Exit;
			}
			GedChildGraft( pGedRec, pTmpNd, GED_LAST);
		}
		if (bResetTree)
		{
			pKeyEditor->setTree( NULL);
			bResetTree = FALSE;
		}
		if (RC_BAD( rc = pKeyEditor->appendTree( pGedRec, NULL)))
		{
			goto Exit;
		}
		
		tmpPool.poolReset();

		// Swap the search key and found key - preparation for
		// the next search.

		pSaveSrchKey = pSrchKey;
		pSrchKey = pFoundKey;
		uiSrchDrn = uiFoundDrn;
		pFoundKey = pSaveSrchKey;

		f_yieldCPU();
#ifdef FLM_WIN
		f_sleep( 0);
#endif
	}

	FTXWinFree( &pStatusWindow);

	/*
	Display the results
	*/

	if( uiKeyCount)
	{
		pKeyEditor->setKeyHook( f_RecEditorViewOnlyKeyHook, 0);
		if( RC_BAD( rc = pKeyEditor->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
		{
			goto Exit;
		}
	}
	else
	{
		displayMessage(
			"No Keys Found Within Specified Range", rc,
			NULL, FLM_RED, FLM_WHITE);
		pKeyEditor->setCurrentNode( pKeyEditor->getTree());
		goto ix_list_retry;
	}

Exit:

	if (pucUntilKeyBuf)
	{
		f_free( &pucUntilKeyBuf);
	}

	if (pFoundKey)
	{
		pFoundKey->Release();
	}

	if (pSrchKey)
	{
		pSrchKey->Release();
	}

	if( pTmpRec)
	{
		pTmpRec->Release();
	}

	if( pFromKeyRec)
	{
		pFromKeyRec->Release();
	}

	if( pUntilKeyRec)
	{
		pUntilKeyRec->Release();
	}

	if( pStatusWindow)
	{
		FTXWinFree( &pStatusWindow);
	}

	if( pKeyEditor)
	{
		pKeyEditor->Release();
	}

	if( RC_BAD( rc))
	{
		if( rc == FERR_EOF_HIT)
		{
			displayMessage( "The index is empty", rc,
				NULL, FLM_RED, FLM_WHITE);
			rc = FERR_OK;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Allows the user to interactively select and manage files
*****************************************************************************/
RCODE F_RecEditor::fileManager(
	const char *	pucTitle,
	FLMUINT			uiModeFlags,
	char *			pszInitialPath,
	char *			pszSelectedPath,
	FLMUINT *		puiTermChar)
{
	FLMUINT				uiFlags;
	NODE *				pRootNd = NULL;
	NODE *				pTmpNd = NULL;
	F_Pool *				pPool = &m_scratchPool;
	void *				pPoolMark = m_scratchPool.poolMark();
	F_RecEditor *		pPathList = NULL;
	IF_DirHdl *			pDirectory = NULL;
	char					szDirPath [F_PATH_MAX_SIZE];
	char					szInitPath [F_PATH_MAX_SIZE];
	char					szTmpPath [F_PATH_MAX_SIZE];
	char					pucTmpBuf[ 128];
	FLMUINT				uiTermChar;
	FLMUINT				uiTmpLen;
	char					pucFileName[ F_PATH_MAX_SIZE];
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_pScreen != NULL);
	flmAssert( pszSelectedPath != NULL);

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if( (pPathList = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pPathList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pPathList->setParent( this);
	pPathList->setReadOnly( TRUE);
	pPathList->setShutdown( m_pbShutdown);
	szDirPath [0] = 0;
	pPathList->setKeyHook( f_RecEditorFileKeyHook, (void *)(&szDirPath [0]));
	pPathList->setEventHook( f_RecEditorFileEventHook, 0);

refresh_list:

	pPathList->setTree( NULL);

	/*
	If no initial path specified, prompt the user to enter a
	starting path
	*/

	if( !pszInitialPath || (uiModeFlags & F_RECEDIT_FSEL_PROMPT))
	{
		char	pucResponse[ F_PATH_MAX_SIZE];

		*pucResponse = '\0';
		if( pszInitialPath)
		{
			f_strcpy( pucResponse, pszInitialPath);
		}

		pPathList->requestInput(
			"Path", pucResponse, sizeof( pucResponse), &uiTermChar);

		if( uiTermChar != FKB_ENTER)
		{
			if( puiTermChar)
			{
				*puiTermChar = uiTermChar;
			}
			goto Exit;
		}

		f_strcpy( szInitPath, pucResponse);
		pszInitialPath = &szInitPath [0];
	}

	/*
	Create a directory object
	*/

	if( m_pFileSystem->isDir( pszInitialPath))
	{
		f_strcpy( szDirPath, pszInitialPath);

		pucFileName[ 0] = '*';
		pucFileName[ 1] = '\0';
	}
	else
	{
		if( RC_BAD( rc = m_pFileSystem->pathReduce( 
			pszInitialPath, szDirPath, pucFileName)))
		{
			goto Exit;
		}
		pszInitialPath = &szDirPath [0];
		if( RC_BAD( m_pFileSystem->doesFileExist( szDirPath)))
		{
			rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
			displayMessage( "The specified path is invalid", rc,
				NULL, FLM_RED, FLM_WHITE);
			goto Exit;
		}
	}

	if( pDirectory)
	{
		pDirectory->Release();
		pDirectory = NULL;
	}

	if( RC_BAD( rc = m_pFileSystem->openDir(
		pszInitialPath, (char *)pucFileName, &pDirectory )))
	{
		goto Exit;
	}

	/*
	Find all files in the directory
	*/

	for( rc = pDirectory->next(); ! RC_BAD( rc) ; rc = pDirectory->next())
	{
		const char *	pucItemName = pDirectory->currentItemName();

		if( (pTmpNd = GedNodeMake( pPool, 0xFFFF, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		f_sprintf( (char *)pucTmpBuf, "%.64s", pucItemName);

		if( RC_BAD( rc = GedPutNATIVE( pPool, pTmpNd, pucTmpBuf)))
		{
			goto Exit;
		}

		if( pDirectory->currentItemIsDir())
		{
			pPathList->insertRecord( pTmpNd, &pRootNd, NULL);
			pPathList->addAnnotation( pRootNd, "DIR");
		}
		else
		{
			pPathList->appendTree( pTmpNd, &pRootNd);
		}

		if( RC_BAD( rc = pPathList->setControlFlags( pRootNd,
			(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
			F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY |
			F_RECEDIT_FLAG_HIDE_SOURCE))))
		{
			goto Exit;
		}
	}

	/*
	Add the parent directory
	*/

	if( RC_BAD( rc = m_pFileSystem->pathReduce( szDirPath, szTmpPath, NULL)))
	{
		goto Exit;
	}

	if( szTmpPath [0])
	{
 		if( (pTmpNd = GedNodeMake( pPool, 0xFFFF, &rc)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		f_sprintf( (char *)pucTmpBuf, "..");

		if( RC_BAD( rc = GedPutNATIVE( pPool, pTmpNd, pucTmpBuf)))
		{
			goto Exit;
		}

		pPathList->insertRecord( pTmpNd, &pRootNd);
		pPathList->addAnnotation( pRootNd, "DIR");

		if( RC_BAD( rc = pPathList->setControlFlags( pRootNd,
			(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
			F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY |
			F_RECEDIT_FLAG_HIDE_SOURCE))))
		{
			goto Exit;
		}
	}

	/*
	Present the directory to the user
	*/

	if( pucTitle)
	{
		pPathList->setTitle( pucTitle);
	}
	else
	{
		pPathList->setTitle( szDirPath);
	}

	pPathList->setCurrentNode( pPathList->getTree());
	pPathList->setCurrentAtTop();

	if( RC_BAD( rc = pPathList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY,
		TRUE, FALSE)))
	{
		goto Exit;
	}

	if( (pTmpNd = pPathList->getTree()) == NULL)
	{
		goto Exit;
	}

	/*
	Make sure the root node is not a system node!
	*/

	if( pPathList->isSystemNode( pTmpNd))
	{
		pTmpNd = pPathList->getNextNode( pTmpNd);
	}

	while( pTmpNd)
	{
		pPathList->getControlFlags( pTmpNd, &uiFlags);
		if( uiFlags & F_RECEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_RECEDIT_FLAG_SELECTED;
			pPathList->setControlFlags( pTmpNd, uiFlags);
			uiTmpLen = sizeof( pucTmpBuf);

			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, pucTmpBuf, &uiTmpLen)))
			{
				goto Exit;
			}

			if( !f_strcmp( pucTmpBuf, ".."))
			{
				if( RC_BAD( rc = m_pFileSystem->pathReduce( 
					szDirPath, pszSelectedPath, NULL)))
				{
					goto Exit;
				}
			}
			else
			{
				f_strcpy( pszSelectedPath, szDirPath);
				if( RC_BAD( rc = m_pFileSystem->pathAppend( 
					pszSelectedPath, pucTmpBuf)))
				{
					goto Exit;
				}
			}

			if( m_pFileSystem->isDir( pszSelectedPath))
			{
				f_strcpy( pszInitialPath, pszSelectedPath);
				pucTitle = NULL;
				goto refresh_list;
			}

			break;
		}
		pTmpNd = pPathList->getNextNode( pTmpNd);
	}

	if( puiTermChar)
	{
		*puiTermChar = pPathList->getLastKey();
	}

Exit:

	if( pPathList)
	{
		pPathList->Release();
		pPathList = NULL;
	}

	if( pDirectory)
	{
		pDirectory->Release();
		pDirectory = NULL;
	}

	pPool->poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Allows the user to view a file (reads the entire file into memory)
*****************************************************************************/
RCODE F_RecEditor::fileViewer(
	const char *	pucTitle,
	const char *	pszFilePath,
	FLMUINT *		puiTermChar)
{
	char *				pucTmpBuf = NULL;
	char *				pucTmp = NULL;
	char *				pucLine = NULL;
	F_Pool				pool;
	F_RecEditor *		pViewer = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	NODE *				pTmpNd = NULL;
	NODE *				pRootNd = NULL;
	FLMUINT				uiFileOffset;
	FLMUINT				uiFlags;
	FLMUINT				uiBytesRead;
	FLMUINT				uiBufOffset;
	FLMBOOL				bReadFromDisk;
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_pScreen != NULL);

	pool.poolInit( 2048);
	
	if( RC_BAD( rc = pool.poolAlloc( 2048, (void **)&pucTmpBuf)))
	{
		goto Exit;
	}
	
	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if( (pViewer = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pViewer->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pViewer->setParent( this);
	pViewer->setReadOnly( TRUE);
	pViewer->setShutdown( m_pbShutdown);

	if( RC_BAD( rc = m_pFileSystem->openFile( pszFilePath, FLM_IO_RDONLY,
		&pFileHdl)))
	{
		displayMessage( "Unable to open file", rc,
			NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	uiFileOffset = 0;
	uiBufOffset = 0;
	bReadFromDisk = TRUE;
	for( ;;)
	{
		if( bReadFromDisk)
		{
			if( RC_BAD( rc = pFileHdl->read( uiFileOffset, 2048 - uiBufOffset,
				&pucTmpBuf[ uiBufOffset], &uiBytesRead)))
			{
				if( rc == FERR_IO_END_OF_FILE)
				{
					bReadFromDisk = FALSE;
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			uiFileOffset += uiBytesRead;
			uiBufOffset += uiBytesRead;
		}

		pucLine = pucTmpBuf;
		pucTmp = pucTmpBuf;
		while( pucTmp < pucTmpBuf + uiBufOffset && *pucTmp != '\r')
		{
			if( *pucTmp == '\0')
			{
				rc = RC_SET( FERR_FAILURE);
				displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
				goto Exit;
			}
			else if( *pucTmp == '\t')
			{
				*pucTmp = ' ';
			}
			pucTmp++;
		}

		if( *pucTmp == '\r')
		{
			*pucTmp = '\0';
		}
		else
		{
			pucTmpBuf[ uiBufOffset] = '\0';
		}

		// convert!!!

		if( (pTmpNd = GedNodeMake( &pool, 1, &rc)) == NULL)
		{
			goto Exit;
		}

		if( RC_BAD( rc = GedPutNATIVE( &pool, pTmpNd, pucLine)))
		{
			goto Exit;
		}

		if( !pRootNd)
		{
			pRootNd = pTmpNd;
		}
		else
		{
			GedSibGraft( pRootNd, pTmpNd, GED_LAST);
		}

		if( pucTmp)
		{
			if( (FLMUINT)((pucTmp - pucTmpBuf) + 1) > uiBufOffset)
			{
				uiBufOffset = 0;
			}
			else
			{
				uiBufOffset -= (pucTmp - pucTmpBuf) + 1;
			}

			if( uiBufOffset > 0 && *(pucTmp + 1) == '\n')
			{
				uiBufOffset--;
				pucTmp++;
			}

			if( uiBufOffset > 0)
			{
				f_memmove( pucTmpBuf, pucTmp + 1, uiBufOffset);
			}
			else
			{
				break;
			}
		}
		else
		{
			break;
		}
	}

	pFileHdl->Release();
	pFileHdl = NULL;

	pViewer->setTree( pRootNd);

	pTmpNd = pViewer->getTree();
	uiFlags = (F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
		F_RECEDIT_FLAG_READ_ONLY);
	while( pTmpNd)
	{
		pViewer->setControlFlags( pTmpNd, uiFlags);
		pTmpNd = pViewer->getNextNode( pTmpNd);
	}
	
	/*
	Show the file to the user
	*/

	if( pucTitle)
	{
		pViewer->setTitle( pucTitle);
	}
	else
	{
		pViewer->setTitle( pszFilePath);
	}

	pViewer->setCurrentNode( pViewer->getTree());
	pViewer->setCurrentAtTop();

	if( RC_BAD( rc = pViewer->interactiveEdit( m_uiULX, m_uiULY,
		m_uiLRX, m_uiLRY, TRUE, FALSE)))
	{
		goto Exit;
	}

	if( puiTermChar)
	{
		*puiTermChar = pViewer->getLastKey();
	}

Exit:

	if( pViewer)
	{
		pViewer->Release();
		pViewer = NULL;
	}

	if( pFileHdl)
	{
		pFileHdl->Release();
		pFileHdl = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts a text string to a number
*****************************************************************************/
RCODE F_RecEditor::getNumber(
	const char *	pucBuf,
	FLMUINT *		puiValue,
	FLMINT *			piValue)
{
	RCODE					rc = FERR_OK;
	const char *		pucTmp = NULL;
	FLMUINT				uiValue = 0;
	FLMUINT				uiDigits = 0;
	FLMUINT				uiHexOffset = 0;
	FLMBOOL				bNeg = FALSE;
	FLMBOOL				bHex = FALSE;

	if( puiValue)
	{
		*puiValue = 0;
	}

	if( piValue)
	{
		*piValue = 0;
	}

	if( f_strnicmp( pucBuf, "0x", 2) == 0)
	{
		uiHexOffset = 2;
		bHex = TRUE;
	}
	else if( *pucBuf == 'x' || *pucBuf == 'X')
	{
		uiHexOffset = 1;
		bHex = TRUE;
	}
	else
	{
		pucTmp = pucBuf;
		while( *pucTmp)
		{
			if( (*pucTmp >= '0' && *pucTmp <= '9') ||
				(*pucTmp >= 'A' && *pucTmp <= 'F') ||
				(*pucTmp >= 'a' && *pucTmp <= 'f'))
			{
				if( (*pucTmp >= 'A' && *pucTmp <= 'F') ||
				(*pucTmp >= 'a' && *pucTmp <= 'f'))
				{
					bHex = TRUE;
				}
			}
			else
			{
				rc = RC_SET( FERR_CONV_ILLEGAL);
				goto Exit;
			}
			pucTmp++;
		}
		uiHexOffset = 0;
	}

	if( bHex)
	{
		pucTmp = &(pucBuf[ uiHexOffset]);
		uiDigits = f_strlen( pucTmp);
		if( !uiDigits || uiDigits > 8)
		{
			rc = RC_SET( FERR_CONV_ILLEGAL);
			goto Exit;
		}

		while( *pucTmp)
		{
			uiValue <<= 4;
			if( *pucTmp >= '0' && *pucTmp <= '9')
			{
				uiValue |= *pucTmp - '0';
			}
			else if( *pucTmp >= 'a' && *pucTmp <= 'f')
			{
				uiValue |= (*pucTmp - 'a') + 10;
			}
			else if( *pucTmp >= 'A' && *pucTmp <= 'F')
			{
				uiValue |= (*pucTmp - 'A') + 10;
			}
			else
			{
				rc = RC_SET( FERR_CONV_ILLEGAL);
				goto Exit;
			}
			pucTmp++;
		}
	}
	else if( (*pucBuf >= '0' && *pucBuf <= '9') || *pucBuf == '-')
	{
		pucTmp = pucBuf;
		if( *pucTmp == '-')
		{
			bNeg = TRUE;
			pucTmp++;
		}
		uiDigits = f_strlen( pucTmp);
		
		while( *pucTmp)
		{
			if( *pucTmp >= '0' && *pucTmp <= '9')
			{
				FLMUINT	uiNewVal = uiValue;

				if( uiNewVal > (0xFFFFFFFF / 10))
				{
					rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				uiNewVal *= 10;
				uiNewVal += *pucTmp - '0';

				if( uiNewVal < uiValue)
				{
					rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				uiValue = uiNewVal;
			}
			else
			{
				rc = RC_SET( FERR_CONV_BAD_DIGIT);
				goto Exit;
			}
			pucTmp++;
		}
	}
	else
	{
		rc = RC_SET( FERR_CONV_BAD_DIGIT);
		goto Exit;
	}

	if( bNeg)
	{
		if( piValue)
		{
			if( uiValue > 0x8FFFFFFF)
			{
				rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
				goto Exit;
			}
			*piValue = -((FLMINT)uiValue);
		}
		else
		{
			rc = RC_SET( FERR_CONV_ILLEGAL);
			goto Exit;
		}
	}
	else
	{
		if( puiValue)
		{
			*puiValue = uiValue;
		}
		else
		{
			rc = RC_SET( FERR_CONV_ILLEGAL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Shows a help screen
*****************************************************************************/
RCODE F_RecEditor::showHelp(
	FLMUINT *	puiKeyRV)
{
	NODE *				pRootNd = NULL;
	NODE *				pTmpNd = NULL;
	FLMUINT				uiFlags;
	F_Pool *				pScratchPool = &m_scratchPool;
	void *				pPoolMark = m_scratchPool.poolMark();
	F_RecEditor *		pHelpList = NULL;
	RCODE					rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( (pHelpList = f_new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pHelpList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pHelpList->setParent( this);
	pHelpList->setReadOnly( TRUE);
	pHelpList->setShutdown( m_pbShutdown);
	pHelpList->setTitle( "HELP");
	pHelpList->setKeyHook( f_RecEditorSelectionKeyHook, 0);

	if( (pTmpNd = GedNodeMake( pScratchPool, 1, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = GedPutNATIVE( pScratchPool, pTmpNd, "Keyboard Commands")))
	{
		goto Exit;
	}

	pRootNd = pTmpNd;

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		(FLMUINT)'?', (void *)"?               Help (this screen)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_UP, (void *)"UP              Position cursor to the previous field",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_DOWN, (void *)"DOWN            Position cursor to the next field",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_PGUP, (void *)"PG UP           Position cursor to the previous page",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_PGDN, (void *)"PG DOWN         Position cursor to the next page",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_CTRL_DOWN, (void *)"CTRL-DOWN, >    Position cursor to the next record",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_CTRL_UP, (void *)"CTRL-UP, <      Position cursor to the previous record",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_HOME, (void *)"HOME            Position cursor to the top of the buffer",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_END, (void *)"END             Position cursor to the bottom of the buffer",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_DELETE, (void *)"DEL             Delete the current field or record",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		(FLMUINT)'#', (void *)"#               Database statistics",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_A, (void *)"ALT-A           Add the current record to the database",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_C, (void *)"ALT-C           Clear all records from the buffer",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_D, (void *)"ALT-D           Delete records by ID or via a query",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_F, (void *)"ALT-F           Find records in the database via a query",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_I, (void *)"ALT-I           Show index keys and references",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_M, (void *)"ALT-M           Update the current record in the database",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_R, (void *)"ALT-R           Retrieve a record from the database",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_S, (void *)"ALT-S           Re-read the current record from the database (sync)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_T, (void *)"ALT-T           Transaction operations",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_F3, (void *)"ALT-F3          Search the buffer for a string",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_F3, (void *)"F3              Find next",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ALT_F10, (void *)"ALT-F10         Toggle display colors on/off",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_SF3, (void *)"SHIFT-F3        Find previous",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		1, (void *)"RIGHT, LEFT     Follow link",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_INSERT, (void *)"INSERT          Insert a new field or record",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ENTER, (void *)"ENTER           Edit the current field's value",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		1, (void *)"'+' or '-'      Toggle expanded/collapsed context",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_F8, (void *)"F8              Index manager",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_F9, (void *)"F9              Memory manager",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pScratchPool, pRootNd,
		FKB_ESCAPE, (void *)"ESC, ALT-Q      Exit",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	/*
	Call the applications help hook to extend the help screen.
	*/

	if( m_pHelpHook)
	{
		if( RC_BAD( rc = m_pHelpHook( this, pHelpList,
			pScratchPool, m_HelpData, &pRootNd)))
		{
			goto Exit;
		}
	}

	/*
	Pass the list to the editor
	*/

	pHelpList->setTree( pRootNd);

	/*
	Call getTree() and then call setControlFlags for each node.  It is
	important to use the tree returned from getTree() rather than setting
	the flags in the loops above where the nodes are created since
	different pools are used.
	*/

	pTmpNd = pHelpList->getTree();
	uiFlags = (F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
		F_RECEDIT_FLAG_LIST_ITEM | F_RECEDIT_FLAG_READ_ONLY);
	while( pTmpNd)
	{
		pHelpList->setControlFlags( pTmpNd, uiFlags);
		pTmpNd = pHelpList->getNextNode( pTmpNd);
	}

	/*
	Show the help screen
	*/

	if( RC_BAD( rc = pHelpList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX,
		m_uiLRY, TRUE, FALSE)))
	{
		goto Exit;
	}

	/*
	Get a pointer to the root of the tree
	*/

	if( (pTmpNd = pHelpList->getTree()) == NULL)
	{
		goto Exit;
	}

	/*
	Make sure the root node is not a system node!
	*/

	if( pHelpList->isSystemNode( pTmpNd))
	{
		pTmpNd = pHelpList->getNextNode( pTmpNd);
	}

	/*
	Find the selected item
	*/

	if( puiKeyRV)
	{
		*puiKeyRV = 0;
		if( pHelpList->getLastKey() == FKB_ENTER)
		{
			while( pTmpNd)
			{
				pHelpList->getControlFlags( pTmpNd, &uiFlags);
				if( uiFlags & F_RECEDIT_FLAG_SELECTED)
				{
					*puiKeyRV = GedTagNum( pTmpNd);
					if( *puiKeyRV == 1)
					{
						*puiKeyRV = 0;
					}
					break;
				}
				pTmpNd = pHelpList->getNextNode( pTmpNd);
			}
		}
	}

Exit:

	if( pHelpList)
	{
		pHelpList->Release();
		pHelpList = NULL;
	}

	m_scratchPool.poolReset( pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Creates a window for displaying an operation's status
*****************************************************************************/
RCODE F_RecEditor::createStatusWindow(
	const char *		pucTitle,
	eColorType			back,
	eColorType			fore,
	FLMUINT *			puiCols,
	FLMUINT *			puiRows,
	FTX_WINDOW **		ppWindow)
{
	FLMUINT			uiNumRows;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumWinRows = 0;
	FLMUINT			uiNumWinCols = 0;
	FTX_WINDOW *	pWindow = NULL;
	RCODE				rc = FERR_OK;

	*ppWindow = NULL;

	FTXScreenGetSize( m_pScreen, &uiNumCols, &uiNumRows);

	if( puiCols)
	{
		uiNumWinCols = *puiCols;
	}

	if( puiRows)
	{
		uiNumWinRows = *puiRows;
	}

	if( uiNumWinCols <= 2 || uiNumWinRows < 3)
	{
		uiNumWinCols = uiNumCols - 2;
		uiNumWinRows = uiNumRows / 2;
	}

	if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumWinCols,
		uiNumWinRows, &pWindow)))
	{
		goto Exit;
	}

	if( puiCols)
	{
		*puiCols = uiNumWinCols;
	}

	if( puiRows)
	{
		*puiRows = uiNumWinRows;
	}

	FTXWinMove( pWindow, (FLMUINT)((uiNumCols - uiNumWinCols) / 2),
		(FLMUINT)((uiNumRows - uiNumWinRows) / 2));
	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetLineWrap( pWindow, FALSE);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_INVISIBLE);

	if( m_bMonochrome)
	{
		back = FLM_LIGHTGRAY;
		fore = FLM_BLACK;
		FTXWinSetBackFore( pWindow, back, fore);
	}
	else
	{
		FTXWinSetBackFore( pWindow, back, fore);
	}

	FTXWinClear( pWindow);
	FTXWinDrawBorder( pWindow);

	if( pucTitle)
	{
		FTXWinSetTitle( pWindow, pucTitle, back, fore);
	}

	*ppWindow = pWindow;

Exit:

	return( rc);
}
	
/****************************************************************************
Desc: Compares two records for equality, ignoring system nodes
*****************************************************************************/
FLMBOOL F_RecEditor::areRecordsEqual(
	NODE *		pRootA,
	NODE *		pRootB)
{
	NODE *		pCurA = pRootA;
	NODE *		pCurB = pRootB;
	FLMUINT		uiStartLevel;
	FLMBOOL		bEqual = FALSE;
	FLMBOOL		bStarting = TRUE;

	uiStartLevel = GedNodeLevel( pCurA);
	for( ;;)
	{
		while( pCurA && GedTagNum( pCurA) == 0)
		{
			pCurA = pCurA->next;
		}

		while( pCurB && GedTagNum( pCurB) == 0)
		{
			pCurB = pCurB->next;
		}

		if( !pCurA || !pCurB)
		{
			if( !pCurA && !pCurB)
			{
				bEqual = TRUE;
			}
			goto Exit;
		}

		if( (GedNodeLevel( pCurA) <= uiStartLevel ||
			GedNodeLevel( pCurB) <= uiStartLevel) &&
			!bStarting)
		{
			if( GedNodeLevel( pCurA) == GedNodeLevel( pCurB) &&
				GedNodeLevel( pCurA) == uiStartLevel)
			{
				bEqual = TRUE;
			}
			goto Exit;
		}
		else
		{
			bStarting = FALSE;
		}

		if( GedTagNum( pCurA) != GedTagNum( pCurB))
		{
			goto Exit;
		}

		if( GedNodeLevel( pCurA) != GedNodeLevel( pCurB))
		{
			goto Exit;
		}

		if( GedValType( pCurA) != GedValType( pCurB))
		{
			goto Exit;
		}

		if( GedValLen( pCurA) != GedValLen( pCurB))
		{
			goto Exit;
		}

		if( f_memcmp( GedValPtr( pCurA), GedValPtr( pCurB),
			GedValLen( pCurA)) != 0)
		{
			goto Exit;
		}

		pCurA = pCurA->next;
		pCurB = pCurB->next;

		/*
		Release CPU to prevent CPU hog
		*/

		f_yieldCPU();
	}

Exit:

	return( bEqual);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_RecEditor::followLink(
	NODE *	pLinkNd,
	FLMUINT 	uiLinkKey)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( m_pLinkHook)
	{
		rc = m_pLinkHook( this, pLinkNd, m_LinkData, uiLinkKey);
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_RecEditorDefaultLinkHook(
	F_RecEditor *		pRecEditor,
	NODE *				pLinkNd,
	void *				UserData,
	FLMUINT				uiLinkKey)
{
	F_Pool			pool;
	FLMUINT			uiDrn;
	FlmRecord *		pRecord = NULL;
	NODE *			pNewNd = NULL;
	NODE *			pGedRec;
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);
	F_UNREFERENCED_PARM( uiLinkKey);

	pool.poolInit( 2048);

	if( GedValType( pLinkNd) == FLM_CONTEXT_TYPE)
	{
		if( RC_BAD( rc = GedGetRecPtr( pLinkNd, &uiDrn)))
		{
			goto Exit;
		}

		if( uiDrn == 0xFFFFFFFF)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}

		if( (pNewNd = pRecEditor->findRecord(
			pRecEditor->getContainer(), uiDrn)) == NULL)
		{
			if( RC_BAD( rc = FlmRecordRetrieve( pRecEditor->getDb(),
				pRecEditor->getContainer(), uiDrn, FO_EXACT, &pRecord, NULL)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRecord->exportRecord( pRecEditor->getDb(), 
				&pool, &pGedRec)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRecEditor->insertRecord( pGedRec, &pNewNd)))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = pRecEditor->setCurrentNode( pNewNd)))
		{
			goto Exit;
		}
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
		pRecord = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_RecEditorViewOnlyKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pCurNd);
	F_UNREFERENCED_PARM( UserData);

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_ALT_F3:
		case FKB_SF3:
		case FKB_F3:
		case FKB_ESCAPE:
		case '>':
		case '<':
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		default:
		{
			*puiKeyOut = 0;
			break;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_KeyEditorKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pCurNd);
	F_UNREFERENCED_PARM( UserData);

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_ENTER:
		case FKB_DELETE:
		case FKB_INSERT:
		case FKB_ALT_F3:
		case FKB_SF3:
		case FKB_F3:
		case FKB_ESCAPE:		/* Quit key editor */
		case FKB_ALT_Q:		/* Done editing keys */
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		default:
		{
			*puiKeyOut = 0;
			break;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_RecEditorSelectionKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pCurNd);
	F_UNREFERENCED_PARM( UserData);

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_F3:
		case FKB_SF3:
		case FKB_ALT_F3:
		case FKB_ESCAPE:
		case FKB_ENTER:
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		default:
		{
			*puiKeyOut = 0;
			break;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_RecEditorFileKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE				rc = FERR_OK;

	*puiKeyOut = 0;

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_F3:
		case FKB_SF3:
		case FKB_ALT_F3:
		case FKB_ESCAPE:
		case FKB_ENTER:
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		case 'V':
		case 'v':
		{
			char		pucTmpBuf[ 256];
			FLMUINT	uiTmpLen = sizeof( pucTmpBuf);
			char		szFilePath [F_PATH_MAX_SIZE];

			if( RC_BAD( rc = GedGetNATIVE( pCurNd, pucTmpBuf, &uiTmpLen)))
			{
				goto Exit;
			}
			f_strcpy( szFilePath, (const char *)UserData);
			f_pathAppend( szFilePath, pucTmpBuf);
			pRecEditor->fileViewer( NULL, szFilePath, NULL);
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE f_RecEditorFileEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData)
{
	char			pucTmpBuf[ 256];
	FLMUINT		uiTextSize;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	switch( eEventType)
	{
		case F_RECEDIT_EVENT_GETDISPVAL:
		{
			DBE_VAL_INFO *		pValInfo = (DBE_VAL_INFO *)EventData;
			NODE *				pNd = pValInfo->pNd;
			NODE *				pCommentNd;
			RCODE					rc2;

			/*
			Tack on the first annotation to the end of
			the display value.
			*/

			if( RC_OK( pRecEditor->getSystemNode( pNd,
				F_RECEDIT_VALANNO_FIELD, 1, &pCommentNd)))
			{
				uiTextSize = sizeof( pucTmpBuf);
				if( RC_OK( rc2 = GedGetNATIVE( pCommentNd, 
					pucTmpBuf, &uiTextSize)) || uiTextSize > 0)
				{
					if( !f_strcmp( pucTmpBuf, "DIR"))
					{
						f_strcat( (char *)pValInfo->pucBuf, " <Directory>");
					}
				}
			}

			break;
		}
		default:
		{
			break;
		}
	}

	return( rc);
}

/***************************************************************************
Desc: 
****************************************************************************/
RCODE F_RecEditor::globalConfig(
	FLMUINT		uiOption)
{
	CS_CONTEXT *		pCSContext = NULL;
	FLMUINT				uiCSOp;
	eFlmConfigTypes	eConfigOp;
	RCODE					rc = FERR_OK;

	if( m_hDefaultDb == HFDB_NULL)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	if( m_hDefaultDb != HFDB_NULL)
	{
		pCSContext = ((FDB *)m_hDefaultDb)->pCSContext;
	}

	switch( uiOption)
	{
		case F_RECEDIT_CONFIG_STATS_START:
		{
			uiCSOp = FCS_OP_GLOBAL_STATS_START;
			eConfigOp = FLM_START_STATS;
			break;
		}

		case F_RECEDIT_CONFIG_STATS_STOP:
		{
			uiCSOp = FCS_OP_GLOBAL_STATS_STOP;
			eConfigOp = FLM_STOP_STATS;
			break;
		}

		case F_RECEDIT_CONFIG_STATS_RESET:
		{
			uiCSOp = FCS_OP_GLOBAL_STATS_RESET;
			eConfigOp = FLM_RESET_STATS;
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	if( !pCSContext)
	{
		if( RC_BAD( rc = FlmConfig( eConfigOp, 0, 0)))
		{
			goto Exit;
		}
	}
	else
	{
		FCL_WIRE		Wire( pCSContext, (FDB *)m_hDefaultDb);

		/*
		Send a C/S request
		*/

		if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_GLOBAL, uiCSOp)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		/* Read the response. */

		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);

Transmission_Error:

	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:	
*****************************************************************************/
class LocalLockInfo : public IF_LockInfoClient
{
public:

	LocalLockInfo( FTX_WINDOW * pWindow)
	{
		m_pWindow = pWindow;
	}

	FLMBOOL FLMAPI setLockCount(
		FLMUINT		uiTotalLocks)
	{
		FTXWinClear( m_pWindow);

		if( !uiTotalLocks)
		{
			FTXWinPrintf( m_pWindow, 
				"There are no entries in the lock table.\n\n", uiTotalLocks);
		}
		else if( uiTotalLocks == 1)
		{
			FTXWinPrintf( m_pWindow, 
				"There is 1 entry in the lock table:\n\n", uiTotalLocks);
		}
		else
		{
			FTXWinPrintf( m_pWindow, 
				"There are %u entries in the lock table:\n\n", uiTotalLocks);
		}
		return( TRUE);
	}

	FLMBOOL FLMAPI addLockInfo(
		FLMUINT		uiLockNum,
		FLMUINT		uiThreadID,
		FLMUINT		uiTime)
	{
		if( uiLockNum != 0)
		{
			FTXWinPrintf( m_pWindow, "   #: %-8u   Thread: %-8u   Time: %-8u\n",
				(unsigned)uiLockNum, (unsigned)uiThreadID, (unsigned)uiTime);
		}

		if( RC_OK( FTXWinTestKB( m_pWindow)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( m_pWindow, &uiChar);
			if( uiChar == FKB_ESCAPE)
			{
				return( FALSE);
			}
		}

		return( TRUE);
	}

private:

	FTX_WINDOW *		m_pWindow;
};

/*============================================================================
Desc:	Accepts a buffer with regular ascii, and 4 ascii chars representing 
      unicode and converts all the chars to unicode.	UC_MARKER defines the
		char which markes the beginning of a unicode sequence.
============================================================================*/
RCODE F_RecEditor::asciiUCMixToUC(
	char *			pucAscii,
	FLMUNICODE *	puzUnicode,
	FLMUINT 			uiMaxUniChars)
{
	char *		pucTmp;
	char *		pucTerm;
	char			pucNumBuf[ 32];
	FLMUINT		uiUniCount = 0;
	FLMUINT		uiValue;
	RCODE			rc = FERR_OK;

	flmAssert( uiMaxUniChars > 0);
	uiMaxUniChars--; // Leave space for the terminator

	while( uiUniCount < uiMaxUniChars && *pucAscii)
	{
		if( pucAscii[ 0] == '~' && pucAscii[ 1] == '[')
		{
			pucAscii += 2;
			if( (pucTerm = f_strchr( pucAscii, ']')) == NULL)
			{
				rc = RC_SET( FERR_CONV_ILLEGAL);
				goto Exit;
			}

			while( *pucAscii && *pucAscii != ']')
			{
				pucTmp = f_strchr( pucAscii, ' ');
				if( !pucTmp || pucTmp > pucTerm)
				{
					pucTmp = pucTerm;
				}

				f_memcpy( pucNumBuf, pucAscii, pucTmp - pucAscii);
				pucNumBuf[ pucTmp - pucAscii] = 0;

				if( RC_BAD( rc = getNumber( pucNumBuf, &uiValue, NULL)))
				{
					goto Exit;
				}

				puzUnicode[ uiUniCount++] = (FLMUNICODE)uiValue;
				pucAscii += (pucTmp - pucAscii);
				while( *pucAscii == ' ')
				{
					pucAscii++;
				}
			}

			if( *pucAscii == ']')
			{
				pucAscii++;
			}
		}
		else
		{
			puzUnicode[ uiUniCount++] = (FLMUNICODE)(*pucAscii);
			pucAscii++;
		}
	}

	puzUnicode[ uiUniCount] = 0;

Exit:

	return rc;
}

/*============================================================================
Desc:
============================================================================*/
RCODE F_RecEditor::UCToAsciiUCMix(
	FLMUNICODE *	puzUnicode,
	char *			pucAscii,
	FLMUINT			uiMaxAsciiChars)
{
	char			pucTmpBuf[ 32];
	FLMUINT		uiAsciiCount = 0;
	FLMBOOL		bEscaped = FALSE;
	RCODE			rc = FERR_OK;

	flmAssert( uiMaxAsciiChars > 0);
	uiMaxAsciiChars--; // Leave space for the terminator

	while( uiAsciiCount < uiMaxAsciiChars && *puzUnicode)
	{
		if( *puzUnicode >= 0x0020 && *puzUnicode <= 0x007E)
		{
			if( bEscaped)
			{
				pucAscii[ uiAsciiCount++] = ']';
				bEscaped = FALSE;
			}

			if( uiAsciiCount == uiMaxAsciiChars)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			pucAscii[ uiAsciiCount++] = (char)*puzUnicode;
		}
		else
		{
			if( !bEscaped)
			{
				if( (uiAsciiCount + 2) >= uiMaxAsciiChars)
				{
					rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				pucAscii[ uiAsciiCount++] = '~';
				pucAscii[ uiAsciiCount++] = '[';
				bEscaped = TRUE;
			}
			else
			{
				pucAscii[ uiAsciiCount++] = ' ';
			}

			pucAscii[ uiAsciiCount] = '\0';
			f_sprintf( (char *)pucTmpBuf, "0x%04X", (unsigned)*puzUnicode);

			if( (uiAsciiCount +f_strlen( pucTmpBuf)) >= uiMaxAsciiChars)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}
			
			f_strcat( &(pucAscii[ uiAsciiCount]), pucTmpBuf);
			uiAsciiCount += f_strlen( pucTmpBuf);
		}

		puzUnicode++;
	}

	if( bEscaped)
	{
		if( uiAsciiCount == uiMaxAsciiChars)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		pucAscii[ uiAsciiCount++] = ']';
		bEscaped = FALSE;
	}

	pucAscii[ uiAsciiCount] = '\0';

Exit:

	return rc;
}
