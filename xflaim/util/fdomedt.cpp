//------------------------------------------------------------------------------
// Desc: DOM editor class
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
#include "domedit.h"
#include "fxpath.h"

static const char * pszDomeditMonths [12] =
{
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec"
};

static const char * pszDomeditFullMonthNames [12] =
{
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December"
};


FSTATIC FLMBOOL domeditIsNum(
	const char *	pszToken,
	FLMUINT *		puiNum);

FSTATIC char * domeditSkipChars(
	const char *	pszStr,
	const char *	pszCharsToSkip);

FSTATIC FLMBOOL domeditStrToDate(
	const char *	s,
	FLMUINT *		puiYear,
	FLMUINT *		puiMonth,
	FLMUINT *		puiDay);

FSTATIC void domGetOutputFileName(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	FLMUINT			uiOutputFileNameBufSize);

FSTATIC RCODE makeNewRow(
	DME_ROW_INFO **			ppTmpRow,
	FLMUNICODE *				puzValue,
	FLMUINT64					ui64Id,
	FLMBOOL						bUseValue = FALSE);

FSTATIC FLMUINT unicodeStrLen(
	FLMUNICODE *		puzStr);

FSTATIC void asciiToUnicode(
	const char *		pszAsciiString,
	FLMUNICODE *		puzString);

FSTATIC RCODE unicodeToAscii(
	FLMUNICODE *	puzString,
	char *			pszString,
	FLMUINT			uiLength);

FSTATIC RCODE formatText(
	FLMUNICODE *		puzBuf,
	FLMBOOL				bQuoted,
	const char *		pszPreText,
	const char *		pszPostText,
	char **				ppszString);

FSTATIC RCODE formatDocumentNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags);

FSTATIC RCODE formatElementNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals,
	FLMUINT				uiFlags);

FSTATIC RCODE formatAttributeNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags);

FSTATIC RCODE formatDataNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags,
	const char *			pszPreText,
	const char *			pszPostText);

FSTATIC RCODE formatProcessingInstruction(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags);

FSTATIC RCODE formatRow(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags);

FSTATIC RCODE formatIndexKeyNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals);

FSTATIC void printNodeType(
	DME_ROW_INFO *		pRow,
	FTX_WINDOW *		pStatusWin);

FSTATIC void releaseRow(
	DME_ROW_INFO **	ppRow);

FSTATIC RCODE getDOMNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow);

FSTATIC RCODE f_KeyEditorKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

FSTATIC RCODE f_QueryEditorKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

FSTATIC RCODE f_ViewOnlyKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

FSTATIC RCODE f_IndexRangeDispHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals);

FSTATIC RCODE setupIndexRow(
	F_DataVector *		pKey,
	FLMUINT				uiElementNumber,
	FLMUINT				uiIndex,
	FLMUINT				uiFlag,
	DME_ROW_INFO **	ppTmpRow);

RCODE _domEditBackgroundThread(
	IF_Thread *			pThread);

RCODE domEditVerifyRun( void);

extern FLMBOOL								gv_bShutdown;

#define FLM_START_STATS
#define FLM_STOP_STATS
#define FLM_RESET_STATS

/****************************************************************************
Desc:	Default constructor
*****************************************************************************/
F_DomEditor::F_DomEditor( void)
{
	m_pEditWindow = NULL;
	m_pEditStatusWin = NULL;
	m_uiEditCanvasRows = 0;
	m_bSetupCalled = FALSE;
	m_pDisplayHook = (F_DOMEDIT_DISP_HOOK)F_DomEditorDefaultDispHook;
	m_pNameTable = NULL;
	m_DisplayData = 0;
	m_pKeyHook = NULL;
	m_KeyData = 0;
	m_bMonochrome = FALSE;
	m_pDb = NULL;
	m_bOpenedDb = FALSE;
	m_pRowAnchor = NULL;
	m_pDocList = NULL;
	m_pCurDoc = NULL;
	m_pScrFirstRow = NULL;
	m_pScrLastRow = NULL;
	m_pCurRow = NULL;
	reset();
}


/****************************************************************************
Desc:	Destructor
*****************************************************************************/
F_DomEditor::~F_DomEditor( void)
{
	DME_ROW_INFO *		pTmpRow = m_pScrFirstRow;
	DME_ROW_INFO *		pDocList = m_pDocList;

	if (m_bOpenedDb && m_pDb != NULL)
	{
		m_pDb->Release();
		m_pDb = NULL;
	}

	if (m_pScrFirstRow)
	{
		releaseAllRows();
	}

	while (pDocList)
	{
		pTmpRow = pDocList->pNext;
		if (pDocList->puzValue)
		{
			f_free( &pDocList->puzValue);
		}
		if (pDocList->pDomNode)
		{
			if (!(pDocList->uiFlags & F_DOMEDIT_FLAG_ENDTAG))
			{
				pDocList->pDomNode->Release();
			}
			else
			{
				pDocList->pDomNode = NULL;
			}
		}
		f_free( &pDocList);
		pDocList = pTmpRow;
	}

	reset();
}

/****************************************************************************
Desc: Prepares the editor for use
*****************************************************************************/
RCODE F_DomEditor::Setup(
	FTX_SCREEN *		pScreen)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( pScreen != NULL);

	m_pScreen = pScreen;
	m_bSetupCalled = TRUE;

	return( rc);
}


/****************************************************************************
Desc:	Reset the DOMEditor object
*****************************************************************************/
void F_DomEditor::reset( void)
{
	m_pScrFirstRow = NULL;
	m_pScrLastRow = NULL;
	m_pCurRow = NULL;
	if (m_bOpenedDb && m_pDb != NULL)
	{
		m_pDb->Release();
	}
	m_bOpenedDb = FALSE;
	m_pDb = NULL;
	m_uiCollection = XFLM_DATA_COLLECTION;
	m_szTitle[ 0] = '\0';
	m_bReadOnly = FALSE;
	m_pbShutdown = NULL;
	m_pParent = NULL;
	m_uiCurRow = 0;
	m_uiEditCanvasRows = 0;
	m_pHelpHook = NULL;
	m_pEventHook = NULL;
	m_bDocList = FALSE;
	m_uiLastKey = 0;
	m_uiULX = 0;
	m_uiULY = 0;
	m_uiLRX = 0;
	m_uiLRY = 0;
	m_uiNumRows = 0;

	if( m_pRowAnchor)
	{
		f_free( &m_pRowAnchor);
	}

	m_pDocList = NULL;
	m_pCurDoc = NULL;

	if (m_pNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}
}

/****************************************************************************
Name:	insertRecord
Desc:
*****************************************************************************/
RCODE F_DomEditor::insertRow(
	DME_ROW_INFO *		pRow,
	DME_ROW_INFO *		pStartRow)
{
	RCODE			rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = _insertRow( pRow, pStartRow)))
	{
		goto Exit;
	}

Exit:

	return( rc);

}


/****************************************************************************
Desc:	_insertRow
*****************************************************************************/
RCODE F_DomEditor::_insertRow(
	DME_ROW_INFO *		pRow,
	DME_ROW_INFO *		pStartRow)
{
	DME_ROW_INFO *		pTmpRow = NULL;
	RCODE					rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( m_pEventHook)
	{

		if( RC_BAD( rc = m_pEventHook( this, F_DOMEDIT_EVENT_RECINSERT,
			(void *)(pRow), m_EventData)))
		{
			goto Exit;
		}
	}

	if( !m_pScrFirstRow)
	{
		m_pScrFirstRow = pRow;
		m_pCurRow = m_pScrFirstRow;
		m_uiNumRows = 1;
	}
	else
	{
		if (pStartRow)
		{
			pTmpRow = pStartRow->pNext;
			pStartRow->pNext = pRow;
		}
		else
		{
			pTmpRow = m_pScrFirstRow;
		}
		pRow->pNext = pTmpRow;
		pRow->pPrev = pStartRow;
		if (pTmpRow)
		{
			pTmpRow->pPrev = pRow;
		}
		if (!pStartRow)
		{
			m_pScrFirstRow = pRow;
		}

		m_uiNumRows++;
	}

	// Adjust the last display row if needed.
	if ( m_uiNumRows == 1)
	{
		m_pScrLastRow = m_pScrFirstRow;
	}
	else
	{
		while (m_pScrLastRow->pNext)
		{
			m_pScrLastRow = m_pScrLastRow->pNext;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Name:	setTitle
Desc:
*****************************************************************************/
RCODE F_DomEditor::setTitle(
	const char *	pszTitle)
{
	RCODE			rc = NE_XFLM_OK;
	eColorType	uiBack;
	eColorType	uiFore;

	flmAssert( m_bSetupCalled == TRUE);

	m_szTitle[ 0] = ' ';
	f_strncpy( &m_szTitle[ 1], pszTitle, F_DOMEDIT_MAX_TITLE_SIZE - 2);
	f_strcat( m_szTitle, " ");
	m_szTitle[ F_DOMEDIT_MAX_TITLE_SIZE] = '\0';

	if (m_pEditWindow)
	{
		uiBack = m_bMonochrome ? FLM_BLACK : FLM_BLUE;
		uiFore = FLM_WHITE;
		FTXWinSetTitle( m_pEditWindow, m_szTitle, uiBack, uiFore);
	}

	return( rc);
}


/****************************************************************************
Desc:	Set the default source Collection...
*****************************************************************************/
RCODE F_DomEditor::setSource(
	F_Db *		pDb,
	FLMUINT		uiCollection)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if (m_bOpenedDb && m_pDb != NULL && pDb != m_pDb)
	{
		m_pDb->Release();
		m_bOpenedDb = FALSE;
	}

	if (m_pDb != pDb)
	{
		if (m_pDb)
		{
			m_pDb->Release();
		}
		m_pDb = pDb;
	}
	m_pDb->AddRef();

	if (pDb)
	{
		m_bOpenedDb = TRUE;
	}
	
	m_uiCollection = uiCollection;

	return( rc);
}

/****************************************************************************
Desc:	QueryStatus class.
*****************************************************************************/
class EditQueryStatus : public IF_QueryStatus
{
public:

#define LABEL_WIDTH					30
#define LABEL_COLUMN					2
#define DATA_COLUMN					(LABEL_COLUMN + LABEL_WIDTH + 1)
#define QUERY_LINE					2
#define SOURCE_LINE					(QUERY_LINE + 1)
#define USING_LINE					(SOURCE_LINE + 1)
#define COST_LINE						(USING_LINE + 1)
#define VERIFY_LINE					(COST_LINE + 1)
#define NODE_MATCH_LINE				(VERIFY_LINE + 1)
#define CAN_COMPARE_KEY_LINE		(NODE_MATCH_LINE + 1)
#define KEYS_READ_LINE				(CAN_COMPARE_KEY_LINE + 1)
#define DUP_KEYS_LINE				(KEYS_READ_LINE + 1)
#define KEYS_PASSED_LINE			(DUP_KEYS_LINE + 1)
#define NODES_READ_LINE				(KEYS_PASSED_LINE + 1)
#define NODES_TESTED_LINE			(NODES_READ_LINE + 1)
#define NODES_PASSED_LINE			(NODES_TESTED_LINE + 1)
#define DOCUMENTS_READ_LINE		(NODES_PASSED_LINE + 1)
#define DOCUMENTS_DUP_LINE			(DOCUMENTS_READ_LINE + 1)
#define NODES_FAILED_VAL_LINE		(DOCUMENTS_DUP_LINE + 1)
#define DOCS_FAILED_VAL_LINE		(NODES_FAILED_VAL_LINE + 1)
#define DOCUMENTS_PASSED_LINE		(DOCS_FAILED_VAL_LINE + 1)
#define TOT_DOCS_READ_LINE			(DOCUMENTS_PASSED_LINE + 2)
#define TOT_DOCS_PASSED_LINE		(TOT_DOCS_READ_LINE + 1)
#define CAN_RETRIEVE_DOCS_LINE	(TOT_DOCS_PASSED_LINE + 1)
#define MESSAGE_LINE					(CAN_RETRIEVE_DOCS_LINE + 2)

	EditQueryStatus()
	{
		m_pWindow = NULL;
		m_uiSourceCnt = 0;
		m_pszQuery = NULL;
		f_memset( &m_optInfo, 0, sizeof( XFLM_OPT_INFO));
		m_ui64TotalDocsRead = 0;
		m_ui64TotalDocsPassed = 0;
		m_bCanRetrieveDocs = FALSE;
		m_bKeepResults = TRUE;
	}

	virtual ~EditQueryStatus()
	{
		if (m_pWindow)
		{
			FTXWinFree( &m_pWindow);
		}
		if (m_pszQuery)
		{
			f_free( &m_pszQuery);
		}
	}

	RCODE XFLAPI queryStatus(
		XFLM_OPT_INFO *	pOptInfo);

	RCODE XFLAPI newSource(
		XFLM_OPT_INFO *	pOptInfo);

	RCODE XFLAPI resultSetStatus(
		FLMUINT64	ui64TotalDocsRead,
		FLMUINT64	ui64TotalDocsPassed,
		FLMBOOL		bCanRetrieveDocs);
		
	RCODE XFLAPI resultSetComplete(
		FLMUINT64	ui64TotalDocsRead,
		FLMUINT64	ui64TotalDocsPassed);
		
	void createQueryStatusWindow(
		FTX_SCREEN *		pScreen,
		eColorType			uiBack,
		eColorType			uiFore,
		char *				pszQuery);

	RCODE testEscape(
		FLMUINT		uiSourceCnt,
		FLMUINT *	puiChar);

	FINLINE FLMBOOL keepResults( void)
	{
		return( m_bKeepResults);
	}

	void refreshStatus(
		FLMBOOL				bRefreshAll,
		FLMUINT				uiSource,
		FLMUINT				uiSourceCnt,
		XFLM_OPT_INFO *	pOptInfo);
		
	void refreshResultSetStatus(
		FLMBOOL				bRefreshAll,
		FLMUINT64			ui64TotalDocsRead,
		FLMUINT64			ui64TotalDocsPassed,
		FLMBOOL				bCanRetrieveDocs);

	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_QueryStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_QueryStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_QueryStatus::Release());
	}

private:

	void outputLabel(
		FLMUINT			uiRow,
		const char *	pszLabel);

	void outputStr(
		FLMUINT			uiRow,
		const char *	pszValue);

	FINLINE void outputBool(
		FLMUINT			uiRow,
		FLMBOOL			bBool)
	{
		outputStr( uiRow, (char *)(bBool ? (char *)"YES" : (char *)"NO"));
	}

	void outputUINT(
		FLMUINT	uiRow,
		FLMUINT	uiValue);

	void outputUINT64(
		FLMUINT		uiRow,
		FLMUINT64	ui64Value);

	FTX_WINDOW *	m_pWindow;
	FLMUINT			m_uiNumCols;
	FLMUINT			m_uiNumRows;
	FLMUINT			m_uiSourceCnt;
	XFLM_OPT_INFO	m_optInfo;
	FLMUINT64		m_ui64TotalDocsRead;
	FLMUINT64		m_ui64TotalDocsPassed;
	FLMBOOL			m_bCanRetrieveDocs;
	char *			m_pszQuery;
	FLMBOOL			m_bKeepResults;
};

/****************************************************************************
Desc: Displays a message window
*****************************************************************************/
void EditQueryStatus::createQueryStatusWindow(
	FTX_SCREEN *		pScreen,
	eColorType			uiBack,
	eColorType			uiFore,
	char *				pszQuery)
{
	FLMBOOL	bOk = FALSE;
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;
	FLMUINT	uiQueryStrLen = f_strlen( pszQuery);

	FTXScreenGetSize( pScreen, &uiNumCols, &uiNumRows);
	m_uiNumCols = uiNumCols - 8;
	m_uiNumRows = uiNumRows - 4;

	if( RC_BAD( FTXWinInit( pScreen, m_uiNumCols, m_uiNumRows, &m_pWindow)))
	{
		goto Exit;
	}

	FTXWinSetScroll( m_pWindow, FALSE);
	FTXWinSetCursorType( m_pWindow, FLM_CURSOR_INVISIBLE);
	FTXWinSetBackFore( m_pWindow, uiBack, uiFore);
	FTXWinClear( m_pWindow);
	FTXWinDrawBorder( m_pWindow);

	FTXWinMove( m_pWindow, (FLMUINT)((uiNumCols - m_uiNumCols) / 2),
		(FLMUINT)((uiNumRows - m_uiNumRows) / 2));

	FTXWinOpen( m_pWindow);

	if (RC_BAD( f_alloc( uiQueryStrLen + 1, &m_pszQuery)))
	{
		goto Exit;
	}
	
	f_memcpy( m_pszQuery, pszQuery, uiQueryStrLen + 1);
	
	if (uiQueryStrLen > m_uiNumCols - 2 - DATA_COLUMN)
	{
		m_pszQuery [m_uiNumCols - 2 - DATA_COLUMN] = 0;
	}

	FTXRefresh();
	bOk = TRUE;

Exit:

	if (!bOk && m_pWindow)
	{
		FTXWinFree( &m_pWindow);
	}
}

/****************************************************************************
Desc:	Output a label on the query status screen.
*****************************************************************************/
void EditQueryStatus::outputLabel(
	FLMUINT			uiRow,
	const char *	pszLabel
	)
{
	char		szLabel [50];
	FLMUINT	uiStrLen = f_strlen( pszLabel);

	f_memset( szLabel, '.', LABEL_WIDTH);
	szLabel [LABEL_WIDTH] = ' ';
	szLabel [LABEL_WIDTH + 1] = 0;
	if (uiStrLen > LABEL_WIDTH)
	{
		uiStrLen = LABEL_WIDTH;
	}
	f_memcpy( szLabel, pszLabel, uiStrLen);
	FTXWinSetCursorPos( m_pWindow, LABEL_COLUMN, uiRow);
	FTXWinPrintf( m_pWindow, "%s", szLabel);
}

/****************************************************************************
Desc:	Output a string value on the query status screen.
*****************************************************************************/
void EditQueryStatus::outputStr(
	FLMUINT			uiRow,
	const char *	pszValue
	)
{
	FTXWinSetCursorPos( m_pWindow, DATA_COLUMN, uiRow);
	FTXWinPrintf( m_pWindow, "%s", pszValue);
	FTXWinClearToEOL( m_pWindow);
}

/****************************************************************************
Desc:	Output a FLMUINT value on the query status screen.
*****************************************************************************/
void EditQueryStatus::outputUINT(
	FLMUINT	uiRow,
	FLMUINT	uiValue
	)
{
	char	szValue [20];

	f_sprintf( szValue, "%,u", (unsigned)uiValue);
	outputStr( uiRow, szValue);
}

/****************************************************************************
Desc:	Output a FLMUINT64 value on the query status screen.
*****************************************************************************/
void EditQueryStatus::outputUINT64(
	FLMUINT		uiRow,
	FLMUINT64	ui64Value
	)
{
	char	szValue [60];

	f_sprintf( szValue, "%,I64u", ui64Value);
	outputStr( uiRow, szValue);
}

/****************************************************************************
Desc:	Refresh query status screen
*****************************************************************************/
void EditQueryStatus::refreshStatus(
	FLMBOOL				bRefreshAll,
	FLMUINT				uiSource,
	FLMUINT				uiSourceCnt,
	XFLM_OPT_INFO *	pOptInfo)
{
	char	szTmp [60];

	if (bRefreshAll)
	{
		f_memcpy( &m_optInfo, pOptInfo, sizeof( XFLM_OPT_INFO));
		outputLabel( QUERY_LINE, "Query");
		outputStr( QUERY_LINE, m_pszQuery);
		outputLabel( SOURCE_LINE, "Source");
		if (!uiSourceCnt)
		{
			outputUINT( SOURCE_LINE, uiSource);
		}
		else
		{
			f_sprintf( szTmp, "%u of %u", (unsigned)uiSource,
					(unsigned)uiSourceCnt);
			outputStr( SOURCE_LINE, szTmp);
		}
		outputLabel( COST_LINE, "  Cost");
		outputLabel( VERIFY_LINE, "  Must Verify Path");
		outputLabel( NODE_MATCH_LINE, "  Do Node Match");
		outputLabel( CAN_COMPARE_KEY_LINE, "  Can Compare Key");
		switch (m_optInfo.eOptType)
		{
			case XFLM_QOPT_USING_INDEX:
				outputLabel( USING_LINE, "Using Index");
				if (m_optInfo.szIxName [0])
				{
					FLMUINT	uiStrLen = f_strlen( (const char *)m_optInfo.szIxName);
					if (uiStrLen > m_uiNumCols - 2 - DATA_COLUMN)
					{
						m_optInfo.szIxName [m_uiNumCols - 2 - DATA_COLUMN] = 0;
					}
					outputStr( USING_LINE, (char *)m_optInfo.szIxName);
				}
				else
				{
					f_sprintf( szTmp, "#%u", (unsigned)m_optInfo.uiIxNum);
					outputStr( USING_LINE, szTmp);
				}
				outputUINT( COST_LINE, m_optInfo.uiCost);
				outputBool( VERIFY_LINE, m_optInfo.bMustVerifyPath);
				outputBool( NODE_MATCH_LINE, m_optInfo.bDoNodeMatch);
				outputBool( CAN_COMPARE_KEY_LINE, m_optInfo.bCanCompareOnKey);
				break;
			case XFLM_QOPT_FULL_COLLECTION_SCAN:
				outputLabel( USING_LINE, "Using");
				outputStr( USING_LINE, "Collection Scan");
				outputStr( COST_LINE, "N/A");
				outputStr( VERIFY_LINE, "N/A");
				outputStr( NODE_MATCH_LINE, "N/A");
				outputStr( CAN_COMPARE_KEY_LINE, "N/A");
				break;
			case XFLM_QOPT_SINGLE_NODE_ID:
				outputLabel( USING_LINE, "Using Node ID");
				f_sprintf( szTmp, "#%I64u", m_optInfo.ui64NodeId);
				outputStr( USING_LINE, szTmp);
				outputUINT( COST_LINE, m_optInfo.uiCost);
				outputBool( VERIFY_LINE, m_optInfo.bMustVerifyPath);
				outputStr( NODE_MATCH_LINE, "N/A");
				outputStr( CAN_COMPARE_KEY_LINE, "N/A");
				break;
			case XFLM_QOPT_NODE_ID_RANGE:
				outputLabel( USING_LINE, "Using Node ID Range");
				f_sprintf( szTmp, "#%I64u to %I64u", m_optInfo.ui64NodeId,
						m_optInfo.ui64EndNodeId);
				outputStr( USING_LINE, szTmp);
				outputUINT( COST_LINE, m_optInfo.uiCost);
				outputBool( VERIFY_LINE, m_optInfo.bMustVerifyPath);
				outputStr( NODE_MATCH_LINE, "N/A");
				outputStr( CAN_COMPARE_KEY_LINE, "N/A");
				break;
			default:
				break;
		}
		outputLabel( KEYS_READ_LINE, "Keys Read");
		outputLabel( DUP_KEYS_LINE, "Keys Elim By Dup Docs");
		outputLabel( KEYS_PASSED_LINE, "Keys Passed");
		outputLabel( NODES_READ_LINE, "Nodes Read");
		outputLabel( NODES_TESTED_LINE, "Nodes Tested");
		outputLabel( NODES_PASSED_LINE, "Nodes Passed");
		outputLabel( DOCUMENTS_READ_LINE, "Documents Read");
		outputLabel( DOCUMENTS_DUP_LINE, "Dup Docs Eliminated");
		outputLabel( NODES_FAILED_VAL_LINE, "Nodes Failed Validation");
		outputLabel( DOCS_FAILED_VAL_LINE, "Docs Failed Validation");
		outputLabel( DOCUMENTS_PASSED_LINE, "Documents Passed");
		refreshResultSetStatus( TRUE, m_ui64TotalDocsRead, m_ui64TotalDocsPassed,
						m_bCanRetrieveDocs);
	}
	if (bRefreshAll || pOptInfo->ui64KeysRead != m_optInfo.ui64KeysRead)
	{
		m_optInfo.ui64KeysRead = pOptInfo->ui64KeysRead;
		outputUINT64( KEYS_READ_LINE, m_optInfo.ui64KeysRead);
	}
	if (bRefreshAll || pOptInfo->ui64KeyHadDupDoc != m_optInfo.ui64KeyHadDupDoc)
	{
		m_optInfo.ui64KeyHadDupDoc = pOptInfo->ui64KeyHadDupDoc;
		outputUINT64( DUP_KEYS_LINE, m_optInfo.ui64KeyHadDupDoc);
	}
	if (bRefreshAll || pOptInfo->ui64KeysPassed != m_optInfo.ui64KeysPassed)
	{
		m_optInfo.ui64KeysPassed = pOptInfo->ui64KeysPassed;
		outputUINT64( KEYS_PASSED_LINE, m_optInfo.ui64KeysPassed);
	}
	if (bRefreshAll || pOptInfo->ui64NodesRead != m_optInfo.ui64NodesRead)
	{
		m_optInfo.ui64NodesRead = pOptInfo->ui64NodesRead;
		outputUINT64( NODES_READ_LINE, m_optInfo.ui64NodesRead);
	}
	if (bRefreshAll || pOptInfo->ui64NodesTested != m_optInfo.ui64NodesTested)
	{
		m_optInfo.ui64NodesTested = pOptInfo->ui64NodesTested;
		outputUINT64( NODES_TESTED_LINE, m_optInfo.ui64NodesTested);
	}
	if (bRefreshAll || pOptInfo->ui64NodesPassed != m_optInfo.ui64NodesPassed)
	{
		m_optInfo.ui64NodesPassed = pOptInfo->ui64NodesPassed;
		outputUINT64( NODES_PASSED_LINE, m_optInfo.ui64NodesPassed);
	}
	if (bRefreshAll || pOptInfo->ui64DocsRead != m_optInfo.ui64DocsRead)
	{
		m_optInfo.ui64DocsRead = pOptInfo->ui64DocsRead;
		outputUINT64( DOCUMENTS_READ_LINE, m_optInfo.ui64DocsRead);
	}
	if (bRefreshAll || pOptInfo->ui64DupDocsEliminated != m_optInfo.ui64DupDocsEliminated)
	{
		m_optInfo.ui64DupDocsEliminated = pOptInfo->ui64DupDocsEliminated;
		outputUINT64( DOCUMENTS_DUP_LINE, m_optInfo.ui64DupDocsEliminated);
	}
	if (bRefreshAll || pOptInfo->ui64NodesFailedValidation != m_optInfo.ui64NodesFailedValidation)
	{
		m_optInfo.ui64NodesFailedValidation = pOptInfo->ui64NodesFailedValidation;
		outputUINT64( NODES_FAILED_VAL_LINE, m_optInfo.ui64NodesFailedValidation);
	}
	if (bRefreshAll || pOptInfo->ui64DocsFailedValidation != m_optInfo.ui64DocsFailedValidation)
	{
		m_optInfo.ui64DocsFailedValidation = pOptInfo->ui64DocsFailedValidation;
		outputUINT64( DOCS_FAILED_VAL_LINE, m_optInfo.ui64DocsFailedValidation);
	}
	if (bRefreshAll || pOptInfo->ui64DocsPassed != m_optInfo.ui64DocsPassed)
	{
		m_optInfo.ui64DocsPassed = pOptInfo->ui64DocsPassed;
		outputUINT64( DOCUMENTS_PASSED_LINE, m_optInfo.ui64DocsPassed);
	}
}

/****************************************************************************
Desc:	Refresh query status screen
*****************************************************************************/
void EditQueryStatus::refreshResultSetStatus(
	FLMBOOL				bRefreshAll,
	FLMUINT64			ui64TotalDocsRead,
	FLMUINT64			ui64TotalDocsPassed,
	FLMBOOL				bCanRetrieveDocs)
{
	if (bRefreshAll)
	{
		m_ui64TotalDocsRead = ui64TotalDocsRead;
		m_ui64TotalDocsPassed = ui64TotalDocsPassed;
		m_bCanRetrieveDocs = bCanRetrieveDocs;
		outputLabel( TOT_DOCS_READ_LINE, "Total Docs Read");
		outputLabel( TOT_DOCS_PASSED_LINE, "Total Docs Passed");
		outputLabel( CAN_RETRIEVE_DOCS_LINE, "Can Retrieve Docs");
	}
	if (bRefreshAll || ui64TotalDocsRead != m_ui64TotalDocsRead)
	{
		m_ui64TotalDocsRead = ui64TotalDocsRead;
		outputUINT64( TOT_DOCS_READ_LINE, m_ui64TotalDocsRead);
	}
	if (bRefreshAll || ui64TotalDocsPassed != m_ui64TotalDocsPassed)
	{
		m_ui64TotalDocsPassed = ui64TotalDocsPassed;
		outputUINT64( TOT_DOCS_PASSED_LINE, m_ui64TotalDocsPassed);
	}
	if (bRefreshAll || bCanRetrieveDocs != m_bCanRetrieveDocs)
	{
		m_bCanRetrieveDocs = bCanRetrieveDocs;
		outputUINT64( CAN_RETRIEVE_DOCS_LINE, m_bCanRetrieveDocs);
	}
}

/****************************************************************************
Desc:	See if user pressed escape
*****************************************************************************/
RCODE EditQueryStatus::testEscape(
	FLMUINT		uiSourceCnt,
	FLMUINT *	puiChar)
{
	RCODE		rc = NE_XFLM_OK;

	if (m_pWindow)
	{
		if (uiSourceCnt)
		{
			FTXWinSetCursorPos( m_pWindow, LABEL_COLUMN, MESSAGE_LINE);
			if (uiSourceCnt == 1)
			{
				FTXWinPrintf( m_pWindow, "Press any character to show query results: ");
			}
			else
			{
				FTXWinPrintf( m_pWindow, "N,P=Show Next/Prev Source, other=Show query results: ");
			}
			FTXWinInputChar( m_pWindow, puiChar);
		}
		else if( RC_OK( FTXWinTestKB( m_pWindow)))
		{
			FLMUINT		uiChar;

			FTXWinInputChar( m_pWindow, &uiChar);

			if (uiChar == FKB_ESCAPE)
			{
				FTXWinSetCursorPos( m_pWindow, LABEL_COLUMN, MESSAGE_LINE);
				FTXWinPrintf( m_pWindow,
					"Escape pressed, exit? (Y=Show results, ESC=quit): ");
				if( RC_BAD( FTXWinInputChar( m_pWindow, &uiChar)) ||
					uiChar == FKB_ESCAPE)
				{
					rc = RC_SET( NE_XFLM_USER_ABORT);
					m_bKeepResults = FALSE;
					goto Exit;
				}
				else if (uiChar == 'Y' || uiChar == 'y')
				{
					rc = RC_SET( NE_XFLM_USER_ABORT);
					goto Exit;
				}
				else
				{
					FTXWinSetCursorPos( m_pWindow, LABEL_COLUMN, MESSAGE_LINE);
					FTXWinPrintf( m_pWindow,
					"                                                  ");
				}
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Query status callback
*****************************************************************************/
RCODE  EditQueryStatus::queryStatus(
	XFLM_OPT_INFO *	pOptInfo)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_pWindow)
	{

		// See if the user pressed escape.

		if (RC_BAD( rc = testEscape( 0, NULL)))
		{
			goto Exit;
		}

		// Update our statistics display

		refreshStatus( FALSE, m_uiSourceCnt, 0, pOptInfo);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Query status callback
*****************************************************************************/
RCODE EditQueryStatus::newSource(
	XFLM_OPT_INFO *	pOptInfo)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_pWindow)
	{

		// See if the user pressed escape.

		if (RC_BAD( rc = testEscape( 0, NULL)))
		{
			goto Exit;
		}

		// Update our statistics display

		m_uiSourceCnt++;
		refreshStatus( TRUE, m_uiSourceCnt, 0, pOptInfo);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Query status callback
*****************************************************************************/
RCODE XFLAPI EditQueryStatus::resultSetStatus(
	FLMUINT64	ui64TotalDocsRead,
	FLMUINT64	ui64TotalDocsPassed,
	FLMBOOL		bCanRetrieveDocs)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_pWindow)
	{

		// See if the user pressed escape.

		if (RC_BAD( rc = testEscape( 0, NULL)))
		{
			goto Exit;
		}
		refreshResultSetStatus( FALSE, ui64TotalDocsRead, ui64TotalDocsPassed,
										bCanRetrieveDocs);
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc:	Query status callback
*****************************************************************************/
RCODE XFLAPI EditQueryStatus::resultSetComplete(
	FLMUINT64	ui64TotalDocsRead,
	FLMUINT64	ui64TotalDocsPassed)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_pWindow)
	{

		// See if the user pressed escape.

		if (RC_BAD( rc = testEscape( 0, NULL)))
		{
			goto Exit;
		}
		refreshResultSetStatus( FALSE, ui64TotalDocsRead, ui64TotalDocsPassed,
										TRUE);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Interact with the user...
*****************************************************************************/
RCODE F_DomEditor::interactiveEdit(
	FLMUINT			uiULX,
	FLMUINT			uiULY,
	FLMUINT			uiLRX,
	FLMUINT			uiLRY,
	FLMBOOL			bBorder,
	FLMUINT			uiStatusLines,
	FLMUINT			uiStartChar)
{
	DME_ROW_INFO *		pTmpRow;
	FLMBOOL				bRefreshEditWindow = FALSE;
	FLMBOOL				bRefreshStatusWindow = FALSE;
	FLMUINT				uiNumRows = 0;
	FLMUINT				uiNumCols = 0;
	FLMUINT				uiMaxRow = 0;
	FLMUINT				uiStartCol = 0;
	eDbTransType			eTransType = XFLM_NO_TRANS;
	FLMUINT				uiLoop;
	FLMUINT				uiCurFlags = 0;
	char					szAction[ 2];
	FLMUINT				uiTermChar;
	FLMUINT				uiHelpKey = uiStartChar;
	FLMBOOL				bDoneEditing = FALSE;
	RCODE					rc = NE_XFLM_OK;
	RCODE					tmpRc = NE_XFLM_OK;
	eColorType			uiFore;
	eColorType			uiBack;
	IF_Thread *			pIxManagerThrd = NULL;
	IF_Thread *			pMemManagerThrd = NULL;
	char *				pszQuery = NULL;
	FLMUINT				uiSzQueryBufSize;
	IF_ThreadMgr *		pThreadMgr = NULL;
	IF_DbSystem *		pDbSystem = NULL;

	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_pScreen != NULL);

	m_uiCurRow = 0;
	m_pScrFirstRow = NULL;
	m_uiLastKey = 0;
	uiSzQueryBufSize = 1024;
	
	if (RC_BAD( rc = f_alloc( uiSzQueryBufSize, &pszQuery)))
	{
		goto Exit;
	}
	
	*pszQuery = 0;
	
	if( RC_BAD( rc = FlmGetThreadMgr( &pThreadMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( !uiLRX && !uiLRY)
	{
		FTXScreenGetSize( m_pScreen, &uiNumCols, &uiNumRows);

		uiNumRows -= uiULY;
		uiNumCols -= uiULX;

	}
	else
	{
		uiNumRows = (uiLRY - uiULY) + 1;
		uiNumCols = (uiLRX - uiULX) + 1;
	}

	uiStartCol = uiULX;

	uiNumRows -= uiStatusLines; // Subtract however many lines are for the status area

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

	uiBack = m_bMonochrome ? FLM_BLACK : FLM_BLUE;
	uiFore = FLM_WHITE;
	
	FTXWinSetBackFore( m_pEditWindow, uiBack, uiFore);
	FTXWinClear( m_pEditWindow);

	if( bBorder)
	{
		FTXWinDrawBorder( m_pEditWindow);
	}

	FTXWinSetTitle( m_pEditWindow, m_szTitle, uiBack, uiFore);

	if( uiStatusLines)
	{
		if( RC_BAD( rc = FTXWinInit( m_pScreen, uiNumCols, uiStatusLines,
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
	m_uiEditCanvasCols = uiNumCols;
	uiMaxRow = uiNumRows - 1;

	bRefreshEditWindow = TRUE;
	bRefreshStatusWindow = TRUE;

	if( m_pDb != NULL)
	{
		eTransType = m_pDb->getTransType();
	}

	/*
	Call the callback to indicate that the interactive
	editor has been invoked
	*/

	if( m_pEventHook)
	{
		 m_pEventHook( this, F_DOMEDIT_EVENT_IEDIT,
			0, m_EventData);
	}

	FTXRefresh();
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
				if( RC_BAD( rc = m_pEventHook( this, F_DOMEDIT_EVENT_REFRESH,
					0, m_EventData)))
				{
					goto Exit;
				}
			}

			refreshEditWindow( &m_pScrFirstRow, m_pCurRow, &m_uiCurRow);
			FTXWinSetCursorPos( m_pEditWindow, 0, m_uiCurRow);
			bRefreshEditWindow = FALSE;
			bRefreshStatusWindow = TRUE;
		}

		if( m_pEditStatusWin && bRefreshStatusWindow)
		{
			/*
			Update the status window
			*/

			FTXWinSetBackFore( m_pEditStatusWin,
				m_bMonochrome ? FLM_LIGHTGRAY : FLM_GREEN,
				m_bMonochrome ? FLM_BLACK : FLM_WHITE);

			FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);

			getControlFlags( m_pCurRow, &uiCurFlags);
			if( !(uiCurFlags & F_DOMEDIT_FLAG_LIST_ITEM))
			{
				switch( eTransType)
				{
					case XFLM_UPDATE_TRANS:
					{
						FTXWinPrintf( m_pEditStatusWin, " | UTRANS");
						break;
					}
					case XFLM_READ_TRANS:
					{
						FTXWinPrintf( m_pEditStatusWin, " | RTRANS");
						break;
					}
					default:
					{
						break;
					}
				}
			}

			if( m_pCurRow)
			{
				FLMUINT64	ui64DocId = m_pCurRow->ui64DocId;
				FLMUINT64	ui64NodeId = m_pCurRow->ui64NodeId;
				FLMUINT		uiNameId = m_pCurRow->uiNameId;

				if ( ui64NodeId || ui64DocId || uiNameId)
				{
					FTXWinPrintf( m_pEditStatusWin, "[");

					if ( ui64DocId)
					{
						FTXWinPrintf( m_pEditStatusWin, "Doc:%,10I64u", ui64DocId);

						if ( ui64NodeId || uiNameId)
						{
							FTXWinPrintf( m_pEditStatusWin, " / ");
						}
					}

					if ( ui64NodeId)
					{
						FTXWinPrintf( m_pEditStatusWin, "Node:%,10I64u", ui64NodeId);

						if ( uiNameId)
						{
							FTXWinPrintf( m_pEditStatusWin, " / ");
						}
					}
					
					if ( uiNameId)
					{
						FTXWinPrintf( m_pEditStatusWin, "Name:%,13u", uiNameId);
					}

					FTXWinPrintf( m_pEditStatusWin, "]");

					if ( ui64NodeId)
					{
						printNodeType( m_pCurRow, m_pEditStatusWin);
					}
				}
			}

			FTXWinClearToEOL( m_pEditStatusWin);
			FTXRefresh();
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
					m_pKeyHook( this, m_pCurRow, uiChar, &uiChar, m_KeyData);
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

			getControlFlags( m_pCurRow, &uiCurFlags);
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

				case FKB_ENTER:
				{
					if( !m_pCurRow)
					{
						break;
					}

					if( uiCurFlags & F_DOMEDIT_FLAG_LIST_ITEM)
					{
						setControlFlags( m_pCurRow,
							uiCurFlags | F_DOMEDIT_FLAG_SELECTED);
						bDoneEditing = TRUE;
					}
					else if( !canEditRow( m_pCurRow))
					{
						displayMessage( "The row cannot be edited",
							RC_SET( NE_XFLM_ILLEGAL_OP), NULL, FLM_RED, FLM_WHITE);
					}
					else if( RC_BAD( tmpRc = editRow( m_uiCurRow, m_pCurRow)))
					{
						displayMessage( "The field could not be edited", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				case 'S':
				case 's':
				{
					bRefreshEditWindow = TRUE;
					break;
				}

				// Just view (edit without saving changes)
				case 'V':
				case 'v':
				{
					if( !m_pCurRow)
					{
						break;
					}

					if( RC_BAD( tmpRc = editRow( m_uiCurRow, m_pCurRow, TRUE)))
					{
						displayMessage( "The field could not be edited", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				// Expand the current row.
				case FKB_RIGHT:
				{
					DME_ROW_INFO *		pLastRow = NULL;
					if (!m_pCurRow->bExpanded)
					{
						if (RC_BAD( tmpRc = expandRow( m_pCurRow, TRUE, &pLastRow)))
						{
							displayMessage( "Error expanding current row", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
							break;
						}
						if (m_pCurRow->bExpanded)
						{
							if ((pLastRow == m_pCurRow) ||
								 (pLastRow->uiLevel != m_pCurRow->uiLevel))
							{
								// We need to mark the starting row since the expansion is
								// incomplete.
								if (m_pRowAnchor)
								{
									f_free( &m_pRowAnchor);
								}

								if( RC_BAD( rc = f_alloc( sizeof( DME_ROW_ANCHOR), &m_pRowAnchor)))
								{
									goto Exit;
								}

								f_memset( m_pRowAnchor, 0, sizeof(DME_ROW_ANCHOR));
								m_pRowAnchor->ui64NodeId = m_pCurRow->ui64NodeId;
								m_pRowAnchor->uiAnchorLevel = m_pCurRow->uiLevel;
								m_pRowAnchor->bSingleLevel = TRUE;
							}
						}
						bRefreshEditWindow = TRUE;
					}
					break;
				}

				// Expand the current row.
				case FKB_PLUS:
				{
					DME_ROW_INFO *		pLastRow = NULL;
					if (!m_pCurRow->bExpanded)
					{
						if (RC_BAD( rc = expandRow( m_pCurRow, FALSE, &pLastRow)))
						{
							goto Exit;
						}
						if (m_pCurRow->bExpanded)
						{
							if ((pLastRow == NULL) ||
								 (pLastRow->uiLevel != m_pCurRow->uiLevel))
							{
								// We need to mark the starting row since the expansion is
								// incomplete.
								if (m_pRowAnchor)
								{
									f_free( &m_pRowAnchor);
								}

								if( RC_BAD( rc = f_alloc( sizeof( DME_ROW_ANCHOR), &m_pRowAnchor)))
								{
									goto Exit;
								}
								f_memset( m_pRowAnchor, 0, sizeof(DME_ROW_ANCHOR));
								m_pRowAnchor->ui64NodeId = m_pCurRow->ui64NodeId;
								m_pRowAnchor->uiAnchorLevel = m_pCurRow->uiLevel;
								m_pRowAnchor->bSingleLevel = FALSE;
							}
						}
						bRefreshEditWindow = TRUE;
					}
					break;
				}

				// Collapse the current row.

				case FKB_LEFT:
				case FKB_MINUS:
				{
					if (m_pCurRow->bExpanded)
					{
						if (RC_BAD( tmpRc = collapseRow( &m_pCurRow)))
						{
							displayMessage( "Error collapsing current row", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
							break;
						}
						bRefreshEditWindow = TRUE;
					}
					break;
				}

				// Move field cursor to the next row

				case FKB_DOWN:
				{
					if (RC_BAD( tmpRc = getNextRow( m_pCurRow,
															  &pTmpRow,
															  TRUE,
															  m_pCurRow ? !m_pCurRow->bExpanded : FALSE)))
					{
						displayMessage( "Failed to retrieve a new row.", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
						break;
					}

					if (pTmpRow != NULL)
					{
						if( m_uiCurRow < uiMaxRow)
						{
							refreshRow( m_uiCurRow, m_pCurRow, FALSE);
							m_uiCurRow++;
							refreshRow( m_uiCurRow, pTmpRow, TRUE);
							bRefreshStatusWindow = TRUE;
						}
						else
						{
							bRefreshEditWindow = TRUE;
						}
						m_pCurRow = pTmpRow;
						checkDocument( &m_pCurDoc, m_pCurRow);
					}
					break;
				}

				// Move field cursor to the prior row

				case FKB_UP:
				{
					if (RC_BAD( tmpRc = getPrevRow( m_pCurRow,
															  &pTmpRow,
															  TRUE)))
					{
						displayMessage( "Failed to retrieve a previous row.", tmpRc,
							NULL, FLM_RED, FLM_WHITE);
						break;
					}
					if( pTmpRow != NULL)
					{
						if( m_uiCurRow > 0)
						{
							refreshRow( m_uiCurRow, m_pCurRow, FALSE);
							m_uiCurRow--;
							refreshRow( m_uiCurRow, pTmpRow, TRUE);
							bRefreshStatusWindow = TRUE;
						}
						else
						{
							bRefreshEditWindow = TRUE;
						}
						m_pCurRow = pTmpRow;
						// Did we change to another document?
						checkDocument( &m_pCurDoc, m_pCurRow);
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

						if (RC_BAD( tmpRc = getPrevRow( m_pCurRow,
																  &pTmpRow,
																  TRUE)))
						{
							displayMessage( "Failed to retrieve a previous row.", tmpRc,
													NULL, FLM_RED, FLM_WHITE);
							break;;
						}
						
						if( pTmpRow != NULL)
						{
							m_pCurRow = pTmpRow;
							// Did we change to another document?
							checkDocument( &m_pCurDoc, m_pCurRow);
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
				Page down
				*/

				case FKB_PGDN:
				{
					FLMBOOL	bIgnoreAnchor = !m_pCurRow->bExpanded;
					for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
					{
						if (RC_BAD( rc = getNextRow( m_pCurRow,
															  &pTmpRow,
															  TRUE,
															  bIgnoreAnchor)))
						{
							displayMessage( "Failed to retrieve a next row.", tmpRc,
												NULL, FLM_RED, FLM_WHITE);
							break;
						}
						if (pTmpRow != NULL)
						{
							m_pCurRow = pTmpRow;
							checkDocument( &m_pCurDoc, m_pCurRow);
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
						bIgnoreAnchor = FALSE;
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Go to the top of the buffer
				*/
#if 0  // Removed functionality
				case FKB_HOME:
				{
					m_pCurRow = m_pScrFirstRow;
					m_uiCurRow = 0;
					bRefreshEditWindow = TRUE;
					checkDocument( &m_pCurDoc, m_pCurRow);

					break;
				}

				/*
				Jump to the end of the buffer
				*/

				case FKB_END:
				{
					m_uiCurRow = uiMaxRow;
					for( ;;)
					{
						if ( RC_BAD( rc = getNextRow( m_pCurRow, &pTmpRow, FALSE)))
						{
							goto Exit;
						}
						if (pTmpRow != NULL)
						{
							m_pCurRow = pTmpRow;
							checkDocument( &m_pCurDoc, m_pCurRow);
						}
						else
						{
							break;
						}
					}

					setCurrentAtBottom();
					bRefreshEditWindow = TRUE;
					break;
				}
#endif

				case FKB_END:
				{
					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);


						FTXWinPrintf( m_pEditStatusWin,
							"Scanning to the end of the display ...");

						FTXWinClearToEOL( m_pEditStatusWin);
					}

					m_uiCurRow = uiMaxRow;
					for( ;;)
					{
						if ( RC_BAD( tmpRc = getNextRow( m_pCurRow,
																	&pTmpRow,
																	TRUE)))
						{
							displayMessage( "Failed to retrieve a next row.", tmpRc,
														NULL, FLM_RED, FLM_WHITE);
							break;
						}
						if (pTmpRow != NULL)
						{
							m_pCurRow = pTmpRow;
							// We need to check to see if we have moved to another document
							checkDocument( &m_pCurDoc, m_pCurRow);
						}
						else
						{
							break;
						}
					}

					setCurrentAtBottom();
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Jump to the top of the buffer
				*/

				case FKB_HOME:
				{
					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);


						FTXWinPrintf( m_pEditStatusWin,
							"Scanning to the beginning of the display ...");

						FTXWinClearToEOL( m_pEditStatusWin);
					}

					m_uiCurRow = uiMaxRow;
					for( ;;)
					{
						if ( RC_BAD( tmpRc = getPrevRow( m_pCurRow,
																	&pTmpRow,
																	TRUE)))
						{
							displayMessage( "Failed to retrieve a previous row.", tmpRc,
													NULL, FLM_RED, FLM_WHITE);
							break;
						}
						if (pTmpRow != NULL)
						{
							m_pCurRow = pTmpRow;
							// We need to check to see if we have moved to another document
							checkDocument( &m_pCurDoc, m_pCurRow);
						}
						else
						{
							m_pScrFirstRow = m_pCurRow;
							break;
						}
					}

					setCurrentAtTop();
					bRefreshEditWindow = TRUE;
					break;
				}

				// Add something
				case 'N':
				case 'n':
				case FKB_ALT_A:
				{
					if (RC_BAD( tmpRc = addSomething( &m_pCurRow)))
					{
						displayMessage( "Add/Insert operation failed",
							RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
						break;
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				// Display attributes
				case 'A':
				case 'a':
				{
					if (RC_BAD( tmpRc = displayAttributes( m_pCurRow)))
					{
						displayMessage(
							"Attributes could not be displayed",
							RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
					}
					break;
				}

				// Display node informatoin
				case 'D':
				case 'd':
				{
					if (RC_BAD( tmpRc = displayNodeInfo( m_pCurRow)))
					{
						displayMessage(
							"Node information could not be displayed",
							RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
					}
					break;
				}

				case 'X':
				case 'x':
				{
					if (RC_BAD( tmpRc = exportNode( m_pCurRow)))
					{
						displayMessage(
							"Node could not be exported",
							RC_SET( tmpRc), NULL, FLM_RED, FLM_WHITE);
					}
					break;
				}
				/*
				Index operations
				*/

				case 'I':
				case 'i':
				case FKB_ALT_I:
				{
					if( m_pDb == NULL)
					{
						break;
					}

					if( RC_BAD( tmpRc = indexList()))
					{
						displayMessage( "Index List Operation Failed",
											 RC_SET(tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				List all documents
				*/
				case 'L':
				case 'l':
				case FKB_ALT_L:
				{

					FLMUINT				uiCollection = m_uiCollection;
					FLMUINT64			ui64NodeId;
					char					szResponse[ 32];
					DME_ROW_INFO *		pDocList = m_pDocList;
					
					szResponse [0] = 0;

					if (!m_pDocList)
					{

						if( RC_BAD( tmpRc = selectCollection( &uiCollection,
																		  &uiTermChar)))
						{
							displayMessage( "Error getting collection",
												 RC_SET(tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
						
						if( uiTermChar != FKB_ENTER)
						{
							break;
						}

						m_uiCollection = uiCollection;
					}

					if( RC_BAD( tmpRc = retrieveDocumentList( uiCollection,
																			&ui64NodeId,
																			&uiTermChar)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Unable to retrieve document list",
											 RC_SET(tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}

					if( uiTermChar != FKB_ENTER)
					{
						break;
					}

					while ( pDocList && pDocList->ui64DocId != ui64NodeId)
					{
						pDocList = pDocList->pNext;
					}

					if (pDocList)
					{
						displayMessage( "Document already selected",
											 NE_XFLM_FAILURE,
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}

					// Retrieve the selected document
					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);

						FTXWinPrintf( m_pEditStatusWin,
							"Retrieving selected document from the database ...");

						FTXWinClearToEOL( m_pEditStatusWin);
					}

					if( RC_BAD( tmpRc = retrieveNodeFromDb( uiCollection, ui64NodeId, 0)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Unable to retrieve selected document",
											 RC_SET( tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}

					// Add the document to the document list.
					if (RC_BAD( tmpRc = addDocumentToList( uiCollection, ui64NodeId)))
					{
						if ( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Unable to add document to document list",
											 RC_SET(tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Retrieve a node
				*/

				case 'R':
				case 'r':
				case FKB_ALT_R:
				{
					FLMUINT		uiCollection;
					char			szResponse[ 32];
					FLMUINT		uiSrcLen;
					FLMUINT		uiNodeId;
					FLMUINT64	ui64Tmp;
					
					szResponse [0] = 0;
					requestInput( "[READ] Node Number",
									  szResponse,
									  sizeof( szResponse),
									  &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( (uiSrcLen = (FLMUINT)f_strlen( szResponse)) == 0)
					{
						uiNodeId = 0;
					}
					else
					{
						if( RC_BAD( tmpRc = getNumber( szResponse, &ui64Tmp, NULL)))
						{
							displayMessage( "Invalid node number",
												 RC_SET( tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
						uiNodeId = (FLMUINT)ui64Tmp;
					}


					if (!m_pDocList)
					{

						if( RC_BAD( tmpRc = selectCollection( &uiCollection,
																		  &uiTermChar)))
						{
							displayMessage( "Error getting collection",
												 RC_SET(tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
						
						if( uiTermChar != FKB_ENTER)
						{
							break;
						}

						m_uiCollection = uiCollection;
					}

					if( RC_BAD( tmpRc = retrieveNodeFromDb( uiCollection, uiNodeId, 0)))
					{
						if( m_pEditStatusWin)
						{
							FTXWinClearLine( m_pEditStatusWin, 0, 0);
						}
						displayMessage( "Unable to retrieve node",
											 RC_SET( tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}

					// VISIT: Check to see if this is a document, we can add it to the list.

					bRefreshEditWindow = TRUE;
					break;
				}

				/*
				Retrieve nodes via XPATH
				*/

				case 'F':
				case 'f':
				case FKB_ALT_F:
					doQuery( pszQuery, uiSzQueryBufSize);
					break;

				/*
				Clear all records from the current editor buffer.
				NOTE: This will discard all changes
				*/

				case 'C':
				case 'c':
				case FKB_ALT_C:
				{
					char			szResponse[ 2];

					szResponse [0] = 0;
					requestInput(
						"Clear buffer and discard modifications? (Y/N)",
						szResponse, 2, &uiTermChar);
					
					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}
					
					if( szResponse [0] == 'y' || szResponse [0] == 'Y')
					{
						setScrFirstRow( NULL);
						if (RC_BAD( rc = setCurrentRow(NULL, 0)))
						{
							goto Exit;
						}

						bRefreshEditWindow = TRUE;
					}
					break;
				}

				/*
				Global administration options (including statistics gathering)
				*/

				case '#':
				{
					szAction[ 0] = '\0';
					requestInput(
						"Statistics (b = begin, e = end, r = reset)",
						szAction, sizeof( szAction), &uiTermChar);

					if( uiTermChar == FKB_ESCAPE)
					{
						break;
					}

					if( m_pEditStatusWin)
					{
						FTXWinSetCursorPos( m_pEditStatusWin, 0, 0);
					}

					if( szAction [0] == 'b' || szAction [0] == 'B')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
								"Starting statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig(
							F_DOMEDIT_CONFIG_STATS_START)))
						{
							displayMessage( "Error Starting Statistics",
												 RC_SET( tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
					}
					else if( szAction [0] == 'e' || szAction [0] == 'E')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
											  "Stopping statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig(
							F_DOMEDIT_CONFIG_STATS_STOP)))
						{
							displayMessage( "Error Stopping Statistics",
												 RC_SET( tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
					}
					else if( szAction [0] == 'r' || szAction [0] == 'R')
					{
						if( m_pEditStatusWin)
						{
							FTXWinPrintf( m_pEditStatusWin,
											  "Resetting statistics ...");
							FTXWinClearToEOL( m_pEditStatusWin);
						}

						if( RC_BAD( tmpRc = globalConfig( F_DOMEDIT_CONFIG_STATS_RESET)))
						{
							displayMessage( "Error Resetting Statistics",
												 RC_SET( tmpRc),
												 NULL,
												 FLM_RED,
												 FLM_WHITE);
							break;
						}
					}
					else
					{
						displayMessage( "Invalid Request",
											 RC_SET( NE_XFLM_FAILURE),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}
					bRefreshStatusWindow = TRUE;
					break;
				}

				case '?':
				{
					showHelp( &uiHelpKey);
					break;
				}

				case FKB_F10:
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

				case FKB_F8: /* Index Manager */
				{
					char			szDbPath [F_PATH_MAX_SIZE];
					F_Db *		pTmpDb = NULL;

					if( m_pDb == NULL)
					{
						break;
					}

					if( pIxManagerThrd)
					{
						pIxManagerThrd->Release();
						pIxManagerThrd = NULL;
					}

					(void)m_pDb->getDbControlFileName( szDbPath, sizeof( szDbPath));

					if (RC_OK( tmpRc = pDbSystem->dbOpen( szDbPath,
																		NULL,
																		NULL, NULL, TRUE,
																		(IF_Db **)&pTmpDb)))
					{
						pThreadMgr->createThread( &pIxManagerThrd,
							flstIndexManagerThread,
							"index_manager", 0, 0, (void *)pTmpDb);
					}
					else
					{
						displayMessage( "Failed to open database",
											 RC_SET( tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
					}
					break;
				}

				case FKB_F9: /* Memory Manager */
				{
					if( pMemManagerThrd)
					{
						pMemManagerThrd->Release();
						pMemManagerThrd = NULL;
					}
					
					pThreadMgr->createThread( &pMemManagerThrd,
						flstMemoryManagerThread, "memory_manager");
					break;
				}

				case FKB_DELETE:
				{
  					if (RC_BAD( tmpRc = deleteRow( &m_pCurRow)))
					{
						displayMessage( "Delete operation failed",
											 RC_SET( tmpRc),
											 NULL,
											 FLM_RED,
											 FLM_WHITE);
						break;
					}
					bRefreshEditWindow = TRUE;
					break;
				}

				case FKB_ESCAPE:
				case 'Q':
				case 'q':
				case 'Z':
				case 'z':
				case FKB_ALT_Q:
				case FKB_ALT_Z:
				{
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

	f_free( &pszQuery);

	if( pIxManagerThrd)
	{
		pIxManagerThrd->Release();
	}

	if( pMemManagerThrd)
	{
		pMemManagerThrd->Release();
	}
	
	if( pThreadMgr)
	{
		pThreadMgr->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

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
Desc:	Draw the screen - refresh the display
*****************************************************************************/
RCODE F_DomEditor::refreshEditWindow(
	DME_ROW_INFO **	ppFirstRow,
	DME_ROW_INFO *		pCursorRow,
	FLMUINT *			puiCurRow)
{
	FLMUINT				uiLoop;
	FLMUINT				uiNumCols;
	FLMUINT				uiNumRows;
	DME_ROW_INFO *		pFirstRow;
	DME_ROW_INFO *		pTmpRow;
	FLMUINT				uiCurRow;
	FLMBOOL				bCurrentVisible = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	RCODE					rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if( pCursorRow == NULL)
	{
		*ppFirstRow = NULL;
		*puiCurRow = 0;
	}

	FTXWinGetCanvasSize( m_pEditWindow, &uiNumCols, &uiNumRows);

	/*
	See if the cursor row is already being displayed.
	*/

	uiCurRow = 0;
	pFirstRow = *ppFirstRow;
	pTmpRow = pFirstRow;
	for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
	{
		if( pCursorRow == pTmpRow)
		{
			uiCurRow = uiLoop;
			bCurrentVisible = TRUE;
			break;
		}

		if (RC_BAD( rc = getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
		if (pTmpRow == NULL)
		{
			break;
		}
		checkDocument(&m_pCurDoc, pTmpRow);
	}

		/*
	If the current node is not displayed, scroll the screen
	so that the node is visible.
	*/

	if( !bCurrentVisible)
	{
		uiCurRow = *puiCurRow;
		pFirstRow = pCursorRow;
		while( uiCurRow && pFirstRow)
		{
			if (RC_BAD( rc = getPrevRow( pFirstRow, &pTmpRow)))
			{
				goto Exit;
			}
			if( pTmpRow)
			{
				pFirstRow = pTmpRow;
			}
			uiCurRow--;
		}
	}

	*ppFirstRow = pFirstRow;
	*puiCurRow = uiCurRow;

	// Turn display refresh off temporarily

	FTXSetRefreshState( TRUE);

	// Start a transaction

	if( RC_BAD( rc = m_pDb->checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Refresh all rows of the edit window.  All rows beyond the end
	// of the tree are cleared.

	pTmpRow = *ppFirstRow;
	for( uiLoop = 0; uiLoop < uiNumRows; uiLoop++)
	{
		if( pTmpRow && pTmpRow == pCursorRow)
		{
			refreshRow( uiLoop, pTmpRow, TRUE);
			*puiCurRow = uiLoop;
		}
		else
		{
			refreshRow( uiLoop, pTmpRow, FALSE);
		}

		if( pTmpRow)
		{
			if (RC_BAD( rc = getNextRow( pTmpRow, &pTmpRow)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( bStartedTrans)
	{
		m_pDb->transAbort();
	}

	/*
	Re-enable display refresh
	*/

	FTXSetRefreshState( FALSE);
	return( rc);
}


/****************************************************************************
Desc: Get the domnode
*****************************************************************************/
RCODE F_DomEditor::getDomNode(
	FLMUINT64		ui64NodeId,
	FLMUINT			uiAttrNameId,
	F_DOMNode **	ppDomNode
	)
{
	RCODE			rc = NE_XFLM_OK;

	if (uiAttrNameId)
	{
		if (RC_BAD( rc = m_pDb->getAttribute( m_uiCollection, ui64NodeId,
											uiAttrNameId,
											(IF_DOMNode **)ppDomNode)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = m_pDb->getNode( m_uiCollection, ui64NodeId, ppDomNode)))
		{
			goto Exit;
		}
	}

Exit:

	return rc;

}
/****************************************************************************
Name:	F_DomEditorDefaultDispHook
Desc: Default line display format routine
*****************************************************************************/
RCODE F_DomEditorDefaultDispHook(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiFlags = 0;

	if( !pRow)
	{
		goto Exit;
	}

	pDomEditor->getControlFlags( pRow, &uiFlags);

	// Are we just displaying a value, or should we use the Dom?
	if (pRow->bUseValue)
	{
		rc = formatRow( pDomEditor, pRow, puiNumVals, uiFlags);
		goto Exit;
	}


	// Need to find out what type of node we have to display.
	if (RC_BAD( rc = getDOMNode( pDomEditor, pRow)))
	{
		goto Exit;
	}

	switch (pRow->eType)
	{
		case DOCUMENT_NODE:
		{
			if (RC_BAD( rc = formatDocumentNode( pDomEditor, pRow, puiNumVals, uiFlags)))
			{
				goto Exit;
			}
			break;
		}
		case ELEMENT_NODE:
		{

			// It is possible to have data embedded in the element node.

			if (uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA)
			{
				if (RC_BAD( rc = formatDataNode( pDomEditor, pRow, puiNumVals, uiFlags, NULL, NULL)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = formatElementNode( pDomEditor, pRow, puiNumVals, uiFlags)))
				{
					goto Exit;
				}
			}
			break;
		}
		case DATA_NODE:
		{
			if (RC_BAD( rc = formatDataNode( pDomEditor, pRow, puiNumVals, uiFlags, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}
		case CDATA_SECTION_NODE:
		{
			if (RC_BAD( rc = formatDataNode( pDomEditor, pRow, puiNumVals, uiFlags,
														"<![CDATA[", "]]>")))
			{
				goto Exit;
			}
			break;
		}
		case COMMENT_NODE:
		{
			if (RC_BAD( rc = formatDataNode( pDomEditor, pRow, puiNumVals, uiFlags,
														"<!--", "-->")))
			{
				goto Exit;
			}
			break;
		}
		case PROCESSING_INSTRUCTION_NODE:
		{
			if (RC_BAD( rc = formatProcessingInstruction( pDomEditor, pRow, puiNumVals, uiFlags)))
			{
				goto Exit;
			}
			break;
		}
		case ATTRIBUTE_NODE:
		{
			if (RC_BAD( rc = formatAttributeNode( pDomEditor, pRow, puiNumVals, uiFlags)))
			{
				goto Exit;
			}
			break;
		}
		case INVALID_NODE:
			break;  // Don't know just yet what to do, but will figure it out soon.
		default:
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}


Exit:
	// Don't hold on to the Dom node when finished.
	if ( pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}
	return( rc);
}


/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_DomEditor::refreshRow(
	FLMUINT				uiRow,
	DME_ROW_INFO *		pRow,
	FLMBOOL				bSelected)
{
	FLMUINT				uiNumCols;
	FLMUINT				uiNumRows;
	FLMUINT				uiNumVals;
	FLMUINT				uiLoop;
	RCODE					rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	FTXWinGetCanvasSize( m_pEditWindow, &uiNumCols, &uiNumRows);

	FTXWinSetCursorPos( m_pEditWindow, 0, uiRow);
	FTXWinClearLine( m_pEditWindow, 0, uiRow);

	if (!pRow)
	{
		goto Exit;
	}

	f_memset( m_dispColumns, 0, sizeof( m_dispColumns));
	uiNumVals = 0;

	/*
	Call the display formatter
	*/

	if( m_pDisplayHook)
	{
		if( RC_BAD( rc = m_pDisplayHook( this, pRow, &uiNumVals)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = F_DomEditorDefaultDispHook( this, pRow, &uiNumVals)))
		{
			goto Exit;
		}
	}

	for( uiLoop = 0; uiLoop < uiNumVals; uiLoop++)
	{
		FTXWinSetCursorPos( m_pEditWindow, m_dispColumns[ uiLoop].uiCol, uiRow);
		FTXWinCPrintf( m_pEditWindow, m_dispColumns[ uiLoop].uiBackground,
			m_dispColumns[ uiLoop].uiForeground, "%s", m_dispColumns[ uiLoop].szString);
	}

	if( bSelected)
	{	
		eColorType 	uiBackground = m_bMonochrome ? FLM_LIGHTGRAY : FLM_CYAN;
		eColorType 	uiForeground = m_bMonochrome ? FLM_BLACK : FLM_WHITE;
		
		FTXWinPaintRow( m_pEditWindow, &uiBackground, &uiForeground, uiRow);
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc: Repositions the cursor to display at the top of
		the editor window
*****************************************************************************/
void F_DomEditor::setCurrentAtTop( void)
{
	m_uiCurRow = 0;
	m_pCurRow = m_pScrFirstRow;
}


/****************************************************************************
Desc: Repositions the cursor (and current node) to display at the bottom of
		the editor window
*****************************************************************************/
void F_DomEditor::setCurrentAtBottom( void)
{
	FLMUINT	uiNumRows;

	flmAssert( m_pEditWindow != NULL);

	FTXWinGetCanvasSize( m_pEditWindow, NULL, &uiNumRows);
	uiNumRows--;

	m_pCurRow = m_pScrLastRow;

	m_uiCurRow = uiNumRows;
}

/****************************************************************************
Name:	displayMessage
Desc:
*****************************************************************************/
RCODE F_DomEditor::displayMessage(
	const char *		pszMessage,
	RCODE					rcOfMessage,
	FLMUINT *			puiTermChar,
	eColorType			uiBackground,
	eColorType			uiForeground)
{
	RCODE				rc = NE_XFLM_OK;
	char				szErr [20];
	
	f_sprintf( szErr, "Error=0x%04X", (unsigned)rcOfMessage);

	flmAssert( m_bSetupCalled == TRUE);

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	FTXDisplayMessage( m_pScreen, m_bMonochrome ? FLM_LIGHTGRAY : uiBackground,
		m_bMonochrome ? FLM_BLACK : uiForeground, pszMessage, szErr, puiTermChar);

	return( rc);
}

/****************************************************************************
Name:	openNewDb
Desc:
*****************************************************************************/
RCODE F_DomEditor::openNewDb( void)
{
	RCODE				rc = NE_XFLM_OK;
	char				szResponse [100];
	FLMUINT			uiChar;
	IF_DbSystem *	pDbSystem = NULL;

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	szResponse [0] = 0;
	flmAssert( m_pDb == NULL);
	
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
		
		if (RC_BAD( rc = pDbSystem->dbOpen( szResponse, NULL, NULL, NULL, TRUE,
			(IF_Db **)&m_pDb)))
		{
			displayMessage( "Unable to open database", rc,
					NULL, FLM_RED, FLM_WHITE);
			m_pDb = NULL;
			continue;
		}
		m_bOpenedDb = TRUE;
		break;
	}

Exit:

	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( rc);
}


/****************************************************************************
Name:	requestInput
Desc:
*****************************************************************************/
RCODE F_DomEditor::requestInput(
	const char *		pszMessage,
	char *				pszResponse,
	FLMUINT				uiMaxRespLen,
	FLMUINT *			puiTermChar)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiNumCols;
	FLMUINT				uiNumRows;
	FLMUINT				uiNumWinRows = 3;
	FLMUINT				uiNumWinCols;
	FTX_WINDOW *		pWindow = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	IF_FileSystem *	pFileSystem = NULL;

	flmAssert( m_bSetupCalled == TRUE);
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

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
		FTXWinPrintf( pWindow, "%s: ", pszMessage);

		if( RC_BAD( rc = FTXLineEdit( pWindow, pszResponse, 
			uiMaxRespLen, uiMaxRespLen, NULL, puiTermChar)))
		{
			goto Exit;
		}

		if( *puiTermChar == FKB_F1)
		{
			FLMUINT		uiBytesRead;
			char *		pszSrc;
			char *		pszDest;

			if( RC_BAD( rc = pFileSystem->openFile( pszResponse, FLM_IO_RDONLY,
				&pFileHdl)))
			{
				displayMessage( "Unable to open file", rc,
					NULL, FLM_RED, FLM_WHITE);
				continue;
			}

			if( RC_BAD( rc = pFileHdl->read( 0, uiMaxRespLen,
				pszResponse, &uiBytesRead)))
			{
				if( rc == NE_FLM_IO_END_OF_FILE)
				{
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}

			pFileHdl->Release();
			pFileHdl = NULL;
			pszResponse[ uiBytesRead] = '\0';
			
			// Convert newlines to spaces.  Multiple consecutive newlines
			// will be converted to a single space.
			
			pszSrc = pszDest = pszResponse;
			while (*pszSrc)
			{
				if (*pszSrc == '\r' || *pszSrc == '\n')
				{
					*pszDest = ' ';
					pszDest++;
					pszSrc++;
					while (*pszSrc == '\r' || *pszSrc == '\n')
					{
						pszSrc++;
					}
				}
				else
				{
					if (pszDest != pszSrc)
					{
						*pszDest = *pszSrc;
					}
					pszSrc++;
					pszDest++;
				}
			}
			*pszDest = 0;
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
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Name:	getControlFlags
Desc:
*****************************************************************************/
RCODE F_DomEditor::getControlFlags(
	DME_ROW_INFO *		pCurRow,
	FLMUINT *			puiFlags)
{
	RCODE			rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);
	if ( !pCurRow)
	{
		goto Exit;
	}

	*puiFlags = pCurRow->uiFlags;
Exit:
	return( rc);
}


/****************************************************************************
Name:	setControlFlags
Desc:
*****************************************************************************/
RCODE F_DomEditor::setControlFlags(
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiFlags)
{
	RCODE			rc = NE_XFLM_OK;
	
	flmAssert( m_bSetupCalled == TRUE);

	pCurRow->uiFlags = uiFlags;

	return( rc);
}


/****************************************************************************
Desc:	Get the previous row to display.
*****************************************************************************/
RCODE F_DomEditor::getPrevRow(
	DME_ROW_INFO *		pCurRow,
	DME_ROW_INFO **	ppPrevRow,
	FLMBOOL				bFetchPrevRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pPrevRow = NULL;
	DME_ROW_INFO *		pDocList = m_pCurDoc;
	F_DOMNode *			pChildDomNode = NULL;
	F_DOMNode *			pSiblingDomNode = NULL;
	F_DOMNode *			pParentDomNode = NULL;
	FLMBOOL				bCurRowDataLocal;

	flmAssert( m_bSetupCalled == TRUE);

	if (pCurRow == NULL)
	{
		goto Exit;
	}

	// Get the previous row.
	pPrevRow = pCurRow->pPrev;

	// Got it, we're done...
	if (pPrevRow)
	{
		goto Exit;
	}


	if (!bFetchPrevRow)
	{
		goto Exit;
	}

	// If we were on the first display line, we will need to see about fetching
	// the previous row
	if (pCurRow == m_pScrFirstRow)
	{
		if (m_bDocList)
		{
			if (RC_BAD( rc = getPrevTitle( pCurRow, &pPrevRow)))
			{
				goto Exit;
			}
			if (pPrevRow == NULL)
			{
				goto Exit;
			}
			goto InsertRow;
		}

		if (pCurRow->uiFlags & F_DOMEDIT_FLAG_NODOM)
		{
			goto Exit;
		}

		if (RC_BAD( rc = getDOMNode( this, pCurRow)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = pCurRow->pDomNode->isDataLocalToNode( m_pDb, &bCurRowDataLocal)))
		{
			goto Exit;
		}

		if ((pCurRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG) &&
			(!(pCurRow->uiFlags & F_DOMEDIT_FLAG_NOCHILD) || bCurRowDataLocal))
		{
			if (bCurRowDataLocal && pCurRow->pDomNode->getNodeType() == ELEMENT_NODE)
			{
				pChildDomNode = pCurRow->pDomNode;
				pChildDomNode->AddRef();
			}
			else if (RC_BAD( rc = pCurRow->pDomNode->getLastChild( m_pDb, (IF_DOMNode **)&pChildDomNode)))
			{
				flmAssert( rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND);
				goto Exit;
			}

			if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel+1,
													pChildDomNode,
													&pPrevRow)))
			{
				goto Exit;
			}

			if (!bCurRowDataLocal)
			{
				pPrevRow->uiFlags = pCurRow->uiFlags | F_DOMEDIT_FLAG_ENDTAG;
				if (RC_BAD( rc = pChildDomNode->isDataLocalToNode( m_pDb, &pPrevRow->bHasElementData)))
				{
					goto Exit;
				}
			}
			else
			{
				pPrevRow->uiFlags = pCurRow->uiFlags & ~F_DOMEDIT_FLAG_ENDTAG;
			}

			if (bCurRowDataLocal)
			{
				pPrevRow->uiFlags |= F_DOMEDIT_FLAG_ELEMENT_DATA;
			}

			if (m_pRowAnchor)
			{
				pPrevRow->bExpanded = !m_pRowAnchor->bSingleLevel;
			}

			goto InsertRow;
		}

		if (pCurRow->eType != DOCUMENT_NODE &&
			 !(pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
		{
			if (RC_BAD( rc = pCurRow->pDomNode->getPreviousSibling(
															m_pDb,
															(IF_DOMNode **)&pSiblingDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
			}

			if (RC_OK( rc))
			{
				// Build a new row.
				if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel,
														pSiblingDomNode,
														&pPrevRow)))
				{
					goto Exit;
				}

				pPrevRow->uiFlags = pCurRow->uiFlags & ~(F_DOMEDIT_FLAG_ENDTAG);
				if (m_pRowAnchor)
				{
					pPrevRow->bExpanded = !m_pRowAnchor->bSingleLevel && pPrevRow->bHasChildren;
				}
				if ( pPrevRow->bExpanded)
				{
					pPrevRow->uiFlags |= F_DOMEDIT_FLAG_ENDTAG;
				}
				goto InsertRow;
			}
		}

		if (!(pCurRow->uiFlags & F_DOMEDIT_FLAG_NOPARENT) ||
			 (bCurRowDataLocal && pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
		{
			if (bCurRowDataLocal && pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA)
			{
				pParentDomNode = pCurRow->pDomNode;
				pParentDomNode->AddRef();
			}
			else if (RC_BAD( rc = pCurRow->pDomNode->getParentNode(
														m_pDb,
														(IF_DOMNode **)&pParentDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
			}

			if (RC_OK( rc))
			{
				if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel - 1,
														pParentDomNode,
														&pPrevRow)))
				{
					goto Exit;
				}

				pPrevRow->uiFlags = pCurRow->uiFlags & ~(F_DOMEDIT_FLAG_ENDTAG | F_DOMEDIT_FLAG_ELEMENT_DATA);
				if (RC_BAD( rc = pParentDomNode->isDataLocalToNode( m_pDb, &pPrevRow->bHasElementData)))
				{
					goto Exit;
				}
				pPrevRow->bExpanded = TRUE;
				goto InsertRow;
			}
		}

		rc = NE_XFLM_OK;

		// Is there a previous document that we can display?
		// The pPrevRow should always be NULL at this point.

		flmAssert( pPrevRow == NULL);

		if (pDocList)
		{
			pDocList = pDocList->pPrev;
		}

		if (pDocList)
		{
			if (RC_BAD( rc = getDOMNode( this, pDocList)))
			{
				goto Exit;
			}

			// Build a new row.
			if (RC_BAD( rc = buildNewRow( -1,
													pDocList->pDomNode,
													&pPrevRow)))
			{
				goto Exit;
			}

			goto InsertRow;

		}
		else
		{
			goto Exit;
		}

InsertRow:

		// Insert at the front.
		if (RC_BAD( rc = insertRow( pPrevRow, NULL)))
		{
			goto Exit;
		}
		m_pCurRow = m_pScrFirstRow;

		// Need to remove the last row.
		if (m_uiNumRows > m_uiEditCanvasRows)
		{
			releaseLastRow();
		}

	}


Exit:


	if (pSiblingDomNode)
	{
		pSiblingDomNode->Release();
	}

	if (pChildDomNode)
	{
		pChildDomNode->Release();
	}

	if (pParentDomNode)
	{
		pParentDomNode->Release();
	}

	if (pCurRow && pCurRow->pDomNode)
	{
		pCurRow->pDomNode->Release();
		pCurRow->pDomNode = NULL;
	}

	if ( m_pCurDoc && m_pCurDoc->pDomNode)
	{
		m_pCurDoc->pDomNode->Release();
		m_pCurDoc->pDomNode = NULL;
	}

	if (pDocList && pDocList->pDomNode)
	{
		pDocList->pDomNode->Release();
		pDocList->pDomNode = NULL;
	}

	if (pPrevRow && pPrevRow->pDomNode)
	{
		pPrevRow->pDomNode->Release();
		pPrevRow->pDomNode = NULL;
	}

	*ppPrevRow = pPrevRow;

	return rc;
}


/****************************************************************************
Desc:	Get the next row.  If the current row is the last row being displayed,
		we need to also remove the first row.
*****************************************************************************/
RCODE F_DomEditor::getNextRow(
	DME_ROW_INFO *		pCurRow,
	DME_ROW_INFO **	ppNextRow,
	FLMBOOL				bFetchNextRow,
	FLMBOOL				bIgnoreAnchor
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pNextRow = NULL;
	DME_ROW_INFO *		pDocList = m_pCurDoc;
	F_DOMNode *			pChildDomNode = NULL;
	F_DOMNode *			pSiblingDomNode = NULL;
	F_DOMNode *			pParentDomNode = NULL;
	FLMBOOL				bCurRowDataLocal;

	flmAssert( m_bSetupCalled == TRUE);

	if (pCurRow == NULL)
	{
		goto Exit;
	}

	pNextRow = pCurRow->pNext;

	// Got it, we're done...
	if (pNextRow)
	{
		goto Exit;
	}


	if (!bFetchNextRow)
	{
		goto Exit;
	}

	// If we were on the last display line, we will need to see about fetching
	// the next row
	if (m_uiNumRows <= m_uiEditCanvasRows)
	{
		if (m_bDocList)
		{
			if (RC_BAD( rc = getNextTitle( pCurRow, &pNextRow)))
			{
				goto Exit;
			}
			if (pNextRow == NULL)
			{
				goto Exit;
			}
			goto InsertRow;
		}

		if (pCurRow->uiFlags & F_DOMEDIT_FLAG_NODOM)
		{
			goto Exit;
		}

		// If the current row was expanded or there is an anchor row, we will
		// need to check the Anchor to see how we are expanding it.  Either
		// deep or single level.
		if (RC_BAD( rc = getDOMNode( this, pCurRow)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pCurRow->pDomNode->isDataLocalToNode( m_pDb, &bCurRowDataLocal)))
		{
			goto Exit;
		}

		if (m_pRowAnchor && !bIgnoreAnchor && pCurRow->bExpanded)
		{
			if (( m_pRowAnchor->bSingleLevel &&
					m_pRowAnchor->uiAnchorLevel == pCurRow->uiLevel) ||
					(!m_pRowAnchor->bSingleLevel))
			{
				if (!(pCurRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG))
				{
					if (bCurRowDataLocal && pCurRow->pDomNode->getNodeType() == ELEMENT_NODE &&
						 !(pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
					{
						pChildDomNode = pCurRow->pDomNode;
						pChildDomNode->AddRef();
					}
					// Get the first child.
					else if (RC_BAD( rc = pCurRow->pDomNode->getFirstChild(
																m_pDb, (IF_DOMNode **)&pChildDomNode)))
					{
						if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
						{
							goto Exit;
						}

					}

					if (RC_OK( rc))
					{
						// Build a new row.
						if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel+1,
																pChildDomNode,
																&pNextRow)))
						{
							goto Exit;
						}

						pNextRow->uiFlags = pCurRow->uiFlags & ~(F_DOMEDIT_FLAG_ENDTAG);
						if (bCurRowDataLocal && pCurRow->pDomNode->getNodeType() == ELEMENT_NODE)
						{
							pNextRow->uiFlags |= F_DOMEDIT_FLAG_ELEMENT_DATA;
						}

						if (!bCurRowDataLocal)
						{
							if (RC_BAD( rc = pChildDomNode->isDataLocalToNode( m_pDb, &pNextRow->bHasElementData)))
							{
								goto Exit;
							}
						}

						pNextRow->bExpanded = !m_pRowAnchor->bSingleLevel;
						goto InsertRow;
					}
				}
			}
		}

		rc = NE_XFLM_OK;

		if (pCurRow->bExpanded && !bIgnoreAnchor)
		{
			if ((!(pCurRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG)) &&
				 (!(pCurRow->uiFlags & F_DOMEDIT_FLAG_NOCHILD) || bCurRowDataLocal))
			{
				if (bCurRowDataLocal && pCurRow->pDomNode->getNodeType() == ELEMENT_NODE &&
					 !(pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
				{
					pChildDomNode = pCurRow->pDomNode;
					pChildDomNode->AddRef();
				}
				// Get the first child.
				else if (RC_BAD( rc = pCurRow->pDomNode->getFirstChild(
													m_pDb, (IF_DOMNode **)&pChildDomNode)))
				{
					if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						goto Exit;
					}
				}

				if (RC_OK( rc))
				{
					// Build a new row.
					if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel+1,
															pChildDomNode,
															&pNextRow)))
					{
						goto Exit;
					}

					pNextRow->uiFlags = pCurRow->uiFlags & ~(F_DOMEDIT_FLAG_ENDTAG);
					if (bCurRowDataLocal && pCurRow->pDomNode->getNodeType() == ELEMENT_NODE)
					{
						pNextRow->uiFlags |= F_DOMEDIT_FLAG_ELEMENT_DATA;
					}

					if (m_pRowAnchor)
					{
						pNextRow->bExpanded = !m_pRowAnchor->bSingleLevel;
					}
					goto InsertRow;
				}
			}
		}

		rc = NE_XFLM_OK;

		if (pCurRow->eType != DOCUMENT_NODE)
		{
			if (RC_BAD( rc = pCurRow->pDomNode->getNextSibling(
															m_pDb,
															(IF_DOMNode **)&pSiblingDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
			}

			if (RC_OK( rc))
			{
				// Build a new row.
				if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel,
														pSiblingDomNode,
														&pNextRow)))
				{
					goto Exit;
				}

				pNextRow->uiFlags = pCurRow->uiFlags & ~(F_DOMEDIT_FLAG_ENDTAG);

				if (m_pRowAnchor)
				{
					pNextRow->bExpanded = !m_pRowAnchor->bSingleLevel && pNextRow->bHasChildren;
				}
				goto InsertRow;
			}
		}

		rc = NE_XFLM_OK;

		// No sibling node found.  Must go up one level to parent if we can
		if (!(pCurRow->uiFlags & F_DOMEDIT_FLAG_NOPARENT) ||
			 (bCurRowDataLocal && pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
		{
 			if (bCurRowDataLocal && (pCurRow->uiFlags & F_DOMEDIT_FLAG_ELEMENT_DATA))
			{
				pParentDomNode = pCurRow->pDomNode;
				pParentDomNode->AddRef();
			}
			else if (RC_BAD( rc = pCurRow->pDomNode->getParentNode(
															m_pDb,
															(IF_DOMNode **)&pParentDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
			}
			
			if (RC_OK( rc))
			{
				if (RC_BAD( rc = buildNewRow( (FLMINT)pCurRow->uiLevel - 1,
														pParentDomNode,
														&pNextRow)))
				{
					goto Exit;
				}
				pNextRow->uiFlags = (pCurRow->uiFlags & ~F_DOMEDIT_FLAG_ELEMENT_DATA) | F_DOMEDIT_FLAG_ENDTAG;
				pNextRow->bExpanded = TRUE;

				if (RC_BAD( rc = pParentDomNode->isDataLocalToNode( m_pDb, &pNextRow->bHasElementData)))
				{
					goto Exit;
				}

				goto InsertRow;
			}
		}

		rc = NE_XFLM_OK;

		// If nothing was found, is there another document in the list that we
		// can display? Getting to this point indicates that we could not find
		// another row to display in the current document.

		flmAssert( pNextRow == NULL);

		if (pDocList)
		{
			pDocList = pDocList->pNext;
		}

		if (pDocList)
		{

			if (RC_BAD( rc = getDOMNode( this, pDocList)))
			{
				goto Exit;
			}

			// Build a new row.
			if (RC_BAD( rc = buildNewRow( -1,
													pDocList->pDomNode,
													&pNextRow)))
			{
				goto Exit;
			}
			goto InsertRow;

		}
		else
		{
			goto Exit;
		}


InsertRow:

		if (RC_BAD( rc = insertRow( pNextRow,
											 pCurRow)))
		{
			goto Exit;
		}

		// Need to remove the first row.
		if (m_uiNumRows > m_uiEditCanvasRows)
		{
			releaseRow( &m_pScrFirstRow);
			m_uiNumRows--;
		}

		// Set the new last row
		m_pScrLastRow = pNextRow;

		// If there is an anchor row, then did we just display it's end tag?
		if (m_pRowAnchor)
		{
			if (m_pRowAnchor->ui64NodeId == pNextRow->ui64NodeId)
			{
				// We can remove the anchor now.

				f_free( &m_pRowAnchor);
			}
		}

	}


Exit:

	if (pSiblingDomNode)
	{
		pSiblingDomNode->Release();
	}

	if (pChildDomNode)
	{
		pChildDomNode->Release();
	}

	if (pParentDomNode)
	{
		pParentDomNode->Release();
	}

	if (pCurRow && pCurRow->pDomNode)
	{
		pCurRow->pDomNode->Release();
		pCurRow->pDomNode = NULL;
	}

	if (m_pCurDoc && m_pCurDoc->pDomNode)
	{
		m_pCurDoc->pDomNode->Release();
		m_pCurDoc->pDomNode = NULL;
	}

	if (pDocList && pDocList->pDomNode)
	{
		pDocList->pDomNode->Release();
		pDocList->pDomNode = NULL;
	}

	if (pNextRow && pNextRow->pDomNode)
	{
		pNextRow->pDomNode->Release();
		pNextRow->pDomNode = NULL;
	}

	*ppNextRow = pNextRow;

	return rc;
}


/****************************************************************************
Desc: Set's root row - first row in the list.
*****************************************************************************/
void F_DomEditor::setScrFirstRow(
	DME_ROW_INFO *		pScrFirstRow
	)
{
	m_uiNumRows = 0;
	if (m_pScrFirstRow)
	{
		while (m_pScrFirstRow)
		{
			releaseRow( &m_pScrFirstRow);
		}
	}

	if (m_pDocList)
	{
		while (m_pDocList)
		{
			releaseRow( &m_pDocList);
		}
		m_pCurDoc = NULL;
	}

	if (pScrFirstRow)
	{
		m_pScrFirstRow = pScrFirstRow;
		m_uiNumRows++;

		m_pScrLastRow = pScrFirstRow;
		while (m_pScrLastRow->pNext)
		{
			m_pScrLastRow = m_pScrLastRow->pNext;
			m_uiNumRows++;
		}
	}
	else
	{
		m_pScrLastRow = m_pScrFirstRow;
		f_free( &m_pRowAnchor);
	}
}




/****************************************************************************
Desc: Build a new row structure.  Note that when deleting the row, it will
		also be necessary to delete the value buffer and release the Dom object.
*****************************************************************************/
RCODE F_DomEditor::buildNewRow(
	FLMINT					iLevel,
	F_DOMNode *				pDomNode,
	DME_ROW_INFO **		ppNewRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pNewRow;
	F_DOMNode *			pCurrDOMNode = NULL;

	// Build the new row

	if( RC_BAD( rc = f_alloc( sizeof( DME_ROW_INFO), &pNewRow)))
	{
		goto Exit;
	}

	f_memset( pNewRow, 0, sizeof( DME_ROW_INFO));

	// Determine the node level if it isn't passed in.
	if (iLevel == -1)
	{
		iLevel = 0;
		pCurrDOMNode = pDomNode;
		pCurrDOMNode->AddRef();
		while (RC_OK( rc = pCurrDOMNode->getParentNode(
								m_pDb, (IF_DOMNode **)&pCurrDOMNode)))
		{
			iLevel++;
		}
		if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		pCurrDOMNode->Release();
		pCurrDOMNode = NULL;
		rc = NE_XFLM_OK;
	}
	
	pNewRow->eType = pDomNode->getNodeType();
	if (pNewRow->eType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = pDomNode->getParentId( m_pDb, &pNewRow->ui64NodeId)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pDomNode->getNodeId( m_pDb, &pNewRow->ui64NodeId)))
		{
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pDomNode->getDocumentId( m_pDb, &pNewRow->ui64DocId)))
	{
		goto Exit;
	}

	pNewRow->uiLevel = (FLMUINT)iLevel;
	pDomNode->getNameId( m_pDb, &pNewRow->uiNameId);
	pNewRow->bExpanded = FALSE;
	if ( pDomNode->getNodeType() != ATTRIBUTE_NODE)
	{
		if (RC_BAD( rc = pDomNode->hasChildren( m_pDb, &pNewRow->bHasChildren)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pDomNode->hasAttributes( m_pDb, &pNewRow->bHasAttributes)))
		{
			goto Exit;
		}
	}
	else
	{
		pNewRow->bHasChildren = FALSE;
		pNewRow->bHasAttributes = FALSE;
	}

	//pNewRow->pDomNode = pDomNode;
	*ppNewRow = pNewRow;
	pNewRow = NULL;

Exit:

	if (pCurrDOMNode)
	{
		pCurrDOMNode->Release();
	}

	if (pNewRow)
	{
		if ( pNewRow->puzValue)
		{
			f_free( pNewRow->puzValue);
		}

		if (pNewRow->pDomNode)
		{
			pNewRow->pDomNode->Release();
		}

		f_free( &pNewRow);
	}
	return rc;
}


/****************************************************************************
Desc: Expands a row to reveal it immediate children.
*****************************************************************************/
RCODE F_DomEditor::expandRow(
	DME_ROW_INFO *				pRow,
	FLMBOOL						bOneLevel,
	DME_ROW_INFO **			ppLastRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pNewRow = NULL;
	DME_ROW_INFO *		pPrevRow = pRow;
	F_DOMNode *			pDomNode = NULL;
	F_DOMNode *			pChildNode = NULL;
	FLMBOOL				bHaveFirstChild = FALSE;
	FLMBOOL				bRowHasElementData = pRow->bHasElementData;

	// Is this row already expanded, or does it not have any children and it doesn't have any data?
	if ( pRow->bExpanded || (!pRow->bHasChildren && !pRow->bHasElementData))
	{
		goto Exit;
	}

	// Get the Dom Node if it isn't already attached to the row.
	if (RC_BAD( rc = getDOMNode( this, pRow)))
	{
		goto Exit;
	}


	while (!(pRow == m_pScrLastRow && m_uiNumRows >= m_uiEditCanvasRows) &&
			 (!(pPrevRow == m_pScrLastRow && m_uiNumRows >= m_uiEditCanvasRows)))
	{
		if (bRowHasElementData)
		{
			pDomNode = pRow->pDomNode;
			pDomNode->AddRef();
		}
		else if (!bHaveFirstChild)
		{
			if (RC_BAD( rc = pRow->pDomNode->getFirstChild( m_pDb,
																			(IF_DOMNode **)&pDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				break;
			}
			bHaveFirstChild = TRUE;
		}
		else
		{
			if (RC_BAD( rc = pChildNode->getNextSibling( m_pDb,
																		(IF_DOMNode **)&pDomNode)))
			{
				if (rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				break;
			}
		}

		if (RC_BAD( rc = buildNewRow( pRow->uiLevel + 1,
												pDomNode,
												&pNewRow)))
		{
			goto Exit;
		}
		
		if (bRowHasElementData)
		{
			pNewRow->uiFlags |= F_DOMEDIT_FLAG_ELEMENT_DATA;
			bRowHasElementData = FALSE;
		}
		else
		{
			// If the new row DOM node is an element node and it has embedded data,
			// we need to set the bHasElementData flag.
			
			if (pDomNode->getNodeType() == ELEMENT_NODE)
			{
				FLMBOOL			bDataIsLocal;
				if (RC_BAD( rc = pDomNode->isDataLocalToNode( m_pDb, &bDataIsLocal)))
				{
					goto Exit;
				}
				if (bDataIsLocal)
				{
					pNewRow->bHasElementData = TRUE;
				}
				else
				{
					pNewRow->bHasElementData = FALSE;
				}
			}
		}

		if (pChildNode)
		{
			pChildNode->Release();
			pChildNode = NULL;
		}
		pChildNode = pDomNode;
		pDomNode = NULL;


		// Link the row in
		if (RC_BAD( rc = insertRow( pNewRow, pPrevRow)))
		{
			goto Exit;
		}

		pPrevRow = pNewRow;
		pNewRow = NULL;

		if (m_uiNumRows > m_uiEditCanvasRows)
		{
			releaseLastRow();
		}

		// Recursively expand...
		if (!bOneLevel && (pPrevRow->bHasChildren || pPrevRow->bHasElementData))
		{
			if (RC_BAD( rc = expandRow( pPrevRow,
												 bOneLevel,
												 &pPrevRow)))
			{
				goto Exit;
			}
		}
	}

	if (!(pPrevRow == m_pScrLastRow && m_uiNumRows >= m_uiEditCanvasRows))
	{
		// Now build the end row
		if (RC_BAD( rc = buildNewRow( pRow->uiLevel,
												pRow->pDomNode,
												&pNewRow)))
		{
			goto Exit;
		}

		// Now set the flags and link this row after the current row.
		pNewRow->uiFlags |= F_DOMEDIT_FLAG_ENDTAG;
		pNewRow->bExpanded = TRUE;
		pNewRow->bHasElementData = pRow->bHasElementData;

		// Link the row in
		if (RC_BAD(rc = insertRow( pNewRow,
											pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pNewRow;
		pNewRow = NULL;

		if (m_uiNumRows > m_uiEditCanvasRows)
		{
			releaseLastRow();
		}
	}

	// Mark that the current row has been expanded...
	pRow->bExpanded = TRUE;

	if (pPrevRow->pDomNode)
	{
		pPrevRow->pDomNode->Release();
		pPrevRow->pDomNode = NULL;
	}

	if (ppLastRow)
	{
		*ppLastRow = pPrevRow;
	}

Exit:

	if (pPrevRow->pDomNode)
	{
		pPrevRow->pDomNode->Release();
		pPrevRow->pDomNode = NULL;
	}

	if (pNewRow)
	{
		if (pNewRow->puzValue)
		{
			f_free( &pNewRow->puzValue);
		}
		if (pNewRow->pDomNode)
		{
			pNewRow->pDomNode->Release();
			pNewRow->pDomNode = NULL;
		}
		f_free( &pNewRow);
	}

	if (pDomNode)
	{
		pDomNode->Release();
	}

	if ( pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	if (pChildNode)
	{
		pChildNode->Release();
	}

	if (pDomNode)
	{
		pDomNode->Release();
	}

	return rc;
}



/****************************************************************************
Desc: Collapses a row to reveal it immediate children.
*****************************************************************************/
RCODE F_DomEditor::collapseRow(
	DME_ROW_INFO **		ppRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pReleaseRow = NULL;
	DME_ROW_INFO *		pRow = *ppRow;
	FLMUINT				uiLevel;
	FLMBOOL				bIgnoreAnchor;

	// Is this row already collapsed, or does it not have any children?
	if ( !pRow->bExpanded || !pRow->bHasChildren && !pRow->bHasElementData)
	{
		goto Exit;
	}

	uiLevel = pRow->uiLevel;

	// Are we looking at the first or last entry to be collapsed?
	if (pRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG)
	{
		for (pTmpRow = pRow->pPrev;
			  pTmpRow && pTmpRow->uiLevel >= uiLevel;
			  )
		{
			FLMUINT		uiOldLevel = pTmpRow->uiLevel;

			pReleaseRow = pTmpRow;
			pTmpRow = pTmpRow->pPrev;
			releaseRow( &pReleaseRow);
			m_uiNumRows--;
			if (uiOldLevel == uiLevel)
			{
				break;
			}
		}

		pRow->bExpanded = FALSE;
		pRow->uiFlags &= ~(FLMUINT)F_DOMEDIT_FLAG_ENDTAG;

		// If we deleted the first row, we must reset it.
		if (pTmpRow == NULL)
		{
			m_pScrFirstRow = pRow;
		}


	}
	else
	{

		pRow->bExpanded = FALSE;
		pTmpRow = pRow->pNext;
		while (pTmpRow && pTmpRow->uiLevel >= uiLevel)
		{
			FLMUINT		uiOldLevel = pTmpRow->uiLevel;
			releaseRow( &pTmpRow);
			m_uiNumRows--;
			if ( uiOldLevel == uiLevel)
			{
				break;
			}
		}

		// If there is no last row, then we need to set it to the current row.
		if (pTmpRow == NULL)
		{
			m_pScrLastRow = pRow;
		}

	}


	// If we just collapsed the anchored row, we don't want the anchor to
	// interfere with getting the rest of the appropriate rows.  If we leave the
	// row anchor there, we will just expand again - oops.
	if (m_pRowAnchor)
	{
		if (m_pRowAnchor->ui64NodeId == pRow->ui64NodeId)
		{
			f_free( &m_pRowAnchor);
		}
	}

	// Now, let's see if we can fill the screen with more rows.
	bIgnoreAnchor = !m_pScrLastRow->bExpanded;
	while (m_uiNumRows < m_uiEditCanvasRows)
	{

		// Set the docList to sync with the last row displayed.
		checkDocument(&m_pCurDoc, m_pScrLastRow);
		if (RC_BAD( rc = getNextRow( m_pScrLastRow, &pTmpRow, TRUE, bIgnoreAnchor)))
		{
			goto Exit;
		}
		if (pTmpRow == NULL)
		{
			break;
		}
		m_pScrLastRow = pTmpRow;
		bIgnoreAnchor = FALSE;
		// Make sure our document list and last row are in sync.  This prevents
		// us from filling the screen with bogus rows.
	}

	// Resync the docList with the cursor row.
	checkDocument(&m_pCurDoc, pRow);

Exit:

	return rc;
}

/****************************************************************************
Desc: 
*****************************************************************************/
/*
NODE_p F_DomEditor::findRow(
	FLMUINT				uiCollection,
	FLMUINT				uiDrn,
	DME_ROW_INFO *		pStartRow)
{
	FLMUINT				uiSourceCont;
	FLMUINT				uiSourceDrn;
	DME_ROW_INFO *		pTmpRow;
	DME_ROW_INFO *		pCurRow = m_pRoot;
	FLMBOOL				bForward = TRUE;

	flmAssert( m_bSetupCalled == TRUE);

	if( pStartRow)
	{
		pCurRow = getRootRow( pStartRow);
	}
	else if( m_pCurRow)
	{
		pTmpRow = getRootRow( m_pCurRow);
		if( RC_OK( GedGetRecSource( pTmpNd, NULL,
			&uiSourceCont, &uiSourceDrn)))
		{
			pCurNd = pTmpNd;
			if( uiSourceCont > uiCollection ||
				(uiSourceCont == uiCollection && uiSourceDrn > uiDrn))
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
			if( uiSourceCont == uiCollection && uiSourceDrn == uiDrn)
			{
				break;
			}

			if( bForward)
			{
				if( uiSourceCont > uiCollection ||
					(uiSourceCont == uiCollection && uiSourceDrn > uiDrn))
				{
					pCurNd = NULL;
					break;
				}
			}
			else
			{
				if( uiSourceCont < uiCollection ||
					(uiSourceCont == uiCollection && uiSourceDrn < uiDrn))
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

		// Release CPU to prevent CPU hog

		f_yieldCPU();
	}

	return( pCurNd);
}
*/

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_DomEditor::clearSelections( void)
{
	RCODE		rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	FLMUINT	uiFlags = 0;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Clear the "selected" flags
	*/

	pTmpRow = m_pScrFirstRow;
	while( pTmpRow)
	{
		(void)getControlFlags( pTmpRow, &uiFlags);
		if( (uiFlags & F_DOMEDIT_FLAG_SELECTED))
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			if( RC_BAD( rc = setControlFlags( pTmpRow, uiFlags)))
			{
				goto Exit;
			}
		}
		pTmpRow = pTmpRow->pNext;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_DomEditor::setCurrentRow(
	DME_ROW_INFO *				pCurRow,
	FLMUINT						uiCurRow
	)
{
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if (pCurRow)
	{

		refreshRow( m_uiCurRow, m_pCurRow, TRUE);
	}

	m_pCurRow = pCurRow;
	if (uiCurRow <= m_uiEditCanvasRows)
	{
		m_uiCurRow = uiCurRow;
	}

	if (pCurRow)
	{

		refreshRow( m_uiCurRow, m_pCurRow, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
DME_ROW_INFO * F_DomEditor::getCurrentRow(
	FLMUINT *			puiCurRow
	)
{

	flmAssert( m_bSetupCalled == TRUE);

	*puiCurRow = m_uiCurRow;

	return( m_pCurRow);
}


/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_DomEditor::setFirstRow(
	DME_ROW_INFO *				pRow
	)
{
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	m_pScrFirstRow = pRow;

	return( rc);
}


/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_DomEditor::getDisplayValue(
	DME_ROW_INFO *			pRow,
	char *					pszBuf,
	FLMUINT					uiBufSize)
{
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	/*
	This is a stupid check, but keep it for now.
	*/

	if( uiBufSize <= 32 || !pRow || !pszBuf)
	{
		flmAssert( 0);
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	*pszBuf = '\0';

	if (pRow->uiLength > uiBufSize)
	{
		f_sprintf( pszBuf, "%*s...", uiBufSize - 3, pRow->puzValue);
	}
	else
	{
		f_sprintf( pszBuf, "%*s", pRow->uiLength, pRow->puzValue);
	}

	if( m_pEventHook)
	{
		if( RC_BAD( rc = m_pEventHook( this, F_DOMEDIT_EVENT_GETDISPVAL,
			(void *)(pRow), m_EventData)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Can we edit this row?
*****************************************************************************/
FLMBOOL F_DomEditor::canEditRow(
	DME_ROW_INFO *		pCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bCanEdit = TRUE;
	eDomNodeType		eType;
	FLMBOOL				bHasAttrs;

	flmAssert( m_bSetupCalled == TRUE);

	if( m_bReadOnly || !pCurRow)
	{
		bCanEdit = FALSE;
		goto Exit;
	}

	if (pCurRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG ||
		pCurRow->uiFlags & F_DOMEDIT_FLAG_COMMENT ||
		pCurRow->uiFlags & F_DOMEDIT_FLAG_READ_ONLY)
	{
		bCanEdit = FALSE;
		goto Exit;
	}

	if (pCurRow->ui64NodeId)
	{
		if (RC_BAD( rc = getDOMNode( this, pCurRow)))
		{
			bCanEdit = FALSE;
			goto Exit;
		}

		eType = pCurRow->pDomNode->getNodeType();

		switch (eType)
		{
		//	case DATA_NODE:
		//	case CDATA_SECTION_NODE:
		//	case COMMENT_NODE:
		//	case ATTRIBUTE_NODE:
			case ELEMENT_NODE:
				if (RC_BAD( rc = pCurRow->pDomNode->hasAttributes( m_pDb, &bHasAttrs)))
				{
					goto Exit;
				}
				if (!bHasAttrs)
				{
					bCanEdit = FALSE;
				}
				else
				{
					bCanEdit = TRUE;
				}
				break;
			case PROCESSING_INSTRUCTION_NODE:
			case DOCUMENT_NODE:
			case INVALID_NODE:
			{
				bCanEdit = FALSE;
				break;
			}
			default:
			{
				bCanEdit = TRUE;
				break;
			}
		}
	}


Exit:

	if (pCurRow->pDomNode)
	{
		pCurRow->pDomNode->Release();
		pCurRow->pDomNode = NULL;
	}

	return( bCanEdit);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL F_DomEditor::canDeleteRow(
	DME_ROW_INFO *		pCurRow
	)
{
	FLMUINT	uiFlags = 0;
	FLMBOOL	bCanDelete = TRUE;

	flmAssert( m_bSetupCalled == TRUE);

	if( m_bReadOnly || !pCurRow)
	{
		bCanDelete = FALSE;
		goto Exit;
	}

	(void)getControlFlags( pCurRow, &uiFlags);
	if( uiFlags & (F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NO_DELETE))
	{
		bCanDelete = FALSE;
		goto Exit;
	}

Exit:

	return( bCanDelete);
}



/****************************************************************************
Desc:	Retrieve the list of Documents in the database - return the selected
		document.
*****************************************************************************/
RCODE F_DomEditor::retrieveDocumentList(
	FLMUINT			uiCollection,
	FLMUINT64 *		pui64NodeId,
	FLMUINT *		puiTermChar
	)
{
	DME_ROW_INFO *			pTmpRow = NULL;
	DME_ROW_INFO *			pPriorRow = NULL;
	FLMUINT					uiFlags;
	F_DomEditor *			pDocumentList = NULL;
	RCODE						rc = NE_XFLM_OK;
	F_DOMNode *				pDOMNode = NULL;
	F_DOMNode *				pSiblingNode = NULL;
	FLMUINT					uiDocumentCount;
	FLMUNICODE *			puzTitle = NULL;
	FLMUINT64				ui64DocumentID;
	FLMBOOL					bGotFirstDoc;


	flmAssert( m_bSetupCalled == TRUE);

	if( pui64NodeId)
	{
		*pui64NodeId = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pDocumentList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pDocumentList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pDocumentList->setParent( this);
	pDocumentList->setReadOnly( TRUE);
	pDocumentList->setShutdown( m_pbShutdown);
	pDocumentList->setTitle( "Document List - Select One");
	pDocumentList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pDocumentList->setSource( m_pDb, uiCollection);

	if( m_pDb == NULL)
	{
		goto Exit;
	}


	// Get the document list.
	uiDocumentCount = 0;

	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NODOM);

	bGotFirstDoc = FALSE;
	while (uiDocumentCount < m_uiEditCanvasRows)
	{
		if (!bGotFirstDoc)
		{
			if (RC_BAD( rc = m_pDb->getFirstDocument( 
				uiCollection, (IF_DOMNode **)&pDOMNode)))
			{
				goto Exit;
			}
			bGotFirstDoc = TRUE;
		}
		else
		{
			if (RC_BAD( rc = pDOMNode->getNextDocument(
									m_pDb, (IF_DOMNode **)&pSiblingNode)))
			{
				if (rc != NE_XFLM_EOF_HIT && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
				break;
			}
			(void)pDOMNode->Release();
			pDOMNode = pSiblingNode;
			pSiblingNode = NULL;
		}
		
		if( RC_BAD( rc = pDOMNode->getNodeId( m_pDb, &ui64DocumentID)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = makeNewRow( &pTmpRow, NULL, ui64DocumentID)))
		{
			goto Exit;
		}

		pTmpRow->uiIndex = uiDocumentCount;
		pTmpRow->ui64DocId = ui64DocumentID;

		pTmpRow->uiFlags = uiFlags;

		// Link the rows into the list of titles to display.
		if (RC_BAD( rc = pDocumentList->insertRow( pTmpRow, pPriorRow)))
		{
			goto Exit;
		}
		pPriorRow = pTmpRow;
		pTmpRow = NULL;
		uiDocumentCount++;
	}
	if (pDOMNode)
	{
		pDOMNode->Release();
		pDOMNode = NULL;
	}

	pDocumentList->setDocList( TRUE);

	// Set the rows...
	pDocumentList->setCurrentAtTop();

	if( RC_BAD( rc = pDocumentList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pDocumentList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pDocumentList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pDocumentList->setControlFlags( pTmpRow, uiFlags);
			if( pui64NodeId)
			{
				*pui64NodeId = pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pDocumentList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pDocumentList->getLastKey();
	}

Exit:

	// Cleanup ...
	f_free( &puzTitle);

	if (pDocumentList)
	{
		pDocumentList->Release();
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	if (pSiblingNode)
	{
		pSiblingNode->Release();
	}
	if (pDOMNode)
	{
		pDOMNode->Release();
	}

	return rc;
}

/****************************************************************************
Desc:	Retrieve the range of nodes from the database...
*****************************************************************************/
RCODE F_DomEditor::retrieveNodeFromDb(
	FLMUINT		uiCollection,
	FLMUINT64	ui64NodeId,
	FLMUINT		uiAttrNameId)
{
	RCODE							rc = NE_XFLM_OK;
	F_DOMNode *					pDomNode = NULL;
	DME_ROW_INFO *				pTmpRow = NULL;
	DME_ROW_INFO *				pPriorRow = NULL;

	// Can we put this node on the screen?
	if (m_uiNumRows < m_uiEditCanvasRows)
	{
		if (uiAttrNameId)
		{
			rc = m_pDb->getAttribute( uiCollection, ui64NodeId, uiAttrNameId,
								(IF_DOMNode **)&pDomNode);
		}
		else
		{
			rc = m_pDb->getNode( uiCollection, ui64NodeId, &pDomNode);
		}

		if( RC_BAD( rc))
		{
			if( rc != NE_XFLM_NOT_FOUND &&
				 rc != NE_XFLM_EOF_HIT &&
				 rc != NE_XFLM_BOF_HIT &&
				 rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_XFLM_OK;
				goto Exit;
			}
		}

		// Find the last row
		pPriorRow = getScrLastRow();
		
		// Now build the new rows
		if (RC_BAD( rc = buildNewRow( -1,
										 pDomNode,
										 &pTmpRow)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = insertRow( pTmpRow, pPriorRow)))
		{
			goto Exit;
		}

		// Make the new node current.
		m_pCurRow = pTmpRow;

		if (RC_BAD( rc = expandRow( pTmpRow, FALSE, NULL)))
		{
			goto Exit;
		}

		// We need to mark the starting row since the expansion is
		// incomplete.
		if (m_pRowAnchor)
		{
			f_free( &m_pRowAnchor);
		}

		if( RC_BAD( rc = f_alloc( sizeof( DME_ROW_ANCHOR), &m_pRowAnchor)))
		{
			goto Exit;
		}
		f_memset( m_pRowAnchor, 0, sizeof(DME_ROW_ANCHOR));
		m_pRowAnchor->ui64NodeId = pTmpRow->ui64NodeId;
		m_pRowAnchor->uiAnchorLevel = pTmpRow->uiLevel;
		m_pRowAnchor->bSingleLevel = FALSE;
	}

Exit:

	if( pDomNode)
	{
		pDomNode->Release();
	}

	return( rc);
}


/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_DomEditor::displayAttributes(
	DME_ROW_INFO *				pRow
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_DomEditor *	pAttrList = NULL;
	F_DOMNode *		pDOMNode = NULL;
	F_DOMNode *		pAttrNode = NULL;
	FLMUINT			uiAttributeCount;
	FLMBOOL			bGotFirstAttr;
	FLMUINT			uiFlags;
	FLMUINT64		ui64DocumentID;
	FLMUINT			uiAttrNameId;
	DME_ROW_INFO *	pTmpRow=NULL;
	DME_ROW_INFO *	pPriorRow=NULL;

	flmAssert( m_bSetupCalled == TRUE);
	if (!pRow)
	{
		displayMessage(
							"Node DOM Nodes to display attributes for",
							NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	if ( pRow->eType != ELEMENT_NODE)
	{
		displayMessage(
							"DOM Node is not an ELEMENT_NODE",
							NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	if (!pRow->bHasAttributes)
	{
		displayMessage(
							"DOM Node does not have attributes",
							NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}


	// Make sure we have a name table....
	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pAttrList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pAttrList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pAttrList->setParent( this);
	pAttrList->setReadOnly( FALSE); // Allow editing these attributes.
	pAttrList->setShutdown( m_pbShutdown);
	pAttrList->setTitle( "Element Attribute List");
	pAttrList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pAttrList->setSource( m_pDb, m_uiCollection);

	if( m_pDb == NULL)
	{
		goto Exit;
	}

	// Get the DOMnode of the current element.
	if (RC_BAD( rc = getDomNode( pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
											&pDOMNode)))
	{
		goto Exit;
	}


	// Get the attribute list.
	uiAttributeCount = 0;

	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
				  F_DOMEDIT_FLAG_NOPARENT | F_DOMEDIT_FLAG_NOCHILD);

	bGotFirstAttr = FALSE;
	while (uiAttributeCount < m_uiEditCanvasRows)
	{
		if (!bGotFirstAttr)
		{
			if (RC_BAD( rc = pDOMNode->getFirstAttribute( m_pDb,
													(IF_DOMNode **)&pAttrNode)))
			{
				goto Exit;
			}
			bGotFirstAttr = TRUE;
			(void)pDOMNode->Release();
			pDOMNode = NULL;
		}
		else
		{
			if (RC_BAD( rc = pAttrNode->getNextSibling(
											m_pDb, (IF_DOMNode **)&pAttrNode)))
			{
				if (rc != NE_XFLM_EOF_HIT && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
				break;
			}
		}
		
		if( RC_BAD( rc = pAttrNode->getParentId( m_pDb, &ui64DocumentID)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pAttrNode->getNameId( m_pDb, &uiAttrNameId)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = makeNewRow( &pTmpRow, NULL, ui64DocumentID)))
		{
			goto Exit;
		}

		pTmpRow->eType = ATTRIBUTE_NODE;
		pTmpRow->uiNameId = uiAttrNameId;
		pTmpRow->uiIndex = uiAttributeCount;
		pTmpRow->ui64DocId = ui64DocumentID;

		pTmpRow->uiFlags = uiFlags;

		// Link the rows into the list of titles to display.
		if (RC_BAD( rc = pAttrList->insertRow( pTmpRow, pPriorRow)))
		{
			goto Exit;
		}
		pPriorRow = pTmpRow;
		pTmpRow = NULL;
		uiAttributeCount++;
	}
	if (pDOMNode)
	{
		pDOMNode->Release();
		pDOMNode = NULL;
	}

	pAttrList->setDocList( FALSE);


	// Set the rows...
	pAttrList->setCurrentAtTop();

	if( RC_BAD( rc = pAttrList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

Exit:


	// Cleanup ...
	if (pAttrList)
	{
		pAttrList->Release();
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}

	if (pAttrNode)
	{
		pAttrNode->Release();
	}
	
	if (pDOMNode)
	{
		pDOMNode->Release();
	}

	return rc;
}


/****************************************************************************
Desc:	
*****************************************************************************/
FSTATIC void domGetOutputFileName(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	FLMUINT			uiOutputFileNameBufSize)
{
	FLMUINT	uiChar;

	*pszOutputFileName = 0;
	FTXWinPrintf( pWindow, "Enter Output File Name: ");
	if( RC_BAD( FTXLineEdit( pWindow, pszOutputFileName, 
		uiOutputFileNameBufSize - 1, uiOutputFileNameBufSize - 1,
		NULL, &uiChar)))
	{
		*pszOutputFileName = 0;
	}
	FTXWinPrintf( pWindow, "\n");
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_DomEditor::displayNodeInfo(
	DME_ROW_INFO *				pRow
	)
{
	RCODE				rc = NE_XFLM_OK;
	FTX_WINDOW *	pWindow = NULL;
	FLMUINT			uiChar;
	FLMUINT			uiTermChar;
	char				szOutputFile [200];

	if (RC_BAD( rc = createStatusWindow(
		"Node Information",
		FLM_GREEN, FLM_WHITE, NULL, NULL, &pWindow)))
	{
		goto Exit;
	}
	FTXWinOpen( pWindow);

	for (;;)
	{
		if (m_uiCollection == XFLM_DATA_COLLECTION)
		{
			FTXWinPrintf( pWindow, "\nInfo Type (N,ENTER=Node Only, S=Subtree, D=directory View, ESC=Cancel): ");
		}
		else
		{
			FTXWinPrintf( pWindow, "\nInfo Type (N,ENTER=Node Only, S=Subtree, ESC=Cancel): ");
		}
		FTXWinInputChar( pWindow, &uiChar);
		FTXWinSetCursorPos( pWindow, 0, 0);
		FTXWinClear( pWindow);

		switch (uiChar)
		{
			case 'D':
			case 'd':
				if (m_uiCollection == XFLM_DATA_COLLECTION)
				{
					domGetOutputFileName( pWindow, szOutputFile, sizeof( szOutputFile));
					domDisplayEntryInfo( pWindow, szOutputFile, m_pDb, pRow->ui64NodeId, TRUE);
				}
				else
				{
					FTXDisplayMessage( m_pScreen, m_bMonochrome ? FLM_LIGHTGRAY : FLM_RED,
						m_bMonochrome ? FLM_BLACK : FLM_WHITE,
						"Invalid option", NULL, &uiTermChar);
					uiChar = 0;
				}
				break;
			case 'S':
			case 's':
				domGetOutputFileName( pWindow, szOutputFile, sizeof( szOutputFile));
				domDisplayNodeInfo( pWindow, szOutputFile, m_pDb, m_uiCollection, pRow->ui64NodeId, TRUE, TRUE);
				break;
			case 'N':
			case 'n':
			case FKB_ENTER:
				domGetOutputFileName( pWindow, szOutputFile, sizeof( szOutputFile));
				domDisplayNodeInfo( pWindow, szOutputFile, m_pDb, m_uiCollection, pRow->ui64NodeId, FALSE, TRUE);
				break;
			case FKB_ESC:
				break;
			default:
				FTXDisplayMessage( m_pScreen, m_bMonochrome ? FLM_LIGHTGRAY : FLM_RED,
					m_bMonochrome ? FLM_BLACK : FLM_WHITE,
					"Invalid option", NULL, &uiTermChar);
				uiChar = 0;
				break;
		}
		if (uiChar)
		{
			break;
		}
	}

Exit:

	if (pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return rc;
}


/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_DomEditor::exportNode(
	DME_ROW_INFO *				pRow
	)
{
	RCODE						rc = NE_XFLM_OK;
	FTX_WINDOW *			pWindow = NULL;
	IF_DOMNode *			pNode = NULL;
	FLMUINT					uiChar;
	FLMUINT					uiTermChar;
	char						szFileName [80];
	eExportFormatType		eFormat = XFLM_EXPORT_INDENT;
	IF_OStream *			pFileOStream = NULL;	
	IF_DbSystem *			pDbSystem = NULL;

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = createStatusWindow(
		"Export Node Subtree",
		FLM_GREEN, FLM_WHITE, NULL, NULL, &pWindow)))
	{
		goto Exit;
	}
	FTXWinOpen( pWindow);

	szFileName [0] = 0;
	FTXWinClear( pWindow);
	for (;;)
	{
		FTXWinSetCursorPos( pWindow, 2, 1);
		FTXWinClearLine( pWindow, 2, 1);
		FTXWinPrintf( pWindow, "Enter Export File Name: ");
		if( RC_BAD( rc = FTXLineEdit( pWindow, szFileName, 
			sizeof( szFileName) - 1, sizeof( szFileName) - 1,
			NULL, &uiChar)))
		{
			goto Exit;
		}
		if (!szFileName [0])
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDbSystem->openFileOStream( szFileName, 
			TRUE, &pFileOStream)))
		{
			displayMessage( "Error creating export file", 
				rc, NULL, FLM_RED, FLM_WHITE);
		}
		else
		{
			break;
		}
	}

	for (;;)
	{
		FTXWinSetCursorPos( pWindow, 2, 2);
		FTXWinClearLine( pWindow, 2, 2);
		FTXWinPrintf( pWindow,
			"Format (I,ENTER=Indent, N=Newline, D=Indent Data, X=None, ESC=Cancel): ");
		FTXWinInputChar( pWindow, &uiChar);

		switch (uiChar)
		{
			case 'I':
			case 'i':
			case FKB_ENTER:
				eFormat = XFLM_EXPORT_INDENT;
				FTXWinPrintf( pWindow, "I");
				break;
			case 'N':
			case 'n':
				eFormat = XFLM_EXPORT_NEW_LINE;
				FTXWinPrintf( pWindow, "N");
				break;
			case 'D':
			case 'd':
				eFormat = XFLM_EXPORT_INDENT_DATA;
				FTXWinPrintf( pWindow, "D");
				break;
			case 'X':
			case 'x':
				eFormat = XFLM_EXPORT_NO_FORMAT;
				FTXWinPrintf( pWindow, "X");
				break;
			case FKB_ESC:
				goto Exit;
			default:
				FTXDisplayMessage( m_pScreen, m_bMonochrome ? FLM_LIGHTGRAY : FLM_RED,
					m_bMonochrome ? FLM_BLACK : FLM_WHITE,
					"Invalid option", NULL, &uiTermChar);
				uiChar = 0;
				break;
		}
		if (uiChar)
		{
			break;
		}
	}

	if (pRow->eType == ATTRIBUTE_NODE)
	{
		rc = m_pDb->getAttribute( m_uiCollection, pRow->ui64NodeId,
									pRow->uiNameId,
									(IF_DOMNode **)&pNode);
	}
	else
	{
		rc = m_pDb->getNode( m_uiCollection, pRow->ui64NodeId, &pNode);
	}
	if (RC_BAD( rc))
	{
		displayMessage( "Error getting node", rc, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}
	if( RC_BAD( rc = m_pDb->exportXML( pNode, pFileOStream, eFormat)))
	{
		displayMessage( "Error exporting data", rc, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}
	FTXWinSetCursorPos( pWindow, 2, 3);
	FTXWinClearLine( pWindow, 2, 3);
	FTXWinPrintf( pWindow, "Export Done, press any character to exit: ");
	FTXWinInputChar( pWindow, &uiChar);

Exit:

	if (pFileOStream)
	{
		pFileOStream->Release();
	}

	if (pNode)
	{
		pNode->Release();
	}

	if (pWindow)
	{
		FTXWinFree( &pWindow);
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return rc;
}


/****************************************************************************
Desc:	Retrieve information about the document and add it to the document linked
		list.
*****************************************************************************/
RCODE F_DomEditor::addDocumentToList(
	FLMUINT			uiCollection,
	FLMUINT64		ui64DocumentId
	)
{
	RCODE						rc = NE_XFLM_OK;
	F_DOMNode *				pDomNode = NULL;
	DME_ROW_INFO *			pTmpRow;
	DME_ROW_INFO *			pLastRow = m_pDocList;

	if( RC_BAD( rc = m_pDb->getNode( uiCollection,
												ui64DocumentId,
												&pDomNode)))
	{
		if( rc != NE_XFLM_NOT_FOUND && rc != NE_XFLM_EOF_HIT &&
			rc != NE_XFLM_BOF_HIT && rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			goto Exit;
		}
		else
		{
			rc = NE_XFLM_OK;
		}
	}


	// Now build the new row
	if (RC_BAD( buildNewRow( -1,
									 pDomNode,
									 &pTmpRow)))
	{
		goto Exit;
	}

	if (m_pDocList == NULL)
	{
		m_pDocList = pTmpRow;
		m_pCurDoc = pTmpRow;
	}
	else
	{
		for (;;)
		{
			if (pLastRow->pNext)
			{
				pLastRow = pLastRow->pNext;
			}
			else
			{
				pLastRow->pNext = pTmpRow;
				pTmpRow->pPrev = pLastRow;
				break;
			}
		}
		if (m_pCurRow->ui64DocId != m_pCurDoc->ui64DocId)
		{
			m_pCurDoc = pTmpRow;
		}
	}


Exit:

	if (pDomNode)
	{
		pDomNode->Release();
	}
	
	return rc;
}

/****************************************************************************
Desc:	Allows the user to interactively select a Collection
*****************************************************************************/
RCODE F_DomEditor::selectCollection(
	FLMUINT *	puiCollection,
	FLMUINT *	puiTermChar)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FLMUINT				uiFlags;
	FLMUNICODE			uzItemName[ 128];
	FLMUINT				uiId;
	FLMUINT				uiNextPos;
	F_DomEditor *		pCollectionList = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiCollection)
	{
		*puiCollection = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pCollectionList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pCollectionList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pCollectionList->setParent( this);
	pCollectionList->setReadOnly( TRUE);
	pCollectionList->setShutdown( m_pbShutdown);
	pCollectionList->setTitle( "Collections - Select One");
	pCollectionList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pCollectionList->setSource( m_pDb, XFLM_DICT_COLLECTION);

	if( m_pDb == NULL)
	{
		goto Exit;
	}
	
	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY);

	asciiToUnicode( "Default Data", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 0, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;
	pTmpRow->uiNameId = XFLM_DATA_COLLECTION;

	if (RC_BAD( rc = pCollectionList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "Dictionary Definitions", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 0, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;
	pTmpRow->uiNameId = XFLM_DICT_COLLECTION;

	if (RC_BAD( rc = pCollectionList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "Maintenance", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 0, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;
	pTmpRow->uiNameId = XFLM_MAINT_COLLECTION;

	if (RC_BAD( rc = pCollectionList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	uiNextPos = 0;
	while( RC_OK( rc = m_pNameTable->getNextTagTypeAndNameOrder(
		ELM_COLLECTION_TAG,
		&uiNextPos, uzItemName, NULL, sizeof( uzItemName), &uiId)))
	{
		if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 0, TRUE)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags = uiFlags;
		pTmpRow->uiNameId = uiId;

		if (RC_BAD( rc = pCollectionList->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;

	}
	if (rc != NE_XFLM_EOF_HIT)
	{
		goto Exit;
	}
	rc = NE_XFLM_OK;


	// Set the start row.
	pCollectionList->setCurrentAtTop();


	if( RC_BAD( rc = pCollectionList->interactiveEdit(
											m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pCollectionList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pCollectionList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pCollectionList->setControlFlags( pTmpRow, uiFlags);
			if( puiCollection)
			{
				*puiCollection = pTmpRow->uiNameId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pCollectionList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pCollectionList->getLastKey();
	}

Exit:

	if( pCollectionList)
	{
		pCollectionList->Release();
		pCollectionList = NULL;
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	return( rc);
}

/****************************************************************************
Desc:	Creates a new DME_ROW_INFO structure and stores the puzValue and Id.
		NOTE:  puzValue must be a null terminated string.
*****************************************************************************/
FSTATIC RCODE makeNewRow(
	DME_ROW_INFO **			ppTmpRow,
	FLMUNICODE *				puzValue,
	FLMUINT64					ui64Id,
	FLMBOOL						bUseValue
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	FLMUINT				uiLength;

	if( RC_BAD( rc = f_alloc( sizeof( DME_ROW_INFO), &pTmpRow)))
	{
		goto Exit;
	}

	f_memset( pTmpRow, 0, sizeof( DME_ROW_INFO));

	if (bUseValue)
	{
		uiLength = unicodeStrLen( puzValue);

		if (RC_BAD( rc = f_calloc( (uiLength*2)+2, &pTmpRow->puzValue)))
		{
			goto Exit;
		}

		f_memcpy( pTmpRow->puzValue, puzValue, (uiLength*2)+2);

		pTmpRow->uiLength = uiLength*2;
	}

	pTmpRow->ui64NodeId = ui64Id;
	pTmpRow->bUseValue = bUseValue;

	*ppTmpRow = pTmpRow;
	pTmpRow = NULL;

Exit:

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}

	return rc;
}

/****************************************************************************
Desc:	Calculates the length of a Unicode string.
*****************************************************************************/
FSTATIC FLMUINT unicodeStrLen(
	FLMUNICODE *		puzStr
	)
{
	FLMUINT		uiLength;
	FLMUNICODE *	puzTmp;

	if (!puzStr)
	{
		return 0;
	}

	for (puzTmp = puzStr, uiLength = 0;
		  *puzTmp != (FLMUNICODE)0;
		  uiLength++, puzTmp++);

	return uiLength;
}


/****************************************************************************
Name:	getNumber
Desc:	Converts a text string to a number
*****************************************************************************/
RCODE F_DomEditor::getNumber(
	char *			pszBuf,
	FLMUINT64 *		pui64Value,
	FLMINT64 *		pi64Value)
{
	char *		pszTmp = NULL;
	FLMUINT64	ui64Value = 0;
	FLMUINT		uiDigits = 0;
	FLMUINT		uiHexOffset = 0;
	FLMBOOL		bNeg = FALSE;
	FLMBOOL		bHex = FALSE;
	RCODE			rc = NE_XFLM_OK;

	if( pui64Value)
	{
		*pui64Value = 0;
	}

	if( pi64Value)
	{
		*pi64Value = 0;
	}

	if( f_strnicmp( pszBuf, "0x", 2) == 0)
	{
		uiHexOffset = 2;
		bHex = TRUE;
	}
	else if( *pszBuf == 'x' || *pszBuf == 'X')
	{
		uiHexOffset = 1;
		bHex = TRUE;
	}
	else
	{
		pszTmp = pszBuf;
		while( *pszTmp)
		{
			if( (*pszTmp >= '0' && *pszTmp <= '9') ||
				(*pszTmp >= 'A' && *pszTmp <= 'F') ||
				(*pszTmp >= 'a' && *pszTmp <= 'f') ||
				(*pszTmp == '-'))
			{
				if( (*pszTmp >= 'A' && *pszTmp <= 'F') ||
				(*pszTmp >= 'a' && *pszTmp <= 'f'))
				{
					bHex = TRUE;
				}
			}
			else
			{
				rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}
			pszTmp++;
		}
		uiHexOffset = 0;
	}

	if( bHex)
	{
		pszTmp = &(pszBuf[ uiHexOffset]);
		uiDigits = f_strlen( pszTmp);
		if( !uiDigits)
		{
			rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}

		while( *pszTmp)
		{
			ui64Value <<= 4;
			if( *pszTmp >= '0' && *pszTmp <= '9')
			{
				ui64Value |= *pszTmp - '0';
			}
			else if( *pszTmp >= 'a' && *pszTmp <= 'f')
			{
				ui64Value |= (*pszTmp - 'a') + 10;
			}
			else if( *pszTmp >= 'A' && *pszTmp <= 'F')
			{
				ui64Value |= (*pszTmp - 'A') + 10;
			}
			else
			{
				rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}
			pszTmp++;
		}
	}
	else if( (*pszBuf >= '0' && *pszBuf <= '9') || *pszBuf == '-')
	{
		pszTmp = pszBuf;
		if( *pszTmp == '-')
		{
			bNeg = TRUE;
			pszTmp++;
		}
		uiDigits = f_strlen( pszTmp);
		
		while( *pszTmp)
		{
			if( *pszTmp >= '0' && *pszTmp <= '9')
			{
				FLMUINT64	ui64NewVal = ui64Value;

				if( ui64NewVal > (~((FLMUINT64)0) / 10))
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				ui64NewVal *= 10;
				ui64NewVal += *pszTmp - '0';

				if( ui64NewVal < ui64Value)
				{
					rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				ui64Value = ui64NewVal;
			}
			else
			{
				rc = RC_SET( NE_XFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
			pszTmp++;
		}
	}
	else
	{
		rc = RC_SET( NE_XFLM_CONV_BAD_DIGIT);
		goto Exit;
	}

	if( bNeg)
	{
		if( pi64Value)
		{
#if defined( FLM_GNUC)
			if( ui64Value > 0x7FFFFFFFFFFFFFFFULL)
#else
			if( ui64Value > 0x7FFFFFFFFFFFFFFF)
#endif
			{
				rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
				goto Exit;
			}
			*pi64Value = -((FLMINT64)ui64Value);
		}
		else
		{
			rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}
	else
	{
		if( pui64Value)
		{
			*pui64Value = ui64Value;
		}
		else
		{
			rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Shows a help screen
*****************************************************************************/
RCODE F_DomEditor::showHelp(
	FLMUINT *		puiKeyRV
	)
{
	RCODE						rc = NE_XFLM_OK;
	DME_ROW_INFO *			pPrevRow = NULL;
	DME_ROW_INFO *			pTmpRow = NULL;
	FLMUINT					uiFlags;
	F_DomEditor *			pHelpList = NULL;
	FLMUNICODE				uzItemName[ 128];

	flmAssert( m_bSetupCalled == TRUE);

	

	if( (pHelpList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
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
	pHelpList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pHelpList->setSource( m_pDb, m_uiCollection);

	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY);

	asciiToUnicode( "Keyboard Commands", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 0, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "?               Help (this screen)", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)'?', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "#               Database statistics", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], '#', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;
	
	asciiToUnicode( "UP              Position cursor to the previous field", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_UP, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "DOWN            Position cursor to the next field", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_DOWN, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "PG UP           Position cursor to the previous page", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_PGUP, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "PG DOWN         Position cursor to the next page", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_PGDN, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "HOME            Position cursor to the top of the buffer", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_HOME, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "END             Position cursor to the bottom of the buffer", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_END, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "DELETE          Delete the current node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_DELETE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "A               Display DOM Node attributes", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'A', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "C               Clear all entries from the buffer", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'C', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;



	asciiToUnicode( "D               Display Node Information", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'C', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;



	asciiToUnicode( "F               Find nodes in the database via an XPATH query", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'F', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "I               Show index keys and references", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'I', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "L               List documents to select", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'L', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "N               add a New node to the database", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'N', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "R               Retrieve a node from the database", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'R', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "S               Refresh the current display window", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'S', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "V               View the current node value", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'V', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "X               Export Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'C', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;



	asciiToUnicode( "RIGHT           Expand context one level", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_RIGHT, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "LEFT            Collapse context", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_LEFT, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "PLUS            Expand to full context", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_PLUS, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "MINUS           collapse context", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_MINUS, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "ENTER           Edit the current node's value", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_ENTER, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "F8              Index manager", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_F8, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "F9              Memory manager", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_F9, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "F10             Toggle display colors on/off", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], FKB_F10, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "ESC, Q,         Exit", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], 'Q', TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pHelpList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	// Call the applications help hook to extend the help screen.

//	if( m_pHelpHook)
//	{
//		if( RC_BAD( rc = m_pHelpHook( this, m_pHelpHook)))
//		{
//			goto Exit;
//		}
//	}


	pHelpList->setCurrentAtTop();

	// Show the help screen

	if( RC_BAD( rc = pHelpList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX,
		m_uiLRY, TRUE, 0)))
	{
		goto Exit;
	}

	if( pHelpList->getLastKey() != FKB_ENTER)
	{
		goto Exit;
	}


	if ( puiKeyRV)
	{
		*puiKeyRV = 0;
	}

	// Find the selected item
	if( (pTmpRow = pHelpList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pHelpList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			if( puiKeyRV)
			{
				*puiKeyRV = (FLMUINT)pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pHelpList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

Exit:

	if( pHelpList)
	{
		pHelpList->Release();
	}

  return( rc);
}


/****************************************************************************
Name:	createStatusWindow
Desc:	Creates a window for displaying an operation's status
*****************************************************************************/
RCODE F_DomEditor::createStatusWindow(
	const char *		pszTitle,
	eColorType			uiBack,
	eColorType			uiFore,
	FLMUINT *			puiCols,
	FLMUINT *			puiRows,
	FTX_WINDOW **		ppWindow)
{
	FLMUINT			uiNumRows;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumWinRows = 0;
	FLMUINT			uiNumWinCols = 0;
	FTX_WINDOW *	pWindow = NULL;
	RCODE				rc = NE_XFLM_OK;

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
		uiNumWinRows = uiNumRows - 2;
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

	FTXWinSetScroll( pWindow, TRUE);
	FTXWinSetLineWrap( pWindow, TRUE);
	FTXWinSetCursorType( pWindow, FLM_CURSOR_INVISIBLE);

	if( m_bMonochrome)
	{
		uiBack = FLM_LIGHTGRAY;
		uiFore = FLM_BLACK;
		FTXWinSetBackFore( pWindow, uiBack, uiFore);
	}
	else
	{
		FTXWinSetBackFore( pWindow, uiBack, uiFore);
	}

	FTXWinClear( pWindow);
	FTXWinDrawBorder( pWindow);

	if( pszTitle)
	{
		FTXWinSetTitle( pWindow, pszTitle, uiBack, uiFore);
	}

	*ppWindow = pWindow;

Exit:

	return( rc);
}
	

/****************************************************************************
Name:	ViewOnlyKeyHook
*****************************************************************************/
FSTATIC RCODE f_ViewOnlyKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				//pvKeyData
	)
{
	RCODE				rc = NE_XFLM_OK;

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_ESCAPE:
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		// Special case
		case FKB_DELETE:
		{
			*puiKeyOut = 0;
			if (pCurRow->uiFlags & F_DOMEDIT_FLAG_COMMENT)
			{
				if (RC_BAD( rc = pDomEditor->viewOnlyDeleteIxKey()))
				{
					goto Exit;
				}
			}
			break;
		}

		default:
		{
			*puiKeyOut = 0;
			break;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Name:	f_KeyEditorKeyHook
*****************************************************************************/
FSTATIC RCODE f_KeyEditorKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				//pvKeyData
	)
{
	RCODE				rc = NE_XFLM_OK;
	DME_ROW_INFO *	pTmpRow = NULL;
	FLMUINT			uiCurRow;

	switch( uiKeyIn)
	{
		case FKB_UP:
		{
			(void)pDomEditor->getCurrentRow( &uiCurRow);
			if (uiCurRow)
			{

				if (RC_BAD( rc = pDomEditor->getPrevRow( pCurRow,
																	  &pTmpRow,
																	  FALSE)))
				{
					goto Exit;
				}

				if (pTmpRow != NULL)
				{
					uiCurRow--;
					if ( RC_BAD( rc = pDomEditor->setCurrentRow( pTmpRow, uiCurRow)))
					{
						goto Exit;
					}
				}
			}
			*puiKeyOut = 0;
			break;
		}
		case FKB_DOWN:
		{
			(void)pDomEditor->getCurrentRow( &uiCurRow);

			if (RC_BAD( rc = pDomEditor->getNextRow(
								pCurRow, &pTmpRow, FALSE)))
			{
				goto Exit;
			}

			if (pTmpRow != NULL)
			{
				uiCurRow++;
				if ( RC_BAD( rc = pDomEditor->setCurrentRow( pTmpRow, uiCurRow)))
				{
					goto Exit;
				}
			}
			*puiKeyOut = 0;
			break;
		}
		case FKB_ENTER:
		{
			if( !pDomEditor->canEditRow( pCurRow))
			{
				pDomEditor->displayMessage( "The row cannot be edited",
					RC_SET( NE_XFLM_ILLEGAL_OP), NULL, FLM_RED, FLM_WHITE);
			}
			else if( RC_BAD( rc = pDomEditor->editIndexRow( pCurRow)))
			{
				pDomEditor->displayMessage( "The field could not be edited", rc,
					NULL, FLM_RED, FLM_WHITE);
			}
			*puiKeyOut = 0;
			break;
		}
		case FKB_ESCAPE:		/* Quit key editor */
//		case FKB_ALT_Q:		/* Done editing keys */
		case FKB_ALT_Z:		/* Done, but don't quit */
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		case 'n':
		case 'N':
		{
			// Option to edit the node id.
			if( !pDomEditor->canEditRow( pCurRow))
			{
				pDomEditor->displayMessage( "The row cannot be edited",
					RC_SET( NE_XFLM_ILLEGAL_OP), NULL, FLM_RED, FLM_WHITE);
			}
			else if( RC_BAD( rc = pDomEditor->editIndexNode( pCurRow)))
			{
				pDomEditor->displayMessage( "The field could not be edited", rc,
					NULL, FLM_RED, FLM_WHITE);
			}
			*puiKeyOut = 0;
			break;
		}
		default:
		{
			*puiKeyOut = 0;
			break;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Name:	SelectionOnlyKeyHook
*****************************************************************************/
RCODE F_DomEditorSelectionKeyHook(
	F_DomEditor *,		//pDomEditor,
	DME_ROW_INFO *,	//pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				//pvKeyData
	)
{
	RCODE				rc = NE_XFLM_OK;

	switch( uiKeyIn)
	{
		case FKB_HOME:
		case FKB_END:
		case FKB_UP:
		case FKB_DOWN:
		case FKB_PGUP:
		case FKB_PGDN:
		case FKB_ESCAPE:
		case FKB_ENTER:
		case FKB_DELETE:
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

/***************************************************************************
Name:
Desc: 
****************************************************************************/
RCODE F_DomEditor::globalConfig(
	FLMUINT		uiOption)
{
	RCODE					rc = NE_XFLM_OK;

	if( m_pDb == NULL)
	{
		rc = RC_SET( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	switch( uiOption)
	{
		case F_DOMEDIT_CONFIG_STATS_START:
		{
			//eConfigOp = FLM_START_STATS;
			break;
		}

		case F_DOMEDIT_CONFIG_STATS_STOP:
		{
			//eConfigOp = FLM_STOP_STATS;
			break;
		}

		case F_DOMEDIT_CONFIG_STATS_RESET:
		{
			//eConfigOp = FLM_RESET_STATS;
			break;
		}

		default:
		{
			rc = RC_SET( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

//	if( RC_BAD( rc = FlmConfig( eConfigOp, 0, 0)))
//	{
//		goto Exit;
//	}

Exit:

	return( rc);
}

/*============================================================================
Name: asciiUCMixToUC
Desc:	Accepts a buffer with regular ascii, and 4 ascii chars representing 
      unicode and converts all the chars to unicode.	UC_MARKER defines the
		char which marks the beginning of a unicode sequence.
============================================================================*/
RCODE F_DomEditor::asciiUCMixToUC(
	char *			pszAscii,
	FLMUNICODE *	puzUnicode,
	FLMUINT 			uiMaxUniChars)
{
	char *		pszTmp;
	char *		pszTerm;
	char			szNumBuf[ 32];
	FLMUINT		uiUniCount = 0;
	FLMUINT64	ui64Value;
	RCODE			rc = NE_XFLM_OK;

	flmAssert( uiMaxUniChars > 0);
	uiMaxUniChars--; // Leave space for the terminator

	while( uiUniCount < uiMaxUniChars && *pszAscii)
	{
		if( pszAscii[ 0] == '~' && pszAscii[ 1] == '[')
		{
			pszAscii += 2;
			if( (pszTerm = f_strchr( pszAscii, ']')) == NULL)
			{
				rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}

			while( *pszAscii && *pszAscii != ']')
			{
				pszTmp = f_strchr( pszAscii, ' ');
				if( !pszTmp || pszTmp > pszTerm)
				{
					pszTmp = pszTerm;
				}

				f_memcpy( szNumBuf, pszAscii, pszTmp - pszAscii);
				szNumBuf[ pszTmp - pszAscii] = 0;

				if( RC_BAD( rc = getNumber( szNumBuf, &ui64Value, NULL)))
				{
					goto Exit;
				}

				puzUnicode[ uiUniCount++] = (FLMUNICODE)ui64Value;
				pszAscii += (pszTmp - pszAscii);
				while( *pszAscii == ' ')
				{
					pszAscii++;
				}
			}

			if( *pszAscii == ']')
			{
				pszAscii++;
			}
		}
		else
		{
			puzUnicode[ uiUniCount++] = (FLMUNICODE)(*pszAscii);
			pszAscii++;
		}
	}

	puzUnicode[ uiUniCount] = 0;

Exit:

	return rc;
}


/*============================================================================
Name: UCToAsciiUCMix
============================================================================*/
RCODE F_DomEditor::UCToAsciiUCMix(
	FLMUNICODE *	puzUnicode,
	char *			pszAscii,
	FLMUINT			uiMaxAsciiChars)
{
	char		szTmpBuf[ 32];
	FLMUINT	uiAsciiCount = 0;
	FLMBOOL	bEscaped = FALSE;
	RCODE		rc = NE_XFLM_OK;

	flmAssert( uiMaxAsciiChars > 0);
	uiMaxAsciiChars--; // Leave space for the terminator

	while( uiAsciiCount < uiMaxAsciiChars && *puzUnicode)
	{
		if( *puzUnicode >= 0x0020 && *puzUnicode <= 0x007E)
		{
			if( bEscaped)
			{
				pszAscii[ uiAsciiCount++] = ']';
				bEscaped = FALSE;
			}

			if( uiAsciiCount == uiMaxAsciiChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			pszAscii[ uiAsciiCount++] = (char)*puzUnicode;
		}
		else
		{
			if( !bEscaped)
			{
				if( (uiAsciiCount + 2) >= uiMaxAsciiChars)
				{
					rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
					goto Exit;
				}

				pszAscii[ uiAsciiCount++] = '~';
				pszAscii[ uiAsciiCount++] = '[';
				bEscaped = TRUE;
			}
			else
			{
				pszAscii[ uiAsciiCount++] = ' ';
			}

			pszAscii[ uiAsciiCount] = '\0';
			f_sprintf( szTmpBuf, "0x%04X", (unsigned)*puzUnicode);

			if( (uiAsciiCount +f_strlen( szTmpBuf)) >= uiMaxAsciiChars)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
			
			f_strcat( &(pszAscii[ uiAsciiCount]), szTmpBuf);
			uiAsciiCount += f_strlen( szTmpBuf);
		}

		puzUnicode++;
	}

	if( bEscaped)
	{
		if( uiAsciiCount == uiMaxAsciiChars)
		{
			rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		pszAscii[ uiAsciiCount++] = ']';
		bEscaped = FALSE;
	}

	pszAscii[ uiAsciiCount] = '\0';

Exit:

	return rc;
}


/***************************************************************************
Name:	domEditVerifyRun
Desc: Makes sure the utility is still allowed to run
*****************************************************************************/
RCODE domEditVerifyRun( void)
{
	F_TMSTAMP		curDate;
	FLMUINT			uiExpireYear = 0;
	FLMUINT			uiExpireMonth = 0;
	FLMUINT			uiExpireDay = 0;
	RCODE				rc = NE_XFLM_OK;
	const char *	pszDate = __DATE__;

	// Get the compilation date of this module.
	// If cannot get it, return NE_XFLM_OK.

	if (!domeditStrToDate( pszDate, &uiExpireYear,
					&uiExpireMonth, &uiExpireDay))
	{
		goto Exit;
	}

	// Add four months to it.

	if (uiExpireMonth <= 8)
	{
		uiExpireMonth += 4;
	}
	else
	{
		uiExpireMonth -= 8;
		uiExpireYear++;
	}

	// Adjust the day if necessary - we don't have to be too precise here
	// because we don't really care if we give them a day or two extra
	// for the expiration date.

	switch (uiExpireMonth)
	{
		case 2:
			if (uiExpireDay > 28)
			{
				uiExpireDay = 28;
			}
			break;
		case 4:
		case 6:
		case 9:
		case 11:
			if (uiExpireDay > 30)
			{
				uiExpireDay = 30;
			}
			break;
		default:
			break;
	}

	f_timeGetTimeStamp( &curDate);
	curDate.month++;
	if(((FLMUINT)curDate.year > uiExpireYear) ||
		((FLMUINT)curDate.year == uiExpireYear &&
		 (FLMUINT)curDate.month > uiExpireMonth) ||
		((FLMUINT)curDate.year == uiExpireYear &&
		 (FLMUINT)curDate.month == uiExpireMonth &&
		 (FLMUINT)curDate.day > uiExpireDay))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	return( rc);
}



/****************************************************************************
Desc: This routine converts the passed in string to a year, month, and day.
****************************************************************************/
FSTATIC FLMBOOL domeditStrToDate(
	const char *	s,
	FLMUINT *		puiYear,
	FLMUINT *		puiMonth,
	FLMUINT *		puiDay)
{
	char		szToken [80];
	FLMUINT	uiLoop;
	FLMBOOL	bHaveNum = FALSE;
	FLMBOOL	bHaveDay = FALSE;
	FLMBOOL	bHaveMonth = FALSE;
	FLMUINT	uiNum1;
	FLMUINT	uiSaveNum = 0;
	FLMBOOL	bSlashFormat = FALSE;
	FLMBOOL	bDashFormat = FALSE;
	FLMBOOL	bNoFormat = FALSE;

	for (;;)
	{

		// Get the next token from the string

		s = domeditSkipChars( s, " \t\n\r");
		uiLoop = 0;
		while (((*s >= 'a') && (*s <= 'z')) ||
				 ((*s >= 'A') && (*s <= 'Z')) ||
				 ((*s >= '0') && (*s <= '9')))
		{
			if (uiLoop < sizeof( szToken) - 1)
			{
				szToken [uiLoop++] = *s;
			}
			s++;
		}
		szToken [uiLoop] = 0;

		// See if it is a number

		if ((uiLoop) &&
			 (domeditIsNum( szToken, &uiNum1)))
		{
			if ((bHaveMonth) && (bHaveDay))
			{
				if (uiNum1 > 65535)
				{
					goto Year_Error;
				}
				*puiYear = uiNum1;

				// Make sure we have a valid day of the month.
				// NOTE: The test for Feb. 29 will allow it in
				// some cases where it shouldn't have, but at
				// least it will never disallow it when it should
				// have allowed it.

				switch (*puiMonth)
				{
					case 2:
						if ((*puiDay > 29) ||
							 ((*puiDay == 29) &&
							  (*puiYear % 4 != 0)))
						{
							goto Day_Error;
						}
						break;
					case 4:
					case 6:
					case 9:
					case 11:
						if (*puiDay > 30)
						{
							goto Day_Error;
						}
						break;
					default:
						break;
				}
				return( TRUE);
			}
			else if (bHaveMonth)
			{
				if ((uiNum1 < 1) || (uiNum1 > 31))
				{
					goto Day_Error;
				}
				bHaveDay = TRUE;
				*puiDay = uiNum1;
			}
			else if (bHaveDay)
			{
				if ((uiNum1 < 1) || (uiNum1 > 12))
				{
					goto Month_Error;
				}
				bHaveMonth = TRUE;
				*puiMonth = uiNum1;
			}
			else if (bHaveNum)
			{
				if ((uiNum1 >= 1) && (uiNum1 <= 12))
				{
					flmAssert( 0);
					return( FALSE);
				}
				else if ((uiNum1 < 1) || (uiNum1 > 31))
				{
					goto MonthDay_Error;
				}
				else
				{
					bHaveDay = TRUE;
					bHaveMonth = TRUE;
					*puiMonth = uiSaveNum;
					*puiDay = uiNum1;
				}
			}
			else if ((uiNum1 < 1) || (uiNum1 > 31))
			{
				goto MonthDay_Error;
			}
			else
			{
				if (uiNum1 <= 12)
				{
					bHaveNum = TRUE;
					uiSaveNum = uiNum1;
				}
				else
				{
					bHaveDay = TRUE;
					*puiDay = uiNum1;
				}
			}
		}
		else if (bHaveMonth)
		{
			if (bHaveDay)
			{
				goto Year_Error;
			}
			else
			{
				goto Day_Error;
			}
		}
		else
		{

			// See if it is a month string

			for (uiLoop = 0; uiLoop < 12; uiLoop++)
			{
				if (f_stricmp( pszDomeditMonths [uiLoop], szToken) == 0)
				{
					bHaveMonth = TRUE;
					*puiMonth = (uiLoop+1);
					break;
				}
			}

			if (uiLoop == 12)
			{
				for (uiLoop = 0; uiLoop < 12; uiLoop++)
				{
					if (f_stricmp( pszDomeditFullMonthNames [uiLoop], 
											szToken) == 0)
					{
						bHaveMonth = TRUE;
						*puiMonth = (uiLoop+1);
						break;
					}
				}
				if (uiLoop == 12)
				{
					goto Month_Error;
				}
			}
		}
		s = domeditSkipChars( s, " \t\n\r");
		if (bNoFormat)
		{
			if ((bHaveMonth) && (bHaveDay) && (*s == ','))
			{
				s++;
			}
		}
		else if (bSlashFormat)
		{
			if (*s != '/')
			{
				goto Invalid_Format;
			}
			s++;
		}
		else if (bDashFormat)
		{
			if (*s != '-')
			{
				goto Invalid_Format;
			}
			s++;
		}
		else if (*s == '/')
		{
			bSlashFormat = TRUE;
			s++;
		}
		else if (*s == '-')
		{
			bDashFormat = TRUE;
			s++;
		}
		else
		{
			bNoFormat = TRUE;
		}
	}
Invalid_Format:
Year_Error:
Month_Error:
Day_Error:
MonthDay_Error:
	flmAssert( 0);
	return( FALSE);
}



/****************************************************************************
Desc: This routine determines if the passed in token is a number.  If so,
		the number is returned.
****************************************************************************/
FSTATIC FLMBOOL domeditIsNum(
	const char *	pszToken,
	FLMUINT *		puiNum)
{
	FLMBOOL			bIsNum = FALSE;
	const char *	pszBuffer;
	FLMUINT			uiLen;

	// Make sure all characters are between 0 and 9

	pszBuffer = pszToken;

	// See if it is a HEX number.

	if ((*pszBuffer == '0') && (*(pszBuffer + 1) == 'x'))
	{
		pszBuffer += 2;
		uiLen = 0;
		while ((*pszBuffer) && 
				 (((*pszBuffer >= '0') && (*pszBuffer <= '9')) ||
				  ((*pszBuffer >= 'a') && (*pszBuffer <= 'f')) ||
				  ((*pszBuffer >= 'A') && (*pszBuffer <= 'F'))))
		{
			uiLen++;
			pszBuffer++;
		}
		if ((!uiLen) || (uiLen > 8) || (*pszBuffer))
		{
			goto Exit;
		}

		// Convert the pszToken to a number

		*puiNum = 0;
		pszBuffer = pszToken + 2;
		while (*pszBuffer)
		{
			if ((*pszBuffer >= '0') && (*pszBuffer <= '9'))
			{
				*puiNum = (*puiNum << 4) + *pszBuffer - '0';
			}
			else if ((*pszBuffer >= 'a') && (*pszBuffer <= 'f'))
			{
				*puiNum = (*puiNum << 4) + *pszBuffer - 'a' + 10;
			}
			else
			{
				*puiNum = (*puiNum << 4) + *pszBuffer - 'A' + 10;
			}
			pszBuffer++;
		}
	}
	else
	{
		FLMUINT	uiLeadingZeroes = 0;

		// Skip leading zeroes

		while (*pszBuffer == '0')
		{
			uiLeadingZeroes++;
			pszBuffer++;
		}
		pszToken = pszBuffer;
		uiLen = 0;
		while ((*pszBuffer) && (*pszBuffer >= '0') && (*pszBuffer <= '9'))
		{
			uiLen++;
			pszBuffer++;
		}
		if ((!uiLen) && (uiLeadingZeroes))
		{
			*puiNum = 0;
			bIsNum = TRUE;
			goto Exit;
		}
		if ((!uiLen) || (uiLen > 10) || (*pszBuffer))
		{
			goto Exit;
		}

		// Make sure there are not more characters than we can handle

		if ((uiLen == 10) && (f_strcmp( pszToken, "4294967295") > 0))
		{
			goto Exit;
		}

		// Convert the pszToken to a number

		*puiNum = 0;
		pszBuffer = pszToken;
		while (*pszBuffer)
		{
			*puiNum = *puiNum * 10 + *pszBuffer - '0';
			pszBuffer++;
		}
	}
	bIsNum = TRUE;
Exit:
	return( bIsNum);
}

/****************************************************************************
Desc: This routine skips the characters in the string specified by
		pszCharsToSkip.
****************************************************************************/
FSTATIC char * domeditSkipChars(
	const char *	pszStr,
	const char *	pszCharsToSkip)
{
	const char *	pszTmp;

	while (*pszStr)
	{
		pszTmp = pszCharsToSkip;
		while ((*pszTmp) && (*pszTmp != *pszStr))
		{
			pszTmp++;
		}
		if (*pszTmp)
		{
			pszStr++;
		}
		else
		{
			break;
		}
	}
	return( (char *)pszStr);
}


/****************************************************************************
Desc:
****************************************************************************/
RCODE _domEditBackgroundThread(
	IF_Thread *			pThread)
{
	FLMUINT		uiCount = 0;

	while( !pThread->getShutdownFlag())
	{
		uiCount++;
		if( !(uiCount % 50))
		{
			if( RC_BAD( domEditVerifyRun()))
			{
				gv_bShutdown = TRUE;
				break;
			}
		}
		f_sleep( 1000);
	}

	return( NE_XFLM_OK);
}


/****************************************************************************
Desc:	Updates the name table from the database
*****************************************************************************/
RCODE F_DomEditor::refreshNameTable( void)
{
	DME_NAME_TABLE_INFO	nametableInfo;
	RCODE						rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled == TRUE);

	if (m_pNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	/*
	Call the callback to build the name table
	*/

	f_memset( &nametableInfo, 0, sizeof( DME_NAME_TABLE_INFO));

	if( m_pEventHook)
	{

		if( RC_BAD( rc = m_pEventHook( this, F_DOMEDIT_EVENT_NAME_TABLE,
			(void *)(&nametableInfo), m_EventData)))
		{
			goto Exit;
		}
		if (nametableInfo.bInitialized)
		{
			if ((m_pNameTable = nametableInfo.pNameTable) != NULL)
			{
				m_pNameTable->AddRef();
			}
		}
	}

	// Try the default initialization if no name table came back from
	// the event hook.

	if (!m_pNameTable)
	{

		// addRef is done by getNameTable.

		if (RC_BAD( rc = m_pDb->getNameTable( &m_pNameTable)))
		{
			if (rc != NE_XFLM_NO_NAME_TABLE)
			{
				goto Exit;
			}

			// If there is no name table (could be that m_pDb == NULL),
			// create one with at least the dictionary tags.

			if ((m_pNameTable = f_new F_NameTable) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
			if (RC_BAD( rc = m_pNameTable->addReservedDictTags()))
			{
				goto Exit;
			}
		}
	}


Exit:

	return( rc);
}

/*=============================================================================
Desc:	Convert an ascii string to unicode
*============================================================================*/
FSTATIC void asciiToUnicode(
	const char *		pszAsciiString,
	FLMUNICODE *		puzString)
{
	FLMUINT uiLoop = 0;
	
	// Convert to a unicode string.
	for( uiLoop=0; *pszAsciiString; pszAsciiString++)
	{
		if (*pszAsciiString == '&')
		{
			const char *	pszTmp = pszAsciiString + 1;
			if (*pszTmp == '#')
			{
				FLMUNICODE *	puzTmpPtr = &puzString[ uiLoop];
				pszTmp++;

				*puzTmpPtr = (FLMUNICODE)f_atoud( pszTmp);
				uiLoop++;
				pszAsciiString += 6;
			}
			else
			{
				puzString[ uiLoop++] = (FLMUNICODE)*pszAsciiString;
			}
		}
		else
		{
			puzString[ uiLoop++] = (FLMUNICODE)*pszAsciiString;
		}
	}
	puzString[ uiLoop] = (FLMUNICODE)0;
	return;
}


/*=============================================================================
Desc:	Convert a unicode string to ascii - separate buffer
*============================================================================*/
FSTATIC RCODE unicodeToAscii(
	FLMUNICODE *	puzString,
	char *			pszString,
	FLMUINT			uiLength
	)
{
	RCODE			rc = NE_XFLM_OK;
	char *		pszDest;
	FLMINT		iLength = uiLength;

	pszDest = pszString;
	while( iLength > 0 && *puzString)
	{

		if (*puzString < 32 || *puzString > 126)
		{
			if (iLength >= 7)
			{
				f_sprintf( pszDest, "&#%04u;",(unsigned)*puzString);
				pszDest += 7;
				iLength -= 7;
			}
			else
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
		}
		else
		{
			*pszDest = (char)*puzString;
			pszDest++;
			iLength--;
		}
		puzString++;
	}

	if (iLength < 0)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	if (pszDest && uiLength)
	{
		*pszDest = '\0';
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Get's a DOM node's name, allocating memory as needed.
****************************************************************************/
RCODE F_DomEditor::getNodeName(
	F_DOMNode *		pDOMNode,
	DME_ROW_INFO *	pRow,
	FLMUNICODE **	ppuzName,
	FLMUINT *		puiBufSize
	)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiLen = 0;
	FLMUINT					uiPrefixLen = 0;
	FLMUINT					uiNameId;
	FLMUINT					uiPrefixId;
	FLMUINT					uiType;
	FLMUNICODE *			puzBuffer = NULL;
	FLMUNICODE *			puzPrefix = NULL;
	FLMUNICODE *			puzAppendBuffer = NULL;

	if (puiBufSize)
	{
		*puiBufSize = 0;
	}

	if (ppuzName)
	{
		*ppuzName = NULL;
	}

	if (!pDOMNode)
	{
		flmAssert( pRow);
	}

	if (pDOMNode)
	{

		// See if there is a prefix...
		if (RC_BAD( rc = pDOMNode->getPrefixId( m_pDb, &uiPrefixId)))
		{
			goto Exit;
		}


		// First we get the name id, then we look up the value in the name table.
		if (RC_BAD( rc = pDOMNode->getNameId( m_pDb, &uiNameId)))
		{
			goto Exit;
		}

		switch (pDOMNode->getNodeType())
		{
			case DOCUMENT_NODE:
			case ELEMENT_NODE:
			case DATA_NODE:
			case COMMENT_NODE:
			case CDATA_SECTION_NODE:
			{
				uiType = ELM_ELEMENT_TAG;
				break;
			}
			case ATTRIBUTE_NODE:
			{
				uiType = ELM_ATTRIBUTE_TAG;
				break;
			}
			default:
				uiType = 0;
				flmAssert( 0);
				break;
		}

		if (uiPrefixId && ppuzName)
		{
			if( RC_BAD( rc = pDOMNode->getPrefix( m_pDb, 
				(FLMUNICODE *)NULL, 0, &uiPrefixLen)))
			{
				goto Exit;
			}

			// Allocate a buffer to hold the data.
			uiPrefixLen = (uiPrefixLen + 1) * sizeof( FLMUNICODE);
			if (RC_BAD( rc = f_calloc( uiPrefixLen, &puzPrefix)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDOMNode->getPrefix( m_pDb, puzPrefix, 
				uiPrefixLen, NULL)))
			{
				goto Exit;
			}
		}

	}
	else if (pRow->pVector)
	{
		uiNameId = pRow->pVector->getNameId( pRow->uiElementNumber);
		uiPrefixLen = 0;
		if (pRow->pVector->isAttr( pRow->uiElementNumber))
		{
			uiType = ELM_ATTRIBUTE_TAG;
		}
		else
		{
			uiType = ELM_ELEMENT_TAG;
		}
	}
	else
	{
		// No name that we can get... Should we return an error?
		goto Exit;
	}

	// Now get the name
	if (RC_BAD( rc = m_pNameTable->getFromTagTypeAndNum(
		m_pDb, uiType, uiNameId, NULL, NULL, &uiLen)))
	{
		goto Exit;
	}
	uiLen *= sizeof( FLMUNICODE);

	// The length returned is the number of characters.  It does
	// not count the null terminator.
	// We must allow for them when allocating a buffer.
	if (uiLen && ppuzName)
	{
		FLMUINT			uiTmpLen = uiLen + sizeof( FLMUNICODE);

		// Allocate a buffer to hold the data.
		uiLen = uiTmpLen;
		if (RC_BAD( rc = f_calloc( uiTmpLen, &puzBuffer)))
		{
			goto Exit;
		}


		// Now get the value...
		if (RC_BAD( rc = m_pNameTable->getFromTagTypeAndNum(
			m_pDb, uiType, uiNameId, puzBuffer, NULL, &uiTmpLen)))
		{
			goto Exit;
		}
	}

	// Append the results
	if (uiPrefixLen && uiLen && ppuzName)
	{
		if (RC_BAD( rc = f_calloc( uiLen + uiPrefixLen, &puzAppendBuffer)))
		{
			goto Exit;
		}
		f_memcpy( puzAppendBuffer, puzPrefix, uiPrefixLen - 2);
		puzAppendBuffer[ (uiPrefixLen >> 1) - 1] = (FLMUNICODE)':';
		f_memcpy( &puzAppendBuffer[ uiPrefixLen >> 1], puzBuffer, uiLen - 2);

		f_free( &puzPrefix);
		puzPrefix = NULL;
		f_free( &puzBuffer);
		puzBuffer = puzAppendBuffer;
		puzAppendBuffer = NULL;
		uiLen = uiLen + uiPrefixLen;
	}

	if (puiBufSize)
	{
		*puiBufSize = uiLen;
	}

	if (ppuzName)
	{
		*ppuzName = puzBuffer;
		puzBuffer = NULL;
	}

Exit:

	if (puzBuffer)
	{
		f_free( &puzBuffer);
	}
	if (puzPrefix)
	{
		f_free( &puzPrefix);
	}
	if (puzAppendBuffer)
	{
		f_free( &puzAppendBuffer);
	}

	return( rc);
}

/****************************************************************************
Desc:	Get's a DOM node's value name, allocating memory as needed.
****************************************************************************/
RCODE F_DomEditor::getNodeValue(
	F_DOMNode *		pDOMNode,
	FLMUNICODE **	ppuzValue,
	FLMUINT *		puiBufSize,
	FLMBOOL			//bStartTrans
	)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLen = 0;
	FLMUINT	uiDataType;

	if (puiBufSize)
	{
		*puiBufSize = 0;
	}

	*ppuzValue = NULL;

	if (RC_BAD( rc = pDOMNode->getDataType( m_pDb, &uiDataType)))
	{
		goto Exit;
	}

	switch (uiDataType)
	{
		case XFLM_NODATA_TYPE:
		{
			break;
		}
		case XFLM_TEXT_TYPE:
		case XFLM_NUMBER_TYPE:
		case XFLM_BINARY_TYPE:
		{
			if (RC_BAD( rc = pDOMNode->getUnicodeChars(m_pDb, &uiLen)))
			{
				goto Exit;
			}

			if (uiLen)
			{

				// The length returned does not allow for 2 NULL terminators.
				// We must allow for them when allocating a buffer.

				uiLen *= sizeof(FLMUNICODE);
				uiLen += sizeof(FLMUNICODE);
				if (RC_BAD( rc = f_calloc( uiLen, ppuzValue)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = pDOMNode->getUnicode( m_pDb, *ppuzValue, uiLen, 0, uiLen, &uiLen)))
				{
					goto Exit;
				}
			}

			break;
		}
		default:
			break;
	}

	if (puiBufSize)
	{
		*puiBufSize = uiLen;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Formats text.
****************************************************************************/
FSTATIC RCODE formatText(
	FLMUNICODE *		puzBuf,
	FLMBOOL				bQuoted,
	const char *		pszPreText,
	const char *		pszPostText,
	char **				ppszString)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUNICODE *			puzTmp = NULL;
	FLMUNICODE				uzNullChar;
	char *					pszString;
	FLMUINT					uiMaxStringLen = MAX_DISPLAY_SEGMENT;
	FLMUINT					uiStringLen = 0;
	F_DynaBuf				tmpBuf;

	flmAssert( ppszString);

	pszString = *ppszString;

	flmAssert( pszString);


	// NOTE: puzBuf may be NULL - treat same as empty string.

	if (!puzBuf)
	{
		uzNullChar = 0;
		puzTmp = &uzNullChar;
	}
	else
	{
		puzTmp = puzBuf;
	}

	// If the first character is a carriage return, skip over it.

	if (*puzTmp == (FLMUNICODE)'\n')
	{
		puzTmp++;
	}
	
	if (pszPreText)
	{
		if (RC_BAD( rc = tmpBuf.appendString( pszPreText)))
		{
			goto Exit;
		}
	}
	if (bQuoted)
	{
		if (RC_BAD( rc = tmpBuf.appendByte( '"')))
		{
			goto Exit;
		}
	}

	while (*puzTmp)
	{

		// Special handling for newline characters - want to add indent
		// spaces to each new line.

		if (*puzTmp == (FLMUNICODE)'\n')
		{
			puzTmp++;

			// Skip newline if it is the last character in the buffer.

			if (*puzTmp == 0)
			{
				break;
			}

			// Output whatever buffer we have built up.

			if (tmpBuf.getDataLength())
			{
				tmpBuf.appendByte( 0);
				
				f_sprintf( pszString, "%*.*s",
					0, (uiStringLen <= uiMaxStringLen 
								? uiMaxStringLen - uiStringLen 
								: 0), 
					tmpBuf.getBufferPtr());
				pszString += tmpBuf.getDataLength() - 1;
				uiStringLen += tmpBuf.getDataLength() - 1;

				if ((uiStringLen + 1) < uiMaxStringLen)
				{
					f_sprintf( pszString, " ");
					pszString ++;
					uiStringLen ++;
				}
				else
				{
					break;
				}

			}

			// Output a newline to end this line
			tmpBuf.truncateData( 0);

			// Have already skipped over the newline, so we just continue.

			continue;
		}

		// Add the character to the buffer.

		if (*puzTmp >= 32 && *puzTmp <= 126)
		{
			if (RC_BAD( rc = tmpBuf.appendByte( (char)*puzTmp)))
			{
				goto Exit;
			}
		}
		else
		{

			// Allow browser to render these - flush what we have in
			// our buffer up to this point, then output.

			if (tmpBuf.getDataLength())
			{
				tmpBuf.appendByte( 0);
				
				f_sprintf( pszString, "%*.*s",
					0, (uiStringLen <= uiMaxStringLen 
								? uiMaxStringLen - uiStringLen 
								: 0), 
					tmpBuf.getBufferPtr());
					
				pszString += tmpBuf.getDataLength() - 1;
				uiStringLen += tmpBuf.getDataLength() - 1;
				tmpBuf.truncateData( 0);
			}

			if (uiStringLen + 7 < uiMaxStringLen)
			{
				f_sprintf( pszString, "&#%04u;", (unsigned)(*puzTmp));
				pszString += 7;
				uiStringLen += 7;
			}
			else
			{
				break;
			}

		}

		puzTmp++;
	}

	if (tmpBuf.getDataLength())
	{
		tmpBuf.appendByte( 0);
		
		f_sprintf( pszString, "%*.*s",
			0, (uiStringLen <= uiMaxStringLen 
						? uiMaxStringLen - uiStringLen 
						: 0), 
			tmpBuf.getBufferPtr());
			
		pszString += tmpBuf.getDataLength() - 1;
		uiStringLen += tmpBuf.getDataLength() - 1;
		tmpBuf.truncateData( 0);
	}
	
	if (bQuoted)
	{
		if (RC_BAD( rc = tmpBuf.appendByte( '"')))
		{
			goto Exit;
		}
	}
	
	if (pszPostText)
	{
		if (RC_BAD( rc = tmpBuf.appendString( pszPostText)))
		{
			goto Exit;
		}
	}

	if (tmpBuf.getDataLength())
	{
		tmpBuf.appendByte( 0);
		f_sprintf( pszString, "%*.*s",
			0, (uiStringLen <= uiMaxStringLen 
					? uiMaxStringLen - uiStringLen 
					: 0), 
			tmpBuf.getBufferPtr());
		pszString += tmpBuf.getDataLength() - 1;
		tmpBuf.truncateData( 0);
	}

	*ppszString = pszString;

Exit:

	return( rc);
}

/*=============================================================================
Desc:	Format a Document Node
*============================================================================*/
FSTATIC RCODE formatDocumentNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzTitle = NULL;
	DME_DISP_COLUMN *	pDispVals;
	FLMUINT				uiCol;
	F_DOMNode *			pAnnotation = NULL;
	FLMUNICODE *		puzAttrValue = NULL;
	char *				pszString = NULL;
	eColorType			uiForeground;

	if (pRow->pDomNode)
	{
		uiForeground = pRow->pDomNode->isQuarantined() 
							? FLM_LIGHTCYAN
							: FLM_YELLOW;
	}
	else
	{
		uiForeground = FLM_YELLOW;
	}


	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;
	/*
	Output the Expand/Collapse symbol - check the flags
	*/

	if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%s", pRow->bExpanded ? "-" : pRow->bHasChildren ? "+" : " ");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + (pRow->uiLevel * 2) + 1;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}


	/*
	Output the display value
	*/
	if ( !(uiFlags & F_DOMEDIT_FLAG_ENDTAG))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, pRow->bExpanded ? "<DOC>" : 
			!pRow->bHasChildren ? "<DOC>" : "<DOC/>");
	}
	else
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, "</DOC>");
	}

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = pDomEditor->isMonochrome() ? FLM_WHITE : uiForeground;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + (pRow->uiLevel * 2) + 1;
	(*puiNumVals)++;

	if (RC_OK( rc = pRow->pDomNode->getAnnotation(
								pDomEditor->getDb(), (IF_DOMNode **)&pAnnotation)))
	{
		if (RC_OK( rc = pDomEditor->getNodeValue( pAnnotation, &puzAttrValue)))
		{
			pszString = &pDispVals[ *puiNumVals].szString[0];
			if (RC_BAD( rc = formatText(
								puzAttrValue, FALSE, "", "", &pszString)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = pDomEditor->isMonochrome() ? FLM_WHITE : uiForeground;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	(*puiNumVals)++;

Exit:

	if (pAnnotation)
	{
		pAnnotation->Release();
	}

	f_free( &puzTitle);
	f_free( &puzAttrValue);

	return rc;

}


/*=============================================================================
Desc:	Format an Element Node
*============================================================================*/
FSTATIC RCODE formatElementNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals,
	FLMUINT				uiFlags
	)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUNICODE *			puzName = NULL;
	FLMUNICODE *			puzAttrName = NULL;
	FLMUNICODE *			puzAttrValue = NULL;
	FLMUINT					uiCol;
	DME_DISP_COLUMN *		pDispVals;
	char *					pszString;
	F_DOMNode *				pAttrNode = NULL;
	F_DOMNode *				pChildNode = NULL;
	FLMBOOL					bGotFirstAttr = FALSE;
	F_Db *					pDb = NULL;
	FLMUINT					uiAttrLen;
	eColorType				uiForeground;
	FLMBOOL					bHasLocalData;

	if (RC_BAD( rc = pRow->pDomNode->isDataLocalToNode( pDomEditor->getDb(), &bHasLocalData)))
	{
		goto Exit;
	}

	if (pRow->pDomNode)
	{
		uiForeground = pRow->pDomNode->isQuarantined() 
							? FLM_LIGHTCYAN
							: FLM_WHITE;
	}
	else
	{
		uiForeground = FLM_WHITE;
	}

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pDb = pDomEditor->getSource();

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;
	/*
	Output the Expand/Collapse symbol - check the flags
	*/

	if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%s", pRow->bExpanded ? "-" : (pRow->bHasChildren || bHasLocalData) ? "+" : " ");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() 
																	? FLM_BLACK
																	: FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() 
																	? FLM_BLACK
																	: FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}


	if (RC_BAD( rc = pDomEditor->getNodeName( pRow->pDomNode, NULL, &puzName)))
	{
		goto Exit;
	}

	pszString = &pDispVals[ *puiNumVals].szString[
									f_strlen(pDispVals[ *puiNumVals].szString)];

	if ( !(uiFlags & F_DOMEDIT_FLAG_ENDTAG))
	{
		f_sprintf( pszString, "<");
		pszString++;
	}
	else
	{
		f_sprintf( pszString, "</");
		pszString += 2;
	}

	// Save the qualified name...
	if (RC_BAD( rc = formatText( puzName, FALSE, NULL, NULL, &pszString)))
	{
		goto Exit;
	}

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = uiForeground;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() 
										? FLM_BLACK
										: FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString);
	(*puiNumVals)++;

	if (!(uiFlags & F_DOMEDIT_FLAG_ENDTAG))
	{

		pszString = &pDispVals[ *puiNumVals].szString[
										f_strlen(pDispVals[ *puiNumVals].szString)];

		uiAttrLen = 0;

		for (;;)
		{
			if (!bGotFirstAttr)
			{
				// Get the first attribute
				if (RC_BAD(
						rc = pRow->pDomNode->getFirstAttribute( pDb,
														(IF_DOMNode **)&pAttrNode)))
				{
					if ( rc == NE_XFLM_NOT_FOUND ||
						  rc == NE_XFLM_EOF_HIT ||
						  rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						goto Exit;
					}
				}
				bGotFirstAttr = TRUE;
			}
			else
			{
				if (RC_BAD( rc = pAttrNode->getNextSibling(
										pDb, (IF_DOMNode **)&pAttrNode)))
				{
					if ( rc == NE_XFLM_NOT_FOUND ||
						  rc == NE_XFLM_EOF_HIT ||
						  rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						goto Exit;
					}
				}
			}

			if (RC_BAD( rc = pDomEditor->getNodeName( pAttrNode, NULL, &puzAttrName)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = formatText(
								puzAttrName, FALSE, " ", "=", &pszString)))
			{
				goto Exit;
			}
			f_free( &puzAttrName);

			// Get the attribute value now.  Look for it in the same node first.
			if (RC_BAD( rc = pDomEditor->getNodeValue( pAttrNode, &puzAttrValue)))
			{
				f_sprintf( pszString, "** NO VALUE FOUND **");
			}
			
			if (RC_BAD( rc = formatText(
								puzAttrValue, FALSE, "", "", &pszString)))
			{
				goto Exit;
			}

			f_free( &puzAttrValue);
			
			if ((uiCol + (pszString - &pDispVals[ *puiNumVals].szString[0])) >= 78)
			{
				f_sprintf( &pDispVals[ *puiNumVals].szString[ 78 - uiCol - 4], "...");
				break;
			}
		}

		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_LIGHTRED;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString);
		(*puiNumVals)++;
	}

	// Add the trailing '>'
	f_sprintf( pDispVals[ *puiNumVals].szString, pRow->bExpanded ? ">" :
			!pRow->bHasChildren ? ">" : "/>");

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = uiForeground;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString);
	(*puiNumVals)++;

Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	if (pChildNode)
	{
		pChildNode->Release();
	}

	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	f_free( &puzName);
	f_free( &puzAttrName);
	f_free( &puzAttrValue);

	return rc;
}

/******************************************************************************
Desc:	Format an attribute for display.
******************************************************************************/
FSTATIC RCODE formatAttributeNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags
	)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUNICODE *			puzName = NULL;
	FLMUNICODE *			puzAttrName = NULL;
	FLMUNICODE *			puzAttrValue = NULL;
	FLMUINT					uiCol;
	DME_DISP_COLUMN *		pDispVals;
	char *					pszString;

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;
	/*
	Output the Expand/Collapse symbol - check the flags
	*/

	if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%s", pRow->bExpanded ? "-" : pRow->bHasChildren ? "+" : " ");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}


	if (RC_BAD( rc = pDomEditor->getNodeName( pRow->pDomNode, NULL, &puzName)))
	{
		goto Exit;
	}

	pszString = &pDispVals[ *puiNumVals].szString[f_strlen(pDispVals[ *puiNumVals].szString)];

	if ( !(uiFlags & F_DOMEDIT_FLAG_ENDTAG))
	{
		f_sprintf( pszString, "<");
		pszString++;
	}
	else
	{
		f_sprintf( pszString, "</");
		pszString += 2;
	}

	// Save the qualified name...
	if (RC_BAD( rc = formatText( puzName, FALSE, " ", "=", &pszString)))
	{
		goto Exit;
	}

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString);
	(*puiNumVals)++;

	// Output the attributes, unless this is a terminator

	if (!(uiFlags & F_DOMEDIT_FLAG_ENDTAG))
	{
		
		pszString = &pDispVals[ *puiNumVals].szString[f_strlen(pDispVals[ *puiNumVals].szString)];

		// Get the attribute value
		if (RC_BAD( rc = pDomEditor->getNodeValue( pRow->pDomNode, &puzAttrValue)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = formatText( puzAttrValue, TRUE, "", "", &pszString)))
		{
			goto Exit;
		}
		f_free( &puzAttrValue);

	}


	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_LIGHTRED;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString)+1;
	(*puiNumVals)++;


	// Add the trailing '>'
	f_sprintf( pDispVals[ *puiNumVals].szString, pRow->bHasChildren ? ">" : "/>");

	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	(*puiNumVals)++;

Exit:

	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	f_free( &puzName);
	f_free( &puzAttrName);
	f_free( &puzAttrValue);

	return rc;
}


/*=============================================================================
Desc:	Format a Text Node
*============================================================================*/
FSTATIC RCODE formatDataNode(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags,
	const char *			pszPreText,
	const char *			pszPostText)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzValue = NULL;
	FLMBOOL				bStringEmpty;
	char *				pszString;
	FLMUINT				uiCol;
	DME_DISP_COLUMN *	pDispVals;
	eColorType			uiForeground;

	if (pRow->pDomNode)
	{
		uiForeground = pRow->pDomNode->isQuarantined() 
							? FLM_LIGHTCYAN
							: FLM_WHITE;
	}
	else
	{
		uiForeground = FLM_WHITE;
	}

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;

	// Output the Expand/Collapse symbol - check the flags

	if (pRow->bHasChildren)
	{

		if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
		{
			f_sprintf( pDispVals[ *puiNumVals].szString,
				"%s", pRow->bExpanded ? "-" : "+");
			pDispVals[ *puiNumVals].uiCol = uiCol;
			pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
			pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
			uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
			(*puiNumVals)++;
		}
	}
	else
	{
		uiCol += 3;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}


	if (RC_BAD( rc = pDomEditor->getNodeValue( pRow->pDomNode, &puzValue)))
	{
		goto Exit;
	}

	bStringEmpty = FALSE;
	if (!puzValue || !(*puzValue) || (*puzValue == '\n' && puzValue [1] == 0))
	{
		bStringEmpty = TRUE;
	}

	if (!bStringEmpty || pszPreText || pszPostText)
	{
		pszString = &pDispVals[ *puiNumVals].szString[
				f_strlen(pDispVals[ *puiNumVals].szString)];
		if (RC_BAD( rc = formatText( puzValue, FALSE, pszPreText, pszPostText, &pszString)))
		{
			goto Exit;
		}

		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = uiForeground;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		(*puiNumVals)++;
	}

Exit:

	f_free( &puzValue);
	return rc;
}


/*=============================================================================
Desc:	Format a Processing Instruction Node
*============================================================================*/
FSTATIC RCODE formatProcessingInstruction(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzName = NULL;
	FLMUNICODE *		puzValue = NULL;
	char *				pszString;
	FLMUINT				uiCol;
	DME_DISP_COLUMN *	pDispVals;

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;
	/*
	Output the Expand/Collapse symbol - check the flags
	*/

	if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%s", pRow->bExpanded ? "-" : "+");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	// Get the target name

	if (RC_BAD( rc = pDomEditor->getNodeName( pRow->pDomNode, NULL, &puzName)))
	{
		goto Exit;
	}

	// Get the target data

	if (RC_BAD( rc = pDomEditor->getNodeValue( pRow->pDomNode, &puzValue)))
	{
		goto Exit;
	}

	pszString = &pDispVals[ *puiNumVals].szString[
				f_strlen(pDispVals[ *puiNumVals].szString)];
	if (RC_BAD( formatText( puzName, FALSE, "<?", " ", &pszString)))
	{
		goto Exit;
	}
	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
	(*puiNumVals)++;

	pszString = &pDispVals[ *puiNumVals].szString[
				f_strlen(pDispVals[ *puiNumVals].szString)];
	if (RC_BAD( formatText( puzValue, FALSE, NULL, "?>", &pszString)))
	{
		goto Exit;
	}
	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	(*puiNumVals)++;

Exit:

	f_free( &puzName);
	f_free( &puzValue);

	return rc;
}

/*=============================================================================
Desc:	Format a generic node without using the DOM.
*============================================================================*/
FSTATIC RCODE formatRow(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals,
	FLMUINT					uiFlags
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCol = 0;
	DME_DISP_COLUMN *	pDispVals;

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));

	uiCol = pRow->uiLevel * 2;
	/*
	Output the Expand/Collapse symbol - check the flags
	*/

	if ( !(uiFlags & F_DOMEDIT_FLAG_HIDE_EXPAND))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%s", pRow->bExpanded ? "-" : "+");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	/*
	Output the level
	*/

	if( !(uiFlags & F_DOMEDIT_FLAG_HIDE_LEVEL))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString,
			"%u", pRow->uiLevel);
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		uiCol += 2;
	}

	if (RC_BAD( rc = unicodeToAscii(
		pRow->puzValue, pDispVals[ *puiNumVals].szString, unicodeStrLen( pRow->puzValue))))
	{
		goto Exit;
	}

	pDispVals[ *puiNumVals].uiCol = (uiFlags & F_DOMEDIT_FLAG_COMMENT ? 0 : uiCol);
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	(*puiNumVals)++;

Exit:
	return rc;
}


/******************************************************************************
Desc:
******************************************************************************/
FSTATIC void printNodeType(
	DME_ROW_INFO *		pRow,
	FTX_WINDOW *		pStatusWin
	)
{

	eDomNodeType		eType = pRow->eType;

	switch (eType)
	{
	case INVALID_NODE :
		FTXWinPrintf( pStatusWin, " | Type: Unknown");
		break;
	case ELEMENT_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Element");
		break;
	case DATA_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Text");
		break;
	case CDATA_SECTION_NODE:
		FTXWinPrintf( pStatusWin, " | Type: CDATA");
		break;
	case PROCESSING_INSTRUCTION_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Processing Instruction");
		break;
	case COMMENT_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Comment");
		break;
	case DOCUMENT_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Document");
		break;
	case ATTRIBUTE_NODE:
		FTXWinPrintf( pStatusWin, " | Type: Attribute");
		break;
	default:
		break;
	}
}


/******************************************************************************
Desc:
******************************************************************************/
FSTATIC void releaseRow(
	DME_ROW_INFO **		ppRow
	)
{
	DME_ROW_INFO *		pRow = *ppRow;

	if (pRow->puzValue)
	{
		f_free( &pRow->puzValue);
	}
	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
	}
	if (pRow->pVector && pRow->uiElementNumber == 0)
	{
		pRow->pVector->reset();
		pRow->pVector->Release();
	}
	*ppRow = pRow->pNext;
	if (pRow->pPrev)
	{
		pRow->pPrev->pNext = pRow->pNext;
	}
	if (pRow->pNext)
	{
		pRow->pNext->pPrev = pRow->pPrev;
	}

	f_free( &pRow);
}


/******************************************************************************
Desc:
******************************************************************************/
void F_DomEditor::releaseLastRow()
{
	DME_ROW_INFO *		pLastRow = m_pScrLastRow->pPrev;

	flmAssert( pLastRow);
	releaseRow( &m_pScrLastRow);
	m_pScrLastRow = pLastRow;
	m_pScrLastRow->pNext = NULL;
	m_uiNumRows--;
}



/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::getNextTitle(
	DME_ROW_INFO *			pRow,
	DME_ROW_INFO **		ppNewRow
	)
{
	RCODE							rc = NE_XFLM_OK;
	FLMUINT						uiIndex = 0;
	F_DOMNode *					pDOMNode = NULL;
	FLMUNICODE *				puzTitle=NULL;
	FLMUINT64					ui64DocumentID;
	FLMUINT64					ui64NodeID;
	DME_ROW_INFO *				pTmpRow = NULL;

	if (RC_BAD( rc = getDomNode(pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
										&pRow->pDomNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pRow->pDomNode->getNextDocument(
											m_pDb, (IF_DOMNode **)&pDOMNode)))
	{
		if (rc == NE_XFLM_NOT_FOUND || rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	if( RC_BAD( rc = pDOMNode->getNodeId( m_pDb, &ui64NodeID)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDOMNode->getDocumentId( m_pDb, &ui64DocumentID)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = makeNewRow( &pTmpRow, NULL, ui64NodeID)))
	{
		goto Exit;
	}

	pTmpRow->ui64DocId = ui64DocumentID;
	pTmpRow->uiIndex = uiIndex;
	pTmpRow->uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | 
							  F_DOMEDIT_FLAG_HIDE_EXPAND |
							  F_DOMEDIT_FLAG_LIST_ITEM |
							  F_DOMEDIT_FLAG_READ_ONLY);

	*ppNewRow = pTmpRow;
	
	pTmpRow = NULL;

Exit:

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	if (pDOMNode)
	{
		pDOMNode->Release();
	}
	f_free( &puzTitle);

	return rc;

}


/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::getPrevTitle(
	DME_ROW_INFO *			pRow,
	DME_ROW_INFO **		ppNewRow
	)
{
	RCODE							rc = NE_XFLM_OK;
	F_DOMNode *					pDOMNode = NULL;
	FLMUNICODE *				puzTitle=NULL;
	FLMUINT64					ui64DocumentID;
	DME_ROW_INFO *				pTmpRow = NULL;

	if (RC_BAD( rc = getDomNode(pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
										&pRow->pDomNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pRow->pDomNode->getPreviousDocument(
												m_pDb, (IF_DOMNode **)&pDOMNode)))
	{
		if (rc == NE_XFLM_NOT_FOUND || rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}
	
	if( RC_BAD( rc = pDOMNode->getNodeId( m_pDb, &ui64DocumentID)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = makeNewRow( &pTmpRow, NULL, ui64DocumentID)))
	{
		goto Exit;
	}

	pTmpRow->uiIndex--;

	pTmpRow->uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | 
							  F_DOMEDIT_FLAG_HIDE_EXPAND |
							  F_DOMEDIT_FLAG_LIST_ITEM |
							  F_DOMEDIT_FLAG_READ_ONLY);

	*ppNewRow = pTmpRow;
	
	pTmpRow = NULL;

Exit:

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	if (pDOMNode)
	{
		pDOMNode->Release();
	}
	f_free( &puzTitle);

	return rc;

}

/******************************************************************************
Desc:	Retrieve the dom node if not already present.
******************************************************************************/
FSTATIC RCODE getDOMNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow
	)
{
	RCODE					rc = NE_XFLM_OK;

	if (!pRow->pDomNode)
	{
		if (RC_BAD( rc = pDomEditor->getDomNode( pRow->ui64NodeId,
													pRow->eType == ATTRIBUTE_NODE
													? pRow->uiNameId
													: 0,
													&pRow->pDomNode)))
		{
			goto Exit;
		}
		pRow->eType = pRow->pDomNode->getNodeType();
	}

Exit:

	return rc;
}


/****************************************************************************
Desc:	Edit a text buffer
*****************************************************************************/
#define MIN_BUFSIZE 80
RCODE F_DomEditor::editTextBuffer(
	char **				ppszBuffer,
	FLMUINT				uiBufSize,
	FLMUINT *			puiTermChar
	)
{
	RCODE				rc = NE_XFLM_OK;
	char *			pszBuffer = *ppszBuffer;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 3;
	FLMUINT			uiNumWinCols;
	FTX_WINDOW *	pWindow = NULL;
	IF_FileHdl *	pFileHdl = NULL;

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
	FTXWinClear( pWindow);

	// Adjust the buffer size if needed.
	
	if (uiBufSize < MIN_BUFSIZE)
	{
		if (RC_BAD( rc = f_realloc( MIN_BUFSIZE, ppszBuffer)))
		{
			goto Exit;
		}
		pszBuffer = *ppszBuffer;
		if (!uiBufSize)
		{
			f_memset( pszBuffer, 0, MIN_BUFSIZE);
		}
		uiBufSize = MIN_BUFSIZE;
	}

	if( RC_BAD( rc = FTXLineEdit( pWindow, pszBuffer, uiBufSize, uiBufSize,
		NULL, puiTermChar)))
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
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

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::editRow(
	FLMUINT,				//uiCurRow,
	DME_ROW_INFO *		pCurRow,
	FLMBOOL				bReadOnly
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzBuffer = NULL;
	char *				pszBuffer = NULL;
	FLMUINT				uiBufSize;
	FLMUINT				uiTermChar;
	eDomNodeType	eType;
	FLMUINT				uiDataType;

	// Get the dom node
	if (RC_BAD( rc = getDOMNode( this, pCurRow)))
	{
		goto Exit;
	}

	eType = pCurRow->pDomNode->getNodeType();
	switch (eType)
	{
		case ATTRIBUTE_NODE:
		{
			if (RC_BAD( rc = getNodeValue( pCurRow->pDomNode, &puzBuffer, &uiBufSize)))
			{
				goto Exit;
			}

			if (bReadOnly)
			{
				if (RC_BAD( rc = f_calloc( uiBufSize + 2, &pszBuffer)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = f_calloc( (uiBufSize * 2) + 2, &pszBuffer)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
				{
					goto Exit;
				}
			}

			if (bReadOnly)
			{
				FLMUINT			uiRows;

				uiRows = uiBufSize / (m_uiEditCanvasCols - 4) + 3;

				if (RC_BAD( rc = FTXDisplayScrollWindow( m_pScreen, "View",
					pszBuffer, m_uiEditCanvasCols - 4, uiRows > 10 ? 10 : uiRows)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = editTextBuffer( &pszBuffer, uiBufSize * 2, &uiTermChar)))
				{
					goto Exit;
				}
			}

			// Save the results?
			if (uiTermChar == FKB_ENTER && !bReadOnly)
			{

				// Begin a transaction
				if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
				{
					goto Exit;
				}

				// Need to know the attribute data type.
				if (RC_BAD( pCurRow->pDomNode->getDataType( m_pDb, &uiDataType)))
				{
					goto Exit;
				}

				switch (uiDataType)
				{
					case XFLM_TEXT_TYPE:
					{
						// Get a new unicode buffer....
						f_free( &puzBuffer);
						if (RC_BAD( rc = f_calloc( (f_strlen( pszBuffer) * 2) + 2, &puzBuffer)))
						{
							goto Exit;
						}

						asciiToUnicode( pszBuffer, puzBuffer);
						if (RC_BAD( rc = pCurRow->pDomNode->setUnicode(
							m_pDb, puzBuffer, unicodeStrLen(puzBuffer), TRUE)))
						{
							(void)abortTransaction();
							goto Exit;
						}
						break;
					}
					case XFLM_NUMBER_TYPE:
					{
						FLMUINT64	ui64Value;
						
						if( RC_BAD( rc = getNumber( pszBuffer, &ui64Value, NULL)))
						{
							displayMessage( "Invalid number", rc,
								NULL, FLM_RED, FLM_WHITE);
							goto Exit;
						}

						if (RC_BAD( rc = pCurRow->pDomNode->setUINT64( m_pDb, ui64Value)))
						{
							goto Exit;
						}
						break;
					}
					case XFLM_BINARY_TYPE:
					{
						rc = RC_SET( NE_XFLM_BAD_DATA_TYPE);
						goto Exit;
					}
				}


				// Commit the transaction
				if (RC_BAD( rc = commitTransaction()))
				{
					goto Exit;
				}
			}

			break;
		}

		case DATA_NODE:
		case CDATA_SECTION_NODE:
		case COMMENT_NODE:
		{
		
			if (RC_BAD( rc = getNodeValue( pCurRow->pDomNode, &puzBuffer, &uiBufSize)))
			{
				goto Exit;
			}

			if (bReadOnly)
			{
				if (RC_BAD( rc = f_calloc( uiBufSize + 2, &pszBuffer)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = f_calloc( (uiBufSize * 2) + 2, &pszBuffer)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
				{
					goto Exit;
				}
			}

			if (bReadOnly)
			{
				FLMUINT			uiRows;

				uiRows = uiBufSize / (m_uiEditCanvasCols - 4) + 3;

				if (RC_BAD( rc = FTXDisplayScrollWindow( m_pScreen, "View",
					pszBuffer, m_uiEditCanvasCols - 4, uiRows > 10 ? 10 : uiRows)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = editTextBuffer( &pszBuffer, uiBufSize, &uiTermChar)))
				{
					goto Exit;
				}
			}

			// Save the results?
			if (uiTermChar == FKB_ENTER && !bReadOnly)
			{
				// Get a new unicode buffer....
				f_free( &puzBuffer);
				if (RC_BAD( rc = f_calloc( (f_strlen( pszBuffer) * 2) + 2, &puzBuffer)))
				{
					goto Exit;
				}

				asciiToUnicode( pszBuffer, puzBuffer);

				// Begin a transaction
				if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = pCurRow->pDomNode->setUnicode(
					m_pDb, puzBuffer, unicodeStrLen( puzBuffer) * sizeof( FLMUNICODE),
					TRUE)))
				{
					goto Exit;
				}

				// Commit the transaction
				if (RC_BAD( rc = commitTransaction()))
				{
					goto Exit;
				}
			}
			break;
		}

		case ELEMENT_NODE:
		{
			rc = selectElementAttribute( pCurRow, &uiTermChar);
			goto Exit;
		}
		
		default:
		{
			break;
		}
	}


Exit:

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}
	if (pCurRow->pDomNode)
	{
		pCurRow->pDomNode->Release();
		pCurRow->pDomNode = NULL;
	}

	f_free( &puzBuffer);
	f_free( &pszBuffer);

	return rc;
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::editIndexRow(
	DME_ROW_INFO *		pCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzBuffer = NULL;
	char *				pszBuffer = NULL;
	FLMUINT				uiBufSize = 0;
	FLMUINT				uiTermChar;
	F_DataVector *		pVector = pCurRow->pVector;
	FLMUINT				uiElementNumber = pCurRow->uiElementNumber;

	if (!pCurRow->bUseValue)
	{

		flmAssert( pVector);

		// Get the current value in the buffer or display the one in the buffer.
		switch( pVector->getDataType( uiElementNumber))
		{
			case XFLM_TEXT_TYPE:
			{

				if ((uiBufSize = pVector->getDataLength( uiElementNumber)) > 0)
				{

					if (RC_BAD( rc = f_calloc( uiBufSize, &pszBuffer)))
					{
						goto Exit;
					}
	
					if (RC_BAD( rc = pVector->getUTF8( uiElementNumber, 
						(FLMBYTE *)pszBuffer, &uiBufSize)))
					{
						goto Exit;
					}
				}
				else
				{
					uiBufSize = 30;
					if (RC_BAD( rc = f_calloc( uiBufSize, &pszBuffer)))
					{
						goto Exit;
					}
					*pszBuffer = 0;
				}
				break;
			}
			case XFLM_NUMBER_TYPE:
			{
				FLMUINT64		ui64Num;
				uiBufSize = 23;  // Largest number of charcters in a 64 bit number + 3.
				if (RC_BAD( rc = f_calloc( uiBufSize, &pszBuffer)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pVector->getUINT64( uiElementNumber, &ui64Num)))
				{
					if (rc == NE_XFLM_NOT_FOUND)
					{
						*pszBuffer = 0;
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
				else
				{
					f_sprintf( pszBuffer, "0x%I64x", ui64Num);
				}
				break;
			}
			case XFLM_NODATA_TYPE:
			case XFLM_BINARY_TYPE:
			default:
			{
				rc = RC_SET( NE_XFLM_BAD_DATA_TYPE);
				break;
			}
		}
	}
	else
	{
		puzBuffer = pCurRow->puzValue;
		pCurRow->puzValue = NULL;
		uiBufSize = unicodeStrLen( puzBuffer);
		if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = editTextBuffer( &pszBuffer, uiBufSize * 2, &uiTermChar)))
	{
		goto Exit;
	}

	// Save the results?
	if (uiTermChar == FKB_ENTER)
	{
		switch( pVector->getDataType( uiElementNumber))
		{
			case XFLM_UNKNOWN_TYPE:
			case XFLM_NODATA_TYPE:
			case XFLM_TEXT_TYPE:
			{
				if (RC_BAD( rc = pVector->setUTF8( uiElementNumber, 
					(FLMBYTE *)pszBuffer)))
				{
					goto Exit;
				}
				break;
			}
			case XFLM_NUMBER_TYPE:
			{
				FLMUINT64		ui64Num = f_atou64( pszBuffer);

				if (RC_BAD( rc = pVector->setUINT64( uiElementNumber, ui64Num)))
				{
					goto Exit;
				}
				break;
			}
		}
	}
	else
	{
		pCurRow->puzValue = puzBuffer;
		puzBuffer = NULL;
	}


Exit:

	f_free( &puzBuffer);
	f_free( &pszBuffer);

	return rc;
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::editIndexNode(
	DME_ROW_INFO *		pCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzBuffer = NULL;
	char *				pszBuffer = NULL;
	FLMUINT				uiBufSize = 0;
	FLMUINT				uiTermChar;
	F_DataVector *		pVector = pCurRow->pVector;
	FLMUINT				uiElementNumber = pCurRow->uiElementNumber;
	FLMUINT64			ui64NodeId;

	if (!pCurRow->bUseValue)
	{

		flmAssert( pVector);

		ui64NodeId = pVector->getID( uiElementNumber);
		uiBufSize = 23;  // Largest number of charcters in a 64 bit number + 3.
		if (RC_BAD( rc = f_calloc( uiBufSize, &pszBuffer)))
		{
			goto Exit;
		}
		f_sprintf( pszBuffer, "%I64u", ui64NodeId);
	}
	else
	{
		puzBuffer = pCurRow->puzValue;
		pCurRow->puzValue = NULL;
		uiBufSize = unicodeStrLen( puzBuffer);
		if (RC_BAD( rc = unicodeToAscii( puzBuffer, pszBuffer, uiBufSize)))
		{
			goto Exit;
		}
	}


	if (RC_BAD( rc = editTextBuffer( &pszBuffer, uiBufSize * 2, &uiTermChar)))
	{
		goto Exit;
	}

	// Save the results?
	if (uiTermChar == FKB_ENTER)
	{
		FLMUINT64		ui64Num = f_atou64( pszBuffer);

		if (RC_BAD( rc = pVector->setID( uiElementNumber, ui64Num)))
		{
			goto Exit;
		}
	}
	else
	{
		pCurRow->puzValue = puzBuffer;
		puzBuffer = NULL;
	}


Exit:

	f_free( &puzBuffer);
	f_free( &pszBuffer);

	return rc;
}



/****************************************************************************
Desc:	Allows the user to interactively select and edit the attributes of an
		element node
*****************************************************************************/
RCODE F_DomEditor::selectElementAttribute(
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiTermChar)
{
	RCODE						rc = NE_XFLM_OK;
	DME_ROW_INFO *			pTmpRow = NULL;
	DME_ROW_INFO *			pPrevRow = NULL;
	FLMUINT					uiFlags;
	F_DomEditor *			pAttributeList = NULL;
	F_DOMNode *				pAttrNode = NULL;
	FLMBOOL					bGotFirstAttr;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if (pRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG)
	{
		rc = displayMessage( "Cannot edit Element end tag",
			NE_XFLM_FAILURE, puiTermChar, FLM_RED, FLM_WHITE);
		goto Exit;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pAttributeList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pAttributeList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pAttributeList->setParent( this);
	pAttributeList->setReadOnly( FALSE);
	pAttributeList->setShutdown( m_pbShutdown);
	pAttributeList->setTitle( "Attributes - Select One");
	pAttributeList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pAttributeList->setSource( m_pDb, m_uiCollection);

	if( m_pDb == NULL)
	{
		goto Exit;
	}
	
	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_NOPARENT | F_DOMEDIT_FLAG_NOCHILD);

	// Get the dom node for the selected row (element)
	if (RC_BAD( rc = getDOMNode( this, pRow)))
	{
		goto Exit;
	}

	// Verify that the node is an element node.
	if (pRow->pDomNode->getNodeType() != ELEMENT_NODE)
	{
		rc = displayMessage( "Invalid node type",
			NE_XFLM_FAILURE, puiTermChar, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	// Get the attributes...

	bGotFirstAttr = FALSE;
	for (;;)
	{
		if (!bGotFirstAttr)
		{
			// Get the first attribute
			if (RC_BAD( rc = pRow->pDomNode->getFirstAttribute( m_pDb,
											(IF_DOMNode **)&pAttrNode)))
			{
				if (rc == NE_XFLM_NOT_FOUND || rc == NE_XFLM_EOF_HIT  || rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			if (RC_BAD( rc = buildNewRow( 0,
													pAttrNode,
													&pTmpRow)))
			{
				goto Exit;
			}
			bGotFirstAttr = TRUE;
		}
		else
		{
			if (RC_BAD( rc = pAttrNode->getNextSibling(
									m_pDb, (IF_DOMNode **)&pAttrNode)))
			{
				if (rc == NE_XFLM_NOT_FOUND ||
					 rc == NE_XFLM_EOF_HIT ||
					 rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			if (RC_BAD( rc = buildNewRow( 0,
													pAttrNode,
													&pTmpRow)))
			{
				goto Exit;
			}
		}


		pTmpRow->uiFlags = uiFlags;

		if (RC_BAD( rc = pAttributeList->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;
	}

	if (pAttrNode)
	{
		pAttrNode->Release();
		pAttrNode = NULL;
	}


	// Set the start row.
	pAttributeList->setCurrentAtTop();

	if( RC_BAD( rc = pAttributeList->interactiveEdit(
											m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pAttributeList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pAttributeList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pAttributeList->setControlFlags( pTmpRow, uiFlags);
			break;
		}
		if (RC_BAD( rc = pAttributeList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if (pTmpRow)
	{
		if (RC_BAD( rc = editRow( 0, pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pAttributeList->getLastKey();
	}

Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	if( pAttributeList)
	{
		pAttributeList->Release();
		pAttributeList = NULL;
	}

	return( rc);
}


/******************************************************************************
Desc:	Method to delete a node (row)
******************************************************************************/
RCODE F_DomEditor::deleteRow(
	DME_ROW_INFO **		ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pRow = *ppCurRow;
	FLMBOOL				bNextRow = TRUE;
	FLMUINT				uiTermChar;
	char					szMessage[ 100];
	char					szResponse[ 3];

	if (!canDeleteRow(pRow))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Confirm this action before deleting...
	switch (pRow->eType)
	{
		case DOCUMENT_NODE:
			f_sprintf( szMessage, "Delete DOCUMENT_NODE [%,I64u]? ",
				pRow->ui64DocId);
			break;
		case ELEMENT_NODE:
			f_sprintf( szMessage, "Delete ELEMENT_NODE [%,I64u]? ",
				pRow->ui64NodeId);
			break;
		case ATTRIBUTE_NODE:
			f_sprintf( szMessage, "Delete ATTRIBUTE_NODE [%,I64u]? ",
				pRow->ui64NodeId);
			break;
		case DATA_NODE:
			f_sprintf( szMessage, "Delete DATA_NODE [%,I64u]? ",
				pRow->ui64NodeId);
			break;
		default:
			f_sprintf( szMessage, "Delete UNKNOWN_NODE [%,I64u]? ",
				pRow->ui64NodeId);
			break;
	}

	f_memset( szResponse, 0, sizeof(szResponse));

	requestInput(	szMessage,
						&szResponse[ 0], sizeof( szResponse), &uiTermChar);

	if( uiTermChar == FKB_ESCAPE || szResponse[ 0] == 'N' || szResponse[ 0] == 'n')
	{
		goto Exit;
	}

	if ( szResponse[ 0] != 'Y' && szResponse[ 0] != 'y')
	{
		goto Exit;
	}
	
	// If the current row is expanded, we first need to collapse it.
  	if (pRow->bExpanded)
	{
		if (RC_BAD( rc = collapseRow( &pRow)))
		{
			goto Exit;
		}
	}

	// Start an update transaction
	if( RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

	// Get the dom for this row but don't start a new transaction.
	if (RC_BAD( rc = getDOMNode( this, pRow)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pRow->pDomNode->deleteNode( m_pDb)))
	{
		(void)abortTransaction();
		goto Exit;
	}
	pRow->pDomNode->Release();
	pRow->pDomNode = NULL;

	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}



	// return the next or previous row.
	if (RC_BAD( rc = getNextRow( pRow, ppCurRow, TRUE)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if (*ppCurRow == NULL)
	{
		if (RC_BAD( rc = getPrevRow( pRow, ppCurRow, TRUE)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		bNextRow = FALSE;
	}

	if (m_pScrFirstRow == pRow)
	{
		m_pScrFirstRow = *ppCurRow;
	}

	if (m_pScrLastRow == pRow)
	{
		m_pScrLastRow = *ppCurRow;
	}


	if (pRow->ui64NodeId == pRow->ui64DocId)
	{
		DME_ROW_INFO *		pDoc = m_pCurDoc;
		// will need to remove this Document from the DocList.
		checkDocument( &pDoc, pRow);

		if (pDoc)
		{

			if (pDoc->pPrev)
			{
				pDoc->pPrev->pNext = pDoc->pNext;
			}
			if (pDoc->pNext)
			{
				pDoc->pNext->pPrev = pDoc->pPrev;
			}

			if (m_pDocList == pDoc)
			{
				m_pDocList = pDoc->pNext;
			}

			m_pCurDoc = pDoc->pNext;
			f_free( &pDoc);
		}
	}

	releaseRow( &pRow);
	m_uiNumRows--;
	
	// See if we can get a new last row.
	if ( *ppCurRow)
	{
		while (m_uiNumRows < m_uiEditCanvasRows)
		{
			if (RC_BAD( rc = getNextRow( pRow, &pRow, TRUE)))
			{
				goto Exit;
			}
			if (pRow == NULL)
			{
				break;
			}
		}
	}

	if (pRow)
	{
		m_pScrLastRow = pRow;
		pRow = NULL;
	}

	// If the new current row is null, then point to the last row - it too may be null
	// but in that case, there is nothing to display.
	if (*ppCurRow == NULL)
	{
		*ppCurRow = m_pScrLastRow;
		bNextRow = FALSE;
	}

	// If the new cursor row is a parent of the node we just deleted,
	// we need to re-set the bHasChildren flag.  There may not be anymore
	// children, so we want this to display correctly.
	pRow = *ppCurRow;

	if (pRow)
	{
		if (pRow->bExpanded && 
			 ((bNextRow && pRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG) ||
			 (!bNextRow && !(pRow->uiFlags & F_DOMEDIT_FLAG_ENDTAG))))
		{
			if (RC_BAD( rc = collapseRow( &pRow)))
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = getDOMNode( this, pRow)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pRow->pDomNode->hasChildren(
							m_pDb, &pRow->bHasChildren)))
		{
			goto Exit;
		}
	}

	// Sync the document list to the row.
	checkDocument(&m_pCurDoc, pRow);

Exit:

	if (pRow)
	{
		if (pRow->pDomNode)
		{
			pRow->pDomNode->Release();
			pRow->pDomNode = NULL;
		}
	}

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return rc;
}


/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::addSomething(
	DME_ROW_INFO **	ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	eDomNodeType		eNodeType;
	FLMUINT				uiTermChar;

	// Select the type of Node that we are going to add.
	if (RC_BAD( rc = selectNodeType( &eNodeType, &uiTermChar)))
	{
		goto Exit;
	}
	
	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	switch (eNodeType)
	{
		case ELEMENT_NODE:
		{
			if (*ppCurRow)
			{
				if (RC_BAD( rc = createElementNode( ppCurRow)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = createRootElementNode( ppCurRow)))
				{
					goto Exit;
				}
			}
			break;
		}
		case ATTRIBUTE_NODE:
		{
			if (RC_BAD( rc = createAttributeNode( ppCurRow)))
			{
				goto Exit;
			}
			break;
		}
		case DATA_NODE:
		case COMMENT_NODE:
		case CDATA_SECTION_NODE:
		{
			if (RC_BAD( rc = createTextNode( ppCurRow, eNodeType)))
			{
				goto Exit;
			}
			break;
		}
		case DOCUMENT_NODE:
		{
			if (RC_BAD( rc = createDocumentNode( ppCurRow)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = addDocumentToList(
						m_uiCollection, (*ppCurRow)->ui64DocId)))
			{
				goto Exit;
			}

			break;
		}
		case PROCESSING_INSTRUCTION_NODE:
		default:
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

Exit:

	return rc;
}


/******************************************************************************
Desc:
******************************************************************************/
RCODE F_DomEditor::createDocumentNode(
	DME_ROW_INFO **			ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	char					szResponse[ 128];
	FLMUNICODE *		puzTitle = NULL;
	FLMUINT				uiTitleLen;
	FLMUINT				uiTermChar;
	F_DOMNode *			pDocNode = NULL;
	DME_ROW_INFO *		pNewRow = NULL;
	IF_DOMNode *		pSource = NULL;

	f_memset( szResponse, 0, sizeof(szResponse));

	requestInput( "Document Title / Comment",
						szResponse, sizeof( szResponse), &uiTermChar);

	if( uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	uiTitleLen = f_strlen( szResponse);

	if (RC_BAD( rc = f_calloc( (uiTitleLen * 2) + 2, &puzTitle)))
	{
		goto Exit;
	}

	asciiToUnicode( szResponse, puzTitle);


	// Begin a transaction
	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	// Create a new document node.
	if (RC_BAD( m_pDb->createDocument(
		m_uiCollection, (IF_DOMNode **)&pDocNode)))
	{
		goto Exit;
	}

	// save the source

	if( RC_BAD( rc = pDocNode->createAttribute( 
		m_pDb, ATTR_SOURCE_TAG, &pSource)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pSource->setUnicode(
		(IF_Db *)m_pDb, puzTitle, unicodeStrLen(puzTitle), TRUE)))
	{
		goto Exit;
	}

	if (m_uiNumRows < m_uiEditCanvasRows)
	{
		// Create a new row and link it to the end of the list.
		if (RC_BAD( rc = buildNewRow( 0,
												(F_DOMNode *)pDocNode,
												&pNewRow)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = insertRow( pNewRow, m_pScrLastRow)))
		{
			goto Exit;
		}

		// May Need to remove the first row.
		if (m_uiNumRows > m_uiEditCanvasRows)
		{
			releaseRow( &m_pScrFirstRow);
			m_uiNumRows--;
		}

		*ppCurRow = pNewRow;
		pNewRow = NULL;
	}

	if (RC_BAD( rc = m_pDb->documentDone( pDocNode)))
	{
		goto Exit;
	}

	// Commit...
	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}

Exit:

	if (pNewRow)
	{
		releaseRow( &pNewRow);
	}

	if (pDocNode)
	{
		pDocNode->Release();
	}

	if (pSource)
	{
		pSource->Release();
	}

	f_free( &puzTitle);

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return rc;
}


/*=============================================================================
Desc:	Method to create a root element node rather than a document node.
=============================================================================*/
RCODE F_DomEditor::createRootElementNode(
	DME_ROW_INFO **			ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	char					szResponse[ 128];
	FLMUNICODE *		puzTitle = NULL;
	FLMUINT				uiTitleLen;
	FLMUINT				uiTermChar;
	F_DOMNode *			pRootNode = NULL;
	DME_ROW_INFO *		pNewRow = NULL;
	IF_DOMNode *			pSource = NULL;
	FLMUINT				uiTag;
	RCODE					tmpRc;
	FLMUINT				uiCollection;

	if( RC_BAD( tmpRc = selectCollection( &uiCollection, &uiTermChar)))
	{
		displayMessage( "Error getting collection", tmpRc,
			NULL, FLM_RED, FLM_WHITE);
	}
	
	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	f_memset( szResponse, 0, sizeof(szResponse));

	requestInput( "Element Title / Comment",
						szResponse,
						sizeof( szResponse),
						&uiTermChar);

	if( uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	uiTitleLen = f_strlen( szResponse);

	if (RC_BAD( rc = f_calloc( (uiTitleLen * 2) + 2, &puzTitle)))
	{
		goto Exit;
	}

	asciiToUnicode( szResponse, puzTitle);


	// Select the name tag for the element node
	if (RC_BAD( rc = selectTag( ELM_ELEMENT_TAG, &uiTag, &uiTermChar)))
	{
		goto Exit;
	}

	if( uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	// Begin a transaction
	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	// Create a new document node.
	if (RC_BAD( m_pDb->createRootElement( uiCollection,
													  uiTag,
													  (IF_DOMNode **)&pRootNode)))
	{
		goto Exit;
	}

	// save the source

	if (RC_BAD( rc = pRootNode->createAttribute( 
		m_pDb, ATTR_SOURCE_TAG, &pSource)))
	{
		goto Exit;
	}


	if (RC_BAD( rc = pSource->setUnicode( (IF_Db *)m_pDb,
															puzTitle,
															unicodeStrLen(puzTitle),
															TRUE)))
	{
		goto Exit;
	}

	if (uiCollection == m_uiCollection)
	{
		if (m_uiNumRows < m_uiEditCanvasRows)
		{
			// Create a new row and link it to the end of the list.
			if (RC_BAD( rc = buildNewRow( 0,
													(F_DOMNode *)pRootNode,
													&pNewRow)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = insertRow( pNewRow,
												 m_pScrLastRow)))
			{
				goto Exit;
			}

			// May Need to remove the first row.
			if (m_uiNumRows > m_uiEditCanvasRows)
			{
				releaseRow( &m_pScrFirstRow);
				m_uiNumRows--;
			}

			*ppCurRow = pNewRow;
			pNewRow = NULL;
		}

		if (RC_BAD( rc = addDocumentToList( m_uiCollection,
														(*ppCurRow)->ui64DocId)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_pDb->documentDone( pRootNode)))
	{
		goto Exit;
	}

	// Commit...
	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}

Exit:

	if (pNewRow)
	{
		releaseRow( &pNewRow);
	}

	if (pRootNode)
	{
		pRootNode->Release();
	}

	if (pSource)
	{
		pSource->Release();
	}

	f_free( &puzTitle);

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return rc;
}


/****************************************************************************
Desc:	Create an ELEMENT_ NODE Dom node.
*****************************************************************************/
RCODE F_DomEditor::createElementNode(
	DME_ROW_INFO **		ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pRow = *ppCurRow;
	char					szResponse[ 5];
	FLMUINT				uiTermChar;
	FLMUINT				uiTag;
	F_DOMNode *			pRefNode = NULL;
	F_DOMNode *			pElementNode = NULL;
	FLMUINT64			ui64RootId;
	FLMBOOL				bAddSibling = FALSE;

	f_memset( szResponse, 0, sizeof(szResponse));

	requestInput( "Create a Root Element?",
						szResponse,
						sizeof( szResponse),
						&uiTermChar);

	if( uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	if (szResponse[0] == 'Y' || szResponse[0] == 'y')
	{
		rc = createRootElementNode( ppCurRow);
		goto Exit;
	}

	if (!pRow)
	{
		displayMessage( "No DOM Node to add to.",
			NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	// Verify that the  node we are looking at can take an element node. Some
	// dom node types can't take a sibling or a child element node.
	if (pRow->eType != DOCUMENT_NODE &&
		 pRow->eType != ELEMENT_NODE)
	{
		displayMessage( "Invalid DOM node type",
			NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	// Select the name tag for the element node
	if (RC_BAD( rc = selectTag( ELM_ELEMENT_TAG, &uiTag, &uiTermChar)))
	{
		goto Exit;
	}

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

	// Get the dom node for this row.
	if (RC_BAD( rc = getDomNode( pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
											&pRefNode)))
	{
		goto Exit;
	}

	// Get the root node.
	
	if( RC_BAD( rc = pRefNode->getDocumentId( m_pDb, &ui64RootId)))
	{
		goto Exit;
	}

	// We need to make sure we have the right root node.
	switch (pRow->eType)
	{
		case DOCUMENT_NODE:
		{
			pRow->bHasChildren = TRUE;
			// Got it already
			break;
		}
		case ELEMENT_NODE:
		{
			// If we are inserting as a sibling to this node, we need to get the parent node.
			// Do we want to add this element as a child or a sibling to the current node.
			f_memset( szResponse, 0, sizeof( szResponse));
			requestInput( "Insert as Sibling node?",
				szResponse, sizeof( szResponse), &uiTermChar);

			if (uiTermChar == FKB_ESCAPE)
			{
				goto Exit;
			}

			if ( szResponse[ 0] == 'Y' || szResponse[ 0] == 'y')
			{
				// Need to get the parent node.
				if (RC_BAD( rc = pRefNode->getParentNode(
									m_pDb, (IF_DOMNode **)&pRefNode)))
				{
					goto Exit;
				}
				bAddSibling = TRUE;
			}
		}
		default:
		{
			break;
		}
	}

	if (RC_BAD( rc = pRefNode->createNode(
					m_pDb, ELEMENT_NODE, uiTag, XFLM_LAST_CHILD, 
					(IF_DOMNode **)&pElementNode)))
	{
		goto Exit;
	}

	if (bAddSibling)
	{
		DME_ROW_INFO *		pNewRow = NULL;
		FLMUINT				uiLevel = pRow->uiLevel;

		// We need to insert the new row into the list
		if (RC_BAD( rc = buildNewRow( (FLMINT)pRow->uiLevel,
												pElementNode,
												&pNewRow)))
		{
			goto Exit;
		}

		if (pRow->pDomNode)
		{
			pRow->pDomNode->Release();
			pRow->pDomNode = NULL;
		}

		while (pRow && pRow->pNext && pRow->pNext->uiLevel >= uiLevel)
		{
			pRow = pRow->pNext;
		}

		if (pRow && (pRow != m_pScrLastRow || m_uiNumRows < m_uiEditCanvasRows))
		{
			if (RC_BAD( rc = insertRow( pNewRow, pRow)))
			{
				goto Exit;
			}
		}

	}
	else
	{
		pRow->bHasChildren = TRUE;

		if (pRow->bExpanded)
		{
			// Collapse the row, then expand it.
			if (RC_BAD( rc = collapseRow( &pRow)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = expandRow( pRow, TRUE, NULL)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = m_pDb->documentDone( pRefNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}
	

Exit:


	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	if (pRefNode)
	{
		pRefNode->Release();
	}

	if (pElementNode)
	{
		pElementNode->Release();
	}

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return rc;
}

/****************************************************************************
Desc:	Method to create text type nodes.
*****************************************************************************/
RCODE F_DomEditor::createTextNode(
	DME_ROW_INFO **		ppCurRow,
	eDomNodeType		eNodeType
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pRow = *ppCurRow;
	DME_ROW_INFO *		pNewRow = NULL;
	char					szResponse[ 5];
	FLMUINT				uiTermChar;
	F_DOMNode *			pRefNode = NULL;
	F_DOMNode *			pTextNode = NULL;
	FLMUINT64			ui64RootId;
	FLMBOOL				bAddSibling = FALSE;
	char					szTextBuffer[ 120];
	FLMUINT				uiTextBufLen = sizeof(szTextBuffer);
	FLMBOOL				bHasChildren = pRow->bHasChildren;

	// Verify that the  node we are looking at can take the type of 
	// node either as a child or sibling...
	if (pRow->eType != DOCUMENT_NODE &&
		 pRow->eType != ATTRIBUTE_NODE &&
		 pRow->eType != ELEMENT_NODE)
	{
		displayMessage( "Invalid DOM node type",
			NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	f_memset( szTextBuffer, 0, uiTextBufLen);
	requestInput( "Text",
				szTextBuffer, uiTextBufLen, &uiTermChar);

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}


	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

	// Get the dom node for this row.
	if (RC_BAD( rc = getDomNode( pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
											&pRefNode)))
	{
		goto Exit;
	}

	// Get the root node.
	
	if( RC_BAD( rc = pRefNode->getDocumentId( m_pDb, &ui64RootId)))
	{
		goto Exit;
	}

	// We need to make sure we have the right root node.
	switch (pRow->eType)
	{
		case DOCUMENT_NODE:
		{
			pRow->bHasChildren = TRUE;
			// No parent node for these
			break;
		}
		default:
		{
			// If we are inserting as a sibling to this node, we need to get the parent node.
			// Do we want to add this element as a child or a sibling to the current node.
			f_memset( szResponse, 0, sizeof( szResponse));
			requestInput( "Insert as Sibling node?",
				szResponse, sizeof( szResponse), &uiTermChar);

			if (uiTermChar == FKB_ESCAPE)
			{
				goto Exit;
			}

			if ( szResponse[ 0] == 'Y' || szResponse[ 0] == 'y')
			{
				// Need to get the parent node.
				if (RC_BAD( pRefNode->getParentNode(
								m_pDb, (IF_DOMNode **)&pRefNode)))
				{
					goto Exit;
				}
				bAddSibling = TRUE;
			}
		}
	}

	if (RC_BAD( rc = pRefNode->createNode(
					m_pDb, eNodeType, 0, XFLM_LAST_CHILD, 
					(IF_DOMNode **)&pTextNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pTextNode->setUTF8( m_pDb, (FLMBYTE *)szTextBuffer)))
	{
		goto Exit;
	}

	if (bAddSibling)
	{
		FLMUINT				uiLevel = pRow->uiLevel;

		// We need to insert the new row into the list
		if (RC_BAD( rc = buildNewRow( (FLMINT)pRow->uiLevel,
												pTextNode,
												&pNewRow)))
		{
			goto Exit;
		}

		pTextNode->AddRef();

		if (pRow->pDomNode)
		{
			pRow->pDomNode->Release();
			pRow->pDomNode = NULL;
		}

		while (pRow && pRow->pNext && pRow->pNext->uiLevel >= uiLevel)
		{
			pRow = pRow->pNext;
		}

		if (pRow && (pRow != m_pScrLastRow || m_uiNumRows < m_uiEditCanvasRows))
		{
			if (RC_BAD( rc = insertRow( pNewRow, pRow)))
			{
				goto Exit;
			}
		}

	}
	else
	{
		pRow->bHasChildren = TRUE;

		if (pRow->bExpanded)
		{
			// Collapse the row, then expand it.
			if (RC_BAD( rc = collapseRow( &pRow)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = expandRow( pRow, TRUE, NULL)))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = m_pDb->documentDone( pRefNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}

Exit:

	// Restore the bHasChildren flag on error.
	if (RC_BAD( rc))
	{
		pRow->bHasChildren = bHasChildren;
	}

	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	if (pRefNode)
	{
		pRefNode->Release();
	}

	if (pTextNode)
	{
		pTextNode->Release();
	}

	if (pNewRow && pNewRow->pDomNode)
	{
		pNewRow->pDomNode->Release();
		pNewRow->pDomNode = NULL;
	}

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}


	return rc;
}

/****************************************************************************
Desc:	Create an ATTRIBUTE_NODE Dom node.
*****************************************************************************/
RCODE F_DomEditor::createAttributeNode(
	DME_ROW_INFO **		ppCurRow
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pRow = *ppCurRow;
	FLMUINT				uiTermChar;
	FLMUINT				uiTag;
	F_DOMNode *			pRefNode = NULL;
	F_DOMNode *			pAttributeNode = NULL;
	char					szAttrValue[ 128];
	FLMUNICODE *		puzValue = NULL;
	FLMUINT				uiValueLen;
	F_Dict *				pDict;
	FLMUINT64			ui64Value;
	F_AttrElmInfo		defInfo;
	FLMUINT				uiEncDefId = 0;

	// Verify that the  node we are looking at can take an element node. Some
	// dom node types can't take a sibling or a child element node.
	if (pRow->eType != ELEMENT_NODE)
	{
		displayMessage( "Invalid DOM node type",
			NE_XFLM_FAILURE, NULL, FLM_RED, FLM_WHITE);
		goto Exit;
	}

	// Select the name tag for the element node
	if (RC_BAD( rc = selectTag( ELM_ATTRIBUTE_TAG, &uiTag, &uiTermChar)))
	{
		goto Exit;
	}

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	f_memset( szAttrValue, 0, sizeof( szAttrValue));
	requestInput( "Attribute Value",
		szAttrValue, sizeof(szAttrValue), &uiTermChar);

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	// Select the encryption algorithm (if any)
	if (RC_BAD( rc = selectEncDef( ELM_ENCDEF_TAG, &uiEncDefId, &uiTermChar)))
	{
		goto Exit;
	}

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

	// Determine the data type for the selected attribute tag.
	if (RC_BAD( rc = m_pDb->getDictionary( &pDict)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDict->getAttribute( m_pDb, uiTag, &defInfo)))
	{
		goto Exit;
	}

	// Get the dom node for this row.
	if (RC_BAD( rc = getDomNode( pRow->ui64NodeId,
											pRow->eType == ATTRIBUTE_NODE
											? pRow->uiNameId
											: 0,
											&pRefNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pRefNode->createAttribute(
					m_pDb, uiTag, (IF_DOMNode **)&pAttributeNode)))
	{
		goto Exit;
	}

	switch (defInfo.getDataType())
	{
		case XFLM_TEXT_TYPE:
		{
			uiValueLen = (f_strlen( szAttrValue) * 2) + 2;

			if (RC_BAD( f_calloc( uiValueLen, &puzValue)))
			{
				goto Exit;
			}

			asciiToUnicode( &szAttrValue[0], puzValue);

			if (RC_BAD( rc = pAttributeNode->setUnicode( m_pDb,
																		puzValue,
																		uiValueLen,
																		TRUE,
																		uiEncDefId)))
			{
				goto Exit;
			}
			
			break;
		}
		case XFLM_NUMBER_TYPE:
		{
			if( RC_BAD( rc = getNumber( &szAttrValue[ 0], &ui64Value, NULL)))
			{
				displayMessage( "Invalid node number", rc,
					NULL, FLM_RED, FLM_WHITE);
				goto Exit;
			}

			if (RC_BAD( rc = pAttributeNode->setUINT64( m_pDb, ui64Value, uiEncDefId)))
			{
				goto Exit;
			}
			break;
		}
		case XFLM_BINARY_TYPE:
		default:
			rc = RC_SET( NE_XFLM_BAD_DATA_TYPE);
			goto Exit;
	}

	pRow->bHasAttributes = TRUE;
	
	if (RC_BAD( rc = m_pDb->documentDone( pRefNode)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}


Exit:

	f_free( &puzValue);

	if (pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}

	if (pRefNode)
	{
		pRefNode->Release();
	}

	if (pAttributeNode)
	{
		pAttributeNode->Release();
	}

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return rc;
}



/****************************************************************************
Desc:	Allows the user to interactively select a Dom Node type
*****************************************************************************/
RCODE F_DomEditor::selectNodeType(
	eDomNodeType *		peNodeType,
	FLMUINT *				puiTermChar
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FLMUINT				uiFlags;
	FLMUNICODE			uzItemName[ 128];
	F_DomEditor *		pNodeTypeList = NULL;

	flmAssert( m_bSetupCalled == TRUE);

	if( peNodeType)
	{
		*peNodeType = INVALID_NODE;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pNodeTypeList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pNodeTypeList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pNodeTypeList->setParent( this);
	pNodeTypeList->setReadOnly( TRUE);
	pNodeTypeList->setShutdown( m_pbShutdown);
	pNodeTypeList->setTitle( "Node Types - Select One");
	pNodeTypeList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
 	pNodeTypeList->setSource( m_pDb, XFLM_DATA_COLLECTION);

	if( m_pDb == NULL)
	{
		goto Exit;
	}
	
	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NODOM);

	asciiToUnicode( "Element Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)ELEMENT_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "Attribute Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)ATTRIBUTE_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "Text Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)DATA_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "CData Section Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)CDATA_SECTION_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "Processing Instruction Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)PROCESSING_INSTRUCTION_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "Comment Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)COMMENT_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;


	asciiToUnicode( "Document Node", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], (FLMUINT)DOCUMENT_NODE, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pNodeTypeList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	// Set the start row.
	
	pNodeTypeList->setCurrentAtTop();


	if( RC_BAD( rc = pNodeTypeList->interactiveEdit(
											m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pNodeTypeList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pNodeTypeList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pNodeTypeList->setControlFlags( pTmpRow, uiFlags);
			if( peNodeType)
			{
				*peNodeType = (eDomNodeType)pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pNodeTypeList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pNodeTypeList->getLastKey();
	}

Exit:

	if( pNodeTypeList)
	{
		pNodeTypeList->Release();
		pNodeTypeList = NULL;
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	return( rc);
}

/*=============================================================================
Desc:
=============================================================================*/
RCODE F_DomEditor::beginTransaction(
	eDbTransType			eTransType)
{
	RCODE			rc = NE_XFLM_OK;

	if (RC_BAD( rc = m_pDb->transBegin( eTransType)))
	{
		if (rc == NE_XFLM_TRANS_ACTIVE)
		{
			if (eTransType != m_pDb->getTransType())
			{
				if (RC_BAD( rc = m_pDb->transCommit()))
				{
					(void)m_pDb->transAbort();
				}

				if (RC_BAD( rc = m_pDb->transBegin( eTransType)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
	}

Exit:

	return rc;
}


/*=============================================================================
Desc:
=============================================================================*/
RCODE F_DomEditor::commitTransaction( void)
{
	RCODE			rc = NE_XFLM_OK;

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}
	
Exit:

	return rc;
}

/*=============================================================================
Desc:
=============================================================================*/
RCODE F_DomEditor::abortTransaction( void)
{
	RCODE			rc = NE_XFLM_OK;

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		if (RC_BAD( rc = m_pDb->transAbort()))
		{
			goto Exit;
		}
		// We need to clear the Abort status flag in the pDb.
		if (RC_BAD( rc = m_pDb->transBegin( XFLM_READ_TRANS)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	return rc;
}



/*=============================================================================
Desc:	Utility method to make sure the row and document list docId match.
=============================================================================*/
void F_DomEditor::checkDocument(
	DME_ROW_INFO **		ppDocRow,
	DME_ROW_INFO *			pRow
	)
{
	DME_ROW_INFO *		pDocList = *ppDocRow;

	// If there is no current row, then we don't care about the document list.
	if (!pRow)
	{
		return;
	}

	if (!pDocList)
	{
		return;
	}

	// If they already match, no need to test.
	if ( pRow->ui64DocId == pDocList->ui64DocId)
	{
		return;
	}

	// Start from the beginning.
	pDocList = m_pDocList;
	while (pDocList && pRow->ui64DocId != pDocList->ui64DocId)
	{
		pDocList = pDocList->pNext;
	}

	flmAssert( pDocList);
	flmAssert( pDocList->ui64DocId == pRow->ui64DocId);

	*ppDocRow = pDocList;

}


/*=============================================================================
Desc:	Method to ask the user to select which tag they want.
=============================================================================*/
RCODE F_DomEditor::selectTag(
	FLMUINT			uiTagType,
	FLMUINT *		puiTag,
	FLMUINT *		puiTermChar
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FLMUINT				uiFlags;
	FLMUNICODE			uzItemName[ 128];
	FLMUINT				uiId;
	FLMUINT				uiNextPos;
	F_DomEditor *		pTagList = NULL;
	FLMUINT				uiDataType;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiTag)
	{
		*puiTag = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pTagList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pTagList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pTagList->setParent( this);
	pTagList->setReadOnly( TRUE);
	pTagList->setShutdown( m_pbShutdown);
	pTagList->setTitle( "Tag Names - Select One");
	pTagList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pTagList->setSource( m_pDb, XFLM_DICT_COLLECTION);

	if( m_pDb == NULL)
	{
		goto Exit;
	}
	
	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NODOM);


	uiNextPos = 0;
	while( RC_OK( rc = m_pNameTable->getNextTagTypeAndNameOrder(
		uiTagType, &uiNextPos, uzItemName, NULL, sizeof( uzItemName), &uiId, &uiDataType)))
	{
		if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], uiId, TRUE)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags = uiFlags;

		if (RC_BAD( rc = pTagList->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;

	}
	if (rc != NE_XFLM_EOF_HIT)
	{
		goto Exit;
	}
	rc = NE_XFLM_OK;


	// Set the start row.
	pTagList->setCurrentAtTop();


	if( RC_BAD( rc = pTagList->interactiveEdit(
					m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pTagList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pTagList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pTagList->setControlFlags( pTmpRow, uiFlags);
			if( puiTag)
			{
				*puiTag = (FLMUINT)pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pTagList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pTagList->getLastKey();
	}

Exit:

	if( pTagList)
	{
		pTagList->Release();
		pTagList = NULL;
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	return( rc);
}

/****************************************************************************
Name:	selectEncDef
*****************************************************************************/
RCODE F_DomEditor::selectEncDef(
	FLMUINT			uiTagType,
	FLMUINT *		puiEncDefId,
	FLMUINT *		puiTermChar)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FLMUINT				uiFlags;
	FLMUNICODE			uzItemName[ 128];
	FLMUNICODE			uzNone[] = 
		{'<','N','o',' ','e','n','c','r','y','p','t','i','o','n','>','\0'};
	FLMUINT				uiId;
	FLMUINT				uiNextPos;
	F_DomEditor *		pEncDefList = NULL;
	FLMBOOL				bFoundEncDef = FALSE;

	flmAssert( m_bSetupCalled == TRUE);

	if( puiEncDefId)
	{
		*puiEncDefId = 0;
	}

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}


	// Initialize the name table.


	if( !m_pNameTable)
	{
		if( RC_BAD( rc = refreshNameTable()))
		{
			goto Exit;
		}
	}

	if( (pEncDefList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pEncDefList->Setup( m_pScreen)))
	{
		goto Exit;
	}

	pEncDefList->setParent( this);
	pEncDefList->setReadOnly( TRUE);
	pEncDefList->setShutdown( m_pbShutdown);
	pEncDefList->setTitle( "Encryption Algorithm Names - Select One");
	pEncDefList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pEncDefList->setSource( m_pDb, XFLM_DICT_COLLECTION);

	if( m_pDb == NULL)
	{
		goto Exit;
	}
	
	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NODOM);

	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzNone[0], 0, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pEncDefList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	uiNextPos = 0;
	while( RC_OK( rc = m_pNameTable->getNextTagTypeAndNameOrder(
		uiTagType, &uiNextPos, uzItemName, NULL, sizeof( uzItemName), &uiId)))
	{
		if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], uiId, TRUE)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags = uiFlags;

		if (RC_BAD( rc = pEncDefList->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;
		bFoundEncDef = TRUE;

	}
	if (rc != NE_XFLM_EOF_HIT)
	{
		goto Exit;
	}
	rc = NE_XFLM_OK;

	// Don't bother to ask if there are no encryption algorithms defined.

	if (!bFoundEncDef)
	{
		goto Exit;
	}

	// Set the start row.
	pEncDefList->setCurrentAtTop();


	if( RC_BAD( rc = pEncDefList->interactiveEdit(
					m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pEncDefList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pEncDefList->getControlFlags( pTmpRow, &uiFlags);
		if( uiFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pEncDefList->setControlFlags( pTmpRow, uiFlags);
			if( puiEncDefId)
			{
				*puiEncDefId = (FLMUINT)pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pEncDefList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pEncDefList->getLastKey();
	}

Exit:

	if( pEncDefList)
	{
		pEncDefList->Release();
		pEncDefList = NULL;
	}

	if (pTmpRow)
	{
		releaseRow( &pTmpRow);
	}
	return( rc);
}


/****************************************************************************
Name:	selectIndex
Desc:	Allows the user to interactively select an index
*****************************************************************************/
RCODE F_DomEditor::selectIndex(
	FLMUINT			uiFlags,
	FLMUINT *		puiIndex,
	FLMUINT *		puiTermChar
	)
{
	RCODE					rc = NE_XFLM_OK;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FLMUINT				uiDispFlags;
	F_DomEditor *		pIndexList = NULL;
	FLMUNICODE			uzItemName[ 80];
	FLMUINT				uiNextPos;
	FLMUINT				uiId;

	flmAssert( m_bSetupCalled == TRUE);

	*puiIndex = 0;

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if( (pIndexList = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
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
	pIndexList->setKeyHook( F_DomEditorSelectionKeyHook, 0);
	pIndexList->setSource( m_pDb, XFLM_DICT_COLLECTION);

	if( m_pDb == NULL)
	{
		goto Exit;
	}

	uiFlags = (F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_LIST_ITEM | F_DOMEDIT_FLAG_READ_ONLY | F_DOMEDIT_FLAG_NODOM);


	asciiToUnicode( "Dictionary Number Index", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], XFLM_DICT_NUMBER_INDEX, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pIndexList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	asciiToUnicode( "Dictionary Name Index", &uzItemName[0]);
	if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], XFLM_DICT_NAME_INDEX, TRUE)))
	{
		goto Exit;
	}

	pTmpRow->uiFlags = uiFlags;

	if (RC_BAD( rc = pIndexList->insertRow( pTmpRow, pPrevRow)))
	{
		goto Exit;
	}
	pPrevRow = pTmpRow;
	pTmpRow = NULL;

	uiNextPos = 0;
	while( RC_OK( rc = m_pNameTable->getNextTagTypeAndNameOrder(
											ELM_INDEX_TAG, &uiNextPos, uzItemName,
											NULL, sizeof( uzItemName), &uiId)))
	{
		if (RC_BAD( rc = makeNewRow( &pTmpRow, &uzItemName[0], uiId, TRUE)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags = uiFlags;

		if (RC_BAD( rc = pIndexList->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;

	}
	if (rc != NE_XFLM_EOF_HIT)
	{
		goto Exit;
	}
	rc = NE_XFLM_OK;


	// Set the start row.
	pIndexList->setCurrentAtTop();

	if( RC_BAD( rc = pIndexList->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( (pTmpRow = pIndexList->getScrFirstRow()) == NULL)
	{
		goto Exit;
	}

	while( pTmpRow)
	{
		pIndexList->getControlFlags( pTmpRow, &uiDispFlags);
		if( uiDispFlags & F_DOMEDIT_FLAG_SELECTED)
		{
			uiDispFlags &= ~F_DOMEDIT_FLAG_SELECTED;
			pIndexList->setControlFlags( pTmpRow, uiDispFlags);
			if (puiIndex)
			{
				*puiIndex = (FLMUINT)pTmpRow->ui64NodeId;
			}
			pTmpRow = NULL;
			break;
		}
		if (RC_BAD( rc = pIndexList->getNextRow( pTmpRow, &pTmpRow)))
		{
			goto Exit;
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = pIndexList->getLastKey();
	}

	if( pIndexList->getLastKey() == FKB_ESCAPE)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		goto Exit;
	}

Exit:

	if( pIndexList)
	{
		pIndexList->Release();
		pIndexList = NULL;
	}

	return( rc);
}


/****************************************************************************
Name:	indexList
Desc:	Allows listing of index keys (and references) by having the user
		interactively build from and until keys.
*****************************************************************************/
RCODE F_DomEditor::indexList( void)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiIndex;
	FLMUINT				uiTermChar;
	F_DomEditor	*		pKeyEditor = NULL;
	F_DataVector *		pFromKeyV = NULL;
	F_DataVector *		pUntilKeyV = NULL;
	F_DataVector *		pFoundKeyV = NULL;
	DME_ROW_INFO *		pTmpRow = NULL;
	DME_ROW_INFO *		pPrevRow = NULL;
	FTX_WINDOW *		pStatusWindow = NULL;
	FLMBYTE *			pucUntilKeyBuf = NULL;
	FLMBYTE *			pucFoundKeyBuf = NULL;
	FLMUINT				uiUntilKeyLen;
	FLMUINT				uiFoundKeyLen;
	FLMUINT				uiSrchFlag;
	FLMUINT				uiKeyCount = 0;
	FLMBOOL				bNewKey;
	FLMUINT				uiElementNumber;
	FLMINT				iCmp;
	FLMBOOL				bFirst;
	IXD *					pIxd;

	flmAssert( m_bSetupCalled == TRUE);

	if( RC_BAD( rc = selectIndex( 0, &uiIndex, &uiTermChar)))
	{
		goto Exit;
	}

	if( uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	// Initialize the key editor

	if( (pKeyEditor = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pKeyEditor->Setup( m_pScreen)))
	{
		goto Exit;
	}


	// Configure the editor

	pKeyEditor->setParent( this);
	pKeyEditor->setShutdown( m_pbShutdown);
	pKeyEditor->setTitle( "Index List [ALT-Z to accept]");
	pKeyEditor->setSource( m_pDb, uiIndex);
	pKeyEditor->setKeyHook( f_KeyEditorKeyHook, NULL);
	pKeyEditor->setDisplayHook( f_IndexRangeDispHook, NULL);

	// Begin a transaction
	if (RC_BAD( rc = beginTransaction( XFLM_READ_TRANS)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDb->getDict()->getIndex( uiIndex, NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	// Get the first key in the index

 	if( (pFromKeyV = f_new F_DataVector) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pFromKeyV->reset();

	if (RC_BAD( rc = m_pDb->keyRetrieve( uiIndex, NULL, XFLM_FIRST, pFromKeyV)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pKeyEditor->addComment( 
		&pPrevRow, "This is the first key in the index")))
	{
		goto Exit;
	}

	pPrevRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;

	uiElementNumber = 0;
	while( pFromKeyV->getNameId( uiElementNumber))
	{
		if (RC_BAD( rc = setupIndexRow(
			pFromKeyV, uiElementNumber, uiIndex, F_DOMEDIT_FLAG_KEY_FROM, &pTmpRow)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;

		uiElementNumber++;

		if (RC_BAD( rc = pKeyEditor->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;
	}
	pFromKeyV = NULL;  // Vector is now in the row list



	// Get the last key in the index

 	if( (pUntilKeyV = f_new F_DataVector) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pUntilKeyV->reset();

	if (RC_BAD( rc = m_pDb->keyRetrieve( uiIndex, NULL, XFLM_LAST, pUntilKeyV)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pKeyEditor->addComment( 
		&pPrevRow, "This is the last key in the index")))
	{
		goto Exit;
	}

	pPrevRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;

	uiElementNumber = 0;
	while ( pUntilKeyV->getNameId( uiElementNumber))
	{
		if (RC_BAD( rc = setupIndexRow(
			pUntilKeyV, uiElementNumber, uiIndex, F_DOMEDIT_FLAG_KEY_UNTIL, &pTmpRow)))
		{
			goto Exit;
		}

		pTmpRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;

		uiElementNumber++;

		if (RC_BAD( rc = pKeyEditor->insertRow( pTmpRow, pPrevRow)))
		{
			goto Exit;
		}
		pPrevRow = pTmpRow;
		pTmpRow = NULL;
	}
	pUntilKeyV = NULL;  // Vector is now in the row list.

	// Show the keys and allow them to be edited
	pKeyEditor->setCurrentAtTop();

ix_list_retry:

	// Allow the user to edit the from / until keys before we start to look for the
	// keys in the index.
	if( RC_BAD( rc = pKeyEditor->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY)))
	{
		goto Exit;
	}

	if( pKeyEditor->getLastKey() == FKB_ESCAPE)
	{
		goto Exit;
	}

	if( RC_BAD( rc = createStatusWindow(
		" Key Retrieval Status (Press ESC to Interrupt) ",
		FLM_GREEN, FLM_WHITE, NULL, NULL, &pStatusWindow)))
	{
		goto Exit;
	}
	FTXWinOpen( pStatusWindow);

	// Get the FROM Key.  It may have been changed, so we need to rebuild the vector and do
	// another keyRetrieve in order to get the search parameters setup.  The display function will
	// make sure that any new value will be stored in the puzValue buffer.


	pTmpRow = pKeyEditor->m_pScrFirstRow;

	while (pTmpRow)
	{
		if (pTmpRow->uiFlags & F_DOMEDIT_FLAG_KEY_FROM)
		{
			pFromKeyV = pTmpRow->pVector;
			pTmpRow->pVector = NULL;
			break;
		}
		pTmpRow = pTmpRow->pNext;
	}

	flmAssert( pTmpRow);


	// Get the UNTIL Key.  It may have been changed, so we need to rebuild the vector and do
	// another keyRetrieve in order to get the search parameters setup.

	pTmpRow = pKeyEditor->m_pScrFirstRow;

	while (pTmpRow)
	{
		if (pTmpRow->uiFlags & F_DOMEDIT_FLAG_KEY_UNTIL)
		{
			pUntilKeyV = pTmpRow->pVector;
			pTmpRow->pVector = NULL;
			break;
		}
		pTmpRow = pTmpRow->pNext;
	}

	flmAssert( pTmpRow);

	pKeyEditor->releaseAllRows();

	bNewKey = TRUE;

	// Allocate key buffers for the from & until key so we
	// can do comparisons.

	if (RC_BAD( rc = f_calloc( XFLM_MAX_KEY_SIZE * 2, &pucUntilKeyBuf)))
	{
		goto Exit;
	}
	pucFoundKeyBuf = &pucUntilKeyBuf [XFLM_MAX_KEY_SIZE];

	// Get the collated until key.

	if (RC_BAD( rc = pUntilKeyV->outputKey(
		m_pDb, uiIndex, 0, pucUntilKeyBuf, XFLM_MAX_KEY_SIZE, &uiUntilKeyLen)))
	{
		goto Exit;
	}
		
	// Read the keys
	uiSrchFlag = XFLM_INCL | XFLM_MATCH_IDS;

	pTmpRow = pPrevRow = NULL;
	bFirst = TRUE;

	while( !isExiting())
	{

		// Update the display

		FTXWinSetCursorPos( pStatusWindow, 0, 1);
		FTXWinPrintf( pStatusWindow, "Keys Retrieved : %u",
			(unsigned)uiKeyCount);
		FTXWinClearToEOL( pStatusWindow);
		FTXWinSetCursorPos( pStatusWindow, 0, 2);
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

		if ((pFoundKeyV = f_new F_DataVector) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		// Look for the next key.
		if (RC_BAD( rc = m_pDb->keyRetrieve(
							uiIndex, pFromKeyV, uiSrchFlag, pFoundKeyV)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				break;
			}
			goto Exit;
		}

		uiSrchFlag = XFLM_EXCL | XFLM_MATCH_IDS;

		// See if we have gone past the until key.

		if (RC_BAD( rc = pFoundKeyV->outputKey(
			m_pDb, uiIndex, 0, pucFoundKeyBuf, XFLM_MAX_KEY_SIZE, &uiFoundKeyLen)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = ixKeyCompare( m_pDb, pIxd, NULL, NULL, NULL,
									TRUE, TRUE, pucFoundKeyBuf, uiFoundKeyLen,
									pucUntilKeyBuf, uiUntilKeyLen, &iCmp)))
		{
			goto Exit;
		}
		if (iCmp > 0)
		{
			break;
		}
		uiKeyCount++;

		// Make a separator row to display the key number
		if (RC_BAD( rc = pKeyEditor->addComment(
			&pPrevRow, "Key [%,10u]   Document [%,10I64u]",
			uiKeyCount, pFoundKeyV->getDocumentID())))
		{
			goto Exit;
		}

		pPrevRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;
	
		// Display the key.

		uiElementNumber = 0;
		while (pFoundKeyV->getNameId( uiElementNumber))
		{
			if (RC_BAD( rc = setupIndexRow( pFoundKeyV, uiElementNumber, uiIndex, 0, &pTmpRow)))
			{
				goto Exit;
			}

			pTmpRow->uiFlags |= F_DOMEDIT_FLAG_LIST_ITEM;
	
			uiElementNumber++;

			if (RC_BAD( rc = pKeyEditor->insertRow( pTmpRow, pPrevRow)))
			{
				goto Exit;
			}
			pPrevRow = pTmpRow;
			pTmpRow = NULL;
		}
		if (bFirst)
		{
			pFromKeyV->Release();
			bFirst = FALSE;
		}
		pFromKeyV = pFoundKeyV;
		pFoundKeyV = NULL;

		f_yieldCPU();
#ifdef FLM_WIN
		f_sleep( 0);
#endif
	}

	pFromKeyV = NULL; // s/b in a row
	if ( pFoundKeyV)
	{
		pFoundKeyV->Release();
		pFoundKeyV = NULL;
	}

	FTXWinFree( &pStatusWindow);

	/*
	Display the results
	*/

	pKeyEditor->setCurrentAtTop();

	if( uiKeyCount)
	{
		pKeyEditor->setTitle( "Index List");
		pKeyEditor->setKeyHook( f_ViewOnlyKeyHook, 0);
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
		pKeyEditor->setCurrentAtTop();
		goto ix_list_retry;
	}

	if (RC_BAD( rc = commitTransaction()))
	{
		goto Exit;
	}

Exit:

	f_free( &pucUntilKeyBuf);

	if( pFromKeyV)
	{
		pFromKeyV->Release();
	}

	if( pFoundKeyV)
	{
		pFoundKeyV->Release();
	}

	if( pUntilKeyV)
	{
		pUntilKeyV->Release();
	}

	if( pStatusWindow)
	{
		FTXWinFree( &pStatusWindow);
	}

	if( pKeyEditor)
	{
		pKeyEditor->releaseAllRows();
		pKeyEditor->Release();
	}

	if( RC_BAD( rc))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			displayMessage( "The index is empty", rc,
				NULL, FLM_RED, FLM_WHITE);
			rc = NE_XFLM_OK;
		}
	}

	if (m_pDb->getTransType() != XFLM_NO_TRANS)
	{
		(void)abortTransaction();
	}

	return( rc);
}

typedef struct QueryDataTag
{
	FLMUINT		uiCollection;
	FLMUINT		uiNodeArraySize;
	FLMUINT64 *	pui64Nodes;
	FLMUINT *	puiAttrNameIds;
	FLMUINT		uiNodeCount;
	FLMUINT		uiCurrNode;
	FLMBOOL		bShowingData;
	char *		pszQuery;
} QUERY_DATA;

/****************************************************************************
Desc:	
*****************************************************************************/
FSTATIC RCODE f_QueryEditorKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *,	// pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData
	)
{
	RCODE				rc = NE_XFLM_OK;
	char				szTitle [100];
	QUERY_DATA *	pQueryData = (QUERY_DATA *)pvKeyData;

	switch (uiKeyIn)
	{
		case 'P':
		case 'p':
			if (pQueryData->uiCurrNode)
			{
				pQueryData->uiCurrNode--;
			}

Retrieve_Node:
			*puiKeyOut = 0;

			// Clear the current buffer.

			pDomEditor->setScrFirstRow( NULL);
			pDomEditor->setCurrentRow(NULL, 0);
			if (RC_BAD( rc = pDomEditor->retrieveNodeFromDb( pQueryData->uiCollection,
									pQueryData->pui64Nodes [pQueryData->uiCurrNode],
									pQueryData->puiAttrNameIds [pQueryData->uiCurrNode])))
			{
				goto Exit;
			}
			f_sprintf( szTitle, "#%u of %u (N,SPACE=Next, P=Prev)",
							(unsigned)(pQueryData->uiCurrNode + 1),
							(unsigned)pQueryData->uiNodeCount);
			pDomEditor->setTitle( szTitle);

			// Show the query

			if (!pQueryData->bShowingData)
			{
				FTX_WINDOW *	pStatusWin = pDomEditor->getStatusWindow();

				FTXWinSetCursorPos( pStatusWin, 0, 1);
				FTXWinPrintf( pStatusWin, "%s", pQueryData->pszQuery);
				pQueryData->bShowingData = TRUE;
			}
			break;

		case 'n':
		case 'N':
		case ' ':
			if (pQueryData->uiCurrNode < pQueryData->uiNodeCount - 1)
			{
				pQueryData->uiCurrNode++;
			}
			goto Retrieve_Node;

		case 'F':
		case 'f':
		case FKB_ALT_F:

			// Don't allow to start another query.
			*puiKeyOut = 0;
			break;

		default:

			*puiKeyOut = uiKeyIn;
			if (!pQueryData->bShowingData)
			{
				goto Retrieve_Node;
			}

			break;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allows doing a query and having the user expand the results of
		each returned node.
*****************************************************************************/
void F_DomEditor::doQuery(
	char *				pszQuery,
	FLMUINT				uiQueryBufSize)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCollection;
	FLMUINT				uiTermChar;
	IF_Query *			pQuery = NULL;
	F_XPath				xpath;
	IF_DOMNode *		pNode = NULL;
	FLMBOOL				bStartedTrans = FALSE;
	XFLM_OPT_INFO *	pOptInfo;
	FLMUINT				uiOptInfoCnt;
	EditQueryStatus	queryStatus;
	F_DomEditor	*		pQueryEditor = NULL;
	QUERY_DATA			QueryData;
	FLMUINT				uiQueryStrLen;
	IF_DbSystem *		pDbSystem = NULL;

	f_memset( &QueryData, 0, sizeof( QueryData));

	requestInput(
		"XPATH Query", pszQuery, uiQueryBufSize, &uiTermChar);

	if (uiTermChar == FKB_ESCAPE)
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	// Create a window for displaying query progress.

	queryStatus.createQueryStatusWindow( m_pScreen,
					FLM_GREEN, FLM_WHITE, pszQuery);

	if (RC_BAD( rc = pDbSystem->createIFQuery( &pQuery)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = beginTransaction( XFLM_READ_TRANS)))
	{
		goto Exit;
	}

	bStartedTrans = TRUE;

	if (RC_BAD( rc = selectCollection( &uiCollection, &uiTermChar)))
	{
		goto Exit;
	}

	if (uiTermChar != FKB_ENTER)
	{
		goto Exit;
	}

	m_uiCollection = uiCollection;

	if (RC_BAD( rc = pQuery->setCollection( uiCollection)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = xpath.parseQuery( m_pDb, pszQuery, pQuery)))
	{
		goto Exit;
	}

	pQuery->setQueryStatusObject( &queryStatus);
	for (;;)
	{
		if (RC_BAD( rc = pQuery->getNext( m_pDb, &pNode, 0, 0)))
		{
			if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_USER_ABORT)
			{
				if (!queryStatus.keepResults())
				{
					QueryData.uiNodeCount = 0;
				}
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}

		// Save the node ID - don't collect any more than 30000

		if (QueryData.uiNodeCount < 30000)
		{
			if (QueryData.uiNodeCount == QueryData.uiNodeArraySize)
			{
				FLMUINT	uiNewSize = QueryData.uiNodeArraySize + 500;

				if (RC_BAD( rc = f_realloc( sizeof( FLMUINT64) * uiNewSize,
											&QueryData.pui64Nodes)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = f_realloc( sizeof( FLMUINT) * uiNewSize,
											&QueryData.puiAttrNameIds)))
				{
					goto Exit;
				}
				QueryData.uiNodeArraySize = uiNewSize;
			}
			
			if (pNode->getNodeType() == ATTRIBUTE_NODE)
			{
				if( RC_BAD( rc = pNode->getParentId( m_pDb, 
					&QueryData.pui64Nodes [QueryData.uiNodeCount])))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pNode->getNameId( m_pDb,
									&QueryData.puiAttrNameIds [QueryData.uiNodeCount])))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = pNode->getNodeId( m_pDb, 
					&QueryData.pui64Nodes [QueryData.uiNodeCount])))
				{
					goto Exit;
				}
				QueryData.puiAttrNameIds [QueryData.uiNodeCount] = 0;
			}
			
			QueryData.uiNodeCount++;
		}
	}

	if (RC_BAD( rc = pQuery->getStatsAndOptInfo( &uiOptInfoCnt,
											&pOptInfo)))
	{
		goto Exit;
	}

	if (uiOptInfoCnt)
	{
		FLMUINT	uiOptToShow = 0;
		FLMUINT	uiChar;

		for (;;)
		{
			queryStatus.refreshStatus( TRUE, uiOptToShow + 1,
						uiOptInfoCnt, &pOptInfo [uiOptToShow]);
			queryStatus.testEscape( uiOptInfoCnt, &uiChar);
			if (uiChar == 'N' || uiChar == 'n')
			{
				if (uiOptToShow < uiOptInfoCnt - 1)
				{
					uiOptToShow++;
				}
			}
			else if (uiChar == 'P' || uiChar == 'P')
			{
				if (uiOptToShow)
				{
					uiOptToShow--;
				}
			}
			else if (uiChar == ' ')
			{
				if (uiOptToShow < uiOptInfoCnt - 1)
				{
					uiOptToShow++;
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
		
		pDbSystem->freeMem( (void **)&pOptInfo);
	}
	
	if (!QueryData.uiNodeCount)
	{
		goto Exit;
	}

	// Initialize the query editor

	if ((pQueryEditor = f_new F_DomEditor) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pQueryEditor->Setup( m_pScreen)))
	{
		goto Exit;
	}

	// Configure the editor

	QueryData.uiCollection = uiCollection;
	pQueryEditor->setParent( this);
	pQueryEditor->setShutdown( m_pbShutdown);
	pQueryEditor->setSource( m_pDb, uiCollection);

	uiQueryStrLen = f_strlen( pszQuery);
	if (RC_BAD( rc = f_alloc( uiQueryStrLen + 1, &QueryData.pszQuery)))
	{
		goto Exit;
	}
	f_memcpy( QueryData.pszQuery, pszQuery, uiQueryStrLen + 1);

	if (uiQueryStrLen > 75)
	{
		f_strcpy( &QueryData.pszQuery [72], "...");
	}
	pQueryEditor->setKeyHook( f_QueryEditorKeyHook, &QueryData);
	if (RC_BAD( rc = pQueryEditor->interactiveEdit( m_uiULX, m_uiULY, m_uiLRX, m_uiLRY,
								TRUE, 2, (FLMUINT)'P')))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		displayMessage( "Error", RC_SET( rc),
				NULL, FLM_RED, FLM_WHITE);
	}

	if (pQuery)
	{
		pQuery->Release();
	}

	if (pNode)
	{
		pNode->Release();
	}

	if (bStartedTrans)
	{
		commitTransaction();
	}

	if (pQueryEditor)
	{
		pQueryEditor->releaseAllRows();
		pQueryEditor->Release();
	}

	if (QueryData.pui64Nodes)
	{
		f_free( &QueryData.pui64Nodes);
	}
	
	if (QueryData.puiAttrNameIds)
	{
		f_free( &QueryData.puiAttrNameIds);
	}
	
	if (QueryData.pszQuery)
	{
		f_free( &QueryData.pszQuery);
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}
}

/****************************************************************************
Desc: Adds a comment line subordinate to the current node
*****************************************************************************/
RCODE F_DomEditor::addComment(
	DME_ROW_INFO **		ppCurRow,
	const char *			pszFormat, ...
	)
{
	RCODE					rc = NE_XFLM_OK;
	char					szBuffer[ 512];
	FLMUNICODE *		puzValue = NULL;
	DME_ROW_INFO *		pCommentRow = NULL;
	f_va_list			args;
	
	flmAssert( m_bSetupCalled == TRUE);

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	if (RC_BAD( rc = f_calloc( (f_strlen(szBuffer) * 2) + 2, &puzValue)))
	{
		goto Exit;
	}

	asciiToUnicode( szBuffer, puzValue);
	
	if (RC_BAD( rc = makeNewRow( &pCommentRow, puzValue, 0, TRUE)))
	{
		goto Exit;
	}

	pCommentRow->uiFlags = F_DOMEDIT_FLAG_COMMENT | F_DOMEDIT_FLAG_READ_ONLY |
		F_DOMEDIT_FLAG_HIDE_LEVEL | F_DOMEDIT_FLAG_NO_DELETE | F_DOMEDIT_FLAG_HIDE_EXPAND |
		F_DOMEDIT_FLAG_NOPARENT | F_DOMEDIT_FLAG_NOCHILD | F_DOMEDIT_FLAG_NODOM;

	if (RC_BAD( rc = insertRow( pCommentRow, *ppCurRow)))
	{
		goto Exit;
	}

	// Return the new row.
	*ppCurRow = pCommentRow;

Exit:

	f_free( &puzValue);

	return( rc);
}

/*=============================================================================
Desc:
=============================================================================*/
FSTATIC RCODE setupIndexRow(
	F_DataVector *		pKey,
	FLMUINT				uiElementNumber,
	FLMUINT				uiIndex,
	FLMUINT				uiFlag,
	DME_ROW_INFO **	ppTmpRow
	)
{
	RCODE				rc = NE_XFLM_OK;
	DME_ROW_INFO *	pTmpRow = NULL;

	// Create the new row.  Make sure we indicate that the display value comes
	// from the local buffer.
	if (RC_BAD( rc = makeNewRow(
				&pTmpRow, NULL, 0, FALSE)))
	{
		goto Exit;
	}

	pTmpRow->pVector = pKey;

	// uiIndex represents the current element number in the vector when there is
	// a vector present.
	pTmpRow->uiIndex = uiIndex;
	pTmpRow->uiElementNumber = uiElementNumber;
	pTmpRow->pDomNode = NULL;

	pTmpRow->uiFlags = uiFlag | F_DOMEDIT_FLAG_HIDE_EXPAND | F_DOMEDIT_FLAG_NODOM;
	
	*ppTmpRow = pTmpRow;

Exit:

	return rc;
}

/*=============================================================================
Desc:	Function to display the Index range selection page
=============================================================================*/
FSTATIC RCODE f_IndexRangeDispHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiFlags = 0;

	if( !pRow)
	{
		goto Exit;
	}

	pDomEditor->getControlFlags( pRow, &uiFlags);

	// Are we just displaying a value, or should we use the Vector?
	if (pRow->bUseValue && uiFlags & F_DOMEDIT_FLAG_COMMENT)
	{
		rc = formatRow( pDomEditor, pRow, puiNumVals, uiFlags);
	}
	else
	{
		flmAssert( pRow->pVector);
		flmAssert( !pRow->pDomNode);
		if (RC_BAD( rc = formatIndexKeyNode( pDomEditor, pRow, puiNumVals)))
		{
			goto Exit;
		}
	}


Exit:
	// Don't hold on to the Dom node when finished.
	if ( pRow->pDomNode)
	{
		pRow->pDomNode->Release();
		pRow->pDomNode = NULL;
	}
	return( rc);
}


/*=============================================================================
Desc:	Function to format the presentation of an index key row
=============================================================================*/
FSTATIC RCODE formatIndexKeyNode(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCol = 4;
	DME_DISP_COLUMN *	pDispVals;
	FLMUNICODE *		puzBuffer = NULL;
	FLMUINT				uiBufSize;
	F_DataVector *		pVector = pRow->pVector;
	FLMUINT				uiElementNumber = pRow->uiElementNumber;
	FLMUINT				uiLen;
	char *				pszData = NULL;
	FLMUINT64			ui64NodeId;

	flmAssert( pVector);

	if ((pDispVals = pDomEditor->getDispColumns()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	f_memset( pDispVals, 0, sizeof(DME_DISP_COLUMN));


	// Prepare the key component number
	if (pVector->isDataComponent( uiElementNumber))
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, "D)");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_CYAN;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}
	else
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, "K)");
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = FLM_CYAN;
		pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
		uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
		(*puiNumVals)++;
	}

	// Get the  name for this element
	if (RC_BAD( rc = pDomEditor->getNodeName( pRow->pDomNode, pRow, &puzBuffer, &uiBufSize)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = unicodeToAscii(
		puzBuffer, pDispVals[ *puiNumVals].szString, uiBufSize)))
	{
		goto Exit;
	}
	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_WHITE;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
	(*puiNumVals)++;

	// Get the current value in the buffer or display the one in the buffer.
	switch( pVector->getDataType( uiElementNumber))
	{
		case XFLM_NODATA_TYPE:
		{
			f_sprintf( pDispVals[ *puiNumVals].szString, "no data");
			break;
		}
		case XFLM_TEXT_TYPE:
		{

			if ((uiLen = pVector->getDataLength( uiElementNumber)) > 0)
			{
				if (RC_BAD( rc = f_calloc( uiLen, &pszData)))
				{
					goto Exit;
				}
	
				if (RC_BAD( rc = pVector->getUTF8( uiElementNumber, 
					(FLMBYTE *)pszData, &uiLen)))
				{
					goto Exit;
				}
				f_sprintf( pDispVals[ *puiNumVals].szString, "%*s", uiLen, pszData);
			}
			else
			{
				f_sprintf( pDispVals[ *puiNumVals].szString, "<empty str>");
			}
			break;
		}
		case XFLM_NUMBER_TYPE:
		{
			FLMUINT64		ui64Num;
			if (RC_BAD( rc = pVector->getUINT64( uiElementNumber, &ui64Num)))
			{
				if (rc == NE_XFLM_NOT_FOUND)
				{
					f_sprintf( pDispVals[ *puiNumVals].szString, "<empty num>");
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				f_sprintf( pDispVals[ *puiNumVals].szString, "0x%I64x", ui64Num);
			}
			break;
		}
		case XFLM_BINARY_TYPE:
		{
			if ((uiLen = pVector->getDataLength( uiElementNumber)) > 0)
			{
				f_sprintf( pDispVals[ *puiNumVals].szString, "binary data type, len=%u",
						(unsigned)uiLen);
			}
			else
			{
				f_sprintf( pDispVals[ *puiNumVals].szString, "<empty binary>");
			}
			break;
		}
		case XFLM_UNKNOWN_TYPE:
		{
			f_sprintf( pDispVals[ *puiNumVals].szString, "unknown data type");
			break;
		}
		default:
		{
			f_sprintf( pDispVals[ *puiNumVals].szString, "invalid data type");
			break;
		}
	}
	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_YELLOW;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
	(*puiNumVals)++;

	// Display the node Id.
	if ((ui64NodeId = pVector->getID( uiElementNumber)) != 0)
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, "%I64u", ui64NodeId);
	}
	else
	{
		f_sprintf( pDispVals[ *puiNumVals].szString, "NULL");
	}
	pDispVals[ *puiNumVals].uiCol = uiCol;
	pDispVals[ *puiNumVals].uiForeground = FLM_LIGHTRED;
	pDispVals[ *puiNumVals].uiBackground = pDomEditor->isMonochrome() ? FLM_BLACK : FLM_BLUE;
	uiCol += f_strlen( pDispVals[ *puiNumVals].szString) + 2;
	(*puiNumVals)++;

	// Check to make sure that the node value does not push the nodeId off the screen.
	if (uiCol > pDomEditor->getCanvasCols())
	{
		FLMUINT	uiValueIdx = *puiNumVals - 2;
		FLMUINT	uiStrOfs = f_strlen(pDispVals[ uiValueIdx].szString) - 
			(uiCol - pDomEditor->getCanvasCols());

		f_sprintf(( char *)&pDispVals[ uiValueIdx].szString[ uiStrOfs], "... ");


	}

Exit:

	f_free( &puzBuffer);
	f_free( &pszData);

	return rc;
}

/*=============================================================================
Desc:	Release all of the rows in the editor.
=============================================================================*/
void F_DomEditor::releaseAllRows()
{

	if (m_pScrFirstRow)
	{

		// Make sure we release from the very beginning of the list of rows.
		// If we do backup, we must increment the row count for each row
		while (m_pScrFirstRow->pPrev != NULL)
		{
			m_pScrFirstRow = m_pScrFirstRow->pPrev;
		}

		while (m_pScrFirstRow)
		{
			releaseRow( &m_pScrFirstRow);
		}
		m_uiNumRows = 0;
		m_pScrLastRow = m_pCurRow = m_pScrFirstRow;
	}
}

/*=============================================================================
Desc:	Deletes an entry from an index.
=============================================================================*/
RCODE F_DomEditor::viewOnlyDeleteIxKey( void	)
{
	RCODE					rc = NE_XFLM_OK;
	F_DataVector *		pVector = NULL;
	DME_ROW_INFO *		pFirstDeleteRow = NULL;
	FLMBYTE				pucKeyBuf[ XFLM_MAX_KEY_SIZE];
	FLMUINT				uiKeyLen;
	F_Btree *			pBtree = NULL;
	IXD *					pIxd;
	LFILE *				pLFile = NULL;
	F_Dict *				pDict;
	FLMBOOL				bHaveCounts;
	FLMBOOL				bHaveData;
	FLMBOOL				bUpdateTranStarted = FALSE;
	IXKeyCompare		compareObject;
	
	if (m_pCurRow->pVector)
	{
		// Return NE_XFLM_OK;
		goto Exit;
	}

	if (!m_pCurRow->pNext)
	{
		// return NE_XFLM_OK;
		goto Exit;
	}

	if (m_pCurRow->pVector)
	{
		// return NE_XFLM_OK;
		goto Exit;
	}

	if ((m_pCurRow->uiFlags & F_DOMEDIT_FLAG_COMMENT) == 0)
	{
		// return NE_XFLM_OK;
		goto Exit;
	}


	// Mark the first row to delete.

	pFirstDeleteRow = m_pCurRow;

	// Get the vector

	pVector = m_pCurRow->pNext->pVector;

	if (!pVector)
	{
		flmAssert( 0);
		goto Exit;
	}

	// Start an update transaction.
	if (RC_BAD( rc = beginTransaction( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bUpdateTranStarted = TRUE;

	// Get the index, then open a btree to the index.
	if (RC_BAD( rc = m_pDb->getDictionary( &pDict)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pVector->outputKey( m_pDb,
													 m_pCurRow->pNext->uiIndex,
													 0,
													 pucKeyBuf,
													 XFLM_MAX_KEY_SIZE,
													 &uiKeyLen)))
	{
		goto Exit;
	}


	if (RC_BAD( rc = pDict->getIndex( m_pCurRow->pNext->uiIndex,
												 &pLFile,
												 &pIxd)))
	{
		goto Exit;
	}

	compareObject.setIxInfo( m_pDb, pIxd);
	compareObject.setCompareNodeIds( TRUE);
	compareObject.setCompareDocId( TRUE);
	compareObject.setSearchKey( NULL);

	if ((pBtree = f_new F_Btree) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	bHaveCounts = (pIxd->uiFlags & IXD_ABS_POS) ? TRUE : FALSE;
	bHaveData = (pIxd->pFirstData) ? TRUE : FALSE;

	if (RC_BAD( rc = pBtree->btOpen( m_pDb,
												pLFile,
												bHaveCounts,
												bHaveData, &compareObject)))
	{
		goto Exit;
	}


	// Delete the key.
	if (RC_BAD( rc = pBtree->btRemoveEntry( pucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}

	// Now remove the rows from the display list.  Release the comment row.
	releaseRow( &m_pCurRow);
	m_uiNumRows--;

	while (m_pCurRow &&
			 (m_pCurRow->uiFlags & F_DOMEDIT_FLAG_COMMENT) == 0)
	{
		releaseRow( &m_pCurRow);
		m_uiNumRows--;
	}


Exit:

	if (pBtree)
	{
		pBtree->btClose();
		pBtree->Release();
	}

	if (bUpdateTranStarted)
	{
		if (RC_OK( rc))
		{
			if (RC_BAD( rc = commitTransaction()))
			{
				(void) abortTransaction();
			}
		}
		else
		{
			(void) abortTransaction();
		}
	}

	return rc;
}

