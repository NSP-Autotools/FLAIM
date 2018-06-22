//------------------------------------------------------------------------------
// Desc:	DOM editor
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

#ifndef DOMEDIT_HPP
#define DOMEDIT_HPP

class F_DomEditor;
typedef F_DomEditor *	F_DomEditor_p;

/*
Constants
*/

#define F_DOMEDIT_BUF_SIZE				0x0000FFFF
#define F_DOMEDIT_MAX_TITLE_SIZE		64

/*
System flags
*/

#define F_DOMEDIT_FLAG_NOPARENT			0x00000001
#define F_DOMEDIT_FLAG_NOCHILD			0x00000002
#define F_DOMEDIT_FLAG_LIST_ITEM			0x00000004
#define F_DOMEDIT_FLAG_HIDE_TAG			0x00000008
#define F_DOMEDIT_FLAG_HIDE_LEVEL		0x00000010
#define F_DOMEDIT_FLAG_HIDE_SOURCE		0x00000020
#define F_DOMEDIT_FLAG_READ_ONLY			0x00000040
#define F_DOMEDIT_FLAG_SELECTED			0x00000080
#define F_DOMEDIT_FLAG_ENDTAG				0x00000100
#define F_DOMEDIT_FLAG_NO_DELETE			0x00000200
#define F_DOMEDIT_FLAG_COLLAPSED			0x00000400
#define F_DOMEDIT_FLAG_HIDE_EXPAND		0x00000800
#define F_DOMEDIT_FLAG_COMMENT			0x00001000
#define F_DOMEDIT_FLAG_KEY_FROM			0x00002000
#define F_DOMEDIT_FLAG_KEY_UNTIL			0x00004000
#define F_DOMEDIT_FLAG_NODOM				0x00008000	// Don't fetch a dom object.
#define F_DOMEDIT_FLAG_ELEMENT_DATA		0x00010000	// Display the data embedded in the element node (acting as the child)

/*
Index selection flags
*/

#define F_DOMEDIT_ISEL_NOIX				0x0001	// Show "no index" as an option

/*
Configuration options
*/

#define F_DOMEDIT_CONFIG_STATS_START	0x0001
#define F_DOMEDIT_CONFIG_STATS_STOP		0x0002
#define F_DOMEDIT_CONFIG_STATS_RESET	0x0003


/*
Types, enums, etc.
*/

typedef enum 
{
	F_DOMEDIT_EVENT_RECREAD,
	F_DOMEDIT_EVENT_RECINSERT,
	F_DOMEDIT_EVENT_GETDISPVAL,
	F_DOMEDIT_EVENT_GETNEXTNODE,
	F_DOMEDIT_EVENT_GETPREVNODE,
	F_DOMEDIT_EVENT_IEDIT,				// Interactive editor invoked
	F_DOMEDIT_EVENT_REFRESH,			// Called prior to refresh
	F_DOMEDIT_EVENT_NAME_TABLE
} eDomEventType;

#define MAX_DISPLAY_SEGMENT		100
typedef struct
{
	char			szString[ MAX_DISPLAY_SEGMENT];
	FLMUINT		uiCol;
	eColorType	uiForeground;
	eColorType	uiBackground;
} DME_DISP_COLUMN;

// Structure to hold an entire row
typedef struct rowInfo
{
	FLMUINT					uiLevel;
	FLMUINT64				ui64NodeId;
	FLMUINT64				ui64DocId;
	FLMUINT					uiNameId;
	FLMBOOL					bExpanded;
	FLMBOOL					bHasChildren;
	FLMBOOL					bHasAttributes;
	FLMBOOL					bHasElementData;	// When this flag is set, the node is acting as the parent.
	F_DOMNode *				pDomNode;
	F_DataVector *			pVector;
	eDomNodeType			eType;
	FLMUNICODE *			puzValue;
	FLMUINT					uiLength;
	FLMUINT					uiIndex;
	FLMUINT					uiElementNumber;
	FLMBOOL					bUseValue;
	FLMUINT					uiFlags;
	struct rowInfo *		pNext;
	struct rowInfo *		pPrev;
} DME_ROW_INFO;

typedef struct
{
	FLMBOOL			bSingleLevel;
	FLMUINT			uiAnchorLevel;
	FLMUINT64		ui64NodeId;
} DME_ROW_ANCHOR;

typedef struct
{
	F_NameTable	*	pNameTable;
	FLMBOOL			bInitialized;
} DME_NAME_TABLE_INFO;

/*
Callbacks
*/

typedef RCODE (* F_DOMEDIT_DISP_HOOK)(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pRow,
	FLMUINT *			puiNumVals);

typedef RCODE (* F_DOMEDIT_KEY_HOOK)(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

typedef RCODE (* F_DOMEDIT_HELP_HOOK)(
	F_DomEditor *		pDomEditor,
	F_DomEditor *		pHelpEditor);

typedef RCODE (* F_DOMEDIT_EVENT_HOOK)(
	F_DomEditor *		pDomEditor,
	eDomEventType		eEventType,
	void *				EventData,
	void *				UserData);

class	F_DomEditor : public F_Object
{
public:

	F_DomEditor( void);
	~F_DomEditor( void);

	void reset( void);

	RCODE Setup(
		FTX_SCREEN *	pScreen);

	void setParent(
		F_DomEditor *	pParent);

	F_DomEditor * getParentEditor( void);

	RCODE setSource(
		F_Db *		pDb,
		FLMUINT		uiCollection);

	F_Db * getSource( void )
	{
		return m_pDb;
	}

	RCODE insertRow(
		DME_ROW_INFO *		pRow,
		DME_ROW_INFO *		pStartRow = NULL);

	RCODE setTitle(
		const char *	pszTitle);

	void setCurrentAtTop( void);

	void setCurrentAtBottom( void);

	void setReadOnly(
		FLMBOOL		bReadOnly);

	RCODE interactiveEdit(
		FLMUINT			uiULX = 0,
		FLMUINT			uiULY = 0,
		FLMUINT			uiLRX = 0,
		FLMUINT			uiLRY = 0,
		FLMBOOL			bBorder = TRUE,
		FLMUINT			uiStatusLines = 1,
		FLMUINT			uiStartChar = 0);

	FLMBOOL isMonochrome( void);

	RCODE getPrevRow(
		DME_ROW_INFO *		pCurRow,
		DME_ROW_INFO **	ppPrevRow,
		FLMBOOL				bFetchPrevRow = FALSE);

	RCODE getNextRow(
		DME_ROW_INFO *		pCurRow,
		DME_ROW_INFO **	ppNextRow,
		FLMBOOL				bFetchNextRow = FALSE,
		FLMBOOL				bIgnoreAnchor = FALSE);

	RCODE getNextDocument(
		DME_ROW_INFO *		pCurRow,
		DME_ROW_INFO **	ppNextRow,
		FLMBOOL				bFetchNextRow = FALSE,
		FLMBOOL				bIgnoreAnchor = FALSE);

	void setScrFirstRow(
		DME_ROW_INFO * pScrFirstRow);

	DME_ROW_INFO * getScrFirstRow( void);

	DME_ROW_INFO * getScrLastRow( void);

	FLMUINT getCursorRow( void);

	FLMUINT getNumRows( void);

	FLMUINT getDispRows( void)
	{
		return m_uiNumRows;
	}

	void setNumDispRows(
		FLMUINT			uiNumDispRows)
	{
		m_uiNumRows = uiNumDispRows;
	}

	RCODE getDisplayValue(
		DME_ROW_INFO *		pRow,
		char *				pszBuf,
		FLMUINT				uiBufSize);

	DME_ROW_INFO * findRecord(
		FLMUINT				uiCollection,
		FLMUINT				uiDrn,
		DME_ROW_INFO *		pStartRow = NULL);

	void setShutdown(
		FLMBOOL *	pbShutdown);

	FLMBOOL * getShutdown( void);

	RCODE setCurrentRow(
		DME_ROW_INFO *		pCurRow,
		FLMUINT				uiCurRow);

	DME_ROW_INFO * getCurrentRow(
		FLMUINT *		puiCurRow);

	RCODE setFirstRow(
		DME_ROW_INFO *		pRow);

	RCODE setControlFlags(
		DME_ROW_INFO *		pCurRow,
		FLMUINT				uiFlags);

	RCODE getControlFlags(
		DME_ROW_INFO *		pCurRow,
		FLMUINT *			puiFlags);

	void setDisplayHook(
		F_DOMEDIT_DISP_HOOK	pDispHook,
		void *					DispData);

	void setKeyHook(
		F_DOMEDIT_KEY_HOOK	pKeyHook,
		void *					KeyData);

	void setHelpHook(
		F_DOMEDIT_HELP_HOOK	pHelpHook,
		void *					HelpData);

	void setEventHook(
		F_DOMEDIT_EVENT_HOOK	pEventHook,
		void *					EventData);

	F_Db * getDb( void);

	FLMUINT getCollection( void);
	
	FLMUINT getLastKey( void);

	RCODE getNumber(
		char *		pszBuf,
		FLMUINT64 *	pui64Value,
		FLMINT64 *	pi64Value);

	RCODE getCollectionNumber(
		char *		pszCollectionName,
		FLMUINT *	puiCollectionNum);

	RCODE getIndexNumber(
		char *		pszIndexName,
		FLMUINT *	puiIndexNum);

	RCODE retrieveNodeFromDb(
		FLMUINT		uiCollection,
		FLMUINT64	ui64NodeId,
		FLMUINT		uiAttrNameId);

	RCODE requestInput(
		const char *	pszMessage,
		char *			pszResponse,
		FLMUINT			uiMaxRespLen,
		FLMUINT *		puiTermChar);

	RCODE displayMessage(
		const char *	pszMessage,
		RCODE				rcOfMessage,
		FLMUINT *		puiTermChar,
		eColorType		uiBackground,
		eColorType		uiForeground);

	RCODE globalConfig(
		FLMUINT		uiOption);

	RCODE createStatusWindow(
		const char *	pszTitle,
		eColorType		uiBack,
		eColorType		uiFore,
		FLMUINT *		puiCols,
		FLMUINT *		puiRows,
		FTX_WINDOW **	ppWindow);


	FLMUINT getULX( void);
	
	FLMUINT getULY( void);

	RCODE openNewDb();

	RCODE refreshNameTable( void);

	RCODE retrieveDocumentList(
		FLMUINT			uiCollection,
		FLMUINT64 *		pui64NodeId,
		FLMUINT *		puiTermChar);

	RCODE getDomNode(
		FLMUINT64		ui64NodeId,
		FLMUINT			uiAttrNameId,
		F_DOMNode **	ppDomNode);

	void setDocList(
		FLMBOOL			bDocList
		)
	{
		m_bDocList = bDocList;
	}

	DME_DISP_COLUMN * getDispColumns( void)
	{
		return &m_dispColumns[0];
	}

	FINLINE FLMUINT getEditCanvasCols( void);

	RCODE getNodeName(
		F_DOMNode *		pDOMNode,
		DME_ROW_INFO *	pRow,
		FLMUNICODE **	ppuzName,
		FLMUINT *		puiBufSize = NULL);

	RCODE getNodeValue(
		F_DOMNode *		pDOMNode,
		FLMUNICODE **	ppuzValue,
		FLMUINT *		puiBufSize = NULL,
		FLMBOOL			bStartTrans = TRUE);

	RCODE beginTransaction(
		eDbTransType			eTransType);

	RCODE commitTransaction( void);

	RCODE abortTransaction( void);

	void doQuery(
		char *		pszQuery,
		FLMUINT		uiQueryBufSize);

	RCODE indexList( void);

	RCODE editIndexRow(
		DME_ROW_INFO *		pCurRow);

	RCODE editIndexNode(
		DME_ROW_INFO *		pCurRow);

	FLMBOOL canEditRow(
		DME_ROW_INFO *		pCurRow);

	void releaseAllRows( void);

	FINLINE FLMBOOL isDocList( void)
	{
		return( m_bDocList);
	}

	FINLINE FLMUINT getCanvasCols( void)
	{
		return m_uiEditCanvasCols;
	}

	RCODE viewOnlyDeleteIxKey( void);

	FINLINE FTX_WINDOW * getStatusWindow( void)
	{
		return( m_pEditStatusWin);
	}

private:

	/*
	Methods
	*/

	void checkDocument(
		DME_ROW_INFO **		ppDocRow,
		DME_ROW_INFO *			pRow);

	RCODE asciiUCMixToUC(
		char *			pszAscii,
		FLMUNICODE *	puzUnicode,
		FLMUINT			uiMaxUniChars);

	RCODE UCToAsciiUCMix(
		FLMUNICODE *	puzUnicode,
		char *			pszAscii,
		FLMUINT			uiMaxAsciiChars);

	FINLINE DME_ROW_ANCHOR * getRowAnchor( void);

	FINLINE void setRowAnchor(
		DME_ROW_ANCHOR * pRowAnchor);

	RCODE editTextBuffer(
		char **				ppszBuffer,
		FLMUINT				uiBufSize,
		FLMUINT *			puiTermChar);

	RCODE addDocumentToList(
		FLMUINT			uiCollection,
		FLMUINT64		ui64DocumentId);

	void releaseLastRow( void);

	RCODE expandRow(
		DME_ROW_INFO *				pRow,
		FLMBOOL						bOneLevel,
		DME_ROW_INFO **			ppLastRow);

	RCODE collapseRow(
		DME_ROW_INFO **		ppRow);

	RCODE refreshEditWindow(
		DME_ROW_INFO **		ppFirstRow,
		DME_ROW_INFO *			pCursorRow,
		FLMUINT *				puiCurRow);

	RCODE refreshRow(
		FLMUINT					uiRow,
		DME_ROW_INFO *			pNd,
		FLMBOOL					bSelected);

	RCODE clearSelections( void);


	RCODE editRow(
		FLMUINT			uiRow,
		DME_ROW_INFO *	pRow,
		FLMBOOL			bReadOnly = FALSE);


	FLMBOOL canDeleteRow(
		DME_ROW_INFO *		pCurRow);

	RCODE addRowToDb(
		DME_ROW_INFO *		pCurRow,
		FLMUINT				uiCollection,
		FLMBOOL				bAddInBackground,
		FLMBOOL				bStartThread,
		FLMUINT *			pudDrn);

	RCODE modifyRowInDb(
		DME_ROW_INFO *		pCurRow,
		FLMBOOL				bAddInBackground,
		FLMBOOL				bStartThread);

	FLMBOOL isExiting( void);

	RCODE _insertRow(
		DME_ROW_INFO *		pRecord,
		DME_ROW_INFO *		pStartNd = NULL);

	RCODE selectCollection(
		FLMUINT *	puiCollection,
		FLMUINT *	puiTermChar);

	RCODE selectIndex(
		FLMUINT		uiFlags,
		FLMUINT *	puiIndex,
		FLMUINT *	puiTermChar);

	RCODE displayAttributes(
		DME_ROW_INFO *				pRow);

	RCODE displayNodeInfo(
		DME_ROW_INFO *				pRow);

	RCODE exportNode(
		DME_ROW_INFO *				pRow);

	RCODE showHelp(
		FLMUINT *	puiKeyRV = NULL);

	RCODE getNextTitle(
		DME_ROW_INFO *			pRow,
		DME_ROW_INFO **		ppNewRow);

	RCODE getPrevTitle(
		DME_ROW_INFO *			pRow,
		DME_ROW_INFO **		ppNewRow);

	RCODE selectElementAttribute(
		DME_ROW_INFO *		pRow,
		FLMUINT *			puiTermChar);

	RCODE deleteRow(
		DME_ROW_INFO **		ppCurRow);

	RCODE addSomething(
		DME_ROW_INFO **			ppCurRow);

	RCODE createDocumentNode(
		DME_ROW_INFO **			ppCurRow);

	RCODE createRootElementNode(
		DME_ROW_INFO **			ppCurRow);

	RCODE createElementNode(
		DME_ROW_INFO **		ppCurRow);

	RCODE createTextNode(
		DME_ROW_INFO **		ppRow,
		eDomNodeType		eNodeType);

	RCODE createAttributeNode(
		DME_ROW_INFO **		ppCurRow);

	RCODE selectNodeType(
		eDomNodeType *		peNodeType,
		FLMUINT *				puiTermChar);

	RCODE buildNewRow(
		FLMINT					iLevel,
		F_DOMNode *				pDomNode,
		DME_ROW_INFO **		ppNewRow);

	RCODE selectTag(
		FLMUINT			uiTagType,
		FLMUINT *		puiTag,
		FLMUINT *		puiTermChar);

	RCODE selectEncDef(
		FLMUINT			uiTagType,
		FLMUINT *		puiEncDefId,
		FLMUINT *		puiTermChar);

	RCODE addComment(
		DME_ROW_INFO **		ppCurRow,
		const char *	pucFormat, ...);

	/*
	Data
	*/

	DME_ROW_INFO *				m_pScrFirstRow;
	DME_ROW_INFO *				m_pScrLastRow;
	DME_ROW_INFO *				m_pCurRow;
	DME_ROW_ANCHOR *			m_pRowAnchor;
	DME_ROW_INFO *				m_pDocList;
	DME_ROW_INFO *				m_pCurDoc;
	F_Db *						m_pDb;
	FLMBOOL						m_bOpenedDb;
	FLMUINT						m_uiCollection;
	FLMUINT						m_uiLastKey;
	char							m_szTitle[	F_DOMEDIT_MAX_TITLE_SIZE + 1];
	FLMUINT						m_uiCurRow;
	FLMUINT						m_uiEditCanvasRows;
	FLMUINT						m_uiEditCanvasCols;
	FLMUINT						m_uiNumRows;
	FLMUINT						m_uiULX;
	FLMUINT						m_uiULY;
	FLMUINT						m_uiLRX;
	FLMUINT						m_uiLRY;
	FLMBOOL						m_bReadOnly;
	FLMBOOL						m_bSetupCalled;
	FLMBOOL *					m_pbShutdown;
	FLMBOOL						m_bMonochrome;
	FTX_SCREEN *				m_pScreen;
	FTX_WINDOW *				m_pEditWindow;
	FTX_WINDOW *				m_pEditStatusWin;
	F_DomEditor *				m_pParent;
	F_DOMEDIT_DISP_HOOK		m_pDisplayHook;
	void *						m_DisplayData;
	F_DOMEDIT_KEY_HOOK		m_pKeyHook;
	void *						m_KeyData;
	F_DOMEDIT_HELP_HOOK		m_pHelpHook;
	void *						m_HelpData;
	F_DOMEDIT_EVENT_HOOK		m_pEventHook;
	void *						m_EventData;
	FLMBOOL						m_bDocList;
	F_NameTable *				m_pNameTable;
	DME_DISP_COLUMN			m_dispColumns[ 16];
};


FINLINE FLMUINT F_DomEditor::getEditCanvasCols( void)
{
	return m_uiEditCanvasCols;
}

FINLINE void F_DomEditor::setShutdown(
	FLMBOOL *		pbShutdown)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pbShutdown = pbShutdown;
}

FINLINE FLMBOOL * F_DomEditor::getShutdown( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pbShutdown);
}

FINLINE FLMBOOL F_DomEditor::isExiting( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	if( m_pbShutdown && *m_pbShutdown)
	{
		return( TRUE);
	}

	return( FALSE);
}

FINLINE FLMBOOL F_DomEditor::isMonochrome( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_bMonochrome);
}

FINLINE void F_DomEditor::setParent(
	F_DomEditor *		pParent)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pParent = pParent;
}

FINLINE F_DomEditor * F_DomEditor::getParentEditor( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pParent);
}

FINLINE void F_DomEditor::setReadOnly(
	FLMBOOL		bReadOnly)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_bReadOnly = bReadOnly;
}

FINLINE DME_ROW_INFO * F_DomEditor::getScrFirstRow( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pScrFirstRow);
}

FINLINE DME_ROW_INFO * F_DomEditor::getScrLastRow( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pScrLastRow);
}

FINLINE FLMUINT F_DomEditor::getCursorRow( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiCurRow);
}

FINLINE DME_ROW_ANCHOR * F_DomEditor::getRowAnchor( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pRowAnchor);
}

FINLINE void F_DomEditor::setRowAnchor(
	DME_ROW_ANCHOR * pRowAnchor)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pRowAnchor = pRowAnchor;
}

FINLINE FLMUINT F_DomEditor::getNumRows( void)
{
	flmAssert( m_bSetupCalled == TRUE && m_pEditWindow != NULL);
	return( m_uiEditCanvasRows);
}


FINLINE F_Db * F_DomEditor::getDb( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pDb);
}

FINLINE FLMUINT F_DomEditor::getCollection( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiCollection);
}

FINLINE FLMUINT F_DomEditor::getLastKey( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiLastKey);
}

FINLINE FLMUINT F_DomEditor::getULX( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiULX);
}

FINLINE FLMUINT F_DomEditor::getULY( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiULY);
}

FINLINE void F_DomEditor::setDisplayHook(
	F_DOMEDIT_DISP_HOOK		pDispHook,
	void *						DispData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pDisplayHook = pDispHook;
	m_DisplayData = DispData;
}

FINLINE void F_DomEditor::setKeyHook(
	F_DOMEDIT_KEY_HOOK		pKeyHook,
	void *						KeyData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pKeyHook = pKeyHook;
	m_KeyData = KeyData;
}

FINLINE void F_DomEditor::setHelpHook(
	F_DOMEDIT_HELP_HOOK		pHelpHook,
	void *						HelpData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pHelpHook = pHelpHook;
	m_HelpData = HelpData;
}

FINLINE void F_DomEditor::setEventHook(
	F_DOMEDIT_EVENT_HOOK		pEventHook,
	void *						EventData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pEventHook = pEventHook;
	m_EventData = EventData;
}


/*
Prototypes
*/
RCODE F_DomEditorDefaultDispHook(
	F_DomEditor *			pDomEditor,
	DME_ROW_INFO *			pRow,
	FLMUINT *				puiNumVals);

RCODE F_DomEditorViewOnlyKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

RCODE F_DomEditorSelectionKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

RCODE F_DomEditorFileKeyHook(
	F_DomEditor *		pDomEditor,
	DME_ROW_INFO *		pCurRow,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvKeyData);

FLMBOOL domDisplayNodeInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	FLMBOOL			bDoSubTree,
	FLMBOOL			bWaitForKeystroke);

FLMBOOL domDisplayEntryInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT64		ui64NodeId,
	FLMBOOL			bWaitForKeystroke);
	
FLMBOOL domDisplayBTreeInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT			uiLfNum,
	FLMBOOL			bDoCollection,
	FLMBOOL			bDoIndex,
	FLMBOOL			bWaitForKeystroke);
	
#endif
