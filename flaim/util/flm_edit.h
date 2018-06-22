//-------------------------------------------------------------------------
// Desc:	GEDCOM editor - definitions.
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

#ifndef FLM_EDIT_H
#define FLM_EDIT_H

#include "flaim.h"

class F_RecEditor;

#define F_RECEDIT_BUF_SIZE				0x0000FFFF
#define F_RECEDIT_MAX_TITLE_SIZE		64

#define F_RECEDIT_SYSTEM_FIELD			0x0000
#define F_RECEDIT_CONTROL_INFO_FIELD	0x0001
#define F_RECEDIT_FLAGS_FIELD				0x0002
#define F_RECEDIT_VAL_VIEW_FIELD			0x0003
#define F_RECEDIT_VISIBLE_FIELD			0x0004
#define F_RECEDIT_CONVTYPE_FIELD			0x0005
#define F_RECEDIT_REFNODE_FIELD			0x0006
#define F_RECEDIT_VIEWTYPE_FIELD			0x0007
#define F_RECEDIT_COMMENT_FIELD			0x0008
#define F_RECEDIT_LINK_DEST_FIELD		0x0009
#define F_RECEDIT_VALANNO_FIELD			0x000A
#define F_RECEDIT_APPDEF_FIELD			0x000B	// Application-specific field
#define F_RECEDIT_INVISIBLE_CNT_FIELD	0x000C

/*
System flags
*/

#define F_RECEDIT_FLAG_FLDMOD				0x00000001
#define F_RECEDIT_FLAG_RECMOD				0x00000002
#define F_RECEDIT_FLAG_LIST_ITEM			0x00000004
#define F_RECEDIT_FLAG_HIDE_TAG			0x00000008
#define F_RECEDIT_FLAG_HIDE_LEVEL		0x00000010
#define F_RECEDIT_FLAG_HIDE_SOURCE		0x00000020
#define F_RECEDIT_FLAG_READ_ONLY			0x00000040
#define F_RECEDIT_FLAG_SELECTED			0x00000080
#define F_RECEDIT_FLAG_NEWFLD				0x00000100
#define F_RECEDIT_FLAG_NO_DELETE			0x00000200
#define F_RECEDIT_FLAG_COLLAPSED			0x00000400

/*
Index selection flags
*/

#define F_RECEDIT_ISEL_NOIX				0x0001	// Show "no index" as an option

/*
Configuration options
*/

#define F_RECEDIT_CONFIG_STATS_START	0x0001
#define F_RECEDIT_CONFIG_STATS_STOP		0x0002
#define F_RECEDIT_CONFIG_STATS_RESET	0x0003

enum eEventType
{
	F_RECEDIT_EVENT_RECREAD,
	F_RECEDIT_EVENT_RECINSERT,
	F_RECEDIT_EVENT_GETDISPVAL,
	F_RECEDIT_EVENT_GETNEXTNODE,
	F_RECEDIT_EVENT_GETPREVNODE,
	F_RECEDIT_EVENT_IEDIT,				// Interactive editor invoked
	F_RECEDIT_EVENT_REFRESH,			// Called prior to refresh
	F_RECEDIT_EVENT_NAME_TABLE
};

typedef struct
{
	FLMUINT		uiContainer;
	FLMUINT		uiDrn;
	NODE *		pRec;
} DBE_REC_INFO;

typedef struct
{
	NODE *		pNd;
	char *		pucBuf;
	FLMUINT		uiBufLen;
	FLMUINT		uiConvType;
	FLMBOOL		bIsSystemNd;
} DBE_VAL_INFO;

typedef struct
{
	/*
	Input
	*/

	NODE *		pCurNd;

	/*
	Output
	*/

	NODE *		pNd;
	FLMBOOL		bUseNd;
} DBE_NODE_INFO;

typedef struct
{
	F_NameTable	*	pNameTable;
	FLMBOOL		bInitialized;
} DBE_NAME_TABLE_INFO;

typedef struct
{
	char			pucString[ 128];
	FLMUINT		uiCol;
	eColorType	foreground;
	eColorType	background;
} DBE_DISP_COLUMN;

/*
Callbacks
*/

typedef RCODE (* F_RECEDIT_DISP_HOOK)(
	F_RecEditor *		pRecEditor,
	NODE *				pNd,
	void *				UserData,
	DBE_DISP_COLUMN *	pDispVals,
	FLMUINT *			puiNumVals);

typedef RCODE (* F_RECEDIT_LINK_HOOK)(
	F_RecEditor *		pRecEditor,
	NODE *				pLinkNd,
	void *				UserData,
	FLMUINT 				uiLinkKey);

typedef RCODE (* F_RECEDIT_KEY_HOOK)(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

typedef RCODE (* F_RECEDIT_HELP_HOOK)(
	F_RecEditor *		pRecEditor,
	F_RecEditor *		pHelpEditor,
	F_Pool *				pPool,
	void *				UserData,
	NODE **				ppRootNd);

typedef RCODE (* F_RECEDIT_EVENT_HOOK)(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

/*
Class definitions
*/

class F_RecEditor : public F_Object
{
	private:

		IF_FileSystem *			m_pFileSystem;
		char *						m_pucTmpBuf;
		F_NameTable *				m_pNameTable;
		NODE * 						m_pTree;
		NODE *						m_pCurNd;
		NODE *						m_pScrFirstNd;
		F_Pool						m_scratchPool;
		F_Pool						m_treePool;
		HFDB							m_hDefaultDb;
		FLMUINT						m_uiDefaultCont;
		FLMUINT						m_uiDefaultStore;
		FLMUINT						m_uiLastKey;
		char							m_pucTitle[	F_RECEDIT_MAX_TITLE_SIZE + 1];
		FLMUINT						m_uiCurRow;
		FLMUINT						m_uiEditCanvasRows;
		char							m_pucAdHocQuery[ 1024];
		FLMUINT						m_uiULX;
		FLMUINT						m_uiULY;
		FLMUINT						m_uiLRX;
		FLMUINT						m_uiLRY;
		FLMBOOL						m_bReadOnly;
		FLMBOOL						m_bSetupCalled;
		FLMBOOL *					m_pbShutdown;
		FLMBOOL						m_bOwnNameTable;
		FLMBOOL						m_bMonochrome;
		FTX_SCREEN *				m_pScreen;
		FTX_WINDOW *				m_pEditWindow;
		FTX_WINDOW *				m_pEditStatusWin;
		F_RecEditor *				m_pParent;
		F_RecEditor *				m_pNameList;
		F_RECEDIT_DISP_HOOK		m_pDisplayHook;
		void *						m_DisplayData;
		F_RECEDIT_LINK_HOOK		m_pLinkHook;
		void *						m_LinkData;
		F_RECEDIT_KEY_HOOK		m_pKeyHook;
		void *						m_KeyData;
		F_RECEDIT_HELP_HOOK		m_pHelpHook;
		void *						m_HelpData;
		F_RECEDIT_EVENT_HOOK		m_pEventHook;
		void *						m_EventData;

		/*
		Methods
		*/

		RCODE refreshEditWindow(
			NODE **			ppFirstNd,
			NODE * 			pCursorNd,
			FLMUINT *		puiCurRow);

		RCODE refreshRow(
			FLMUINT			uiRow,
			NODE *			pNd,
			FLMBOOL			bSelected);

		RCODE clearSelections( void);

		RCODE editNode(
			FLMUINT	uiNdRow,
			NODE *	pNd);

		RCODE editTextNode(
			FTX_WINDOW *	pWindow,
			NODE *			pNd,
			FLMBOOL *		pbModified);

		RCODE editNumberNode(
			FTX_WINDOW *	pWindow,
			NODE *			pNd,
			FLMBOOL *		pbModified);

		RCODE editContextNode(
			FTX_WINDOW *	pWindow,
			NODE *			pNd,
			FLMBOOL *		pbModified);

		RCODE editBinaryNode(
			FTX_WINDOW *		pWindow,
			NODE *				pNd,
			FLMBOOL *			pbModified);

		RCODE createSystemNode(
			NODE *	pCurNd,
			FLMUINT	uiTagNum,
			NODE **	ppSystemNd);

		RCODE getControlNode(
			NODE *		pCurNd,
			FLMBOOL		bCreate,
			NODE **		ppControlNd);

		RCODE addAltView(
			NODE *		pCurNd,
			FLMUINT 		uiViewType);

		FLMBOOL canEditRecord(
			NODE *		pCurNd);

		FLMBOOL canEditNode(
			NODE *		pCurNd);

		FLMBOOL canDeleteRecord(
			NODE *		pCurNd);

		FLMBOOL canDeleteNode(
			NODE *		pCurNd);

		RCODE addRecordToDb(
			NODE *		pCurNd,
			FLMUINT		uiContainer,
			FLMBOOL		bAddInBackground,
			FLMBOOL		bStartThread,
			FLMUINT *	pudDrn);

		RCODE deleteRecordFromDb(
			NODE *		pCurNd);

		RCODE deleteRecordFromDb(
			HFDB			hSourceDb,
			FLMUINT		uiSourceCont,
			FLMUINT		uiSourceDrn);
	
		RCODE modifyRecordInDb(
			NODE *		pCurNd,
			FLMBOOL		bAddInBackground,
			FLMBOOL		bStartThread);

		FLMBOOL isExiting( void);

		RCODE createNewField(
			FLMBOOL		bAllocSource,
			NODE **		ppNewField);

		RCODE refreshNameTable( void);

		RCODE followLink(
			NODE *		pLinkNd,
			FLMUINT		uiLinkKey);

		RCODE _insertRecord(
			NODE *		pRecord,
			NODE *		pStartNd = NULL);

		RCODE selectContainer(
			FLMUINT *	puiContainer,
			FLMUINT *	puiTermChar);

		RCODE selectIndex(
			FLMUINT		uiContainer,
			FLMUINT		uiFlags,
			FLMUINT *	puiIndex,
			FLMUINT *	puiContainer,
			FLMUINT *	puiTermChar);

		RCODE showHelp(
			FLMUINT *	puiKeyRV = NULL);

		RCODE adHocQuery(
			FLMBOOL			bRetrieve = TRUE,
			FLMBOOL			bPurge = FALSE);

	public:

		F_RecEditor( void);
		~F_RecEditor( void);

		void reset( void);

		RCODE Setup(
			FTX_SCREEN *	pScreen);

		void setParent(
			F_RecEditor *	pParent);

		F_RecEditor * getParentEditor( void);

		RCODE setDefaultSource(
			HFDB			hDb,
			FLMUINT		uiContainer);

		RCODE setTree(
			NODE *		pTree,
			NODE **		ppNewNd = NULL);

		RCODE pruneTree(
			NODE *			pCurNd);

		RCODE appendTree(
			NODE *		pTree,
			NODE **		ppNewRoot);

		RCODE insertRecord(
			NODE *		pRecord,
			NODE **		ppNewRoot,
			NODE *		pStartNd = NULL);

		RCODE retrieveRecordFromDb(
			FLMUINT		uiContainer,
			FLMUINT		uiDrn);

		RCODE markRecordModified(
			NODE *	pCurNd);

		FLMBOOL isRecordModified(
			NODE *	pCurNd);

		RCODE clearRecordModified(
			NODE *	pCurNd);

		NODE * getTree( void);

		RCODE setTitle(
			const char *	pucTitle);

		RCODE setCurrentAtTop( void);

		RCODE setCurrentAtBottom( void);

		void setReadOnly(
			FLMBOOL		bReadOnly);

		RCODE copyCleanRecord(
			F_Pool *		pPool,
			NODE *		pRecNd,
			NODE **		ppCopiedRec);

		RCODE copyCleanTree(
			F_Pool *		pPool,
			NODE *		pTreeNd,
			NODE **		ppCopiedTree);

		RCODE interactiveEdit(
			FLMUINT		uiULX = 0,
			FLMUINT		uiULY = 0,
			FLMUINT		uiLRX = 0,
			FLMUINT		uiLRY = 0,
			FLMBOOL		bBorder = TRUE,
			FLMBOOL		bStatus = TRUE);

		FTX_SCREEN * getScreen( void);

		FLMBOOL isMonochrome( void);

		NODE * getPrevNode(
			NODE *	pCurNd,
			FLMBOOL	bUseCallback = TRUE);

		NODE * getNextNode(
			NODE *	pCurNd,
			FLMBOOL	bUseCallback = TRUE);

		NODE * getPrevRecord(
			NODE *	pCurNd);

		NODE * getNextRecord(
			NODE *	pCurNd);

		NODE * getRootNode(
			NODE *	pCurNd);

		NODE * getChildNode(
			NODE *	pCurNd);

		NODE * getCurrentNode( void);

		NODE * getFirstNode( void);

		FLMUINT getCursorRow( void);

		FLMUINT getNumRows( void);

		RCODE getDisplayValue(
			NODE *		pNd,
			FLMUINT		uiConvType,
#define F_RECEDIT_DEFAULT_TYPE		0x0000
#define F_RECEDIT_TEXT_TYPE			0x0001
#define F_RECEDIT_BINARY_TYPE			0x0002
			char *		pucBuf,
			FLMUINT		uiBufSize);

		NODE * findRecord(
			FLMUINT		uiContainer,
			FLMUINT		uiDrn,
			NODE *		pStartNd = NULL);

		FLMBOOL isNodeVisible(
			NODE *		pCurNd);

		FLMBOOL isSystemNode(
			NODE *		pCurNd);

		void setShutdown(
			FLMBOOL *	pbShutdown);

		FLMBOOL * getShutdown( void);

		RCODE setCurrentNode(
			NODE *		pCurNd);

		RCODE setFirstNode(
			NODE *		pNd);

		RCODE setControlFlags(
			NODE *		pCurNd,
			FLMUINT 		uiFlags);

		RCODE setNameTable(
			F_NameTable *	pNameTable);

		RCODE getControlFlags(
			NODE *		pCurNd,
			FLMUINT *	puiFlags);

		void setDisplayHook(
			F_RECEDIT_DISP_HOOK	pDispHook,
			void *					DispData);

		void setLinkHook(
			F_RECEDIT_LINK_HOOK	pLinkHook,
			void *					LinkData);

		void setKeyHook(
			F_RECEDIT_KEY_HOOK	pKeyHook,
			void *					KeyData);

		void setHelpHook(
			F_RECEDIT_HELP_HOOK	pHelpHook,
			void *					HelpData);

		void setEventHook(
			F_RECEDIT_EVENT_HOOK	pEventHook,
			void *					EventData);

		HFDB getDb( void);
		
		FLMUINT getContainer( void);
		
		FLMUINT getLastKey( void);
		
		IF_FileSystem * getFileSystem( void);

		RCODE getNumber(
			const char *	pucBuf,
			FLMUINT *		puiValue,
			FLMINT *			piValue);

		RCODE getDictionaryName(
			FLMUINT			uiNum,
			char *			pucName);

		RCODE getFieldType(
			FLMUINT			uiFieldNum,
			FLMUINT *		puiFieldType);

		RCODE getFieldNumber(
			const char *	pucFieldName,
			FLMUINT *		puiFieldNum);

		RCODE getContainerNumber(
			const char *	pucContainerName,
			FLMUINT *		puiContainerNum);

		RCODE getIndexNumber(
			const char *	pucIndexName,
			FLMUINT *		puiIndexNum);

		RCODE addComment(
			NODE *			pCurNd,
			FLMBOOL			bVisible,
			const char *	pucFormat, ...);

		RCODE addAnnotation(
			NODE *			pCurNd,
			const char *	pucFormat, ...);

		RCODE setLinkDestination(
			NODE *			pCurNd,
			FLMUINT			uiContainer,
			FLMUINT			uiDrn);

		RCODE getLinkDestination(
			NODE *			pCurNd,
			FLMUINT *		puiContainer,
			FLMUINT *		puiDrn);

		FLMBOOL areRecordsEqual(
			NODE *		pRootA,
			NODE *		pRootB);

		RCODE retrieveRecordsFromDb(
			FLMUINT		uiContainer,
			FLMUINT		uiFirstDrn,
			FLMUINT		uiLastDrn);

		RCODE getSystemNode(
			NODE *		pCurNd,
			FLMUINT		uiTagNum,
			FLMUINT		uiNth,
			NODE **		ppSystemNd);

		RCODE indexList( void);

		RCODE fileManager(
			const char *	pucTitle,
			FLMUINT			uiModeFlags,
#define	F_RECEDIT_FSEL_PROMPT				0x00000001
			char *			pszInitialPath,
			char *			pszSelectedPath,
			FLMUINT *		puiTermChar);

		RCODE fileViewer(
			const char *	pucTitle,
			const char *	pszFilePath,
			FLMUINT *		puiTermChar);

		RCODE requestInput(
			const char *	pucMessage,
			char *			pucResponse,
			FLMUINT			uiMaxRespLen,
			FLMUINT *		puiTermChar);

		RCODE copyBuffer(
			F_Pool *			pPool,
			NODE *			pStartNd,
			NODE **			ppNewTree);

		RCODE displayMessage(
			const char *	pucMessage,
			RCODE				rcOfMessage,
			FLMUINT *		puiTermChar,
			eColorType		background,
			eColorType		foreground);

		RCODE globalConfig(
			FLMUINT			uiOption);

		RCODE createStatusWindow(
			const char *	pucTitle,
			eColorType		back,
			eColorType		fore,
			FLMUINT *		puiCols,
			FLMUINT *		puiRows,
			FTX_WINDOW **	ppWindow);

		RCODE asciiUCMixToUC(
			char *			pucAscii,
			FLMUNICODE *	puzUnicode,
			FLMUINT			uiMaxUniChars);

		RCODE UCToAsciiUCMix(
			FLMUNICODE *	puzUnicode,
			char *			pucAscii,
			FLMUINT			uiMaxAsciiChars);

		RCODE expandNode(
			NODE *			pNode,
			FLMBOOL *		pbExpanded);

		RCODE collapseNode(
			NODE *			pNode,
			FLMBOOL *		pbCollapsed);

		FLMUINT getULX( void);
		
		FLMUINT getULY( void);

		RCODE openNewDb();
};

FINLINE void F_RecEditor::setShutdown(
	FLMBOOL *		pbShutdown)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pbShutdown = pbShutdown;
}

FINLINE FLMBOOL * F_RecEditor::getShutdown( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pbShutdown);
}

FINLINE FLMBOOL F_RecEditor::isExiting( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	if( m_pbShutdown && *m_pbShutdown)
	{
		return( TRUE);
	}

	return( FALSE);
}

FINLINE FLMBOOL F_RecEditor::isMonochrome( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_bMonochrome);
}

FINLINE RCODE F_RecEditor::setNameTable(
	F_NameTable *			pNameTable)
{
	flmAssert( m_bSetupCalled == TRUE);

	if( m_pNameTable && m_bOwnNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}
	m_pNameTable = pNameTable;
	m_bOwnNameTable = FALSE;

	return( FERR_OK);
}

FINLINE void F_RecEditor::setParent(
	F_RecEditor *		pParent)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pParent = pParent;
}

FINLINE F_RecEditor * F_RecEditor::getParentEditor( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pParent);
}

FINLINE void F_RecEditor::setReadOnly(
	FLMBOOL		bReadOnly)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_bReadOnly = bReadOnly;
}

FINLINE NODE * F_RecEditor::getTree( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pTree);
}

FINLINE NODE * F_RecEditor::getCurrentNode( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pCurNd);
}

FINLINE NODE * F_RecEditor::getFirstNode( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pScrFirstNd);
}

FINLINE FLMUINT F_RecEditor::getCursorRow( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiCurRow);
}

FINLINE FLMUINT F_RecEditor::getNumRows( void)
{
	flmAssert( m_bSetupCalled == TRUE && m_pEditWindow != NULL);
	return( m_uiEditCanvasRows);
}

FINLINE HFDB F_RecEditor::getDb( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_hDefaultDb);
}

FINLINE FLMUINT F_RecEditor::getContainer( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiDefaultCont);
}

FINLINE FTX_SCREEN * F_RecEditor::getScreen( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pScreen);
}

FINLINE FLMUINT F_RecEditor::getLastKey( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiLastKey);
}

FINLINE FLMUINT F_RecEditor::getULX( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiULX);
}

FINLINE FLMUINT F_RecEditor::getULY( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_uiULY);
}

FINLINE IF_FileSystem * F_RecEditor::getFileSystem( void)
{
	flmAssert( m_bSetupCalled == TRUE);
	return( m_pFileSystem);
}

FINLINE void F_RecEditor::setDisplayHook(
	F_RECEDIT_DISP_HOOK		pDispHook,
	void *						DispData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pDisplayHook = pDispHook;
	m_DisplayData = DispData;
}

FINLINE void F_RecEditor::setLinkHook(
	F_RECEDIT_LINK_HOOK		pLinkHook,
	void *						LinkData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pLinkHook = pLinkHook;
	m_LinkData = LinkData;
}

FINLINE void F_RecEditor::setKeyHook(
	F_RECEDIT_KEY_HOOK		pKeyHook,
	void *						KeyData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pKeyHook = pKeyHook;
	m_KeyData = KeyData;
}

FINLINE void F_RecEditor::setHelpHook(
	F_RECEDIT_HELP_HOOK		pHelpHook,
	void *						HelpData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pHelpHook = pHelpHook;
	m_HelpData = HelpData;
}

FINLINE void F_RecEditor::setEventHook(
	F_RECEDIT_EVENT_HOOK		pEventHook,
	void *						EventData)
{
	flmAssert( m_bSetupCalled == TRUE);
	m_pEventHook = pEventHook;
	m_EventData = EventData;
}

RCODE f_RecEditorDefaultDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals);

RCODE f_RecEditorDefaultLinkHook(
	F_RecEditor *		pRecEditor,
	NODE *				pLinkNd,
	void *				UserData,
	FLMUINT				uiLinkKey);

RCODE f_RecEditorViewOnlyKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE f_RecEditorSelectionKeyHook(
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

#endif	// FLM_EDIT_H
