//------------------------------------------------------------------------------
// Desc: Command-line environment for FLAIM utilities
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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

#include "fshell.h"
#include "domedit.h"

// Imported global variables.

extern FLMBOOL								gv_bShutdown;

// XML Import Error Handling

#define	flmErrorCodeEntry( c)		{ c, #c }
#define	MAX_IMPORT_ERROR_STRING		65

typedef struct
{
	RCODE				rc;
	const char *	pszErrorStr;
} XSHELL_ERROR_CODE_MAP;

XSHELL_ERROR_CODE_MAP gv_XMLParseErrors[
		XML_NUM_ERRORS - 1] =
{
		flmErrorCodeEntry( XML_ERR_BAD_ELEMENT_NAME),
		flmErrorCodeEntry( XML_ERR_XMLNS_IN_ELEMENT_NAME),
		flmErrorCodeEntry( XML_ERR_ELEMENT_NAME_MISMATCH),
		flmErrorCodeEntry( XML_ERR_PREFIX_NOT_DEFINED),
		flmErrorCodeEntry( XML_ERR_EXPECTING_GT),
		flmErrorCodeEntry( XML_ERR_EXPECTING_ELEMENT_LT),
		flmErrorCodeEntry( XML_ERR_EXPECTING_EQ),
		flmErrorCodeEntry( XML_ERR_MULTIPLE_XMLNS_DECLS),
		flmErrorCodeEntry( XML_ERR_MULTIPLE_PREFIX_DECLS),
		flmErrorCodeEntry( XML_ERR_EXPECTING_QUEST_GT),
		flmErrorCodeEntry( XML_ERR_INVALID_XML_MARKUP),
		flmErrorCodeEntry( XML_ERR_MUST_HAVE_ONE_ATT_DEF),
		flmErrorCodeEntry( XML_ERR_EXPECTING_NDATA),
		flmErrorCodeEntry( XML_ERR_EXPECTING_SYSTEM_OR_PUBLIC),
		flmErrorCodeEntry( XML_ERR_EXPECTING_LPAREN),
		flmErrorCodeEntry( XML_ERR_EXPECTING_RPAREN_OR_PIPE),
		flmErrorCodeEntry( XML_ERR_EXPECTING_NAME),
		flmErrorCodeEntry( XML_ERR_INVALID_ATT_TYPE),
		flmErrorCodeEntry( XML_ERR_INVALID_DEFAULT_DECL),
		flmErrorCodeEntry( XML_ERR_EXPECTING_PCDATA),
		flmErrorCodeEntry( XML_ERR_EXPECTING_ASTERISK),
		flmErrorCodeEntry( XML_ERR_EMPTY_CONTENT_INVALID),
		flmErrorCodeEntry( XML_ERR_CANNOT_MIX_CHOICE_AND_SEQ),
		flmErrorCodeEntry( XML_ERR_XML_ILLEGAL_PI_NAME),
		flmErrorCodeEntry( XML_ERR_ILLEGAL_FIRST_NAME_CHAR),
		flmErrorCodeEntry( XML_ERR_ILLEGAL_COLON_IN_NAME),
		flmErrorCodeEntry( XML_ERR_EXPECTING_VERSION),
		flmErrorCodeEntry( XML_ERR_INVALID_VERSION_NUM),
		flmErrorCodeEntry( XML_ERR_ENCODING_NOT_SUPPORTED),
		flmErrorCodeEntry( XML_ERR_EXPECTING_YES_OR_NO),
		flmErrorCodeEntry( XML_ERR_EXPECTING_QUOTE_BEFORE_EOL),
		flmErrorCodeEntry( XML_ERR_EXPECTING_SEMI),
		flmErrorCodeEntry( XML_ERR_UNEXPECTED_EOL_IN_ENTITY),
		flmErrorCodeEntry( XML_ERR_INVALID_CHARACTER_NUMBER),
		flmErrorCodeEntry( XML_ERR_UNSUPPORTED_ENTITY),
		flmErrorCodeEntry( XML_ERR_EXPECTING_QUOTE),
		flmErrorCodeEntry( XML_ERR_INVALID_PUBLIC_ID_CHAR),
		flmErrorCodeEntry( XML_ERR_EXPECTING_WHITESPACE),
		flmErrorCodeEntry( XML_ERR_EXPECTING_HEX_DIGIT),
		flmErrorCodeEntry( XML_ERR_INVALID_BINARY_ATTR_VALUE),
		flmErrorCodeEntry( XML_ERR_CREATING_CDATA_NODE),
		flmErrorCodeEntry( XML_ERR_CREATING_COMMENT_NODE),
		flmErrorCodeEntry( XML_ERR_CREATING_PI_NODE),
		flmErrorCodeEntry( XML_ERR_CREATING_DATA_NODE),
		flmErrorCodeEntry( XML_ERR_CREATING_ROOT_ELEMENT),
		flmErrorCodeEntry( XML_ERR_CREATING_ELEMENT_NODE),
		flmErrorCodeEntry( XML_ERR_XML_PREFIX_REDEFINITION)
};

#define FSMI_ENTRY_ELEMENT												2
#define FSMI_INTERNAL_ENTRY_FLAGS_ATTR								21
#define FSMI_ENTRY_MODIFY_TIME_ATTR									22
#define FSMI_ENTRY_FLAGS_ATTR											23
#define FSMI_PARTITION_ID_ATTR										24
#define FSMI_CLASS_ID_ATTR												25
#define FSMI_PARENT_ID_ATTR											26
#define FSMI_ALT_ID_ATTR												27
#define FSMI_SUBORDINATE_COUNT_ATTR									28
#define FSMI_RDN_ATTR													29
#define FSMI_FIRST_CHILD_ATTR											30
#define FSMI_LAST_CHILD_ATTR											31
#define FSMI_NEXT_SIBLING_ATTR										32
#define FSMI_PREV_SIBLING_ATTR										33

#define FSMI_ATTR_GVTS_ATTR											50
#define FSMI_ATTR_DTS_ATTR												51
#define FSMI_VALUE_FLAGS_ATTR											52
#define FSMI_VALUE_MTS_ATTR											53
#define FSMI_TTL_ATTR													54
#define FSMI_POLICY_DN_ATTR											55

typedef struct OVERHEAD_INFO
{
	FLMUINT64	ui64DOMOverhead;
	FLMUINT64	ui64ValueBytes;
} OVERHEAD_INFO;

typedef struct ATTR_NODE_INFO
{
	FLMUINT			uiAttrNameId;
	char *			pszAttrName;
	FLMUINT64		ui64NumValues;
	OVERHEAD_INFO	GVTS;
	OVERHEAD_INFO	DTS;
	OVERHEAD_INFO	ValueFlags;
	OVERHEAD_INFO	ValueMTS;
	OVERHEAD_INFO	TTL;
	OVERHEAD_INFO	PolicyDN;
	OVERHEAD_INFO	OtherNodes;
} ATTR_NODE_INFO;

class Entry_Info
{
public:

	Entry_Info();

	~Entry_Info();

	FINLINE void resetOverheadInfo(
		OVERHEAD_INFO *	pOverhead)
	{
		pOverhead->ui64DOMOverhead = 0;
		pOverhead->ui64ValueBytes = 0;
	}

	FINLINE void initAttrInfo(
		FLMUINT				uiAttrNameId,
		ATTR_NODE_INFO *	pAttrInfo)
	{
		pAttrInfo->uiAttrNameId = uiAttrNameId;
		pAttrInfo->pszAttrName = NULL;
		pAttrInfo->ui64NumValues = 0;
		resetOverheadInfo( &pAttrInfo->GVTS);
		resetOverheadInfo( &pAttrInfo->DTS);
		resetOverheadInfo( &pAttrInfo->ValueFlags);
		resetOverheadInfo( &pAttrInfo->ValueMTS);
		resetOverheadInfo( &pAttrInfo->TTL);
		resetOverheadInfo( &pAttrInfo->PolicyDN);
		resetOverheadInfo( &pAttrInfo->OtherNodes);
	}

	void getOverheadInfo(
		OVERHEAD_INFO *	pOverhead);

	RCODE getXMLAttrInfo(
		IF_Db *				pDb,
		IF_DOMNode *		pNode,
		OVERHEAD_INFO *	pOverhead,
		FLMUINT				uiAttrNameId);

	RCODE getAttrValueInfo(
		IF_Db *				pDb,
		IF_DOMNode *		pValueNode,
		ATTR_NODE_INFO *	pAttrNodeInfo);

	RCODE processAttrValues(
		IF_Db *				pDb,
		IF_DOMNode *		pAttrNode,
		ATTR_NODE_INFO *	pAttrNodeInfo);

	FLMBOOL findDirAttr(
		FLMUINT		uiAttrNameId,
		FLMUINT *	puiInsertPos);

	RCODE getDirAttrInfo(
		IF_Db *			pDb,
		IF_DOMNode *	pAttrNode);

	RCODE processEntryAttrs(
		IF_Db *			pDb,
		IF_DOMNode *	pEntryNode);

	RCODE addEntryInfo(
		IF_Db *			pDb,
		IF_DOMNode *	pEntryNode);

	FLMUINT64			m_ui64NumEntries;
	FLMUINT64			m_ui64TotalBytes;
	OVERHEAD_INFO		m_EntryNode;
	OVERHEAD_INFO		m_AttrNode;
	OVERHEAD_INFO		m_InternalEntryFlags;
	OVERHEAD_INFO		m_ModifyTime;
	OVERHEAD_INFO		m_EntryFlags;
	OVERHEAD_INFO		m_PartitionID;
	OVERHEAD_INFO		m_ClassID;
	OVERHEAD_INFO		m_ParentID;
	OVERHEAD_INFO		m_AlternateID;
	OVERHEAD_INFO		m_SubordinateCount;
	OVERHEAD_INFO		m_RDN;
	OVERHEAD_INFO		m_FirstChild;
	OVERHEAD_INFO		m_LastChild;
	OVERHEAD_INFO		m_NextSibling;
	OVERHEAD_INFO		m_PrevSibling;
	ATTR_NODE_INFO *	m_pAttrList;
	FLMUINT				m_uiAttrListSize;
	FLMUINT				m_uiNumAttrs;
	F_NodeInfo			m_nodeInfo;
	F_Pool				m_pool;
};

// Local prototypes

RCODE flmBackupProgFunc(
	FLMUINT		uiStatusType,
	void *		Parm1,
	void *		Parm2,
	void *		UserData);

FLMINT fshellFileSystemTest(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell);

FSTATIC void format64BitNum(
	FLMUINT64	ui64Num,
	char *		pszBuf,
	FLMBOOL		bOutputHex,
	FLMBOOL		bAddCommas = FALSE);

RCODE flmXmlImportStatus(
	eXMLStatus		eStatusType,
	void *			pvArg1,
	void *			pvArg2,
	void *			pvArg3,
	void *			pvUserData);

RCODE importXmlFiles(
	IF_Db *						pDb,
	FLMUINT						uiCollection,
	FlmShell *					pShell,
	char *						pszPath,
	XFLM_IMPORT_STATS *		pImportStats);

RCODE	getErrFromFile(
	IF_PosIStream *	pFileIStream,
	FLMUINT				uiCurrLineFilePos,
	FLMUINT				uiCurrLineBytes,
	XMLEncoding			eXMLEncoding,
	char *				pszErrorString);

FSTATIC void domDisplayError(
	FTX_WINDOW *	pWindow,
	const char *	pszError,
	RCODE				errRc);

FSTATIC void domOutputValues(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLabel,
	FLMUINT64		ui64Bytes,
	FLMUINT			uiPercent,
	FLMUINT64		ui64Count);

FSTATIC void domDisplayLine(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLine,
	const char *	pszWaitPrompt = NULL);

FSTATIC void domDisplayValue(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLabel,
	FLMUINT			uiPercent,
	FLMUINT64		ui64Value);

FSTATIC void domDisplayInfo(
	FTX_WINDOW *		pWindow,
	FLMUINT *			puiLineCount,
	IF_FileHdl *		pFileHdl,
	const char *		pszDomOverheadLabel,
	const char *		pszValueBytesLabel,
	OVERHEAD_INFO *	pInfo,
	FLMUINT64			ui64TotalBytes);

const char * errorToString(
	XMLParseError	errorType);

/****************************************************************************
Desc:
*****************************************************************************/
class F_LocalRestore : public F_FSRestore
{
public:

	~F_LocalRestore() {}
	F_LocalRestore()
	{
	}

	FINLINE RCODE setup(
		char *			pszDbPath,
		char *			pszBackupSetPath,
		char *			pszRflDir)
	{
		return( F_FSRestore::setup( pszDbPath,
			pszBackupSetPath, pszRflDir));
	}

	FINLINE RCODE XFLAPI openRflFile(
		FLMUINT	uiFileNum)
	{
		return( F_FSRestore::openRflFile( uiFileNum));
	}

	FINLINE RCODE XFLAPI read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead)
	{
		return( F_FSRestore::read( uiLength, pvBuffer, puiBytesRead));
	}

private:

};

/****************************************************************************
Desc:
*****************************************************************************/
class F_LocalRestoreStatus : public F_DefaultRestoreStatus
{
public:

	F_LocalRestoreStatus(
		FlmShell *		pShell)
	{
		m_pShell = pShell;
		m_bFirstStatus = TRUE;
		m_uiTransCount = 0;
		m_uiAddCount = 0;
		m_uiDeleteCount = 0;
		m_uiModifyCount = 0;
		m_uiReserveCount = 0;
		m_uiIndexCount = 0;
		m_uiRflFileNum = 0;
		m_ui64RflBytesRead = 0;
	}

	RCODE XFLAPI reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64			ui64BytesToDo,
		FLMUINT64			ui64BytesDone);

	RCODE XFLAPI reportError(
		eRestoreAction *	peAction,
		RCODE					rcErr);

	RCODE XFLAPI reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum)
	{
		m_uiRflFileNum = uiFileNum;
		updateCountDisplay();
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;

		return( NE_XFLM_OK);
	}

	RCODE XFLAPI reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum,
		FLMUINT				uiBytesRead)
	{
		(void)uiFileNum;

		m_ui64RflBytesRead += uiBytesRead;
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;

		return( NE_XFLM_OK);
	}

	RCODE XFLAPI reportBlockChainFree(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT64,				// ui64MaintDocNum,
		FLMUINT,					// uiStartBlkAddr,
		FLMUINT,					// uiEndBlkAddr,
		FLMUINT					// uiCount
		)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLAPI reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLAPI reportRollOverDbKey(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportDocumentDone(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64)				// ui64NodeId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeCreate(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64RefNodeId,
		eDomNodeType,			// eNodeType,
		FLMUINT,					// uiNameId,
		eNodeInsertLoc)		// eLocation)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeDelete(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64)				// ui64NodeId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeChildrenDelete(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64NodeId,
		FLMUINT)					// uiNameId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportInsertBefore(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64ParentId,
		FLMUINT64,				// ui64NewChildId,
		FLMUINT64)				// ui64RefChildId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeUpdate(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64)				// ui64NodeId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64)				// ui64NodeId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeFlagsUpdate(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64NodeId,
		FLMUINT,					// uiFlags,
		FLMBOOL)					// bAdd)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportNodeSetPrefixId(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64NodeId,
		FLMUINT)					// uiPrefixId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
	RCODE XFLAPI reportSetNextNodeId(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64)				// ui64NextNodeId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLAPI reportNodeSetMetaValue(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64NodeId,
		FLMUINT64				// ui64MetaValue
		)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
	
	RCODE XFLAPI reportAttributeDelete(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64ElementId,
		FLMUINT)					// uiAttrName)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLAPI reportAttributeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64ElementId,
		FLMUINT)					// uiNameId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
	
	RCODE XFLAPI reportNodeSetPrefixId(
		eRestoreAction *		peAction,
		FLMUINT64,				// ui64TransId,
		FLMUINT,					// uiCollection,
		FLMUINT64,				// ui64NodeId,
		FLMUINT,					// uiAttrName,
		FLMUINT)					// uiPrefixId)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}
		
private:

	RCODE report_preamble(
		FTX_WINDOW *	pWin);

	RCODE report_postamble(
		FTX_WINDOW *	pWin);

	void updateCountDisplay( void);

	FlmShell *			m_pShell;
	FLMBOOL				m_bFirstStatus;
	FLMUINT				m_uiTransCount;
	FLMUINT				m_uiAddCount;
	FLMUINT				m_uiDeleteCount;
	FLMUINT				m_uiModifyCount;
	FLMUINT				m_uiReserveCount;
	FLMUINT				m_uiIndexCount;
	FLMUINT				m_uiRflFileNum;
	FLMUINT64			m_ui64RflBytesRead;
};

/****************************************************************************
Desc:
*****************************************************************************/
class F_LocalBackupClient : public F_DefaultBackupClient
{
public:

	F_LocalBackupClient(
		FlmShell *	pShell,
		char *		pszBackupPath) : F_DefaultBackupClient( pszBackupPath)
	{
		m_pShell = pShell;
	}
private:

	FlmShell *	m_pShell;
};

/****************************************************************************
Desc:
*****************************************************************************/
class F_LocalBackupStatus : public IF_BackupStatus
{
public:

	F_LocalBackupStatus(
		FlmShell *	pShell)
	{
		m_pShell = pShell;
	}

	RCODE XFLAPI backupStatus(
		FLMUINT64		ui64BytesToDo,
		FLMUINT64		ui64BytesDone);

	FINLINE FLMINT XFLAPI getRefCount( void)
	{
		return( IF_BackupStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLAPI AddRef( void)
	{
		return( IF_BackupStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLAPI Release( void)
	{
		return( IF_BackupStatus::Release());
	}

private:

	FlmShell *	m_pShell;
};

/****************************************************************************
Desc:
*****************************************************************************/
FlmShell::FlmShell( void) : FlmThreadContext()
{
	m_histPool.poolInit( 512);
	m_argPool.poolInit( 512);
	
	f_memset( m_DbList, 0, MAX_SHELL_OPEN_DB * sizeof( IF_Db *));
	m_pTitleWin = NULL;
	m_iCurrArgC = 0;
	m_ppCurrArgV = NULL;
	m_iLastCmdExitCode = 0;
	m_bPagingEnabled = FALSE;
	m_pSharedContext = NULL;
	f_memset( m_ppCmdList, 0, sizeof( FlmCommand *) * MAX_REGISTERED_COMMANDS);
	f_memset( m_ppHistory, 0, sizeof( char *) * MAX_SHELL_HISTORY_ITEMS);
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmShell::~FlmShell( void)
{
	FLMUINT		uiLoop;

	m_histPool.poolFree();
	m_argPool.poolFree();

	// Free the command objects.

	for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
	{
		if( m_ppCmdList[ uiLoop] != NULL)
		{
			m_ppCmdList[ uiLoop]->Release();
		}
	}

	// Free the history items

	for( uiLoop = 0; uiLoop < MAX_SHELL_HISTORY_ITEMS; uiLoop++)
	{
		if( m_ppHistory[ uiLoop])
		{
			f_free( &m_ppHistory[ uiLoop]);
		}
	}

	// Close all open databases

	for( uiLoop = 0; uiLoop < MAX_SHELL_OPEN_DB; uiLoop++)
	{
		if( m_DbList[ uiLoop])
		{
			m_DbList[ uiLoop]->Release();
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::setup(
	FlmSharedContext *		pSharedContext)
{
	FlmCommand *	pCommand = NULL;
	RCODE				rc = NE_XFLM_OK;

	flmAssert( pSharedContext != NULL);

	m_pSharedContext = pSharedContext;

	if( RC_BAD( rc = FlmThreadContext::setup( m_pSharedContext,
		"X-FLAIM Shell", NULL, NULL)))
	{
		goto Exit;
	}

	// Register dbopen command

	if( (pCommand = f_new FlmDbOpenCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register dbclose command

	if( (pCommand = f_new FlmDbCloseCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register dbcopy, dbrename, and dbremove command handler

	if( (pCommand = f_new FlmDbManageCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register trans command

	if( (pCommand = f_new FlmTransCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register backup command

	if( (pCommand = f_new FlmBackupCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register restore command

	if( (pCommand = f_new FlmRestoreCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register database config command

	if( (pCommand = f_new FlmDbConfigCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register database get config command

	if( (pCommand = f_new FlmDbGetConfigCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register sysinfo command

	if( (pCommand = f_new FlmSysInfoCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the hex conversion command

	if( (pCommand = f_new FlmHexConvertCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the base64 conversion command

	if( (pCommand = f_new FlmBase64ConvertCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;


	// Register the file delete command

	if( (pCommand = f_new FlmFileSysCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the import command

	if( (pCommand = f_new FlmImportCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the dom edit command

	if( (pCommand = f_new FlmDomEditCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the export command

	if( (pCommand = f_new FlmExportCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;
	
	// Register the wrap key command

	if( (pCommand = f_new FlmWrapKeyCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register nodeinfo command

	if( (pCommand = f_new FlmNodeInfoCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register btreeinfo command

	if( (pCommand = f_new FlmBTreeInfoCommand) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;
	
Exit:

	if( pCommand)
	{
		pCommand->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::registerDatabase(
	IF_Db *			pDb,
	FLMUINT *		puiDbId)
{
	FLMUINT		uiLoop;
	RCODE			rc = NE_XFLM_OK;

	*puiDbId = 0xFFFFFFFF;

	for( uiLoop = 0; uiLoop < MAX_SHELL_OPEN_DB; uiLoop++)
	{
		if( !m_DbList[ uiLoop])
		{
			m_DbList[ uiLoop] = pDb;
			*puiDbId = uiLoop;
			break;
		}
	}

	if( *puiDbId == 0xFFFFFFFF)
	{
		rc = RC_SET( NE_XFLM_TOO_MANY_OPEN_DATABASES);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::getDatabase(
	FLMUINT			uiDbId,
	IF_Db **			ppDb)
{
	RCODE			rc = NE_XFLM_OK;

	if( uiDbId >= MAX_SHELL_OPEN_DB)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	*ppDb = m_DbList[ uiDbId];

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::deregisterDatabase(
	FLMUINT			uiDbId)
{
	RCODE			rc = NE_XFLM_OK;

	if( uiDbId >= MAX_SHELL_OPEN_DB)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	m_DbList[ uiDbId] = NULL;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::con_printf(
	const char *	pszFormat, ...)
{
	char				szBuffer[ 4096];
	f_va_list		args;

	if( m_pWindow)
	{
		f_va_start( args, pszFormat);
		f_vsprintf( szBuffer, pszFormat, &args);
		f_va_end( args);
		FTXWinPrintStr( m_pWindow, szBuffer);
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::parseCmdLine(
	char *		pszString)
{
	FLMUINT		uiArgCount = 0;
	FLMUINT		uiCurrToken = 0;
	FLMUINT		uiTokenLen;
	char *		pszCurrToken;
	FLMBOOL		bQuoted;
	FlmParse		Parser;
	RCODE			rc = NE_XFLM_OK;

	m_argPool.poolReset( NULL);
	m_iCurrArgC = 0;
	m_ppCurrArgV = NULL;
	m_pszOutputFile = NULL;

	Parser.setString( pszString);
	while( Parser.getNextToken())
	{
		uiArgCount++;
	}

	if (RC_BAD( rc = m_argPool.poolCalloc( uiArgCount * sizeof( char *),
		(void **)&m_ppCurrArgV)))
	{
		goto Exit;
	}

	uiCurrToken = 0;
	Parser.setString( pszString);
	while( (pszCurrToken = Parser.getNextToken()) != NULL)
	{
		bQuoted = FALSE;
		if( *pszCurrToken == '\"')
		{
			// Skip the quote character
			pszCurrToken++;
			bQuoted = TRUE;
		}

		uiTokenLen = f_strlen( pszCurrToken);
		if (!bQuoted && uiTokenLen >= 2 && *pszCurrToken == '>' && !m_pszOutputFile)
		{
			if (RC_BAD( rc = m_argPool.poolCalloc( uiTokenLen,
				(void **)&m_pszOutputFile)))
			{
				goto Exit;
			}
			f_strcpy( m_pszOutputFile, pszCurrToken + 1);
		}
		else
		{
			if (RC_BAD( rc = m_argPool.poolCalloc( uiTokenLen + 1,
				(void **)&m_ppCurrArgV [uiCurrToken])))
			{
				goto Exit;
			}

			f_strcpy( m_ppCurrArgV[ uiCurrToken], pszCurrToken);

			if( bQuoted)
			{
				// Strip off the trailing quote
				m_ppCurrArgV[ uiCurrToken][ uiTokenLen - 1] = '\0';
			}
			uiCurrToken++;
		}
	}

	m_iCurrArgC = (FLMINT)uiCurrToken;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::executeCmdLine( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bValidCommand = FALSE;
	FLMUINT					uiLoop;
	IF_BufferIStream *	pBufIStream = NULL;

	if( !m_iCurrArgC)
	{
		goto Exit;
	}

	// Process internal commands

	if( f_stricmp( m_ppCurrArgV[ 0], "cls") == 0)
	{
		FTXWinClear( m_pWindow);
		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "exit") == 0)
	{
		setShutdownFlag();
		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "echo") == 0)
	{
		FLMBOOL		bNewline = FALSE;

		if( m_iCurrArgC > 1 &&
			f_stricmp( m_ppCurrArgV[ 1], "-n") == 0)
		{
			bNewline = TRUE;
			uiLoop = 2;
		}
		else
		{
			uiLoop = 1;
		}

		for( ; uiLoop < (FLMUINT)m_iCurrArgC; uiLoop++)
		{
			con_printf( "%s", (char *)m_ppCurrArgV[ uiLoop]);
		}

		if( bNewline)
		{
			con_printf( "\n");
		}

		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "shell") == 0)
	{
		FlmShell	*	pTmpShell;

		if( (pTmpShell = f_new FlmShell) != NULL)
		{
			if( RC_BAD( pTmpShell->setup( m_pSharedContext)))
			{
				pTmpShell->Release();
			}
			else
			{
				m_pSharedContext->spawn( pTmpShell);
			}
		}
		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "qp") == 0)
	{
		F_XPath					xpathObj;
		F_Query					query;
		RCODE						tmpRc;

		bValidCommand = TRUE;
		if( m_iCurrArgC != 3)
		{
			con_printf( "Invalid number of arguments.\n\n");
		}
		else
		{
			FLMUINT		uiDbId;
			F_Db *		pDb;

			uiDbId = f_atol( m_ppCurrArgV[ 1]);
			if( RC_BAD( rc = getDatabase( uiDbId, (IF_Db **)&pDb)))
			{
				con_printf( "Invalid database ID.\n\n");
				goto Exit;
			}
			
			if( !pBufIStream)
			{
				if( RC_BAD( rc = FlmAllocBufferIStream( &pBufIStream)))
				{
					goto Exit;
				}
			}
			
			pBufIStream->openStream( m_ppCurrArgV[ 2], 
							f_strlen( m_ppCurrArgV[ 2]));
			tmpRc = xpathObj.parseQuery( pDb, pBufIStream, &query);
			pBufIStream->closeStream();
			
			con_printf( "Result: 0x%08X\n\n", (unsigned)tmpRc);
		}
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "meta") == 0)
	{
		FLMUINT				uiMeta;
		FLMUINT				uiAltMeta;

		bValidCommand = TRUE;
		if( m_iCurrArgC != 2)
		{
			con_printf( "Invalid number of arguments.\n\n");
		}
		else
		{
			if( !pBufIStream)
			{
				if( RC_BAD( rc = FlmAllocBufferIStream( &pBufIStream)))
				{
					goto MetaExit;
				}
			}
			
			pBufIStream->openStream( m_ppCurrArgV[ 1], f_strlen( m_ppCurrArgV[ 1]));
			for( ;;)
			{
				RCODE	tmpRc;
				
				if( RC_BAD( tmpRc = f_getNextMetaphone( pBufIStream, 
					&uiMeta, &uiAltMeta)))
				{
					if( tmpRc != NE_XFLM_EOF_HIT)
					{
						con_printf( "Error: 0x%04X\n", tmpRc);
					}
					break;
				}
				con_printf( "Meta = 0x%04X, AltMeta = 0x%04X\n",
					uiMeta, uiAltMeta);
			}

MetaExit:

			if( pBufIStream)
			{
				pBufIStream->closeStream();
			}
		}
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "help") == 0 ||
				f_stricmp( m_ppCurrArgV[ 0], "?") == 0 ||
				f_stricmp( m_ppCurrArgV[ 0], "h") == 0)
	{
		if( m_iCurrArgC < 2)
		{
			con_printf( "Commands:\n");
			displayCommand( "help, ?, h", "Show help");
			displayCommand( "shell", "Start a new shell");
			displayCommand( "echo", "Echo typed in command");
			displayCommand( "cls", "Clear screen");
			displayCommand( "exit", "Exit shell");
			displayCommand( "echo", "Echo typed in command");
			for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
			{
				if( m_ppCmdList[ uiLoop] != NULL)
				{
					m_ppCmdList[ uiLoop]->displayHelp( this, NULL);
				}
			}
		}
		else
		{
			for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
			{
				if( m_ppCmdList[ uiLoop] != NULL)
				{
					if (m_ppCmdList[ uiLoop]->canPerformCommand(
								(char *)m_ppCurrArgV [1]))
					{
						m_ppCmdList[ uiLoop]->displayHelp( this,
								(char *)m_ppCurrArgV [1]);
						break;
					}
				}
			}
		}
		bValidCommand = TRUE;
	}
	else
	{
		for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
		{
			if( m_ppCmdList[ uiLoop] != NULL)
			{
				if( m_ppCmdList[ uiLoop]->canPerformCommand(
										(char *)m_ppCurrArgV[ 0]))
				{
					m_ppCmdList[ uiLoop]->execute( m_iCurrArgC, m_ppCurrArgV, this);
					bValidCommand = TRUE;
					break;
				}
			}
		}
	}

	if( !bValidCommand)
	{
		FTXWinPrintf( m_pWindow, "Unrecognized command: %s\n", m_ppCurrArgV[ 0]);
		rc = RC_SET( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

Exit:

	if( pBufIStream)
	{
		pBufIStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::registerCmd(
	FlmCommand *	pCmd)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bRegistered = FALSE;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
	{
		if( m_ppCmdList[ uiLoop] == NULL)
		{
			m_ppCmdList[ uiLoop] = pCmd;
			bRegistered = TRUE;
			break;
		}
	}

	if( !bRegistered)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::addCmdHistory(
	char *	pszCmd)
{
	FLMUINT		uiLoop;
	FLMUINT		uiSlot;
	FLMUINT		uiCmdLen;
	RCODE			rc = NE_XFLM_OK;

	// If the command line is too long, don't store it in the
	// history buffer

	if( (uiCmdLen = f_strlen( pszCmd)) > MAX_CMD_LINE_LEN)
	{
		goto Exit;
	}

	// Look for a duplicate history item

	for( uiLoop = 0; uiLoop < MAX_SHELL_HISTORY_ITEMS; uiLoop++)
	{
		if( m_ppHistory[ uiLoop] &&
			f_strcmp( pszCmd, m_ppHistory[ uiLoop]) == 0)
		{
			// Remove the command from the history list and compress
			// the history table

			f_free( &m_ppHistory[ uiLoop]);

			if( uiLoop < MAX_SHELL_HISTORY_ITEMS - 1)
			{
				f_memmove( &m_ppHistory[ uiLoop], &m_ppHistory[ uiLoop + 1],
					sizeof( char *) * (MAX_SHELL_HISTORY_ITEMS - uiLoop - 1));
				m_ppHistory[ MAX_SHELL_HISTORY_ITEMS - 1] = NULL;
				break;
			}
		}
	}

	// Find an empty slot for the new history item

	for( uiSlot = MAX_SHELL_HISTORY_ITEMS; uiSlot > 0; uiSlot--)
	{
		if( m_ppHistory[ uiSlot - 1])
		{
			break;
		}
	}

	if( uiSlot == MAX_SHELL_HISTORY_ITEMS)
	{
		f_free( &m_ppHistory[ 0]);
		f_memmove( &m_ppHistory[ 0], &m_ppHistory[ 1],
			sizeof( char *) * (MAX_SHELL_HISTORY_ITEMS - 1));
		m_ppHistory[ MAX_SHELL_HISTORY_ITEMS - 1] = NULL;
		uiSlot = MAX_SHELL_HISTORY_ITEMS - 1;
	}

	if( RC_BAD( rc = f_alloc( uiCmdLen + 1, &m_ppHistory[ uiSlot])))
	{
		goto Exit;
	}

	f_strcpy( m_ppHistory[ uiSlot], pszCmd);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::execute( void)
{
	char						szBuffer[ MAX_CMD_LINE_LEN + 1];
	char						szThreadName[ MAX_THREAD_NAME_LEN + 1];
	FLMUINT					uiTermChar;
	FLMUINT					uiRow;
	FLMUINT					uiLastHistorySlot = MAX_SHELL_HISTORY_ITEMS;
	RCODE						rc = NE_XFLM_OK;
	char						szDir [F_PATH_MAX_SIZE];
	DirectoryIterator		directoryIterator;
	char *					pszTabCompleteBegin = NULL;
	IF_PosIStream *		pFileIStream = NULL;


	if( RC_BAD( rc = FTXScreenInit( "xshell main", &m_pScreen)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXScreenInitStandardWindows( m_pScreen, 
		FLM_RED, FLM_WHITE, FLM_BLUE, FLM_WHITE, FALSE, FALSE, 
		NULL, &m_pTitleWin, &m_pWindow)))
	{
		goto Exit;
	}

	FTXScreenDisplay( m_pScreen);
	FTXScreenSetShutdownFlag( m_pScreen, getShutdownFlagAddr());

	szBuffer[ 0] = '\0';
	for( ;;)
	{
		// Refresh the title bar
		getName( szThreadName);
		FTXWinSetCursorPos( m_pTitleWin, 0, 0);
		FTXWinPrintf( m_pTitleWin, "%5.5u: %s",
			(unsigned)getID(), szThreadName);
		FTXWinClearToEOL( m_pTitleWin);

		// Check for shutdown
		if( getShutdownFlag())
		{
			break;
		}

		FTXWinGetCursorPos( m_pWindow, NULL, &uiRow);
		FTXWinSetCursorPos( m_pWindow, 0, uiRow);
		FTXWinClearToEOL( m_pWindow);

		if( RC_BAD( f_getcwd( szDir)))
		{
			szDir [0] = '\0';
		}

		FTXWinPrintf( m_pWindow, "%s>", szDir);

		if( FTXLineEdit( m_pWindow, szBuffer,
			MAX_CMD_LINE_LEN, 255, 0, &uiTermChar))
		{
			break;
		}

		if( uiTermChar == FKB_TAB)
		{
			char	szBase[ 255];
			char	szWildcard[ 255];

			szWildcard[0] = '\0';

			pszTabCompleteBegin = positionToPath( szBuffer);

			if ( f_strchr( pszTabCompleteBegin, '\"'))
			{
				// remove quotes
				removeChars( pszTabCompleteBegin, '\"');
			}

			// If we have not initialized our iterator to scan this directory
			// or if the command-line does not contain a path that we provided
			// we need to reinitialize the iterator.

			if( !directoryIterator.isInitialized() || 
				 !pszTabCompleteBegin || 
				 !directoryIterator.isInSet( pszTabCompleteBegin))
			{

				extractBaseDirAndWildcard( pszTabCompleteBegin, szBase, szWildcard);

				directoryIterator.reset();
				directoryIterator.setupForSearch( szDir, szBase, szWildcard);
			}

			if ( !directoryIterator.isEmpty())
			{
				// Copy in the next entry along with its full path.

				directoryIterator.next( pszTabCompleteBegin, TRUE);

			}
			else
			{
				FTXBeep();
			}

			// If the completed path contains spaces, quote it
			if ( f_strchr( pszTabCompleteBegin, ASCII_SPACE))
			{
				f_memmove( pszTabCompleteBegin + 1, pszTabCompleteBegin, 
					f_strlen( pszTabCompleteBegin) + 1);
				pszTabCompleteBegin[0] = '\"';

				f_strcat( pszTabCompleteBegin, "\"");
			}
			continue;
		}

		directoryIterator.reset();

		if( uiTermChar == FKB_UP)
		{
			for(; uiLastHistorySlot > 0; uiLastHistorySlot--)
			{
				if( m_ppHistory[ uiLastHistorySlot - 1])
				{
					f_strcpy( szBuffer, m_ppHistory[ uiLastHistorySlot - 1]);
					uiLastHistorySlot--;
					break;
				}
			}

			continue;
		}

		if( uiTermChar == FKB_DOWN)
		{
			for(; uiLastHistorySlot < MAX_SHELL_HISTORY_ITEMS - 1; uiLastHistorySlot++)
			{
				if( m_ppHistory[ uiLastHistorySlot + 1])
				{
					f_strcpy( szBuffer, m_ppHistory[ uiLastHistorySlot + 1]);
					uiLastHistorySlot++;
					break;
				}
			}
			continue;
		}

		if( uiTermChar == FKB_ESCAPE)
		{
			szBuffer[ 0] = '\0';
			continue;
		}

		if( uiTermChar == FKB_F1)
		{
			FLMUINT				uiLen;

			addCmdHistory( szBuffer);

			if( RC_BAD( rc = FlmOpenFileIStream( szBuffer, &pFileIStream)))
			{
				goto BatchError;
			}
			
			for( ;;)
			{
				FTXWinPrintf( m_pWindow, "\n");

				uiLen = MAX_CMD_LINE_LEN;
				if( RC_BAD( rc = flmReadLine( 
					pFileIStream, (FLMBYTE *)szBuffer, &uiLen)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						if( !uiLen)
						{
							break;
						}
					}
					goto BatchError;
				}

				parseCmdLine( szBuffer);
				szBuffer[ 0] = '\0';

				if( RC_BAD( rc = executeCmdLine()))
				{
					goto BatchError;
				}
			}

BatchError:

			if( RC_BAD( rc))
			{
				FTXWinPrintf( m_pWindow, 
					"Error: %e.  Batch execution halted.\n", rc);
				rc = NE_XFLM_OK;
			}
			continue;
		}

		uiLastHistorySlot = MAX_SHELL_HISTORY_ITEMS;

		if( szBuffer [0])
		{
			FTXWinPrintf( m_pWindow, "\n");
			addCmdHistory( szBuffer);
			parseCmdLine( szBuffer);
			executeCmdLine();
			szBuffer[0] = '\0';

			continue;
		}

		FTXWinPrintf( m_pWindow, "\n");
	}

Exit:

	if( pFileIStream)
	{
		pFileIStream->Release();
	}

	if( m_pWindow)
	{
		FTXWinFree( &m_pWindow);
	}

	if( m_pScreen)
	{
		FTXScreenFree( &m_pScreen);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmParse::FlmParse( void)
{
	m_szString [0] = 0;
	m_pszCurPos = &m_szString [0];
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmParse::setString(
	char *		pszString)
{
	if( pszString)
	{
		f_strcpy( m_szString, pszString);
	}
	else
	{
		m_szString  [0]= 0;
	}

	m_pszCurPos = &m_szString [0];
}

/****************************************************************************
Desc:
*****************************************************************************/
char * FlmParse::getNextToken( void)
{
	char *		pszTokenPos = &m_szToken [0];
	FLMBOOL		bQuoted = FALSE;

	while( *m_pszCurPos && *m_pszCurPos == ' ')
	{
		m_pszCurPos++;
	}

	if( *m_pszCurPos == '$')
	{
		*pszTokenPos++ = *m_pszCurPos++;
		while( *m_pszCurPos)
		{
			if( (*m_pszCurPos >= 'A' && *m_pszCurPos <= 'Z') ||
				(*m_pszCurPos >= 'a' && *m_pszCurPos <= 'z') ||
				(*m_pszCurPos >= '0' && *m_pszCurPos <= '9') ||
				(*m_pszCurPos == '_'))
			{
				*pszTokenPos++ = *m_pszCurPos++;
			}
			else
			{
				break;
			}
		}
	}
	else if( *m_pszCurPos == '=')
	{
		*pszTokenPos++ = *m_pszCurPos++;
	}
	else
	{
		while( *m_pszCurPos && (*m_pszCurPos != ' ' || bQuoted))
		{
			if( *m_pszCurPos == '\"')
			{
				*pszTokenPos++ = *m_pszCurPos++;
				if( bQuoted)
				{
					break;
				}
				else
				{
					bQuoted = TRUE;
				}
			}
			else
			{
				*pszTokenPos++ = *m_pszCurPos++;
			}
		}
	}

	*pszTokenPos = '\0';

	if( m_szToken [0] == 0)
	{
		return( NULL);
	}

	return( &m_szToken [0]);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbOpenCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	IF_Db *			pDb = NULL;
	FLMUINT			uiDbId;
	RCODE				rc = NE_XFLM_OK;
	char *			pszRflDir = NULL;
	char *			pszPassword = NULL;
	char *			pszAllowLimited;
	FLMBOOL			bAllowLimited = FALSE;
	IF_DbSystem *				pDbSystem = NULL;
	
	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( iArgC >= 3)
	{
		pszRflDir = ppszArgV[ 2];
	}
	
	if (iArgC >=4)
	{
		pszPassword = ppszArgV[ 3];
	}

	if (iArgC >=5)
	{
		pszAllowLimited = ppszArgV[ 4];
		
		if (f_strnicmp( pszAllowLimited, "TRUE", 4) == 0)
		{
			bAllowLimited = TRUE;
		}
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDbSystem->dbOpen( ppszArgV[ 1],
		NULL, pszRflDir, pszPassword, bAllowLimited, &pDb)))
	{
		if( rc != NE_FLM_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}

		if( RC_BAD( rc = pDbSystem->dbCreate( 
			ppszArgV[ 1], NULL, pszRflDir, NULL, NULL, NULL, &pDb)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pShell->registerDatabase( pDb, &uiDbId)))
	{
		goto Exit;
	}
	pDb = NULL;

	pShell->con_printf( "Database #%u opened.\n", (unsigned)uiDbId);

Exit:

	if( pDb)
	{
		pDb->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "Error opening database: %e\n", rc);
		iExitCode = -1;
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbOpenCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbopen", "Open a database");
	}
	else
	{
		pShell->con_printf("Usage:\n"
								 "  dbopen <DbFileName> [<RflPath> [<Password> [<AllowLimited>]]]\n");
		pShell->con_printf("  <AllowLimited> : TRUE | FALSE \n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbOpenCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbopen", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbCloseCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iExitCode = 0;
	FLMUINT			uiDbId;
	IF_Db *			pDb = NULL;
	IF_DbSystem *	pDbSystem = NULL;

	if( iArgC != 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( !f_stricmp( ppszArgV[ 1], "kill"))
	{
		pDbSystem->deactivateOpenDb( NULL, NULL);
		pShell->con_printf( "All handles killed, but not necessarily closed.\n");
	}
	else if( !f_stricmp( ppszArgV[ 1], "all"))
	{
		for( uiDbId = 0; uiDbId < MAX_SHELL_OPEN_DB; uiDbId++)
		{
			if( RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
			{
				goto Exit;
			}

			if( pDb)
			{
				if( RC_BAD( rc = pShell->deregisterDatabase( uiDbId)))
				{
					goto Exit;
				}
				pDb->Release();
				pShell->con_printf( "Database #%u closed.\n", (unsigned)uiDbId);
			}
		}

		pDbSystem->closeUnusedFiles( 0);
	}
	else
	{
		uiDbId = f_atol( ppszArgV[ 1]);
		if( RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
		{
			goto Exit;
		}

		if( pDb)
		{
			if( RC_BAD( rc = pShell->deregisterDatabase( uiDbId)))
			{
				goto Exit;
			}
			pDb->Release();
			pShell->con_printf( "Database #%u closed.\n", (unsigned)uiDbId);
		}
		else
		{
			pShell->con_printf( "Database #%u already closed.\n", (unsigned)uiDbId);
		}
	}

Exit:

	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "Error closing database: %e\n", rc);
		iExitCode = -1;
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbCloseCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbclose", "Close a database");
	}
	else
	{
		pShell->con_printf("Usage:\n"
								 "  dbclose <db# | ALL>\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbCloseCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbclose", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmTransCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	FLMUINT			uiDbId;
	FLMUINT			uiTimeout;
	eDbTransType		eTransType;
	IF_Db *			pDb;
	RCODE				rc = NE_XFLM_OK;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle

	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &pDb)) || !pDb)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}

	eTransType = ((F_Db *)pDb)->getTransType();
	if( f_stricmp( ppszArgV [0], "trbegin") == 0)
	{
		if( iArgC < 3)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( eTransType != XFLM_NO_TRANS)
		{
			pShell->con_printf( "%s transaction is already active on database %u.\n",
				(char *)(eTransType == XFLM_READ_TRANS
							? "A read"
							: "An update"), (unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( !f_stricmp( ppszArgV[ 2], "read"))
		{
			if( RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS)))
			{
				goto Exit;
			}
		}
		else if( !f_stricmp( ppszArgV[ 2], "update"))
		{
			uiTimeout = 10;
			if( iArgC > 4)
			{
				uiTimeout = f_atol( ppszArgV[ 3]);
			}

			if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS, uiTimeout)))
			{
				goto Exit;
			}
		}
		else
		{
			pShell->con_printf( "Invalid parameter: %s\n", ppszArgV[ 3]);
			iExitCode = -1;
			goto Exit;
		}

		pShell->con_printf( "Transaction on %u started.\n", (unsigned)uiDbId);
	}
	else if( f_stricmp( ppszArgV[ 0], "trcommit") == 0)
	{
		if( eTransType == XFLM_NO_TRANS)
		{
			pShell->con_printf( "There is no active transaction on database %u.\n",
				(unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
		pShell->con_printf( "Transaction committed on database %u.\n",
			(unsigned)uiDbId);
	}
	else if( f_stricmp( ppszArgV[ 0], "trabort") == 0)
	{
		if( eTransType == XFLM_NO_TRANS)
		{
			pShell->con_printf( "There is no active transaction on database %u.\n",
				(unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = pDb->transAbort()))
		{
			goto Exit;
		}
		pShell->con_printf( "Transaction aborted on database %u.\n",
			(unsigned)uiDbId);
	}
	else
	{
		// should never be able to get here!
		flmAssert( 0);
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmTransCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "trbegin", "Begin a transaction");
		pShell->displayCommand( "trcommit", "Commit a transaction");
		pShell->displayCommand( "trabort", "Abort a transaction");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( pszCommand, "trbegin") == 0)
		{
			pShell->con_printf( "  trbegin db# [read | update <timeout>]\n");
		}
		else
		{
			pShell->con_printf( "  %s db#\n", pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmTransCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "trbegin", pszCommand) == 0 ||
				f_stricmp( "trcommit", pszCommand) == 0 ||
				f_stricmp( "trabort", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Status class for reporting progress of database copy
*****************************************************************************/
class FSHELL_CopyStatus : public IF_DbCopyStatus
{
public:
	FSHELL_CopyStatus(
		FlmShell *	pShell)
	{
		m_pShell = pShell;
		m_pWin = m_pShell->getWindow();
	}

	virtual ~FSHELL_CopyStatus()
	{
	}

	RCODE XFLAPI dbCopyStatus(
		FLMUINT64		ui64BytesToCopy,
		FLMUINT64		ui64BytesCopied,
		FLMBOOL			bNewSrcFile,
		const char *	pszSrcFileName,
		const char *	pszDestFileName)
	{
		RCODE	rc = NE_XFLM_OK;

		if (bNewSrcFile)
		{
			FTXWinPrintf( m_pWin, "\nCopying %s to %s ...\n",
					pszSrcFileName, pszDestFileName);
		}

		if( m_pShell->getShutdownFlag())
		{
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}

		FTXWinPrintf( m_pWin, "  %,I64u of %,I64u bytes copied\r",
						ui64BytesCopied, ui64BytesToCopy);
		f_yieldCPU();

		if( RC_OK( FTXWinTestKB( m_pWin)))
		{
			FLMUINT	uiChar;

			FTXWinInputChar( m_pWin, &uiChar);
			if (uiChar == FKB_ESC)
			{
				rc = RC_SET( NE_XFLM_USER_ABORT);
				goto Exit;
			}
		}

	Exit:

		return( rc);
	}

private:

	FlmShell *		m_pShell;
	FTX_WINDOW *	m_pWin;
};

/****************************************************************************
Desc: Status class for reporting progress of database rename
*****************************************************************************/
class FSHELL_RenameStatus : public IF_DbRenameStatus
{
public:
	FSHELL_RenameStatus(
		FlmShell *	pShell)
	{
		m_pShell = pShell;
	}

	virtual ~FSHELL_RenameStatus()
	{
	}

	FINLINE RCODE XFLAPI dbRenameStatus(
		const char *	pszSrcFileName,
		const char *	pszDstFileName)
	{
		m_pShell->con_printf( "Renaming %s to %s ...\n", pszSrcFileName,
				pszDstFileName);
		return( NE_XFLM_OK);
	}

private:
	FlmShell *	m_pShell;
};

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbManageCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iExitCode = 0;
	IF_DbSystem *	pDbSystem = NULL;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if (f_stricmp( ppszArgV [0], "dbremove") == 0)
	{
		if (RC_BAD( rc = pDbSystem->dbRemove( ppszArgV[ 1], NULL, NULL, TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		if( iArgC < 3)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}
		if (f_stricmp( ppszArgV [0], "dbcopy") == 0)
		{
			FSHELL_CopyStatus	copyStatus( pShell);

			if (RC_BAD( rc = pDbSystem->dbCopy( ppszArgV [1], NULL, NULL,
										ppszArgV [2], NULL, NULL, &copyStatus)))
			{
				goto Exit;
			}
			pShell->con_printf( "\n\n");
		}
		else
		{
			FSHELL_RenameStatus	renameStatus( pShell);

			if (RC_BAD( rc = pDbSystem->dbRename( ppszArgV [1], NULL, NULL,
										ppszArgV [2], TRUE, &renameStatus)))
			{
				goto Exit;
			}
			pShell->con_printf( "\n\n");
		}
	}

Exit:

	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbManageCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbcopy", "Copy a database");
		pShell->displayCommand( "dbrename", "Rename a database");
		pShell->displayCommand( "dbremove", "Delete a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( pszCommand, "dbremove") == 0)
		{
			pShell->con_printf( "  dbremove <DbFileName>\n");
		}
		else
		{
			pShell->con_printf( "  %s <SrcDbFileName> <DestDbFileName>\n",
				pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbManageCommand::canPerformCommand(
	char *		pszCommand )
{
	return( (f_stricmp( "dbcopy", pszCommand) == 0 ||
				f_stricmp( "dbrename", pszCommand) == 0 ||
				f_stricmp( "dbremove", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmBackupCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT					uiDbId;
	FLMUINT					uiIncSeqNum;
	IF_Db *					pDb;
	IF_Backup *				pBackup = NULL;
	IF_BackupClient *		pBackupClient = NULL;
	IF_BackupStatus *		pBackupStatus = NULL;
	FLMINT					iExitCode = 0;
	eDbBackupType			eBackupType = XFLM_FULL_BACKUP;
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bUsePasswd = FALSE;

	if( iArgC < 3)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}
	
	if (iArgC > 3)
	{
		bUsePasswd = TRUE;
	}
	
	if( iArgC > 4)
	{
		if( f_strnicmp( ppszArgV[ 3], "inc", 3) == 0)
		{
			eBackupType = XFLM_INCREMENTAL_BACKUP;
		}
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &pDb)) || !pDb)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = pDb->backupBegin( eBackupType,
		XFLM_READ_TRANS, 0, &pBackup)))
	{
		goto Exit;
	}

	if( (pBackupClient = f_new F_LocalBackupClient( 
		pShell, ppszArgV[ 2])) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( (pBackupStatus = f_new F_LocalBackupStatus( pShell)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pBackup->backup( 
			ppszArgV[ 2],
			bUsePasswd?ppszArgV[3]:NULL,
			pBackupClient, pBackupStatus, &uiIncSeqNum)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBackup->endBackup()))
	{
		goto Exit;
	}

	pShell->con_printf( "\nBackup complete.\n");

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	if( pBackup)
	{
		pBackup->Release();
	}

	if( pBackupClient)
	{
		pBackupClient->Release();
	}

	if( pBackupStatus)
	{
		pBackupStatus->Release();
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmBackupCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbbackup", "Backup a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <database_path> <backup_name> [<password> [\"INC\"]]\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmBackupCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbbackup", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalBackupStatus::backupStatus(
		FLMUINT64		ui64BytesToDo,
		FLMUINT64		ui64BytesDone)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	if( m_pShell->getShutdownFlag())
	{
		rc = RC_SET( NE_XFLM_USER_ABORT);
		goto Exit;
	}

	FTXWinPrintf( pWin, "%,I64u / %,I64u bytes backed up\r",
					  ui64BytesDone, ui64BytesToDo);
	f_yieldCPU();

	if( pWin && RC_OK( FTXWinTestKB( pWin)))
	{
		FLMUINT	uiChar;

		FTXWinInputChar( pWin, &uiChar);
		if (uiChar == FKB_ESC)
		{
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/************************************************************************
If we want status info for backups, we need to implement an IF_BackupClient.....

RCODE flmBackupProgFunc(
	FLMUINT		uiStatusType,
	void *		Parm1,
	void *		Parm2,
	void *		UserData)
{
	RCODE							rc = NE_XFLM_OK;
	FLMUINT64					ui64BytesDone;
	FLMUINT64					ui64BytesToDo;
	FlmShell *					pShell = (FlmShell *)UserData;
	FTX_WINDOW *				pWin = pShell->getWindow();

	F_UNREFERENCED_PARM( Parm2);
	F_UNREFERENCED_PARM( UserData);

	if( pShell->getShutdownFlag())
	{
		rc = RC_SET( NE_XFLM_USER_ABORT);
		goto Exit;
	}

	if( uiStatusType == FLM_DB_BACKUP_STATUS)
	{
		pDbBackupInfo = (DB_BACKUP_INFO *)Parm1;
		FTXWinPrintf( pWin, "%,I64u / %,I64u bytes backed up\r",
						  ui64BytesDone, ui64BytesToDo);
	}

	f_yieldCPU();

	if( pWin && FTXWinTestKB( pWin) == FTXRC_SUCCESS)
	{
		FLMUINT	uiChar;

		FTXWinInputChar( pWin, &uiChar);
		if (uiChar == FKB_ESC)
		{
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}
***********************************************************************/

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmRestoreCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	char *						pszRflDir = NULL;
	FLMINT						iExitCode = 0;
	F_LocalRestore *			pRestore = NULL;
	F_LocalRestoreStatus 	restoreStatus( pShell);
	RCODE							rc = NE_XFLM_OK;
	FLMBOOL						bUsePasswd = FALSE;
	IF_DbSystem *				pDbSystem = NULL;

	if( iArgC < 3)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}
	
	if( iArgC > 3)
	{
		bUsePasswd = TRUE;
	}
	
	if( iArgC > 4)
	{
		pszRflDir = ppszArgV[ 4];
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( (pRestore = f_new F_LocalRestore) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pRestore->setup( 
		ppszArgV[ 1], ppszArgV[ 2], pszRflDir)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDbSystem->dbRestore( ppszArgV[ 1], NULL, NULL, NULL,
						 bUsePasswd?ppszArgV[3]:NULL, pRestore, &restoreStatus)))
	{
		goto Exit;
	}

	pShell->con_printf( "\nRestore complete.\n");

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	if( pRestore)
	{
		pRestore->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmRestoreCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbrestore", "Restore a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <RestoreToDbName> <BackupPath> [<password> [<RFL Dir>]]\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmRestoreCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbrestore", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:  The report* functions are all status callbacks that allow XFlaim to 
		 pass some information back out
*****************************************************************************/
void F_LocalRestoreStatus::updateCountDisplay( void)
{
	FTX_WINDOW *	pWin = m_pShell->getWindow();

	if (pWin)
	{
		FTXWinSetCursorPos( pWin, 0, 2);
		FTXWinPrintf( pWin,
			"RFLFile#: %-10u   TotalCnt: %-10u  RflKBytes: %uK\n"
			"AddCnt:   %-10u   DelCnt:   %-10u  ModCnt:    %u\n"
			"TrCnt:    %-10u   RsrvCnt:  %-10u  IxSetCnt:  %u",
			m_uiRflFileNum,
			m_uiTransCount + m_uiAddCount + m_uiDeleteCount +
			m_uiModifyCount + m_uiReserveCount + m_uiIndexCount,
			(unsigned)(m_ui64RflBytesRead / 1024),
			m_uiAddCount, m_uiDeleteCount, m_uiModifyCount,
			m_uiTransCount, m_uiReserveCount, m_uiIndexCount);
	}
}

// Contains some common code that all of the report* functions call after
// their primary processing...
RCODE F_LocalRestoreStatus::report_preamble(
		FTX_WINDOW *	pWin)
{
	RCODE	rc = NE_XFLM_OK;

	if( m_pShell->getShutdownFlag())
	{
		rc = RC_SET( NE_XFLM_USER_ABORT);
		goto Exit;
	}

	if( m_bFirstStatus)
	{
		FTXWinClear( pWin);
		m_bFirstStatus = FALSE;
	}
Exit:
	return rc;
}

// Contains some common code that all of the report* functions call after
// their primary processing...
RCODE F_LocalRestoreStatus::report_postamble(
	FTX_WINDOW *	pWin)
{
	RCODE		rc = NE_XFLM_OK;

	FTXWinSetCursorPos( pWin, 0, 5);

	f_yieldCPU();

	if( pWin && RC_OK( FTXWinTestKB( pWin)))
	{
		FLMUINT	uiChar;

		FTXWinInputChar( pWin, &uiChar);
		if (uiChar == FKB_ESC)
		{
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}
	}
Exit:
	return rc;

}

RCODE F_LocalRestoreStatus::reportProgress(
	eRestoreAction *	peAction,
	FLMUINT64			ui64BytesToDo,
	FLMUINT64			ui64BytesDone)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 1);
	FTXWinPrintf( pWin, "%,I64u / %,I64u bytes restored", ui64BytesDone,
		ui64BytesToDo);
	FTXWinClearToEOL( pWin);

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:

	return rc;
}

RCODE F_LocalRestoreStatus::reportError(
	eRestoreAction *	peAction,
	RCODE					rcErr)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();
	FLMUINT						uiChar;

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 6);
	FTXWinClearToEOL( pWin);
	FTXWinPrintf( pWin, "Error: 0x%04X.  Retry (Y/N): ",
		(unsigned)rcErr);
		
	if( RC_BAD( FTXWinInputChar( pWin, &uiChar)))
	{
		uiChar = 0;
		goto Exit;
	}

	if( uiChar == 'Y' || uiChar == 'y')
	{
		*peAction = XFLM_RESTORE_ACTION_RETRY;
	}

	FTXWinClearToEOL( pWin);

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;

}

RCODE F_LocalRestoreStatus::reportBeginTrans(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 5);
	FTXWinPrintf( pWin, "BEGIN_TRANS: ID = 0x%I64X", ui64TransId);
	FTXWinClearToEOL( pWin);

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;
}

RCODE F_LocalRestoreStatus::reportCommitTrans(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 5);
	FTXWinPrintf( pWin, "COMMIT_TRANS: ID = 0x%I64X", ui64TransId);
	FTXWinClearToEOL( pWin);
	m_uiTransCount++;
	updateCountDisplay();

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;
}

RCODE F_LocalRestoreStatus::reportAbortTrans(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 5);
	FTXWinPrintf( pWin, "ABORT_TRANS: ID = 0x%I64X", ui64TransId);
	FTXWinClearToEOL( pWin);
	m_uiTransCount++;
	updateCountDisplay();

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;
}


RCODE F_LocalRestoreStatus::reportEnableEncryption(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 5);
	FTXWinPrintf( pWin, "ENABLE_ENCRYPTION: ID = 0x%I64X", ui64TransId);
	FTXWinClearToEOL( pWin);
	m_uiTransCount++;
	updateCountDisplay();

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;
}


RCODE F_LocalRestoreStatus::reportWrapKey(
	eRestoreAction *	peAction,
	FLMUINT64			ui64TransId)
{
	RCODE							rc = NE_XFLM_OK;
	FTX_WINDOW *				pWin = m_pShell->getWindow();

	*peAction = XFLM_RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble( pWin)))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWin, 0, 5);
	FTXWinPrintf( pWin, "WRAP_KEY: ID = 0x%I64X", ui64TransId);
	FTXWinClearToEOL( pWin);
	m_uiTransCount++;
	updateCountDisplay();

	if (RC_BAD(rc = report_postamble( pWin)))
	{
		goto Exit;
	}

Exit:
	return rc;
}


/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbConfigCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT			uiDbId;
	IF_Db *			pDb;
	FLMINT			iExitCode = 0;
	RCODE				rc = NE_XFLM_OK;

	if( iArgC < 3)
	{
		pShell->con_printf( "Too few parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &pDb)) || !pDb)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}
	if( f_stricmp( ppszArgV[ 2], "rflkeepfiles") == 0)
	{
		FLMBOOL	bEnable;

		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if (f_stricmp( ppszArgV[ 3], "on") == 0)
		{
			bEnable = TRUE;
		}
		else if (f_stricmp( ppszArgV[ 3], "off") == 0)
		{
			bEnable = FALSE;
		}
		else
		{
			pShell->con_printf( "Invalid value, must be 'on' or 'off'.\n");
			iExitCode = -1;
			goto Exit;
		}
		if( RC_BAD( rc = ((F_Db *)pDb)->setRflKeepFilesFlag( bEnable)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rfldir") == 0)
	{
		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = ((F_Db *)pDb)->setRflDir( ppszArgV[ 3])))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rflfilelimits") == 0)
	{
		FLMUINT	uiRflMinSize;
		FLMUINT	uiRflMaxSize;

		if( iArgC < 5)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		uiRflMinSize = f_atol( ppszArgV[ 3]);
		uiRflMaxSize = f_atol( ppszArgV[ 4]);
		if( RC_BAD( rc = ((F_Db *)pDb)->setRflFileSizeLimits(
			uiRflMinSize, uiRflMaxSize)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rflrolltonextfile") == 0)
	{
		if( RC_BAD( rc = ((F_Db *)pDb)->rflRollToNextFile()))
		{
			goto Exit;
		}
	}
	else
	{
		pShell->con_printf( "Invalid option.\n");
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbConfigCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbconfig", "Configure a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <db#> rflkeepfiles <on|off>\n", pszCommand);
		pShell->con_printf( "  %s <db#> rfldir <DirectoryName>\n", pszCommand);
		pShell->con_printf( "  %s <db#> rflfilelimits <MinRflSize> <MaxRflSize>\n",
						pszCommand);
		pShell->con_printf( "  %s <db#> rolltonextfile\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbConfigCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbconfig", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC void format64BitNum(
	FLMUINT64	ui64Num,
	char *		pszBuf,
	FLMBOOL		bOutputHex,
	FLMBOOL		bAddCommas
	)
{
	char		szTmpBuf [60];
	FLMUINT	uiDigit;
	FLMUINT	uiChars = 0;
	FLMUINT	uiCharsBetweenCommas;

	if (bOutputHex)
	{
		while (ui64Num)
		{
			uiDigit = (FLMUINT)(ui64Num & 0xF);
			szTmpBuf [uiChars++] = (char)(uiDigit + '0');
			ui64Num >>= 4;
		}
	}
	else
	{
		uiCharsBetweenCommas = 0;
		while (ui64Num)
		{
			if (bAddCommas && uiCharsBetweenCommas == 3)
			{
				szTmpBuf [uiChars++] = ',';
				uiCharsBetweenCommas = 0;
			}
			uiDigit = (FLMUINT)(ui64Num % 10);
			szTmpBuf [uiChars++] = (char)(uiDigit + '0');
			ui64Num /= 10;
			uiCharsBetweenCommas++;
		}
	}

	// Need to reverse the numbers going back out.

	while (uiChars)
	{
		uiChars--;
		*pszBuf++ = szTmpBuf [uiChars];
	}
	*pszBuf = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbGetConfigCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT			uiDbId;
	IF_Db *			pIDb;
	F_Db *			pDb;
	FLMINT			iExitCode = 0;
	FLMUINT64		ui64Arg;
	FLMUINT			uiArg;
	FLMUINT			uiArg2;
	FLMBOOL			bArg;
	char				szTmpPath[ F_PATH_MAX_SIZE];
	char				ucBuf[ 256];
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bDoAll = FALSE;
	FLMBOOL			bValidOption = FALSE;

	if( iArgC < 3)
	{
		pShell->con_printf( "Too few parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &pIDb)) || !pIDb)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}
	pDb = (F_Db *)pIDb;
	if (f_stricmp( ppszArgV [2], "all") == 0)
	{
		bDoAll = TRUE;
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rfldir") == 0)
	{
		pDb->getRflDir( szTmpPath);
		pShell->con_printf( "RFL directory = %s\n", szTmpPath);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflfilenum") == 0)
	{
		pDb->getRflFileNum( &uiArg);
		pShell->con_printf( "Current RFL file # = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflsizelimits") == 0)
	{
		pDb->getRflFileSizeLimits( &uiArg, &uiArg2);
		pShell->con_printf( "RFL file size limits = min:%u, max:%u\n",
			(unsigned)uiArg, (unsigned)uiArg2);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "diskusage") == 0)
	{
		FLMUINT64	ui64DbSize;
		FLMUINT64	ui64RollbackSize;
		FLMUINT64	ui64RflSize;
		char			szBuf1 [40];
		char			szBuf2 [40];
		char			szBuf3 [40];

		if( RC_BAD( rc = pDb->getDiskSpaceUsage(
			&ui64DbSize, &ui64RollbackSize, &ui64RflSize)))
		{
			goto Exit;
		}

		format64BitNum( ui64DbSize, szBuf1, FALSE);
		format64BitNum( ui64RollbackSize, szBuf2, FALSE);
		format64BitNum( ui64RflSize, szBuf3, FALSE);
		pShell->con_printf( "Sizes = db:%s, rollback:%s, rfl:%s",
			szBuf1, szBuf2, szBuf3);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflkeepfiles") == 0)
	{
		if( RC_BAD( rc = pDb->getRflKeepFlag( &bArg)))
		{
			goto Exit;
		}

		pShell->con_printf( "Keep RFL files = %s\n",
			bArg ? "Yes" : "No");
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "lastbackuptransid") == 0)
	{
		if( RC_BAD( rc = pDb->getLastBackupTransID( &ui64Arg)))
		{
			goto Exit;
		}

		//VISIT: Use formatter for 64 bit unsigned
		pShell->con_printf( "Last backup transaction ID = %u\n",
			(unsigned)ui64Arg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "blockschangedsincebackup") == 0)
	{
		if( RC_BAD( rc = pDb->getBlocksChangedSinceBackup( &uiArg)))
		{
			goto Exit;
		}

		pShell->con_printf( "Blocks changed since last backup = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "serialnumber") == 0)
	{
		pDb->getSerialNumber( ucBuf);

		pShell->con_printf(
			"Serial number = "
			"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
			(unsigned)ucBuf[ 0],
			(unsigned)ucBuf[ 1],
			(unsigned)ucBuf[ 2],
			(unsigned)ucBuf[ 3],
			(unsigned)ucBuf[ 4],
			(unsigned)ucBuf[ 5],
			(unsigned)ucBuf[ 6],
			(unsigned)ucBuf[ 7],
			(unsigned)ucBuf[ 8],
			(unsigned)ucBuf[ 9],
			(unsigned)ucBuf[ 10],
			(unsigned)ucBuf[ 11],
			(unsigned)ucBuf[ 12],
			(unsigned)ucBuf[ 13],
			(unsigned)ucBuf[ 14],
			(unsigned)ucBuf[ 15]);
		bValidOption = TRUE;
	}

	if (!bValidOption)
	{
		pShell->con_printf( "Invalid option.\n");
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbGetConfigCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbgetconfig", "Display DB configuration");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <db#> <option>\n", pszCommand);
		pShell->con_printf( "  <option> may be one of the following:\n");
		pShell->con_printf( "    rflfilenum\n");
		pShell->con_printf( "    diskusage\n");
		pShell->con_printf( "    rflkeepfiles\n");
		pShell->con_printf( "    lastbackuptransid\n");
		pShell->con_printf( "    blockschangedsincebackup\n");
		pShell->con_printf( "    serialnumber\n");
		pShell->con_printf( "    all (will print all of the above)\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbGetConfigCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbgetconfig", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmSysInfoCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT				iExitCode = 0;
	RCODE					rc = NE_XFLM_OK;
	FTX_WINDOW *		pWin = pShell->getWindow();
	FLMUINT				uiLoop;
	IF_ThreadInfo *	pThreadInfo = NULL;
	IF_DbSystem *		pDbSystem = NULL;

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( iArgC < 2)
	{
		FLMUINT64			ui64TotalMem;
		FLMUINT64			ui64AvailMem;
	
		f_getMemoryInfo( &ui64TotalMem, &ui64AvailMem);
		
		pShell->con_printf( "Total Memory ............. %,I64u\n",
			ui64TotalMem);
		pShell->con_printf( "Available Memory ......... %,I64u\n",
			ui64AvailMem);
	}
	else
	{
		if( !f_stricmp( ppszArgV[ 1], "memtest"))
		{
			FLMUINT		uiBlockSize;
			FLMUINT		uiCount;
			FLMUINT		uiStartTime;
			FLMUINT		uiMilli;
			void *		pvHead = NULL;
			void *		pvAlloc = NULL;

			if( iArgC >= 4)
			{
				uiBlockSize = f_atol( ppszArgV[ 2]);
				if( uiBlockSize < sizeof( void *))
				{
					uiBlockSize = sizeof( void *);
				}

				uiCount = f_atol( ppszArgV[ 3]);
				if( uiCount < 1)
				{
					uiCount = 1;
				}

				uiStartTime = FLM_GET_TIMER();
				for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
				{
					if( RC_BAD( f_alloc( uiBlockSize, &pvAlloc)))
					{
						pShell->con_printf( "Unable to allocate block %u.\n",
							(unsigned)uiLoop);
						break;
					}

					if( !pvHead)
					{
						pvHead = pvAlloc;
						*((FLMUINT *)pvAlloc) = 0;
					}
					else
					{
						*((FLMUINT *)pvAlloc) = (FLMUINT)pvHead;
						pvHead = pvAlloc;
					}
					f_yieldCPU();
				}

				uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
				pShell->con_printf( "Allocations: %u ms, %,u count, %,u bytes\n",
					(unsigned)uiMilli, (unsigned)uiLoop,
					(unsigned)(uiCount * uiBlockSize));

				if( iArgC > 4 && !f_stricmp( ppszArgV[ 4], "pause"))
				{
					FTXDisplayMessage( pShell->getScreen(), FLM_BLUE, FLM_WHITE,
						"Press <ENTER> to continue ...",
						NULL, NULL);
				}

				uiStartTime = FLM_GET_TIMER();
				while( pvHead)
				{
					pvAlloc = pvHead;
					pvHead = (void *)(*((FLMUINT *)pvHead));
					f_free( &pvAlloc);
					f_yieldCPU();
				}
				uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
				pShell->con_printf( "Frees: %u ms, %u count, %u bytes\n",
					(unsigned)uiMilli, (unsigned)uiLoop,
					(unsigned)(uiLoop * uiBlockSize));
			}
			else
			{
				pShell->con_printf( "Wrong number of arguments.\n");
			}
		}
		else if( !f_stricmp( ppszArgV[ 1], "fstest"))
		{
			fshellFileSystemTest( iArgC + 1, ppszArgV + 1, pShell);
		}
		else if( !f_stricmp( ppszArgV[ 1], "guid"))
		{
			FLMBYTE	ucGuid[ XFLM_SERIAL_NUM_SIZE];

			if( RC_BAD( rc = f_createSerialNumber( ucGuid)))
			{
				goto Exit;
			}

			pShell->con_printf(
				"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
				(unsigned)ucGuid[ 0],
				(unsigned)ucGuid[ 1],
				(unsigned)ucGuid[ 2],
				(unsigned)ucGuid[ 3],
				(unsigned)ucGuid[ 4],
				(unsigned)ucGuid[ 5],
				(unsigned)ucGuid[ 6],
				(unsigned)ucGuid[ 7],
				(unsigned)ucGuid[ 8],
				(unsigned)ucGuid[ 9],
				(unsigned)ucGuid[ 10],
				(unsigned)ucGuid[ 11],
				(unsigned)ucGuid[ 12],
				(unsigned)ucGuid[ 13],
				(unsigned)ucGuid[ 14],
				(unsigned)ucGuid[ 15]);
		}
		else if( !f_stricmp( ppszArgV[ 1], "threads"))
		{
			FLMUINT		uiNumThreads;
			FLMUINT		uiCurrentTime;
			FLMUINT		uiRow;

			for( ;;)
			{
				if( RC_OK( FTXWinTestKB( pWin)))
				{
					break;
				}

				FTXWinSetCursorPos( pWin, 0, 0);

				if (pThreadInfo)
				{
					pThreadInfo->Release();
					pThreadInfo = NULL;
				}
				if( RC_BAD( rc = pDbSystem->getThreadInfo( &pThreadInfo)))
				{
					goto Exit;
				}
				uiNumThreads = pThreadInfo->getNumThreads();

				f_timeGetSeconds( &uiCurrentTime);
				for( uiLoop = 0; uiLoop < uiNumThreads; uiLoop++)
				{
					FLMUINT			uiThreadId;
					FLMUINT			uiThreadGroup;
					FLMUINT			uiAppId;
					FLMUINT			uiStartTime;
					const char *	pszThreadName;
					const char *	pszThreadStatus;

					pThreadInfo->getThreadInfo( uiLoop,
										&uiThreadId, &uiThreadGroup, &uiAppId,
										&uiStartTime, &pszThreadName,
										&pszThreadStatus);

					pShell->con_printf( "0x%08X 0x%08X (%-6u): 0x%08X %-20.20s %-15.15s",
						(unsigned)uiThreadId,
						(unsigned)uiThreadGroup,
						(unsigned)(uiCurrentTime - uiStartTime),
						(unsigned)uiAppId,
						pszThreadName
							? pszThreadName
							: "Unknown",
						pszThreadStatus
							? (char *)pszThreadStatus
							: "Unknown");
					FTXWinClearToEOL( pWin);
					pShell->con_printf( "\n");
				}

				uiRow = FTXWinGetCurrRow( pWin);
				FTXWinClearXY( pWin, 0, uiRow);
				f_sleep( 300);
			}
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	if (pThreadInfo)
	{
		pThreadInfo->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmSysInfoCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "sysinfo", "Display system information");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmSysInfoCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "sysinfo", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT fshellFileSystemTest(
	FLMINT				iArgC,
	char **				ppszArgV,
	FlmShell *			pShell)
{
	FLMINT						iExitCode = 0;
	RCODE							rc = NE_XFLM_OK;
	IF_FileHdl *				pFileHdl = NULL;
	FLMUINT						uiBlockSize = 4096;
	FLMUINT						uiFileSize = (1024 * 1024 * 100); // 100 MB
	FLMUINT						uiOffset = 0;
	FLMUINT						uiBytesWritten;
	FLMUINT						uiBytesRead;
	FLMUINT						uiStartTime;
	FLMUINT						uiMilli;
	FLMUINT						uiTotal;
	FLMUINT						uiCount;
	FLMBYTE *					pucBuf = NULL;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of arguments.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( uiBlockSize, &pucBuf)))
	{
		goto Exit;
	}

	f_memset( pucBuf, 0xFF, uiBlockSize);
	
	if( RC_OK( f_getFileSysPtr()->doesFileExist( ppszArgV[ 1])))
	{
		if( RC_BAD( rc = f_getFileSysPtr()->openFile( ppszArgV[ 1],
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &pFileHdl)))
		{
			goto Exit;
		}
		
	}
	else
	{
		pShell->con_printf( "Creating %s\n", ppszArgV[ 1]);

		uiStartTime = FLM_GET_TIMER();
		if( RC_BAD( rc = f_getFileSysPtr()->createFile( ppszArgV[ 1],
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
			&pFileHdl)))
		{
			goto Exit;
		}
		
		uiOffset = 0;
		while( uiOffset < uiFileSize)
		{
			if( RC_BAD( rc = pFileHdl->write( uiOffset, uiBlockSize,
				pucBuf, &uiBytesWritten)))
			{
				goto Exit;
			}

			uiOffset += uiBytesWritten;

			uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
			pShell->con_printf( "%u / %u (%u bytes/sec)          \r",
				(unsigned)uiOffset, (unsigned)uiFileSize,
				(unsigned)uiMilli > 1000 ? (uiOffset / (uiMilli / 1000)) : 0);
			if( pShell->getShutdownFlag())
			{
				rc = RC_SET( NE_XFLM_USER_ABORT);
				goto Exit;
			}
		}
		uiFileSize = uiOffset;

		pShell->con_printf( "\nFile created.\n");
	}

	pShell->con_printf( "\nRandom writes ...\n");
	uiCount = (uiFileSize / uiBlockSize);
	uiTotal = 0;
	uiStartTime = FLM_GET_TIMER();
	while( uiCount)
	{
		uiOffset = (FLMUINT)((f_getRandomUINT32( 1,
			(FLMUINT32)uiCount) - 1) * uiBlockSize);
		if( RC_BAD( rc = pFileHdl->write( uiOffset, uiBlockSize,
			pucBuf, &uiBytesWritten)))
		{
			goto Exit;
		}

		uiCount--;
		uiTotal += uiBytesWritten;

		uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
		pShell->con_printf( "%u / %u (%u bytes/sec)          \r",
			(unsigned)uiTotal, (unsigned)uiFileSize,
			(unsigned)uiMilli > 1000 ? (uiTotal / (uiMilli / 1000)) : 0);
	}

	pShell->con_printf( "\nFinished random writes.\n");

	pShell->con_printf( "\nSequential scan ...\n");
	uiOffset = 0;
	uiStartTime = FLM_GET_TIMER();
	while( uiOffset < uiFileSize)
	{
		if( RC_BAD( rc = pFileHdl->read( uiOffset, uiBlockSize,
			pucBuf, &uiBytesRead)))
		{
			goto Exit;
		}

		uiOffset += uiBytesRead;
		uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
		pShell->con_printf( "%u / %u (%u bytes/sec)          \r",
			(unsigned)uiOffset, (unsigned)uiFileSize,
			(unsigned)uiMilli > 1000 ? (uiOffset / (uiMilli / 1000)) : 0);
		if( pShell->getShutdownFlag())
		{
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}
	}

	pShell->con_printf( "\nFinished sequential scan.\n");

	pShell->con_printf( "\nRandom scan ...\n");
	uiCount = (uiFileSize / uiBlockSize);
	uiTotal = 0;
	uiStartTime = FLM_GET_TIMER();
	while( uiCount)
	{
		uiOffset = (FLMUINT)((f_getRandomUINT32( 1, (FLMUINT32)uiCount)
			- 1) * uiBlockSize);
		if( RC_BAD( rc = pFileHdl->read( uiOffset, uiBlockSize,
			pucBuf, &uiBytesRead)))
		{
			goto Exit;
		}

		uiCount--;
		uiTotal += uiBytesRead;

		uiMilli = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER() - uiStartTime);
		pShell->con_printf( "%u / %u (%u bytes/sec)          \r",
			(unsigned)uiTotal, (unsigned)uiFileSize,
			(unsigned)uiMilli > 1000 ? (uiTotal / (uiMilli / 1000)) : 0);
	}

	pShell->con_printf( "\nFinished random scan.\n");

	if( RC_BAD( rc = pFileHdl->closeFile()))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	if( pucBuf)
	{
		f_free( &pucBuf);
	}

	if( !iExitCode && RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		iExitCode = -1;
	}

	return( iExitCode);
}

/****************************************************************************
Desc:	Constructor
*****************************************************************************/
FlmDbContext::FlmDbContext()
{
	f_memset( m_DbContexts, 0, sizeof( m_DbContexts));
	m_uiCurrDbId = 0;
}

/****************************************************************************
Desc:	Destructor
*****************************************************************************/
FlmDbContext::~FlmDbContext( void)
{
	FLMUINT			uiDbId;

	for( uiDbId = 0; uiDbId < MAX_DBCONTEXT_OPEN_DB &&
			 m_DbContexts [uiDbId].pDb; uiDbId++)
	{
		m_DbContexts[ uiDbId].pDb->Release();
	}
}


/****************************************************************************
Desc:	Get an available database ID.
*****************************************************************************/
FLMBOOL FlmDbContext::getAvailDbId(
	FLMUINT *	puiDbId
	)
{
	FLMUINT	uiDbId = 0;

	while (uiDbId < MAX_DBCONTEXT_OPEN_DB && m_DbContexts [uiDbId].pDb)
	{
		uiDbId++;
	}
	*puiDbId = uiDbId;
	return( (FLMBOOL)((uiDbId < MAX_DBCONTEXT_OPEN_DB) ? TRUE : FALSE));
}

/****************************************************************************
Desc:	Set the database handle for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setDb(
	FLMUINT		uiDbId,
	IF_Db *		pDb)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].pDb = pDb;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the database handle for a database ID
*****************************************************************************/
IF_Db * FlmDbContext::getDb(
	FLMUINT	uiDbId)
{
	return( (uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].pDb
						: NULL);
}

/****************************************************************************
Desc:	Set the current container for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrCollection(
	FLMUINT	uiDbId,
	FLMUINT	uiCollection)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrCollection = uiCollection;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current container for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrCollection(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrCollection
						: (FLMUINT)0));
}

/****************************************************************************
Desc:	Set the current index for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrIndex(
	FLMUINT	uiDbId,
	FLMUINT	uiIndex)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrIndex = uiIndex;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current index for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrIndex(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrIndex
						: (FLMUINT)0));
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrId(
	FLMUINT	uiDbId,
	FLMUINT	uiId)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrId = uiId;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMUINT FlmDbContext::getCurrId(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrId
						: (FLMUINT)0));
}

/****************************************************************************
Desc:	Set the current search flagsfor a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrSearchFlags(
	FLMUINT	uiDbId,
	FLMUINT	uiSearchFlags)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrSearchFlags = uiSearchFlags;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current search flags for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrSearchFlags(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrSearchFlags
						: (FLMUINT)XFLM_INCL));
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmFileSysCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiLoop = 0;
	FLMBOOL			bForce = FALSE;
	FLMBOOL			bOverwrite;
	IF_DirHdl *		pDir = NULL;
	FLMUINT64		ui64BytesCopied;

	// Delete the file

	if (f_stricmp( ppszArgV [0], "delete") == 0 ||
		 f_stricmp( ppszArgV [0], "del") == 0 ||
		 f_stricmp( ppszArgV [0], "rm") == 0)
	{
		for ( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if( ppszArgV[uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s.\n", ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		if( ( iArgC - uiLoop) < 1)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( bForce)
		{
			f_getFileSysPtr()->setReadOnly( 
				ppszArgV[uiLoop], FALSE);	
		}

		if( RC_BAD( rc = f_getFileSysPtr()->deleteFile( ppszArgV[ uiLoop])))
		{
			goto Exit;
		}
		pShell->con_printf( "\nFile deleted.\n");
	}
	else if (f_stricmp( ppszArgV [0], "cd") == 0 ||
				f_stricmp( ppszArgV [0], "chdir") == 0)
	{
		if (iArgC > 1)
		{
			if( RC_BAD( rc = f_chdir( (const char *)ppszArgV [1])))
			{
				pShell->con_printf( "Error changing directory\n");
			}
		}
	}

	else if(f_stricmp( "rename", ppszArgV[0]) == 0 ||
			f_stricmp( "move", ppszArgV[0]) == 0 ||
			f_stricmp( "mv", ppszArgV[0]) == 0 )
	{
		bForce = FALSE;
		bOverwrite = FALSE;

		for( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if ( ppszArgV[ uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				case 'o':
				case 'O':
					bOverwrite = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s\n", &ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		// The remaining two parameters are the source and destination

		if ( ( iArgC - uiLoop) < 2)
		{
			pShell->con_printf( "You must specify a source and destination.\n");
			iExitCode = -1;
			goto Exit;
		}

		if ( RC_BAD( rc = f_getFileSysPtr()->doesFileExist( ppszArgV[uiLoop])))
		{
			goto Exit;
		}

		if ( f_getFileSysPtr()->isDir( ppszArgV[uiLoop + 1]))
		{
			char	szFilename[ F_FILENAME_SIZE];

			// If the second param is a directory we'll assume the user wants to
			// move the file into it with the same filename.

			f_getFileSysPtr()->pathReduce( ppszArgV[uiLoop], NULL, szFilename);
			f_getFileSysPtr()->pathAppend( szFilename, ppszArgV[uiLoop + 1]);
		}

		if( RC_OK( f_getFileSysPtr()->doesFileExist( ppszArgV[uiLoop + 1])))
		{
			if ( !bOverwrite)
			{
				FLMUINT	uiChar;

				pShell->con_printf( "%s exists. Overwrite? (Y/N)", ppszArgV[ uiLoop + 1]);
				for(;;)
				{
					if( RC_OK( FTXWinTestKB( pShell->getWindow())))
					{
						FTXWinInputChar( pShell->getWindow(), &uiChar);

						// Echo char back to the user
						pShell->con_printf( "%c\n", uiChar);

						if ( uiChar == 'Y' || uiChar == 'y')
						{
							bOverwrite = TRUE;
						}
						else
						{
							rc = RC_SET( NE_XFLM_USER_ABORT);
							goto Exit;
						}
						break;
					}
				}
			}
			if ( bOverwrite)
			{
				if ( bForce)
				{
					f_getFileSysPtr()->setReadOnly( ppszArgV[uiLoop + 1], FALSE);
					pShell->con_printf( "Error changing file attributes. ");
					goto Exit;
				}

				if ( RC_BAD( rc = f_getFileSysPtr()->deleteFile( 
					ppszArgV[uiLoop + 1])))
				{
					pShell->con_printf( "Error removing destination file. ");
					goto Exit;
				}
			}
		}

		if ( RC_BAD( rc = f_getFileSysPtr()->renameFile( ppszArgV[uiLoop], 
			ppszArgV[uiLoop + 1])))
		{
			goto Exit;
		}

		pShell->con_printf( "%s -> %s\n", 
			ppszArgV[uiLoop], ppszArgV[uiLoop + 1]);
	}
	else if(f_stricmp( "copy", ppszArgV[0]) == 0 ||
				f_stricmp( "cp", ppszArgV[0]) == 0)
	{
		bForce = FALSE;
		bOverwrite = FALSE;
		ui64BytesCopied = 0;

		for( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if ( ppszArgV[ uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				case 'o':
				case 'O':
					bOverwrite = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s\n", &ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		// The remaining two parameters are the source and destination

		if ( ( iArgC - uiLoop) < 2)
		{
			pShell->con_printf( "You must specify a source and destination.\n");
			iExitCode = -1;
			goto Exit;
		}

		if ( RC_BAD( rc = f_getFileSysPtr()->doesFileExist( ppszArgV[uiLoop])))
		{
			goto Exit;
		}

		if ( f_getFileSysPtr()->isDir( ppszArgV[uiLoop + 1]))
		{
			char	szFilename[ F_FILENAME_SIZE];

			// If the second param is a directory we'll assume the user wants to
			// copy the file into it with the same filename.

			f_getFileSysPtr()->pathReduce( ppszArgV[uiLoop], NULL, szFilename);
			f_getFileSysPtr()->pathAppend( szFilename, ppszArgV[uiLoop + 1]);
		}

		if ( RC_OK( f_getFileSysPtr()->doesFileExist( ppszArgV[uiLoop + 1])))
		{
			if ( !bOverwrite)
			{
				FLMUINT	uiChar;

				pShell->con_printf( "%s exists. Overwrite? (Y/N)", ppszArgV[ uiLoop + 1]);
				for(;;)
				{
					if( RC_OK( FTXWinTestKB( pShell->getWindow())))
					{
						FTXWinInputChar( pShell->getWindow(), &uiChar);

						// Echo char back to the user
						pShell->con_printf( "%c\n", uiChar);

						if ( uiChar == 'Y' || uiChar == 'y')
						{
							bOverwrite = TRUE;
						}
						else
						{
							rc = RC_SET( NE_XFLM_USER_ABORT);
							goto Exit;
						}
						break;
					}
				}
			}

			if ( bOverwrite)
			{

				// There's no sense in changing a file's attributes if we aren't
				// going to overwrite it.

				if ( bForce)
				{
					f_getFileSysPtr()->setReadOnly( 
						ppszArgV[uiLoop + 1], FALSE);
				}
			}
		}

		if ( RC_BAD( rc = f_getFileSysPtr()->copyFile( 
			ppszArgV[uiLoop], ppszArgV[uiLoop +1], bOverwrite, &ui64BytesCopied)))
		{
			goto Exit;
		}

		pShell->con_printf( "%s copied to %s (%I64u bytes copied)\n", 
			ppszArgV[uiLoop], ppszArgV[uiLoop + 1], ui64BytesCopied);
	}
	else if (f_stricmp( "ls", ppszArgV [0]) == 0 ||
				f_stricmp( "dir", ppszArgV [0]) == 0)
	{
		char				szDir [F_PATH_MAX_SIZE];
		char				szBaseName [F_FILENAME_SIZE];
		FLMUINT			uiLineCount;
		FTX_WINDOW *	pWindow = pShell->getWindow();
		FLMUINT			uiMaxLines;
		FLMUINT			uiNumCols;
		FLMUINT			uiChar;

		FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiMaxLines);
		uiMaxLines--;

		if( iArgC > 1)
		{
			if (RC_BAD( rc = f_getFileSysPtr()->pathReduce( 
				ppszArgV [1], szDir, szBaseName)))
			{
				goto Exit;
			}
			if (!szDir [0])
			{
				f_strcpy( szDir, ".");
			}
			
			if (RC_BAD( rc = f_getFileSysPtr()->openDir( szDir, 
				(char *)szBaseName, &pDir)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = f_getFileSysPtr()->openDir( ".", NULL, &pDir)))
			{
				goto Exit;
			}
		}
		pShell->con_printf( "%-20s %25s\n", "File Name", "File Size");
		uiLineCount = 1;
		for (;;)
		{
			if (RC_BAD( rc = pDir->next()))
			{
				if (rc == NE_FLM_IO_NO_MORE_FILES)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			if (uiLineCount == uiMaxLines)
			{
				pShell->con_printf(
					"...(more, press any character to continue, ESC to quit)");
				uiChar = 0;
				for (;;)
				{
					if( RC_OK( FTXWinTestKB( pWindow)))
					{
						FTXWinInputChar( pWindow, &uiChar);
						break;
					}
					if (pShell->getShutdownFlag())
					{
						uiChar = FKB_ESC;
						break;
					}
					f_yieldCPU();
				}
				if (uiChar == FKB_ESC)
				{
					break;
				}
				pShell->con_printf(
					"\r                                                       \r");
				uiLineCount = 0;
			}
			if (pDir->currentItemIsDir())
			{
				pShell->con_printf( "%-20s %25s\n", pDir->currentItemName(),
					"<DIR>");
			}
			else
			{
				char	szTmpBuf [60];

				format64BitNum( pDir->currentItemSize(), szTmpBuf, FALSE, TRUE);
				pShell->con_printf( "%-20s %25s\n", pDir->currentItemName(),
					szTmpBuf);
			}
			uiLineCount++;
		}
	}
	else
	{
		// Should never happen!
		flmAssert( 0);
	}

Exit:

	if( pDir)
	{
		pDir->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmFileSysCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "delete, del, rm", "Delete a file. Options: "
			"/O - Overwrite /F - Force");
		pShell->displayCommand( "cd (or chdir)", "Change directories");
		pShell->displayCommand( "dir, ls", "List files");
		pShell->displayCommand( "copy, cp", "Copy files. Options: "
			"/O - Overwrite /F - Force");
		pShell->displayCommand( "rename, move, mv", "Move a file. Options: "
			"/O - Overwrite /F - Force");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( "cd", pszCommand) == 0)
		{
			pShell->con_printf( "  %s <Directory>\n", pszCommand);
		}
		else if (f_stricmp( "ls", pszCommand) == 0 ||
					f_stricmp( "dir", pszCommand) == 0)
		{
			pShell->con_printf( "  %s [FileMask]\n", pszCommand);
		}
		else
		{
			pShell->con_printf( "  %s <filename>\n", pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmFileSysCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "delete", pszCommand) == 0 ||
				f_stricmp( "del", pszCommand) == 0 ||
				f_stricmp( "rm", pszCommand) == 0 ||
				f_stricmp( "ls", pszCommand) == 0 ||
				f_stricmp( "dir", pszCommand) == 0 ||
				f_stricmp( "cd", pszCommand) == 0 ||
				f_stricmp( "chdir", pszCommand) == 0 ||
				f_stricmp( "del", pszCommand) == 0 ||
				f_stricmp( "rm", pszCommand) == 0 ||
				f_stricmp( "copy", pszCommand) == 0 ||
				f_stricmp( "cp", pszCommand) == 0 ||
				f_stricmp( "move", pszCommand) == 0 ||
				f_stricmp( "mv", pszCommand) == 0 ||
				f_stricmp( "rename", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmHexConvertCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	IF_FileHdl *	pSrcFile = NULL;
	IF_FileHdl *	pDestFile = NULL;
	FLMBYTE			ucTmpBuf[ 64];
	char				szOutputBuf[ 64];
	char				szPreamble[ 64];
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;
	FLMUINT			uiSrcOffset;
	FLMUINT			uiDestOffset;
	FLMUINT			uiLineOffset;
	FLMINT			iExitCode = 0;
	RCODE				rc = NE_XFLM_OK;

	if( iArgC < 3)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Open the source file

	if( RC_BAD( rc = f_getFileSysPtr()->openFile( ppszArgV[ 1],
		FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &pSrcFile)))
	{
		goto Exit;
	}

	// Create the destination file

	if( RC_BAD( rc = f_getFileSysPtr()->createFile( ppszArgV[ 2],
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pDestFile)))
	{
		goto Exit;
	}

	uiSrcOffset = 0;
	uiDestOffset = 0;
	uiLineOffset = 0;

	for( ;;)
	{
		if( RC_BAD( rc = pSrcFile->read( uiSrcOffset, 1,
			ucTmpBuf, &uiBytesRead)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
		uiSrcOffset += uiBytesRead;

		szOutputBuf[ 0] = 0;
		szPreamble[ 0] = 0;

		if( uiLineOffset > 60)
		{
			uiLineOffset = 0;
			f_sprintf( szOutputBuf,",\n");

			if( RC_BAD( rc = pDestFile->write( uiDestOffset,
				f_strlen( szOutputBuf), (FLMBYTE *)szOutputBuf, &uiBytesWritten)))
			{
				goto Exit;
			}

			uiDestOffset += uiBytesWritten;
		}

		if( uiLineOffset)
		{
			f_sprintf( szPreamble, ", ");
		}

		f_sprintf( szOutputBuf,
			"%s0x%02X", szPreamble, (unsigned)ucTmpBuf[ 0]);

		if( RC_BAD( rc = pDestFile->write( uiDestOffset,
			f_strlen( szOutputBuf), (FLMBYTE *)szOutputBuf, &uiBytesWritten)))
		{
			goto Exit;
		}

		uiDestOffset += uiBytesWritten;
		uiLineOffset += uiBytesWritten;
	}

	pShell->con_printf( "\nConversion complete (Source size = %u).\n",
		(unsigned)uiSrcOffset);

Exit:

	if( pSrcFile)
	{
		pSrcFile->Release();
	}

	if( pDestFile)
	{
		pDestFile->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmHexConvertCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "hexconvert", "Convert file to HEX");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <SourceFile> <DestFile>\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmHexConvertCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "hexconvert", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmBase64ConvertCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	RCODE								rc = NE_XFLM_OK;
	FLMINT							iExitCode = 0;
	FLMBYTE							ucReadBuf[ 256];
	FLMUINT							uiBytesRead;
	FLMUINT							uiDestOffset = 0;
	FLMUINT							uiBytesWritten;
	IF_FileHdl *					pDestFile = NULL;
	IF_PosIStream *				pSrcIStream = NULL;
	IF_IStream *					pBase64Stream = NULL;

	if( iArgC < 2 || iArgC > 4)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Open the source file
	
	if( RC_BAD( rc = FlmOpenFileIStream( ppszArgV[ 1], &pSrcIStream)))
	{
		goto Exit;
	}

	if( iArgC == 4)
	{
		if( f_stricmp( ppszArgV[ 3], "-d") != 0)
		{
			pShell->con_printf( "Invalid parameter (%s).\n", ppszArgV[ 3]);
			iExitCode = -1;
			goto Exit;
		}
		
		if( RC_BAD( rc = FlmOpenBase64DecoderIStream( 
			pSrcIStream, &pBase64Stream)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = FlmOpenBase64EncoderIStream( 
			pSrcIStream, TRUE, &pBase64Stream)))
		{
			goto Exit;
		}
	}

	// Create the destination file

	if( iArgC >= 3)
	{
		if( RC_BAD( rc = f_getFileSysPtr()->createFile( ppszArgV[ 2],
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pDestFile)))
		{
			goto Exit;
		}

		uiDestOffset = 0;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pBase64Stream->read( 
			ucReadBuf, sizeof( ucReadBuf) - 1, &uiBytesRead)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;

			if( !uiBytesRead)
			{
				break;
			}
		}

		if( pDestFile)
		{
			if( RC_BAD( rc = pDestFile->write( uiDestOffset,
				uiBytesRead, ucReadBuf, &uiBytesWritten)))
			{
				goto Exit;
			}

			uiDestOffset += uiBytesWritten;
		}
	}

	pShell->con_printf( "Done.\n");

Exit:

	if( pBase64Stream)
	{
		pBase64Stream->Release();
	}

	if( pSrcIStream)
	{
		pSrcIStream->Release();
	}

	if( pDestFile)
	{
		pDestFile->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmBase64ConvertCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "base64", "Convert a file to base64");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <SourceFile> [<DestFile> [-d]]\n", pszCommand);
		pShell->con_printf("\n");
		pShell->con_printf("-d = Decode base64 source file\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmBase64ConvertCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "base64", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE flmXmlImportStatus(
	eXMLStatus		eStatusType,
	void *			pvArg1,
	void *			pvArg2,
	void *			pvArg3,
	void *			pvUserData)
{
	RCODE				rc = NE_XFLM_OK;
	FlmShell *		pShell = (FlmShell *)pvUserData;

	F_UNREFERENCED_PARM( pvArg2);
	F_UNREFERENCED_PARM( pvArg3);

	if( eStatusType == XML_STATS)
	{
		XFLM_IMPORT_STATS *		pStats = (XFLM_IMPORT_STATS *)pvArg1;

		pShell->con_printf( "Processed: Nodes = %u, Chars = %u\r",
			pStats->uiElements + pStats->uiAttributes + pStats->uiText,
			pStats->uiChars);
	}

// Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE	getErrFromFile(
	IF_PosIStream *	pFileIStream,
	FLMUINT				uiCurrLineFilePos,
	FLMUINT				uiCurrLineBytes,
	FLMUINT *			puiErrLineOffset,
	XMLEncoding			eXMLEncoding,
	char *				pszErrorString)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiBytesRead = 0;
	FLMBYTE *			pucSrcBuff = NULL;
	FLMBYTE *			pucDstBuff = NULL;
	const FLMBYTE *	pucSrcCur = NULL;
	FLMBYTE *			pucDstCur = NULL;
	FLMUNICODE 			uzChar = 0;
	FLMUINT				uiCountChars = 0;
	FLMUINT				uiErrStringBegin = 0;
	FLMUINT				uiErrStringEnd = 0;
	FLMUINT				uiErrLineOffsetSave = *puiErrLineOffset;
  
   if ( RC_BAD( rc = f_alloc( uiCurrLineBytes + 1, &pucSrcBuff)))
	{
		goto Exit;	
	}
	pucSrcCur = pucSrcBuff;
	if ( RC_BAD( rc = f_alloc( uiCurrLineBytes * 8, &pucDstBuff)))
	{
		goto Exit;	
	}
	pucDstCur = pucDstBuff;

	if ( RC_BAD( rc = pFileIStream->positionTo( uiCurrLineFilePos)))
	{
		goto Exit;
	}
	if ( RC_BAD( rc = pFileIStream->read( pucSrcBuff, 
		uiCurrLineBytes, &uiBytesRead)))
	{
		goto Exit;
	}
	pucSrcBuff[ uiBytesRead] = 0;


	if( eXMLEncoding == XFLM_XML_USASCII_ENCODING ||
				eXMLEncoding == XFLM_XML_UTF8_ENCODING)
	{
		for( ;;)
		{
			if( RC_BAD( rc = f_getCharFromUTF8Buf( &pucSrcCur, 
					pucSrcBuff + uiBytesRead,
					&uzChar)))
			{
				goto Exit;
			}

			if( uzChar <= 0x007F)
			{

				*pucDstCur++ = ( FLMBYTE)uzChar;
			}
			else
			{
				f_sprintf( ( char *)pucDstCur, "&#x%04x;", uzChar);
				pucDstCur += 8;
				if( uiErrLineOffsetSave > uiCountChars)
				{
					*puiErrLineOffset += 7;
				}
			}
			uiCountChars++;

			if( !uzChar)
			{
				break;
			}
		}

		// Set Begin and End for error string that will be returned		

		uiErrStringBegin = 0;
		uiErrStringEnd = pucDstCur - pucDstBuff;
		
		if( *puiErrLineOffset < ( MAX_IMPORT_ERROR_STRING / 2))
		{
			if( uiErrStringEnd > MAX_IMPORT_ERROR_STRING)
			{
            uiErrStringEnd = MAX_IMPORT_ERROR_STRING;
			}
		}
		else
		{
			if( *puiErrLineOffset + ( MAX_IMPORT_ERROR_STRING / 2) < uiErrStringEnd)
			{
				uiErrStringEnd = *puiErrLineOffset + ( MAX_IMPORT_ERROR_STRING / 2);
			}
			if( uiErrStringEnd > MAX_IMPORT_ERROR_STRING)
			{
				uiErrStringBegin = uiErrStringEnd - MAX_IMPORT_ERROR_STRING;
			}
		}
		*puiErrLineOffset -= uiErrStringBegin;
		

		f_memcpy( pszErrorString, &( pucDstBuff[ uiErrStringBegin]), 
				uiErrStringEnd - uiErrStringBegin);
	}
	else
	{

		// Encoding scheme not supported.  Error String will not be set.

		goto Exit;
	}

Exit:
	
	if( pucSrcBuff)
	{
		f_free( &pucSrcBuff);
	}
	if( pucDstBuff)
	{
		f_free( &pucDstBuff);
	}

	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE importXmlFiles(
	IF_Db *						pDb,
	FLMUINT						uiCollection,
	FlmShell *					pShell,
	char *						pszPath,
	XFLM_IMPORT_STATS *		pImportStats)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DirHdl *				pDirHdl = NULL;
	char						szTmpPath[ F_PATH_MAX_SIZE];
	char						szTmpPath2[ F_PATH_MAX_SIZE];
	char						szFile[ F_FILENAME_SIZE];
	FTX_WINDOW *			pWin = pShell->getWindow();
	FLMBOOL					bTransActive = FALSE;
	IF_DOMNode *			pRoot = NULL;
	IF_DOMNode *			pSource = NULL;
	F_Pool					pool;
	IF_PosIStream *		pFileIStream = NULL;
	FLMBOOL					bUseSafeMode = FALSE;
	char						szErrorString[ MAX_IMPORT_ERROR_STRING + 1];
	FLMUINT					uiIndentCount = 0;
	FLMUINT					uiNewErrLineOffset = 0;
	
	pool.poolInit( 256);

RetryLoad:

	f_memset( szErrorString, 0, sizeof( szErrorString));
	if( pDirHdl)
	{
		pDirHdl->Release();
		pDirHdl = NULL;
	}

	flmAssert( !bTransActive);
	pool.poolReset( NULL);

	if( f_getFileSysPtr()->isDir( pszPath))
	{
		if( RC_BAD( rc = f_getFileSysPtr()->openDir(
			pszPath, (char *)"*", &pDirHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		f_strcpy( szTmpPath, pszPath);
	}

	for( ;;)
	{
		pool.poolReset( NULL);
		if( pWin && RC_OK( FTXWinTestKB( pWin)))
		{
			FLMUINT	uiChar;

			FTXWinInputChar( pWin, &uiChar);
			if (uiChar == FKB_ESC)
			{
				rc = RC_SET( NE_XFLM_USER_ABORT);
				goto Exit;
			}
		}

		if( pDirHdl)
		{
			if( RC_BAD( rc = pDirHdl->next()))
			{
				if (rc == NE_FLM_IO_NO_MORE_FILES)
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

		if( pDirHdl)
		{
			pDirHdl->currentItemPath( szTmpPath);
			if( pDirHdl->currentItemIsDir())
			{
				if( bTransActive)
				{
					if( RC_BAD( rc = pDb->transCommit()))
					{
						goto Exit;
					}
					bTransActive = FALSE;
				}

				if( RC_BAD( rc = importXmlFiles( pDb, uiCollection, pShell,
					szTmpPath, pImportStats)))
				{
					goto Exit;
				}
				continue;
			}
		}

		if( !bTransActive && !bUseSafeMode)
		{
			if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS)))
			{
				goto Exit;
			}
			bTransActive = TRUE;
		}
		
		if (pFileIStream)
		{
			pFileIStream->Release();
			pFileIStream = NULL;
		}
		if( RC_BAD( rc = FlmOpenFileIStream( szTmpPath, &pFileIStream)))
		{
			goto Exit;
		}

		pShell->con_printf( "Importing %s ...\n", szTmpPath);

		for( ;;)
		{
			if( bUseSafeMode)
			{
				if( RC_BAD( rc = pDb->transBegin( XFLM_UPDATE_TRANS)))
				{
					goto Exit;
				}
				bTransActive = TRUE;
			}

			if( RC_BAD( rc = pDb->importDocument( 
				pFileIStream, uiCollection, &pRoot, pImportStats)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					pShell->con_printf( "Error importing: %s ... 0x%04X\n",
						szTmpPath, (unsigned)rc);

					if( pImportStats)
					{
						pShell->con_printf( "Line Number: %d\tOffset: %d\n",
							pImportStats->uiErrLineNum, pImportStats->uiErrLineOffset);
	
						if( pImportStats->eErrorType)
						{
							pShell->con_printf( "Import Error: %s\n",
								errorToString( pImportStats->eErrorType));
						}

						uiNewErrLineOffset = pImportStats->uiErrLineOffset;
						if( RC_OK( rc = getErrFromFile( pFileIStream,
														pImportStats->uiErrLineFilePos,
														pImportStats->uiErrLineBytes,
														&uiNewErrLineOffset,
														pImportStats->eXMLEncoding,
														szErrorString)))
						{
							pShell->con_printf( "%s\n", szErrorString);
							for( uiIndentCount = 0; 
										uiIndentCount < uiNewErrLineOffset;
										uiIndentCount++)
							{
								pShell->con_printf(" ");
							}
							pShell->con_printf("^\n");
						}

					}

					pDb->transAbort();
					bTransActive = FALSE;

					if( !bUseSafeMode)
					{
						bUseSafeMode = TRUE;
						goto RetryLoad;
					}
				}
				else if( bUseSafeMode)
				{
					pDb->transAbort();
					bTransActive = FALSE;
				}

				rc = NE_XFLM_OK;
				break;
			}

			f_getFileSysPtr()->pathReduce( szTmpPath, szTmpPath2, szFile);

			if( RC_BAD( rc = pRoot->createAttribute( pDb, ATTR_SOURCE_TAG, &pSource)))
			{
				if( rc == NE_XFLM_EXISTS)
				{
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = pSource->setUTF8( pDb, (FLMBYTE *)szFile)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = ((F_Db *)pDb)->documentDone( pRoot)))
			{
				goto Exit;
			}

			pRoot->Release();
			pRoot = NULL;

			if( pSource)
			{
				pSource->Release();
				pSource = NULL;
			}

			if( bUseSafeMode)
			{
				if( RC_BAD( rc = pDb->transCommit()))
				{
					goto Exit;
				}
				bTransActive = FALSE;
			}

			pShell->con_printf( "Documents  = %u\n", pImportStats->uiDocuments);
			pShell->con_printf( "Elements   = %u\n", pImportStats->uiElements);
			pShell->con_printf( "Attributes = %u\n", pImportStats->uiAttributes);
			pShell->con_printf( "Text Nodes = %u\n", pImportStats->uiText);
			pShell->con_printf( "Characters = %u\n", pImportStats->uiChars);
		}

		pShell->con_printf( "Import complete (%s).\n\n", szTmpPath);

		if( !pDirHdl)
		{
			break;
		}
	}

	if( bTransActive)
	{
		flmAssert( !bUseSafeMode);
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
		bTransActive = FALSE;
	}

Exit:

	if (pFileIStream)
	{
		pFileIStream->Release();
	}

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	if( pSource)
	{
		pSource->Release();
	}

	if( pRoot)
	{
		pRoot->Release();
	}

	if( bTransActive)
	{
		pDb->transAbort();
	}
	
	pool.poolFree();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmImportCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT					iExitCode = 0;
	FLMUINT					uiDbId;
	IF_Db *					pDb = NULL;
	FLMUINT					uiCollection = XFLM_DATA_COLLECTION;
	RCODE						rc = NE_XFLM_OK;
	XFLM_IMPORT_STATS	importStats;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if (iArgC >= 3)
	{
		uiDbId = f_atol( ppszArgV[ 2]);
	}
	else
	{
		uiDbId = 0;
	}
	if( RC_BAD( pShell->getDatabase( uiDbId, &pDb)) || !pDb)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}

	if (iArgC >= 4)
	{
		if (f_stricmp( ppszArgV [3], "Data") == 0 ||
			 f_stricmp( ppszArgV [3], "DefaultData") == 0)
		{
			uiCollection = XFLM_DATA_COLLECTION;
		}
		else if (f_stricmp( ppszArgV [3], "Dict") == 0 ||
					f_stricmp( ppszArgV [3], "Dictionary") == 0)
		{
			uiCollection = XFLM_DICT_COLLECTION;
		}
		else
		{
			uiCollection = f_atol( ppszArgV [3]);
			if (!uiCollection)
			{
				uiCollection = XFLM_DATA_COLLECTION;
			}
		}
	}

	f_memset( &importStats, 0, sizeof( XFLM_IMPORT_STATS));
	rc = importXmlFiles( pDb, uiCollection, pShell,
			ppszArgV [1], &importStats);

	pShell->con_printf( "\n\nDone.\n");

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmImportCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "import", "Import XML into a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s <filename or directory> [db#] [collection]\n"
			"  [db#] defaults to zero if omitted\n"
			"  [collection] defaults to Default Data Collection if omitted\n"
			"  It can also be one of the following:\n"
			"    Data - Default data collection\n"
			"    Dict - Dictionary collection\n"
			"    #    - Collection number\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmImportCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "import", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDomEditCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iExitCode = 0;
	IF_Db *			pDb = NULL;
	FLMUINT			uiDbId;
	F_DomEditor *	pDomEditor = NULL;
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	char				szTitle[ 80];
	FLMUINT			Cols;
	FLMUINT			Rows;
	IF_Db *			pNewDb = NULL;
	char *			pszRflDir = NULL;
	char *			pszPassword = NULL;
	char *			pszAllowLimited;
	FLMBOOL			bAllowLimited = FALSE;
	IF_DbSystem *	pDbSystem = NULL;

	if( iArgC < 1)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( iArgC >= 2)
	{
		char *	pszName = ppszArgV [1];

		while (*pszName)
		{
			if (*pszName < '0' || *pszName > '9')
			{
				break;
			}
			pszName++;
		}
		if (*pszName)
		{

			if( iArgC >= 3)
			{
				pszRflDir = ppszArgV[ 2];
			}
			
			if (iArgC >=4)
			{
				pszPassword = ppszArgV[ 3];
			}

			if (iArgC >=5)
			{
				pszAllowLimited = ppszArgV[ 4];
				
				if (f_strnicmp( pszAllowLimited, "TRUE", 4) == 0)
				{
					bAllowLimited = TRUE;
				}
			}

			if( RC_BAD( rc = pDbSystem->dbOpen( ppszArgV[ 1],
				NULL, pszRflDir, pszPassword, bAllowLimited, &pNewDb)))
			{
				pShell->con_printf( "Error opening database: %e.\n", rc);
				iExitCode = -1;
				goto Exit;
			}

			if( RC_BAD( rc = pShell->registerDatabase( pNewDb, &uiDbId)))
			{
				pShell->con_printf( "Error registering database: %e.\n", rc);
				iExitCode = -1;
				goto Exit;
			}
			pNewDb = NULL;

			pShell->con_printf( "Database #%u opened.\n", (unsigned)uiDbId);
		}
		else
		{
			uiDbId = f_atoi( ppszArgV[ 1]);
		}
	}
	else
	{
		uiDbId = 0;
	}

	if (RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
	{
		pShell->con_printf( "Error getting database handle: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	if (pDb == NULL)
	{
		pShell->con_printf( "Database %u not open.\n", uiDbId);
		iExitCode = -1;
		goto Exit;
	}

	f_sprintf( szTitle,
		"DOMEdit for XFLAIM [DB=%s/BUILD=%s]",
		XFLM_CURRENT_VER_STR, __DATE__);

	if( RC_BAD( FTXScreenInit( szTitle, &pScreen)))
	{
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( FTXWinInit( pScreen, 0, 1, &pTitleWin)))
	{
		iExitCode = -1;
		goto Exit;
	}

	FTXWinPaintBackground( pTitleWin, FLM_RED);
	FTXWinPrintStr( pTitleWin, szTitle);
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if ((pDomEditor = f_new F_DomEditor()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		pShell->con_printf( "Error allocating DOM editor object: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = pDomEditor->Setup( pScreen)))
	{
		pShell->con_printf( "Error setting up DOM editor object: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	pDomEditor->setSource( (F_Db *)pDb, XFLM_DATA_COLLECTION);
	pDomEditor->setShutdown( &gv_bShutdown);

	// Start up the editor

	FTXScreenGetSize( pScreen, &Cols, &Rows);

	FTXScreenDisplay( pScreen);

	pDomEditor->interactiveEdit( 0, 1, Cols - 1, Rows - 1);

Exit:

	if( pNewDb)
	{
		pNewDb->Release();
	}

	if( pDomEditor)
	{
		pDomEditor->Release();
		pDomEditor = NULL;
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	FTXScreenFree( &pScreen);
	return( iExitCode);
}

/****************************************************************************
Desc:	displayHelp - print a help message.
*****************************************************************************/
void FlmDomEditCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "edit", "Edit a database (DOM Editor)");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s [db#] (if db# is omitted, 0 is used)\n",
							  pszCommand);
		pShell->con_printf( "  OR\n");
		pShell->con_printf( "  %s <DbFileName> [<RflPath> [<Password> [<AllowLimited>]]]\n", pszCommand);
		pShell->con_printf( "      <AllowLimited> = TRUE | FALSE\n");
		pShell->con_printf( "      This form of the command will open the database before editing.\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDomEditCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "edit", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Executes an export command
*****************************************************************************/
FLMINT FlmExportCommand::execute(
	FLMINT iArgC,
	char ** ppszArgV,
	FlmShell * pShell)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DOMNode *			pDoc = NULL;
	IF_Db *					pDb = NULL;
	IF_OStream *			pFileOStream = NULL;	
	FLMINT					iExitCode = 0;
	FLMUINT					uiDbId = 0;
	FLMUINT					uiDocNum = 0;
	FLMUINT					uiCollection = XFLM_DATA_COLLECTION;
	FLMBOOL					bAllDocs = TRUE;
	eExportFormatType		eFormat = XFLM_EXPORT_INDENT;
	IF_DbSystem *			pDbSystem = NULL;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDbSystem->openFileOStream( ppszArgV[ 1], 
		TRUE, &pFileOStream)))
	{
		pShell->con_printf( "Unable to create file: %s.\n", ppszArgV[1]);
		iExitCode = -1;
		goto Exit;
	}

	if ( iArgC >= 3)
	{
		uiDbId = f_atoi( ppszArgV[ 2]);
	}
	else
	{
		uiDbId = 0;
	}
	if( RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
	{
		goto Exit;
	}

	if( !pDb)
	{
		pShell->con_printf( "Database %u not open.\n", uiDbId);
		iExitCode = -1;
		goto Exit;
	}

	if ( iArgC >= 4)
	{
		if ( f_stricmp( ppszArgV[ 3], "data") == 0)
		{
			uiCollection = XFLM_DATA_COLLECTION;
		}
		else if ( f_stricmp( ppszArgV[ 3], "dict") == 0)
		{
			uiCollection = XFLM_DICT_COLLECTION;
		}
		else
		{
			uiCollection = f_atoi( ppszArgV[ 3]);
		}
	}
	
	if ( iArgC >= 5)
	{
		if ( f_stricmp( ppszArgV[ 4], "all") == 0)
		{
			bAllDocs = TRUE;
		}
		else
		{
			uiDocNum = f_atoi( ppszArgV[ 4]);
			bAllDocs = FALSE;
		}
	}

	if ( iArgC >= 6)
	{
		if ( f_stricmp( ppszArgV[ 5], "none") == 0)
		{
			eFormat = XFLM_EXPORT_NO_FORMAT;
		}
		else if ( f_stricmp( ppszArgV[ 5], "newline") == 0)
		{
			eFormat = XFLM_EXPORT_NEW_LINE;
		}
		else if ( f_stricmp( ppszArgV[ 5], "indent") == 0)
		{
			eFormat = XFLM_EXPORT_INDENT;
		}
		else if ( f_stricmp( ppszArgV[ 5], "idata") == 0)
		{
			eFormat = XFLM_EXPORT_INDENT_DATA;
		}
		else
		{
			eFormat = ( eExportFormatType)f_atoi( ppszArgV[ 5]);
		}
	}

	if( bAllDocs)
	{
		if( RC_BAD( rc = pDb->getFirstDocument( uiCollection, &pDoc)))
		{
			pShell->con_printf( "Unable to get first document. rc == %e.", rc);
			iExitCode = -1;
			goto Exit;
		}

		do
		{
			if( RC_BAD( rc = pDb->exportXML( pDoc, pFileOStream, eFormat)))
			{
				goto Exit;
			}
		} while( RC_OK( rc = pDoc->getNextDocument( pDb, &pDoc)));

		if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			pShell->con_printf( "Error while iterating documents. rc == %x\n",
				rc);
			iExitCode = -1;
			goto Exit;
		}
		
		rc = RC_SET( NE_XFLM_OK);
	}
	else
	{
		if ( RC_BAD( rc = pDb->getDocument(
			uiCollection, 
			XFLM_EXACT, 
			uiDocNum, 
			&pDoc)))
		{
			pShell->con_printf( "Could not retrieve document %u\n", uiDocNum);
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = pDb->exportXML( pDoc, pFileOStream, eFormat)))
		{
			goto Exit;
		}
	}

	pShell->con_printf( "Export complete\n");

Exit:

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pFileOStream)
	{
		pFileOStream->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmExportCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "export", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Displays help for the export command
*****************************************************************************/
void FlmExportCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if ( !pszCommand)
	{
		pShell->displayCommand( "export", "Export a document from"
			" a database to a file.");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s <outputfile> [<db#> [<collection> [<doc#> [<export format>]]]]\n"
			"  <outputfile> name of file to export data to\n"
			"  <db#> defaults to zero if omitted\n"
			"  <collection> defaults to \"Data\" if omitted\n"
			"    Data - Default data collection\n"
			"    Dict - Dictionary collection\n"
			"    #    - Collection number\n"
			"  <doc#> defaults to all documents if omitted\n"
			"  <export format> defaults to \"indent\" if omitted)\n"
			"    none    - no special formatting used\n"
			"    newline - newlines are inserted after elements\n"
			"    indent  - same as \"newline\" + sub-elements are indented\n"
			"    idata   - same as \"indent\" + element data is also indented\n",
			pszCommand);
	}
}

/****************************************************************************
Desc: Executes a wrapkey command
*****************************************************************************/
FLMINT FlmWrapKeyCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	int					iExitCode = 0;
	RCODE					rc = NE_XFLM_OK;
	IF_Db *				pDb = NULL;
	FLMUINT				uiDbId = 0;

	if (iArgC >= 2)
	{
		uiDbId = f_atoi( ppszArgV[ 1]);
	}
	else
	{
		uiDbId = 0;
	}
	
	if( RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
	{
		pShell->con_printf( "Error %X getting database %u\n",
			(unsigned)rc, (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}
	
	if (!pDb)
	{
		pShell->con_printf( "Database %u is not open\n", (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}
	if (iArgC >= 3)
	{
		if (RC_BAD( rc = pDb->wrapKey( ppszArgV[2])))
		{
			pShell->con_printf( "wrapKey failed with error: %X\n",
				(unsigned)rc);
			iExitCode = -1;
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->wrapKey( NULL)))
		{
			pShell->con_printf( "wrapKey failed with error: %X\n",
				(unsigned)rc);
			iExitCode = -1;
			goto Exit;
		}
	}

Exit:

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmWrapKeyCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "wrapkey", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Displays help for the wrapkey command
*****************************************************************************/
void FlmWrapKeyCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if ( !pszCommand)
	{
		pShell->displayCommand( "wrapkey", "(Re)Wrap the database key in"
			" either a password or the NICI server key");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s [db#] [<password>]\n"
			"  If a password is present, the database key is wrapped\n"
			"  in that password.  Otherwise, it is wrapped in the NICI\n"
			"  server key.\n", pszCommand);
	}
}

/****************************************************************************
Desc: Display an error message on the screen
*****************************************************************************/
FSTATIC void domDisplayError(
	FTX_WINDOW *	pWindow,
	const char *	pszError,
	RCODE				errRc)
{
	FTX_SCREEN *	pScreen;
	FLMUINT			uiTermChar;
	char				szErrBuf [100];

	FTXWinGetScreen( pWindow, &pScreen);
	if (errRc == NE_XFLM_OK)
	{
		FTXDisplayMessage( pScreen, FLM_RED, FLM_WHITE,
								pszError, NULL, &uiTermChar);
	}
	else
	{
		f_sprintf( szErrBuf, "%s: %e", pszError, errRc);
		FTXDisplayMessage( pScreen, FLM_RED, FLM_WHITE,
								szErrBuf, NULL, &uiTermChar);
	}
}

/****************************************************************************
Desc: Output values for DOM info line.
*****************************************************************************/
FSTATIC void domOutputValues(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLabel,
	FLMUINT64		ui64Bytes,
	FLMUINT			uiPercent,
	FLMUINT64		ui64Count)
{
	char	szBuf [100];

	szBuf [0] = ' ';
	szBuf [1] = ' ';
	f_memset( &szBuf [2], '.', 22);
	f_memcpy( &szBuf [2], pszLabel, (FLMSIZET)f_strlen( pszLabel));
	szBuf [24] = ' ';
	if (uiPercent <= 10000)
	{
		f_sprintf( &szBuf [25], "%3u.%02u  ",
		(unsigned)(uiPercent / 100), (unsigned)(uiPercent % 100));
	}
	else
	{
		f_memset( &szBuf [25], ' ', 8);
	}
	if (ui64Bytes != FLM_MAX_UINT64)
	{
		f_sprintf( &szBuf [33], "%,20I64u", ui64Bytes);
	}
	else
	{
		f_memset( &szBuf [33], ' ', 20);
	}
	if (ui64Count != FLM_MAX_UINT64)
	{
		f_sprintf( &szBuf [53], "%,20I64u", ui64Count);
	}
	else
	{
		szBuf [53] = 0;
	}
	domDisplayLine( pWindow, puiLineCount, pFileHdl, szBuf);
}

/****************************************************************************
Desc: Output a node info item.
*****************************************************************************/
FINLINE void domOutputNodeInfoItem(
	FTX_WINDOW *				pWindow,
	FLMUINT *					puiLineCount,
	IF_FileHdl *				pFileHdl,
	const char *				pszInfoType,
	XFLM_NODE_INFO_ITEM *	pInfoItem,
	FLMUINT64					ui64TotalBytes,
	FLMBOOL						bForce = FALSE)
{
	if (bForce || pInfoItem->ui64Count)
	{
		domOutputValues( pWindow, puiLineCount, pFileHdl, pszInfoType,
			pInfoItem->ui64Bytes,
			(FLMUINT)((pInfoItem->ui64Bytes * 10000) / ui64TotalBytes),
			pInfoItem->ui64Count);
	}
}

/****************************************************************************
Desc: Gathers and then displays node information in a window.
*****************************************************************************/
FLMBOOL domDisplayNodeInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	FLMBOOL			bDoSubTree,
	FLMBOOL			bWaitForKeystroke)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bOk = FALSE;
	IF_NodeInfo *	pNodeInfo = NULL;
	XFLM_NODE_INFO	nodeInfo;
	IF_DOMNode *	pNode = NULL;
	FLMUINT64		ui64TotalOverheadBytes;
	FLMUINT64		ui64TotalBytes;
	FLMUINT			uiPercent;
	FLMUINT64		ui64DocumentCount = 1;
	FLMUINT			uiChar;
	IF_FileHdl *	pFileHdl = NULL;
	FLMUINT			uiLineCount;
	IF_DbSystem *	pDbSystem = NULL;
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	// If there is a file name, attempt to create it

	if (pszOutputFileName && *pszOutputFileName)
	{
		if (RC_BAD( rc = f_getFileSysPtr()->deleteFile( pszOutputFileName)))
		{
			if (rc != NE_FLM_IO_PATH_NOT_FOUND)
			{
				domDisplayError( pWindow, "Error deleting output file", rc);
				goto Exit;
			}
		}
		if (RC_BAD( rc = f_getFileSysPtr()->createFile( pszOutputFileName,
								FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE |
								FLM_IO_CREATE_DIR, &pFileHdl)))
		{
			domDisplayError( pWindow, "Error creating output file", rc);
			goto Exit;
		}
	}

	if (RC_BAD( rc = pDbSystem->createIFNodeInfo( &pNodeInfo)))
	{
		domDisplayError( pWindow, "Error calling createIFNodeInfo", rc);
		goto Exit;
	}

	if (ui64NodeId)
	{
		if (RC_BAD( rc = pDb->getNode( uiCollection, ui64NodeId, &pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				domDisplayError( pWindow, "Node does not exist", NE_XFLM_OK);
			}
			else
			{
				domDisplayError( pWindow, "Error calling getNode", rc);
			}
			goto Exit;
		}
		
		if (RC_BAD( rc = pNodeInfo->addNodeInfo( pDb, pNode, bDoSubTree)))
		{
			domDisplayError( pWindow, "Error calling addNodeInfo", rc);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getFirstDocument( uiCollection, &pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				domDisplayError( pWindow, "Collection is empty", NE_XFLM_OK);
			}
			else
			{
				domDisplayError( pWindow, "Error calling getFirstDocument", rc);
			}
			goto Exit;
		}

		ui64DocumentCount = 0;
		FTXWinPrintf( pWindow, "\n");
		for (;;)
		{
			if( RC_OK( FTXWinTestKB( pWindow)))
			{
				FTXWinInputChar( pWindow, &uiChar);
				if (uiChar == FKB_ESC)
				{
					FTXWinPrintf( pWindow, "\nESCAPE PRESSED, stopped reading documents\n");
					break;
				}
			}

			ui64DocumentCount++;
			FTXWinPrintf( pWindow, "\rDocuments: %,20I64u", ui64DocumentCount);
			if (RC_BAD( rc = pNodeInfo->addNodeInfo( pDb, pNode, TRUE)))
			{
				domDisplayError( pWindow, "Error calling addNodeInfo", rc);
				goto Exit;
			}
			if (RC_BAD( rc = pNode->getNextDocument( pDb, &pNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					domDisplayError( pWindow, "Error calling getNextDocument", rc);
				}
				goto Exit;
			}
		}
		FTXWinPrintf( pWindow, "\n\n");
	}
	if (!ui64DocumentCount)
	{
		goto Exit;
	}

	// Print out the information
	
	pNodeInfo->getNodeInfo( &nodeInfo);
	ui64TotalBytes = (nodeInfo.attributeNode.ui64Bytes +
							nodeInfo.elementNode.ui64Bytes +
							nodeInfo.dataNode.ui64Bytes +
							nodeInfo.commentNode.ui64Bytes +
							nodeInfo.otherNode.ui64Bytes);

	ui64TotalOverheadBytes = nodeInfo.totalOverhead.ui64Bytes;

	uiLineCount = FTXWinGetCurrRow( pWindow);
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"OVERHEAD DETAILS              %                 BYTES               COUNT");

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Header Size", &nodeInfo.headerSize, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Type & Data Type", &nodeInfo.nodeAndDataType, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Flags", &nodeInfo.flags, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Name ID", &nodeInfo.nameId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Prefix ID", &nodeInfo.prefixId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Base ID", &nodeInfo.baseId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Document ID", &nodeInfo.documentId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Parent ID", &nodeInfo.parentId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Previous Sibling ID", &nodeInfo.prevSibId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Next Sibling ID", &nodeInfo.nextSibId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"First Child ID", &nodeInfo.firstChildId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Last Child ID", &nodeInfo.lastChildId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Child Element Count", &nodeInfo.childElmCount, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Data Child Count", &nodeInfo.dataChildCount, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Attr Count", &nodeInfo.attrCount, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Attr Base Name Id", &nodeInfo.attrBaseId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Attr Flags", &nodeInfo.attrFlags, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Attr Payload Len", &nodeInfo.attrPayloadLen, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Annotation Id", &nodeInfo.attrPayloadLen, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Meta Value", &nodeInfo.metaValue, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Encryption Id", &nodeInfo.encDefId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Data Length", &nodeInfo.unencDataLen, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Child Element Name Id", &nodeInfo.childElmNameId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Child Element Node Id", &nodeInfo.childElmNodeId, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Encryption IV", &nodeInfo.encIV, ui64TotalBytes);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Encryption Padding", &nodeInfo.encPadding, ui64TotalBytes);

	// Print out summary information

	domDisplayLine( pWindow, &uiLineCount, pFileHdl, " ");
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"SUMMARY                       %                 BYTES               COUNT");
	uiPercent = (FLMUINT)((ui64TotalOverheadBytes * 10000) / ui64TotalBytes);
	domOutputValues( pWindow, &uiLineCount, pFileHdl, "Overhead",
		ui64TotalOverheadBytes, uiPercent, FLM_MAX_UINT64);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Numeric Data", &nodeInfo.dataNumeric, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"String Data", &nodeInfo.dataString, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Binary Data", &nodeInfo.dataBinary, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"No Data", &nodeInfo.dataNodata, ui64TotalBytes, TRUE);

	domOutputValues( pWindow, &uiLineCount, pFileHdl, "TOTAL BYTES",
		ui64TotalBytes, 10000, FLM_MAX_UINT64);

	domDisplayLine( pWindow, &uiLineCount, pFileHdl, " ");

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Element Nodes", &nodeInfo.elementNode, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Attribute Nodes", &nodeInfo.attributeNode, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Data Nodes", &nodeInfo.dataNode, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Comment Nodes", &nodeInfo.commentNode, ui64TotalBytes, TRUE);

	domOutputNodeInfoItem( pWindow, &uiLineCount, pFileHdl,
		"Other Nodes", &nodeInfo.otherNode, ui64TotalBytes, TRUE);

	domDisplayLine( pWindow, &uiLineCount, pFileHdl, " ");

	domOutputValues( pWindow, &uiLineCount, pFileHdl, "TOTAL DOCUMENTS",
		ui64DocumentCount, FLM_MAX_UINT, FLM_MAX_UINT64);
	domOutputValues( pWindow, &uiLineCount, pFileHdl, "AVG. BYTES PER DOC",
		ui64TotalBytes / ui64DocumentCount, FLM_MAX_UINT, FLM_MAX_UINT64);
	domOutputValues( pWindow, &uiLineCount, pFileHdl, "AVG. OVHD. PER DOC",
		ui64TotalOverheadBytes / ui64DocumentCount, FLM_MAX_UINT, FLM_MAX_UINT64);
	domOutputValues( pWindow, &uiLineCount, pFileHdl, "AVG. DATA PER DOC",
		(ui64TotalBytes - ui64TotalOverheadBytes) / ui64DocumentCount,
		FLM_MAX_UINT, FLM_MAX_UINT64);

	bOk = TRUE;

	if (bWaitForKeystroke)
	{
		domDisplayLine( pWindow, &uiLineCount, NULL, NULL, "Press any character to exit: ");
	}

Exit:

	if (pNodeInfo)
	{
		pNodeInfo->Release();
	}
	
	if (pNode)
	{
		pNode->Release();
	}
	
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}
	
	return( bOk);
}

/****************************************************************************
Desc: Constructor
*****************************************************************************/
Entry_Info::Entry_Info()
{
	m_ui64NumEntries = 0;
	m_ui64TotalBytes = 0;
	resetOverheadInfo( &m_EntryNode);
	resetOverheadInfo( &m_AttrNode);
	resetOverheadInfo( &m_InternalEntryFlags);
	resetOverheadInfo( &m_ModifyTime);
	resetOverheadInfo( &m_EntryFlags);
	resetOverheadInfo( &m_PartitionID);
	resetOverheadInfo( &m_ClassID);
	resetOverheadInfo( &m_ParentID);
	resetOverheadInfo( &m_AlternateID);
	resetOverheadInfo( &m_SubordinateCount);
	resetOverheadInfo( &m_RDN);
	resetOverheadInfo( &m_FirstChild);
	resetOverheadInfo( &m_LastChild);
	resetOverheadInfo( &m_NextSibling);
	resetOverheadInfo( &m_PrevSibling);
	m_pAttrList = NULL;
	m_uiAttrListSize = 0;
	m_uiNumAttrs = 0;
	
	m_pool.poolInit( 512);
}
	
/****************************************************************************
Desc: Destructor
*****************************************************************************/
Entry_Info::~Entry_Info()
{
	if (m_pAttrList)
	{
		f_free( &m_pAttrList);
	}
	m_pool.poolFree();
}
	
/****************************************************************************
Desc: Transfer node information to OVERHEAD_INFO structure.
*****************************************************************************/
void Entry_Info::getOverheadInfo(
	OVERHEAD_INFO *	pOverhead)
{
	XFLM_NODE_INFO	nodeInfo;

	m_nodeInfo.getNodeInfo( &nodeInfo);
	pOverhead->ui64DOMOverhead += nodeInfo.totalOverhead.ui64Bytes;
	m_ui64TotalBytes += nodeInfo.totalOverhead.ui64Bytes;

	pOverhead->ui64ValueBytes += nodeInfo.dataNumeric.ui64Bytes;
	m_ui64TotalBytes += nodeInfo.dataNumeric.ui64Bytes;

	pOverhead->ui64ValueBytes += nodeInfo.dataBinary.ui64Bytes;
	m_ui64TotalBytes += nodeInfo.dataBinary.ui64Bytes;

	pOverhead->ui64ValueBytes += nodeInfo.dataString.ui64Bytes;
	m_ui64TotalBytes += nodeInfo.dataString.ui64Bytes;

	pOverhead->ui64ValueBytes += nodeInfo.dataNodata.ui64Bytes;
	m_ui64TotalBytes += nodeInfo.dataNodata.ui64Bytes;

	// Reset the node info

	m_nodeInfo.clearNodeInfo();
}

/****************************************************************************
Desc: Gathers information about a node's XML attributes.
*****************************************************************************/
RCODE Entry_Info::getXMLAttrInfo(
	IF_Db *				pDb,
	IF_DOMNode *		pNode,
	OVERHEAD_INFO *	pOverhead,
	FLMUINT				uiAttrNameId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pAttrNode = NULL;

	// Retrieve the node

	if (RC_BAD( rc = pNode->getAttribute( pDb, uiAttrNameId, &pAttrNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Get the information on the node.

	if (RC_BAD( rc = m_nodeInfo.addNodeInfo( pDb, pAttrNode, FALSE)))
	{
		goto Exit;
	}
	getOverheadInfo( pOverhead);

Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Gathers information about an attribute's value.
*****************************************************************************/
RCODE Entry_Info::getAttrValueInfo(
	IF_Db *				pDb,
	IF_DOMNode *		pValueNode,
	ATTR_NODE_INFO *	pAttrNodeInfo)
{
	RCODE				rc = NE_XFLM_OK;
	XFLM_NODE_INFO	nodeInfo;

	// Get the information on the value node.

	if (RC_BAD( rc = m_nodeInfo.addNodeInfo( pDb, pValueNode, FALSE)))
	{
		goto Exit;
	}
	m_nodeInfo.getNodeInfo( &nodeInfo);
	getOverheadInfo( &pAttrNodeInfo->OtherNodes);

	// Subtract off meta bytes DOM overhead and add it on for value flags.

	pAttrNodeInfo->OtherNodes.ui64DOMOverhead -= nodeInfo.metaValue.ui64Bytes;
	pAttrNodeInfo->ValueFlags.ui64ValueBytes += nodeInfo.metaValue.ui64Bytes;

	pAttrNodeInfo->ui64NumValues++;

	// Get the XML information for the meta data on the value node.

	if (RC_BAD( rc = getXMLAttrInfo( pDb, pValueNode, &pAttrNodeInfo->ValueFlags, FSMI_VALUE_FLAGS_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pValueNode, &pAttrNodeInfo->ValueMTS, FSMI_VALUE_MTS_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pValueNode, &pAttrNodeInfo->TTL, FSMI_TTL_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pValueNode, &pAttrNodeInfo->PolicyDN, FSMI_POLICY_DN_ATTR)))
	{
		goto Exit;
	}

	// Get the information on all other nodes below the value node.  This is where
	// the actual data for the value will be stored.  NOTE: There may not be any
	// sub-elements.

	if (RC_BAD( rc = m_nodeInfo.addNodeInfo( pDb, pValueNode, TRUE, FALSE)))
	{
		goto Exit;
	}
	getOverheadInfo( &pAttrNodeInfo->OtherNodes);

Exit:

	return( rc);
}

/****************************************************************************
Desc: Gathers information about an entry's attribute.
*****************************************************************************/
RCODE Entry_Info::processAttrValues(
	IF_Db *				pDb,
	IF_DOMNode *		pAttrNode,
	ATTR_NODE_INFO *	pAttrNodeInfo)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pValueNode = NULL;

	if (RC_BAD( rc = pAttrNode->getFirstChild( pDb, &pValueNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Loop through processing all of the values on the attribute

	for (;;)
	{

		// Process the information for the value

		if (RC_BAD( getAttrValueInfo( pDb, pValueNode, pAttrNodeInfo)))
		{
			goto Exit;
		}

		// Get the next value, if any

		if (RC_BAD( rc = pValueNode->getNextSibling( pDb, &pValueNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
	}

Exit:

	if (pValueNode)
	{
		pValueNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Find a directory attribute in the attribute array by attribute name id.
****************************************************************************/
FLMBOOL Entry_Info::findDirAttr(
	FLMUINT		uiAttrNameId,
	FLMUINT *	puiInsertPos)
{
	FLMBOOL				bFound = FALSE;
	FLMUINT				uiLoop;
	ATTR_NODE_INFO *	pAttrNodeInfo;
	FLMUINT				uiTblSize;
	FLMUINT				uiLow;
	FLMUINT				uiMid;
	FLMUINT				uiHigh;
	FLMUINT				uiTblNameId;
	
	// If the count is <= 4, do a sequential search through
	// the array.  Otherwise, do a binary search.
	
	if ((uiTblSize = m_uiNumAttrs) <= 4)
	{
		for (uiLoop = 0, pAttrNodeInfo = m_pAttrList;
			  uiLoop < m_uiNumAttrs && pAttrNodeInfo->uiAttrNameId < uiAttrNameId;
			  uiLoop++, pAttrNodeInfo++)
		{
			;
		}
		if (uiLoop < m_uiNumAttrs)
		{
			*puiInsertPos = uiLoop;
			if (pAttrNodeInfo->uiAttrNameId == uiAttrNameId)
			{
				bFound = TRUE;
			}
		}
		else
		{
			*puiInsertPos = uiLoop;
		}
	}
	else
	{
		uiHigh = --uiTblSize;
		uiLow = 0;
		for (;;)
		{
			uiMid = (uiLow + uiHigh) / 2;
			uiTblNameId = m_pAttrList [uiMid].uiAttrNameId;
			if (uiTblNameId == uiAttrNameId)
			{
				// Found Match
	
				*puiInsertPos = uiMid;
				bFound = TRUE;
				goto Exit;
			}
	
			// Check if we are done
	
			if (uiLow >= uiHigh)
			{
				// Done, item not found
	
				*puiInsertPos = (uiAttrNameId < uiTblNameId)
										 ? uiMid
										 : uiMid + 1;
				goto Exit;
			}
	
			if (uiAttrNameId < uiTblNameId)
			{
				if (uiMid == 0)
				{
					*puiInsertPos = 0;
					goto Exit;
				}
				uiHigh = uiMid - 1;
			}
			else
			{
				if (uiMid == uiTblSize)
				{
					*puiInsertPos = uiMid + 1;
					goto Exit;
				}
				uiLow = uiMid + 1;
			}
		}
	}
	
Exit:

	return( bFound);
}

/****************************************************************************
Desc: Gathers information about an entry's attribute.
*****************************************************************************/
RCODE Entry_Info::getDirAttrInfo(
	IF_Db *			pDb,
	IF_DOMNode *	pAttrNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiAttrNameId;
	FLMUINT				uiPos;
	ATTR_NODE_INFO *	pAttrNodeInfo;
	FLMUINT				uiNameSize;

	// Get the information on the attribute node.

	if (RC_BAD( rc = m_nodeInfo.addNodeInfo( pDb, pAttrNode, FALSE)))
	{
		goto Exit;
	}
	getOverheadInfo( &m_AttrNode);

	// Locate the attribute in our array.  If the attribute is not
	// present, create a new one for it and insert it into the array.

	if (RC_BAD( rc = pAttrNode->getNameId( pDb, &uiAttrNameId)))
	{
		goto Exit;
	}

	// Try to find the attribute in our list

	if (!findDirAttr( uiAttrNameId, &uiPos))
	{
		// Need to make room in the list

		if (m_uiAttrListSize == m_uiNumAttrs)
		{
			if (RC_BAD( rc = f_realloc( (m_uiAttrListSize + 20) * sizeof( ATTR_NODE_INFO),
									&m_pAttrList)))
			{
				goto Exit;
			}
			m_uiAttrListSize += 20;
		}

		// Scoot everything in the list up in the array.

		if (m_uiNumAttrs && uiPos < m_uiNumAttrs - 1)
		{
			f_memmove( &m_pAttrList [uiPos + 1], &m_pAttrList [uiPos],
				sizeof( ATTR_NODE_INFO) * (m_uiNumAttrs - uiPos));
		}
		
		// Initialize the new slot.

		pAttrNodeInfo = &m_pAttrList [uiPos];
		initAttrInfo( uiAttrNameId, pAttrNodeInfo);

		// Get the attribute name

		uiNameSize = FLM_MAX_UINT;
		if (RC_BAD( rc = pDb->getDictionaryName( ELM_ELEMENT_TAG, uiAttrNameId,
									(char *)NULL, &uiNameSize, NULL, NULL)))
		{
			goto Exit;
		}
		uiNameSize++;
		if (RC_BAD( rc = m_pool.poolAlloc( uiNameSize,
			(void **)&pAttrNodeInfo->pszAttrName)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDb->getDictionaryName( ELM_ELEMENT_TAG, uiAttrNameId,
									pAttrNodeInfo->pszAttrName, &uiNameSize, NULL, NULL)))
		{
			goto Exit;
		}

		// Truncate the name so it will print out - only care about the first 35 or so characters.

		if (uiNameSize > 35)
		{
			pAttrNodeInfo->pszAttrName [35] = 0;
		}
		m_uiNumAttrs++;
	}
	else
	{
		pAttrNodeInfo = &m_pAttrList [uiPos];
	}

	// Get the XLM attribute overhead.

	if (RC_BAD( rc = getXMLAttrInfo( pDb, pAttrNode, &pAttrNodeInfo->GVTS, FSMI_ATTR_GVTS_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pAttrNode, &pAttrNodeInfo->DTS, FSMI_ATTR_DTS_ATTR)))
	{
		goto Exit;
	}

	// Cycle through all of the attribute values.

	if (RC_BAD( rc = processAttrValues( pDb, pAttrNode, pAttrNodeInfo)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Gathers information about an entry's attributes.
*****************************************************************************/
RCODE Entry_Info::processEntryAttrs(
	IF_Db *			pDb,
	IF_DOMNode *	pEntryNode)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pAttrNode = NULL;

	if (RC_BAD( rc = pEntryNode->getFirstChild( pDb, &pAttrNode)))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Loop through processing all of the attributes.

	for (;;)
	{

		// Gather the attribute information.

		if (RC_BAD( getDirAttrInfo( pDb, pAttrNode)))
		{
			goto Exit;
		}

		// Get the next attribute, if any

		if (RC_BAD( rc = pAttrNode->getNextSibling( pDb, &pAttrNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
	}

Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Gathers information about an entry.
*****************************************************************************/
RCODE Entry_Info::addEntryInfo(
	IF_Db *			pDb,
	IF_DOMNode *	pEntryNode)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNameId;

	// Verify that pEntryNode is, in fact, an entry node.
	
	if( pEntryNode->getNodeType() != ELEMENT_NODE)
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pEntryNode->getNameId( pDb, &uiNameId)))
	{
		goto Exit;
	}
	if (uiNameId != FSMI_ENTRY_ELEMENT)
	{
		goto Exit;
	}

	// Get the entry node information.

	if (RC_BAD( rc = m_nodeInfo.addNodeInfo( pDb, pEntryNode, FALSE)))
	{
		goto Exit;
	}
	getOverheadInfo( &m_EntryNode);
	m_ui64NumEntries++;

	// Get the overhead for all of the node's attributes.

	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_InternalEntryFlags, FSMI_INTERNAL_ENTRY_FLAGS_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_ModifyTime, FSMI_ENTRY_MODIFY_TIME_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_EntryFlags, FSMI_ENTRY_FLAGS_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_PartitionID, FSMI_PARTITION_ID_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_ClassID, FSMI_CLASS_ID_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_ParentID, FSMI_PARENT_ID_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_AlternateID, FSMI_ALT_ID_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_SubordinateCount, FSMI_SUBORDINATE_COUNT_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_RDN, FSMI_RDN_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_FirstChild, FSMI_FIRST_CHILD_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_LastChild, FSMI_LAST_CHILD_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_NextSibling, FSMI_NEXT_SIBLING_ATTR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = getXMLAttrInfo( pDb, pEntryNode, &m_PrevSibling, FSMI_PREV_SIBLING_ATTR)))
	{
		goto Exit;
	}

	// Now get all of the directory attribute information for the node

	if (RC_BAD( rc = processEntryAttrs( pDb, pEntryNode)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Display a line, pausing when the window gets near to full.
*****************************************************************************/
FSTATIC void domDisplayLine(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLine,
	const char *	pszWaitPrompt)
{
	RCODE		rc;
	FLMUINT	uiNumRows;
	FLMUINT	uiNumCols;
	FLMUINT	uiChar;
	FLMUINT	uiBytesWritten;

	if (*puiLineCount == FLM_MAX_UINT)
	{
		return;
	}
	if (pFileHdl)
	{
		if (RC_BAD( rc = pFileHdl->write( FLM_IO_CURRENT_POS,
						f_strlen( pszLine), pszLine, &uiBytesWritten)))
		{
			domDisplayError( pWindow, "Error writing to output file", rc);
			*puiLineCount = FLM_MAX_UINT;
			return;
		}
		if (RC_BAD( rc = pFileHdl->write( FLM_IO_CURRENT_POS, 1, (void *)"\n",
						&uiBytesWritten)))
		{
			domDisplayError( pWindow, "Error writing to output file", rc);
			*puiLineCount = FLM_MAX_UINT;
			return;
		}
	}
	else
	{

		FTXWinGetSize( pWindow, &uiNumCols, &uiNumRows);
		if (*puiLineCount >= uiNumRows - 5 || pszWaitPrompt)
		{
			if (pszWaitPrompt)
			{
				FTXWinPrintf( pWindow, "%s", pszWaitPrompt);
			}
			else
			{
				FTXWinPrintf( pWindow, "...more, press any key to continue, ESC to quit: ");
			}
			
			if( RC_BAD( FTXWinInputChar( pWindow, &uiChar)))
			{
				*puiLineCount = FLM_MAX_UINT;
				return;
			}
			if (uiChar == FKB_ESC)
			{
				*puiLineCount = FLM_MAX_UINT;
				return;
			}
			*puiLineCount = 0;
			FTXWinClearLine( pWindow, 0, FTXWinGetCurrRow( pWindow));
		}
		if (pszLine)
		{
			FTXWinPrintf( pWindow, "%s\n", pszLine);
			(*puiLineCount)++;
		}
	}
}

/****************************************************************************
Desc: Display label, %, value
*****************************************************************************/
FSTATIC void domDisplayValue(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiLineCount,
	IF_FileHdl *	pFileHdl,
	const char *	pszLabel,
	FLMUINT			uiPercent,
	FLMUINT64		ui64Value)
{
	char		szBuf [100];

	szBuf [0] = ' ';
	szBuf [1] = ' ';
	f_memset( &szBuf[2], '.', 38);
	f_memcpy( &szBuf[2], pszLabel, (FLMSIZET)f_strlen( pszLabel));
	if (uiPercent <= 10000)
	{
		f_sprintf( &szBuf [40], "  %3u.%02u  %,20I64u",
				(unsigned)(uiPercent / 100), (unsigned)(uiPercent % 100),
				ui64Value);
	}
	else
	{
		f_sprintf( &szBuf [40], "          %,20I64u", ui64Value);
	}
	domDisplayLine( pWindow, puiLineCount, pFileHdl, szBuf);
}

/****************************************************************************
Desc: Display information from an OVERHEAD_INFO structure.
*****************************************************************************/
FSTATIC void domDisplayInfo(
	FTX_WINDOW *		pWindow,
	FLMUINT *			puiLineCount,
	IF_FileHdl *		pFileHdl,
	const char *		pszDomOverheadLabel,
	const char *		pszValueBytesLabel,
	OVERHEAD_INFO *	pInfo,
	FLMUINT64			ui64TotalBytes)
{
	FLMUINT	uiPercent;

	// Display the DOM overhead value

	if (pInfo->ui64DOMOverhead || pInfo->ui64ValueBytes)
	{
		uiPercent = (FLMUINT)((pInfo->ui64DOMOverhead * 10000) / ui64TotalBytes);
		domDisplayValue( pWindow, puiLineCount, pFileHdl, pszDomOverheadLabel,
			uiPercent, pInfo->ui64DOMOverhead);

		// Display the value bytes

		uiPercent = (FLMUINT)((pInfo->ui64ValueBytes * 10000) / ui64TotalBytes);
		domDisplayValue( pWindow, puiLineCount, pFileHdl, pszValueBytesLabel,
			uiPercent, pInfo->ui64ValueBytes);
	}
}

/****************************************************************************
Desc: Gathers and then displays directory entry information in a window.  This
		assumes that the nodes we are looking at are entries for the NVDS
		directory.
*****************************************************************************/
FLMBOOL domDisplayEntryInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT64		ui64NodeId,
	FLMBOOL			bWaitForKeystroke)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bOk = FALSE;
	Entry_Info			entryInfo;
	FLMUINT				uiNameId;
	IF_DOMNode *		pEntryNode = NULL;
	FLMUINT				uiLoop;
	FLMUINT64			ui64EntryDOMOverhead = 0;
	FLMUINT64			ui64EntryMetaDataOverhead = 0;
	FLMUINT64			ui64AttrDOMOverhead = 0;
	FLMUINT64			ui64AttrMetaDataOverhead = 0;
	FLMUINT64			ui64AttrData = 0;
	FLMUINT64			ui64GVTSDOMOverhead = 0;
	FLMUINT64			ui64GVTSDataOverhead = 0;
	FLMUINT64			ui64DTSDOMOverhead = 0;
	FLMUINT64			ui64DTSDataOverhead = 0;
	FLMUINT64			ui64ValueFlagsDOMOverhead = 0;
	FLMUINT64			ui64ValueFlagsDataOverhead = 0;
	FLMUINT64			ui64ValueMTSDOMOverhead = 0;
	FLMUINT64			ui64ValueMTSDataOverhead = 0;
	FLMUINT64			ui64TTLDOMOverhead = 0;
	FLMUINT64			ui64TTLDataOverhead = 0;
	FLMUINT64			ui64PolicyDNDOMOverhead = 0;
	FLMUINT64			ui64PolicyDNDataOverhead = 0;
	FLMUINT				uiChar;
	ATTR_NODE_INFO *	pAttrInfo;
	char					szBuf [80];
	FLMUINT				uiPercent;
	FLMUINT				uiLineCount;
	IF_FileHdl *		pFileHdl = NULL;

	// If there is a file name, attempt to create it

	if (pszOutputFileName && *pszOutputFileName)
	{
		if (RC_BAD( rc = f_getFileSysPtr()->deleteFile( pszOutputFileName)))
		{
			if (rc != NE_FLM_IO_PATH_NOT_FOUND)
			{
				domDisplayError( pWindow, "Error deleting output file", rc);
				goto Exit;
			}
		}
		if (RC_BAD( rc = f_getFileSysPtr()->createFile( pszOutputFileName,
								FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE |
								FLM_IO_CREATE_DIR, &pFileHdl)))
		{
			domDisplayError( pWindow, "Error creating output file", rc);
			goto Exit;
		}
	}

	if (ui64NodeId)
	{
		if (RC_BAD( rc = pDb->getNode( XFLM_DATA_COLLECTION, ui64NodeId, &pEntryNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				domDisplayError( pWindow, "Node does not exist", NE_XFLM_OK);
			}
			else
			{
				domDisplayError( pWindow, "Error calling getNode", rc);
			}
			goto Exit;
		}

		// Verify that pEntryNode is, in fact, an entry node.
		
		if( pEntryNode->getNodeType() != ELEMENT_NODE)
		{
			domDisplayError( pWindow, "Node is not an entry document", NE_XFLM_OK);
			goto Exit;
		}
		
		if (RC_BAD( rc = pEntryNode->getNameId( pDb, &uiNameId)))
		{
			domDisplayError( pWindow, "Error calling getNameId", rc);
			goto Exit;
		}
		
		if (uiNameId != FSMI_ENTRY_ELEMENT)
		{
			domDisplayError( pWindow, "Node is not an entry document", NE_XFLM_OK);
			goto Exit;
		}
		
		if (RC_BAD( rc = entryInfo.addEntryInfo( pDb, pEntryNode)))
		{
			domDisplayError( pWindow, "Error getting entry information", rc);
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getFirstDocument( XFLM_DATA_COLLECTION, &pEntryNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				domDisplayError( pWindow, "Collection is empty", NE_XFLM_OK);
			}
			else
			{
				domDisplayError( pWindow, "Error calling getFirstDocument", rc);
			}
			goto Exit;
		}

		FTXWinPrintf( pWindow, "\n");
		for (;;)
		{
			if( RC_OK( FTXWinTestKB( pWindow)))
			{
				FTXWinInputChar( pWindow, &uiChar);
				if (uiChar == FKB_ESC)
				{
					FTXWinPrintf( pWindow, "\nESCAPE PRESSED, stopped reading documents\n");
					break;
				}
			}

			if (RC_BAD( rc = entryInfo.addEntryInfo( pDb, pEntryNode)))
			{
				domDisplayError( pWindow, "Error getting entry information", rc);
				goto Exit;
			}
			FTXWinPrintf( pWindow, "\rEntries: %,20I64u", entryInfo.m_ui64NumEntries);
			if (RC_BAD( rc = pEntryNode->getNextDocument( pDb, &pEntryNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					domDisplayError( pWindow, "Error calling getNextDocument", rc);
				}
				goto Exit;
			}
		}
		FTXWinPrintf( pWindow, "\n\n");
	}
	if (!entryInfo.m_ui64NumEntries)
	{
		goto Exit;
	}

	// Print out the information
	
	uiLineCount = FTXWinGetCurrRow( pWindow);
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"ENTRY OVERHEAD                                 %                 BYTES");

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Entry DOM Node", "Entry Node Data",
		&entryInfo.m_EntryNode, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_EntryNode.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_EntryNode.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Attribute DOM Node", "Attribute Node Data",
		&entryInfo.m_AttrNode, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_AttrNode.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_AttrNode.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Entry Flags DOM", "Entry Flags Data",
		&entryInfo.m_EntryFlags, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_EntryFlags.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_EntryFlags.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "RDN DOM", "RDN Data",
		&entryInfo.m_RDN, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_RDN.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_RDN.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Internal Flags DOM", "Internal Flags Data",
		&entryInfo.m_InternalEntryFlags, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_InternalEntryFlags.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_InternalEntryFlags.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Modify Time DOM", "Modify Time Data",
		&entryInfo.m_ModifyTime, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_ModifyTime.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_ModifyTime.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Partition ID DOM", "Partition ID Data",
		&entryInfo.m_PartitionID, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_PartitionID.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_PartitionID.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Class ID DOM", "Class ID Data",
		&entryInfo.m_ClassID, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_ClassID.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_ClassID.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Parent ID DOM", "Parent ID Data",
		&entryInfo.m_ParentID, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_ParentID.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_ParentID.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Alternate ID DOM", "Alternate ID Data",
		&entryInfo.m_AlternateID, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_AlternateID.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_AlternateID.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Subordinate Count DOM", "Subordinate Count Data",
		&entryInfo.m_SubordinateCount, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_SubordinateCount.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_SubordinateCount.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "First Child ID DOM", "First Child ID Data",
		&entryInfo.m_FirstChild, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_FirstChild.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_FirstChild.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Last Child ID DOM", "Last Child ID Data",
		&entryInfo.m_LastChild, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_LastChild.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_LastChild.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Next Sibling ID DOM", "Next Sibling ID Data",
		&entryInfo.m_NextSibling, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_NextSibling.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_NextSibling.ui64ValueBytes;

	domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "Previous Sibling ID DOM", "Previous Sibling ID Data",
		&entryInfo.m_PrevSibling, entryInfo.m_ui64TotalBytes);
	ui64EntryDOMOverhead += entryInfo.m_PrevSibling.ui64DOMOverhead;
	ui64EntryMetaDataOverhead += entryInfo.m_PrevSibling.ui64ValueBytes;

	domDisplayLine( pWindow, &uiLineCount, pFileHdl, " ");
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"ATTRIBUTE INFORMATION                          %                 BYTES");

	for (uiLoop = 0, pAttrInfo = entryInfo.m_pAttrList;
		  uiLoop < entryInfo.m_uiNumAttrs;
		  uiLoop++, pAttrInfo++)
	{

		f_sprintf( szBuf, "  %s [%u]", pAttrInfo->pszAttrName, (unsigned)pAttrInfo->uiAttrNameId);
		domDisplayLine( pWindow, &uiLineCount, pFileHdl, szBuf);

		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Total Values", FLM_MAX_UINT,
			pAttrInfo->ui64NumValues);

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  GVTS DOM Overhead", "  GVTS Data Overhead",
			&pAttrInfo->GVTS, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->GVTS.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->GVTS.ui64ValueBytes;
		ui64GVTSDOMOverhead += pAttrInfo->GVTS.ui64DOMOverhead;
		ui64GVTSDataOverhead += pAttrInfo->GVTS.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  DTS DOM Overhead", "  DTS Data Overhead",
			&pAttrInfo->DTS, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->DTS.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->DTS.ui64ValueBytes;
		ui64DTSDOMOverhead += pAttrInfo->DTS.ui64DOMOverhead;
		ui64DTSDataOverhead += pAttrInfo->DTS.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  Value Flags DOM Overhead", "  Value Flags Data Overhead",
			&pAttrInfo->ValueFlags, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->ValueFlags.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->ValueFlags.ui64ValueBytes;
		ui64ValueFlagsDOMOverhead += pAttrInfo->ValueFlags.ui64DOMOverhead;
		ui64ValueFlagsDataOverhead += pAttrInfo->ValueFlags.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  Value MTS DOM Overhead", "  Value MTS Data Overhead",
			&pAttrInfo->ValueMTS, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->ValueMTS.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->ValueMTS.ui64ValueBytes;
		ui64ValueMTSDOMOverhead += pAttrInfo->ValueMTS.ui64DOMOverhead;
		ui64ValueMTSDataOverhead += pAttrInfo->ValueMTS.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  TTL DOM Overhead", "  TTL Data Overhead",
			&pAttrInfo->TTL, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->TTL.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->TTL.ui64ValueBytes;
		ui64TTLDOMOverhead += pAttrInfo->TTL.ui64DOMOverhead;
		ui64TTLDataOverhead += pAttrInfo->TTL.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  Policy DN DOM Overhead", "  Policy DN Data Overhead",
			&pAttrInfo->PolicyDN, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->PolicyDN.ui64DOMOverhead;
		ui64AttrMetaDataOverhead += pAttrInfo->PolicyDN.ui64ValueBytes;
		ui64PolicyDNDOMOverhead += pAttrInfo->PolicyDN.ui64DOMOverhead;
		ui64PolicyDNDataOverhead += pAttrInfo->PolicyDN.ui64ValueBytes;

		domDisplayInfo( pWindow, &uiLineCount, pFileHdl, "  Value Data DOM Overhead", "  Value Data",
			&pAttrInfo->OtherNodes, entryInfo.m_ui64TotalBytes);
		ui64AttrDOMOverhead += pAttrInfo->OtherNodes.ui64DOMOverhead;
		ui64AttrData += pAttrInfo->OtherNodes.ui64ValueBytes;
	}

	// Print out summary information

	domDisplayLine( pWindow, &uiLineCount, pFileHdl, " ");
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"SUMMARY                                        %                 BYTES");

	uiPercent = (FLMUINT)((ui64EntryDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ENTRY DOM OVERHEAD", uiPercent,
		ui64EntryDOMOverhead);

	uiPercent = (FLMUINT)((ui64EntryMetaDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ENTRY METADATA OVERHEAD", uiPercent,
		ui64EntryMetaDataOverhead);

	uiPercent = (FLMUINT)((ui64AttrDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ATTR DOM OVERHEAD", uiPercent,
		ui64AttrDOMOverhead);

	if (ui64GVTSDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64GVTSDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  GVTS", uiPercent,
			ui64GVTSDOMOverhead);
	}

	if (ui64DTSDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64DTSDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  DTS", uiPercent,
			ui64DTSDOMOverhead);
	}

	if (ui64ValueFlagsDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64ValueFlagsDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Flags", uiPercent,
			ui64ValueFlagsDOMOverhead);
	}

	if (ui64ValueMTSDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64ValueMTSDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  MTS", uiPercent,
			ui64ValueMTSDOMOverhead);
	}

	if (ui64TTLDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64TTLDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  TTL", uiPercent,
			ui64TTLDOMOverhead);
	}

	if (ui64PolicyDNDOMOverhead)
	{
		uiPercent = (FLMUINT)((ui64PolicyDNDOMOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Policy DN", uiPercent,
			ui64PolicyDNDOMOverhead);
	}

	uiPercent = (FLMUINT)((ui64AttrMetaDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ATTR METADATA OVERHEAD", uiPercent,
		ui64AttrMetaDataOverhead);

	if (ui64GVTSDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64GVTSDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  GVTS", uiPercent,
			ui64GVTSDataOverhead);
	}

	if (ui64DTSDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64DTSDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  DTS", uiPercent,
			ui64DTSDataOverhead);
	}

	if (ui64ValueFlagsDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64ValueFlagsDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Flags", uiPercent,
			ui64ValueFlagsDataOverhead);
	}

	if (ui64ValueMTSDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64ValueMTSDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  MTS", uiPercent,
			ui64ValueMTSDataOverhead);
	}

	if (ui64TTLDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64TTLDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  TTL", uiPercent,
			ui64TTLDataOverhead);
	}

	if (ui64PolicyDNDataOverhead)
	{
		uiPercent = (FLMUINT)((ui64PolicyDNDataOverhead * 10000) / entryInfo.m_ui64TotalBytes);
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Policy DN", uiPercent,
			ui64PolicyDNDataOverhead);
	}

	uiPercent = (FLMUINT)((ui64AttrData * 10000) / entryInfo.m_ui64TotalBytes);
	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL VALUE DATA", uiPercent,
		ui64AttrData);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ENTRY BYTES", 10000,
		entryInfo.m_ui64TotalBytes);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "TOTAL ENTRIES", FLM_MAX_UINT,
		entryInfo.m_ui64NumEntries);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ENTRY SIZE", FLM_MAX_UINT,
		entryInfo.m_ui64TotalBytes / entryInfo.m_ui64NumEntries);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ENTRY DOM OVERHEAD", FLM_MAX_UINT,
		ui64EntryDOMOverhead / entryInfo.m_ui64NumEntries);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ENTRY METADATA OVERHEAD", FLM_MAX_UINT,
		ui64EntryMetaDataOverhead / entryInfo.m_ui64NumEntries);

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ATTR DOM OVERHEAD", FLM_MAX_UINT,
		ui64AttrDOMOverhead / entryInfo.m_ui64NumEntries);

	if (ui64GVTSDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  GVTS", FLM_MAX_UINT,
			ui64GVTSDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64DTSDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  DTS", FLM_MAX_UINT,
			ui64DTSDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64ValueFlagsDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Flags", FLM_MAX_UINT,
			ui64ValueFlagsDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64ValueMTSDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  MTS", FLM_MAX_UINT,
			ui64ValueMTSDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64TTLDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  TTL", FLM_MAX_UINT,
			ui64TTLDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64PolicyDNDOMOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Policy DN", FLM_MAX_UINT,
			ui64PolicyDNDOMOverhead / entryInfo.m_ui64NumEntries);
	}

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ATTR METADATA OVERHEAD", FLM_MAX_UINT,
		ui64AttrMetaDataOverhead / entryInfo.m_ui64NumEntries);

	if (ui64GVTSDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  GVTS", FLM_MAX_UINT,
			ui64GVTSDataOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64DTSDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  DTS", FLM_MAX_UINT,
			ui64DTSDataOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64ValueFlagsDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Flags", FLM_MAX_UINT,
			ui64ValueFlagsDataOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64ValueMTSDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  MTS", FLM_MAX_UINT,
			ui64ValueMTSDataOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64TTLDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  TTL", FLM_MAX_UINT,
			ui64TTLDataOverhead / entryInfo.m_ui64NumEntries);
	}

	if (ui64PolicyDNDataOverhead)
	{
		domDisplayValue( pWindow, &uiLineCount, pFileHdl, "  Policy DN", FLM_MAX_UINT,
			ui64PolicyDNDataOverhead / entryInfo.m_ui64NumEntries);
	}

	domDisplayValue( pWindow, &uiLineCount, pFileHdl, "AVG. ATTR VALUE DATA", FLM_MAX_UINT,
		ui64AttrData / entryInfo.m_ui64NumEntries);

	bOk = TRUE;

	if (bWaitForKeystroke)
	{
		domDisplayLine( pWindow, &uiLineCount, NULL, NULL, "Press any character to exit: ");
	}

Exit:

	if (pEntryNode)
	{
		pEntryNode->Release();
	}
	
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	
	return( bOk);
}

/****************************************************************************
Desc: Executes a nodeinfo command
*****************************************************************************/
FLMINT FlmNodeInfoCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	int				iExitCode = 0;
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = NULL;
	FLMUINT			uiDbId = 0;
	FLMUINT			uiCollection;
	FLMUINT64		ui64NodeId;
	FLMBOOL			bDoSubTree = FALSE;
	FLMBOOL			bDoDirTree = FALSE;
	FTX_WINDOW *	pWin = pShell->getWindow();
	
	if (iArgC >= 2)
	{
		if (f_stricmp( ppszArgV [1], "all") == 0)
		{
			ui64NodeId = 0;
		}
		else
		{
			ui64NodeId = (FLMUINT64)f_atol( ppszArgV[ 1]);
		}
	}
	else
	{
		pShell->con_printf( "Must specify a node id\n");
		iExitCode = -1;
		goto Exit;
	}
	
	if (iArgC >= 3)
	{
		if (f_stricmp( ppszArgV [2], "subtree") == 0)
		{
			bDoSubTree = TRUE;
		}
		else if (f_stricmp( ppszArgV [2], "dirtree") == 0)
		{
			bDoDirTree = TRUE;
		}
		else
		{
			pShell->con_printf( "Invalid info option - must be subtree or dirtree.\n");
			iExitCode = -1;
			goto Exit;
		}
	}
	
	if (iArgC >= 4)
	{
		if (f_stricmp( ppszArgV [3], "data") == 0)
		{
			uiCollection = XFLM_DATA_COLLECTION;
		}
		else if (f_stricmp( ppszArgV [3], "dict") == 0)
		{
			uiCollection = XFLM_DICT_COLLECTION;
		}
		else
		{
			uiCollection = f_atol( ppszArgV[ 3]);
		}
	}
	else
	{
		uiCollection = XFLM_DATA_COLLECTION;
	}
	
	if (iArgC >= 5)
	{
		uiDbId = f_atol( ppszArgV[ 4]);
	}
	else
	{
		uiDbId = 0;
	}
	
	if (RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
	{
		pShell->con_printf( "Error %e getting database %u\n",
			rc, (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}
	
	if (!pDb)
	{
		pShell->con_printf( "Database %u is not open\n", (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}

	if (bDoDirTree)
	{
		if (uiCollection != XFLM_DATA_COLLECTION)
		{
			pShell->con_printf( "dirtree option valid only for data collection\n");
			iExitCode = -1;
			goto Exit;
		}
		if (!domDisplayEntryInfo( pWin, pShell->getOutputFileName(),
					pDb, ui64NodeId, FALSE))
		{
			iExitCode = -1;
			goto Exit;
		}
	}
	else
	{
		if (!domDisplayNodeInfo( pWin, pShell->getOutputFileName(),
				pDb, uiCollection, ui64NodeId, bDoSubTree, FALSE))
		{
			iExitCode = -1;
			goto Exit;
		}
	}

Exit:

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmNodeInfoCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "nodeinfo", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Displays help for the nodeinfo command
*****************************************************************************/
void FlmNodeInfoCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if ( !pszCommand)
	{
		pShell->displayCommand( "nodeinfo", "Show information about a node,"
			" optionally including sub-tree nodes");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s <node# | all> [<info option>] [<collection#>] [<db#>]\n", pszCommand);
		pShell->con_printf(
			"     <info option> may be one of the following:\n");
		pShell->con_printf(
			"        subtree - print information for the node and all of its descendants.\n");
		pShell->con_printf(
			"        dirtree - assume node points to a directory object, and print out\n");
		pShell->con_printf(
			"                  information for directory object.\n");
		pShell->con_printf(
			"        NOTE: Default is single node if omitted.\n");
		pShell->con_printf(
			"     <collection#> may be a collection number or \"data\" for the default data\n");
		pShell->con_printf(
			"        collection or \"dict\" for the dictionary collection.  Default is\n");
		pShell->con_printf(
			"        default data collection if omitted.\n");
		pShell->con_printf(
			"     <db#> is the database ID.  Zero if omitted.\n");
	}
}

/****************************************************************************
Desc: Callback object while gathering b-tree information.
*****************************************************************************/
#define DATA_COL					40
#define LABEL_COL					2
#define NUMBER_ROW				2
#define NAME_ROW					(NUMBER_ROW + 1)
#define LF_BLOCK_COUNT_ROW		(NAME_ROW + 1)
#define LEVEL_ROW					(LF_BLOCK_COUNT_ROW + 1)
#define LEVEL_BLOCK_COUNT_ROW	(LEVEL_ROW + 1)
#define TOTAL_BLOCK_COUNT_ROW	(LEVEL_BLOCK_COUNT_ROW + 1)
class SH_BTreeInfoStatus : public IF_BTreeInfoStatus
{
public:

	SH_BTreeInfoStatus(
		FTX_WINDOW *	pWindow)
	{
		m_pWindow = pWindow;
		m_bFirstStatus = TRUE;
		m_uiCurrLfNum = 0;
		m_uiCurrLevel = 0;
		m_ui64CurrLfBlockCount = 0;
		m_ui64CurrLevelBlockCount = 0;
		m_ui64TotalBlockCount = 0;
	}
	
	virtual ~SH_BTreeInfoStatus()
	{
	}
	
	RCODE XFLAPI infoStatus(
		FLMUINT		uiCurrLfNum,
		FLMBOOL		bIsCollection,
		char *		pszCurrLfName,
		FLMUINT		uiCurrLevel,
		FLMUINT64	ui64CurrLfBlockCount,
		FLMUINT64	ui64CurrLevelBlockCount,
		FLMUINT64	ui64TotalBlockCount);
		
private:

	void outputLabel(
		const char *	pszLabel,
		FLMUINT			uiRow)
	{
		char	szTmp [80];
		FLMUINT	uiNumDots = DATA_COL - LABEL_COL;
		
		f_memset( szTmp, '.', uiNumDots);
		f_memcpy( szTmp, pszLabel, (FLMSIZET)f_strlen( pszLabel));
		szTmp [uiNumDots] = 0;
		
		FTXWinSetCursorPos( m_pWindow, LABEL_COL, uiRow);
		FTXWinPrintf( m_pWindow, "%s", szTmp);
	}

	void outputStr(
		const char *	pszStr,
		FLMUINT			uiRow,
		FLMBOOL			bClearToEOL)
	{
		FTXWinSetCursorPos( m_pWindow, DATA_COL, uiRow);
		if (bClearToEOL)
		{
			FTXWinClearToEOL( m_pWindow);
		}
		FTXWinPrintf( m_pWindow, "%s", pszStr);
	}

	void outputUINT(
		FLMUINT			uiValue,
		FLMUINT			uiRow)
	{
		char	szTmp [80];
		
		f_sprintf( szTmp, "%,u", (unsigned)uiValue);
		outputStr( szTmp, uiRow, FALSE);
	}
		
	void outputUINT64(
		FLMUINT64		ui64Value,
		FLMUINT			uiRow)
	{
		char	szTmp [80];
		
		f_sprintf( szTmp, "%,I64u", ui64Value);
		outputStr( szTmp, uiRow, FALSE);
	}
		
	FTX_WINDOW *	m_pWindow;
	FLMBOOL			m_bFirstStatus;
	FLMUINT			m_uiCurrLfNum;
	FLMUINT			m_uiCurrLevel;
	FLMUINT64		m_ui64CurrLfBlockCount;
	FLMUINT64		m_ui64CurrLevelBlockCount;
	FLMUINT64		m_ui64TotalBlockCount;
};

/****************************************************************************
Desc: Callback function that is called while gathering data on an index
		or collection.
*****************************************************************************/
RCODE XFLAPI SH_BTreeInfoStatus::infoStatus(
	FLMUINT		uiCurrLfNum,
	FLMBOOL		bIsCollection,
	char *		pszCurrLfName,
	FLMUINT		uiCurrLevel,
	FLMUINT64	ui64CurrLfBlockCount,
	FLMUINT64	ui64CurrLevelBlockCount,
	FLMUINT64	ui64TotalBlockCount)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiChar;
	
	// See if user pressed escape
	
	if( RC_OK( FTXWinTestKB( m_pWindow)))
	{
		FTXWinInputChar( m_pWindow, &uiChar);
		if (uiChar == FKB_ESC)
		{
			FTXWinPrintf( m_pWindow, "\nESCAPE PRESSED, stopped reading documents\n");
			rc = RC_SET( NE_XFLM_USER_ABORT);
			goto Exit;
		}
	}
	
	if (m_bFirstStatus)
	{
		m_bFirstStatus = FALSE;
		FTXWinClear( m_pWindow);
		if (bIsCollection)
		{
			outputLabel( "Collection Number", NUMBER_ROW);
			outputLabel( "Collection Name", NAME_ROW);
			outputLabel( "Collection Block Count", LF_BLOCK_COUNT_ROW);
		}
		else
		{
			outputLabel( "Index Number", NUMBER_ROW);
			outputLabel( "Index Name", NAME_ROW);
			outputLabel( "Index Block Count", LF_BLOCK_COUNT_ROW);
		}
		outputLabel( "Current Level", LEVEL_ROW);
		outputLabel( "Level Block Count", LEVEL_BLOCK_COUNT_ROW);
		outputLabel( "Total Block Count", TOTAL_BLOCK_COUNT_ROW);
	}
	
	// Update the display
	
	if (uiCurrLfNum != m_uiCurrLfNum || m_bFirstStatus)
	{
		m_uiCurrLfNum = uiCurrLfNum;
		outputUINT( uiCurrLfNum, NUMBER_ROW);
		outputStr( pszCurrLfName, NAME_ROW, TRUE);
	}
	if (uiCurrLevel != m_uiCurrLevel || m_bFirstStatus)
	{
		m_uiCurrLevel = uiCurrLevel;
		outputUINT( uiCurrLevel, LEVEL_ROW);
	}
	if (ui64CurrLfBlockCount != m_ui64CurrLfBlockCount || m_bFirstStatus)
	{
		m_ui64CurrLfBlockCount = ui64CurrLfBlockCount;
		outputUINT64( ui64CurrLfBlockCount, LF_BLOCK_COUNT_ROW);
	}
	if (ui64CurrLevelBlockCount != m_ui64CurrLevelBlockCount || m_bFirstStatus)
	{
		m_ui64CurrLevelBlockCount = ui64CurrLevelBlockCount;
		outputUINT64( ui64CurrLevelBlockCount, LEVEL_BLOCK_COUNT_ROW);
	}
	if (ui64TotalBlockCount != m_ui64TotalBlockCount || m_bFirstStatus)
	{
		m_ui64TotalBlockCount = ui64TotalBlockCount;
		outputUINT64( ui64CurrLevelBlockCount, TOTAL_BLOCK_COUNT_ROW);
	}
		
Exit:

	return( rc);
}

/****************************************************************************
Desc: Gathers and then displays node information in a window.
*****************************************************************************/
FLMBOOL domDisplayBTreeInfo(
	FTX_WINDOW *	pWindow,
	char *			pszOutputFileName,
	IF_Db *			pDb,
	FLMUINT			uiLfNum,
	FLMBOOL			bDoCollection,
	FLMBOOL			bDoIndex,
	FLMBOOL			bWaitForKeystroke)
{
	RCODE							rc = NE_XFLM_OK;
	FLMBOOL						bOk = FALSE;
	IF_BTreeInfo *				pBTreeInfo = NULL;
	SH_BTreeInfoStatus		infoStatus( pWindow);
	FLMUINT						uiIndexCount;
	FLMUINT						uiCollectionCount;
	FLMUINT						uiBTreeNum;
	char *						pszBTreeName;
	FLMUINT						uiNumLevels;
	FLMUINT						uiLoop;
	FLMUINT						uiLevel;
	IF_FileHdl *				pFileHdl = NULL;
	FLMUINT64					ui64BlockBytes;
	FLMUINT64					ui64FreeBlockBytes;
	FLMUINT64					ui64DataOnlyBytes;
	FLMUINT64					ui64TotalBytes;
	FLMUINT						uiLineCount;
	char							szBuf [100];
	XFLM_BTREE_LEVEL_INFO	levelInfo;
	IF_DbSystem *				pDbSystem = NULL;
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}
	
	// If there is a file name, attempt to create it

	if (pszOutputFileName && *pszOutputFileName)
	{
		if (RC_BAD( rc = f_getFileSysPtr()->deleteFile( pszOutputFileName)))
		{
			if (rc != NE_FLM_IO_PATH_NOT_FOUND)
			{
				domDisplayError( pWindow, "Error deleting output file", rc);
				goto Exit;
			}
		}
		if (RC_BAD( rc = f_getFileSysPtr()->createFile( pszOutputFileName,
								FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE |
								FLM_IO_CREATE_DIR, &pFileHdl)))
		{
			domDisplayError( pWindow, "Error creating output file", rc);
			goto Exit;
		}
	}

	if (RC_BAD( rc = pDbSystem->createIFBTreeInfo( &pBTreeInfo)))
	{
		domDisplayError( pWindow, "Error calling createIFNodeInfo", rc);
		goto Exit;
	}
	
	if (bDoCollection)
	{
		if (RC_BAD( rc = pBTreeInfo->collectCollectionInfo( pDb, uiLfNum,
									&infoStatus)))
		{
			if (rc != NE_XFLM_USER_ABORT)
			{
				domDisplayError( pWindow, "Error calling collectCollectionInfo", rc);
			}
			goto Exit;
		}
	}
	if (bDoIndex)
	{
		if (RC_BAD( rc = pBTreeInfo->collectIndexInfo( pDb, uiLfNum,
									&infoStatus)))
		{
			if (rc != NE_XFLM_USER_ABORT)
			{
				domDisplayError( pWindow, "Error calling collectIndexInfo", rc);
			}
			goto Exit;
		}
	}
	
	uiIndexCount = pBTreeInfo->getNumIndexes();
	uiCollectionCount = pBTreeInfo->getNumCollections();
	if (!uiIndexCount && !uiCollectionCount)
	{
		goto Exit;
	}

	FTXWinSetCursorPos( pWindow, 0,TOTAL_BLOCK_COUNT_ROW+1);
	FTXWinClearToEOL( pWindow);
	uiLineCount = FTXWinGetCurrRow( pWindow);
	domDisplayLine( pWindow, &uiLineCount, pFileHdl,
		"DETAILS                       %                 BYTES               COUNT");
		
	// Output index information
	
	for (uiLoop = 0; uiLoop < uiIndexCount; uiLoop++)
	{
		(void)pBTreeInfo->getIndexInfo( uiLoop, &uiBTreeNum, &pszBTreeName,
							&uiNumLevels);
		f_sprintf( szBuf, "Index: %s (%u)", pszBTreeName, (unsigned)uiBTreeNum);
		domDisplayLine( pWindow, &uiLineCount, pFileHdl, szBuf);
		
		ui64TotalBytes = 0;
		for (uiLevel = 0; uiLevel < uiNumLevels; uiLevel++)
		{
			pBTreeInfo->getIndexLevelInfo( uiLoop, uiLevel, &levelInfo);
			ui64TotalBytes += levelInfo.ui64BlockLength;
		}
		for (uiLevel = 0; uiLevel < uiNumLevels; uiLevel++)
		{
			f_sprintf( szBuf, "  Level %u", (unsigned)uiLevel);
			domDisplayLine( pWindow, &uiLineCount, pFileHdl, szBuf);
			
			pBTreeInfo->getIndexLevelInfo( uiLoop, uiLevel, &levelInfo);
			ui64BlockBytes = levelInfo.ui64BlockLength;
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Blocks",
					ui64BlockBytes,
					(FLMUINT)((ui64BlockBytes * 10000) / ui64TotalBytes),
					levelInfo.ui64BlockCount);
					
			ui64FreeBlockBytes = levelInfo.ui64BlockFreeSpace;
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Free Space",
					ui64FreeBlockBytes,
					(FLMUINT)((ui64FreeBlockBytes * 10000) / ui64TotalBytes),
					FLM_MAX_UINT64);
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    % of Block Used",
					FLM_MAX_UINT64,
					10000 - (FLMUINT)((ui64FreeBlockBytes * 10000) / ui64BlockBytes),
					FLM_MAX_UINT64);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Elements",
					ui64BlockBytes - ui64FreeBlockBytes,
					(FLMUINT)(((ui64BlockBytes - ui64FreeBlockBytes) * 10000) / ui64TotalBytes),
					levelInfo.ui64ElmCount);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Cont. Elements",
					FLM_MAX_UINT64,
					FLM_MAX_UINT,
					levelInfo.ui64ContElmCount);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Offset Ovhd.",
					levelInfo.ui64ElmOffsetOverhead,
					(FLMUINT)((levelInfo.ui64ElmOffsetOverhead * 10000) / ui64TotalBytes),
					FLM_MAX_UINT64);
					
			if (levelInfo.ui64ElmFlagOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Flag Ovhd.",
						levelInfo.ui64ElmFlagOvhd,
						(FLMUINT)((levelInfo.ui64ElmFlagOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmKeyLengthOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Key Len Ovhd.",
						levelInfo.ui64ElmKeyLengthOvhd,
						(FLMUINT)((levelInfo.ui64ElmKeyLengthOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmCountsOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Counts Ovhd.",
						levelInfo.ui64ElmCountsOvhd,
						(FLMUINT)((levelInfo.ui64ElmCountsOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmChildAddrsOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Child Addr Ovhd.",
						levelInfo.ui64ElmChildAddrsOvhd,
						(FLMUINT)((levelInfo.ui64ElmChildAddrsOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmDataLenOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Len Ovhd.",
						levelInfo.ui64ElmDataLenOvhd,
						(FLMUINT)((levelInfo.ui64ElmDataLenOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmOADataLenOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      DO Len Ovhd.",
						levelInfo.ui64ElmOADataLenOvhd,
						(FLMUINT)((levelInfo.ui64ElmOADataLenOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmKeyLength)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Key Bytes",
						levelInfo.ui64ElmKeyLength,
						(FLMUINT)((levelInfo.ui64ElmKeyLength * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64KeyDataSize)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "        Component Bytes",
						levelInfo.ui64KeyDataSize,
						(FLMUINT)((levelInfo.ui64KeyDataSize * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64KeyIdSize)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "        ID Bytes",
						levelInfo.ui64KeyIdSize,
						(FLMUINT)((levelInfo.ui64KeyIdSize * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64KeyComponentLengthsSize)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "        Comp. Len Bytes",
						levelInfo.ui64KeyComponentLengthsSize,
						(FLMUINT)((levelInfo.ui64KeyComponentLengthsSize * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmDataLength)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Bytes",
						levelInfo.ui64ElmDataLength,
						(FLMUINT)((levelInfo.ui64ElmDataLength * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
		}
	}
	
	// Output collection information

	for (uiLoop = 0; uiLoop < uiCollectionCount; uiLoop++)
	{
		(void)pBTreeInfo->getCollectionInfo( uiLoop, &uiBTreeNum, &pszBTreeName,
							&uiNumLevels);
		f_sprintf( szBuf, "Collection: %s (%u)", pszBTreeName, (unsigned)uiBTreeNum);
		domDisplayLine( pWindow, &uiLineCount, pFileHdl, szBuf);
		
		ui64TotalBytes = 0;
		for (uiLevel = 0; uiLevel < uiNumLevels; uiLevel++)
		{
			pBTreeInfo->getCollectionLevelInfo( uiLoop, uiLevel, &levelInfo);
			ui64TotalBytes += (levelInfo.ui64BlockLength +
									 levelInfo.ui64DataOnlyBlockLength);
		}
		for (uiLevel = 0; uiLevel < uiNumLevels; uiLevel++)
		{
			f_sprintf( szBuf, "  Level %u", (unsigned)uiLevel);
			domDisplayLine( pWindow, &uiLineCount, pFileHdl, szBuf);
			
			pBTreeInfo->getCollectionLevelInfo( uiLoop, uiLevel, &levelInfo);
			ui64BlockBytes = levelInfo.ui64BlockLength;
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Blocks",
					ui64BlockBytes,
					(FLMUINT)((ui64BlockBytes * 10000) / ui64TotalBytes),
					levelInfo.ui64BlockCount);
					
			ui64FreeBlockBytes = levelInfo.ui64BlockFreeSpace;
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Free Space",
					ui64FreeBlockBytes,
					(FLMUINT)((ui64FreeBlockBytes * 10000) / ui64TotalBytes),
					FLM_MAX_UINT64);
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    % Of Block Used",
					FLM_MAX_UINT64,
					10000 - (FLMUINT)((ui64FreeBlockBytes * 10000) / ui64BlockBytes),
					FLM_MAX_UINT64);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "    Elements",
					ui64BlockBytes - ui64FreeBlockBytes,
					(FLMUINT)(((ui64BlockBytes - ui64FreeBlockBytes) * 10000) / ui64TotalBytes),
					levelInfo.ui64ElmCount);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Cont. Elements",
					FLM_MAX_UINT64,
					FLM_MAX_UINT,
					levelInfo.ui64ContElmCount);
					
			domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Offset Ovhd.",
					levelInfo.ui64ElmOffsetOverhead,
					(FLMUINT)((levelInfo.ui64ElmOffsetOverhead * 10000) / ui64TotalBytes),
					FLM_MAX_UINT64);
					
			if (levelInfo.ui64ElmFlagOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Flag Ovhd.",
						levelInfo.ui64ElmFlagOvhd,
						(FLMUINT)((levelInfo.ui64ElmFlagOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmKeyLengthOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Key Len Ovhd.",
						levelInfo.ui64ElmKeyLengthOvhd,
						(FLMUINT)((levelInfo.ui64ElmKeyLengthOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmCountsOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Counts Ovhd.",
						levelInfo.ui64ElmCountsOvhd,
						(FLMUINT)((levelInfo.ui64ElmCountsOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmChildAddrsOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Child Addr Ovhd.",
						levelInfo.ui64ElmChildAddrsOvhd,
						(FLMUINT)((levelInfo.ui64ElmChildAddrsOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmDataLenOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Len Ovhd.",
						levelInfo.ui64ElmDataLenOvhd,
						(FLMUINT)((levelInfo.ui64ElmDataLenOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmOADataLenOvhd)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      DO Len Ovhd.",
						levelInfo.ui64ElmOADataLenOvhd,
						(FLMUINT)((levelInfo.ui64ElmOADataLenOvhd * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmKeyLength)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Key Bytes",
						levelInfo.ui64ElmKeyLength,
						(FLMUINT)((levelInfo.ui64ElmKeyLength * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			if (levelInfo.ui64ElmDataLength)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Bytes",
						levelInfo.ui64ElmDataLength,
						(FLMUINT)((levelInfo.ui64ElmDataLength * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
			}
			ui64DataOnlyBytes = levelInfo.ui64DataOnlyBlockLength;
			if (ui64DataOnlyBytes)
			{
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Only Blocks",
						ui64DataOnlyBytes,
						(FLMUINT)((ui64DataOnlyBytes * 10000) / ui64TotalBytes),
						levelInfo.ui64DataOnlyBlockCount);
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Only Used Bytes",
						ui64DataOnlyBytes - levelInfo.ui64DataOnlyBlockFreeSpace,
						(FLMUINT)((ui64DataOnlyBytes - levelInfo.ui64DataOnlyBlockFreeSpace * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      Data Only Free Bytes",
						levelInfo.ui64DataOnlyBlockFreeSpace,
						(FLMUINT)((levelInfo.ui64DataOnlyBlockFreeSpace * 10000) / ui64TotalBytes),
						FLM_MAX_UINT64);
				domOutputValues( pWindow, &uiLineCount, pFileHdl, "      % DO Blocks Used",
						FLM_MAX_UINT64,
						10000 - (FLMUINT)((levelInfo.ui64DataOnlyBlockFreeSpace * 10000) / ui64DataOnlyBytes),
						FLM_MAX_UINT64);
			}
		}
	}

	bOk = TRUE;

	if (bWaitForKeystroke)
	{
		domDisplayLine( pWindow, &uiLineCount, NULL, NULL, "Press any character to exit: ");
	}

Exit:

	if (pBTreeInfo)
	{
		pBTreeInfo->Release();
	}
	
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}
	
	return( bOk);
}

/****************************************************************************
Desc: Executes a nodeinfo command
*****************************************************************************/
FLMINT FlmBTreeInfoCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	int				iExitCode = 0;
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = NULL;
	FLMUINT			uiDbId = 0;
	FLMUINT			uiLfNum;
	FLMBOOL			bDoCollection = FALSE;
	FLMBOOL			bDoIndex = FALSE;
	FTX_WINDOW *	pWin = pShell->getWindow();
	
	if (f_stricmp( ppszArgV [0], "btreeinfo") == 0)
	{
		uiLfNum = 0;
		bDoCollection = TRUE;
		bDoIndex = TRUE;
		if (iArgC >= 2)
		{
			uiDbId = f_atol( ppszArgV[ 1]);
		}
		else
		{
			uiDbId = 0;
		}
	}
	else if( f_stricmp( ppszArgV [0], "collectioninfo") == 0)
	{
		if (iArgC >= 2)
		{
			if (f_stricmp( ppszArgV [1], "all") == 0)
			{
				uiLfNum = 0;
			}
			else if (f_stricmp( ppszArgV [1], "data") == 0)
			{
				uiLfNum = XFLM_DATA_COLLECTION;
			}
			else if (f_stricmp( ppszArgV [1], "dict") == 0)
			{
				uiLfNum = XFLM_DICT_COLLECTION;
			}
			else
			{
				uiLfNum = f_atol( ppszArgV[ 1]);
			}
		}
		else
		{
			pShell->con_printf( "Must specify a collection\n");
			iExitCode = -1;
			goto Exit;
		}
		bDoCollection = TRUE;
		bDoIndex = FALSE;
		if (iArgC >= 3)
		{
			uiDbId = f_atol( ppszArgV[ 2]);
		}
		else
		{
			uiDbId = 0;
		}
	}
	else
	{
		if (iArgC >= 2)
		{
			if (f_stricmp( ppszArgV [1], "all") == 0)
			{
				uiLfNum = 0;
			}
			else if (f_stricmp( ppszArgV [1], "dictnum") == 0)
			{
				uiLfNum = XFLM_DICT_NUMBER_INDEX;
			}
			else if (f_stricmp( ppszArgV [1], "dictname") == 0)
			{
				uiLfNum = XFLM_DICT_NAME_INDEX;
			}
			else
			{
				uiLfNum = f_atol( ppszArgV[ 1]);
			}
		}
		else
		{
			pShell->con_printf( "Must specify an index\n");
			iExitCode = -1;
			goto Exit;
		}
		bDoCollection = FALSE;
		bDoIndex = TRUE;
		if (iArgC >= 3)
		{
			uiDbId = f_atol( ppszArgV[ 2]);
		}
		else
		{
			uiDbId = 0;
		}
	}
	
	if (RC_BAD( rc = pShell->getDatabase( uiDbId, &pDb)))
	{
		pShell->con_printf( "Error %e getting database %u\n",
			rc, (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}
	
	if (!pDb)
	{
		pShell->con_printf( "Database %u is not open\n", (unsigned)uiDbId);
		iExitCode = -1;
		goto Exit;
	}

	if (!domDisplayBTreeInfo( pWin, pShell->getOutputFileName(),
				pDb, uiLfNum, bDoCollection, bDoIndex, FALSE))
	{
		iExitCode = -1;
		goto Exit;
	}

Exit:

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmBTreeInfoCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "collectioninfo", pszCommand) == 0 ||
				f_stricmp( "indexinfo", pszCommand) == 0 ||
				f_stricmp( "btreeinfo", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Displays help for the btreeinfo command
*****************************************************************************/
void FlmBTreeInfoCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if ( !pszCommand)
	{
		pShell->displayCommand( "collectioninfo", "Show information about a collection");
		pShell->displayCommand( "indexinfo", "Show information about an index");
		pShell->displayCommand( "btreeinfo", "Show information about ALL indexes and collections");
	}
	else if (f_stricmp( "collectioninfo", pszCommand) == 0)
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s <collection# | all> [<db#>]\n", pszCommand);
		pShell->con_printf(
			"     <collection#> may be a collection number or \"data\" for the default data\n");
		pShell->con_printf(
			"        collection or \"dict\" for the dictionary collection.  Default is\n");
		pShell->con_printf(
			"        default data collection if omitted.\n");
		pShell->con_printf(
			"     <db#> is the database ID.  Zero if omitted.\n");
	}
	else if (f_stricmp( "indexinfo", pszCommand) == 0)
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s <index# | all> [<db#>]\n", pszCommand);
		pShell->con_printf(
			"     <index#> may be an index number or \"dictnum\" for the dictionary\n");
		pShell->con_printf(
			"        number index or \"dictname\" for the dictionary name index.\n");
		pShell->con_printf(
			"        default is dictionary number index if omitted.\n");
		pShell->con_printf(
			"     <db#> is the database ID.  Zero if omitted.\n");
	}
	else if (f_stricmp( "btreeinfo", pszCommand) == 0)
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf(
			"  %s [<db#>]\n", pszCommand);
		pShell->con_printf(
			"     <db#> is the database ID.  Zero if omitted.\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::setupDirectories( 
	char *	pszBaseDir,
	char *	pszExtendedDir)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( !m_pszBaseDir && !m_pszExtendedDir && !m_pszResolvedDir);

	if( RC_BAD( rc = f_alloc( MAX_PATH_SIZE, &m_pszBaseDir)))
	{
		goto Exit;
	}

	f_strcpy( m_pszBaseDir, pszBaseDir);
	if ( m_pszBaseDir[ f_strlen(m_pszBaseDir) - 1] != SLASH)
	{
		f_strcat( m_pszBaseDir, SSLASH);
	}

	if( *pszExtendedDir)
	{
		if( f_strlen( pszExtendedDir) < MAX_PATH_SIZE)
		{
			if( RC_BAD( f_alloc( MAX_PATH_SIZE, &m_pszExtendedDir)))
			{
				goto Exit;
			}
			f_strcpy( m_pszExtendedDir, pszExtendedDir);
		}
		else
		{
			rc = RC_SET( NE_XFLM_FAILURE);
			goto Exit;
		}
	}

	if ( !m_pszResolvedDir)
	{
		if( RC_BAD( rc = f_alloc( MAX_PATH_SIZE, &m_pszResolvedDir)))
		{
			goto Exit;
		}
	}
	
	if ( RC_BAD( rc = resolveDir()))
	{
		goto Exit;
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL DirectoryIterator::isInSet(
	char *	pszFilename)
{
	char *		pszTemp = NULL;
	char *		pszSave = NULL;
	FLMBOOL		bFound = FALSE;
	FLMUINT		uiLoop = 0;

	if( RC_BAD( f_alloc( f_strlen( m_pszResolvedDir) + 
		f_strlen( pszFilename) + 1, &pszTemp)))
	{
		goto Exit;
	}

	// Move past the prefix since that is not stored in the match list.

	if( f_strstr( pszFilename, m_pszResolvedDir) == pszFilename)
	{
		f_strcpy( pszTemp, pszFilename + f_strlen( m_pszResolvedDir));
	}

	for( uiLoop = 0; uiLoop < m_uiTotalMatches; uiLoop++)
	{
		if ( f_stricmp( pszTemp, m_ppszMatchList[uiLoop]) == 0)
		{
			bFound = TRUE;
			break;
		}
	}

	if ( pszTemp)
	{
		f_free( &pszTemp);
	}
	if ( pszSave)
	{
		f_strcpy( pszFilename, pszSave);
		f_free( &pszSave);
	}

Exit:

	return bFound;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::setupForSearch( 
	char *	pszBaseDir,
	char *	pszExtendedDir,
	char *	pszPattern)
{
	RCODE		rc = NE_XFLM_OK;

	if ( RC_BAD( rc = setupDirectories( pszBaseDir, pszExtendedDir)))
	{
		goto Exit;
	}

	flmAssert( !m_pDirHdl && !m_ppszMatchList);

	if ( !m_pDirHdl)
	{
		if( RC_BAD( rc = f_getFileSysPtr()->openDir( m_pszResolvedDir, 
			pszPattern, &m_pDirHdl)))
		{
			goto Exit;
		}
	}

	// First pass - determine the number of matches

	while( RC_OK( m_pDirHdl->next()))
	{
		m_uiTotalMatches++;
	}

	if( RC_BAD( rc = f_alloc( 
		sizeof( char *) * m_uiTotalMatches, &m_ppszMatchList)))
	{
		goto Exit;
	}

	f_memset( m_ppszMatchList, 0, m_uiTotalMatches * sizeof( char *));

	// Reopen the directory and copy the matches
	
	m_pDirHdl->Release();
	m_pDirHdl = NULL;
		
	if( RC_BAD( rc = f_getFileSysPtr()->openDir( m_pszResolvedDir, 
		pszPattern, &m_pDirHdl)))
	{
		goto Exit;
	}
	
	m_uiCurrentMatch = 0;
	while ( RC_OK( m_pDirHdl->next()))
	{
		if( RC_BAD( rc = f_alloc( 
			f_strlen( m_pDirHdl->currentItemName()) + 1,
			&m_ppszMatchList[m_uiCurrentMatch])))
		{
			goto Exit;
		}

		f_strcpy( m_ppszMatchList[m_uiCurrentMatch], 
			m_pDirHdl->currentItemName());

		m_uiCurrentMatch++;
	}
	m_uiCurrentMatch = 0;
	m_bInitialized = TRUE;

Exit:
	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::next( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( m_uiCurrentMatch >= m_uiTotalMatches)
	{
		m_uiCurrentMatch = 0;
	}
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiCurrentMatch]); 
	}
	m_uiCurrentMatch++;
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::prev( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if( m_uiCurrentMatch == 0)
	{
		m_uiCurrentMatch = m_uiTotalMatches;
	}
	m_uiCurrentMatch--;

	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiCurrentMatch]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::first( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ 0]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::last( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiTotalMatches - 1]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::resolveDir()
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiIndex = 0;
	char				szTemp[ MAX_PATH_SIZE];

	if ( !m_pszExtendedDir)
	{
		f_strcpy( m_pszResolvedDir, m_pszBaseDir);
		goto Exit;
	}

	// Examine the extended dir.

	// If it begins with a SLASH, just go to the root
	if ( m_pszExtendedDir[0] == SLASH) 
	{
		if( RC_BAD( rc = extractRoot( m_pszBaseDir, szTemp)))
		{
			goto Exit;
		}

		f_strcpy( m_pszResolvedDir, szTemp); 
		f_strcat( m_pszResolvedDir, &m_pszExtendedDir[1]);

		goto Exit;
	}

	// I think this can only happen on windows and NetWare
	else if (isDriveSpec( m_pszExtendedDir))
	{
		//We have been given a fully-specified drive path
		f_strcpy( m_pszResolvedDir, m_pszExtendedDir);
		goto Exit;
	}

	// For each ".." reduce the base path by one
	for(;;)
	{
		if( (f_strlen( &m_pszExtendedDir[uiIndex]) >= f_strlen( PARENT_DIR)) &&
			f_memcmp( 
				&m_pszExtendedDir[uiIndex], 
				PARENT_DIR, 
				f_strlen( PARENT_DIR)) == 0)
		{
			uiIndex += f_strlen( PARENT_DIR);
			if ( m_pszExtendedDir[uiIndex] == SLASH)
			{
				uiIndex++;
			}
	
			f_getFileSysPtr()->pathReduce( m_pszBaseDir, szTemp, NULL);
			f_strcpy( m_pszBaseDir, szTemp);
			if ( m_pszBaseDir[ f_strlen(m_pszBaseDir) - 1] != SLASH)
			{
				f_strcat( m_pszBaseDir, SSLASH);
			}
		}
		else
		{
			break;
		}
	}

	// Tack on whatever's left
	f_strcpy( m_pszResolvedDir, m_pszBaseDir);
	if( m_pszResolvedDir[f_strlen( m_pszResolvedDir) - 1] != SLASH)
	{
		// Put the slash back on. f_pathReduce likes to take it off.
		f_strcat( m_pszResolvedDir, SSLASH);
	}
	f_strcat( m_pszResolvedDir, &m_pszExtendedDir[uiIndex]);

Exit:

	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::reset()
{
	if( m_pszBaseDir)
	{
		f_free( &m_pszBaseDir);
	}

	if (m_pszExtendedDir)
	{
		f_free( &m_pszExtendedDir);
	}

	if (m_pszResolvedDir)
	{
		f_free( &m_pszResolvedDir);
		m_pszResolvedDir = NULL;
	}

	if ( m_pDirHdl)
	{
		m_pDirHdl->Release();
		m_pDirHdl = NULL;
	}

	if ( m_ppszMatchList)
	{
		for ( FLMUINT uiLoop = 0; uiLoop < m_uiTotalMatches; uiLoop++)
		{
			f_free( &m_ppszMatchList[uiLoop]);
		}
		f_free( &m_ppszMatchList);
		m_ppszMatchList = NULL;
	}

	m_uiCurrentMatch = 0;
	m_uiTotalMatches = 0;
	m_bInitialized = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL DirectoryIterator::isDriveSpec(
	char *	pszPath)
{
	FLMBOOL		bIsDriveSpec = FALSE;
	char *		pszTemp = NULL;

	if ((pszTemp = (char *)f_strchr( pszPath, ':')) != NULL)
	{
		if( *(pszTemp + 1) == SLASH)
		{
			bIsDriveSpec = TRUE;
		}
#ifdef FLM_NLM
		// Netware accepts both front and back slashes
		else if( *(pszTemp + 1) == FWSLASH)
		{
			bIsDriveSpec = TRUE;
		}
#endif
	}
	return bIsDriveSpec;
}

/****************************************************************************
Desc: 
*****************************************************************************/
RCODE DirectoryIterator::extractRoot( 
	char *	pszPath,
	char *	pszRoot)
{
	FLMUINT		uiIndex = 0;
	FLMUINT		uiLen	= f_strlen( pszPath);
	RCODE			rc = NE_XFLM_OK;

	for ( uiIndex = 0; uiIndex < uiLen; uiIndex++)
	{
		if( pszPath[uiIndex] == '\\')
		{
			f_strncpy( pszRoot, pszPath, uiIndex + 1); 
			pszRoot[uiIndex + 1] = '\0';
			goto Exit;
		}
	}
	rc = RC_SET( NE_XFLM_NOT_FOUND);
	pszRoot[0] = '\0';
Exit:
	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
void removeChars(
	char *	pszString,
	char		cChar)
{
	char *	pszFrom = pszString;
	char *	pszTo = pszString;

	for ( ;;)
	{
		if ( *pszFrom != cChar)
		{
			*pszTo = *pszFrom;
			if ( *pszTo)
			{
				pszTo++;
				pszFrom++;
			}
			else
			{
				break;
			}
		}
		else
		{
			pszFrom++;
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
char * positionToPath(
	char *	pszCommandLine)
{
	char *	pszPathBegin = 
		pszCommandLine + f_strlen( pszCommandLine);

	if ( f_strlen( pszCommandLine) != 0 && *(pszPathBegin - 1) != ASCII_SPACE)
	{
		if ( *(pszPathBegin - 1))
		{
			// Move to the beginning of the last token

			while( ( pszPathBegin != pszCommandLine) && 
				( *(pszPathBegin - 1) != ASCII_SPACE))
			{
				if ( *(pszPathBegin - 1) == '\"')
				{
					// Find first whitespace after begin quote

					FLMBOOL		bInQuotes = TRUE;

					pszPathBegin--;
					
					while ( ( pszPathBegin != pszCommandLine) &&	bInQuotes)
					{
						pszPathBegin--;
						if ( *pszPathBegin == '\"')
						{
							bInQuotes = FALSE;
						}
					}
				}
				else
				{
					pszPathBegin--;
				}
			}
		}
	}

	return pszPathBegin;
}

/****************************************************************************
Desc: Given a path, extract the base directory and a wildcard for searching
*****************************************************************************/
void extractBaseDirAndWildcard( 
	char *	pszPath, 
	char *	pszBase, 
	char *	pszWildcard)
{

	flmAssert( pszBase && pszWildcard);

	pszBase[0] = '\0';
	pszWildcard[0] = '\0';

	// If the extended directory is a path but does not end with a 
	// slash, this means that we will use the last portion of the
	// path as our search pattern.

	if ( pszPath && //we have a path
		f_strchr( pszPath, SLASH) && //it contains directories
		pszPath[ f_strlen( pszPath) - 1] != SLASH) //does not end with a slash
	{
		f_getFileSysPtr()->pathReduce( pszPath,
			pszBase, pszWildcard);

		// Darn thing sometimes removes the trailing slash. Put it back.

		if ( pszPath[ f_strlen( pszBase) - 1] != SLASH)
		{
			f_strcat( pszBase, SSLASH);
		}
	}
	else if ( pszPath && !f_strchr( pszPath, SLASH))
	{
		// We will assume that what we have is just a part of a filename
		// since it contains no slashes.
		f_strcpy( pszWildcard, pszPath);
		//pszTabCompleteBegin[0] = '\0';
	}
	else if ( pszPath && pszPath[ f_strlen( pszPath) - 1] == SLASH)
	{
		// We were given only a path
		f_strcpy( pszBase, pszPath);
	}
	f_strcat( pszWildcard, "*");
}

const char * errorToString(
	XMLParseError	errorType)
{
	const char *		pszErrorStr;

	if( errorType == XML_NO_ERROR)
	{
		pszErrorStr = "NE_XFLM_OK";
	}
	else if( errorType < XML_NUM_ERRORS)
	{
		pszErrorStr = gv_XMLParseErrors[ errorType -1].pszErrorStr;
	}
	else
	{
		pszErrorStr = "Unknown error";
		flmAssert(0);
	}

	return( pszErrorStr);
}
