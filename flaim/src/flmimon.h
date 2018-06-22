//-------------------------------------------------------------------------
// Desc:	HTML callback function for displaying monitoring web pages.
// Tabs:	3
//
// Copyright (c) 2001-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FLMIMON_H
#define FLMIMON_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

int flmHttpCallback(
	HRequest *	pHRequest,
	void *		pvUserData);

// HTML definitions
#define	HTML_DOCTYPE "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">\n"
#define	TABLE			"<center><TABLE BORDER WIDTH=100%%>\n"
#define	TABLE_END	"</TABLE></center>\n"
#define	TR				"<TR>\n"
#define  TR_END		"</TR>\n"
#define	TD_4x			"<TD>0x%0.4X</TD>\n"
#define	TD_s			"<TD>%s</TD>\n"
#define	TD_a_s_s		"<TD><a href=\"%s\">%s</a></TD>\n"
#define	TD_a_s_x		"<TD><a href=\"%s\">0x%.8X</a></TD>\n"
#define	TD_a_p_s		"<TD><a href=\"javascript:openPopup(\'%s\')\">%s</a></TD>\n"
#define	TD_a_p_x		"<TD><a href=\"javascript:openPopup(\'%s\')\">0x%.8X</a></TD>\n"
#define	TD_ui			"<TD>%u</TD>\n"
#define	TD_i			"<TD>%d</TD>\n"
#define	TD_lu			"<TD>%lu</TD>\n"
#define	TD_ld			"<TD>%ld</TD>\n"
#define	TD_8x			"<TD>0x%0.8X</TD>\n"
#define	TD				"<TD></TD>\n"
#define	HEAD			"<THEAD>\n"
#define	HEAD_END		"</THEAD>\n"
#define	HEADING		"<TH>%s</TH>\n"

// Colors

#define FLM_IMON_COLOR_PUTTY_1		("#dfddd5")
#define FLM_IMON_COLOR_PUTTY_2		("#efeee9")

// RCache Link name definitions

#define	MANAGER			"RCacheMgr"
#define	AVAILGROUPS		"pAvailGroups"
#define	USEDGROUPS		"pUsedGroups"
#define	NEXT				"pNext"
#define	PREV				"pPrev"
#define	PREVINBUCKET	"pPrevInBucket"
#define	NEXTINBUCKET	"pNextInBucket"
#define	PREVINFILE		"pPrevInFile"
#define	NEXTINFILE		"pNextInFile"
#define	PREVINGLOBAL	"pPrevInGlobal"
#define	NEXTINGLOBAL	"pNextInGlobal"
#define	NEWERVERSION	"pNewerVersion"
#define	OLDERVERSION	"pOlderVersion"

// Operations for the printOperationButton

#define OPERATION_QUERY				"doQuery"
#define OPERATION_DELETE			"doDelete"
#define OPERATION_STOP				"doStop"
#define OPERATION_ABORT				"doAbort"
#define OPERATION_CHECK				"doCheck"
#define OPERATION_INDEX_LIST		"doIndexList"

// Enum. types

typedef enum
{
	URL_PATH_ENCODING = 1,
	URL_QUERY_ENCODING,
	HTML_ENCODING
} FStringEncodeType;

typedef enum
{
	JUSTIFY_LEFT = 1,
	JUSTIFY_CENTER,
	JUSTIFY_RIGHT
} JustificationType;

enum	ButtonTypes 
{ 
	BT_Submit,
	BT_Reset,
	BT_Button
};

/****************************************************************************
Desc: The F_WebPage class, from which all of the various web page classes 
		will be defined.
*****************************************************************************/
class F_WebPage : public F_Object
{
public:
	
	F_WebPage()
	{
		m_pszFormData = NULL;
		m_pszURLString = gv_FlmSysData.HttpConfigParms.pszURLString;
		fnPrintf = gv_FlmSysData.HttpConfigParms.fnPrintf;
		m_uiSessionRC = FERR_NOT_IMPLEMENTED; // Indicates not setup yet.
	}

	virtual ~F_WebPage()
	{
		if( m_pszFormData)
		{
			f_free( &m_pszFormData);
		}
		// If the session has not been released yet, then release it.
		if (m_pFlmSession)
		{
			flmAssert( 0);
			releaseSession();
		}
	}

	FINLINE const char * fnReqPath( void)
	{
		return( gv_FlmSysData.HttpConfigParms.fnReqPath( m_pHRequest));
	}

	FINLINE const char * fnReqQuery( void)
	{
		return( gv_FlmSysData.HttpConfigParms.fnReqQuery( m_pHRequest));
	}

	FINLINE const char * fnReqHdrValue(
		const char *	pszName)
	{
		return( gv_FlmSysData.HttpConfigParms.fnReqHdrValue( m_pHRequest,
					pszName));
	}

	FINLINE int fnSetHdrValue(
		const char *	pszName,
		const char *	pszValue)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSetHdrValue( m_pHRequest,
					pszName, pszValue));
	}

	FINLINE int fnEmit( void)
	{
		return( gv_FlmSysData.HttpConfigParms.fnEmit( m_pHRequest));
	}

	FINLINE void fnSetNoCache(
		const char *	pszHeader)
	{
		gv_FlmSysData.HttpConfigParms.fnSetNoCache( m_pHRequest,
						pszHeader);
	}

	FINLINE int fnSendHeader(
		int			iStatus)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSendHeader( m_pHRequest,
						iStatus));
	}

	FINLINE int fnSetIOMode(
		int		bRaw,
		int		bOutput)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSetIOMode( m_pHRequest,
						bRaw, bOutput));
	}

	FINLINE int fnSendBuffer(
		const void *	pvBuf,
		FLMSIZET			bufsz)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSendBuffer( m_pHRequest,
						pvBuf, bufsz));
	}

	FINLINE void * fnAcquireSession( void)
	{
		return( gv_FlmSysData.HttpConfigParms.fnAcquireSession( m_pHRequest));
	}

	FINLINE void fnReleaseSession(
		void *	pvHSession)
	{
		gv_FlmSysData.HttpConfigParms.fnReleaseSession( pvHSession);
	}

	FINLINE void * fnAcquireUser(
		void *		pvHSession)
	{
		return( gv_FlmSysData.HttpConfigParms.fnAcquireUser( pvHSession, m_pHRequest));
	}

	FINLINE void fnReleaseUser(
		void *		pvHUser)
	{
		gv_FlmSysData.HttpConfigParms.fnReleaseUser( pvHUser);
	}

	FINLINE int fnSetSessionValue(
		void *				pvHSession,
		const char *		pcTag,
		const void *		pvData,
		FLMSIZET				uiSize)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSetSessionValue( pvHSession,
					pcTag, pvData, uiSize));
	}

	FINLINE int fnGetSessionValue(
		void *				pvHSession,
		const char *		pcTag,
		void *				pvData,
		FLMSIZET *				puiSize)
	{
		return( gv_FlmSysData.HttpConfigParms.fnGetSessionValue( pvHSession,
					pcTag, pvData, puiSize));
	}

	FINLINE int fnGetGblValue(
		const char *		pcTag,
		void *				pvData,
		FLMSIZET *				puiSize)
	{
		return( gv_FlmSysData.HttpConfigParms.fnGetGblValue( pcTag,
			pvData, puiSize));
	}

	FINLINE int fnSetGblValue(
		const char *		pcTag,
		const void *		pvData,
		FLMSIZET				uiSize)
	{
		return( gv_FlmSysData.HttpConfigParms.fnSetGblValue( pcTag,
			pvData, uiSize));
	}

	FINLINE int fnRecvBuffer(
		void *		pvBuf,
		FLMSIZET *		puiBufSize)
	{
		return( gv_FlmSysData.HttpConfigParms.fnRecvBuffer( m_pHRequest,
			pvBuf, puiBufSize));
	}

	void setMembers( 
		HRequest *		pHRequest)
	{	
		m_pHRequest = pHRequest;

		// Get the session object for this page.
		m_uiSessionRC = acquireSession();
	}

	virtual RCODE display(
		FLMUINT			uiNumParams, 
		const char **	ppszParams) = 0;

	RCODE ExtractParameter(
		FLMUINT			uiNumParams, 
		const char **	ppszParams, 
		const char *	pszParamName, 
		FLMUINT			uiParamLen,
		char *			pszParam);

	FLMBOOL DetectParameter(
		FLMUINT			uiNumParams, 
		const char **	ppszParams, 
		const char *	pszParamName);

	RCODE getDatabaseHandleParam(
		FLMUINT			uiNumParams,
		const char **	ppszParams,
		F_Session *		pFlmSession,
		HFDB *			phDb,
		char *			pszKey = NULL);

	void FormatTime(
		FLMUINT		uiTimerUnits, 
		char *		pszFormattedTime);

	void popupFrame( void);

	RCODE writeUsage( 
		FLM_CACHE_USAGE *	pUsage,
		FLMBOOL				bRefresh,
		const char *		pszURL,
		const char *		pszTitle);

	FINLINE void stdHdr( void)
	{
		fnSetHdrValue( "Content-Type", "text/html");
		fnSetNoCache( NULL);
		fnSendHeader( HTS_OK);
	}

	void printHTMLLink(
		const char *		pszName,
		const char *		pszType,
		void *				pvBase,
		void *				pvAddress,
		void *				pvValue,
		const char *		pszLink,
		FLMBOOL				bHighlight = FALSE);

	void printHTMLString(
		const char *		pszName,
		const char *		pszType,
		void *				pvBase,
		void *				pvAddress,
		const char *		pszValue,
		FLMBOOL				bHighlight = FALSE);

	void printHTMLUint(
		const char *		pszName,
		const char *		pszType,
		void *				pvBase,
		void *				pvAddress,
		FLMUINT				uiValue,
		FLMBOOL				bHighlight = FALSE);

	void printHTMLInt(
		const char *		pszName,
		const char *		pszType,
		void *				pvBase,
		void *				pvAddress,
		FLMINT				iValue,
		FLMBOOL				bHighlight = FALSE);

	void printHTMLUlong(
		const char *		pszName,
		const char *		pszType,
		void *				pvBase,
		void *				pvAddress,
		unsigned long		luValue,
		FLMBOOL				bHighlight = FALSE);

	void printStyle( void);

	void printColumnHeading(
		const char *		pszHeading,
		JustificationType	eJustification = JUSTIFY_LEFT,
		const char *		pszBackground = NULL,
		FLMUINT				uiColSpan = 1,
		FLMUINT				uiRowSpan = 1,
		FLMBOOL				bClose = TRUE,
		FLMUINT				uiWidth = 0);

	void printColumnHeadingClose( void);

	void printEncodedString(
		const char *		pszString,
		FStringEncodeType	eEncodeType = HTML_ENCODING,
		FLMBOOL				bMapSlashes = TRUE);

	void printDocStart(
		const char *	pszTitle,
		FLMBOOL			bPrintTitle = TRUE,
		FLMBOOL			bStdHeader = TRUE,
		const char *	pszBackground = NULL);

	void printDocEnd( void);

	void printMenuReload( void);

	void printTableStart( 
		const char *	pszTitle,
		FLMUINT			uiColumns,
		FLMUINT			uiWidthFactor = 100);

	void printTableEnd( void);

	void printTableRowStart( 
		FLMBOOL		bHighlight = FALSE);

	void printTableRowEnd( void);

	void printTableDataStart(
		FLMBOOL				bNoWrap = TRUE,
		JustificationType	eJustification = JUSTIFY_LEFT,
		FLMUINT				uiWidth = 0);

	void printTableDataEnd( void);

	void printTableDataEmpty( void);

	void printErrorPage(
		RCODE				rc,
		FLMBOOL			bStdHeader = TRUE,
		const char *	pszWhat = "Unable to process request ... ");

	void printErrorPage(
		const char *	pszErrMsg,
		const char *	pszErrStr2 = NULL,
		FLMBOOL			bStdHeader = TRUE);

	RCODE getFormValueByName(
		const char *	pszValueTag,
		char **			ppszBuf,
		FLMUINT			uiBufLen,
		FLMUINT *		puiDataLen);

	void printDate(
		FLMUINT		uiGMTTime,
		char *		pszBuffer = NULL);

	void printYesNo(
		FLMBOOL		bYes);

	void printCommaNum(
		FLMUINT64			ui64Num,
		JustificationType	eJustify = JUSTIFY_RIGHT,
		FLMBOOL				bChangedValue = FALSE);

	void printCommaNumText(
		FLMUINT64			ui64Num);

	void printStartInputForm(
		const char *	pszFormName,
		const char *	pszPage,
		FLMUINT			uiFormValue);

	void printEndInputForm( void);

	void printButton(
			const char *	pszContents,
			ButtonTypes		eBType,
			const char *	pszName = NULL,
			const char *	pszValue = NULL,
			const char *	pszExtra = NULL,
			FLMBOOL			bDisabled = FALSE,
			FLMBYTE			ucAccessKey ='\0',
			FLMUINT			uiTabIndex = 0);
	// pszExtra is used for text that doesn't appear too often - event handler
	// scripts are a good example.  The text of pszExtra appears just before
	// the closing > of the <BUTTON> tag
	
	RCODE acquireSession( void);

	void releaseSession( void);

	void printSelectOption(
		FLMUINT			uiSelectedValue,
		FLMUINT			uiOptionValue,
		const char *	pszOptionName,
		FLMBOOL			bPrintOptionVal = TRUE);

	void printContainerPulldown(
		F_NameTable *	pNameTable,
		FLMUINT			uiSelectedContainer);

	void printFieldPulldown(
		F_NameTable *	pNameTable,
		FLMUINT			uiSelectedField);

	void printIndexPulldown(
		F_NameTable *	pNameTable,
		FLMUINT			uiSelectedIndex,
		FLMBOOL			bIncludeNoIndex = TRUE,
		FLMBOOL			bIncludeChooseBestIndex = TRUE,
		FLMBOOL			bPrintSelect = FALSE,
		const char *	pszExtra = NULL);


	void printRetrievalFlagsPulldown(
		FLMUINT			uiSelectedFlag = FO_EXACT);

	void printSetOperationScript( void);

	void printOperationButton(
		const char *	pszFormName,
		const char *	pszButtonLabel,
		const char *	pszButtonValue);

	void printStartCenter( void);

	void printEndCenter(
		FLMBOOL	bPrintCR = TRUE);

	void printSpaces(
		FLMUINT			uiCount);

	void printElapTime(
		FLMUINT64			ui64ElapTime,
		char *				pszBuffer = NULL,
		JustificationType	eJustify = JUSTIFY_RIGHT,
		FLMBOOL				bTimeIsMilli = TRUE);

	void printRecord(
		const char *		pszDbKey,
		FlmRecord *			pRec,
		F_NameTable *		pNameTable,
		FLMUINT *			puiContext,
		FLMBOOL				bReadOnly = TRUE,
		FLMUINT				uiSelectedField = 0,
		FLMUINT				uiFlags = FO_EXACT);

	void printRecordStyle( void);

	void printRecordScripts( void);

	void printTextField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);
		
	void printNumberField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);

	void printBinaryField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);

	void printContextField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);

	void printBlobField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);

	void printDefaultField(
		FlmRecord *			pRec,
		void *				pvField,
		FLMUINT				uiFieldCounter,
		FLMBOOL				bReadOnly);

	void printHiddenField(
		const char *		pszName,
		const char *		pszValue);

	void printHiddenField(
		const char *		pszName,
		FLMUINT				uiValue);

	void printFieldIds(
		FLMUINT			uiFieldCounter,
		FLMUINT			uiFieldLevel,
		FLMUINT			uiType,
		FLMUINT			uiTagNum);

	void printLogHeaders(
		FLMBYTE *		pucLastCommitted,
		FLMBYTE *		pucCheckpoint,
		FLMBYTE *		pucUncommitted);

	void printSerialNum(
		FLMBYTE *		pucSerialNum);

	void printLogFileEntryUDX(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLogFileEntryUD_X(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLogFileEntryUD(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLogFileEntryUW(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLogFileEntryUC(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLogFileEntryBool(
		FLMBYTE *		pucLog,
		FLMUINT			uiOffset);

	void printLanguagePulldown(
		FLMUINT			uiSelectedLang);

	RCODE displayLogFileHdr(
		const char *	pszPath);

protected:

	HRequest *			m_pHRequest;
	char *				m_pszFormData;
	char *				m_pszURLString;
	PRINTF_FN			fnPrintf;
	F_Session *			m_pFlmSession;
	RCODE					m_uiSessionRC;
};

/****************************************************************************
Desc: The ErrorPage class, which is displayed when no other classes 
		are found that satisfy the request
*****************************************************************************/
class F_ErrorPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:

	void printRandomHaiku();
};

/****************************************************************************
Desc: Page that is displayed when a URL is requested for a page requireing
		secure access but the Global security is not enabled or has expired.
*****************************************************************************/
class F_GblAccessPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

};

/****************************************************************************
Desc: Page that is displayed when a URL is requested for a page requireing
		secure access but the Session security is not enabled.  A password 
		is required.
*****************************************************************************/
class F_SessionAccessPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: The NameTableMgr class
*****************************************************************************/
#define TABLE_ARRAY_SIZE 10

class F_NameTableMgr : public F_Object
{
public:

	F_NameTableMgr();

	virtual ~F_NameTableMgr();

	F_NameTable * getNameTable(
		FFILE *		pFile);

	RCODE releaseNameTable(
		FFILE *		pFile);

private:

	FLMUINT initNameTable(
		FFILE *		pFile);

	struct
	{ 
		F_MUTEX			hMutex;
		FFILE *			pFile;
		FLMUINT			uiDictSeq;
		F_NameTable *	pNameTable;
	} m_tablearray[ TABLE_ARRAY_SIZE];

	IF_RandomGenerator *		m_pRandomGen;
};

/****************************************************************************
Desc:	The WebPageFactory class, which is used to create the various web 
		page classes on demand.
*****************************************************************************/
typedef F_WebPage * (* CREATE_FN)();

typedef struct
{
	const char *	pszName;
	CREATE_FN		fnCreate;
	FLMBOOL			bSecure;
} RegistryEntry;

#define FLM_SECURE_PASSWORD "SecureCoreDbPassword"
#define FLM_SECURE_EXPIRATION "SecureCoreDbExpiration"

class F_WebPageFactory : public F_Object
{
public:
	F_WebPageFactory() { sortRegistry(); }
	// Default destructor will do just fine...

	RCODE create( 
		const char *		pszName,
		F_WebPage **		ppPage,
		HRequest *			pHRequest);

	void Release( 
		F_WebPage **		ppPage);

	FINLINE FLMINT FLMAPI Release( void)
	{
		flmAssert( 0);
		return( 0);
	}

private:

	void	sortRegistry();

	FLMINT searchRegistry( 
		const char *		pszName);

	FLMBOOL isSecurePage( FLMUINT uiEntryNum)
	{
		return m_Registry[ uiEntryNum].bSecure;
	}

	FLMBOOL isSecurePasswordEntered(
		void *			pvSession);

	FLMBOOL isValidSecurePassword(
		const char *	pszData);

	FLMBOOL isSecureAccessEnabled();

	static RegistryEntry		m_Registry[];
	FLMUINT						m_uiNumEntries;
	static CREATE_FN			m_fnDefault;
	static CREATE_FN			m_fnError;
	static CREATE_FN			m_fnGblAccess;
	static CREATE_FN			m_fnSessionAccess;

	// m_Registry will contain a list of all names and creation functions
	// m_fnDefault is used for the "home page".  Specifically, if the create
	// function receives a pszName consisting of a single NULL, this 
	// function will be called.
	// m_fnError will contain only a pointer to be used only if none of the
	// entries in m_pRegistry match the requested name.
	// m_fnGblAccess will be used when a secure page is requested, but the secure
	// access is not enabled.
	//VISIT: What s m_fnSessionAccess used for???
	// m_fnSessionAccess will be used 
};

/****************************************************************************
Desc:	The class that displays the FDB structure
*****************************************************************************/
class F_FDBPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:

	void write_data(
		FDB *				pDb,
		const char *	pszFDBAddress,
		FLMUINT			uiBucket);
};

/****************************************************************************
Desc: The class that displays the FFILE structures.
*****************************************************************************/
typedef struct
{
		FLMUINT		SCacheBlkAddress;
		FLMUINT		SCacheLowTransID;
		FLMUINT		SCacheHighTransID;
		FLMUINT		PendingWriteBlkAddress;
		FLMUINT		PendingWriteLowTransID;
		FLMUINT		PendingWriteHighTransID;
		FLMUINT		LastDirtyBlkAddress;
		FLMUINT		LastDirtyLowTransID;
		FLMUINT		LastDirtyHighTransID;
		FLMUINT		FirstRecordContainer;
		FLMUINT		FirstRecordDrn;
		FLMUINT		FirstRecordLowTransId;
		FLMUINT		LastRecordContainer;
		FLMUINT		LastRecordDrn;
		FLMUINT		LastRecordLowTransId;
} DATASTRUCT;

class F_FFilePage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:

	void write_data(
		FFILE *				pFile,
		void *				pvFFileAddress,
		DATASTRUCT *		pDataStruct);
};

/****************************************************************************
Desc:	The class that displays the gv_FlmSysData.pFileHashTbl hash table.
*****************************************************************************/
class F_FileHashTblPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/*********************************************************
Desc:	Return HTML code that defines the welcome page frames.
		There are two framesets.  The first has one frame
		that references "Header.htm".  The second frameset
		has two frames.  The first frame references
		"Nav.htm" and "Welcome.htm".  This class is invoked
		following a successful login.
 **********************************************************/
class F_FrameMain : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char	**	ppszParams);
};

/*********************************************************
Desc: Return HTML code that defines the Header.htm frame.
**********************************************************/
class F_FrameHeader : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/*********************************************************
Desc:	Return HTML code that defines the Nav.htm frame.
**********************************************************/
class F_FrameNav : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/*********************************************************
Desc:	Return HTML code that defines the Welcome.htm frame.
**********************************************************/
class F_FrameWelcome : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

};

/****************************************************************************
Desc: The class that displays the gv_FlmSysData struct
*****************************************************************************/
class F_FlmSysDataPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:
	
	void write_data(
		FLMBOOL			bRefresh);
};

/****************************************************************************
Desc: The class that displays the HttpConfigParms struct
*****************************************************************************/
class F_HttpConfigParmsPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

};

/*********************************************************
 Desc:	Return HTML code that defines the SecureDbInfo
			popup window contents.
 **********************************************************/
class F_SecureDbInfo : public F_WebPage
{
public:
	RCODE display(
	FLMUINT				uiNumParams,
	const char **		ppszParams);

};

/****************************************************************************
Desc: The class that displays information about FLAIM threads
*****************************************************************************/
class F_FlmThreadsPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: The class that returns an image, applet, etc.
*****************************************************************************/
class F_HttpFile : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: Performs a database backup
*****************************************************************************/
class F_HttpDbBackup : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:

	static RCODE backupWriteHook(
		void *		pvBuffer,
		FLMUINT		uiBytesToWrite,
		void *		pvUserData);
};

/****************************************************************************
Desc: The class that displays information about FLAIM indexes
*****************************************************************************/
class F_FlmIndexPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: The class that allows interaction with a database
*****************************************************************************/
class F_DatabasePage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

	void printSessionDatabaseList(
		F_Session *		pFlmSession);

	void printGlobalDatabaseList( void);

private:

	void printDbOption(
		FLMBOOL			bOpenPopup,
		const char *	pszMenuOption,
		const char *	pszPage,
		const char *	pszUrlOption1,
		const char *	pszDbKey);

};

/****************************************************************************
Desc: Database configuration page
*****************************************************************************/
class F_DatabaseConfigPage : public F_WebPage
{
	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

	void outputValue(
		FLMBOOL *		pbHighlight,
		HFDB				hDb,
		const char *	pszDbKey,
  		FLMUINT			uiType,
		const char *	pszParamDescription,
		FLMUINT			uiConfigVal = 0);
};

/****************************************************************************
Desc: The class that allows interaction with records
*****************************************************************************/
class F_RecordMgrPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/*********************************************************
 Desc:	Return HTML code that defines the SecureDbAccess
			popup window contents.
 **********************************************************/
class F_SecureDbAccess : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: The class that displays the RCacheMgr struct
*****************************************************************************/
class F_RCacheMgrPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams);

private:

	void write_data( void);
};

/*********************************************************
Desc:	Displays the RCache structures
**********************************************************/
class F_RCachePage : public F_WebPage
{
public:
	
	RCODE display(
		FLMUINT			uiNumParams,
		const char **	pszParams);

private:

	void write_data(
		RCACHE *			pRCache);
};

/*********************************************************
Desc:	Displays a Record 
**********************************************************/
class F_RecordPage : public F_WebPage
{
public:

	F_RecordPage()
	{
		m_pFlmSession = NULL;
	}

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	pszParams);

private:

	void write_data(
		FlmRecord *				pRecord,
		RCACHE *					pRCache);

	void write_links(
		RCACHE *			pRCache);

	void printRecordFields(
		FlmRecord *		pRecord,
		RCACHE *			pRCache);
};

/*********************************************************
Desc: Displays the RCache Hash Table
**********************************************************/
class F_RCHashBucketPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	pszParams);
};

/****************************************************************************
Desc: A base class for SCache related stuff.  Mostly needed for
		the implementation of the locateSCacheBlock() function...
*****************************************************************************/
class F_SCacheBase : public F_WebPage
{
public:
	virtual RCODE display(
		FLMUINT			uiNumParams,
		const char **	ppszParams) = 0;

protected:
	
	RCODE locateSCacheBlock(
		FLMUINT			uiNumParams,
		const char **	ppszParams,
		SCACHE *			pLocalSCache,
		FLMUINT *		puiBlkAddress,
		FLMUINT *		puiLowTransID,
		FLMUINT *		puiHighTransID,
		FFILE * *		ppFile);

	void notFoundErr();
	
	void malformedUrlErr();
};

/****************************************************************************
Desc: Class for displaying SCache structs
*****************************************************************************/
class F_SCacheBlockPage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};


/****************************************************************************
Desc: Class for displaying SCache notify lists
*****************************************************************************/
class F_SCacheNotifyListPage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};


/****************************************************************************
Desc: Class for displaying SCache use lists
*****************************************************************************/
class F_SCacheUseListPage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams,
		const char **	ppszParams);
};

/****************************************************************************
Desc: Class for displaying the SCacheMgr struct
*****************************************************************************/
class F_SCacheMgrPage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);
};

/****************************************************************************
Desc: Class for displaying data blocks in the SCache
*****************************************************************************/
class F_SCacheDataPage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);
};


/****************************************************************************
Desc: Class for displaying the SCache Hash Table
*****************************************************************************/
class F_SCacheHashTablePage : public F_SCacheBase
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);
};

/****************************************************************************
Desc: Class for displaying a query
*****************************************************************************/
class F_QueryPage : public F_WebPage
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);
};

/****************************************************************************
Desc: Class for displaying a query's statistics
*****************************************************************************/
class F_QueryStatsPage : public F_WebPage
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);
};

/****************************************************************************
Desc: Class for displaying list of queries
*****************************************************************************/
class F_QueriesPage : public F_WebPage
{
public:

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

};

/****************************************************************************
Desc: Class for doing system configuration.
*****************************************************************************/
class F_SysConfigPage : public F_WebPage
{
public:

	F_SysConfigPage()
	{
		m_bHighlight = FALSE;
	}

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

private:

	void outputParams( void);

	FINLINE void beginRow( void)
	{
		printTableRowStart( m_bHighlight = !m_bHighlight);
	}

	FINLINE void endRow( void)
	{
		printTableRowEnd();
	}

	FINLINE void beginInputForm(
  		eFlmConfigTypes	eConfigType)
	{
		fnPrintf( m_pHRequest,
			"<form type=\"submit\" method=\"get\" action=\"%s/SysConfig\">\n"
			"<input name=\"Action\" type=\"hidden\" value=\"%u\">\n",
			m_pszURLString,
			(unsigned)eConfigType);
	}

	FINLINE void endInputForm( void)
	{
		fnPrintf( m_pHRequest, "</form>");
	}

	// A thin wrapper around the printButton function that puts the
	// button in its own element in a table row.
	
	void addSubmitButton(
		const char *		pszLabel)
	{
		printTableDataStart();
		printButton( pszLabel, BT_Submit);
		printTableDataEnd();
	}

	FINLINE void addStrInputField(
		eFlmConfigTypes	eConfigType,
		FLMUINT				uiMaxStrLen,
		const char *		pszFieldValue)
	{
		fnPrintf( m_pHRequest,
				"<TD><input name=\"U%u\" maxlength=\"%u\" "
				"type=\"text\" value=\"%s\"></TD>\n",
				(unsigned)eConfigType, (unsigned)uiMaxStrLen, pszFieldValue);
	}

	void outputButton(
	  	eFlmConfigTypes	eConfigType,
		const char *		pszParamAction,
		FLMUINT				uiValue1 = 0,
		FLMUINT				uiValue2 = 0);

	void outputUINT(
  		eFlmConfigTypes	eConfigType,
		const char *		pszParamDescription,
		FLMBOOL				bParamIsSettable = TRUE,
		FLMBOOL				bParamIsGettable = TRUE,
		FLMUINT				uiDefaultValue = 0);

	void outputBOOL(
  		eFlmConfigTypes	eConfigType,
		const char *		pszParamDescription,
		const char *		pszOnState = "Enabled",
		const char *		pszOffState = "Disabled",
		const char *		pszTurnOnAction = "Enable",
		const char *		pszTurnOffAction = "Disable");

	void outputString(
  		eFlmConfigTypes	eConfigType,
		const char *		pszParamDescription,
		FLMUINT				uiStrMaxLen,
		FLMBOOL				bParamIsSettable = TRUE,
		FLMBOOL				bParamIsGettable = TRUE,
		const char *		pszDefaultValue = "");

	RCODE getConfigValue(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams,
		char **				ppszValue,
		FLMUINT				uiMaxStrLen);

	RCODE getConfigValue(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams,
		FLMUINT				uiValueLen,
		char *				pszValue);

	RCODE configButton(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams);

	RCODE configUINT(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams);

	RCODE configBOOL(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams);

	RCODE configString(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams,
		FLMUINT				uiAllocLen);

	RCODE doConfig(
  		eFlmConfigTypes	eConfigType,
		FLMUINT				uiNumParams,
		const char **		ppszParams);

	FLMBOOL				m_bHighlight;
};

/****************************************************************************
Desc:	Statistics gather structure.
****************************************************************************/
typedef struct focusTag
{
	char								szFileName[50];
	FLMUINT							uiLFileNum;
} FOCUS_BLOCK, * FOCUS_BLOCK_p;

typedef struct LockUserHeader
{
	FLMBYTE							szFileName[50];
	F_LOCK_USER *					pDbLockUser;
	F_LOCK_USER *					pTxLockUser;
	struct LockUserHeader *		pNext;
} LOCK_USER_HEADER, * LOCK_USER_HEADER_p;

typedef struct CheckPointInfo
{
	FLMBYTE							szFileName[50];
	CHECKPOINT_INFO *				pCheckpointInfo;
	struct CheckPointInfo *		pNext;
} CP_INFO_HEADER, * CP_INFO_HEADER_p;

typedef struct StatGatherTag
{
	FLMBOOL					bCollectingStats;
	FLMUINT					uiStartTime;
	FLMUINT					uiStopTime;
	FLMUINT					uiNumDbStats;
	FLMUINT					uiNumLFileStats;
	F_COUNT_TIME_STAT		CommittedUpdTrans;
	F_COUNT_TIME_STAT		GroupCompletes;
	FLMUINT64				ui64GroupFinished;
	F_COUNT_TIME_STAT		AbortedUpdTrans;
	F_COUNT_TIME_STAT		CommittedReadTrans;
	F_COUNT_TIME_STAT		AbortedReadTrans;
	F_COUNT_TIME_STAT		Reads;
	F_COUNT_TIME_STAT		Adds;
	F_COUNT_TIME_STAT		Modifies;
	F_COUNT_TIME_STAT		Deletes;
	F_COUNT_TIME_STAT		Queries;
	F_COUNT_TIME_STAT		QueryReads;
	FLMUINT64				ui64BlockCombines;
	FLMUINT64				ui64BlockSplits;
	DISKIO_STAT				IOReads;
	DISKIO_STAT				IORootBlockReads;
	DISKIO_STAT				IONonLeafBlockReads;
	DISKIO_STAT				IOLeafBlockReads;
	DISKIO_STAT				IOAvailBlockReads;
	DISKIO_STAT				IOLFHBlockReads;
	DISKIO_STAT				IORollbackBlockReads;
	FLMUINT					uiReadErrors;
	FLMUINT					uiCheckErrors;
	DISKIO_STAT				IOWrites;
	DISKIO_STAT				IORootBlockWrites;
	DISKIO_STAT				IONonLeafBlockWrites;
	DISKIO_STAT				IOLeafBlockWrites;
	DISKIO_STAT				IOAvailBlockWrites;
	DISKIO_STAT				IOLFHBlockWrites;
	DISKIO_STAT				IORollBackLogWrites;
	DISKIO_STAT				IOLogHdrWrites;
	DISKIO_STAT				IORolledbackBlockWrites;
	FLMUINT					uiWriteErrors;
	F_LOCK_STATS			LockStats;
	FLM_CACHE_USAGE		BlockCache;
	FLM_CACHE_USAGE		RecordCache;
	FLMUINT					uiDirtyBlocks;
	FLMUINT					uiDirtyBytes;
	FLMUINT					uiLogBlocks;
	FLMUINT					uiLogBytes;
	FLMUINT					uiFreeCount;
	FLMUINT					uiFreeBytes;
	FLMUINT					uiReplaceableCount;
	FLMUINT					uiReplaceableBytes;
	CP_INFO_HEADER_p		pCPHeader;
	LOCK_USER_HEADER_p	pLockUsers;
} STAT_GATHER;

/****************************************************************************
Desc: Class for displaying statistics.
*****************************************************************************/
class F_StatsPage : public F_WebPage
{
public:

	F_StatsPage()
	{
		m_pFocusBlock = NULL;
	}

	virtual ~F_StatsPage()
	{
		if (m_pFocusBlock)
		{
			f_free( &m_pFocusBlock);
		}
	}

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

private:

	void gatherBlockIOStats(
		STAT_GATHER *		pStatGather,
		DISKIO_STAT *		pReadStat,
		DISKIO_STAT *		pWriteStat,
		BLOCKIO_STATS *	pBlockIOStats);

	void gatherLFileStats(
		STAT_GATHER *	pStatGather,
		LFILE_STATS *	pLFileStats);

	void gatherDbStats(
		STAT_GATHER *	pStatGather,
		DB_STATS *		pDbStats);

	void gatherStats(
		STAT_GATHER *	pStatGather);

	void displayStats(
		STAT_GATHER *	pStatGather,
		STAT_GATHER *	pOldStatGather,
		FLMUINT *		puiStatOrders);

	void printIORow(
		FLMBOOL			bHighlight,
		const char *	pszIOCategory,
		DISKIO_STAT *	pIOStat,
		DISKIO_STAT *	pOldIOStat);

	void printCountTimeRow(
		FLMBOOL					bHighlight,
		const char *			pszCategory,
		F_COUNT_TIME_STAT *	pStat,
		F_COUNT_TIME_STAT *	pOldStat,
		FLMBOOL					bPrintCountOnly = FALSE);

	void printCacheStatRow(
		FLMBOOL				bHighlight,
		const char *		pszCategory,
		FLMUINT				uiBlockCacheValue,
		FLMUINT				uiRecordCacheValue,
		FLMBOOL				bRecordCacheValueApplicable = TRUE,
		FLMBOOL				bBChangedValue = FALSE,
		FLMBOOL				bRChangedValue = FALSE);

	void formatStatsHeading(
		STAT_GATHER *		pStatGather,
		const char *		pszHeading);

	void freeLockUsers(
		STAT_GATHER *		pStatGather);

	void gatherLockStats(
		STAT_GATHER *			pStatGather,
		FFILE *					pFile);

	void gatherCPStats(
		STAT_GATHER *			pStatGather,
		FFILE *					pFile);

	void freeCPInfoHeaders(
		STAT_GATHER *		pStatGather);

	void printCacheStats(
		STAT_GATHER *		pStatGather,
		STAT_GATHER *		pOldStatGather);

	void printOperationStats(
		STAT_GATHER *		pStatGather,
		STAT_GATHER *		pOldStatGather);

	void printLockStats(
		STAT_GATHER *		pStatGather,
		STAT_GATHER *		pOldStatGather);

	void printDiskStats(
		STAT_GATHER *		pStatGather,
		STAT_GATHER *		pOldStatGather);

	void printCPStats(
		STAT_GATHER *		pStatGather);

	void displayFocus(
		FLMUINT				uiNumParams,
		const char **		ppszParams);

	RCODE setFocus(
		char *				pszFocus);

	FOCUS_BLOCK_p			m_pFocusBlock;
};

typedef struct QueryStatusTag
{
	FLMBOOL		bHaveQueryStatus;
	HFDB			hDb;
	FLMUINT		uiContainer;
	FLMUINT		uiIndex;
	FLMUINT		uiOptIndex;
	FLMUINT		uiIndexInfo;
	HFCURSOR		hCursor;
	FLMBOOL		bDoDelete;
	FLMBOOL		bStopQuery;
	FLMBOOL		bAbortQuery;
	FLMBOOL		bQueryRunning;
	FLMUINT		uiProcessedCnt;
	FLMUINT *	puiDrnList;
	FLMUINT		uiDrnListSize;
	FLMUINT		uiDrnCount;
	FLMUINT		uiLastTimeChecked;
	FLMUINT		uiQueryTimeout;
} QUERY_STATUS;

/****************************************************************************
Desc: Class for running queries.
*****************************************************************************/
class F_SelectPage : public F_WebPage
{
public:

	F_SelectPage()
	{
	}

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

private:

	void outputSelectForm(
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiContainer,
		FLMUINT			uiIndex,
		FLMBOOL			bQueryRunning,
		FLMUINT			uiQueryThreadId,
		F_NameTable *	pNameTable,
		const char *	pszQueryCriteria,
		QUERY_STATUS *	pQueryStatus);

	void outputQueryStatus(
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiContainer,
		F_NameTable *	pNameTable,
		QUERY_STATUS *	pQueryStatus);

	RCODE parseQuery(
		HFDB				hDb,
		FLMUINT			uiContainer,
		FLMUINT			uiIndex,
		F_NameTable *	pNameTable,
		const char *	pszQueryCriteria,
		HFCURSOR *		phCursor);

	RCODE runQuery(
		HFDB			hDb,
		FLMUINT		uiContainer,
		FLMUINT		uiIndex,
		HFCURSOR		hCursor,
		FLMBOOL		bDoDelete,
		FLMUINT *	puiQueryThreadId);

	void getQueryStatus(
		FLMUINT			uiQueryThreadId,
		FLMBOOL			bStopQuery,
		FLMBOOL			bAbortQuery,
		QUERY_STATUS *	pQueryStatus);
};

typedef struct CheckStatusTag
{
	FLMBOOL					bHaveCheckStatus;
	HFDB						hDb;
	RCODE						CheckRc;
	char *					pszDbName;
	char *					pszDataDir;
	char *					pszRflDir;
	char *					pszLogFileName;
	IF_FileHdl *			pLogFile;
	F_NameTable *			pNameTable;
	FLMBOOL					bCheckingIndexes;
	FLMBOOL					bRepairingIndexes;
	FLMBOOL					bDetailedStatistics;
	FLMBOOL					bStopCheck;
	FLMBOOL					bCheckRunning;
	FLMUINT					uiLastTimeBrowserChecked;
	FLMUINT					uiCheckTimeout;
	FLMUINT					uiCorruptCount;
	FLMUINT					uiOldViewCount;
	DB_CHECK_PROGRESS		Progress;
	FLMUINT					uiLastTimeSetStatus;
	FLMUINT					uiUpdateStatusInterval;
	IF_Thread *				pThread;
} CHECK_STATUS;

/****************************************************************************
Desc: Class for checking a database.
*****************************************************************************/
class F_CheckDbPage : public F_WebPage
{
public:

	F_CheckDbPage()
	{
	}

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

private:

	void outputStrParam(
		CHECK_STATUS *	pCheckStatus,
		FLMBOOL			bHighlight,
		const char *	pszParamName,
		const char *	pszFieldName,
		FLMUINT			uiMaxValueLen,
		const char *	pszFieldValue);

	void outputFlagParam(
		CHECK_STATUS *	pCheckStatus,
		FLMBOOL			bHighlight,
		const char *	pszParamName,
		const char *	pszFieldName,
		FLMBOOL			bFieldValue);

	void outputNum64Param(
		FLMBOOL			bHighlight,
		const char *	pszParamName,
		FLMUINT64		ui64Num);

	void outputCheckForm(
		HFDB				hDb,
		const char *	pszDbKey,
		CHECK_STATUS *	pCheckStatus,
		F_NameTable *	pNameTable,
		FLMUINT			uiCheckThreadId);

	RCODE runCheck(
		F_Session *		pFlmSession,
		HFDB *			phDb,
		char *			pszDbKey,
		const char *	pszDbName,
		const char *	pszDataDir,
		const char *	pszRflDir,
		const char *	pszLogFileName,
		FLMBOOL			bCheckingIndexes,
		FLMBOOL			bRepairingIndexes,
		FLMBOOL			bDetailedStatistics,
		FLMUINT *		puiCheckThreadId);

	void getCheckStatus(
		FLMUINT			uiQueryThreadId,
		FLMBOOL			bStopCheck,
		CHECK_STATUS *	pCheckStatus);
};

typedef struct KeyElementTag
{
	FlmRecord *	pKey;
	FLMUINT		uiRefStartOffset;
	FLMUINT		uiRefCnt;
} KEY_ELEMENT;

typedef struct IndexListStatusTag
{
	FLMBOOL					bHaveIndexListStatus;
	HFDB						hDb;
	FLMUINT					uiIndex;
	FlmRecord *				pFromKey;
	FlmRecord *				pUntilKey;
	FLMUINT					uiKeyCount;
	KEY_ELEMENT *			pKeyList;
	FLMUINT					uiKeyListSize;
	FLMUINT					uiRefCount;
	FLMUINT *				puiRefList;
	FLMUINT					uiRefListSize;
	FLMBOOL					bStopIndexList;
	FLMBOOL					bIndexListRunning;
	FLMUINT					uiLastTimeBrowserChecked;
	FLMUINT					uiIndexListTimeout;
	FLMUINT					uiLastTimeSetStatus;
	FLMUINT					uiUpdateStatusInterval;
	char						szEndStatus [80];
	IF_Thread *				pThread;
} IXLIST_STATUS;

/****************************************************************************
Desc: Class for listing an index's keys.
*****************************************************************************/
class F_IndexListPage : public F_WebPage
{
public:

	F_IndexListPage()
	{
	}

	RCODE display( 
		FLMUINT			uiNumParams, 
		const char **	ppszParams);

private:

	FLMBOOL getKey(
		HFDB				hDb,
		FLMUINT			uiIndex,
		FlmRecord **	ppKey,
		FLMUINT			uiKeyId);

	void outputKey(
		const char *	pszKeyName,
		HFDB				hDb,
		FLMUINT			uiIndex,
		FLMUINT			uiContainer,
		F_NameTable *	pNameTable,
		FlmRecord *		pKey,
		FLMUINT			uiRefCnt,
		FLMBOOL			bRunningIndexList,
		FLMUINT			uiKeyId);

	void outputIndexListForm(
		HFDB					hDb,
		const char *		pszDbKey,
		FLMUINT				uiIndex,
		FLMUINT				uiContainer,
		FLMUINT				uiIndexListThreadId,
		F_NameTable *		pNameTable,
		IXLIST_STATUS *	pIndexListStatus);

	RCODE runIndexList(
		HFDB			hDb,
		FLMUINT		uiIndex,
		FlmRecord *	pFromKey,
		FlmRecord *	pUntilKey,
		FLMUINT *	puiIndexListThreadId);

	void getIndexListStatus(
		FLMUINT				uiIndexListThreadId,
		FLMBOOL				bStopIndexList,
		IXLIST_STATUS *	pIndexListStatus);
};

/****************************************************************************
Desc:	Function to print the difference between two addresses
*****************************************************************************/
void printOffset(
	void *		pBase,
	void *		pAddress,
	char *		pszOffset);

/****************************************************************************
Desc:	Function to print an address as a string.
*****************************************************************************/
void printAddress(
	void *		pAddress,
	char *		pszBuff);

/****************************************************************************
Desc:	Utility class to help prepare a character buffer for printing when the
		output buffer size is not known in advance.
*****************************************************************************/
class F_DynamicBuffer : public F_Object
{
public:

	F_DynamicBuffer()
	{
		m_bSetup = FALSE;
		m_pucBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
		if (RC_OK( f_mutexCreate( &m_hMutex)))
		{
			m_bSetup = TRUE;
		}
	}

	virtual ~F_DynamicBuffer()
	{
		f_free( &m_pucBuffer);
		m_pucBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
		
		if (m_bSetup)
		{
			f_mutexDestroy( &m_hMutex);
			m_bSetup = FALSE;
		}
	}

	RCODE addChar( 
		char				ucCharacter);
	
	RCODE addString( 
		const char * 	pszString);
	
	const char * printBuffer( void);

	FLMUINT getBufferSize( void)
	{
		return m_uiUsedChars;
	}

	void reset( void)
	{
		f_free( &m_pucBuffer);
		m_pucBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
	}

private:

	FLMBOOL		m_bSetup;
	FLMBYTE *	m_pucBuffer;
	FLMUINT		m_uiBuffSize;
	FLMUINT		m_uiUsedChars;
	F_MUTEX		m_hMutex;
};


/*********************************************************
Desc:	Processes Record add, modify, delete, retrieve requests
		as well as Record field copy, insert (child & sibling) and
		clip requests.
**********************************************************/
class F_ProcessRecordPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	pszParams);

private:

	void addRecord(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	void newRecord(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	void modifyRecord(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	void deleteRecord(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	void retrieveRecord(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly,
		FLMUINT			uiFlag = 0xFFFFFFFF);

	void displayRecordPage(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FlmRecord *		pRec,
		FLMBOOL			bReadOnly,
		RCODE				uiRc);

	void copyField(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	void insertField(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly,
		FLMUINT			uiInsertAt);

	void clipField(
		F_Session *		pFlmSession,
		HFDB				hDb,
		const char *	pszDbKey,
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FLMBOOL			bReadOnly);

	RCODE constructRecord(
		FLMUINT			uiDrn,
		FLMUINT			uiContainer,
		FlmRecord **	ppRec,
		HFDB				hDb);

	RCODE storeBinaryField(
		FlmRecord *		pRec,
		void *			pvField,
		const char *	pszFldValue);

	RCODE storeUnicodeField(
		FlmRecord *		pRec,
		void *			pvField,
		const char *	pszFldValue);

	RCODE extractFieldInfo(
		FLMUINT			uiFieldCounter,
		char **			ppucBuf,
		FLMUINT *		puiLevel,
		FLMUINT *		puiType,
		FLMUINT *		puiTag);

	RCODE storeNumberField(
		FlmRecord *		pRec,
		void *			pvField,
		const char *	pszFldValue);

	RCODE storeBlobField(
		FlmRecord *		pRec,
		void *			pvField,
		const char *	pszFldValue,
		HFDB				hDb);

	RCODE copyFieldsFromTo(
		FlmRecord *		pRec,
		void *			pvOrigField,
		void *			pvMarkerField);
};

/*********************************************************
Desc:	Displays the Log Headers.
**********************************************************/
class F_LogHeaderPage : public F_WebPage
{
public:

	RCODE display(
		FLMUINT			uiNumParams,
		const char **	pszParams);
};

#include "fpackoff.h"

#endif
