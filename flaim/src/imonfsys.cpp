//-------------------------------------------------------------------------
// Desc:	Class for displaying the gv_FlmSysData structure in HTML on a web page.
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

#include "flaimsys.h"

/****************************************************************************
Desc:	This implements the display method of the F_FlmSysDataPage class
*****************************************************************************/
RCODE F_FlmSysDataPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bRefresh = FALSE;

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest,  "<html>\n");

	// Determine if we are being requested to refresh this page or  not.

	if ((bRefresh = DetectParameter( uiNumParams,
											   ppszParams, 
											   "Refresh")) == TRUE)
	{
		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=FlmSysData?Refresh\">"
			"<TITLE>Database iMonitor - gv_FlmSysData</TITLE>\n");
	
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD>\n");
	}
	
	printStyle();
	popupFrame();  //Spits out a Javascript function that will open a new window..
	fnPrintf( m_pHRequest, "</HEAD>\n");
	
	// Insert a new table into the page to display the gv_FlmSysData fields
	fnPrintf( m_pHRequest,  "<body>\n");

	write_data(bRefresh);
	
	fnPrintf( m_pHRequest,  "</body></html>\n");

	fnEmit();

	return( rc);
}

/****************************************************************************
 Desc: Generate the HTML that will display the contents of the gv_FlmSysData
		 structure.
 ****************************************************************************/
void F_FlmSysDataPage::write_data(
	FLMBOOL				bRefresh)
{
	RCODE			rc = FERR_OK;
	char *		pszTemp;
	char *		pszTemp2;
	char			szAddress[20];
	FLMBOOL		bHighlight = TRUE;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 150, &pszTemp2)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// If we are not to refresh this page, then don't include the
	// refresh meta command
	if (!bRefresh)
	{
		f_sprintf( (char *)pszTemp,
			       "<A HREF=%s/FlmSysData?Refresh>Start Auto-refresh (5 sec.)</A>",
					 m_pszURLString);
	}
	else
	{
		f_sprintf( (char *)pszTemp,
			       "<A HREF=%s/FlmSysData>Stop Auto-refresh</A>",
					 m_pszURLString);
	}

	// Print out a formal header and the refresh option.
	printTableStart("Database System Data", 4);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=%s/FlmSysData>Refresh</A>, ", 
								  m_pszURLString);
	fnPrintf( m_pHRequest, "%s\n", pszTemp);
	printColumnHeadingClose();
	printTableRowEnd();



	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();

	
	
	// pMrnuFile - Most recently used file address
	if (gv_FlmSysData.pMrnuFile)
	{
		printAddress( (void *)gv_FlmSysData.pMrnuFile, szAddress);
		f_sprintf( (char *)pszTemp, 
				   "%s/FFile?From=FlmSysData?Link=pMrnuFile?Address=%s",
					m_pszURLString,
				   szAddress);
	}
	
	printHTMLLink(
		"pMrnuFile",
		"FFILE *",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pMrnuFile,
		(void *)gv_FlmSysData.pMrnuFile,
		(char *)pszTemp,
		(bHighlight = !bHighlight));
	
	
	
	
	// pLrnuFile - Least recently used file address
	if (gv_FlmSysData.pLrnuFile)
	{
		printAddress( (void *)gv_FlmSysData.pLrnuFile, szAddress);
		f_sprintf( (char *)pszTemp, 
					"%s/FFile?From=FlmSysData?Link=pLrnuFile?Address=%s",
					m_pszURLString,
					szAddress);
	}
	
	printHTMLLink(
		"pLrnuFile",
		"FFILE *",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pLrnuFile,
		(void *)gv_FlmSysData.pLrnuFile,
		(char *)pszTemp,
		(bHighlight = !bHighlight));



	// pFileHashTbl - File name hash table
	f_sprintf( (char *)pszTemp, "%s/FileHashTbl",
		m_pszURLString);

	printHTMLLink(
		"pFileHashTbl",
		"FFILE *",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pFileHashTbl,
		(void *)gv_FlmSysData.pFileHashTbl,
		(char *)pszTemp,
		(bHighlight = !bHighlight));
	
	
	
	// hShareMutex - Shared File Mutex
	printAddress( (void *)&gv_FlmSysData.hShareMutex, szAddress);
	printHTMLString(
		"hShareMutex",
		"F_MUTEX",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.hShareMutex,
		(char *)szAddress,
		(bHighlight = !bHighlight));

	
	
	// pFileSystem - File system
	printAddress( (void *)gv_FlmSysData.pFileSystem, szAddress);
	printHTMLString(
		"pFileSystem",
		"F_FileSystem *",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pFileSystem,
		(char *)szAddress,
		(bHighlight = !bHighlight));




	// bTempDirSet - Temporary directory
	printHTMLString(
		"bTempDirSet",
		"FLMBOOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.bTempDirSet,
		(char *)(gv_FlmSysData.bTempDirSet ? "Yes" : "No"),
		(bHighlight = !bHighlight));

	
	
	
	// bCheckCache - Check cache
	printHTMLString(
		"bCheckCache",
		"FLMBOOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.bCheckCache,
		(char *)(gv_FlmSysData.bCheckCache ? "Yes" : "No"),
		(bHighlight = !bHighlight));


	
	// uiMaxCPInterval - Maximum checkpoint interval
	printHTMLUint(
		"uiMaxCPInterval",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxCPInterval,
		gv_FlmSysData.uiMaxCPInterval,
		(bHighlight = !bHighlight));	

	
	
	
	// uiMaxTransTime - Maximum Transaction Time
	printHTMLUint(
		"uiMaxTransTime",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxTransTime,
		gv_FlmSysData.uiMaxTransTime,
		(bHighlight = !bHighlight));	



	// uiMaxTransInactiveTime - Maximum transaction inactive time
	printHTMLUint(
		"uiMaxTransInactiveTime",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxTransInactiveTime,
		gv_FlmSysData.uiMaxTransInactiveTime,
		(bHighlight = !bHighlight));	




	// bDynamicCacheAdjust - Dynamic Cache Adjust
	printHTMLString(
		"bDynamicCacheAdjust",
		"FLMBOOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.bDynamicCacheAdjust,
		(char *)(gv_FlmSysData.bDynamicCacheAdjust ? "Yes" : "No"),
		(bHighlight = !bHighlight));




	// uiBlockCachePercentage - Block cache percentage
	printHTMLUint(
		"uiBlockCachePercentage",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiBlockCachePercentage,
		gv_FlmSysData.uiBlockCachePercentage,
		(bHighlight = !bHighlight));	




	// uiCacheAdjustPercent - Cache adjust percentage
	printHTMLUint(
		"uiCacheAdjustPercent",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheAdjustPercent,
		gv_FlmSysData.uiCacheAdjustPercent,
		(bHighlight = !bHighlight));	




	// uiCacheAdjustMin - Cache adjust minimum
	printHTMLUint(
		"uiCacheAdjustMin",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheAdjustMin,
		gv_FlmSysData.uiCacheAdjustMin,
		(bHighlight = !bHighlight));	



	// uiCacheAdjustMax - Cache Adjust Maximum
	printHTMLUint(
		"uiCacheAdjustMax",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheAdjustMax,
		gv_FlmSysData.uiCacheAdjustMax,
		(bHighlight = !bHighlight));	



	// uiCacheAdjustMinToLeave - Cache adjust minimum to leave
	printHTMLUint(
		"uiCacheAdjustMinToLeave",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheAdjustMinToLeave,
		gv_FlmSysData.uiCacheAdjustMinToLeave,
		(bHighlight = !bHighlight));	



	// uiCacheAdjustInterval - Cache adjust interval
	printHTMLUint(
		"uiCacheAdjustInterval",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheAdjustInterval,
		gv_FlmSysData.uiCacheAdjustInterval,
		(bHighlight = !bHighlight));	
	
	
	
	// uiCacheCleanupInterval - Cache Cleanup Interval
	printHTMLUint(
		"uiCacheCleanupInterval",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiCacheCleanupInterval,
		gv_FlmSysData.uiCacheCleanupInterval,
		(bHighlight = !bHighlight));	



	// uiUnusedCleanupInterval - Unused cleanup interval
	printHTMLUint(
		"uiUnusedCleanupInterval",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiUnusedCleanupInterval,
		gv_FlmSysData.uiUnusedCleanupInterval,
		(bHighlight = !bHighlight));	




	// SCacheMgr - Block Cache Manager
	f_sprintf( (char *)pszTemp, "%s/SCacheMgr",
		m_pszURLString);

	printHTMLLink(
		"SCacheMgr",
		"SCACHE_MGR",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.SCacheMgr,
		(void *)&gv_FlmSysData.SCacheMgr,
		(char *)pszTemp,
		(bHighlight = !bHighlight));

	
	
	
	// RCacheMgr - Record cache manager
	f_sprintf( (char *)pszTemp, "%s/RCacheMgr",
		m_pszURLString);

	printHTMLLink(
		"RCacheMgr",
		"RCACHE_MGR",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.RCacheMgr,
		(void *)&gv_FlmSysData.RCacheMgr,
		(char *)pszTemp,
		(bHighlight = !bHighlight));


	// pMonitorThrd - Monitor Thread
	f_sprintf( (char *)pszTemp, "%s/MonitorThrd",
		m_pszURLString);

	printHTMLLink(
		"pMonitorThrd",
		"F_Thread *",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pMonitorThrd,
		(void *)gv_FlmSysData.pMonitorThrd,
		(char *)pszTemp,
		(bHighlight = !bHighlight));



	// Stats
	f_sprintf( (char *)pszTemp, "<A HREF=\"javascript:openPopup('%s/Stats')\">Stats</A>",
		m_pszURLString);
	printAddress( (void *)&gv_FlmSysData.Stats, szAddress);
	f_sprintf( (char *)pszTemp2, "<A HREF=\"javascript:openPopup('%s/Stats')\">%s</A>",
		m_pszURLString, szAddress);

	printHTMLString(
		(char *)pszTemp,
		"FLM_STATS",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.Stats,
		(char *)pszTemp2,
		(bHighlight = !bHighlight));

	
	// hQueryMutex
	printAddress( (void *)&gv_FlmSysData.hQueryMutex, szAddress);
	printHTMLString(
		"hQueryMutex",
		"F_MUTEX",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.hQueryMutex,
		(char *)szAddress,
		(bHighlight = !bHighlight));

	// pNewestQuery
	printAddress( (void *)&gv_FlmSysData.pNewestQuery, szAddress);
	printHTMLString(
		"pNewestQuery",
		"QUERY_HDR_p",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pNewestQuery,
		(char *)szAddress,
		(bHighlight = !bHighlight));

	// pOldestQuery
	printAddress( (void *)&gv_FlmSysData.pOldestQuery, szAddress);
	printHTMLString(
		"pOldestQuery",
		"QUERY_HDR_p",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.pOldestQuery,
		(char *)szAddress,
		(bHighlight = !bHighlight));

	// uiQueryCnt
	printHTMLUint(
		"uiQueryCnt",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiQueryCnt,
		gv_FlmSysData.uiQueryCnt,
		(bHighlight = !bHighlight));	

	// uiMaxQueries
	printHTMLUint(
		"uiMaxQueries",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxQueries,
		gv_FlmSysData.uiMaxQueries,
		(bHighlight = !bHighlight));	

	// bNeedToUnsetMaxQueries
	printHTMLString(
		"bNeedToUnsetMaxQueries",
		"FLMBOOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.bNeedToUnsetMaxQueries,
		(char *)(gv_FlmSysData.bNeedToUnsetMaxQueries ? "Yes" : "No"),
		(bHighlight = !bHighlight));


	// bStatsInitialized - Statistics initialized
	printHTMLString(
		"bStatsInitialized",
		"FLMBOOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.bStatsInitialized,
		(char *)(gv_FlmSysData.bStatsInitialized ? "Yes" : "No"),
		(bHighlight = !bHighlight));

	
	
	
	// pszTempDir - Temporary Working Directory
	printHTMLString(
		"pszTempDir",
		"FLMBYTE",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.szTempDir[0],
		(char *)gv_FlmSysData.szTempDir,
		(bHighlight = !bHighlight));

	
	
	
	// uiMaxUnusedTime - Maximum unused structures time
	printHTMLUint(
		"uiMaxUnusedTime",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxUnusedTime,
		gv_FlmSysData.uiMaxUnusedTime,
		(bHighlight = !bHighlight));	


	
	
	// ucBlobExt - Blob Override Extension
	printHTMLString(
		"ucBlobExt",
		"FLMBYTE",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.ucBlobExt[0],
		(char *)gv_FlmSysData.ucBlobExt,
		(bHighlight = !bHighlight));

	
	
	
	// KRefPool - Update Pool
	printAddress( (void *)&gv_FlmSysData.KRefPool, szAddress);
	printHTMLString(
		"KRefPool",
		"POOL",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.KRefPool,
		(char *)szAddress,
		(bHighlight = !bHighlight));


	// HttpConfigParms
	f_sprintf( (char *)pszTemp, "<A HREF=\"javascript:openPopup('%s/HttpConfigParms')\">HttpConfigParms</A>",
		m_pszURLString);
	printAddress( (void *)&gv_FlmSysData.Stats, szAddress);
	f_sprintf( (char *)pszTemp2, "<A HREF=\"javascript:openPopup('%s/HttpConfigParms')\">%s</A>",
		m_pszURLString, szAddress);

	printAddress( (void *)&gv_FlmSysData.HttpConfigParms, szAddress);
	printHTMLString(
		(char *)pszTemp,
		"HTTPCONFIGPARMS",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.HttpConfigParms,
		(char *)pszTemp2,
		(bHighlight = !bHighlight));




	// uiMaxFileSize - Maximum File Size
	printHTMLUint(
		"uiMaxFileSize",
		"FLMUINT",
		(void *)&gv_FlmSysData,
		(void *)&gv_FlmSysData.uiMaxFileSize,
		gv_FlmSysData.uiMaxFileSize,
		(bHighlight = !bHighlight));

	printTableEnd();

Exit:
	
	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	if (pszTemp2)
	{
		f_free( &pszTemp2);
	}

	return;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_HttpConfigParmsPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	char			szAddress[20];
	char			szOffset[8];
	FLMBOOL		bHighlight = FALSE;

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	printDocStart(	"HttpConfigParams");
	printStyle();

	//printTableStart("HttpConfigParms", 4);
	fnPrintf( m_pHRequest, "<table border=0 cellpadding=2"
								  " cellspacing=0 width=100%%>\n");


	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	
	printAddress( (void *)gv_FlmSysData.HttpConfigParms.hMutex, szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.hMutex, szOffset);
	printTableRowStart();
	fnPrintf( m_pHRequest, TD_s " <TD>hMutex</TD>	<TD>F_MUTEX</TD>"	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.uiUseCount, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>uiUseCount</TD>	<TD>FLMUINT</TD> "	TD_lu,
				 szOffset, gv_FlmSysData.HttpConfigParms.uiUseCount);
	printTableRowEnd();

	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.pszURLString, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>pszURLString</TD>	<TD>FLMBYTE *</TD> "	TD_s,
				 szOffset, gv_FlmSysData.HttpConfigParms.pszURLString);
	printTableRowEnd();

	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.uiURLStringLen, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>uiURLStringLen</TD>	<TD>FLMUINT</TD> "	TD_lu,
				 szOffset, gv_FlmSysData.HttpConfigParms.uiURLStringLen);
	printTableRowEnd();

	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.bRegistered, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>bRegistered</TD>	<TD>FLMBOOL</TD> "	TD_s,
		szOffset, (char *)(gv_FlmSysData.HttpConfigParms.bRegistered ? "Yes" : "No"));
	printTableRowEnd();
	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReg), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReg, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReg</TD>	<TD>REG_URL_HANDLER_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnDereg), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnDereg, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnDereg</TD>	<TD>DEREG_URL_HANDLER_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReqPath), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReqPath, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReqPath</TD>	<TD>REQ_PATH_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReqQuery), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReqQuery, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReqQuery</TD>	<TD>REQ_QUERY_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReqHdrValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReqHdrValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReqHdrValue</TD>	<TD>REQ_HDR_VALUE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSetHdrValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSetHdrValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSetHdrValue</TD>	<TD>SET_HDR_VAL_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnPrintf), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnPrintf, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnPrintf</TD>	<TD>PRINTF_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnEmit), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnEmit, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnEmit</TD>	<TD>EMIT_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
				
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSetNoCache), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSetNoCache, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSetNoCache</TD>	<TD>SET_NO_CACHE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
				
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSendHeader), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSendHeader, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSendHeader</TD>	<TD>SEND_HDR_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();
				
	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSetIOMode), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSetIOMode, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSetIOMode</TD>	<TD>SET_IO_MODE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSendBuffer), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSendBuffer, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSendBuffer</TD>	<TD>SEND_BUFF_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnAcquireSession), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnAcquireSession, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnAcquireSession</TD>	<TD>ACQUIRE_SESSION_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReleaseSession), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReleaseSession, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReleaseSession</TD>	<TD>RELEASE_SESSION_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnAcquireUser), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnAcquireUser, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnAcquireUser</TD>	<TD>ACQUIRE_USER_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnReleaseUser), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnReleaseUser, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnReleaseUser</TD>	<TD>RELEASE_USER_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSetSessionValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSetSessionValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSetSessionValue</TD>	<TD>SET_SESSION_VALUE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnGetSessionValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnGetSessionValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnGetSessionValue</TD>	<TD>GET_SESSION_VALUE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnGetGblValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnGetGblValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnGetGblValue</TD>	<TD>GET_GBL_VALUE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnSetGblValue), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnSetGblValue, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnSetGblValue</TD>	<TD>SET_GBL_VALUE_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();

	printAddress( *((void **)&gv_FlmSysData.HttpConfigParms.fnRecvBuffer), szAddress);
	printOffset( &gv_FlmSysData.HttpConfigParms, &gv_FlmSysData.HttpConfigParms.fnRecvBuffer, szOffset);
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, TD_s " <TD>fnRecvBuffer</TD>	<TD>RECV_BUFFER_FN</TD> "	TD_s,
				 szOffset, szAddress);
	printTableRowEnd();


	printTableEnd();
	fnPrintf( m_pHRequest, " </BODY> </HTML>\n");
	fnEmit();


	return( rc);
}
