//-------------------------------------------------------------------------
// Desc:	Check a database via HTTP monitoring.
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

#define CHECK_FORM_NAME	"CheckForm"

#define DATABASE_NAME_FIELD		"databasename"
#define DATA_DIR_FIELD				"datadir"
#define RFL_DIR_FIELD				"rfldir"
#define LOG_FILE_NAME_FIELD		"logfilename"
#define CHECK_INDEXES_FIELD		"checkindexes"
#define REPAIR_INDEXES_FIELD		"repairindexes"
#define DETAILED_STATS_FIELD		"detailedstats"

FSTATIC void format64Num(
	FLMUINT64		ui64Num,
	char *			pszNum);

FSTATIC RCODE copyStr(
	char **			ppszDestStr,
	const char *	pszSrcStr);

FSTATIC void copyNames(
	CHECK_STATUS *	pDestCheckStatus,
	CHECK_STATUS *	pSrcCheckStatus);

FSTATIC void freeCheckStatus(
	CHECK_STATUS *	pCheckStatus,
	FLMBOOL			bFreeStruct);

FSTATIC void imonLogField(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiStartCol,
	FLMUINT			uiLevelOffset);

FSTATIC void imonLogKeyError(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	CORRUPT_INFO *	pCorrupt);

FSTATIC void imonLogCorruptError(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	CORRUPT_INFO *	pCorrupt);

FSTATIC RCODE CheckStatusCB(
	eStatusType	eStatus,
	void *		pvParm1,
	void *		pvParm2,
	void *		pvAppData);

FSTATIC RCODE FLMAPI imonDoCheck(
	IF_Thread *		pThread);

FSTATIC void imonLogStr(
	IF_FileHdl *	pLogFile,
	FLMUINT			uiIndent,
	const char *	pszStr);

/****************************************************************************
Desc:	Prints the web page for checking a database.
****************************************************************************/
RCODE F_CheckDbPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{

	RCODE				rc = FERR_OK;
	const char *	pszErrType = NULL;
	RCODE				runRc = FERR_OK;
	F_Session *		pFlmSession = m_pFlmSession;
	HFDB				hDb = HFDB_NULL;
	F_NameTable *	pNameTable = NULL;
	char				szTmp[ 32];
	char *			pszTmp;
	char *			pszOperation = NULL;
	char *			pszDbName = NULL;
	char *			pszDataDir = NULL;
	char *			pszRflDir = NULL;
	FLMBOOL			bCheckingIndexes = FALSE;
	FLMBOOL			bRepairingIndexes = FALSE;
	FLMBOOL			bDetailedStatistics = FALSE;
	char *			pszLogFileName = NULL;
	FLMBOOL			bPerformCheck = FALSE;
	FLMBOOL			bStopCheck = FALSE;
	FLMUINT			uiCheckThreadId;
	CHECK_STATUS	CheckStatus;
	char				szDbKey[ F_SESSION_DB_KEY_LEN];

	f_memset( &CheckStatus, 0, sizeof( CHECK_STATUS));

	// Acquire a FLAIM session

	if (!pFlmSession)
	{
		rc = RC_SET( m_uiSessionRC);
		goto ReportErrorExit;
	}

	// Get the database handle, if any

	if( RC_BAD( rc = getDatabaseHandleParam( uiNumParams, 
		ppszParams, pFlmSession, &hDb, szDbKey)))
	{
		hDb = HFDB_NULL;
	}
	else
	{
		if( IsInCSMode( hDb))
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto ReportErrorExit;
		}

		if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
		{
			goto ReportErrorExit;
		}
	}

	// Get the value of the Operation field, if present.

	getFormValueByName( "Operation", &pszOperation, 0, NULL);
	if (pszOperation)
	{
		if (f_stricmp( pszOperation, OPERATION_CHECK) == 0)
		{
			bPerformCheck = TRUE;
		}
		else if (f_stricmp( pszOperation, OPERATION_STOP) == 0)
		{
			bStopCheck = TRUE;
		}
	}

	// Get the database name, if any

	if (getFormValueByName( DATABASE_NAME_FIELD, &pszDbName, 0, NULL) == 0)
	{
		if (pszDbName && *pszDbName)
		{
			fcsDecodeHttpString( pszDbName);
		}
	}

	// Get the database directory, if any

	if (getFormValueByName( DATA_DIR_FIELD, &pszDataDir, 0, NULL) == 0)
	{
		if (pszDataDir && *pszDataDir)
		{
			fcsDecodeHttpString( pszDataDir);
		}
	}

	// Get the RFL directory, if any

	if (getFormValueByName( RFL_DIR_FIELD, &pszRflDir, 0, NULL) == 0)
	{
		if (pszRflDir && *pszRflDir)
		{
			fcsDecodeHttpString( pszRflDir);
		}
	}

	// Get the log file name, if any

	if (getFormValueByName( LOG_FILE_NAME_FIELD, &pszLogFileName, 0, NULL) == 0)
	{
		if (pszLogFileName && *pszLogFileName)
		{
			fcsDecodeHttpString( pszLogFileName);
		}
	}

	// Get the flag for whether or not to check indexes.

	szTmp [0] = 0;
	pszTmp = &szTmp [0];
	
	if( RC_BAD( getFormValueByName( CHECK_INDEXES_FIELD,
		&pszTmp, sizeof( szTmp), NULL)))
	{
		if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			CHECK_INDEXES_FIELD, sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (f_strcmp( szTmp, "yes") == 0)
	{
		bCheckingIndexes = TRUE;
	}

	// Get the flag for whether or not to repair indexes

	szTmp [0] = 0;
	pszTmp = &szTmp [0];
	if (RC_BAD( getFormValueByName( REPAIR_INDEXES_FIELD,
			&pszTmp, sizeof( szTmp), NULL)))
	{
		if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			REPAIR_INDEXES_FIELD, sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (f_strcmp( szTmp, "yes") == 0)
	{
		bRepairingIndexes = TRUE;
	}

	// Get the flag for whether or not to collect detailed statistics.

	szTmp [0] = 0;
	pszTmp = &szTmp [0];
	if (RC_BAD( getFormValueByName( DETAILED_STATS_FIELD,
			&pszTmp, sizeof( szTmp), NULL)))
	{
		if (RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			DETAILED_STATS_FIELD, sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (f_strcmp( szTmp, "yes") == 0)
	{
		bDetailedStatistics = TRUE;
	}

	// See if we had a check running.  Get the check thread ID
	// if any.

	szTmp [0] = '\0';
	uiCheckThreadId = 0;
	if (RC_OK( ExtractParameter( uiNumParams, ppszParams, 
		"Running", sizeof( szTmp), szTmp)))
	{
		if (szTmp [0])
		{
			uiCheckThreadId = f_atoud( szTmp);
			CheckStatus.bCheckRunning = TRUE;
		}
	}

	if (bPerformCheck)
	{

		// Better not have both bCheckRunning and bPerformCheck set!

		flmAssert( !CheckStatus.bCheckRunning);

		if (RC_BAD( runRc = runCheck( pFlmSession,
											&hDb, szDbKey, pszDbName, pszDataDir,
											pszRflDir, pszLogFileName,
											bCheckingIndexes, bRepairingIndexes,
											bDetailedStatistics, &uiCheckThreadId)))
		{
			pszErrType = "RUNNING CHECK";
		}
		else
		{
			CheckStatus.bCheckRunning = TRUE;
		}
	}

	// Stop the check, if requested, or get the check data.

	if (CheckStatus.bCheckRunning)
	{

		// getCheckStatus could change CheckStatus.bCheckRunning
		// to FALSE.

		getCheckStatus( uiCheckThreadId, bStopCheck, &CheckStatus);
	}

	// Output the web page.

	if (!CheckStatus.bCheckRunning && CheckStatus.bHaveCheckStatus)
	{

		// If we have check results, output a page for viewing/editing them.

		printDocStart( "Check Results");
	}
	else if (!CheckStatus.bCheckRunning)
	{
		printDocStart( "Run Check");
		if (pszErrType)
		{
			fnPrintf( m_pHRequest,
			"<br><font color=\"Red\">ERROR %04X (%s) %s</font><br><br>\n",
			(unsigned)runRc, FlmErrorString( runRc), pszErrType);
		}
	}
	else
	{
		stdHdr();
		fnPrintf( m_pHRequest, HTML_DOCTYPE);
		fnPrintf( m_pHRequest, "<html>\n"
									  "<head>\n");
		printStyle();

		// Output html that will cause a refresh to occur.

		fnPrintf( m_pHRequest, 
			"<META http-equiv=\"refresh\" content=\"3; "
			"url=%s/checkdb?Running=%u&dbhandle=%s\">"
			"<TITLE>Check Status</TITLE>\n",
			m_pszURLString, (unsigned)uiCheckThreadId, szDbKey);

		fnPrintf( m_pHRequest, "</head>\n"
									  "<body>\n");
	}

	// Output the form for entering the check parameters and
	// check status.

	outputCheckForm( hDb, szDbKey, &CheckStatus, pNameTable, uiCheckThreadId);

	// End the document

	printDocEnd();

Exit:

	fnEmit();

	if (pszOperation)
	{
		f_free( &pszOperation);
	}

	if (pszDbName)
	{
		f_free( &pszDbName);
	}

	if (pszDataDir)
	{
		f_free( &pszDataDir);
	}

	if (pszRflDir)
	{
		f_free( &pszRflDir);
	}

	if (pszLogFileName)
	{
		f_free( &pszLogFileName);
	}

	freeCheckStatus( &CheckStatus, FALSE);

	return( FERR_OK);

ReportErrorExit:

	printErrorPage( rc);
	goto Exit;
}

/****************************************************************************
Desc:	Output a string parameter.
****************************************************************************/
void F_CheckDbPage::outputStrParam(
	CHECK_STATUS *	pCheckStatus,
	FLMBOOL			bHighlight,
	const char *	pszParamName,
	const char *	pszFieldName,
	FLMUINT			uiMaxValueLen,
	const char *	pszFieldValue)
{
	printTableRowStart( bHighlight);

	printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
	fnPrintf( m_pHRequest, "%s", pszParamName);
	printTableDataEnd();

	printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
	if (pCheckStatus->bCheckRunning || !pszFieldName)
	{
		if (pszFieldValue)
		{
			printEncodedString( pszFieldValue, HTML_ENCODING);
		}
		else
		{
			fnPrintf( m_pHRequest, " ");
		}
	}
	else
	{
		fnPrintf( m_pHRequest,
				"<input name=\"%s\" maxlength=\"%u\" "
				"type=\"text\"",
				pszFieldName, (unsigned)uiMaxValueLen);
		if (pCheckStatus->bHaveCheckStatus &&
			 pszFieldValue && *pszFieldValue)
		{
			fnPrintf( m_pHRequest,
				" value=\"", pszFieldValue);
			printEncodedString( pszFieldValue, HTML_ENCODING);
			fnPrintf( m_pHRequest, "\">\n");
		}
		else
		{
			fnPrintf( m_pHRequest, ">\n");
		}
	}
	printTableDataEnd();
	printTableRowEnd();
}

/****************************************************************************
Desc:	Output a flag parameter.
****************************************************************************/
void F_CheckDbPage::outputFlagParam(
	CHECK_STATUS *	pCheckStatus,
	FLMBOOL			bHighlight,
	const char *	pszParamName,
	const char *	pszFieldName,
	FLMBOOL			bFieldValue)
{
	printTableRowStart( bHighlight);

	if (pCheckStatus->bCheckRunning)
	{
		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
		fnPrintf( m_pHRequest, "%s", pszParamName);
		printTableDataEnd();

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		fnPrintf( m_pHRequest, "%s", (char *)(bFieldValue ? "yes" : "no"));
		printTableDataEnd();
	}
	else
	{
		printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
		fnPrintf( m_pHRequest,
				"<input name=\"%s\" type=\"checkbox\"",
				pszFieldName);
		if (pCheckStatus->bHaveCheckStatus && bFieldValue)
		{
			fnPrintf( m_pHRequest, " checked");
		}
		fnPrintf( m_pHRequest, " value=\"yes\">&nbsp;%s\n",
			pszParamName);
		printTableDataEnd();

		printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
		fnPrintf( m_pHRequest, "&nbsp;");
		printTableDataEnd();
	}
	printTableRowEnd();
}

/****************************************************************************
Desc:	format a 64 bit number with commas.
****************************************************************************/
FSTATIC void format64Num(
	FLMUINT64	ui64Num,
	char *		pszNum
	)
{
	FLMUINT	uiNums [15];
	FLMUINT	uiNumNums = 0;
	FLMBOOL	bFirstNum;

	// Format the number with commas.

	do
	{
		uiNums [uiNumNums++] = (FLMUINT)(ui64Num % (FLMUINT64)1000);
		ui64Num /= 1000;
	} while (ui64Num);

	bFirstNum = TRUE;
	while (uiNumNums)
	{
		uiNumNums--;
		if (bFirstNum)
		{
			f_sprintf( pszNum, "%u", (unsigned)uiNums [uiNumNums]);
			bFirstNum = FALSE;
		}
		else
		{
			f_sprintf( pszNum, ",%03u", (unsigned)uiNums [uiNumNums]);
		}
		while (*pszNum)
		{
			pszNum++;
		}
	}
}

/****************************************************************************
Desc:	Output a flag parameter.
****************************************************************************/
void F_CheckDbPage::outputNum64Param(
	FLMBOOL			bHighlight,
	const char *	pszParamName,
	FLMUINT64		ui64Num)
{
	char	szNum [60];

	printTableRowStart( bHighlight);

	printTableDataStart( TRUE, JUSTIFY_LEFT, 35);
	fnPrintf( m_pHRequest, "%s", pszParamName);
	printTableDataEnd();

	// Format the number with commas

	format64Num( ui64Num, szNum);

	// Output the number

	printTableDataStart( TRUE, JUSTIFY_LEFT, 65);
	fnPrintf( m_pHRequest, "%s", szNum);
	printTableDataEnd();

	printTableRowEnd();
}

/****************************************************************************
Desc:	Output the form for the user to run a check.
****************************************************************************/
void F_CheckDbPage::outputCheckForm(
	HFDB					hDb,
	const char *		pszDbKey,
	CHECK_STATUS *		pCheckStatus,
	F_NameTable *		pNameTable,
	FLMUINT				uiCheckThreadId)
{
	FLMBOOL			bHighlight = FALSE;
	char				szTmp [128];
	char *			pszTmp;
	char *			pszName;
	IF_FileHdl *	pFileHdl = NULL;

	fnPrintf( m_pHRequest, "<form name=\""
		CHECK_FORM_NAME "\" type=\"submit\" "
	  "method=\"post\" action=\"%s/checkdb", m_pszURLString);
	if (pCheckStatus->bCheckRunning)
	{
		fnPrintf( m_pHRequest, "?Running=%u&dbhandle=%s\">\n",
			(unsigned)uiCheckThreadId, pszDbKey);
	}
	else
	{
		if (hDb != HFDB_NULL)
		{
			fnPrintf( m_pHRequest, "?dbhandle=%s\">\n",
				pszDbKey);
		}
		else
		{
			fnPrintf( m_pHRequest, "\">\n");
		}
	}

	printStartCenter();

	if (pCheckStatus->bCheckRunning)
	{
		printTableStart( "CHECK PROGRESS", 2, 75);
	}
	else if (pCheckStatus->bHaveCheckStatus)
	{
		printTableStart( "CHECK RESULTS", 2, 75);
	}
	else
	{
		printTableStart( "CHECK PARAMETERS", 2, 74);
	}

	// Column headers

	printTableRowStart();
	printColumnHeading( "Parameter", JUSTIFY_LEFT, NULL, 1, 1, TRUE, 35);
	printColumnHeading( "Value", JUSTIFY_LEFT, NULL, 1, 1, TRUE, 65);
	printTableRowEnd();

	if (hDb == HFDB_NULL)
	{

		// Output the database name

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"Database Name", DATABASE_NAME_FIELD,
				F_PATH_MAX_SIZE + 1, pCheckStatus->pszDbName);

		// Output the data directory

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"Data Directory", DATA_DIR_FIELD,
				F_PATH_MAX_SIZE + 1, pCheckStatus->pszDataDir);

		// Output the rfl directory

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"RFL Directory", RFL_DIR_FIELD,
				F_PATH_MAX_SIZE + 1, pCheckStatus->pszRflDir);
	}
	else
	{
		FDB *	pDb = (FDB *)hDb;

		// Output the database name

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"Database Name", NULL, 0, pDb->pFile->pszDbPath);

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"Data Directory", NULL, 0, pDb->pFile->pszDataDir);
	}

	// Output the log file name

	outputStrParam( pCheckStatus, bHighlight = !bHighlight,
			"Log File Name", LOG_FILE_NAME_FIELD,
			F_PATH_MAX_SIZE + 1, pCheckStatus->pszLogFileName);

	// Output the checking indexes flag.

	outputFlagParam( pCheckStatus, bHighlight = !bHighlight,
			"Check Indexes", CHECK_INDEXES_FIELD,
			pCheckStatus->bCheckingIndexes);

	// Output the repairing indexes flag.

	outputFlagParam( pCheckStatus, bHighlight = !bHighlight,
			"Repair Indexes", REPAIR_INDEXES_FIELD,
			pCheckStatus->bRepairingIndexes);

#if 0
	// Output the collecting detailed stats flag.

	outputFlagParam( pCheckStatus, bHighlight = !bHighlight,
			"Detailed Statistics", DETAILED_STATS_FIELD,
			pCheckStatus->bDetailedStatistics);
#endif

	if (pCheckStatus->bHaveCheckStatus)
	{

		// Output what we are currently doing

		switch (pCheckStatus->Progress.iCheckPhase)
		{
			case CHECK_LFH_BLOCKS:
				pszTmp = (char *)"LFH BLOCKS";
				break;
			case CHECK_B_TREE:
				pszTmp = &szTmp [0];
				if (pCheckStatus->Progress.uiLfType == LF_INDEX)
				{
					if (pCheckStatus->Progress.bUniqueIndex)
					{
						f_strcpy( pszTmp, "UNIQUE INDEX: ");
					}
					else
					{
						f_strcpy( pszTmp, "INDEX: ");
					}
				}
				else
				{
					f_strcpy( pszTmp, "CONTAINER: ");
				}
				pszName = &pszTmp [f_strlen( pszTmp)];
				if (!pNameTable ||
					 !pNameTable->getFromTagNum( pCheckStatus->Progress.uiLfNumber,
											NULL, pszName,
											sizeof( szTmp) - (pszName - &szTmp [0])))
				{
					f_sprintf( pszName, "#%u",
						(unsigned)pCheckStatus->Progress.uiLfNumber);
				}
				else
				{
					f_sprintf( &pszTmp [f_strlen( pszTmp)], " (%u)",
						(unsigned)pCheckStatus->Progress.uiLfNumber);
				}
				break;
			case CHECK_AVAIL_BLOCKS:
				pszTmp = (char *)"AVAIL BLOCKS";
				break;
			case CHECK_RS_SORT:
				pszTmp = (char *)"SORTING INDEX KEYS";
				break;
			default:
				pszTmp = &szTmp [0];
				f_sprintf( pszTmp, "UNKNOWN: %u",
					(unsigned)pCheckStatus->Progress.iCheckPhase);
				break;
		}

		outputStrParam( pCheckStatus, bHighlight = !bHighlight,
			"Doing", NULL, 0, pszTmp);

		// Output various statistics as we go

		outputNum64Param( bHighlight = !bHighlight, "Database Size",
			pCheckStatus->Progress.ui64DatabaseSize);
		if (pCheckStatus->Progress.iCheckPhase == CHECK_RS_SORT)
		{
			FLMUINT		uiPercent = 0;

			if (pCheckStatus->Progress.ui64NumRSUnits > (FLMUINT64)0)
			{
				uiPercent = 
						(FLMUINT)((pCheckStatus->Progress.ui64NumRSUnitsDone *
										(FLMUINT64)100) /
							pCheckStatus->Progress.ui64NumRSUnits);
			}
			outputNum64Param( bHighlight = !bHighlight, 
				"Percent Sorted", (FLMUINT64)uiPercent);
		}
		else
		{
			outputNum64Param( bHighlight = !bHighlight, "Bytes Checked",
				pCheckStatus->Progress.ui64BytesExamined);
		}
		outputNum64Param( bHighlight = !bHighlight, "Total Index Keys",
				pCheckStatus->Progress.ui64NumKeys);
		outputNum64Param( bHighlight = !bHighlight, "Num. Keys Checked",
				pCheckStatus->Progress.ui64NumKeysExamined);
		outputNum64Param( bHighlight = !bHighlight, "Invalid Index Keys",
				pCheckStatus->Progress.ui64NumKeysNotFound);
		outputNum64Param( bHighlight = !bHighlight, "Missing Index Keys",
				pCheckStatus->Progress.ui64NumRecKeysNotFound);
		outputNum64Param( bHighlight = !bHighlight, "Non-unique Index Keys",
				pCheckStatus->Progress.ui64NumNonUniqueKeys);
		outputNum64Param( bHighlight = !bHighlight, "Key Conflicts",
				pCheckStatus->Progress.ui64NumConflicts);
		outputNum64Param( bHighlight = !bHighlight, "Total Corruptions",
				(FLMUINT64)pCheckStatus->uiCorruptCount);
		outputNum64Param( bHighlight = !bHighlight, "Problems Repaired",
				(FLMUINT64)pCheckStatus->Progress.uiNumProblemsFixed);
		outputNum64Param( bHighlight = !bHighlight, "Old View Count",
				(FLMUINT64)pCheckStatus->uiOldViewCount);

		// Output the return status if the check is finished.

		if (!pCheckStatus->bCheckRunning)
		{
			if (pCheckStatus->CheckRc == FERR_OK)
			{
				pszTmp = (char *)"Database OK";
			}
			else if (pCheckStatus->CheckRc == FERR_USER_ABORT)
			{
				pszTmp = (char *)"User Halted";
			}
			else
			{
				pszTmp = &szTmp [0];
				f_sprintf( pszTmp, "Error %04X, (%s)",
					(unsigned)pCheckStatus->CheckRc,
					FlmErrorString( pCheckStatus->CheckRc));
			}
			outputStrParam( pCheckStatus, bHighlight = !bHighlight,
				"Check Status", NULL, 0, pszTmp);
		}
	}

	// End the table

	printTableEnd();

	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output the setOperation function

	printSetOperationScript();

	printStartCenter();
	if (!pCheckStatus->bCheckRunning)
	{

		// If we are not running a check, add a Perform Check button

		printOperationButton( CHECK_FORM_NAME,
			"Perform Check", OPERATION_CHECK);
	}
	else
	{

		// If we are running a check, output a stop button.

		printOperationButton( CHECK_FORM_NAME,
			"Stop Check", OPERATION_STOP);
	}
	printEndCenter( TRUE);

	// Close the form

	fnPrintf( m_pHRequest, "</form>\n");

	// If the check is done, and we have a log file, output it.

	if (!pCheckStatus->bCheckRunning && pCheckStatus->bHaveCheckStatus &&
		 pCheckStatus->uiCorruptCount && pCheckStatus->pszLogFileName)
	{
		fnPrintf( m_pHRequest, "<br><br><pre>------LOG FILE CONTENTS------\n");

		// Open the log file

		if (RC_OK( gv_FlmSysData.pFileSystem->openFile( 
			pCheckStatus->pszLogFileName, FLM_IO_RDWR | FLM_IO_SH_DENYNONE,
			&pFileHdl)))
		{
			RCODE		rc;
			FLMUINT	uiBytesRead;

			// Read and output until we run out of data

			for (;;)
			{
				if (RC_BAD( rc = pFileHdl->read( FLM_IO_CURRENT_POS,
						sizeof( szTmp) - 1, szTmp, &uiBytesRead)))
				{
					if (rc != FERR_IO_END_OF_FILE || !uiBytesRead)
					{
						break;
					}
				}
				if (uiBytesRead)
				{
					szTmp [uiBytesRead] = 0;
					fnPrintf( m_pHRequest, "%s", szTmp);
				}
				if (uiBytesRead < sizeof( szTmp) - 1)
				{
					break;
				}
			}
			
			pFileHdl->Release();
			pFileHdl = NULL;
		}
		
		fnPrintf( m_pHRequest, "\n------END OF LOG FILE------\n");
		fnPrintf( m_pHRequest, "</pre>\n");
	}
	
	if( pFileHdl)
	{
		pFileHdl->Release();
	}
}

/****************************************************************************
Desc:	Copy one string into another - allocating memory if needed.
****************************************************************************/
FSTATIC RCODE copyStr(
	char **			ppszDestStr,
	const char *	pszSrcStr)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLen;

	if (pszSrcStr && *pszSrcStr)
	{
		uiLen = f_strlen( pszSrcStr) + 1;
		if (RC_BAD( rc = f_alloc( uiLen, ppszDestStr)))
		{
			goto Exit;
		}
		f_memcpy( *ppszDestStr, pszSrcStr, uiLen);
	}
	else
	{
		*ppszDestStr = NULL;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copy the database names, etc. from one check status to another.
****************************************************************************/
FSTATIC void copyNames(
	CHECK_STATUS *	pDestCheckStatus,
	CHECK_STATUS *	pSrcCheckStatus)
{
	(void)copyStr( &pDestCheckStatus->pszDbName,
		pSrcCheckStatus->pszDbName);
	(void)copyStr( &pDestCheckStatus->pszDataDir,
		pSrcCheckStatus->pszDataDir);
	(void)copyStr( &pDestCheckStatus->pszRflDir,
		pSrcCheckStatus->pszRflDir);
	(void)copyStr( &pDestCheckStatus->pszLogFileName,
		pSrcCheckStatus->pszLogFileName);
}

/****************************************************************************
Desc:	Free a CHECK_STATUS structure and all associated memory.
****************************************************************************/
FSTATIC void freeCheckStatus(
	CHECK_STATUS *	pCheckStatus,
	FLMBOOL			bFreeStruct
	)
{
	f_free( &pCheckStatus->pszDbName);
	f_free( &pCheckStatus->pszDataDir);
	f_free( &pCheckStatus->pszRflDir);
	f_free( &pCheckStatus->pszLogFileName);

	if (bFreeStruct)
	{
		if (pCheckStatus->hDb != HFDB_NULL)
		{
			FlmDbClose( &pCheckStatus->hDb);
		}
		
		if (pCheckStatus->pLogFile)
		{
			pCheckStatus->pLogFile->Release();
			pCheckStatus->pLogFile = NULL;
		}
		
		if (pCheckStatus->pNameTable)
		{
			pCheckStatus->pNameTable->Release();
			pCheckStatus->pNameTable = NULL;
		}
		
		f_free( &pCheckStatus);
	}
}

/****************************************************************************
Desc:	Run a database check.
****************************************************************************/
RCODE F_CheckDbPage::runCheck(
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
	FLMUINT *		puiCheckThreadId)
{
	RCODE					rc = FERR_OK;
	CHECK_STATUS *		pCheckStatus = NULL;
	IF_Thread *			pThread;
	HFDB					hDb = HFDB_NULL;
	FDB *					pDb;

	if (*phDb == HFDB_NULL)
	{

		// Open the database

		if (RC_BAD( rc = FlmDbOpen( pszDbName, pszDataDir, pszRflDir,
											  0, NULL, phDb)))
		{
			goto Exit;
		}
		else
		{

			// Insert the handle into the session

			if (RC_BAD( rc = pFlmSession->addDbHandle( *phDb, pszDbKey)))
			{
				FlmDbClose( phDb);
				goto Exit;
			}
		}
	}
	else
	{
		pDb = (FDB *)(*phDb);
		pszDbName = pDb->pFile->pszDbPath;
		if ((pszDataDir = pDb->pFile->pszDataDir) != NULL)
		{
			if (!(*pszDataDir))
			{
				pszDataDir = NULL;
			}
		}
		pszRflDir = NULL;
	}

	// Open the database for the thread - so it doesn't have
	// to worry about the handle going away.  The thread will close the
	// new handle when it exits.
	
	if (RC_BAD( rc = flmOpenFile( ((FDB *)(*phDb))->pFile, NULL, NULL, NULL,
							0, TRUE, NULL, NULL,
							(((FDB *)(*phDb))->pFile)->pszDbPassword, &pDb)))
	{
		goto Exit;
	}
	hDb = (HFDB)pDb;

	// Create an object to track the check.

	if (RC_BAD( rc = f_calloc( sizeof( CHECK_STATUS), &pCheckStatus)))
	{
		goto Exit;
	}
	pCheckStatus->hDb = hDb;

	// Set hDb to HFDB_NULL so it won't be closed below.

	hDb = HFDB_NULL;

	// Copy database names.

	if (RC_BAD( rc = copyStr( &pCheckStatus->pszDbName,
								pszDbName)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = copyStr( &pCheckStatus->pszDataDir,
								pszDataDir)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = copyStr( &pCheckStatus->pszRflDir,
								pszRflDir)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = copyStr( &pCheckStatus->pszLogFileName,
								pszLogFileName)))
	{
		goto Exit;
	}

	// Create the log file, if one was specified.

	if (pCheckStatus->pszLogFileName)
	{
		gv_FlmSysData.pFileSystem->deleteFile( pCheckStatus->pszLogFileName);
		
		if (RC_BAD( gv_FlmSysData.pFileSystem->createFile(
			pCheckStatus->pszLogFileName, FLM_IO_RDWR | FLM_IO_SH_DENYNONE,
			&pCheckStatus->pLogFile)))
		{
			f_free( &pCheckStatus->pszLogFileName);
		}
	}

	// Get a name table for the database - if we can.

	if ((pCheckStatus->pNameTable = f_new F_NameTable) != NULL)
	{
		if (RC_BAD( pCheckStatus->pNameTable->setupFromDb( hDb)))
		{
			pCheckStatus->pNameTable->Release();
			pCheckStatus->pNameTable = NULL;
		}
	}

	pCheckStatus->bCheckingIndexes = bCheckingIndexes;
	pCheckStatus->bRepairingIndexes = bRepairingIndexes;
	pCheckStatus->bDetailedStatistics = bDetailedStatistics;

	pCheckStatus->bCheckRunning = TRUE;
	pCheckStatus->uiLastTimeBrowserChecked = FLM_GET_TIMER();

	// If browser does not check status at least every 15 seconds, we will
	// assume it has gone away and the thread will terminate itself.

	pCheckStatus->uiCheckTimeout = FLM_SECS_TO_TIMER_UNITS( 15);

	// Start a thread to do the check.

	if (RC_BAD( rc = f_threadCreate( &pThread, imonDoCheck,
							"IMON DB CHECK", gv_uiDbThrdGrp, 1,
							(void *)pCheckStatus, (void *)hDb)))
	{
		goto Exit;
	}

	*puiCheckThreadId = pThread->getThreadId();
	
	// Set pCheckStatus to NULL so it won't be freed below.  The thread
	// will free it when it stops.

	pCheckStatus = NULL;

Exit:

	if (pThread)
	{
		pThread->Release();
	}

	if (pCheckStatus)
	{
		freeCheckStatus( pCheckStatus, TRUE);
	}

	if (hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( rc);

}

/****************************************************************************
Desc:	Output the current thread status to the web page.
****************************************************************************/
void F_CheckDbPage::getCheckStatus(
	FLMUINT			uiCheckThreadId,
	FLMBOOL			bStopCheck,
	CHECK_STATUS *	pCheckStatus)
{
	FLMUINT			uiThreadId;
	IF_Thread *		pThread = NULL;
	CHECK_STATUS *	pThreadCheckStatus;
	FLMBOOL			bMutexLocked = FALSE;

	// pCheckStatus->bHaveCheckStatus should be set to FALSE by the caller.

	flmAssert( !pCheckStatus->bHaveCheckStatus);

	// See if the thread is still running.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	uiThreadId = 0;
	for (;;)
	{
		if (RC_BAD( gv_FlmSysData.pThreadMgr->getNextGroupThread( &pThread,
						gv_uiDbThrdGrp, &uiThreadId)))
		{
			pCheckStatus->bCheckRunning = FALSE;
			goto Exit;
		}
		if (uiThreadId == uiCheckThreadId)
		{

			// If the app ID is zero, the thread is on its way out or already
			// out.  Can no longer get thread status.

			if (!pThread->getThreadAppId())
			{
				pCheckStatus->bCheckRunning = FALSE;
				goto Exit;
			}

			// Found thread, get its check status data

			pThreadCheckStatus = (CHECK_STATUS *)pThread->getParm1();
			pThreadCheckStatus->uiLastTimeBrowserChecked = FLM_GET_TIMER();

			// Tell the thread to stop the check before telling it
			// to stop.  This is so we can get partial results.

			if (bStopCheck)
			{
				pThreadCheckStatus->bStopCheck = TRUE;

				// Go into a while loop, waiting for the thread
				// to finish its check.

				while (pThreadCheckStatus->bCheckRunning)
				{
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					bMutexLocked = FALSE;
					f_sleep( 200);
					f_mutexLock( gv_FlmSysData.hShareMutex);
					bMutexLocked = TRUE;

					// If the thread app ID goes to zero, it has been
					// told to shut down, and has either already gone
					// away or is in the process of doing so, in which
					// case pThreadCheckStatus has either already been
					// deleted, or will be - so it is not safe to access
					// it any more!

					if (!pThread->getThreadAppId())
					{
						pCheckStatus->bCheckRunning = FALSE;
						goto Exit;
					}
				}
			}

			break;
		}
		pThread->Release();
		pThread = NULL;
	}

	// Mutex better still be locked at this point.

	flmAssert( bMutexLocked);

	// Note that we test pThreadCheckStatus->bCheckRunning BEFORE
	// doing the memcpy.  This is because puiDrnList is not guaranteed
	// to be set until bCheckRunning is FALSE.  If bCheckRunning is TRUE,
	// we will NULL out whatever got copied into puiDrnList.

	if (!pThreadCheckStatus->bCheckRunning)
	{
		f_memcpy( pCheckStatus, pThreadCheckStatus, sizeof( CHECK_STATUS));
		copyNames( pCheckStatus, pThreadCheckStatus);

		// Need to unlock the mutex so that the thread can stop.

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		pThread->stopThread();
	}
	else
	{
		f_memcpy( pCheckStatus, pThreadCheckStatus, sizeof( CHECK_STATUS));
		copyNames( pCheckStatus, pThreadCheckStatus);

		// Set bCheckRunning to TRUE.  This takes care of a race
		// race condition of pThreadCheckStatus->bCheckRunning getting
		// set to FALSE by the check thread after we test it above.
		// we make the test on pThreadCheckStatus->bCheckRunning.  We will
		// simply get that fact next time we get status.

		pCheckStatus->bCheckRunning = TRUE;
	}

	// NULL out certain members so we won't attempt to use them.  They
	// may go away if the background thread has gone away.

	pCheckStatus->hDb = HFDB_NULL;
	pCheckStatus->pLogFile = NULL;
	pCheckStatus->pNameTable = NULL;

	pCheckStatus->bHaveCheckStatus = TRUE;

Exit:
	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if (pThread)
	{
		pThread->Release();
	}
}

/********************************************************************
Desc:	Log a string to the log file.
*********************************************************************/
FSTATIC void imonLogStr(
	IF_FileHdl *	pLogFile,
	FLMUINT			uiIndent,
	const char *	pszStr)
{
	char		szBuffer [100];
	FLMUINT	uiLoop;
	FLMUINT	uiBytesWritten;

	if ((uiLoop = uiIndent) != 0)
	{
		f_memset( szBuffer, ' ', uiIndent);
		uiLoop = uiIndent;
	}
	if (pszStr)
	{
		while (*pszStr)
		{
			if (uiLoop == sizeof( szBuffer))
			{
				pLogFile->write( FLM_IO_CURRENT_POS, uiLoop, 
					szBuffer, &uiBytesWritten);
				uiLoop = 0;
			}
			szBuffer [uiLoop++] = *pszStr;
			pszStr++;
		}
	}
	if (uiLoop >= sizeof( szBuffer) - 2)
	{
		pLogFile->write( FLM_IO_CURRENT_POS,
			uiLoop, szBuffer, &uiBytesWritten);
		uiLoop = 0;
	}
	szBuffer [uiLoop++] = '\r';
	szBuffer [uiLoop++] = '\n';
	pLogFile->write( FLM_IO_CURRENT_POS,
		uiLoop, szBuffer, &uiBytesWritten);
}

/***************************************************************************
Desc:	Log a field's data.
*****************************************************************************/
FSTATIC void imonLogField(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiStartCol,
	FLMUINT			uiLevelOffset)
{
	char			szTmpBuf [200];
	char *		pszTmp;
	FLMUINT		uiFieldNum;
	FLMUINT		uiLen;
	FLMUINT		uiBinLen;
	FLMUINT		uiTmpLen;
	FLMBYTE *	pucTmp;
	FLMBYTE		ucTmpBin [80];
	FLMUINT		uiNum;
	FLMUINT		uiLevel = pRecord->getLevel( pvField) + uiLevelOffset;
	FLMUINT		uiIndent = (uiLevel * 2) + uiStartCol;

	// Insert leading spaces to indent for level

	if (uiIndent)
	{
		f_memset(szTmpBuf, ' ', uiIndent);
	}

	// Output level and tag

	f_sprintf( &szTmpBuf [uiIndent], "%u ", (unsigned)uiLevel);
	pszTmp = &szTmpBuf [f_strlen( szTmpBuf)];
	uiFieldNum = pRecord->getFieldID( pvField);
	if (!pNameTable ||
		 !pNameTable->getFromTagNum( uiFieldNum,
								NULL, pszTmp,
								sizeof( szTmpBuf) - (pszTmp - &szTmpBuf [0])))
	{
		f_sprintf( pszTmp, "#%u", (unsigned)uiFieldNum);
	}

	// Output what will fit of the value on the rest of the line

	uiLen = f_strlen( szTmpBuf);
	szTmpBuf [uiLen++] = ' ';
	szTmpBuf [uiLen] = 0;
	if (!pRecord->getDataLength( pvField))
	{
		goto Exit;
	}
	switch (pRecord->getDataType( pvField))
	{
		case FLM_TEXT_TYPE:
			pszTmp = &szTmpBuf [uiLen];
			uiLen = 80 - uiLen;
			pRecord->getNative( pvField, pszTmp, &uiLen);
			break;
		case FLM_NUMBER_TYPE:
			pRecord->getUINT( pvField, &uiNum);
			f_sprintf( &szTmpBuf [uiLen], "%u", (unsigned)uiNum);
			break;
		case FLM_BINARY_TYPE:
			pRecord->getBinaryLength( pvField, &uiBinLen);
			uiTmpLen = sizeof( ucTmpBin);
			pRecord->getBinary( pvField, ucTmpBin, &uiTmpLen);
			pucTmp = &ucTmpBin [0];
			while (uiBinLen && uiLen < 77)
			{
				f_sprintf( &szTmpBuf [uiLen], "%02X ", (unsigned)*pucTmp);
				uiBinLen--;
				pucTmp++;
				uiLen += 3;
			}
			szTmpBuf [uiLen - 1] = 0;
			break;
		case FLM_CONTEXT_TYPE:
			pRecord->getUINT( pvField, &uiNum);
			f_sprintf( &szTmpBuf[ uiLen], "@%u@", (unsigned)uiNum);
			break;
	}

Exit:

	imonLogStr( pLogFile, 0, szTmpBuf);
}

/********************************************************************
Desc:	Log an index key corruption error.
*********************************************************************/
FSTATIC void imonLogKeyError(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	CORRUPT_INFO *	pCorrupt)
{
	FLMUINT		uiLogItem;
	FlmRecord *	pRecord = NULL;
	void *		pvField;
	REC_KEY *	pTempKeyList = NULL;
	FLMUINT		uiIndent;
	FLMUINT		uiLevelOffset;
	char			szNameBuf [128];
	char			szTmpBuf [128];
	
	if (!pNameTable ||
		 !pNameTable->getFromTagNum( pCorrupt->uiErrLfNumber,
								NULL, szNameBuf, sizeof( szNameBuf)))
	{
		f_sprintf( (char *)szNameBuf, "#%u", (unsigned)pCorrupt->uiErrLfNumber);
	}
	imonLogStr( pLogFile, 0, NULL);
	imonLogStr( pLogFile, 0, NULL);
	
	f_sprintf( szTmpBuf, "ERROR IN INDEX: %s", szNameBuf);
	imonLogStr( pLogFile, 0, szTmpBuf);
	
	uiLogItem = 'R';
	uiLevelOffset = 0;
	for (;;)
	{
		uiIndent = 2;
		if (uiLogItem == 'K')
		{
			if ((pRecord = pCorrupt->pErrIxKey) == NULL)
			{
				uiLogItem = 'L';
				continue;
			}
			imonLogStr( pLogFile, 0, NULL);
			imonLogStr( pLogFile, 0, " PROBLEM KEY");
		}
		else if (uiLogItem == 'R')
		{
			if ((pRecord = pCorrupt->pErrRecord) == NULL)
			{
				uiLogItem = 'K';
				continue;
			}
			imonLogStr( pLogFile, 0, NULL);
			imonLogStr( pLogFile, 0, " RECORD");
		}
		else if (uiLogItem == 'L')
		{
			if ((pTempKeyList =
				pCorrupt->pErrRecordKeyList) == NULL)
			{
				break;
			}
			pRecord = pTempKeyList->pKey;
			imonLogStr( pLogFile, 0, NULL);
			imonLogStr( pLogFile, 0, " RECORD KEYS");
			imonLogStr( pLogFile, 0, "  0 Key");
			uiLevelOffset = 1;
		}

		for (pvField = pRecord->root();;)
		{
			if (!pvField)
			{
				if (uiLogItem != 'L')
				{
					break;
				}
				if ((pTempKeyList = pTempKeyList->pNextKey) == NULL)
				{
					break;
				}
				pRecord = pTempKeyList->pKey;
				pvField = pRecord->root();
				imonLogStr( pLogFile, 0, "  0 Key");
				continue;
			}
			else
			{
				imonLogField( pLogFile, pNameTable, pRecord, pvField,
							uiIndent, uiLevelOffset);
			}
			pvField = pRecord->next( pvField);
		}

		if (uiLogItem == 'L')
		{
			break;
		}
		else if (uiLogItem == 'R')
		{
			uiLogItem = 'K';
		}
		else
		{
			uiLogItem = 'L';
		}
	}
}

/********************************************************************
Desc:	Log corruptions to log file.
*********************************************************************/
FSTATIC void imonLogCorruptError(
	IF_FileHdl *	pLogFile,
	F_NameTable *	pNameTable,
	CORRUPT_INFO *	pCorrupt)
{
	char	szWhat [20];
	char	szTmpBuf [100];

	switch (pCorrupt->eErrLocale)
	{
		case LOCALE_LFH_LIST:
			imonLogStr( pLogFile, 0, "ERROR IN LFH LINKED LIST:");
			break;
		case LOCALE_AVAIL_LIST:
			imonLogStr( pLogFile, 0, "ERROR IN AVAIL LINKED LIST:");
			break;
		case LOCALE_B_TREE:
			if (pCorrupt->eCorruption == FLM_OLD_VIEW)
			{
				imonLogStr( pLogFile, 0, "OLD VIEW");
			}
			else
			{
				if (pCorrupt->uiErrFieldNum)
				{
					f_strcpy( szWhat, "FIELD");
				}
				else if (pCorrupt->uiErrElmOffset)
				{
					f_strcpy( szWhat, "ELEMENT");
				}
				else if (pCorrupt->uiErrBlkAddress)
				{
					f_strcpy( szWhat, "BLOCK");
				}
				else
				{
					f_strcpy( szWhat, "LAST BLOCK");
				}
				f_sprintf( szTmpBuf, "BAD %s", szWhat);
				imonLogStr( pLogFile, 0, szTmpBuf);
			}

			// Log the logical file number, name, and type

			f_sprintf( szTmpBuf, "Logical File Number: %u",
				(unsigned)pCorrupt->uiErrLfNumber);
			imonLogStr( pLogFile, 2, szTmpBuf);
			
			switch( pCorrupt->uiErrLfType)
			{
				case LF_CONTAINER:
					f_strcpy( szWhat, "Container");
					break;
				case LF_INDEX:
					f_strcpy( szWhat, "Index");
					break;
				default:
					f_sprintf( (char *)szWhat, "?%u", 
							(unsigned)pCorrupt->uiErrLfType);
					break;
			}
			f_sprintf( szTmpBuf, "Logical File Type: %s", szWhat);
			imonLogStr( pLogFile, 2, szTmpBuf);

			// Log the level in the B-Tree, if known

			if (pCorrupt->uiErrBTreeLevel != 0xFF)
			{
				f_sprintf( szTmpBuf, "Level in B-Tree: %u",
					(unsigned)pCorrupt->uiErrBTreeLevel);
				imonLogStr( pLogFile, 2, szTmpBuf);
			}
			break;
		case LOCALE_IXD_TBL:
			f_sprintf( szTmpBuf, "ERROR IN IXD TABLE, Index Number: %u",
				(unsigned)pCorrupt->uiErrLfNumber);
			imonLogStr( pLogFile, 0, szTmpBuf);
			break;
		case LOCALE_INDEX:
			f_strcpy( szWhat, "Index");
			imonLogKeyError( pLogFile, pNameTable, pCorrupt);
			break;
		default:
			pCorrupt->eErrLocale = LOCALE_NONE;
			break;
	}

	// Log the block address, if known

	if (pCorrupt->uiErrBlkAddress)
	{
		f_sprintf( szTmpBuf, "Block Address: 0x%08X (%u)",
			(unsigned)pCorrupt->uiErrBlkAddress,
			(unsigned)pCorrupt->uiErrBlkAddress);
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	// Log the parent block address, if known

	if (pCorrupt->uiErrParentBlkAddress)
	{
		if (pCorrupt->uiErrParentBlkAddress != 0xFFFFFFFF)
		{
			f_sprintf( szTmpBuf, "Parent Block Address: 0x%08X (%u)",
				(unsigned)pCorrupt->uiErrParentBlkAddress,
				(unsigned)pCorrupt->uiErrParentBlkAddress);
		}
		else
		{
			f_sprintf( szTmpBuf,
				"Parent Block Address: NONE, Root Block");
		}
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	// Log the element offset, if known

	if (pCorrupt->uiErrElmOffset)
	{
		f_sprintf( szTmpBuf, "Element Offset: %u", 
				(unsigned)pCorrupt->uiErrElmOffset);
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	// Log the record number, if known

	if (pCorrupt->uiErrDrn)
	{
		f_sprintf( szTmpBuf, 
			"Record Number: %u", (unsigned)pCorrupt->uiErrDrn);
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	// Log the offset within the element record, if known

	if (pCorrupt->uiErrElmRecOffset != 0xFFFF)
	{
		f_sprintf( szTmpBuf, "Offset Within Element: %u",
			(unsigned)pCorrupt->uiErrElmRecOffset);
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	// Log the field number, if known

	if (pCorrupt->uiErrFieldNum)
	{
		f_sprintf( szTmpBuf, 
				"Field Number: %u", (unsigned)pCorrupt->uiErrFieldNum);
		imonLogStr( pLogFile, 2, szTmpBuf);
	}

	f_strcpy( szTmpBuf, FlmVerifyErrToStr( pCorrupt->eCorruption));
	f_sprintf( &szTmpBuf[ f_strlen( szTmpBuf)], " (%d)",
		(int)pCorrupt->eCorruption);
	imonLogStr( pLogFile, 2, szTmpBuf);
	imonLogStr( pLogFile, 0, NULL);
	
	pLogFile->flush();
}

/***************************************************************************
Desc:	Check status callback.
***************************************************************************/
FSTATIC RCODE CheckStatusCB(
	eStatusType	eStatus,
	void *		pvParm1,
	void *		pvParm2,
	void *		pvAppData)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCurrTime;
	CORRUPT_INFO *	pCorrupt;
	CHECK_STATUS *	pCheckStatus = (CHECK_STATUS *)pvAppData;

	uiCurrTime = FLM_GET_TIMER();
	if (pCheckStatus->bStopCheck)
	{
		rc = RC_SET( FERR_USER_ABORT);
		goto Exit;
	}
	else if (FLM_ELAPSED_TIME( uiCurrTime,
					pCheckStatus->uiLastTimeBrowserChecked) >=
			pCheckStatus->uiCheckTimeout)
	{
		rc = RC_SET( FERR_TIMEOUT);
		goto Exit;
	}

	// Handle each of the status types

	if (eStatus == FLM_PROBLEM_STATUS)
	{
		FLMBOOL *	pbFixCorruptions = (FLMBOOL *)pvParm2;

		pCorrupt = (CORRUPT_INFO *)pvParm1;
		if (pCheckStatus->pLogFile &&
			 pCorrupt->eCorruption != FLM_OLD_VIEW)
		{
			imonLogCorruptError( pCheckStatus->pLogFile,
							pCheckStatus->pNameTable, pCorrupt);
		}

		if (pCorrupt->eCorruption == FLM_OLD_VIEW)
		{
			pCheckStatus->uiOldViewCount++;
		}
		else
		{
			pCheckStatus->uiCorruptCount++;
		}
		if (pbFixCorruptions)
		{
			*pbFixCorruptions = pCheckStatus->bRepairingIndexes;
		}
	}
	else if (eStatus == FLM_CHECK_STATUS)
	{

		// Capture the progress information.

		f_memcpy( &pCheckStatus->Progress, pvParm1,
						sizeof( DB_CHECK_PROGRESS));

		// Update thread status

		if (FLM_ELAPSED_TIME( uiCurrTime, pCheckStatus->uiLastTimeSetStatus) >=
					pCheckStatus->uiUpdateStatusInterval)
		{
			if (pCheckStatus->Progress.iCheckPhase == CHECK_RS_SORT)
			{
				FLMUINT		uiPercent = 0;

				if (pCheckStatus->Progress.ui64NumRSUnits > (FLMUINT64)0)
				{
					uiPercent = 
							(FLMUINT)((pCheckStatus->Progress.ui64NumRSUnitsDone *
											(FLMUINT64)100) /
								pCheckStatus->Progress.ui64NumRSUnits);
				}
				pCheckStatus->pThread->setThreadStatus( "Sorting, %u percent done",
					(unsigned)uiPercent);
			}
			else
			{
				char	szFileSize [60];
				char	szBytesDone [60];

				format64Num( pCheckStatus->Progress.ui64DatabaseSize, szFileSize);
				format64Num( pCheckStatus->Progress.ui64BytesExamined, szBytesDone);
				pCheckStatus->pThread->setThreadStatus( "%s of %s bytes checked",
					szBytesDone, szFileSize);
			}
			pCheckStatus->uiLastTimeSetStatus = uiCurrTime;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Thread to perform a database check for a web page.
****************************************************************************/
FSTATIC RCODE FLMAPI imonDoCheck(
	IF_Thread *		pThread)
{
	RCODE						rc;
	CHECK_STATUS *			pCheckStatus = (CHECK_STATUS *)pThread->getParm1();
	FLMUINT					uiFlags;
	F_Pool					pool;
	DB_CHECK_PROGRESS		CheckProgress;
	FLMUINT					uiCurrTime;

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);
	pCheckStatus->pThread = pThread;

	pCheckStatus->uiUpdateStatusInterval = FLM_SECS_TO_TIMER_UNITS( 5);

	uiFlags = FLM_CHK_FIELDS;
	if (pCheckStatus->bCheckingIndexes)
	{
		uiFlags |= FLM_CHK_INDEX_REFERENCING;
	}
	pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);

	pool.poolInit( 512);
	
	rc = FlmDbCheck( pCheckStatus->hDb, NULL, NULL, NULL, uiFlags,
					&pool, &CheckProgress, CheckStatusCB, pCheckStatus);
	
	pool.poolFree();

	// Close the database and log file before doing anything else.

	FlmDbClose( &pCheckStatus->hDb);
	if (pCheckStatus->pLogFile)
	{
		pCheckStatus->pLogFile->Release();
		pCheckStatus->pLogFile = NULL;
	}

	pCheckStatus->CheckRc = rc;
	pCheckStatus->bCheckRunning = FALSE;

	if (RC_BAD( rc))
	{
		if (rc == FERR_USER_ABORT)
		{

			// Callback forced us to quit.

			pThread->setThreadStatus( "User halted");
		}
		else if (rc == FERR_TIMEOUT)
		{
			pThread->setThreadStatus( "Timed out");
			goto Exit;
		}
		else
		{
			pThread->setThreadStatus( "Check Error %04X,", (unsigned)rc);
		}
	}

	// Wait for the user to tell us to quit.

	for (;;)
	{

		// See if we should shut down. 

		if (pThread->getShutdownFlag())
		{

			// Transaction will be aborted below

			pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);
			goto Exit;
		}

		// See if we timed out

		uiCurrTime = FLM_GET_TIMER();
		if (FLM_ELAPSED_TIME( uiCurrTime,
						pCheckStatus->uiLastTimeBrowserChecked) >=
				pCheckStatus->uiCheckTimeout)
		{
			goto Exit;
		}

		// Pause one second

		pThread->sleep( 1000);
	}

Exit:

	// Set the thread's app ID to 0, so that it will not
	// be found now that the thread is terminating (we don't
	// want getCheckStatus() to find the thread).

	pThread->setThreadAppId( 0);

	// Free the check status.  Must do inside mutex lock so
	// that it doesn't go away after getCheckStatus finds the
	// thread.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	freeCheckStatus( pCheckStatus, TRUE);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	return( FERR_OK);
}
