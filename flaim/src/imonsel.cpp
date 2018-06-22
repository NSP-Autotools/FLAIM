//-------------------------------------------------------------------------
// Desc:	Class for doing queries via web pages - for monitoring stuff.
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

#define MAX_RECORDS_TO_OUTPUT		100
#define DRN_LIST_INCREASE_SIZE	4096

#define SELECT_FORM_NAME	"SelectForm"

#define QUERY_CRITERIA_LEN	"querycriterialen"
#define QUERY_CRITERIA		"querycriteria"

FSTATIC RCODE queryStatusCB(
	FLMUINT	uiStatusType,
	void *	pvParm1,
	void *	pvParm2,
	void *	pvUserData);

FSTATIC RCODE FLMAPI imonDoQuery(
	IF_Thread *		pThread);

/****************************************************************************
Desc:	Prints the web page for running queries.
****************************************************************************/
RCODE F_SelectPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	const char *		pszErrType = NULL;
	RCODE					runRc = FERR_OK;
	F_Session *			pFlmSession = m_pFlmSession;
	HFDB					hDb;
	FLMUINT				uiContainer;
	FLMUINT				uiIndex;
	char					szTmp [32];
	char *				pszTmp;
	char *				pszQueryCriteria = NULL;
	char					szQueryCriteria [100];
	char *				pszOperation = NULL;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bPerformQuery = FALSE;
	FLMBOOL				bDoDelete = FALSE;
	FLMBOOL				bStopQuery = FALSE;
	FLMBOOL				bAbortQuery = FALSE;
	HFCURSOR				hCursor = HFCURSOR_NULL;
	FLMUINT				uiQueryThreadId;
	QUERY_STATUS		QueryStatus;
	void *				pvHttpSession;
	FLMUINT				uiCriteriaLen;
	FLMUINT				uiSize;
	char					szDbKey[ F_SESSION_DB_KEY_LEN];

	QueryStatus.bQueryRunning = FALSE;
	QueryStatus.bHaveQueryStatus = FALSE;

	// Check the FLAIM session

	if( !pFlmSession)
	{
		rc = RC_SET( m_uiSessionRC);
		goto ReportErrorExit;
	}

	// Get the database handle, if any

	if( RC_BAD( rc = getDatabaseHandleParam( uiNumParams, 
		ppszParams, pFlmSession, &hDb, szDbKey)))
	{
		goto ReportErrorExit;
	}

	if (RC_BAD( rc = pFlmSession->getNameTable( hDb, &pNameTable)))
	{
		goto ReportErrorExit;
	}

	// Get the container, if any - look in the form first - because
	// it can be in both the form and the header.  The one in the form
	// takes precedence over the one in the header.

	szTmp [0] = '\0';
	uiContainer = 0;
	pszTmp = &szTmp [0];
	if (RC_BAD( getFormValueByName( "container", 
			&pszTmp, sizeof( szTmp), NULL)))
	{
		if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			"container", sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (szTmp[ 0])
	{
		uiContainer = f_atoud( szTmp);
	}

	// Get the index, if any - look in the form first - because
	// it can be in both the form and the header.  The one in the form
	// takes precedence over the one in the header.

	szTmp [0] = '\0';
	uiIndex = FLM_SELECT_INDEX;
	pszTmp = &szTmp [0];
	if (RC_BAD( getFormValueByName( "index",
			&pszTmp, sizeof( szTmp), NULL)))
	{
		if( RC_BAD( ExtractParameter( uiNumParams, ppszParams, 
			"index", sizeof( szTmp), szTmp)))
		{
			szTmp [0] = 0;
		}
	}
	if (szTmp[ 0])
	{
		uiIndex = f_atoud( szTmp);
	}

	// Get the query, as constructed so far.

	if (getFormValueByName( "querycriteria",
				&pszQueryCriteria, 0, NULL) == 0)
	{
		if( pszQueryCriteria && *pszQueryCriteria)
		{
			fcsDecodeHttpString( pszQueryCriteria);
		}
		else if (!pszQueryCriteria)
		{
			szQueryCriteria[0] = 0;
			pszQueryCriteria = &szQueryCriteria [0];
		}

		// Store the criteria into the session.

		if (gv_FlmSysData.HttpConfigParms.fnAcquireSession &&
			 ((pvHttpSession = fnAcquireSession()) != NULL))
		{
			uiCriteriaLen = f_strlen( pszQueryCriteria) + 1;
			fnSetSessionValue( pvHttpSession,
				QUERY_CRITERIA_LEN, &uiCriteriaLen, sizeof( uiCriteriaLen));
			fnSetSessionValue( pvHttpSession,
				QUERY_CRITERIA, pszQueryCriteria, (FLMSIZET)uiCriteriaLen);
			fnReleaseSession( pvHttpSession);
		}
	}
	else
	{

		// See if the query criteria was stored in the http session.

		if (gv_FlmSysData.HttpConfigParms.fnAcquireSession &&
			 ((pvHttpSession = fnAcquireSession()) != NULL))
		{
			uiSize = sizeof( uiCriteriaLen);
			if (fnGetSessionValue( pvHttpSession,
							QUERY_CRITERIA_LEN, (void *)&uiCriteriaLen,
							(FLMSIZET *)&uiSize) == 0)
			{
				if (uiCriteriaLen <= sizeof( szQueryCriteria))
				{
					pszQueryCriteria = &szQueryCriteria [0];
				}
				else
				{
					if (RC_BAD( f_alloc( uiCriteriaLen, &pszQueryCriteria)))
					{
						pszQueryCriteria = NULL;
					}
				}
				if (pszQueryCriteria)
				{
					if (fnGetSessionValue( pvHttpSession, QUERY_CRITERIA,
									pszQueryCriteria,
									(FLMSIZET *)&uiCriteriaLen) != 0)
					{
						if (pszQueryCriteria != &szQueryCriteria [0])
						{
							f_free( &pszQueryCriteria);
						}
					}
				}
			}
			fnReleaseSession( pvHttpSession);
		}
	}

	// Get the value of the Operation field, if present.

	getFormValueByName( "Operation",
				&pszOperation, 0, NULL);
	if( pszOperation)
	{
		if (f_stricmp( pszOperation, OPERATION_QUERY) == 0)
		{
			bPerformQuery = TRUE;
		}
		else if (f_stricmp( pszOperation, OPERATION_DELETE) == 0)
		{
			bPerformQuery = TRUE;
			bDoDelete = TRUE;
		}
		else if (f_stricmp( pszOperation, OPERATION_STOP) == 0)
		{
			bStopQuery = TRUE;
		}
		else if (f_stricmp( pszOperation, OPERATION_ABORT) == 0)
		{
			bStopQuery = TRUE;
			bAbortQuery = TRUE;
		}
	}

	// See if we had a query running.  Get the query object ID
	// if any.

	szTmp [0] = '\0';
	uiQueryThreadId = 0;
	if (RC_OK( ExtractParameter( uiNumParams, ppszParams, 
		"Running", sizeof( szTmp), szTmp)))
	{
		if (szTmp [0])
		{
			uiQueryThreadId = f_atoud( szTmp);
			QueryStatus.bQueryRunning = TRUE;
		}
	}

	if (bPerformQuery)
	{

		// Better not have both bQueryRunning and bPerformQuery set!

		flmAssert( !QueryStatus.bQueryRunning);

		// Parse the query.

		if (RC_BAD( runRc = parseQuery( hDb, uiContainer, uiIndex,
									pNameTable,
									pszQueryCriteria, &hCursor)))
		{
			pszErrType = "PARSING QUERY";
		}
		else if (RC_BAD( runRc = runQuery( hDb, uiContainer, uiIndex,
										hCursor, bDoDelete, &uiQueryThreadId)))
		{
			pszErrType = "RUNNING QUERY";
		}
		else
		{
			QueryStatus.bQueryRunning = TRUE;

			// Set hCursor to null because the query thread will destroy it when
			// it finishes.

			hCursor = HFCURSOR_NULL;
		}
	}

	// Stop the query, if requested, or get the query data.

	if (QueryStatus.bQueryRunning)
	{

		// Give query a fifth of a second to complete.  If it doesn't complet
		// in this amount of time, we will catch it on a refresh.

		f_sleep( 200);

		// getQueryStatus could change QueryStatus.bQueryRunning
		// to FALSE.

		getQueryStatus( uiQueryThreadId, bStopQuery, bAbortQuery, &QueryStatus);
	}

	// Output the web page.

	if (!QueryStatus.bQueryRunning && QueryStatus.bHaveQueryStatus)
	{

		// If we have query results, output a page for viewing/editing them.

		printDocStart( "Query Results");
	}
	else if (!QueryStatus.bQueryRunning)
	{
		printDocStart( "Run Query");
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
		printRecordStyle();
		printStyle();

		// Output html that will cause a refresh to occur.

		fnPrintf( m_pHRequest, 
			"<META http-equiv=\"refresh\" content=\"1; "
			"url=%s/select?Running=%u&dbhandle=%s&container=%u&index=%u\">"
			"<TITLE>Query Status</TITLE>\n",
			m_pszURLString,
			(unsigned)uiQueryThreadId, szDbKey, (unsigned)uiContainer,
			(unsigned)uiIndex);

		fnPrintf( m_pHRequest, "</head>\n"
									  "<body>\n");
	}

	// Output the form for entering the query criteria
	// and query status, if the query is currently running.

	outputSelectForm( hDb, szDbKey, uiContainer, uiIndex,
						QueryStatus.bQueryRunning,
						uiQueryThreadId, pNameTable,
					pszQueryCriteria, &QueryStatus);

	// Output any query status information we have.

	if (QueryStatus.bHaveQueryStatus)
	{
		outputQueryStatus( hDb, szDbKey, uiContainer,
									pNameTable, &QueryStatus);
	}

	// End the document

	printDocEnd();

Exit:

	fnEmit();

	if (pszQueryCriteria && pszQueryCriteria != &szQueryCriteria [0])
	{
		f_free( &pszQueryCriteria);
	}

	if (pszOperation)
	{
		f_free( &pszOperation);
	}

	if (hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	return( FERR_OK);

ReportErrorExit:

	printErrorPage( rc);
	goto Exit;
}

/****************************************************************************
Desc:	Output the form for the user to input a query.  If the query is
		running, the query criteria cannot be changed.
****************************************************************************/
void F_SelectPage::outputSelectForm(
	HFDB					hDb,
	const char *		pszDbKey,
	FLMUINT				uiContainer,
	FLMUINT				uiIndex,
	FLMBOOL				bQueryRunning,
	FLMUINT				uiQueryThreadId,
	F_NameTable *		pNameTable,
	const char *		pszQueryCriteria,
	QUERY_STATUS *		pQueryStatus)
{
	char *	pszName;
	char		szName [128];

	fnPrintf( m_pHRequest, "<form name=\""
		SELECT_FORM_NAME "\" type=\"submit\" "
	  "method=\"post\" action=\"%s/select", m_pszURLString);
	if (bQueryRunning)
	{
		fnPrintf( m_pHRequest, "?Running=%u&",
			(unsigned)uiQueryThreadId);
	}
	else
	{
		fnPrintf( m_pHRequest, "?");
	}

	fnPrintf( m_pHRequest, "dbhandle=%s&container=%u&index=%u\">\n",
		pszDbKey, (unsigned)uiContainer, (unsigned)uiIndex);

	// Output the database name

	printStartCenter();
	fnPrintf( m_pHRequest, "Database&nbsp;");
	printEncodedString( ((FDB *)hDb)->pFile->pszDbPath, HTML_ENCODING);
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output container name or a pulldown list to select a container.

	printStartCenter();
	fnPrintf( m_pHRequest, "Container&#%u;&nbsp;", (unsigned)':');
	if (pQueryStatus->bQueryRunning)
	{
		switch (uiContainer)
		{
			case FLM_DATA_CONTAINER:
				pszName = (char *)"Data";
				break;
			case FLM_DICT_CONTAINER:
				pszName = (char *)"Dictionary";
				break;
			case FLM_TRACKER_CONTAINER:
				pszName = (char *)"Tracker";
				break;
			default:
				if (!pNameTable ||
					 !pNameTable->getFromTagNum( uiContainer, NULL,
									szName,
									sizeof( szName)))
				{
					f_sprintf( szName, "Cont_%u", (unsigned)uiContainer);
				}
				pszName = &szName [0];
				break;
		}
		printEncodedString( pszName, HTML_ENCODING);
		fnPrintf( m_pHRequest, " (%u)", (unsigned)uiContainer);
	}
	else
	{
		printContainerPulldown( pNameTable, uiContainer);
	}
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output pulldown list to select an index

	if (!pQueryStatus->bQueryRunning)
	{
		printStartCenter();
		fnPrintf( m_pHRequest, "Index&#%u;&nbsp;", (unsigned)':');
		printIndexPulldown( pNameTable, uiIndex, TRUE, TRUE);
		printEndCenter( FALSE);
		fnPrintf( m_pHRequest, "<br>\n");
	}

	// Output text box for query.

	printStartCenter();
	fnPrintf( m_pHRequest, "<textarea name=\"querycriteria\" wrap=off rows=4 cols=80");
	if (pQueryStatus->bQueryRunning)
	{
		fnPrintf( m_pHRequest, " readonly");
	}
	fnPrintf( m_pHRequest, ">\n");
	if (pszQueryCriteria && *pszQueryCriteria)
	{
		printEncodedString( pszQueryCriteria, HTML_ENCODING);
	}
	fnPrintf( m_pHRequest, "</textarea>");
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output a text box for the field list - so user can copy and paste
	// field names into the query box.

	if (!pQueryStatus->bQueryRunning && pNameTable)
	{
		FLMUINT	uiNextPos;
		FLMUINT	uiFieldNum;
		FLMUINT	uiType;

		printStartCenter();
		fnPrintf( m_pHRequest,
			"<textarea name=\"ListOfFields\" wrap=off rows=8 cols=80>\n");

		uiNextPos = 0;
		while (pNameTable->getNextTagNameOrder( &uiNextPos, NULL, 
			szName, sizeof( szName), &uiFieldNum, &uiType))
		{
			if( uiType != FLM_FIELD_TAG)
			{
				continue;
			}
			printEncodedString( szName, HTML_ENCODING);
			fnPrintf( m_pHRequest, " (%u)\n", (unsigned)uiFieldNum);
		}
		fnPrintf( m_pHRequest, "</textarea>");
		printEndCenter();
		fnPrintf( m_pHRequest, "<br>\n");
	}

	// Output the setOperation function

	printSetOperationScript();

	printStartCenter();
	if (!pQueryStatus->bQueryRunning)
	{

		// If we are not running a query, add a Perform Query button
		// and a Query & Delete button.

		printOperationButton( SELECT_FORM_NAME,
			"Perform Query", OPERATION_QUERY);
		printSpaces( 1);
		printOperationButton( SELECT_FORM_NAME,
			"Query & Delete", OPERATION_DELETE);
	}
	else
	{

		// Output a stop button if not doing a delete.  Otherwise, output a stop and
		// commit button and a stop and abort button.

		if (!pQueryStatus->bDoDelete)
		{
			printOperationButton( SELECT_FORM_NAME,
				"Stop Query", OPERATION_STOP);
		}
		else
		{
			printOperationButton( SELECT_FORM_NAME,
				"Stop Query & Commit Transaction", OPERATION_STOP);
			printSpaces( 1);
			printOperationButton( SELECT_FORM_NAME,
				"Stop Query & Abort Transaction", OPERATION_ABORT);
		}
	}
	printEndCenter( TRUE);

	// Close the form

	fnPrintf( m_pHRequest, "</form>\n");
}

/****************************************************************************
Desc:	Outputs information on a query that is either running or has
		finished.
****************************************************************************/
void F_SelectPage::outputQueryStatus(
	HFDB					hDb,
	const char *		pszDbKey,
	FLMUINT				uiContainer,
	F_NameTable *		pNameTable,
	QUERY_STATUS *		pQueryStatus)
{
	RCODE					rc;
	char					szName[ 128];
	FLMUINT				uiMax;
	FLMUINT				uiLoop;
	FlmRecord *			pRec = NULL;
	FLMUINT				uiContext;

	fnPrintf( m_pHRequest, "<br>\n");

	// Output index optimization information

	printStartCenter();
	fnPrintf( m_pHRequest, "Index ");
	if (pQueryStatus->uiIndex == FLM_SELECT_INDEX)
	{
		fnPrintf( m_pHRequest, "(Selected by DB)&#%u; ", (unsigned)':');
	}
	else
	{
		fnPrintf( m_pHRequest, "(Set by User)&#%u; ", (unsigned)':');
	}
	if (pQueryStatus->uiIndexInfo == HAVE_NO_INDEX)
	{
		fnPrintf( m_pHRequest, "None");
	}
	else
	{
		if (!pNameTable ||
			 !pNameTable->getFromTagNum( pQueryStatus->uiOptIndex, NULL,
							szName, sizeof( szName)))
		{
			f_sprintf( (char *)szName, "Index_%u",
				(unsigned)pQueryStatus->uiOptIndex);
		}
		printEncodedString( szName, HTML_ENCODING);
		fnPrintf( m_pHRequest, " (%u)", (unsigned)pQueryStatus->uiOptIndex);

		if (pQueryStatus->uiIndexInfo == HAVE_MULTIPLE_INDEXES)
		{
			fnPrintf( m_pHRequest, " (Using multiple indexes)");
		}
		else if (pQueryStatus->uiIndexInfo == HAVE_ONE_INDEX_MULT_PARTS)
		{
			fnPrintf( m_pHRequest, " (Multiple subqueries use index)");
		}
	}
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	printStartCenter();
	if (pQueryStatus->bQueryRunning)
	{
		printTableStart( "QUERY PROGRESS", 2, 50);
	}
	else
	{
		printTableStart( "QUERY RESULTS", 2, 50);
	}

	// Column headers

	printTableRowStart();
	if (pQueryStatus->bDoDelete)
	{
		printColumnHeading( "Records Deleted", JUSTIFY_RIGHT);
	}
	else
	{
		printColumnHeading( "Records Matched", JUSTIFY_RIGHT);
	}
	printColumnHeading( "Processed Count", JUSTIFY_RIGHT);
	printTableRowEnd();

	if (pQueryStatus->uiProcessedCnt < pQueryStatus->uiDrnCount)
	{
		pQueryStatus->uiProcessedCnt = pQueryStatus->uiDrnCount;
	}
	printTableRowStart( TRUE);
	printTableDataStart( TRUE, JUSTIFY_RIGHT);
	fnPrintf( m_pHRequest, "%u", (unsigned)pQueryStatus->uiDrnCount);
	printTableDataEnd();
	printTableDataStart( TRUE, JUSTIFY_RIGHT);
	fnPrintf( m_pHRequest, "%u", (unsigned)pQueryStatus->uiProcessedCnt);
	printTableDataEnd();
	printTableRowEnd();
	printTableEnd();
	printEndCenter( FALSE);
	fnPrintf( m_pHRequest, "<br>\n");

	// Output the records if the query is done.  Then free the data.

	if (!pQueryStatus->bQueryRunning && pQueryStatus->puiDrnList)
	{

		// List the retrieved records.  They should not be in there if
		// we are doing a delete.

		flmAssert( !pQueryStatus->bDoDelete);

		// Output up to the first 100 records retrieved.  We
		// don't output more than one hundred because it will just
		// take too long.

		printTableStart( "RECORDS RETRIEVED", 1, 100);
		printTableEnd();
		fnPrintf( m_pHRequest, "<br>\n");

		if ((uiMax = pQueryStatus->uiDrnCount) > MAX_RECORDS_TO_OUTPUT)
		{
			uiMax = MAX_RECORDS_TO_OUTPUT;
		}

		uiContext = 0;
		for (uiLoop = 0; uiLoop < uiMax; uiLoop++)
		{
			if (RC_BAD( rc = FlmRecordRetrieve( hDb, uiContainer,
										pQueryStatus->puiDrnList [uiLoop],
										FO_EXACT, &pRec, NULL)))
			{
				if (rc != FERR_NOT_FOUND)
				{
					fnPrintf( m_pHRequest,
							"<br><font color=\"Red\">ERROR %04X (%s) retrieving "
							"record #%u</font><br><br>\n",
							(unsigned)rc, FlmErrorString( rc),
							(unsigned)pQueryStatus->puiDrnList [uiLoop]);
				}
			}
			else
			{
				printRecord( pszDbKey, pRec, pNameTable, &uiContext,
									TRUE, 0, FO_EXACT);
			}
		}
		f_free( &pQueryStatus->puiDrnList);
	}

//Exit:

	if (pRec)
	{
		pRec->Release();
	}
}

/****************************************************************************
Desc:	Set up a query from a query criteria string.
****************************************************************************/
RCODE F_SelectPage::parseQuery(
	HFDB				hDb,
	FLMUINT			uiContainer,
	FLMUINT			uiIndex,
	F_NameTable *	pNameTable,
	const char *	pszQueryCriteria,
	HFCURSOR *		phCursor)
{
	RCODE				rc = FERR_OK;

	*phCursor = HFCURSOR_NULL;
	if (RC_BAD( rc = FlmCursorInit( hDb, uiContainer, phCursor)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = FlmCursorConfig( *phCursor, FCURSOR_SET_FLM_IX,
						(void *)uiIndex, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = FlmParseQuery( *phCursor, pNameTable, pszQueryCriteria)))
	{
		goto Exit;
	}

	// Do a final validation on the cursor to ensure it is valid.

	if (RC_BAD( rc = FlmCursorValidate( *phCursor)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc) && *phCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( phCursor);
	}

	return( rc);
}

/****************************************************************************
Desc:	Run a query.
****************************************************************************/
RCODE F_SelectPage::runQuery(
	HFDB			hDb,
	FLMUINT		uiContainer,
	FLMUINT		uiIndex,
	HFCURSOR		hCursor,
	FLMBOOL		bDoDelete,
	FLMUINT *	puiQueryThreadId
	)
{
	RCODE					rc = FERR_OK;
	QUERY_STATUS *		pQueryStatus = NULL;
	IF_Thread *			pThread;
	FDB *					pDb = NULL;

	// Open the database for the thread - so it doesn't have
	// to worry about the handle going away.  The thread will close the
	// new handle when it exits.
	
	if (RC_BAD( rc = flmOpenFile( ((FDB *)hDb)->pFile, NULL, NULL, NULL,
							0, TRUE, NULL, NULL,
							(((FDB *)hDb)->pFile)->pszDbPassword, &pDb)))
	{
		goto Exit;
	}

	// Create an object to track the query.

	if (RC_BAD( rc = f_calloc( sizeof( QUERY_STATUS), &pQueryStatus)))
	{
		goto Exit;
	}

	pQueryStatus->hDb = (HFDB)pDb;
	pQueryStatus->uiContainer = uiContainer;
	pQueryStatus->uiIndex = uiIndex;
	pQueryStatus->hCursor = hCursor;
	pQueryStatus->bDoDelete = bDoDelete;
	pQueryStatus->bQueryRunning = TRUE;
	pQueryStatus->uiLastTimeChecked = FLM_GET_TIMER();
	FlmCursorGetConfig( hCursor, FCURSOR_GET_FLM_IX, &pQueryStatus->uiOptIndex,
		&pQueryStatus->uiIndexInfo);

	// If browser does not check query status at least every 15 seconds, we will
	// assume it has gone away and the thread will terminate itself.

	pQueryStatus->uiQueryTimeout = FLM_SECS_TO_TIMER_UNITS( 15);

	// Start a thread to do the query.

	if( RC_BAD( rc = f_threadCreate( &pThread, imonDoQuery,
							"IMON QUERY",
							gv_uiDbThrdGrp, 1,
							(void *)pQueryStatus, (void *)hDb)))
	{
		goto Exit;
	}

	*puiQueryThreadId = pThread->getThreadId();
	
	// Set pQueryStatus to NULL so it won't be freed below.  The thread
	// will free it when it stops.

	pQueryStatus = NULL;

	// Set pDb to NULL so it won't be closed below.  The thread will
	// close it when it stops.

	pDb = NULL;

Exit:

	if (pThread)
	{
		pThread->Release();
	}

	if (pQueryStatus)
	{
		f_free( &pQueryStatus);
	}

	if (pDb)
	{
		FlmDbClose( (HFDB *)&pDb);
	}

	return( rc);

}

/****************************************************************************
Desc:	Output the current thread status to the web page.
****************************************************************************/
void F_SelectPage::getQueryStatus(
	FLMUINT			uiQueryThreadId,
	FLMBOOL			bStopQuery,
	FLMBOOL			bAbortQuery,
	QUERY_STATUS *	pQueryStatus
	)
{
	FLMUINT			uiThreadId;
	IF_Thread *		pThread = NULL;
	QUERY_STATUS *	pThreadQueryStatus;
	FLMBOOL			bMutexLocked = FALSE;

	// pQueryStatus->bHaveQueryStatus should be set to FALSE by the caller.

	flmAssert( !pQueryStatus->bHaveQueryStatus);

	// See if the thread is still running.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	uiThreadId = 0;
	for (;;)
	{
		if (RC_BAD( gv_FlmSysData.pThreadMgr->getNextGroupThread( &pThread,
						gv_uiDbThrdGrp, &uiThreadId)))
		{
			pQueryStatus->bQueryRunning = FALSE;
			goto Exit;
		}
		if (uiThreadId == uiQueryThreadId)
		{

			// If the app ID is zero, the thread is on its way out or already
			// out.  Can no longer get thread status.

			if (!pThread->getThreadAppId())
			{
				pQueryStatus->bQueryRunning = FALSE;
				goto Exit;
			}

			// Found thread, get its query data

			pThreadQueryStatus = (QUERY_STATUS *)pThread->getParm1();
			pThreadQueryStatus->uiLastTimeChecked = FLM_GET_TIMER();

			// Tell the thread to stop the query before telling it
			// to stop.  This is so we can get partial results.

			if (bStopQuery)
			{
				pThreadQueryStatus->bStopQuery = TRUE;
				pThreadQueryStatus->bAbortQuery = bAbortQuery;

				// Go into a while loop, waiting for the thread
				// to finish its query.

				while (pThreadQueryStatus->bQueryRunning)
				{
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					bMutexLocked = FALSE;
					f_sleep( 200);
					f_mutexLock( gv_FlmSysData.hShareMutex);
					bMutexLocked = TRUE;

					// If the thread app ID goes to zero, it has been
					// told to shut down, and has either already gone
					// away or is in the process of doing so, in which
					// case pThreadQueryStatus has either already been
					// deleted, or will be - so it is not safe to access
					// it any more!

					if (!pThread->getThreadAppId())
					{
						pQueryStatus->bQueryRunning = FALSE;
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

	// If the query is not done, return everything except the DRN list.
	// Note that we test pThreadQueryStatus->bQueryRunning BEFORE
	// doing the memcpy.  This is because puiDrnList is not guaranteed
	// to be set until bQueryRunning is FALSE.  If bQueryRunning is TRUE,
	// we will NULL out whatever got copied into puiDrnList.

	if (!pThreadQueryStatus->bQueryRunning)
	{
		f_memcpy( pQueryStatus, pThreadQueryStatus, sizeof( QUERY_STATUS));

		// NULL out the puiDrnList member in the thread's copy of
		// the query data so it won't free the list when it exits.
		// The caller of this routine must be sure to do it instead!

		pThreadQueryStatus->puiDrnList = NULL;

		// Need to unlock the mutex so that the thread can stop.

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
		pThread->stopThread();
	}
	else
	{
		f_memcpy( pQueryStatus, pThreadQueryStatus, sizeof( QUERY_STATUS));

		// NULL out the DRN list and set bQueryRunning to TRUE.  This takes
		// care of a race condition of pThreadQueryStatus->bQueryRunning getting
		// set to FALSE by the query thread after we test it above.
		// we make the test on pThreadQueryStatus->bQueryRunning.  We will
		// simply get that fact next time we get status.

		pQueryStatus->bQueryRunning = TRUE;
		pQueryStatus->puiDrnList = NULL;
	}
	pQueryStatus->bHaveQueryStatus = TRUE;

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

/***************************************************************************
Desc:	Query status callback.
***************************************************************************/
FSTATIC RCODE queryStatusCB(
	FLMUINT	uiStatusType,
	void *	pvParm1,
	void *,	// pvParm2,
	void *	pvUserData)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiCurrTime;

	switch (uiStatusType)
	{
		case FLM_SUBQUERY_STATUS:
		{
			FCURSOR_SUBQUERY_STATUS *	pQueryInfo = (FCURSOR_SUBQUERY_STATUS *)pvParm1;
			QUERY_STATUS *					pQueryStatus = (QUERY_STATUS *)pvUserData;

			pQueryStatus->uiProcessedCnt = pQueryInfo->uiProcessedCnt;
			uiCurrTime = FLM_GET_TIMER();
			if (pQueryStatus->bStopQuery)
			{
				rc = RC_SET( FERR_USER_ABORT);
				goto Exit;
			}
			else if (FLM_ELAPSED_TIME( uiCurrTime,
							pQueryStatus->uiLastTimeChecked) >=
					pQueryStatus->uiQueryTimeout)
			{
				rc = RC_SET( FERR_TIMEOUT);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Thread to perform a query for a web page.
****************************************************************************/
FSTATIC RCODE FLMAPI imonDoQuery(
	IF_Thread *		pThread)
{
	RCODE				rc;
	QUERY_STATUS *	pQueryStatus = (QUERY_STATUS *)pThread->getParm1();
	HFCURSOR			hCursor = pQueryStatus->hCursor;
	HFDB				hDb = pQueryStatus->hDb;
	FLMUINT			uiContainer = pQueryStatus->uiContainer;
	FLMUINT			uiDrn;
	FLMUINT			uiCurrTime;
	FLMUINT			uiLastTimeSetStatus = 0;
	FLMUINT			ui5SecsTime;
	FLMBOOL			bTransStarted = FALSE;

	pThread->setThreadStatus( FLM_THREAD_STATUS_INITIALIZING);
	ui5SecsTime = FLM_SECS_TO_TIMER_UNITS( 5);

	// Start a transaction.

	if (pQueryStatus->bDoDelete)
	{
		rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, FLM_NO_TIMEOUT);
	}
	else
	{
		rc = FlmDbTransBegin( hDb, FLM_READ_TRANS, 0);
	}
	if (RC_BAD( rc))
	{
		pThread->setThreadStatus( "Trans Error %04X", (unsigned)rc);
		pQueryStatus->bQueryRunning = FALSE;
	}
	else
	{
		bTransStarted = TRUE;

		// Set the cursor callback

		(void)FlmCursorConfig( hCursor, FCURSOR_SET_STATUS_HOOK,
										(void *)(FLMUINT)queryStatusCB,
										(void *)pQueryStatus);

		// Disconnect the cursor from its current database handle.

		(void)FlmCursorConfig( hCursor, FCURSOR_DISCONNECT, 0, 0);

		// Connect the cursor to the database handle passed in.

		(void)FlmCursorConfig( hCursor, FCURSOR_SET_HDB, (void *)hDb, 0);

		pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);
	}

	for (;;)
	{

		// See if we should shut down. 

		if (pThread->getShutdownFlag())
		{
			pQueryStatus->bQueryRunning = FALSE;

			// Transaction will be aborted below

			pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);
			goto Exit;
		}

		// See if the browser quit asking for status.

		uiCurrTime = FLM_GET_TIMER();
		if (FLM_ELAPSED_TIME( uiCurrTime, pQueryStatus->uiLastTimeChecked) >=
						pQueryStatus->uiQueryTimeout)
		{
			if (pQueryStatus->bQueryRunning)
			{
				pThread->setThreadStatus( "Timed out, Cnt=%u",
					(unsigned)pQueryStatus->uiDrnCount);
				pQueryStatus->bQueryRunning = FALSE;
			}

			// Transaction will be aborted below

			goto Exit;
		}

		// If the query is not running, just pause one second at a time
		// until we are told to shut down or until we time out.

		if (!pQueryStatus->bQueryRunning)
		{
			pThread->sleep( 1000);
			continue;
		}

		// See if we should stop the query.

		if (pQueryStatus->bStopQuery)
		{
Stop_Query:
			if (pQueryStatus->bAbortQuery)
			{
				pThread->setThreadStatus( "User aborted, Cnt=%u",
					(unsigned)pQueryStatus->uiDrnCount);
				goto Abort_Query;
			}
			else
			{
				pThread->setThreadStatus( "User halted, Cnt=%u",
					(unsigned)pQueryStatus->uiDrnCount);
				goto Commit_Query;
			}
		}

		// Get the next record.

		if (RC_BAD( rc = FlmCursorNextDRN( hCursor, &uiDrn)))
		{
			if (rc == FERR_EOF_HIT || rc == FERR_BOF_HIT || rc == FERR_NOT_FOUND)
			{
				pThread->setThreadStatus( "Query done, Cnt=%u",
					(unsigned)pQueryStatus->uiDrnCount);

				// Finish the query.  If doing a delete, the transaction
				// will be committed.

				goto Commit_Query;
			}
			else if (rc == FERR_USER_ABORT)
			{

				// Callback forced us to quit.

				goto Stop_Query;
			}
			else if (rc == FERR_TIMEOUT)
			{
				pThread->setThreadStatus( "Timed out, Cnt=%u",
					(unsigned)pQueryStatus->uiDrnCount);
				goto Abort_Query;
			}
			else if (rc == FERR_OLD_VIEW)
			{

				// Better not happen if we are in an update transaction!

				flmAssert( !pQueryStatus->bDoDelete);

				// Start a new read transaction.

				FlmDbTransAbort( hDb);
				bTransStarted = FALSE;
				if (RC_BAD( rc = FlmDbTransBegin( hDb, FLM_READ_TRANS, 0)))
				{
					pThread->setThreadStatus( "Trans Error %04X, Cnt=%u",
						(unsigned)rc, (unsigned)pQueryStatus->uiDrnCount);
					goto Abort_Query;
				}
				bTransStarted = TRUE;
			}
			else
			{
				pThread->setThreadStatus( "Read Error %04X, Cnt=%u",
					(unsigned)rc, (unsigned)pQueryStatus->uiDrnCount);
				goto Abort_Query;
			}
		}
		else	// FERR_OK
		{

			// If we are not deleting, add DRN to the list of DRNs.

			if (!pQueryStatus->bDoDelete)
			{
				if (pQueryStatus->uiDrnCount == pQueryStatus->uiDrnListSize)
				{
					FLMUINT *	puiTmp;

					if( RC_BAD( rc = f_alloc( sizeof( FLMUINT) * 
						(pQueryStatus->uiDrnListSize + DRN_LIST_INCREASE_SIZE),
						&puiTmp)))
					{
						pThread->setThreadStatus( "Mem Alloc Error, Cnt=%u",
								(unsigned)pQueryStatus->uiDrnCount);
						goto Abort_Query;
					}

					if (pQueryStatus->puiDrnList)
					{
						f_memcpy( puiTmp, pQueryStatus->puiDrnList,
							sizeof( FLMUINT) * pQueryStatus->uiDrnCount);
						f_free( &pQueryStatus->puiDrnList);
					}
					pQueryStatus->puiDrnList = puiTmp;
					pQueryStatus->uiDrnListSize += DRN_LIST_INCREASE_SIZE;
				}
				pQueryStatus->puiDrnList [pQueryStatus->uiDrnCount] = uiDrn;
				pQueryStatus->uiDrnCount++;
			}
			else
			{
				pQueryStatus->uiDrnCount++;

				// Delete the record.

				if (RC_BAD( rc = FlmRecordDelete( hDb, uiContainer, uiDrn, 0)))
				{
					if (rc != FERR_NOT_FOUND)
					{
						pThread->setThreadStatus( "Delete Error %04X, Cnt=%u",
							(unsigned)rc, (unsigned)pQueryStatus->uiDrnCount);
						goto Abort_Query;
					}
				}
			}

			// Update thread status every 5 seconds

			uiCurrTime = FLM_GET_TIMER();
			if (FLM_ELAPSED_TIME( uiCurrTime, uiLastTimeSetStatus) >=
						ui5SecsTime)
			{
				if (pQueryStatus->uiProcessedCnt < pQueryStatus->uiDrnCount)
				{
					pQueryStatus->uiProcessedCnt = pQueryStatus->uiDrnCount;
				}
				pThread->setThreadStatus( "Found %u, Processed %u", 
					(unsigned)pQueryStatus->uiDrnCount,
					(unsigned)pQueryStatus->uiProcessedCnt);
				uiLastTimeSetStatus = uiCurrTime;
			}
		}

		continue;

Abort_Query:

		// This label is jumped to whenever a condition occurs which
		// causes the query to be stopped - error, user stops, etc.

		// Free the cursor.

		FlmCursorFree( &hCursor);

		// Abort the transaction if we still have one going.

		if (bTransStarted)
		{
			(void)FlmDbTransAbort( hDb);
			bTransStarted = FALSE;
		}

		// Close the database.

		FlmDbClose( &hDb);

		// Continue until told to shut down or until we
		// timeout.

		pQueryStatus->bQueryRunning = FALSE;

		continue;

Commit_Query:

		// Free the cursor.

		FlmCursorFree( &hCursor);

		bTransStarted = FALSE;
		if (pQueryStatus->bDoDelete)
		{
			if (RC_BAD( rc = FlmDbTransCommit( hDb)))
			{
				pThread->setThreadStatus( "Commit Error %04X, Cnt=%u",
					(unsigned)rc, (unsigned)pQueryStatus->uiDrnCount);
			}
		}
		else
		{
			// Only a read transaction - don't care if committed or aborted

			(void)FlmDbTransCommit( hDb);
		}

		// Close the database.

		FlmDbClose( &hDb);

		pQueryStatus->bQueryRunning = FALSE;

		// Continue until told to shut down or until we
		// timeout.

		continue;
	}

Exit:

	// Free the cursor.

	if (hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	// Abort the transaction if we still have one going.

	if (bTransStarted)
	{
		(void)FlmDbTransAbort( hDb);
	}

	// Close the database.

	if (hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	// Set the thread's app ID to 0, so that it will not
	// be found now that the thread is terminating (we don't
	// want getQueryStatus() to find the thread).

	pThread->setThreadAppId( 0);

	// Free the query status.  Must do inside mutex lock so
	// that it doesn't go away after getQueryStatus finds the
	// thread.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	if (pQueryStatus->puiDrnList)
	{
		f_free( &pQueryStatus->puiDrnList);
	}
	f_free( &pQueryStatus);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	return( FERR_OK);
}
