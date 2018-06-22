//-------------------------------------------------------------------------
// Desc:	Class for displaying database log header in HTML on a web page.
// Tabs:	3
//
// Copyright (c) 2002-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

/*********************************************************
Desc:	Displays the Log Headers.
**********************************************************/
RCODE F_LogHeaderPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	HFDB			hDb = HFDB_NULL;
	char			szDbKey[ F_SESSION_DB_KEY_LEN];
	char			szTmp[ 128];
	char			szTmp1[ 128];
	FLMUINT		uiBucket = 0;
	FFILE *		pFile = NULL;
	char			szAddress[ 30];
	void *		pvAddress = NULL;
	FLMBOOL		bFlmLocked = FALSE;
	FLMBYTE *	pucLastCommitted = NULL;
	FLMBYTE *	pucCheckpoint = NULL;
	FLMBYTE *	pucUncommitted = NULL;
	char			szFilename[ 128];
	FLMBOOL		bRefresh;
	F_Session *	pFlmSession = m_pFlmSession;

	// We need to check our session.
	if (!pFlmSession)
	{
		printErrorPage( m_uiSessionRC,  TRUE, "No session available for this request");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( LOG_HEADER_SIZE, &pucLastCommitted)))
	{
		printErrorPage( rc, TRUE, "Failed to allocate a temporary log header buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( LOG_HEADER_SIZE, &pucCheckpoint)))
	{
		printErrorPage( rc, TRUE, "Failed to allocate a temporary log header buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( LOG_HEADER_SIZE, &pucUncommitted)))
	{
		printErrorPage( rc, TRUE, "Failed to allocate a temporary log header buffer");
		goto Exit;
	}

	if (DetectParameter( uiNumParams, ppszParams, "dbhandle"))
	{
		// The hDb (Database File Handle)
		if (RC_BAD( rc = getDatabaseHandleParam( uiNumParams, ppszParams, 
			pFlmSession, &hDb, szDbKey)))
		{
			printErrorPage( rc, TRUE, "Invalid Database Handle");
			goto Exit;
		}

		if( IsInCSMode( hDb))
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			printErrorPage( rc, TRUE, "Unsupported client/server operation.");
			goto Exit;
		}

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bFlmLocked = TRUE;

		// Get the pFile.
		pFile = ((FDB *)hDb)->pFile;
	}
	else
	{

		if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
													  "Bucket", sizeof( szTmp),
													  szTmp)))
		{
			printErrorPage( rc, TRUE, "Missing Bucket parameter from request");
			goto Exit;
		}
		uiBucket = f_atoud( szTmp);

		if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
													  "Address", sizeof( szAddress),
													  szAddress)))
		{
			printErrorPage( rc, TRUE, "Missing Address parameter from request");
			goto Exit;
		}
		pvAddress = (void *)f_atoud( szAddress);

		f_mutexLock( gv_FlmSysData.hShareMutex);
		bFlmLocked = TRUE;

		pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket;
		while (pFile && (void *)pFile != pvAddress)
		{
			pFile = pFile->pNext;
		}
		
		if (pFile == NULL)
		{
			printErrorPage( rc, TRUE, "Cannot locate required FFILE");
			goto Exit;
		}
	}

	f_memcpy( pucLastCommitted, pFile->ucLastCommittedLogHdr, LOG_HEADER_SIZE);
	f_memcpy( pucCheckpoint, pFile->ucCheckpointLogHdr, LOG_HEADER_SIZE);
	f_memcpy( pucUncommitted, pFile->ucUncommittedLogHdr, LOG_HEADER_SIZE);

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bFlmLocked = FALSE;

	// Start the document.
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest,  "<html>\n");


	// Determine if we are being requested to refresh this page or  not.
	if ((bRefresh = DetectParameter(
		uiNumParams, ppszParams, "Refresh")) == TRUE)
	{
		// Send back the page with a refresh command in the header
		if (hDb == HFDB_NULL)
		{
			f_sprintf( (char *)szTmp,
				"%s/LogHdr?Refresh&Bucket=%lu&Address=%s",
				m_pszURLString, uiBucket, szAddress);
		}
		else
		{
			f_sprintf( (char *)szTmp,
				"%s/LogHdr?Refresh&dbhandle=%s", m_pszURLString, (char *)szDbKey);
		}

		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>Log File Header</TITLE>\n", szTmp);
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>Log File Header</TITLE>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");


	// If we are not to refresh this page, then don't include the
	// refresh meta command
	if (!bRefresh)
	{
		if (hDb == HFDB_NULL)
		{
			f_sprintf( (char *)szTmp,
				"<A HREF=%s/LogHdr?Refresh&Bucket=%lu&Address=%s>Start auto-refresh (5 sec.)</A>",
				m_pszURLString, uiBucket, szAddress);
		}
		else
		{
			f_sprintf( (char *)szTmp,
				"<A HREF=%s/LogHdr?Refresh&dbhandle=%s>Start auto-refresh (5 sec.)</A>",
				m_pszURLString, (char *)szDbKey);
		}
	}
	else
	{
		if (hDb == HFDB_NULL)
		{
			f_sprintf( (char *)szTmp,
				"<A HREF=%s/LogHdr?Bucket=%lu&Address=%s>Stop auto-refresh</A>",
				m_pszURLString, uiBucket, szAddress);
		}
		else
		{
			f_sprintf( (char *)szTmp,
				"<A HREF=%s/LogHdr?dbhandle=%s>Stop auto-refresh</A>",
				m_pszURLString, (char *)szDbKey);
		}
	}
	// Prepare the refresh link.
	if (hDb == HFDB_NULL)
	{
		f_sprintf( (char *)szTmp1,
			"<A HREF=%s/LogHdr?Bucket=%lu&Address=%s>Refresh</A>",
			m_pszURLString, uiBucket, szAddress);
	}
	else
	{
		f_sprintf( (char *)szTmp1,
			"<A HREF=%s/LogHdr?dbhandle=%s>Refresh</A>",
			m_pszURLString, (char *)szDbKey);
	}

	// Write out the table headings
	f_sprintf( (char *)szFilename, "Log File Header - %s", pFile->pszDbPath);
	printTableStart( (char *)szFilename, 2, 100);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 2, 1, FALSE);
	fnPrintf( m_pHRequest, "%s, ", szTmp1);
	fnPrintf( m_pHRequest, "%s\n", szTmp);
	printColumnHeadingClose();
	printTableRowEnd();

	printTableEnd();

	// Display the log header in the Log Header table.
	printLogHeaders( pucLastCommitted, pucCheckpoint, pucUncommitted);

	printDocEnd();
	fnEmit();

Exit:

	if (bFlmLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (pucLastCommitted)
	{
		f_free( &pucLastCommitted);
	}
	if (pucCheckpoint)
	{
		f_free( &pucCheckpoint);
	}
	if (pucUncommitted)
	{
		f_free( &pucUncommitted);
	}

	return( rc);
}
