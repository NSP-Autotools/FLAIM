//-------------------------------------------------------------------------
// Desc:	Class for displaying an RCACHE structure in HTML on a web page.
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

#define GENERIC_SIZE 1024

FSTATIC void flmBuildRCacheLink(
	char *			pszString,
	RCACHE *			pRCache,
	char *			pszURLString);

/****************************************************************************
 Desc:	procedure to display the contents of the RCache Manager
 ****************************************************************************/
RCODE F_RCacheMgrPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bUsage;
	FLMBOOL		bRefresh;
	FLMBYTE *	pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Determine if we are being requested to refresh this page or  not and if
	// we are being asked to show the Usage Statistics.
	bRefresh = DetectParameter( uiNumParams, ppszParams, "Refresh");
	bUsage = DetectParameter( uiNumParams, ppszParams, "Usage");

	// Invoke the public function to display the Usage structure.
	if (bUsage)
	{
		RCACHE_MGR			LocalRCacheMgr;
		RCACHE_MGR *		pRCacheMgr;

		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

		f_memcpy(
			&LocalRCacheMgr, 
			(char *)&gv_FlmSysData.RCacheMgr, 
			sizeof(LocalRCacheMgr));
		
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	
		pRCacheMgr = &LocalRCacheMgr;

		rc = writeUsage(
					&pRCacheMgr->Usage,
					bRefresh,
					"/RCacheMgr?Usage",
					"RCache Manager Usage Statistics");

		goto Exit;

	}

	stdHdr();
	
	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	if (bRefresh)
	{
		// Send back the page with a refresh command in the header
		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s/RCacheMgr?Refresh\">"
			"<TITLE>gv_FlmSysData.RCacheMgr</TITLE>\n",
			m_pszURLString);
		
		// Add the function to generate a popup framed window only if this is not
		// showing the Usage statistics.
		printStyle();
		popupFrame();

		fnPrintf( m_pHRequest, "</HEAD>\n");
		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
			"<A HREF=%s/RCacheMgr>Stop Auto-refresh</A>",
			m_pszURLString);
	}
	else
	{

		// Add the function to generate a popup framed window if not
		// displaying the Usage.
		fnPrintf( m_pHRequest, "<HEAD><TITLE>gv_FlmSysData.RCacheMgr</TITLE>\n");
		printStyle();
		popupFrame();
		fnPrintf( m_pHRequest, "</HEAD>\n");

		// Send back a page without the refresh command
		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
			"<A HREF=%s/RCacheMgr?Refresh>Start Auto-refresh (5 sec.)</A>",
			m_pszURLString);
	}


	// Begin the table
	printTableStart( "RCache Manager", 4, 100);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=%s/RCacheMgr>Refresh</A>, ",
									m_pszURLString);
	fnPrintf( m_pHRequest, "%s\n", pszTemp);
	printColumnHeadingClose();
	printTableRowEnd();
	
	// Write out the table headings
	
	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();

	
	write_data();

	fnPrintf( m_pHRequest, "</body></html>\n");

Exit:

	fnEmit();

	if (pszTemp)
	{
		f_free( &pszTemp);
	}
	
	return( rc);
}

/****************************************************************************
 Desc:	private procedure to generate the HTML page
 ****************************************************************************/
void F_RCacheMgrPage::write_data( void)
{

	RCODE					rc = FERR_OK;
	char					szTemp[100];
	RCACHE_MGR			LocalRCacheMgr;
	RCACHE_MGR *		pRCacheMgr;
	FLMBOOL				bFlaimLocked = FALSE;
	RCACHE *				pPurgeList = NULL;
	RCACHE *				pMRURecord = NULL;
	RCACHE *				pLRURecord = NULL;
	char					szAddress[20];
	char					szOffset[8];
	FLMBOOL				bHighlight = FALSE;


	// First, get a local copy of the RCacheMgr structure so we don't interfere 
	// with the operation of the database.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	bFlaimLocked = TRUE;

	f_memcpy(&LocalRCacheMgr, 
				&gv_FlmSysData.RCacheMgr, 
				sizeof(LocalRCacheMgr));

	pRCacheMgr = &LocalRCacheMgr;

	// Make copies of needed records so we can use them later.
	if (pRCacheMgr->pPurgeList)
	{
		if (RC_BAD( rc = f_alloc( sizeof(RCACHE), &pPurgeList)))
		{
			goto Exit;
		}
		f_memcpy( pPurgeList, pRCacheMgr->pPurgeList, sizeof(RCACHE));
	}

	if (pRCacheMgr->pMRURecord)
	{
		if (RC_BAD( rc = f_alloc( sizeof(RCACHE), &pMRURecord)))
		{
			goto Exit;
		}
		f_memcpy( pMRURecord, pRCacheMgr->pMRURecord, sizeof(RCACHE));
	}
	
	if (pRCacheMgr->pLRURecord)
	{
		if (RC_BAD( rc = f_alloc( sizeof(RCACHE), &pLRURecord)))
		{
			goto Exit;
		}
		f_memcpy( pLRURecord, pRCacheMgr->pLRURecord, sizeof(RCACHE));
	}

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bFlaimLocked = FALSE;


	
	// pPurgeList
	if (pRCacheMgr->pPurgeList)
	{
		printAddress( pPurgeList->pFile, szAddress);
		f_sprintf((char *)szTemp, 
					"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
					m_pszURLString,
					(unsigned long)pPurgeList->uiContainer,
					(unsigned long)pPurgeList->uiDrn,
					szAddress,
					(unsigned long)pPurgeList->uiLowTransId);
	}

	printHTMLLink(
		"pPurgeList",
		"RCACHE *",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->pPurgeList,
		(void *)pRCacheMgr->pPurgeList,
		(char*)szTemp,
		(bHighlight = ~bHighlight));




	// pMRURecord
	if (pRCacheMgr->pMRURecord)
	{
		printAddress( pMRURecord->pFile, szAddress);
		f_sprintf((char *)szTemp, 
					"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
					m_pszURLString,
					(unsigned long)pMRURecord->uiContainer,
					(unsigned long)pMRURecord->uiDrn,
					szAddress,
					(unsigned long)pMRURecord->uiLowTransId);
	}

	printHTMLLink(
		"pMRURecord",
		"RCACHE *",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->pMRURecord,
		(void *)pRCacheMgr->pMRURecord,
		(char*)szTemp,
		(bHighlight = ~bHighlight));




	// pLRURecord
	if (pRCacheMgr->pLRURecord)
	{
		printAddress( pRCacheMgr->pLRURecord->pFile, szAddress);
		f_sprintf((char *)szTemp,
					"%s/RCache?Container=%lu?DRN=%ld?File=%s?Version=%ld",
					m_pszURLString,
					(unsigned long)pLRURecord->uiContainer,
					(unsigned long)pLRURecord->uiDrn,
					szAddress,
					(unsigned long)pLRURecord->uiLowTransId);
	}

	printHTMLLink(
		"pLRURecord",
		"RCACHE *",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->pLRURecord,
		(void *)pRCacheMgr->pLRURecord,
		(char*)szTemp,
		(bHighlight = ~bHighlight));

	


	
	// Usage
	printTableRowStart( (bHighlight = ~bHighlight));
	f_sprintf( (char *)szTemp, "%s/RCacheMgr?Usage",
				m_pszURLString);
	printOffset( (void *)pRCacheMgr, (void *)&pRCacheMgr->Usage, szOffset);
	fnPrintf( m_pHRequest, TD_s, szOffset);									// Field offset
	fnPrintf( m_pHRequest, TD_a_p_s, szTemp, "Usage");						// Link & Name 
	fnPrintf( m_pHRequest, TD_s, "FLM_CACHE_USAGE");						// Type
	fnPrintf( m_pHRequest, TD_a_p_x, szTemp, (FLMUINT)&pRCacheMgr->Usage);	// Link & Value
	printTableRowEnd();
	
	// ppHashBuckets
	if (pRCacheMgr->ppHashBuckets)
	{
		f_sprintf( (char *)szTemp, "%s/RCHashBucket?Start=0",
					m_pszURLString);
	}

	printHTMLLink(
		"ppHashBuckets",
		"RCACHE **",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->ppHashBuckets,
		(void *)pRCacheMgr->ppHashBuckets,
		(char*)szTemp,
		(bHighlight = ~bHighlight));



	// uiNumBuckets
	printHTMLUint(
		"uiNumBuckets",
		"FLMUINT",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->uiNumBuckets,
		pRCacheMgr->uiNumBuckets,
		(bHighlight = ~bHighlight));



	// uiHashMask
	printHTMLUint(
		"uiHashMask",
		"FLMUINT",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->uiHashMask,
		pRCacheMgr->uiHashMask,
		(bHighlight = ~bHighlight));




	// uiPendingReads
	printHTMLUint(
		"uiPendingReads",
		"FLMUINT",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->uiPendingReads,
		pRCacheMgr->uiPendingReads,
		(bHighlight = ~bHighlight));





	// uiIoWaits
	printHTMLUint(
		"uiIoWaits",
		"FLMUINT",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->uiIoWaits,
		pRCacheMgr->uiIoWaits,
		(bHighlight = ~bHighlight));





	// hMutex
	printAddress( &pRCacheMgr->hMutex, szAddress);
	printHTMLString(
		"hMutex",
		"F_MUTEX",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->hMutex,
		(char*)szAddress,
		(bHighlight = ~bHighlight));




#ifdef FLM_DEBUG


	// bDebug
	printHTMLString(
		"bDebug",
		"F_MUTEX",
		(void *)pRCacheMgr,
		(void *)&pRCacheMgr->bDebug,
		(char *)(pRCacheMgr->bDebug ? "Yes" : "No"),
		(bHighlight = ~bHighlight));


#endif

Exit:

	printTableEnd();
	if (bFlaimLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bFlaimLocked = FALSE;
	}

	if (pPurgeList)
	{
		f_free( &pPurgeList);
	}

	if (pMRURecord)
	{
		f_free( &pMRURecord);
	}
	
	if (pLRURecord)
	{
		f_free( &pLRURecord);
	}

}

/****************************************************************************
Desc:		Implements the RCache display function.
*****************************************************************************/
RCODE F_RCachePage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	char					szContainer[GENERIC_SIZE];
	FLMUINT				uiContainer;
	char					szDrn[GENERIC_SIZE];
	FLMUINT				uiDrn;
	char					szVersion[GENERIC_SIZE];
	FLMUINT				uiVersion;
	char					szTemp[GENERIC_SIZE];
	RCACHE *				pRCache = NULL;
	RCACHE *				pOlderRCache;
	RCACHE *				pNewerRCache;
	char					szFile[GENERIC_SIZE];
	FFILE *				pFile;
	char					szFrom[GENERIC_SIZE];
	char					szBucket[GENERIC_SIZE];
	FLMUINT				uiBucket;
	FLMBOOL				bRCLocked = FALSE;
	FLMBOOL				bpFileInc = FALSE;
	FLMBYTE *			pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Check to see if this page is being invoked from the RCHashBucket
	// Page.  If it is, then all we will need is the Bucket and we should
	// be able to find the RCache block.
	if (RC_BAD( rc = ExtractParameter( uiNumParams, 
												  ppszParams, 
												  "From", 
												  sizeof( szFrom),
												  szFrom)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
		else
		{
			szFrom[0] = '\0';
		}
	}

	if (f_strcmp( szFrom, "RCHashBucket")==0)
	{
		if (RC_BAD(rc = ExtractParameter(uiNumParams, 
													ppszParams, 
													"Bucket", 
													sizeof( szBucket),
													szBucket)))
		{
			goto Exit;
		}
		uiBucket = f_atoud( szBucket);

		// We need to lock the RCache Manager before messing with the RCache records

		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		bRCLocked = TRUE;

		if ((pRCache = gv_FlmSysData.RCacheMgr.ppHashBuckets[uiBucket]) == NULL)
		{
			goto Exit;
		}
		
		uiContainer = pRCache->uiContainer;
		uiDrn = pRCache->uiDrn;
		uiVersion = pRCache->uiLowTransId;
		pFile = pRCache->pFile;

		pRCache = NULL; // We will retrieve this again later.

		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bRCLocked = FALSE;

	}
	else
	{

		// Container tag
		if (RC_BAD( rc = ExtractParameter( uiNumParams, 
													  ppszParams, 
													  "Container", 
													  sizeof( szContainer),
													  szContainer)))
		{
			goto Exit;
		}

		uiContainer = f_atoud( szContainer);
		
		// DRN tag
		if (RC_BAD(rc = ExtractParameter( uiNumParams, 
													 ppszParams, 
													 "DRN", 
													 sizeof( szDrn),
													 szDrn)))
		{
			goto Exit;
		}

		uiDrn = f_atoud( szDrn);

		// File tag
		if (RC_BAD(rc = ExtractParameter( uiNumParams, 
													 ppszParams, 
													 "File", 
													 sizeof( szFile),
													 szFile)))
		{
			goto Exit;
		}
		pFile = (FFILE *)f_atoud( szFile);

		// Version tag
		if (RC_BAD(rc = ExtractParameter( uiNumParams, 
													 ppszParams, 
													 "Version", 
													 sizeof( szVersion),
													 szVersion)))
		{
			goto Exit;
		}
		uiVersion = f_atoud( szVersion);
	}


	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Determine if we are being requested to refresh this page or  not.
	if (DetectParameter( uiNumParams, ppszParams, "Refresh"))
	{
		// Send back the page with a refresh command in the header
		f_sprintf((char *)szTemp, 
					"%s/RCache?Refresh?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);

		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>RCache</TITLE>\n",
			szTemp);
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n");


		f_sprintf((char *)szTemp,
					"%s/RCache?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);

		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
               "<A HREF=%s>Stop Auto-refresh</A>", szTemp);
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>RCache</TITLE>\n");
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n");

		f_sprintf((char *)szTemp,
					"%s/RCache?Refresh?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);
		
		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf( (char *)pszTemp,
			         "<A HREF=%s>Start Auto-refresh (5 sec.)</A>", szTemp);
	}

	// Prepare the Refresh link.
	f_sprintf((char *)szTemp,
					"%s/RCache?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);


	// Need to lock the record Cache mutex first...
	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
	bRCLocked = TRUE;

	flmRcaFindRec( pFile, F_SEM_NULL, uiContainer, uiDrn, uiVersion, TRUE, 
					  0, &pRCache, &pNewerRCache, &pOlderRCache);

	// We want to hold the RCache and pFile in memory while we render this page.
	if (pRCache)
	{

		RCA_INCR_USE_COUNT( pRCache->uiFlags);
		if (++pRCache->pFile->uiUseCount == 1)
		{
			flmUnlinkFileFromNUList( pFile);
		}
		bpFileInc = TRUE;

	}

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bRCLocked = FALSE;

	if (pRCache)
	{
		// Begin the table
		printTableStart( "RCache", 4);

		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
		fnPrintf( m_pHRequest, "<a href=%s>Refresh</a>, ", szTemp);
		fnPrintf( m_pHRequest, "%s\n", pszTemp);
		printColumnHeadingClose();
		printTableRowEnd();
		
		// Write out the table headings
		
		printTableRowStart();
		printColumnHeading( "Byte Offset (hex)");
		printColumnHeading( "Field Name");
		printColumnHeading( "Field Type");
		printColumnHeading( "Value");
		printTableRowEnd();

		write_data(pRCache);
	}
	else
	{
		// Return an error page
		fnPrintf( m_pHRequest,
			"<P>Unable to find the RCache structure that you requested."
			"  This is probably because the state of the cache changed "
			"between the time that you displayed the previous page and the time "
			"that you clicked on the link that brought you here.\n"
			"<P>Click on your browser's \"Back\" button, then click \"Reload\" "
			"and then try the link again.</P>\n");

	}


	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();

	// Now decrement the use count on the pFile and on the RCache
	if (pRCache)
	{
		if (bpFileInc)
		{
			if (--pRCache->pFile->uiUseCount == 0)
			{
				flmLinkFileToNUList( pRCache->pFile);
			}
			bpFileInc = FALSE;
		}
		RCA_DECR_USE_COUNT( pRCache->uiFlags);
	}

Exit:

	if ( bRCLocked)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);
}




/****************************************************************************
 Desc: This is procedure for generating HTML content for an RCACHE structure
 ****************************************************************************/
void F_RCachePage::write_data(
	RCACHE *				pRCache)
{
	char					szTemp[GENERIC_SIZE];
	char					szAddress[20];
	FLMBOOL				bHighlight = FALSE;

	F_UNREFERENCED_PARM( fnPrintf);

	if (!pRCache)
	{
		flmAssert(0);
		return;
	}
	else
	{

		// pRecord
		if (pRCache->pRecord)
		{
			printAddress( pRCache->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/Record?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->uiContainer,
						(unsigned long)pRCache->uiDrn, 
						szAddress,
						(unsigned long)pRCache->uiLowTransId);
		}
		
		printHTMLLink(
			"pRecord",
			"FlmRecord *",
			(void *)pRCache,
			(void *)&pRCache->pRecord,
			(void *)pRCache->pRecord,
			(char*)szTemp,
			(bHighlight = ~bHighlight));




		// pFile
		if (pRCache->pFile)
		{
			printAddress( pRCache->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/FFile?From=RCache?Bucket=%lu?Address=%s",
						m_pszURLString,
						(unsigned long)pRCache->pFile->uiBucket,
						szAddress);
		}
		
		printHTMLLink(
			"pFile",
			"FFILE *",
			(void *)pRCache,
			(void *)&pRCache->pFile,
			(void *)pRCache->pFile,
			(char*)szTemp,
			(bHighlight = ~bHighlight));




		// uiContainer
		printHTMLUint(
			"uiContainer",
			"FLMUINT",
			(void *)pRCache,
			(void *)&pRCache->uiContainer,
			pRCache->uiContainer,
			(bHighlight = ~bHighlight));



		// uiDrn
		printHTMLUint(
			"uiDrn",
			"FLMUINT",
			(void *)pRCache,
			(void *)&pRCache->uiDrn,
			pRCache->uiDrn,
			(bHighlight = ~bHighlight));




		// uiLowTransId
		printHTMLUint(
			"uiLowTransId",
			"FLMUINT",
			(void *)pRCache,
			(void *)&pRCache->uiLowTransId,
			pRCache->uiLowTransId,
			(bHighlight = ~bHighlight));


		// uiHighTransId
		printHTMLUint(
			"uiHighTransId",
			"FLMUINT",
			(void *)pRCache,
			(void *)&pRCache->uiHighTransId,
			pRCache->uiHighTransId,
			(bHighlight = ~bHighlight));




		// pNextInBucket
		if (pRCache->pNextInBucket)
		{
			printAddress( pRCache->pNextInBucket->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pNextInBucket->uiContainer,
						(unsigned long)pRCache->pNextInBucket->uiDrn,
						szAddress,
						(unsigned long)pRCache->pNextInBucket->uiLowTransId);
		}


		printHTMLLink(
			"pNextInBucket",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pNextInBucket,
			(void *)pRCache->pNextInBucket,
			(char*)szTemp,
			(bHighlight = ~bHighlight));





		// pPrevInBucket
		if (pRCache->pPrevInBucket)
		{
			printAddress( pRCache->pPrevInBucket->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pPrevInBucket->uiContainer,
						(unsigned long)pRCache->pPrevInBucket->uiDrn,
						szAddress,
						(unsigned long)pRCache->pPrevInBucket->uiLowTransId);
		}

		printHTMLLink(
			"pPrevInBucket",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pPrevInBucket,
			(void *)pRCache->pPrevInBucket,
			(char*)szTemp,
			(bHighlight = ~bHighlight));




		// pNextInFile
		if (pRCache->pNextInFile)
		{
			printAddress( pRCache->pNextInFile->pFile, szAddress);
			f_sprintf((char *)szTemp, 
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pNextInFile->uiContainer,
						(unsigned long)pRCache->pNextInFile->uiDrn,
						szAddress,
						(unsigned long)pRCache->pNextInFile->uiLowTransId);
		}
		
		printHTMLLink(
			"pNextInFile",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pNextInFile,
			(void *)pRCache->pNextInFile,
			(char*)szTemp,
			(bHighlight = ~bHighlight));




		// pPrevInFile
		if (pRCache->pPrevInFile)
		{
			printAddress( pRCache->pPrevInFile->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pPrevInFile->uiContainer,
						(unsigned long)pRCache->pPrevInFile->uiDrn,
						szAddress,
						(unsigned long)pRCache->pPrevInFile->uiLowTransId);
		}
		
		printHTMLLink(
			"pPrevInFile",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pPrevInFile,
			(void *)pRCache->pPrevInFile,
			(char*)szTemp,
			(bHighlight = ~bHighlight));




		// pNextInGlobal
		if (pRCache->pNextInGlobal)
		{
			printAddress( pRCache->pNextInGlobal->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pNextInGlobal->uiContainer,
						(unsigned long)pRCache->pNextInGlobal->uiDrn,
						szAddress,
						(unsigned long)pRCache->pNextInGlobal->uiLowTransId);
		}

		printHTMLLink(
			"pNextInGlobal",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pNextInGlobal,
			(void *)pRCache->pNextInGlobal,
			(char*)szTemp,
			(bHighlight = ~bHighlight));



		// pPrevInGlobal
		if (pRCache->pPrevInGlobal)
		{
			printAddress( pRCache->pPrevInGlobal->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pPrevInGlobal->uiContainer,
						(unsigned long)pRCache->pPrevInGlobal->uiDrn,
						szAddress,
						(unsigned long)pRCache->pPrevInGlobal->uiLowTransId);
		}
		
		printHTMLLink(
			"pPrevInGlobal",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pPrevInGlobal,
			(void *)pRCache->pPrevInGlobal,
			(char*)szTemp,
			(bHighlight = ~bHighlight));



		// pOlderVersion
		if (pRCache->pOlderVersion)
		{
			printAddress( pRCache->pOlderVersion->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pOlderVersion->uiContainer,
						(unsigned long)pRCache->pOlderVersion->uiDrn,
						szAddress,
						(unsigned long)pRCache->pOlderVersion->uiLowTransId);
		
		}
		
		printHTMLLink(
			"pOlderVersion",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pOlderVersion,
			(void *)pRCache->pOlderVersion,
			(char*)szTemp,
			(bHighlight = ~bHighlight));



		// pNewerVersion
		if (pRCache->pNewerVersion)
		{
			printAddress( pRCache->pNewerVersion->pFile, szAddress);
			f_sprintf((char *)szTemp,
						"%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
						m_pszURLString,
						(unsigned long)pRCache->pNewerVersion->uiContainer,
						(unsigned long)pRCache->pNewerVersion->uiDrn,
						szAddress,
						(unsigned long)pRCache->pNewerVersion->uiLowTransId);
		}
		
		printHTMLLink(
			"pNewerVersion",
			"RCACHE *",
			(void *)pRCache,
			(void *)&pRCache->pNewerVersion,
			(void *)pRCache->pNewerVersion,
			(char*)szTemp,
			(bHighlight = ~bHighlight));



		// pNotifyList
		if (pRCache->pNotifyList)
		{
			printAddress( pRCache->pNotifyList, szAddress);
			f_sprintf((char *)szTemp, "%s/FNOTIFY?From=RCache?Address=%s",
						m_pszURLString,
						szAddress);
		}
		
		printHTMLLink(
			"pNotifyList",
			"FNOTIFY *",
			(void *)pRCache,
			(void *)&pRCache->pNotifyList,
			(void *)pRCache->pNotifyList,
			(char*)szTemp,
			(bHighlight = ~bHighlight));



		// uiFlags
		printHTMLUint(
			"uiFlags",
			"FLMUINT",
			(void *)pRCache,
			(void *)&pRCache->uiFlags,
			pRCache->uiFlags,
			(bHighlight = ~bHighlight));

		printTableEnd();

	}
}


/****************************************************************************
Desc:		Implements the Record display function.
*****************************************************************************/
RCODE F_RecordPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{

	RCODE					rc = FERR_OK;
	char					szContainer[GENERIC_SIZE];
	FLMUINT				uiContainer;
	char					szDrn[GENERIC_SIZE];
	FLMUINT				uiDrn;
	char					szVersion[GENERIC_SIZE];
	FLMUINT				uiVersion;
	char					szTemp[GENERIC_SIZE];
	RCACHE *				pRCache = NULL;
	RCACHE *				pOlderRCache;
	RCACHE *				pNewerRCache;
	char					szFile[GENERIC_SIZE];
	FFILE *				pFile;
	FlmRecord *			pRecord = NULL;
	FLMBOOL				bpFileInc = FALSE;
	FLMBYTE *			pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Container tag
	if (RC_BAD( rc = ExtractParameter( uiNumParams, 
												  ppszParams, 
												  "Container", 
												  sizeof( szContainer),
												  szContainer)))
	{
		goto Exit;
	}
	uiContainer = f_atoud( szContainer);

	// DRN tag
	if (RC_BAD( rc = ExtractParameter( uiNumParams, 
												  ppszParams, 
												  "DRN", 
												  sizeof( szDrn),
												  szDrn)))
	{
		goto Exit;
	}
	uiDrn = f_atoud( szDrn);

	// File tag
	if (RC_BAD( rc = ExtractParameter( uiNumParams, 
												  ppszParams, 
												  "File", 
												  sizeof( szFile),
												  szFile)))
	{
		goto Exit;
	}
	pFile = (FFILE *)f_atoud( szFile);



	// Version tag
	if (RC_BAD( rc = ExtractParameter( uiNumParams, 
												  ppszParams, 
												  "Version", 
												  sizeof( szVersion),
												  szVersion)))
	{
		goto Exit;
	}
	uiVersion = f_atoud( szVersion);

	stdHdr();
	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Determine if we are being requested to refresh this page or  not.
	if (DetectParameter( uiNumParams, ppszParams, "Refresh"))
	{
		// Send back the page with a refresh command in the header
		f_sprintf((char *)szTemp, 
					"%s/Record?Refresh?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);

		fnPrintf( m_pHRequest, 
			"<HEAD><META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>Database iMonitor - gv_FlmSysData</TITLE>\n", szTemp);
		printRecordStyle();
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n");

		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf((char *)szTemp,
					"%s/Record?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);

		f_sprintf( (char *)pszTemp,
					"<A HREF=%s>Stop Auto-refresh</A>", szTemp);
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>Database iMonitor - gv_FlmSysData</TITLE>\n");
		printRecordStyle();
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n");

		fnPrintf( m_pHRequest, "<body>\n");

		f_sprintf((char *)szTemp,
					"%s/Record?Refresh?Container=%s?DRN=%s?File=%s?Version=%s",
					m_pszURLString,
					szContainer, szDrn, szFile, szVersion);

		f_sprintf( (char *)pszTemp,
					"<A HREF=%s>Start Auto-refresh (5 sec.)</A>", szTemp);
	}

	// Prepare the refresh link.
	f_sprintf((char *)szTemp,
				"%s/Record?Container=%s?DRN=%s?File=%s?Version=%s",
				m_pszURLString,
				szContainer, szDrn, szFile, szVersion);
	

	// Need to lock the record Cache mutex first...
	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

	flmRcaFindRec( pFile, F_SEM_NULL, uiContainer, uiDrn, uiVersion, 
					   TRUE, 0, &pRCache, &pNewerRCache, &pOlderRCache);

	if (pRCache)
	{
		// Keep the record in memory until we finish rendering this page
		RCA_INCR_USE_COUNT( pRCache->uiFlags);
		if (++pRCache->pFile->uiUseCount == 1)
		{
			flmUnlinkFileFromNUList( pRCache->pFile);
		}
		bpFileInc = TRUE;
		pRecord = pRCache->pRecord;
	}

	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);


	printTableStart( "DB Record", 1, 100);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 1, 1, FALSE);
	fnPrintf( m_pHRequest, "<a href=%s>Refresh</a>, ", szTemp);
	fnPrintf( m_pHRequest, "%s\n", pszTemp);
	printColumnHeadingClose();
	printTableRowEnd();

	printTableEnd();

	write_links( pRCache);

	write_data( pRecord, pRCache);

	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();
	
Exit:

	if (pRCache)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		// Decrement the use count on the record
		if (bpFileInc)
		{
			if (--pRCache->pFile->uiUseCount == 0)
			{
				flmLinkFileToNUList( pRCache->pFile);
			}
		}
		RCA_DECR_USE_COUNT( pRCache->uiFlags);
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (pszTemp)
	{
		f_free( &pszTemp);
	}


	return( rc);
}


/****************************************************************************
 Desc: This is a procedure for generating HTML content to display links to
		 next and previous records from the Record page, rather than having to
		 backup to the RCache page first.
 ****************************************************************************/
void F_RecordPage::write_links(
	RCACHE *				pRCache)
{
	FLMUINT		uiContainer;
	FLMUINT		uiDrn;
	FLMUINT		uiVersion;
	void *		pFile;
	RCACHE *		pTmpRCache;
	char			szAddress[20];

	
	// Do nothing if there is no RCache to report on.
	if (!pRCache)
	{
		return;
	}
	
	// Begin the table
	printTableStart( "DB Record - Links", 8, 100);

	printTableRowStart();
	// pNextInBucket
	if (pRCache->pNextInBucket)
	{
		pTmpRCache = pRCache->pNextInBucket;

		// Extract the container etc.
		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pNextInBucket</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pNextInBucket</TD>\n");
	}
	




	// pPrevInBucket
	if (pRCache->pPrevInBucket)
	{
		pTmpRCache = pRCache->pPrevInBucket;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pPrevInBucket</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pPrevInBucket</TD>\n");
	}
	




	// pNextInFile
	if (pRCache->pNextInFile)
	{
		pTmpRCache = pRCache->pNextInFile;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pNextInFile</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pNextInFile</TD>\n");
	}





	// pPrevInFile
	if (pRCache->pPrevInFile)
	{	
		pTmpRCache = pRCache->pPrevInFile;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pPrevInFile</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pPrevInFile</TD>\n");
	}






	// pNextInGlobal
	if (pRCache->pNextInGlobal)
	{
		pTmpRCache = pRCache->pNextInGlobal;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%u&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pNextInGlobal</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pNextInGlobal</TD>\n");
	}







	// pPrevInGlobal
	if (pRCache->pPrevInGlobal)
	{
		pTmpRCache = pRCache->pPrevInGlobal;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pPrevInGlobal</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pPrevInGlobal</TD>\n");
	}




	// pOlderVersion
	if (pRCache->pOlderVersion)
	{	
		pTmpRCache = pRCache->pOlderVersion;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pOlderVersion</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pOlderVersion</TD>\n");
	}





	// pNewerVersion
	if (pRCache->pNewerVersion)
	{	
		pTmpRCache = pRCache->pNewerVersion;

		uiContainer = pTmpRCache->uiContainer;
		uiDrn = pTmpRCache->uiDrn; 
		pFile = (void *)pTmpRCache->pFile;
		uiVersion = pTmpRCache->uiLowTransId;

		printAddress( pFile, szAddress);
		fnPrintf( m_pHRequest, 
			"<TD>"
			"<a href=%s/Record?"
			"Container=%lu&DRN=%lu&"
			"File=%s&Version=%lu>"
			"pNewerVersion</a></TD>\n",
			m_pszURLString,
			(unsigned long)uiContainer,
			(unsigned long)uiDrn,
			szAddress,
			(unsigned long)uiVersion);
	}
	else
	{
		fnPrintf( m_pHRequest, "<TD>pNewerVersion</TD>\n");
	}
	printTableRowEnd();

	printTableEnd();

}


/****************************************************************************
 Desc: This is a procedure for generating HTML content to display the results
       of querying the various methods in a FlmRecord
 ****************************************************************************/
void F_RecordPage::write_data(
	FlmRecord *				pRecord,
	RCACHE *					pRCache)
{
	FLMBOOL				bHighlight = FALSE;

	// Do we have a valid reference to an record?
	if (!pRecord)
	{
		// Return an error page
		fnPrintf( m_pHRequest,
			"<P> Unable to find the Record that you requested."
			"  This is probably because the state of the cache changed "
			"between the time that you displayed the previous page and the time "
			"that you clicked on the link that brought you here.\n"
			"<P>Click on your browser's \"Back\" button, then click \"Reload\" "
			"and then try the link again.\n");
	}
	else
	{

		// Begin the table
		printTableStart( "DB Record - Methods", 2, 100);
		
		
		// Write out the table headings
		printTableRowStart();
		printColumnHeading( "Method Name");
		printColumnHeading( "Value");
		printTableRowEnd();
		
		
		
		// getID()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "getID");
		fnPrintf( m_pHRequest, TD_ui, pRecord->getID());
		printTableRowEnd();
		
		
		
		
		// getContainerID()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "getContainerID");
		fnPrintf( m_pHRequest, TD_ui, pRecord->getContainerID());
		printTableRowEnd();
		
		// isReadOnly()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "isReadOnly");
		fnPrintf( m_pHRequest, TD_s, pRecord->isReadOnly() ? "Yes" : "No");
		printTableRowEnd();

		// getTotalMemory()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "getTotalMemory");
		fnPrintf( m_pHRequest, TD_ui, pRecord->getTotalMemory());
		printTableRowEnd();
		
		// getFreeMemory()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "getFreeMemory");
		fnPrintf( m_pHRequest, TD_ui, pRecord->getFreeMemory());
		printTableRowEnd();

		// getRefCount()
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "getRefCount");
		fnPrintf( m_pHRequest, TD_ui, pRecord->getRefCount());
		printTableRowEnd();


		// End the table
		printTableEnd();

		// Begin a new table to display the fields
		printTableStart( "DB Record - Fields", 4);
		
		
		// Write out the new table headings
		printTableRowStart();
		printColumnHeading( "Byte Offset (hex)");
		printColumnHeading( "Field Name");
		printColumnHeading( "Field Type");
		printColumnHeading( "Value");
		printTableRowEnd();
		
		// End the table
		printTableEnd();

		// At this point, we will extract the various fields and display each
		// one according to the structure of the record.

		printRecordFields( pRecord, pRCache);
	}
}

/****************************************************************************
 Desc: This is a procedure for generating HTML content to display
		 the fields of a FlmRecord.
 ****************************************************************************/
void F_RecordPage::printRecordFields(
	FlmRecord *				pRecord,
	RCACHE *					pRCache)
{
	RCODE				rc = FERR_OK;
	F_NameTable *	pNameTable = NULL;
	FLMUINT			uiContext = 0;

	// Return if we have nothing to display.
	if( !pRecord)
	{
		goto Exit;
	}

	if (!m_pFlmSession)
	{
		fnPrintf( m_pHRequest, 
			"<center>Cannot display record data.  No session object available. "
			"Return Code = 0x%04X (%s)</center>\n", m_uiSessionRC, FlmErrorString( m_uiSessionRC));
		goto Exit;
	}

	if (RC_BAD( rc = m_pFlmSession->getNameTable( pRCache->pFile, &pNameTable)))
	{
		fnPrintf( m_pHRequest,
			"<center>Cannot display record data.  Could not get a Name Table."
			"Return Code = 0x%04X (%s)</center>\n", m_uiSessionRC, FlmErrorString( m_uiSessionRC));
		goto Exit;
	}

	// We will call this when the code is ready...
	printRecord( NULL, pRecord, pNameTable, &uiContext, TRUE);


Exit:

	return;

}


/****************************************************************************
Desc:	Prints the web page for the RCHashBucket
****************************************************************************/
RCODE F_RCHashBucketPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bRefresh;
	FLMBOOL		bHighlight = TRUE;
  	FLMUINT		uiLoop;
	FLMUINT		uiHashTableSize;
	FLMUINT		uiUsedEntries = 0;
	char			szStart[10];
	char			szRefresh[] = "&Refresh";
	FLMUINT		uiStart;
	FLMUINT		uiNewStart;
	FLMBYTE *	pszTemp;
	FLMBOOL		bNextUsed;
	FLMUINT		uiNextStart;

	// We display 20 hash table entries at a time, some of which might need
	// to be hyperlinked.
#define NUM_ENTRIES 20
	FLMBYTE *	pszHTLinks[NUM_ENTRIES];

	// Check for the refresh parameter
	bRefresh = DetectParameter( uiNumParams, ppszParams, "Refresh");
	if (!bRefresh)
	{
		szRefresh[0]='\0';  // Effectively turns szRefresh into a null string
	}

	// Get the starting entry number...
	if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
												  "Start", sizeof( szStart),
												  szStart)))
	{  
		flmAssert( 0);  
		goto Exit;
	}
	uiStart = f_atoud( szStart);

	// Allocate space for the hyperlink text
	for (uiLoop = 0; uiLoop < NUM_ENTRIES; uiLoop++)
	{
		if( RC_BAD( rc = f_alloc( 250, &pszHTLinks[ uiLoop])))
		{
			printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
			goto Exit;
		}

		pszHTLinks[uiLoop][0] = '\0';
	}

	if( RC_BAD( rc = f_alloc( 250, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Find out if we want to go to the next or previous used entry.
	bNextUsed = DetectParameter( uiNumParams, ppszParams, "NextUsed");

	// Lock the database
	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);

	// Get the number of entries in the hash table
	uiHashTableSize = gv_FlmSysData.RCacheMgr.uiNumBuckets;
	
	// May need to modify starting number if it's out of range...
	if ((uiStart + NUM_ENTRIES) >= uiHashTableSize)
	{
		uiStart = uiHashTableSize - NUM_ENTRIES;
	}

	if (bNextUsed)
	{
		uiNextStart = uiStart + NUM_ENTRIES;
		if ((uiNextStart + NUM_ENTRIES) >= uiHashTableSize)
		{
			uiNextStart = uiHashTableSize - NUM_ENTRIES;
		}
	}

	// We need to find out if there are any used entries after our uiNextStart
	// index into the hash table.  To do this, we will need to scan the table
	// first, just to find out if there are any used entries...
	if (bNextUsed)
	{
		for (uiLoop = 0; uiLoop < uiHashTableSize; uiLoop++)
		{
			if (gv_FlmSysData.RCacheMgr.ppHashBuckets[ uiLoop])
			{
				if ( uiLoop >= uiNextStart)
				{
					uiStart = uiLoop - (uiLoop % NUM_ENTRIES);
					break;  // No need to go further
				}
			}
		}
	}

	// Loop through the entire table counting the number of entries in use
	// If the entry is one of the one's we're going to display, store the 
	// appropriate text in pszHTLinks
	for (uiLoop = 0; uiLoop < uiHashTableSize; uiLoop++)
	{
		if (gv_FlmSysData.RCacheMgr.ppHashBuckets[ uiLoop])
		{
			uiUsedEntries++;
		}

		if (	(uiLoop >= uiStart) &&
				(uiLoop < (uiStart + NUM_ENTRIES)) )
		{
			// This is one of the entries that we will display
			if (gv_FlmSysData.RCacheMgr.ppHashBuckets[ uiLoop])
			{
				flmBuildRCacheLink( (char *)pszHTLinks[uiLoop - uiStart],
					gv_FlmSysData.RCacheMgr.ppHashBuckets[ uiLoop], m_pszURLString);
			}
		}
	}

	// Unlock the database
	f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);


	// Begin rendering the page...
	stdHdr();

	printStyle();
	fnPrintf( m_pHRequest, HTML_DOCTYPE "<html>\n");

	// Determine if we are being requested to refresh this page or  not.

	if (bRefresh)
	{
		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s/RCHashBucket?Start=%lu%s\">"
			"<TITLE>Database iMonitor - RCache Hash Bucket</TITLE>\n", m_pszURLString, uiStart, szRefresh);
	
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD>\n");
	}


	// If we are not to refresh this page, then don't include the
	// refresh meta command
	if (!bRefresh)
	{
		f_sprintf( (char *)pszTemp,
			       "<A HREF=%s/RCHashBucket?Start=%lu&Refresh>Start Auto-refresh (5 sec.)</A>",
					 m_pszURLString, uiStart);
	}
	else
	{
		f_sprintf( (char *)pszTemp,
			       "<A HREF=%s/RCHashBucket?Start=%lu>Stop Auto-refresh</A>",
					 m_pszURLString, uiStart);
	}

	// Print out a formal header and the refresh option.
	printTableStart("RCache Hash Bucket", 4);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/RCHashBucket?Start=%lu%s>Refresh</A>, %s\n",
				 m_pszURLString, uiStart, szRefresh, pszTemp);
	printColumnHeadingClose();
	printTableRowEnd();
		
	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, "<TD>Table Size: %lu </TD>\n", uiHashTableSize);
	printTableRowEnd();

	printTableRowStart( (bHighlight = !bHighlight));
	fnPrintf( m_pHRequest, "<TD>Entries Used: %lu (%lu%%) </TD>\n", uiUsedEntries,
				 ((uiUsedEntries * 100) / uiHashTableSize) );
	printTableRowEnd();
	
	// The rest of the table is going to be a single row with two columns:  
	// one for the list of hash buckets and the other for everything else

	printTableRowStart( FALSE);
	fnPrintf( m_pHRequest, " <TD>\n");
	// Print out the hash buckets
	for (uiLoop = 0; uiLoop < NUM_ENTRIES; uiLoop++)
	{
		if (pszHTLinks[uiLoop][0] != '\0')
		{
			fnPrintf( m_pHRequest, "<A HREF=%s%s>%lu</A> <br>\n",
						 pszHTLinks[uiLoop], szRefresh, uiStart+uiLoop);
		}
		else
		{
			fnPrintf( m_pHRequest, "%lu<br>\n", uiStart+uiLoop);
		}
	}

	fnPrintf( m_pHRequest, "</ul>\n</TD>\n<TD>\n");

	// Print out the other stuff...
	uiNewStart = (uiStart > 1000)?(uiStart - 1000):0;
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Previous 1000</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);
	uiNewStart = (uiStart > 100)?(uiStart - 100):0;
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Previous 100</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);
	uiNewStart = (uiStart > 10)?(uiStart - 10):0;
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Previous 10</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);

	fnPrintf( m_pHRequest, "<BR>\n");
	uiNewStart = (uiStart + 10);
	if (uiNewStart >= (uiHashTableSize - NUM_ENTRIES))
	{
		uiNewStart = (uiHashTableSize - NUM_ENTRIES);
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Next 10</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);

	uiNewStart = (uiStart + 100);
	if (uiNewStart >= (uiHashTableSize - NUM_ENTRIES))
	{
		uiNewStart = (uiHashTableSize - NUM_ENTRIES);
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Next 100</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);

	uiNewStart = (uiStart + 1000);
	if (uiNewStart >= (uiHashTableSize - NUM_ENTRIES))
	{
		uiNewStart = (uiHashTableSize - NUM_ENTRIES);
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/RCHashBucket?Start=%lu%s>Next 1000</A> <BR>\n"
				"<A HREF=%s/RCHashBucket?Start=%lu%s&NextUsed>Next Used Bucket</A> <BR>\n"
				"<form type=\"submit\" method=\"get\" action=\"%s/RCHashBucket\">\n"
				"<BR> Jump to specific bucket:<BR> \n"
				"<INPUT type=\"text\" size=\"10\" maxlength=\"10\" name=\"Start\"></INPUT> <BR>\n",
				m_pszURLString, uiNewStart, szRefresh, m_pszURLString, uiStart, szRefresh, m_pszURLString);
	printButton( "Jump", BT_Submit);
	fnPrintf( m_pHRequest, "<BR>\n");
				
	// We use a hidden field to pass the refresh parameter back the the server
	if (bRefresh)
	{
		fnPrintf( m_pHRequest, "<INPUT type=\"hidden\" name=\"Refresh\"></INPUT>\n");
	}
	fnPrintf( m_pHRequest, "</form>\n</TD>\n");

	printTableRowEnd();

	printTableEnd();
	printDocEnd();
	fnEmit();

Exit:
	// Free the space for the hyperlink text
	for (uiLoop = 0; uiLoop < NUM_ENTRIES; uiLoop++)
	{
		f_free( &pszHTLinks[uiLoop]);
	}

	f_free( &pszTemp);

	return( rc);

}

/****************************************************************************
 Desc:	Determines the values of the parameters needed to reference
			a specific RCache structure.  Must be called from within a mutex
****************************************************************************/
FSTATIC void flmBuildRCacheLink(
	char *		pszString,
	RCACHE *		pRCache,
	char *		pszURLString)
{
	char		szAddress[20];

	if (pRCache == NULL)
	{
		pszString[0] = 0;
	}
	else
	{
		printAddress( pRCache->pFile, szAddress);
		f_sprintf((char *)pszString, 
					"%s/RCache?Container=%lu&DRN=%lu&File=%s&Version=%lu",
					pszURLString,
					(unsigned long)pRCache->uiContainer,
					(unsigned long)pRCache->uiDrn,
					szAddress,
					(unsigned long)pRCache->uiLowTransId);
	}
	return;
}
