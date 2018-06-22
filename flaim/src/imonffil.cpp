//-------------------------------------------------------------------------
// Desc:	Class for displaying an FFILE structure in HTML on a web page.
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
 Desc:	This function handle the details of extracting the parameters
			needed to interpret the request and then generating the response
			HTML page
 ****************************************************************************/
RCODE F_FFilePage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
#define GENERIC_SIZE_B 20
	char				szFrom[ GENERIC_SIZE_B];
	char				szBucket[ 4];
	FLMUINT			uiBucket;
	FFILE				localFFile;
	FFILE *			pFile;
	FLMBOOL			bRefresh;
	void *			pvAddress;
	char				szAddress[GENERIC_SIZE_B];
	char				szLink[GENERIC_SIZE_B];
	FLMBOOL			bFlmLocked = FALSE;
	DATASTRUCT		DataStruct;
	FLMBYTE *		pszTemp = NULL;
	FLMBYTE *		pszTemp1 = NULL;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 150, &pszTemp1)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Initialize a few variables first...
	szFrom[0] = '\0';
	szBucket[0] = '\0';

	pFile = NULL;

	
	// Get the "From" parameter.  We use this to determine everything else.
	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "From",
												  sizeof( szFrom),
												  szFrom)))
	{
		goto Exit;
	}
	
	f_mutexLock( gv_FlmSysData.hShareMutex);
	bFlmLocked = TRUE;


	if (!f_stricmp( szFrom, "FileHashTbl"))
	{

		//  Get the hash bucket index
		if (RC_BAD( rc = ExtractParameter( uiNumParams, 
													 ppszParams, 
													 "Bucket", 
													 sizeof( szBucket),
													 szBucket)))
		{
			goto Exit;
		}

		uiBucket = f_atoud( szBucket);
		pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket;
	}
	else if ( (f_stricmp( szFrom, "SCacheBlock") == 0) || 
				 (f_stricmp( szFrom, "RCache") == 0) ||
				 (f_stricmp( szFrom, "FDB") == 0))
	{
		// Get the FFile address and the Hash Bucket
		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "Bucket",
													  sizeof( szBucket),
													  szBucket)))
		{
			goto Exit;
		}
		
		uiBucket = f_atoud( szBucket);
		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "Address",
													  sizeof( szAddress),
													  szAddress)))
		{
			goto Exit;
		}
		
		pvAddress = (void *)f_atoud( szAddress);

		pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket;

		while (pFile && (void *)pFile != pvAddress)
		{
			pFile = pFile->pNext;
		}
		
	}
	else if (f_stricmp( szFrom, "FlmSysData") == 0)
	{
		// Get the Link and the FFile address
		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "Link",
													  sizeof( szLink),
													  szLink)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "Address",
													  sizeof( szAddress),
													  szAddress)))
		{
			goto Exit;
		}
		
		pvAddress = (void *)f_atoud( szAddress);

		if (f_stricmp( szLink, "pMrnuFile") == 0)
		{
			pFile = gv_FlmSysData.pMrnuFile;
			
			// Now let's make sure we are looking at the right FFile...
			while (pFile && (void *)pFile != pvAddress)
			{
				pFile = pFile->pNextNUFile;
			}
		}
		else if (f_stricmp( szLink, "pLrnuFile") == 0)
		{
			pFile = gv_FlmSysData.pLrnuFile;

			// Now let's make sure we are looking at the right FFile...
			while (pFile && (void *)pFile != pvAddress)
			{
				pFile = pFile->pPrevNUFile;
			}
		}

	}
	else if (f_stricmp( szFrom, "FFile") == 0)
	{
		// We need to get the Link, Bucket & Address

		if (RC_BAD(rc = ExtractParameter( uiNumParams,
													 ppszParams,
													 "Link",
													 sizeof( szLink),
													 szLink)))
		{
			goto Exit;
		}

		if (RC_BAD(rc = ExtractParameter( uiNumParams,
													 ppszParams,
													 "Address",
													 sizeof( szAddress),
													 szAddress)))
		{
			goto Exit;
		}
		
		pvAddress = (void *)f_atoud( szAddress);

		if (RC_BAD(rc = ExtractParameter( uiNumParams,
													 ppszParams,
													 "Bucket",
													 sizeof( szBucket),
													 szBucket)))
		{
			goto Exit;
		}

		uiBucket = f_atoud( szBucket);

		// First, let's get a reference to an FFile from the specified bucket

		if (gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket)
		{
			pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket;
		}

		// Now let's make sure we are looking at the right FFile...
		while (pFile && (void *)pFile != pvAddress)
		{
			pFile = pFile->pNext;
		}


		// Now what link are we supposed to follow?
		if (f_stricmp( szLink, "pNext") == 0)
		{
			pFile = pFile->pNext;
		}
		else if (f_stricmp( szLink, "pPrev") == 0)
		{
			pFile = pFile->pPrev;
		}
		else if (f_stricmp( szLink, "pNextNUFile") == 0)
		{
			pFile = pFile->pNextNUFile;
		}
		else if (f_stricmp( szLink, "pPrevNUFile") == 0)
		{
			pFile = pFile->pPrevNUFile;
		}

	}

	// Gather additional data if present. Initialize the structure before
	// using it.
	f_memset( &DataStruct, 0, sizeof(DataStruct));

	if (pFile)
	{
		f_memcpy( &localFFile, pFile, sizeof(localFFile));

		if (pFile->pSCacheList)
		{
			DataStruct.SCacheBlkAddress = pFile->pSCacheList->uiBlkAddress;
			DataStruct.SCacheLowTransID = scaGetLowTransID( pFile->pSCacheList),
			DataStruct.SCacheHighTransID = pFile->pSCacheList->uiHighTransID;
		}
		if (pFile->pPendingWriteList)
		{
			DataStruct.PendingWriteBlkAddress = pFile->pPendingWriteList->uiBlkAddress;
			DataStruct.PendingWriteLowTransID = scaGetLowTransID( pFile->pPendingWriteList),
			DataStruct.PendingWriteHighTransID = pFile->pPendingWriteList->uiHighTransID;
		}
		if (pFile->pLastDirtyBlk)
		{
			DataStruct.LastDirtyBlkAddress = pFile->pLastDirtyBlk->uiBlkAddress;
			DataStruct.LastDirtyLowTransID = scaGetLowTransID( pFile->pLastDirtyBlk),
			DataStruct.LastDirtyHighTransID = pFile->pLastDirtyBlk->uiHighTransID;
		}
		
		if (pFile->pFirstRecord)
		{
			DataStruct.FirstRecordContainer = pFile->pFirstRecord->uiContainer;
			DataStruct.FirstRecordDrn = pFile->pFirstRecord->uiDrn;
			DataStruct.FirstRecordLowTransId = pFile->pFirstRecord->uiLowTransId;
		}

		if (pFile->pLastRecord)
		{
			DataStruct.LastRecordContainer = pFile->pLastRecord->uiContainer;
			DataStruct.LastRecordDrn = pFile->pLastRecord->uiDrn;
			DataStruct.LastRecordLowTransId = pFile->pLastRecord->uiLowTransId;
		}
	}
	

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bFlmLocked = FALSE;

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest,  "<html>\n");


	// Determine if we are being requested to refresh this page or  not.
	if ((bRefresh = DetectParameter( uiNumParams,
											   ppszParams,
											   "Refresh")) == TRUE)
	{
		// Send back the page with a refresh command in the header
		f_sprintf( (char *)pszTemp, "%s/FFile?Refresh&From=%s&Bucket=%s",
					m_pszURLString,
					szFrom, szBucket);

		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>FFile Structure</TITLE>\n", pszTemp);
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>FFile Structure</TITLE>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");

	fnPrintf( m_pHRequest,  "<body>\n");

	// If we are not to refresh this page, then don't include the
	// refresh meta command
	if (!bRefresh)
	{
		f_sprintf( (char *)pszTemp,
					"<A HREF=%s/FFile?Refresh&From=%s&Bucket=%s>Start Auto-refresh (5 sec.)</A>",
					m_pszURLString, szFrom, szBucket);
	}
	else
	{
		f_sprintf( (char *)pszTemp,
               "<A HREF=%s/FFile?From=%s&Bucket=%s>Stop Auto-refresh</A>",
					m_pszURLString, szFrom, szBucket);
	}
	// Prepare the refresh link.
	f_sprintf( (char *)pszTemp1,
            "<A HREF=%s/FFile?From=%s&Bucket=%s>Refresh</A>",
				m_pszURLString, szFrom, szBucket);



	// Show the table headings and the refresh option.

	if (pFile)
	{
		// Write out the table headings
		printTableStart( "FFile Structure", 4, 100);

		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
		fnPrintf( m_pHRequest, "%s, ", pszTemp1);
		fnPrintf( m_pHRequest, "%s\n", pszTemp);
		printColumnHeadingClose();
		printTableRowEnd();

		// Write out the table headings.
		printTableRowStart();
		printColumnHeading( "Byte Offset (hex)");
		printColumnHeading( "Field Name");
		printColumnHeading( "Field Type");
		printColumnHeading( "Value");
		printTableRowEnd();

		write_data( (pFile ? &localFFile: NULL), (void *)pFile, &DataStruct);

	}
	else
	{
		// Write out an error page...
		fnPrintf( m_pHRequest, 
			"<P>Unable to find the FFile structure that you requested."
			"  This is probably because the state of the cache changed between "
			"the time that you displayed the previous page and the time that you "
			"clicked on the link that brought you here.\n"
			"<P>Click on your browser's \"Back\" button, then click \"Reload\" "
			"and then try the link again.\n");
	}


	fnPrintf( m_pHRequest,  "</body></html>\n");

	fnEmit();

Exit:

	if (bFlmLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bFlmLocked = FALSE;
	}

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	if (pszTemp1)
	{
		f_free( &pszTemp1);
	}

	return( rc);


}

/****************************************************************************
 Desc:	This procedure generates the HTML page to display
			the contents of the FFile structure
 ****************************************************************************/
void F_FFilePage::write_data(
	FFILE *			pFile,
	void *			pvFFileAddress,
	DATASTRUCT *	pDataStruct)
{
	char			szFormattedTime[13];
	char			szTemp[100];
	char			szAddress[20];
	char			szFFileAddress[20];
	FLMBOOL		bHighlight = FALSE;


	F_UNREFERENCED_PARM( fnPrintf);

	if (pFile == NULL)
	{
		flmAssert(0);
		return;
	}
	else
	{

		printAddress( pvFFileAddress, szFFileAddress);

		// pNext
		if ( pFile->pNext)
		{
			f_sprintf( (char *)szTemp,
						"%s/FFile?From=FFile?Link=pNext?Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress, (unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pNext", 
				"FFILE *",
				(void *)pFile,
				(void *)&pFile->pNext,
				(void *)pFile->pNext,
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		
		


		// pPrev - previous file in hash bucket.
		if (pFile->pPrev)
		{
			f_sprintf( (char *)szTemp,
						"%s/FFile?From=FFile?Link=pPrev?Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress, (unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pPrev", 
				"FFILE *",
				(void *)pFile,
				(void *)&pFile->pPrev,
				(void *)pFile->pPrev,
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		


		// uiZeroUseCountTime - Time Use Count went to zero
		FormatTime(pFile->uiZeroUseCountTime, szFormattedTime);
		printHTMLString(
				"uiZeroUseCountTime", 
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiZeroUseCountTime,
				(char *)szFormattedTime,
				(bHighlight = ~bHighlight));


		// uiInternalUseCount - Internal Use Count
		printHTMLUint(
				"uiInternalUseCount", 
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiInternalUseCount,
				pFile->uiInternalUseCount,
				(bHighlight = ~bHighlight));



		// uiUseCount - Current Use Count
		printHTMLUint(
				"uiUseCount",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiUseCount,
				pFile->uiUseCount,
				(bHighlight = ~bHighlight));





		// pFirstDb
		if (pFile->pFirstDb)
		{
			char		szFDBAddr[20];

			printAddress( pFile->pFirstDb, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			f_sprintf( (char *)szTemp,
						"%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket, szFDBAddr);
		}

		printHTMLLink(
				"pFirstDb", 
				"FDB *",
				(void *)pFile,
				(void *)&pFile->pFirstDb,
				(void *)pFile->pFirstDb,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// pszDbPath - Database File Name
		printHTMLString(
				"pszDbPath",
				"FLMBYTE *",
				(void *)pFile,
				(void *)&pFile->pszDbPath,
				(char *)(pFile->pszDbPath ? (char *)pFile->pszDbPath : "Null"),
				(bHighlight = ~bHighlight));



		// pszDataDir
		printHTMLString(
				"pszDataDir",
				"FLMBYTE *",
				(void *)pFile,
				(void *)&pFile->pszDataDir,
				(char *)(pFile->pszDataDir ? (char *)pFile->pszDataDir : "Null"),
				(bHighlight = ~bHighlight));


		

		// pNextNUFile - Next Not Used File
		if (pFile->pNextNUFile)
		{
			f_sprintf( (char *)szTemp,
						"%s/FFile?From=FFile?Link=pNextNUFile?Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress, (unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pNextNUFile", 
				"FFILE *",
				(void *)pFile,
				(void *)&pFile->pNextNUFile,
				(void *)pFile->pNextNUFile,
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		

		
		
		// pPrevNUFile - Previous Not Used File
		if (pFile->pPrevNUFile)
		{
			f_sprintf( (char *)szTemp,
						"%s/FFile?From=FFile?Link=pPrevNUFile?Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress, (unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pPrevNUFile", 
				"FFILE *",
				(void *)pFile,
				(void *)&pFile->pPrevNUFile,
				(void *)pFile->pPrevNUFile,
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		
		

		// pSCacheList - Shared Cache Blocks
		if (pFile->pSCacheList)
		{
			f_sprintf( (char *)szTemp,
					  "%s/SCacheBlock?"
					  "BlockAddress=%ld&File=%s&LowTransID=%ld&HighTransID=%ld",
					  m_pszURLString,
					  pDataStruct->SCacheBlkAddress,
					  szFFileAddress,
					  pDataStruct->SCacheLowTransID,
					  pDataStruct->SCacheHighTransID);
		}

		printHTMLLink(
				"pSCacheList", 
				"SCACHE *",
				(void *)pFile,
				(void *)&pFile->pSCacheList,
				(void *)pFile->pSCacheList,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// pPendingWriteList
		if (pFile->pPendingWriteList)
		{
			f_sprintf( (char *)szTemp,
					  "%s/SCacheBlock?"
					  "BlockAddress=%ld&File=%s&LowTransID=%ld&HighTransID=%ld",
					  m_pszURLString,
					  pDataStruct->PendingWriteBlkAddress,
					  szFFileAddress,
					  pDataStruct->PendingWriteLowTransID,
					  pDataStruct->PendingWriteHighTransID);
		}

		printHTMLLink(
				"pPendingWriteList", 
				"SCACHE *",
				(void *)pFile,
				(void *)&pFile->pPendingWriteList,
				(void *)pFile->pPendingWriteList,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// pLastDirtyBlk
		if (pFile->pLastDirtyBlk)
		{
			f_sprintf( (char *)szTemp,
					  "%s/SCacheBlock?"
					  "BlockAddress=%ld&File=%s&LowTransID=%ld&HighTransID=%ld",
					  m_pszURLString,
					  pDataStruct->LastDirtyBlkAddress,
					  szFFileAddress,
					  pDataStruct->LastDirtyLowTransID,
					  pDataStruct->LastDirtyHighTransID);
		}

		printHTMLLink(
				"pLastDirtyBlk", 
				"SCACHE *",
				(void *)pFile,
				(void *)&pFile->pLastDirtyBlk,
				(void *)pFile->pLastDirtyBlk,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// uiDirtyCacheCount
		printHTMLUint(
				"uiDirtyCacheCount",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiDirtyCacheCount,
				pFile->uiDirtyCacheCount,
				(bHighlight = ~bHighlight));


		// uiLogCacheCount
		printHTMLUint(
				"uiLogCacheCount",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiLogCacheCount,
				pFile->uiLogCacheCount,
				(bHighlight = ~bHighlight));


		// pFirstRecord - First Record Cache Block
		// **Do we need to rework this by passing in the Container, Drn, pFile & LowTransId? ** //
		if (pFile->pFirstRecord)
		{
			f_sprintf( (char *)szTemp,
					  "%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
					  m_pszURLString,
					  pDataStruct->FirstRecordContainer,
					  pDataStruct->FirstRecordDrn,
					  szFFileAddress,
					  pDataStruct->FirstRecordLowTransId);
		}

		printHTMLLink(
				"pFirstRecord", 
				"RCACHE_p",
				(void *)pFile,
				(void *)&pFile->pFirstRecord,
				(void *)pFile->pFirstRecord,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// pLastRecord - Last Record Cache Block
		if (pFile->pLastRecord)
		{
			f_sprintf( (char *)szTemp,
					  "%s/RCache?Container=%lu?DRN=%lu?File=%s?Version=%lu",
					  m_pszURLString,
					  pDataStruct->LastRecordContainer,
					  pDataStruct->LastRecordDrn,
					  szFFileAddress,
					  pDataStruct->LastRecordLowTransId);
		}
			
		printHTMLLink(
				"pLastRecord", 
				"RCACHE_p",
				(void *)pFile,
				(void *)&pFile->pLastRecord,
				(void *)pFile->pLastRecord,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// ppBlocksDone - List of blocks to be written to the Rollback
		if (pFile->ppBlocksDone)
		{
			f_sprintf( (char *)szTemp,
					  "%s/SCache?From=FFile?Link=ppBlocksDone?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"ppBlocksDone", 
				"SCACHE **",
				(void *)pFile,
				(void *)&pFile->ppBlocksDone,
				(void *)pFile->ppBlocksDone,
				(char *)szTemp,
				(bHighlight = ~bHighlight));

		
		
		// uiBlocksDoneArraySize
		printHTMLUint(
				"uiBlocksDoneArraySize",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiBlocksDoneArraySize,
				pFile->uiBlocksDoneArraySize,
				(bHighlight = ~bHighlight));




		// uiBlocksDone - Number of Blocks in Blocks Done Array
		printHTMLUint(
				"uiBlocksDone",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiBlocksDone,
				pFile->uiBlocksDone,
				(bHighlight = ~bHighlight));


		
		
		// pTransLogList - Shared Cache Log List
		if (pFile->pTransLogList != NULL)
		{
			f_sprintf( (char *)szTemp,
					  "%s/SCache?From=FFile?Link=pTransLogList?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);
		}
		
		
		printHTMLLink(
				"pTransLogList", 
				"SCACHE *",
				(void *)pFile,
				(void *)&pFile->pTransLogList,
				(void *)pFile->pTransLogList,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// pOpenNotifies - Open Notifies Threads
		if (pFile->pOpenNotifies != NULL)
		{
			f_sprintf( (char *)szTemp,
					  "%s/FNOTIFY?From=FFile?Link=pOpenNotifies?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pOpenNotifies", 
				"FNOTIFY *",
				(void *)pFile,
				(void *)&pFile->pOpenNotifies,
				(void *)pFile->pOpenNotifies,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// pCloseNotifies
		if (pFile->pCloseNotifies != NULL)
		{
			f_sprintf( (char *)szTemp,
					  "%s/FNOTIFY?From=FFile?Link=pCloseNotifies?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pCloseNotifies", 
				"FNOTIFY *",
				(void *)pFile,
				(void *)&pFile->pCloseNotifies,
				(void *)pFile->pCloseNotifies,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// pDictList - Dictionaries List
		if (pFile->pDictList != NULL)
		{
			f_sprintf( (char *)szTemp,
					  "%s/FDICT?From=FFile?Link=pDictList?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pDictList", 
				"FDICT *",
				(void *)pFile,
				(void *)&pFile->pDictList,
				(void *)pFile->pDictList,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// krefPool - Kref pool
		printAddress( &pFile->krefPool, szAddress);
		printHTMLString(
				"krefPool", 
				"POOL",
				(void *)pFile,
				(void *)&pFile->krefPool,
				(char *)szAddress,
				(bHighlight = ~bHighlight));


		// FileHdr - File Header
		f_sprintf( (char *)szTemp,
				  "%s/FILE_HDR?From=FFile?Link=FileHdr?Address=%s?Bucket=%lu",
				  m_pszURLString,
				  szFFileAddress, (unsigned long)pFile->uiBucket);

		printHTMLLink(
				"FileHdr", 
				"FILE_HDR",
				(void *)pFile,
				(void *)&pFile->FileHdr,
				(void *)&pFile->FileHdr,
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// uiMaxFileSize - Maximum File Size
		printHTMLUint(
				"uiMaxFileSize",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiMaxFileSize,
				pFile->uiMaxFileSize,
				(bHighlight = ~bHighlight));



		// uiFileExtendSize - File Extend Size
		printHTMLUint(
				"uiFileExtendSize",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiFileExtendSize,
				pFile->uiFileExtendSize,
				(bHighlight = ~bHighlight));

		

		// uiUpdateTransID - Update Transaction Id
		printHTMLUint(
				"uiUpdateTransID",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiUpdateTransID,
				pFile->uiUpdateTransID,
				(bHighlight = ~bHighlight));

		
		
		// pRfl - Roll Forward Log Object
		if (pFile->pRfl)
		{
			f_sprintf( (char *)szTemp,
					  "%s/Rfl?From=FFile?Link=pRfl?Address=%s?Bucket=%lu",
					  m_pszURLString,
					  szFFileAddress, (unsigned long)pFile->uiBucket);

		}
		
		printHTMLLink(
				"pRfl", 
				"F_Rfl *",
				(void *)pFile,
				(void *)&pFile->pRfl,
				(void *)pFile->pRfl,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// ucLastCommittedLogHdr - Last Committed Log Header
		f_sprintf( (char *)szTemp,
				  "%s/LogHdr?From=FFile?"
				  "Link=ucLastCommittedLogHdr?"
				  "Address=%s?Bucket=%ld",
				  m_pszURLString,
				  szFFileAddress, pFile->uiBucket);

		printHTMLLink(
				"ucLastCommittedLogHdr", 
				"FLMBYTE",
				(void *)pFile,
				(void *)&pFile->ucLastCommittedLogHdr[0],
				(void *)&pFile->ucLastCommittedLogHdr[0],
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// ucCheckpointLogHdr - Checkpoint Log Header
		f_sprintf( (char *)szTemp,
				  "%s/LogHdr?From=FFile?"
				  "Link=ucCheckpointLogHdr?"
				  "Address=%s?Bucket=%ld",
				  m_pszURLString,
				  szFFileAddress, pFile->uiBucket);
		
		printHTMLLink(
				"ucCheckpointLogHdr", 
				"FLMBYTE",
				(void *)pFile,
				(void *)&pFile->ucCheckpointLogHdr[0],
				(void *)&pFile->ucCheckpointLogHdr[0],
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		
		
		

		// ucUncommittedLogHdr - Uncommitted Log Header
		f_sprintf( (char *)szTemp,
				  "%s/LogHdr?From=FFile?Link=ucUncommittedLogHdr?Address=%s?Bucket=%lu",
				  m_pszURLString,
				  szFFileAddress, (unsigned long)pFile->uiBucket);
		
		printHTMLLink(
				"ucUncommittedLogHdr", 
				"FLMBYTE",
				(void *)pFile,
				(void *)&pFile->ucUncommittedLogHdr[0],
				(void *)&pFile->ucUncommittedLogHdr[0],
				(char *)szTemp,
				(bHighlight = ~bHighlight));


		// pBufferMgr
		f_sprintf( (char *)szTemp,
				  "%s/F_IOBufferMgr?From=FFile?"
				  "Link=pBufferMgr?"
				  "Address=%s?Bucket=%lu",
				  m_pszURLString,
				  szFFileAddress,
				  (unsigned long)pFile->uiBucket);

		printHTMLLink(
				"pBufferMgr", 
				"F_IOBufferMgr *",
				(void *)pFile,
				(void *)&pFile->pBufferMgr,
				(void *)pFile->pBufferMgr,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// pCurrLogBuffer
		f_sprintf( (char *)szTemp,
				  "%s/F_IOBuffer?From=FFile?"
				  "Link=pCurrLogBuffer?"
				  "Address=%s?Bucket=%lu",
				  m_pszURLString,
				  szFFileAddress,
				  (unsigned long)pFile->uiBucket);

		printHTMLLink(
				"pCurrLogBuffer", 
				"F_IOBuffer *",
				(void *)pFile,
				(void *)&pFile->pCurrLogBuffer,
				(void *)pFile->pCurrLogBuffer,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// uiCurrLogWriteOffset
		printHTMLUint(
				"uiCurrLogWriteOffset",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiCurrLogWriteOffset,
				pFile->uiCurrLogWriteOffset,
				(bHighlight = ~bHighlight));





		// uiCurrLogBlkAddr
		printHTMLUint(
				"uiCurrLogBlkAddr",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiCurrLogBlkAddr,
				pFile->uiCurrLogBlkAddr,
				(bHighlight = ~bHighlight));

				
				

		// pFileLockObj - File Locking Object
		if (pFile->pFileLockObj)
		{
			f_sprintf( (char *)szTemp,
						"%s/ServerLockObject?From=FFile?"
						"Link=pFileLockObj?"
						"Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pFileLockObj", 
				"ServerLockObject_p",
				(void *)pFile,
				(void *)&pFile->pFileLockObj,
				(void *)pFile->pFileLockObj,
				(char *)szTemp,
				(bHighlight = ~bHighlight));
		


		// pWriteLockObj
		if (pFile->pWriteLockObj)
		{
			f_sprintf( (char *)szTemp,
						"%s/ServerLockObject?From=FFile?"
						"Link=pWriteLockObj?"
						"Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}
	
		printHTMLLink(
				"pWriteLockObj", 
				"ServerLockObject_p",
				(void *)pFile,
				(void *)&pFile->pWriteLockObj,
				(void *)pFile->pWriteLockObj,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// pLockFileHdl - File Lock Handle (3.x Db)
		if (pFile->pLockFileHdl)
		{
			f_sprintf( (char *)szTemp,
						"%s/F_FileHdl?From=FFile?"
						"Link=pLockFileHdl?"
						"Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pLockFileHdl", 
				"F_FileHdl_p",
				(void *)pFile,
				(void *)&pFile->pLockFileHdl,
				(void *)pFile->pLockFileHdl,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// pLockNotifies - Notifies List
		if (pFile->pLockNotifies)
		{
			f_sprintf( (char *)szTemp,
						"%s/FNOTIFY?From=FFile?"
						"Link=pLockNotifies?"
						"Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}

		printHTMLLink(
				"pLockNotifies", 
				"FNOTIFY *",
				(void *)pFile,
				(void *)&pFile->pLockNotifies,
				(void *)pFile->pLockNotifies,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// bBeingLocked - File   being locked
		printHTMLString(
				"bBeingLocked",
				"FLMBOOL",
				(void *)pFile,
				(void *)&pFile->bBeingLocked,
				(char *)(pFile->bBeingLocked ? "Yes" : "No"),
				(bHighlight = ~bHighlight));




		// pFirstReadTrans - First Read Transaction
		if (pFile->pFirstReadTrans)
		{
			char		szFDBAddr[20];

			printAddress( pFile->pFirstReadTrans, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			f_sprintf( (char *)szTemp,
						"%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
						m_pszURLString,
						szFFileAddress, 
						(unsigned long)pFile->uiBucket, szFDBAddr);
		}
		
		printHTMLLink(
				"pFirstReadTrans", 
				"FDB *",
				(void *)pFile,
				(void *)&pFile->pFirstReadTrans,
				(void *)pFile->pFirstReadTrans,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// pLastReadTrans - Last Read Transaction
		if (pFile->pLastReadTrans)
		{
			char		szFDBAddr[20];

			printAddress( pFile->pLastReadTrans, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			f_sprintf(
				(char *)szTemp, "%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szFFileAddress,
				(unsigned long)pFile->uiBucket, szFDBAddr);
		}

		printHTMLLink(
				"pLastReadTrans", 
				"FDB *",
				(void *)pFile,
				(void *)&pFile->pLastReadTrans,
				(void *)pFile->pLastReadTrans,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// pFirstKilledTrans - First Killed Transaction
		if (pFile->pFirstKilledTrans)
		{
			char		szFDBAddr[20];

			printAddress( pFile->pFirstKilledTrans, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			f_sprintf(
				(char *)szTemp, "%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szFFileAddress,
				(unsigned long)pFile->uiBucket, szFDBAddr);
		}

		printHTMLLink(
				"pFirstKilledTrans", 
				"FDB *",
				(void *)pFile,
				(void *)&pFile->pFirstKilledTrans,
				(void *)pFile->pFirstKilledTrans,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// uiFirstLogBlkAddress
		printHTMLUint(
				"uiFirstLogBlkAddress",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiFirstLogBlkAddress,
				pFile->uiFirstLogBlkAddress,
				(bHighlight = ~bHighlight));


		// uiFirstLogCPBlkAddress - First Log Checkpoint Block Address
		printHTMLUint(
				"uiFirstLogCPBlkAddress",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiFirstLogCPBlkAddress,
				pFile->uiFirstLogCPBlkAddress,
				(bHighlight = ~bHighlight));




		// uiLastCheckpointTime - Last Checkpoint Time
		FormatTime( pFile->uiLastCheckpointTime, szFormattedTime);
		printHTMLString(
				"uiLastCheckpointTime",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiLastCheckpointTime,
				(char *)szFormattedTime,
				(bHighlight = ~bHighlight));





		// pCPThrd
		if (pFile->pCPThrd)
		{
			f_sprintf( (char *)szTemp,
						"%s/F_Thread?From=FFile?"
						"Link=pCPThrd?"
						"Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pCPThrd", 
				"F_Thread *",
				(void *)pFile,
				(void *)&pFile->pCPThrd,
				(void *)pFile->pCPThrd,
				(char *)szTemp,
				(bHighlight = ~bHighlight));




		// pCPInfo - Checkpoint Info Buffer
		if (pFile->pCPInfo)
		{
			f_sprintf( (char *)szTemp,
						"%s/CP_INFO?From=FFile?Link=pCPInfo?Address=%s?Bucket=%lu",
						m_pszURLString,
						szFFileAddress,
						(unsigned long)pFile->uiBucket);
		}
		
		printHTMLLink(
				"pCPInfo", 
				"CP_INFO_p",
				(void *)pFile,
				(void *)&pFile->pCPInfo,
				(void *)pFile->pCPInfo,
				(char *)szTemp,
				(bHighlight = ~bHighlight));



		// CheckpointRc - Last Checkpoint Return Code
		printHTMLUint(
				"CheckpointRc",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->CheckpointRc,
				pFile->CheckpointRc,
				(bHighlight = ~bHighlight));




		// uiBucket - Hash Table Bucket
		printHTMLUint(
				"uiBucket",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiBucket,
				pFile->uiBucket,
				(bHighlight = ~bHighlight));



		// uiFlags - Flags
		if (pFile->uiFlags)
		{
			FLMBOOL		bTest = FALSE;
			char *		pTemp = (char *)szTemp;

			f_sprintf( (char *)szTemp, "%08X<br>", (unsigned)pFile->uiFlags);
			pTemp += 12;

			if (pFile->uiFlags & DBF_BEING_OPENED)
			{
				f_sprintf(pTemp, "Being Opened");
				pTemp += f_strlen("Being Opened");
				bTest=TRUE;
			}

			if (pFile->uiFlags & DBF_IN_NU_LIST)
			{
				if (bTest)
				{
					f_sprintf(pTemp, "<br>");
					pTemp += f_strlen("<br>");
				}

				f_sprintf(pTemp, "In Not Used List");
				pTemp += f_strlen("In Not Used List");
				bTest = TRUE;
			}

			if (pFile->uiFlags & DBF_BEING_CLOSED)
			{
				if (bTest)
				{
					f_sprintf(pTemp, "<br>");
					pTemp += f_strlen("<br>");
				}

				f_sprintf(pTemp, "Being Closed");
				pTemp += f_strlen("Being Closed");
			}

		}
		else
		{
			f_sprintf( (char *)szTemp, "%08X<br>Normal", (unsigned)pFile->uiFlags);
		}

		printHTMLString(
				"uiFlags",
				"FLMUINT",
				(void *)pFile,
				(void *)&pFile->uiFlags,
				(char *)szTemp,
				(bHighlight = ~bHighlight));

		

		
		
		// bBackupActive - Backup Active
		printHTMLString(
				"bBackupActive",
				"FLMBOOL",
				(void *)pFile,
				(void *)&pFile->bBackupActive,
				(char *)(pFile->bBackupActive ? "Yes" : "No"),
				(bHighlight = ~bHighlight));
		
		printTableEnd();

	}
}
