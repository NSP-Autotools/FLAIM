//-------------------------------------------------------------------------
// Desc:	Class for displaying an SCACHE structure in HTML on a web page.
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

FSTATIC void flmPrintCacheLine(
	HRequest *		pHRequest,
	const char *	pszHREF,
	const char *	pszName,
	void *			pBaseAddr,
	SCACHE **		ppSCache);


FSTATIC void flmBuildSCacheBlockString(
	char *			pszString,
	SCACHE *			pScache);

/****************************************************************************
Desc:	Prints the web page for an SCACHE struct
****************************************************************************/
RCODE F_SCacheBlockPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBlkAddress = 0;
	FLMUINT			uiLowTransID = 0;
	FLMUINT			uiHighTransID = 0;
	FFILE *			pFile;
	FLMBOOL			bHighlight = FALSE;
	char *			pszTemp = NULL;
	char *			pszTemp1 = NULL;
	FLMUINT			uiLoop = 0;
	char				szOffsetTable[10][6];
	char				szAddressTable[4][20];
	SCACHE			LocalSCacheBlock;
	FLMUINT			uiPFileBucket = 0;
	char *			pszSCacheRequestString[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	char *			pszSCacheDataRequest = NULL;
	char *			pszSCacheAutoRequest = NULL; 
	char *			pszSCacheUseListRequest = NULL;
	char *			pszSCacheNotifyListRequest = NULL;
	char *			pszFFileRequest = NULL;
	char *			pszFlagNames = NULL;

	if( RC_BAD( rc = f_alloc( 200, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 200, &pszTemp1)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// Allocate memory for all those string pointers we declared above...
	for (uiLoop = 0; uiLoop < 8; uiLoop++)
	{
		if( RC_BAD( rc = f_alloc( 150, &pszSCacheRequestString[ uiLoop])))
		{
			goto Exit;
		}
	}
		
	if( RC_BAD( rc = f_alloc( 150, &pszSCacheDataRequest)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 150, &pszSCacheAutoRequest)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 150, &pszSCacheUseListRequest)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 150, &pszSCacheNotifyListRequest)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 100, &pszFFileRequest)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 100, &pszFlagNames)))
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	rc = locateSCacheBlock( uiNumParams, ppszParams, &LocalSCacheBlock,
									&uiBlkAddress, &uiLowTransID,	&uiHighTransID,
									&pFile);

	if (RC_OK(rc) && LocalSCacheBlock.pFile)
	{
		uiPFileBucket = LocalSCacheBlock.pFile->uiBucket;
	}

	
	if (RC_OK( rc))
	{
		// Build the proper strings to request various other SCache blocks
		flmBuildSCacheBlockString( pszSCacheRequestString[0], LocalSCacheBlock.pPrevInFile);
		flmBuildSCacheBlockString( pszSCacheRequestString[1], LocalSCacheBlock.pNextInFile);
		flmBuildSCacheBlockString( pszSCacheRequestString[2], LocalSCacheBlock.pPrevInGlobalList);
		flmBuildSCacheBlockString( pszSCacheRequestString[3], LocalSCacheBlock.pNextInGlobalList);
		flmBuildSCacheBlockString( pszSCacheRequestString[4], LocalSCacheBlock.pPrevInHashBucket);
		flmBuildSCacheBlockString( pszSCacheRequestString[5], LocalSCacheBlock.pNextInHashBucket);
		flmBuildSCacheBlockString( pszSCacheRequestString[6], LocalSCacheBlock.pPrevInVersionList);
		flmBuildSCacheBlockString( pszSCacheRequestString[7], LocalSCacheBlock.pNextInVersionList);

		// Build the proper string to request the current Page
		flmBuildSCacheBlockString( pszSCacheAutoRequest, &LocalSCacheBlock);
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	if (RC_BAD( rc))
	{
		if (rc == FERR_NOT_FOUND)
		{
			
			// The block wasn't there, print an error message and exit
			notFoundErr();
			rc = FERR_OK;
		}
		else if (rc == FERR_MEM)
		{
			// Parameters were too long to store in the space provided.
			// Probably means that the URL was malformed...
			malformedUrlErr();
			rc = FERR_OK;
		}
		goto Exit;
	}
	
	//Build the proper string to request this block's data...
	printAddress( pFile, szAddressTable[0]);
	f_sprintf( (char *)pszSCacheDataRequest,
		"%s/SCacheData?BlockAddress=%lu&File=%s&LowTransID=%lu&HighTransID=%lu",
		m_pszURLString, LocalSCacheBlock.uiBlkAddress, szAddressTable[0],
		uiLowTransID, uiHighTransID);
	
#ifdef FLM_DEBUG
	//Build the proper string to request this block's use list
	if( LocalSCacheBlock.pUseList)
	{
		f_sprintf( (char *)pszSCacheUseListRequest,
			"%s/SCacheUseList?BlockAddress=%lu&File=%s&LowTransID=%lu&HighTransID=%lu",
			m_pszURLString, LocalSCacheBlock.uiBlkAddress, szAddressTable[0],
			uiLowTransID, uiHighTransID);
	}
	else
	{
		pszSCacheUseListRequest[0] = '\0';
	}
#endif

	//Build the proper string to request the notify list data...
	if (LocalSCacheBlock.pNotifyList)
	{
		f_sprintf( (char *)pszSCacheNotifyListRequest,
			"%s/SCacheNotifyList?BlockAddress=%lu&File=%s&LowTransID=%lu&HighTransID=%lu",
			m_pszURLString, LocalSCacheBlock.uiBlkAddress, szAddressTable[0],
			uiLowTransID, uiHighTransID);
	}
	else
	{
		pszSCacheNotifyListRequest[0] = '\0';
	}

	//Build the proper string to request the FFile
	printAddress( LocalSCacheBlock.pFile, szAddressTable[0]);
	f_sprintf( (char *)pszFFileRequest, "%s/FFile?From=SCacheBlock&Bucket=%lu&Address=%s",
				 m_pszURLString, uiPFileBucket, szAddressTable[0]);
					

	// Build a string with the names of all the flags that have been set...
	pszFlagNames[0]='\0';
	if (LocalSCacheBlock.ui16Flags & CA_DIRTY)
	{
		f_strcat( pszFlagNames, "<BR> CA_DIRTY");
	}
	if (LocalSCacheBlock.ui16Flags & CA_READ_PENDING)
	{
		f_strcat( pszFlagNames, "<BR> CA_READ_PENDING");
	}
	if (LocalSCacheBlock.ui16Flags & CA_WRITE_TO_LOG)
	{
		f_strcat( pszFlagNames, "<BR> CA_WRITE_TO_LOG");
	}
	if (LocalSCacheBlock.ui16Flags & CA_LOG_FOR_CP)
	{
		f_strcat( pszFlagNames, "<BR> CA_LOG_FOR_CP");
	}
	if (LocalSCacheBlock.ui16Flags & CA_WAS_DIRTY)
	{
		f_strcat( pszFlagNames, "<BR> CA_WAS_DIRTY");
	}
	if (LocalSCacheBlock.ui16Flags & CA_WRITE_PENDING)
	{
		f_strcat( pszFlagNames, "<BR> CA_WRITE_PENDING");
	}
	if (LocalSCacheBlock.ui16Flags & CA_IN_WRITE_PENDING_LIST)
	{
		f_strcat( pszFlagNames, "<BR> CA_IN_WRITE_PENDING_LIST");
	}


	// OK - Start outputting HTML...
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE "<html>\n");
		
	// Determine if we are being requested to refresh this page or  not.
	if (DetectParameter( uiNumParams, ppszParams, "Refresh"))
	{
		// Send back the page with a refresh command in the header
		fnPrintf( m_pHRequest, 
			"<HEAD>\n"
			"<META http-equiv=\"refresh\" content=\"5; url=\"%s\">"
			"<TITLE>SCache Block</TITLE>\n", pszSCacheAutoRequest);
		printStyle();
		popupFrame();  //Spits out a Javascript function that will open a new window..
		fnPrintf( m_pHRequest, "</HEAD>\n<body>\n");
		
		f_sprintf( (char*)pszTemp,
					"<A HREF=\"%s\">Stop Auto-refresh</A>", pszSCacheAutoRequest);
	}
	else
	{
		// Send back a page without the refresh command
		
		fnPrintf( m_pHRequest, "<HEAD>\n");
		printStyle();
		popupFrame();  //Spits out a Javascript function that will open a new window..
		fnPrintf( m_pHRequest, "</HEAD>\n<body>\n");
		
		f_sprintf( (char *)pszTemp,
					"<A HREF=\"%s?Refresh\">Start Auto-refresh (5 sec.)</A>", pszSCacheAutoRequest);
	}

	// Write out the table headings
	printTableStart( "SCache Block Structure", 4, 100);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=\"%s\">Refresh</A>, %s\n",
				 pszSCacheAutoRequest, pszTemp);
	printColumnHeadingClose();
	printTableRowEnd();

	// Write out the table headings.
	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();

	// Print the two rows for pPrevInFile and pNextInFile
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[0], "pPrevInFile", &LocalSCacheBlock, &LocalSCacheBlock.pPrevInFile);
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[1], "pNextInFile", &LocalSCacheBlock, &LocalSCacheBlock.pNextInFile);


	// Format the strings that are displayed in the Offset and Address
	// columns of the table
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.pucBlk, szOffsetTable[0]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.pFile, szOffsetTable[1]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.uiBlkAddress, szOffsetTable[2]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.pNotifyList, szOffsetTable[3]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.uiHighTransID, szOffsetTable[4]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.uiUseCount, szOffsetTable[5]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.ui16Flags, szOffsetTable[6]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.ui16BlkSize, szOffsetTable[7]);
#ifdef FLM_DEBUG
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.uiChecksum, szOffsetTable[8]);
	printOffset( &LocalSCacheBlock, &LocalSCacheBlock.pUseList, szOffsetTable[9]);
#endif

	printAddress( LocalSCacheBlock.pucBlk, szAddressTable[0]);
	printAddress( LocalSCacheBlock.pFile, szAddressTable[1]);
	printAddress( LocalSCacheBlock.pNotifyList, szAddressTable[2]);
#ifdef FLM_DEBUG
	printAddress( LocalSCacheBlock.pUseList, szAddressTable[3]);
#endif


	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td><A HREF=\"javascript:openPopup('%s')\">pucBlk</A></td>\n"
				"<td>FLMBYTE *</td>\n<td><A HREF=\"javascript:openPopup('%s')\">%s</A></td>\n",
				szOffsetTable[0],	pszSCacheDataRequest, pszSCacheDataRequest, szAddressTable[0] );
	printTableRowEnd();

	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td><A href=%s>pFile</A></td>\n"
				"<td>FFILE *</td>\n<td><A HREF=%s>%s</a></td>\n",
				szOffsetTable[1], pszFFileRequest, pszFFileRequest, szAddressTable[1]);
	printTableRowEnd();

	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>uiBlkAddress</td>\n<td>FLMUINT</td>\n"
				"<td>0x%lX</td>\n", szOffsetTable[2], LocalSCacheBlock.uiBlkAddress);
	printTableRowEnd();

	//Print the rows for the remaining SCache * fields
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[2], "pPrevInGlobalList", &LocalSCacheBlock, &LocalSCacheBlock.pPrevInGlobalList);		
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[3], "pNextInGlobalList", &LocalSCacheBlock, &LocalSCacheBlock.pNextInGlobalList);
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[4], "pPrevInHashBucket", &LocalSCacheBlock, &LocalSCacheBlock.pPrevInHashBucket);		
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[5], "pNextInHashBucket", &LocalSCacheBlock, &LocalSCacheBlock.pNextInHashBucket);
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[6], "pPrevInVersionList", &LocalSCacheBlock, &LocalSCacheBlock.pPrevInVersionList);		
	printTableRowStart( bHighlight = ~bHighlight);
	flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[7], "pNextInVersionList", &LocalSCacheBlock, &LocalSCacheBlock.pNextInVersionList);

	//Notify list line
	printTableRowStart( bHighlight = ~bHighlight);
	if (LocalSCacheBlock.pNotifyList)
	{
		fnPrintf( m_pHRequest,
			TD_s
			" <td> <A HREF=\"javascript:openPopup('%s')\"> pNotifyList </A> </td>	<td>FNOTIFY *</td> "
			"<td> <A HREF=\"javascript:openPopup('%s')\"> %s </A> </td>",
			szOffsetTable[3], pszSCacheNotifyListRequest,
			pszSCacheNotifyListRequest, szAddressTable[2]);
	}
	else
	{
		fnPrintf( m_pHRequest,
			TD_s " <td> pNotifyList </td>	<td>FNOTIFY *</td> "
			"<td> 0x0 </td>", szOffsetTable[3]);
	}
	printTableRowEnd();


	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>uiHighTransID</td>\n"
				"<td>FLMUINT</td>\n" TD_8x, szOffsetTable[4],
				LocalSCacheBlock.uiHighTransID);
	printTableRowEnd();

	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>uiUseCount</td>\n<td>FLMUINT</td>\n"
				 TD_lu,  szOffsetTable[5], LocalSCacheBlock.uiUseCount);
	printTableRowEnd();

	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>ui16Flags</td>\n<td>FLMUINT16</td>\n"
					"<td>0x%04X %s</td>\n", szOffsetTable[6],
					LocalSCacheBlock.ui16Flags, pszFlagNames);
	printTableRowEnd();

	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>ui16BlkSize</td>\n<td>FLMUINT16</td>\n" TD_i,
					szOffsetTable[7], LocalSCacheBlock.ui16BlkSize);
	printTableRowEnd();

#ifdef FLM_DEBUG
	printTableRowStart( bHighlight = ~bHighlight);
	fnPrintf( m_pHRequest, TD_s "<td>uiChecksum</td>\n"
					"<td>FLMUINT</td>\n" TD_8x,
					szOffsetTable[8], LocalSCacheBlock.uiChecksum);
	printTableRowEnd();
#endif


#ifdef FLM_DEBUG
		//Last line - the use list...
		printTableRowStart( bHighlight = ~bHighlight);
		if (LocalSCacheBlock.pUseList)
		{
			fnPrintf( m_pHRequest,
				TD_s " <td> <A href=\"javascript:openPopup('%s')> pUseList </A> </td>	<td> SCACHE_USE_p </td>		<td> <A href=\"javascript:openPopup('%s')> %s </A></td>",
				szOffsetTable[9], pszSCacheUseListRequest, pszSCacheUseListRequest, szAddressTable[3]);
		}
		else
		{
			fnPrintf( m_pHRequest,
				TD_s " <td> pUseList </td>	<td> SCACHE_USE_p </td>	<td> 0x0 </td>",
				szOffsetTable[9]);
		}
		printTableRowEnd();

#endif

	fnPrintf( m_pHRequest, TABLE_END "</BODY></HTML>\n");
	fnEmit();

Exit:

	// Even though uiLoop2 is not in the same scope as uiLoop, VC6 still
	// complains if this is called uiLoop....
	for (FLMUINT uiLoop2 = 0; uiLoop2 < 8; uiLoop2++)
	{
		if (pszSCacheRequestString[uiLoop2])
		{
			f_free( &pszSCacheRequestString[uiLoop2]);
		}
	}

	if (pszSCacheDataRequest)
	{
		f_free( &pszSCacheDataRequest);
	}

	if (pszSCacheAutoRequest)
	{
		f_free( &pszSCacheAutoRequest);
	}

	if (pszSCacheUseListRequest)
	{
		f_free( &pszSCacheUseListRequest);
	}

	if (pszSCacheNotifyListRequest)
	{
		f_free( &pszSCacheNotifyListRequest);
	}
	
	if( pszFFileRequest)
	{
		f_free( &pszFFileRequest);
	}

	if( pszFlagNames)
	{
		f_free( &pszFlagNames);
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
Desc:	Prints the web page for an SCACHE use list

		This function is essentially unimplemented because I have yet to see
		an SCache page where the use_list value was non-null!!
****************************************************************************/
RCODE F_SCacheUseListPage::display( 
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	RCODE rc = FERR_OK;
	
	stdHdr();

	fnPrintf( m_pHRequest,
		HTML_DOCTYPE "\n<html>\n <body>\n"
		"Congratulations!  You've managed to find an SCache block with a valid "
		"use list!   Too bad we haven't implemented a page to dislay use " 
		"lists yet...\n </body> </html>");
	fnEmit();
	return( rc);
}


/****************************************************************************
Desc:	Prints the web page for an SCACHE notify list

  		This function is essentially unimplemented because I have yet to see
		an SCache page where the notify list value was non-null!!
****************************************************************************/
RCODE F_SCacheNotifyListPage::display( 
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	RCODE rc = FERR_OK;
	
	stdHdr();

	fnPrintf( m_pHRequest,
		HTML_DOCTYPE "\n<html>\n <body>\n"
		"Congratulations!  You've managed to find an SCache block with a valid "
		"notify list!   Too bad we haven't implemented a page to dislay use " 
		"lists yet...\n </body> </html>");
	fnEmit();
	return( rc);
}



/****************************************************************************
Desc:	Prints the web page showing the binary data in an SCache block
****************************************************************************/
RCODE F_SCacheDataPage::display( 
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;

	FLMUINT		uiBlkAddress = 0;
	FLMUINT		uiLowTransID = 0;
	FLMUINT		uiHighTransID = 0;
	FFILE *		pFile = NULL;
	FLMBOOL		bFlaimLocked = FALSE;
	SCACHE		LocalSCacheBlock;
	char *		pucData = NULL;
	char *		pucDataLine;
	char			szData[97];
	char			szOneChar[7];
	FLMUINT		uiCurrentOffset = 0;
	FLMUINT		uiLoop = 0;
	
	f_mutexLock( gv_FlmSysData.hShareMutex);
	bFlaimLocked = TRUE;
	rc = locateSCacheBlock( uiNumParams, ppszParams, &LocalSCacheBlock,
									&uiBlkAddress, &uiLowTransID,	&uiHighTransID,
									&pFile);
	if (RC_BAD( rc))
	{
		if(rc == FERR_NOT_FOUND)
		{
			notFoundErr();
			rc = FERR_OK;
		}
		goto Exit;
	}
	else
	{
		// Store the data in a local variable...
		if( RC_BAD( rc = f_alloc( 
			LocalSCacheBlock.ui16BlkSize, &pucData)))
		{
			goto Exit;
		}

		f_memcpy( pucData, LocalSCacheBlock.pucBlk, LocalSCacheBlock.ui16BlkSize);
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bFlaimLocked = FALSE;

	// Start the HTML...
	
	stdHdr();
	fnPrintf( m_pHRequest, HTML_DOCTYPE 
				"<HTML> <BODY>\n<font face=arial><PRE>\n");

	while (uiCurrentOffset < LocalSCacheBlock.ui16BlkSize)
	{
		szData[0] = '\0';
		pucDataLine =  pucData + uiCurrentOffset;
		fnPrintf( m_pHRequest, "<font color=blue>0x%04X</font>    "
						"%02X %02X %02X %02X  %02X %02X %02X %02X  %02X %02X %02X %02X  %02X %02X %02X %02X    ",
						uiCurrentOffset,
						pucDataLine[ 0], pucDataLine[ 1], pucDataLine[ 2], pucDataLine[ 3],
						pucDataLine[ 4], pucDataLine[ 5], pucDataLine[ 6], pucDataLine[ 7],
						pucDataLine[ 8], pucDataLine[ 9], pucDataLine[10], pucDataLine[11],
						pucDataLine[12], pucDataLine[13], pucDataLine[14], pucDataLine[15]);
	
		for (uiLoop = 0; uiLoop < 16; uiLoop++)
		{
			if (	(pucDataLine[uiLoop] >= 32) &&  // 32 is a space
					(pucDataLine[uiLoop] <= 126)  ) // 126 is a ~
			{
				f_sprintf( szOneChar, "&#%d;", pucDataLine[uiLoop]);
			}
			else
			{
				f_strcpy( szOneChar, "&#46;"); // 46 is a .
			}
			f_strcat(szData, szOneChar);

			// The reason for all the &#xxx; nonsence is because if we just put
			// the characters into a string, when the brower comes across a <
			// character, it will try to interpret what follows as an HTML
			// tag...
		}

		fnPrintf( m_pHRequest, "<font color=green>%s</font>\n", szData);
		
		uiCurrentOffset += 16;
	}

	fnPrintf( m_pHRequest, "</PRE></font>\n</BODY> </HTML>\n");
	fnEmit();

Exit:
	if (bFlaimLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (pucData)
	{
		f_free( &pucData);
	}

	return( rc);
}

/****************************************************************************
Desc:	Prints the web page for an SCACHEMGR struct
		(The URL for this page requires no parameters since there is only
		one SCACHE_MGR per copy of FLAIM.)
****************************************************************************/
RCODE F_SCacheMgrPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	SCACHE_MGR	LocalSCacheMgr;
	FLMBOOL		bAutoRefresh;
#define NUM_CACHE_REQ_STRINGS			4
	char	*		pszSCacheRequestString[ NUM_CACHE_REQ_STRINGS];
	char			szOffsetTable[12][6];
	char			szAddressTable[2][20];
	FLMBOOL		bHighlight = FALSE;
	char *		pszTemp = NULL;
	FLMUINT		uiLoop;

	// Note: The SCacheBlock requests need the following params:
	// "BlockAddress", "File", "LowTransID" and "HighTransID"
	// ex:  <A href="SCacheBlock?BlockAddress=100?File=5?LowTransID=30?HighTransID=100"> pMRUCache </A>
	
	if( RC_BAD( rc = f_alloc( 200, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	// First thing that we need to do is grab a local copy of gv_FlmSysData.SCacheMgr,
	// and of the data for the three SCache blocks that it has pointers to...
	for (uiLoop = 0; uiLoop < NUM_CACHE_REQ_STRINGS; uiLoop++)
	{
		if( RC_BAD( rc = f_alloc( 150,
									&pszSCacheRequestString[ uiLoop])))
		{
			printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
			goto Exit;
		}
	}
	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_memcpy (&LocalSCacheMgr, &gv_FlmSysData.SCacheMgr, sizeof (LocalSCacheMgr));
	flmBuildSCacheBlockString( pszSCacheRequestString[0], LocalSCacheMgr.pMRUCache);
	flmBuildSCacheBlockString( pszSCacheRequestString[1], LocalSCacheMgr.pLRUCache);
	flmBuildSCacheBlockString( pszSCacheRequestString[2], LocalSCacheMgr.pFirstFree);
	flmBuildSCacheBlockString( pszSCacheRequestString[3], LocalSCacheMgr.pLastFree);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	bAutoRefresh = DetectParameter( uiNumParams, ppszParams, "Refresh");

	// Now - are we being asked to display the usage stats?  Or is this a regular page...
	if (DetectParameter( uiNumParams, ppszParams, "Usage"))
	{
		// There's a function to handle display the usage info (because both
		// RCacheMgr and SCacheMgr have usage stats).
		writeUsage( &LocalSCacheMgr.Usage, bAutoRefresh,
						"/SCacheMgr?Usage",
						"Usage Statistics for the SCache");
	}
	else // This is a regular SCacheMgr page...
	{
		// Determine if we are being requested to refresh this page or  not.

		stdHdr();

		fnPrintf( m_pHRequest, HTML_DOCTYPE "<HTML>\n");

		if (bAutoRefresh)
		{
			// Send back the page with a refresh command in the header

			fnPrintf( m_pHRequest, 
				"<HEAD>"
				"<META http-equiv=\"refresh\" content=\"5; url=%s/SCacheMgr?Refresh\">"
				"<TITLE>gv_FlmSysData.SCacheMgr</TITLE>\n", m_pszURLString);

			printStyle();
			popupFrame();  //Spits out a Javascript function that will open a new window..
	
			fnPrintf( m_pHRequest, "\n</HEAD>\n<body>\n");


			f_sprintf( (char *)pszTemp,
							"<A HREF=%s/SCacheMgr>Stop Auto-refresh</A>", m_pszURLString);
		}
		else  // bAutoRefresh == FALSE
		{
			// Send back a page without the refresh command
			fnPrintf( m_pHRequest, 
				"<HEAD>"
				"<TITLE>gv_FlmSysData.SCacheMgr</TITLE>\n");

			printStyle();
			popupFrame();  //Spits out a Javascript function that will open a new window..
	
			fnPrintf( m_pHRequest, "\n</HEAD>\n<body>\n");

			f_sprintf( (char *)pszTemp,
						"<A HREF=%s/SCacheMgr?Refresh>Start Auto-refresh (5 sec.)</A>",
						m_pszURLString);
		}

		// Write out the table headings
		printTableStart( "SCache Manager Structure", 4);

		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
		fnPrintf( m_pHRequest, "<A HREF=%s/SCacheMgr>Refresh</A>, %s\n", m_pszURLString, pszTemp);
		printColumnHeadingClose();
		printTableRowEnd();

		// Write out the table headings.
		printTableRowStart();
		printColumnHeading( "Byte Offset (hex)");
		printColumnHeading( "Field Name");
		printColumnHeading( "Field Type");
		printColumnHeading( "Value");
		printTableRowEnd();
	
		//Now - we have three rows in the table that may or may not have hyperlinks in them.  
		printTableRowStart( bHighlight = ~bHighlight);
		flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[0], "pMRUCache", &LocalSCacheMgr, &LocalSCacheMgr.pMRUCache);
		printTableRowStart( bHighlight = ~bHighlight);
		flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[1], "pLRUCache", &LocalSCacheMgr, &LocalSCacheMgr.pLRUCache);
		printTableRowStart( bHighlight = ~bHighlight);
		flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[2], "pFirstFree", &LocalSCacheMgr, &LocalSCacheMgr.pFirstFree);
		printTableRowStart( bHighlight = ~bHighlight);
		flmPrintCacheLine(m_pHRequest, pszSCacheRequestString[3], "pLastFree", &LocalSCacheMgr, &LocalSCacheMgr.pLastFree);

		//Format the strings that are displayed in the Offset column on of the table
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.ppHashTbl, szOffsetTable[0]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.Usage, szOffsetTable[1]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.bAutoCalcMaxDirty, szOffsetTable[2]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiMaxDirtyCache, szOffsetTable[3]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiLowDirtyCache, szOffsetTable[4]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiTotalUses, szOffsetTable[5]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiBlocksUsed, szOffsetTable[6]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiPendingReads, szOffsetTable[7]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiIoWaits, szOffsetTable[8]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiHashTblSize, szOffsetTable[9]);
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.uiHashTblBits, szOffsetTable[10]);
#ifdef FLM_DEBUG
		printOffset(&LocalSCacheMgr, &LocalSCacheMgr.bDebug, szOffsetTable[11]);
#endif


		printAddress( LocalSCacheMgr.ppHashTbl, szAddressTable[0]);
		printAddress( &LocalSCacheMgr.Usage, szAddressTable[1]);

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s
				"<td><A HREF=\"%s/SCacheHashTable?Start=0\">ppHashTbl</A></td>\n"
				"<td>SCACHE **</td>\n"
				"<td><A href=\"%s/SCacheHashTbl\">%s</A></td>\n",
				szOffsetTable[0], m_pszURLString, m_pszURLString, szAddressTable[0]);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s
			"<td><A href=\"javascript:openPopup('%s/SCacheMgr?Usage')\">Usage</A></td>\n"
			"<td>FLM_CACHE_USAGE</td>\n"
			"<td><A href=\"javascript:openPopup('%s/SCacheMgr?Usage')\">%s</A></td>\n",
			szOffsetTable[1], m_pszURLString, m_pszURLString, szAddressTable[1]);
		printTableRowEnd();

		// uiFreeCount
		printHTMLUint(
			(char *)"uiFreeCount",
			(char *)"FLMUINT",
			(void *)&LocalSCacheMgr,
			(void *)&LocalSCacheMgr.uiFreeCount,
			LocalSCacheMgr.uiFreeCount,
			(bHighlight = ~bHighlight));

		// uiFreeBytes
		printHTMLUint(
			(char *)"uiFreeBytes",
			(char *)"FLMUINT",
			(void *)&LocalSCacheMgr,
			(void *)&LocalSCacheMgr.uiFreeBytes,
			LocalSCacheMgr.uiFreeBytes,
			(bHighlight = ~bHighlight));

		// uiReplaceableCount
		printHTMLUint(
			(char *)"uiReplaceableCount",
			(char *)"FLMUINT",
			(void *)&LocalSCacheMgr,
			(void *)&LocalSCacheMgr.uiReplaceableCount,
			LocalSCacheMgr.uiReplaceableCount,
			(bHighlight = ~bHighlight));

		// uiReplaceableBytes
		printHTMLUint(
			(char *)"uiReplaceableBytes",
			(char *)"FLMUINT",
			(void *)&LocalSCacheMgr,
			(void *)&LocalSCacheMgr.uiReplaceableBytes,
			LocalSCacheMgr.uiReplaceableBytes,
			(bHighlight = ~bHighlight));

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>bAutoCalcMaxDirty</td>\n"
						"<td>FLMBOOL</td>\n" TD_i, szOffsetTable[2],
						LocalSCacheMgr.bAutoCalcMaxDirty);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiMaxDirtyCache</td>\n"
						"<td>FLMUINT</td>\n" TD_lu, szOffsetTable[3],
						LocalSCacheMgr.uiMaxDirtyCache);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiLowDirtyCache</td>\n"
						"<td>FLMUINT</td>\n" TD_lu, szOffsetTable[4],
						LocalSCacheMgr.uiLowDirtyCache);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiTotalUses</td>\n"
						"<td>FLMUINT</td>\n" TD_lu, szOffsetTable[5],
						LocalSCacheMgr.uiTotalUses);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiBlocksUsed</td> <td>FLMUINT</td>\n"
					TD_lu, szOffsetTable[6], LocalSCacheMgr.uiBlocksUsed);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiPendingReads</td>\n"
			"<td>FLMUINT</td>\n" TD_lu,  szOffsetTable[7],
			LocalSCacheMgr.uiPendingReads);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiIoWaits</td>\n <td>FLMUINT</td>\n" TD_lu,
						szOffsetTable[8], LocalSCacheMgr.uiIoWaits);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiHashTableSize</td>\n"
					"<td>FLMUINT</td>\n" TD_lu, szOffsetTable[9],
					LocalSCacheMgr.uiHashTblSize);
		printTableRowEnd();

		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>uiHashTableBits</td>\n"
					"<td>FLMUINT</td>\n" TD_lu, szOffsetTable[10],
					LocalSCacheMgr.uiHashTblBits);
		printTableRowEnd();

#ifdef FLM_DEBUG
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s "<td>bDebug</td>\n" "<td>FLMBOOL</td>\n"
					TD_i, szOffsetTable[11], LocalSCacheMgr.bDebug);
		printTableRowEnd();
#endif

		printTableEnd();
		
		fnPrintf( m_pHRequest, "</BODY></HTML>\n");

		fnEmit();

	}

Exit:

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	for (uiLoop = 0; uiLoop < NUM_CACHE_REQ_STRINGS; uiLoop++)
	{
		if( pszSCacheRequestString[uiLoop])
		{
			f_free( &pszSCacheRequestString[uiLoop]);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Prints the web page for the SCacheHashTable
****************************************************************************/
RCODE F_SCacheHashTablePage::display(
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
	char *		pszTemp;
#define NUM_ENTRIES 20
	char *		pszHTLinks[NUM_ENTRIES];

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

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

	// Lock the database
	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Get the number of entries in the hash table
	uiHashTableSize = gv_FlmSysData.SCacheMgr.uiHashTblSize;
	
	// May need to modify starting number if it's out of range...
	if ((uiStart + NUM_ENTRIES) >= uiHashTableSize)
	{
		uiStart = uiHashTableSize - NUM_ENTRIES;
	}


	// Loop through the entire table counting the number of entries in use
	// If the entry is one of the one's we're going to display, store the 
	// appropriate text in pszHTLinks
	for (uiLoop = 0; uiLoop < uiHashTableSize; uiLoop++)
	{
		if (gv_FlmSysData.SCacheMgr.ppHashTbl[uiLoop])
		{
			uiUsedEntries++;
		}

		if (	(uiLoop >= uiStart) &&
				(uiLoop < (uiStart + NUM_ENTRIES)) )
		{
			// This is one of the entries that we will display
			if (gv_FlmSysData.SCacheMgr.ppHashTbl[uiLoop])
			{
				flmBuildSCacheBlockString( pszHTLinks[uiLoop - uiStart], 
					gv_FlmSysData.SCacheMgr.ppHashTbl[uiLoop]);
			}

		}


	}

	// Unlock the database
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
			"<META http-equiv=\"refresh\" content=\"5; url=%s/SCacheHashTable?Start=%lu%s\">"
			"<TITLE>Database iMonitor - SCache Hash Table</TITLE>\n", m_pszURLString, uiStart, szRefresh);
	
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
			       "<A HREF=%s/SCacheHashTable?Start=%lu&Refresh>Start Auto-refresh (5 sec.)</A>",
					 m_pszURLString, uiStart);
	}
	else
	{
		f_sprintf( (char *)pszTemp,
			       "<A HREF=%s/SCacheHashTable?Start=%lu>Stop Auto-refresh</A>",
					 m_pszURLString, uiStart);
	}

	// Print out a formal header and the refresh option.
	printTableStart("SCache Hash Table", 4);

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/SCacheHashTable?Start=%lu%s>Refresh</A>, %s\n",
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
	uiNewStart = (uiStart > 100)?(uiStart - 100):0;
	fnPrintf( m_pHRequest, "<A HREF=%s/SCacheHashTable?Start=%lu%s>Previous 100</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);
	uiNewStart = (uiStart > 10)?(uiStart - 10):0;
	fnPrintf( m_pHRequest, "<A HREF=%s/SCacheHashTable?Start=%lu%s>Previous 10</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);

	fnPrintf( m_pHRequest, "<BR>\n");
	uiNewStart = (uiStart + 10);
	if (uiNewStart >= (uiHashTableSize - NUM_ENTRIES))
	{
		uiNewStart = (uiHashTableSize - NUM_ENTRIES);
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/SCacheHashTable?Start=%lu%s>Next 10</A> <BR>\n",
					m_pszURLString, uiNewStart, szRefresh);

	uiNewStart = (uiStart + 100);
	if (uiNewStart >= (uiHashTableSize - NUM_ENTRIES))
	{
		uiNewStart = (uiHashTableSize - NUM_ENTRIES);
	}
	fnPrintf( m_pHRequest, "<A HREF=%s/SCacheHashTable?Start=%lu%s>Next 100</A> <BR>\n"
				"<form type=\"submit\" method=\"get\" action=\"/coredb/SCacheHashTable\">\n"
				"<BR> Jump to specific bucket:<BR> \n"
				"<INPUT type=\"text\" size=\"10\" maxlength=\"10\" name=\"Start\"></INPUT> <BR>\n",
				m_pszURLString, uiNewStart, szRefresh);
	printButton( "Jump", BT_Submit);
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
 Desc:	Searches the SCache for the block referenced by the parameters.  If
			found, it copies the data into pLocalSCache.  Assumes that the 
			mutex has already been locked!
****************************************************************************/
RCODE F_SCacheBase::locateSCacheBlock(
	FLMUINT			uiNumParams,
	const char **	ppszParams,
	SCACHE *			pLocalSCache,
	FLMUINT *		puiBlkAddress,
	FLMUINT *		puiLowTransID,
	FLMUINT *		puiHighTransID,
	FFILE * *		ppFile)
{
	RCODE				rc = FERR_OK;

	FLMUINT			uiSigBitsInBlkSize;
	SCACHE *			pSCache;
	SCACHE **		ppSCache;
#define MAXPARAMLEN 15
	char				szBlkAddress[MAXPARAMLEN];
	char				szLowTransID[MAXPARAMLEN];
	char				szHighTransID[MAXPARAMLEN];
	char				szFile[MAXPARAMLEN];

	// Grab the block address, low and high trans id's and FFile pointer, which
	// we need to uniquely identify an scache block...
	
	if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
												  "BlockAddress", sizeof( szBlkAddress),
												  &szBlkAddress[0])))
	{  
		goto Exit;
	}
	*puiBlkAddress = f_atoi( szBlkAddress);

	if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
												  "LowTransID", sizeof( szLowTransID),
												  &szLowTransID[0])))
	{
		goto Exit;
	}
	*puiLowTransID = f_atoi( szLowTransID);

	if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams,
												  "HighTransID", sizeof( szHighTransID),
												  &szHighTransID[0])))
	{
		goto Exit;
	}
	*puiHighTransID = f_atoi( szHighTransID);

	if (RC_BAD( rc = ExtractParameter( uiNumParams, ppszParams, 
												  "File", sizeof( szFile),
												  &szFile[0])))
	{ 
		goto Exit;
	}
	*ppFile = (FFILE *)f_atoud( szFile);

	flmAssert( *ppFile);
	uiSigBitsInBlkSize = (*ppFile)->FileHdr.uiSigBitsInBlkSize;

	// ScaHash actually returns a pointer to the first scache in the hash
	// bucket. It's up to us to traverse this list to find the proper block
	// address and FFile (and potentially high and low trans id)
	ppSCache = ScaHash( uiSigBitsInBlkSize,	*puiBlkAddress);
	pSCache = *ppSCache;

	while (	pSCache &&
				(	(pSCache->uiBlkAddress != *puiBlkAddress)	||
					(pSCache->pFile != *ppFile) )					)
	{
		pSCache = pSCache->pNextInHashBucket;
	}

	// Ok - we've found the right address and ffile.  Do we need a different
	// version?
	while (	(pSCache)	&&
				(pSCache->uiHighTransID != *puiHighTransID)	&&
				(scaGetLowTransID( pSCache) != *puiLowTransID) )
	{
		pSCache = pSCache->pNextInVersionList;
	}


	// Now, if we've found the right block, copy it's contents to local memory...
	if (pSCache)
	{
		f_memcpy( pLocalSCache, pSCache, sizeof( SCACHE));
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:
	return( rc);
}

/****************************************************************************
 Desc:	Spits out a short message saying that the SCache block you were 
			looking for wasn't found.  Used when locateSCacheBlock()
			returns FERR_NOT_FOUND.
****************************************************************************/
void F_SCacheBase::notFoundErr()
{
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE "<html>\n<HEAD>\n");
	printStyle();
	fnPrintf( m_pHRequest,
		"</HEAD><BODY>\n"
		"<H2 ALIGN=CENTER>SCache Block Not Found</H2>"
		"<HR><P> Unable to find the SCache Block that you requested."
		"  This is probably because the state of the cache changed between the time"
		" that you displayed the previous page and the time that you clicked on the"
		" link that brought you here.  (It's also possible that the link you selected"
		" was a NULL pointer.) \n"
		" <P>Your best bet is probably to click on the \"Database System Data\" link on"
		" the left.</P>\n</BODY></HTML>\n");
	fnEmit();
}


/****************************************************************************
 Desc:	Spits out a short message saying that the URL for the SCache
			block that was requested was badly formed.  Used when 
			locateSCacheBlock() returns FERR_MEM.
****************************************************************************/
void F_SCacheBase::malformedUrlErr()
{
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE "<html>\n<HEAD>\n");
	printStyle();
	fnPrintf( m_pHRequest,
		"</HEAD><BODY>\n"
		"<H2 ALIGN=CENTER>Bad SCache Block URL</H2>"
		"<HR><P> Couldn't process requested URL.  Is"
		" the query string properly formed?.</P>\n</BODY></HTML>\n");
	fnEmit();
}

/****************************************************************************
 Desc:	Generates the html to display a row in a table.  If *ppSCache is
			a valid pointer, then that line will have hyperlinks in it.
****************************************************************************/
FSTATIC void flmPrintCacheLine(
	HRequest *		pHRequest,
	const char *	pszHREF,
	const char *	pszName,
	void *			pBaseAddr,
	SCACHE **		ppSCache)
{
	char				szAddress[20];
	char				szOffset[8];
	PRINTF_FN		fnPrintf = gv_FlmSysData.HttpConfigParms.fnPrintf;

	printAddress( *ppSCache, szAddress);
	printOffset( pBaseAddr, ppSCache, szOffset);

	if ((*ppSCache) && (*ppSCache)->pFile && pszHREF)
	{
		// We have a pointer to a valid SCache block and a valid HREF string,
		// so we need hyperlinks to appear in the browser...
		fnPrintf( pHRequest, TD_s TD_a_s_s " </td>	<td> SCACHE * </td> " TD_a_s_s TR_END,
					 szOffset, pszHREF, pszName, pszHREF, szAddress);
	}
	else
	{
		fnPrintf( pHRequest, TD_s TD_s " <td> SCACHE * </td> " TD_s TR_END,
					 szOffset, pszName, szAddress );
	}
}

/****************************************************************************
 Desc:	Determines the values of the parameters needed to reference
			a specific SCache block.  Must be called from within a mutex
****************************************************************************/
FSTATIC void flmBuildSCacheBlockString(
	char *			pszString,
	SCACHE *			pSCache)
{
	char				szAddress[ 20];

	if ((pSCache == NULL) || (pSCache->pFile == NULL))
	{
		pszString[0] = 0;
	}
	else
	{
		printAddress( pSCache->pFile, szAddress);
		f_sprintf( (char *)pszString,
				"%s/SCacheBlock?BlockAddress=%lu&File=%s&LowTransID=%lu&HighTransID=%lu",
				gv_FlmSysData.HttpConfigParms.pszURLString, pSCache->uiBlkAddress, szAddress,
				scaGetLowTransID( pSCache), pSCache->uiHighTransID);
	}
	
	return;
}
