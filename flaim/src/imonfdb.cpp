//-------------------------------------------------------------------------
// Desc:	Class for displaying an FDB structure in HTML on a web page.
// Tabs:	3
//
// Copyright (c) 2001-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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
 Desc:	This procedure generates the HTML page to display
			the contents of the FDB structure
 ****************************************************************************/
RCODE F_FDBPage::display(
	FLMUINT			uiNumParams,
	const char **	ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bRefresh = FALSE;
#define GENERIC_SIZE_A 100
	char			szAddressParm[ GENERIC_SIZE_A];
	void *		pvFFileAddress;
	void *		pvFDBAddress;
	char			szFFileAddress[20];
	char			szFDBAddress[20];
	char			szBucket[20];
	FLMINT		uiBucket;
	FDB			LocalFDB;
	FDB *			pDb=NULL;
	FFILE *		pFile = NULL;
	char			szTemp[GENERIC_SIZE_A];
	FLMBOOL		bpFileInc = FALSE;
	char			szAddress[20];
	char *		pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 200, &pszTemp)))
	{
		printErrorPage( rc, TRUE, "Failed to allocate temporary buffer");
		goto Exit;
	}

	// Let's extract as much information as we can from the parameters
	// before we proceed.

	// Determine if we are being requested to refresh this page or not.
	
	bRefresh = DetectParameter( uiNumParams, ppszParams, "Refresh");


	// FFileAddress - required
	if (RC_BAD(rc = ExtractParameter( uiNumParams,
												 ppszParams,
												 "FFileAddress",
												 sizeof( szAddressParm),
												 szAddressParm)))
	{
		goto Exit;
	}
	else
	{
		pvFFileAddress = (void *)f_atoud( szAddressParm);
	}


	// FDBAddress - required
	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "FDBAddress",
												  sizeof( szAddressParm),
												  szAddressParm)))
	{
		goto Exit;
	}
	else
	{
		pvFDBAddress = (void *)f_atoud( szAddressParm);
	}


	// Bucket - index into the file hash table - required
	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "Bucket",
												  sizeof( szBucket),
												  szBucket)))
	{
		goto Exit;
	}
	else
	{
		uiBucket = f_atoud( szBucket);
	}

	// Now we will search for the FFile first, then look for the FDB.		
	f_mutexLock( gv_FlmSysData.hShareMutex);

	pFile = (FFILE *)gv_FlmSysData.pFileHashTbl[uiBucket].pFirstInBucket;
			
	while ((pFile != NULL) && ((void *)pFile != pvFFileAddress))
	{
		pFile = pFile->pNext;
	}
		
	if (pFile)
	{
		pDb = pFile->pFirstDb;

		//Now let's look for the FDB we want...
		while (pDb && ((void *)pDb != pvFDBAddress))
		{
			pDb = pDb->pNextForFile;
		}


		if (pDb)
		{
			f_memcpy( &LocalFDB, pDb, sizeof(LocalFDB));
		}

		// Now we want to make sure the pFile doesn't go away while we are 
		// using it.
		if (++pFile->uiUseCount == 1)
		{
			flmUnlinkFileFromNUList( pFile);
		}
		bpFileInc = TRUE;
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	// Save the FFileAddress and the FDBAddress.
	printAddress( pvFFileAddress, szAddress);
	f_sprintf( szFFileAddress, "%s", szAddress);
	printAddress( pvFDBAddress, szAddress);
	f_sprintf( szFDBAddress, "%s", szAddress);

	// At this point we will either have a valid FDB or we will have not been able
	// to find it.
	
	stdHdr();
	
	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	
	if (bRefresh)
	{
		// Send back the page with a refresh command in the header
		f_sprintf( szTemp,
					"%s/FDB?Refresh?FFileAddress=%s?Bucket=%s?FDBAddress=%s",
					m_pszURLString,
					szFFileAddress, szBucket, szFDBAddress);

		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>FDB - Database Context Structure</TITLE></HEAD>\n",
			szTemp);

	}
	else
	{
		fnPrintf( m_pHRequest, 
			"<HEAD><TITLE>FDB - Database Context Structure</TITLE></HEAD>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");

	
	fnPrintf( m_pHRequest, "<body>\n");


	// Prepare the Auto-refresh link.
	if (bRefresh)
	{
		f_sprintf( szTemp, "%s/FDB?FFileAddress=%s?Bucket=%s?FDBAddress=%s",
					m_pszURLString,
					szFFileAddress, szBucket, szFDBAddress);

		f_sprintf( pszTemp,
					"<A HREF=%s>Stop Auto-refresh</A>", szTemp);
	}
	else
	{

		// Send back a page without the refresh command
		f_sprintf( szTemp, 
					"%s/FDB?Refresh?FFileAddress=%s?Bucket=%s?FDBAddress=%s",
					m_pszURLString,
					szFFileAddress, szBucket, szFDBAddress);

		f_sprintf( pszTemp,
					"<a href=%s>Start Auto-refresh (5 sec.)</a>", szTemp);
	}

	// Prepare the Refresh link.
	f_sprintf( szTemp, "%s/FDB?FFileAddress=%s?Bucket=%s?FDBAddress=%s",
					m_pszURLString,
					szFFileAddress, szBucket, szFDBAddress);

	if (!pDb)
	{
		// Write out an error page...
		fnPrintf( m_pHRequest,  "<P> Unable to find the FDB structure that "
									"you requested. This is probably because the state"
									" of the system changed between the time that you "
									"displayed the previous page and the time that "
									"you clicked on the link that brought you here.\n"
									"<P>Click on your browser's \"Back\" button, then"
									" click \"Reload\" and then try the link again.");
	}
	else
	{

		printTableStart( "FDB Database Context", 4, 100);
		
		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
		fnPrintf( m_pHRequest, "<a href=%s>Refresh</a>, ", szTemp);
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
	

		// Insert a new table into the page to display the FDB fields
		write_data( (pDb ? &LocalFDB : NULL), szFDBAddress, uiBucket);

	}

	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();


Exit:

	// Free up the pFile now.
	if (bpFileInc)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);

		if (--pFile->uiUseCount == 0)
		{
			flmLinkFileToNUList( pFile);
		}

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);
}


/****************************************************************************
 Desc:	This procedure writes the actual HTML page to display
			the contents of the FDB structure
 ****************************************************************************/
void F_FDBPage::write_data(
	FDB *				pDb,
	const char *	pszFDBAddress,
	FLMUINT			uiBucket)
{
	FLMUINT		uiFlag;
	char			szTemp[100];
	char			szAddress[20];
	char			szOffset[8];
	FLMBOOL		bHighlight = FALSE;

	if (!pDb)
	{
		flmAssert(0);
		return;
	}
	else
	{

		// pFile
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf( szTemp,
						"%s/FFile?From=FDB?Address=%s?Bucket=%lu",
						m_pszURLString,
						szAddress, (unsigned long)uiBucket);

		}

		printHTMLLink( "pFile", "FFILE *", (void *)pDb, (void *)&pDb->pFile,
			(void *)pDb->pFile, szTemp, (bHighlight = ~bHighlight));

		
		// pDict
		if (pDb->pDict && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "%s/FDICT?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress, (unsigned long)uiBucket,
				pszFDBAddress);
			
			printHTMLLink(
				"pDict",
				"FDICT *",
				(void *)pDb,
				(void *)&pDb->pDict,
				(void *)pDb->pDict,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pDict, szAddress);
			printHTMLString(
				"pDict",
				"FDICT *",
				(void *)pDb,
				(void *)&pDb->pDict,
				szAddress,
				(bHighlight = ~bHighlight));
		}

		// pNextForFile
		if (pDb->pNextForFile && pDb->pFile)
		{
			char		szFDBAddr[ 20];

			printAddress( pDb->pNextForFile, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket, szFDBAddr);
		
			printHTMLLink(
				"pNextForFile",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pNextForFile,
				(void *)pDb->pNextForFile,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pNextForFile, szAddress);
			printHTMLString(
				"pNextForFile",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pNextForFile,
				szAddress,
				(bHighlight = ~bHighlight));
		}
		


		// pPrevForFile
		if (pDb->pPrevForFile && pDb->pFile)
		{
			char			szFDBAddr[20];

			printAddress( pDb->pPrevForFile, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket, szFDBAddr);
		
			printHTMLLink(
				"pPrevForFile",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pPrevForFile,
				(void *)pDb->pPrevForFile,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pPrevForFile, szAddress);
			printHTMLString(
				"pPrevForFile",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pPrevForFile,
				szAddress,
				(bHighlight = ~bHighlight));
		}
		


		// pvAppData
		printAddress( pDb->pvAppData, szAddress);
		printHTMLString(
			"pvAppData",
			"void *",
			(void *)pDb,
			(void *)&pDb->pvAppData,
			szAddress,
			(bHighlight = ~bHighlight));



		// uiThreadId
		printHTMLUint(
			"uiThreadId",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiThreadId,
			pDb->uiThreadId,
			(bHighlight = ~bHighlight));




		// uiInitNestLevel
		printHTMLUint(
			"uiInitNestLevel",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiInitNestLevel,
			pDb->uiInitNestLevel,
			(bHighlight = ~bHighlight));




		// uiInFlmFunc
		printHTMLUint(
			"uiInFlmFunc",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiInFlmFunc,
			pDb->uiInFlmFunc,
			(bHighlight = ~bHighlight));

		

		// pSFileHdl
		if (pDb->pSFileHdl && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/SFileHdl?FFileAddress=%s?Link=pSFileHdl?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"pSFileHdl",
				"F_SuperFileHdl *",
				(void *)pDb,
				(void *)&pDb->pSFileHdl,
				(void *)pDb->pSFileHdl,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pSFileHdl, szAddress);
			printHTMLString(
				"pSFileHdl",
				"F_SuperFileHdl *",
				(void *)pDb,
				(void *)&pDb->pSFileHdl,
				szAddress,
				(bHighlight = ~bHighlight));
		}

		// uiFlags
		printOffset( (void *)pDb, 
						 (void *)&pDb->uiFlags,
						 szOffset);
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest,  TD_s, szOffset);
		fnPrintf( m_pHRequest,  TD_s, "uiFlags");
		fnPrintf( m_pHRequest,  TD_s, "FLMUINT");
		uiFlag = 0;
		if (pDb->uiFlags & FDB_UPDATED_DICTIONARY)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_UPDATED_DICTIONARY" : 
												  "<td>FDB_UPDATED_DICTIONARY");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_DO_TRUNCATE)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_DO_TRUNCATE" :
												  "<td>FDB_DO_TRUNCATE");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_INVISIBLE_TRANS)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_INVISIBLE_TRANS" :
												  "<td>FDB_INVISIBLE_TRANS");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_HAS_FILE_LOCK)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_HAS_FILE_LOCK" :
												  "<td>FDB_HAS_FILE_LOCK");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_FILE_LOCK_SHARED)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_FILE_LOCK_SHARED" :
												  "<td>FDB_FILE_LOCK_SHARED");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_FILE_LOCK_IMPLICIT" :
												  "<td>FDB_FILE_LOCK_IMPLICIT");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_DONT_KILL_TRANS)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_DONT_KILL_TRANS" :
												  "<td>FDB_DONT_KILL_TRANS");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_INTERNAL_OPEN)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_INTERNAL_OPEN" :
												  "<td>FDB_INTERNAL_OPEN");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_DONT_POISON_CACHE)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_DONT_POISON_CACHE" :
												  "<td>FDB_DONT_POISON_CACHE");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_UPGRADING)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_UPGRADING" :
												  "<td>FDB_UPGRADING");
			uiFlag++;
		}
		if (pDb->uiFlags & FDB_REPLAYING_RFL)
		{
			fnPrintf( m_pHRequest, uiFlag ? "<BR>FDB_REPLAYING_RFL" :
												  "<td>FDB_REPLAYING_RFL");
			uiFlag++;
		}
		if (!uiFlag)
		{
			fnPrintf( m_pHRequest,  TD_8x, pDb->uiFlags);
		}
		else
		{
			fnPrintf( m_pHRequest, "</td>\n");
		}
		printTableRowEnd();




		// uiTransCount
		printHTMLUint(
			"uiTransCount",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiTransCount,
			pDb->uiTransCount,
			(bHighlight = ~bHighlight));

		

		// uiTransType
		switch (FLM_GET_TRANS_TYPE(pDb->uiTransType))
		{
		case FLM_NO_TRANS:
			f_sprintf( szTemp, "No Transaction");
			break;
		case FLM_UPDATE_TRANS:
			f_sprintf( szTemp, "Update Transaction");
			break;
		case FLM_READ_TRANS:
			f_sprintf( szTemp, "Read Transaction");
			break;
		default:
			f_sprintf( szTemp, "%lu", pDb->uiTransType);
			break;
		}

		printHTMLString(
			"uiTransType",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiTransType,
			szTemp,
			(bHighlight = ~bHighlight));




		// AbortRc
		f_sprintf( szTemp, "%04X", (unsigned)pDb->AbortRc);
		printHTMLString(
			"AbortRc",
			"RCODE",
			(void *)pDb,
			(void *)&pDb->AbortRc,
			szTemp,
			(bHighlight = ~bHighlight));


		// LogHdr
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/LogHdr?FileAddress=%s?Link=LogHdr?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"LogHdr",
				"FlmRecordFactory *",
				(void *)pDb,
				(void *)&pDb->LogHdr,
				(void *)&pDb->LogHdr,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->LogHdr, szAddress);
			printHTMLString(
				"LogHdr",
				"FLMUINT",
				(void *)pDb,
				(void *)&pDb->LogHdr,
				szAddress,
				(bHighlight = ~bHighlight));
		}
		



		// uiUpgradeCPFileNum
		printHTMLUint(
			"uiUpgradeCPFileNum",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiUpgradeCPFileNum,
			pDb->uiUpgradeCPFileNum,
			(bHighlight = ~bHighlight));




		// uiUpgradeCPOffset
		printHTMLUint(
			"uiUpgradeCPOffset",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiUpgradeCPOffset,
			pDb->uiUpgradeCPOffset,
			(bHighlight = ~bHighlight));




		// uiTransEOF
		printHTMLUint(
			"uiTransEOF",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiTransEOF,
			pDb->uiTransEOF,
			(bHighlight = ~bHighlight));



		// KrefCntrl
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, 
				"%s/KREF_CNTRL?FFileAddress=%s?Link=KrefCntrl?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);

			printHTMLLink(
				"KrefCntrl",
				"KREF_CNTRL",
				(void *)pDb,
				(void *)&pDb->KrefCntrl,
				(void *)&pDb->KrefCntrl,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->KrefCntrl, szAddress);
			printHTMLString(
				"KrefCntrl",
				"KREF_CNTRL",
				(void *)pDb,
				(void *)&pDb->KrefCntrl,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// pIxStats
		if ((pDb->pIxStats) &&
			 (pDb->pFile))
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/IX_STATS?FFileAddress=%s?Link=pIxStats?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);

			printHTMLLink(
				"pIxStats",
				"IX_STATS *",
				(void *)pDb,
				(void *)&pDb->pIxStats,
				(void *)pDb->pIxStats,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pIxStats, szAddress);
			printHTMLString(
				"pIxStats",
				"IX_STATS *",
				(void *)pDb,
				(void *)&pDb->pIxStats,
				szAddress,
				(bHighlight = ~bHighlight));
		}
		
		


		// bHadUpdOper
		printHTMLString(
			"bHadUpdOper",
			"FLMBOOL",
			(void *)pDb,
			(void *)&pDb->bHadUpdOper,
			(pDb->bHadUpdOper ? "Yes" : "No"),
			(bHighlight = ~bHighlight));

			


		// uiBlkChangeCnt
		printHTMLUint(
			"uiBlkChangeCnt",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiBlkChangeCnt,
			pDb->uiBlkChangeCnt,
			(bHighlight = ~bHighlight));




		// pBlobList
		if (pDb->pBlobList && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/FlmBlob?FFileAddress=%s?Link=pBlobList?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);

			printHTMLLink(
				"pBlobList",
				"FlmBlob *",
				(void *)pDb,
				(void *)&pDb->pBlobList,
				(void *)pDb->pBlobList,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pBlobList, szAddress);
			printHTMLString(
				"pBlobList",
				"FlmBlob *",
				(void *)pDb,
				(void *)&pDb->pBlobList,
				szAddress,
				(bHighlight = ~bHighlight));
		}



		// pIxdFixups
		if (pDb->pIxdFixups && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/IXD_FIXUP?FFileAddress=%s?Link=pIxdFixups?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);

			printHTMLLink(
				"pIxdFixups",
				"IXD_FIXUP_p",
				(void *)pDb,
				(void *)&pDb->pIxdFixups,
				(void *)pDb->pIxdFixups,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pIxdFixups, szAddress);
			printHTMLString(
				"pIxdFixups",
				"IXD_FIXUP_p",
				(void *)pDb,
				(void *)&pDb->pIxdFixups,
				szAddress,
				(bHighlight = ~bHighlight));
		}



		// pNextReadTrans
		if (pDb->pNextReadTrans && pDb->pFile)
		{
			char			szFDBAddr[20];

			printAddress( pDb->pNextReadTrans, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket, szFDBAddr);

			printHTMLLink(
				"pNextReadTrans",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pNextReadTrans,
				(void *)pDb->pNextReadTrans,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pNextReadTrans, szAddress);
			printHTMLString(
				"pNextReadTrans",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pNextReadTrans,
				szAddress,
				(bHighlight = ~bHighlight));
		}





		// pPrevReadTrans
		if (pDb->pPrevReadTrans && pDb->pFile)
		{
			char			szFDBAddr[20];

			printAddress( pDb->pPrevReadTrans, szAddress);
			f_sprintf( szFDBAddr, "%s", szAddress);
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/FDB?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket, szFDBAddr);
		
			printHTMLLink(
				"pPrevReadTrans",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pPrevReadTrans,
				(void *)pDb->pPrevReadTrans,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pPrevReadTrans, szAddress);
			printHTMLString(
				"pPrevReadTrans",
				"FDB *",
				(void *)pDb,
				(void *)&pDb->pPrevReadTrans,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// uiInactiveTime
		FormatTime(pDb->uiInactiveTime, szTemp);
		printHTMLString(
			"uiInactiveTime",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiInactiveTime,
			szTemp,
			(bHighlight = ~bHighlight));




		// uiKilledTime
		FormatTime(pDb->uiKilledTime, szTemp);
		printHTMLString(
			"uiKilledTime",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiKilledTime,
			szTemp,
			(bHighlight = ~bHighlight));




		// tmpKrefPool
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
			szTemp,
			"%s/POOL?FFileAddress=%s?Link=tmpKrefPool?Bucket=%lu?FDBAddress=%s",
			m_pszURLString,
			szAddress,
			(unsigned long)uiBucket,
			pszFDBAddress);
		
			printHTMLLink(
				"tmpKrefPool",
				"POOL",
				(void *)pDb,
				(void *)&pDb->tmpKrefPool,
				(void *)&pDb->tmpKrefPool,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->tmpKrefPool, szAddress);
			printHTMLString(
				"tmpKrefPool",
				"POOL",
				(void *)pDb,
				(void *)&pDb->tmpKrefPool,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// bFldStateUpdOk
		printHTMLString(
			"bFldStateUpdOk",
			"FLMBOOL",
			(void *)pDb,
			(void *)&pDb->bFldStateUpdOk,
			(pDb->bFldStateUpdOk ? "Yes" : "No"),
			(bHighlight = ~bHighlight));




		// Diag
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/FDIAG?FFileAddress=%s?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"Diag",
				"FDIAG",
				(void *)pDb,
				(void *)&pDb->Diag,
				(void *)&pDb->Diag,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->Diag, szAddress);
			printHTMLString(
				"Diag",
				"FDIAG",
				(void *)pDb,
				(void *)&pDb->Diag,
				szAddress,
				(bHighlight = ~bHighlight));
		}



		// TempPool
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp,
				"%s/POOL?FFileAddress=%s?Link=TempPool?Bucket=%lu?FDBAddress=%s",
				m_pszURLString,
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"TempPool",
				"POOL",
				(void *)pDb,
				(void *)&pDb->TempPool,
				(void *)&pDb->TempPool,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->TempPool, szAddress);
			printHTMLString(
				"TempPool",
				"POOL",
				(void *)pDb,
				(void *)&pDb->TempPool,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// fnRecValidator
		printAddress( *((void **)&pDb->fnRecValidator), szAddress);
		printHTMLString(
			"fnRecValidator",
			"REC_VALIDATOR_HOOK",
			(void *)pDb,
			(void *)&pDb->fnRecValidator,
			szAddress,
			(bHighlight = ~bHighlight));




		// RecValData
		printAddress( pDb->RecValData, szAddress);
		printHTMLString(
			"RecValData",
			"void *",
			(void *)pDb,
			(void *)&pDb->RecValData,
			szAddress,
			(bHighlight = ~bHighlight));




		// fnStatus
		printAddress( *((void **)&pDb->fnStatus), szAddress);
		printHTMLString(
			"fnStatus",
			"STATUS_HOOK",
			(void *)pDb,
			(void *)&pDb->fnStatus,
			szAddress,
			(bHighlight = ~bHighlight));



		// StatusData
		printAddress( pDb->StatusData, szAddress);
		printHTMLString(
			"StatusData",
			"void *",
			(void *)pDb,
			(void *)&pDb->StatusData,
			szAddress,
			(bHighlight = ~bHighlight));




		// fnIxCallback
		printAddress( *((void **)&pDb->fnIxCallback), szAddress);
		printHTMLString(
			"fnIxCallback",
			"IX_CALLBACK",
			(void *)pDb,
			(void *)&pDb->fnIxCallback,
			szAddress,
			(bHighlight = ~bHighlight));




		// IxCallbackData
		printAddress( pDb->IxCallbackData, szAddress);
		printHTMLString(
			"IxCallbackData",
			"void *",
			(void *)pDb,
			(void *)&pDb->IxCallbackData,
			szAddress,
			(bHighlight = ~bHighlight));




		// pStats
		if (pDb->pStats && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "FLM_STATS?FFileAddress=%s?Link=pStats?Bucket=%lu?FDBAddress=%s",
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);

			printHTMLLink(
				"pStats",
				"FLM_STATS *",
				(void *)pDb,
				(void *)&pDb->pStats,
				(void *)pDb->pStats,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pStats, szAddress);
			printHTMLString(
				"pStats",
				"FLM_STATS *",
				(void *)pDb,
				(void *)&pDb->pStats,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// pDbStats
		if (pDb->pDbStats && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "DB_STATS?FFileAddress=%s?Link=pDbStats?Bucket=%lu?FDBAddr=%s",
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"pDbStats",
				"DB_STATS *",
				(void *)pDb,
				(void *)&pDb->pDbStats,
				(void *)pDb->pDbStats,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pDbStats, szAddress);
			printHTMLString(
				"pDbStats",
				"DB_STATS *",
				(void *)pDb,
				(void *)&pDb->pDbStats,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// pLFileStats
		if (pDb->pLFileStats && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "LFILE_STATS?FFileAddress=%s?Link=pLFileStats?Bucket=%lu?FDBAddress=%s",
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"pLFileStats",
				"LFILE_STATS *",
				(void *)pDb,
				(void *)&pDb->pLFileStats,
				(void *)pDb->pLFileStats,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pLFileStats, szAddress);
			printHTMLString(
				"pLFileStats",
				"LFILE_STATS *",
				(void *)pDb,
				(void *)&pDb->pLFileStats,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// uiLFileAllocSeq
		printHTMLUint(
			"uiLFileAllocSeq",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiLFileAllocSeq,
			pDb->uiLFileAllocSeq,
			(bHighlight = ~bHighlight));




		// Stats
		if (pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "FLM_STATS?FFileAddress=%s?Link=Stats?Bucket=%lu?FDBAddress=%s",
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"Stats",
				"FLM_STATS",
				(void *)pDb,
				(void *)&pDb->Stats,
				(void *)&pDb->Stats,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( &pDb->Stats, szAddress);
			printHTMLString(
				"Stats",
				"FLM_STATS",
				(void *)pDb,
				(void *)&pDb->Stats,
				szAddress,
				(bHighlight = ~bHighlight));
		}



		// bStatsInitialized
		printHTMLString(
			"bStatsInitialized",
			"FLMBOOL",
			(void *)pDb,
			(void *)&pDb->bStatsInitialized,
			(pDb->bStatsInitialized ? "Yes" : "No"),
			(bHighlight = ~bHighlight));


		// pCSContext
		if (pDb->pCSContext && pDb->pFile)
		{
			printAddress( pDb->pFile, szAddress);
			f_sprintf(
				szTemp, "CS_CONTEXT?FFileAddress=%s?Link=pCSContext?Bucket=%lu?FDBAddress=%s",
				szAddress,
				(unsigned long)uiBucket,
				pszFDBAddress);
		
			printHTMLLink(
				"pCSContext",
				"CS_CONTECT_p",
				(void *)pDb,
				(void *)&pDb->pCSContext,
				(void *)pDb->pCSContext,
				szTemp,
				(bHighlight = ~bHighlight));
		}
		else
		{
			printAddress( pDb->pCSContext, szAddress);
			printHTMLString(
				"pCSContext",
				"CS_CONTECT_p",
				(void *)pDb,
				(void *)&pDb->pCSContext,
				szAddress,
				(bHighlight = ~bHighlight));
		}




		// pIxStartList
		printAddress( pDb->pIxStartList, szAddress);
		printHTMLString(
			"pIxStartList",
			"F_BKGND_IX *",
			(void *)pDb,
			(void *)&pDb->pIxStartList,
			szAddress,
			(bHighlight = ~bHighlight));

		
		
		// pIxStopList
		printAddress( pDb->pIxStopList, szAddress);
		printHTMLString(
			"pIxStopList",
			"F_BKGND_IX *",
			(void *)pDb,
			(void *)&pDb->pIxStopList,
			szAddress,
			(bHighlight = ~bHighlight));



#ifdef FLM_DEBUG

		// hMutex
		printAddress( pDb->hMutex, szAddress);
		printHTMLString(
			"hMutex",
			"F_MUTEX",
			(void *)pDb,
			(void *)&pDb->hMutex,
			szAddress,
			(bHighlight = ~bHighlight));




		// uiUseCount
		printHTMLUint(
			"uiUseCount",
			"FLMUINT",
			(void *)pDb,
			(void *)&pDb->uiUseCount,
			pDb->uiUseCount,
			(bHighlight = ~bHighlight));

#endif


		printTableEnd();

	}
}
