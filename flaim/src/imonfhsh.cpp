//-------------------------------------------------------------------------
// Desc:	Class for displaying an file hash table in HTML on a web page.
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
Desc: This function implements the display method of the F_FileHashTblPage
		class.
*****************************************************************************/
RCODE F_FileHashTblPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;
	FLMINT		iIndex;
	FLMINT		iSindex;
	FLMINT		iNindex;
	FLMBOOL		found = FALSE;
	F_BUCKET *	pFileHashTbl;
	FLMBOOL		buckets[FILE_HASH_ENTRIES];
	FLMINT		next[FILE_HASH_ENTRIES];
	FLMBOOL		bRefresh;

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Determine if we are being requested to refresh this page or not.

	if ((bRefresh = DetectParameter( uiNumParams,
											   ppszParams, 
											   "Refresh")) == TRUE)
	{
		fnPrintf( 
			m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s/FileHashTbl?Refresh\">"
			"<TITLE>gv_FlmSysData.pFileHashTbl</TITLE>\n",
			m_pszURLString);
	}
	else
	{
		fnPrintf( 
			m_pHRequest, 
			"<HEAD><TITLE>gv_FlmSysData.pFileHashTbl</TITLE>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");
	
	// Insert a new table into the page to display the gv_FlmSysData fields
	fnPrintf( m_pHRequest, "<body>\n");

	printTableStart( "File Hash Table", 1, 100);
	printTableEnd();

	if (gv_FlmSysData.pFileHashTbl == NULL)
	{
		fnPrintf( m_pHRequest, "<CENTER>No File Hash Table entries exist.  "
								  "Please ensure that a database has been opened."
								  "</CENTER>\n");
	}
	else
	{

		for (pFileHashTbl = gv_FlmSysData.pFileHashTbl, iIndex=0;
			  iIndex < FILE_HASH_ENTRIES;
			  iIndex++)
		{
			buckets[iIndex] = (pFileHashTbl[iIndex].pFirstInBucket) ? 
																			TRUE : FALSE;
		}


		for (iIndex=0; iIndex < FILE_HASH_ENTRIES; iIndex++)
		{
			if (buckets[iIndex])
			{
				// We need to look for a next valid index to match with this one
				// for when the "Next Bucket" button is pressed.
				for (iNindex = (iIndex+1 < FILE_HASH_ENTRIES ? iIndex+1 : 0);
					  iNindex != iIndex ; )
				{
					if (buckets[iNindex])
					{
						break;
					}
					else
					{
						iNindex = (iNindex+1 < FILE_HASH_ENTRIES ? iNindex+1 : 0);
					}
				}

				// We will either have a valid next iIndex, or we will be pointing
				// to the same index, which means there is only one valid index
				// altogether.  So that is okay too.

				next[iIndex] = iNindex;

			}
		}

		//  Let's check to make sure we actually got at least one valid index.
		for (iIndex = 0; iIndex < FILE_HASH_ENTRIES; iIndex++)
		{
			if (buckets[iIndex])
			{
				found = TRUE;
				break;
			}
		}


		if (!found)
		{
			fnPrintf( m_pHRequest, "<CENTER>No File Hash Table entries exist.  "
									  "Please ensure that a database has been opened."
									  "</CENTER>\n");
		}
		else
		{


			fnPrintf( m_pHRequest, "<form name=\"HashSelection\" type=\"submit\" "
									  "method=\"get\" action=\"%s/FFile\">\n",
									  m_pszURLString);
			fnPrintf( m_pHRequest, "<CENTER>"
								"Only Buckets that are not empty are listed below."
								" You may use the \"Next Bucket\" button to choose "
								"the next available bucket to display, or you may "
								"select a specific bucket by selecting it from the "
								"list below. To display the chosen bucket, press "
								"the \"Submit\" button. </CENTER>\n");
			fnPrintf( m_pHRequest, "<BR>\n");
			fnPrintf( m_pHRequest, "<CENTER>\n");
			printButton( "Next Bucket", BT_Button, NULL, NULL,
				"ONCLICK='nextBucket(document.HashSelection.SelectionOption)'");
			fnPrintf( m_pHRequest, "&nbsp&nbspor select a specific bucket to "
									  "view&nbsp&nbsp\n");
			
			//  Only present the non-empty hash buckets...
			
			fnPrintf( m_pHRequest, "<SELECT NAME=\"SelectionOption\" onChange=\""
									  "this.form.Bucket.value = this.form.Selection"
									  "Option.options[this.form.SelectionOption."
									  "selectedIndex].text\">\n");
			
			for (iIndex = 0; iIndex < FILE_HASH_ENTRIES; iIndex++)
			{
				if (buckets[iIndex])
				{
					fnPrintf( m_pHRequest, "<OPTION> %d\n", iIndex);
				}
			}
			fnPrintf( m_pHRequest, "</SELECT>\n");
			
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");
			printButton( "Submit", BT_Submit);
			fnPrintf( m_pHRequest, "</CENTER>\n");
			fnPrintf( m_pHRequest, "<INPUT name=\"From\" type=hidden "
									  "value=\"FileHashTbl\"></INPUT>\n");
			
			// Pre-load the Bucket with the first non-empty index
			
			for (iIndex = 0; iIndex < FILE_HASH_ENTRIES; iIndex++)
			{
				if (buckets[iIndex])
				{
					fnPrintf( m_pHRequest, "<INPUT name=\"Bucket\" type=hidden "
											  "value=%d></INPUT>\n", iIndex);
					break;  // only do this once for the first valid entry...
				}
			}
			
			fnPrintf( m_pHRequest, "</form>\n");
			
			// Prepare the javascript functions...
			fnPrintf( m_pHRequest, "<SCRIPT>\n");
			fnPrintf( m_pHRequest, "function nextBucket(selectObj) {\n");
			fnPrintf( m_pHRequest, "var Bucket\n");
			fnPrintf( m_pHRequest, "switch (selectObj.selectedIndex) {\n");
			
			// Only identify the non-empty buckets...
			for (iSindex=0, iIndex = 0; iIndex < FILE_HASH_ENTRIES; iIndex++)
			{
				if (buckets[iIndex])
				{
					fnPrintf( m_pHRequest, "case %d:{\nBucket=%d\nselectObj."
											  "selectedIndex=%d\nbreak\n}\n",
												iSindex,
												next[iIndex],
												(next[iIndex] < iIndex) ? 0 :
													(next[iIndex] == iIndex ? iSindex :
																					  iSindex+1));
					iSindex++;
				}
			}
			
			
			fnPrintf( m_pHRequest, "default: break;}\n");
			fnPrintf( m_pHRequest, "document.HashSelection.Bucket.value = Bucket\n");
			fnPrintf( m_pHRequest, "}\n</SCRIPT>\n");
			
		}
	}

	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();

	return( rc);
}
