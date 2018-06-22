//-------------------------------------------------------------------------
// Desc:	Base class for monitoring code.
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

#define RESP_WRITE_BUF_SIZE		1024
#define FLM_SESSION_ID_NAME		"flmsessionid"

#define MAX_FIELD_SIZE(uiSize) \
		(uiSize > 100 ? 100 : (uiSize < 20 ? 20 : uiSize))

/****************************************************************************
 Desc:	Outputs a javascript function that, when called, causes a new
			window to open and a page to be displayed in it.
****************************************************************************/
void F_WebPage::popupFrame( void)
{
	fnPrintf( m_pHRequest, "<SCRIPT LANGUAGE=\"JavaScript\">\n");
	fnPrintf( m_pHRequest, "var windowW=600\n");		// wide
	fnPrintf( m_pHRequest, "var windowH=500\n");		// high
	fnPrintf( m_pHRequest, "var windowX = 100\n");	// from left
	fnPrintf( m_pHRequest, "var windowY = 100\n");	// from top
	fnPrintf( m_pHRequest, "var title =  \"Popup Window\"\n");
	fnPrintf( m_pHRequest, "var autoclose = false\n");
	fnPrintf( m_pHRequest, "function openPopup( urlPop) {\n");

	fnPrintf( m_pHRequest, "if (openPopup.arguments.length == 5)\n");
	fnPrintf( m_pHRequest, "{\nwindowW = openPopup.arguments[1];\n");
	fnPrintf( m_pHRequest, "windowH = openPopup.arguments[2];\n");
	fnPrintf( m_pHRequest, "windowX = openPopup.arguments[3];\n");
	fnPrintf( m_pHRequest, "windowY = openPopup.arguments[4];\n}\n");
	fnPrintf( m_pHRequest, "s = \"width=\"+windowW+\",height=\"+windowH;\n");
	fnPrintf( m_pHRequest,
				"NFW = window.open(urlPop,\"popFrameless\",\"scrollbars,resizable,\"+s);\n");
	fnPrintf( m_pHRequest, "NFW.blur();\n");
	fnPrintf( m_pHRequest, "window.focus();\n");
	fnPrintf( m_pHRequest, "NFW.resizeTo(windowW,windowH);\n");
	fnPrintf( m_pHRequest, "NFW.moveTo(windowX,windowY);\n");
	fnPrintf( m_pHRequest, "NFW.focus();\n");
	fnPrintf( m_pHRequest, "}\n</script>\n");
}

/******************************************************************************
Desc:	This method will extract the value of the parameter in the form of
		PARAMETER=VALUE
*******************************************************************************/
RCODE F_WebPage::ExtractParameter(
	FLMUINT			uiNumParams,
	const char **	ppszParams,
	const char *	pszParamName,
	FLMUINT			uiParamLen,
	char*				pszParamValue)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLoop;
	FLMUINT			uiParamNameLen;
	const char*		pszTemp;
	FLMBOOL			bFound = FALSE;

	uiParamNameLen = f_strlen( pszParamName);
	for (uiLoop = 0; uiLoop < uiNumParams; uiLoop++)
	{
		if( f_strncmp( (char*) ppszParams[uiLoop], pszParamName, uiParamNameLen) == 0 &&
			 (ppszParams[uiLoop][uiParamNameLen] == '\0' ||
			  ppszParams[uiLoop][uiParamNameLen] == '='))
		{
			pszTemp = &ppszParams[uiLoop][uiParamNameLen];
			if (*pszTemp == '=')
			{
				// Skip past the equal sign
				
				pszTemp++;
				
				f_strncpy( pszParamValue, pszTemp, uiParamLen - 1);

				// See if the param was too long to store

				if (f_strlen( pszTemp) >= uiParamLen)
				{
					pszParamValue[uiParamLen] = '\0';
					rc = RC_SET( FERR_MEM);
				}
			}
			else
			{
				*pszParamValue = 0;
			}

			bFound = TRUE;
			break;
		}
	}

	return (bFound ? rc : RC_SET( FERR_NOT_FOUND));
}

/****************************************************************************
Desc:	This method will detect the presence of a parameter in the form
		PARAMETER - no value will be checked.  A TRUE or FALSE value will
		be returned.
*****************************************************************************/
FLMBOOL F_WebPage::DetectParameter(
	FLMUINT			uiNumParams,
	const char **	ppszParams,
	const char *	pszParamName)
{
	for (FLMUINT uiLoop = 0; uiLoop < uiNumParams; uiLoop++)
	{
		if (f_strncmp( (char*) ppszParams[uiLoop], pszParamName,
						  f_strlen( (char*) pszParamName)) == 0)
		{
			return (TRUE);
		}
	}

	return (FALSE);
}

/****************************************************************************
Desc:	
*****************************************************************************/
RCODE F_WebPage::getDatabaseHandleParam(
	FLMUINT			uiNumParams,
	const char **	ppszParams,
	F_Session *		pFlmSession,
	HFDB*				phDb,
	char*				pszKey)
{
	RCODE		rc = FERR_OK;
	HFDB		hDb;
	char		szTmp[ 64];
	char *	pTmp;

	if (phDb)
	{
		*phDb = hDb = HFDB_NULL;
	}

	if (pszKey)
	{
		*pszKey = 0;
	}

	// Need to memset the first F_SESSION_DB_KEY_LEN bytes of szTmp
	// because the hash lookup algorithm expects the buffer to be padded
	// with zeros at the end of the key.

	f_memset( szTmp, 0, F_SESSION_DB_KEY_LEN);

	if (RC_BAD( ExtractParameter( uiNumParams, ppszParams, "dbhandle",
				  sizeof(szTmp), szTmp)))
	{
		pTmp = &szTmp[0];
		if (RC_BAD( getFormValueByName( "dbhandle", &pTmp, sizeof(szTmp), NULL)))
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}

	if (szTmp[0])
	{
		if (RC_BAD( rc = pFlmSession->getDbHandle( szTmp, &hDb)))
		{
			goto Exit;
		}

		if (pszKey)
		{
			f_memcpy( pszKey, szTmp, F_SESSION_DB_KEY_LEN);
		}
	}

	if (phDb)
	{
		*phDb = hDb;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Function to display the formatted time value in the form dd hh:mm:ss.ccc
*****************************************************************************/
void F_WebPage::FormatTime(
	FLMUINT		uiTimerUnits,
	char *		pszFormattedTime)
{
	FLMUINT		uiMilli;
	FLMUINT		uiSec;
	FLMUINT		uiMin;
	FLMUINT		uiHr;
	FLMUINT		uiDays;
	FLMUINT		uiTemp;

	// Initialize to NULL;

	pszFormattedTime[0] = '\0';

	// Convert the timer units to milliseconds

	uiMilli = FLM_TIMER_UNITS_TO_MILLI( uiTimerUnits);

	// Determine the number of days

	uiDays = uiMilli / 86400000;
	uiTemp = uiMilli % 86400000;

	// Now the hours

	uiHr = uiTemp / 3600000;
	uiTemp = uiTemp % 3600000;

	// Determine the minutes

	uiMin = uiTemp / 60000;
	uiTemp = uiTemp % 60000;

	// Determine seconds

	uiSec = uiTemp / 1000;

	// Determine the milliseconds

	uiMilli = uiTemp % 1000;

	// Put it all together - hh:mm:ss

	f_sprintf( (char *) pszFormattedTime, "%ld %2.2ld:%2.2ld:%2.2ld.%3.3ld",
				 uiDays, uiHr, uiMin, uiSec, uiMilli);
}

/****************************************************************************
 Desc:	Procedure to generate the HTML page that displays the usage statistics
			structure.
****************************************************************************/
RCODE F_WebPage::writeUsage(
	FLM_CACHE_USAGE*	pUsage,
	FLMBOOL				bRefresh,
	const char*			pszURL,
	const char*			pszTitle)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bHighlight = FALSE;
	char		szTemp[100];

	stdHdr();
	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Setup the page header & refresh control... Assuming the the first
	// parameter ?Usage is already contained in the pszURL string.

	if (bRefresh)
	{
		fnPrintf( m_pHRequest, "<HEAD>"
					"<META http-equiv=\"refresh\" content=\"5; url=%s%s&Refresh\">"
				"<TITLE>%s</TITLE>\n", m_pszURLString, pszURL, pszTitle);
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n<body>\n");

		f_sprintf( (char*) szTemp, "<A HREF=%s%s>Stop Auto-refresh</A>",
					 m_pszURLString, pszURL);
	}
	else
	{
		fnPrintf( m_pHRequest, "<HEAD><TITLE>%s</TITLE>\n", pszTitle);
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n<body>\n");

		f_sprintf( (char*) szTemp,
					 "<A HREF=%s%s&Refresh>Start Auto-refresh (5 sec.)</A>", 
					 m_pszURLString, pszURL);
	}

	// Begin the table

	printTableStart( (char*) pszTitle, 4);
	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=%s%s>Refresh</A>, ", m_pszURLString, pszURL);
	fnPrintf( m_pHRequest, "%s\n", szTemp);
	printColumnHeadingClose();
	printTableRowEnd();

	// Write out the table headings.

	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Byte Offset");
	printColumnHeading( "Value");
	printTableRowEnd();

	// uiMaxBytes

	printHTMLUint( (char*) "uiMaxBytes", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiMaxBytes, pUsage->uiMaxBytes,
					  (bHighlight = ~bHighlight));

	// uiTotalBytesAllocated

	printHTMLUint( (char*) "uiTotalBytesAllocated", (char*) "FLMUINT",
					  (void*) pUsage, (void*) &pUsage->uiTotalBytesAllocated,
					  pUsage->uiTotalBytesAllocated, (bHighlight = ~bHighlight));

	// uiCount

	printHTMLUint( (char*) "uiCount", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiCount, pUsage->uiCount,
					  (bHighlight = ~bHighlight));

	// uiOldVerCount

	printHTMLUint( (char*) "uiOldVerCount", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiOldVerCount, pUsage->uiOldVerCount,
					  (bHighlight = ~bHighlight));

	// uiOldVerBytes

	printHTMLUint( (char*) "uiOldVerBytes", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiOldVerBytes, pUsage->uiOldVerBytes,
					  (bHighlight = ~bHighlight));

	// uiCacheHits

	printHTMLUint( (char*) "uiCacheHits", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiCacheHits, pUsage->uiCacheHits,
					  (bHighlight = ~bHighlight));

	// uiCacheHitLooks

	printHTMLUint( (char*) "uiCacheHitLooks", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiCacheHitLooks, pUsage->uiCacheHitLooks,
					  (bHighlight = ~bHighlight));

	// uiCacheFaults

	printHTMLUint( (char*) "uiCacheFaults", (char*) "FLMUINT", (void*) pUsage,
					  (void*) &pUsage->uiCacheFaults, pUsage->uiCacheFaults,
					  (bHighlight = ~bHighlight));

	// uiCacheFaultLooks

	printHTMLUint( (char*) "uiCacheFaultLooks", (char*) "FLMUINT",
					  (void*) pUsage, (void*) &pUsage->uiCacheFaultLooks,
					  pUsage->uiCacheFaultLooks, (bHighlight = ~bHighlight));

	printTableEnd();

	fnPrintf( m_pHRequest, "<form>\n");
	fnPrintf( m_pHRequest,
				"<center><input type=\"button\" value=\"Close\" onClick=\"window.close()\"></center>\n");
	fnPrintf( m_pHRequest, "</form>\n");

	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();

	return (rc);
}

/*********************************************************************
Desc: This function prints a linkable field in HTML
*********************************************************************/
void F_WebPage::printHTMLLink(
	const char *		pszName,
	const char *		pszType,
	void *				pvBase,
	void *				pvAddress,
	void *				pvValue,
	const char *		pszLink,
	FLMBOOL				bHighlight)
{
	char	szAddress[20];
	char	szOffset[8];

	printOffset( pvBase, pvAddress, szOffset);
	printTableRowStart( bHighlight);
	fnPrintf( m_pHRequest, TD_s, szOffset);						// Field offset
	
	if (pvValue)
	{
		printAddress( pvValue, szAddress);
		fnPrintf( m_pHRequest, TD_a_s_s, pszLink, pszName);	// Link & Name
		fnPrintf( m_pHRequest, TD_s, pszType);						// Type
		fnPrintf( m_pHRequest, TD_a_s_s, pszLink, szAddress); // Link & Value
	}
	else
	{
		fnPrintf( m_pHRequest, TD_s, pszName);
		fnPrintf( m_pHRequest, TD_s, pszType);
		fnPrintf( m_pHRequest, TD_s, "Null");
	}

	printTableRowEnd();
}

/*********************************************************************
Desc: This function prints a text field in HTML
*********************************************************************/
void F_WebPage::printHTMLString(
	const char *		pszName,
	const char *		pszType,
	void *				pvBase,
	void *				pvAddress,
	const char *		pszValue,
	FLMBOOL				bHighlight)
{
	char	szOffset[8];

	printOffset( pvBase, pvAddress, szOffset);
	printTableRowStart( bHighlight);
	fnPrintf( m_pHRequest, TD_s, szOffset);	// Field offset
	fnPrintf( m_pHRequest, TD_s, pszName);		// Name
	fnPrintf( m_pHRequest, TD_s, pszType);		// Type
	fnPrintf( m_pHRequest, TD_s, pszValue);	// Value
	printTableRowEnd();
}

/*********************************************************************
Desc: This function prints a unsigned long (FLMUINT) field in HTML
*********************************************************************/
void F_WebPage::printHTMLUint(
	const char *		pszName,
	const char *		pszType,
	void *				pvBase,
	void *				pvAddress,
	FLMUINT				uiValue,
	FLMBOOL				bHighlight)
{
	char	szOffset[8];

	printOffset( pvBase, pvAddress, szOffset);
	printTableRowStart( bHighlight);
	fnPrintf( m_pHRequest, TD_s, szOffset);	// Field offset
	fnPrintf( m_pHRequest, TD_s, pszName);		// Name
	fnPrintf( m_pHRequest, TD_s, pszType);		// Type
	fnPrintf( m_pHRequest, TD_ui, uiValue);	// Value
	printTableRowEnd();
}

/*********************************************************************
Desc: This function prints a signed long (FLMINT) field in HTML
*********************************************************************/
void F_WebPage::printHTMLInt(
	const char *		pszName,
	const char *		pszType,
	void *				pvBase,
	void *				pvAddress,
	FLMINT				iValue,
	FLMBOOL				bHighlight)
{
	char	szOffset[8];

	printOffset( pvBase, pvAddress, szOffset);
	printTableRowStart( bHighlight);
	fnPrintf( m_pHRequest, TD_s, szOffset);	// Field offset
	fnPrintf( m_pHRequest, TD_s, pszName);		// Name
	fnPrintf( m_pHRequest, TD_s, pszType);		// Type
	fnPrintf( m_pHRequest, TD_i, iValue);		// Value
	printTableRowEnd();
}

/*********************************************************************
Desc: This function prints a unsigned long field in HTML
*********************************************************************/
void F_WebPage::printHTMLUlong(
	const char *		pszName,
	const char *		pszType,
	void *				pvBase,
	void *				pvAddress,
	unsigned long		luValue,
	FLMBOOL				bHighlight)
{
	char	szOffset[8];

	printOffset( pvBase, pvAddress, szOffset);
	printTableRowStart( bHighlight);
	fnPrintf( m_pHRequest, TD_s, szOffset);	// Field offset
	fnPrintf( m_pHRequest, TD_s, pszName);		// Name
	fnPrintf( m_pHRequest, TD_s, pszType);		// Type
	fnPrintf( m_pHRequest, TD_lu, luValue);	// Value
	printTableRowEnd();
}

/*********************************************************************
Desc: This function takes the name of a form field, and returns a
		pointer to it.  This will allocate the buffer returned.  The
		calling function is responsible for freeing that buffer.
*********************************************************************/
RCODE F_WebPage::getFormValueByName(
	const char *		pszValueTag,
	char **			 	ppszBuf,
	FLMUINT				uiBufLen,
	FLMUINT *			puiDataLen)
{
	RCODE			rc = FERR_OK;
	char			szTag[128];
	char *		pszValue;
	FLMUINT		uiLen;
	FLMBOOL		bFreeFormData = FALSE;
	FLMBOOL		bFreeUserData = FALSE;

	if (puiDataLen)
	{
		*puiDataLen = 0;
	}

#ifdef FLM_DEBUG
	if (!uiBufLen)
	{
		flmAssert( ppszBuf && *ppszBuf == NULL);
	}
#endif

	if (f_strlen( pszValueTag) + 1 >= sizeof(szTag))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	f_sprintf( (char*) szTag, "%s=", pszValueTag);

	if (!m_pszFormData)
	{
		char *		pszContentLength;
		FLMUINT	uiContentLength;

		// First we need to determine how much form data there is.

		if ((pszContentLength = (char*) fnReqHdrValue( "Content-Length")) == NULL)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}

		if ((uiContentLength = f_atoi( pszContentLength)) == 0)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}

		// Now allocate a buffer to hold the form data

		if (RC_BAD( rc = f_alloc( uiContentLength + 1, &m_pszFormData)))
		{
			goto Exit;
		}

		bFreeFormData = TRUE;

		if (fnRecvBuffer( m_pszFormData, (FLMSIZET *) &uiContentLength) != 0)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		m_pszFormData[uiContentLength] = 0;
		bFreeFormData = FALSE;
	}

	// Now, parse through the buffer until we find the field we are
	// looking for. The data is in the form name=value:name=value...

	if ((pszValue = f_strstr( m_pszFormData, szTag)) != NULL)
	{
		pszValue += f_strlen( szTag);
		for (uiLen = 0;
			  pszValue[uiLen] && pszValue[uiLen] != ':' && pszValue[uiLen] != '&';
			  uiLen++);

		if (ppszBuf)
		{
			if (!uiBufLen)
			{
				uiBufLen = uiLen + 1;
				bFreeUserData = TRUE;
				*ppszBuf = NULL;
				if (RC_BAD( rc = f_alloc( uiBufLen, ppszBuf)))
				{
					goto Exit;
				}
			}

			if (uiLen >= uiBufLen)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			f_memcpy( *ppszBuf, pszValue, uiLen);
			(*ppszBuf)[uiLen] = 0;
			bFreeUserData = FALSE;
		}

		if (puiDataLen)
		{
			*puiDataLen = uiLen + 1;
		}

		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	if (bFreeFormData)
	{
		f_free( &m_pszFormData);
	}

	if (bFreeUserData && *ppszBuf)
	{
		f_free( ppszBuf);
	}

	return (rc);
}

/****************************************************************************
Desc: Prints the standard style sheet
****************************************************************************/
void F_WebPage::printStyle(void)
{
	fnPrintf( m_pHRequest,
				"<link REL=stylesheet TYPE=text/css HREF=%s/staticfile/style.css>\n",
				m_pszURLString);
}

/****************************************************************************
Desc: Outputs a column heading using elements from the standard style sheet
****************************************************************************/
void F_WebPage::printColumnHeading(
	const char *			pszHeading,
	JustificationType 	eJustification,
	const char *			pszBackground,
	FLMUINT					uiColSpan,
	FLMUINT					uiRowSpan,
	FLMBOOL					bClose,
	FLMUINT					uiWidth)
{
	fnPrintf( m_pHRequest,
				"<td class=\"tablecolumnhead1\" colspan=%u rowspan=%u", 
					(unsigned) uiColSpan,
					(unsigned) uiRowSpan);

	if (uiWidth)
	{
		fnPrintf( m_pHRequest, " width=\"%u%%\"", (unsigned) uiWidth);
	}

	if (pszBackground)
	{
		fnPrintf( m_pHRequest, " bgColor=\"%s\"", pszBackground);
	}

	if (eJustification == JUSTIFY_CENTER)
	{
		fnPrintf( m_pHRequest, " align=\"center\"");
	}
	else if (eJustification == JUSTIFY_RIGHT)
	{
		fnPrintf( m_pHRequest, " align=\"right\"");
	}
	else
	{
		fnPrintf( m_pHRequest, " align=\"left\"");
	}

	fnPrintf( m_pHRequest, ">\n");
	
	if (pszHeading)
	{
		printEncodedString( pszHeading);
	}

	if (bClose)
	{
		fnPrintf( m_pHRequest, "</td>\n");
	}
}

/****************************************************************************
Desc: Closes a column heading
****************************************************************************/
void F_WebPage::printColumnHeadingClose(void)
{
	fnPrintf( m_pHRequest, "</td>\n");
}

/****************************************************************************
Desc: Encodes a string for rendering in an HTML page or for inclusion in
		an URL
****************************************************************************/
void F_WebPage::printEncodedString(
	const char *		pszString,
	FStringEncodeType eEncodeType,
	FLMBOOL				bMapSlashes)
{
	char	ucChar;

	while ((ucChar = *pszString) != 0)
	{
		if ((ucChar >= '0' && ucChar <= '9') ||
			 (ucChar >= 'A' && ucChar <= 'Z') ||
			 (ucChar >= 'a' && ucChar <= 'z') ||
			 ucChar == '_' ||
			 (
				 eEncodeType == URL_PATH_ENCODING &&
			 (ucChar == '.' || (bMapSlashes && (ucChar == '/' || ucChar == '\\')))
		 ))
		{
			if (ucChar == '\\')
			{
				ucChar = '/';
			}

			fnPrintf( m_pHRequest, "%c", ucChar);
		}
		else if (eEncodeType == URL_PATH_ENCODING)
		{
			fnPrintf( m_pHRequest, "%%%02X", (unsigned) ucChar);
		}
		else if (eEncodeType == URL_QUERY_ENCODING)
		{
			if (ucChar == ' ')
			{
				ucChar = '+';
			}

			fnPrintf( m_pHRequest, "%%%02X", (unsigned) ucChar);
		}
		else	// HTML encoding
		{
			fnPrintf( m_pHRequest, "&#%u;", (unsigned) ucChar);
		}

		pszString++;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printDocStart(
	const char *	pszTitle,
	FLMBOOL			bPrintTitle,
	FLMBOOL			bStdHeader,
	const char *	pszBackground)
{
	if (bStdHeader)
	{
		stdHdr();
	}

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	printRecordStyle();
	printStyle();
	fnPrintf( m_pHRequest, "<title>Database iMonitor - ");
	printEncodedString( pszTitle);
	fnPrintf( m_pHRequest, "</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body bgcolor=\"%s\">\n",
				pszBackground ? pszBackground : "white");

	if (bPrintTitle)
	{
		printTableStart( pszTitle, 1);
		printTableEnd();
		fnPrintf( m_pHRequest, "<BR>\n");
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printDocEnd(void)
{
	fnPrintf( m_pHRequest, "</body>\n");
	fnPrintf( m_pHRequest, "</html>\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printMenuReload(void)
{
	fnPrintf( m_pHRequest, "<script>parent.Menu.location.reload( true)\n");
	fnPrintf( m_pHRequest, "</script>\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableStart(
	const char*		pszTitle,
	FLMUINT			uiColumns,
	FLMUINT			uiWidthFactor)
{
	fnPrintf( m_pHRequest, "<table border=0 cellpadding=2 cellspacing=0");
	if (uiWidthFactor)
	{
		fnPrintf( m_pHRequest, " width=%u%%", (unsigned) uiWidthFactor);
	}

	fnPrintf( m_pHRequest, ">\n");

	if (pszTitle)
	{
		printTableRowStart();
		fnPrintf( m_pHRequest, "<td colspan=%u class=\"tablehead1\"",
					(unsigned) uiColumns);
		fnPrintf( m_pHRequest, ">\n");
		printEncodedString( pszTitle);
		fnPrintf( m_pHRequest, "</td>");
		printTableRowEnd();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableEnd(void)
{
	fnPrintf( m_pHRequest, "</table>\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableRowStart(
	FLMBOOL	bHighlight)
{
	fnPrintf( m_pHRequest, "<tr class=\"mediumtext\"");
	if (bHighlight)
	{
		fnPrintf( m_pHRequest, " bgColor=\"%s\"", FLM_IMON_COLOR_PUTTY_2);
	}

	fnPrintf( m_pHRequest, ">\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableRowEnd(void)
{
	fnPrintf( m_pHRequest, "</tr>\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableDataStart(
	FLMBOOL				bNoWrap,
	JustificationType eJustification,
	FLMUINT				uiWidth)
{
	fnPrintf( m_pHRequest, "<td");

	if (uiWidth)
	{
		fnPrintf( m_pHRequest, " width=%u%%", (unsigned) uiWidth);
	}

	if (bNoWrap)
	{
		fnPrintf( m_pHRequest, " nowrap");
	}

	if (eJustification == JUSTIFY_CENTER)
	{
		fnPrintf( m_pHRequest, " align=\"center\"");
	}
	else if (eJustification == JUSTIFY_RIGHT)
	{
		fnPrintf( m_pHRequest, " align=\"right\"");
	}
	else
	{
		fnPrintf( m_pHRequest, " align=\"left\"");
	}

	fnPrintf( m_pHRequest, ">\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableDataEnd(void)
{
	fnPrintf( m_pHRequest, "</td>\n");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printTableDataEmpty(void)
{
	fnPrintf( m_pHRequest, "&nbsp;");
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printErrorPage(
	RCODE				rc,
	FLMBOOL			bStdHeader,
	const char *	pszWhat)
{
	printDocStart( "Error", TRUE, bStdHeader);

	fnPrintf( m_pHRequest, "<center><h2>\n");
	fnPrintf( m_pHRequest, "%s<br>%s (0x%04X).\n", pszWhat, FlmErrorString( rc),
				(unsigned) rc);
	fnPrintf( m_pHRequest, "</h2></center>\n");

	printDocEnd();
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printErrorPage(
	const char *	pszErrStr,
	const char *	pszErrStr2,
	FLMBOOL			bStdHeader)
{
	printDocStart( "Error", TRUE, bStdHeader);

	fnPrintf( m_pHRequest, "<center><h2>\n");
	fnPrintf( m_pHRequest, "%s\n", pszErrStr);
	if (pszErrStr2 && *pszErrStr2)
	{
		fnPrintf( m_pHRequest, "<BR>%s\n", pszErrStr2);
	}

	fnPrintf( m_pHRequest, "</h2></center>\n");
	
	printDocEnd();
}

/****************************************************************************
Desc:	Start an input form
****************************************************************************/
void F_WebPage::printStartInputForm(
	const char *	pszFormName,
	const char *	pszPage,
	FLMUINT			uiFormValue)
{
	fnPrintf( m_pHRequest,
				"<form name=\"%s\" type=\"submit\" method=\"get\" action=\"%s/%s\">\n"
				"<input name=\"Action\" type=\"hidden\" value=\"%u\">\n", 
				pszFormName, m_pszURLString, pszPage, (unsigned) uiFormValue);
}

/****************************************************************************
Desc:	End an input form
****************************************************************************/
void F_WebPage::printEndInputForm(void)
{
	fnPrintf( m_pHRequest, "</form>");
}

/****************************************************************************
Desc:	Generic function to output HTML that gererates a button
****************************************************************************/
void F_WebPage::printButton(
	const char *	pszContents,
	ButtonTypes		eBType,
	const char *	pszName,
	const char *	pszValue,
	const char *	pszExtra,
	FLMBOOL			bDisabled,
	FLMBYTE			ucAccessKey,
	FLMUINT			uiTabIndex)
{
	fnPrintf( m_pHRequest, "<BUTTON TYPE=");

	switch (eBType)
	{
		case BT_Submit:
			fnPrintf( m_pHRequest, "submit");
			break;
		case BT_Reset:
			fnPrintf( m_pHRequest, "reset");
			break;
		case BT_Button:
			fnPrintf( m_pHRequest, "button");
			break;
		default:
			flmAssert( 0);
	}

	if (pszName && pszName[0])
	{
		fnPrintf( m_pHRequest, " NAME=%s", pszName);
	}

	if (pszValue && pszValue[0])
	{
		fnPrintf( m_pHRequest, " VALUE=%s", pszValue);
	}

	if (bDisabled)
	{
		fnPrintf( m_pHRequest, " DISABLED");
	}

	if (ucAccessKey != '\0')
	{
		fnPrintf( m_pHRequest, " ACCESSKEY=%c", ucAccessKey);
	}

	if (uiTabIndex)
	{
		fnPrintf( m_pHRequest, " uiTabIndex=%d", uiTabIndex);
	}

	if (pszExtra)
	{
		fnPrintf( m_pHRequest, " %s ", pszExtra);
	}

	fnPrintf( m_pHRequest, ">%s</BUTTON>\n", 
				(char*) (pszContents ? pszContents : ""));
}


/****************************************************************************
Desc:	Format and output date.
****************************************************************************/
void F_WebPage::printDate(
	FLMUINT		uiGMTTime,
	char *		pszBuffer)
{
	F_TMSTAMP		timeStamp;
	FLMUINT			uiLocalTime;
	char *			pszAmPm;
	const char *	pszMonth;

	uiLocalTime = (FLMUINT) (uiGMTTime - f_timeGetLocalOffset());
	f_timeSecondsToDate( uiLocalTime, &timeStamp);

	pszAmPm = (char*) ((timeStamp.hour >= 12) ? (char*) "pm" : (char*) "am");
	if (timeStamp.hour > 12)
	{
		timeStamp.hour -= 12;
	}

	if (timeStamp.hour == 0)
	{
		timeStamp.hour = 12;
	}

	switch (timeStamp.month)
	{
		case 0:
			pszMonth = "Jan";
			break;
		case 1:
			pszMonth = "Feb";
			break;
		case 2:
			pszMonth = "Mar";
			break;
		case 3:
			pszMonth = "Apr";
			break;
		case 4:
			pszMonth = "May";
			break;
		case 5:
			pszMonth = "Jun";
			break;
		case 6:
			pszMonth = "Jul";
			break;
		case 7:
			pszMonth = "Aug";
			break;
		case 8:
			pszMonth = "Sep";
			break;
		case 9:
			pszMonth = "Oct";
			break;
		case 10:
			pszMonth = "Nov";
			break;
		default:
		case 11:
			pszMonth = "Dec";
			break;
	}

	if (pszBuffer != NULL)
	{
		f_sprintf( (char*) pszBuffer, "%s %u, %u  %u:%02u:%02u %s", pszMonth,
					 (unsigned) timeStamp.day, (unsigned) timeStamp.year,
					 (unsigned) timeStamp.hour, (unsigned) timeStamp.minute,
					 (unsigned) timeStamp.second, pszAmPm);
	}
	else
	{
		fnPrintf( m_pHRequest, "%s %u, %u  %u:%02u:%02u %s", pszMonth,
					(unsigned) timeStamp.day, (unsigned) timeStamp.year,
					(unsigned) timeStamp.hour, (unsigned) timeStamp.minute,
					(unsigned) timeStamp.second, pszAmPm);
	}
}

/****************************************************************************
Desc:	Outputs a Yes or No value based on the passed-in boolean
****************************************************************************/
void F_WebPage::printYesNo(
	FLMBOOL	bYes)
{
	fnPrintf( m_pHRequest, "%s", bYes ? "Yes" : "No");
}

/****************************************************************************
Desc:	Outputs a number with commas, for easier reading.
****************************************************************************/
void F_WebPage::printCommaNumText(
	FLMUINT64	ui64Num)
{
	FLMUINT		uiTerm;
	FLMUINT64	ui64Divisor = 1;
	FLMBOOL		bFirstPass = TRUE;

	while ((FLMUINT64) (ui64Num / (ui64Divisor * (FLMUINT64) 1000)))
	{
		ui64Divisor *= 1000;
	}

	while (ui64Divisor)
	{
		uiTerm = (FLMUINT) (ui64Num / ui64Divisor);
		ui64Num -= ((FLMUINT64) uiTerm) * ui64Divisor;
		
		if (bFirstPass)
		{
			fnPrintf( m_pHRequest, "%u", (unsigned) uiTerm);
			bFirstPass = FALSE;
		}
		else
		{
			fnPrintf( m_pHRequest, "%03u", (unsigned) uiTerm);
		}

		if ((ui64Divisor /= (FLMUINT64) 1000) > (FLMUINT64) 0)
		{
			fnPrintf( m_pHRequest, ",");
		}
	}
}

/****************************************************************************
Desc:	Outputs a number with commas, for easier reading.
****************************************************************************/
void F_WebPage::printCommaNum(
	FLMUINT64			ui64Num,
	JustificationType eJustify,
	FLMBOOL				bChangedValue)
{
	printTableDataStart( TRUE, eJustify);
	if (bChangedValue)
	{
		fnPrintf( m_pHRequest, "<font color=red>");
	}

	printCommaNumText( ui64Num);

	if (bChangedValue)
	{
		fnPrintf( m_pHRequest, "</font>");
	}

	printTableDataEnd();
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_WebPage::acquireSession(void)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bHttpSessionMutexLocked = FALSE;
	FLMUINT		uiSize;
	void *		pvHttpSession = NULL;
	char			szSessionKey[F_SESSION_KEY_LEN];

	m_pFlmSession = NULL;

	if (!gv_FlmSysData.HttpConfigParms.fnAcquireSession)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	if ((pvHttpSession = fnAcquireSession()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hHttpSessionMutex);
	bHttpSessionMutexLocked = TRUE;

	uiSize = sizeof(szSessionKey);
	if (fnGetSessionValue( pvHttpSession, FLM_SESSION_ID_NAME,
								 (void*) szSessionKey, (FLMSIZET *) &uiSize) != 0)
	{
CreateSession:

		if (RC_BAD( rc = gv_FlmSysData.pSessionMgr->createSession( &m_pFlmSession)))
		{
			goto Exit;
		}

		fnSetSessionValue( pvHttpSession, FLM_SESSION_ID_NAME,
								m_pFlmSession->getKey(), sizeof(szSessionKey));
	}
	else
	{
		if (RC_BAD( rc = gv_FlmSysData.pSessionMgr->getSession( szSessionKey,
					  &m_pFlmSession)))
		{
			if (rc == FERR_NOT_FOUND)
			{
				goto CreateSession;
			}
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		if (m_pFlmSession)
		{
			releaseSession();
		}
	}

	if (pvHttpSession)
	{
		fnReleaseSession( pvHttpSession);
	}

	if (bHttpSessionMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hHttpSessionMutex);
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::releaseSession(void)
{
	if (m_pFlmSession)
	{
		gv_FlmSysData.pSessionMgr->releaseSession( &m_pFlmSession);
		m_pFlmSession = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_WebPage::printSpaces(
	FLMUINT	uiCount)
{
	while (uiCount--)
	{
		fnPrintf( m_pHRequest, "&nbsp;");
	}
}

/****************************************************************************
Desc:	Outputs elapsed milliseconds as seconds.milli.  The optional parameter
		pszBuffer will cause the time to be written to pszBuffer instead of the
		web page.  That way it can be incorporated into more complex structures
		if desired.
****************************************************************************/
void F_WebPage::printElapTime(
	FLMUINT64			ui64ElapTime,
	char*					pszBuffer,
	JustificationType eJustify,
	FLMBOOL				bTimeIsMilli)
{
	FLMUINT	uiHours;
	FLMUINT	uiMinutes;
	FLMUINT	uiSeconds;
	FLMUINT	uiMilli = 0;

	if (bTimeIsMilli)
	{
		uiHours = (FLMUINT) (ui64ElapTime / (FLMUINT64) (1000 * 3600));
		uiMinutes = (FLMUINT) ((ui64ElapTime / (FLMUINT64) (1000 * 60)) % (FLMUINT64) 60);
		uiSeconds = (FLMUINT) ((ui64ElapTime / (FLMUINT64) 1000) % (FLMUINT64) 60);
		uiMilli = (FLMUINT) (ui64ElapTime % (FLMUINT64) 1000);
	}
	else
	{
		uiHours = (FLMUINT) (ui64ElapTime / (FLMUINT64) 3600);
		uiMinutes = (FLMUINT) ((ui64ElapTime / (FLMUINT64) 60) % (FLMUINT64) 60);
		uiSeconds = (FLMUINT) (ui64ElapTime % (FLMUINT64) 60);
	}

	if (!pszBuffer)
	{
		printTableDataStart( TRUE, eJustify);
	}

	if (pszBuffer)
	{
		f_sprintf( (char*) pszBuffer, "%02u:%02u:%02u", (unsigned) uiHours,
					 (unsigned) uiMinutes, (unsigned) uiSeconds);
	}
	else
	{
		fnPrintf( m_pHRequest, "%02u:%02u:%02u", (unsigned) uiHours,
					(unsigned) uiMinutes, (unsigned) uiSeconds);
	}

	if (bTimeIsMilli)
	{
		if (pszBuffer)
		{
			char	szTemp[5];

			f_sprintf( szTemp, ".%03u", (unsigned) uiMilli);
			f_strncat( (char*) pszBuffer, szTemp, 4);
		}
		else
		{
			fnPrintf( m_pHRequest, ".%03u", (unsigned) uiMilli);
		}
	}

	if (!pszBuffer)
	{
		printTableDataEnd();
	}
}

/****************************************************************************
Desc:	This function will output a form in an already existing page that will
		present a formatted display of the record passed in.  An optional parameter
		bReadOnly will determine if the form should allow editing of the record
		or not.  The default value is TRUE (No editing capability).  A null may
		be passed in for the FlmRecord pointer, in which case an empty display
		will be created.  Normally, a bReadOnly value of false would accompany
		a null record so that a record can be created.  The printRecordStyle() 
		must be called in the header section of the page prior to calling this
		function.  Multiple calls may be made to display multiple records per page.
		The puiContext parameter must be initialized to zero before the first 
		(and possibly the only) call, otherwise the scripts need to run this page
		will not be loaded.
****************************************************************************/
void F_WebPage::printRecord(
	const char *		pszDbKey,
	FlmRecord *			pRec,
	F_NameTable *		pNameTable,
	FLMUINT *			puiContext,
	FLMBOOL				bReadOnly,
	FLMUINT				uiSelectedField,
	FLMUINT				uiFlags)
{
	#define SP	"&nbsp;"

	FLMBOOL			bEmpty = FALSE;
	FLMUINT			uiContainer;
	FLMUINT			uiDrn;
	void *			pvField;
	FLMUINT			uiFieldCounter;
	FLMUINT			uiFldCnt;
	FLMUINT			uiTagNum;
	FLMUINT			uiLevel;
	FLMUINT			uiType;
	FLMUINT			uiContext = 0;
	char				szNameBuf[128];

	flmAssert( pNameTable);

	if (puiContext)
	{
		uiContext = *puiContext;
		(*puiContext)++;
	}

	// See if we need to write out the scripts

	if (uiContext == 0)
	{
		printRecordScripts();
	}

	if (!pRec)
	{
		bEmpty = TRUE;
		uiDrn = 0;
		uiContainer = 0;
	}
	else
	{

		// Get the Drn & Container

		uiDrn = pRec->getID();
		uiContainer = pRec->getContainerID();
	}

	// Count the fields

	uiFldCnt = 0;
	if (pRec != NULL)
	{
		pvField = pRec->root();
		while (pvField)
		{
			pvField = pRec->next( pvField);
			uiFldCnt++;
		}
	}

	// Begin the form that displays the record data (if any)

	fnPrintf( m_pHRequest,
				"<form name=\"Record%d\" method=\"post\" action=\"%s/ProcessRecord\">\n",
			uiContext, gv_FlmSysData.HttpConfigParms.pszURLString);
	printHiddenField( "ReadOnly", (char*) (bReadOnly ? "TRUE" : "FALSE"));

	if (pszDbKey)
	{
		printHiddenField( "dbhandle", (char*) (pszDbKey));
	}

	printHiddenField( (char*) "Action", "none");
	printHiddenField( (char*) "FieldLevel", (FLMUINT) 0);
	printHiddenField( (char*) "FieldNumber", (FLMUINT) 0);
	printHiddenField( (char*) "FieldCount", uiFldCnt);

	// Print out the block that displays the DRN and Container list

	fnPrintf( m_pHRequest, "<div id=\"recordselect\">\n");
	fnPrintf( m_pHRequest,
				"<table border=\"1\" cellpadding=\"0\" cellspacing=\"0\" width=\"170\" "
				"frame=\"box\">\n");
	fnPrintf( m_pHRequest, "<tr>\n<td align=right>DRN&nbsp;<input name=\"DRN\" "
				"type=\"text\" value=\"%u\" size=10 maxlength=20 %s>&nbsp;</td>\n</tr>\n",
				uiDrn, pszDbKey == NULL ? "disabled" : "");

	if (pszDbKey != NULL)
	{
		fnPrintf( m_pHRequest, "<tr>\n<td align=right>Flags&nbsp;\n");
		printRetrievalFlagsPulldown( uiFlags);
		fnPrintf( m_pHRequest, "</td>\n</tr>\n");
	}

	fnPrintf( m_pHRequest, "<tr>\n<td align=right>Container&nbsp;\n");
	if (pszDbKey != NULL)
	{
		printContainerPulldown( pNameTable, uiContainer);
	}
	else
	{
		fnPrintf( m_pHRequest,
					"<input name=\"container\" type=\"text\" value=\"%u\" "
					"size=10 maxlength=20 disabled>&nbsp;</td>\n</tr>\n", uiContainer);
	}

	fnPrintf( m_pHRequest, "</td>\n</tr>\n");

	if (pszDbKey != NULL)
	{

		// Print out the field list drop down box.

		fnPrintf( m_pHRequest, "<tr>\n<td align=left nowrap>Field list&nbsp;");
		printFieldPulldown( pNameTable, uiSelectedField);
		fnPrintf( m_pHRequest, "</td>\n</tr>\n");

		// Print out the Add, Modify, Delete action buttons.

		fnPrintf( m_pHRequest,
					"<tr><td align=left><input type=\"button\" value=\"new record\" "
					"onClick=\"doNewRecord(document.Record%u)\">", uiContext);
		if (pRec != NULL)
		{
			if (!bReadOnly)
			{
				fnPrintf( m_pHRequest, "<input type=\"button\" value=\"add\" "
							"onClick=\"doAddRecord(document.Record%u)\">", uiContext);
				fnPrintf( m_pHRequest, "<input type=\"button\" value=\"modify\" "
							"onClick=\"doModRecord(document.Record%u)\">", uiContext);
			}

			fnPrintf( m_pHRequest, "<input type=\"button\" value=\"delete\" "
						"onClick=\"doDelRecord(document.Record%u)\">", uiContext);
		}

		fnPrintf( m_pHRequest, "<input type=\"button\" value=\"retrieve\" "
					"onClick=\"doRetrieveRecord(document.Record%u)\">", uiContext);
		if (pRec != NULL && bReadOnly)
		{
			fnPrintf( m_pHRequest, "<input type=\"button\" value=\"edit record\" "
						"onClick=\"doEdit(document.Record%u)\">", uiContext);
		}

		fnPrintf( m_pHRequest, "</td></tr>\n");
	}

	fnPrintf( m_pHRequest, "</table>\n</div>\n");

	// Print out the record fields (if there are any)

	if (pRec != NULL)
	{
		fnPrintf( m_pHRequest, "<div id=\"fieldcontrol\">\n");
		if (!bReadOnly)
		{
			fnPrintf( m_pHRequest, "<input type=\"button\" value=\"ins-c\" "
						"onClick=\"doInsertChild(document.Record%u)\">", uiContext);
			if (uiFldCnt > 1)
			{
				fnPrintf( m_pHRequest, "<input type=\"button\" value=\"ins-s\" "
							"onClick=\"doInsertSibling(document.Record%u)\">", uiContext);
				fnPrintf( m_pHRequest, "<input type=\"button\" value=\"copy\" "
							"onClick=\"doCopy(document.Record%u)\">", uiContext);
				fnPrintf( m_pHRequest, "<input type=\"button\" value=\"clip\" "
							"onClick=\"doClip(document.Record%u)\">\n", uiContext);
			}
		}

		fnPrintf( m_pHRequest, "<pre>\n");

		// Now for the actual data. Start with the root field.

		pvField = pRec->root();

		uiFieldCounter = 0;

		while (pvField)
		{
			uiTagNum = pRec->getFieldID( pvField);
			uiLevel = pRec->getLevel( pvField);
			uiType = pRec->getDataType( pvField);

			if (uiLevel != 0 && !bReadOnly)
			{
				fnPrintf( m_pHRequest,
							"<input name=\"radioSel\" type=\"radio\" value=\"%u\" "
							"onClick=\"setFieldLevel(document.Record%u, %u, %u)\">",
						uiFieldCounter, uiContext, uiFieldCounter, uiLevel);
			}

			pNameTable->getFromTagNum( uiTagNum, NULL, szNameBuf, sizeof(szNameBuf));
			printSpaces( uiLevel + 5);

			fnPrintf( m_pHRequest, "%s<font color=black>%d</font>%s%s%s", SP,
						uiLevel, SP, szNameBuf, SP);

			if (pRec->getDataLength( pvField))
			{
				switch (uiType)
				{
					case FLM_TEXT_TYPE:
						printTextField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
					case FLM_NUMBER_TYPE:
						printNumberField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
					case FLM_BINARY_TYPE:
						printBinaryField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
					case FLM_CONTEXT_TYPE:
						printContextField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
					case FLM_BLOB_TYPE:
						printBlobField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
					default:
						printDefaultField( pRec, pvField, uiFieldCounter, bReadOnly);
						break;
				}
			}
			else if (!bReadOnly)
			{
				fnPrintf( m_pHRequest,
							"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"\" size=\"%d\">",
						uiFieldCounter, MAX_FIELD_SIZE( 0));
			}

			// Print the hidden field Ids

			printFieldIds( uiFieldCounter, uiLevel, uiType, uiTagNum);
			fnPrintf( m_pHRequest, "\n");

			pvField = pRec->next( pvField);
			uiFieldCounter++;
		}

		fnPrintf( m_pHRequest, "</pre>\n</div>\n<hr width=75%%>\n");
	}

	fnPrintf( m_pHRequest, "</form>\n");

	return;
}


/****************************************************************************
Desc:	Prints out a style sheet specific to displaying records.
****************************************************************************/
void F_WebPage::printRecordStyle(void)
{
	fnPrintf( m_pHRequest, "<style media=\"screen\" type=\"text/css\"><!--\n");
	fnPrintf( m_pHRequest, "#recordselect { background-color: #e8e8e8; "
				"position: relative; left: 15px; width: 150px; visibility: visible}\n");
	fnPrintf( m_pHRequest, "#fieldlist { position: relative; width: 200px; "
				"visibility: visible}\n");
	fnPrintf( m_pHRequest, "#fieldcontrol { background-color: #e5e5e5; "
				"color: #357977; font-weight: bold; position: relative; top: 5px; "
			"left: 15px; visibility: visible}\n");
	fnPrintf( m_pHRequest, ".fieldclass { color: #0db3ae }\n--></style>\n");
}

/****************************************************************************
Desc:	Prints out the required scripts for displaying and updating records.
****************************************************************************/
void F_WebPage::printRecordScripts( void)
{
	fnPrintf( m_pHRequest, "<script><!-- Hide script from old browsers\n");
	fnPrintf( m_pHRequest, "function doEdit(myForm)\n"
		"{\n"
			"myForm.Action.value=\"Retrieve\";\n"
			"myForm.ReadOnly.value = \"FALSE\";\n"
			"myForm.submit();\n"
		"}\n");
	fnPrintf( m_pHRequest, "function confirmAction(action)\n"
		"{\n"
			"return confirm(\"Are you sure you want to \" + action + \" this record?\");\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doAddRecord(myForm)\n"
		"{\n"
			"if (confirmAction(\"Add\"))\n"
			"{\n"
				"myForm.Action.value=\"Add\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doNewRecord(myForm)\n"
		"{\n"
			"myForm.ReadOnly.value=\"FALSE\";\n"
			"myForm.Action.value=\"New\";\n"
			"myForm.submit();\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doModRecord(myForm)\n"
		"{\n"
			"if (confirmAction(\"Modify\"))\n"
			"{\n"
				"myForm.Action.value=\"Modify\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doDelRecord(myForm)\n"
		"{\n"
			"if (confirmAction(\"Delete\"))\n"
			"{\n"
				"myForm.Action.value=\"Delete\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doRetrieveRecord(myForm)\n"
		"{\n"
			"myForm.ReadOnly.value=\"TRUE\";\n"
			"myForm.Action.value=\"Retrieve\";\n"
			"myForm.submit();\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doInsertSibling(myForm)\n"
		"{\n"
			"if (validateFieldLevel(myForm))\n"
			"{\n"
				"myForm.Action.value=\"InsertSibling\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doInsertChild(myForm)\n"
		"{\n"
			"if (validateFieldLevel(myForm))\n"
			"{\n"
				"myForm.Action.value=\"InsertChild\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doCopy(myForm)\n"
		"{\n"
			"if (validateFieldLevel(myForm))\n"
			"{\n"
				"myForm.Action.value=\"Copy\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function doClip(myForm)\n"
		"{\n"
			"if (validateFieldLevel(myForm))\n"
			"{\n"
				"myForm.Action.value=\"Clip\";\n"
				"myForm.submit();\n"
			"}\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "function validateFieldLevel(myForm)\n"
		"{\n"
			"if (myForm.FieldCount.value>1)\n"
			"{\n"
				"if ((myForm.FieldLevel.value==0)||(myForm.FieldNumber.value==0))\n"
				"{\n"
					"alert(\"You must select a field radio button\");\n"
					"return false;\n"
				"}\n"
			"}\n"
			"return true;\n"
		"}\n");

	fnPrintf( m_pHRequest, "function setFieldLevel(myForm,field,level)\n"
		"{\n"
			"myForm.FieldLevel.value=level;\n"
			"myForm.FieldNumber.value=field;\n"
		"}\n");
	
	fnPrintf( m_pHRequest, "// End hiding here -->\n</script>\n");
}

/****************************************************************************
Desc:	Prints out a hidden field with a character string value
****************************************************************************/
void F_WebPage::printHiddenField(
	const char *		pszName,
	const char *		pszValue)
{
	fnPrintf( m_pHRequest, "<input name=\"%s\" type=\"hidden\" value=\"%s\">",
				pszName, pszValue);
}

//
/****************************************************************************
Desc:	Prints out a hidden with an unsigned long value
****************************************************************************/
void F_WebPage::printHiddenField(
	const char *		pszName,
	FLMUINT				uiValue)
{
	fnPrintf( m_pHRequest, "<input name=\"%s\" type=\"hidden\" value=\"%u\">",
				pszName, (unsigned) uiValue);
}

//
/****************************************************************************
Desc:	Prints out the value for the field, assuming the field is a text field.
****************************************************************************/
void F_WebPage::printTextField(
	FlmRecord *		pRec,
	void *			pvField,
	FLMUINT			uiFieldCounter,
	FLMBOOL			bReadOnly)
{
	RCODE					rc = FERR_OK;
	FLMUNICODE *		puzBuf = NULL;
	FLMUNICODE *		puzTmp = NULL;
	F_DynamicBuffer *	pBuffer = NULL;
	FLMUINT				uiLen;

	if (RC_BAD( rc = pRec->getUnicodeLength( pvField, &uiLen)))
	{
		fnPrintf( m_pHRequest,
					"** Error retrieving Unicode field length (Return Code = 0x%04X, %s) **",
				(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	// The length returned does not allow for 2 NULL terminators. We must
	// allow for them when allocating a buffer.

	uiLen += 2;
	if (RC_BAD( rc = f_alloc( uiLen, &puzBuf)))
	{
		fnPrintf( m_pHRequest,
					"** Error allocating memory buffer (Return Code = 0x%04X, %s) **",
					(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	if (RC_BAD( rc = pRec->getUnicode( pvField, puzBuf, &uiLen)))
	{
		fnPrintf( m_pHRequest,
					"** Error retrieving Unicode field (Return Code = 0x%04X, %s) **",
					(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	puzTmp = puzBuf;
	if ((pBuffer = f_new F_DynamicBuffer) == NULL)
	{
		fnPrintf( m_pHRequest, "** Error allocating memory **");
		goto Exit;
	}

	// Start the text field if not read only mode.

	if (!bReadOnly)
	{
		fnPrintf( m_pHRequest,
					"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"",
				uiFieldCounter);
	}
	else
	{
		fnPrintf( m_pHRequest, "<font color=\"0db3ae\">");
	}

	while (*puzTmp)
	{

		// Check for ASCII characters

		if ((*puzTmp >= 32) && (*puzTmp <= 126))
		{
			if (RC_BAD( rc = pBuffer->addChar( (char) *puzTmp)))
			{
				fnPrintf( m_pHRequest,
							"** Error adding Unicode character to buffer (Return Code = 0x%04X, %s) **",
						(unsigned) rc, FlmErrorString( rc));
				goto Exit;
			}
		}
		else
		{

			// Treat as though these are NON-ASCII. They will be printed in
			// the form ~[0x ]

			char	szTempBuff[20];

			f_sprintf( szTempBuff, "~[0x%04X]", (unsigned) (*puzTmp));

			if (RC_BAD( rc = pBuffer->addString( szTempBuff)))
			{
				fnPrintf( m_pHRequest,
							"** Error formatting Unicode string (Return Code = 0x%04X, %s) **",
						(unsigned) rc, FlmErrorString( rc));
				goto Exit;
			}
		}

		// We are attempting to not let our buffer get any larger than the
		// Http stack buffer size. We don't really know what the limit is,
		// but we are using what seems to reasonable to us...

		if ((pBuffer->getBufferSize() + 9) >= RESP_WRITE_BUF_SIZE)
		{
			fnPrintf( m_pHRequest, "%s", pBuffer->printBuffer());
			pBuffer->reset();
		}

		puzTmp++;
	}

	if (bReadOnly)
	{
		fnPrintf( m_pHRequest, "%s</font>", pBuffer->printBuffer());
	}
	else
	{
		fnPrintf( m_pHRequest, "%s\" size=\"%d\">", pBuffer->printBuffer(),
					MAX_FIELD_SIZE( uiLen));
	}

Exit:

	if (puzBuf)
	{
		f_free( &puzBuf);
	}

	if (pBuffer)
	{
		pBuffer->Release();
	}
}

//
/****************************************************************************
Desc:	Prints out the value for the field, assuming the field is a number field.
****************************************************************************/
void F_WebPage::printNumberField(
	FlmRecord *		pRec,
	void *			pvField,
	FLMUINT			uiFieldCounter,
	FLMBOOL			bReadOnly)
{
	RCODE			rc = FERR_OK;
	FLMINT		iVal;
	FLMUINT		uiVal;

	if (RC_BAD( rc = pRec->getUINT( pvField, &uiVal)))
	{
		if (RC_OK( rc = pRec->getINT( pvField, &iVal)))
		{
			if (bReadOnly)
			{
				fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%d</font>", (int) iVal);
			}
			else
			{
				fnPrintf( m_pHRequest,
							"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"%d\" size=\"%d\">",
						uiFieldCounter, (int) iVal, MAX_FIELD_SIZE( 0));
			}
		}
		else
		{
			fnPrintf( m_pHRequest,
						"** Error retrieving number field (Return Code = 0x%04X, %s)**\n",
					(unsigned) rc, FlmErrorString( rc));
		}
	}
	else
	{
		if (bReadOnly)
		{
			fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%lu</font>",
						(unsigned long) uiVal);
		}
		else
		{
			fnPrintf( m_pHRequest,
						"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"%lu\" size=\"20\">",
					uiFieldCounter, (unsigned long) uiVal);
		}
	}
}

/****************************************************************************
Desc:	Prints out the value for the field, assuming the field is a binary field.
****************************************************************************/
void F_WebPage::printBinaryField(
	FlmRecord *		pRec,
	void *			pvField,
	FLMUINT			uiFieldCounter,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucBuf = NULL;
	FLMBYTE *		pszTmpBuf = NULL;
	FLMBYTE *		pszTmp = NULL;
	FLMUINT			uiLoop;
	FLMUINT			uiLen;
	FLMUINT			uiBufLen;

	uiLen = pRec->getDataLength( pvField);

	if (RC_BAD( rc = f_alloc( uiLen, &pucBuf)))
	{
		fnPrintf( m_pHRequest,
					"** Error occured allocating memory to retrieve binary field (Return Code = 0x%04X, %s) **\n",
				(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	if (RC_BAD( rc = pRec->getBinary( pvField, pucBuf, &uiLen)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			fnPrintf( m_pHRequest,
						"** Error occured retrieving binary field (Return Code = 0x%04X, %s) **\n",
					(unsigned) rc, FlmErrorString( rc));
			goto Exit;
		}
	}

	if (RC_BAD( rc = f_alloc( RESP_WRITE_BUF_SIZE + 1, &pszTmpBuf)))
	{
		fnPrintf( m_pHRequest,
					"** Error occured allocating memory to format binary field (Return Code = 0x%04X, %s) **\n",
				(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	if (!bReadOnly)
	{
		fnPrintf( m_pHRequest,
					"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"",
				uiFieldCounter);
	}
	else
	{
		fnPrintf( m_pHRequest, "<font color=\"0db3ae\">");
	}

	// Scan through the binary data, present all data as Hex.

	for (pszTmp = pszTmpBuf, uiLoop = 0, uiBufLen = 0; uiLoop < uiLen; uiLoop++)
	{
		if (uiLoop)
		{
			*pszTmp++ = ' ';
			uiBufLen++;
		}

		f_sprintf( (char*) pszTmp, "%2.2X", (unsigned) pucBuf[uiLoop]);
		
		pszTmp += 2;
		uiBufLen += 2;
		
		if ((uiBufLen + 3) >= RESP_WRITE_BUF_SIZE)
		{

			// Flush the current buffer

			*pszTmp = '\0';
			fnPrintf( m_pHRequest, "%s", pszTmpBuf);
			pszTmp = pszTmpBuf;
			uiBufLen = 0;
		}
	}

	*pszTmp = '\0';

	if (bReadOnly)
	{
		fnPrintf( m_pHRequest, "%s</font>", pszTmpBuf);
	}
	else
	{
		fnPrintf( m_pHRequest, "%s\" size=\"%d\">", pszTmpBuf,
					MAX_FIELD_SIZE( uiLen * 3));
	}

Exit:

	if (pucBuf)
	{
		f_free( &pucBuf);
	}

	if (pszTmpBuf)
	{
		f_free( &pszTmpBuf);
	}
}

/****************************************************************************
Desc:	Prints out the value for the field, assuming the field is a context field.
****************************************************************************/
void F_WebPage::printContextField(
	FlmRecord *	pRec,
	void *		pvField,
	FLMUINT		uiFieldCounter,
	FLMBOOL		bReadOnly)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiRecPointer;

	if (RC_OK( rc = pRec->getRecPointer( pvField, &uiRecPointer)))
	{
		if (bReadOnly)
		{
			fnPrintf( m_pHRequest, "<font color=\"0db3ae\">%lu</font>",
						(unsigned long) uiRecPointer);
		}
		else
		{
			fnPrintf( m_pHRequest,
						"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" "
						"value=\"%lu\" size=\"d\">", uiFieldCounter,
						(unsigned long) uiRecPointer, MAX_FIELD_SIZE( 0));
		}
	}
	else
	{
		fnPrintf( m_pHRequest,
					"** Error retrieving context field (Return Code = 0x%04X, %s) **",
					(unsigned) rc, FlmErrorString( rc));
	}
}

/****************************************************************************
Desc:	Prints out the value for the field, assuming the field is a blob field.
****************************************************************************/
void F_WebPage::printBlobField(
	FlmRecord *	pRec,
	void *		pvField,
	FLMUINT		uiFieldCounter,
	FLMBOOL		bReadOnly)
{
	RCODE			rc = FERR_OK;
	FlmBlob *	pBlob = NULL;
	char			szPath[F_PATH_MAX_SIZE];
	FLMUINT		uiLen;

	if (RC_BAD( rc = pRec->getBlob( pvField, &pBlob)))
	{
		fnPrintf( m_pHRequest,
					"** Failed to retrieve Blob object (Return Code = 0x%04X, %s) **",
					(unsigned long) rc, FlmErrorString( rc));
		goto Exit;
	}

	uiLen = ((FlmBlobImp *) pBlob)->getDataLength();
	if (uiLen == 0)
	{
		if (!bReadOnly)
		{
			fnPrintf( m_pHRequest,
						"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" "
						"value=\"\" size=\"%d\">", uiFieldCounter,
						MAX_FIELD_SIZE( 0));
		}

		goto Exit;
	}

	if (RC_BAD( rc = pBlob->buildFileName( szPath)))
	{
		fnPrintf( m_pHRequest,
					"** Failed to retrieve Blob filename (Return Code = 0x%04X, %s) **",
					(unsigned) rc, FlmErrorString( rc));
		goto Exit;
	}

	if (bReadOnly)
	{
		fnPrintf( m_pHRequest, "<font color=\"0db3ae\">");
		printEncodedString( szPath, HTML_ENCODING);
		fnPrintf( m_pHRequest, "</font>");
	}
	else
	{
		fnPrintf( m_pHRequest,
					"<input class=\"fieldclass\" name=\"field%d\" type=\"text\" value=\"",
				uiFieldCounter);
		printEncodedString( szPath, HTML_ENCODING);
		fnPrintf( m_pHRequest, "\" size=\"20\">");
	}

Exit:

	if (pBlob)
	{
		pBlob->Release();
	}
}

/****************************************************************************
Desc:	Prints out a string identifying this as a default field - error condition.
****************************************************************************/
void F_WebPage::printDefaultField(
	FlmRecord *,
	void *,
	FLMUINT,
	FLMBOOL)
{
	fnPrintf( m_pHRequest, "<font color=\"0db3ae\">**Default Field**</font>");
}

/****************************************************************************
Desc:	Prints out the hidden field identifiers
****************************************************************************/
void F_WebPage::printFieldIds(
	FLMUINT	uiFieldCounter,
	FLMUINT	uiFieldLevel,
	FLMUINT	uiType,
	FLMUINT	uiTagNum)
{
	char	szTmp[20];

	f_sprintf( szTmp, "fieldLevel%u", (unsigned) uiFieldCounter);
	printHiddenField( szTmp, uiFieldLevel);
	f_sprintf( szTmp, "fieldType%u", (unsigned) uiFieldCounter);
	printHiddenField( szTmp, uiType);
	f_sprintf( szTmp, "fieldTag%u", (unsigned) uiFieldCounter);
	printHiddenField( szTmp, uiTagNum);
}

/****************************************************************************
Desc:	Prints a table listing the fields of the supplied Log Headers.  Any of
		the log header pointers may be null, in which case a series of blank
		entries will be created in the table for that entry.
****************************************************************************/
void F_WebPage::printLogHeaders(
	FLMBYTE *		pucLastCommitted,
	FLMBYTE *		pucCheckpoint,
	FLMBYTE *		pucUncommitted)
{
	FLMBOOL	bHighlight = FALSE;

	// Start the table and headings...

	printTableStart( NULL, 5, 100);

	printTableRowStart( FALSE);
	printColumnHeading( "Offset (hex)");
	printColumnHeading( "Field");
	printColumnHeading( "Last Committed");
	printColumnHeading( "Checkpoint");
	printColumnHeading( "Uncommitted");
	printTableRowEnd();

	// Fill in the table here. LOG_RFL_FILE_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_FILE_NUM);
	fnPrintf( m_pHRequest, "<td>Current RFL file</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_FILE_NUM);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_FILE_NUM);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_FILE_NUM);
	printTableRowEnd();

	// LOG_RFL_LAST_TRANS_OFFSET

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_LAST_TRANS_OFFSET);
	fnPrintf( m_pHRequest, "<td>Current RFL offset</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_LAST_TRANS_OFFSET);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_LAST_TRANS_OFFSET);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_LAST_TRANS_OFFSET);
	printTableRowEnd();

	// LOG_RFL_LAST_CP_FILE_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_LAST_CP_FILE_NUM);
	fnPrintf( m_pHRequest, "<td>Last CP RFL file</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_LAST_CP_FILE_NUM);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_LAST_CP_FILE_NUM);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_LAST_CP_FILE_NUM);
	printTableRowEnd();

	// LOG_RFL_LAST_CP_OFFSET

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_LAST_CP_OFFSET);
	fnPrintf( m_pHRequest, "<td>Last CP RFL offset</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_LAST_CP_OFFSET);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_LAST_CP_OFFSET);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_LAST_CP_OFFSET);
	printTableRowEnd();

	// LOG_ROLLBACK_EOF

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_ROLLBACK_EOF);
	fnPrintf( m_pHRequest, "<td>End of file</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_ROLLBACK_EOF);
	printLogFileEntryUD( pucCheckpoint, LOG_ROLLBACK_EOF);
	printLogFileEntryUD( pucUncommitted, LOG_ROLLBACK_EOF);
	printTableRowEnd();

	// LOG_INC_BACKUP_SEQ_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_INC_BACKUP_SEQ_NUM);
	fnPrintf( m_pHRequest, "<td>Incremental backup sequence number</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_INC_BACKUP_SEQ_NUM);
	printLogFileEntryUD( pucCheckpoint, LOG_INC_BACKUP_SEQ_NUM);
	printLogFileEntryUD( pucUncommitted, LOG_INC_BACKUP_SEQ_NUM);
	printTableRowEnd();

	// LOG_CURR_TRANS_ID

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_CURR_TRANS_ID);
	fnPrintf( m_pHRequest, "<td>Transaction ID</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_CURR_TRANS_ID);
	printLogFileEntryUD( pucCheckpoint, LOG_CURR_TRANS_ID);
	printLogFileEntryUD( pucUncommitted, LOG_CURR_TRANS_ID);
	printTableRowEnd();

	// LOG_COMMIT_COUNT

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_COMMIT_COUNT);
	fnPrintf( m_pHRequest, "<td>Commit count</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_COMMIT_COUNT);
	printLogFileEntryUD( pucCheckpoint, LOG_COMMIT_COUNT);
	printLogFileEntryUD( pucUncommitted, LOG_COMMIT_COUNT);
	printTableRowEnd();

	// LOG_PL_FIRST_CP_BLOCK_ADDR

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_PL_FIRST_CP_BLOCK_ADDR);
	fnPrintf( m_pHRequest, "<td>First CP block address</td>");
	printLogFileEntryUDX( pucLastCommitted, LOG_PL_FIRST_CP_BLOCK_ADDR);
	printLogFileEntryUDX( pucCheckpoint, LOG_PL_FIRST_CP_BLOCK_ADDR);
	printLogFileEntryUDX( pucUncommitted, LOG_PL_FIRST_CP_BLOCK_ADDR);
	printTableRowEnd();

	// LOG_LAST_RFL_FILE_DELETED

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LAST_RFL_FILE_DELETED);
	fnPrintf( m_pHRequest, "<td>Last RFL file deleted</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_LAST_RFL_FILE_DELETED);
	printLogFileEntryUD( pucCheckpoint, LOG_LAST_RFL_FILE_DELETED);
	printLogFileEntryUD( pucUncommitted, LOG_LAST_RFL_FILE_DELETED);
	printTableRowEnd();

	// LOG_RFL_MIN_FILE_SIZE

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_MIN_FILE_SIZE);
	fnPrintf( m_pHRequest, "<td>Minimum RFL file size</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_MIN_FILE_SIZE);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_MIN_FILE_SIZE);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_MIN_FILE_SIZE);
	printTableRowEnd();

	// LOG_HDR_CHECKSUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_HDR_CHECKSUM);
	fnPrintf( m_pHRequest, "<td>Header checksum</td>");
	printLogFileEntryUW( pucLastCommitted, LOG_HDR_CHECKSUM);
	printLogFileEntryUW( pucCheckpoint, LOG_HDR_CHECKSUM);
	printLogFileEntryUW( pucUncommitted, LOG_HDR_CHECKSUM);
	printTableRowEnd();

	// LOG_FLAIM_VERSION

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_FLAIM_VERSION);
	fnPrintf( m_pHRequest, "<td>Flaim version</td>");
	printLogFileEntryUW( pucLastCommitted, LOG_FLAIM_VERSION);
	printLogFileEntryUW( pucCheckpoint, LOG_FLAIM_VERSION);
	printLogFileEntryUW( pucUncommitted, LOG_FLAIM_VERSION);
	printTableRowEnd();

	// LOG_LAST_BACKUP_TRANS_ID

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LAST_BACKUP_TRANS_ID);
	fnPrintf( m_pHRequest, "<td>Last backup trans ID</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_LAST_BACKUP_TRANS_ID);
	printLogFileEntryUD( pucCheckpoint, LOG_LAST_BACKUP_TRANS_ID);
	printLogFileEntryUD( pucUncommitted, LOG_LAST_BACKUP_TRANS_ID);
	printTableRowEnd();

	// LOG_BLK_CHG_SINCE_BACKUP

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_BLK_CHG_SINCE_BACKUP);
	fnPrintf( m_pHRequest, "<td>Blocks changed since backup</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_BLK_CHG_SINCE_BACKUP);
	printLogFileEntryUD( pucCheckpoint, LOG_BLK_CHG_SINCE_BACKUP);
	printLogFileEntryUD( pucUncommitted, LOG_BLK_CHG_SINCE_BACKUP);
	printTableRowEnd();

	// LOG_LAST_CP_TRANS_ID

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LAST_CP_TRANS_ID);
	fnPrintf( m_pHRequest, "<td>Last CP trans ID</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_LAST_CP_TRANS_ID);
	printLogFileEntryUD( pucCheckpoint, LOG_LAST_CP_TRANS_ID);
	printLogFileEntryUD( pucUncommitted, LOG_LAST_CP_TRANS_ID);
	printTableRowEnd();

	// LOG_PF_FIRST_BACKCHAIN

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_PF_FIRST_BACKCHAIN);
	fnPrintf( m_pHRequest, "<td>Backchain block address</td>");
	if (pucLastCommitted &&
		 FB2UD( &pucLastCommitted[LOG_PF_FIRST_BACKCHAIN]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucLastCommitted, LOG_PF_FIRST_BACKCHAIN);
	}

	if (pucCheckpoint &&
		 FB2UD( &pucCheckpoint[LOG_PF_FIRST_BACKCHAIN]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucCheckpoint, LOG_PF_FIRST_BACKCHAIN);
	}

	if (pucUncommitted &&
		 FB2UD( &pucUncommitted[LOG_PF_FIRST_BACKCHAIN]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucUncommitted, LOG_PF_FIRST_BACKCHAIN);
	}

	printTableRowEnd();

	// LOG_PF_AVAIL_BLKS

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_PF_AVAIL_BLKS);
	fnPrintf( m_pHRequest, "<td>Available blocks</td>");
	if (pucLastCommitted &&
		 FB2UD( &pucLastCommitted[LOG_PF_AVAIL_BLKS]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucLastCommitted, LOG_PF_AVAIL_BLKS);
	}

	if (pucCheckpoint && FB2UD( &pucCheckpoint[LOG_PF_AVAIL_BLKS]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucCheckpoint, LOG_PF_AVAIL_BLKS);
	}

	if (pucUncommitted && FB2UD( &pucUncommitted[LOG_PF_AVAIL_BLKS]) == BT_END)
	{
		fnPrintf( m_pHRequest, "<td>none</td>");
	}
	else
	{
		printLogFileEntryUDX( pucUncommitted, LOG_PF_AVAIL_BLKS);
	}

	printTableRowEnd();

	// LOG_LOGICAL_EOF

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LOGICAL_EOF);
	fnPrintf( m_pHRequest, "<td>Logical EOF</td>");
	printLogFileEntryUD_X( pucLastCommitted, LOG_LOGICAL_EOF);
	printLogFileEntryUD_X( pucCheckpoint, LOG_LOGICAL_EOF);
	printLogFileEntryUD_X( pucUncommitted, LOG_LOGICAL_EOF);
	printTableRowEnd();

	// LOG_LAST_RFL_COMMIT_ID

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LAST_RFL_COMMIT_ID);
	fnPrintf( m_pHRequest, "<td>Last RFL commit ID</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_LAST_RFL_COMMIT_ID);
	printLogFileEntryUD( pucCheckpoint, LOG_LAST_RFL_COMMIT_ID);
	printLogFileEntryUD( pucUncommitted, LOG_LAST_RFL_COMMIT_ID);
	printTableRowEnd();

	// LOG_KEEP_ABORTED_TRANS_IN_RFL

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_KEEP_ABORTED_TRANS_IN_RFL);
	fnPrintf( m_pHRequest, "<td>Keep aborted trans in RFL</td>");
	printLogFileEntryBool( pucLastCommitted, LOG_KEEP_ABORTED_TRANS_IN_RFL);
	printLogFileEntryBool( pucCheckpoint, LOG_KEEP_ABORTED_TRANS_IN_RFL);
	printLogFileEntryBool( pucUncommitted, LOG_KEEP_ABORTED_TRANS_IN_RFL);
	printTableRowEnd();

	// LOG_PF_FIRST_BC_CNT

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_PF_FIRST_BC_CNT);
	fnPrintf( m_pHRequest, "<td>First BC count</td>");
	printLogFileEntryUC( pucLastCommitted, LOG_PF_FIRST_BC_CNT);
	printLogFileEntryUC( pucCheckpoint, LOG_PF_FIRST_BC_CNT);
	printLogFileEntryUC( pucUncommitted, LOG_PF_FIRST_BC_CNT);
	printTableRowEnd();

	// LOG_KEEP_RFL_FILES

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_KEEP_RFL_FILES);
	fnPrintf( m_pHRequest, "<td>Keep RFL files</td>");
	printLogFileEntryBool( pucLastCommitted, LOG_KEEP_RFL_FILES);
	printLogFileEntryBool( pucCheckpoint, LOG_KEEP_RFL_FILES);
	printLogFileEntryBool( pucUncommitted, LOG_KEEP_RFL_FILES);
	printTableRowEnd();

	// LOG_AUTO_TURN_OFF_KEEP_RFL

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_AUTO_TURN_OFF_KEEP_RFL);
	fnPrintf( m_pHRequest, "<td>Auto turn off keep RFL</td>");
	printLogFileEntryBool( pucLastCommitted, LOG_AUTO_TURN_OFF_KEEP_RFL);
	printLogFileEntryBool( pucCheckpoint, LOG_AUTO_TURN_OFF_KEEP_RFL);
	printLogFileEntryBool( pucUncommitted, LOG_AUTO_TURN_OFF_KEEP_RFL);
	printTableRowEnd();

	// LOG_PF_NUM_AVAIL_BLKS

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_PF_NUM_AVAIL_BLKS);
	fnPrintf( m_pHRequest, "<td>Avail Blocks</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_PF_NUM_AVAIL_BLKS);
	printLogFileEntryUD( pucCheckpoint, LOG_PF_NUM_AVAIL_BLKS);
	printLogFileEntryUD( pucUncommitted, LOG_PF_NUM_AVAIL_BLKS);
	printTableRowEnd();

	// LOG_RFL_MAX_FILE_SIZE

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_MAX_FILE_SIZE);
	fnPrintf( m_pHRequest, "<td>Max file size</td>");
	printLogFileEntryUD( pucLastCommitted, LOG_RFL_MAX_FILE_SIZE);
	printLogFileEntryUD( pucCheckpoint, LOG_RFL_MAX_FILE_SIZE);
	printLogFileEntryUD( pucUncommitted, LOG_RFL_MAX_FILE_SIZE);
	printTableRowEnd();

	// LOG_DB_SERIAL_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_DB_SERIAL_NUM);
	fnPrintf( m_pHRequest, "<td>DB serial number</td>");
	printSerialNum( pucLastCommitted ? &pucLastCommitted[LOG_DB_SERIAL_NUM] : NULL);
	printSerialNum( pucCheckpoint ? &pucCheckpoint[LOG_DB_SERIAL_NUM] : NULL);
	printSerialNum( pucUncommitted ? &pucUncommitted[LOG_DB_SERIAL_NUM] : NULL);
	printTableRowEnd();

	// LOG_LAST_TRANS_RFL_SERIAL_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_LAST_TRANS_RFL_SERIAL_NUM);
	fnPrintf( m_pHRequest, "<td>Last Trans RFL serial number</td>");
	printSerialNum( pucLastCommitted ? &pucLastCommitted[LOG_LAST_TRANS_RFL_SERIAL_NUM] :
							NULL);
	printSerialNum( pucCheckpoint ? &pucCheckpoint[LOG_LAST_TRANS_RFL_SERIAL_NUM] : NULL);
	printSerialNum( pucUncommitted ? &pucUncommitted[LOG_LAST_TRANS_RFL_SERIAL_NUM] : NULL);
	printTableRowEnd();

	// LOG_RFL_NEXT_SERIAL_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_RFL_NEXT_SERIAL_NUM);
	fnPrintf( m_pHRequest, "<td>Next RFL serial number</td>");
	printSerialNum( pucLastCommitted ? &pucLastCommitted[LOG_RFL_NEXT_SERIAL_NUM] : NULL);
	printSerialNum( pucCheckpoint ? &pucCheckpoint[LOG_RFL_NEXT_SERIAL_NUM] : NULL);
	printSerialNum( pucUncommitted ? &pucUncommitted[LOG_RFL_NEXT_SERIAL_NUM] : NULL);
	printTableRowEnd();

	// LOG_INC_BACKUP_SERIAL_NUM

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_INC_BACKUP_SERIAL_NUM);
	fnPrintf( m_pHRequest, "<td>Incremental backup serial number</td>");
	printSerialNum( pucLastCommitted ? &pucLastCommitted[LOG_INC_BACKUP_SERIAL_NUM] : NULL);
	printSerialNum( pucCheckpoint ? &pucCheckpoint[LOG_INC_BACKUP_SERIAL_NUM] : NULL);
	printSerialNum( pucUncommitted ? &pucUncommitted[LOG_INC_BACKUP_SERIAL_NUM] : NULL);
	printTableRowEnd();

	// LOG_MAX_FILE_SIZE

	printTableRowStart( bHighlight = !bHighlight);
	fnPrintf( m_pHRequest, "<td>0x%X</td>", LOG_MAX_FILE_SIZE);
	fnPrintf( m_pHRequest, "<td>Maximum file size (64K units)</td>");
	printLogFileEntryUW( pucLastCommitted, LOG_MAX_FILE_SIZE);
	printLogFileEntryUW( pucCheckpoint, LOG_MAX_FILE_SIZE);
	printLogFileEntryUW( pucUncommitted, LOG_MAX_FILE_SIZE);
	printTableRowEnd();

	printTableEnd();
}

/*******************************************************************
Desc:
********************************************************************/
void F_WebPage::printSerialNum(
	FLMBYTE *		pucSerialNum)
{
	if (pucSerialNum)
	{
		printTableDataStart( FALSE, JUSTIFY_LEFT);

		// fnPrintf( m_pHRequest, "0x");

		for (int iLoop = 0; iLoop < F_SERIAL_NUM_SIZE; iLoop++)
		{
			fnPrintf( m_pHRequest, "%02X ", pucSerialNum[iLoop]);
		}

		printTableDataEnd();
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as unsigned 4 digit hex minimum.
********************************************************************/
void F_WebPage::printLogFileEntryUDX(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		fnPrintf( m_pHRequest, "<td>0x%04X</td>", FB2UD( &pucLog[uiOffset]));
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as unsigned decimal hex in parenthesis.
********************************************************************/
void F_WebPage::printLogFileEntryUD_X(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		printCommaNumText( (FLMUINT64) FB2UD( &pucLog[uiOffset]));
		fnPrintf( m_pHRequest, " (0x%X)", FB2UD( &pucLog[uiOffset]));
		printTableDataEnd();
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as unsigned decimal.
********************************************************************/
void F_WebPage::printLogFileEntryUD(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		printCommaNum( (FLMUINT64) FB2UD( &pucLog[uiOffset]), JUSTIFY_LEFT);
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as unsigned word.
********************************************************************/
void F_WebPage::printLogFileEntryUW(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		printCommaNum( (FLMUINT64) FB2UW( &pucLog[uiOffset]), JUSTIFY_LEFT);
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as unsigned char.
********************************************************************/
void F_WebPage::printLogFileEntryUC(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		fnPrintf( m_pHRequest, "<td>%u</td>", (unsigned char) pucLog[uiOffset]);
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}

/*******************************************************************
Desc: Print a table entry as yes or no (FLMBOOL)
********************************************************************/
void F_WebPage::printLogFileEntryBool(
	FLMBYTE *	pucLog,
	FLMUINT		uiOffset)
{
	if (pucLog)
	{
		printTableDataStart( TRUE, JUSTIFY_LEFT);
		printYesNo( (FLMBOOL) pucLog[uiOffset]);
		printTableDataEnd();
	}
	else
	{
		fnPrintf( m_pHRequest, "<td>-</td>");
	}
}
