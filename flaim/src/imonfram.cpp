//-------------------------------------------------------------------------
// Desc:	Class for displaying the framesets used by the monitoring code
//			to display web pages.
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

/*********************************************************
 Desc:	Return HTML code that defines the welcome page frames.
			There are two framesets.  The first has one frame
			that references "Header.htm".  The second frameset
			has two frames.  The first frame references
			"Nav.htm" and "Welcome.htm".  This class is invoked
			following a successful login.
 **********************************************************/
RCODE F_FrameMain::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM(uiNumParams);
	F_UNREFERENCED_PARM(ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	printStyle();
	fnPrintf( m_pHRequest, "<title>Database iMonitor</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest,
				 "<frameset rows=\"80,*\""
				 " framespacing=\"0\" border=\"0\" frameborder=\"0\">\n");
	fnPrintf( m_pHRequest,
				 "<frame name=\"Header\" SRC=\"%s/Header.htm\" "
				 "TITLE=\"Header\" border=0 frameborder=0 "
				 "marginwidth=0 marginheight=0 scrolling=\"no\">\n", m_pszURLString);
	fnPrintf( m_pHRequest, 
				 "<frameset cols=220,* "
				 "framespacing=0 border=0 frameborder=0>\n");
	fnPrintf( m_pHRequest,
				 "<frame name=\"Menu\" SRC=\"%s/Nav.htm\" "
				 "TITLE=\"Menu\" border=0 frameborder=0 framespacing=0 "
				 "marginwidth=0 marginheight=0 width=220>\n", m_pszURLString);
	fnPrintf( m_pHRequest,
				 "<frame name=\"Content\" SRC=\"%s/Welcome.htm\" "
				 "TITLE=\"Content\" border=0 frameborder=0 "
				 "marginwidth=0 marginheight=0>\n", m_pszURLString);
   fnPrintf( m_pHRequest, "</frameset>\n");
   fnPrintf( m_pHRequest, "</frameset>\n");
	fnPrintf( m_pHRequest, "</html>\n");

	fnEmit();
	return( rc);
}


/*********************************************************
 Desc:	Return HTML code that defines the Header.htm frame.
 **********************************************************/
RCODE F_FrameHeader::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM(uiNumParams);
	F_UNREFERENCED_PARM(ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	fnPrintf( m_pHRequest, "<style type=\"text/css\"><!--\n");
	fnPrintf( m_pHRequest, "#Headgraphic   { position: absolute; z-index: 0; top: 0px; left: 0px; width: 650px; visibility: visible }\n");
	fnPrintf( m_pHRequest, "#logo    { text-align: right; position: absolute; z-index: 1; top: 30px; left: 0px; width: 100%%; height: 22px; visibility: visible }\n");
	fnPrintf( m_pHRequest, "#title   { position: absolute; z-index: 1; top: 22px; left: 12px; width: 208px; visibility: visible }\n");
	fnPrintf( m_pHRequest, "body { background: white url(%s/staticfile/head_bg.gif) repeat-x 0%% 0%% }\n", m_pszURLString);
	fnPrintf( m_pHRequest, "-->\n");
	fnPrintf( m_pHRequest, "</style>\n");
	printStyle();

	fnPrintf( m_pHRequest, "<title>Header</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body marginwidth=\"0\" "
								  "marginheight=\"0\" leftmargin=\"0\" topmargin=\"0\">\n");

	fnPrintf( m_pHRequest, "<div id=\"Headgraphic\">\n");
	fnPrintf( m_pHRequest, "<img src=\"%s/staticfile/imonhdr.gif\" width=\"650\" height=\"59\" border=\"0\">\n", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div id=\"logo\">\n");
	fnPrintf( m_pHRequest, "<table border=\"0\" cellpadding=\"0\" cellspacing=\"0\" align=\"left\">\n");
	fnPrintf( m_pHRequest, "<tr>\n");
	fnPrintf( m_pHRequest, "<td width=\"450\" align=\"left\">\n");
	fnPrintf( m_pHRequest, "<img height=\"10\" width=\"600\" src=\"%s/staticfile/spacer.gif\" border=\"0\">", m_pszURLString);
	fnPrintf( m_pHRequest, "</td>\n");
	fnPrintf( m_pHRequest, "<td align=\"right\" width=\"100%%\">\n");
	fnPrintf( m_pHRequest, "<a href=\"http://www.novell.com/\" target=\"_blank\">\n");
	fnPrintf( m_pHRequest, "<img height=\"22\" width=\"100\" src=\"%s/staticfile/novlogo.gif\" border=\"0\" alt=\"Novell Home Page\">\n", m_pszURLString);
	fnPrintf( m_pHRequest, "</a>\n");
	fnPrintf( m_pHRequest, "</td>\n");
	fnPrintf( m_pHRequest, "</tr>\n");
	fnPrintf( m_pHRequest, "</table>\n");
	fnPrintf( m_pHRequest, "</div>\n");
	fnPrintf( m_pHRequest,
				 "<div id=\"title\" class=\"subtitle2\">Database <i>i</i>Monitor</div>\n");
	fnPrintf( m_pHRequest, "</body>\n");
	fnPrintf( m_pHRequest, "</html>\n");

	fnEmit();

	return( rc);
}

/*********************************************************
 Desc:	Return HTML code that defines the Nav.htm frame.
 *********************************************************/
RCODE F_FrameNav::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE						rc = FERR_OK;
	void *					pvSession = NULL;
	char						szValue[20];
	char						szGblPassword[20];
	char *					pszPassword = NULL;
	FLMUINT					uiSize;
	F_Session *				pFlmSession = m_pFlmSession;
	
	if (gv_FlmSysData.HttpConfigParms.fnAcquireSession)
	{
		pvSession = fnAcquireSession();
	}

	printDocStart( "Navigator", FALSE, TRUE, FLM_IMON_COLOR_PUTTY_1);

	// Configuration

	fnPrintf( m_pHRequest, "<div class=\"head4tm6\">");
	fnPrintf( m_pHRequest, "Configuration");
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/SysConfig "
				 "title=\"System Configuration\" "
				 "TARGET=\"Content\">System</A>", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	// Monitoring

	fnPrintf( m_pHRequest, "<div class=\"head4tm6\">");
	fnPrintf( m_pHRequest, "Monitoring");
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest, "<A HREF=\"%s/Queries\" "
							  "title=\"View information about queries\" "
							  "TARGET=\"Content\">Queries</A>",
							  m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest, "<A HREF=\"%s/threads\" "
							  "title=\"View information about threads\" "
							  "TARGET=\"Content\">Threads</A>",
							  m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest, "<A HREF=\"%s/Stats\" "
							  "title=\"View statistics\" "
							  "TARGET=\"Content\">Statistics</A>",
							  m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	// Database

	if( pFlmSession)
	{
		fnPrintf( m_pHRequest, "<div class=\"head4tm6\">");
		fnPrintf( m_pHRequest,
					 "<A HREF=%s/database "
					 "title=\"Database\" "
					 "TARGET=\"Content\">Database</A>", m_pszURLString);
		fnPrintf( m_pHRequest, "</div>\n");

		fnPrintf( m_pHRequest, "<div class=\"task1\">");
		fnPrintf( m_pHRequest,
					 "<A HREF=%s/dbbackup "
					 "title=\"Database backup\" "
					 "TARGET=\"Content\">Backup</A>", m_pszURLString);
		fnPrintf( m_pHRequest, "</div>\n");

		fnPrintf( m_pHRequest, "<div class=\"task1\">");
		fnPrintf( m_pHRequest,
					 "<A HREF=%s/checkdb "
					 "title=\"Database check\" "
					 "TARGET=\"Content\">Check</A>", m_pszURLString);
		fnPrintf( m_pHRequest, "</div>\n");

	}

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/index "
				 "TARGET=\"Content\">Index Manager</A>", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	// Internal structures
	
	fnPrintf( m_pHRequest, "<div class=\"head4tm6\">");
	fnPrintf( m_pHRequest, "Internal Structures");
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/FlmSysData "
				 "title=\"View gv_FlmSysData and associated structures\" "
				 "TARGET=\"Content\">Database System Data</A>", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");

	// Misc.

	fnPrintf( m_pHRequest, "<div class=\"head4tm6\">");
	fnPrintf( m_pHRequest, "Misc.");
	fnPrintf( m_pHRequest, "</div>\n");

	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/file "
				 "title=\"File Manager\" "
				 "TARGET=\"Content\">File Manager</A>", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");
	
	fnPrintf( m_pHRequest, "<div class=\"task1\">");
	fnPrintf( m_pHRequest,
				 "<A HREF=%s/returncode "
				 "title=\"Return Code Lookup\" "
				 "TARGET=\"Content\">Return Code Lookup</A>", m_pszURLString);
	fnPrintf( m_pHRequest, "</div>\n");


	// The rest of this function is password related stuff...

	
	f_memset( szGblPassword, 0, sizeof( szGblPassword));
	if (DetectParameter( uiNumParams, ppszParams,
							   "StopSecureDbAccess"))
	{
		fnSetGblValue( FLM_SECURE_PASSWORD, "", (FLMSIZET)0);
		fnSetGblValue( FLM_SECURE_EXPIRATION, "", (FLMSIZET )0);
		uiSize = 0;

	}
	else
	{
		// Get the session global access password if it has been entered.
		// We are going to ignore any error, as the password may not have been entered.
		uiSize = sizeof( szGblPassword);
		(void)fnGetGblValue( FLM_SECURE_PASSWORD,
									(void *)szGblPassword,
									(FLMSIZET *)&uiSize);
	}

	fnPrintf( m_pHRequest, "<div class=\"head4tm6\">Secure Control</div>\n"
								  "<div class=\"task1\">");
	if (f_strlen( szGblPassword) == 0)
	{			
		fnPrintf( m_pHRequest, "<a href=%s/SecureDbAccess "
				 "title=\"Enter the Secure area access code\" "
				 "TARGET=\"Content\">Access Code</a>", m_pszURLString);			
	}
	else
	{
		fnPrintf( m_pHRequest, "<a href=%s/Nav.htm?StopSecureDbAccess>"
					 "Disallow Secure DB Access</a>", m_pszURLString);
	}
	fnPrintf( m_pHRequest, "</div>\n");


	// Check to see if we just entered the secure password.
	if (pvSession && DetectParameter( uiNumParams,
								ppszParams,	"SecurePassword"))
	{
		// We need to get the password entered, but it is being passsed as form data.
		// pszPassword will be allocated within the getFormValueByName function and
		// will need to be released using f_free.

		if (RC_BAD( rc = getFormValueByName( 
			FLM_SECURE_PASSWORD, &pszPassword, 0, &uiSize)))
		{
			goto Exit;
		}

		if (pszPassword && f_strlen( pszPassword) > 0)  // Do we want to have a minimum password length?
		{

			if (f_strcmp( szGblPassword, pszPassword) == 0)
			{
				if (fnSetSessionValue( pvSession,
											  FLM_SECURE_PASSWORD,
											  pszPassword,
											  uiSize))
				{
					flmAssert( 0);
				}
			}
			else
			{
				// They don't match - need to tell the user.
				fnPrintf( m_pHRequest, "<div class=\"task1\">");
				fnPrintf( m_pHRequest, "<font color=red>Invalid access code</font>");
				fnPrintf( m_pHRequest, "</div>\n");
			}
		}

		if (pszPassword)
		{
			f_free( &pszPassword);
		}
	}
	// Did the user ask to log off?
	else if (pvSession && DetectParameter( uiNumParams,
								ppszParams, "Logoff"))
	{
		// Pull the password out of the session data.
		if (fnSetSessionValue( pvSession, FLM_SECURE_PASSWORD, "", 0))
		{
			flmAssert( 0);
		}
	
		// Close any database handles associated with our session...
		if (pFlmSession)
		{
			RCODE				tmpRC = FERR_OK;
			F_SessionDb *	pSessionDb = NULL;
			char *			pDbKey;

			tmpRC = pFlmSession->getNextDb( &pSessionDb);
			while (RC_OK( tmpRC))
			{
				pDbKey = (char *)pSessionDb->getKey();
				tmpRC = pFlmSession->getNextDb( &pSessionDb);			
				pFlmSession->closeDb( pDbKey);
			}
			
			if (tmpRC != FERR_EOF_HIT)
			{
				flmAssert( 0);
			}

		}


		// Reload the content window...
		fnPrintf( m_pHRequest, "<script> parent.Content.location.href=\"%s"
									  "/Welcome.htm\" </script>", m_pszURLString);

	}

	// Do we display the Secure Password option?

	if (pvSession)
	{
		uiSize = sizeof( szValue);
		f_memset( szValue, 0, uiSize);
		(void)fnGetSessionValue( pvSession,
								 FLM_SECURE_PASSWORD,
								 (void *)szValue,
								 (FLMSIZET *)&uiSize);

		if (f_strlen( szValue) != 0)
		{
			fnPrintf( m_pHRequest, "<div class=\"task1\"> "
						 "<A HREF=Nav.htm?Logoff>Log Off</A> </div>\n");
		}
		else
		{
			// Only want to display the password entry box if the secure mode
			// has been enabled
			uiSize = sizeof( szGblPassword);
			(void)fnGetGblValue( FLM_SECURE_PASSWORD,
									(void *)szGblPassword,
									(FLMSIZET *)&uiSize);
			if (f_strlen( szGblPassword) != 0)
			{
				fnPrintf( m_pHRequest, "<div class=\"task1\">");
				fnPrintf( m_pHRequest,
							 "<form action=\"Nav.htm?SecurePassword\" method=\"post\" "
							 "title=\"SuperPassword\">\n");
				fnPrintf( m_pHRequest,
							 "<input name=\"%s\" type=\"password\" size=19 maxlength=19 "
							 "title=\"Password\"><br>\n", FLM_SECURE_PASSWORD);
				printButton( "Login", BT_Submit);
				fnPrintf( m_pHRequest, "</form>\n");
				fnPrintf( m_pHRequest, "</div>\n");
			}
		}
	}	 
	
	printDocEnd();
	fnEmit();

Exit:

	if (pszPassword)
	{
		f_free( &pszPassword);
	}

	if (pvSession)
	{
		(void)fnReleaseSession( pvSession);
	}
	
	return( rc);
}

/*********************************************************
 Desc:	Return HTML code that defines the Welcome.htm frame.
 **********************************************************/
RCODE F_FrameWelcome::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE	rc = FERR_OK;

	F_UNREFERENCED_PARM(uiNumParams);
	F_UNREFERENCED_PARM(ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	fnPrintf( m_pHRequest, "<title>Welcome</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body>\n");
	fnPrintf( m_pHRequest, "<TABLE ALIGN=\"CENTER\">\n");
	fnPrintf( m_pHRequest,
				 "<TD HEIGHT=10 WIDTH=80%% ALIGN=\"CENTER\" "
				 "VALIGN=\"TOP\" BGCOLOR=\"#FFFFFF\">\n");
	fnPrintf( m_pHRequest,
				 "<STRONG><FONT color=\"black\" SIZE=+2>"
				 "<H1 ALIGN=\"CENTER\">Welcome to the Database <i>i</i>Monitor."
				 "</H1></FONT></STRONG>\n");
	fnPrintf( m_pHRequest,
				 "<center><br><br><p>This is a tool for examining and, if necessary, "
				 "altering various data elements of the database <br> "
				 "internal structures as well as the database records "
				 "themselves.\n");
	fnPrintf( m_pHRequest,
				 "<br><br><p><strong><font color=\"#57FF26\">"
				 "Please exercise caution when using this tool."		
				 "</font></strong></center>\n");
	fnPrintf( m_pHRequest, "</TD></TABLE>\n");
	fnPrintf( m_pHRequest, "</body>\n");
	fnPrintf( m_pHRequest, "</html>\n");

	fnEmit();

	return( rc);
}


/*********************************************************
 Desc:	Return HTML code that defines the SecureDbAccess
			popup window contents.
 **********************************************************/
RCODE F_SecureDbAccess::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM(uiNumParams);
	F_UNREFERENCED_PARM(ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE
					"<html>\n<head>\n<title>Secure Database Access</title>\n"
					"</head>\n<body>\n"
					"<center><strong><h2>Please paste the private access "
					"enabling data in the text area  below, then click "
					"on the Submit button</h2></strong></center><br>\n"
					"<form name=\"form1\" method=\"post\" action=\"/coredb/SecureDbInfo\">\n"
					"<center><textarea name=\"SecureData\" rows=12 cols=65 >"
					"</textarea></center><br>\n<center>");
	printButton( "Submit", BT_Submit);
	printButton( "Reset", BT_Reset);
	fnPrintf( m_pHRequest, "</center></form>\n"
					"<SCRIPT>\ndocument.form1.SecureData.focus()\n</SCRIPT>\n"
					"</body>\n</html>\n");

	fnEmit();

	return( rc);
}

/*********************************************************
 Desc:	Return HTML code that defines the SecureDbInfo
			popup window contents.
 **********************************************************/
RCODE F_SecureDbInfo::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pszBuffer = NULL;
	FLMUINT			uiLen;
	char *			pszPassword;
	char *			pszExpiration;
	FLMUINT			uiPwdLen = 0;
	FLMUINT			uiExpLen = 0;
	FLMBYTE *		pszData = NULL;
	FLMUINT			uiDataSize;
	char *			pTmp;
	FLMBOOL			bDataOk = FALSE;
	void *			pvSession = NULL;
	FLMUINT			uiExpTime;
	FLMUINT			uiCurrTime;

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	if (gv_FlmSysData.HttpConfigParms.fnAcquireSession)
	{
		pvSession = fnAcquireSession();
	}

	if (RC_BAD( rc = getFormValueByName( "SecureData",
		(char **)&pszBuffer, 0, &uiLen)))
	{
		printErrorPage( FERR_INVALID_PARM, TRUE, "Could not retrieve required data.");
		goto Exit;
	}

	// Decode the data.

	fcsDecodeHttpString( (char *)pszBuffer);
	if ( RC_BAD( rc = flmExtractHexPacketData(
								pszBuffer,
								&pszData,
								&uiDataSize)))
	{
		goto SkipPwdExp;
	}

   // Extract the password field...The data should be in the format
	// password=Password,expire=Date

	if (( pszPassword = f_strstr( (char *)pszData, "password")) != NULL)
	{
		pszPassword += f_strlen("password") + 1; // Allow for '='
		for ( pTmp = pszPassword, uiPwdLen = 0;
				*pTmp && *pTmp != ',';
				pTmp++, uiPwdLen++);
	}
	else
	{
		goto SkipPwdExp;
	}
	
	if (( pszExpiration = f_strstr( (char *)pszData, "expire")) != NULL)
	{
		pszExpiration += f_strlen("expire") + 1; // Allow for '='
		for ( pTmp = pszExpiration, uiExpLen = 0;
				*pTmp && *pTmp != ',';
				pTmp++, uiExpLen++);
	}
	else
	{
		goto SkipPwdExp;
	}
	
	pszPassword[ uiPwdLen] = '\0';
	pszExpiration[ uiExpLen] = '\0';

	// Let's determine if the Expiration date is still valid.
	uiExpTime = f_atoud( pszExpiration);
	f_timeGetSeconds( &uiCurrTime);

	if (uiCurrTime > uiExpTime)
	{
		goto SkipPwdExp;
	}

	
	// Now store the data...
	if (gv_FlmSysData.HttpConfigParms.fnSetGblValue)
	{
		if (fnSetGblValue( FLM_SECURE_PASSWORD, pszPassword, (FLMSIZET)uiPwdLen))
		{
			flmAssert( 0);
		}
		if (fnSetGblValue( FLM_SECURE_EXPIRATION, pszExpiration, (FLMSIZET)uiExpLen))
		{
			flmAssert( 0);
		}

		// Now, reset the session password if it exists.

		pszPassword = '\0';
		if (fnSetSessionValue( pvSession,
									  FLM_SECURE_PASSWORD,
									  pszPassword,
									  0))
		{
			flmAssert( 0);
		}

	}

	bDataOk = TRUE;

SkipPwdExp:


	if (bDataOk)
	{
		stdHdr();
		fnPrintf( m_pHRequest, HTML_DOCTYPE);
		fnPrintf( m_pHRequest, "<html>\n");
		fnPrintf( m_pHRequest, "<body>\n");
		fnPrintf( m_pHRequest, "<script>parent.Menu.location.href=\"%s/Nav.htm\";\n", m_pszURLString);
		fnPrintf( m_pHRequest, "parent.Content.location.replace(\"%s/Welcome.htm\")</script>\n",
					m_pszURLString);
		fnPrintf( m_pHRequest, "</body>\n");
		fnPrintf( m_pHRequest, "</html>\n");

	}
	else
	{
		printErrorPage( FERR_INVALID_PARM, TRUE, "The data you entered could not been accepted."
															  "The information may be invalid or expired."
															  " Please try again with new data.");
	}
	

Exit:

	fnEmit();

	if (pszBuffer)
	{
		f_free( &pszBuffer);
	}

	if (pszData)
	{
		f_free( &pszData);
	}

	if (pvSession)
	{
		(void)fnReleaseSession( pvSession);
	}

	return( rc);
}
