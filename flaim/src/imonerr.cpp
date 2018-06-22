//-------------------------------------------------------------------------
// Desc:	Error page for HTTP monitoring.
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
 Desc:	Page that is displayed when a URL is requested that we don't
			know how to fill
****************************************************************************/
RCODE F_ErrorPage::display(
	FLMUINT 			uiNumParams,
	const char **	ppszParams)
{
	RCODE 		rc = FERR_OK;

	// Can't use a call to stdHdr() because we want to send back a 404 error
	
	fnSetHdrValue( "Content-Type", "text/html");
	fnSetNoCache( NULL);
	fnSendHeader( HTS_NOT_FOUND);

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	fnPrintf( m_pHRequest, "<title>Error Page</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body>\n");
	fnPrintf( m_pHRequest, "<H2 ALIGN=CENTER>That which you seek is not available.</H2>\n");
	fnPrintf( m_pHRequest, "<br><br> \n Number of Parameters: %ld <br>\n", uiNumParams);
	
	for (FLMUINT uiLoop = 0; uiLoop < uiNumParams; uiLoop++)
	{
		fnPrintf( m_pHRequest, "Parameter %ld:\t%s<BR>\n", uiLoop, ppszParams[uiLoop]);
	}

	fnPrintf( m_pHRequest, "<BR><BR>\n");
	fnPrintf( m_pHRequest, "</BODY></HTML>\n");
	fnEmit();

	return( rc);
}

/****************************************************************************
Desc:	Page that is displayed when a URL is requested for a page requireing
		secure access but the Global security is not enabled or has expired.
****************************************************************************/
RCODE F_GblAccessPage::display(
	FLMUINT 				uiNumParams,
	const char **		ppszParams)
{
	RCODE		rc = FERR_OK;

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	fnPrintf( m_pHRequest, "<title>Global Access Error Page</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body>\n");
	// We're putting this script here to force the Nav bar to reload.  The
	// reason for this because of the case where one user disables the secure
	// access stuff after a second user has successfully logged in.  When the
	// second user attempts to load a page requiring secure access he or she
	// will get this page.  If the nav bar is not reloaded then it will still
	// be indicating that the user is logged in and the user will have no idea
	// why this page is coming up.  If the nav bar is reloaded, it will at
	// least give a clue (though not an obvious one) as to what has happened.
	fnPrintf( m_pHRequest, "<script>parent.Menu.location.href=\"%s/Nav.htm\" "
								  "</script>\n", m_pszURLString);
	fnPrintf( m_pHRequest, "<STRONG>The page you are attempting to view requires "
		"secure access. The secure access either has not been enabled or "
		"it has expired. To activate secure access, you must select the "
		"\"Access Code\" link in the  navigation bar and enter the "
		"enabling data provided to you by Novell Inc. You will then need "
		"to enter the secure access password.</STRONG>\n");
	fnPrintf( m_pHRequest, "</BODY></HTML>\n");

	fnEmit();

	return( rc);
}


/****************************************************************************
Desc:	Page that is displayed when a URL is requested for a page requireing
		secure access but the Session security is not enabled.  A password 
		is required.
****************************************************************************/
RCODE F_SessionAccessPage::display(
	FLMUINT 			uiNumParams,
	const char **	ppszParams)
{
	RCODE rc = FERR_OK;

	F_UNREFERENCED_PARM( uiNumParams);
	F_UNREFERENCED_PARM( ppszParams);

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");
	fnPrintf( m_pHRequest, "<head>\n");
	fnPrintf( m_pHRequest, "<title>Session Access Error Page</title>\n");
	fnPrintf( m_pHRequest, "</head>\n");
	fnPrintf( m_pHRequest, "<body>\n");
	fnPrintf( m_pHRequest, "<STRONG>The page you are attempting to view requires "
		"secure access. The session level access has not been enabled."
		" To activate session level secure access, you must enter the "
		"secure access password.</STRONG>\n");
	fnPrintf( m_pHRequest, "</BODY></HTML>\n");

	fnEmit();

	return( rc);
}


