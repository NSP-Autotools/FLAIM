//-------------------------------------------------------------------------
// Desc:	HTML callback function for displaying monitoring web pages.
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

FSTATIC char * tokenizer( 
	const char *	pszString,
	FLMBYTE			ucToken1,
	FLMBYTE			ucToken2);

F_WebPageFactory	*	gv_pWPFact = NULL;

/****************************************************************************
Desc:	This is the function that the HTTP server calls when it wants to
			display one of our pages
****************************************************************************/
int flmHttpCallback(
	HRequest *			pHRequest,
	void *				//pvUserData
	)
{
	RCODE							rc = FERR_OK;
	F_WebPage *					pPage = NULL;
	char *						pszPath = NULL;
	char *						pszQuery = NULL;
	char *						pszTemp = NULL;
	const char *				pszConstTemp = NULL;
#define MAX_PARAMS 10
	const char *				pszParams[ MAX_PARAMS];
	FLMUINT						uiNumParams;

	// If we get a NULL for the pHRequest object, then we are shutting down...

	if (pHRequest == NULL)
	{
		// Remove the globals that enable the secure pages...
		gv_FlmSysData.HttpConfigParms.fnSetGblValue( 
				FLM_SECURE_PASSWORD, "", 0);
		gv_FlmSysData.HttpConfigParms.fnSetGblValue(
				FLM_SECURE_EXPIRATION, "", 0);

		// Delete the web page factory object
		if (gv_pWPFact)
		{
			gv_pWPFact->Release( NULL);
		}
		gv_pWPFact = NULL;
		goto Exit;
	}

	// Increment the use count (helps ensure that the function pointers
	// that display() references don't go away while display() still needs
	// them.

	f_mutexLock( gv_FlmSysData.HttpConfigParms.hMutex);
	gv_FlmSysData.HttpConfigParms.uiUseCount++;
	f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);

	// Must not access any HRequest function pointers prior to incrementing the
	// use count.

	if( !gv_FlmSysData.HttpConfigParms.fnReqPath)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// If the web page factory does not exist yet, then we need to create it.
	if (!gv_pWPFact)
	{
		f_mutexLock( gv_FlmSysData.HttpConfigParms.hMutex);
		// In the time it took us to get the lock, some other thread might
		// have come along and created the factory already...
		if (!gv_pWPFact)
		{
			if ((gv_pWPFact = f_new F_WebPageFactory) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);
				goto Exit;
			}
		}
		f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);
	}
	
	pszConstTemp = gv_FlmSysData.HttpConfigParms.fnReqPath( pHRequest);
	flmAssert( pszConstTemp);

	if( RC_BAD( rc = f_alloc( 
		f_strlen( pszConstTemp) + 1, &pszPath)))
	{
		goto Exit;
	}

	f_strcpy( pszPath, pszConstTemp);
	
	pszConstTemp = gv_FlmSysData.HttpConfigParms.fnReqQuery( pHRequest);
	if( pszConstTemp)
	{
		if( RC_BAD( rc = f_alloc( f_strlen( pszConstTemp) + 1, &pszQuery)))
		{
			goto Exit;
		}

		f_strcpy( pszQuery, pszConstTemp);
		pszConstTemp = pszQuery;
	}
	else  // This URL had no query string...
	{
		// If pszQuery is NULL, it causes problems further down, so we'll
		// make it a pointer to a null string...

		if( RC_BAD( rc = f_alloc( 1, &pszQuery)))
		{
			goto Exit;
		}
		pszQuery[0] = '\0';				
	}

	// Strip off pszURLString (and the next '/', if there is one) from the request and store 
	// what's left as pszParams[0].
	// (ie: /coredb/FlmSysData --> FlmSysData)

	// Note: The reason we're checking for the URL string first is because if
	// we're using our own http stack, then this callback is called for every
	// http request and we don't want to crash if we've got a short URI.
	// When we're running under DS, we're guarenteed that the URLString will
	// be part of the URI.

	if( f_strlen( pszPath) >= gv_FlmSysData.HttpConfigParms.uiURLStringLen)
	{
		pszConstTemp = pszPath + gv_FlmSysData.HttpConfigParms.uiURLStringLen;
		if( *pszConstTemp == '/')
		{
			pszConstTemp++;
		}
	}
	else
	{
		pszConstTemp = pszPath;
	}

	pszParams[0] = pszConstTemp;
	uiNumParams = 1;


	// Parse parameters in the query string 
	// Note that it's technically incorrect to have more than one ? in a
	// URL, but we didn't know that when we first started creating some of 
	// these pages and as a result, some queries are in the form of:
	// ?name1=value1?name2=value2?name3=value3...  (which is improper) and
	// some have the form:
	// ?name1=value1&name2=value2&name3=value3... (which is correct).
	
	pszTemp = pszQuery;
	
	while( *pszTemp != 0)
	{
		flmAssert( uiNumParams < MAX_PARAMS);
		pszParams[ uiNumParams] = pszTemp;
		uiNumParams++;
		
		pszTemp = tokenizer( pszTemp, '?', '&');
		
		if (*pszTemp)
		{
			*pszTemp = '\0';
			pszTemp++;
		}
	}

	// Tell the factory to create the page
	
	if (RC_BAD( rc = gv_pWPFact->create( pszParams[0], &pPage, pHRequest)))
	{
		goto Exit;
	}
	
	
	pPage->setMembers( pHRequest);

	// display the page
	if( RC_BAD( rc = pPage->display (uiNumParams, &pszParams[0])))
	{
		goto Exit;
	}

Exit:

	// Decrement the use count

	if( pHRequest)
	{
		f_mutexLock( gv_FlmSysData.HttpConfigParms.hMutex);
		if( gv_FlmSysData.HttpConfigParms.uiUseCount > 0)
		{
			gv_FlmSysData.HttpConfigParms.uiUseCount--;
		}
		else
		{
			flmAssert( 0);
		}
		f_mutexUnlock( gv_FlmSysData.HttpConfigParms.hMutex);
	}
	
	if (pPage)
	{
		gv_pWPFact->Release( &pPage);
	}

	if (pszPath)
	{
		f_free( &pszPath);
	}

	if (pszQuery)
	{
		f_free( &pszQuery);
	}

	return (int)rc;
}

/****************************************************************************
 Desc:	Given a string, returns a pointer to the next occurance of either
			of two specific characters.
****************************************************************************/
FSTATIC char * tokenizer(
	const char *	pszString,
	FLMBYTE			ucToken1,
	FLMBYTE			ucToken2)
{
	while (*pszString != ucToken1 && *pszString != ucToken2 && *pszString != 0)
	{
		pszString++;
	}

	return( (char *)pszString);
}

/****************************************************************************
 Desc:	Given two address, calculates the difference between them and
			converts them to a string in hex (including the 0x).
			This is in its own function because of some possible issues 
			with win64 and maybe other 64bit OS's
****************************************************************************/
void printOffset(
	void *		pBase,
	void *		pAddress,
	char *		pszOffset)
{
	FLMUINT uiBase = (FLMUINT)pBase;
	FLMUINT uiAddress = (FLMUINT)pAddress;

	f_sprintf( pszOffset, "0x%lX", (FLMUINT)(uiAddress - uiBase));
}

/****************************************************************************
 Desc:	Takes a pointer and converts its address to a string in hex
			(including the 0x).  This is in its own function because of some
			possible issues with win64 and maybe other 64bit OS's
****************************************************************************/
void printAddress(
	void *		pAddress,
	char *		pszBuff)
{
	FLMUINT64		ui64Addr = (FLMUINT64)((FLMUINT)pAddress);
	FLMUINT			uiHigh = (FLMUINT)(ui64Addr >> 32);
	FLMUINT			uiLow = (FLMUINT)(ui64Addr & (FLMUINT64)0xFFFFFFFF);

	if( uiHigh)
	{
		f_sprintf( pszBuff, "0x%X%08X", 
			(unsigned)uiHigh, (unsigned)uiLow);
	}
	else
	{
		f_sprintf( pszBuff, "0x%X", (unsigned)uiLow);
	}
}

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
RCODE F_DynamicBuffer::addChar(
	char			ucCharacter)
{
	RCODE			rc = FERR_OK;

	if (!m_bSetup)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;

	}

	f_mutexLock( m_hMutex);

	// Is there room for just one more character plus a terminator?
	if ((m_uiBuffSize - m_uiUsedChars) > 1)
	{
		m_pucBuffer[ m_uiUsedChars++] = ucCharacter;
		m_pucBuffer[ m_uiUsedChars] = 0;
	}
	else
	{
		// Allocate a new buffer or increase the size of the existing one.
		if( !m_uiBuffSize)
		{
			if( RC_BAD( rc = f_alloc( 50, &m_pucBuffer)))
			{
				goto Exit;
			}
			m_uiBuffSize = 50;
		}
		else
		{
			if( RC_BAD( rc = f_realloc( m_uiBuffSize + 50,  &m_pucBuffer)))
			{
				goto Exit;
			}
			m_uiBuffSize += 50;
		}


		m_pucBuffer[ m_uiUsedChars++] = ucCharacter;
		m_pucBuffer[ m_uiUsedChars] = 0;
	}

Exit:

	if ( m_bSetup)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
RCODE F_DynamicBuffer::addString( const char * pszString)
{
	RCODE					rc = FERR_OK;
	const	char *		pTemp = pszString;
	FLMUINT				uiTmpPos = m_uiUsedChars;


	while( *pTemp)
	{
		if (RC_BAD( rc = addChar( *pTemp)))
		{
			// Reset the buffer to its state prior to this call.
			
			m_uiUsedChars = uiTmpPos;
			if (m_uiBuffSize > 0)
			{
				m_pucBuffer[ m_uiUsedChars] = 0;
			}
			goto Exit;
		}
		pTemp++;
	}

Exit:

	return( rc);
}

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
const char * F_DynamicBuffer::printBuffer()
{
	return( (const char *)m_pucBuffer);
}
