//-------------------------------------------------------------------------
// Desc:	URL parsing.
// Tabs:	3
//
// Copyright (c) 1998-2000, 2002-2003, 2005-2007 Novell, Inc.
// All Rights Reserved.
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

#define FLAIM_PROTOCOL_NAME				"x-flaim"
#define FLAIM_PROTOCOL_NAME_LEN			7
#define TCP_SUB_PROTOCOL_NAME				":tcp://"
#define TCP_SUB_PROTOCOL_NAME_LEN		7
#define STREAM_SUB_PROTOCOL_NAME			":stream://"
#define STREAM_SUB_PROTOCOL_NAME_LEN	10
#define NO_SUB_PROTOCOL_NAME				"://"
#define NO_SUB_PROTOCOL_NAME_LEN			3
#define FLAIM_DEFAULT_PORT					1677

/****************************************************************************
Public:	FUrl constructor, destructor
****************************************************************************/
FUrl::FUrl()
{
	m_iPort = -1;
	m_iSubProtocol = NO_SUB_PROTOCOL;
	m_bRelative = FALSE;
	m_pucAlloc = NULL;
	m_pszHostName = NULL;
	m_pszFileName = NULL;
	m_uiAddrType = FLM_CS_NO_ADDR;
	m_pszIPName = NULL;
	m_ui32IPAddr = 0;
	f_memset( m_pszAddr, 0, FLM_CS_MAX_ADDR_LEN);
}

FUrl::~FUrl()
{
	Reset();
}

/****************************************************************************
Public:	FUrl::Reset
Desc:		Reset members of the URL - freeing memory, etc.
****************************************************************************/
void FUrl::Reset()
{
	m_iPort = -1;
	m_iSubProtocol = NO_SUB_PROTOCOL;
	m_bRelative = FALSE;
	f_free( &m_pucAlloc);
	m_pszHostName = NULL;
	m_pszFileName = NULL;
	m_bLocal = FALSE;
	f_free( (void **)&m_pszIPName);
	m_ui32IPAddr = 0;
	m_uiAddrType = FLM_CS_NO_ADDR;
	f_memset( m_pszAddr, 0, FLM_CS_MAX_ADDR_LEN);
}

/****************************************************************************
Public:	FUrl::SetUrl
Desc:		Parse the string passed in and setup the members of the URL.
****************************************************************************/
RCODE FUrl::SetUrl(
	const char *		pszUrlStr)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiAllocLen;
	char *				pucTmp;
	const char *		pucCurrent;
	const char *		pszFileName = NULL;
	const char *		pszHostName = NULL;
	FLMINT				iHostNameLen = 0;
	FLMINT				iPort = -1;
	FLMINT				iSubProtocol = NO_SUB_PROTOCOL;
	FLMBOOL				bRelative = FALSE;

	/* Clear out the current URL. */

	Reset();

	/*
	See if the string begins with our protocol name.  If not, assume that
	it is a file name.
	*/

	if ((f_strncmp( pszUrlStr, FLAIM_PROTOCOL_NAME,
							FLAIM_PROTOCOL_NAME_LEN) != 0) ||
		 ((f_strncmp( pszUrlStr + FLAIM_PROTOCOL_NAME_LEN,
								TCP_SUB_PROTOCOL_NAME,
								TCP_SUB_PROTOCOL_NAME_LEN) != 0) &&
		  (f_strncmp( pszUrlStr + FLAIM_PROTOCOL_NAME_LEN,
								STREAM_SUB_PROTOCOL_NAME,
								STREAM_SUB_PROTOCOL_NAME_LEN) != 0) &&
		  (f_strncmp( pszUrlStr + FLAIM_PROTOCOL_NAME_LEN,
		  						NO_SUB_PROTOCOL_NAME,
								NO_SUB_PROTOCOL_NAME_LEN) != 0)))
	{
		bRelative = FALSE;
		pszFileName = pszUrlStr;
		m_bLocal = TRUE;
	}
	else
	{
		pucCurrent = pszUrlStr + FLAIM_PROTOCOL_NAME_LEN;
		if( f_strncmp( pucCurrent, TCP_SUB_PROTOCOL_NAME,
									TCP_SUB_PROTOCOL_NAME_LEN) == 0)
		{
			iSubProtocol = TCP_SUB_PROTOCOL;
			pucCurrent += TCP_SUB_PROTOCOL_NAME_LEN;
		}
		else if( f_strncmp( pucCurrent, STREAM_SUB_PROTOCOL_NAME,
									STREAM_SUB_PROTOCOL_NAME_LEN) == 0)
		{
			iSubProtocol = STREAM_SUB_PROTOCOL;
			pucCurrent += STREAM_SUB_PROTOCOL_NAME_LEN;
			m_uiAddrType = FLM_CS_STREAM_ADDR;
		}
		else
		{
			iSubProtocol = NO_SUB_PROTOCOL;
			pucCurrent += NO_SUB_PROTOCOL_NAME_LEN;
		}

		/*
		If the next character is NOT a slash, we have a host name
		(or stream name) and optionally a port number
		*/
	
		if( *pucCurrent != '/')
		{
			FLMINT				iAddrPos;
			FLMINT				iAddrSlot;
			FLMUINT				uiTmpVal;
			const char *		pucNamePos;

			iPort = FLAIM_DEFAULT_PORT;
			pszHostName = pucCurrent;
			while( (*pucCurrent) && (*pucCurrent != '/') && (*pucCurrent != ':'))
			{
				iHostNameLen++;
				pucCurrent++;
			}

			/*
			It is invalid to have an empty host name
			*/

			if( !iHostNameLen)
			{
				rc = RC_SET( FERR_IO_INVALID_PATH);
				goto Exit;
			}

			/*
			If this is a URL that does not include a file path,
			we are done
			*/

			if( !(*pucCurrent))
			{
				goto Done;
			}

			/* See if we got a colon.  If so, extract the port number. */

			if( *pucCurrent == ':')
			{
				iPort = 0;

				/* Increment past colon. */

				pucCurrent++;

				/* Make sure we have at least one digit. */

				if( (*pucCurrent < '0') || (*pucCurrent > '9'))
				{
					rc = RC_SET( FERR_IO_INVALID_PATH);
					goto Exit;
				}

				/* Go until string termination or another backslash. */

				while( (*pucCurrent) && (*pucCurrent != '/'))
				{
					if ((*pucCurrent < '0') || (*pucCurrent > '9'))
					{
						rc = RC_SET( FERR_IO_INVALID_PATH);
						goto Exit;
					}
					
					iPort *= 10;
					iPort += (FLMINT)(*pucCurrent - '0');
					pucCurrent++;
				}
			}

			if( iSubProtocol != STREAM_SUB_PROTOCOL)
			{
				m_uiAddrType = FLM_CS_IP_ADDR;
			}

			/*
			See if the host name is an IP address.  If it is an
			address, extract it.
			*/

			/*
			Dotted IP Address.  The format of an IP address is
			000.000.000.000 where each dot-separated value can vary
			in length from 1 to 3 characters.
			*/

			iAddrSlot = 0;
			uiTmpVal = 0;
			pucNamePos = pszHostName;
			for( iAddrPos = 0; iAddrPos < iHostNameLen; iAddrPos++)
			{
				switch( *pucNamePos)
				{
					case '0':
					case '1':
					case '2':
					case '3':
					case '4':
					case '5':
					case '6':
					case '7':
					case '8':
					case '9':
					{
						uiTmpVal *= 10;
						uiTmpVal += (FLMUINT)(*pucNamePos - '0');
						break;
					}
					case '.':
					{
						if( iAddrSlot >= 4)
						{
							rc = RC_SET( FERR_IO_INVALID_PATH);
							goto Exit;
						}
						
						((FLMBYTE *)(&m_ui32IPAddr))[ iAddrSlot++] = (FLMBYTE)uiTmpVal;
						uiTmpVal = 0;
						break;
					}
					default:
					{
						// The buffer does not contain an address.  Cause the
						// loop to terminate.

						iAddrPos = iHostNameLen;
						iAddrSlot = 3;
						continue;
					}
				}
				pucNamePos++;
			}

			if( iAddrSlot != 3)
			{
				rc = RC_SET( FERR_IO_INVALID_PATH);
				goto Exit;
			}

			/* By now we better be on another slash */

			if( *pucCurrent != '/')
			{
				rc = RC_SET( FERR_IO_INVALID_PATH);
				goto Exit;
			}
		}

		/* Increment past slash. */

		pucCurrent++;

		/* See if there are any keywords. */

		if( f_strnicmp( pucCurrent, "RELATIVE.", 9) == 0)
		{
			pucCurrent += 9;

			/*
			Better have specified a host name - RELATIVE is only valid when
			talking to a server.
			*/

			if (!iHostNameLen)
			{
				rc = RC_SET( FERR_IO_INVALID_PATH);
				goto Exit;
			}

			/*
			Treat the rest of the string after the "RELATIVE."
			as a relative file name.
			*/

			bRelative = TRUE;
			pszFileName = pucCurrent;
		}
		else if( f_strnicmp( pucCurrent, "ABSOLUTE.", 9) == 0)
		{
			pucCurrent += 9;

			/*
			Treat the rest of the string after the "ABSOLUTE."
			as an absolute file name.
			*/

			bRelative = FALSE;
			pszFileName = pucCurrent;
		}
		else
		{

			/*
			No keywords, so default case is to treat the rest of the
			string as a relative file name.
			*/

			bRelative = TRUE;
			pszFileName = pucCurrent;
		}
	}

	/* Make one allocation for everything that is to be copied. */

Done:

	uiAllocLen = 0;
	if( pszFileName)
	{
		uiAllocLen += (FLMUINT)(f_strlen( pszFileName) + 1);
	}

	if (iHostNameLen)
	{
		uiAllocLen += (FLMUINT)(iHostNameLen + 1);
	}

	if (RC_BAD( rc = f_alloc( uiAllocLen, &m_pucAlloc)))
	{
		goto Exit;
	}

	m_iSubProtocol = iSubProtocol;
	m_bRelative = bRelative;
	pucTmp = m_pucAlloc;
	m_iPort = iPort;
	
	if (iHostNameLen)
	{
		f_memcpy( pucTmp, pszHostName, iHostNameLen);
		pucTmp[ iHostNameLen] = 0;
		m_pszHostName = pucTmp;
		pucTmp += (iHostNameLen + 1);

		if( iHostNameLen + 1 <= FLM_CS_MAX_ADDR_LEN)
		{
			f_strcpy( m_pszAddr, m_pszHostName);
		}
		else
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	
	if( pszFileName)
	{
		f_strcpy( pucTmp, pszFileName);
		m_pszFileName = pucTmp;
	}
	else
	{
		m_pszFileName = NULL;
	}

Exit:

	return( rc);
}
