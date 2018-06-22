//-------------------------------------------------------------------------
// Desc:	URL parsing - definitions.
// Tabs:	3
//
// Copyright (c) 1998-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FURL_H
#define FURL_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define NO_SUB_PROTOCOL				-1
#define TCP_SUB_PROTOCOL			1
#define STREAM_SUB_PROTOCOL		2

class FUrl;
typedef FUrl *			FUrl_p;

/****************************************************************************
Desc:		The FUrl class is for dealing with URLs.
****************************************************************************/
class FUrl: public F_Object
{
private:

	FLMINT		m_iSubProtocol;		// Sub-Protocol to use, -1 if none.
	FLMBOOL		m_bRelative;			// Relative to server directory or area?
	FLMBOOL		m_bLocal;				// URL references a local file
	char *		m_pucAlloc;				// Memory allocation.
	char *		m_pszHostName;			// Host name string, NULL if none.
	char *		m_pszFileName;			// File name string.
	char *		m_pszIPName;			// IP name string, NULL if none.
	FLMUINT		m_uiAddrType;			// Address type: IP, IPX, etc.
	FLMUINT32	m_ui32IPAddr;			// IP Address, 0 if not yet set.
	FLMINT		m_iPort;					// Port number, -1 if none.
	char			m_pszAddr[ FLM_CS_MAX_ADDR_LEN];

public:

	FUrl();

	virtual ~FUrl();

	void Reset();

	RCODE SetUrl(
		const char * 		pszUrlStr);

	FINLINE const char * GetFile( void)
	{
		return( (const char *)m_pszFileName);
	}

	FINLINE const char * GetIPHost( void)
	{
		return( (const char *)m_pszHostName);
	}

	FINLINE const char * GetAddress( void)
	{
		return( (const char *)m_pszAddr);
	}

	FINLINE FLMINT GetIPPort( void)
	{
		return m_iPort;
	}

	FINLINE FLMBOOL GetRelative( void)
	{
		return m_bRelative;
	}

	FINLINE FLMINT GetSubProtocol( void)
	{
		return m_iSubProtocol;
	}

	FINLINE FLMUINT GetAddrType( void)
	{
		return m_uiAddrType;
	}

	FINLINE FLMBOOL IsLocal( void)
	{
		return m_bLocal;
	}
};

#include "fpackoff.h"

#endif
