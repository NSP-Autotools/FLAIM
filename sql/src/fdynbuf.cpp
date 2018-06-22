//------------------------------------------------------------------------------
// Desc: This file contains the routines for the Flaim Dynamic Buffer Class.
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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
#include "fdynbuf.h"

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
RCODE F_DynamicBuffer::addChar(
	FLMBYTE			ucCharacter)
{
	RCODE			rc = NE_SFLM_OK;

	if (!m_bSetup)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
		goto Exit;

	}

	f_mutexLock( m_hMutex);

	// Is there room for just one more character plus a terminator?
	if ((m_uiBuffSize - m_uiUsedChars) > 1)
	{
		m_psBuffer[m_uiUsedChars++] = ucCharacter;
		m_psBuffer[m_uiUsedChars] = 0;
	}
	else
	{
		// Allocate a new buffer or increase the size of the existing one.
		if( !m_uiBuffSize)
		{
			if( RC_BAD( rc = f_alloc( 50, &m_psBuffer)))
			{
				goto Exit;
			}
			m_uiBuffSize = 50;
		}
		else
		{
			if( RC_BAD( rc = f_realloc( m_uiBuffSize + 50,  &m_psBuffer)))
			{
				goto Exit;
			}
			m_uiBuffSize += 50;
		}


		m_psBuffer[m_uiUsedChars++] = ucCharacter;
		m_psBuffer[m_uiUsedChars] = 0;
	}

Exit:

	if ( m_bSetup)
	{
		f_mutexUnlock( m_hMutex);
	}

	return rc;
}

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
RCODE F_DynamicBuffer::addString( const char * pszString)
{
	RCODE					rc = NE_SFLM_OK;
	const	char *		pTemp = pszString;
	FLMUINT				uiTmpPos = m_uiUsedChars;


	while( *pTemp)
	{
		if (RC_BAD( rc = addChar( (FLMBYTE)*pTemp)))
		{
			// Reset the buffer to its state prior to this call.
			m_uiUsedChars = uiTmpPos;
			if (m_uiBuffSize > 0)
			{
				m_psBuffer[ m_uiUsedChars] = 0;
			}
			goto Exit;
		}
		pTemp++;
	}

Exit:

	return rc;
}

/******************************************************************
Desc:	Implements the addChar function of the DynamicBuffer class
*******************************************************************/
const char * F_DynamicBuffer::printBuffer()
{
	return (char *)m_psBuffer;
}
