//------------------------------------------------------------------------------
// Desc:	Dynamic buffer
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

/****************************************************************************
Desc:	Utility class to help prepare a character buffer for printing when the
		output buffer size is not known in advance.
*****************************************************************************/
class F_DynamicBuffer : public F_Object
{
public:

	F_DynamicBuffer()
	{
		m_bSetup = FALSE;
		m_psBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
		if (RC_OK( f_mutexCreate( &m_hMutex)))
		{
			m_bSetup = TRUE;
		}
	}

	~F_DynamicBuffer()
	{
		f_free( &m_psBuffer);
		m_psBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
		if (m_bSetup)
		{
			f_mutexDestroy( &m_hMutex);
			m_bSetup = FALSE;
		}
	}

	RCODE addChar( FLMBYTE ucCharacter);

	RCODE addString( const char * pszString);

	const char * printBuffer();

	FLMUINT getBufferSize( void)
	{
		return m_uiUsedChars;
	}

	void reset( void)
	{
		f_free( &m_psBuffer);
		m_psBuffer = NULL;
		m_uiBuffSize = 0;
		m_uiUsedChars = 0;
	}

private:

	FLMBOOL		m_bSetup;
	FLMBYTE *	m_psBuffer;
	FLMUINT		m_uiBuffSize;
	FLMUINT		m_uiUsedChars;
	F_MUTEX		m_hMutex;
};
