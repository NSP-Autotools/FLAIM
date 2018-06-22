//-------------------------------------------------------------------------
// Desc:	Utility routines shared among various utilities - definitions.
// Tabs:	3
//
// Copyright (c) 1997-2001, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#ifndef SHARUTIL_H
#define SHARUTIL_H

#include "flaim.h"

#if defined( FLM_WIN)
	// Conversion from XXX to YYY, possible loss of data
	#pragma warning( disable : 4244) 

	// Local variable XXX may be used without having been initialized
	#pragma warning( disable : 4701)

	// Function XXX not inlined
	#pragma warning( disable : 4710) 

	// Unreferenced inline function has been removed
	#pragma warning (disable : 4514)
#endif

void flmUtilParseParams(
	char *			pszCommandBuffer,
	FLMINT			iMaxArgs,
	FLMINT *			iArgcRV,
	const char **	ppArgvRV);

RCODE  flmUtilStatusHook(
	FLMUINT			uiStatusType,
	void *			Parm1,
	void *			Parm2,
	void *			UserData);

#ifdef FLM_NLM
	#define flmUtilGiveUpCPU()     f_yieldCPU()
#else
	#define flmUtilGiveUpCPU()     f_sleep( 0)
#endif

#define TEST_ALLOC(ptr)									\
if ((ptr) == NULL)										\
{																\
	rc = RC_SET( FERR_MEM);								\
	goto Exit;												\
}
#define TEST_RC(rc)										\
if (RC_BAD( (rc)))										\
{																\
	goto Exit;												\
}
#define TEST_RC_LOCAL(rc)								\
if (RC_BAD( (rc)))										\
{																\
	goto Exit_local;										\
}
#define MAKE_BAD_RC_JUMP()								\
{																\
	rc = RC_SET( FERR_FAILURE);						\
	goto Exit;												\
}

/****************************************************************************
Name:	FlmVector
Desc:	treat this vector class like an array, except that you will never
		write to an item out-of-bounds.  This is because the vector
		dynamically allocates enough space to cover at least up through the
		index you are setting.  If you try to read out-of-bounds you will
		hit an assert rather than an access violation.  You will need to
		keep track of your own length, as there is no concept of "length"
		internal to this class.  You can exploit the fact that if you
		leave holes in the elements, the intermediate elements will
		be filled with 0's.
****************************************************************************/
class FlmVector : public F_Object
{
public:
	FlmVector()
	{
		m_pElementArray = NULL;
		m_uiArraySize = 0;
	}
	~FlmVector()
	{
		if ( m_pElementArray)
		{
			f_free( &m_pElementArray);
		}
	}
	RCODE setElementAt( void * pData, FLMUINT uiIndex);
	void * getElementAt( FLMUINT uiIndex);
private:
	void **	m_pElementArray;
	FLMUINT	m_uiArraySize;
};

/****************************************************************************
Name:	FlmStringAcc
Desc:	a class to safely build up a string accumulation, without worrying
		about buffer overflows.
****************************************************************************/
#define FSA_QUICKBUF_BUFFER_SIZE 128
class FlmStringAcc
{
public:

	FlmStringAcc()
	{
		commonInit();
	}
	
	FlmStringAcc( char * pszStr)
	{
		commonInit();
		this->appendTEXT( pszStr);
	}
	
	~FlmStringAcc()
	{
		if ( m_pszVal)
		{
			f_free( &m_pszVal);
		}
	}
	
	void clear()
	{
		if ( m_pszVal)
		{
			m_pszVal[ 0] = 0;
		}
		m_szQuickBuf[ 0] = 0;
		m_uiValStrLen = 0;
	}
	
	RCODE printf( const char * pszFormatString, ...);
	
	RCODE appendCHAR( char ucChar, FLMUINT uiHowMany = 1);
	
	RCODE appendTEXT( const char * pszVal);
	
	RCODE appendf( const char * pszFormatString, ...);
	
	const char * getTEXT()
	{
		if ( m_bQuickBufActive)
		{
			return( m_szQuickBuf);
		}
		else if( m_pszVal)
		{
			return( m_pszVal);
		}
		else
		{
			return( "");
		}
	}
	
private:

	void commonInit()
	{
		m_pszVal = NULL;
		m_uiValStrLen = 0;
		m_szQuickBuf[ 0] = 0;
		m_bQuickBufActive = FALSE;
	}
	
	RCODE formatNumber( FLMUINT uiNum, FLMUINT uiBase);
	
	char			m_szQuickBuf[ FSA_QUICKBUF_BUFFER_SIZE];
	FLMBOOL		m_bQuickBufActive;
	char *		m_pszVal;
	FLMUINT		m_uiBytesAllocatedForPszVal;
	FLMUINT		m_uiValStrLen;
};

void utilOutputLine(
	const char *		pszData,
	void *				pvUserData);

void utilPressAnyKey(
	const char *		pszPressAnyKeyMessage,
	void *				pvUserData);

RCODE utilInitWindow(
	const char *		pszTitle,
	FLMUINT *			puiScreenRows,
	FTX_WINDOW **		ppMainWindow,
	FLMBOOL *			pbShutdown);
	
void utilShutdownWindow( void);

FLMUINT utilGetTimeString(
	char *				pszOutString,
	FLMUINT				uiBufferSize,
	FLMUINT				uiInSeconds = 0);

RCODE utilWriteProperty(
	const char *		pszFile,
	const char *		pszProp,
	const char *		pszValue);

RCODE utilReadProperty(
	const char *		pszFile,
	const char *		pszProp,
	FlmStringAcc *		pAcc);

#endif
