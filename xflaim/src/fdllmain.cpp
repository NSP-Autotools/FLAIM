//------------------------------------------------------------------------------
// Desc: This is the standard functionality that all com servers must export
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

static FLMATOMIC				gv_lockCount = 0;

XFLXPC RCODE XFLAPI DllCanUnloadNow( void);
XFLXPC RCODE XFLAPI DllStart( void);
XFLXPC RCODE XFLAPI DllStop( void);

#if defined( FLM_UNIX)

#elif defined( FLM_WIN)

	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	
	#pragma comment(linker, "/export:DllCanUnloadNow=_DllCanUnloadNow@0,PRIVATE")
	#pragma comment(linker, "/export:DllStart=_DllStart@0,PRIVATE")
	#pragma comment(linker, "/export:DllStop=_DllStop@0,PRIVATE")

#elif !defined( FLM_NLM)
	#error platform not supported.
#endif

/******************************************************************************
Desc:
******************************************************************************/
void LockModule(void)
{
	f_atomicInc( &gv_lockCount);
}

/******************************************************************************
Desc:
******************************************************************************/
void UnlockModule(void)
{
	f_atomicDec( &gv_lockCount);
}

/******************************************************************************
Desc:	Returns 0 if it's okay to unload, or a non-zero status
		code if not.
******************************************************************************/
XFLXPC RCODE XFLAPI DllCanUnloadNow( void)
{
	if( gv_lockCount > 1)
	{
		return( RC_SET( NE_XFLM_FAILURE));
	}

	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:	Called by PSA when it loads the library.  Must return 0 for
		success, or a non-zero error code.
******************************************************************************/
XFLXPC RCODE XFLAPI DllStart( void)
{
	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:	Called by PSA when it unloads the library.  The return value
		is ignored.
******************************************************************************/
XFLXPC RCODE XFLAPI DllStop( void)
{
	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
XFLXPC RCODE XFLAPI DllRegisterServer(
	const char *)
{
	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
XFLXPC RCODE XFLAPI DllUnregisterServer( void) 
{
	return( NE_XFLM_OK);
}
