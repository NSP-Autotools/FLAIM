//------------------------------------------------------------------------------
// Desc:	Interactive database shell
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
#include "fshell.h"

static FlmSharedContext *				gv_pSharedContext = NULL;
FLMBOOL										gv_bShutdown = FALSE;
FLMBOOL										gv_bRunning = TRUE;

#ifdef FLM_RING_ZERO_NLM
	#define main		nlm_main
#endif

/***************************************************************************
Desc:	Program entry point (main)
****************************************************************************/
extern "C" int main(
	int, //			iArgC,
	char **)		// ppucArgV
{
	RCODE						rc = NE_XFLM_OK;
	FlmShell *				pShell = NULL;
	IF_DbSystem *			pDbSystem = NULL;
	
	if( RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXInit( "X-FLAIM Shell", (FLMBYTE)80, (FLMBYTE)50,
		FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		goto Exit;
	}

	FTXSetShutdownFlag( &gv_bShutdown);

	if( (gv_pSharedContext = f_new FlmSharedContext) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_pSharedContext->init( NULL)))
	{
		goto Exit;
	}

	gv_pSharedContext->setShutdownFlag( &gv_bShutdown);

	if( (pShell = f_new FlmShell) != NULL)
	{
		if( RC_OK( pShell->setup( gv_pSharedContext)))
		{
			gv_pSharedContext->spawn( pShell);
		}
	}

	gv_pSharedContext->wait();

Exit:

	gv_bShutdown = TRUE;

	if( gv_pSharedContext)
	{
		gv_pSharedContext->Release();
	}

	FTXExit();
	
	if( pDbSystem)
	{
		pDbSystem->Release();
	}

	gv_bRunning = FALSE;
	return( 0);
}
