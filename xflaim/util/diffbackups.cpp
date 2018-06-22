//------------------------------------------------------------------------------
// Desc: Check differences between backups
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

#include "dart_backup.h"
#include "ftx.h"
#include "flmarg.h"

FLMBOOL						gv_bShutdown = FALSE;
FSTATIC FLMBOOL			gv_bRunning = FALSE;
FSTATIC FTX_INFO *		gv_pFtxInfo = NULL;
FSTATIC FTX_WINDOW *		gv_pMainWindow = NULL;

FINLINE FLMBOOL breakCallback(
	void * pvData)
{
	F_UNREFERENCED_PARM( pvData);
	return FALSE;
}

FSTATIC RCODE utilMain( FLMUINT uiArgc, char ** ppszArgv);

#ifdef FLM_NLM

// Prototypes

FSTATIC void utilCleanup( void);

// End prototypes
											
/****************************************************************************
Desc: This routine shuts down all threads in the NLM.
****************************************************************************/
FSTATIC void utilCleanup(
	void
	)
{
	gv_bShutdown = TRUE;
	while( gv_bRunning)
	{
		f_yieldCPU();
	}
}
#endif

#ifdef FLM_NLM
FLMBOOL						gv_bSynchronized = FALSE;
#endif

#ifdef FLM_WATCOM_NLM
	#define main		nlm_main
#endif

/********************************************************************
Desc: main
*********************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
int main(
	int			iArgC,
	char **		ppucArgV
	)
#else
int __cdecl main(
	int			iArgC,
	char **		ppucArgV
	)
#endif   
{
	RCODE			rc = NE_XFLM_OK;
	int			iResCode = 0;


	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	
#ifdef FLM_NLM
	/* Setup the routine to be called when the NLM exits itself */
	atexit( utilCleanup);
#endif

	if ( RC_BAD( rc = dbSystem.init()))
	{
		goto Exit;
	}

	//main code which varies from util to util goes here
	rc = utilMain( iArgC, ppucArgV);

Exit:

	dbSystem.exit();
	
#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
	gv_bRunning = FALSE;
#endif

	if ( iResCode == 0)
	{
		iResCode = (int)rc;
	}
	return( iResCode);
}

/****************************************************************************
NOTE:  UTILITY-SPECIFIC CODE STARTS HERE
****************************************************************************/

/****************************************************************************
Name:	utilMain
Desc:	the 'main'-type method which varies per utility
****************************************************************************/
FSTATIC RCODE utilMain( FLMUINT uiArgc, char ** ppszArgv)
{
	RCODE				rc = NE_XFLM_OK;
	FlmArgSet *		pArgSet = NULL;
	FLMBOOL			bPrintedUsage;
	FLMBOOL			bBatchMode = FALSE;
	FLMUINT			uiScreenRows;

	char				szBackupRoot[128];
	char				szRflRoot[128];
	char				szDestDib1[16];
	char				szDestDib2[16];

	FLMUINT			uiSet1 = 0;
	FLMUINT			uiSet2 = 0;

	FLMUINT64		ui64LastTrans = 0;

	F_RandomGenerator	randomGen;

	randomGen.randomSetSeed( 123);

	//initialize the windowing
	TEST_RC( rc = utilInitWindow( "ezutil", &uiScreenRows,
		&gv_pFtxInfo, &gv_pMainWindow, &gv_bShutdown));
	
	if ( (pArgSet = new FlmArgSet(
		"diff backups test",	//description of utility
		utilOutputLine, gv_pMainWindow,		//output callback and data
		utilPressAnyKey, gv_pMainWindow,		//pager callback and data
		uiScreenRows))								//rows per screen
		== NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}



	TEST_RC( rc = pArgSet->addArg(
		"root", "root directory where backups are",
		TRUE, FLMARG_OPTION, FLMARG_CONTENT_STRING));

	TEST_RC( rc = pArgSet->addArg(
		"rflroot", "root directory where rfl backups are",
		TRUE, FLMARG_OPTION, FLMARG_CONTENT_STRING));

	TEST_RC( rc = pArgSet->addArg(
		"firstDb", "name of first database",
		FALSE, FLMARG_OPTION, FLMARG_CONTENT_STRING));

	TEST_RC( rc = pArgSet->addArg(
		"secondDb", "name of second database",
		FALSE, FLMARG_OPTION, FLMARG_CONTENT_STRING));

#ifdef FLM_LINUX
	TEST_RC( rc = pArgSet->addArg(
		"trans", "last trans to replay",
		FALSE, FLMARG_OPTION, FLMARG_CONTENT_UNSIGNED_INT_64, (FLMUINT64)0, 0xFFFFFFFFFFFFFFFFLL));
#else
	TEST_RC( rc = pArgSet->addArg(
		"trans", "last trans to replay",
		FALSE, FLMARG_OPTION, FLMARG_CONTENT_UNSIGNED_INT_64, (FLMUINT64)0, 0xFFFFFFFFFFFFFFFF));
#endif

	//options
#ifdef FLM_NLM
	TEST_RC( rc = pArgSet->addArg(
		"waitToSync", "wait to sync on Netware",
		TRUE, FLMARG_OPTION, FLMARG_CONTENT_NONE));
#endif

	//required args
	TEST_RC( rc = pArgSet->addArg(
		"firstSet", "first backup set number",
		FALSE, FLMARG_REQUIRED_ARG, FLMARG_CONTENT_UNSIGNED_INT,
		0, 0xFFFFFFFF));

	TEST_RC( rc = pArgSet->addArg(
		"secondSet", "second backup set number",
		FALSE, FLMARG_REQUIRED_ARG, FLMARG_CONTENT_UNSIGNED_INT,
		0, 0xFFFFFFFF));

	//feed in the true command line
	TEST_RC( rc = pArgSet->parseCommandLine( uiArgc, ppszArgv, &bPrintedUsage));

#ifdef FLM_NLM
	if (!gv_bSynchronized && !(pArgSet->argIsPresent( "waitToSync")))
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif

	if( pArgSet->argIsPresent("root"))
	{
		f_strcpy( szBackupRoot, pArgSet->getString("root"));
	}
	else
	{
		f_strcpy( szBackupRoot, "hcbstage\\backups");
	}

	if( pArgSet->argIsPresent("rflroot"))
	{
		f_strcpy( szRflRoot, pArgSet->getString("rflroot"));
	}
	else
	{
		f_strcpy( szRflRoot, "hcbstage\\rfls");
	}

	if( pArgSet->argIsPresent("firstDb"))
	{
		f_strcpy( szDestDib1, pArgSet->getString("firstDb"));
	}
	else
	{
		f_strcpy( szDestDib1, "db1.db");
	}

	if( pArgSet->argIsPresent("secondDb"))
	{
		f_strcpy( szDestDib2, pArgSet->getString("secondDb"));
	}
	else
	{
		f_strcpy( szDestDib2, "db2.db");
	}

	if ( pArgSet->argIsPresent("firstSet"))
	{
		uiSet1 = pArgSet->getUINT( "firstSet");
	}
	else
	{
		flmAssert(0);
	}

	if ( pArgSet->argIsPresent("secondSet"))
	{
		uiSet2 = pArgSet->getUINT( "secondSet");
	}
	else
	{
		flmAssert(0);
	}

	if( pArgSet->argIsPresent("trans"))
	{
		ui64LastTrans = pArgSet->getUINT64( "trans");
	}

	if ( RC_BAD( rc = utilDiffBackupSets(
		szBackupRoot,
		szRflRoot,
		szDestDib1,
		szDestDib2,
		&randomGen,
		utilOutputLine,
		gv_pMainWindow,
		breakCallback,
		gv_pMainWindow,
		uiSet1,		
		uiSet2,
		ui64LastTrans)))
	{
		goto Exit;
	}
	
Exit:

	if ( !bBatchMode)
	{
		utilPressAnyKey( "press any key to exit...", gv_pMainWindow);
	}
	if ( pArgSet)
	{
		pArgSet->Release();
	}
	utilShutdownWindow( gv_pFtxInfo);
	return rc;
}
