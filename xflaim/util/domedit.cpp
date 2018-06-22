//------------------------------------------------------------------------------
// Desc: Standalone DOM editor utility
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
#include "ftx.h"
#include "domedit.h"

#if defined( FLM_UNIX)
	#include <ctype.h>
#endif

// NetWare hooks

#ifdef FLM_NLM
		FLMBOOL	gv_bSynchronized = FALSE;
	void domEditCleanup( void);
#endif

/*
DOMEdit prototypes
*/

void UIMain( void * pData);

RCODE _domEditBackgroundThread(
	IF_Thread *			pThread);

RCODE domEditVerifyRun( void);

/*
Imported global data
*/

#ifdef FLM_DEBUG
	extern RCODE		gv_CriticalFSError;
#endif

/*--------------------------------------------------------
** Local (to this file only) global variables.
**-------------------------------------------------------*/

FTX_INFO *									gv_pFtxInfo = NULL;
FLMBOOL										gv_bShutdown = FALSE;
static IF_Thread *						gv_pBackgroundThrd = NULL;
char											gv_szDbPath[ F_PATH_MAX_SIZE];
char											gv_szRflDir[ F_PATH_MAX_SIZE];
char 											gv_szPassword[ 128];
FLMBOOL										gv_bAllowLimited;
static FLMBOOL								gv_bRunning = TRUE;

#ifdef FLM_WATCOM_NLM
	#define main		nlm_main
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
extern "C" int main(
	int				ArgC,
	char **			ArgV)
#else
int __cdecl main(
	int				ArgC,
	char **			ArgV)
#endif
{
	int				iResCode = 0;

#if defined( FLM_UNIX)
	struct rlimit l;

	if (getrlimit(RLIMIT_NOFILE, &l) < 0)
	{
		fprintf(stderr, "Could not get the maximum number of open files: %s",
				strerror(errno));
		exit(1);
	}

	if (geteuid() == 0)
		l.rlim_max = 65536;		// big enough for our needs

	l.rlim_cur = l.rlim_max;

    // increase the fd table
	if (setrlimit(RLIMIT_NOFILE, &l) < 0)
		fprintf(stderr, "Could not increase the number of open files to %ld",
				l.rlim_max);
#endif

#ifdef FLM_NLM

	/* Setup the routines to be called when the NLM exits itself */

	atexit( domEditCleanup);

	/* Register to see the DOWN server event. */

	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif

	gv_szDbPath[ 0] = 0;
	gv_szRflDir[ 0] = 0;
	gv_szPassword[0] = 0;
	gv_bAllowLimited = FALSE;

	if( ArgC >= 2)
	{
		f_strcpy( gv_szDbPath, ArgV[ 1]);
	}
	if( ArgC >= 3)
	{
		f_strcpy( gv_szRflDir, ArgV[ 2]);
	}
	
	if (ArgC >= 4)
	{
		f_strcpy( gv_szPassword, ArgV[ 3]);
	}

	if (ArgC >=5)
	{
		if (f_strnicmp( ArgV[ 4], "TRUE", 4) == 0)
		{
			gv_bAllowLimited = TRUE;
		}
	}

	UIMain( NULL);

//Exit:

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif
	gv_bRunning = FALSE;
	

	return( iResCode);
}

/****************************************************************************
Name: UIMain
****************************************************************************/
void UIMain( void * pData)
{
	F_Db *			pDb = NULL;
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	F_DomEditor	*	pDomEditor = NULL;
	char				szTitle[ 80];
	FLMUINT			uiDummy;
	char				szDbPath [F_PATH_MAX_SIZE];
	FLMUINT			Cols;
	FLMUINT			Rows;
	RCODE				rc;
	int				iResCode = 0;

	F_UNREFERENCED_PARM( pData);

	if( RC_BAD( dbSystem.init()))
	{
		iResCode = -1;
		goto Exit;
	}

	f_sprintf( szTitle,
		"DOMEdit for XFLAIM [DB=%s/BUILD=%s]",
		XFLM_CURRENT_VER_STR, __DATE__);

	if( RC_BAD( FTXInit( szTitle, 80, 50, FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		iResCode = 1;
		goto Exit;
	}

	FTXSetShutdownFlag( gv_pFtxInfo, &gv_bShutdown);


	if( FTXScreenInit( gv_pFtxInfo, szTitle, &pScreen) != FTXRC_SUCCESS)
	{
		iResCode = 1;
		goto Exit;
	}

	if( FTXWinInit( pScreen, 0, 1, &pTitleWin) != FTXRC_SUCCESS)
	{
		iResCode = 1;
		goto Exit;
	}

	if( FTXWinPaintBackground( pTitleWin, FLM_RED) != FTXRC_SUCCESS)
	{
		iResCode = 1;
		goto Exit;
	}

	if( FTXWinPrintStr( pTitleWin, szTitle) != FTXRC_SUCCESS)
	{
		iResCode = 1;
		goto Exit;
	}

	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);

	if( FTXWinOpen( pTitleWin) != FTXRC_SUCCESS)
	{
		iResCode = 1;
		goto Exit;
	}


	if( RC_BAD( pThreadMgr->createThread( &gv_pBackgroundThrd,
		_domEditBackgroundThread, "domedit_refresh")))
	{
		iResCode = 1;
		goto Exit;
	}

	/*
	Check expiration date
	*/

	if( RC_BAD( rc = domEditVerifyRun()))
	{
		FTXDisplayMessage( pScreen, FLM_RED, FLM_WHITE,
			"This Utility Has Expired",
			"NE_XFLM_ILLEGAL_OP", &uiDummy);
		f_sleep( 5000);
		iResCode = 1;
		goto Exit;
	}

	/*
	Open the database
	*/

	if( gv_szDbPath[ 0])
	{

		if( RC_BAD( rc = dbSystem.dbOpen( gv_szDbPath, NULL, gv_szRflDir,
			(IF_Db **)&pDb, gv_szPassword, gv_bAllowLimited)))
		{
			char	szErr [20];
			
			f_sprintf( szErr, "Error=0x%04X", (unsigned)rc);
			FTXDisplayMessage( pScreen, FLM_RED, FLM_WHITE,
				"Unable to open the database", szErr, &uiDummy);
			iResCode = 1;
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = dbSystem.dbOpen( szDbPath, NULL, gv_szRflDir,
			(IF_Db **)&pDb, gv_szPassword, gv_bAllowLimited)))
		{
			char	szErr [20];
			
			f_sprintf( szErr, "Error=0x%04X", (unsigned)rc);
			FTXDisplayMessage( pScreen, FLM_RED, FLM_WHITE,
				"Unable to open the database", szErr, &uiDummy);
			iResCode = 1;
			goto Exit;
		}
		else
		{
			FTXWinClear( pTitleWin);
			if( FTXWinPrintf( pTitleWin, "%s (Direct)", szTitle) != FTXRC_SUCCESS)
			{
				iResCode = 1;
				goto Exit;
			}
		}
	}

	if( (pDomEditor = f_new F_DomEditor) == NULL)
	{
		iResCode = 1;
		goto Exit;
	}

	if( RC_BAD( pDomEditor->Setup( pScreen)))
	{
		iResCode = 1;
		goto Exit;
	}

	pDomEditor->setSource( pDb, XFLM_DATA_COLLECTION);
	pDomEditor->setShutdown( &gv_bShutdown);

	/*
	Fire up the editor
	*/

	FTXScreenGetSize( pScreen, &Cols, &Rows);
	pDomEditor->interactiveEdit( 0, 1, Cols - 1, Rows - 1);





Exit:

	if( pDomEditor)
	{
		pDomEditor->Release();
		pDomEditor = NULL;
	}

	gv_bShutdown = TRUE;
	
	if (pDb)
	{
		pDb->Release();
	}

	if( gv_pBackgroundThread)
	{
		gv_pBackgroundThrd->Release();
	}
	
	if( pThreadMgr)
	{
		pThreadMgr->Release();
	}

	dbSystem.exit();
}


#ifdef FLM_NLM
/****************************************************************************
Desc: This routine shuts down all threads in the NLM.
****************************************************************************/
void domEditCleanup( void)
{
	gv_bShutdown = TRUE;
	while( gv_bRunning)
	{
		f_sleep( 10);
	}
}
#endif

