//-------------------------------------------------------------------------
// Desc:	Gigatest
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

#include "flaim.h"
#include "flm_lutl.h"
#include "gigatest.h"

// Columns/Rows where things go on the screen.

#define LABEL_COLUMN					5
#define DATA_COLUMN					35

#define MAX_CACHE_ROW				1
#define USED_CACHE_ROW				2
#define ITEMS_CACHED_ROW			3
#define DIRTY_CACHE_ROW				4
#define LOG_CACHE_ROW				5
#define FREE_CACHE_ROW				6
#define CP_STATE_ROW					7
#define DB_NAME_ROW					8
#define TOTAL_TO_LOAD_ROW			9
#define TRANS_SIZE_ROW				10
#define TOTAL_LOADED_ROW			11
#define ADDS_PER_SEC_CURRENT		12
#define ADDS_PER_SEC_OVERALL		13
#define ELAPSED_TIME_ROW			14

char						gv_szDibName[ 200];
char						gv_szDataDir[ 200];
char						gv_szRflDir[ 200];
char						gv_szDirectoryPath[ F_PATH_MAX_SIZE];
char						gv_pszFileName[ F_FILENAME_SIZE];
FLMUINT					gv_ui10Secs;
FLMUINT					gv_ui1Sec;
IF_Thread *				gv_pScreenThrd = NULL;
FLMUINT					gv_uiCacheSize;
FLMUINT         		gv_uiBlockCachePercentage;
FLMUINT					gv_uiMaxDirtyCache;
FLMUINT					gv_uiLowDirtyCache;
FLMUINT					gv_uiTotalToLoad;
FLMUINT					gv_uiTransSize;
FLMUINT					gv_uiTotalLoaded;
FLMUINT					gv_ui10SecTotal;
FLMUINT					gv_ui10SecStartTime;
FLMUINT					gv_uiStartTime;
HFDB						gv_hDb;
FLMBOOL					gv_bShutdown;
FLMBOOL					gv_bRunning;
FTX_SCREEN *			gv_pScreen;
FTX_WINDOW *			gv_pWindow;
F_MUTEX					gv_hWindowMutex;
char *					gv_pszTitle;
FLMUINT					gv_uiNumCols;
FLMUINT					gv_uiNumRows;
FLMUINT					gv_uiMaxMemory;
FLMUINT					gv_uiCPInterval;
FLMUINT					gv_uiNumFields;
FLMUINT					gv_uiPreallocSpace;
const char **			gv_ppszCurrGiven;
const char **			gv_ppszCurrFamily;
IF_Thread *				gv_pIxManagerThrd;
FLMBOOL					gv_bBatchMode;
FLMBOOL					gv_bDisableDirectIO;
#ifdef FLM_NLM
FLMBOOL					gv_bSynchronized;
#endif

const char * gv_pszGigaDictionary =	
	"0 @1@ field Person\n"
	" 1 type text\n"
	"0 @2@ field LastName\n"
	" 1 type text\n"
	"0 @3@ field FirstName\n"
	" 1 type text\n"
	"0 @4@ field Age\n"
	" 1 type number\n"
	"0 @100@ index LastFirst_IX\n"
	" 1 language US\n"
	" 1 key\n"
	"  2 field 2\n"
	"   3 required\n"
	"  2 field 3\n"
	"   3 required\n";

#define PERSON_TAG						1
#define LAST_NAME_TAG					2
#define FIRST_NAME_TAG					3
#define AGE_TAG							4
#define LAST_NAME_FIRST_NAME_IX		100

FLMUINT gigaGetInput(
	const char *		pszMsg1,
	const char *		pszMsg2,
	FLMBOOL				bMutexLocked = FALSE);

void gigaOutputErrMsg(
	const char *		pszErrMsg,
	FLMBOOL				bMutexLocked = FALSE);

void gigaOutputRcErr(
	const char *		pszWhat,
	RCODE					rc,
	FLMBOOL				bMutexLocked = FALSE);

void gigaOutputLabel(
	FLMUINT				uiRow,
	const char *		pszLabel,
	FLMBOOL				bMutexLocked = FALSE);

void gigaOutputStr(
	FLMUINT				uiRow,
	const char *		pszStr,
	FLMBOOL				bMutexLocked = FALSE);

void gigaOutputUINT(
	FLMUINT				uiRow,
	FLMUINT				uiNum,
	FLMBOOL				bMutexLocked = FALSE);
	
#ifdef FLM_NLM
	extern "C"
	{
		void SynchronizeStart();

		int nlm_main(
			int		ArgC,
			char **	ArgV);

		int atexit( void (*)( void ) );
	}

	void gigaCleanup( void);
#endif

void gigaInitGlobalVars( void);

void gigaShowHelp( void);

FLMBOOL gigaGetParams(
	FLMINT				iArgC,
	const char **		ppszArgV);

FLMUINT gigaSeeIfQuit( void);

RCODE gigaStartTrans( void);

RCODE gigaMakeNewRecord(
	FlmRecord **		ppRecord);

RCODE gigaCommitTrans( void);

void gigaUpdateMemInfo( void);

RCODE gigaLoadDatabase( void);

void gigaCheckpointDisplay( void);

RCODE FLMAPI gigaScreenThread(
	IF_Thread *			pThread);

RCODE gigaStartScreenThread( void);

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_UNIX)
int main(
	int			iArgC,
	char **		ppszArgV)
#elif defined( FLM_NLM)
int nlm_main(
	int			iArgC,
	char **		ppszArgV)
#else
int __cdecl main(
	int			iArgC,
	char **		ppszArgV)
#endif   
{
	int			iRetCode = 0;

	gigaInitGlobalVars();

#ifdef FLM_NLM

	// Setup the routines to be called when the NLM exits itself
	
	atexit( gigaCleanup);

#endif

	if( RC_BAD( FlmStartup()))
	{
		iRetCode = 1;
		goto Exit;
	}

	if( RC_BAD( FTXInit( gv_pszTitle, (FLMBYTE)80, (FLMBYTE)50,
					FLM_BLUE, FLM_LIGHTGRAY, NULL, NULL)))
	{
		iRetCode = 1;
		goto Exit;
	}

	FTXSetShutdownFlag( &gv_bShutdown);

	if( RC_BAD( FTXScreenInit( gv_pszTitle, &gv_pScreen)))
	{
		iRetCode = 1;
		goto Exit;
	}

	if( RC_BAD( FTXScreenInitStandardWindows( gv_pScreen, FLM_RED, FLM_WHITE,
		FLM_BLUE, FLM_WHITE, FALSE, TRUE, gv_pszTitle, NULL, &gv_pWindow)))
	{
		iRetCode = 1;
		goto Exit;
	}
	
	FTXWinGetCanvasSize( gv_pWindow, &gv_uiNumCols, &gv_uiNumRows);

	if( RC_BAD( f_mutexCreate( &gv_hWindowMutex)))
	{
		iRetCode = 99;
		goto Exit;
	}

	if( !gigaGetParams( iArgC, (const char **)ppszArgV))
	{
		iRetCode = 2;
		goto Exit;
	}

	f_pathReduce( gv_szDibName, gv_szDirectoryPath, gv_pszFileName);
	if( !gv_szDirectoryPath [0])
	{
		f_strcpy( gv_szDirectoryPath, ".");
	}
	
	if( RC_BAD( gigaLoadDatabase()))
	{
		iRetCode = 7;
		goto Exit;
	}

	if( !gv_bBatchMode)
	{
		gigaOutputErrMsg( "Load complete");
	}

Exit:

	if( gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hWindowMutex);
	}
	
	FTXExit();
	FlmShutdown();

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif

	gv_bRunning = FALSE;
	return( iRetCode);
}

/********************************************************************
Desc:	Initialize global variables.
*********************************************************************/
void gigaInitGlobalVars( void)
{
	gv_hDb = HFDB_NULL;
	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	gv_pScreen = NULL;
	gv_pWindow = NULL;
	gv_hWindowMutex = F_MUTEX_NULL;
	gv_pszTitle = (char *)"GigaTest 1.0 for FLAIM";
	gv_uiNumCols = 0;
	gv_uiNumRows = 0;
	gv_uiMaxMemory = 0;
	gv_uiCPInterval = 0xFFFFFFFF;
	gv_uiNumFields = 0;
	gv_uiPreallocSpace = 0;
	gv_ppszCurrGiven = &gv_pszGivenNames [0];
	gv_ppszCurrFamily = &gv_pszFamilyNames [0];
	gv_pIxManagerThrd = NULL;
	gv_bBatchMode = FALSE;
	gv_bDisableDirectIO = FALSE;
#ifdef FLM_NLM
	gv_bSynchronized = FALSE;
#endif
}

/********************************************************************
Desc:	Show help for the gigaloader.
*********************************************************************/
void gigaShowHelp( void)
{
#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif
	f_mutexLock( gv_hWindowMutex);
	FTXWinClear( gv_pWindow);
	FTXWinPrintStr( gv_pWindow, "\n");
	FTXWinPrintStr( gv_pWindow, 
"Parameters: [Number To Create] [Options]\n\n");
	FTXWinPrintStr( gv_pWindow, 
"\nNumber To Create\n");
	FTXWinPrintStr( gv_pWindow, 
"   Number of object to create (default = 100,000)\n");
	FTXWinPrintStr( gv_pWindow, 
"\nOptions (may be specified anywhere on command line):\n");
	FTXWinPrintStr( gv_pWindow, 
"   -b         = Run in batch mode.\n");
	FTXWinPrintStr( gv_pWindow, 
"   -c<n>      = Cache (bytes) to use, 0=Use default mode\n");
	FTXWinPrintStr( gv_pWindow, 
"   -p<n>      = Block cache percentage (0-100) to use (default 50)\n");
	FTXWinPrintStr( gv_pWindow, 
"   -i<n>      = Checkpoint interval (seconds) to use.\n");
#ifdef FLM_NLM
	FTXWinPrintStr( gv_pWindow, 
"   -n<DbName> = Database name (default = sys:\\_netware\\gigatest.db).\n");
#else
	FTXWinPrintStr( gv_pWindow, 
"   -n<DbName> = Database name (default = gigatest.db).\n");
#endif
	FTXWinPrintStr( gv_pWindow, 
"   -dr<Dir>   = Directory where rfl files are located (default=same as db)\n");
	FTXWinPrintStr( gv_pWindow, 
"   -dd<Dir>   = Directory where data files are located (default=same as db)\n");
	FTXWinPrintStr( gv_pWindow, 
"   -t<n>      = Transaction Size (objects per transaction, default=100).\n");
#ifdef FLM_NLM
	FTXWinPrintStr( gv_pWindow, 
"   -w         = Wait to end to synchronize\n");
#endif
	FTXWinPrintStr( gv_pWindow, 
"   -md<n>     = Set maximum dirty cache (bytes), 0=Use default mode\n");
	FTXWinPrintStr( gv_pWindow, 
"   -ld<n>     = Set low dirty cache (bytes), default=0\n");
	FTXWinPrintStr( gv_pWindow, 
"   -?         = A '?' anywhere in the command line will cause this help\n");
	FTXWinPrintStr( gv_pWindow, 
"                screen to be displayed, with or without the leading '-'.\n");
	f_mutexUnlock( gv_hWindowMutex);
	gigaOutputErrMsg( "");
}

/********************************************************************
Desc:	Get command line parameters.
*********************************************************************/
FLMBOOL gigaGetParams(
	FLMINT			iArgC,
	const char **	ppszArgV)
{
	FLMBOOL			bOk = FALSE;
	FLMUINT			uiLoop;
	const char *	pszPtr;
	FLMBOOL			bHaveNumToLoad = FALSE;
#ifdef FLM_NLM
	FLMBOOL			bWaitToSync = FALSE;
#endif
	char				szMsg [100];

	gv_uiCacheSize = 0;
	gv_uiBlockCachePercentage = 50;
	gv_uiMaxDirtyCache = 0;
	gv_uiLowDirtyCache = 0;
	gv_uiTotalToLoad = 100000;
	gv_uiTransSize = 100;
	gv_uiTotalLoaded = 0;
#ifdef FLM_NLM
	f_strcpy( gv_szDibName, "sys:\\_netware\\gigatest.db");
#else
	f_strcpy( gv_szDibName, "gigatest.db");
#endif
	gv_szDataDir[ 0] = 0;
	gv_szRflDir[ 0] = 0;

	// If no parameters were entered, show a help screen.

	if( iArgC < 2)
	{
		gigaShowHelp();
		goto Exit;
	}

	uiLoop = 1;
	while( uiLoop < (FLMUINT)iArgC)
	{
		pszPtr = ppszArgV [uiLoop];

		// See if they specified an option

#ifdef FLM_UNIX
		if( *pszPtr == '-')
#else
		if( (*pszPtr == '-') || (*pszPtr == '/'))
#endif
		{
			pszPtr++;
			if ((*pszPtr == 'c') || (*pszPtr == 'C'))
			{
				gv_uiCacheSize = f_atol( (pszPtr + 1));
			}
			else if ((*pszPtr == 'i') || (*pszPtr == 'I'))
			{
				gv_uiCPInterval = f_atol( (pszPtr + 1));
			}
			else if ((*pszPtr == 'p') || (*pszPtr == 'P'))
			{
				gv_uiBlockCachePercentage = f_atol( (pszPtr + 1));
			}
			else if ((*pszPtr == 'm') || (*pszPtr == 'M'))
			{
				pszPtr++;
				if ((*pszPtr == 'd') || (*pszPtr == 'D'))
				{
					gv_uiMaxDirtyCache = f_atol( (pszPtr + 1));
				}
			}
			else if ((*pszPtr == 'l') || (*pszPtr == 'L'))
			{
				pszPtr++;
				if ((*pszPtr == 'd') || (*pszPtr == 'D'))
				{
					gv_uiLowDirtyCache = f_atol( (pszPtr + 1));
				}
			}
			else if ((*pszPtr == 't') || (*pszPtr == 'T'))
			{
				gv_uiTransSize = f_atol( (pszPtr + 1));
			}
			else if ((*pszPtr == 'n') || (*pszPtr == 'N'))
			{
				f_strcpy( gv_szDibName, pszPtr + 1);
			}
			else if (*pszPtr == 'd' || *pszPtr == 'D')
			{
				if( f_stricmp( pszPtr, "dio") == 0)
				{
					gv_bDisableDirectIO = TRUE;
				}
				else
				{
					pszPtr++;
					if (*pszPtr == 'r' || *pszPtr == 'R')
					{
						f_strcpy( gv_szRflDir, pszPtr + 1);
					}
					else if (*pszPtr == 'd' || *pszPtr == 'D')
					{
						f_strcpy( gv_szDataDir, pszPtr + 1);
					}
					else
					{
						f_sprintf( szMsg, "Invalid option %s", (pszPtr - 1));
						gigaOutputErrMsg( szMsg);
						goto Exit;
					}
				}
			}
			else if (f_stricmp( pszPtr, "B") == 0)
			{
				gv_bBatchMode = TRUE;
			}
#ifdef FLM_NLM
			else if (f_stricmp( pszPtr, "W") == 0)
			{
				bWaitToSync = TRUE;
			}
#endif
			else if (f_stricmp( pszPtr, "?") == 0 ||
						f_stricmp( pszPtr, "HELP") == 0)
			{
				gigaShowHelp();
				goto Exit;
			}
			else
			{
				f_sprintf( szMsg, "Invalid option %s", pszPtr);
				gigaOutputErrMsg( szMsg);
				goto Exit;
			}
		}
		else if (f_stricmp( pszPtr, "?") == 0)
		{
			gigaShowHelp();
			return( FALSE);
		}
		else if (!bHaveNumToLoad)
		{
			gv_uiTotalToLoad = f_atol( pszPtr);
			bHaveNumToLoad = TRUE;
		}
		uiLoop++;
	}

#ifdef FLM_NLM
	if (!bWaitToSync && !gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif

	bOk = TRUE;
Exit:
	return( bOk);
}

/********************************************************************
Desc: Checks to see if the user pressed ESCAPE to exit the loader.
		Also updates the total loaded counter on the screen.
*********************************************************************/
FLMUINT gigaSeeIfQuit( void)
{
	FLMUINT	uiChar = 0;;

	f_mutexLock( gv_hWindowMutex);
	gigaOutputUINT( TOTAL_LOADED_ROW, gv_uiTotalLoaded, TRUE);
	if (RC_OK( FTXWinTestKB( gv_pWindow)))
	{
		FTXWinInputChar( gv_pWindow, &uiChar);
		if (uiChar == FKB_ESCAPE)
		{
			uiChar = gigaGetInput(
							"ESCAPE pressed, quit? (ESC,Q,Y=Quit, other=continue): ",
							NULL, TRUE);
			switch (uiChar)
			{
				case 'Q':
				case 'q':
				case 'y':
				case 'Y':
					uiChar = FKB_ESCAPE;
					break;
				case FKB_ESCAPE:
					break;
				default:
					uiChar = 0;
					break;
			}
		}
		else if( uiChar == 'i' || uiChar == 'I')
		{
			HFDB			hDb;

			f_threadDestroy( &gv_pIxManagerThrd);
			if (RC_OK( FlmDbOpen( gv_szDibName, gv_szDataDir,
										gv_szRflDir, 0, NULL, &hDb)))
			{
				f_threadCreate( &gv_pIxManagerThrd,
					flstIndexManagerThread, "index_manager",
					F_DEFAULT_THREAD_GROUP, 0, (void *)hDb);
			}
		}
	}

	if (gv_bShutdown)
	{
		uiChar = FKB_ESCAPE;
	}
	
	f_mutexUnlock( gv_hWindowMutex);
	return( uiChar);
}

/********************************************************************
Desc: Starts a transaction and does a few modifications that are
		necessary at the beginning of a transaction.
*********************************************************************/
RCODE gigaStartTrans( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiChar;

	while( !gv_bShutdown)
	{
		if( RC_BAD( rc = FlmDbTransBegin( gv_hDb, 
			FLM_UPDATE_TRANS, FLM_NO_TIMEOUT)))
		{
			if( rc != FERR_MUST_WAIT_CHECKPOINT)
			{
				gigaOutputRcErr( "starting transaction", rc);
				goto Exit;
			}
			
			f_yieldCPU();
			
			if( (uiChar = gigaSeeIfQuit()) != 0)
			{
				if (uiChar == FKB_ESCAPE)
				{
					gigaOutputRcErr( "starting transaction", rc);
					goto Exit;
				}
			}
		}
		else
		{
			break;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE gigaMakeNewRecord(
	FlmRecord **		ppRecord)
{
	RCODE					rc = NE_FLM_OK;
	FlmRecord *			pRecord = NULL;
	void *				pvField;

	if( *ppRecord && !(*ppRecord)->isReadOnly())
	{
		f_assert( (*ppRecord)->getRefCount() == 1);
		pRecord = *ppRecord;
		*ppRecord = NULL;
		pRecord->clear();
	}
	else
	{
		if( (pRecord = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}

	if( RC_BAD( rc = pRecord->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->insertLast( 1, FIRST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->setNative( pvField, *gv_ppszCurrGiven)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->insertLast( 1, LAST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->setNative( pvField, *gv_ppszCurrFamily)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->insertLast( 1, AGE_TAG,
		FLM_NUMBER_TYPE, &pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->setUINT( pvField, f_getRandomUINT32( 1, 100))))
	{
		goto Exit;
	}

	if( *ppRecord)
	{
		(*ppRecord)->Release();
	}
	
	*ppRecord = pRecord;
	pRecord = NULL;
	
	gv_ppszCurrGiven++;
	if( *gv_ppszCurrGiven == NULL)
	{
		gv_ppszCurrGiven = &gv_pszGivenNames [0];
		gv_ppszCurrFamily++;
		
		if (*gv_ppszCurrFamily == NULL)
		{
			gv_ppszCurrFamily = &gv_pszFamilyNames[0];
		}
	}
	
Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	return( rc);
}

/********************************************************************
Desc: Commits the current transaction - fixes up the parent object to
		point to the last child added before committing.
*********************************************************************/
RCODE gigaCommitTrans( void)
{
	RCODE				rc = NE_FLM_OK;

	gigaOutputUINT( TOTAL_LOADED_ROW, gv_uiTotalLoaded);

	// Commit the transaction.

	if( RC_BAD( rc =  FlmDbTransCommit( gv_hDb)))
	{
		gigaOutputRcErr( "committing transaction", rc);
		goto Exit;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Update the memory information on the screen.
*********************************************************************/
void gigaCheckpointDisplay( void)
{
	char					szBuf[ 80];
	CHECKPOINT_INFO	cpInfo;

	FlmDbGetConfig( gv_hDb, FDB_GET_CHECKPOINT_INFO, &cpInfo, NULL, NULL);
	if (!cpInfo.bRunning)
	{
		f_strcpy( szBuf, "Idle                  ");
	}
	else
	{
		if (cpInfo.bForcingCheckpoint)
		{
			f_sprintf( szBuf, "Forcing (%ums)          ",
				(unsigned)cpInfo.uiRunningTime);
		}
		else
		{
			f_sprintf( szBuf, "Running (%ums)           ",
				(unsigned)cpInfo.uiRunningTime);
		}
	}
	
	gigaOutputStr( CP_STATE_ROW, szBuf);
}

/********************************************************************
Desc: Update the memory information on the screen.
*********************************************************************/
void gigaUpdateMemInfo( void)
{
	FLM_MEM_INFO	MemInfo;
	char				szBuf [50];

	FlmGetMemoryInfo( &MemInfo);
	
	f_sprintf( (char *)szBuf, "Blk: %-10u  Record: %-10u",
								(unsigned)MemInfo.BlockCache.uiMaxBytes,
								(unsigned)MemInfo.RecordCache.uiMaxBytes);
	gigaOutputStr( MAX_CACHE_ROW, szBuf);
	
	f_sprintf( (char *)szBuf, "Blk: %-10u  Record: %-10u",
								(unsigned)MemInfo.BlockCache.uiTotalBytesAllocated,
								(unsigned)MemInfo.RecordCache.uiTotalBytesAllocated);
	gigaOutputStr( USED_CACHE_ROW, szBuf);

	f_sprintf( (char *)szBuf, "Blk: %-10u  Record: %-10u",
								(unsigned)MemInfo.BlockCache.uiCount,
								(unsigned)MemInfo.RecordCache.uiCount);
	gigaOutputStr( ITEMS_CACHED_ROW, szBuf);
	
	f_sprintf( (char *)szBuf, "Cnt: %-10u  Bytes : %-10u", 
		(unsigned)MemInfo.uiDirtyCount, (unsigned)MemInfo.uiDirtyBytes);
	gigaOutputStr( DIRTY_CACHE_ROW, szBuf);

	f_sprintf( (char *)szBuf, "Cnt: %-10u  Bytes : %-10u", 
		(unsigned)MemInfo.uiLogCount, (unsigned)MemInfo.uiLogBytes);
	gigaOutputStr( LOG_CACHE_ROW, szBuf);

	f_sprintf( (char *)szBuf, "Cnt: %-10u  Bytes : %-10u", 
		(unsigned)MemInfo.uiFreeCount, 
		(unsigned)MemInfo.uiFreeBytes);
	gigaOutputStr( FREE_CACHE_ROW, szBuf);

	gigaCheckpointDisplay();
}

/****************************************************************************
Desc: This routine functions as a thread.  It keeps the gigaload screen
		up to date.
****************************************************************************/
RCODE FLMAPI gigaScreenThread(
	IF_Thread *		pThread)
{
	FLMUINT		uiCurrTime;

	for (;;)
	{
		// See if we should shut down.

		if( pThread->getShutdownFlag())
		{
			break;
		}

		uiCurrTime = FLM_GET_TIMER();

		// Update the display

		gigaUpdateMemInfo();

		pThread->sleep( 1000);
	}

	return( FERR_OK);
}

/********************************************************************
Desc: Start the screen thread.
*********************************************************************/
RCODE gigaStartScreenThread( void)
{
	RCODE			rc = FERR_OK;

	// Start the screen thread

	if( RC_BAD( rc = f_threadCreate( &gv_pScreenThrd,
		gigaScreenThread, "Gigaload Monitor")))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc: Shutdown the screen thread.
*********************************************************************/
void gigaStopScreenThread( void)
{
	f_threadDestroy( &gv_pScreenThrd);
}

/********************************************************************
Desc:
*********************************************************************/
void gigaUpdateLoadTimes( void)
{
	FLMUINT		uiElapsedTime;
	FLMUINT		uiCurrTime;
	FLMUINT		uiSecs;
	FLMUINT		uiAddsPerSec;
	char			szElapsedTime [20];

	uiCurrTime = FLM_GET_TIMER();
	uiElapsedTime = FLM_ELAPSED_TIME( uiCurrTime, gv_ui10SecStartTime);

	// Calculate and display the average for the last 10 seconds.

	uiSecs = FLM_TIMER_UNITS_TO_SECS( uiElapsedTime);
	uiAddsPerSec = (gv_uiTotalLoaded - gv_ui10SecTotal) / uiSecs;

	f_mutexLock( gv_hWindowMutex);
	gigaOutputUINT( ADDS_PER_SEC_CURRENT, uiAddsPerSec, TRUE);

	gv_ui10SecTotal = gv_uiTotalLoaded;
	gv_ui10SecStartTime = uiCurrTime;

	// Calculate and display the overall average

	uiElapsedTime = FLM_ELAPSED_TIME( uiCurrTime, gv_uiStartTime);
	uiSecs = FLM_TIMER_UNITS_TO_SECS( uiElapsedTime);
	uiAddsPerSec = gv_uiTotalLoaded / uiSecs;

	gigaOutputUINT( ADDS_PER_SEC_OVERALL, uiAddsPerSec, TRUE);

	f_sprintf( szElapsedTime, "%u:%02u:%02u",
		(unsigned)uiSecs / 3600,
		(unsigned)(uiSecs % 3600) / 60,
		(unsigned)uiSecs % 60);

	gigaOutputStr( ELAPSED_TIME_ROW, szElapsedTime, TRUE);

	f_mutexUnlock( gv_hWindowMutex);
}

/********************************************************************
Desc: Loads the database with objects.
*********************************************************************/
RCODE gigaLoadDatabase( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bCommitTrans = FALSE;
	FLMUINT			uiObjsInTrans = 0;
	FLMUINT			uiChar = 0;
	FLMUINT			bSuspend = FALSE;
	FlmRecord *		pNewRec = NULL;

	// Set cache size, if specified on command line.

	if( gv_uiCacheSize)
	{
		if( RC_BAD( rc = FlmSetHardMemoryLimit( 0, FALSE, 0,
			gv_uiCacheSize, 0)))
		{
			gigaOutputRcErr( "setting cache size", rc);
			goto Exit;
		}
	}

	// Set block cache percentage, if it is not default.
	
	if( gv_uiBlockCachePercentage != 50)
	{
		if( RC_BAD( rc = FlmConfig( FLM_BLOCK_CACHE_PERCENTAGE,
			(void *)gv_uiBlockCachePercentage, (void *)0)))
		{
			gigaOutputRcErr( "setting block cache percentage", rc);
			goto Exit;
		}
	}

	// Set the maximum and low dirty cache, if one was specified

	if( gv_uiMaxDirtyCache)
	{
		if( RC_BAD( rc = FlmConfig( FLM_MAX_DIRTY_CACHE,
			(void *)gv_uiMaxDirtyCache, (void *)gv_uiLowDirtyCache)))
		{
			gigaOutputRcErr( "setting maximum dirty cache", rc);
			goto Exit;
		}
	}

	// Set checkpoint interval, if one is specified.

	if( gv_uiCPInterval != 0xFFFFFFFF)
	{
		if( RC_BAD( rc = FlmConfig( FLM_MAX_CP_INTERVAL, 
			(void *)gv_uiCPInterval, (void *)0)))
		{
			gigaOutputRcErr( "setting checkpoint interval", rc);
			goto Exit;
		}
	}
	
	// Enable/Disable direct I/O
	
	if( RC_BAD( rc = FlmConfig( FLM_DIRECT_IO_STATE, 
		(void *)!gv_bDisableDirectIO, NULL)))
	{
		goto Exit;
	}

	// Create the database.
	
	(void)FlmDbRemove( gv_szDibName, gv_szDataDir, gv_szRflDir, TRUE);
	
	if( RC_BAD( rc = FlmDbCreate( gv_szDibName, gv_szDataDir, gv_szRflDir, 
		NULL, gv_pszGigaDictionary, NULL, &gv_hDb)))
	{
		gigaOutputRcErr( "creating database", rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmDbConfig( gv_hDb, 
		FDB_RFL_FOOTPRINT_SIZE, (void *)(512 * 1024 * 1024), NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbConfig( gv_hDb, 
		FDB_RBL_FOOTPRINT_SIZE, (void *)(512 * 1024 * 1024), NULL)))
	{
		goto Exit;
	}

	// Create the display

	gv_uiTotalLoaded = 0;
	gv_ui10SecTotal = 0;
	
	f_mutexLock( gv_hWindowMutex);
	FTXWinClear( gv_pWindow);
	f_mutexUnlock( gv_hWindowMutex);

	gigaOutputLabel( MAX_CACHE_ROW, "Maximum Cache Size (bytes)");
	gigaOutputLabel( USED_CACHE_ROW, "Cache Used (bytes)");
	gigaOutputLabel( ITEMS_CACHED_ROW, "Cache Used (items)");
	gigaOutputLabel( DIRTY_CACHE_ROW, "Dirty Cache (bytes)");
	gigaOutputLabel( LOG_CACHE_ROW, "Log Cache (bytes)");
	gigaOutputLabel( FREE_CACHE_ROW, "Free Cache (bytes)");
	gigaOutputLabel( CP_STATE_ROW, "Checkpoint State");
	
	gigaUpdateMemInfo();

	gigaOutputLabel( DB_NAME_ROW, "Database Name");
	gigaOutputStr( DB_NAME_ROW, gv_szDibName);

	gigaOutputLabel( TOTAL_TO_LOAD_ROW, "Total To Load");
	gigaOutputUINT( TOTAL_TO_LOAD_ROW, gv_uiTotalToLoad);

	gigaOutputLabel( TRANS_SIZE_ROW, "Transaction Size");
	gigaOutputUINT( TRANS_SIZE_ROW, gv_uiTransSize);

	gigaOutputLabel( TOTAL_LOADED_ROW, "Total Loaded");
	gigaOutputUINT( TOTAL_LOADED_ROW, gv_uiTotalLoaded);

	gigaOutputLabel( ADDS_PER_SEC_CURRENT, "Adds/Sec. (10 secs)");
	gigaOutputUINT( ADDS_PER_SEC_CURRENT, 0);

	gigaOutputLabel( ADDS_PER_SEC_OVERALL, "Adds/Sec. (overall)");
	gigaOutputUINT( ADDS_PER_SEC_OVERALL, 0);

	gigaOutputLabel( ELAPSED_TIME_ROW, "Elapsed Time");
	gigaOutputStr( ELAPSED_TIME_ROW, "<none>");

	if( RC_BAD( rc = gigaStartScreenThread()))
	{
		goto Exit;
	}
	
	gv_ui10SecStartTime = gv_uiStartTime = FLM_GET_TIMER();
	gv_ui10Secs = FLM_SECS_TO_TIMER_UNITS( 10);
	gv_ui1Sec = FLM_SECS_TO_TIMER_UNITS( 1);
	
	for( ;;)
	{
		// See if we have been told to shut down, or if the user 
		// has pressed escape.

		if( gv_bShutdown)
		{
			break;
		}

		// Every 127 objects, see if character was pressed and update 
		// count on screen.

		if( (gv_uiTotalLoaded & 0x7F) == 0)
		{
			f_yieldCPU();
			
			if( (uiChar = gigaSeeIfQuit()) != 0)
			{
				if( uiChar == FKB_ESCAPE)
				{
					break;
				}
				else if( uiChar == 's' || uiChar == 'S')
				{
					bSuspend = TRUE;
				}
			}

			// Check for other keyboard options
		}
		else if( (gv_uiTotalLoaded & 0x7) == 0)
		{
			FLMUINT		uiElapsedTime;
			FLMUINT		uiCurrTime;

			uiCurrTime = FLM_GET_TIMER();

			// If at least 10 seconds have elapsed, redisplay the average
			// rate values.

			if( (uiElapsedTime = FLM_ELAPSED_TIME( uiCurrTime,
				gv_ui10SecStartTime)) >= gv_ui10Secs)
			{
				gigaUpdateLoadTimes();
			}
		}

		// Start a transaction, if one is not going.

		if( !bTransActive)
		{
			if( bSuspend)
			{
				uiChar = gigaGetInput(
					"Load suspended, press any character to continue loading: ",
					NULL);
				bSuspend = FALSE;
			}
			
			if( RC_BAD( rc = gigaStartTrans()))
			{
				goto Exit;
			}
			
			bTransActive = TRUE;
			bCommitTrans = FALSE;
			uiObjsInTrans = 0;
		}

		// Increment the load counters and determine if this will be the
		// last object of the transaction.

		gv_uiTotalLoaded++;
		uiObjsInTrans++;
		
		if( uiObjsInTrans == gv_uiTransSize ||
			 gv_uiTotalLoaded == gv_uiTotalToLoad)
		{
			bCommitTrans = TRUE;
		}

		// Create a new object.

		if( RC_BAD( rc = gigaMakeNewRecord( &pNewRec)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = FlmRecordAdd( gv_hDb, FLM_DATA_CONTAINER, 
			NULL, pNewRec, FLM_DONT_INSERT_IN_CACHE)))
		{
			goto Exit;
		}
		
		// Commit when we reach the transaction size or the total to load.
		// NOTE: The bCommitTrans flag is set above.

		if( bCommitTrans)
		{
			if( RC_BAD( rc = gigaCommitTrans()))
			{
				goto Exit;
			}
			
			bTransActive = FALSE;
		}

		// See if we are done.

		if( gv_uiTotalLoaded == gv_uiTotalToLoad)
		{
			flmAssert( !bTransActive);
			break;
		}
	}

Exit:

	if( pNewRec)
	{
		pNewRec->Release();
	}

	if( bTransActive)
	{
		(void)FlmDbTransAbort( gv_hDb);
	}
	
	if( gv_hDb != HFDB_NULL)
	{
		FlmDbCheckpoint( gv_hDb, FLM_NO_TIMEOUT);
		gigaStopScreenThread();
		FlmDbClose( &gv_hDb);

		// This will cause us to wait for the last checkpoint
		// to finish.

		(void)FlmConfig( FLM_CLOSE_FILE, (void *)gv_szDibName,
								(void *)gv_szDataDir);
	}
	
	gigaUpdateLoadTimes();
	gigaStopScreenThread();
	f_threadDestroy( &gv_pIxManagerThrd);
	
	return( rc);
}

#ifdef FLM_NLM
/****************************************************************************
Desc: This routine shuts down all threads in the NLM.
****************************************************************************/
void gigaCleanup( void)
{
	gv_bShutdown = TRUE;
	while( gv_bRunning)
	{
		f_yieldCPU();
	}
}
#endif

/****************************************************************************
Desc:	Displays two lines of message and gets the user's input.
*****************************************************************************/
FLMUINT gigaGetInput(
	const char *	pszMsg1,
	const char *	pszMsg2,
	FLMBOOL			bMutexLocked)
{
	eColorType		eSaveBack;
	eColorType		eSaveFore;
	FLMUINT			uiChar;

	if (!bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexLock( gv_hWindowMutex);
	}

	// Get the background and foreground color so we can restore them.

	FTXWinGetBackFore( gv_pWindow, &eSaveBack, &eSaveFore);

	// Clear the last one or two lines on the screen

	if (pszMsg2 && *pszMsg2)
	{
		FTXWinClearXY( gv_pWindow, 0, gv_uiNumRows - 2);
	}
	else
	{
		FTXWinClearXY( gv_pWindow, 0, gv_uiNumRows - 1);
	}

	// Change to WHITE on RED.

	FTXWinSetBackFore( gv_pWindow, FLM_RED, FLM_WHITE);

	// Display messages on last two lines of screen.

	if (pszMsg2 && *pszMsg2)
	{
		FTXWinPrintStrXY( gv_pWindow, pszMsg1,
				0, gv_uiNumRows - 2);
		FTXWinPrintStrXY( gv_pWindow, pszMsg2,
				0, gv_uiNumRows - 1);
	}
	else
	{
		FTXWinPrintStrXY( gv_pWindow, pszMsg1,
				0, gv_uiNumRows - 1);
	}

	// Wait for user to press key.

	for (;;)
	{
		if (gv_bShutdown)
		{
			uiChar = 0;
			break;
		}
		if (RC_OK( FTXWinTestKB( gv_pWindow)))
		{
			FTXWinInputChar( gv_pWindow, &uiChar);
			break;
		}
		f_sleep(50);
	}

	// Clear out last one or two lines of screen.

	FTXWinSetBackFore( gv_pWindow, eSaveBack, eSaveFore);
	
	if (pszMsg2 && *pszMsg2)
	{
		FTXWinClearXY( gv_pWindow, 0, gv_uiNumRows - 2);
	}
	else
	{
		FTXWinClearXY( gv_pWindow, 0, gv_uiNumRows - 1);
	}

	if (!bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( gv_hWindowMutex);
	}

	return( uiChar);
}

/****************************************************************************
Desc:	Displays an error message.
*****************************************************************************/
void gigaOutputErrMsg(
	const char *	pszErrMsg,
	FLMBOOL			bMutexLocked)
{
	(void)gigaGetInput( pszErrMsg, "Press any character to continue: ",
			bMutexLocked);
}

/****************************************************************************
Desc:	Displays an error message with an RCODE.
*****************************************************************************/
void gigaOutputRcErr(
	const char *	pszWhat,
	RCODE				rc,
	FLMBOOL			bMutexLocked)
{
	char	szMsg [100];

	f_sprintf( szMsg, "Error %s: %s (%04X)", pszWhat,
		FlmErrorString( rc), (unsigned)rc);
	gigaOutputErrMsg( szMsg, bMutexLocked);
}

/********************************************************************
Desc: Output a label in the LABEL_COLUMN
*********************************************************************/
void gigaOutputLabel(
	FLMUINT			uiRow,
	const char *	pszLabel,
	FLMBOOL			bMutexLocked)
{
	char			szLabel [DATA_COLUMN - LABEL_COLUMN];
	char *		pszTmp;
	FLMUINT		uiNumDots = sizeof( szLabel) - 1;

	f_memset( szLabel, '.', uiNumDots);
	szLabel [uiNumDots] = 0;

	pszTmp = &szLabel [0];
	uiNumDots -= 2;
	
	while( *pszLabel && uiNumDots)
	{
		*pszTmp++ = (FLMBYTE)(*pszLabel++);
		uiNumDots--;
	}
	
	if( !bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexLock( gv_hWindowMutex);
	}
	
	FTXWinPrintStrXY( gv_pWindow, szLabel, LABEL_COLUMN, uiRow);
	
	if( !bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( gv_hWindowMutex);
	}

}

/********************************************************************
Desc: Output a string in the DATA_COLUMN
*********************************************************************/
void gigaOutputStr(
	FLMUINT			uiRow,
	const char *	pszStr,
	FLMBOOL			bMutexLocked)
{
	if( !bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexLock( gv_hWindowMutex);
	}
	
	FTXWinPrintStrXY( gv_pWindow, pszStr, DATA_COLUMN, uiRow);
	
	if( !bMutexLocked && gv_hWindowMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( gv_hWindowMutex);
	}
}

/********************************************************************
Desc: Output a FLMUINT in the DATA_COLUMN
*********************************************************************/
void gigaOutputUINT(
	FLMUINT		uiRow,
	FLMUINT		uiNum,
	FLMBOOL		bMutexLocked)
{
	char	szBuf [20];

	f_sprintf( szBuf, "%-10u", (unsigned)uiNum);
	gigaOutputStr( uiRow, szBuf, bMutexLocked);
}
