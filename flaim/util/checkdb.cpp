//-------------------------------------------------------------------------
// Desc:	Check database for corruptions.
// Tabs:	3
//
// Copyright (c) 1992-2007 Novell, Inc. All Rights Reserved.
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
#include "sharutil.h"

#define UTIL_ID					"CHECKDB"
#define MAX_LOG_BUF				(16 * 1024)

#define  LABEL_COLUMN   		5
#define  VALUE_COLUMN   		30

#define LOG_FILE_ROW				1
#define SOURCE_ROW				2
#define DATA_DIR_ROW				3
#define RFL_DIR_ROW				4
#define CACHE_USED_ROW			5
#define DOING_ROW					7
#define DB_SIZE_ROW				8
#define AMOUNT_DONE_ROW			9
#define TOTAL_KEYS_ROW			10
#define TOTAL_KEYS_EXAM_ROW	11
#define BAD_IXREF_ROW			12
#define MISSING_IXREF_ROW		13
#define NONUNIQUE_ROW			14
#define CONFLICT_ROW				15
#define CORRUPT_ROW				16
#define TOTAL_CORRUPT_ROW		17
#define REPAIR_ROW				18
#define OLD_VIEW_ROW				19
#define MISMATCH_ROW				20

#define MAX_LOG_BUFF          2048

#define MISMATCH_READ_ERR        0
#define MISMATCH_ERROR_CODE      1
#define MISMATCH_ERR_LOCALE      2
#define MISMATCH_LF_NUMBER       3
#define MISMATCH_LF_NAME         4
#define MISMATCH_LF_TYPE         5
#define MISMATCH_LF_LEVEL        6
#define MISMATCH_BLK_ADDRESS     7
#define MISMATCH_PARENT_ADDRESS  8
#define MISMATCH_ELM_OFFSET      9
#define MISMATCH_DRN             10
#define MISMATCH_ELM_REC_OFFSET  11
#define MISMATCH_FIELD_NUM       12
#define MISMATCH_ERR_NOT_LOGGED  13

FSTATIC FLMBOOL CheckDatabase( void);

FSTATIC FLMBOOL DoCheck( void);

FSTATIC void CheckShowHelp(
	FLMBOOL					bShowFullUsage);

FSTATIC FLMBOOL GetParams(
	FLMINT					iArgC,
	char **					ppucArgV);

FSTATIC void OutLabel(
	FLMUINT					uiCol,
	FLMUINT					uiRow,
	const char *			pucLabel,
	const char *			pucValue,
	FLMUINT64				ui64NumValue,
	FLMBOOL					bLogIt);

FSTATIC void OutLine(
	const char *			pucString);

FSTATIC FLMBOOL GetDatabaseTags( void);

FSTATIC void LogFlush( void);

FSTATIC void LogString(
	const char *			pszString);

FSTATIC void DisplayValue(
	FLMUINT					uiRow,
	const char *			pucValue);

FSTATIC void DisplayNumValue(
	FLMUINT					uiRow,
	FLMUINT64				ui64Number);

FSTATIC void OutValue(
	const char *			pucLabel,
	const char *			pucValue);

FSTATIC void OutUINT(
	const char *			pucLabel,
	FLMUINT					uiNum);

FSTATIC void OutUINT64(
	const char *			pucLabel,
	FLMUINT64				ui64Num);

FSTATIC void OutBlkHeader( void);

FSTATIC void OutOneBlockStat(
	const char *			pucLabel,
	FLMUINT					uiBlockSize,
	BLOCK_INFO *			pBlockInfo,
	FLMUINT64				ui64KeyCount,
	FLMUINT64				ui64RefCount,
	FLMUINT64				ui64FldCount);

FSTATIC void OutLogicalFile(
	DB_CHECK_PROGRESS *	pCheckProgress,
	FLMUINT					uiIndex);

FSTATIC void PrintInfo(
	DB_CHECK_PROGRESS *	pCheckProgress);

FSTATIC FLMUINT CheckShowError(
	const char *			pucMessage,
	FLMBOOL					bLogIt);

FSTATIC RCODE GetUserInput( void);

RCODE ProgFunc(
	eStatusType				eStatus,
	void *					pvParm1,
	void *					pvParm2,
	void *					pvAppData);

FSTATIC void LogStr(
	FLMUINT					uiIndent,
	const char *			pszStr);

FSTATIC void LogCorruptError(
	CORRUPT_INFO *			pCorrupt);

FSTATIC void LogKeyError(
	CORRUPT_INFO *			pCorrupt);

FSTATIC FLMBOOL DisplayField(
	FlmRecord *				pRecord,
	void *					pvField,
	FLMUINT					uiStartCol,
	FLMUINT					uiLevelOffset);

FSTATIC void NumToName(
	FLMUINT					uiNum,
	char *					pucBuf);

FLMBOOL						gv_bShutdown = FALSE;
static F_Pool				gv_pool;

static IF_FileHdl *		gv_pLogFile = NULL;

static HFDB					gv_hDb = HFDB_NULL;
static F_NameTable *		gv_pNameTable = NULL;

static FLMUINT				gv_uiMaxRow;
static FLMUINT				gv_uiLineCount;

static char					gv_pucLogFileName[ F_PATH_MAX_SIZE];
static char					gv_pucTmpDir[ F_PATH_MAX_SIZE];
static char					gv_pucLastError[ 256];
static char					gv_pucDbFileName[ F_PATH_MAX_SIZE];
static char					gv_szDataDir[ F_PATH_MAX_SIZE];
static char					gv_szRflDir[ F_PATH_MAX_SIZE];
static char *				gv_pucLogBuffer = NULL;
static FLMUINT64			gv_ui64DatabaseSize;
static FLMUINT64			gv_ui64BytesDone;
static FLMUINT				gv_uiCorruptCount;
static FLMUINT				gv_uiRepairCount;
static FLMUINT				gv_uiTotalCorruptions;
static FLMUINT				gv_uiOldViewCount;
static FLMUINT				gv_uiMismatchCount;
static FLMUINT				gv_uiLogBufferCount = 0;
static FLMBOOL				gv_bMultiplePasses = FALSE;
static FLMBOOL				gv_bBatchMode;
static FLMBOOL				gv_bContinue;
static FLMBOOL				gv_bStartUpdate = FALSE;
static FLMBOOL				gv_bRepairCorruptions = FALSE;
static FLMBOOL				gv_bDoLogicalCheck = FALSE;
static FLMBOOL				gv_bLoggingEnabled;
static FLMBOOL				gv_bShowStats;
static FLMBOOL				gv_bRunning;
static FLMBOOL				gv_bPauseBeforeExiting = FALSE;
static IF_FileSystem *	gv_pFileSystem = NULL;
static char 				gv_szPassword[ 256];

/********************************************************************
Desc: ?
*********************************************************************/
#ifdef FLM_RING_ZERO_NLM
extern "C" int nlm_main(
#else
int main(
#endif
	int				iArgC,
	char **			ppucArgV)
{
	int				iResCode = 0;
	F_Pool *			pLogPool = NULL;

	gv_bBatchMode = FALSE;
	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	gv_pucLastError[ 0] = '\0';

	if( RC_BAD( FlmStartup()))
	{
		f_conStrOut( "\nCould not initialize FLAIM.\n");
		goto Exit;
	}

	f_conInit( 0xFFFF, 0xFFFF, "FLAIM Database Check");
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conDrawBorder();
	f_conClearScreen( 0, 0);
	f_conGetScreenSize( NULL, &gv_uiMaxRow);

	if( RC_BAD( FlmGetFileSystem( &gv_pFileSystem)))
	{
		f_conStrOut( "\nCould not allocate a file system object.\n");
		goto Exit;
	}

	if( (pLogPool = f_new F_Pool) == NULL)
	{
		f_conStrOut( "\nCould not allocate a pool object.\n");
		goto Exit;
	}

	pLogPool->poolInit( 1024);
	
	if( RC_BAD( pLogPool->poolAlloc( MAX_LOG_BUF, (void **)&gv_pucLogBuffer)))
	{
		f_conStrOut( "\nCould not allocat memory.\n");
		goto Exit;
	}
	
	if( GetParams( iArgC, ppucArgV))
	{
		if (!DoCheck())
		{
			iResCode = 1;
		}
	}

	if( gv_pucTmpDir[ 0] != '\0')
	{
		(void)FlmConfig( FLM_TMPDIR, (void *)(&gv_pucTmpDir [0]), 0);
	}

	if( (gv_bPauseBeforeExiting) && (!gv_bShutdown))
	{
		f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		f_conClearScreen( 0, (FLMBYTE)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_RED, FLM_WHITE);
		
		if( gv_pucLastError[ 0] != '\0')
		{
			f_conStrOut( gv_pucLastError);
		}
		
		f_conSetCursorPos( 0, gv_uiMaxRow - 1);
		f_conStrOut( "Press any character to exit CHECKDB: ");
		
		for (;;)
		{
			if( gv_bShutdown)
			{
				break;
			}
			
			if( f_conHaveKey())
			{
				f_conGetKey();
				break;
			}
		}
	}

Exit:

	if( gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}

	if( pLogPool)
	{
		pLogPool->Release();
	}

	f_conExit();
	FlmShutdown();
	
	gv_bRunning = FALSE;
	return( iResCode);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBOOL CheckDatabase( void)
{
	RCODE						rc = FERR_OK;
	FLMUINT					uiStatus;
	char						pucTmpBuf[ 100];
	FLMUINT					uiCheckFlags;
	FLMBOOL					bOk = TRUE;
	FLMBOOL					bCheckDb = TRUE;
	FLMBOOL					bStartedTrans;
	DB_CHECK_PROGRESS		CheckProgress;

	// Open the database - if not already open

	if( gv_hDb == HFDB_NULL)
	{
		if( RC_BAD( rc = FlmDbOpen( gv_pucDbFileName, gv_szDataDir,
			gv_szRflDir, 0, &gv_szPassword[ 0], &gv_hDb)))
		{
			f_strcpy( pucTmpBuf, "Error opening database: ");
			f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], FlmErrorString( rc));
			CheckShowError( pucTmpBuf, TRUE);
			bOk = FALSE;
			goto Exit;
		}
	}

	// Get the tag numbers for the database we are doing

	if( !GetDatabaseTags())
	{
		bOk = FALSE;
		goto Exit;
	}

	gv_uiCorruptCount = 0;
	gv_ui64BytesDone = 0;
	gv_ui64DatabaseSize = 0;
	gv_uiOldViewCount = 0;
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);

	if( gv_bLoggingEnabled)
	{
		LogString( NULL);
		LogString( NULL);
		LogString( NULL);
		LogString( "==========================================================================");
		LogString( "CHECK PARAMETERS:");
	}
	
	OutLabel( LABEL_COLUMN, SOURCE_ROW, "Database", gv_pucDbFileName, 0, TRUE);
	
	OutLabel( LABEL_COLUMN, DATA_DIR_ROW, "Data Files Dir.",
						gv_szDataDir [0]
							? &gv_szDataDir [0]
							: "<Same as DB>", 0, TRUE);
						
	OutLabel( LABEL_COLUMN, RFL_DIR_ROW, "RFL Files Dir.",
						gv_szRflDir [0]
							? &gv_szRflDir [0]
							: "<Same as DB>", 0, TRUE);
						
	OutLabel( LABEL_COLUMN, LOG_FILE_ROW, "Log File",
						(gv_pucLogFileName[ 0])
								? &gv_pucLogFileName[ 0]
								: "<NONE>", 0, FALSE);

	OutLabel( LABEL_COLUMN, CACHE_USED_ROW, "Cache Bytes Used", NULL, 0, FALSE);
		
	OutLabel( LABEL_COLUMN, DOING_ROW, "Doing", "Opening Database", 0, FALSE);
		
	OutLabel( LABEL_COLUMN, DB_SIZE_ROW, "DB Size", NULL, (FLMUINT)gv_ui64DatabaseSize, FALSE);
			
	OutLabel( LABEL_COLUMN, CORRUPT_ROW, "Database Corruptions", NULL, gv_uiCorruptCount, FALSE);
			
	OutLabel( LABEL_COLUMN, TOTAL_CORRUPT_ROW, "Total Corruptions", NULL, gv_uiTotalCorruptions, FALSE);
			
	OutLabel( LABEL_COLUMN, OLD_VIEW_ROW, "Old View Count", NULL, gv_uiOldViewCount, FALSE);
			
	OutLabel( LABEL_COLUMN, MISMATCH_ROW, "Mismatch Count", NULL, gv_uiMismatchCount, FALSE);
			
	OutLabel( LABEL_COLUMN, REPAIR_ROW, "Problems Repaired", NULL, gv_uiRepairCount, FALSE);
			
	OutLabel( LABEL_COLUMN, TOTAL_KEYS_ROW, "Total Index Keys", NULL, 0, FALSE);
			
	OutLabel( LABEL_COLUMN, CONFLICT_ROW, "Key Conflicts", NULL, 0, FALSE);
			
	OutLabel( LABEL_COLUMN, TOTAL_KEYS_EXAM_ROW, "Num. Keys Checked", NULL, 0, FALSE);
			
	OutLabel( LABEL_COLUMN, BAD_IXREF_ROW, "Invalid Index Keys", NULL, 0, FALSE);
		
	OutLabel( LABEL_COLUMN, MISSING_IXREF_ROW, "Missing Index Keys", NULL, 0, FALSE);
		
	OutLabel( LABEL_COLUMN, NONUNIQUE_ROW, "Non-unique Index Keys", NULL, 0, FALSE);
	
	if( gv_bLoggingEnabled)
	{
		LogString( NULL);
		LogString( "CHECK DETAILED RESULTS:");
		LogString( NULL);
	}

	uiCheckFlags = FLM_CHK_FIELDS;

	if( gv_bDoLogicalCheck)
	{
		uiCheckFlags |= FLM_CHK_INDEX_REFERENCING;
	}
 
	if( RC_OK( rc))
	{
		// Start an update transaction for the duration of the check.

		bStartedTrans = FALSE;
		if( gv_bStartUpdate)
		{
			if( RC_BAD( rc = FlmDbTransBegin( gv_hDb,
				FLM_UPDATE_TRANS, 15)))
			{
				bCheckDb = FALSE;
			}
			bStartedTrans = TRUE;
		}

		if( bCheckDb)
		{
			rc = FlmDbCheck( gv_hDb, NULL, NULL, NULL, 
				uiCheckFlags, &gv_pool, &CheckProgress, ProgFunc, (void *)0);
		}

		if( bStartedTrans)
		{
			if( RC_BAD( rc))
			{
				(void)FlmDbTransAbort( gv_hDb);
			}
			else
			{
				rc = FlmDbTransCommit( gv_hDb);
			}
		}
	}

	if( rc == FERR_FAILURE)
	{
		f_sprintf( pucTmpBuf, "User pressed ESCAPE, check halted");
		gv_bShutdown = TRUE;
	}
	else
	{
		f_strcpy( pucTmpBuf, "RETURN CODE: ");
		f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], FlmErrorString( rc));
	}
	
	uiStatus = CheckShowError( pucTmpBuf, TRUE);

	if( ((uiStatus != FKB_ESCAPE) || (gv_bLoggingEnabled)) &&
		 (gv_bShowStats) &&
		 ((rc == FERR_OK) ||
		  (rc == FERR_DATA_ERROR) ||
		  (rc == FERR_TRANS_ACTIVE)))
	{
		PrintInfo( &CheckProgress);
	}
	
	if( gv_bLoggingEnabled)
	{
		LogString( NULL);
		LogFlush();
	}

Exit:

	if( gv_pNameTable)
	{
		gv_pNameTable->Release();
		gv_pNameTable = NULL;
	}

	if( gv_hDb != HFDB_NULL)
	{
		FlmDbClose( &gv_hDb);
	}

	gv_pool.poolReset();
	return( bOk);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBOOL DoCheck( void)
{
	RCODE       rc = FERR_OK;
	FLMBOOL		bOk = TRUE;
	char			pucTmpBuf[ 100];

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conClearScreen( 0, 0);
	gv_bContinue = TRUE;
	gv_uiLineCount = 0;
	gv_bLoggingEnabled = FALSE;
	gv_uiCorruptCount = 0;
	gv_uiTotalCorruptions = 0;
	gv_uiRepairCount = 0;
	gv_uiMismatchCount = 0;
	gv_uiLogBufferCount = 0;
	gv_pool.poolInit( 1024);
	if( gv_pucLogFileName[ 0])
	{
		gv_pFileSystem->deleteFile( gv_pucLogFileName);
		if( RC_OK( rc = gv_pFileSystem->createFile( gv_pucLogFileName,
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &gv_pLogFile)))
		{
			gv_bLoggingEnabled = TRUE;
		}
		else
		{
			f_strcpy( pucTmpBuf, "Error creating log file: ");
			f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], FlmErrorString( rc));
			CheckShowError( pucTmpBuf, FALSE);
			bOk = FALSE;
			goto Exit;
		}
	}

	f_conSetCursorType( FLM_CURSOR_INVISIBLE);
	for( ;;)
	{
		if( !CheckDatabase())
		{
			bOk = FALSE;
			break;
		}

		if( (!gv_bMultiplePasses) || (gv_bShutdown))
		{
			break;
		}
	}
	f_conSetCursorType( FLM_CURSOR_UNDERLINE);

	if( gv_bLoggingEnabled)
	{
		LogFlush();
		gv_pLogFile->Release();
		gv_pLogFile = NULL;
	}

Exit:

	gv_pool.poolFree();
	return( bOk);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void CheckShowHelp(
	FLMBOOL		bShowFullUsage)
{
	f_conStrOut( "\n");
	
	if( bShowFullUsage)
	{
		f_conStrOut( "Usage: checkdb <FileName> [Options]\n");
	}
	else
	{
		f_conStrOut( "Parameters: <FileName> [Options]\n\n");
	}

	f_conStrOut( "   FileName = Name of database to check.\n");
	f_conStrOut( "   Options\n");
	f_conStrOut( "        -b           = Run in Batch Mode.\n");
	f_conStrOut( "        -c           = Repair logical corruptions.\n");
	f_conStrOut( "        -d           = Display/log detailed statistics.\n");
	f_conStrOut( "        -dr<Dir>     = RFL directory.\n");
	f_conStrOut( "        -dd<Dir>     = Data directory.\n");
	f_conStrOut( "        -i           = Perform a logical (index) check.\n");
	f_conStrOut( "        -l<FileName> = Log detailed information to <FileName>.\n");
	f_conStrOut( "        -m           = Multiple passes (continuous check).\n");
	f_conStrOut( "        -o<FileName> = Output binary log information to <FileName>.\n");
	f_conStrOut( "        -p           = Pause before exiting.\n");
	f_conStrOut( "        -pw<password>= Open database with password.\n");
	f_conStrOut( "        -t<Path>     = Temporary directory.\n");
	f_conStrOut( "        -u           = Run check in an update transaction.\n");
	f_conStrOut( "        -v<FileName> = Verify binary log information in <FileName>.  NOTE:\n");
	f_conStrOut( "                       The -v and -o options cannot both be specified.\n");
	f_conStrOut( "        -?           = A '?' anywhere in the command line will cause this\n");
	f_conStrOut( "                       screen to be displayed.\n");
	f_conStrOut( "Options may be specified anywhere in the command line.\n");
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBOOL GetParams(
	FLMINT		iArgC,
	char **		ppucArgV)
{
#define MAX_ARGS     30
	FLMUINT		uiLoop;
	char			pucTmpBuf[ 100];
	char *		pucTmp;
	char *		ppArgs[ MAX_ARGS];
	char			pucCommandBuffer[ 300];

	gv_pucDbFileName[ 0] = '\0';
	gv_szDataDir[ 0] = '\0';
	gv_szRflDir[ 0] = '\0';
	gv_pucLogFileName[ 0] = '\0';
	gv_pucTmpDir[ 0] = '\0';
	gv_szPassword[ 0] = '\0';
	gv_bShowStats = FALSE;

	// Ask the user to enter parameters if none were entered on the command
	// line.

	if( iArgC < 2)
	{
		for( ;;)
		{
			f_conStrOut( "CheckDB Params (enter ? for help): ");
			
			pucCommandBuffer[ 0] = '\0';
			f_conLineEdit( pucCommandBuffer, sizeof( pucCommandBuffer) - 1);
			
			if( gv_bShutdown)
			{
				return( FALSE);
			}
			
			if( f_stricmp( pucCommandBuffer, "?") == 0)
			{
				CheckShowHelp( FALSE);
			}
			else
			{
				break;
			}
		}
		
		flmUtilParseParams( pucCommandBuffer, MAX_ARGS, &iArgC, 
			(const char **)&ppArgs [1]);
		ppArgs[ 0] = ppucArgV [0];
		iArgC++;
		ppucArgV = &ppArgs[ 0];
	}

	uiLoop = 1;
	while( uiLoop < (FLMUINT)iArgC)
	{
		pucTmp = ppucArgV[ uiLoop];

		// See if they specified an option

#ifdef FLM_UNIX
		if( *pucTmp == '-')
#else
		if( (*pucTmp == '-') || (*pucTmp == '/'))
#endif
		{
			pucTmp++;
			if( (*pucTmp == 'l') || (*pucTmp == 'L'))
			{
				pucTmp++;
				if( *pucTmp)
				{
					f_strcpy( gv_pucLogFileName, pucTmp);
				}
				else
				{
					if( CheckShowError( "Log file name not specified in parameter", FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if( (*pucTmp == 't') || (*pucTmp == 'T'))
			{
				pucTmp++;
				if( *pucTmp)
				{
					f_strcpy( gv_pucTmpDir, pucTmp);
				}
				else
				{
					if( CheckShowError( "Temporary directory not specified in parameter", FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if( (*pucTmp == 'd') || (*pucTmp == 'D'))
			{
				pucTmp++;
				if (!(*pucTmp))
				{
					gv_bShowStats = TRUE;
				}
				else if (*pucTmp == 'r' || *pucTmp == 'R')
				{
					f_strcpy( gv_szRflDir, pucTmp + 1);
				}
				else if (*pucTmp == 'd' || *pucTmp == 'D')
				{
					f_strcpy( gv_szDataDir, pucTmp + 1);
				}
				else
				{
					f_sprintf( pucTmpBuf, "Invalid option %s", pucTmp - 1);
					if( CheckShowError( pucTmpBuf, FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if (f_stricmp( pucTmp, "B") == 0)
			{
				gv_bBatchMode = TRUE;
			}
			else if (f_stricmp( pucTmp, "C") == 0)
			{
				gv_bRepairCorruptions = TRUE;
			}
			else if (f_stricmp( pucTmp, "I") == 0)
			{
				gv_bDoLogicalCheck = TRUE;
			}
			else if (f_stricmp( pucTmp, "M") == 0)
			{
				gv_bMultiplePasses = TRUE;
			}
			else if ((*pucTmp == 'p') || (*pucTmp == 'P'))
			{
				pucTmp++;
				if (!(*pucTmp))
				{
					gv_bPauseBeforeExiting = TRUE;
				}
				else if (*pucTmp == 'w' || *pucTmp == 'W')
				{
					f_strcpy( gv_szPassword, pucTmp + 1);
				}
			}
			else if (f_stricmp( pucTmp, "U") == 0)
			{
				gv_bStartUpdate = TRUE;
			}
			else if (f_stricmp( pucTmp, "?") == 0 ||
						f_stricmp( pucTmp, "HELP") == 0)
			{
				CheckShowHelp( TRUE);
				gv_bPauseBeforeExiting = TRUE;
				return( FALSE);
			}
			else
			{
				f_sprintf( pucTmpBuf, "Invalid option %s", pucTmp);
				if( CheckShowError( pucTmpBuf, FALSE) == FKB_ESCAPE)
				{
					return( FALSE);
				}
			}
		}
		else if( f_stricmp( pucTmp, "?") == 0)
		{
Show_Help:
			CheckShowHelp( TRUE);
			gv_bPauseBeforeExiting = TRUE;
			return( FALSE);
		}
		else if( !gv_pucDbFileName[ 0])
		{
			f_strcpy( gv_pucDbFileName, pucTmp);
		}
		uiLoop++;
	}

	if( !gv_pucDbFileName[ 0])
	{
		goto Show_Help;
	}
	else
	{
		return( TRUE);
	}
}

/***************************************************************************
Desc:    This routine gets the tag names for the store we are checking
*****************************************************************************/
FSTATIC FLMBOOL GetDatabaseTags( void)
{
	FLMBOOL		bOk = TRUE;
	RCODE			rc = FERR_OK;

	// Build the path and open the database

	f_conSetBackFore( FLM_BLUE, FLM_LIGHTGRAY);
	f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 1));
	f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 1));
	if( gv_pNameTable)
	{
		gv_pNameTable->Release();
		gv_pNameTable = NULL;
	}

	f_conStrOut( "Initializing tag table ...");

	if ((gv_pNameTable = f_new F_NameTable) == NULL)
	{
		CheckShowError( "Error creating tag table.", TRUE);
		goto Exit;
	}

	if( RC_BAD( rc = gv_pNameTable->setupFromDb( gv_hDb)))
	{
		CheckShowError( "Error initializing tag table.", TRUE);
		goto Exit;
	}

Exit:

	f_conSetBackFore( FLM_BLUE, FLM_LIGHTGRAY);
	f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 1));
	return( bOk);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogFlush( void)
{
	FLMUINT	uiBytesWritten;

	if( gv_uiLogBufferCount)
	{
		gv_pLogFile->write( FLM_IO_CURRENT_POS,
							 gv_uiLogBufferCount, gv_pucLogBuffer, &uiBytesWritten);
		gv_uiLogBufferCount = 0;
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogString(
	const char *	pucString)
{
	FLMUINT		uiLen;
	FLMUINT		uiLoop;

	if( (gv_bLoggingEnabled) && (gv_pucLogBuffer != NULL))
	{
		uiLen = (FLMUINT)((pucString != NULL)
							  ? (FLMUINT)(f_strlen( pucString))
							  : 0);
		for( uiLoop = 0; uiLoop < uiLen; uiLoop++)
		{
			gv_pucLogBuffer[ gv_uiLogBufferCount++] = *pucString++;
			if( gv_uiLogBufferCount == MAX_LOG_BUFF)
			{
				LogFlush();
			}
		}
		gv_pucLogBuffer[ gv_uiLogBufferCount++] = '\r';
		if( gv_uiLogBufferCount == MAX_LOG_BUFF)
		{
			LogFlush();
		}
		gv_pucLogBuffer[ gv_uiLogBufferCount++] = '\n';
		if( gv_uiLogBufferCount == MAX_LOG_BUFF)
		{
			LogFlush();
		}
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutLine(
	const char *		pucBuf)
{
	FLMUINT		uiChar;

	if( gv_bLoggingEnabled)
	{
		LogString( pucBuf);
	}

	if( !gv_bBatchMode)
	{
		if( gv_bContinue)
		{
			if( gv_uiLineCount == 20)
			{
				f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 1));
				f_conSetBackFore( FLM_BLUE, FLM_WHITE);
				f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 1));
				f_conSetBackFore( FLM_RED, FLM_WHITE);
				f_conStrOut( "Press: ESC to quit, anything else to continue");
				for( ;;)
				{
					if( gv_bShutdown)
					{
						uiChar = FKB_ESCAPE;
						break;
					}
					else if( f_conHaveKey())
					{
						uiChar = f_conGetKey();
						break;
					}
				}
				if( uiChar == FKB_ESCAPE)
				{
					gv_bContinue = FALSE;
				}
				else
				{
					f_conSetBackFore( FLM_BLUE, FLM_WHITE);
					f_conClearScreen( 0, 0);
				}
				gv_uiLineCount = 0;
			}
		}
		
		if( gv_bContinue)
		{
			f_conStrOutXY( pucBuf, 0, gv_uiLineCount);
			gv_uiLineCount++;
		}
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutValue(
	const char *	pucLabel,
	const char *	pucValue)
{
	char		pucTmpBuf[ 100];

	f_strcpy( pucTmpBuf, "...................................... ");
	f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], pucValue);
	f_memcpy( pucTmpBuf, pucLabel, f_strlen( pucLabel));
	OutLine( pucTmpBuf);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutUINT(
	const char *	pucLabel,
	FLMUINT			uiNum)
{
	char		pucValue[ 12];

	if( uiNum == 0xFFFFFFFF)
	{
		f_strcpy( pucValue, "0xFFFFFFFF");
	}
	else
	{
		f_sprintf( pucValue, "%u", (unsigned)uiNum);
	}

	OutValue( pucLabel, pucValue);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutUINT64(
	const char *	pucLabel,
	FLMUINT64		ui64Num)
{
	char		pucValue [24];

	if( ui64Num == (FLMUINT64)-1)
	{
		f_strcpy( pucValue, "0xFFFFFFFFFFFFFFFF");
	}
	else
	{
		f_sprintf( pucValue, "%u", (unsigned)ui64Num);
	}

	OutValue( pucLabel, pucValue);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutBlkHeader(	void)
{
	OutLine( "  Blk Type   Blk Count  Total Bytes  Bytes Used  Prcnt  Element Cnt  Avg Elem");
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutOneBlockStat(
	const char *	pucLabel,
	FLMUINT			uiBlockSize,
	BLOCK_INFO *   pBlockInfo,
	FLMUINT64		ui64KeyCount,
	FLMUINT64		ui64RefCount,
	FLMUINT64		ui64FldCount)
{
	char				pucTmpBuf[ 100];
	FLMUINT64		ui64TotalBytes;
	FLMUINT			uiPercent;
	FLMUINT			uiAvgElementSize;

	ui64TotalBytes = (FLMUINT64)pBlockInfo->uiBlockCount * 
								(FLMUINT64)uiBlockSize;
	
	if( pBlockInfo->ui64ElementCount)
	{
		uiAvgElementSize = (FLMUINT)( pBlockInfo->ui64BytesUsed /
											pBlockInfo->ui64ElementCount);
	}
	else
	{
		uiAvgElementSize = 0;
	}

	if( pBlockInfo->ui64BytesUsed > (FLMUINT64)40000000)
	{
		uiPercent = (FLMUINT)( pBlockInfo->ui64BytesUsed / 
			(ui64TotalBytes / (FLMUINT64)100));
	}
	else if( ui64TotalBytes)
	{
		uiPercent = (FLMUINT)(( pBlockInfo->ui64BytesUsed * 
			(FLMUINT64)100) / ui64TotalBytes);
	}
	else
	{
		uiPercent = 0;
	}

	f_sprintf( pucTmpBuf, "%-12s %10u  %11u  %10u  %5u  %11u  %8u",
		pucLabel, (unsigned)pBlockInfo->uiBlockCount, (unsigned)ui64TotalBytes,
		(unsigned)pBlockInfo->ui64BytesUsed,
		(unsigned)uiPercent, (unsigned)pBlockInfo->ui64ElementCount,
		(unsigned)uiAvgElementSize);
		
	OutLine( pucTmpBuf);

	if( pBlockInfo->ui64ContElementCount)
	{
		uiAvgElementSize = (FLMUINT)( pBlockInfo->ui64ContElmBytes /
															pBlockInfo->ui64ContElementCount);
															
		if( pBlockInfo->ui64ContElmBytes > (FLMUINT64)40000000)
		{
			uiPercent =
				(FLMUINT)( pBlockInfo->ui64ContElmBytes / 
				(ui64TotalBytes / (FLMUINT64)100));
		}
		else if( ui64TotalBytes)
		{
			uiPercent =
				(FLMUINT)(( pBlockInfo->ui64ContElmBytes * (FLMUINT64)100) /
				ui64TotalBytes);
		}
		else
		{
			uiPercent = 0;
		}

		f_sprintf( pucTmpBuf, "%-12s                         "
			"%10u  %5u  %11u  %8u", "    ContElm",
			(unsigned)pBlockInfo->ui64ContElmBytes,
			(unsigned)uiPercent, (unsigned)pBlockInfo->ui64ContElementCount,
			(unsigned)uiAvgElementSize);
			
		OutLine( pucTmpBuf);
	}

	if( ui64KeyCount)
	{
		f_sprintf( pucTmpBuf, "%-12s                         %10u",
			"		KeyCnt", (unsigned)ui64KeyCount);
		OutLine( pucTmpBuf);
	}

	if( ui64RefCount)
	{
		f_sprintf( pucTmpBuf, "%-12s                         %10u",
			"    RefCnt", (unsigned)ui64RefCount);
		OutLine( pucTmpBuf);
	}

	if( ui64FldCount)
	{
		f_sprintf( pucTmpBuf, "%-12s                         %10u",
			"    FldCnt", (unsigned)ui64FldCount);
		OutLine( pucTmpBuf);
	}

	if( pBlockInfo->uiNumErrors)
	{
		f_strcpy( pucTmpBuf, "    LAST ERROR: ");
		f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)],
			FlmVerifyErrToStr( pBlockInfo->eCorruption));
		OutLine( pucTmpBuf);
		f_sprintf( pucTmpBuf,
			"    TOTAL ERRORS: %u", (unsigned)pBlockInfo->uiNumErrors);
		OutLine( pucTmpBuf);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutLogicalFile(
	DB_CHECK_PROGRESS *	pCheckProgress,
	FLMUINT					uiIndex)
{
	char			pucTmpBuf[ 100];
	LF_STATS *	pLfStats;
	FLMUINT		uiLoop;

	pLfStats = &pCheckProgress->pLfStats[ uiIndex];

	switch( pLfStats->uiLfType)
	{
		case LF_CONTAINER:
			OutUINT( "CONTAINER", pLfStats->uiContainerNum);
			break;
		case LF_INDEX:
			OutUINT( "INDEX", pLfStats->uiIndexNum);
			OutUINT( "  Index Container Number", pLfStats->uiContainerNum);
			break;
	}
	
	if( !pLfStats->uiNumLevels)
	{
		OutUINT( "  Levels", pLfStats->uiNumLevels);
	}
	else
	{
		OutBlkHeader();
		
		for( uiLoop = 0; uiLoop < pLfStats->uiNumLevels; uiLoop++)
		{
			f_sprintf( pucTmpBuf, "  Level %u", (unsigned)uiLoop);
			
			if( !uiLoop)
			{
				OutOneBlockStat( pucTmpBuf,
					pCheckProgress->uiBlockSize,
					&pLfStats->pLevelInfo[ uiLoop].BlockInfo,
					pLfStats->pLevelInfo[ uiLoop].ui64KeyCount,
					(FLMUINT64)((pLfStats->uiLfType == LF_INDEX)
								 ? pLfStats->ui64FldRefCount
								 : (FLMUINT64)0),
					(FLMUINT64)((pLfStats->uiLfType == LF_INDEX)
								 ? (FLMUINT64)0
								 : pLfStats->ui64FldRefCount));
			}
			else
			{
				OutOneBlockStat( pucTmpBuf,
					pCheckProgress->uiBlockSize,
					&pLfStats->pLevelInfo[ uiLoop].BlockInfo,
					pLfStats->pLevelInfo[ uiLoop].ui64KeyCount,
					(FLMUINT64)0, (FLMUINT64)0);
			}
		}
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void PrintInfo(
	DB_CHECK_PROGRESS *	pCheckProgress)
{
	FLMUINT		uiLoop;

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conClearScreen( 0, 0);

	OutUINT( "Default Language", pCheckProgress->uiDefaultLanguage);
	OutUINT64( "DB Size", pCheckProgress->ui64DatabaseSize);
	OutUINT( "Field Count", pCheckProgress->uiNumFields);
	OutUINT( "Index Count", pCheckProgress->uiNumIndexes);
	OutUINT( "Non-Default Container Count", pCheckProgress->uiNumContainers);
	OutUINT( "Block Size", pCheckProgress->uiBlockSize);

	if( (pCheckProgress->AvailBlocks.uiBlockCount) ||
		(pCheckProgress->LFHBlocks.uiBlockCount))
	{
		OutLine( "MISCELLANEOUS BLOCK STATISTICS");
		OutBlkHeader();

		if( pCheckProgress->AvailBlocks.uiBlockCount)
		{
			OutOneBlockStat( "  Avail", pCheckProgress->uiBlockSize,
				&pCheckProgress->AvailBlocks, 0, 0, 0);
		}

		if( pCheckProgress->LFHBlocks.uiBlockCount)
		{
			OutOneBlockStat( "  LFH", pCheckProgress->uiBlockSize,
				&pCheckProgress->LFHBlocks, 0, 0, 0);
		}
	}

	for( uiLoop = 0; uiLoop < pCheckProgress->uiNumLogicalFiles; uiLoop++)
	{
		OutLogicalFile( pCheckProgress, uiLoop);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMUINT CheckShowError(
	const char *	pucMessage,
	FLMBOOL			bLogIt)
{
	FLMUINT			uiResKey;

	f_sprintf( gv_pucLastError, "%s", pucMessage);
	
	if( bLogIt)
	{
		LogString( pucMessage);
	}

	if( gv_bBatchMode)
	{
		uiResKey = 0;
	}
	else
	{
		f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_RED, FLM_WHITE);
		f_conStrOut( pucMessage);
		f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 1));
		f_conStrOut( "Press ENTER to continue, ESC to quit");
		
		for( ;;)
		{
			if( gv_bShutdown)
			{
				uiResKey = FKB_ESCAPE;
				break;
			}
			else if( f_conHaveKey())
			{
				uiResKey = f_conGetKey();
				if( (uiResKey == FKB_ENTER) || (uiResKey == FKB_ESCAPE))
				{
					break;
				}
			}
		}
		
		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 2));
	}

	return( uiResKey);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutLabel(
	FLMUINT			uiCol,
	FLMUINT			uiRow,
	const char *	pucLabel,
	const char *	pucValue,
	FLMUINT64		ui64NumValue,
	FLMBOOL			bLogIt)
{
	char			pucTmpBuf[ 100];
	FLMUINT		uiLoop;

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conStrOutXY( pucLabel, uiCol, uiRow);

	for( uiLoop = f_conGetCursorColumn(); uiLoop < VALUE_COLUMN - 1; uiLoop++)
	{
		f_conStrOut( ".");
	}

	if( pucValue != NULL)
	{
		DisplayValue( uiRow, pucValue);
	}
	else
	{
		DisplayNumValue( uiRow, ui64NumValue);
	}

	if( (bLogIt) && (gv_bLoggingEnabled))
	{
		f_strcpy( pucTmpBuf, pucLabel);
		f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], ": ");
		if( pucValue != NULL)
		{
			f_strcpy( &pucTmpBuf[ f_strlen( pucTmpBuf)], pucValue);
		}
		else
		{
			f_sprintf( (&pucTmpBuf[ f_strlen( pucTmpBuf)]),
				"%u", (unsigned)ui64NumValue);
		}
		LogString( pucTmpBuf);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void DisplayValue(
	FLMUINT			uiRow,
	const char *	pucValue)
{
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conStrOutXY( pucValue, VALUE_COLUMN, uiRow);
	f_conClearLine( 255, 255);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void DisplayNumValue(
	FLMUINT		uiRow,
	FLMUINT64	ui64Number)
{
	FLMUINT		uiDigit;
	char			pucTmpBuf[ 80];
	char			pucDisplayBuf[ 80];
	FLMUINT		uiOffset;
	FLMUINT64	ui64Tmp;
	FLMUINT		uiLen;

	ui64Tmp = ui64Number;
	uiOffset = 0;
	do
	{
		uiDigit = (FLMUINT)(ui64Tmp % (FLMUINT64)10);
		ui64Tmp /= (FLMUINT64)10;
		pucTmpBuf[ uiOffset++] = (FLMBYTE)(uiDigit + NATIVE_ZERO);
	} while( ui64Tmp);
	pucTmpBuf[ uiOffset] = 0;
	uiLen = uiOffset;

	f_memset( pucDisplayBuf, NATIVE_SPACE, sizeof( pucDisplayBuf));
	uiOffset = 0;
	while( uiLen)
	{
		pucDisplayBuf[ uiOffset++] = pucTmpBuf[ --uiLen];
	}
	pucDisplayBuf[ 16] = 0;

	f_sprintf( pucTmpBuf, "   0x%08X%08X",
			(unsigned)(ui64Number >> 32),
			(unsigned)(ui64Number & (FLMUINT64)0xFFFFFFFF));
	f_strcat( &pucDisplayBuf[ 16], pucTmpBuf);	

	DisplayValue( uiRow, pucDisplayBuf);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC RCODE GetUserInput( void)
{
	FLMUINT		uiChar;

	f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 1));
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 1));
	f_conSetBackFore( FLM_RED, FLM_WHITE);

	f_conStrOut( "Q,ESC=Quit, Other=Continue");
	
	for( ;;)
	{
		if( gv_bShutdown)
		{
			uiChar = FKB_ESCAPE;
			break;
		}
		else if( f_conHaveKey())
		{
			uiChar = f_conGetKey();
			break;
		}
	}

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 1));

	switch( uiChar)
	{
		case 'q':
		case 'Q':
		case FKB_ESCAPE:
			return( RC_SET( FERR_FAILURE));
		default:
			break;
	}

	return( FERR_OK);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE ProgFunc(
	eStatusType	eStatus,
	void *		Parm1,
	void *		Parm2,
	void *		pvAppData)
{
	RCODE							rc = FERR_OK;
	DB_CHECK_PROGRESS *		pProgress;
	DB_COPY_INFO *				pDbCopyInfo;
	CORRUPT_INFO *				pCorrupt;
	FLM_MEM_INFO				memInfo;
	char							pucWhat[ 256];
	char							pucLfName[ 128];

	F_UNREFERENCED_PARM( pvAppData);

	FlmGetMemoryInfo( &memInfo);
	DisplayNumValue( CACHE_USED_ROW, memInfo.BlockCache.uiTotalBytesAllocated +
		memInfo.RecordCache.uiTotalBytesAllocated);

	if (eStatus == FLM_DB_COPY_STATUS)
	{
		if( gv_bShutdown)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
		
		pDbCopyInfo = (DB_COPY_INFO *)Parm1;
		
		if (pDbCopyInfo->bNewSrcFile)
		{
			gv_ui64DatabaseSize = pDbCopyInfo->ui64BytesToCopy;
			DisplayNumValue( DB_SIZE_ROW, (FLMUINT)gv_ui64DatabaseSize);
			f_sprintf( pucWhat, "SAVING FILE: %s",
								pDbCopyInfo->szSrcFileName);
			DisplayValue( DOING_ROW, pucWhat);
		}
		
		OutLabel( LABEL_COLUMN, AMOUNT_DONE_ROW, "Bytes Saved",
						NULL, pDbCopyInfo->ui64BytesCopied, FALSE);
						
		goto Exit;
	}
	else if (eStatus == FLM_PROBLEM_STATUS)
	{
		FLMBOOL *	pbFixCorruptions = (FLMBOOL *)Parm2;

		pCorrupt = (CORRUPT_INFO *)Parm1;
		if( (gv_bLoggingEnabled) &&
			 ((gv_bShowStats) ||
			 (pCorrupt->eCorruption != FLM_OLD_VIEW)))
		{
			LogCorruptError( pCorrupt);
		}

		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		if( pCorrupt->eCorruption == FLM_OLD_VIEW)
		{
			gv_uiOldViewCount++;
			DisplayNumValue( OLD_VIEW_ROW, gv_uiOldViewCount);
		}
		else
		{
			gv_uiCorruptCount++;
			gv_uiTotalCorruptions++;
			DisplayNumValue( CORRUPT_ROW, gv_uiCorruptCount);
			DisplayNumValue( TOTAL_CORRUPT_ROW, gv_uiTotalCorruptions);
		}
		if (pbFixCorruptions)
		{
			*pbFixCorruptions = gv_bRepairCorruptions;
		}
	}
	else if (eStatus == FLM_CHECK_STATUS)
	{
		pProgress = (DB_CHECK_PROGRESS *)Parm1;
		if( gv_bShutdown)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		// Update the display first

		gv_ui64BytesDone = pProgress->ui64BytesExamined;
		DisplayNumValue( TOTAL_KEYS_ROW, pProgress->ui64NumKeys);
		DisplayNumValue( TOTAL_KEYS_EXAM_ROW, pProgress->ui64NumKeysExamined);
		DisplayNumValue( CONFLICT_ROW, pProgress->ui64NumConflicts);
		DisplayNumValue( BAD_IXREF_ROW, pProgress->ui64NumKeysNotFound);
		DisplayNumValue( MISSING_IXREF_ROW, pProgress->ui64NumRecKeysNotFound);
		DisplayNumValue( NONUNIQUE_ROW, pProgress->ui64NumNonUniqueKeys);

		DisplayNumValue( REPAIR_ROW, pProgress->uiNumProblemsFixed);
		gv_uiRepairCount = pProgress->uiNumProblemsFixed;

		if( pProgress->iCheckPhase == CHECK_RS_SORT)
		{
			FLMUINT		uiPercent = 0;

			if( pProgress->ui64NumRSUnits > (FLMUINT64)0)
			{
					uiPercent = 
						(FLMUINT)((pProgress->ui64NumRSUnitsDone * (FLMUINT64)100) /
							pProgress->ui64NumRSUnits);
			}

			OutLabel( LABEL_COLUMN, AMOUNT_DONE_ROW, "Percent Sorted",
					NULL, uiPercent, FALSE);
		}
		else
		{
			OutLabel( LABEL_COLUMN, AMOUNT_DONE_ROW, "Bytes Checked",
					NULL, gv_ui64BytesDone, FALSE);
		}

		if( pProgress->bStartFlag)
		{
			gv_ui64DatabaseSize = pProgress->ui64DatabaseSize;
			DisplayNumValue( DB_SIZE_ROW, gv_ui64DatabaseSize);

			switch( pProgress->iCheckPhase)
			{
				case CHECK_LFH_BLOCKS:
					f_strcpy( pucWhat, "LFH BLOCKS");
					break;
				case CHECK_B_TREE:
					*pucLfName = '\0';
					if( pProgress->uiLfType == LF_INDEX)
					{
						if( pProgress->bUniqueIndex)
						{
							f_strcpy( pucWhat, "UNIQUE INDEX: ");
						}
						else
						{
							f_strcpy( pucWhat, "INDEX: ");
						}

						NumToName( pProgress->uiLfNumber, pucLfName);
					}
					else if( pProgress->uiLfType == LF_CONTAINER)
					{
						f_strcpy( pucWhat, "CONTAINER: ");
						NumToName( pProgress->uiLfNumber, pucLfName);
					}
					else
					{
						f_strcpy( pucWhat, "DICT CONTAINER: ");
						NumToName( pProgress->uiLfNumber, pucLfName);
					}

					f_strcpy( &pucWhat[ f_strlen( pucWhat)], pucLfName);

					f_sprintf( (&pucWhat[ f_strlen( pucWhat)]), " (%u)",
						(unsigned)pProgress->uiLfNumber);
					pucWhat[ 50] = '\0';
					break;
				case CHECK_AVAIL_BLOCKS:
					f_strcpy( pucWhat, "AVAIL BLOCKS");
					break;
				case CHECK_RS_SORT:
					f_sprintf( pucWhat, "SORTING INDEX KEYS");
					break;
				default:
					break;
			}

			pucWhat[ 45] = '\0';
			f_conSetBackFore( FLM_BLUE, FLM_WHITE);
			DisplayValue( DOING_ROW, pucWhat);
		}
		else if( (f_conHaveKey()) && (f_conGetKey() == FKB_ESCAPE))
		{
			f_conSetBackFore( FLM_BLUE, FLM_WHITE);
			f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 2));
			f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 2));
			f_conSetBackFore( FLM_RED, FLM_WHITE);
			f_conStrOut( "ESCAPE key pressed.\n");
			
			rc = GetUserInput();
			
			f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 2));
			f_conSetBackFore( FLM_BLUE, FLM_WHITE);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogStr(
	FLMUINT			uiIndent,
	const char *	pucStr)
{
	FLMUINT		uiLoop;

	if( gv_bLoggingEnabled)
	{
		for( uiLoop = 0; uiLoop < uiIndent; uiLoop++)
		{
			gv_pucLogBuffer[ gv_uiLogBufferCount++] = ' ';
			if( gv_uiLogBufferCount == MAX_LOG_BUFF)
			{
				LogFlush();
			}
		}
		LogString( pucStr);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogCorruptError(
	CORRUPT_INFO *	pCorrupt)
{
	char		pucWhat[ 20];
	char		pucTmpBuf[ 100];

	switch( pCorrupt->eErrLocale)
	{
		case LOCALE_LFH_LIST:
			LogStr( 0, "ERROR IN LFH LINKED LIST:");
			break;
		case LOCALE_AVAIL_LIST:
			LogStr( 0, "ERROR IN AVAIL LINKED LIST:");
			break;
		case LOCALE_B_TREE:
			if( pCorrupt->eCorruption == FLM_OLD_VIEW)
			{
				LogStr( 0, "OLD VIEW");
			}
			else
			{
				if( pCorrupt->uiErrFieldNum)
				{
					f_strcpy( pucWhat, "FIELD");
				}
				else if( pCorrupt->uiErrElmOffset)
				{
					f_strcpy( pucWhat, "ELEMENT");
				}
				else if( pCorrupt->uiErrBlkAddress)
				{
					f_strcpy( pucWhat, "BLOCK");
				}
				else
				{
					f_strcpy( pucWhat, "LAST BLOCK");
				}
				f_sprintf( pucTmpBuf, "BAD %s", pucWhat);
				LogStr( 0, pucTmpBuf);
			}

			// Log the logical file number, name, and type

			f_sprintf( pucTmpBuf, "Logical File Number: %u",
				(unsigned)pCorrupt->uiErrLfNumber);
			LogStr( 2, pucTmpBuf);
			
			switch( pCorrupt->uiErrLfType)
			{
				case LF_CONTAINER:
					f_strcpy( pucWhat, "Container");
					break;
				case LF_INDEX:
					f_strcpy( pucWhat, "Index");
					break;
				default:
					f_sprintf( pucWhat, "?%u", (unsigned)pCorrupt->uiErrLfType);
					break;
			}
			f_sprintf( pucTmpBuf, "Logical File Type: %s", pucWhat);
			LogStr( 2, pucTmpBuf);

			// Log the level in the B-Tree, if known

			if( pCorrupt->uiErrBTreeLevel != 0xFF)
			{
				f_sprintf( pucTmpBuf, "Level in B-Tree: %u",
					(unsigned)pCorrupt->uiErrBTreeLevel);
				LogStr( 2, pucTmpBuf);
			}
			break;
		case LOCALE_IXD_TBL:
			f_sprintf( pucTmpBuf, "ERROR IN IXD TABLE, Index Number: %u",
				(unsigned)pCorrupt->uiErrLfNumber);
			LogStr( 0, pucTmpBuf);
			break;
		case LOCALE_INDEX:
			f_strcpy( pucWhat, "Index");
			LogKeyError( pCorrupt);
			break;
		default:
			pCorrupt->eErrLocale = LOCALE_NONE;
			break;
	}

	// Log the block address, if known

	if( pCorrupt->uiErrBlkAddress)
	{
		f_sprintf( pucTmpBuf, "Block Address: 0x%08X (%u)",
			(unsigned)pCorrupt->uiErrBlkAddress,
			(unsigned)pCorrupt->uiErrBlkAddress);
		LogStr( 2, pucTmpBuf);
	}

	// Log the parent block address, if known

	if( pCorrupt->uiErrParentBlkAddress)
	{
		if( pCorrupt->uiErrParentBlkAddress != 0xFFFFFFFF)
		{
			f_sprintf( pucTmpBuf, "Parent Block Address: 0x%08X (%u)",
				(unsigned)pCorrupt->uiErrParentBlkAddress,
				(unsigned)pCorrupt->uiErrParentBlkAddress);
		}
		else
		{
			f_sprintf( pucTmpBuf,
				"Parent Block Address: NONE, Root Block");
		}
		LogStr( 2, pucTmpBuf);
	}

	// Log the element offset, if known

	if( pCorrupt->uiErrElmOffset)
	{
		f_sprintf( pucTmpBuf, "Element Offset: %u", 
				(unsigned)pCorrupt->uiErrElmOffset);
	}

	// Log the record number, if known

	if( pCorrupt->uiErrDrn)
	{
		f_sprintf( pucTmpBuf, 
			"Record Number: %u", (unsigned)pCorrupt->uiErrDrn);
		LogStr( 2, pucTmpBuf);
	}

	// Log the offset within the element record, if known

	if( pCorrupt->uiErrElmRecOffset != 0xFFFF)
	{
		f_sprintf( pucTmpBuf, "Offset Within Element: %u",
			(unsigned)pCorrupt->uiErrElmRecOffset);
		LogStr( 2, pucTmpBuf);
	}

	// Log the field number, if known

	if( pCorrupt->uiErrFieldNum)
	{
		f_sprintf( pucTmpBuf, 
				"Field Number: %u", (unsigned)pCorrupt->uiErrFieldNum);
		LogStr( 2, pucTmpBuf);
	}

	f_strcpy( pucTmpBuf, FlmVerifyErrToStr( pCorrupt->eCorruption));
	f_sprintf( (&pucTmpBuf[ f_strlen( pucTmpBuf)]), " (%d)",
		(int)pCorrupt->eCorruption);
	LogStr( 2, pucTmpBuf);
	LogStr( 0, NULL);
	
	if( gv_bLoggingEnabled)
	{
		gv_pLogFile->flush();
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogKeyError(
	CORRUPT_INFO *	pCorrupt
	)
{
	FLMUINT		uiLogItem;
	FlmRecord *	pRecord = NULL;
	void *		pvField;
	REC_KEY *	pTempKeyList = NULL;
	FLMUINT		uiIndent;
	FLMUINT		uiLevelOffset;
	char			pucNameBuf[ 200];
	char			pucTmpBuf[ 200];
	
	NumToName( pCorrupt->uiErrLfNumber, pucNameBuf);
	
	LogString( NULL);
	LogString( NULL);
	
	f_sprintf( pucTmpBuf, "ERROR IN INDEX: %s", pucNameBuf);
	LogString( pucTmpBuf);
	
	uiLogItem = 'R';
	uiLevelOffset = 0;
	for( ;;)
	{
		uiIndent = 2;
		if( uiLogItem == 'K')
		{
			if( (pRecord = pCorrupt->pErrIxKey) == NULL)
			{
				uiLogItem = 'L';
				continue;
			}
			LogString( NULL);
			LogString( " PROBLEM KEY");
		}
		else if( uiLogItem == 'R')
		{
			if( (pRecord = pCorrupt->pErrRecord) == NULL)
			{
				uiLogItem = 'K';
				continue;
			}
			LogString( NULL);
			LogString( " RECORD");
		}
		else if( uiLogItem == 'L')
		{
			if( (pTempKeyList =
				pCorrupt->pErrRecordKeyList) == NULL)
			{
				break;
			}
			pRecord = pTempKeyList->pKey;
			LogString( NULL);
			LogString( " RECORD KEYS");
			LogString( "  0 Key");
			uiLevelOffset = 1;
		}

		for ( pvField = pRecord->root();;)
		{
			if (!pvField)
			{
				if (uiLogItem != 'L')
					break;
				if ((pTempKeyList = pTempKeyList->pNextKey) == NULL)
					break;
				pRecord = pTempKeyList->pKey;
				pvField = pRecord->root();
				LogString( "  0 Key");
				continue;
			}
			else
			{
				DisplayField( pRecord, pvField, uiIndent, uiLevelOffset);
			}
			pvField = pRecord->next( pvField);
		}

		if( uiLogItem == 'L')
		{
			break;
		}
		else if( uiLogItem == 'R')
		{
			uiLogItem = 'K';
		}
		else
		{
			uiLogItem = 'L';
		}
	}
}

/***************************************************************************
Desc:    This routine displays a field to the screen.
*****************************************************************************/
FSTATIC FLMBOOL DisplayField(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiStartCol,
	FLMUINT			uiLevelOffset)
{
	char				pucTmpBuf[ 200];
	FLMUINT			uiLoop;
	FLMUINT			uiLen;
	FLMUINT			uiBinLen;
	FLMUINT			uiTmpLen;
	char *			pucTmp;
	char				ucTmpBin [80];
	FLMUINT			uiNum;
	FLMUINT			uiLevel = pRecord->getLevel( pvField) + uiLevelOffset;
	FLMUINT			uiIndent = (uiLevel * 2) + uiStartCol;

	// Insert leading spaces to indent for level

	for( uiLoop = 0; uiLoop < uiIndent; uiLoop++)
	{
		pucTmpBuf[ uiLoop] = ' ';
	}

	// Output level and tag

	f_sprintf( (&pucTmpBuf[ uiIndent]), "%u ", (unsigned)uiLevel);
	NumToName( pRecord->getFieldID( pvField), &pucTmpBuf[ f_strlen( pucTmpBuf)]);

	// Output what will fit of the value on the rest of the line

	uiLen = f_strlen( pucTmpBuf);
	pucTmpBuf[ uiLen++] = ' ';
	pucTmpBuf[ uiLen] = 0;
	
	if (!pRecord->getDataLength( pvField))
	{
		goto Exit;
	}
	
	switch( pRecord->getDataType( pvField))
	{
		case FLM_TEXT_TYPE:
			pucTmp = &pucTmpBuf[ uiLen];
			uiLen = 80 - uiLen;
			pRecord->getNative( pvField, pucTmp, &uiLen);
			break;
		case FLM_NUMBER_TYPE:
			pRecord->getUINT( pvField, &uiNum);
			f_sprintf( (&pucTmpBuf [uiLen]), "%u", (unsigned)uiNum);
			break;
		case FLM_BINARY_TYPE:
			pRecord->getBinaryLength( pvField, &uiBinLen);
			uiTmpLen = sizeof( ucTmpBin);
			pRecord->getBinary( pvField, ucTmpBin, &uiTmpLen);
			pucTmp = &ucTmpBin [0];
			while (uiBinLen && uiLen < 77)
			{
				f_sprintf( &pucTmpBuf [uiLen], "%02X ", (unsigned)*pucTmp);
				uiBinLen--;
				pucTmp++;
				uiLen += 3;
			}
			pucTmpBuf [uiLen - 1] = 0;
			break;
		case FLM_CONTEXT_TYPE:
			pRecord->getUINT( pvField, &uiNum);
			f_sprintf( (&pucTmpBuf[ uiLen]), "@%u@", (unsigned)uiNum);
			break;
	}

Exit:

	LogString( pucTmpBuf);
	return( TRUE);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void NumToName(
	FLMUINT			uiNum,
	char *			pucBuf)
{
	if( !gv_pNameTable || 
		 !gv_pNameTable->getFromTagNum( uiNum, NULL, pucBuf, 128))
	{
		f_sprintf( pucBuf, "#%u", (unsigned)uiNum);
	}
}
