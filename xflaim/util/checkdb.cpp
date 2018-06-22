//------------------------------------------------------------------------------
// Desc: Checks a database for corruptions
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

#define UTIL_ID		"CHECKDB"

#define LABEL_COLUMN   				5
#define VALUE_COLUMN   				30

#define LOG_FILE_ROW					1
#define SOURCE_ROW					2
#define DATA_DIR_ROW					3
#define RFL_DIR_ROW					4
#define CACHE_USED_ROW				5
#define DOING_ROW						6
#define FILE_SIZE_ROW				7
#define AMOUNT_DONE_ROW				8

#define TOTAL_DOM_NODES_ROW		9
#define DOM_LINKS_VERIFIED_ROW	10
#define TOTAL_BROKEN_LINKS_ROW	11

#define TOTAL_KEYS_ROW				12
#define TOTAL_DUPS_ROW				13
#define TOTAL_KEYS_EXAM_ROW		14
#define BAD_IXREF_ROW				15
#define MISSING_IXREF_ROW			16
#define CONFLICT_ROW					17
#define CORRUPT_ROW					18
#define TOTAL_CORRUPT_ROW			19
#define REPAIR_ROW					20
#define OLD_VIEW_ROW					21
#define MISMATCH_ROW					22

#define MAX_LOG_BUFF					2048

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

#define SCREEN_REFRESH_RATE 		100

/********************************************************************
Desc:
*********************************************************************/
class F_LocalCheckStatus : public IF_DbCheckStatus
{
public:

	F_LocalCheckStatus()
	{
		m_uiLastRefresh = 0;
	}

	RCODE XFLAPI reportProgress(
		XFLM_PROGRESS_CHECK_INFO *	pProgCheck);
	
	RCODE XFLAPI reportCheckErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo,
		FLMBOOL *				pbFix);

private:
	FLMUINT	m_uiLastRefresh;
};

FSTATIC FLMBOOL CheckDatabase( void);

FSTATIC FLMBOOL DoCheck( void);

FSTATIC void CheckShowHelp(
	FLMBOOL		bShowFullUsage);

FSTATIC FLMBOOL GetParams(
	FLMINT		iArgC,
	char **		ppszArgV);

FSTATIC void OutLabel(
	FLMUINT			uiCol,
	FLMUINT			uiRow,
	const char *	pszLabel,
	const char *	pszValue,
	FLMUINT64		ui64NumValue,
	FLMBOOL			bLogIt);

FSTATIC void OutLine(
	const char *	pszString);

FSTATIC void LogFlush( void);

FSTATIC void LogString(
	const char *	pszString);

FSTATIC void DisplayValue(
	FLMUINT			uiRow,
	const char *	pszValue);

FSTATIC void DisplayNumValue(
	FLMUINT			uiRow,
	FLMUINT64		ui64Number);

FSTATIC void OutValue(
	const char *	pszLabel,
	const char *	pszValue);

FSTATIC void OutUINT(
	const char *	pszLabel,
	FLMUINT			uiNum);

FSTATIC void OutUINT64(
	const char *	pszLabel,
	FLMUINT64		ui64Num);

FSTATIC void OutBlkHeader( void);

FSTATIC void OutOneBlockStat(
	const char *	pszLabel,
	FLMUINT			uiBlockSize,
	FLMUINT64		ui64KeyCount,
	FLMUINT64		ui64BytesUsed,
	FLMUINT64		ui64ElementCount,
	FLMUINT64		ui64ContElementCount,
	FLMUINT64		ui64ContElmBytes,
	FLMUINT			uiBlockCount,
	FLMINT32			i32LastError,
	FLMUINT			uiNumErrors);

FSTATIC void OutLogicalFile(
	IF_DbInfo *		pDbInfo,
	FLMUINT			uiIndex);

FSTATIC void PrintInfo(
	IF_DbInfo *		pDbInfo);

FSTATIC FLMUINT CheckShowError(
	const char *	pszMessage,
	FLMBOOL			bLogIt);

FSTATIC RCODE GetUserInput( void);

FSTATIC void LogStr(
	FLMUINT			uiIndent,
	const char *	pszStr);

FSTATIC void LogCorruptError(
	XFLM_CORRUPT_INFO *	pCorrupt);

FSTATIC void LogKeyError(
	XFLM_CORRUPT_INFO *	pCorrupt);

FSTATIC FLMBOOL DisplayField(
	IF_DataVector *	ifpKey,
	FLMUINT				uiElementNumber,
	FLMUINT				uiStartCol,
	FLMUINT				uiLevelOffset);

FSTATIC FLMBOOL NumToName(
	FLMUINT		uiNum,
	FLMUINT		uiType,
	char *		pszBuf);

FLMBOOL						gv_bShutdown = FALSE;
static IF_FileHdl *		gv_pLogFile = NULL;
static IF_DbInfo *		gv_pDbInfo = NULL;
static F_Db *				gv_pDb = NULL;
static F_NameTable *		gv_pNameTable = NULL;
static FLMUINT				gv_uiMaxRow;
static char					gv_szLogFileName[ F_PATH_MAX_SIZE];
static char					gv_szTmpDir[ F_PATH_MAX_SIZE];
static char					gv_szLastError[ 256];
static FLMUINT				gv_uiLineCount;
static char					gv_szDbFileName[ F_PATH_MAX_SIZE];
static char					gv_szDataDir[ F_PATH_MAX_SIZE];
static char					gv_szRflDir[ F_PATH_MAX_SIZE];
static char *				gv_pszLogBuffer = NULL;
static FLMUINT64			gv_ui64FileSize;
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
static FLMBOOL				gv_bSkipDomLinkVerify = FALSE;
static char					gv_szPassword[256];
static IF_DbSystem *		gv_pDbSystem = NULL;

#ifdef FLM_RING_ZERO_NLM
	#define main		nlm_main
#endif

/********************************************************************
Desc:
*********************************************************************/
extern "C" int main(
	int					iArgC,
	char **				ppszArgV)
{
	int		iResCode = 0;
	F_Pool	logPool;

	logPool.poolInit( 1024);
	gv_bBatchMode = FALSE;
	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	gv_szLastError[ 0] = '\0';

	if( RC_BAD( FlmAllocDbSystem( &gv_pDbSystem)))
	{
		f_conStrOut( "\nCould not initialize FLAIM.\n");
		goto Exit;
	}
	
	f_conInit( 0xFFFF, 0xFFFF, "FLAIM Database Check");
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conDrawBorder();
	f_conClearScreen( 0, 0);
	f_conGetScreenSize( NULL, &gv_uiMaxRow);
	
	if (RC_BAD( logPool.poolAlloc( MAX_LOG_BUFF, (void **)&gv_pszLogBuffer)))
	{
		f_conStrOut( "\nFailed to allocatae memory pool\n");
		goto Exit;
	}

	if( GetParams( iArgC, ppszArgV))
	{
		if (!DoCheck())
		{
			iResCode = 1;
		}
	}
	
	logPool.poolReset( NULL);

	if( (gv_bPauseBeforeExiting) && (!gv_bShutdown))
	{
		f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		f_conClearScreen( 0, (FLMUINT)(gv_uiMaxRow - 2));
		f_conSetBackFore( FLM_RED, FLM_WHITE);
		if( gv_szLastError[ 0] != '\0')
		{
			f_conStrOut( gv_szLastError);
		}
		
		f_conSetCursorPos( 0, (FLMUINT)(gv_uiMaxRow - 1));
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

	if (gv_pDbInfo)
	{
		gv_pDbInfo->Release();
	}

	logPool.poolFree();	

	f_conExit();
	
	if( gv_pDbSystem)
	{
		gv_pDbSystem->Release();
	}

	gv_bRunning = FALSE;
	return( iResCode);
}

/********************************************************************
Desc: Check the database...
*********************************************************************/
FSTATIC FLMBOOL CheckDatabase( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiStatus;
	char						szTmpBuf[ 100];
	FLMUINT					uiCheckFlags;
	FLMBOOL					bOk = TRUE;
	IF_DbCheckStatus *	pDbCheckStatus = NULL;

	// Open the database - so we can have access to its name table

	if (!gv_pDb)
	{
		if( RC_BAD( rc = gv_pDbSystem->dbOpen(
						gv_szDbFileName, gv_szDataDir,
						gv_szRflDir, gv_szPassword, XFLM_ALLOW_LIMITED_MODE,
						(IF_Db **)&gv_pDb)))
		{
			f_sprintf( szTmpBuf, "Error opening database: 0x%04X", (unsigned)rc);
			CheckShowError( szTmpBuf, TRUE);
			bOk = FALSE;
			goto Exit;
		}
	}

	if (gv_pNameTable)
	{
		gv_pNameTable->Release();
		gv_pNameTable = NULL;
	}
	(void)gv_pDb->getNameTable( &gv_pNameTable);

	gv_uiCorruptCount = 0;
	gv_ui64BytesDone = 0;
	gv_ui64FileSize = 0;
	gv_uiOldViewCount = 0;
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);

	if (gv_bLoggingEnabled)
	{
		LogString( NULL);
		LogString( NULL);
		LogString( NULL);
		LogString(
"==========================================================================");
		LogString( "CHECK PARAMETERS:");
	}
	OutLabel( LABEL_COLUMN,
				 SOURCE_ROW,
				 "Database",
				 gv_szDbFileName,
				 0,
				 TRUE);
				 
	OutLabel( LABEL_COLUMN,
				 DATA_DIR_ROW,
				 "Data Files Dir.",
				 gv_szDataDir [0]
						? &gv_szDataDir [0]
						: "<Same as DB>",
				 0,
				 TRUE);
				 
	OutLabel( LABEL_COLUMN,
				 RFL_DIR_ROW,
				 "RFL Files Dir.",
				 gv_szRflDir [0]
						? &gv_szRflDir [0]
						: "<Same as DB>",
				 0,
				 TRUE);
				 
	OutLabel( LABEL_COLUMN,
				 LOG_FILE_ROW,
				 "Log File",
				 (gv_szLogFileName[ 0])
						? &gv_szLogFileName[ 0]
						: "<NONE>",
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 CACHE_USED_ROW,
				 "Cache Bytes Used",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 DOING_ROW,
				 "Doing",
				 "Opening Database File",
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 FILE_SIZE_ROW,
				 "File Size",
				 NULL,
				 (FLMUINT)gv_ui64FileSize,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 CORRUPT_ROW,
				 "File Corruptions",
				 NULL,
				 gv_uiCorruptCount,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_CORRUPT_ROW,
				 "Total Corruptions",
				 NULL,
				 gv_uiTotalCorruptions,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 OLD_VIEW_ROW,
				 "Old View Count",
				 NULL,
				 gv_uiOldViewCount,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 MISMATCH_ROW,
				 "Mismatch Count",
				 NULL,
				 gv_uiMismatchCount,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 REPAIR_ROW,
				 "Problems Repaired",
				 NULL,
				 gv_uiRepairCount,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_KEYS_ROW,
				 "Total Index Keys",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_DUPS_ROW,
				 "Total Duplicate Keys",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 CONFLICT_ROW,
				 "Key Conflicts",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_KEYS_EXAM_ROW,
				 "Num. Keys Checked",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 BAD_IXREF_ROW,
				 "Invalid Index Keys",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 MISSING_IXREF_ROW,
				 "Missing Index Keys",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_DOM_NODES_ROW,
				 "Total DOM Nodes",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 DOM_LINKS_VERIFIED_ROW,
				 "DOM Links Verified",
				 NULL,
				 0,
				 FALSE);
				 
	OutLabel( LABEL_COLUMN,
				 TOTAL_BROKEN_LINKS_ROW,
				 "DOM Links Broken",
				 NULL,
				 0,
				 FALSE);
		
	if( gv_bLoggingEnabled)
	{
		LogString( NULL);
		LogString( "CHECK DETAILED RESULTS:");
		LogString( NULL);
	}

	uiCheckFlags = 0;
	if( gv_bRepairCorruptions == TRUE)
	{
		uiCheckFlags |= XFLM_ONLINE;
	}

	if( gv_bDoLogicalCheck == TRUE)
	{
		uiCheckFlags |= XFLM_DO_LOGICAL_CHECK;
	}

	if (gv_bSkipDomLinkVerify)
	{
		uiCheckFlags |= XFLM_SKIP_DOM_LINK_CHECK;
	}

	if (RC_OK( rc))
	{
		F_LocalCheckStatus dbCheckStatus;

		rc = gv_pDbSystem->dbCheck( gv_szDbFileName, gv_szDataDir, gv_szRflDir, NULL,
										uiCheckFlags, &gv_pDbInfo, &dbCheckStatus);
	}

	if( rc == NE_XFLM_FAILURE)
	{
		f_sprintf( szTmpBuf, "User pressed ESCAPE, check halted");
		gv_bShutdown = TRUE;
	}
	else
	{
		f_sprintf( szTmpBuf, "RETURN CODE: 0x%04X", (unsigned)rc);
	}
	
	uiStatus = CheckShowError( szTmpBuf, TRUE);

	if( ((uiStatus != FKB_ESCAPE) || (gv_bLoggingEnabled)) &&
		 (gv_bShowStats) &&
		 ((rc == NE_XFLM_OK) ||
		  (rc == NE_XFLM_DATA_ERROR) ||
		  (rc == NE_XFLM_TRANS_ACTIVE)))
	{
		PrintInfo( gv_pDbInfo);
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

	if( gv_pDb)
	{
		gv_pDb->Release();
	}

	if (pDbCheckStatus)
	{
		pDbCheckStatus->Release();
	}

	return( bOk);
}

/********************************************************************
Desc: Function to coordinate check of the database.
*********************************************************************/
FSTATIC FLMBOOL DoCheck( void)
{
	RCODE       		rc = NE_XFLM_OK;
	FLMBOOL				bOk = TRUE;
	char					szTmpBuf[ 100];
	IF_FileSystem *	pFileSystem = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

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

	if( gv_szLogFileName[ 0])
	{
		pFileSystem->deleteFile( gv_szLogFileName);
		if( RC_OK( rc = pFileSystem->createFile( gv_szLogFileName,
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &gv_pLogFile)))
		{
			gv_bLoggingEnabled = TRUE;
		}
		else
		{
			f_sprintf( szTmpBuf, "Error creating log file: 0x%04X", (unsigned)rc);
			CheckShowError( szTmpBuf, FALSE);
			bOk = FALSE;
			goto Exit;
		}
	}

	f_conSetCursorType( FLM_CURSOR_INVISIBLE);
	for( ;;)
	{
		// Check the database...
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

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	return( bOk);
}

/********************************************************************
Desc: Show the help screen.
*********************************************************************/
FSTATIC void CheckShowHelp(
	FLMBOOL bShowFullUsage
	)
{
	f_conStrOut( "\n");
	if (bShowFullUsage)
	{
		f_conStrOut( "Usage: checkdb <FileName> [Options]\n");
	}
	else
	{
		f_conStrOut( "Parameters: <FileName> [Options]\n\n");
	}

	f_conStrOut(
"   FileName = Name of database to check.\n");
	f_conStrOut(
"   Options\n");
	f_conStrOut(
"        -b           = Run in Batch Mode.\n");
	f_conStrOut(
"        -c           = Repair logical corruptions.\n");
	f_conStrOut(
"        -d           = Display/log detailed statistics.\n");
	f_conStrOut(
"        -dr<Dir>     = RFL directory.\n");
	f_conStrOut(
"        -dd<Dir>     = Data directory.\n");
	f_conStrOut(
"        -i           = Perform a logical (index) check.\n");
	f_conStrOut(
"        -l<FileName> = Log detailed information to <FileName>.\n");
	f_conStrOut(
"        -m           = Multiple passes (continuous check).\n");
	f_conStrOut(
"        -o<FileName> = Output binary log information to <FileName>.\n");
	f_conStrOut(
"        -p           = Pause before exiting.\n");
	f_conStrOut(
"        -s           = Skip DOM link verification.\n");
	f_conStrOut(
"        -t<Path>     = Temporary directory.\n");
	f_conStrOut(
"        -u           = Run check in an update transaction.\n");
	f_conStrOut(
"        -v<FileName> = Verify binary log information in <FileName>.  NOTE:\n");
	f_conStrOut(
"                       The -v and -o options cannot both be specified.\n");
	f_conStrOut(
"        -a<Password> = Database password.\n");
	f_conStrOut(
"        -?           = A '?' anywhere in the command line will cause this\n");
	f_conStrOut(
"                       screen to be displayed.\n");
	f_conStrOut(
"Options may be specified anywhere in the command line.\n");
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBOOL GetParams(
	FLMINT		iArgC,
	char **		ppszArgV
	)
{
#define MAX_ARGS     30
	FLMUINT		uiLoop;
	char			szTmpBuf[ 100];
	char *		pszTmp;
	char *		ppszArgs[ MAX_ARGS];
	char			szCommandBuffer[ 300];

	gv_szDbFileName [0] = '\0';
	gv_szDataDir [0] = '\0';
	gv_szRflDir [0] = '\0';
	gv_szLogFileName[ 0] = '\0';
	gv_szTmpDir[ 0] = '\0';
	gv_bShowStats = FALSE;
	gv_szPassword[0] = '\0';

	/*
	Ask the user to enter parameters if none were entered on the command
	line.
	*/

	if( iArgC < 2)
	{
		for( ;;)
		{
			f_conStrOut( "CheckDB Params (enter ? for help): ");
			szCommandBuffer[ 0] = '\0';
			f_conLineEdit( szCommandBuffer, sizeof( szCommandBuffer) - 1);
			if( gv_bShutdown)
			{
				return( FALSE);
			}
			if( f_stricmp( szCommandBuffer, "?") == 0)
			{
				CheckShowHelp( FALSE);
			}
			else
			{
				break;
			}
		}
		flmUtilParseParams( szCommandBuffer, MAX_ARGS, &iArgC, &ppszArgs [1]);
		ppszArgs [0] = ppszArgV [0];
		iArgC++;
		ppszArgV = &ppszArgs [0];
	}

	uiLoop = 1;
	while( uiLoop < (FLMUINT)iArgC)
	{
		pszTmp = ppszArgV[ uiLoop];

		/* See if they specified an option */

#ifdef FLM_UNIX
		if( *pszTmp == '-')
#else
		if( (*pszTmp == '-') || (*pszTmp == '/'))
#endif
		{
			pszTmp++;
			if( (*pszTmp == 'l') || (*pszTmp == 'L'))
			{
				pszTmp++;
				if( *pszTmp)
				{
					f_strcpy( gv_szLogFileName, pszTmp);
				}
				else
				{
					if( CheckShowError( 
						"Log file name not specified in parameter",
						FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if( (*pszTmp == 't') || (*pszTmp == 'T'))
			{
				pszTmp++;
				if( *pszTmp)
				{
					f_strcpy( gv_szTmpDir, pszTmp);
				}
				else
				{
					if( CheckShowError(
						"Temporary directory not specified in parameter",
						FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if( (*pszTmp == 'd') || (*pszTmp == 'D'))
			{
				pszTmp++;
				if (!(*pszTmp))
				{
					gv_bShowStats = TRUE;
				}
				else if (*pszTmp == 'r' || *pszTmp == 'R')
				{
					f_strcpy( gv_szRflDir, pszTmp + 1);
				}
				else if (*pszTmp == 'd' || *pszTmp == 'D')
				{
					f_strcpy( gv_szDataDir, pszTmp + 1);
				}
				else
				{
					f_sprintf( szTmpBuf, "Invalid option %s", pszTmp - 1);
					if( CheckShowError( szTmpBuf, FALSE) == FKB_ESCAPE)
					{
						return( FALSE);
					}
				}
			}
			else if (*pszTmp == 'a' || *pszTmp == 'A')
			{
				f_strcpy( gv_szPassword, pszTmp + 1);
			}
			else if (f_stricmp( pszTmp, "B") == 0)
			{
				gv_bBatchMode = TRUE;
			}
			else if (f_stricmp( pszTmp, "C") == 0)
			{
				gv_bRepairCorruptions = TRUE;
			}
			else if (f_stricmp( pszTmp, "I") == 0)
			{
				gv_bDoLogicalCheck = TRUE;
			}
			else if (f_stricmp( pszTmp, "M") == 0)
			{
				gv_bMultiplePasses = TRUE;
			}
			else if (f_stricmp( pszTmp, "P") == 0)
			{
				gv_bPauseBeforeExiting = TRUE;
			}
			else if (f_stricmp( pszTmp, "S") == 0)
			{
				gv_bSkipDomLinkVerify = TRUE;
			}
			else if (f_stricmp( pszTmp, "U") == 0)
			{
				gv_bStartUpdate = TRUE;
			}
			else if (f_stricmp( pszTmp, "?") == 0 ||
						f_stricmp( pszTmp, "HELP") == 0)
			{
				CheckShowHelp( TRUE);
				gv_bPauseBeforeExiting = TRUE;
				return( FALSE);
			}
			else
			{
				f_sprintf( szTmpBuf, "Invalid option %s", pszTmp);
				if( CheckShowError( szTmpBuf, FALSE) == FKB_ESCAPE)
				{
					return( FALSE);
				}
			}
		}
		else if( f_stricmp( pszTmp, "?") == 0)
		{
Show_Help:
			CheckShowHelp( TRUE);
			gv_bPauseBeforeExiting = TRUE;
			return( FALSE);
		}
		else if( !gv_szDbFileName[ 0])
		{
			f_strcpy( gv_szDbFileName, pszTmp);
		}
		uiLoop++;
	}

	if( !gv_szDbFileName[ 0])
	{
		goto Show_Help;
	}
	else
	{
		return( TRUE);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogFlush(
	void
	)
{
	FLMUINT	uiBytesWritten;

	if( gv_uiLogBufferCount)
	{
		gv_pLogFile->write( FLM_IO_CURRENT_POS,
							 gv_uiLogBufferCount, (FLMBYTE *)gv_pszLogBuffer,
							 &uiBytesWritten);
		gv_uiLogBufferCount = 0;
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogString(
	const char *	pszString)
{
	FLMUINT		uiLen;
	FLMUINT		uiLoop;

	if( (gv_bLoggingEnabled) && (gv_pszLogBuffer != NULL))
	{
		uiLen = (FLMUINT)((pszString != NULL)
							  ? (FLMUINT)(f_strlen( pszString))
							  : 0);
		for( uiLoop = 0; uiLoop < uiLen; uiLoop++)
		{
			gv_pszLogBuffer[ gv_uiLogBufferCount++] = *pszString++;
			if( gv_uiLogBufferCount == MAX_LOG_BUFF)
			{
				LogFlush();
			}
		}
		gv_pszLogBuffer[ gv_uiLogBufferCount++] = '\r';
		if( gv_uiLogBufferCount == MAX_LOG_BUFF)
		{
			LogFlush();
		}
		gv_pszLogBuffer[ gv_uiLogBufferCount++] = '\n';
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
	const char *	pszBuf)
{
	FLMUINT		uiChar;

	if( gv_bLoggingEnabled)
	{
		LogString( pszBuf);
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
				f_conStrOut(
					"Press: ESC to quit, anything else to continue");
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
			f_conStrOutXY( pszBuf, 0, gv_uiLineCount);
			gv_uiLineCount++;
		}
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutValue(
	const char *	pszLabel,
	const char *	pszValue)
{
	char	szTmpBuf[ 100];

	f_strcpy( szTmpBuf, "...................................... ");
	f_strcpy( &szTmpBuf[ f_strlen( szTmpBuf)], pszValue);
	f_memcpy( szTmpBuf, pszLabel, (FLMSIZET)f_strlen( pszLabel));
	OutLine( szTmpBuf);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutUINT(
	const char *	pszLabel,
	FLMUINT			uiNum)
{
	char	szValue [12];

	if( uiNum == 0xFFFFFFFF)
	{
		f_strcpy( szValue, "0xFFFFFFFF");
	}
	else
	{
		f_sprintf( szValue, "%u", (unsigned)uiNum);
	}

	OutValue( pszLabel, szValue);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutUINT64(
	const char *	pszLabel,
	FLMUINT64		ui64Num)
{
	char	szValue [24];

	if( ui64Num == (FLMUINT64)-1)
	{
		f_strcpy( szValue, "0xFFFFFFFFFFFFFFFF");
	}
	else
	{
		f_sprintf( szValue, "%I64u", ui64Num);
	}

	OutValue( pszLabel, szValue);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutBlkHeader( void)
{
	OutLine(
"  Blk Type    Blk Count  Total Bytes  Bytes Used  Prcnt  Element Cnt  Avg Elem");
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutOneBlockStat(
	const char *	pszLabel,
	FLMUINT			uiBlockSize,
	FLMUINT64		ui64KeyCount,
	FLMUINT64		ui64BytesUsed,
	FLMUINT64		ui64ElementCount,
	FLMUINT64		ui64ContElementCount,
	FLMUINT64		ui64ContElmBytes,
	FLMUINT			uiBlockCount,
	FLMINT32			i32LastError,
	FLMUINT			uiNumErrors)
{
	char			szTmpBuf[ 100];
	FLMUINT64	ui64TotalBytes;
	FLMUINT		uiPercent;
	FLMUINT		uiAvgElementSize;

	ui64TotalBytes = (FLMUINT64)uiBlockCount * (FLMUINT64)uiBlockSize;
	if( ui64ElementCount)
	{
		uiAvgElementSize = (FLMUINT)( ui64BytesUsed / ui64ElementCount);
	}
	else
	{
		uiAvgElementSize = 0;
	}

	if( ui64BytesUsed > 40000000)
	{
		uiPercent = (FLMUINT)( ui64BytesUsed / (ui64TotalBytes / 100));
	}
	else if( ui64TotalBytes)
	{
		uiPercent = (FLMUINT)((ui64BytesUsed * 100) / ui64TotalBytes);
	}
	else
	{
		uiPercent = 0;
	}

	f_sprintf( szTmpBuf, "%-12s %10u  %11u  %10u  %5u  %11u  %8u",
		pszLabel, (unsigned)uiBlockCount, (unsigned)ui64TotalBytes,
		(unsigned)ui64BytesUsed,
		(unsigned)uiPercent, (unsigned)ui64ElementCount,
		(unsigned)uiAvgElementSize);
	OutLine( szTmpBuf);

	if( ui64ContElementCount)
	{
		uiAvgElementSize = (FLMUINT)( ui64ContElmBytes / ui64ContElementCount);
		if( ui64ContElmBytes > 40000000)
		{
			uiPercent =
				(FLMUINT)( ui64ContElmBytes / (ui64TotalBytes / 100));
		}
		else if( ui64TotalBytes)
		{
			uiPercent = (FLMUINT)((ui64ContElmBytes * 100) / ui64TotalBytes);
		}
		else
		{
			uiPercent = 0;
		}

		f_sprintf( szTmpBuf, "%-12s                         "
			"%10u  %5u  %11u  %8u", "    ContElm",
			(unsigned)ui64ContElmBytes,
			(unsigned)uiPercent, (unsigned)ui64ContElementCount,
			(unsigned)uiAvgElementSize);
		OutLine( szTmpBuf);
	}

	if( ui64KeyCount)
	{
		f_sprintf( szTmpBuf, "%-12s                         %10u",
			"		KeyCnt", (unsigned)ui64KeyCount);
		OutLine( szTmpBuf);
	}

	if( uiNumErrors)
	{
		f_strcpy( szTmpBuf, "    LAST ERROR: ");
		f_strcpy( &szTmpBuf[ f_strlen( szTmpBuf)],
			gv_pDbSystem->checkErrorToStr( (FLMINT)i32LastError));
		OutLine( szTmpBuf);
		f_sprintf( szTmpBuf,
			"    TOTAL ERRORS: %u", (unsigned)uiNumErrors);
		OutLine( szTmpBuf);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void OutLogicalFile(
	IF_DbInfo *	pDbInfo,
	FLMUINT		uiIndex
	)
{
	char			szTmpBuf[ 100];
	FLMUINT		uiLoop;
	char			szLfName[ 30];
	FLMUINT		uiLfNum;
	eLFileType	eLfType;
	FLMUINT		uiRootBlkAddress;
	FLMUINT		uiNumLevels;
	FLMUINT64	ui64KeyCount;
	FLMUINT64	ui64BytesUsed;
	FLMUINT64	ui64ElementCount;
	FLMUINT64	ui64ContElementCount;
	FLMUINT64	ui64ContElmBytes;
	FLMUINT		uiBlockCount;
	FLMINT32		i32LastError;
	FLMUINT		uiNumErrors;

	pDbInfo->getBTreeInfo( uiIndex, &uiLfNum, &eLfType,
						&uiRootBlkAddress,
						&uiNumLevels);

	switch( eLfType)
	{
		case XFLM_LF_COLLECTION: /* Data collection */
			f_strcpy( szTmpBuf, "COLLECTION");
			break;
		case XFLM_LF_INDEX: /* Index */
			f_strcpy( szTmpBuf, "INDEX");
			break;
		default:
			break;
	}
	(void)NumToName( uiLfNum,
							  eLfType == XFLM_LF_COLLECTION 
									? ELM_COLLECTION_TAG
									: ELM_INDEX_TAG,
							  szLfName);
	OutValue( szTmpBuf, szLfName);

	OutUINT( "  Logical File Number", uiLfNum);
	OutUINT( "  Root Block Address", uiRootBlkAddress);
	
	if (!uiNumLevels)
	{
		OutUINT( "  Levels", uiNumLevels);
	}
	else
	{
		OutBlkHeader();
		for( uiLoop = 0; uiLoop < uiNumLevels; uiLoop++)
		{
			f_sprintf( szTmpBuf, "  Level %u", (unsigned)uiLoop);
			pDbInfo->getBTreeBlockStats( uiIndex, uiLoop,
							&ui64KeyCount, &ui64BytesUsed,
							&ui64ElementCount, &ui64ContElementCount,
							&ui64ContElmBytes, &uiBlockCount,
							&i32LastError, &uiNumErrors);
			OutOneBlockStat( szTmpBuf,
				pDbInfo->getDbHdr()->ui16BlockSize, ui64KeyCount,
							ui64BytesUsed, ui64ElementCount, ui64ContElementCount,
							ui64ContElmBytes, uiBlockCount, i32LastError, uiNumErrors);
		}
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void PrintInfo(
	IF_DbInfo *	pDbInfo
	)
{
	FLMUINT					uiLoop;
	FLMUINT					uiNumLogicalFiles;
	FLMUINT64				ui64BytesUsed;
	FLMUINT64				ui64ElementCount;
	FLMUINT64				ui64ContElementCount;
	FLMUINT64				ui64ContElmBytes;
	FLMUINT					uiBlockCount;
	FLMINT32					i32LastError;
	FLMUINT					uiNumErrors;
	const XFLM_DB_HDR *	pDbHdr = pDbInfo->getDbHdr();

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conClearScreen( 0, 0);

	OutUINT( "Default Language",
		(FLMUINT)pDbHdr->ui8DefaultLanguage);
	OutUINT64( "File Size", pDbInfo->getFileSize());
	OutUINT( "Index Count", pDbInfo->getNumIndexes());
	OutUINT( "Collection Count",
		pDbInfo->getNumCollections());
	OutUINT( "Block Size",
		(FLMUINT)pDbHdr->ui16BlockSize);
	OutLine( "LOG HEADER");
	OutUINT( "  First LFH Block Address",
		(FLMUINT)pDbHdr->ui32FirstLFBlkAddr);

	OutLine( "MISCELLANEOUS BLOCK STATISTICS");
	OutBlkHeader();

	ui64ElementCount = 0;
	ui64ContElementCount = 0;
	ui64ContElmBytes = 0;
	pDbInfo->getAvailBlockStats( &ui64BytesUsed, &uiBlockCount,
						&i32LastError, &uiNumErrors);
	if( uiBlockCount)
	{
		OutOneBlockStat( "  Avail",
			(FLMUINT)pDbHdr->ui16BlockSize,
			0, ui64BytesUsed, ui64ElementCount, ui64ContElementCount,
			ui64ContElmBytes, uiBlockCount, i32LastError, uiNumErrors);
	}

	ui64ElementCount = 0;
	ui64ContElementCount = 0;
	ui64ContElmBytes = 0;
	pDbInfo->getLFHBlockStats( &ui64BytesUsed, &uiBlockCount,
						&i32LastError, &uiNumErrors);
	if( uiBlockCount)
	{
		OutOneBlockStat( "  LFH",
			(FLMUINT)pDbHdr->ui16BlockSize,
			0, ui64BytesUsed, ui64ElementCount, ui64ContElementCount,
			ui64ContElmBytes, uiBlockCount, i32LastError, uiNumErrors);
	}

	uiNumLogicalFiles = pDbInfo->getNumLogicalFiles();
	for( uiLoop = 0; uiLoop < uiNumLogicalFiles; uiLoop++)
	{
		OutLogicalFile( pDbInfo, uiLoop);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMUINT CheckShowError(
	const char *	pszMessage,
	FLMBOOL			bLogIt)
{
	FLMUINT		uiResKey;

	f_sprintf( gv_szLastError, "%s", pszMessage);
	
	if( bLogIt)
	{
		LogString( pszMessage);
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
		f_conStrOut( pszMessage);
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
	const char *	pszLabel,
	const char *	pszValue,
	FLMUINT64		ui64NumValue,
	FLMBOOL			bLogIt)
{
	char			szTmpBuf[ 100];
	FLMUINT		uiLoop;

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conStrOutXY( pszLabel, uiCol, uiRow);

	for( uiLoop = f_conGetCursorColumn(); uiLoop < VALUE_COLUMN - 1; uiLoop++)
	{
		f_conStrOut( ".");
	}
	

	if( pszValue != NULL)
	{
		DisplayValue( uiRow, pszValue);
	}
	else
	{
		DisplayNumValue( uiRow, ui64NumValue);
	}

	if( (bLogIt) && (gv_bLoggingEnabled))
	{
		f_strcpy( szTmpBuf, pszLabel);
		f_strcpy( &szTmpBuf[ f_strlen( szTmpBuf)], ": ");
		if( pszValue != NULL)
		{
			f_strcpy( &szTmpBuf[ f_strlen( szTmpBuf)], pszValue);
		}
		else
		{
			f_sprintf( &szTmpBuf[ f_strlen( szTmpBuf)],
				"%I64u", ui64NumValue);
		}
		LogString( szTmpBuf);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void DisplayValue(
	FLMUINT			uiRow,
	const char *	pszValue)
{
	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	f_conStrOutXY( pszValue, VALUE_COLUMN, uiRow);
	f_conClearLine( 255, 255);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void DisplayNumValue(
	FLMUINT		uiRow,
	FLMUINT64	ui64Number)
{
	char	szTmpBuf[ 128];

	f_sprintf( szTmpBuf, "%,23I64u   0x%016I64X", 
		ui64Number, ui64Number);
	DisplayValue( uiRow, szTmpBuf);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC RCODE GetUserInput(
	void
	)
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
			return( RC_SET( NE_XFLM_FAILURE));
		default:
			break;
	}

	return( NE_XFLM_OK);
}

/********************************************************************
Desc:
*********************************************************************/
RCODE F_LocalCheckStatus::reportProgress(
	XFLM_PROGRESS_CHECK_INFO *	pProgCheck)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_CACHE_INFO	cacheInfo;
	char					szWhat[ 256];
	char					szLfName[ 128];
	FLMUINT				uiCurrentTime;

	uiCurrentTime = FLM_TIMER_UNITS_TO_MILLI( FLM_GET_TIMER());

	if( (uiCurrentTime - m_uiLastRefresh < SCREEN_REFRESH_RATE) &&
		!pProgCheck->bStartFlag)
	{
		goto Exit;
	}

	// We have exceeded our refresh interval or we have changed check phases,
	// therefore, we should refresh the screen.

	m_uiLastRefresh = uiCurrentTime;

	gv_pDbSystem->getCacheInfo( &cacheInfo);
	DisplayNumValue( CACHE_USED_ROW, cacheInfo.BlockCache.uiByteCount);

	if( gv_bShutdown)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Update the display first

	gv_ui64BytesDone = pProgCheck->ui64BytesExamined;
	DisplayNumValue( TOTAL_KEYS_ROW, pProgCheck->ui64NumKeys);
	DisplayNumValue( TOTAL_DUPS_ROW, pProgCheck->ui64NumDuplicateKeys);
	DisplayNumValue( TOTAL_KEYS_EXAM_ROW, pProgCheck->ui64NumKeysExamined);
	DisplayNumValue( CONFLICT_ROW, pProgCheck->ui64NumConflicts);
	DisplayNumValue( BAD_IXREF_ROW, pProgCheck->ui64NumKeysNotFound);
	DisplayNumValue( MISSING_IXREF_ROW, pProgCheck->ui64NumDocKeysNotFound);

	DisplayNumValue( TOTAL_DOM_NODES_ROW, pProgCheck->ui64NumDomNodes);
	DisplayNumValue( DOM_LINKS_VERIFIED_ROW, pProgCheck->ui64NumDomLinksVerified);
	DisplayNumValue( TOTAL_BROKEN_LINKS_ROW, pProgCheck->ui64NumBrokenDomLinks);

	DisplayNumValue( REPAIR_ROW, pProgCheck->ui32NumProblemsFixed);
	gv_uiRepairCount = (FLMUINT)pProgCheck->ui32NumProblemsFixed;

	if( pProgCheck->i32CheckPhase != XFLM_CHECK_RS_SORT)
	{
		OutLabel( LABEL_COLUMN, AMOUNT_DONE_ROW, "Bytes Checked",
				NULL, gv_ui64BytesDone, FALSE);
	}

	if( pProgCheck->bStartFlag)
	{
		gv_ui64FileSize = pProgCheck->ui64FileSize;
		DisplayNumValue( FILE_SIZE_ROW, gv_ui64FileSize);

		switch( pProgCheck->i32CheckPhase)
		{
			case XFLM_CHECK_LFH_BLOCKS:
				f_strcpy( szWhat, "LFH BLOCKS");
				break;
			case XFLM_CHECK_B_TREE:
				*szLfName = '\0';
				if( pProgCheck->ui32LfType == XFLM_LF_INDEX)
				{
					f_strcpy( szWhat, "INDEX: ");
					(void)NumToName( pProgCheck->ui32LfNumber,
								ELM_INDEX_TAG, szLfName);
				}
				else if( pProgCheck->ui32LfType == XFLM_LF_COLLECTION)
				{
					f_strcpy( szWhat, "COLLECTION: ");
					(void)NumToName( pProgCheck->ui32LfNumber,
								ELM_COLLECTION_TAG, szLfName);
				}
				else
				{
					f_strcpy( szWhat, "DICTIONARY: ");
					(void)NumToName( pProgCheck->ui32LfNumber,
										ELM_INDEX_TAG, szLfName);
				}

				f_strcpy( &szWhat[ f_strlen( szWhat)], szLfName);

				f_sprintf( &szWhat[ f_strlen( szWhat)], " (%u)",
					(unsigned)pProgCheck->ui32LfNumber);
				szWhat[ 50] = '\0';
				break;
			case XFLM_CHECK_AVAIL_BLOCKS:
				f_strcpy( szWhat, "AVAIL BLOCKS");
				break;
			case XFLM_CHECK_RS_SORT:
				f_strcpy( szWhat, "SORTING INDEX KEYS");
				break;
			case XFLM_CHECK_DOM_LINKS:
				f_strcpy( szWhat, "COLLECTION: ");
				(void)NumToName( pProgCheck->ui32LfNumber,
									ELM_COLLECTION_TAG, szLfName);
				f_strcpy( &szWhat[ f_strlen( szWhat)], szLfName);
				f_sprintf( &szWhat[ f_strlen( szWhat)], " (%u)",
					(unsigned)pProgCheck->ui32LfNumber);
				szWhat[ 50] = '\0';
				break;
			default:
				break;
		}

		szWhat[ 45] = '\0';
		f_conSetBackFore( FLM_BLUE, FLM_WHITE);
		DisplayValue( DOING_ROW, szWhat);
	}
	else if( f_conHaveKey() && (f_conGetKey() == FKB_ESCAPE))
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

Exit:

	return rc;
}

RCODE F_LocalCheckStatus::reportCheckErr(
	XFLM_CORRUPT_INFO *	pCorruptInfo,
	FLMBOOL *				pbFix)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_CACHE_INFO	cacheInfo;

	gv_pDbSystem->getCacheInfo( &cacheInfo);
	DisplayNumValue( CACHE_USED_ROW, cacheInfo.BlockCache.uiByteCount);
	
	if( (gv_bLoggingEnabled) &&
		 ((gv_bShowStats) ||
		 (pCorruptInfo->i32ErrCode != FLM_OLD_VIEW)))
	{
		LogCorruptError( pCorruptInfo);
	}

	f_conSetBackFore( FLM_BLUE, FLM_WHITE);
	if( pCorruptInfo->i32ErrCode == FLM_OLD_VIEW)
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
	if (pbFix)
	{
		*pbFix = gv_bRepairCorruptions;
	}

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogStr(
	FLMUINT			uiIndent,
	const char *	pszStr)
{
	FLMUINT		uiLoop;

	if( gv_bLoggingEnabled)
	{
		for( uiLoop = 0; uiLoop < uiIndent; uiLoop++)
		{
			gv_pszLogBuffer[ gv_uiLogBufferCount++] = ' ';
			if( gv_uiLogBufferCount == MAX_LOG_BUFF)
			{
				LogFlush();
			}
		}
		LogString( pszStr);
	}
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC void LogCorruptError(
	XFLM_CORRUPT_INFO *	pCorrupt)
{
	char			szWhat[ 20];
	char			szTmpBuf[ 100];

	switch( pCorrupt->ui32ErrLocale)
	{
		case XFLM_LOCALE_LFH_LIST:
		{
			LogStr( 0, "ERROR IN LFH LINKED LIST:");
			break;
		}
		
		case XFLM_LOCALE_AVAIL_LIST:
		{
			LogStr( 0, "ERROR IN AVAIL LINKED LIST:");
			break;
		}
		
		case XFLM_LOCALE_B_TREE:
		{
			if( pCorrupt->i32ErrCode == FLM_OLD_VIEW)
			{
				LogStr( 0, "OLD VIEW");
			}
			else
			{
				if( pCorrupt->ui64ErrNodeId)
				{
					f_strcpy( szWhat, "NODE");
				}
				else if( pCorrupt->ui32ErrElmOffset)
				{
					f_strcpy( szWhat, "ELEMENT");
				}
				else if( pCorrupt->ui32ErrBlkAddress)
				{
					f_strcpy( szWhat, "BLOCK");
				}
				else
				{
					f_strcpy( szWhat, "LAST BLOCK");
				}
				f_sprintf( szTmpBuf, "BAD %s", szWhat);
				LogStr( 0, szTmpBuf);
			}

			// Log the logical file number, name, and type

			f_sprintf( szTmpBuf, "Logical File Number: %u",
				(unsigned)pCorrupt->ui32ErrLfNumber);
			LogStr( 2, szTmpBuf);
			
			switch( pCorrupt->ui32ErrLfType)
			{
				case XFLM_LF_COLLECTION:
				{
					f_strcpy( szWhat, "Collection");
					break;
				}
				
				case XFLM_LF_INDEX:
				{
					f_strcpy( szWhat, "Index");
					break;
				}
				
				default:
				{
					f_sprintf( szWhat, "?%u", 
							(unsigned)pCorrupt->ui32ErrLfType);
					break;
				}
			}
			
			f_sprintf( szTmpBuf, "Logical File Type: %s", szWhat);
			LogStr( 2, szTmpBuf);

			// Log the level in the B-Tree, if known

			if( pCorrupt->ui32ErrBTreeLevel != 0xFF)
			{
				f_sprintf( szTmpBuf, "Level in B-Tree: %u",
					(unsigned)pCorrupt->ui32ErrBTreeLevel);
				LogStr( 2, szTmpBuf);
			}
			
			break;
		}
		
		case XFLM_LOCALE_INDEX:
		{
			f_strcpy( szWhat, "Index");
			LogKeyError( pCorrupt);
			break;
		}
		
		default:
		{
			pCorrupt->ui32ErrLocale = 0;
			break;
		}
	}

	// Log the block address, if known

	if( pCorrupt->ui32ErrBlkAddress)
	{
		f_sprintf( szTmpBuf, "Block Address: 0x%08X (%u)",
			(unsigned)pCorrupt->ui32ErrBlkAddress,
			(unsigned)pCorrupt->ui32ErrBlkAddress);
		LogStr( 2, szTmpBuf);
	}

	// Log the parent block address, if known

	if( pCorrupt->ui32ErrParentBlkAddress)
	{
		if( pCorrupt->ui32ErrParentBlkAddress != FLM_MAX_UINT32)
		{
			f_sprintf( szTmpBuf, "Parent Block Address: 0x%08X (%u)",
				(unsigned)pCorrupt->ui32ErrParentBlkAddress,
				(unsigned)pCorrupt->ui32ErrParentBlkAddress);
		}
		else
		{
			f_sprintf( szTmpBuf,
				"Parent Block Address: NONE, Root Block");
		}
		LogStr( 2, szTmpBuf);
	}

	// Log the element offset, if known

	if( pCorrupt->ui32ErrElmOffset != FLM_MAX_UINT32)
	{
		f_sprintf( szTmpBuf, "Element Offset: %u", 
				(unsigned)pCorrupt->ui32ErrElmOffset);
		LogStr( 2, szTmpBuf);
	}

	// Log the NodeId, if known

	if( pCorrupt->ui64ErrNodeId)
	{
		f_sprintf( szTmpBuf, 
				"NodeId: %u", (unsigned)pCorrupt->ui64ErrNodeId);
		LogStr( 2, szTmpBuf);
	}

	f_strcpy( szTmpBuf, gv_pDbSystem->checkErrorToStr( (FLMINT)pCorrupt->i32ErrCode));
	f_sprintf( &szTmpBuf[ f_strlen( szTmpBuf)], " (%d)",
		(int)pCorrupt->i32ErrCode);
	LogStr( 2, szTmpBuf);
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
	XFLM_CORRUPT_INFO *	pCorrupt)
{
	FLMUINT				uiLogItem;
	IF_DataVector *	ifpKey = NULL;
	FLMUINT				uiIndent;
	FLMUINT				uiLevelOffset;
	char					szNameBuf[ 200];
	char					szTmpBuf[ 200];
	FLMUINT				uiElementNumber;
	
	(void)NumToName( (FLMUINT)pCorrupt->ui32ErrLfNumber, ELM_INDEX_TAG, szNameBuf);
	LogString( NULL);
	LogString( NULL);
	
	f_sprintf( szTmpBuf, "ERROR IN INDEX: %s", szNameBuf);
	LogString( szTmpBuf);
	
	uiLogItem = 'K';
	uiLevelOffset = 0;
	for( ;;)
	{
		uiIndent = 2;
		if( uiLogItem == 'K')
		{
			if( (ifpKey = pCorrupt->ifpErrIxKey) == NULL)
			{
				uiLogItem = 'L';
				continue;
			}
			LogString( NULL);
			LogString( " PROBLEM KEY");
		}

		uiElementNumber = 0;
		while( ifpKey->getNameId( uiElementNumber))
		{
			DisplayField( ifpKey, uiElementNumber, uiIndent, uiLevelOffset);
			uiElementNumber++;
		}

		if( uiLogItem == 'L')
		{
			break;
		}
		else
		{
			uiLogItem = 'L';
		}
	}
}

/***************************************************************************
Name:    DisplayField
Desc:    This routine displays a field to the screen.
*****************************************************************************/
FSTATIC FLMBOOL DisplayField(
	IF_DataVector *	ifpKey,
	FLMUINT				uiElementNumber,
	FLMUINT				uiStartCol,
	FLMUINT				uiLevelOffset
	)
{
	char			szTmpBuf[ 220];
	FLMUINT		uiLoop;
	FLMUINT		uiLen;
	FLMUINT		uiBinLen;
	FLMUINT		uiTmpLen;
	char *		pszTmp;
	FLMBYTE *	pucTmp;
	FLMBYTE		ucTmpBin [80];
	FLMUINT		uiNum;
	FLMUINT		uiIndent = (uiLevelOffset * 2) + uiStartCol;
	FLMUINT64	ui64NodeId;

	// Insert leading spaces to indent for level

	for( uiLoop = 0; uiLoop < uiIndent; uiLoop++)
	{
		szTmpBuf[ uiLoop] = ' ';
	}

	// Output level and tag
	if (ifpKey->isKeyComponent( uiElementNumber))
	{
		f_sprintf( &szTmpBuf[ uiIndent], "K) ");
	}
	else
	{
		f_sprintf( &szTmpBuf[ uiIndent], "D) ");
	}

	(void)NumToName( ifpKey->getNameId( uiElementNumber),
						  ifpKey->isAttr( uiElementNumber)
										? ELM_ATTRIBUTE_TAG
										: ELM_ELEMENT_TAG,
						  &szTmpBuf[ f_strlen( szTmpBuf)]);

	// Output what will fit of the value on the rest of the line

	uiLen = f_strlen( szTmpBuf);
	szTmpBuf[ uiLen++] = ' ';
	szTmpBuf[ uiLen] = 0;
	if (!ifpKey->getDataLength( uiElementNumber))
	{
		goto Exit;
	}
	switch( ifpKey->getDataType( uiElementNumber))
	{
		case XFLM_TEXT_TYPE:
			pszTmp = &szTmpBuf[ uiLen];
			uiLen = 80 - uiLen;
			ifpKey->getUTF8( uiElementNumber, (FLMBYTE *)pszTmp, &uiLen);
			break;
		case XFLM_NUMBER_TYPE:
			ifpKey->getUINT( uiElementNumber, &uiNum);
			f_sprintf( &szTmpBuf [uiLen], "%u", (unsigned)uiNum);
			break;
		case XFLM_BINARY_TYPE:
			ifpKey->getBinary( uiElementNumber, NULL, &uiBinLen);
			uiTmpLen = sizeof( ucTmpBin);
			ifpKey->getBinary( uiElementNumber, ucTmpBin, &uiTmpLen);
			pucTmp = &ucTmpBin [0];
			while (uiBinLen && uiLen < 77)
			{
				f_sprintf( &szTmpBuf [uiLen], "%02X ", (unsigned)*pucTmp);
				uiBinLen--;
				pucTmp++;
				uiLen += 3;
			}
			szTmpBuf [uiLen - 1] = 0;
			break;
		default:
			break;
	}

	// Get the Id to display
	if ((ui64NodeId = ifpKey->getID( uiElementNumber)) != 0)
	{
		uiLen = f_strlen( szTmpBuf);
		f_sprintf( &szTmpBuf[ uiLen], " %I64u", ui64NodeId);
	}

	// Output the line

Exit:
	LogString( szTmpBuf);
	return( TRUE);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBOOL NumToName(
	FLMUINT		uiNum,
	FLMUINT		uiType,
	char *		pszBuf
	)
{
	FLMUINT	uiLen = 128;

	if (gv_pNameTable &&
		 RC_OK( gv_pNameTable->getFromTagTypeAndNum(
								gv_pDb, uiType, uiNum,
								NULL, pszBuf, &uiLen)))
	{
		return( TRUE);
	}

	switch (uiType)
	{
		case ELM_INDEX_TAG:
		{
			if (uiNum == XFLM_DICT_NUMBER_INDEX)
			{
				f_strcpy( pszBuf, "Dictionary Number Index");
				return( TRUE);
			}
			else if (uiNum == XFLM_DICT_NAME_INDEX)
			{
				f_strcpy( pszBuf, "Dictionary Name Index");
				return( TRUE);
			}
		}
		case ELM_COLLECTION_TAG:
		{
			if (uiNum == XFLM_DATA_COLLECTION)
			{
				f_strcpy( pszBuf, "Data Collection");
				return( TRUE);
			}
			else if (uiNum == XFLM_DICT_COLLECTION)
			{
				f_strcpy( pszBuf, "Dictionary Collection");
				return( TRUE);
			}
		}
	}

	f_sprintf( pszBuf, "#%u", (unsigned)uiNum);
	return( FALSE);

}
