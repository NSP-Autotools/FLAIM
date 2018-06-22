//-------------------------------------------------------------------------
// Desc:	Database rebuild utility.
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

#define UTIL_ID					"REBUILD"

#define LABEL_COLUMN    		5
#define VALUE_COLUMN    		35

#define PARAM_ROW					1
#define SOURCE_ROW				(PARAM_ROW + 1)
#define SOURCE_DATA_DIR_ROW	(SOURCE_ROW + 1)
#define DEST_ROW					(SOURCE_DATA_DIR_ROW + 1)
#define DEST_DATA_DIR_ROW		(DEST_ROW + 1)
#define DEST_RFL_ROW				(DEST_DATA_DIR_ROW + 1)
#define DICT_ROW					(DEST_RFL_ROW + 1)
#define CACHE_ROW					(DICT_ROW + 1)
#define LOG_FILE_ROW				(CACHE_ROW + 1)
#define DOING_ROW					(LOG_FILE_ROW + 1)
#define DB_SIZE_ROW				(DOING_ROW + 1)
#define BYTES_DONE_ROW			(DB_SIZE_ROW + 1)
#define TOTAL_REC_ROW			(BYTES_DONE_ROW + 1)
#define RECOV_ROW					(TOTAL_REC_ROW + 1)
#define DICT_RECOV_ROW			(RECOV_ROW + 1)

#define MAX_LOG_BUFF       	2048

FSTATIC FLMBOOL bldDoRebuild( void);

FSTATIC void bldShowResults(
	const char *	pucFuncName,
	RCODE				rc,
	FLMUINT			uiTotalRecords,
	FLMUINT			uiRecordsRecovered,
	FLMUINT			uiDictRecordsRecovered);

FSTATIC void bldShowHelp( void);

FSTATIC FLMBOOL bldGetParams(
	FLMINT			iArgC,
	const char **	ppszArgV);

FSTATIC FLMBOOL bldParseHdrInfo(
	const char *	pszBuffer);

FSTATIC void bldOutLabel(
	FLMUINT			uiCol,
	FLMUINT			uiRow,
	const char *	pszLabel,
	const char *	pszValue,
	FLMUINT			uiNumValue,
	FLMBOOL			bLogIt);

FSTATIC void bldLogFlush( void);

FSTATIC void bldLogString(
	const char *	pszStr);

FSTATIC void bldOutValue(
	FLMUINT			uiRow,
	const char *	pszValue);

FSTATIC void bldOutNumValue(
	FLMUINT			uiRow,
	FLMUINT			uiNumber);

FSTATIC RCODE bldGetUserInput( void);

FSTATIC void bldLogStr(
	FLMUINT			uiIndent,
	const char *	pszStr);

FSTATIC void bldLogCorruptError(
	CORRUPT_INFO *	pCorruptInfo);

FSTATIC RCODE bldProgFunc(
	eStatusType		eStatus,
	void *			Parm1,
	void *			Parm2,
	void *			pvAppData);

FSTATIC void bldShowError(
	const char *	pszMessage);

FSTATIC RCODE bldGetCreateOpts(
	const char *		pszFileName,
	CREATE_OPTS *		pCreateOpts);

FLMBOOL						gv_bShutdown = FALSE;
static char *				gv_pucLogBuffer = NULL;
static FLMUINT				gv_uiLogBufferCount = 0;
static FLMBOOL				gv_bBatchMode;
static FLMUINT64			gv_ui64DatabaseSize;
static FLMINT				gv_iLastDoing;
static FLMUINT64			gv_ui64BytesDone;
static FLMUINT				gv_uiTotalRecs;
static FLMUINT				gv_uiRecsRecovered;
static FLMUINT				gv_uiDictRecsRecovered;
static char					gv_szSrcFileName[ F_PATH_MAX_SIZE];
static char					gv_szSrcDataDir [F_PATH_MAX_SIZE];
static char					gv_szDestFileName[ F_PATH_MAX_SIZE];
static char					gv_szDestDataDir [F_PATH_MAX_SIZE];
static char					gv_szDestRflDir [F_PATH_MAX_SIZE];
static char					gv_szDictFileName[ F_PATH_MAX_SIZE];
static char					gv_szLogFileName[ F_PATH_MAX_SIZE];
static FLMUINT				gv_uiCacheSize = 30000;
static IF_FileHdl *		gv_pLogFile = NULL;
static FLMBOOL				gv_bLoggingEnabled;
static char *				gv_pszDictPath;
static CREATE_OPTS		gv_DefaultCreateOpts;
static FLMBOOL				gv_bFixHdrInfo;
static FLMBOOL				gv_bRunning;
static FLMBOOL				gv_bPauseBeforeExiting = FALSE;
static IF_FileSystem *	gv_pFileSystem = NULL;

/********************************************************************
Desc: ?
*********************************************************************/
#ifdef FLM_RING_ZERO_NLM
extern "C" int nlm_main(
#else
int main(
#endif
	int			iArgC,
	char **		ppszArgV)
{
	int		iRetCode = 0;
	F_Pool	LogPool;

	gv_bBatchMode = FALSE;
	gv_bRunning = TRUE;

	if( RC_BAD( FlmStartup()))
	{
		iRetCode = -1;
		goto Exit;
	}

	f_conInit( 0xFFFF, 0xFFFF, "FLAIM Database Rebuild");

	if( RC_BAD( FlmGetFileSystem( &gv_pFileSystem)))
	{
		f_conStrOut( "\nCould not allocate a file system object.\n");
		goto Exit;
	}

	LogPool.poolInit( 1024);
	
	if( RC_BAD( LogPool.poolAlloc( MAX_LOG_BUFF, (void **)&gv_pucLogBuffer)))
	{
		goto Exit;
	}
	
	if( bldGetParams( iArgC, (const char **)ppszArgV))
	{
		if (!bldDoRebuild())
		{
			iRetCode = 1;
		}
	}
	
Exit:

	if (gv_bPauseBeforeExiting && !gv_bShutdown)
	{
		f_conStrOut( "\nPress any character to exit REBUILD: ");
		for (;;)
		{
			if (gv_bShutdown)
			{
				break;
			}
			
			if (f_conHaveKey())
			{
				f_conGetKey();
				break;
			}
			
			f_yieldCPU();
		}
	}

	if (gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}
	
	f_conExit();
	FlmShutdown();

	gv_bRunning = FALSE;
	return( iRetCode);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC FLMBOOL bldDoRebuild( void)
{
	RCODE				rc;
	FLMBOOL			bOk = TRUE;
	char				szErrMsg[ 100];
	CREATE_OPTS		createOpts;

	gv_ui64DatabaseSize = 0;
	gv_ui64BytesDone = 0;
	gv_uiDictRecsRecovered = 0;
	gv_iLastDoing = -1;
	gv_uiTotalRecs = 0;
	gv_uiRecsRecovered = 0;

	f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
	f_conClearScreen( 0, 0);

	gv_bLoggingEnabled = FALSE;
	gv_uiLogBufferCount = 0;

	if( gv_szLogFileName[ 0])
	{
		gv_pFileSystem->deleteFile( gv_szLogFileName);
		if (RC_OK( rc = gv_pFileSystem->createFile( 
			gv_szLogFileName, FLM_IO_RDWR, &gv_pLogFile)))
		{
			gv_bLoggingEnabled = TRUE;
		}
		else
		{
			f_strcpy( szErrMsg, "Error creating log file: ");
			f_strcpy( &szErrMsg[ f_strlen( szErrMsg)], FlmErrorString( rc));
			bldShowError( szErrMsg);
			bOk = FALSE;
			goto Exit;
		}
	}

	/* Configure FLAIM */

	if (RC_BAD( rc = FlmConfig( FLM_CACHE_LIMIT,
							(void *)(gv_uiCacheSize * 1024), (void *)0)))
	{
		f_strcpy( szErrMsg, "Error setting cache size for FLAIM share: ");
		f_strcpy( &szErrMsg[ f_strlen( szErrMsg)], FlmErrorString( rc));
		bldShowError( szErrMsg);
		bOk = FALSE;
		goto Exit;
	}

	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	if( gv_bLoggingEnabled)
	{
		bldLogString( NULL);
		bldLogString( NULL);
		bldLogString( NULL);
		bldLogString( 
"==========================================================================");
		bldLogString( "REBUILD PARAMETERS:");
	}
	f_conClearScreen( 0, PARAM_ROW);
	f_conStrOutXY( "REBUILD PARAMETERS:", LABEL_COLUMN, PARAM_ROW);
	bldOutLabel( LABEL_COLUMN + 2, SOURCE_ROW, "Source DB",
					 gv_szSrcFileName, 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, SOURCE_DATA_DIR_ROW,
		"Src. Data Dir",
		(gv_szSrcDataDir [0])
		? &gv_szSrcDataDir [0]
		: "<NONE>", 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, DEST_ROW, "Destination DB",
		gv_szDestFileName, 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, DEST_DATA_DIR_ROW,
		"Dest. Data Dir",
		(gv_szDestDataDir [0])
		? &gv_szDestDataDir [0]
		: "<NONE>", 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, DEST_RFL_ROW, "Dest. RFL Dir",
		(gv_szDestRflDir [0])
		? &gv_szDestRflDir [0]
		: "<NONE>", 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, DICT_ROW, "Dictionary File",
		(gv_szDictFileName [0])
		? &gv_szDictFileName [0]
		: "<NONE>", 0, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, CACHE_ROW, "Cache (kb)", NULL,
		gv_uiCacheSize, TRUE);
	bldOutLabel( LABEL_COLUMN + 2, LOG_FILE_ROW, "Log File",
		(gv_szLogFileName [0])
		? &gv_szLogFileName [0]
		: "<NONE>", 0, TRUE);
	bldOutLabel( LABEL_COLUMN, DOING_ROW, "Current Action",
		"Startup                  ", 0L, FALSE);
	bldOutLabel( LABEL_COLUMN, DB_SIZE_ROW, "Database Size",
		NULL, (FLMUINT)gv_ui64DatabaseSize, FALSE);
	bldOutLabel( LABEL_COLUMN, BYTES_DONE_ROW, "Bytes Processed",
		NULL, (FLMUINT)gv_ui64BytesDone, FALSE);
	bldOutLabel( LABEL_COLUMN, TOTAL_REC_ROW, "Total Records",
		NULL, gv_uiTotalRecs, FALSE);
	bldOutLabel( LABEL_COLUMN, RECOV_ROW, "Records Recovered",
		NULL, gv_uiRecsRecovered, FALSE);
	bldOutLabel( LABEL_COLUMN, DICT_RECOV_ROW, "Dict Items Recov",
		NULL, gv_uiDictRecsRecovered, FALSE);

	if( gv_szDictFileName [0])
	{
		gv_pszDictPath = &gv_szDictFileName [0];
	}
	else
	{
		gv_pszDictPath = NULL;
	}

	/*
	Open the database ONLY to get the createOpts.
	Rebuild the exact prefix and other create options.
	*/

	rc = bldGetCreateOpts( gv_szSrcFileName, &createOpts);
	if ((!gv_bShutdown) && (RC_OK( rc)))
	{
		char *	pszDestRflDir;

		pszDestRflDir = ((gv_szDestRflDir [0])
											 ? &gv_szDestRflDir [0]
											 : NULL);

		rc = FlmDbRebuild( gv_szSrcFileName, gv_szSrcDataDir,
									gv_szDestFileName, gv_szDestDataDir,
									pszDestRflDir,
									gv_pszDictPath, &createOpts,
									&gv_uiTotalRecs,
									&gv_uiRecsRecovered,
									bldProgFunc, NULL);
		bldShowResults( "FlmDbRebuild",
									rc, gv_uiTotalRecs, gv_uiRecsRecovered,
									gv_uiDictRecsRecovered);
	}

Exit:

	if( gv_bLoggingEnabled)
	{
		bldLogFlush();
		gv_pLogFile->Release();
		gv_pLogFile = NULL;
	}

	return( bOk);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldShowResults(
	const char *	FuncName,
	RCODE				rc,
	FLMUINT			uiTotalRecords,
	FLMUINT			uiRecordsRecovered,
	FLMUINT			uiDictRecordsRecovered)
{
	char		szErrMsg[ 100];

	if( RC_BAD( rc))
	{
		if( rc != FERR_FAILURE)
		{
			f_strcpy( szErrMsg, "Error calling ");
			f_strcpy( &szErrMsg[ f_strlen( szErrMsg)], FuncName);
			f_strcpy( &szErrMsg[ f_strlen( szErrMsg)], ": ");
			f_strcpy( &szErrMsg[ f_strlen( szErrMsg)], FlmErrorString( rc));
			bldShowError( szErrMsg);
			if( gv_bLoggingEnabled)
			{
				bldLogString( szErrMsg);
			}
		}
		else if( gv_bLoggingEnabled)
		{
			bldLogString( "REBUILD HALTED BY USER");
			gv_bShutdown = TRUE;
		}
	}
	else
	{
		bldOutNumValue( TOTAL_REC_ROW, uiTotalRecords);
		bldOutNumValue( RECOV_ROW, uiRecordsRecovered);
		bldOutNumValue( DICT_RECOV_ROW, uiDictRecordsRecovered);
		if( gv_bLoggingEnabled)
		{
			f_sprintf( (char *)szErrMsg, "Total Records:     %u", (unsigned)uiTotalRecords);
			bldLogString( szErrMsg);
			f_sprintf( (char *)szErrMsg, "Records Recovered: %u", (unsigned)uiRecordsRecovered);
			bldLogString( szErrMsg);
			f_sprintf( (char *)szErrMsg, "Dict Items Recovered: %u",
					(unsigned)uiDictRecordsRecovered);
			bldLogString( szErrMsg);
		}
		f_strcpy( szErrMsg, "Recovery completed successfully");
		bldShowError( szErrMsg);
		if( gv_bLoggingEnabled)
		{
			bldLogString( szErrMsg);
		}
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldShowHelp(
	void
	)
{
	f_conStrOut( "\n");
	f_conStrOut( 
"Parameters: <SourceName> <DestName> [Options]\n\n");
	f_conStrOut( 
"SourceName = Name of database which is to be recovered.\n");
	f_conStrOut( 
"DestName   = Name of destination database to recover data to.  Recovered\n");
	f_conStrOut( 
"             records are put in this database.\n");
	f_conStrOut( 
"Options    = (may be specified anywhere on command line): \n");
	f_conStrOut( 
"  -c<n>         = Cache (kilobytes) to use.\n");
	f_conStrOut( 
"  -sd<DirName>  = Data directory for source DB.\n");
	f_conStrOut( 
"  -dc<DictName> = Dictionary file to use to create destination DB.\n");
	f_conStrOut( 
"  -dd<DirName>  = Data directory for destination DB.\n");
	f_conStrOut( 
"  -dr<DirName>  = RFL directory for destination DB.\n");
	f_conStrOut( 
"  -l<FileName>  = Log detailed information to <FileName>.\n");
	f_conStrOut( 
"  -b            = Run in Batch Mode.\n");
	f_conStrOut( 
"  -h<HdrInfo>   = Fix file header information. HdrInfo is in the format\n");
	f_conStrOut( 
"                  BlkSiz:Prod:FType:MajVer:MinVer:InitLog:LogExt:Lang:FlmVer\n");
	f_conStrOut( 
"  -q<FileName>  = Output binary log information to <FileName>.\n");
	f_conStrOut( 
"  -v<FileName>  = Verify binary log information in <FileName>.  NOTE: The\n");
	f_conStrOut( 
"                  -v and -q options cannot both be specified.\n");
	f_conStrOut( 
"  -p            = Pause before exiting.\n");
	f_conStrOut( 
"  -?           = A '?' anywhere in the command line will cause this help\n");
	f_conStrOut( 
"                 screen to be displayed, with or without the leading '-'.\n");
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC FLMBOOL bldGetParams(
	FLMINT			iArgC,
	const char **	ppszArgV)
{
#define MAX_ARGS     30
	FLMUINT			uiLoop;
	char				szErrMsg [100];
	const char *	pszPtr;
	const char *	ppszArgs[ MAX_ARGS];
	char				szCommandBuffer [300];

	gv_szSrcFileName [0] = 0;
	gv_szSrcDataDir [0] = 0;
	gv_szDestFileName [0] = 0;
	gv_szDestDataDir [0] = 0;
	gv_szDestRflDir [0] = 0;
	gv_szDictFileName [0] = 0;
	gv_szLogFileName [0] = 0;
	gv_bFixHdrInfo = FALSE;
	gv_DefaultCreateOpts.uiBlockSize = DEFAULT_BLKSIZ;
	gv_DefaultCreateOpts.uiMinRflFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
	gv_DefaultCreateOpts.uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
	gv_DefaultCreateOpts.bKeepRflFiles = DEFAULT_KEEP_RFL_FILES_FLAG;
	gv_DefaultCreateOpts.bLogAbortedTransToRfl = DEFAULT_LOG_ABORTED_TRANS_FLAG;
	gv_DefaultCreateOpts.uiDefaultLanguage = DEFAULT_LANG;
	gv_DefaultCreateOpts.uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
	gv_DefaultCreateOpts.uiAppMajorVer = 
	gv_DefaultCreateOpts.uiAppMinorVer = 0;
	gv_uiCacheSize = 30000;
	gv_bBatchMode = FALSE;

	// Ask the user to enter parameters if none were entered on the command
	// line.

	if( iArgC < 2)
	{
		for (;;)
		{
			f_conStrOut( "\nRebuild Params (enter ? for help): ");
			szCommandBuffer[ 0] = 0;
			f_conLineEdit( szCommandBuffer, sizeof( szCommandBuffer) - 1);
			if( gv_bShutdown)
			{
				return( FALSE);
			}

			if( f_stricmp( szCommandBuffer, "?") == 0)
			{
				bldShowHelp();
			}
			else
			{
				break;
			}
		}
		flmUtilParseParams( szCommandBuffer, MAX_ARGS, &iArgC, &ppszArgs [1]);
		ppszArgs[ 0] = ppszArgV[ 0];
		iArgC++;
		ppszArgV = &ppszArgs[ 0];
	}

	uiLoop = 1;
	while (uiLoop < (FLMUINT)iArgC)
	{
		pszPtr = ppszArgV [uiLoop];

		// See if they specified an option

#ifdef FLM_UNIX
		if (*pszPtr == '-')
#else
		if (*pszPtr == '-' || *pszPtr == '/')
#endif
		{
			pszPtr++;
			if (*pszPtr == 'c' || *pszPtr == 'C')
			{
				gv_uiCacheSize = f_atoi( (pszPtr + 1));
			}
			else if (*pszPtr == 'd' || *pszPtr == 'D')
			{
				pszPtr++;
				if (*pszPtr == 'r' || *pszPtr == 'R')
				{
					pszPtr++;
					if (*pszPtr)
					{
						f_strcpy( gv_szDestRflDir, pszPtr);
					}
					else
					{
						bldShowError(
							"Destination RFL directory not specified");
						return( FALSE);
					}
				}
				else if (*pszPtr == 'd' || *pszPtr == 'D')
				{
					pszPtr++;
					if (*pszPtr)
					{
						f_strcpy( gv_szDestDataDir, pszPtr);
					}
					else
					{
						bldShowError(
							"Destination data directory not specified");
						return( FALSE);
					}
				}
				else if (*pszPtr == 'c' || *pszPtr == 'C')
				{
					pszPtr++;
					if (*pszPtr)
					{
						f_strcpy( gv_szDictFileName, pszPtr);
					}
					else
					{
						bldShowError(
							"Dictionary file name not specified");
						return( FALSE);
					}
				}
				else
				{
					f_sprintf( szErrMsg, "Invalid option %s", pszPtr-1);
					bldShowError( szErrMsg);
					return( FALSE);
				}
			}
			else if (*pszPtr == 's' || *pszPtr == 'S')
			{
				pszPtr++;
				if (*pszPtr == 'd' || *pszPtr == 'D')
				{
					pszPtr++;
					if (*pszPtr)
					{
						f_strcpy( gv_szSrcDataDir, pszPtr);
					}
					else
					{
						bldShowError(
							"Source data directory not specified");
						return( FALSE);
					}
				}
				else
				{
					f_sprintf( szErrMsg, "Invalid option %s", pszPtr-1);
					bldShowError( szErrMsg);
					return( FALSE);
				}
			}
			else if (*pszPtr == 'h' || *pszPtr == 'H')
			{
				pszPtr++;
				if( *pszPtr)
				{
					if( !bldParseHdrInfo( pszPtr))
					{
						return( FALSE);
					}
				}
				else
				{
					bldShowError( "Block sizes not specified");
					return( FALSE);
				}
			}
			else if (*pszPtr == 'l' || *pszPtr == 'L')
			{
				pszPtr++;
				if (*pszPtr)
				{
					f_strcpy( gv_szLogFileName, pszPtr);
				}
				else
				{
					bldShowError( "Log file name not specified");
					return( FALSE);
				}
			}
			else if (f_stricmp( pszPtr, "P") == 0)
			{
				gv_bPauseBeforeExiting = TRUE;
			}
			else if (f_stricmp( pszPtr, "B") == 0)
			{
				gv_bBatchMode = TRUE;
			}
			else if (f_stricmp( pszPtr, "?") == 0)
			{
				goto Show_Help;
			}
			else
			{
				f_sprintf( szErrMsg, "Invalid option %s", pszPtr);
				bldShowError( szErrMsg);
				return( FALSE);
			}
		}
		else if (f_stricmp( pszPtr, "?") == 0)
		{
Show_Help:
			bldShowHelp();
			gv_bPauseBeforeExiting = TRUE;
			return( FALSE);
		}
		else if (!gv_szSrcFileName[ 0])
		{
			f_strcpy( gv_szSrcFileName, pszPtr);
		}
		else if (!gv_szDestFileName[ 0])
		{
			f_strcpy( gv_szDestFileName, pszPtr);
		}
		uiLoop++;
	}

	if (!gv_szSrcFileName [0] || !gv_szDestFileName [0])
	{
		goto Show_Help;
	}
	
	return( TRUE);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC FLMBOOL bldParseHdrInfo(
	const char *	pucBuffer)
{
	FLMUINT			uiNum;
	FLMBOOL			bHaveParam;
	CREATE_OPTS		CreateOpts;
	FLMUINT			uiFieldNum;

	f_memcpy( &CreateOpts, &gv_DefaultCreateOpts, sizeof( CREATE_OPTS));
	uiFieldNum = 1;
	for (;;)
	{
		uiNum = 0;
		bHaveParam = FALSE;
		while ((*pucBuffer == ' ') ||
				 (*pucBuffer == ':') ||
				 (*pucBuffer == ',') ||
				 (*pucBuffer == ';') ||
				 (*pucBuffer == '\t'))
		{
			pucBuffer++;
		}

		if( uiFieldNum == 8)	// Language
		{
			char		pszTmpBuf[ 100];
			FLMUINT	uiTmpLen = 0;

			while ((*pucBuffer) &&
					 (*pucBuffer != ':') &&
					 (*pucBuffer != ',') &&
					 (*pucBuffer != ';') &&
					 (*pucBuffer != ' ') &&
					 (*pucBuffer != '\t'))
			{
				pszTmpBuf[ uiTmpLen++] = *pucBuffer++;
			}

			pszTmpBuf[ uiTmpLen] = 0;
			if( uiTmpLen)
			{
				uiNum = f_languageToNum( pszTmpBuf);
				if( (!uiNum) && (f_stricmp( pszTmpBuf, "US") != 0))
				{
					bldShowError( "Illegal language in header information");
					return( FALSE);
				}
				bHaveParam = TRUE;
			}
		}
		else
		{
			while( (*pucBuffer >= '0') && (*pucBuffer <= '9'))
			{
				uiNum *= 10;
				uiNum += (FLMUINT)(*pucBuffer - '0');
				pucBuffer++;
				bHaveParam = TRUE;
			}
		}

		if( ((*pucBuffer != 0) &&
			  (*pucBuffer != ' ') &&
			  (*pucBuffer != ':') &&
			  (*pucBuffer != ',') &&
			  (*pucBuffer != ';') &&
			  (*pucBuffer != '\t')))
		{
			bldShowError( "Illegal value in header information");
			return( FALSE);
		}

		if( bHaveParam)
		{
			switch( uiFieldNum)
			{
				case 1:
					if( uiNum != 0 && !VALID_BLOCK_SIZE( uiNum))
					{
						bldShowError( "Illegal block size");
						return( FALSE);
					}
					CreateOpts.uiBlockSize = uiNum;
					break;
				case 4:
					if( uiNum > 255)
					{
						bldShowError( "Illegal application major version");
						return( FALSE);
					}
					CreateOpts.uiAppMajorVer = uiNum;
					break;
				case 5:
					if( uiNum > 255)
					{
						bldShowError( "Illegal application minor version");
						return( FALSE);
					}
					CreateOpts.uiAppMinorVer = uiNum;
					break;
				case 6:
					CreateOpts.uiMaxRflFileSize = uiNum;
					break;
				case 7:
					CreateOpts.uiDefaultLanguage = uiNum;
					break;
				case 8:
					CreateOpts.uiVersionNum = uiNum;
					break;
				default:
					bldShowError( "Too many parameters in header information");
					return( FALSE);
			}
		}

		if( !(*pucBuffer))
		{
			break;
		}

		uiFieldNum++;
	}

	gv_bFixHdrInfo = TRUE;
	f_memcpy( &gv_DefaultCreateOpts, &CreateOpts, sizeof( CREATE_OPTS));
	return( TRUE);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldOutLabel(
	FLMUINT			uiCol,
	FLMUINT			uiRow,
	const char *	pucLabel,
	const char *	pucValue,
	FLMUINT			uiNumValue,
	FLMBOOL			bLogIt)
{
	char			szMsg[ 100];
	FLMUINT		uiLen = (FLMUINT)(VALUE_COLUMN - uiCol - 1);

	f_memset( szMsg, '.', uiLen);
	szMsg[ uiLen] = 0;
	f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
	f_conStrOutXY( szMsg, uiCol, uiRow);
	f_conStrOutXY( pucLabel, uiCol, uiRow);

	if( pucValue != NULL)
	{
		bldOutValue( uiRow, pucValue);
	}
	else
	{
		bldOutNumValue( uiRow, uiNumValue);
	}

	if( (bLogIt) && (gv_bLoggingEnabled))
	{
		f_strcpy( szMsg, pucLabel);
		f_strcpy( &szMsg[ f_strlen( szMsg)], ": ");
		if( pucValue != NULL)
		{
			f_strcpy( &szMsg[ f_strlen( szMsg)], pucValue);
		}
		else
		{
			f_sprintf( (char *)(&szMsg[ f_strlen( szMsg)]), "%u",
				(unsigned)uiNumValue);
		}
		bldLogString( szMsg);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldOutValue(
	FLMUINT			uiRow,
	const char *	pucValue)
{
	f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
	f_conStrOutXY( pucValue, VALUE_COLUMN, uiRow);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldOutNumValue(
	FLMUINT		uiRow,
	FLMUINT		uiNumber)
{
	char		szMsg[ 80];

	f_sprintf( (char *)szMsg, "%-10u  (0x%08X)", 
		(unsigned)uiNumber, (unsigned)uiNumber);
	bldOutValue( uiRow, szMsg);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC RCODE bldGetUserInput(
	void
	)
{
	FLMUINT		uiChar;

	f_conStrOutXY( "Q,ESC=Quit, Other=Continue: ", 0, 23);
	for (;;)
	{
		if( gv_bShutdown)
		{
			uiChar = FKB_ESCAPE;
			break;
		}
		else if( f_conHaveKey())
		{
			uiChar = f_conGetKey();
			if( uiChar)
			{
				break;
			}
		}
		
		f_yieldCPU();
	}

	f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
	f_conClearScreen( 0, 22);
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
Desc: ?
*********************************************************************/
FSTATIC void bldLogStr(
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
				bldLogFlush();
			}
		}
		bldLogString( pucStr);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldLogCorruptError(
	CORRUPT_INFO *			pCorruptInfo)
{
	char		szBuf[ 100];

	/* Log the container number */

	bldLogString( NULL);
	bldLogString( "ERROR IN DATABASE");
	f_sprintf( (char *)szBuf, "Container Number: %u",
					(unsigned)pCorruptInfo->uiErrLfNumber);
	bldLogStr( 2, szBuf);

	/* Log the block address, if known */

	if (pCorruptInfo->uiErrBlkAddress)
	{
		f_sprintf( (char *)szBuf, "Block Address: 0x%08X (%u)",
						 (unsigned)pCorruptInfo->uiErrBlkAddress,
						 (unsigned)pCorruptInfo->uiErrBlkAddress);
		bldLogStr( 2, szBuf);
	}

	/* Log the parent block address, if known */

	if (pCorruptInfo->uiErrParentBlkAddress)
	{
		f_sprintf( (char *)szBuf, "Parent Block Address: 0x%08X (%u)",
						 (unsigned)pCorruptInfo->uiErrParentBlkAddress,
						 (unsigned)pCorruptInfo->uiErrParentBlkAddress);
		bldLogStr( 2, szBuf);
	}

	/* Log the element offset, if known */

	if (pCorruptInfo->uiErrElmOffset)
	{
		f_sprintf( (char *)szBuf, "Offset of Element within Block: %u",
						 (unsigned)pCorruptInfo->uiErrElmOffset);
		bldLogStr( 2, szBuf);
	}

	/* Log the elment record offset, if known */

	if (pCorruptInfo->uiErrElmRecOffset != 0xFFFF)
	{
		f_sprintf( (char *)szBuf, "Offset within Element Record: %u",
						 (unsigned)pCorruptInfo->uiErrElmRecOffset);
		bldLogStr( 2, szBuf);
	}

	/* Log the record number, if known */

	if (pCorruptInfo->uiErrDrn)
	{
		f_sprintf( (char *)szBuf, "Record Number: %u",
			(unsigned)pCorruptInfo->uiErrDrn);
		bldLogStr( 2, szBuf);
	}

	/* Log the field number, if known */

	if (pCorruptInfo->uiErrFieldNum)
	{
		f_sprintf( (char *)szBuf, "Field Number: %u",
			(unsigned)pCorruptInfo->uiErrFieldNum);
		bldLogStr( 2, szBuf);
	}

	/* Log the error message */

	f_strcpy( szBuf, FlmVerifyErrToStr( pCorruptInfo->eCorruption));
	f_sprintf( (char *)(&szBuf [f_strlen( szBuf)]), " (%d)",
		(int)pCorruptInfo->eCorruption);
	bldLogStr( 2, szBuf);
	bldLogStr( 0, NULL);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC RCODE bldProgFunc(
	eStatusType	eStatus,
	void *		Parm1,
	void *		Parm2,
	void *		pvAppData
	)
{
	RCODE	rc = FERR_OK;

	F_UNREFERENCED_PARM( Parm2);
	F_UNREFERENCED_PARM( pvAppData);

	if( eStatus == FLM_DB_COPY_STATUS)
	{
		DB_COPY_INFO *	pCopyInfo = (DB_COPY_INFO *)Parm1;
		char				ucDoing [200];

		if( gv_ui64DatabaseSize != pCopyInfo->ui64BytesToCopy)
		{
			gv_ui64DatabaseSize = pCopyInfo->ui64BytesToCopy;
			bldOutNumValue( DB_SIZE_ROW, (FLMUINT)gv_ui64DatabaseSize);
		}
		gv_ui64BytesDone = pCopyInfo->ui64BytesCopied;
		bldOutNumValue( BYTES_DONE_ROW, (FLMUINT)gv_ui64BytesDone);
		gv_iLastDoing = -1;
		if (pCopyInfo->bNewSrcFile)
		{
			f_sprintf( (char *)ucDoing, "Saving File %-15s",
				(char *)pCopyInfo->szSrcFileName);
			ucDoing [25] = 0;
			bldOutValue( DOING_ROW, ucDoing);
		}
	}
	else if( eStatus == FLM_REBUILD_STATUS)
	{
		REBUILD_INFO *	Progress = (REBUILD_INFO *)Parm1;

		/* First update the display */

		if( gv_iLastDoing != Progress->iDoingFlag)
		{
			gv_ui64DatabaseSize = Progress->ui64DatabaseSize;
			bldOutNumValue( DB_SIZE_ROW, (FLMUINT)gv_ui64DatabaseSize);
			gv_iLastDoing = Progress->iDoingFlag;

			if( gv_iLastDoing == REBUILD_GET_BLK_SIZ)
			{
				bldOutValue( DOING_ROW, "Determining Block Size   ");
			}
			else if( gv_iLastDoing == REBUILD_RECOVER_DICT)
			{
				bldOutValue( DOING_ROW, "Recovering Dictionaries  ");
			}
			else
			{
				bldOutValue( DOING_ROW, "Recovering Data          ");
			}
		}
		if( gv_iLastDoing == REBUILD_GET_BLK_SIZ)
		{
			if( gv_ui64DatabaseSize != Progress->ui64DatabaseSize)
			{
				gv_ui64DatabaseSize = Progress->ui64DatabaseSize;
				bldOutNumValue( DB_SIZE_ROW, (FLMUINT)gv_ui64DatabaseSize);
			}
			gv_ui64BytesDone = Progress->ui64BytesExamined;
			bldOutNumValue( BYTES_DONE_ROW, (FLMUINT)gv_ui64BytesDone);
		}
		else
		{
			if( gv_ui64DatabaseSize != Progress->ui64DatabaseSize)
			{
				gv_ui64DatabaseSize = Progress->ui64DatabaseSize;
				bldOutNumValue( DB_SIZE_ROW, (FLMUINT)gv_ui64DatabaseSize);
			}
			gv_ui64BytesDone = Progress->ui64BytesExamined;
			bldOutNumValue( BYTES_DONE_ROW, (FLMUINT)gv_ui64BytesDone);
			if( gv_uiTotalRecs != Progress->uiTotRecs)
			{
				gv_uiTotalRecs = Progress->uiTotRecs;
				bldOutNumValue( TOTAL_REC_ROW, gv_uiTotalRecs);
			}

			if( gv_iLastDoing == REBUILD_RECOVER_DICT)
			{
				if( gv_uiDictRecsRecovered != Progress->uiRecsRecov)
				{
					gv_uiDictRecsRecovered = Progress->uiRecsRecov;
					bldOutNumValue( DICT_RECOV_ROW, gv_uiDictRecsRecovered);
				}
			}
			else
			{
				if( gv_uiRecsRecovered != Progress->uiRecsRecov)
				{
					gv_uiRecsRecovered = Progress->uiRecsRecov;
					bldOutNumValue( RECOV_ROW, gv_uiRecsRecovered);
				}
			}
		}
	}
	else if( eStatus == FLM_PROBLEM_STATUS)
	{
		CORRUPT_INFO *	pCorruptInfo = (CORRUPT_INFO *)Parm1;

		bldLogCorruptError( pCorruptInfo);
		goto Exit;
	}
	else if( eStatus == FLM_CHECK_RECORD_STATUS)
	{
		CHK_RECORD *	pChkRec = (CHK_RECORD *)Parm1;

		if (pChkRec->pDictRecSet)
		{
			pChkRec->pDictRecSet->clear();
		}
	}

	if ((f_conHaveKey()) && (f_conGetKey() == FKB_ESCAPE))
	{
		f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
		f_conClearScreen( 0, 22);
		f_conSetBackFore (FLM_RED, FLM_WHITE);
		f_conStrOutXY( "ESCAPE key pressed", 0, 22);
		rc = bldGetUserInput();
		goto Exit;
	}
	
	f_yieldCPU();
	
Exit:

	return( rc);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldShowError(
	const char * 	Message)
{

	if( !gv_bBatchMode)
	{
		f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
		f_conClearScreen( 0, 22);
		f_conSetBackFore( FLM_RED, FLM_WHITE);
		f_conStrOutXY( Message, 0, 22);
		f_conStrOutXY( "Press any character to continue, ESCAPE to quit: ", 0, 23);
		
		for (;;)
		{
			if (gv_bShutdown)
			{
				break;
			}
			else if (f_conHaveKey())
			{
				if (f_conGetKey() == FKB_ESCAPE)
				{
					gv_bShutdown = TRUE;
				}
				break;
			}
			
			f_yieldCPU();
		}
		
		f_conSetBackFore( FLM_BLACK, FLM_LIGHTGRAY);
		f_conClearScreen( 0, 22);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldLogFlush(
	void
	)
{
	FLMUINT		uiBytesWritten;

	if( gv_uiLogBufferCount)
	{
		gv_pLogFile->write( FLM_IO_CURRENT_POS,
			gv_uiLogBufferCount, gv_pucLogBuffer, &uiBytesWritten);
		gv_uiLogBufferCount = 0;
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void bldLogString(
	const char *	pucStr)
{
	FLMUINT		uiLen;
	FLMUINT		uiLoop;

	if( (gv_bLoggingEnabled) && (gv_pucLogBuffer != NULL))
	{
		uiLen = (FLMUINT)((pucStr != NULL) ? (FLMUINT)f_strlen( pucStr) : 0);
		for( uiLoop = 0; uiLoop < uiLen; uiLoop++)
		{
			gv_pucLogBuffer[ gv_uiLogBufferCount++] = *pucStr++;
			if( gv_uiLogBufferCount == MAX_LOG_BUFF)
			{
				bldLogFlush();
			}
		}
		gv_pucLogBuffer[ gv_uiLogBufferCount++] = '\r';
		if( gv_uiLogBufferCount == MAX_LOG_BUFF)
		{
			bldLogFlush();
		}
		gv_pucLogBuffer[ gv_uiLogBufferCount++] = '\n';
		if( gv_uiLogBufferCount == MAX_LOG_BUFF)
		{
			bldLogFlush();
		}
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC RCODE bldGetCreateOpts(
	const char *		pszFileName,
	CREATE_OPTS *		pCreateOpts)
{
	RCODE					rc = FERR_OK;
	char					szBuf[ 80];
	FLMBYTE				ucLogHdr [LOG_HEADER_SIZE];
	HDR_INFO				HdrInfo;
	FLMUINT				uiVersion;
	IF_FileSystem *	pFileSystem = NULL;
	IF_FileHdl *		pCFileHdl = NULL;

	f_memset( pCreateOpts, 0, sizeof( CREATE_OPTS));
	if( gv_bFixHdrInfo)
	{
		f_memcpy( pCreateOpts, &gv_DefaultCreateOpts, sizeof( CREATE_OPTS));
		goto Exit;
	}

	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pFileSystem->openFile( pszFileName, 
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &pCFileHdl)))
	{
		goto Exit;
	}
	
	if( (rc = flmGetHdrInfo( pCFileHdl, &HdrInfo.FileHdr,
									 &HdrInfo.LogHdr, ucLogHdr)) == FERR_NOT_FLAIM)
	{
		uiVersion = gv_DefaultCreateOpts.uiVersionNum;
		f_memcpy( pCreateOpts, &gv_DefaultCreateOpts, sizeof( CREATE_OPTS));
		rc = FERR_OK;
	}
	else
	{
		uiVersion = HdrInfo.FileHdr.uiVersionNum;
		flmGetCreateOpts( &HdrInfo.FileHdr, ucLogHdr, pCreateOpts);
	}

	if (rc != FERR_OK &&
		 rc != FERR_INCOMPLETE_LOG &&
		 rc != FERR_BLOCK_CHECKSUM)
	{
		if (((rc == FERR_UNSUPPORTED_VERSION) || (rc == FERR_NEWER_FLAIM)) &&
			 (uiVersion == 999))
		{
			rc = FERR_OK;
		}
		else
		{
			f_strcpy( szBuf, "Error reading header info from ");
			f_strcpy( &szBuf[ f_strlen( szBuf)], pszFileName);
			f_strcpy( &szBuf[ f_strlen( szBuf)], ": ");
			f_strcpy( &szBuf[ f_strlen( szBuf)], FlmErrorString( rc));
			bldShowError( szBuf);
			if( gv_bLoggingEnabled)
			{
				bldLogString( szBuf);
			}
			goto Exit;
		}
	}
	else
	{
		rc = FERR_OK;
	}
	
Exit:

	if( pCFileHdl)
	{
		pCFileHdl->Release();
	}
	if (pFileSystem)
	{
		pFileSystem->Release();
	}
	
	return( rc);
}
