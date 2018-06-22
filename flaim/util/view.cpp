//-------------------------------------------------------------------------
// Desc:	Database viewer utility main.
// Tabs:	3
//
// Copyright (c) 1992-1995, 1997-2007 Novell, Inc. All Rights Reserved.
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

#define MAIN_MODULE

#include "view.h"
#include "sharutil.h"

#define UTIL_ID	"VIEW"

// Main Menu options

#define MAIN_MENU_FILE_HEADER    1
#define MAIN_MENU_LOG_HEADER     2
#define MAIN_MENU_LOGICAL_FILES  3

FSTATIC void ViewShowHelp(
	FLMBOOL bShowFullUsage);

FSTATIC FLMUINT16 ViewGetChar(
	const char *	pszMessage1,
	const char *	pszMessage2,
	FLMUINT16		ui16DefaultChar);

FSTATIC RCODE ViewReadAndVerifyHdrInfo( void);

FSTATIC FLMBOOL ViewGetFileName(
	FLMUINT		uiCol,
	FLMUINT		uiRow,
	FLMBOOL		bDispOnly);

FSTATIC FLMBOOL ViewOpenFile( void);

FSTATIC void ViewDoMainMenu( void);

FSTATIC FLMINT ViewSetupMainMenu( void);

FSTATIC FLMBOOL ViewOpenFileDirect( void);

static FLMBOOL		bPauseBeforeExiting = FALSE;
FLMUINT				gv_uiTopLine = 0;
FLMUINT				gv_uiBottomLine = 0;
static char			gv_szPassword[ 256];


/********************************************************************
Desc: ?
*********************************************************************/
#ifdef FLM_RING_ZERO_NLM
extern "C" int nlm_main(
#else
int main(
#endif
	int			argc,
	char **		ArgV)
{
#define MAX_ARGS     30
	FLMINT    		i;
	FLMINT    		ArgC = argc;
	const char *	ppArgs [MAX_ARGS];
	char   			CommandBuffer [300];

	if( RC_BAD( FlmStartup()))
	{
		goto Exit;
	}


	/* Setup defaults for fixing the file header if necessary */

	gv_ViewFixOptions.uiBlockSize = 2048;
	gv_ViewFixOptions.uiAppMajorVer = 0;
	gv_ViewFixOptions.uiAppMinorVer = 0;
	gv_ViewFixOptions.uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
	gv_ViewFixOptions.uiDefaultLanguage = 0;

	/* See if a file name was passed in */

	gv_szViewFileName [0] = '\0';
	gv_szDataDir [0] = 0;
	gv_szRflDir [0] = 0;
	gv_bViewExclusive = FALSE;
	gv_bViewFileOpened = FALSE;
	gv_bViewHdrRead = FALSE;
	gv_bViewHaveDictInfo = FALSE;
	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	gv_pSFileHdl = NULL;
	gv_szPassword[ 0] = 0;

	f_conInit( 0xFFFF, 0xFFFF,  "FLAIM Database Viewer");
	f_conGetScreenSize( NULL, &gv_uiBottomLine);
	
	gv_uiTopLine = 2;
	gv_uiBottomLine -= 3;

	if( RC_BAD( FlmGetFileSystem( &gv_pFileSystem)))
	{
		f_conStrOut( "\nCould not allocate a file system object.\n");
		goto Exit;
	}

	// Ask the user to enter parameters if none were entered on the command
	// line.

	if (ArgC < 2)
	{
		for (;;)
		{
			f_conStrOut( "\nView Params (enter ? for help): ");
			CommandBuffer [0] = 0;
			f_conLineEdit( CommandBuffer, sizeof( CommandBuffer) - 1);
			if (f_stricmp( CommandBuffer, "?") == 0)
				ViewShowHelp( FALSE);
			else
				break;
			if (gv_bShutdown)
				goto Exit;
		}
		flmUtilParseParams( CommandBuffer, MAX_ARGS, &ArgC, &ppArgs [1]);
		ppArgs [0] = ArgV [0];
		ArgC++;
		ArgV = (char **)&ppArgs [0];
	}

	i = 1;
	while( i < ArgC)
	{
#ifdef FLM_UNIX
		if (ArgV [i][0] == '-')
#else
		if ((ArgV [i][0] == '/') || (ArgV [i][0] == '-'))
#endif
		{
			switch( ArgV [i][1])
			{
				case 'x':
				case 'X':
					gv_bViewExclusive = TRUE;
					break;
				case 'b':
				case 'B':
					gv_ViewFixOptions.uiBlockSize = f_atoi( (&ArgV [i][2]));
					break;
				case 'd':
				case 'D':
					switch (ArgV [i][2])
					{
						case 'r':
						case 'R':
							f_strcpy( gv_szRflDir, &ArgV [i][3]);
							break;
						case 'd':
						case 'D':
							f_strcpy( gv_szDataDir, &ArgV [i][3]);
							break;
						default:
							break;
					}
					break;
				case 'l':
				case 'L':
					gv_ViewFixOptions.uiMaxRflFileSize = f_atol( (&ArgV [i][2]));
					break;
				case 'p':
				case 'P':
					switch( ArgV [i][2])
					{
						case 0:
							bPauseBeforeExiting = TRUE;
							break;
						case 'M':
							gv_ViewFixOptions.uiAppMajorVer = f_atoi( (&ArgV [i][3]));
							break;
						case 'm':
							gv_ViewFixOptions.uiAppMinorVer = f_atoi( (&ArgV [i][3]));
							break;
						case 'w':
							f_strcpy(gv_szPassword, &ArgV [i][3]);
							break;
						default:
							break;
					}
					break;
				case '?':
					ViewShowHelp( TRUE);
					bPauseBeforeExiting = TRUE;
					goto Exit;
				default:
					break;
			}
		}
		else if (f_stricmp( (ArgV [i]), "?") == 0)
		{
			ViewShowHelp( TRUE);
			bPauseBeforeExiting = TRUE;
			goto Exit;
		}
		else if (!gv_szViewFileName [0])
			f_strcpy( gv_szViewFileName, ArgV [i]);
		i++;
	}

	gv_ViewPool.poolInit( 2048);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, 0);

	// Open the file

	if (ViewOpenFile())
	{

		// Execute the main menu

		ViewDoMainMenu();
		ViewFreeMenuMemory();

		// Close the file

		if (gv_bViewDbInitialized)
		{
			fdbExit( (FDB *)gv_hViewDb);
			(void)FlmDbClose( &gv_hViewDb);
		}
	}

Exit:

	if (gv_pSFileHdl)
	{
		gv_pSFileHdl->Release();
		gv_pSFileHdl = NULL;
	}

	if ((bPauseBeforeExiting) && (!gv_bShutdown))
	{
		f_conStrOut( "\nPress any character to exit VIEW: ");
		for (;;)
		{
			if (gv_bShutdown)
				break;
			if (f_conHaveKey())
			{
				f_conGetKey();
				break;
			}
			viewGiveUpCPU();
		}
	}

	if( gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}

	f_conExit();
	FlmShutdown();

	gv_bRunning = FALSE;
	return 0;
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void ViewShowHelp(
	FLMBOOL bShowFullUsage
	)
{
	f_conStrOut( "\n");
	if (bShowFullUsage)
		f_conStrOut( "Usage: view <FileName> [Options]\n");
	else
		f_conStrOut( "Parameters: <FileName> [Options]\n\n");
	f_conStrOut( 
"   FileName = Name of database to view.\n");
	f_conStrOut( 
"              @<FileName>, where FileName is the name of the file containing\n");
	f_conStrOut( 
"   Options  =\n");
	f_conStrOut( 
"        -dr<Dir>     = RFL directory.\n");
	f_conStrOut( 
"        -dd<Dir>     = Data directory.\n");
	f_conStrOut( 
"        -x           = Open file in exclusive mode.\n");
	f_conStrOut( 
"        -f           = Fix file header.  If the options below are not set,\n");
	f_conStrOut( 
"                       defaults will be used.\n");
	f_conStrOut( 
"        -b<Size>     = Set block size to Size (only used if -f is specified).\n");
	f_conStrOut( 
"        -l<Size>     = Set maximum RFL file size to Size (only used if -f\n");
	f_conStrOut( 
"                       option is used).\n");
	f_conStrOut( 
"                       used).\n");
	f_conStrOut( 
"        -p           = Pause before exiting.\n");
	f_conStrOut( 
"        -pM<MajorVer>= Set application major version number (only used if -f\n");
	f_conStrOut( 
"                       option is used).\n");
	f_conStrOut( 
"        -pm<MinorVer>= Set application minor version number (only used if -f\n");
	f_conStrOut( 
"                       option is used).\n");
	f_conStrOut( 
"        -pw<Password>= Use Password when opening the database\n");
	f_conStrOut( 
"        -?           = A '?' anywhere in the command line will cause this\n");
	f_conStrOut( 
"                       screen to be displayed, with or without the leading '-'.\n");
	f_conStrOut( 
"Options may be specified anywhere in the command line.\n");
}

/***************************************************************************
Desc: Prompt user for a single character response and get the response.
*****************************************************************************/
FSTATIC FLMUINT16 ViewGetChar(
	const char *		pszMessage1,
	const char *		pszMessage2,
	FLMUINT16   		ui16DefaultChar)
{
	FLMUINT16   ui16Char;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	f_conSetBackFore( FLM_RED, FLM_WHITE);
	if (pszMessage1)
		f_conStrOutXY( pszMessage1, 0, uiNumRows - 2);
	f_conStrOutXY( pszMessage2, 0, 23);
	for (;;)
	{
		if (gv_bShutdown)
		{
			ui16Char = FKB_ESCAPE;
			break;
		}
		else if (f_conHaveKey())
		{
			ui16Char = (FLMUINT16)f_conGetKey();
			break;
		}
		viewGiveUpCPU();
	}
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	if (ui16Char == FKB_ENTER)
		ui16Char = ui16DefaultChar;
	if ((ui16Char >= 'a') && (ui16Char <= 'z'))
		ui16Char = ui16Char - 'a' + 'A';
	return( ui16Char);
}

/***************************************************************************
Desc:	This routine reads and verifies the information contained in the
		file header and log header of a FLAIM database.
*****************************************************************************/
FSTATIC RCODE ViewReadAndVerifyHdrInfo( void)
{
	RCODE					rc = FERR_OK;
	RCODE					rc0;
	RCODE					rc1;
	FLMUINT				uiBytesRead;
	FLMBYTE *			pReadBuf = NULL;
	IF_FileHdl *		pCFileHdl = NULL;
	IF_FileSystem *	pFileSystem = NULL;
	FLMUINT				uiTmpLen;

	if( RC_BAD( rc = f_calloc( 2048, &pReadBuf)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = pFileSystem->openFile( gv_szViewFileName, 
		FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &pCFileHdl)))
	{
		goto Exit;
	}
	
	// Read the fixed information area  -- except for the 1st byte --
	// because it might be locked by an active transaction.  We don't
	// care what is in this byte anyway.

	rc0 = pCFileHdl->read( 1L, 2047, &pReadBuf [1], &uiBytesRead);

	// Increment bytes read - to account for byte zero, which
	// was not really read in.

	uiBytesRead++;
	*pReadBuf = 0xFF;

	// Before doing any checking, get whatever we can from the
	// first 2048 bytes.  For the flmGetHdrInfo routine, we want
	// to get whatever we can from the headers, even if it is
	// invalid.

	rc1 = flmGetFileHdrInfo( pReadBuf, &pReadBuf [FLAIM_HEADER_START],
									&gv_ViewHdrInfo.FileHdr);

	// Get the log file header information

	f_memcpy( gv_ucViewLogHdr, &pReadBuf [DB_LOG_HEADER_START],
						LOG_HEADER_SIZE);
	
	flmGetLogHdrInfo( gv_ucViewLogHdr, &gv_ViewHdrInfo.LogHdr);

	// Get some additional information from the file header.

	uiTmpLen = FLAIM_NAME_LEN;
	if (uiTmpLen > sizeof( gv_szFlaimName) - 1)
	{
		uiTmpLen = sizeof( gv_szFlaimName) - 1;
	}
	
	f_memcpy( gv_szFlaimName, &pReadBuf [FLAIM_HEADER_START + FLAIM_NAME_POS],
						uiTmpLen);
						
	gv_szFlaimName [uiTmpLen] = 0;

	uiTmpLen = FLM_FILE_FORMAT_VER_LEN;
	if (uiTmpLen > sizeof( gv_szFlaimVersion) - 1)
	{
		uiTmpLen = sizeof( gv_szFlaimVersion) - 1;
	}
	
	f_memcpy( gv_szFlaimVersion,
		&pReadBuf [FLAIM_HEADER_START + FLM_FILE_FORMAT_VER_POS], uiTmpLen);
	gv_szFlaimVersion [uiTmpLen] = 0;

	// If there is not enough data to satisfy the read, this
	// is probably not a FLAIM file.

	if (RC_BAD( rc0))
	{
		if (rc0 != FERR_IO_END_OF_FILE)
		{
			rc = rc0;
			goto Exit;
		}
		if (uiBytesRead < 2048)
		{
			rc = RC_SET( FERR_NOT_FLAIM);
			goto Exit;
		}
	}

	// See if we got any other errors where we might want to retry
	// the read.

	if (RC_BAD( rc1))
	{
		rc = rc1;
		goto Exit;
	}

Exit:

	if (pReadBuf)
	{
		f_free( &pReadBuf);
	}
	
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

/***************************************************************************
Desc: Read the header information from the database -- this includes
		the file header and the log header.
*****************************************************************************/
void ViewReadHdr( void)
{
	RCODE			rc;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);

	gv_bViewHdrRead = TRUE;
	if (RC_OK( rc = ViewReadAndVerifyHdrInfo()))
	{
		return;
	}

	ViewShowRCError( "reading header information", rc);

	if (!VALID_BLOCK_SIZE( gv_ViewHdrInfo.FileHdr.uiBlockSize))
		gv_ViewHdrInfo.FileHdr.uiBlockSize = DEFAULT_BLKSIZ;
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewAskInput(
	const char *	Prompt,
	char  *			Buffer,
	FLMUINT			BufLen)
{
	char		TempBuf [80];

	f_conStrOut( Prompt);
	if (BufLen > sizeof( TempBuf))
		BufLen = sizeof( TempBuf);
	TempBuf [0] = 0;
	f_conLineEdit( TempBuf, BufLen);
	f_strcpy( Buffer, TempBuf);
}

/***************************************************************************
Desc:    This routine asks the user for the file name to be viewed.
*****************************************************************************/
FSTATIC FLMBOOL ViewGetFileName(
	FLMUINT		uiCol,
	FLMUINT		uiRow,
	FLMBOOL		bDispOnly)
{
	const char *		Prompt = "Enter database file name: ";

	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( uiCol, uiRow);
	
	if (bDispOnly)
	{
		f_conStrOutXY( Prompt, uiCol, uiRow);
		f_conStrOutXY( gv_szViewFileName,
							(FLMBYTE)(uiCol + f_strlen( Prompt)), uiRow);
	}
	else
	{
		f_conSetCursorPos( uiCol, uiRow);
		ViewAskInput( Prompt, gv_szViewFileName, 40);
		if ((!gv_szViewFileName [0]) ||
			 (f_strcmp( gv_szViewFileName, "\\") == 0))
			return( FALSE);
	}
	return( TRUE);
}

/****************************************************************************
Desc: This routine opens a database file in DIRECT mode - because we couldn't
		get it open by calling the normal FLAIM functions.
****************************************************************************/
FSTATIC FLMBOOL ViewOpenFileDirect(
	void
	)
{
	RCODE					rc;
	IF_FileHdl *		pCFileHdl;

	if( RC_BAD( rc = gv_pSFileHdl->getFileHdl( 0, FALSE, &pCFileHdl)))
	{
		ViewShowRCError( "opening file in direct mode", rc);
		return( FALSE);
	}

	gv_bViewFileOpened = TRUE;
	return( TRUE);
}

/***************************************************************************
Desc:    This routine opens the database file which is to be viewed.
*****************************************************************************/
FSTATIC FLMBOOL ViewOpenFile( void)
{
	RCODE						rc;
	FLMBOOL					bOk = FALSE;
	FLMBOOL					bIgnore;
	F_SuperFileClient *	pSFileClient = NULL;

Get_File_Name:

	// Prompt for file name if necessary

	f_conClearScreen( 0, 1);
	if( !gv_szViewFileName [0])
	{
		if( !ViewGetFileName( 5, 5, FALSE))
		{
			goto Exit;
		}
	}
	else
	{
		if( !ViewGetFileName( 5, 5, TRUE))
		{
			goto Exit;
		}
	}
	
	if (gv_pSFileHdl)
	{
		gv_pSFileHdl->Release();
		gv_pSFileHdl = NULL;
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
		pSFileClient = NULL;
	}
	
	if( (pSFileClient = f_new F_SuperFileClient) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( pSFileClient->setup( 
		gv_szViewFileName, gv_szDataDir, gv_ViewHdrInfo.FileHdr.uiVersionNum)))
	{
		goto Exit;
	}
	
	if ((gv_pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		ViewShowRCError( "creating super file handle", rc);
		goto Exit;
	}
	
	if (RC_BAD( rc = gv_pSFileHdl->setup( pSFileClient, 
		gv_FlmSysData.pFileHdlCache, 
		gv_FlmSysData.uiFileOpenFlags, gv_FlmSysData.uiFileCreateFlags)))
	{
		ViewShowRCError( "setting up super file handle", rc);
		goto Exit;
	}
	
	if( RC_BAD( rc = ViewReadAndVerifyHdrInfo()))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND)
		{
			goto Path_Not_Found;
		}
		else
		{
			goto Other_Error;
		}
	}

	gv_pSFileHdl->releaseFiles();

	if (RC_BAD( rc = FlmDbOpen( gv_szViewFileName, gv_szDataDir,
										 gv_szRflDir, FO_DONT_REDO_LOG,
										 gv_szPassword, &gv_hViewDb)))
	{
		char		TBuf[ 100];

		if (rc == FERR_IO_PATH_NOT_FOUND)
		{
Path_Not_Found:
			if (ViewGetChar( NULL,
				"File not found, try another file name? (Y/N, Default=Y): ",
				'Y') != 'Y')
				goto Exit;
			gv_szViewFileName [0] = 0;
			goto Get_File_Name;
		}
		else
		{
Other_Error:
			f_strcpy( TBuf, "Error opening file: ");
			f_strcpy( &TBuf [f_strlen( TBuf)], FlmErrorString( rc));
			if (ViewGetChar( TBuf,
				"Open file in DIRECT MODE anyway? (Y/N, Default=Y): ",
				'Y') == 'Y')
			{
				if (!ViewOpenFileDirect())
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}
	}
	else
	{
		gv_bViewDbInitialized = TRUE;
		if (RC_BAD( rc = fdbInit( (FDB *)gv_hViewDb, FLM_NO_TRANS, TRUE, 0,
										&bIgnore)))
		{
			ViewShowRCError( "calling fdbInit", rc);
			goto Exit;
		}
		
		if (!ViewOpenFileDirect())
		{
			goto Exit;
		}
	}

	bOk = TRUE;
	
Exit:

	if (!bOk)
	{
		if (gv_pSFileHdl)
		{
			gv_pSFileHdl->Release();
			gv_pSFileHdl = NULL;
		}
		
		gv_bViewFileOpened = FALSE;
	}
	
	if( pSFileClient)
	{
		pSFileClient->Release();
	}
	
	return( bOk);
}

/***************************************************************************
Desc:    This routine gets the dictionary information for a database and
			locks it into memory.
*****************************************************************************/
void ViewGetDictInfo( void)
{
	FDB *		pDb = (FDB *)gv_hViewDb;
	FLMUINT	uiSaveFlags;

	if (gv_bViewDbInitialized)
	{

		/* If we have a transaction going, abort it and start another one. */

		if (pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( pDb);
		}

		// Need to fake out flmBeginDbTrans to avoid an assert.
		// This may be the first time we read in a dictionary due to the
		// fact that we did not do recovery and rollback.  flmBeginDbTrans
		// expects the DBF_BEING_OPENED flag to be set the first time
		// a dictionary is read in.  Otherwise, it will assert.

		uiSaveFlags = pDb->pFile->uiFlags;
		pDb->pFile->uiFlags |= DBF_BEING_OPENED;

		// Start a read transaction.

		gv_bViewHaveDictInfo = (RC_OK( flmBeginDbTrans( pDb, FLM_READ_TRANS, 0)))
										? TRUE
										: FALSE;
		pDb->pFile->uiFlags = uiSaveFlags;
	}
}

/***************************************************************************
Desc:    This routine sets up the main menu for the VIEW program.
*****************************************************************************/
FSTATIC FLMINT ViewSetupMainMenu( void)
{
	FLMUINT		uiRow;
	FLMUINT		uiCol;

	/* Initialize the menu structures */

	if (!ViewMenuInit( "Main Menu"))
	{
		return( 0);
	}
	
	uiRow = 3;
	uiCol = 20;

	/* Add each menu item to the menu */

	if (!ViewAddMenuItem( LBL_FILE_HEADER, 0,
												VAL_IS_EMPTY, 0, 0,
												0, 0xFFFFFFFF, 0, MOD_DISABLED,
												uiCol, uiRow++, MAIN_MENU_FILE_HEADER,
												FLM_BLACK, FLM_WHITE,
												FLM_BLUE, FLM_WHITE))
		return( 0);
	if (!ViewAddMenuItem( LBL_LOG_HEADER, 0,
												VAL_IS_EMPTY, 0, 0,
												0, 0xFFFFFFFF, 0, MOD_DISABLED,
												uiCol, uiRow++, MAIN_MENU_LOG_HEADER,
												FLM_BLACK, FLM_WHITE,
												FLM_BLUE, FLM_WHITE))
		return( 0);

	if (gv_ViewHdrInfo.FileHdr.uiFirstLFHBlkAddr == 0xFFFFFFFF)
	{
		if (!ViewAddMenuItem( LBL_LOGICAL_FILES, 0,
									VAL_IS_LABEL_INDEX, (FLMUINT)LBL_NONE, 0,
									0, 0xFFFFFFFF, 0, MOD_DISABLED,
									uiCol, uiRow++, 0,
									FLM_BLACK, FLM_LIGHTGRAY,
									FLM_BLUE, FLM_LIGHTGRAY))
			return( 0);
	}
	else
	{
		if (!ViewAddMenuItem( LBL_LOGICAL_FILES, 0,
									VAL_IS_EMPTY, 0, 0,
									0, 0xFFFFFFFF, 0, MOD_DISABLED,
									uiCol, uiRow++, MAIN_MENU_LOGICAL_FILES,
									FLM_BLACK, FLM_WHITE,
									FLM_BLUE, FLM_WHITE))
			return( 0);
	}
	return( 1);
}

/***************************************************************************
Desc: This routine executes the main menu of the VIEW program.  From here
		the user may view various parts of the database until he presses
		the ESC key.
*****************************************************************************/
FSTATIC void ViewDoMainMenu(
	void
	)
{
	FLMUINT     Option;
	VIEW_INFO   SaveView;
	FLMUINT     Repaint = 1;
	FLMUINT     BlkAddress;
	FLMUINT		Type;
	BLK_EXP     BlkExp;

	/* Loop getting commands until the ESC key is pressed */

	ViewReset( &SaveView);
	for( ;;)
	{

		/* Redisplay the main menu each time, because the other options will */
		/* have destroyed the menu. */

		if (gv_bViewPoppingStack)
		{
			if (!gv_bViewHdrRead)
			{
				ViewReadHdr();
			}
			
			ViewSearch();
		}
		
		if (Repaint)
		{
			if (!ViewSetupMainMenu())
			{
				return;
			}
		}
		
		Repaint = 1;
		Option = ViewGetMenuOption();
		
		switch( Option)
		{
			case ESCAPE_OPTION:
				return;
			case MAIN_MENU_FILE_HEADER:
				ViewFileHeader();
				break;
			case MAIN_MENU_LOG_HEADER:
				ViewLogHeader();
				break;
			case MAIN_MENU_LOGICAL_FILES:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				
				ViewLogicalFiles();
				break;
			case SEARCH_OPTION:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				
				gv_uiViewSearchLfNum = FLM_DATA_CONTAINER;
				if (ViewGetKey())
					ViewSearch();
				break;
			case GOTO_BLOCK_OPTION:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				
				if (GetBlockAddrType( &BlkAddress, &Type))
				{
					BlkExp.Type = Type;
					BlkExp.Level = 0xFF;
					BlkExp.NextAddr = 0;
					BlkExp.PrevAddr = 0;
					BlkExp.LfNum = 0;
					ViewBlocks( BlkAddress, BlkAddress, &BlkExp);
				}
				else
					Repaint = 0;
				break;
			case EDIT_OPTION:
			default:
				Repaint = 0;
				break;
		}
	}
}
