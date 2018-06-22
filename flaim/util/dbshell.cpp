//------------------------------------------------------------------------------
// Desc: Command-line environment for FLAIM utilities
//
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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

#include "dbshell.h"
#include "flm_edit.h"

// Imported global variables.

FLMBOOL			gv_bShutdown = FALSE;

FSTATIC RCODE copyStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *			pvParm2,
	void *			pvAppData);
	
FSTATIC RCODE renameStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *			pvParm2,
	void *			pvAppData);
	
FSTATIC RCODE backupStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *			pvParm2,
	void *			pvAppData);
	
FSTATIC void format64BitNum(
	FLMUINT64	ui64Num,
	char *		pszBuf,
	FLMBOOL		bOutputHex,
	FLMBOOL		bAddCommas = FALSE);

FSTATIC void removeChars(
	char *	pszString,
	char		cChar);
	
FSTATIC char * positionToPath(
	char *	pszCommandLine);
	
FSTATIC void extractBaseDirAndWildcard(
	IF_FileSystem *	pFileSystem,
	char *				pszPath, 
	char *				pszBase, 
	char *				pszWildcard);
	
// Methods

/****************************************************************************
Desc:
*****************************************************************************/
FlmShell::FlmShell()
{
	m_pScreen = NULL;
	m_pWindow = NULL;
	
	m_ArgPool.poolInit( 512);

	f_memset( &m_DbList [0], 0, sizeof( m_DbList));
	m_pTitleWin = NULL;
	m_iCurrArgC = 0;
	m_ppCurrArgV = NULL;
	m_iLastCmdExitCode = 0;
	m_bPagingEnabled = FALSE;
	f_memset( &m_ppCmdList [0], 0, sizeof( m_ppCmdList));
	f_memset( &m_ppHistory [0], 0, sizeof( m_ppHistory));
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmShell::~FlmShell()
{
	FLMUINT		uiLoop;

	m_ArgPool.poolFree();

	// Free the command objects.

	for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
	{
		if( m_ppCmdList[ uiLoop] != NULL)
		{
			m_ppCmdList[ uiLoop]->Release();
		}
	}

	// Free the history items

	for( uiLoop = 0; uiLoop < MAX_SHELL_HISTORY_ITEMS; uiLoop++)
	{
		if( m_ppHistory[ uiLoop])
		{
			f_free( &m_ppHistory[ uiLoop]);
		}
	}

	// Close all open databases

	for( uiLoop = 0; uiLoop < MAX_SHELL_OPEN_DB; uiLoop++)
	{
		if( m_DbList[ uiLoop] != HFDB_NULL)
		{
			(void)FlmDbClose( &m_DbList[ uiLoop]);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::setup( void)
{
	FlmCommand *	pCommand = NULL;
	RCODE				rc = FERR_OK;

	// Register dbopen command

	if( (pCommand = f_new FlmDbOpenCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register dbclose command

	if( (pCommand = f_new FlmDbCloseCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register dbcopy, dbrename, and dbremove command handler

	if( (pCommand = f_new FlmDbManageCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register trans command

	if( (pCommand = f_new FlmTransCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register backup command

	if( (pCommand = f_new FlmBackupCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register restore command

	if( (pCommand = f_new FlmRestoreCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register database config command

	if( (pCommand = f_new FlmDbConfigCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register database get config command

	if( (pCommand = f_new FlmDbGetConfigCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the file delete command

	if( (pCommand = f_new FlmFileSysCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

	// Register the edit command

	if( (pCommand = f_new FlmEditCommand) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = registerCmd( pCommand)))
	{
		goto Exit;
	}
	pCommand = NULL;

Exit:

	if( pCommand)
	{
		pCommand->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::registerDatabase(
	HFDB			hDb,
	FLMUINT *	puiDbId)
{
	FLMUINT		uiLoop;
	RCODE			rc = FERR_OK;

	for( uiLoop = 0; uiLoop < MAX_SHELL_OPEN_DB; uiLoop++)
	{
		if( m_DbList[ uiLoop] == HFDB_NULL)
		{
			m_DbList[ uiLoop] = hDb;
			*puiDbId = uiLoop;
			goto Exit;
		}
	}

	rc = RC_SET( FERR_TOO_MANY_OPEN_DBS);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::getDatabase(
	FLMUINT	uiDbId,
	HFDB *	phDb)
{
	RCODE			rc = FERR_OK;

	if( uiDbId >= MAX_SHELL_OPEN_DB)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	*phDb = m_DbList[ uiDbId];

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::deregisterDatabase(
	FLMUINT			uiDbId)
{
	RCODE			rc = FERR_OK;

	if( uiDbId >= MAX_SHELL_OPEN_DB)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	m_DbList[ uiDbId] = HFDB_NULL;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmShell::con_printf(
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	f_va_list	args;

	if( m_pWindow)
	{
		f_va_start( args, pszFormat);
		f_vsprintf( szBuffer, pszFormat, &args);
		f_va_end( args);
		FTXWinPrintStr( m_pWindow, szBuffer);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::parseCmdLine(
	char *		pszString)
{
	FLMUINT		uiArgCount = 0;
	FLMUINT		uiCurrToken = 0;
	FLMUINT		uiTokenLen;
	char *		pszCurrToken;
	FLMBOOL		bQuoted;
	FlmParse		Parser;
	RCODE			rc = FERR_OK;

	m_ArgPool.poolReset();
	m_iCurrArgC = 0;
	m_ppCurrArgV = NULL;
	m_pszOutputFile = NULL;

	Parser.setString( pszString);
	while( Parser.getNextToken())
	{
		uiArgCount++;
	}
	
	if( RC_BAD( rc = m_ArgPool.poolCalloc( uiArgCount * sizeof( char *),
		(void **)&m_ppCurrArgV)))
	{
		goto Exit;
	}

	uiCurrToken = 0;
	Parser.setString( pszString);
	while( (pszCurrToken = Parser.getNextToken()) != NULL)
	{
		bQuoted = FALSE;
		if( *pszCurrToken == '\"')
		{
			// Skip the quote character
			pszCurrToken++;
			bQuoted = TRUE;
		}

		uiTokenLen = f_strlen( pszCurrToken);
		if (!bQuoted && uiTokenLen >= 2 && *pszCurrToken == '>' && !m_pszOutputFile)
		{
			if( RC_BAD( rc = m_ArgPool.poolCalloc( uiTokenLen, 
				(void **)&m_pszOutputFile)))
			{
				goto Exit;
			}
			
			f_strcpy( m_pszOutputFile, pszCurrToken + 1);
		}
		else
		{
			if( RC_BAD( rc = m_ArgPool.poolCalloc( uiTokenLen + 1,
				(void **)&m_ppCurrArgV [uiCurrToken])))
			{
				goto Exit;
			}
			
			f_strcpy( m_ppCurrArgV[ uiCurrToken], pszCurrToken);

			if( bQuoted)
			{
				// Strip off the trailing quote
				m_ppCurrArgV[ uiCurrToken][ uiTokenLen - 1] = '\0';
			}
			uiCurrToken++;
		}
	}

	m_iCurrArgC = (FLMINT)uiCurrToken;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::executeCmdLine( void)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bValidCommand = FALSE;
	FLMUINT				uiLoop;

	if( !m_iCurrArgC)
	{
		goto Exit;
	}

	// Process internal commands

	if( f_stricmp( m_ppCurrArgV[ 0], "cls") == 0)
	{
		FTXWinClear( m_pWindow);
		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "exit") == 0)
	{
		gv_bShutdown = TRUE;
		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "echo") == 0)
	{
		FLMBOOL		bNewline = FALSE;

		if( m_iCurrArgC > 1 &&
			f_stricmp( m_ppCurrArgV[ 1], "-n") == 0)
		{
			bNewline = TRUE;
			uiLoop = 2;
		}
		else
		{
			uiLoop = 1;
		}

		for( ; uiLoop < (FLMUINT)m_iCurrArgC; uiLoop++)
		{
			con_printf( "%s", (char *)m_ppCurrArgV[ uiLoop]);
		}

		if( bNewline)
		{
			con_printf( "\n");
		}

		bValidCommand = TRUE;
	}
	else if( f_stricmp( m_ppCurrArgV[ 0], "help") == 0 ||
				f_stricmp( m_ppCurrArgV[ 0], "?") == 0 ||
				f_stricmp( m_ppCurrArgV[ 0], "h") == 0)
	{
		if( m_iCurrArgC < 2)
		{
			con_printf( "Commands:\n");
			displayCommand( "help, ?, h", "Show help");
			displayCommand( "echo", "Echo typed in command");
			displayCommand( "cls", "Clear screen");
			displayCommand( "exit", "Exit shell");
			for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
			{
				if( m_ppCmdList[ uiLoop] != NULL)
				{
					m_ppCmdList[ uiLoop]->displayHelp( this, NULL);
				}
			}
		}
		else
		{
			for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
			{
				if( m_ppCmdList[ uiLoop] != NULL)
				{
					if (m_ppCmdList[ uiLoop]->canPerformCommand(
								(char *)m_ppCurrArgV [1]))
					{
						m_ppCmdList[ uiLoop]->displayHelp( this,
								(char *)m_ppCurrArgV [1]);
						break;
					}
				}
			}
		}
		bValidCommand = TRUE;
	}
	else
	{
		for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
		{
			if( m_ppCmdList[ uiLoop] != NULL)
			{
				if( m_ppCmdList[ uiLoop]->canPerformCommand(
										(char *)m_ppCurrArgV[ 0]))
				{
					m_ppCmdList[ uiLoop]->execute( m_iCurrArgC, m_ppCurrArgV, this);
					bValidCommand = TRUE;
					break;
				}
			}
		}
	}

	if( !bValidCommand)
	{
		FTXWinPrintf( m_pWindow, "Unrecognized command: %s\n", m_ppCurrArgV[ 0]);
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::registerCmd(
	FlmCommand *	pCmd)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bRegistered = FALSE;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop < MAX_REGISTERED_COMMANDS; uiLoop++)
	{
		if( m_ppCmdList[ uiLoop] == NULL)
		{
			m_ppCmdList[ uiLoop] = pCmd;
			bRegistered = TRUE;
			break;
		}
	}

	if( !bRegistered)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::addCmdHistory(
	char *	pszCmd)
{
	FLMUINT		uiLoop;
	FLMUINT		uiSlot;
	FLMUINT		uiCmdLen;
	RCODE			rc = FERR_OK;

	// If the command line is too long, don't store it in the
	// history buffer

	if( (uiCmdLen = f_strlen( pszCmd)) > MAX_CMD_LINE_LEN)
	{
		goto Exit;
	}

	// Look for a duplicate history item

	for( uiLoop = 0; uiLoop < MAX_SHELL_HISTORY_ITEMS; uiLoop++)
	{
		if( m_ppHistory[ uiLoop] &&
			f_strcmp( pszCmd, m_ppHistory[ uiLoop]) == 0)
		{
			// Remove the command from the history list and compress
			// the history table

			f_free( &m_ppHistory[ uiLoop]);

			if( uiLoop < MAX_SHELL_HISTORY_ITEMS - 1)
			{
				f_memmove( &m_ppHistory[ uiLoop], &m_ppHistory[ uiLoop + 1],
					sizeof( char *) * (MAX_SHELL_HISTORY_ITEMS - uiLoop - 1));
				m_ppHistory[ MAX_SHELL_HISTORY_ITEMS - 1] = NULL;
				break;
			}
		}
	}

	// Find an empty slot for the new history item

	for( uiSlot = MAX_SHELL_HISTORY_ITEMS; uiSlot > 0; uiSlot--)
	{
		if( m_ppHistory[ uiSlot - 1])
		{
			break;
		}
	}

	if( uiSlot == MAX_SHELL_HISTORY_ITEMS)
	{
		f_free( &m_ppHistory[ 0]);
		f_memmove( &m_ppHistory[ 0], &m_ppHistory[ 1],
			sizeof( char *) * (MAX_SHELL_HISTORY_ITEMS - 1));
		m_ppHistory[ MAX_SHELL_HISTORY_ITEMS - 1] = NULL;
		uiSlot = MAX_SHELL_HISTORY_ITEMS - 1;
	}

	if( RC_BAD( rc = f_alloc( uiCmdLen + 1, &m_ppHistory[ uiSlot])))
	{
		goto Exit;
	}

	f_strcpy( m_ppHistory[ uiSlot], pszCmd);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmShell::execute( void)
{
	char						szBuffer[ MAX_CMD_LINE_LEN + 1];
	char						szThreadName[ MAX_THREAD_NAME_LEN + 1];
	FLMUINT					uiTermChar;
	FLMUINT					uiRow;
	FLMUINT					uiLastHistorySlot = MAX_SHELL_HISTORY_ITEMS;
	RCODE						rc = FERR_OK;
	char						szDir [F_PATH_MAX_SIZE];
	DirectoryIterator		directoryIterator;
	char *					pszTabCompleteBegin = NULL;
	IF_FileSystem *		pFileSystem = NULL;

	if( RC_BAD( rc = FTXScreenInit( "dbshell main", &m_pScreen)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXScreenInitStandardWindows( m_pScreen, 
		FLM_RED, FLM_WHITE, FLM_BLUE, FLM_WHITE, FALSE, FALSE,
		NULL, &m_pTitleWin, &m_pWindow)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXScreenDisplay( m_pScreen)))
	{
		goto Exit;
	}

	FTXScreenSetShutdownFlag( m_pScreen, &gv_bShutdown);

	szBuffer[ 0] = '\0';
	for( ;;)
	{
		// Refresh the title bar
		
		f_strcpy( szThreadName, "Flaim Database Shell");
		FTXWinSetCursorPos( m_pTitleWin, 0, 0);
		FTXWinPrintf( m_pTitleWin, "%s", szThreadName);
		FTXWinClearToEOL( m_pTitleWin);

		// Check for shutdown
		if( gv_bShutdown)
		{
			break;
		}

		FTXWinGetCursorPos( m_pWindow, NULL, &uiRow);
		FTXWinSetCursorPos( m_pWindow, 0, uiRow);
		FTXWinClearToEOL( m_pWindow);

		if( RC_BAD( f_getcwd( szDir)))
		{
			szDir [0] = '\0';
		}

		FTXWinPrintf( m_pWindow, "%s>", szDir);

		if( FTXLineEdit( m_pWindow, szBuffer,
			MAX_CMD_LINE_LEN, 255, 0, &uiTermChar))
		{
			break;
		}

		if( uiTermChar == FKB_TAB)
		{
			char	szBase[ 255];
			char	szWildcard[ 255];

			szWildcard[0] = '\0';

			pszTabCompleteBegin = positionToPath( szBuffer);

			if ( f_strchr( pszTabCompleteBegin, '\"'))
			{
				// remove quotes
				removeChars( pszTabCompleteBegin, '\"');
			}

			// If we have not initialized our iterator to scan this directory
			// or if the command-line does not contain a path that we provided
			// we need to reinitialize the iterator.

			if( !directoryIterator.isInitialized() || 
				 !pszTabCompleteBegin || 
				 !directoryIterator.isInSet( pszTabCompleteBegin))
			{

				extractBaseDirAndWildcard( pFileSystem, pszTabCompleteBegin,
					szBase, szWildcard);

				directoryIterator.reset();
				directoryIterator.setupForSearch( pFileSystem,
					szDir, szBase, szWildcard);
			}

			if ( !directoryIterator.isEmpty())
			{
				// Copy in the next entry along with its full path.

				directoryIterator.next( pszTabCompleteBegin, TRUE);

			}
			else
			{
				FTXBeep();
			}

			// If the completed path contains spaces, quote it
			if ( f_strchr( pszTabCompleteBegin, ASCII_SPACE))
			{
				f_memmove( pszTabCompleteBegin + 1, pszTabCompleteBegin, 
					f_strlen( pszTabCompleteBegin) + 1);
				pszTabCompleteBegin[0] = '\"';

				f_strcat( pszTabCompleteBegin, "\"");
			}
			continue;
		}

		directoryIterator.reset();

		if( uiTermChar == FKB_UP)
		{
			for(; uiLastHistorySlot > 0; uiLastHistorySlot--)
			{
				if( m_ppHistory[ uiLastHistorySlot - 1])
				{
					f_strcpy( szBuffer, m_ppHistory[ uiLastHistorySlot - 1]);
					uiLastHistorySlot--;
					break;
				}
			}

			continue;
		}

		if( uiTermChar == FKB_DOWN)
		{
			for(; uiLastHistorySlot < MAX_SHELL_HISTORY_ITEMS - 1; uiLastHistorySlot++)
			{
				if( m_ppHistory[ uiLastHistorySlot + 1])
				{
					f_strcpy( szBuffer, m_ppHistory[ uiLastHistorySlot + 1]);
					uiLastHistorySlot++;
					break;
				}
			}
			continue;
		}

		if( uiTermChar == FKB_ESCAPE)
		{
			szBuffer[ 0] = '\0';
			continue;
		}

		uiLastHistorySlot = MAX_SHELL_HISTORY_ITEMS;

		if( szBuffer [0])
		{
			FTXWinPrintf( m_pWindow, "\n");
			addCmdHistory( szBuffer);
			parseCmdLine( szBuffer);
			executeCmdLine();
			szBuffer[0] = '\0';

			continue;
		}

		FTXWinPrintf( m_pWindow, "\n");
	}

Exit:

	if( m_pWindow)
	{
		FTXWinFree( &m_pWindow);
	}

	if( m_pScreen)
	{
		FTXScreenFree( &m_pScreen);
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmParse::FlmParse( void)
{
	m_szString [0] = 0;
	m_pszCurPos = &m_szString [0];
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmParse::setString(
	char *		pszString)
{
	if( pszString)
	{
		f_strcpy( m_szString, pszString);
	}
	else
	{
		m_szString  [0]= 0;
	}

	m_pszCurPos = &m_szString [0];
}

/****************************************************************************
Desc:
*****************************************************************************/
char * FlmParse::getNextToken( void)
{
	char *		pszTokenPos = &m_szToken [0];
	FLMBOOL		bQuoted = FALSE;

	while( *m_pszCurPos && *m_pszCurPos == ' ')
	{
		m_pszCurPos++;
	}

	if( *m_pszCurPos == '$')
	{
		*pszTokenPos++ = *m_pszCurPos++;
		while( *m_pszCurPos)
		{
			if( (*m_pszCurPos >= 'A' && *m_pszCurPos <= 'Z') ||
				(*m_pszCurPos >= 'a' && *m_pszCurPos <= 'z') ||
				(*m_pszCurPos >= '0' && *m_pszCurPos <= '9') ||
				(*m_pszCurPos == '_'))
			{
				*pszTokenPos++ = *m_pszCurPos++;
			}
			else
			{
				break;
			}
		}
	}
	else if( *m_pszCurPos == '=')
	{
		*pszTokenPos++ = *m_pszCurPos++;
	}
	else
	{
		while( *m_pszCurPos && (*m_pszCurPos != ' ' || bQuoted))
		{
			if( *m_pszCurPos == '\"')
			{
				*pszTokenPos++ = *m_pszCurPos++;
				if( bQuoted)
				{
					break;
				}
				else
				{
					bQuoted = TRUE;
				}
			}
			else
			{
				*pszTokenPos++ = *m_pszCurPos++;
			}
		}
	}

	*pszTokenPos = '\0';

	if( m_szToken [0] == 0)
	{
		return( NULL);
	}

	return( &m_szToken [0]);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbOpenCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	HFDB				hDb = HFDB_NULL;
	FLMUINT			uiDbId;
	RCODE				rc = FERR_OK;
	char *			pszRflDir = NULL;
	char *			pszPassword = NULL;
	char *			pszAllowLimited;
	FLMUINT			uiOpenFlags = 0;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( iArgC >= 3)
	{
		pszRflDir = ppszArgV[ 2];
	}
	
	if (iArgC >=4)
	{
		pszPassword = ppszArgV[ 3];
	}

	if (iArgC >=5)
	{
		pszAllowLimited = ppszArgV[ 4];
		
		if (f_strnicmp( pszAllowLimited, "TRUE", 4) == 0)
		{
			uiOpenFlags |= FO_ALLOW_LIMITED;
		}
	}

	if( RC_BAD( rc = FlmDbOpen( ppszArgV[ 1],
		NULL, pszRflDir, uiOpenFlags, pszPassword, &hDb)))
	{
		if( rc != FERR_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}

		if( RC_BAD( rc = FlmDbCreate( 
			ppszArgV[ 1], NULL, pszRflDir, NULL, NULL, NULL, &hDb)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pShell->registerDatabase( hDb, &uiDbId)))
	{
		goto Exit;
	}
	hDb = HFDB_NULL;

	pShell->con_printf( "Database #%u opened.\n", (unsigned)uiDbId);

Exit:

	if( hDb != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb);
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "Error opening database: %e\n", rc);
		iExitCode = -1;
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbOpenCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbopen", "Open a database");
	}
	else
	{
		pShell->con_printf("Usage:\n"
								 "  dbopen <DbFileName> [<RflPath> [<Password> [<AllowLimited>]]]\n");
		pShell->con_printf("  <AllowLimited> : TRUE | FALSE \n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbOpenCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbopen", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbCloseCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	FLMUINT			uiDbId;
	HFDB				hDb;
	RCODE				rc = FERR_OK;

	if( iArgC != 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( f_stricmp( ppszArgV[ 1], "kill") == 0)
	{
		(void)FlmConfig( FLM_KILL_DB_HANDLES, NULL, NULL);
		pShell->con_printf( "All handles killed, but not necessarily closed.\n");
	}
	else if( f_stricmp( ppszArgV[ 1], "all") == 0)
	{
		for( uiDbId = 0; uiDbId < MAX_SHELL_OPEN_DB; uiDbId++)
		{
			if( RC_BAD( rc = pShell->getDatabase( uiDbId, &hDb)))
			{
				goto Exit;
			}

			if( hDb != HFDB_NULL)
			{
				if( RC_BAD( rc = pShell->deregisterDatabase( uiDbId)))
				{
					goto Exit;
				}
				(void)FlmDbClose( &hDb);
				pShell->con_printf( "Database #%u closed.\n", (unsigned)uiDbId);
			}
		}

		(void)FlmConfig( FLM_CLOSE_UNUSED_FILES, (void *)0, NULL);
	}
	else
	{
		uiDbId = f_atol( ppszArgV[ 1]);
		if( RC_BAD( rc = pShell->getDatabase( uiDbId, &hDb)))
		{
			goto Exit;
		}

		if( hDb != HFDB_NULL)
		{
			if( RC_BAD( rc = pShell->deregisterDatabase( uiDbId)))
			{
				goto Exit;
			}
			(void)FlmDbClose( &hDb);
			pShell->con_printf( "Database #%u closed.\n", (unsigned)uiDbId);
		}
		else
		{
			pShell->con_printf( "Database #%u already closed.\n", (unsigned)uiDbId);
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "Error closing database: %e\n", rc);
		iExitCode = -1;
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbCloseCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbclose", "Close a database");
	}
	else
	{
		pShell->con_printf("Usage:\n"
								 "  dbclose <db# | ALL>\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbCloseCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbclose", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmTransCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	FLMUINT			uiDbId;
	FLMUINT			uiTimeout;
	#define FLM_NO_TRANS				0
	#define FLM_UPDATE_TRANS		1
	#define FLM_READ_TRANS			2
	FLMUINT			uiTransType;
	HFDB				hDb;
	RCODE				rc = FERR_OK;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle

	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &hDb)) || hDb == HFDB_NULL)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}
	
	(void)FlmDbGetTransType( hDb, &uiTransType);
	if( f_stricmp( ppszArgV [0], "trbegin") == 0)
	{
		if( iArgC < 3)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( uiTransType != FLM_NO_TRANS)
		{
			pShell->con_printf( "%s transaction is already active on database %u.\n",
				(char *)(uiTransType == FLM_READ_TRANS
							? "A read"
							: "An update"), (unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( !f_stricmp( ppszArgV[ 2], "read"))
		{
			if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_READ_TRANS,
												FLM_NO_TIMEOUT, NULL)))
			{
				goto Exit;
			}
		}
		else if( !f_stricmp( ppszArgV[ 2], "update"))
		{
			if( iArgC > 4)
			{
				uiTimeout = f_atol( ppszArgV[ 3]);
			}
			else
			{
				uiTimeout = FLM_NO_TIMEOUT;
			}

			if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS,
												uiTimeout, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			pShell->con_printf( "Invalid parameter: %s\n", ppszArgV[ 3]);
			iExitCode = -1;
			goto Exit;
		}

		pShell->con_printf( "Transaction on %u started.\n", (unsigned)uiDbId);
	}
	else if( f_stricmp( ppszArgV[ 0], "trcommit") == 0)
	{
		if( uiTransType == FLM_NO_TRANS)
		{
			pShell->con_printf( "There is no active transaction on database %u.\n",
				(unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = FlmDbTransCommit( hDb)))
		{
			goto Exit;
		}
		pShell->con_printf( "Transaction committed on database %u.\n",
			(unsigned)uiDbId);
	}
	else if( f_stricmp( ppszArgV[ 0], "trabort") == 0)
	{
		if( uiTransType == FLM_NO_TRANS)
		{
			pShell->con_printf( "There is no active transaction on database %u.\n",
				(unsigned)uiDbId);
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = FlmDbTransAbort( hDb)))
		{
			goto Exit;
		}
		pShell->con_printf( "Transaction aborted on database %u.\n",
			(unsigned)uiDbId);
	}
	else
	{
		// should never be able to get here!
		flmAssert( 0);
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmTransCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "trbegin", "Begin a transaction");
		pShell->displayCommand( "trcommit", "Commit a transaction");
		pShell->displayCommand( "trabort", "Abort a transaction");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( pszCommand, "trbegin") == 0)
		{
			pShell->con_printf( "  trbegin db# [read | update <timeout>]\n");
		}
		else
		{
			pShell->con_printf( "  %s db#\n", pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmTransCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "trbegin", pszCommand) == 0 ||
				f_stricmp( "trcommit", pszCommand) == 0 ||
				f_stricmp( "trabort", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: Status function for reporting progress of database copy
*****************************************************************************/
FSTATIC RCODE copyStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *,			// pvParm2,
	void *			pvAppData)
{
	RCODE	rc = FERR_OK;
	
	if (eStatus == FLM_DB_COPY_STATUS)
	{
		DB_COPY_INFO *		pCopyInfo = (DB_COPY_INFO *)pvParm1;
		FlmShell *			pShell = (FlmShell *)pvAppData;
		FTX_WINDOW *		pWin = pShell->getWindow();
		
		if (pCopyInfo->bNewSrcFile)
		{
			FTXWinPrintf( pWin, "\nCopying %s to %s ...\n",
					pCopyInfo->szSrcFileName, pCopyInfo->szDestFileName);
		}

		if( gv_bShutdown)
		{
			rc = RC_SET( FERR_USER_ABORT);
			goto Exit;
		}

		FTXWinPrintf( pWin, "  %,I64u of %,I64u bytes copied\r",
						pCopyInfo->ui64BytesCopied, pCopyInfo->ui64BytesToCopy);

		if( RC_OK( FTXWinTestKB( pWin)))
		{
			FLMUINT	uiChar;

			FTXWinInputChar( pWin, &uiChar);
			if (uiChar == FKB_ESC)
			{
				rc = RC_SET( FERR_USER_ABORT);
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Status function for reporting progress of database rename
*****************************************************************************/
FSTATIC RCODE renameStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *,			// pvParm2,
	void *			pvAppData)
{
	RCODE	rc = FERR_OK;
	
	if (eStatus == FLM_DB_RENAME_STATUS)
	{
		DB_RENAME_INFO *	pRenameInfo = (DB_RENAME_INFO *)pvParm1;
		FlmShell *			pShell = (FlmShell *)pvAppData;
		
		pShell->con_printf( "Renaming %s to %s ...\n",
				pRenameInfo->szSrcFileName,
				pRenameInfo->szDstFileName);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbManageCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT			iExitCode = 0;
	RCODE				rc = FERR_OK;

	if( iArgC < 2)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if (f_stricmp( ppszArgV [0], "dbremove") == 0)
	{
		if (RC_BAD( rc = FlmDbRemove( ppszArgV[ 1], NULL, NULL, TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		if( iArgC < 3)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}
		if (f_stricmp( ppszArgV [0], "dbcopy") == 0)
		{
			if (RC_BAD( rc = FlmDbCopy( ppszArgV [1], NULL, NULL,
										ppszArgV [2], NULL, NULL, copyStatusFunc,
										pShell)))
			{
				goto Exit;
			}
			pShell->con_printf( "\n\n");
		}
		else
		{
			if (RC_BAD( rc = FlmDbRename( ppszArgV [1], NULL, NULL,
										ppszArgV [2], TRUE, renameStatusFunc, pShell)))
			{
				goto Exit;
			}
			pShell->con_printf( "\n\n");
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbManageCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbcopy", "Copy a database");
		pShell->displayCommand( "dbrename", "Rename a database");
		pShell->displayCommand( "dbremove", "Delete a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( pszCommand, "dbremove") == 0)
		{
			pShell->con_printf( "  dbremove <DbFileName>\n");
		}
		else
		{
			pShell->con_printf( "  %s <SrcDbFileName> <DestDbFileName>\n",
				pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbManageCommand::canPerformCommand(
	char *		pszCommand )
{
	return( (f_stricmp( "dbcopy", pszCommand) == 0 ||
				f_stricmp( "dbrename", pszCommand) == 0 ||
				f_stricmp( "dbremove", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE backupStatusFunc(
	eStatusType		eStatus,
	void *			pvParm1,
	void *,			// pvParm2,
	void *			pvAppData)
{
	RCODE	rc = FERR_OK;
	
	if (eStatus == FLM_DB_BACKUP_STATUS)
	{
		DB_BACKUP_INFO *	pBackupInfo = (DB_BACKUP_INFO *)pvParm1;
		FlmShell *			pShell = (FlmShell *)pvAppData;
		FTX_WINDOW *		pWin = pShell->getWindow();

		if (gv_bShutdown)
		{
			rc = RC_SET( FERR_USER_ABORT);
			goto Exit;
		}

		FTXWinPrintf( pWin, "%,I64u / %,I64u bytes backed up\r",
						  pBackupInfo->ui64BytesDone, pBackupInfo->ui64BytesToDo);

		if( RC_OK( FTXWinTestKB( pWin)))
		{
			FLMUINT	uiChar;
	
			FTXWinInputChar( pWin, &uiChar);
			if (uiChar == FKB_ESC)
			{
				rc = RC_SET( FERR_USER_ABORT);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmBackupCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT					uiDbId;
	FLMUINT					uiIncSeqNum;
	HFDB						hDb;
	HFBACKUP					hBackup = HFBACKUP_NULL;
	FLMINT					iExitCode = 0;
	FBackupType				eBackupType = FLM_FULL_BACKUP;
	RCODE						rc = FERR_OK;
	FLMBOOL					bUsePasswd = FALSE;

	if( iArgC < 3)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}
	
	if (iArgC > 3)
	{
		bUsePasswd = TRUE;
	}
	
	if( iArgC > 4)
	{
		if( f_strnicmp( ppszArgV[ 3], "inc", 3) == 0)
		{
			eBackupType = FLM_INCREMENTAL_BACKUP;
		}
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &hDb)) || hDb == HFDB_NULL)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbBackupBegin( hDb, eBackupType, TRUE, &hBackup)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbBackup( hBackup, ppszArgV [2], 
								(const char *)(bUsePasswd ? ppszArgV[3] : NULL),
								NULL, backupStatusFunc, pShell, &uiIncSeqNum)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmDbBackupEnd( &hBackup)))
	{
		goto Exit;
	}

	pShell->con_printf( "\nBackup complete.\n");

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	if( hBackup != HFBACKUP_NULL)
	{
		(void)FlmDbBackupEnd( &hBackup);
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmBackupCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbbackup", "Backup a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <database_path> <backup_name> [<password> [\"INC\"]]\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmBackupCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbbackup", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
class F_LocalRestore : public F_FSRestore
{
public:

	F_LocalRestore(
		FlmShell *		pShell)
	{
		m_pShell = pShell;
		m_pWin = pShell->getWindow();
		m_bFirstStatus = TRUE;
		m_uiTransCount = 0;
		m_uiAddCount = 0;
		m_uiDeleteCount = 0;
		m_uiModifyCount = 0;
		m_uiReserveCount = 0;
		m_uiIndexCount = 0;
		m_uiRflFileNum = 0;
		m_ui64RflBytesRead = 0;
	}

	virtual ~F_LocalRestore()
	{
	}
	
	FINLINE RCODE setup(
		char *			pszDbPath,
		char *			pszBackupSetPath,
		char *			pszRflDir)
	{
		return( F_FSRestore::setup( pszDbPath,
			pszBackupSetPath, pszRflDir));
	}

	FINLINE RCODE openBackupSet( void)
	{
		return( F_FSRestore::openBackupSet());
	}

	FINLINE RCODE openIncFile(
		FLMUINT	uiFileNum)
	{
		return( F_FSRestore::openIncFile( uiFileNum));
	}

	FINLINE RCODE openRflFile(
		FLMUINT	uiFileNum)
	{
		RCODE	rc;
		
		m_uiRflFileNum = uiFileNum;
		if (RC_OK( rc = updateCountDisplay()))
		{
			rc = F_FSRestore::openRflFile( uiFileNum);
		}
		return( rc);
	}

	FINLINE RCODE read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead)
	{
		RCODE	rc;
		
		if (RC_OK( rc =F_FSRestore::read( uiLength, pvBuffer, puiBytesRead)))
		{
			if (m_uiRflFileNum)
			{
				m_ui64RflBytesRead += (*puiBytesRead);
			}
		}
		return( rc);
	}

	FINLINE RCODE close( void)
	{
		return( F_FSRestore::close());
	}

	FINLINE RCODE abortFile( void)
	{
		return( F_FSRestore::abortFile());
	}

	FINLINE RCODE processUnknown(
		F_UnknownStream *		pUnkStrm)
	{
		return( F_FSRestore::processUnknown( pUnkStrm));
	}

	RCODE status(
		eRestoreStatusType	eStatusType,
		FLMUINT					uiTransId,
		void *					pvValue1,
		void *					pvValue2,
		void *					pvValue3,
		eRestoreActionType *	peRestoreAction);

private:

	RCODE report_preamble( void);

	RCODE report_postamble( void);

	RCODE updateCountDisplay( void);

	RCODE reportProgress(
		BYTE_PROGRESS *	pProgress);

	RCODE reportError(
		eRestoreActionType *	peRestoreAction,
		RCODE						rcErr);

	RCODE reportBeginTrans(
		FLMUINT	uiTransId);

	RCODE reportEndTrans(
		const char *	pszAction,
		FLMUINT			uiTransId);

	FlmShell *			m_pShell;
	FTX_WINDOW *		m_pWin;
	FLMBOOL				m_bFirstStatus;
	FLMUINT				m_uiTransCount;
	FLMUINT				m_uiAddCount;
	FLMUINT				m_uiDeleteCount;
	FLMUINT				m_uiModifyCount;
	FLMUINT				m_uiReserveCount;
	FLMUINT				m_uiIndexCount;
	FLMUINT				m_uiRflFileNum;
	FLMUINT64			m_ui64RflBytesRead;

};

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::report_preamble( void)
{
	RCODE	rc = FERR_OK;

	if( gv_bShutdown)
	{
		rc = RC_SET( FERR_USER_ABORT);
		goto Exit;
	}

	if( m_bFirstStatus)
	{
		FTXWinClear( m_pWin);
		m_bFirstStatus = FALSE;
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::report_postamble( void)
{
	RCODE	rc = FERR_OK;

	FTXWinSetCursorPos( m_pWin, 0, 5);

	f_yieldCPU();

	if( RC_OK( FTXWinTestKB( m_pWin)))
	{
		FLMUINT	uiChar;

		FTXWinInputChar( m_pWin, &uiChar);
		if (uiChar == FKB_ESC)
		{
			rc = RC_SET( FERR_USER_ABORT);
			goto Exit;
		}
	}
	
Exit:

	return rc;

}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::updateCountDisplay( void)
{
	RCODE	rc = FERR_OK;

	if (m_pWin)
	{
		if (RC_BAD(rc = report_preamble()))
		{
			goto Exit;
		}
		FTXWinSetCursorPos( m_pWin, 0, 2);
		FTXWinPrintf( m_pWin,
			"RFLFile#: %-10u   TotalCnt: %-10u  RflKBytes: %uK\n"
			"AddCnt:   %-10u   DelCnt:   %-10u  ModCnt:    %u\n"
			"TrCnt:    %-10u   RsrvCnt:  %-10u  IxSetCnt:  %u",
			m_uiRflFileNum,
			m_uiTransCount + m_uiAddCount + m_uiDeleteCount +
			m_uiModifyCount + m_uiReserveCount + m_uiIndexCount,
			(unsigned)(m_ui64RflBytesRead / 1024),
			m_uiAddCount, m_uiDeleteCount, m_uiModifyCount,
			m_uiTransCount, m_uiReserveCount, m_uiIndexCount);
		if (RC_BAD(rc = report_postamble()))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::reportProgress(
	BYTE_PROGRESS *	pProgress)
{
	RCODE	rc = FERR_OK;

	if (RC_BAD(rc = report_preamble()))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( m_pWin, 0, 1);
	FTXWinPrintf( m_pWin, "%,I64u / %,I64u bytes restored", pProgress->ui64BytesDone,
		pProgress->ui64BytesToDo);
	FTXWinClearToEOL( m_pWin);

	if (RC_BAD(rc = report_postamble()))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::reportError(
	eRestoreActionType *	peAction,
	RCODE						rcErr)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiChar;

	*peAction = RESTORE_ACTION_CONTINUE;

	if (RC_BAD(rc = report_preamble()))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( m_pWin, 0, 6);
	FTXWinClearToEOL( m_pWin);
	FTXWinPrintf( m_pWin, "Error: %s.  Retry (Y/N): ",
		FlmErrorString( rcErr));
		
	if( RC_BAD( FTXWinInputChar( m_pWin, &uiChar)))
	{
		uiChar = 0;
		goto Exit;
	}

	if( uiChar == 'Y' || uiChar == 'y')
	{
		*peAction = RESTORE_ACTION_RETRY;
	}

	FTXWinClearToEOL( m_pWin);

	if (RC_BAD(rc = report_postamble()))
	{
		goto Exit;
	}

Exit:

	return rc;

}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::reportBeginTrans(
	FLMUINT	uiTransId)
{
	RCODE	rc = FERR_OK;

	if (RC_BAD(rc = report_preamble()))
	{
		goto Exit;
	}

	FTXWinSetCursorPos( m_pWin, 0, 5);
	FTXWinPrintf( m_pWin, "BEGIN_TRANS: ID = 0x%X", uiTransId);
	FTXWinClearToEOL( m_pWin);

	if (RC_BAD(rc = report_postamble()))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::reportEndTrans(
	const char *	pszAction,
	FLMUINT			uiTransId)
{
	if( m_bFirstStatus)
	{
		FTXWinClear( m_pWin);
		m_bFirstStatus = FALSE;
	}
	FTXWinSetCursorPos( m_pWin, 0, 5);
	FTXWinPrintf( m_pWin, "%s: ID = 0x%X", pszAction, uiTransId);
	FTXWinClearToEOL( m_pWin);
	m_uiTransCount++;
	return( updateCountDisplay());
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_LocalRestore::status(
	eRestoreStatusType	eStatusType,
	FLMUINT					uiTransId,
	void *					pvValue1,
	void *,					// pvValue2,
	void *,					// pvValue3,
	eRestoreActionType *	peRestoreAction)
{
	RCODE	rc = FERR_OK;
	
	*peRestoreAction = RESTORE_ACTION_CONTINUE;
	switch (eStatusType)
	{
		case RESTORE_BEGIN_TRANS:
			rc = reportBeginTrans( uiTransId);
			break;
		case RESTORE_COMMIT_TRANS:
			rc = reportEndTrans( "COMMIT_TRANS", uiTransId);
			break;
		case RESTORE_ABORT_TRANS:
			rc = reportEndTrans( "ABORT_TRANS", uiTransId);
			break;
		case RESTORE_PROGRESS:
			rc = reportProgress( (BYTE_PROGRESS *)pvValue1);
			break;
		case RESTORE_WRAP_KEY:
			rc = reportEndTrans( "WRAP_KEY", uiTransId);
			break;
		case RESTORE_ENABLE_ENCRYPTION:
			rc = reportEndTrans( "ENABLE_ENCRYPTION", uiTransId);
			break;
		case RESTORE_ADD_REC:
			m_uiAddCount++;
			rc = updateCountDisplay();
			break;
		case RESTORE_DEL_REC:
			m_uiDeleteCount++;
			rc = updateCountDisplay();
			break;
		case RESTORE_MOD_REC:
			m_uiModifyCount++;
			rc = updateCountDisplay();
			break;
		case RESTORE_RESERVE_DRN:
			m_uiReserveCount++;
			rc = updateCountDisplay();
			break;
		case RESTORE_INDEX_SET:
			m_uiIndexCount++;
			rc = updateCountDisplay();
			break;
		case RESTORE_ERROR:
			rc = reportError( peRestoreAction, (RCODE)((FLMUINT)pvValue1));
			break;
		case RESTORE_REDUCE:
			rc = reportEndTrans( "REDUCE", uiTransId);
			break;
		case RESTORE_UPGRADE:
			rc = reportEndTrans( "UPGRADE_DB", uiTransId);
			break;
		case RESTORE_INDEX_SUSPEND:
		case RESTORE_INDEX_RESUME:
		case RESTORE_BLK_CHAIN_DELETE:
		default:
			break;
	}
	
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmRestoreCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	char *			pszRflDir = NULL;
	FLMINT			iExitCode = 0;
	F_LocalRestore	restoreObj( pShell);
	RCODE				rc = FERR_OK;
	FLMBOOL			bUsePasswd = FALSE;

	if( iArgC < 3)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}
	if( iArgC > 3)
	{
		bUsePasswd = TRUE;
	}
	if( iArgC > 4)
	{
		pszRflDir = ppszArgV[ 4];
	}

	if( RC_BAD( rc = FlmDbRestore( ppszArgV [1], NULL, ppszArgV [2], pszRflDir,
						 	(const char *)(bUsePasswd ? ppszArgV [3] : NULL),
							&restoreObj)))
	{
		goto Exit;
	}

	pShell->con_printf( "\nRestore complete.\n");

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmRestoreCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbrestore", "Restore a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <RestoreToDbName> <BackupPath> [<password> [<RFL Dir>]]\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmRestoreCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbrestore", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbConfigCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT			uiDbId;
	HFDB				hDb;
	FLMINT			iExitCode = 0;
	RCODE				rc = FERR_OK;

	if( iArgC < 3)
	{
		pShell->con_printf( "Too few parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &hDb)) || hDb == HFDB_NULL)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}
	if( f_stricmp( ppszArgV[ 2], "rflkeepfiles") == 0)
	{
		FLMBOOL	bEnable;

		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if (f_stricmp( ppszArgV[ 3], "on") == 0)
		{
			bEnable = TRUE;
		}
		else if (f_stricmp( ppszArgV[ 3], "off") == 0)
		{
			bEnable = FALSE;
		}
		else
		{
			pShell->con_printf( "Invalid value, must be 'on' or 'off'.\n");
			iExitCode = -1;
			goto Exit;
		}
		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_RFL_KEEP_FILES, (void *)((FLMUINT)bEnable), NULL)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rfldir") == 0)
	{
		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_RFL_DIR, ppszArgV[ 3], NULL)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rflfilelimits") == 0)
	{
		FLMUINT	uiRflMinSize;
		FLMUINT	uiRflMaxSize;

		if( iArgC < 5)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		uiRflMinSize = f_atol( ppszArgV[ 3]);
		uiRflMaxSize = f_atol( ppszArgV[ 4]);
		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_RFL_FILE_LIMITS, (void *)uiRflMinSize,
									(void *)uiRflMaxSize)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rflrolltonextfile") == 0)
	{
		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_RFL_ROLL_TO_NEXT_FILE, NULL, NULL)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "rflkeepabortedtrans") == 0)
	{
		FLMBOOL	bEnable;

		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if (f_stricmp( ppszArgV[ 3], "on") == 0)
		{
			bEnable = TRUE;
		}
		else if (f_stricmp( ppszArgV[ 3], "off") == 0)
		{
			bEnable = FALSE;
		}
		else
		{
			pShell->con_printf( "Invalid value, must be 'on' or 'off'.\n");
			iExitCode = -1;
			goto Exit;
		}
		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_KEEP_ABORTED_TRANS_IN_RFL, (void *)((FLMUINT)bEnable), NULL)))
		{
			goto Exit;
		}
	}
	else if( f_stricmp( ppszArgV[ 2], "fileextendsize") == 0)
	{
		FLMUINT	uiFileExtendSize;

		if( iArgC < 4)
		{
			pShell->con_printf( "Too few parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		uiFileExtendSize = f_atol( ppszArgV[ 3]);
		if( RC_BAD( rc = FlmDbConfig( hDb, FDB_FILE_EXTEND_SIZE,
								(void *)uiFileExtendSize, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		pShell->con_printf( "Invalid option.\n");
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbConfigCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbconfig", "Configure a database");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <db#> rflkeepfiles <on|off>\n", pszCommand);
		pShell->con_printf( "  %s <db#> rfldir <DirectoryName>\n", pszCommand);
		pShell->con_printf( "  %s <db#> rflfilelimits <MinRflSize> <MaxRflSize>\n",
						pszCommand);
		pShell->con_printf( "  %s <db#> rolltonextfile\n", pszCommand);
		pShell->con_printf( "  %s <db#> rflkeepabortedtrans <on|off>\n", pszCommand);
		pShell->con_printf( "  %s <db#> fileextendsize <FileExtendSize>\n", pszCommand);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbConfigCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbconfig", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC void format64BitNum(
	FLMUINT64	ui64Num,
	char *		pszBuf,
	FLMBOOL		bOutputHex,
	FLMBOOL		bAddCommas
	)
{
	char		szTmpBuf [60];
	FLMUINT	uiDigit;
	FLMUINT	uiChars = 0;
	FLMUINT	uiCharsBetweenCommas;

	if (bOutputHex)
	{
		while (ui64Num)
		{
			uiDigit = (FLMUINT)(ui64Num & 0xF);
			szTmpBuf [uiChars++] = (char)(uiDigit + '0');
			ui64Num >>= 4;
		}
	}
	else
	{
		uiCharsBetweenCommas = 0;
		while (ui64Num)
		{
			if (bAddCommas && uiCharsBetweenCommas == 3)
			{
				szTmpBuf [uiChars++] = ',';
				uiCharsBetweenCommas = 0;
			}
			uiDigit = (FLMUINT)(ui64Num % 10);
			szTmpBuf [uiChars++] = (char)(uiDigit + '0');
			ui64Num /= 10;
			uiCharsBetweenCommas++;
		}
	}

	// Need to reverse the numbers going back out.

	while (uiChars)
	{
		uiChars--;
		*pszBuf++ = szTmpBuf [uiChars];
	}
	*pszBuf = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmDbGetConfigCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMUINT			uiDbId;
	HFDB				hDb;
	FLMINT			iExitCode = 0;
	FLMUINT			uiArg;
	FLMUINT			uiArg2;
	FLMBOOL			bArg;
	char				szTmpPath[ F_PATH_MAX_SIZE];
	char				ucBuf[ 256];
	RCODE				rc = FERR_OK;
	FLMBOOL			bDoAll = FALSE;
	FLMBOOL			bValidOption = FALSE;

	if( iArgC < 3)
	{
		pShell->con_printf( "Too few parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	// Get the database ID and handle
	uiDbId = f_atol( ppszArgV[ 1]);
	if( RC_BAD( pShell->getDatabase( uiDbId, &hDb)) || hDb == HFDB_NULL)
	{
		pShell->con_printf( "Invalid database.\n");
		iExitCode = -1;
		goto Exit;
	}
	if (f_stricmp( ppszArgV [2], "all") == 0)
	{
		bDoAll = TRUE;
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "transid") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_TRANS_ID,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Current Transaction ID = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}
	if( bDoAll || f_stricmp( ppszArgV[ 2], "dbversion") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_VERSION,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Database Version = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "blocksize") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_BLKSIZ,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Database Block Size = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "language") == 0)
	{
		char	szLang [20];
		
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_DEFAULT_LANG,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		
		f_languageToStr( uiArg, szLang);
		pShell->con_printf( "Database Language = %s\n",
			szLang);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rfldir") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_RFL_DIR,
								szTmpPath, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "RFL directory = %s\n", szTmpPath);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflfilenum") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_RFL_FILE_NUM,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Current RFL file # = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflsizelimits") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_RFL_FILE_SIZE_LIMITS,
								&uiArg, &uiArg2, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "RFL file size limits = min:%u, max:%u\n",
			(unsigned)uiArg, (unsigned)uiArg2);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "diskusage") == 0)
	{
		FLMUINT64	ui64DbSize;
		FLMUINT64	ui64RollbackSize;
		FLMUINT64	ui64RflSize;
		char			szBuf1 [40];
		char			szBuf2 [40];
		char			szBuf3 [40];

		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_SIZES,
					&ui64DbSize, &ui64RollbackSize, &ui64RflSize)))
		{
			goto Exit;
		}

		format64BitNum( ui64DbSize, szBuf1, FALSE);
		format64BitNum( ui64RollbackSize, szBuf2, FALSE);
		format64BitNum( ui64RflSize, szBuf3, FALSE);
		pShell->con_printf( "Sizes = db:%s, rollback:%s, rfl:%s",
			szBuf1, szBuf2, szBuf3);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflkeepfiles") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_RFL_KEEP_FLAG,
					&bArg, NULL, NULL)))
		{
			goto Exit;
		}

		pShell->con_printf( "Keep RFL files = %s\n",
			bArg ? "Yes" : "No");
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "lastbackuptransid") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_LAST_BACKUP_TRANS_ID,
					&uiArg, NULL, NULL)))
		{
			goto Exit;
		}

		pShell->con_printf( "Last backup transaction ID = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "blockschangedsincebackup") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP,
					&uiArg, NULL, NULL)))
		{
			goto Exit;
		}

		pShell->con_printf( "Blocks changed since last backup = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "highestnotusedrflnum") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_RFL_HIGHEST_NU,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Highest Non-Used RFL Number = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}
	if( bDoAll || f_stricmp( ppszArgV[ 2], "nextincbackupseqnum") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_NEXT_INC_BACKUP_SEQ_NUM,
								&uiArg, NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf( "Next Incremental Backup Sequence # = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}
	if( bDoAll || f_stricmp( ppszArgV[ 2], "serialnumber") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_SERIAL_NUMBER,
					&ucBuf [0], NULL, NULL)))
		{
			goto Exit;
		}
		pShell->con_printf(
			"Serial number = "
			"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
			(unsigned)ucBuf[ 0],
			(unsigned)ucBuf[ 1],
			(unsigned)ucBuf[ 2],
			(unsigned)ucBuf[ 3],
			(unsigned)ucBuf[ 4],
			(unsigned)ucBuf[ 5],
			(unsigned)ucBuf[ 6],
			(unsigned)ucBuf[ 7],
			(unsigned)ucBuf[ 8],
			(unsigned)ucBuf[ 9],
			(unsigned)ucBuf[ 10],
			(unsigned)ucBuf[ 11],
			(unsigned)ucBuf[ 12],
			(unsigned)ucBuf[ 13],
			(unsigned)ucBuf[ 14],
			(unsigned)ucBuf[ 15]);
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "rflkeepabortedtrans") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG,
					&bArg, NULL, NULL)))
		{
			goto Exit;
		}

		pShell->con_printf( "Keep Aborted Trans in RFL = %s\n",
			bArg ? "Yes" : "No");
		bValidOption = TRUE;
	}

	if( bDoAll || f_stricmp( ppszArgV[ 2], "fileextendsize") == 0)
	{
		if (RC_BAD( rc = FlmDbGetConfig( hDb, FDB_GET_FILE_EXTEND_SIZE,
					&uiArg, NULL, NULL)))
		{
			goto Exit;
		}

		pShell->con_printf( "File Extend Size = %u\n",
			(unsigned)uiArg);
		bValidOption = TRUE;
	}

	if (!bValidOption)
	{
		pShell->con_printf( "Invalid option.\n");
		iExitCode = -1;
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmDbGetConfigCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "dbgetconfig", "Display DB configuration");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s <db#> <option>\n", pszCommand);
		pShell->con_printf( "  <option> may be one of the following:\n");
		pShell->con_printf( "    transid\n");
		pShell->con_printf( "    dbversion\n");
		pShell->con_printf( "    language\n");
		pShell->con_printf( "    blocksize\n");
		pShell->con_printf( "    rflfilenum\n");
		pShell->con_printf( "    diskusage\n");
		pShell->con_printf( "    rflkeepfiles\n");
		pShell->con_printf( "    lastbackuptransid\n");
		pShell->con_printf( "    blockschangedsincebackup\n");
		pShell->con_printf( "    highestnotusedrflnum\n");
		pShell->con_printf( "    nextincbackupseqnum\n");
		pShell->con_printf( "    serialnumber\n");
		pShell->con_printf( "    rflkeepabortedtrans\n");
		pShell->con_printf( "    fileextendsize\n");
		pShell->con_printf( "    all (will print all of the above)\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbGetConfigCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "dbgetconfig", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:	Constructor
*****************************************************************************/
FlmDbContext::FlmDbContext()
{
	f_memset( m_DbContexts, 0, sizeof( m_DbContexts));
	m_uiCurrDbId = 0;
}

/****************************************************************************
Desc:	Destructor
*****************************************************************************/
FlmDbContext::~FlmDbContext( void)
{
	FLMUINT	uiDbId;

	for( uiDbId = 0; uiDbId < MAX_DBCONTEXT_OPEN_DB; uiDbId++)
	{
		if (m_DbContexts [uiDbId].hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_DbContexts[ uiDbId].hDb);
		}
	}
}


/****************************************************************************
Desc:	Get an available database ID.
*****************************************************************************/
FLMBOOL FlmDbContext::getAvailDbId(
	FLMUINT *	puiDbId
	)
{
	FLMUINT	uiDbId = 0;

	while (uiDbId < MAX_DBCONTEXT_OPEN_DB &&
			 m_DbContexts [uiDbId].hDb != HFDB_NULL)
	{
		uiDbId++;
	}
	*puiDbId = uiDbId;
	return( (FLMBOOL)((uiDbId < MAX_DBCONTEXT_OPEN_DB) ? TRUE : FALSE));
}

/****************************************************************************
Desc:	Set the database handle for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setDb(
	FLMUINT	uiDbId,
	HFDB		hDb)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].hDb = hDb;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the database handle for a database ID
*****************************************************************************/
HFDB FlmDbContext::getDb(
	FLMUINT	uiDbId)
{
	return( (uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].hDb
						: HFDB_NULL);
}

/****************************************************************************
Desc:	Set the current container for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrContainer(
	FLMUINT	uiDbId,
	FLMUINT	uiContainer)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrContainer = uiContainer;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current container for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrContainer(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrContainer
						: (FLMUINT)0));
}

/****************************************************************************
Desc:	Set the current index for a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrIndex(
	FLMUINT	uiDbId,
	FLMUINT	uiIndex)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrIndex = uiIndex;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current index for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrIndex(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrIndex
						: (FLMUINT)0));
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrId(
	FLMUINT	uiDbId,
	FLMUINT	uiId)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrId = uiId;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMUINT FlmDbContext::getCurrId(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrId
						: (FLMUINT)0));
}

/****************************************************************************
Desc:	Set the current search flagsfor a database ID
*****************************************************************************/
FLMBOOL FlmDbContext::setCurrSearchFlags(
	FLMUINT	uiDbId,
	FLMUINT	uiSearchFlags)
{
	if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
	{
		m_DbContexts [uiDbId].uiCurrSearchFlags = uiSearchFlags;
		return( TRUE);
	}
	else
	{
		return( FALSE);
	}
}

/****************************************************************************
Desc:	Get the current search flags for a database ID
*****************************************************************************/
FLMUINT FlmDbContext::getCurrSearchFlags(
	FLMUINT	uiDbId)
{
	return( (FLMUINT)((uiDbId < MAX_DBCONTEXT_OPEN_DB)
						? m_DbContexts [uiDbId].uiCurrSearchFlags
						: (FLMUINT)FO_INCL));
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmFileSysCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	FLMINT				iExitCode = 0;
	RCODE					rc = FERR_OK;
	FLMUINT				uiLoop = 0;
	FLMBOOL				bForce = FALSE;
	IF_FileSystem *	pFileSystem = NULL;
	IF_DirHdl *			pDir = NULL;

	if (RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	// Delete the file

	if (f_stricmp( ppszArgV [0], "delete") == 0 ||
		 f_stricmp( ppszArgV [0], "del") == 0 ||
		 f_stricmp( ppszArgV [0], "rm") == 0)
	{
		for ( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if( ppszArgV[uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s.\n", ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		if( ( iArgC - uiLoop) < 1)
		{
			pShell->con_printf( "Wrong number of parameters.\n");
			iExitCode = -1;
			goto Exit;
		}

		if( bForce)
		{
			pFileSystem->setReadOnly( ppszArgV[uiLoop], FALSE);	
		}

		if( RC_BAD( rc = pFileSystem->deleteFile( ppszArgV[ uiLoop])))
		{
			goto Exit;
		}
		pShell->con_printf( "\nFile deleted.\n");
	}
	else if (f_stricmp( ppszArgV [0], "cd") == 0 ||
				f_stricmp( ppszArgV [0], "chdir") == 0)
	{
		if (iArgC > 1)
		{
			if( RC_BAD( f_chdir( (const char *)ppszArgV [1])))
			{
				pShell->con_printf( "Error changing directory\n");
			}
		}
	}

	else if(f_stricmp( "rename", ppszArgV[0]) == 0 ||
			f_stricmp( "move", ppszArgV[0]) == 0 ||
			f_stricmp( "mv", ppszArgV[0]) == 0 )
	{
		FLMBOOL		bOverwrite = FALSE;

		for( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if ( ppszArgV[ uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				case 'o':
				case 'O':
					bOverwrite = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s\n", &ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		// The remaining two parameters are the source and destination

		if ( ( iArgC - uiLoop) < 2)
		{
			pShell->con_printf( "You must specify a source and destination.\n");
			iExitCode = -1;
			goto Exit;
		}

		if ( RC_BAD( rc = pFileSystem->doesFileExist( ppszArgV[uiLoop])))
		{
			goto Exit;
		}

		if ( pFileSystem->isDir( ppszArgV[uiLoop + 1]))
		{
			char	szFilename[ F_FILENAME_SIZE];

			// If the second param is a directory we'll assume the user wants to
			// move the file into it with the same filename.

			pFileSystem->pathReduce( ppszArgV[uiLoop], NULL, szFilename);
			pFileSystem->pathAppend( szFilename, ppszArgV[uiLoop + 1]);
		}

		if( RC_OK( pFileSystem->doesFileExist( ppszArgV[uiLoop + 1])))
		{
			if ( !bOverwrite)
			{
				FLMUINT	uiChar;

				pShell->con_printf( "%s exists. Overwrite? (Y/N)", ppszArgV[ uiLoop + 1]);
				for(;;)
				{
					if( RC_OK( FTXWinTestKB( pShell->getWindow())))
					{
						FTXWinInputChar( pShell->getWindow(), &uiChar);

						// Echo char back to the user
						pShell->con_printf( "%c\n", uiChar);

						if ( uiChar == 'Y' || uiChar == 'y')
						{
							bOverwrite = TRUE;
						}
						else
						{
							rc = RC_SET( FERR_USER_ABORT);
							goto Exit;
						}
						break;
					}
				}
			}
			if ( bOverwrite)
			{
				if ( bForce)
				{
					pFileSystem->setReadOnly( ppszArgV[uiLoop + 1], FALSE);
					pShell->con_printf( "Error changing file attributes. ");
					goto Exit;
				}

				if ( RC_BAD( rc = pFileSystem->deleteFile( ppszArgV[uiLoop + 1])))
				{
					pShell->con_printf( "Error removing destination file. ");
					goto Exit;
				}
			}
		}

		if ( RC_BAD( rc = pFileSystem->renameFile( ppszArgV[uiLoop], 
			ppszArgV[uiLoop + 1])))
		{
			goto Exit;
		}

		pShell->con_printf( "%s -> %s\n", 
			ppszArgV[uiLoop], ppszArgV[uiLoop + 1]);
	}
	else if(f_stricmp( "copy", ppszArgV[0]) == 0 ||
				f_stricmp( "cp", ppszArgV[0]) == 0)
	{
		FLMBOOL			bOverwrite = FALSE;
		FLMUINT64		ui64BytesCopied = 0;

		for( uiLoop = 1; uiLoop < (FLMUINT)iArgC; uiLoop++)
		{
			if ( ppszArgV[ uiLoop][0] == '/')
			{
				switch( ppszArgV[uiLoop][1])
				{
				case 'f':
				case 'F':
					bForce = TRUE;
					break;
				case 'o':
				case 'O':
					bOverwrite = TRUE;
					break;
				default:
					pShell->con_printf( "Unknown option: %s\n", &ppszArgV[uiLoop]);
					iExitCode = -1;
					goto Exit;
				}
			}
			else
			{
				break;
			}
		}

		// The remaining two parameters are the source and destination

		if ( ( iArgC - uiLoop) < 2)
		{
			pShell->con_printf( "You must specify a source and destination.\n");
			iExitCode = -1;
			goto Exit;
		}

		if ( RC_BAD( rc = pFileSystem->doesFileExist( ppszArgV[uiLoop])))
		{
			goto Exit;
		}

		if ( pFileSystem->isDir( ppszArgV[uiLoop + 1]))
		{
			char	szFilename[ F_FILENAME_SIZE];

			// If the second param is a directory we'll assume the user wants to
			// copy the file into it with the same filename.

			pFileSystem->pathReduce( ppszArgV[uiLoop], NULL, szFilename);
			pFileSystem->pathAppend( szFilename, ppszArgV[uiLoop + 1]);
		}

		if ( RC_OK( pFileSystem->doesFileExist( ppszArgV[uiLoop + 1])))
		{
			if ( !bOverwrite)
			{
				FLMUINT	uiChar;

				pShell->con_printf( "%s exists. Overwrite? (Y/N)", ppszArgV[ uiLoop + 1]);
				for(;;)
				{
					if( RC_OK( FTXWinTestKB( pShell->getWindow())))
					{
						FTXWinInputChar( pShell->getWindow(), &uiChar);

						// Echo char back to the user
						pShell->con_printf( "%c\n", uiChar);

						if ( uiChar == 'Y' || uiChar == 'y')
						{
							bOverwrite = TRUE;
						}
						else
						{
							rc = RC_SET( FERR_USER_ABORT);
							goto Exit;
						}
						break;
					}
				}
			}

			if ( bOverwrite)
			{

				// There's no sense in changing a file's attributes if we aren't
				// going to overwrite it.

				if ( bForce)
				{
					pFileSystem->setReadOnly( ppszArgV[ uiLoop + 1], FALSE);
				}
			}
		}

		if ( RC_BAD( rc = pFileSystem->copyFile( 
			ppszArgV[uiLoop], ppszArgV[uiLoop +1], bOverwrite, &ui64BytesCopied)))
		{
			goto Exit;
		}

		pShell->con_printf( "%s copied to %s (%I64u bytes copied)\n", 
			ppszArgV[uiLoop], ppszArgV[uiLoop + 1], ui64BytesCopied);
	}
	else if (f_stricmp( "ls", ppszArgV [0]) == 0 ||
				f_stricmp( "dir", ppszArgV [0]) == 0)
	{
		char				szDir [F_PATH_MAX_SIZE];
		char				szBaseName [F_FILENAME_SIZE];
		FLMUINT			uiLineCount;
		FTX_WINDOW *	pWindow = pShell->getWindow();
		FLMUINT			uiMaxLines;
		FLMUINT			uiNumCols;
		FLMUINT			uiChar;

		FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiMaxLines);
		uiMaxLines--;

		if( iArgC > 1)
		{
			if (RC_BAD( rc = pFileSystem->pathReduce( 
				ppszArgV [1], szDir, szBaseName)))
			{
				goto Exit;
			}
			
			if (!szDir [0])
			{
				f_strcpy( szDir, ".");
			}
			
			if (RC_BAD( rc = pFileSystem->openDir( szDir, 
				(char *)szBaseName, &pDir)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = pFileSystem->openDir( ".", NULL, &pDir)))
			{
				goto Exit;
			}
		}
		pShell->con_printf( "%-20s %25s\n", "File Name", "File Size");
		uiLineCount = 1;
		for (;;)
		{
			if (RC_BAD( rc = pDir->next()))
			{
				if (rc == FERR_IO_NO_MORE_FILES)
				{
					rc = FERR_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			if (uiLineCount == uiMaxLines)
			{
				pShell->con_printf(
					"...(more, press any character to continue, ESC to quit)");
				uiChar = 0;
				for (;;)
				{
					if( RC_OK( FTXWinTestKB( pWindow)))
					{
						FTXWinInputChar( pWindow, &uiChar);
						break;
					}
					if (gv_bShutdown)
					{
						uiChar = FKB_ESC;
						break;
					}
					f_yieldCPU();
				}
				if (uiChar == FKB_ESC)
				{
					break;
				}
				pShell->con_printf(
					"\r                                                       \r");
				uiLineCount = 0;
			}
			if (pDir->currentItemIsDir())
			{
				pShell->con_printf( "%-20s %25s\n", pDir->currentItemName(),
					"<DIR>");
			}
			else
			{
				char	szTmpBuf [60];

				format64BitNum( (FLMUINT64)pDir->currentItemSize(),
					szTmpBuf, FALSE, TRUE);
				pShell->con_printf( "%-20s %25s\n", pDir->currentItemName(),
					szTmpBuf);
			}
			uiLineCount++;
		}
	}
	else
	{
		// Should never happen!
		flmAssert( 0);
	}

Exit:

	if( pDir)
	{
		pDir->Release();
	}

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if( RC_BAD( rc))
	{
		pShell->con_printf( "\nError: %e\n", rc);
		if( !iExitCode)
		{
			iExitCode = rc;
		}
	}

	return( iExitCode);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmFileSysCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "delete, del, rm", "Delete a file. Options: "
			"/O - Overwrite /F - Force");
		pShell->displayCommand( "cd (or chdir)", "Change directories");
		pShell->displayCommand( "dir, ls", "List files");
		pShell->displayCommand( "copy, cp", "Copy files. Options: "
			"/O - Overwrite /F - Force");
		pShell->displayCommand( "rename, move, mv", "Move a file. Options: "
			"/O - Overwrite /F - Force");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		if (f_stricmp( "cd", pszCommand) == 0)
		{
			pShell->con_printf( "  %s <Directory>\n", pszCommand);
		}
		else if (f_stricmp( "ls", pszCommand) == 0 ||
					f_stricmp( "dir", pszCommand) == 0)
		{
			pShell->con_printf( "  %s [FileMask]\n", pszCommand);
		}
		else
		{
			pShell->con_printf( "  %s <filename>\n", pszCommand);
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmFileSysCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "delete", pszCommand) == 0 ||
				f_stricmp( "del", pszCommand) == 0 ||
				f_stricmp( "rm", pszCommand) == 0 ||
				f_stricmp( "ls", pszCommand) == 0 ||
				f_stricmp( "dir", pszCommand) == 0 ||
				f_stricmp( "cd", pszCommand) == 0 ||
				f_stricmp( "chdir", pszCommand) == 0 ||
				f_stricmp( "del", pszCommand) == 0 ||
				f_stricmp( "rm", pszCommand) == 0 ||
				f_stricmp( "copy", pszCommand) == 0 ||
				f_stricmp( "cp", pszCommand) == 0 ||
				f_stricmp( "move", pszCommand) == 0 ||
				f_stricmp( "mv", pszCommand) == 0 ||
				f_stricmp( "rename", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMINT FlmEditCommand::execute(
	FLMINT		iArgC,
	char **		ppszArgV,
	FlmShell *	pShell)
{
	RCODE				rc = FERR_OK;
	FLMINT			iExitCode = 0;
	HFDB				hDb = HFDB_NULL;
	FLMUINT			uiDbId;
	F_RecEditor *	pRecEditor = NULL;
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	char				szTitle[ 80];
	FLMUINT			Cols;
	FLMUINT			Rows;
	HFDB				hNewDb = HFDB_NULL;
	char *			pszRflDir = NULL;
	char *			pszPassword = NULL;
	char *			pszAllowLimited;
	FLMUINT			uiOpenFlags = 0;

	if( iArgC < 1)
	{
		pShell->con_printf( "Wrong number of parameters.\n");
		iExitCode = -1;
		goto Exit;
	}

	if( iArgC >= 2)
	{
		char *	pszName = ppszArgV [1];

		while (*pszName)
		{
			if (*pszName < '0' || *pszName > '9')
			{
				break;
			}
			pszName++;
		}
		if (*pszName)
		{

			if( iArgC >= 3)
			{
				pszRflDir = ppszArgV[ 2];
			}
			
			if (iArgC >=4)
			{
				pszPassword = ppszArgV[ 3];
			}

			if (iArgC >=5)
			{
				pszAllowLimited = ppszArgV[ 4];
				
				if (f_strnicmp( pszAllowLimited, "TRUE", 4) == 0)
				{
					uiOpenFlags |= FO_ALLOW_LIMITED;
				}
			}

			if( RC_BAD( rc = FlmDbOpen( ppszArgV[ 1],
				NULL, pszRflDir, uiOpenFlags, pszPassword, &hNewDb)))
			{
				pShell->con_printf( "Error opening database: %e.\n", rc);
				iExitCode = -1;
				goto Exit;
			}

			if( RC_BAD( rc = pShell->registerDatabase( hNewDb, &uiDbId)))
			{
				pShell->con_printf( "Error registering database: %e.\n", rc);
				iExitCode = -1;
				goto Exit;
			}
			hNewDb = HFDB_NULL;

			pShell->con_printf( "Database #%u opened.\n", (unsigned)uiDbId);
		}
		else
		{
			uiDbId = f_atoi( ppszArgV[ 1]);
		}
	}
	else
	{
		uiDbId = 0;
	}

	if (RC_BAD( rc = pShell->getDatabase( uiDbId, &hDb)))
	{
		pShell->con_printf( "Error getting database handle: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	if (hDb == HFDB_NULL)
	{
		pShell->con_printf( "Database %u not open.\n", uiDbId);
		iExitCode = -1;
		goto Exit;
	}

	f_sprintf( szTitle,
		"Database Edit for FLAIM [DB=%s/BUILD=%s]",
		FLM_CUR_FILE_FORMAT_VER_STR, __DATE__);

	if( RC_BAD( FTXScreenInit( szTitle, &pScreen)))
	{
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( FTXWinInit( pScreen, 0, 1, &pTitleWin)))
	{
		iExitCode = -1;
		goto Exit;
	}

	FTXWinPaintBackground( pTitleWin, FLM_RED);
	FTXWinPrintStr( pTitleWin, szTitle);
	FTXWinSetCursorType( pTitleWin, FLM_CURSOR_INVISIBLE);
	FTXWinOpen( pTitleWin);

	if ((pRecEditor = f_new F_RecEditor()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		pShell->con_printf( "Error allocating editor object: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	if( RC_BAD( rc = pRecEditor->Setup( pScreen)))
	{
		pShell->con_printf( "Error setting up editor object: %e.\n", rc);
		iExitCode = -1;
		goto Exit;
	}

	pRecEditor->setDefaultSource( hDb, FLM_DATA_CONTAINER);
	pRecEditor->setShutdown( &gv_bShutdown);

	// Start up the editor

	FTXScreenGetSize( pScreen, &Cols, &Rows);

	FTXScreenDisplay( pScreen);

	pRecEditor->interactiveEdit( 0, 1, Cols - 1, Rows - 1);

Exit:

	if( hNewDb != HFDB_NULL)
	{
		(void)FlmDbClose( &hNewDb);
	}

	if( pRecEditor)
	{
		pRecEditor->Release();
		pRecEditor = NULL;
	}

	FTXScreenFree( &pScreen);

	return( iExitCode);
}

/****************************************************************************
Desc:	displayHelp - print a help message.
*****************************************************************************/
void FlmEditCommand::displayHelp(
	FlmShell *	pShell,
	char *		pszCommand)
{
	if (!pszCommand)
	{
		pShell->displayCommand( "edit", "Edit a database (DOM Editor)");
	}
	else
	{
		pShell->con_printf("Usage:\n");
		pShell->con_printf( "  %s [db#] (if db# is omitted, 0 is used)\n",
							  pszCommand);
		pShell->con_printf( "  OR\n");
		pShell->con_printf( "  %s <DbFileName> [<RflPath> [<Password> [<AllowLimited>]]]\n", pszCommand);
		pShell->con_printf( "      <AllowLimited> = TRUE | FALSE\n");
		pShell->con_printf( "      This form of the command will open the database before editing.\n");
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmEditCommand::canPerformCommand(
	char *		pszCommand)
{
	return( (f_stricmp( "edit", pszCommand) == 0)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::setupDirectories( 
	char *	pszBaseDir,
	char *	pszExtendedDir)
{
	RCODE		rc = FERR_OK;

	flmAssert( !m_pszBaseDir && !m_pszExtendedDir && !m_pszResolvedDir);

	if( RC_BAD( rc = f_alloc( MAX_PATH_SIZE, &m_pszBaseDir)))
	{
		goto Exit;
	}

	f_strcpy( m_pszBaseDir, pszBaseDir);
	if ( m_pszBaseDir[ f_strlen(m_pszBaseDir) - 1] != SLASH)
	{
		f_strcat( m_pszBaseDir, SSLASH);
	}

	if( *pszExtendedDir)
	{
		if( f_strlen( pszExtendedDir) < MAX_PATH_SIZE)
		{
			if( RC_BAD( f_alloc( MAX_PATH_SIZE, &m_pszExtendedDir)))
			{
				goto Exit;
			}
			f_strcpy( m_pszExtendedDir, pszExtendedDir);
		}
		else
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}

	if ( !m_pszResolvedDir)
	{
		if( RC_BAD( rc = f_alloc( MAX_PATH_SIZE, &m_pszResolvedDir)))
		{
			goto Exit;
		}
	}
	
	if ( RC_BAD( rc = resolveDir()))
	{
		goto Exit;
	}
Exit:
	return rc;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL DirectoryIterator::isInSet(
	char *	pszFilename)
{
	char *		pszTemp = NULL;
	char *		pszSave = NULL;
	FLMBOOL		bFound = FALSE;
	FLMUINT		uiLoop = 0;

	if( RC_BAD( f_alloc( f_strlen( m_pszResolvedDir) + 
		f_strlen( pszFilename) + 1, &pszTemp)))
	{
		goto Exit;
	}

	// Move past the prefix since that is not stored in the match list.

	if( f_strstr( pszFilename, m_pszResolvedDir) == pszFilename)
	{
		f_strcpy( pszTemp, pszFilename + f_strlen( m_pszResolvedDir));
	}

	for( uiLoop = 0; uiLoop < m_uiTotalMatches; uiLoop++)
	{
		if ( f_stricmp( pszTemp, m_ppszMatchList[uiLoop]) == 0)
		{
			bFound = TRUE;
			break;
		}
	}

	if ( pszTemp)
	{
		f_free( &pszTemp);
	}
	if ( pszSave)
	{
		f_strcpy( pszFilename, pszSave);
		f_free( &pszSave);
	}

Exit:

	return bFound;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::setupForSearch(
	IF_FileSystem *	pFileSystem,
	char *				pszBaseDir,
	char *				pszExtendedDir,
	char *				pszPattern)
{
	RCODE		rc = FERR_OK;

	if ( RC_BAD( rc = setupDirectories( pszBaseDir, pszExtendedDir)))
	{
		goto Exit;
	}

	flmAssert( !m_pDirHdl && !m_ppszMatchList);

	if ( RC_BAD( pFileSystem->openDir( m_pszResolvedDir, 
		(char *)pszPattern, &m_pDirHdl)))
	{
		goto Exit;
	}

	// First pass - determine the number of matches

	while( RC_OK( m_pDirHdl->next()))
	{
		m_uiTotalMatches++;
	}
	
	m_pDirHdl->Release();
	m_pDirHdl = NULL;

	if( RC_BAD( rc = f_alloc( 
		sizeof( char *) * m_uiTotalMatches, &m_ppszMatchList)))
	{
		goto Exit;
	}

	f_memset( m_ppszMatchList, 0, m_uiTotalMatches * sizeof( char *));
	
	// Reopen the directory and copy the matches
		
	if ( RC_BAD( pFileSystem->openDir( m_pszResolvedDir, 
		(char *)pszPattern, &m_pDirHdl)))
	{
		goto Exit;
	}

	m_uiCurrentMatch = 0;
	while ( RC_OK( m_pDirHdl->next()))
	{
		if( RC_BAD( rc = f_alloc( 
			f_strlen( m_pDirHdl->currentItemName()) + 1,
			&m_ppszMatchList[m_uiCurrentMatch])))
		{
			goto Exit;
		}

		f_strcpy( m_ppszMatchList[m_uiCurrentMatch], 
			m_pDirHdl->currentItemName());

		m_uiCurrentMatch++;
	}
	
	m_uiCurrentMatch = 0;
	m_bInitialized = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::next( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( m_uiCurrentMatch >= m_uiTotalMatches)
	{
		m_uiCurrentMatch = 0;
	}
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiCurrentMatch]); 
	}
	m_uiCurrentMatch++;
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::prev( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if( m_uiCurrentMatch == 0)
	{
		m_uiCurrentMatch = m_uiTotalMatches;
	}
	m_uiCurrentMatch--;

	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiCurrentMatch]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::first( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ 0]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::last( 
	char *	pszReturn, 
	FLMBOOL	bCompletePath)
{
	if ( pszReturn)
	{
		*pszReturn = '\0';
		if ( bCompletePath)
		{
			f_strcpy( pszReturn, getResolvedPath());
		}
		f_strcat( pszReturn, m_ppszMatchList[ m_uiTotalMatches - 1]); 
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE DirectoryIterator::resolveDir( void)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiIndex = 0;
	char					szTemp[ MAX_PATH_SIZE];
	IF_FileSystem *	pFileSystem = NULL;

	if( !m_pszExtendedDir)
	{
		f_strcpy( m_pszResolvedDir, m_pszBaseDir);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}
	
	// If it begins with a SLASH, just go to the root
	
	if ( m_pszExtendedDir[0] == SLASH) 
	{
		if( RC_BAD( rc = extractRoot( m_pszBaseDir, szTemp)))
		{
			goto Exit;
		}

		f_strcpy( m_pszResolvedDir, szTemp); 
		f_strcat( m_pszResolvedDir, &m_pszExtendedDir[1]);

		goto Exit;
	}

	// I think this can only happen on windows and NetWare
	
	else if (isDriveSpec( m_pszExtendedDir))
	{
		// We have been given a fully-specified drive path
		
		f_strcpy( m_pszResolvedDir, m_pszExtendedDir);
		goto Exit;
	}

	// For each ".." reduce the base path by one
	
	for(;;)
	{
		if( (f_strlen( &m_pszExtendedDir[uiIndex]) >= f_strlen( PARENT_DIR)) &&
			f_memcmp( &m_pszExtendedDir[uiIndex], PARENT_DIR, f_strlen( PARENT_DIR)) == 0)
		{
			uiIndex += f_strlen( PARENT_DIR);
			if ( m_pszExtendedDir[uiIndex] == SLASH)
			{
				uiIndex++;
			}
	
			pFileSystem->pathReduce( m_pszBaseDir, szTemp, NULL);
			f_strcpy( m_pszBaseDir, szTemp);
			if ( m_pszBaseDir[ f_strlen(m_pszBaseDir) - 1] != SLASH)
			{
				f_strcat( m_pszBaseDir, SSLASH);
			}
		}
		else
		{
			break;
		}
	}

	// Tack on whatever's left
	
	f_strcpy( m_pszResolvedDir, m_pszBaseDir);
	if( m_pszResolvedDir[f_strlen( m_pszResolvedDir) - 1] != SLASH)
	{
		// Put the slash back on. pathReduce likes to take it off.
		
		f_strcat( m_pszResolvedDir, SSLASH);
	}
	
	f_strcat( m_pszResolvedDir, &m_pszExtendedDir[uiIndex]);

Exit:

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void DirectoryIterator::reset( void)
{
	if( m_pszBaseDir)
	{
		f_free( &m_pszBaseDir);
	}

	if( m_pszExtendedDir)
	{
		f_free( &m_pszExtendedDir);
	}

	if( m_pszResolvedDir)
	{
		f_free( &m_pszResolvedDir);
		m_pszResolvedDir = NULL;
	}

	if( m_pDirHdl)
	{
		m_pDirHdl->Release();
		m_pDirHdl = NULL;
	}

	if( m_ppszMatchList)
	{
		for( FLMUINT uiLoop = 0; uiLoop < m_uiTotalMatches; uiLoop++)
		{
			f_free( &m_ppszMatchList[uiLoop]);
		}
		
		f_free( &m_ppszMatchList);
		m_ppszMatchList = NULL;
	}

	m_uiCurrentMatch = 0;
	m_uiTotalMatches = 0;
	m_bInitialized = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL DirectoryIterator::isDriveSpec(
	char *		pszPath)
{
	FLMBOOL		bIsDriveSpec = FALSE;
	char *		pszTemp = NULL;

	if ((pszTemp = (char *)f_strchr( pszPath, ':')) != NULL)
	{
		if( *(pszTemp + 1) == SLASH)
		{
			bIsDriveSpec = TRUE;
		}
#ifdef FLM_NLM
		// Netware accepts both front and back slashes
		else if( *(pszTemp + 1) == FWSLASH)
		{
			bIsDriveSpec = TRUE;
		}
#endif
	}
	return bIsDriveSpec;
}

/****************************************************************************
Desc: 
*****************************************************************************/
RCODE DirectoryIterator::extractRoot( 
	char *		pszPath,
	char *		pszRoot)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiIndex = 0;
	FLMUINT		uiLen	= f_strlen( pszPath);

	for ( uiIndex = 0; uiIndex < uiLen; uiIndex++)
	{
		if( pszPath[uiIndex] == '\\')
		{
			f_strncpy( pszRoot, pszPath, uiIndex + 1); 
			pszRoot[uiIndex + 1] = '\0';
			goto Exit;
		}
	}
	
	rc = RC_SET( FERR_NOT_FOUND);
	pszRoot[0] = '\0';
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC void removeChars(
	char *	pszString,
	char		cChar)
{
	char *	pszFrom = pszString;
	char *	pszTo = pszString;

	for ( ;;)
	{
		if ( *pszFrom != cChar)
		{
			*pszTo = *pszFrom;
			if ( *pszTo)
			{
				pszTo++;
				pszFrom++;
			}
			else
			{
				break;
			}
		}
		else
		{
			pszFrom++;
		}
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC char * positionToPath(
	char *	pszCommandLine)
{
	char *	pszPathBegin = pszCommandLine + f_strlen( pszCommandLine);

	if( f_strlen( pszCommandLine) != 0 && *(pszPathBegin - 1) != ASCII_SPACE)
	{
		if( *(pszPathBegin - 1))
		{
			// Move to the beginning of the last token

			while( (pszPathBegin != pszCommandLine) && 
				(*(pszPathBegin - 1) != ASCII_SPACE))
			{
				if( *(pszPathBegin - 1) == '\"')
				{
					// Find first whitespace after begin quote

					FLMBOOL		bInQuotes = TRUE;

					pszPathBegin--;
					
					while ( ( pszPathBegin != pszCommandLine) &&	bInQuotes)
					{
						pszPathBegin--;
						if ( *pszPathBegin == '\"')
						{
							bInQuotes = FALSE;
						}
					}
				}
				else
				{
					pszPathBegin--;
				}
			}
		}
	}

	return( pszPathBegin);
}

/****************************************************************************
Desc: Given a path, extract the base directory and a wildcard for searching
*****************************************************************************/
FSTATIC void extractBaseDirAndWildcard(
	IF_FileSystem *	pFileSystem,
	char *				pszPath, 
	char *				pszBase, 
	char *				pszWildcard)
{
	flmAssert( pszBase && pszWildcard);

	pszBase[0] = '\0';
	pszWildcard[0] = '\0';

	// If the extended directory is a path but does not end with a 
	// slash, this means that we will use the last portion of the
	// path as our search pattern.

	if( pszPath &&
		f_strchr( pszPath, SLASH) &&
		pszPath[ f_strlen( pszPath) - 1] != SLASH)
	{
		pFileSystem->pathReduce( pszPath, pszBase, pszWildcard);

		// Darn thing sometimes removes the trailing slash. Put it back.

		if ( pszPath[ f_strlen( pszBase) - 1] != SLASH)
		{
			f_strcat( pszBase, SSLASH);
		}
	}
	else if ( pszPath && !f_strchr( pszPath, SLASH))
	{
		// We will assume that what we have is just a part of a filename
		// since it contains no slashes.
		f_strcpy( pszWildcard, pszPath);
		//pszTabCompleteBegin[0] = '\0';
	}
	else if ( pszPath && pszPath[ f_strlen( pszPath) - 1] == SLASH)
	{
		// We were given only a path
		f_strcpy( pszBase, pszPath);
	}
	f_strcat( pszWildcard, "*");
}

/***************************************************************************
Desc:	Program entry point (main)
****************************************************************************/
#ifdef FLM_RING_ZERO_NLM
extern "C" int nlm_main(
#else
int main(
#endif
	int,		// iArgC,
	char **	// ppszArgV
	)
{
	RCODE			rc = FERR_OK;
	FlmShell *	pShell = NULL;

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXInit( "FLAIM DB Shell", (FLMBYTE)80, (FLMBYTE)50,
		FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		goto Exit;
	}

	FTXSetShutdownFlag( &gv_bShutdown);

	if( (pShell = f_new FlmShell) != NULL)
	{
		if( RC_OK( pShell->setup()))
		{
			(void)pShell->execute();
		}
	}

Exit:

	if (pShell)
	{
		pShell->Release();
	}

	gv_bShutdown = TRUE;
	
	FTXExit();
	FlmShutdown();
	
	return( 0);
}
