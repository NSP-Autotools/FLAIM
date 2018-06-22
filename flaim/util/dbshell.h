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

#ifndef DBSHELL_H
#define DBSHELL_H

#include "flaimsys.h"
#include "sharutil.h"

// Types of clipboard data

enum eClipboardDataType
{
	CLIPBOARD_EMPTY,
	CLIPBOARD_GEDCOM,
	CLIPBOARD_TEXT
};

typedef enum eClipboardDataType ClipboardDataType;
class FlmShell;

/*===========================================================================
struct:	DB_CONTEXT
Desc:		This structure contains information for a particular database.
===========================================================================*/
typedef struct DBContextTag
{
	HFDB		hDb;
	FLMUINT	uiCurrContainer;
	FLMUINT	uiCurrIndex;
	FLMUINT	uiCurrId;
	FLMUINT	uiCurrSearchFlags;
} DB_CONTEXT;

/*===========================================================================
Desc:		This class is used by the shell to perform commands it has parsed.
===========================================================================*/
class FlmCommand : public F_Object
{
public:
	FlmCommand( void){}

	virtual ~FlmCommand( void) {}

	// Methods that must be implemented in classes that extend this class.

	virtual FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell) = 0;

	virtual void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand) = 0;

	virtual FLMBOOL canPerformCommand(
		char *	pszCommand) = 0;

};

/*===========================================================================
Desc:		This class manages a database context - FLAIM session and #N
			open databases.
===========================================================================*/
class FlmDbContext : public F_Object
{
#define MAX_DBCONTEXT_OPEN_DB		9
public:
	FlmDbContext( void);
	~FlmDbContext( void);

	FINLINE FLMUINT getCurrDbId(
		void)
	{
		return m_uiCurrDbId;
	}

	FINLINE void setCurrDbId(
		FLMUINT uiDbId)
	{
		if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
		{
			m_uiCurrDbId = uiDbId;
		}
	}

	FLMBOOL getAvailDbId(
		FLMUINT *		puiDbId);

	FLMBOOL setDb(
		FLMUINT	uiDbId,
		HFDB		hDb);

	HFDB getDb(
		FLMUINT	uiDbId);

	FLMBOOL setCurrContainer(
		FLMUINT	uiDbId,
		FLMUINT	uiContainer);

	FLMUINT getCurrContainer(
		FLMUINT	uiDbId);

	FLMBOOL setCurrIndex(
		FLMUINT	uiDbId,
		FLMUINT	uiIndex);

	FLMUINT getCurrIndex(
		FLMUINT	uiDbId);

	FLMBOOL setCurrId(
		FLMUINT	uiDbId,
		FLMUINT	uiId);

	FLMUINT getCurrId(
		FLMUINT	uiDbId);

	FLMBOOL setCurrSearchFlags(
		FLMUINT	uiDbId,
		FLMUINT	uiSearchFlags);

	FLMUINT getCurrSearchFlags(
		FLMUINT	uiDbId);

private:
	FLMUINT					m_uiCurrDbId;
	DB_CONTEXT				m_DbContexts [MAX_DBCONTEXT_OPEN_DB];
};

/*===========================================================================
Desc:		This class parses a command line
===========================================================================*/
class FlmParse : public F_Object
{
public:

	FlmParse( void);

	~FlmParse( void)
	{
	}

	void setString(
		char *		pszString);

	char * getNextToken( void);

private:

	char			m_szString[ 512];
	char			m_szToken[ 512];
	char *		m_pszCurPos;
};

/*===========================================================================
Desc:		This class manages a command-line shell
===========================================================================*/
class FlmShell : public F_Object
{
public:

	FlmShell();
	virtual ~FlmShell();

	RCODE setup( void);

	// Methods that are invoked by the command objects

	RCODE registerDatabase(
		HFDB			hDb,
		FLMUINT *	puiDbId);

	RCODE getDatabase(
		FLMUINT	uiDbId,
		HFDB *	phDb);

	RCODE deregisterDatabase(
		FLMUINT	uiDbId);

	void con_printf(
		const char *	pszFormat, ...);

	FINLINE void displayCommand(
		const char *	pszCommand,
		const char *	pszDescription)
	{
		con_printf( "  %-20s  -- %s\n", pszCommand, pszDescription);
	}

	RCODE execute( void);

	RCODE registerCmd(
		FlmCommand *	pCmd);

	RCODE addCmdHistory(
		char *		pszCmd);

	FINLINE char * getOutputFileName( void)
	{
		return m_pszOutputFile;
	}

	FINLINE void setScreen(
		FTX_SCREEN *	pScreen)
	{
		m_pScreen = pScreen;
	}
	
	FINLINE FTX_SCREEN * getScreen( void)
	{
		return( m_pScreen);
	}

	FINLINE void setWindow(
		FTX_WINDOW *	pWindow)
	{
		m_pWindow = pWindow; 
	}
	
	FINLINE FTX_WINDOW * getWindow( void)
	{
		return( m_pWindow);
	}

private:

#define MAX_SHELL_OPEN_DB				10
#define MAX_SHELL_HISTORY_ITEMS		5
#define MAX_REGISTERED_COMMANDS		50
#define MAX_CMD_LINE_LEN				256
#define MAX_THREAD_NAME_LEN			64

	FTX_SCREEN *			m_pScreen;
	FTX_WINDOW *			m_pWindow;
	FTX_WINDOW *			m_pTitleWin;
	HFDB						m_DbList[ MAX_SHELL_OPEN_DB];
	F_Pool					m_ArgPool;
	FLMINT					m_iCurrArgC;
	char **					m_ppCurrArgV;
	char *					m_pszOutputFile;
	FLMINT					m_iLastCmdExitCode;
	FLMBOOL					m_bPagingEnabled;
	FlmCommand *			m_ppCmdList[ MAX_REGISTERED_COMMANDS];
	char *					m_ppHistory[ MAX_SHELL_HISTORY_ITEMS];

	// Private methods
	RCODE parseCmdLine(
		char *		pszString);

	RCODE executeCmdLine( void);

	RCODE selectCmdLineFromList(	// Pops up a selection list and allows
											// the user to choose a command line
											// from the history list
		char *	pszCmdLineRV);
};

/*===========================================================================
Desc:		This class implements the database open command
===========================================================================*/
class FlmDbOpenCommand : public FlmCommand
{
public:
	FlmDbOpenCommand( void) {}
	~FlmDbOpenCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the database close command
===========================================================================*/
class FlmDbCloseCommand : public FlmCommand
{
public:
	FlmDbCloseCommand( void) {}
	~FlmDbCloseCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the trans command (for transactions)
===========================================================================*/
class FlmTransCommand : public FlmCommand
{
public:
	FlmTransCommand( void) {}
	~FlmTransCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the dbCopy, dbRename, and dbRemove commands.
==========================================================================*/
class FlmDbManageCommand : public FlmCommand
{
public:
	FlmDbManageCommand( void) {}
	~FlmDbManageCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the backup command (FlmDbBackup)
===========================================================================*/
class FlmBackupCommand : public FlmCommand
{
public:
	FlmBackupCommand( void) {}
	~FlmBackupCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the restore command (FlmDbRestore)
===========================================================================*/
class FlmRestoreCommand : public FlmCommand
{
public:
	FlmRestoreCommand( void) {}
	~FlmRestoreCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:
===========================================================================*/
class FlmDbConfigCommand : public FlmCommand
{
public:
	FlmDbConfigCommand( void) {}
	~FlmDbConfigCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:
===========================================================================*/
class FlmDbGetConfigCommand : public FlmCommand
{
public:
	FlmDbGetConfigCommand( void) {}
	~FlmDbGetConfigCommand( void) {}

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the file system command.
===========================================================================*/

class FlmFileSysCommand : public FlmCommand
{
public:

	FlmFileSysCommand( void)
	{
	}

	~FlmFileSysCommand( void)
	{
	}

	// Methods that must be implemented in classes that extend this class.

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);


	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		This class implements the file delete command.
===========================================================================*/
class FlmEditCommand : public FlmCommand
{
public:
	FlmEditCommand( void)
	{
	}

	~FlmEditCommand( void)
	{
	}

	// Methods that must be implemented in classes that extend this class.

	FLMINT execute(
		FLMINT		iArgC,
		char **		ppszArgV,
		FlmShell *	pShell);

	void displayHelp(
		FlmShell *	pShell,
		char *		pszCommand);

	FLMBOOL canPerformCommand(
		char *	pszCommand);

private:
};

/*===========================================================================
Desc:		
===========================================================================*/
class DirectoryIterator
{
public:

	DirectoryIterator()
	{
		m_bInitialized = FALSE;
		m_pszBaseDir = NULL;
		m_pszExtendedDir = NULL;
		m_pszResolvedDir = NULL;
		m_pDirHdl = NULL;
		m_ppszMatchList = NULL;
		m_uiCurrentMatch = 0;
		m_uiTotalMatches = 0;
	}

	~DirectoryIterator()
	{
		reset();
	}

	FINLINE char * getResolvedPath()
	{
		return m_pszResolvedDir;
	}

	void reset();

	RCODE setupForSearch(
		IF_FileSystem *	pFileSystem,
		char *				pszBaseDir,
		char *				pszExtendedDir,
		char *				pszPattern);

	FINLINE FLMBOOL isInitialized()
	{
		return m_bInitialized;
	}

	void next(
		char *	pszReturn,
		FLMBOOL	bCompletePath);

	void prev(
		char *	pszReturn,
		FLMBOOL	bCompletePath);

	void first(
		char *	pszReturn,
		FLMBOOL	bCompletePath);

	void last(
		char *	pszReturn,
		FLMBOOL	bCompletePath);

	FLMBOOL isInSet(
		char *	pszFilename);

	FINLINE FLMBOOL isEmpty( void)
	{
		return m_uiTotalMatches == 0;
	}

private:
	enum {MAX_PATH_SIZE = 640};

	RCODE setupDirectories( 
		char *	pszBaseDir,
		char *	pszExtendedDir);

	static FINLINE FLMBOOL isQuoted(
		char *	pszString)
	{
		return pszString[0] == '\"' && 
			pszString[ f_strlen( pszString) - 1] == '\"'; 
	}

	FLMBOOL isDriveSpec(
		char *	pszPath);

	RCODE extractRoot(
		char *	pszPath,
		char *	pszRoot);

	RCODE resolveDir( void);

	FLMBOOL		m_bInitialized;
	char *		m_pszBaseDir;
	char *		m_pszExtendedDir;
	char *		m_pszResolvedDir;
	IF_DirHdl *	m_pDirHdl;
	char **		m_ppszMatchList;
	FLMUINT		m_uiCurrentMatch;
	FLMUINT		m_uiTotalMatches;
};

#endif 		// #ifndef DBSHELL_H
