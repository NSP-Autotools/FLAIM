//------------------------------------------------------------------------------
// Desc:	This file contains the F_DbSystem::dropDatabase method.
// Tabs:	3
//
// Copyright (c) 2001-2007 Novell, Inc. All Rights Reserved.
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

/****************************************************************************
Desc:	Drops a database - all physical files will be deleted.
****************************************************************************/
RCODE F_DbSystem::dropDatabase(
	const char *		pszDbName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMBOOL				bRemoveRflFiles)
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiFileNumber;
	char *				pszTmpName = NULL;
	char *				pszRflDirName;
	char *				pszDataName;
	char *				pszBaseName;
	char *				pszExt;
	char *				pszDataExt;
	IF_DirHdl *			pDirHdl = NULL;

	// Cannot handle empty database name.

	if( !pszDbName || !(*pszDbName))
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	// Allocate memory, so as to not consume stack.

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 3 + F_FILENAME_SIZE,
		&pszTmpName)))
	{
		goto Exit;
	}

	pszRflDirName = pszTmpName + F_PATH_MAX_SIZE;
	pszDataName = pszRflDirName + F_PATH_MAX_SIZE;
	pszBaseName = pszDataName + F_PATH_MAX_SIZE;

	// First make sure we have closed this database and gotten rid of
	// it from our internal memory tables - in case it had been open.

	if (RC_BAD( rc = checkDatabaseClosed( pszDbName, pszDataDir)))
	{
		goto Exit;
	}

	if (pszDataDir && *pszDataDir)
	{
		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( 
			pszDbName, pszDataName, pszBaseName)))
		{
			goto Exit;
		}
		f_strcpy( pszDataName, pszDataDir);
		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
			pszDataName, pszBaseName)))
		{
			goto Exit;
		}
	}
	else
	{
		f_strcpy( pszDataName, pszDbName);
	}
	f_strcpy( pszTmpName, pszDbName);

	// Start deleting files, beginning with the main DB file.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->deleteFile( pszDbName)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// Find where the extension of the database name is

	pszExt = pszTmpName + f_strlen( pszTmpName) - 1;
	pszDataExt = pszDataName + f_strlen( pszDataName) - 1;
	while (pszExt != pszTmpName && *pszExt != '.')
	{
		pszExt--;

		// Both the db name and data name have the same
		// base name, so we can decrement pszDataExt
		// at the same time we decrement pszExt.

		pszDataExt--;
	}
	if (*pszExt != '.')
	{
		pszExt = pszTmpName + f_strlen( pszTmpName);
		pszDataExt = pszDataName + f_strlen( pszDataName);
	}

	// Delete the .lck file, if any

	f_strcpy( pszExt, ".lck");
	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->deleteFile( pszTmpName)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			rc = NE_SFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	// Delete block (data) files.

	uiFileNumber = 1;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszDataExt);

		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->deleteFile( pszDataName)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		if (uiFileNumber == MAX_DATA_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiFileNumber++;
	}

	// Delete rollback log files.

	uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszExt);

		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->deleteFile( pszTmpName)))
		{
			if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		if (uiFileNumber == MAX_LOG_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiFileNumber++;
	}

	if (bRemoveRflFiles)
	{

		// Delete roll-forward log files.

		FLMBOOL	bCanDeleteDir;

		// Scan the RFL directory for RFL files.

		if (RC_BAD( rc = rflGetDirAndPrefix( pszDbName, pszRflDir, pszRflDirName)))
		{
			goto Exit;
		}

		// See if the directory exists.  If not, we are done.

		if (!gv_SFlmSysData.pFileSystem->isDir( pszRflDirName))
		{
			goto Exit;	// Should return NE_SFLM_OK
		}

		// Open the directory and scan for RFL files.
		// NOTE: DO NOT just call removeDir.  There may be other
		// things in the directory that we do not want to delete.
		// Look specifically for files that match our expected
		// name format for RFL files.

		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->openDir( pszRflDirName,
										"*", &pDirHdl)))
		{
			goto Exit;
		}

		// Assume that we can delete the directory.  This will only
		// be set to FALSE if we can't delete all of the files in
		// the directory - i.e., some don't look like RFL log files.

		bCanDeleteDir = TRUE;
		for (;;)
		{
			if (RC_BAD( rc = pDirHdl->next()))
			{
				if (rc == NE_FLM_IO_NO_MORE_FILES)
				{
					rc = NE_SFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}
			pDirHdl->currentItemPath( pszTmpName);
			if (pDirHdl->currentItemIsDir())
			{
				bCanDeleteDir = FALSE;
			}
			else if (!rflGetFileNum( pszTmpName, &uiFileNumber))
			{
				bCanDeleteDir = FALSE;
			}
			else
			{
				if( RC_BAD( rc =
							gv_SFlmSysData.pFileSystem->deleteFile( pszTmpName)))
				{
					if (rc == NE_FLM_IO_PATH_NOT_FOUND ||
						 rc == NE_FLM_IO_INVALID_FILENAME)
					{
						rc = NE_SFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}

		// Attempt to delete the directory - if allowed.

		if (bCanDeleteDir)
		{

			// Need to release the directory handle so the
			// directory will be closed when we try to delete it
			// below.

			if (pDirHdl)
			{
				pDirHdl->Release();
				pDirHdl = NULL;
			}
			
			if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->removeDir( pszRflDirName)))
			{
				if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
				{
					rc = NE_SFLM_OK;
				}
				goto Exit;
			}
		}
	}

Exit:

	if( pszTmpName)
	{
		f_free( &pszTmpName);
	}

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the drop database statement.  The "DROP DATABASE" keywords
//			have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processDropDatabase( void)
{
	RCODE					rc = NE_SFLM_OK;
	char					szDatabaseName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDatabaseNameLen;
	char					szDataDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiDataDirNameLen;
	char					szRflDirName [F_PATH_MAX_SIZE + 1];
	FLMUINT				uiRflDirNameLen;
	FLMBOOL				bRemoveRflFiles;
	F_DbSystem			dbSystem;
	char					szToken [MAX_SQL_TOKEN_SIZE + 1];
	FLMUINT				uiTokenLineOffset;
	
	// SYNTAX: DROP DATABASE databasename
	// [DATA_DIR=<DataDirName>] [RFL_DIR=<RflDirName>]
	// [REMOVE_RFL_FILES]

	// Whitespace must follow the "DROP DATABASE"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the database name.

	if (RC_BAD( rc = getUTF8String( FALSE, TRUE, (FLMBYTE *)szDatabaseName,
							sizeof( szDatabaseName),
							&uiDatabaseNameLen, NULL, NULL)))
	{
		goto Exit;
	}
	
	szDataDirName [0] = 0;
	szRflDirName [0] = 0;
	bRemoveRflFiles = FALSE;
	
	// See if there are any options
	
	for (;;)
	{
		if (RC_BAD( rc = getToken( szToken, sizeof( szToken), TRUE,
									&uiTokenLineOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if (f_stricmp( szToken, SFLM_DATA_DIR_STR) == 0)
		{
			if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szDataDirName,
									sizeof( szDataDirName),
									&uiDataDirNameLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, SFLM_RFL_DIR_STR) == 0)
		{
			if (RC_BAD( rc = getUTF8String( TRUE, TRUE, (FLMBYTE *)szRflDirName,
									sizeof( szRflDirName),
									&uiRflDirNameLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else if (f_stricmp( szToken, "remove_rfl_files") == 0)
		{
			bRemoveRflFiles = TRUE;
		}
		else
		{
			// Move the line offset back to the beginning of the token
			// so it can be processed by the next SQL statement in the
			// stream.
			
			m_uiCurrLineOffset = uiTokenLineOffset;
			break;
		}
	}
	
	// Drop the database
	
	if (RC_BAD( rc = dbSystem.dropDatabase( szDatabaseName, szDataDirName,
										szRflDirName, bRemoveRflFiles)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

