//------------------------------------------------------------------------------
// Desc:	This file contains the F_DbSystem::dbRename method.
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

typedef struct
{
	char	szSrcFileName [F_PATH_MAX_SIZE];
	char	szDstFileName [F_PATH_MAX_SIZE];
} DB_RENAME_INFO, * DB_RENAME_INFO_p;

typedef struct DBRenameInfoTag
{
	DB_RENAME_INFO		Info;
	DBRenameInfoTag *	pNext;
} DBRenameInfo;

FSTATIC RCODE flmRenameFile(
	const char *			pszSrcFileName,
	const char *			pszDstFileName,
	FLMBOOL					bOverwriteDestOk,
	FLMBOOL					bPathNotFoundOk,
	DBRenameInfo **		ppRenameList,
	FLMBOOL *				pbFileFound,
	IF_DbRenameStatus *	ifpStatus);

/****************************************************************************
Desc:		Renames all files of a database
****************************************************************************/
RCODE F_DbSystem::dbRename(
	const char *			pszDbName,
		// [IN] Database to be renamed.
	const char *			pszDataDir,
		// [IN] Directory for data files.
	const char *			pszRflDir,
		// [IN] RFL directory of database. NULL can be
		// passed to indicate that the log files are located
		// in the same directory as the other database files.
	const char *			pszNewDbName,
		// [IN] New name to be given to the database.  May be
		// the short name only, or include a directory.  If it
		// includes a directory, it must be the same directory
		// as the directory given in pszDbName.
	FLMBOOL					bOverwriteDestOk,
		// [IN] Ok to overwrite existing file with rename?
	IF_DbRenameStatus *	ifpStatus)
		// [IN] Status callback function.
{
	RCODE					rc = NE_SFLM_OK;
	FLMUINT				uiFileNumber;
	DBRenameInfo *		pRenameList = NULL;
	FLMBOOL				bFileFound;
	char *				pszOldName = NULL;
	char *				pszNewName;
	char *				pszOldDataName;
	char *				pszNewDataName;
	char *				pszFullNewName;
	char					szOldBase [F_FILENAME_SIZE];
	char					szNewBase [F_FILENAME_SIZE];
	char *				pszExtOld;
	char *				pszExtNew;
	char *				pszDataExtOld;
	char *				pszDataExtNew;

	// Cannot handle empty database name.

	flmAssert( pszDbName && *pszDbName);
	flmAssert( pszNewDbName && *pszNewDbName);

	// Allocate memory for a read buffer, the log header, and various
	// file names.

	if (RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 5, &pszOldName)))
	{
		goto Exit;
	}
	pszNewName = pszOldName + F_PATH_MAX_SIZE;
	pszOldDataName = pszNewName + F_PATH_MAX_SIZE;
	pszNewDataName = pszOldDataName + F_PATH_MAX_SIZE;
	pszFullNewName = pszNewDataName + F_PATH_MAX_SIZE;

	// There must be either no directory specified for the new name, or
	// it must be identical to the old directory.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( 
		pszDbName, pszOldName, szOldBase)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathReduce( 
		pszNewDbName, pszNewName, szNewBase)))
	{
		goto Exit;
	}

	// Directories must be the same.

	if (*pszNewName && f_stricmp( pszOldName, pszNewName) != 0)
	{
		rc = RC_SET( NE_SFLM_INVALID_PARM);
		goto Exit;
	}
	f_strcpy( pszNewName, pszOldName);
	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
		pszNewName, szNewBase)))
	{
		goto Exit;
	}

	f_strcpy( pszFullNewName, pszNewName);
	f_strcpy( pszOldName, pszDbName);

	if (pszDataDir && *pszDataDir)
	{
		f_strcpy( pszOldDataName, pszDataDir);
		f_strcpy( pszNewDataName, pszDataDir);
		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
			pszOldDataName, szOldBase)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->pathAppend( 
			pszNewDataName, szNewBase)))
		{
			goto Exit;
		}
	}
	else
	{
		f_strcpy( pszNewDataName, pszNewName);
		f_strcpy( pszOldDataName, pszOldName);
	}

	// First make sure we have closed the databases and gotten rid of
	// them from our internal memory tables - in case they had been open.

	if (RC_BAD( rc = checkDatabaseClosed( pszDbName, pszDataDir)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = checkDatabaseClosed( pszFullNewName, pszDataDir)))
	{
		goto Exit;
	}

	// Start renaming files, beginning with the main DB file.

	if (RC_BAD( rc = flmRenameFile( pszDbName, pszFullNewName,
								bOverwriteDestOk, FALSE,
								&pRenameList, &bFileFound,
								ifpStatus)))
	{
		goto Exit;
	}

	// Find where the extension of the old and new database names are

	pszExtOld = pszOldName + f_strlen( pszOldName) - 1;
	pszDataExtOld = pszOldDataName + f_strlen( pszOldDataName) - 1;
	while (pszExtOld != pszOldName && *pszExtOld != '.')
	{
		pszExtOld--;

		// Both the old db name and old data name have the same
		// base name, so we can decrement pszDataExtOld
		// at the same time we decrement pszExtOld.

		pszDataExtOld--;
	}
	if (*pszExtOld != '.')
	{
		pszExtOld = pszOldName + f_strlen( pszOldName);
		pszDataExtOld = pszOldDataName + f_strlen( pszOldDataName);
	}

	pszExtNew = pszNewName + f_strlen( pszNewName) - 1;
	pszDataExtNew = pszNewDataName + f_strlen( pszNewDataName) - 1;
	while (pszExtNew != pszOldName && *pszExtNew != '.')
	{
		pszExtNew--;

		// Both the new db name and new data name have the same
		// base name, so we can decrement pszDataExtNew
		// at the same time we decrement pszExtNew.

		pszDataExtNew--;
	}
	if (*pszExtNew != '.')
	{
		pszExtNew = pszNewName + f_strlen( pszNewName);
		pszDataExtNew = pszNewDataName + f_strlen( pszNewDataName);
	}

	// Rename the .lck file, if any.  This is necessary for UNIX.

	f_strcpy( pszExtOld, ".lck");
	f_strcpy( pszExtNew, ".lck");
	if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
								bOverwriteDestOk, TRUE,
								&pRenameList, &bFileFound,
								ifpStatus)))
	{
		goto Exit;
	}

	// Rename block (data) files.

	uiFileNumber = 1;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszDataExtOld);
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszDataExtNew);

		if (RC_BAD( rc = flmRenameFile( pszOldDataName, pszNewDataName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									ifpStatus)))
		{
			goto Exit;
		}
		if (!bFileFound)
		{
			break;
		}
		if (uiFileNumber == MAX_DATA_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiFileNumber++;
	}

	// Rename rollback log files.

	uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszExtOld);
		F_SuperFileClient::bldSuperFileExtension( uiFileNumber, pszExtNew);

		if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									ifpStatus)))
		{
			goto Exit;
		}
		if (!bFileFound)
		{
			break;
		}
		if (uiFileNumber == MAX_LOG_BLOCK_FILE_NUMBER)
		{
			break;
		}
		uiFileNumber++;
	}

	// Rename the RFL directory.

	if (RC_BAD( rc = rflGetDirAndPrefix( pszDbName, pszRflDir, pszOldName)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = rflGetDirAndPrefix( pszFullNewName, pszRflDir,
									pszNewName)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
								bOverwriteDestOk, TRUE,
								&pRenameList, &bFileFound,
								ifpStatus)))
	{
		goto Exit;
	}

Exit:
	if (pszOldName)
	{
		f_free( &pszOldName);
	}

	// Free the list of renamed files.

	while (pRenameList)
	{
		DBRenameInfo *		pRenameFile;

		pRenameFile = pRenameList;
		pRenameList = pRenameList->pNext;

		// If we had an error of some sort, attempt to un-rename
		// the file that had been renamed.

		if (RC_BAD( rc))
		{
			gv_SFlmSysData.pFileSystem->renameFile( 
				pRenameFile->Info.szDstFileName, pRenameFile->Info.szSrcFileName);
		}
		f_free( &pRenameFile);
	}
	return( rc);
}

/****************************************************************************
Desc:	Rename a database file and add to list of renamed files.
****************************************************************************/
FSTATIC RCODE flmRenameFile(
	const char *			pszSrcFileName,
	const char *			pszDstFileName,
	FLMBOOL					bOverwriteDestOk,
	FLMBOOL					bPathNotFoundOk,
	DBRenameInfo **		ppRenameList,
	FLMBOOL *				pbFileFound,
	IF_DbRenameStatus *	ifpStatus)
{
	RCODE				rc = NE_SFLM_OK;
	DBRenameInfo *	pRenameFile = NULL;

	*pbFileFound = FALSE;

	// Should not do anything if the source and destination names
	// are the same.

	if (f_stricmp( pszSrcFileName, pszDstFileName) == 0)
	{
		if (gv_SFlmSysData.pFileSystem->doesFileExist( 
			pszSrcFileName) == NE_SFLM_OK)
		{
			*pbFileFound = TRUE;
		}
		goto Exit;
	}

	if (RC_BAD( rc = f_alloc( sizeof( DBRenameInfo), &pRenameFile)))
	{
		goto Exit;
	}

	// If a destination file exists, and it is OK to overwrite
	// it, it must be deleted.

	if (bOverwriteDestOk)
	{
		if (gv_SFlmSysData.pFileSystem->isDir( pszDstFileName))
		{
			if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->removeDir( 
				pszDstFileName, TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->deleteFile( 
				pszDstFileName)))
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
		}
	}

	// If names are the same, no need to actually do the
	// rename.

	if (RC_BAD( rc = gv_SFlmSysData.pFileSystem->renameFile( 
			pszSrcFileName, pszDstFileName)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_FILENAME)
		{
			if (bPathNotFoundOk)
			{
				rc = NE_SFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		*pbFileFound = TRUE;
		pRenameFile->pNext = *ppRenameList;
		*ppRenameList = pRenameFile;

		// Do user callback.  User could choose to stop the rename
		// from continuing.
		if (ifpStatus)
		{
			f_strcpy( pRenameFile->Info.szSrcFileName, pszSrcFileName);
			f_strcpy( pRenameFile->Info.szDstFileName, pszDstFileName);
			if (RC_BAD( rc = ifpStatus->dbRenameStatus(
					pRenameFile->Info.szSrcFileName,
					pRenameFile->Info.szDstFileName)))
			{
				goto Exit;
			}
		}

		// So it won't get deallocated at exit.

		pRenameFile = NULL;
	}
Exit:
	if (pRenameFile)
	{
		f_free( &pRenameFile);
	}
	return( rc);
}
