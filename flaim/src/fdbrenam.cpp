//-------------------------------------------------------------------------
// Desc:	Rename a database.
// Tabs:	3
//
// Copyright (c) 2001,2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

typedef struct DBRenameInfo
{
	DB_RENAME_INFO		Info;
	DBRenameInfo *		pNext;
} DBRenameInfo;

FSTATIC RCODE flmRenameFile(
	const char *		pszSrcFileName,
	const char *		pszDstFileName,
	FLMBOOL				bOverwriteDestOk,
	FLMBOOL				bPathNotFoundOk,
	DBRenameInfo **	ppRenameList,
	FLMBOOL *			pbFileFound,
	STATUS_HOOK			fnStatusCallback,
	void *				UserData);

/****************************************************************************
Desc:	Rename a database file and add to list of renamed files.
****************************************************************************/
FSTATIC RCODE flmRenameFile(
	const char *		pszSrcFileName,
	const char *		pszDstFileName,
	FLMBOOL				bOverwriteDestOk,
	FLMBOOL				bPathNotFoundOk,
	DBRenameInfo **	ppRenameList,
	FLMBOOL *			pbFileFound,
	STATUS_HOOK			fnStatusCallback,
	void *				UserData)
{
	RCODE				rc = FERR_OK;
	DBRenameInfo *	pRenameFile = NULL;

	*pbFileFound = FALSE;

	// Should not do anything if the source and destination names
	// are the same.

	if (f_stricmp( pszSrcFileName, pszDstFileName) == 0)
	{
		if (RC_OK( gv_FlmSysData.pFileSystem->doesFileExist( pszSrcFileName)))
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
		if (gv_FlmSysData.pFileSystem->isDir( pszDstFileName))
		{
			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->removeDir(
											pszDstFileName, TRUE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile(
											pszDstFileName)))
			{
				if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
				{
					rc = FERR_OK;
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

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->renameFile( 
		pszSrcFileName, pszDstFileName)))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			if (bPathNotFoundOk)
			{
				rc = FERR_OK;
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

		f_strcpy( pRenameFile->Info.szSrcFileName, pszSrcFileName);
		f_strcpy( pRenameFile->Info.szDstFileName, pszDstFileName);
		if (fnStatusCallback)
		{
			if (RC_BAD( rc = (*fnStatusCallback)( FLM_DB_RENAME_STATUS,
										(void *)&pRenameFile->Info,
										(void *)0, UserData)))
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

/*******************************************************************************
Desc:	Renames a database
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmDbRename(
	const char *		pszDbName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszNewDbName,
	FLMBOOL				bOverwriteDestOk,
	STATUS_HOOK			fnStatusCallback,
	void *				UserData)
{
	RCODE					rc = FERR_OK;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT				uiFileNumber;
	FILE_HDR				FileHdr;
	LOG_HDR				LogHdr;
	DBRenameInfo *		pRenameList = NULL;
	FLMBOOL				bFileFound;
	FLMBYTE *			pucBuffer = NULL;
	FLMBYTE *			pucLogHdr;
	char *				pszOldName;
	char *				pszNewName;
	char *				pszOldDataName;
	char *				pszNewDataName;
	char *				pszFullNewName;
	char					szOldBase[ F_FILENAME_SIZE];
	char					szNewBase[ F_FILENAME_SIZE];
	char *				pszExtOld;
	char *				pszExtNew;
	char *				pszDataExtOld;
	char *				pszDataExtNew;

	// Cannot handle empty database name.

	flmAssert( pszDbName && *pszDbName);
	flmAssert( pszNewDbName && *pszNewDbName);

	// Allocate memory for a read buffer, the log header, and various
	// file names.

	if( RC_BAD( rc = f_allocAlignedBuffer( 
		2048 + LOG_HEADER_SIZE + F_PATH_MAX_SIZE * 5, &pucBuffer)))
	{
		goto Exit;
	}
	pucLogHdr = pucBuffer + 2048;
	pszOldName = (char *)(pucLogHdr + LOG_HEADER_SIZE);
	pszNewName = pszOldName + F_PATH_MAX_SIZE;
	pszOldDataName = pszNewName + F_PATH_MAX_SIZE;
	pszNewDataName = pszOldDataName + F_PATH_MAX_SIZE;
	pszFullNewName = pszNewDataName + F_PATH_MAX_SIZE;

	// There must be either no directory specified for the new name, or
	// it must be identical to the old directory.

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
		pszDbName, pszOldName, szOldBase)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
		pszNewDbName, pszNewName, szNewBase)))
	{
		goto Exit;
	}

	// Directories must be the same.

	if (*pszNewName && f_stricmp( pszOldName, pszNewName) != 0)
	{
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	
	f_strcpy( pszNewName, pszOldName);
	
	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
		pszNewName, szNewBase)))
	{
		goto Exit;
	}

	f_strcpy( pszFullNewName, pszNewName);
	f_strcpy( pszOldName, pszDbName);

	if( pszDataDir && *pszDataDir)
	{
		f_strcpy( pszOldDataName, pszDataDir);
		f_strcpy( pszNewDataName, pszDataDir);
		
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
			pszOldDataName, szOldBase)))
		{
			goto Exit;
		}
		
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
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

	if( RC_BAD( rc = FlmConfig( FLM_CLOSE_FILE, 
		(void *)pszDbName, (void *)pszDataDir)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FlmConfig( FLM_CLOSE_FILE, 
		(void *)pszFullNewName, (void *)pszDataDir)))
	{
		goto Exit;
	}
	
	gv_FlmSysData.pFileHdlCache->closeUnusedFiles();

	// Open the file so we can get the log header.

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( pszDbName, 
		gv_FlmSysData.uiFileOpenFlags, &pFileHdl)))
	{
		goto Exit;
	}

	// Read the header to get the low and high RFL log
	// file numbers.

	if (RC_BAD( flmReadAndVerifyHdrInfo( NULL, pFileHdl,
								pucBuffer, &FileHdr, &LogHdr, pucLogHdr)))
	{
		goto Exit;
	}

	// Close the file.

	pFileHdl->Release();
	pFileHdl = NULL;

	// Start renaming files, beginning with the main DB file.

	if( RC_BAD( rc = flmRenameFile( pszDbName, pszFullNewName,
								bOverwriteDestOk, FALSE,
								&pRenameList, &bFileFound,
								fnStatusCallback, UserData)))
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
								fnStatusCallback, UserData)))
	{
		goto Exit;
	}

	// Rename block (data) files.

	uiFileNumber = 1;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( FileHdr.uiVersionNum,
			uiFileNumber, pszDataExtOld);
		F_SuperFileClient::bldSuperFileExtension( FileHdr.uiVersionNum,
			uiFileNumber, pszDataExtNew);

		if (RC_BAD( rc = flmRenameFile( pszOldDataName, pszNewDataName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									fnStatusCallback, UserData)))
		{
			goto Exit;
		}
		if (!bFileFound)
		{
			break;
		}
		if (uiFileNumber ==
				MAX_DATA_BLOCK_FILE_NUMBER( FileHdr.uiVersionNum))
		{
			break;
		}
		uiFileNumber++;
	}

	// Rename rollback log files.

	uiFileNumber =
			FIRST_LOG_BLOCK_FILE_NUMBER (FileHdr.uiVersionNum);
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( FileHdr.uiVersionNum,
			uiFileNumber, pszExtOld);
		F_SuperFileClient::bldSuperFileExtension( FileHdr.uiVersionNum,
			uiFileNumber, pszExtNew);

		if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									fnStatusCallback, UserData)))
		{
			goto Exit;
		}
		
		if (!bFileFound)
		{
			break;
		}
		
		if (uiFileNumber ==
				MAX_LOG_BLOCK_FILE_NUMBER( FileHdr.uiVersionNum))
		{
			break;
		}
		
		uiFileNumber++;
	}

	// Rename roll-forward log files.

	if (FileHdr.uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
	{

		// For pre-4.3 versions, only need to rename one RFL file.

		if (RC_BAD( rc = rflGetFileName( FileHdr.uiVersionNum,
									pszDbName, pszRflDir, 1, pszOldName)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = rflGetFileName( FileHdr.uiVersionNum,
									pszFullNewName, pszRflDir, 1, pszNewName)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									fnStatusCallback, UserData)))
		{
			goto Exit;
		}
	}
	else
	{

		// For 4.3 and greater, rename the RFL directory.

		if (RC_BAD( rc = rflGetDirAndPrefix( FileHdr.uiVersionNum,
									pszDbName, pszRflDir, pszOldName, szOldBase)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = rflGetDirAndPrefix( FileHdr.uiVersionNum,
									pszFullNewName, pszRflDir, pszNewName,
									szNewBase)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = flmRenameFile( pszOldName, pszNewName,
									bOverwriteDestOk, TRUE,
									&pRenameList, &bFileFound,
									fnStatusCallback, UserData)))
		{
			goto Exit;
		}
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pucBuffer)
	{
		f_freeAlignedBuffer( &pucBuffer);
	}

	// Free the list of renamed files.

	while( pRenameList)
	{
		DBRenameInfo *		pRenameFile;

		pRenameFile = pRenameList;
		pRenameList = pRenameList->pNext;

		// If we had an error of some sort, attempt to un-rename
		// the file that had been renamed.

		if (RC_BAD( rc))
		{
			gv_FlmSysData.pFileSystem->renameFile( 
				pRenameFile->Info.szDstFileName, pRenameFile->Info.szSrcFileName);
		}
		
		f_free( &pRenameFile);
	}
	
	return( rc);
}
