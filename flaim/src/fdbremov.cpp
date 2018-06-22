//-------------------------------------------------------------------------
// Desc:	Delete a database.
// Tabs:	3
//
// Copyright (c) 2001-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

/*******************************************************************************
Desc:	Removes a database, including roll-forward log files, if requested.
*******************************************************************************/
FLMEXP RCODE FLMAPI FlmDbRemove(
	const char *		pszDbName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMBOOL				bRemoveRflFiles)
{
	RCODE					rc = FERR_OK;
	IF_FileHdl *		pFileHdl = NULL;
	FLMBYTE *			pucBuffer = NULL;
	FLMUINT				uiFileNumber;
	FILE_HDR				FileHdr;
	FLMUINT				uiVersionNum;
	char *				pszTmpName = NULL;
	char *				pszRflDirName;
	char *				pszDataName;
	char *				pszBaseName;
	char					szPrefix[ F_FILENAME_SIZE];
	char *				pszExt;
	char *				pszDataExt;
	IF_DirHdl *			pDirHdl = NULL;

	// Cannot handle empty database name.

	if( !pszDbName || !(*pszDbName))
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
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

	if (RC_BAD( rc = FlmConfig( FLM_CLOSE_FILE, (void *)pszDbName,
								(void *)pszDataDir)))
	{
		goto Exit;
	}

	gv_FlmSysData.pFileHdlCache->closeUnusedFiles();

	// Open the file so we can get the log header.

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->openFile( 
		pszDbName, gv_FlmSysData.uiFileOpenFlags, &pFileHdl)))
	{
		goto Exit;
	}

	// Allocate a buffer for reading the header stuff.
	
	if( RC_BAD( rc = f_allocAlignedBuffer( 2048, &pucBuffer)))
	{
		goto Exit;
	}

	// Read the header to get the low and high RFL log
	// file numbers.

	if (RC_BAD( flmReadAndVerifyHdrInfo( NULL, pFileHdl,
								pucBuffer, &FileHdr, NULL, NULL)))
	{
		uiVersionNum = FLM_CUR_FILE_FORMAT_VER_NUM;
	}
	else
	{
		uiVersionNum = FileHdr.uiVersionNum;
	}

	// Close the file.

	pFileHdl->Release();
	pFileHdl = NULL;

	if (pszDataDir && *pszDataDir)
	{
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
			pszDbName, pszDataName, pszBaseName)))
		{
			goto Exit;
		}
		
		f_strcpy( pszDataName, pszDataDir);
		
		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->pathAppend( 
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

	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile( pszDbName)))
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
	if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile( pszTmpName)))
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

	// Delete block (data) files.

	uiFileNumber = 1;
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( 
			uiVersionNum, uiFileNumber, pszDataExt);

		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile( pszDataName)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if (uiFileNumber == MAX_DATA_BLOCK_FILE_NUMBER( uiVersionNum))
		{
			break;
		}
		uiFileNumber++;
	}

	// Delete rollback log files.

	uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER( uiVersionNum);
	for (;;)
	{
		F_SuperFileClient::bldSuperFileExtension( 
			uiVersionNum, uiFileNumber, pszExt);

		if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile( pszTmpName)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if (uiFileNumber == MAX_LOG_BLOCK_FILE_NUMBER( uiVersionNum))
		{
			break;
		}
		
		uiFileNumber++;
	}

	if (bRemoveRflFiles)
	{

		// Delete roll-forward log files.

		if (uiVersionNum < FLM_FILE_FORMAT_VER_4_3)
		{

			// For pre-4.3 versions, only need to delete one RFL file.

			if (RC_BAD( rc = rflGetFileName( uiVersionNum, pszDbName, 
				pszRflDir, 1, pszTmpName)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->deleteFile( pszTmpName)))
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
		else
		{
			FLMBOOL	bCanDeleteDir;

			// For 4.3 and greater, need to scan the RFL directory for
			// RFL files.

			if (RC_BAD( rc = rflGetDirAndPrefix( uiVersionNum,
										pszDbName, pszRflDir, pszRflDirName, szPrefix)))
			{
				goto Exit;
			}

			// See if the directory exists.  If not, we are done.

			if (!gv_FlmSysData.pFileSystem->isDir( pszRflDirName))
			{
				goto Exit;	// Should return FERR_OK
			}

			// Open the directory and scan for RFL files.
			// NOTE: DO NOT just call RemoveDir.  There may be other
			// things in the directory that we do not want to delete.
			// Look specifically for files that match our expected
			// name format for RFL files.

			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->openDir( pszRflDirName,
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
				
				pDirHdl->currentItemPath( pszTmpName);
				
				if (pDirHdl->currentItemIsDir())
				{
					bCanDeleteDir = FALSE;
				}
				else if (!rflGetFileNum( uiVersionNum,
													szPrefix, pszTmpName, &uiFileNumber))
				{
					bCanDeleteDir = FALSE;
				}
				else
				{
					if( RC_BAD( rc =
								gv_FlmSysData.pFileSystem->deleteFile( pszTmpName)))
					{
						if (rc == FERR_IO_PATH_NOT_FOUND ||
							 rc == FERR_IO_INVALID_PATH)
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

				if (RC_BAD( rc = gv_FlmSysData.pFileSystem->removeDir( 
					pszRflDirName)))
				{
					if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
					{
						rc = FERR_OK;
					}
					goto Exit;
				}
			}
		}
	}

Exit:

	if( pszTmpName)
	{
		f_free( &pszTmpName);
	}
	
	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pucBuffer)
	{
		f_freeAlignedBuffer( &pucBuffer);
	}
	
	if( pDirHdl)
	{
		pDirHdl->Release();
	}
	
	return( rc);
}
