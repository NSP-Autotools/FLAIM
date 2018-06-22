//------------------------------------------------------------------------------
// Desc: This file contains the F_64BitFile class
// Tabs:	3
//
// Copyright (c) 2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

#define F_MULTI_FHDL_LIST_SIZE								8
#define F_MULTI_FHDL_DEFAULT_MAX_FILE_SIZE				((FLMUINT)0xFFFFFFFF)

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	IF_FileHdl *	pFileHdl;
	FLMUINT			uiFileNum;
	FLMBOOL			bDirty;
} FH_INFO;

/****************************************************************************
Desc:
****************************************************************************/
class F_MultiFileHdl : public IF_MultiFileHdl
{
public:

	F_MultiFileHdl(
		FLMUINT			uiMaxFileSize = F_MULTI_FHDL_DEFAULT_MAX_FILE_SIZE);

	virtual ~F_MultiFileHdl();

	void FTKAPI closeFile(
		FLMBOOL			bDelete = FALSE);

	RCODE FTKAPI createFile(
		const char *	pszPath);

	RCODE FTKAPI createUniqueFile(
		const char *	pszPath,
		const char *	pszFileExtension);

	RCODE FTKAPI deleteMultiFile(
		const char *	pszPath);

	RCODE FTKAPI openFile(
		const char *	pszPath);

	RCODE FTKAPI flush( void);

	RCODE FTKAPI read(
		FLMUINT64	ui64Offset,
		FLMUINT		uiLength,
		void *		pvBuffer,
		FLMUINT *	puiBytesRead);

	RCODE FTKAPI write(
		FLMUINT64	ui64Offset,
		FLMUINT		uiLength,
		void *		pvBuffer,
		FLMUINT *	puiBytesWritten);

	RCODE FTKAPI getPath(
		char *	pszFilePath);

	FINLINE RCODE FTKAPI size(
		FLMUINT64 *	pui64FileSize)
	{
		*pui64FileSize = m_ui64EOF;
		return( NE_FLM_OK);
	}

	RCODE FTKAPI truncateFile(
		FLMUINT64	ui64NewSize);

private:

	RCODE getFileHdl(
		FLMUINT				uiFileNum,
		FLMBOOL				bGetForWrite,
		IF_FileHdl **		ppFileHdl);

	RCODE createLockFile(
		const char *		pszBasePath);

	FINLINE void releaseLockFile(
		const char *		pszBasePath,
		FLMBOOL				bDelete)
	{
#ifndef FLM_UNIX
		F_UNREFERENCED_PARM( bDelete);
		F_UNREFERENCED_PARM( pszBasePath);
#endif

		if( m_pLockFileHdl)
		{

			// Release the lock file

			(void)m_pLockFileHdl->closeFile();
			m_pLockFileHdl->Release();
			m_pLockFileHdl = NULL;

#ifdef FLM_UNIX
			if( bDelete)
			{
				IF_FileSystem *	pFileSystem = f_getFileSysPtr();
				char					szTmpPath[ F_PATH_MAX_SIZE];

				// Delete the lock file

				f_strcpy( szTmpPath, pszBasePath);
				pFileSystem->pathAppend( szTmpPath, "64.LCK");
				pFileSystem->deleteFile( szTmpPath);
			}
#endif
		}
	}

	FINLINE void formatFileNum(
		FLMUINT	uiFileNum,
		char *	pszStr)
	{
		f_sprintf( pszStr, "%08X.64", (unsigned)uiFileNum);
	}

	RCODE getFileNum(
		const char *	pszFileName,
		FLMUINT *		puiFileNum);

	FINLINE void dataFilePath(
		FLMUINT		uiFileNum,
		char *		pszPath)
	{
		char					szFileName[ 13];
		IF_FileSystem *	pFileSystem = f_getFileSysPtr();

		f_strcpy( pszPath, m_szPath);
		formatFileNum( uiFileNum, szFileName);
		pFileSystem->pathAppend( pszPath, szFileName);
	}

	FINLINE FLMUINT getFileNum(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset / m_uiMaxFileSize));
	}

	FINLINE FLMUINT getFileOffset(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset % m_uiMaxFileSize));
	}

	FH_INFO				m_pFileHdlList[ F_MULTI_FHDL_LIST_SIZE];
	char					m_szPath[ F_PATH_MAX_SIZE];
	FLMBOOL				m_bOpen;
	FLMUINT64			m_ui64EOF;
	FLMUINT				m_uiMaxFileSize;
	IF_FileHdl *		m_pLockFileHdl;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmAllocMultiFileHdl(
	IF_MultiFileHdl **		ppFileHdl)
{
	if( (*ppFileHdl = f_new F_MultiFileHdl) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:
****************************************************************************/
F_MultiFileHdl::F_MultiFileHdl(
	FLMUINT			uiMaxFileSize)
{
	m_bOpen = FALSE;
	m_szPath[ 0] = 0;
	m_ui64EOF = 0;
	m_pLockFileHdl = NULL;
	f_memset( m_pFileHdlList, 0, sizeof( FH_INFO) * F_MULTI_FHDL_LIST_SIZE);
	m_uiMaxFileSize = uiMaxFileSize;
	
	if( !m_uiMaxFileSize)
	{
		m_uiMaxFileSize = F_MULTI_FHDL_DEFAULT_MAX_FILE_SIZE;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_MultiFileHdl::~F_MultiFileHdl()
{
	if( m_bOpen)
	{
		closeFile();
	}

	f_assert( !m_pLockFileHdl);
}

/****************************************************************************
Desc:	Closes all data files associated with the object
****************************************************************************/
void F_MultiFileHdl::closeFile(
	FLMBOOL			bDelete)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLoop;
	IF_DirHdl *			pDir = NULL;
	char					szTmpPath[ F_PATH_MAX_SIZE];
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( !m_bOpen)
	{
		return;
	}

	for( uiLoop = 0; uiLoop < F_MULTI_FHDL_LIST_SIZE; uiLoop++)
	{
		if( m_pFileHdlList[ uiLoop].pFileHdl)
		{
			if( m_pFileHdlList[ uiLoop].bDirty)
			{
				(void)m_pFileHdlList[ uiLoop].pFileHdl->flush();
			}
			m_pFileHdlList[ uiLoop].pFileHdl->closeFile();
			m_pFileHdlList[ uiLoop].pFileHdl->Release();
			f_memset( &m_pFileHdlList[ uiLoop], 0, sizeof( FH_INFO));
		}
	}

	m_ui64EOF = 0;
	m_bOpen = FALSE;

	if( bDelete)
	{
		if( RC_OK( pFileSystem->openDir(
			m_szPath, "*.64", &pDir)))
		{
			// Remove all data files

			for( rc = pDir->next(); !RC_BAD( rc) ; rc = pDir->next())
			{
				pDir->currentItemPath( szTmpPath);
				f_assert( f_strstr( szTmpPath, ".64") != 0);
				(void)pFileSystem->deleteFile( szTmpPath);
			}

			pDir->Release();
			pDir = NULL;
		}

		// Release and delete the lock file

		(void)releaseLockFile( m_szPath, TRUE);

		// Remove the directory

		(void)pFileSystem->removeDir( m_szPath);
	}
	else
	{
		(void)releaseLockFile( m_szPath, FALSE);
	}
}

/****************************************************************************
Desc:	Removes a 64-bit file
****************************************************************************/
RCODE F_MultiFileHdl::deleteMultiFile(
	const char *	pszPath)
{
	RCODE					rc = NE_FLM_OK;
	IF_DirHdl *			pDir = NULL;
	char					szTmpPath[ F_PATH_MAX_SIZE];
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	// Can't use this handle to delete something if we already
	// have a file open.

	if( m_bOpen)
	{

		// Can't jump to exit, because it calls releaseLockFile

		return( RC_SET_AND_ASSERT( NE_FLM_FAILURE));
	}

	if( RC_BAD( rc = pFileSystem->doesFileExist( pszPath)))
	{
		goto Exit;
	}

	if( !pFileSystem->isDir( pszPath))
	{
		// If the path specifies a single file rather than a
		// 64-bit directory, just go ahead and delete the file.

		rc = pFileSystem->deleteFile( pszPath);
		goto Exit;
	}

	if( RC_BAD( rc = createLockFile( pszPath)))
	{
		goto Exit;
	}

	if( RC_OK( pFileSystem->openDir( pszPath, "*.64", &pDir)))
	{
		// Remove all data files

		for( rc = pDir->next(); !RC_BAD( rc) ; rc = pDir->next())
		{
			pDir->currentItemPath( szTmpPath);
			f_assert( f_strstr( szTmpPath, ".64") != 0);
			(void)pFileSystem->deleteFile( szTmpPath);
		}

		pDir->Release();
		pDir = NULL;
		rc = NE_FLM_OK;
	}

	// Release and delete the lock file

	(void)releaseLockFile( pszPath, TRUE);

	// Remove the directory

	(void)pFileSystem->removeDir( pszPath);

Exit:

	(void)releaseLockFile( pszPath, FALSE);

	return( rc);
}

/****************************************************************************
Desc: Creates a new 64-bit "file"
****************************************************************************/
RCODE F_MultiFileHdl::createFile(
	const char *	pszPath)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bCreatedDir = FALSE;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	if( RC_BAD( rc = pFileSystem->createDir( pszPath)))
	{
		goto Exit;
	}

	f_strcpy( m_szPath, pszPath);
	bCreatedDir = TRUE;

	// Create the lock file

	if( RC_BAD( rc = createLockFile( m_szPath)))
	{
		goto Exit;
	}

	// Initialize the EOF to 0 and set the state to open

	m_ui64EOF = 0;
	m_bOpen = TRUE;

Exit:

	// Release the lock file

	if( RC_BAD( rc))
	{
		(void)releaseLockFile( m_szPath, TRUE);
		if( bCreatedDir)
		{
			(void)pFileSystem->removeDir( m_szPath);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Creates a new 64-bit file with a unique, generated name
****************************************************************************/
RCODE F_MultiFileHdl::createUniqueFile(
	const char *		pszPath,					// Directory where the file is to be created
	const char *		pszFileExtension)		// Extension to be used on the new file.
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiCount;
	FLMBOOL				bModext = TRUE;
	FLMBOOL				bCreatedDir = FALSE;
	FLMUINT				uiBaseTime = 0;
	FLMBYTE				ucHighByte = 0;
	char					szDirName[ F_FILENAME_SIZE];
	char					szTmpPath[ F_PATH_MAX_SIZE];
	char					szBasePath[ F_PATH_MAX_SIZE];
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	if( !pszPath || pszPath[ 0] == '\0')
	{
#if defined( FLM_UNIX)
		f_strcpy( szBasePath, "./");
#elif defined( FLM_NLM)
		f_strcpy( szBasePath, "SYS:_NETWARE");
#else
		szBasePath[ 0] = '\0';
#endif
	}
	else
	{
		f_strcpy( szBasePath, pszPath);
	}

	if ((pszFileExtension) && (f_strlen( pszFileExtension) >= 3))
	{
		bModext = FALSE;
	}

	uiCount = 0;
	szDirName[ 0] = '\0';
	do
	{
		pFileSystem->pathCreateUniqueName( &uiBaseTime, szDirName, 
				pszFileExtension, &ucHighByte, bModext);

		f_strcpy( szTmpPath, szBasePath);
		pFileSystem->pathAppend( szTmpPath, szDirName);
		rc = pFileSystem->createDir( szTmpPath);
	} while ((rc != NE_FLM_OK) && (uiCount++ < 20));

	if( RC_BAD( rc))
	{
		goto Exit;
	}

	f_strcpy( m_szPath, szTmpPath);
	bCreatedDir = TRUE;

	// Create the lock file

	if( RC_BAD( rc = createLockFile( m_szPath)))
	{
		goto Exit;
	}

	// Initialize the EOF to 0 and set the state to open

	m_ui64EOF = 0;
	m_bOpen = TRUE;

Exit:

	// Release the lock file

	if( RC_BAD( rc))
	{
		releaseLockFile( m_szPath, TRUE);

		if( bCreatedDir)
		{
			(void)pFileSystem->removeDir( m_szPath);
		}
	}

	return( rc);
}

/****************************************************************************
Desc: Opens an existing 64-bit file
****************************************************************************/
RCODE F_MultiFileHdl::openFile(
	const char *	pszPath)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	IF_DirHdl *			pDir = NULL;
	FLMUINT				uiTmp;
	FLMUINT				uiHighFileNum = 0;
	FLMUINT64			ui64HighOffset = 0;

	if( m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	if( RC_BAD( pFileSystem->doesFileExist( pszPath)) ||
		!pFileSystem->isDir( pszPath))
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( m_szPath, pszPath);

	// Create the lock file

	if( RC_BAD( rc = createLockFile( m_szPath)))
	{
		goto Exit;
	}

	// Need to determine the current EOF

	if( RC_BAD( rc = pFileSystem->openDir( m_szPath, (char *)"*.64", &pDir)))
	{
		goto Exit;
	}

	// Find all data files to determine the EOF

	for( rc = pDir->next(); !RC_BAD( rc); rc = pDir->next())
	{
		if( RC_OK( getFileNum( pDir->currentItemName(), &uiTmp)))
		{
			if( uiTmp >= uiHighFileNum)
			{
				uiHighFileNum = uiTmp;
				ui64HighOffset = pDir->currentItemSize();
			}
		}
	}
	rc = NE_FLM_OK;

	m_ui64EOF = (((FLMUINT64)uiHighFileNum) * m_uiMaxFileSize) + ui64HighOffset;
	m_bOpen = TRUE;

Exit:

	if( pDir)
	{
		pDir->Release();
	}

	// Release the lock file

	if( RC_BAD( rc))
	{
		releaseLockFile( m_szPath, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc: Flushes cached data to the data file(s)
****************************************************************************/
RCODE F_MultiFileHdl::flush( void)
{
	FLMUINT		uiLoop;
	RCODE			rc = NE_FLM_OK;

	if( !m_bOpen)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < F_MULTI_FHDL_LIST_SIZE; uiLoop++)
	{
		if( m_pFileHdlList[ uiLoop].bDirty)
		{
			if( RC_BAD( rc = m_pFileHdlList[ uiLoop].pFileHdl->flush()))
			{
				goto Exit;
			}
			m_pFileHdlList[ uiLoop].bDirty = FALSE;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Reads data from the file
****************************************************************************/
RCODE F_MultiFileHdl::read(
	FLMUINT64	ui64Offset,				// Offset to begin reading
	FLMUINT		uiLength,				// Number of bytes to read
	void *		pvBuffer,				// Buffer
	FLMUINT *	puiBytesRead)			// [out] Number of bytes read
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiFileNum = getFileNum( ui64Offset);
	FLMUINT			uiFileOffset = getFileOffset( ui64Offset);
	FLMUINT			uiTmp;
	FLMUINT			uiTotalBytesRead = 0;
	FLMUINT			uiBytesToRead;
	FLMUINT			uiMaxReadLen;
	IF_FileHdl *	pFileHdl;

	// Handle the case of a 0-byte read

	if( !uiLength)
	{
		if( ui64Offset >= m_ui64EOF)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		}
		goto Exit;
	}

	// Read the data file(s), moving to new files as needed.

	for( ;;)
	{
		if( ui64Offset >= m_ui64EOF)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			goto Exit;
		}

		uiMaxReadLen = m_uiMaxFileSize - uiFileOffset;
		f_assert( uiMaxReadLen != 0);
		uiTmp = (uiLength >= uiMaxReadLen ? uiMaxReadLen : uiLength);
		uiBytesToRead = (((FLMUINT64)uiTmp > (FLMUINT64)(m_ui64EOF - ui64Offset))
								? (FLMUINT)(m_ui64EOF - ui64Offset)
								: uiTmp);

		if( RC_BAD( rc = getFileHdl( uiFileNum, FALSE, &pFileHdl)))
		{
			if( rc == NE_FLM_IO_PATH_NOT_FOUND)
			{
				// Handle the case of a sparse file by filling the unread
				// portion of the buffer with zeros.

				f_memset( pvBuffer, 0, uiBytesToRead);
				uiTmp = uiBytesToRead;
				rc = NE_FLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pFileHdl->read( uiFileOffset, uiBytesToRead,
				pvBuffer, &uiTmp)))
			{
				if( rc == NE_FLM_IO_END_OF_FILE)
				{
					// Handle the case of a sparse file by filling the unread
					// portion of the buffer with zeros.

					f_memset( &(((FLMBYTE *)(pvBuffer))[ uiTmp]),
						0, (FLMUINT)(uiBytesToRead - uiTmp));
					uiTmp = uiBytesToRead;
					rc = NE_FLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}

		uiTotalBytesRead += uiTmp;
		uiLength -= uiTmp;
		if( !uiLength)
		{
			break;
		}

		// Set up for next read

		pvBuffer = ((FLMBYTE *)pvBuffer) + uiTmp;
		ui64Offset += uiTmp;
		uiFileNum = getFileNum( ui64Offset);
		uiFileOffset = getFileOffset( ui64Offset);
	}

Exit:

	*puiBytesRead = uiTotalBytesRead;
	return( rc);
}

/****************************************************************************
Desc: Writes data to the file
****************************************************************************/
RCODE F_MultiFileHdl::write(
	FLMUINT64	ui64Offset,				// Offset
	FLMUINT		uiLength,				// Number of bytes to write.
	void *		pvBuffer,				// Buffer that contains bytes to be written
	FLMUINT *	puiBytesWritten)		// Number of bytes written.
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiFileNum = getFileNum( ui64Offset);
	FLMUINT			uiFileOffset = getFileOffset( ui64Offset);
	FLMUINT			uiTmp;
	FLMUINT			uiTotalBytesWritten = 0;
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiMaxWriteLen;
	IF_FileHdl *	pFileHdl;

	// Don't allow zero-length writes

	f_assert( uiLength);

	// Write to the data file(s), moving to new files as needed.

	for( ;;)
	{
		if( RC_BAD( rc = getFileHdl( uiFileNum, TRUE, &pFileHdl)))
		{
			goto Exit;
		}

		uiMaxWriteLen = m_uiMaxFileSize - uiFileOffset;
		f_assert( uiMaxWriteLen != 0);
		uiBytesToWrite = uiLength >= uiMaxWriteLen ? uiMaxWriteLen : uiLength;

		uiTmp = 0;
		rc = pFileHdl->write( uiFileOffset, uiBytesToWrite, pvBuffer, &uiTmp);

		uiTotalBytesWritten += uiTmp;
		uiLength -= uiTmp;
		ui64Offset += uiTmp;

		if( RC_BAD( rc))
		{
			goto Exit;
		}

		if( !uiLength)
		{
			break;
		}

		// Set up for next write

		pvBuffer = ((FLMBYTE *)pvBuffer) + uiTmp;
		uiFileNum = getFileNum( ui64Offset);
		uiFileOffset = getFileOffset( ui64Offset);
	}

Exit:

	if( ui64Offset > m_ui64EOF)
	{
		m_ui64EOF = ui64Offset;
	}

	*puiBytesWritten = uiTotalBytesWritten;
	return( rc);
}

/****************************************************************************
Desc: Returns the requested file handle
****************************************************************************/
RCODE F_MultiFileHdl::getFileHdl(
	FLMUINT				uiFileNum,
	FLMBOOL				bGetForWrite,
	IF_FileHdl **		ppFileHdl)
{
	RCODE					rc	= NE_FLM_OK;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	FLMUINT				uiSlot;
	IF_FileHdl *		pTmpHdl;
	char					szPath[ F_PATH_MAX_SIZE];

	f_assert( m_bOpen);

	*ppFileHdl = NULL;

	uiSlot = uiFileNum % F_MULTI_FHDL_LIST_SIZE;
	pTmpHdl = m_pFileHdlList[ uiSlot].pFileHdl;

	if( pTmpHdl && m_pFileHdlList[ uiSlot].uiFileNum != uiFileNum)
	{
		if( RC_BAD( rc = pTmpHdl->flush()))
		{
			goto Exit;
		}

		pTmpHdl->closeFile();
		pTmpHdl->Release();
		pTmpHdl = NULL;

		f_memset( &m_pFileHdlList[ uiSlot], 0, sizeof( FH_INFO));
	}

	if( !pTmpHdl)
	{
		dataFilePath( uiFileNum, szPath);
		if( RC_BAD( rc = pFileSystem->openFile( szPath, 
			FLM_IO_RDWR, &pTmpHdl)))
		{
			if( rc == NE_FLM_IO_PATH_NOT_FOUND && bGetForWrite)
			{
				if( RC_BAD( rc = pFileSystem->createFile( szPath,
 					FLM_IO_RDWR, &pTmpHdl)))
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		m_pFileHdlList[ uiSlot].pFileHdl = pTmpHdl;
		m_pFileHdlList[ uiSlot].uiFileNum = uiFileNum;
		f_assert( !m_pFileHdlList[ uiSlot].bDirty);
	}

	*ppFileHdl = m_pFileHdlList[ uiSlot].pFileHdl;
	if( bGetForWrite)
	{
		m_pFileHdlList[ uiSlot].bDirty = TRUE;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Given a data file name, returns the file's number
****************************************************************************/
RCODE F_MultiFileHdl::getFileNum(
	const char *	pszFileName,
	FLMUINT *		puiFileNum)
{
	FLMUINT		uiCnt = 0;
	FLMUINT		uiDigit;
	FLMUINT		uiFileNum = 0;
	RCODE			rc = NE_FLM_OK;

	if( f_strlen( pszFileName) != 11) // XXXXXXXX.64
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	if( f_strcmp( &pszFileName[ 8], ".64") != 0)
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	while( uiCnt < 8)
	{
		uiDigit = pszFileName[ uiCnt];
		if( uiDigit >= NATIVE_LOWER_A && uiDigit <= NATIVE_LOWER_F)
		{
			uiDigit = (FLMUINT)(uiDigit - NATIVE_LOWER_A) + 10;
		}
		else if( uiDigit >= NATIVE_UPPER_A && uiDigit <= NATIVE_UPPER_F)
		{
			uiDigit = (FLMUINT)(uiDigit - NATIVE_UPPER_A) + 10;
		}
		else if( uiDigit >= NATIVE_ZERO && uiDigit <= NATIVE_NINE)
		{
			uiDigit -= NATIVE_ZERO;
		}
		else
		{
			rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
			goto Exit;
		}

		uiFileNum <<= 4;
		uiFileNum += uiDigit;
		uiCnt++;
	}

	*puiFileNum = uiFileNum;

Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a 64-bit file by creating
		a .lck file.  The object holds the .lck file open as long as the
		64-bit file is open.
****************************************************************************/
RCODE F_MultiFileHdl::createLockFile(
	const char *		pszBasePath)
{
	RCODE					rc = NE_FLM_OK;
	char					szLockPath [F_PATH_MAX_SIZE];
	F_FileHdl *			pLockFileHdl = NULL;
	FLMUINT				uiIoFlags = FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_strcpy( szLockPath, pszBasePath);
	pFileSystem->pathAppend( szLockPath, "64.LCK");

	// Attempt to create the lock file.  If it fails, the lock file
	// may have been left because of a crash.  Hence, we first try
	// to delete the file.  If that succeeds, we then attempt to
	// create the file again.  If it, or the 2nd create fail, we simply
	// return an access denied error.
	
	if( RC_BAD( rc = f_allocFileHdl( &pLockFileHdl)))
	{
		goto Exit;
	}

#ifndef FLM_UNIX
	// On Unix, we do not want to delete the file because it
	// will succeed even if someone else has the file open.
	
	uiIoFlags |= FLM_IO_DELETE_ON_RELEASE;
#endif

	if( RC_BAD( pLockFileHdl->createFile( szLockPath, uiIoFlags)))
	{
#ifndef FLM_UNIX
		if (RC_BAD( pFileSystem->deleteFile( szLockPath)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if (RC_BAD( pLockFileHdl->createFile( szLockPath, uiIoFlags)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
#else

		if( RC_BAD( pLockFileHdl->openFile( szLockPath, uiIoFlags)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
#endif
	}

#ifdef FLM_UNIX
	if( RC_BAD( pLockFileHdl->lock()))
	{
		rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
		goto Exit;
	}
#endif

	m_pLockFileHdl = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->closeFile();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	This is a private method that will truncate the spill file back to
		the specified size.
****************************************************************************/
RCODE F_MultiFileHdl::truncateFile(
	FLMUINT64		ui64NewSize)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiFileNum = getFileNum( ui64NewSize);
	IF_FileHdl *	pFileHdl;

	if( RC_BAD( rc = getFileHdl( uiFileNum, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->truncateFile( getFileOffset( ui64NewSize))))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_MultiFileHdl::getPath(
	char *	pszPath)
{
	f_strcpy( pszPath, m_szPath);
	return( NE_FLM_OK);
}
