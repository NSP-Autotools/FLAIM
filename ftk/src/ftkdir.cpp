//------------------------------------------------------------------------------
//	Desc:	Class for doing file directory operations.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

#define ERR_NO_FILES_FOUND                      0xFF
#define ERR_INVALID_PATH                        0x9C

#if defined( FLM_WIN)

	#define F_IO_FA_NORMAL			FILE_ATTRIBUTE_NORMAL		// Normal file
	#define F_IO_FA_RDONLY			FILE_ATTRIBUTE_READONLY		// Read only attribute
	#define F_IO_FA_HIDDEN			FILE_ATTRIBUTE_HIDDEN		// Hidden file
	#define F_IO_FA_SYSTEM			FILE_ATTRIBUTE_SYSTEM		// System file
	#define F_IO_FA_VOLUME			FILE_ATTRIBUTE_VOLUME		// Volume label
	#define F_IO_FA_DIRECTORY		FILE_ATTRIBUTE_DIRECTORY	// Directory
	#define F_IO_FA_ARCHIVE			FILE_ATTRIBUTE_ARCHIVE		// Archive
	
	FSTATIC FLMBOOL f_fileMeetsFindCriteria(
		F_IO_FIND_DATA *		pFindData);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	#define F_IO_FA_NORMAL			0x01	// Normal file, no attributes
	#define F_IO_FA_RDONLY			0x02	// Read only attribute
	#define F_IO_FA_HIDDEN			0x04	// Hidden file
	#define F_IO_FA_SYSTEM			0x08	// System file
	#define F_IO_FA_VOLUME			0x10	// Volume label
	#define F_IO_FA_DIRECTORY		0x20	// Directory
	#define F_IO_FA_ARCHIVE			0x40	// Archive
	
	FSTATIC int Find1(
		char *				FindTemplate,
		F_IO_FIND_DATA *	DirInfo);

	FSTATIC int Find2(
		F_IO_FIND_DATA *	DirStuff);

	FSTATIC FLMBYTE ReturnAttributes(
		mode_t		FileMode,
		char *		pszFileName);

	FSTATIC int RetrieveFileStat(
		char *			FilePath,
		struct stat	*	StatusRec);

#elif !defined( FLM_RING_ZERO_NLM)

	#error Platform not supported

#endif

RCODE f_fileFindFirst(
	char *				pszSearchPath,
	FLMUINT				uiSearchAttrib,
	F_IO_FIND_DATA	*	find_data,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib);

RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib);

void f_fileFindClose(
	F_IO_FIND_DATA *		pFindData);

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocDirHdl(
	F_DirHdl **				ppDirHdl)
{
	if( (*ppDirHdl = f_new F_DirHdl) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:
****************************************************************************/
F_DirHdl::F_DirHdl()
{
	m_rc = NE_FLM_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;
	m_szPattern[ 0] = '\0';
}

/****************************************************************************
Desc:
****************************************************************************/
F_DirHdl::~F_DirHdl()
{
#ifndef FLM_RING_ZERO_NLM
	if( m_bFindOpen)
	{
		f_fileFindClose( &m_FindData);
	}
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
const char * FTKAPI F_DirHdl::currentItemName( void)
{
	const char *	pszName = NULL;
	
#ifndef FLM_RING_ZERO_NLM

	if( RC_OK( m_rc))
	{
		pszName = m_szFileName;
	}
	
#else

	FLMUINT			uiLength;

	if( RC_OK( m_rc))
	{
		if( !m_FindData.pCurrentItem)
		{
			return( NULL);
		}
		
		uiLength = sizeof( m_FindData.ucTempBuffer) - 1;
		if( m_FindData.pCurrentItem->DFileNameLength < uiLength)
		{
			uiLength = m_FindData.pCurrentItem->DFileNameLength;
		}
		
		f_strncpy( m_FindData.ucTempBuffer, 
			(const char *)m_FindData.pCurrentItem->DFileName, uiLength);
		m_FindData.ucTempBuffer[ uiLength] = 0;
		pszName = m_FindData.ucTempBuffer;
	}
	
#endif

	return( pszName);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void FTKAPI F_DirHdl::currentItemPath(
	char *		pszPath)
{
	if( RC_OK( m_rc))
	{
		f_strcpy( pszPath, m_szDirectoryPath);
#ifdef FLM_RING_ZERO_NLM
		f_pathAppend( pszPath, currentItemName());
#else
		f_pathAppend( pszPath, m_szFileName);
#endif
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI F_DirHdl::currentItemIsDir( void)
{
#if defined( FLM_WIN) || defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	return( ((m_uiAttrib & F_IO_FA_DIRECTORY)
						 ? TRUE
						 : FALSE));
						 
#elif defined( FLM_RING_ZERO_NLM)

	if( !m_FindData.pCurrentItem)
	{
		return( FALSE);
	}

	return( (m_FindData.pCurrentItem->DFileAttributes & SUBDIRECTORY_BIT)
							? TRUE 
							: FALSE);
							
#else

	#error Platform not supported
		
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT64 FTKAPI F_DirHdl::currentItemSize( void)
{
	FLMUINT64	ui64Size = 0;

	if( RC_OK( m_rc))
	{
#if defined( FLM_WIN)
		ui64Size = (((FLMUINT64)m_FindData.findBuffer.nFileSizeHigh) << 32) + 
						m_FindData.findBuffer.nFileSizeLow;
#elif defined( FLM_UNIX) || defined ( FLM_LIBC_NLM)
		ui64Size = m_FindData.FileStat.st_size;
#elif defined( FLM_RING_ZERO_NLM)
		char		szTmpPath[ F_PATH_MAX_SIZE];

		currentItemPath( szTmpPath);
		(void)f_getFileSysPtr()->getFileSize( szTmpPath, &ui64Size);
#endif
	}
	return( ui64Size);
}

/****************************************************************************
Desc:	Get the next item in a directory
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE FTKAPI F_DirHdl::next( void)
{
	char					szFoundPath[ F_PATH_MAX_SIZE];
	char					szDummyPath[ F_PATH_MAX_SIZE];
	FLMUINT				uiSearchAttributes;
	FLMUINT				uiFoundAttrib;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( RC_BAD( m_rc))
	{
		goto Exit;
	}

	uiSearchAttributes =
		F_IO_FA_NORMAL | F_IO_FA_RDONLY | F_IO_FA_ARCHIVE | F_IO_FA_DIRECTORY;

	for( ;;)
	{
		if( m_bFirstTime)
		{
			m_bFirstTime = FALSE;

			if( RC_BAD( m_rc = f_fileFindFirst( m_szDirectoryPath, 
				uiSearchAttributes, &m_FindData, szFoundPath, &uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_bFindOpen = TRUE;
			m_uiAttrib = uiFoundAttrib;
		}
		else
		{
			if( RC_BAD( m_rc = f_fileFindNext( &m_FindData, 
				szFoundPath, &uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_uiAttrib = uiFoundAttrib;
		}

		if( RC_BAD( m_rc = pFileSystem->pathReduce( szFoundPath, 
			szDummyPath, m_szFileName)))
		{
			goto Exit;
		}

		if( pFileSystem->doesFileMatch( m_szFileName, m_szPattern))
		{
			break;
		}
	}

Exit:

	return( m_rc);
}
#endif

/****************************************************************************
Desc:	Get the next item in a directory
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_DirHdl::next( void)
{
	LONG					lError = 0;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	if( RC_BAD( m_rc))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( (lError = DirectorySearch( 0, m_FindData.lVolumeNumber, 
			m_FindData.lDirectoryNumber, LONGNameSpace, 
			m_FindData.lCurrentEntryNumber, (BYTE *)"\x02\xFF*",
			-1, &m_FindData.pCurrentItem, 
			&m_FindData.lCurrentEntryNumber)) != 0)
		{
			if( (lError == ERR_NO_FILES_FOUND) || (lError == ERR_INVALID_PATH))
			{
				m_rc = RC_SET( NE_FLM_IO_NO_MORE_FILES);
			}
			else
			{
				m_rc = f_mapPlatformError( lError, NE_FLM_READING_FILE);
			}
			
			break;
		}

		if( pFileSystem->doesFileMatch( 
			(const char *)m_FindData.pCurrentItem->DFileName, m_szPattern))
		{
			break;
		}
	}
	
Exit:

	return( m_rc);
}
#endif

/****************************************************************************
Desc:	Open a directory
****************************************************************************/
#ifndef FLM_RING_ZERO_NLM
RCODE FTKAPI F_DirHdl::openDir(
	const char *	pszDirName,
	const char *	pszPattern)
{
	RCODE		rc = NE_FLM_OK;

	m_rc = NE_FLM_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;

	f_memset( &m_FindData, 0, sizeof( m_FindData));
	f_strcpy( m_szDirectoryPath, pszDirName);

	if( pszPattern)
	{
		if( f_strlen( pszPattern) >= (FLMINT)sizeof( m_szPattern))
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}

		f_strcpy( m_szPattern, pszPattern);
	}
	else
	{
		m_szPattern[ 0] = 0;
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:		Open a directory
Notes:
			1. DOS file names, not long file names !  If we want to support long
				file names, then increase the size of the filename buffer and change 
				the name space.
			2. '*.*' doesn't work as a pattern.  '*' seems to do the trick.
			3. These Netware APIs are case sensitive.  If you want to specify a 
				a pattern like "*.db"  make sure that the files you are looking for
				were created with lowercase "db" extensions.
				
				The path needs to match the case also.  For example, 
				sys:\_netware won't work.  SYS:\_NETWARE will.
			4. Server names are not supported by ConvertPathString
				'Connecting to remote servers' is not supported by this code.
****************************************************************************/
#ifdef FLM_RING_ZERO_NLM
RCODE FTKAPI F_DirHdl::openDir(
	const char *	pszDirName,
	const char *	pszPattern)
{
	RCODE			rc = NE_FLM_OK;
	LONG			lError = 0;
	LONG			unused;	
	FLMBYTE		pseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		LNamePath[ F_PATH_MAX_SIZE];
	LONG			lLNamePathCount;

	m_rc = NE_FLM_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;
	f_strcpy( m_szDirectoryPath, pszDirName);
	
	f_memset( &m_FindData, 0, sizeof( m_FindData));
	m_FindData.lVolumeNumber = F_NW_DEFAULT_VOLUME_NUMBER;
	m_FindData.lCurrentEntryNumber = 0xFFFFFFFF;
	m_FindData.lDirectoryNumber = 0xFFFFFFFF;
	
	LNamePath[0] = 0;
	lLNamePathCount = 0;

	f_strcpy( (char *)&pseudoLNamePath[1], pszDirName);
	pseudoLNamePath[ 0] = (FLMBYTE)f_strlen( (const char *)&pseudoLNamePath[ 1]);
	
	if( (lError = ConvertPathString( 0, 0, pseudoLNamePath, 
		&m_FindData.lVolumeNumber,
		&unused, (BYTE *)LNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}
											
	if( (lError = MapPathToDirectoryNumber( 0, m_FindData.lVolumeNumber, 0, 
		(BYTE *)LNamePath, lLNamePathCount, LONGNameSpace, 
		&m_FindData.lDirectoryNumber, &unused)) != 0)
	{
		goto Exit;
	}
		
	if( pszPattern)
	{
		if( f_strlen( pszPattern) >= (FLMINT)sizeof( m_szPattern))
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}

		f_strcpy( m_szPattern, pszPattern);
	}
	else
	{
		m_szPattern[ 0] = 0;
	}
	
Exit:
	
	if( lError != 0)
	{
		m_rc = f_mapPlatformError( lError, NE_FLM_OPENING_FILE);
	}

	return( m_rc);
}
#endif

/****************************************************************************
Desc:	Create a directory (and parent directories if necessary).
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE FTKAPI F_DirHdl::createDir(
	const char *	pszDirPath)
{
	RCODE					rc = NE_FLM_OK;
	char *				pszParentDir = NULL;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &pszParentDir)))
	{
		goto Exit;
	}

	// Discover the parent directory of the given one

	if( RC_BAD( rc = pFileSystem->pathReduce( pszDirPath, 
		pszParentDir, NULL)))
	{
		goto Exit;
	}

	// If pathReduce couldn't reduce the path at all, then an
	// invalid path was supplied.

	if( f_strcmp( pszDirPath, pszParentDir) == 0)
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	// If a parent directory was found, and it doesn't already exist, create it

	if( *pszParentDir)
	{
		// If the "parent" is actually a regular file we need to return an error

		if( RC_OK( pFileSystem->doesFileExist( pszParentDir)))
		{
			if( !pFileSystem->isDir( pszParentDir))
			{
				rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
				goto Exit;
			}
		}
		else if( RC_BAD( rc = createDir( pszParentDir)))
		{
			goto Exit;
		}
	}

#if defined( FLM_WIN)

	if( !CreateDirectory((LPTSTR)pszDirPath, NULL))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_CREATING_FILE);
	}

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	if( mkdir( (char *)pszDirPath, 0777) == -1)
	{
		rc = f_mapPlatformError( errno, NE_FLM_CREATING_FILE);
	}

#endif

Exit:

	if( pszParentDir)
	{
		f_free( &pszParentDir);
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FTKAPI F_DirHdl::createDir(
	const char *	pszDirPath)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE			pucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE			pucLNamePath[ F_PATH_MAX_SIZE];
	LONG				lVolumeID;
	LONG				lPathID;
	LONG				lLNamePathCount;
	LONG				lNewDirectoryID;
	void *			pNotUsed;
	LONG				lErrorCode;
	
	f_strcpy( (char *)&pucPseudoLNamePath[1], pszDirPath);
	pucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszDirPath);
	
	if( (lErrorCode = ConvertPathString( 0, 0, pucPseudoLNamePath, &lVolumeID,		
		&lPathID, pucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = CreateDirectory( 0, lVolumeID, lPathID, pucLNamePath,
		lLNamePathCount, LONGNameSpace, MaximumDirectoryAccessBits, 
		&lNewDirectoryID, &pNotUsed)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode)
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_CREATING_FILE);
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:		Remove a directory
Notes:	The directory must be empty.
****************************************************************************/
RCODE FTKAPI F_DirHdl::removeDir(
	const char *	pszDirName)
{
#if defined( FLM_WIN)

	if( !RemoveDirectory((LPTSTR)pszDirName))
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	 if( rmdir( (char *)pszDirName) == -1)
	 {
		 return( f_mapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	 }

    return( NE_FLM_OK);
	 
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareRemoveDir( pszDirName));

#endif
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE f_netwareRemoveDir( 
	const char *		pszDirName)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		pucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		pucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lErrorCode;
	
	f_strcpy( (char *)&pucPseudoLNamePath[1], pszDirName);
	pucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszDirName);
	
	if( (lErrorCode = ConvertPathString( 0, 0, pucPseudoLNamePath, &lVolumeID,		
		&lPathID, pucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = DeleteDirectory( 0, lVolumeID, lPathID, pucLNamePath,
		lLNamePathCount, LONGNameSpace)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode)
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_IO_DELETING_FILE);
	}
	
	return( rc);
}
#endif
	
/****************************************************************************
Desc:		Find the first file that matches the supplied criteria
****************************************************************************/
#ifdef FLM_WIN
RCODE f_fileFindFirst(
	char *				pszSearchPath,
   FLMUINT				uiSearchAttrib,
	F_IO_FIND_DATA	*	pFindData,
   char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE					rc = NE_FLM_OK;
	char 					szTmpPath[ F_PATH_MAX_SIZE];
   char *				pszWildCard = "*.*";
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_memset( pFindData, 0, sizeof( F_IO_FIND_DATA));
	pFindData->findHandle = INVALID_HANDLE_VALUE;
	pFindData->uiSearchAttrib = uiSearchAttrib;

	if( !pszSearchPath)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( pFindData->szSearchPath, pszSearchPath);

	if( uiSearchAttrib & F_IO_FA_NORMAL )
	{
		uiSearchAttrib |= F_IO_FA_ARCHIVE;
	}

	f_strcpy( szTmpPath, pszSearchPath);
	
	if( RC_BAD( rc = pFileSystem->pathAppend( szTmpPath, pszWildCard)))
	{
		goto Exit;
	}

   if( (pFindData->findHandle = FindFirstFile( (LPTSTR)szTmpPath,
	      &(pFindData->findBuffer))) == INVALID_HANDLE_VALUE)
   {
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

	// Loop until a file with correct attributes is found

	for( ;;)
	{
		if( f_fileMeetsFindCriteria( pFindData))
		{
			break;
		}

		if( FindNextFile( pFindData->findHandle,
			&(pFindData->findBuffer)) == FALSE)
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name

	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	
	if( RC_BAD( rc = pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->findBuffer.cFileName)))
	{
		goto Exit;
	}

	// Return the found file attribute

   *puiFoundAttrib = pFindData->findBuffer.dwFileAttributes;

Exit:

	if( RC_BAD( rc) && pFindData &&
		pFindData->findHandle != INVALID_HANDLE_VALUE)
	{
		f_fileFindClose( pFindData);
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:		Find the first file that matches the supplied criteria
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_fileFindFirst(
	char *				pszSearchPath,
   FLMUINT				uiSearchAttrib,
	F_IO_FIND_DATA	*	pFindData,
   char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE					rc = NE_FLM_OK;
	char 					szTmpPath[ F_PATH_MAX_SIZE];
	FSTATIC char		pszWildCard[] = {'*',0};
	int					iRetVal;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if( !pszSearchPath)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( szTmpPath, pszSearchPath);
	if( RC_BAD( rc = pFileSystem->pathAppend( szTmpPath, pszWildCard)))
	{
		goto Exit;
	}

	f_memset( pFindData, 0, sizeof( F_IO_FIND_DATA));
	if( uiSearchAttrib & F_IO_FA_DIRECTORY)
	{
		pFindData->mode_flag |= S_IFDIR;
	}

	if( uiSearchAttrib & F_IO_FA_RDONLY)
	{
		pFindData->mode_flag |= S_IREAD;
	}

	iRetVal = Find1( (char*)szTmpPath, pFindData);

	if( iRetVal != 0)
	{
		// If there were no more files found then return no more files
		// instead of mapping to error path not found or io error.
		// To return no more files ret_val is ENOENT (set in Find2)
		// and errno is not set

		if( iRetVal == ENOENT && errno == 0)
		{
			rc = RC_SET( NE_FLM_IO_NO_MORE_FILES);
		}
		else
		{
			rc = f_mapPlatformError( errno, NE_FLM_READING_FILE);
		}
		
		goto Exit;
	}

	// filter out ".." (PARENT) and "." (CURRENT) directories
	
	if( uiSearchAttrib & F_IO_FA_DIRECTORY )
	{
		while( (f_strcmp( pFindData->name, "..") == 0) ||
			   (f_strcmp( pFindData->name, ".") == 0))
		{
			if( (iRetVal = Find2( pFindData)) != 0)
			{
				// If there were no more files found then return no more files
				// instead of mapping to error path not found or io error.
				// To return no more files ret_val is ENOENT (set in Find2)
				// and errno is not set
				
				if( iRetVal == ENOENT && errno == 0)
				{
					rc = RC_SET( NE_FLM_IO_NO_MORE_FILES);
				}
				else
				{
					rc = f_mapPlatformError( errno, NE_FLM_READING_FILE);
				}
				
				goto Exit;
			}
		}
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pszSearchPath);
	
	if( RC_BAD( rc = pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = (FLMUINT)ReturnAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);

	// Save the search path in the NE_FLM_IO_FIND_DATA struct
	// for a find next call

	f_strcpy( pFindData->search_path, pszSearchPath);

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
#ifdef FLM_WIN
RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

   if( FindNextFile( pFindData->findHandle,
		&(pFindData->findBuffer)) == FALSE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_READING_FILE);
		goto Exit;
	}

	// Loop until a file with correct attributes is found

	for( ;;)
	{
		if( f_fileMeetsFindCriteria( pFindData))
		{
			break;
		}

		if( FindNextFile( pFindData->findHandle,
			&(pFindData->findBuffer)) == FALSE)
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name

	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	
	if( RC_BAD( rc = pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->findBuffer.cFileName)))
	{
		goto Exit;
	}

	// Return the found file attribute

   *puiFoundAttrib = pFindData->findBuffer.dwFileAttributes;

Exit:

   return( rc);
}
#endif

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	int					iRetVal;

	if( (iRetVal =  Find2( pFindData)) != 0)
	{
		// If there were no more files found then return no more files
		// instead of mapping to error path not found or io error.
		// To return no more files ret_val is ENOENT (set in Find2)
		// and errno is not set
		
		if( iRetVal == ENOENT && errno == 0)
		{
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));
		}
		
		return( f_mapPlatformError( errno, NE_FLM_READING_FILE));
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pFindData->search_path);
	
	if( RC_BAD( rc = pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = (FLMUINT)ReturnAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);

Exit:

   return( rc);
}
#endif

/****************************************************************************
Desc:		Releases any memory allocated to an F_IO_FIND_DATA structure
****************************************************************************/
#ifdef FLM_WIN
void f_fileFindClose(
	F_IO_FIND_DATA *	pFindData)
{
	// Don't call it on an already closed or invalid handle.

	if( pFindData->findHandle != INVALID_HANDLE_VALUE)
	{
		FindClose( pFindData->findHandle );
		pFindData->findHandle = INVALID_HANDLE_VALUE;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined ( FLM_LIBC_NLM)
void f_fileFindClose(
	F_IO_FIND_DATA *	pFindData)
{
	if( pFindData->globbuf.gl_pathv)
	{
		pFindData->globbuf.gl_offs = 0;
		globfree( &pFindData->globbuf);
		pFindData->globbuf.gl_pathv = 0;
	}
}
#endif

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
#ifdef FLM_WIN
FSTATIC FLMBOOL f_fileMeetsFindCriteria(
	F_IO_FIND_DATA *		pFindData)
{
	// Fail ".." (PARENT) and "." (CURRENT) directories.  Then,
	// if the file found possesses any of the search attributes, it's
	// a match.

	if( !((f_strcmp( pFindData->findBuffer.cFileName, "..") == 0) ||
    (f_strcmp( pFindData->findBuffer.cFileName, ".") == 0) ||
	 (!(pFindData->uiSearchAttrib & F_IO_FA_DIRECTORY) &&
		(pFindData->findBuffer.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))))
	{
		if( (pFindData->findBuffer.dwFileAttributes &
				pFindData->uiSearchAttrib) ||
			((pFindData->uiSearchAttrib & F_IO_FA_NORMAL) &&
				(pFindData->findBuffer.dwFileAttributes == 0)))
		{
			return( TRUE);
		}
	}

	return( FALSE);
}
#endif

/****************************************************************************
Desc:		Search for file names matching FindTemplate (UNIX)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
FSTATIC int Find1(
	char *				FindTemplate,
	F_IO_FIND_DATA *	DirInfo)
{
	char  	MaskNam[ F_PATH_MAX_SIZE];
	char  	*PathSeparator;
	FLMINT	uiFindLen;
	FLMINT	uiLen;
#ifdef FLM_LIBC_NLM
	char  	szPosixNam[ F_PATH_MAX_SIZE];
	FLMINT	uiCount;				
#endif

	// If supplied template is illegal, return immediately

	if( (FindTemplate == NULL) || (uiFindLen = f_strlen( FindTemplate)) == 0)
	{
		return( EINVAL);
	}

	// Now separate the template into a PATH and a template MASK
	// If no separating slash character found, use current directory
	// as path!

	f_strcpy( DirInfo->full_path, FindTemplate);
	
#ifdef FLM_LIBC_NLM
	if( (( PathSeparator = strrchr( DirInfo->full_path, '/')) == NULL) &&
		( PathSeparator = strrchr( DirInfo->full_path, '\\')) == NULL)
#else
	if( (PathSeparator = strrchr( DirInfo->full_path, '/')) == NULL)
#endif
	{
		FLM_UNCHECKED_RV(getcwd( DirInfo->full_path, F_PATH_MAX_SIZE));
		uiLen = f_strlen( DirInfo->full_path );
		DirInfo->full_path[uiLen] = '/';
		DirInfo->full_path[uiLen+1] = '\0';
		(void) f_strcat( DirInfo->full_path, FindTemplate );
		PathSeparator = strrchr( DirInfo->full_path, '/');
	}

	// Copy the template MASK, and null terminate the PATH
	
	f_strcpy( MaskNam, PathSeparator + 1);
	
	if( ! f_strlen(MaskNam))
	{
		(void) f_strcpy( MaskNam, "*");
	}

	*PathSeparator = '\0';

	// Use ROOT directory if PATH is empty
	
	if( ! f_strlen(DirInfo->full_path))
	{
		(void) f_strcpy( DirInfo->full_path, "/");
	}

	f_strcpy( DirInfo->dirpath, DirInfo->full_path );

	// Open the specified directory.  Return immediately
	// if error detected!

	errno = 0;
	DirInfo->globbuf.gl_pathv = 0;

#ifdef FLM_LIBC_NLM
	// glob does not seem to be able to handle a non-posix path
	// on NetWare.
	for( uiCount = 0; uiCount <= uiFindLen; uiCount++)
	{
		if( FindTemplate[ uiCount] == '\\')
		{
			szPosixNam[ uiCount] = '/';
		}
		else
		{
			szPosixNam[ uiCount] = FindTemplate[ uiCount];
		}
	}
	if( glob( szPosixNam, GLOB_NOSORT, 0, &DirInfo->globbuf) != 0 &&
		 !DirInfo->globbuf.gl_pathc)
#else
	if( glob( FindTemplate, GLOB_NOSORT, 0, &DirInfo->globbuf) != 0 &&
		 !DirInfo->globbuf.gl_pathc)
#endif
	{
		globfree(&DirInfo->globbuf);
		DirInfo->globbuf.gl_pathv = 0;
		return ENOENT;
	}
	
	// Call Find2 to get the 1st matching file

	return( Find2(DirInfo) );
}
#endif


/****************************************************************************
Desc:		Search for file names matching FindTemplate (UNIX)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
FSTATIC int Find2(
	F_IO_FIND_DATA *		DirStuff)
{
	int			stat;
	glob_t *		pglob = &DirStuff->globbuf;
	char *		pszTmp;
	char *		pszLastSlash;

	errno = 0;

	for( ;;)
	{
		if( pglob->gl_offs == pglob->gl_pathc)
		{
			pglob->gl_offs = 0;
			globfree(pglob);
			pglob->gl_pathv = 0;
			return ENOENT;
		}
		
		// Get status of file

		f_strcpy(DirStuff->full_path, pglob->gl_pathv[pglob->gl_offs++]);
		if( (stat = RetrieveFileStat( DirStuff->full_path,
											 &DirStuff->FileStat)) != 0 )
		{
			// If file name just read from directory is NO
			// longer there (deleted by another process)
			// then just advance to the next file in
			// directory!

			if( stat == ENOENT)
			{
				continue;
			}
			else
			{
				break;
			}
		}

		// If we don't want directories, and current entry
		// is a directory, then skip it!

		if( (! S_ISDIR(DirStuff->mode_flag)) &&
			  S_ISDIR(DirStuff->FileStat.st_mode))
		{
			continue;
		}

		// If we only want regular files and file is NOT
		// regular, then skip it!  This means there is no
		// way to retrieve named pipes, sockets, or links!

		if ( (DirStuff->mode_flag == F_IO_FA_NORMAL) &&
			  (! S_ISREG(DirStuff->FileStat.st_mode)) )
		{
			continue;
		}

		pszTmp = &DirStuff->full_path[ 0];
		pszLastSlash = NULL;
		while( *pszTmp)
		{
			if( *pszTmp == '/')
			{
				pszLastSlash = pszTmp;
			}
			pszTmp++;
		}

		if( pszLastSlash)
		{
			f_strcpy( DirStuff->name, &pszLastSlash[ 1]);
		}
		else
		{
			f_strcpy( DirStuff->name, DirStuff->full_path);
		}
		stat = 0;
		break;
	}
	
	return( stat);
}
#endif

/****************************************************************************
Desc: Return file's attributes (UNIX)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
FSTATIC FLMBYTE ReturnAttributes(
	mode_t	FileMode,
	char *	fileName)
{
	FLMBYTE  IOmode = 0;

	// Return the found file attribute
	
	if( S_ISDIR( FileMode ) )
	{
		IOmode |= F_IO_FA_DIRECTORY;
	}
	else
	{
		if( access( (char *)fileName, (int)(R_OK | W_OK)) == 0)
		{
			IOmode |= F_IO_FA_NORMAL;
		}
		else if( access( (char *)fileName, (int)R_OK ) == 0)
		{
			IOmode |= F_IO_FA_RDONLY;
		}
	}

	return( IOmode);
}
#endif

/****************************************************************************
Desc: Return file's attributes (UNIX) || (NetWare)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
FSTATIC int RetrieveFileStat(
	char *			FilePath,
	struct stat	*	StatusRec)
{
	// Get status of last file read from directory, using the standard
	// UNIX stat call

	errno = 0;
	if( stat( FilePath, StatusRec ) == -1)
	{
		if( errno == ENOENT || errno == ELOOP)
		{
			// Get status of symbolic link rather than referenced file!

			errno = 0;
			if( lstat( FilePath, StatusRec ) == -1)
			{
				return( errno);
			}
		}
		else
		{
			return( errno);
		}
	}

	return( 0);
}
#endif
