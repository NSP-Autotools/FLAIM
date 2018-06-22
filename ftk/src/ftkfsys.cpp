//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileSystem class.
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

class F_FileHdlCache;

FSTATIC FLMBOOL f_canReducePath(
	const char *	pszSource);

FSTATIC const char * f_findFileNameStart(
	const char * 	pszPath);

FSTATIC char * f_getPathComponent(
	char **			ppszPath,
	FLMUINT *		puiEndChar);
	
/****************************************************************************
Desc:
****************************************************************************/
class F_CachedFileHdl : public F_FileHdl, public F_HashObject
{
public:

	F_CachedFileHdl()
	{
		m_pucKey = NULL;
		m_uiKeyLen = 0;
		m_bInAvailList = FALSE;
		m_pFileHdlCache = NULL;
	}
	
	virtual ~F_CachedFileHdl()
	{
		if( m_pucKey)
		{
			f_free( &m_pucKey);
		}
	}
	
	FLMINT FTKAPI AddRef( void)
	{
		return( f_atomicInc( &m_refCnt));
	}

	FLMINT FTKAPI Release( void);
	
	FINLINE const void * FTKAPI getKey( void)
	{
		return( m_pucKey);
	}
	
	FINLINE FLMUINT FTKAPI getKeyLength( void)
	{
		return( m_uiKeyLen);
	}
	
	FINLINE FLMUINT FTKAPI getObjectType( void)
	{
		return( 0);
	}

private:

	FLMBYTE *					m_pucKey;
	FLMUINT						m_uiKeyLen;
	FLMBOOL						m_bInAvailList;
	F_FileHdlCache *			m_pFileHdlCache;
	
	friend class F_FileHdlCache;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_FileHdlCache : public IF_FileHdlCache
{
public:

	F_FileHdlCache();
	
	virtual ~F_FileHdlCache();
	
	RCODE setup(
		FLMUINT					uiMaxCachedFiles,
		FLMUINT					uiIdleTimeoutSecs);
		
	FINLINE RCODE FTKAPI openFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		IF_FileHdl **			ppFile)
	{
		return( openOrCreate( pszFileName, uiIoFlags, FALSE, ppFile));
	}
	
	FINLINE RCODE FTKAPI createFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		IF_FileHdl **			ppFile)
	{
		return( openOrCreate( pszFileName, uiIoFlags, TRUE, ppFile));
	}

	void FTKAPI closeUnusedFiles(
		FLMUINT					uiUnusedTime);
	
	FLMUINT FTKAPI getOpenThreshold( void)
	{
		return( m_pAvailList->getMaxObjects());
	}
	
	RCODE FTKAPI setOpenThreshold(
		FLMUINT					uiMaxOpenFiles);
			
private:

	RCODE openOrCreate(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		FLMBOOL					bCreate,
		IF_FileHdl **			ppFile);
		
	static RCODE FTKAPI timeoutThread(
		IF_Thread *				pThread);
	
	IF_Thread *					m_pTimeoutThread;
	F_HashTable *				m_pAvailList;
	FLMUINT						m_uiMaxIdleTime;
	
	friend class F_CachedFileHdl;
};

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void f_hexToNative(
	FLMBYTE		ucHexVal,
	char *		pszNativeChar)
{
	*pszNativeChar = (char)(ucHexVal < 10
										? ucHexVal + NATIVE_ZERO
										: (ucHexVal - 10) + NATIVE_LOWER_A);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void f_setupTime(
	FLMUINT *	puiBaseTime,
	FLMBYTE *	pbyHighByte)
{
	FLMUINT		uiSdTime = 0;
	
	f_timeGetSeconds( &uiSdTime);
	*pbyHighByte = (FLMBYTE)(uiSdTime >> 24);
	uiSdTime = uiSdTime << 5;
	
	if( *puiBaseTime < uiSdTime)
	{
		*puiBaseTime = uiSdTime;
	}
}

/****************************************************************************
Desc:	Returns TRUE if character is a "slash" separator
****************************************************************************/
FINLINE FLMBOOL f_isSlashSeparator(
	char	cChar)
{
#ifdef FLM_UNIX
	return( cChar == '/' ? TRUE : FALSE);
#else
	return( cChar == '/' || cChar == '\\' ? TRUE : FALSE);
#endif
}

/****************************************************************************
Desc:	Return a pointer to the next path component in ppszPath.
****************************************************************************/
FSTATIC char * f_getPathComponent(
	char **			ppszPath,
	FLMUINT *		puiEndChar)
{
	char *	pszComponent;
	char *	pszEnd;
	
	pszComponent = pszEnd = *ppszPath;
	if (f_isSlashSeparator( *pszEnd))
	{
		// handle the condition of sys:\system   the colon would have terminated
		// the previous token, to pComponent would now be pointing at the '\'.
		// We need to move past the '\' to find the next token.

		pszEnd++;
	}

	// Find the end of the path component

	while (*pszEnd)
	{
		if (f_isSlashSeparator( *pszEnd)
#ifndef FLM_UNIX
			|| *pszEnd == ':'
#endif
			)
		{
			break;
		}
		pszEnd++;
	}

	if (*pszEnd)
	{

		// A delimiter was found, assume that there is another path component
		// after this one.
		// Return a pointer to the beginning of the next path component

		*ppszPath = pszEnd + 1;

		*puiEndChar = *pszEnd;

		// NULL terminate the path component

		*pszEnd = 0;
	}
	else
	{

		// There is no "next path component" so return a pointer to the 
		// NULL-terminator

		*ppszPath = pszEnd;
		*puiEndChar = 0;
	}	
	
	// Return the path component

	return( pszComponent);
}

/****************************************************************************
Desc:	Will determine whether any format of (UNC, drive based, NetWare
		UNC) path can be reduced any further.
****************************************************************************/
FSTATIC FLMBOOL f_canReducePath(
	const char *	pszSource)
{
#if defined FLM_UNIX
	F_UNREFERENCED_PARM( pszSource);
	return( TRUE);
#else
	FLMBOOL			bCanReduce;
	const char *	pszTemp = pszSource;

	// Determine whether the passed path is UNC or not
	// (UNC format is: \\FileServer\Volume\Path).

	if (f_strncmp( "\\\\", pszSource, 2 ) == 0)
	{
		FLMUINT	uiSlashCount = 0;

		pszTemp += 2;

		// Search forward for at least two slash separators
		// If we find at least two, the path can be reduced.

		bCanReduce = FALSE;
		while (*pszTemp)
		{
			pszTemp++;
			if (f_isSlashSeparator( *pszTemp))
			{
				++uiSlashCount;
				if (uiSlashCount == 2)
				{
					bCanReduce = TRUE;
					break;
				}
			}
		}
	}
	else
	{
		bCanReduce = TRUE;

		// Search forward for the colon.

		while (*pszTemp)
		{
			if (*pszTemp == ':')
			{

				// If nothing comes after the colon,
				// we can't reduce any more.

				if (*(pszTemp + 1) == 0)
				{
					bCanReduce = FALSE;
				}
				break;
			}
			pszTemp++;
		}
	}

	return( bCanReduce);
#endif
}

/****************************************************************************
Desc:	Return pointer to start of filename part of path.
		Search for the last slash separator.
****************************************************************************/
FSTATIC const char * f_findFileNameStart(
	const char * 	pszPath)
{
	const char *	pszFileNameStart;

	pszFileNameStart = pszPath;
	while (*pszPath)
	{
		if (f_isSlashSeparator( *pszPath))
		{
			pszFileNameStart = pszPath + 1;
		}
		pszPath++;
	}
	return( pszFileNameStart);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocFileSystem(
	IF_FileSystem **		ppFileSystem)
{
	RCODE						rc = NE_FLM_OK;
	F_FileSystem *			pFileSystem = NULL;

	if( (pFileSystem = f_new F_FileSystem) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	if( RC_BAD( rc = pFileSystem->setup()))
	{
		goto Exit;
	}

	*ppFileSystem = pFileSystem;
	pFileSystem = NULL;

Exit:

	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI FlmGetFileSystem(
	IF_FileSystem **		ppFileSystem)
{
	*ppFileSystem = f_getFileSysPtr();
	(*ppFileSystem)->AddRef();
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocFileHdl(
	F_FileHdl **		ppFileHdl)
{
	if( (*ppFileHdl = f_new F_FileHdl) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileSystem::setup( void)
{
#if defined( FLM_WIN)

	OSVERSIONINFO		versionInfo;

	versionInfo.dwOSVersionInfoSize = sizeof( OSVERSIONINFO);
	if( !GetVersionEx( &versionInfo))
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_FAILURE));
	}

	// Async writes are not supported on Win32s (3.1) or
	// Win95, 98, ME, etc.

	m_bCanDoAsync =
		(versionInfo.dwPlatformId != VER_PLATFORM_WIN32_WINDOWS &&
		 versionInfo.dwPlatformId != VER_PLATFORM_WIN32s)
		 ? TRUE
		 : FALSE;

#else
	m_bCanDoAsync = TRUE;
#endif

#if !defined( FLM_HAS_ASYNC_IO)
	m_bCanDoAsync = FALSE;
#endif

	return( NE_FLM_OK);
}
	
/****************************************************************************
Desc:    Create a file, return a file handle to created file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::createFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;
	
	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->createFile( pszFileName, uiIoFlags)))
	{
		goto Exit;
	}
	
	*ppFileHdl = pFileHdl;
	pFileHdl = NULL;

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Create a unique file, return a file handle to created file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::createUniqueFile(
	char *			pszPath,
	const char *	pszFileExtension,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pFileHdl->createUniqueFile( pszPath, 
			pszFileExtension,	uiIoFlags)))
	{
		goto Exit;
	}
	
	*ppFileHdl = pFileHdl;
	pFileHdl = NULL;

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Open a file, return a file handle to opened file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::openFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pFileHdl->openFile( pszFileName, uiIoFlags)))
	{
		goto Exit;
	}
	
	*ppFileHdl = pFileHdl;
	pFileHdl = NULL;

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:    Open a directory, return a file handle to opened directory.
****************************************************************************/
RCODE FTKAPI F_FileSystem::openDir(
	const char *	pszDirName,
	const char *	pszPattern,
	IF_DirHdl **	ppDirHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_DirHdl *		pDirHdl = NULL;
	
	if( RC_BAD( rc = f_allocDirHdl( &pDirHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDirHdl->openDir( pszDirName, pszPattern)))
	{
		goto Exit;
	}

	*ppDirHdl = (IF_DirHdl *)pDirHdl;
	pDirHdl = NULL;
	
Exit:

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:    Create a directory.
****************************************************************************/
RCODE FTKAPI F_FileSystem::createDir(
	const char *	pszDirName)
{
	RCODE				rc = NE_FLM_OK;
	F_DirHdl *		pDirHdl = NULL;
	
	if( RC_BAD( rc = f_allocDirHdl( &pDirHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDirHdl->createDir( pszDirName)))
	{
		goto Exit;
	}

Exit:

	if (pDirHdl)
	{
		pDirHdl->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Remove a directory
****************************************************************************/
RCODE FTKAPI F_FileSystem::removeDir(
	const char *	pszDirName,
	FLMBOOL			bClear)
{
	RCODE				rc = NE_FLM_OK;
	IF_DirHdl *		pDirHdl = NULL;
	char				szFilePath[ F_PATH_MAX_SIZE];

	if( bClear)
	{
		if( RC_BAD( rc = openDir( pszDirName, (char *)"*", &pDirHdl)))
		{
			goto Exit;
		}

		for( rc = pDirHdl->next(); RC_OK( rc) ; rc = pDirHdl->next())
		{
			pDirHdl->currentItemPath( szFilePath);
			if( !pDirHdl->currentItemIsDir())
			{
				if( RC_BAD( rc = deleteFile( szFilePath)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND || 
						 rc == NE_FLM_IO_INVALID_FILENAME)
					{
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
			else
			{
				if( RC_BAD( rc = removeDir( szFilePath, bClear)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND ||
						 rc == NE_FLM_IO_INVALID_FILENAME)
					{
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}

		// Need to release the directory handle so the
		// directory will be closed when we try to delete it
		// below.

		pDirHdl->Release();
		pDirHdl = NULL;
	}

	if( RC_BAD( rc = removeEmptyDir( pszDirName)))
	{
		goto Exit;
	}

Exit:

	if (pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Remove an empty directory
****************************************************************************/
RCODE F_FileSystem::removeEmptyDir(
	const char *			pszDirPath)
{
#if defined( FLM_WIN)

	if( !RemoveDirectory( (LPTSTR)pszDirPath))
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	 if( rmdir( pszDirPath) == -1 )
	 {
		 return( f_mapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	 }

    return( NE_FLM_OK);
	 
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareRemoveDir( pszDirPath));

#endif
}

/****************************************************************************
Desc:	Determine if a file or directory exists.
****************************************************************************/
RCODE FTKAPI F_FileSystem::doesFileExist(
	const char *	pszPath)
{
#if defined( FLM_WIN)

	DWORD		dwFileAttr = GetFileAttributes( (LPTSTR)pszPath);

	if( dwFileAttr == (DWORD)-1)
		return RC_SET( NE_FLM_IO_PATH_NOT_FOUND);

	return NE_FLM_OK;

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

   if( access( pszPath, F_OK) == -1)
	{
		return( f_mapPlatformError( errno, NE_FLM_CHECKING_FILE_EXISTENCE));
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareTestIfFileExists( pszPath));
	
#endif
}

/****************************************************************************
Desc:    Get the time stamp of the last modification to this file.
Notes:
	puiTimeStamp is assumed to point to a DWORD.

NLM Notes:
	We could call MapPathToDirectoryNumber and GetDirectoryEntry directly.
	This works, providing that the high byte of the directory entry (returned
	by MapPathToDirectoryNumber) is masked off.  Otherwise, GetDirectoryEntry
	will generate an abend.
	We have opted to call a higher level function, GetEntryFromPathStringBase,
	which calls the lower level functions for us.
****************************************************************************/
RCODE FTKAPI F_FileSystem::getFileTimeStamp(
	const char *	pszPath,
	FLMUINT *		puiTimeStamp)
{
#if defined( FLM_WIN)

	WIN32_FIND_DATA find_data;
	FILETIME			ftLocalFileTime;
	SYSTEMTIME		stLastFileWriteTime;
	HANDLE			hSearch = INVALID_HANDLE_VALUE;
	RCODE				rc = NE_FLM_OK;
	F_TMSTAMP		tmstamp;

	hSearch = FindFirstFile( (LPTSTR)pszPath, &find_data);
	if( hSearch == INVALID_HANDLE_VALUE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		switch( rc)
		{
	   	case NE_FLM_IO_NO_MORE_FILES:
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
				goto Exit;
			default:
				goto Exit;
		}
	}

	// Convert it to a local time
		
	if( FileTimeToLocalFileTime( &(find_data.ftLastWriteTime),
											&ftLocalFileTime) == FALSE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

	// Convert the local time to a system time
		
	if( FileTimeToSystemTime( &ftLocalFileTime,
									   &stLastFileWriteTime) == FALSE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

   // Fill the time stamp structure
	
   f_memset( &tmstamp, 0, sizeof( F_TMSTAMP));
	
   tmstamp.hour = (FLMBYTE)stLastFileWriteTime.wHour;
   tmstamp.minute = (FLMBYTE)stLastFileWriteTime.wMinute;
   tmstamp.second = (FLMBYTE)stLastFileWriteTime.wSecond;
	tmstamp.hundredth = (FLMBYTE)stLastFileWriteTime.wMilliseconds;
   tmstamp.year = (FLMUINT16)stLastFileWriteTime.wYear;
	tmstamp.month = (FLMBYTE)(stLastFileWriteTime.wMonth - 1);
   tmstamp.day = (FLMBYTE)stLastFileWriteTime.wDay;

   // Convert and return the file time stamp as seconds since January 1, 1970
	
   f_timeDateToSeconds( &tmstamp, puiTimeStamp);

Exit:

	if( hSearch != INVALID_HANDLE_VALUE)
	{
	   FindClose( hSearch);
	}

	if( RC_OK(rc))
	{
		*puiTimeStamp = f_localTimeToUTC( *puiTimeStamp);
	}

   return( rc);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat   	filestatus;

	if( stat( pszPath, &filestatus) == -1)
	{
       return( f_mapPlatformError( errno, NE_FLM_GETTING_FILE_INFO));
	}
	
	// Return the UTC time

	*puiTimeStamp = (FLMUINT)filestatus.st_mtime;
	return NE_FLM_OK;
	
#elif defined( FLM_RING_ZERO_NLM)

	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiTmp;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	LONG			lErrorCode;
	struct DirectoryStructure * pFileInfo = NULL;

	flmAssert( puiTimeStamp);
	*puiTimeStamp = 0;
	
	f_strcpy( (char *)&ucPseudoLNamePath[1], pszPath);
	ucPseudoLNamePath[ 0] = (FLMBYTE)f_strlen( pszPath );
	
	if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,    
		&lPathID, ucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}
	
	if( (lErrorCode = GetEntryFromPathStringBase( 0, lVolumeID, 0, ucLNamePath,
		lLNamePathCount, LONGNameSpace, LONGNameSpace, &pFileInfo, 
		&lDirectoryID)) != 0)
	{
		goto Exit;
	}

	if( pFileInfo)
	{
		FLMUINT			uiTime;
		FLMUINT			uiDate;
		F_TMSTAMP		dateTime;
		LONG				DayMask = 0x001F; 
		LONG				MonthMask = 0x01E0; 
		LONG				YearMask = 0xFE00;
		LONG				SecMask = 0x001F; 
		LONG				MinMask = 0x07E0;
		LONG				HourMask = 0xF800;
		
		// Get the low-order 16 bits
		
		uiTime = (FLMUINT)pFileInfo->DLastUpdatedDateAndTime;
		
		// Get the high-order 16 bits
		
		uiDate = (FLMUINT)(pFileInfo->DLastUpdatedDateAndTime >> 16);

		f_memset( &dateTime, 0, sizeof( dateTime));
		dateTime.second = (FLMBYTE) ((uiTime & SecMask) * 2);
		dateTime.minute = (FLMBYTE) ((uiTime & MinMask) >> 5);
		dateTime.hour = (FLMBYTE) ((uiTime & HourMask) >> 11);
		dateTime.day = (FLMBYTE) (uiDate & DayMask);
		dateTime.month = (FLMBYTE) ((uiDate & MonthMask) >> 5)-1;
		dateTime.year = (FLMUINT16)(((uiDate & YearMask) >> 9) + 1980);
		
		f_timeDateToSeconds( &dateTime, &uiTmp);
		*puiTimeStamp = uiTmp;
		*puiTimeStamp = f_localTimeToUTC(*puiTimeStamp);
	}

Exit:

	if( lErrorCode != 0 )
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_PARSING_FILE_NAME);
	}
	
	return( rc);

#endif
}

/****************************************************************************
Desc: Determine if a path is a directory.
****************************************************************************/
RCODE FTKAPI F_FileSystem::getFileSize(
	const char *			pszFileName,
	FLMUINT64 *				pui64FileSize)
{
	RCODE						rc = NE_FLM_OK;
	IF_FileHdl *			pFileHdl = NULL;
	
	if( RC_BAD( rc = openFile( pszFileName, 
			FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &pFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pFileHdl->size( pui64FileSize)))
	{
		goto Exit;
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}
	
/****************************************************************************
Desc: Determine if a path is a directory.
****************************************************************************/
FLMBOOL FTKAPI F_FileSystem::isDir(
	const char *		pszDirName)
{
#if defined( FLM_WIN)

	DWORD	FileAttr = GetFileAttributes( (LPTSTR)pszDirName);
	
	if( FileAttr == 0xFFFFFFFF)
	{
		return( FALSE);
	}

	return (FileAttr & FILE_ATTRIBUTE_DIRECTORY) ? TRUE : FALSE;

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat     filestatus;

	if( stat( (char *)pszDirName, &filestatus) == -1)
	{
		return FALSE;
	}

	return ( S_ISDIR( filestatus.st_mode)) ? TRUE : FALSE;
	
#elif defined( FLM_RING_ZERO_NLM)
	
	LONG			lIsFile;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	FLMBOOL		bIsDir = FALSE;

	f_strcpy( (char *)&ucPseudoLNamePath[1], pszDirName);
	ucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszDirName);
	if( ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID, &lPathID,      
		ucLNamePath, &lLNamePathCount) == 0)
	{
		if( MapPathToDirectoryNumber( 0, lVolumeID, 0, ucLNamePath, 
			lLNamePathCount, LONGNameSpace, &lDirectoryID, &lIsFile) == 0)
		{
			bIsDir = (FLMBOOL)((lIsFile == 0) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
		}
	}
	
	return( bIsDir);
	
#endif
}

/****************************************************************************
Desc:    Delete a file or directory
****************************************************************************/
RCODE FTKAPI F_FileSystem::deleteFile(
	const char *		pszFileName)
{
#if defined( FLM_WIN)

	if( DeleteFile( (LPTSTR)pszFileName) == FALSE)
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}
	
	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat FileStat;

	if( stat( (char *)pszFileName, &FileStat) == -1)
	{
		return( f_mapPlatformError( errno, NE_FLM_GETTING_FILE_INFO));
	}

	// Ensure that the path does NOT designate a directory for deletion
	
	if( S_ISDIR(FileStat.st_mode))
	{
		return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
	}

	// Delete the file
	
	if( unlink( (char *)pszFileName) == -1)
	{
       return( f_mapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareDeleteFile( pszFileName));
	 
#endif
}

/****************************************************************************
Desc:	Copy a file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::copyFile(
	const char *	pszSrcFileName,
	const char *	pszDestFileName,
	FLMBOOL			bOverwrite,
	FLMUINT64 *		pui64BytesCopied)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pSrcFileHdl = NULL;
	IF_FileHdl *	pDestFileHdl = NULL;
	FLMBOOL			bCreatedDest = FALSE;
	FLMUINT64		ui64SrcSize;

	// See if the destination file exists.  If it does, see if it is
	// OK to overwrite it.  If so, delete it.

	if (doesFileExist( pszDestFileName) == NE_FLM_OK)
	{
		if (!bOverwrite)
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		if (RC_BAD( rc = deleteFile( pszDestFileName)))
		{
			goto Exit;
		}
	}

	// Open the source file.

	if( RC_BAD( rc = openFile( pszSrcFileName, 
			FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &pSrcFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pSrcFileHdl->size( &ui64SrcSize)))
	{
		goto Exit;
	}

	// Create the destination file.

	if( RC_BAD( rc = createFile( pszDestFileName, 
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pDestFileHdl)))
	{
		goto Exit;
	}
	bCreatedDest = TRUE;

	// Do the copy.

	if( RC_BAD( rc = copyPartialFile( pSrcFileHdl, 0, ui64SrcSize, 
				pDestFileHdl, 0, pui64BytesCopied)))
	{
		goto Exit;
	}
	
Exit:

	if( pSrcFileHdl)
	{
		pSrcFileHdl->closeFile();
		pSrcFileHdl->Release();
	}
	
	if( pDestFileHdl)
	{
		pDestFileHdl->closeFile();
		pDestFileHdl->Release();
	}
	
	if( RC_BAD( rc))
	{
		if( bCreatedDest)
		{
			(void)deleteFile( pszDestFileName);
		}
		
		*pui64BytesCopied = 0;
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Do a partial copy from one file into another file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::copyPartialFile(
	IF_FileHdl *	pSrcFileHdl,
	FLMUINT64		ui64SrcOffset,
	FLMUINT64		ui64SrcSize,
	IF_FileHdl *	pDestFileHdl,
	FLMUINT64		ui64DestOffset,
	FLMUINT64 *		pui64BytesCopiedRV)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucBuffer = NULL;
	FLMUINT			uiAllocSize = 65536;
	FLMUINT			uiBytesToRead;
	FLMUINT64		ui64CopySize;
	FLMUINT64		ui64FileOffset;
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;

	ui64CopySize = ui64SrcSize;
	*pui64BytesCopiedRV = 0;

	// Set the buffer size for use during the file copy

	if( ui64CopySize < uiAllocSize)
	{
		uiAllocSize = (FLMUINT)ui64CopySize;
	}

	// Allocate a buffer

	if( RC_BAD( rc = f_alloc( uiAllocSize, &pucBuffer)))
	{
		goto Exit;
	}

	// Position the file pointers

	if( RC_BAD( rc = pSrcFileHdl->seek( ui64SrcOffset, FLM_IO_SEEK_SET,
								&ui64FileOffset)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDestFileHdl->seek( ui64DestOffset, FLM_IO_SEEK_SET,
								&ui64FileOffset)))
	{
		goto Exit;
	}

	// Begin copying the data

	while( ui64CopySize)
	{
		if( ui64CopySize > uiAllocSize)
		{
			uiBytesToRead = uiAllocSize;
		}
		else
		{
			uiBytesToRead = (FLMUINT)ui64CopySize;
		}
		
		rc = pSrcFileHdl->read( FLM_IO_CURRENT_POS, uiBytesToRead,
										pucBuffer, &uiBytesRead);
										
		if (rc == NE_FLM_IO_END_OF_FILE)
		{
			rc = NE_FLM_OK;
		}
		
		if (RC_BAD( rc))
		{
			rc = RC_SET( NE_FLM_IO_COPY_ERR);
			goto Exit;
		}

		uiBytesWritten = 0;
		if( RC_BAD( rc = pDestFileHdl->write( FLM_IO_CURRENT_POS, uiBytesRead,
									pucBuffer, &uiBytesWritten)))
		{
			if (rc == NE_FLM_IO_DISK_FULL)
			{
				*pui64BytesCopiedRV += uiBytesWritten;
			}
			else
			{
				rc = RC_SET( NE_FLM_IO_COPY_ERR);
			}
			
			goto Exit;
		}
		
		*pui64BytesCopiedRV += uiBytesWritten;

		if( uiBytesRead < uiBytesToRead)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			goto Exit;
		}

		ui64CopySize -= uiBytesRead;
	}
	
Exit:

	if (pucBuffer)
	{
		(void)f_free( &pucBuffer);
	}

	return( rc);
}

/****************************************************************************
Desc: Rename a file.
****************************************************************************/
RCODE FTKAPI F_FileSystem::renameFile(
	const char *		pszFileName,
	const char *		pszNewFileName)
{
#if defined( FLM_WIN)

	DWORD			error;
	RCODE			rc = NE_FLM_OK;
	FLMUINT64	ui64BytesCopied;

   // Try to move the file by doing a rename first, otherwise copy the file

	if( (MoveFile( (LPTSTR)pszFileName, (LPTSTR)pszNewFileName)) != TRUE)
	{
		error = GetLastError();
		switch( error)
		{
			case ERROR_NOT_SAME_DEVICE:
			case ERROR_NO_MORE_FILES:
			case NO_ERROR:
				if( copyFile( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
				{
					rc = RC_SET( NE_FLM_IO_COPY_ERR);
				}
				else
				{
					rc = F_FileSystem::deleteFile( pszFileName);
				}
				break;
			default:
				rc = f_mapPlatformError( error, NE_FLM_RENAMING_FILE);
				break;
		}
	}

	return( rc);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	RCODE			rc;
	FLMBOOL		bSrcIsDir;
	FLMUINT64	ui64BytesCopied;

	if( RC_BAD( rc = targetIsDir( (char*)pszFileName, &bSrcIsDir)))
	{
		return( rc);
	}

	errno = 0;

	if( RC_BAD( renameSafe( pszFileName, pszNewFileName)))
	{
		switch( errno)
		{
			case EXDEV:
			{
				if( bSrcIsDir)
				{
					return( RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE));
				}
				else
				{
					if( copyFile( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
					{
						return( RC_SET( NE_FLM_IO_COPY_ERR));
					}
					
					F_FileSystem::deleteFile( pszFileName);
					return( NE_FLM_OK);
				}
			}

			default:
			{
				if( errno == ENOENT)
				{
					return( RC_SET( NE_FLM_IO_RENAME_FAILURE));
				}
				else
				{
					return( f_mapPlatformError( errno, NE_FLM_RENAMING_FILE));
				}
			}
		}
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareRenameFile( pszFileName, pszNewFileName));

#endif
}

/****************************************************************************
Desc: Get the sector size (not supported on all platforms).
****************************************************************************/
RCODE FTKAPI F_FileSystem::getSectorSize(
	const char *	pszFileName,
	FLMUINT *		puiSectorSize)
{
#ifdef FLM_NLM

	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = FLM_NLM_SECTOR_SIZE;
	return( NE_FLM_OK);
	
#elif defined( FLM_WIN)

	RCODE			rc = NE_FLM_OK;
	DWORD			udSectorsPerCluster;
	DWORD			udBytesPerSector;
	DWORD			udNumberOfFreeClusters;
	DWORD			udTotalNumberOfClusters;
	char			szVolume [256];
	char *		pszVolume;
	FLMUINT		uiLen;

	if (!pszFileName)
	{
		pszVolume = NULL;
	}
	else
	{
		pathParse( pszFileName, NULL, szVolume, NULL, NULL);
		if (!szVolume [0])
		{
			pszVolume = NULL;
		}
		else
		{
			uiLen = f_strlen( szVolume);
			if (szVolume [uiLen - 1] == ':')
			{
				szVolume [uiLen] = '\\';
				szVolume [uiLen + 1] = 0;
			}
			pszVolume = &szVolume [0];
		}
	}

	if( !GetDiskFreeSpace( (LPCTSTR)pszVolume, &udSectorsPerCluster,
			&udBytesPerSector, &udNumberOfFreeClusters,
			&udTotalNumberOfClusters))
	{
		f_assert( 0);
		rc = f_mapPlatformError( GetLastError(), NE_FLM_INITIALIZING_IO_SYSTEM);
		*puiSectorSize = 0;
		goto Exit;
	}
	
	*puiSectorSize = (FLMUINT)udBytesPerSector;
	
Exit:

	return( rc);

#elif defined( DEV_BSIZE)
	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = (FLMUINT)DEV_BSIZE;
	return( NE_FLM_OK);
#else

	RCODE				rc = NE_FLM_OK;
	int				hFile = -1;
	struct stat		filestats;
	
	if( (hFile = open( pszFileName, O_RDONLY, 0600)) == -1)
	{
		rc = f_mapPlatformError( errno, NE_FLM_OPENING_FILE);
		goto Exit;
	}
	
	if( fstat( hFile, &filestats) != 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_OPENING_FILE);
		goto Exit;
	}

	*puiSectorSize = (FLMUINT)filestats.st_blksize;
	
Exit:

	if( hFile != -1)
	{
		close( hFile);
	}
	
	return( rc);
#endif
}

/****************************************************************************
Desc: Set the Read-Only Attribute (not supported on all platforms).
****************************************************************************/
RCODE FTKAPI F_FileSystem::setReadOnly(
	const char *	pszFileName,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = NE_FLM_OK;

#if defined( FLM_WIN)

	DWORD				dwAttr;

	dwAttr = GetFileAttributes( (LPTSTR)pszFileName);
	if( dwAttr == (DWORD)-1)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	if( bReadOnly)
	{
		dwAttr |= FILE_ATTRIBUTE_READONLY;
	}
	else
	{
		dwAttr &= ~FILE_ATTRIBUTE_READONLY;
	}
	
	if( !SetFileAttributes( (LPTSTR)pszFileName, dwAttr))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	struct stat		filestatus;

	if( stat( (char *)pszFileName, &filestatus))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	if( bReadOnly)
	{
		filestatus.st_mode &= ~S_IWUSR;
	}
	else
	{
		filestatus.st_mode |= S_IWUSR;
	}
	
	if( chmod( (char *)pszFileName, filestatus.st_mode))
	{
		rc = RC_SET( NE_FLM_FAILURE);
		goto Exit;
	}
#else
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( bReadOnly);

	rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
	goto Exit;
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FTKAPI F_FileSystem::canDoAsync( void)
{
	return( m_bCanDoAsync);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE F_FileSystem::targetIsDir(
	const char	*	pszPath,
	FLMBOOL *		pbIsDir)
{
	struct stat		sbuf;
	RCODE				rc = NE_FLM_OK;

	*pbIsDir = FALSE;
	if( stat( pszPath, &sbuf) < 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_IO_ACCESS_DENIED);
	}
	else if( (sbuf.st_mode & S_IFMT) == S_IFDIR)
	{
		*pbIsDir = TRUE;
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:	Rename an existing file (typically an "X" locked file to an
		unlocked file) using a safe (non-race) method.  To ensure that
		an existing file is not being overwritten by a rename operation,
		we will first create a new file with the desired name (using the
		CREAT and EXCL options, (ensuring a unique file name)).  Then,
		the source file will be renamed to new name.
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE F_FileSystem::renameSafe(
	const char *	pszSrcFile,
	const char *	pszDestFile)
{
	RCODE			rc = NE_FLM_OK;
	struct stat	temp_stat_buf;
	
	errno = 0;
	if( stat( pszDestFile, &temp_stat_buf) != -1)
	{
		// If we were able to stat it, then the file obviously exists...
		
		rc = RC_SET( NE_FLM_IO_RENAME_FAILURE);
		goto Exit;
	}
	else
	{
		if (errno != ENOENT)
		{
			// ENOENT means the file didn't exist, which is what we were
			// hoping for.
			
			rc = f_mapPlatformError( errno, NE_FLM_IO_RENAME_FAILURE);
			goto Exit;
		}
	}

	errno = 0;
	if( rename( pszSrcFile, pszDestFile) != 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_IO_RENAME_FAILURE);
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;
	F_FileSystem	fileSystem;
	FLMUINT 			uiBytesWritten = 0;

	if( RC_OK( rc = fileSystem.doesFileExist( pszSourceFile)))
	{
		if( RC_BAD( rc = fileSystem.deleteFile( pszSourceFile)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = fileSystem.createFile( pszSourceFile, FLM_IO_RDWR,
		&pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->write( 0, f_strlen( pszData), (void *)pszData,
		&uiBytesWritten)))
	{
		goto Exit;
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->closeFile();
		pFileHdl->Release();
		pFileHdl = NULL;
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_filecat(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT64 			ui64FileSize = 0;
	FLMUINT 				uiBytesWritten = 0;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if (RC_BAD( rc = pFileSystem->doesFileExist( pszSourceFile)))
	{
		if( rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = pFileSystem->createFile( 
				pszSourceFile, FLM_IO_RDWR, &pFileHdl)))
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
		if( RC_BAD( rc = pFileSystem->openFile( pszSourceFile,
			FLM_IO_RDWR, &pFileHdl)))
		{
			goto Exit;
		}
	}

	if ( RC_BAD( rc = pFileHdl->size( &ui64FileSize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->write( ui64FileSize, f_strlen( pszData),
		(void *)pszData, &uiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Split the path into its components
Output:
		pServer - pointer to a buffer to hold the server name
		pVolume - pointer to a buffer to hold the volume name
		pDirPath  - pointer to a buffer to hold the path
		pFileName pointer to a buffer to hold the filename

		All of the output parameters are optional.  If you do not want one
		of the components, simply give a NULL pointer.

Note: if the input path has no file name, d:\dir_1 for example, then
		pass a NULL pointer for pFileName.  Otherwise dir_1 will be returned
		as pFileName.

		The server name may be ommitted in the input path:
			sys:\system\autoexec.ncf

		UNC paths of the form:
			\\server-name\volume-name\dir_1\dir_2\file.ext
		are supported.

		DOS paths of the form:
			d:\dir_1\dir_2\file.ext
		are also supported.

Example:
		Given this input:  orm-prod48/sys:\system\autoexec.ncf
		The output would be:
			pServer = "orm-prod48"
			pVolume = "sys:"
			pDirPath  = "\system"
			pFileName "autoexec.ncf"
****************************************************************************/
void FTKAPI F_FileSystem::pathParse(
	const char *		pszInputPath,
	char *				pszServer,
	char *				pszVolume,
	char *				pszDirPath,
	char *				pszFileName)
{
	char					szInput[ F_PATH_MAX_SIZE];
	char *				pszNext;
	char *				pszColon;
	char *				pszComponent;
	FLMUINT				uiEndChar;
	FLMBOOL				bUNC = FALSE;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	// Initialize return buffers

	if (pszServer)
	{
		*pszServer = 0;
	}
	
	if (pszVolume)
	{
		*pszVolume = 0;
	}
	
	if (pszDirPath)
	{
		*pszDirPath = 0;
	}
	
	if (pszFileName)
	{
		// Get the file name

		*pszFileName = 0;
		pFileSystem->pathReduce( pszInputPath, szInput, pszFileName);
	}
	else
	{
		f_strcpy( szInput, pszInputPath);
	}
	
	// Split out the rest of the components

	pszComponent = &szInput [0];

	// Is this a UNC path?

	if (szInput[0] == '\\' && szInput[1] == '\\')
	{

		// Yes, assume a UNC path

		pszComponent += 2;
		bUNC = TRUE;
	}

	pszNext = pszColon = pszComponent;

	// Is there a ':' in the szInput path?

	while (*pszColon && *pszColon != ':')
	{
		pszColon++;
	}
	
	if (*pszColon || bUNC)
	{
		
		// Yes, assume there is a volume in the path
		
		pszComponent = f_getPathComponent( &pszNext, &uiEndChar);
		if (uiEndChar != ':')
		{
			// Assume that this component is the server

			if (pszServer)
			{
				f_strcpy( pszServer, pszComponent);
			}

			// Get the next component

			pszComponent = f_getPathComponent( &pszNext, &uiEndChar);
		}

		// Assume that this component is the volume

		if (pszVolume)
		{
			char *	pszSrc = pszComponent;
			char *	pszDst = pszVolume;

			while (*pszSrc)
			{
				*pszDst++ = *pszSrc++;
			}
			*pszDst++ = ':';
			*pszDst = 0;
		}

		// For UNC paths, the leading '\' of the path is set to 0 by
		// f_getPathComponent.  This code restores the leading '\'.

		if (f_isSlashSeparator( (char)uiEndChar))
		{
			*(--pszNext) = (char)uiEndChar;
		}
	}

	// Assume that all that is left of the input is the path

	if (pszDirPath)
	{
		f_strcpy( pszDirPath, pszNext);
	}
}	

/****************************************************************************
Desc:		This function will strip off the filename or trailing
			directory of a path.  The stripped component of the path will
			be placed into the area pointed at by string.  The source
			path will not be modified.  The dest path will contain the
			remainder of the stripped path.  A stripped path can be processed
			repeatedly by this function until there is no more path to reduce.
			If the string is set to NULL, the copying of the stripped portion of
			the path will be bypassed by the function.

Notes:	This function handles drive based, UNC, Netware, and UNIX type
			paths.
****************************************************************************/
RCODE FTKAPI F_FileSystem::pathReduce(
	const char *	pszPath,
	char *			pszDir,
	char * 			pszPathComponent)
{
	RCODE				rc = NE_FLM_OK;
	const char *	pszFileNameStart;
	char				szLocalPath[ F_PATH_MAX_SIZE];
	FLMUINT			uiLen;

	// Check for valid path pointers

	if( !pszPath || !pszDir)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if ((uiLen = f_strlen( pszPath)) == 0)
	{
		rc = RC_SET( NE_FLM_IO_CANNOT_REDUCE_PATH);
		goto Exit;
	}

	// Trim out any trailing slash separators
	
	if( f_isSlashSeparator( pszPath [uiLen - 1]))
	{
		f_strcpy( szLocalPath, pszPath);
		
		while( f_isSlashSeparator( szLocalPath[ uiLen - 1]))
		{
			szLocalPath[ --uiLen] = 0;
			if( !uiLen)
			{
				rc = RC_SET( NE_FLM_IO_CANNOT_REDUCE_PATH);
				goto Exit;
			}
		}
		
		pszPath = szLocalPath;
	}

	if( f_canReducePath( pszPath))
	{
		// Search for a slash or beginning of path

		pszFileNameStart = f_findFileNameStart( pszPath);

		// Copy the sliced portion of the path if requested by caller

		if( pszPathComponent)
		{
			f_strcpy( pszPathComponent, pszFileNameStart);
		}

		// Copy the reduced source path to the dir path

		if (pszFileNameStart > pszPath)
		{
			uiLen = (FLMUINT)(pszFileNameStart - pszPath);
			f_memcpy( pszDir, pszPath, uiLen);

			if (uiLen >= 2 && f_isSlashSeparator( pszDir [uiLen - 1])
#ifndef FLM_UNIX
				 && pszDir [uiLen - 2] != ':'
#endif
				 )
			{
				// Trim off the trailing path separator

				pszDir [uiLen - 1] = 0;
			}
			else
			{
				pszDir [uiLen] = 0;
			}
		}
		else
		{
			*pszDir = 0;
		}
	}
	else
	{
		// We've found the drive id or server\volume specifier.

		if (pszPathComponent)
		{
			f_strcpy( pszPathComponent, pszPath);
		}
		
		*pszDir = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:   Internal function for WpioPathBuild() and WpioPathModify().
	     Appends string the path & adds a path delimiter if necessary.
In:     *path     = pointer to an IO_PATH
	     *string   = pointer to a NULL terminated string
	     *end_ptr  = pointer to the end of the IO_PATH which is being built.
****************************************************************************/
RCODE FTKAPI F_FileSystem::pathAppend(
	char *			pszPath,
	const char *	pszPathComponent)
{

	// Don't put a slash separator if pszPath is empty

	if (*pszPath)
	{
		FLMUINT		uiStrLen = f_strlen( pszPath);
		char *		pszEnd = pszPath + uiStrLen - 1;

		if (!f_isSlashSeparator( *pszEnd))
		{

		   // Check for maximum path size - 2 is for slash separator
			// and null byte.

			if (uiStrLen + 2 + f_strlen( pszPathComponent) > F_PATH_MAX_SIZE)
			{
				return RC_SET_AND_ASSERT( NE_FLM_IO_PATH_TOO_LONG);
			}

			pszEnd++;
#if defined( FLM_UNIX)
			*pszEnd = '/';
#else
			*pszEnd = '\\';
#endif
		}
		else
		{

		   // Check for maximum path size +1 is for null byte.

			if (uiStrLen + 1 + f_strlen( pszPathComponent) > F_PATH_MAX_SIZE)
			{
				return RC_SET_AND_ASSERT( NE_FLM_IO_PATH_TOO_LONG);
			}
		}

		f_strcpy( pszEnd + 1, pszPathComponent);
	}
	else
	{
		f_strcpy( pszPath, pszPathComponent);
	}

   return( NE_FLM_OK);
}

/****************************************************************************
Desc:	Convert an PATH into a fully qualified, storable C string
		reference to a file or directory.
In:	pszPath - the path to convert.
		pszStorageString - a pointer to a string that is atleast 
		F_PATH_MAX_SIZE in size
****************************************************************************/
RCODE FTKAPI F_FileSystem::pathToStorageString(
	const char *	pszPath,
	char *			pszStorageString)
{
#ifdef FLM_WIN

	char *	pszNamePart;

	if (GetFullPathName( (LPCSTR)pszPath,
							(DWORD)F_PATH_MAX_SIZE - 1,
							(LPSTR)pszStorageString,
							(LPSTR *)&pszNamePart) != 0)
	{

	}
	else
	{
		// Convert to upper case.

		while (*pszPath)
		{
			*pszStorageString++ = *pszPath;
			pszPath++;
		}
		
		*pszStorageString = 0;
	}
	
	return( NE_FLM_OK);

#elif defined( FLM_RING_ZERO_NLM)

	while (*pszPath)
	{
		*pszStorageString++ = f_toupper( *pszPath);
		pszPath++;
	}
	*pszStorageString = 0;
	return( NE_FLM_OK);
	
#else

	char			szFile[ F_PATH_MAX_SIZE];
	char			szDir[ F_PATH_MAX_SIZE];
	char *		pszRealPath = NULL;
	RCODE			rc = NE_FLM_OK;

	if (RC_BAD( rc = pathReduce( pszPath, szDir, szFile)))
	{
		goto Exit;
	}

	if (!szDir [0])
	{
		szDir [0] = '.';
		szDir [1] = '\0';
	}

	if (RC_BAD( rc = f_alloc( (FLMUINT)PATH_MAX, &pszRealPath)))
	{
		goto Exit;
	}

	if (!realpath( (char *)szDir, (char *)pszRealPath))
	{
		rc = f_mapPlatformError( errno, NE_FLM_PARSING_FILE_NAME);
		goto Exit;
	}

	if (f_strlen( pszRealPath) >= F_PATH_MAX_SIZE)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_IO_PATH_TOO_LONG);
		goto Exit;
	}

	f_strcpy( pszStorageString, pszRealPath);

	if (RC_BAD( rc = pathAppend( pszStorageString, szFile)))
	{
		goto Exit;
	}

Exit:

	if (pszRealPath)
	{
		f_free( &pszRealPath);
	}

	return( rc);

#endif
}

/****************************************************************************
Desc:		Generates a file name given a seed and some modifiers, it is built
			to be called in a loop until the file can be sucessfully written or
			created with the increment being changed every time.
In:		bModext		-> if TRUE then we will use the extension for collisions.
In\Out:	puiTime		-> a modified time stamp which is used as the base
								filename.  To properly set up this value, make sure
								the puiTime points to a 0 the first time this routine
								is called and it will be set up for you.  Thereafter,
                        do not change it between calls.
			pHighChars->	these are the 8 bits that were shifted off the top of
								the time struct.  It will be set up for you the first
								time you call this routine if puiTime points to a 0
								the first time this routine is called.  Do not change
								this value between calls.
			pszFileName -> should be pointing to a null string on the way in.
								going out it will be the complete filename.
			pszFileExt	-> the last char of the ext will be used for collisions,
								depending on the bModext flag.  If null then
								the extension will be .00x where x is the collision
								counter.
Notes:	The counter on the collision is 0-9, a-z.
****************************************************************************/
void FTKAPI F_FileSystem::pathCreateUniqueName(
	FLMUINT *		puiTime,
	char *			pszFileName,
	const char *	pszFileExt,
	FLMBYTE *		pHighChars,
	FLMBOOL			bModext)
{
	FLMINT		iCount, iLength;
	FLMUINT		uiSdTmp = 0;
	FLMUINT		uiIncVal = 1;

	f_setupTime( puiTime, pHighChars);
	uiSdTmp = *puiTime;

	*(pszFileName + 8) = NATIVE_DOT;
	f_memset( (pszFileName + 9), NATIVE_ZERO, 3 );
	
	if( (pszFileExt != NULL))
	{
		if ((iLength = f_strlen(pszFileExt)) > 3)
		{
			iLength = 3;
		}
		
      f_memmove( (pszFileName + 9), pszFileExt, iLength);
   }

	if( bModext)
	{
		f_hexToNative((FLMBYTE)(uiSdTmp & 0x0000001F), pszFileName+(11));
   }
	else
	{
	 	uiIncVal = 32;
	}
	
	uiSdTmp = uiSdTmp >> 5;
	for( iCount = 0; iCount < 6; iCount++)
	{
		f_hexToNative((FLMBYTE)(uiSdTmp & 0x0000000F), pszFileName+(7-iCount));
		uiSdTmp = uiSdTmp >> 4;
	}

	for( iCount = 0; iCount < 2; iCount++)
	{
		f_hexToNative((FLMBYTE)(*pHighChars & 0x0000000F), pszFileName+(1-iCount));
		*pHighChars = *pHighChars >> 4;
	}

   *(pszFileName + 12) = '\0';
	*puiTime += uiIncVal;
}

/****************************************************************************
Desc:		Compares the current file against a pattern template
****************************************************************************/
FLMBOOL FTKAPI F_FileSystem::doesFileMatch(
	const char *	pszFileName,
	const char *	pszTemplate)
{
	FLMUINT		uiPattern;
	FLMUINT		uiChar;

	if( !*pszTemplate)
	{
		return( TRUE);
	}

	while( *pszTemplate)
	{
		uiPattern = *pszTemplate++;
		switch( uiPattern)
		{
			case NATIVE_WILDCARD:
			{
				if( *pszTemplate == 0)
				{
					return( TRUE);
				}

				while( *pszFileName)
				{
					if( doesFileMatch( pszFileName, pszTemplate))
					{
						return( TRUE);
					}
					pszFileName++;
				}
				
				return( FALSE);
			}
			
			case NATIVE_QUESTIONMARK:
			{
				if( *pszFileName++ == 0)
				{
					return( FALSE);
				}
				break;
			}
			
			default:
			{
				uiChar = *pszFileName++;
				if( f_toupper( uiPattern) != f_toupper( uiChar))
				{
					return( FALSE);
				}
				break;
			}
		}
	}

	return( (*pszFileName != 0) ? FALSE : TRUE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FTKAPI F_FileSystem::deleteMultiFileStream(
	const char *			pszDirectory,
	const char *			pszBaseName)
{
	RCODE						rc = NE_FLM_OK;
	F_MultiFileOStream *	pMultiStream = NULL;
	
	if( (pMultiStream = f_new F_MultiFileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pMultiStream->processDirectory( 
		pszDirectory, pszBaseName, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	if( pMultiStream)
	{
		pMultiStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE FTKAPI F_FileSystem::createLockFile(
	const char *		pszPath,
	IF_FileHdl **		ppLockFileHdl)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileHdl *		pLockFileHdl = NULL;

	// Attempt to create the lock file.  If that succeeds, we have 
	// the lock.  If it fails, the lock file may have been left because
	// of a crash.  Hence, we first try to delete the file.  If that succeeds,
	// we then attempt to create the file again.  If that, or the 2nd create
	// fail, we simply return an access denied error.

#ifdef FLM_UNIX
	if( RC_BAD( createFile( pszPath,
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW, 
			&pLockFileHdl)))
	{
		if( RC_BAD( openFile( pszPath, 
			FLM_IO_RDWR | FLM_IO_SH_DENYRW, &pLockFileHdl)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
	}

	if( RC_BAD( pLockFileHdl->lock()))
	{
		rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
		goto Exit;
	}
#else
	if( RC_BAD( createFile( pszPath,
		FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW | FLM_IO_DELETE_ON_RELEASE,
		&pLockFileHdl)))
	{
		if( RC_BAD( deleteFile( pszPath)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if( RC_BAD( createFile( pszPath, 
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW | FLM_IO_DELETE_ON_RELEASE,
			&pLockFileHdl)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
	}
#endif

	*ppLockFileHdl = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_filetobuf(
	const char *		pszSourceFile,
	char **				ppszBuffer)
{
	RCODE					rc = NE_FLM_OK;
	char *				pszBuffer = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT64			ui64FileSize;
	FLMUINT				uiBytesRead;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	if( RC_BAD(rc = pFileSystem->openFile( pszSourceFile,
		FLM_IO_RDONLY, &pFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pFileHdl->size( &ui64FileSize)))
	{
		goto Exit;
	}
	
	if( !ui64FileSize)
	{
		*ppszBuffer = NULL;
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( (FLMUINT)(ui64FileSize + 1), &pszBuffer)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->read( 0, (FLMUINT)ui64FileSize,
		pszBuffer, &uiBytesRead)))
	{
		goto Exit;
	}
	
	f_assert( (FLMUINT)ui64FileSize == uiBytesRead);
	pszBuffer[ ui64FileSize] = 0;
	*ppszBuffer = pszBuffer;
	pszBuffer = NULL;
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pszBuffer)
	{
		f_free( &pszBuffer);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_pathReduce(
	const char *			pszSourcePath,
	char *					pszDestPath,
	char *					pszString)
{
	return( f_getFileSysPtr()->pathReduce( 
		pszSourcePath, pszDestPath, pszString));
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI f_pathAppend(
	char *					pszPath,
	const char *			pszPathComponent)
{
	return( f_getFileSysPtr()->pathAppend( pszPath, pszPathComponent)); 
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdlCache::F_FileHdlCache()
{
	m_pTimeoutThread = NULL;
	m_pAvailList = NULL;
	m_uiMaxIdleTime = 0;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_FileHdlCache::~F_FileHdlCache()
{
	if( m_pTimeoutThread)
	{
		m_pTimeoutThread->stopThread();
		m_pTimeoutThread->Release();
		m_pTimeoutThread = NULL;
	}
	
	if( m_pAvailList)
	{
		m_pAvailList->Release();
		m_pAvailList = NULL;
	}
	
	m_uiMaxIdleTime = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlCache::setup(
	FLMUINT		uiMaxCachedFiles,
	FLMUINT		uiIdleTimeoutSecs)
{
	RCODE			rc = NE_FLM_OK;
	
	if( (m_pAvailList = f_new F_HashTable) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pAvailList->setupHashTable( TRUE, 
		uiMaxCachedFiles, uiMaxCachedFiles)))
	{
		goto Exit;
	}
	
	m_uiMaxIdleTime = uiIdleTimeoutSecs;
	
	if( RC_BAD( rc = f_threadCreate( &m_pTimeoutThread,
		timeoutThread, "F_FileHdlCache Timeout", F_DEFAULT_THREAD_GROUP, 0,
		this, NULL, F_THREAD_DEFAULT_STACK_SIZE)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdlCache::setOpenThreshold(
	FLMUINT					uiMaxOpenFiles)
{
	return( m_pAvailList->setMaxObjects( uiMaxOpenFiles));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlCache::openOrCreate(
	const char *			pszFileName,
	FLMUINT					uiIoFlags,
	FLMBOOL					bCreate,
	IF_FileHdl **			ppFile)
{
	RCODE						rc = NE_FLM_OK;
	F_CachedFileHdl *		pFileHdl = NULL;
	F_HashObject *			pHashObject;
	FLMUINT					uiNameLen = f_strlen( pszFileName);
	FLMUINT					uiKeyLen = uiNameLen + 4;
	FLMBYTE					ucKeyBuf[ F_PATH_MAX_SIZE + 4];
	
	UD2FBA( uiIoFlags, ucKeyBuf);
	f_memcpy( &ucKeyBuf[ 4], pszFileName, uiNameLen);
	
	if( RC_BAD( rc = m_pAvailList->getObject( ucKeyBuf, uiKeyLen, 
		&pHashObject, TRUE)))
	{
		if( rc != NE_FLM_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( (pFileHdl = f_new F_CachedFileHdl) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = pFileHdl->openOrCreate( pszFileName, 
			uiIoFlags, bCreate)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = f_alloc( uiKeyLen, &pFileHdl->m_pucKey)))
		{
			goto Exit;
		}
		
		f_memcpy( pFileHdl->m_pucKey, ucKeyBuf, uiKeyLen);
		pFileHdl->m_uiKeyLen = uiKeyLen;
		pFileHdl->m_pFileHdlCache = this;
	}
	else
	{
		pFileHdl = (F_CachedFileHdl *)pHashObject;
		pFileHdl->m_bInAvailList = FALSE;

		if( bCreate)
		{
			if( RC_BAD( rc = pFileHdl->truncateFile()))
			{
				goto Exit;
			}
		}
	}
	
	*ppFile = pFileHdl;
	pFileHdl = NULL;
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_FileHdlCache::closeUnusedFiles(
	FLMUINT			uiUnusedTime)
{
	if( !uiUnusedTime)
	{
		m_pAvailList->removeAllObjects();
	}
	else
	{
		m_pAvailList->removeAgedObjects( uiUnusedTime);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlCache::timeoutThread(
	IF_Thread *			pThread)
{
	FLMUINT				uiCurrentTime;
	FLMUINT				uiLastPurgeTime = FLM_GET_TIMER();
	F_FileHdlCache *	pThis = (F_FileHdlCache *)pThread->getParm1();
	
	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}
		
		uiCurrentTime = FLM_GET_TIMER();
		
		if( FLM_TIMER_UNITS_TO_SECS( 
			FLM_ELAPSED_TIME( uiCurrentTime, uiLastPurgeTime)) >= 
				pThis->m_uiMaxIdleTime)
		{
			pThis->m_pAvailList->removeAgedObjects( pThis->m_uiMaxIdleTime);
			uiLastPurgeTime = uiCurrentTime;
		}

		f_sleep( 100);
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI F_CachedFileHdl::Release( void)
{
	FLMINT					iRefCnt = f_atomicDec( &m_refCnt);
	F_FileHdlCache *		pFileHdlCache = m_pFileHdlCache;
	
	if( !iRefCnt)
	{
		if( pFileHdlCache)
		{
			f_assert( getHashBucket() == F_INVALID_HASH_BUCKET);

			if( m_bInAvailList)
			{
				// This should only happen if the object is being released
				// by the avail list hash table
				
				m_bInAvailList = FALSE;
			}
			else if( isOpen())
			{
				if( RC_OK( pFileHdlCache->m_pAvailList->addObject( this, TRUE)))
				{
					m_bInAvailList = TRUE;
				}
			}
		}

		if( !m_refCnt)
		{
			delete this;
		}
	}

	return( iRefCnt);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_FileSystem::allocFileHandleCache(
	FLMUINT						uiMaxCachedFiles,
	FLMUINT						uiIdleTimeoutSecs,
	IF_FileHdlCache **		ppFileHdlCache)
{
	RCODE							rc = NE_FLM_OK;
	F_FileHdlCache *			pFileHdlCache = NULL;
	
	if( (pFileHdlCache = f_new F_FileHdlCache) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pFileHdlCache->setup( 
		uiMaxCachedFiles, uiIdleTimeoutSecs)))
	{
		goto Exit;
	}
	
	*ppFileHdlCache = pFileHdlCache;
	pFileHdlCache = NULL;
	
Exit:

	if( pFileHdlCache)
	{
		pFileHdlCache->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FTKAPI F_FileSystem::allocIOBuffer(
	FLMUINT				uiMinSize,
	IF_IOBuffer **		ppIOBuffer)
{
	RCODE					rc = NE_FLM_OK;
	F_IOBuffer *		pIOBuffer = NULL;
	
	if( (pIOBuffer = f_new F_IOBuffer) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pIOBuffer->setupBuffer( uiMinSize, NULL)))
	{
		goto Exit;
	}
	
	*ppIOBuffer = pIOBuffer;
	pIOBuffer = NULL;
	
Exit:

	if( pIOBuffer)
	{
		pIOBuffer->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_FileHdl::initCommonData( void)
{
	m_pszFileName = NULL;
	m_uiBytesPerSector = 0;
	m_ui64NotOnSectorBoundMask = 0;
	m_ui64GetSectorBoundMask = 0;
	m_uiExtendSize = 0;
	m_uiMaxAutoExtendSize = 0;
	m_pucAlignedBuff = NULL;
	m_uiAlignedBuffSize = 0;
	m_ui64CurrentPos = 0;
	m_bFileOpened = FALSE;
	m_bDeleteOnRelease = FALSE;
	m_bOpenedReadOnly = FALSE;
	m_bOpenedExclusive = FALSE;
	m_bOpenedInAsyncMode = FALSE;
	m_bRequireAlignedIO = FALSE;
	m_bDoDirectIO = FALSE;
	m_numAsyncPending = 0;
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_FileHdl::freeCommonData( void)
{
	f_assert( !m_numAsyncPending);
	
	if( m_pucAlignedBuff)
	{
		f_freeAlignedBuffer( (void **)&m_pucAlignedBuff);
		m_uiAlignedBuffSize = 0;
	}
	
	if( m_pszFileName)
	{
		f_free( &m_pszFileName);
	}
}

/****************************************************************************
Desc:	Open a file
****************************************************************************/
RCODE F_FileHdl::openFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	return( openOrCreate( pszFileName, uiIoFlags, FALSE));
}

/****************************************************************************
Desc:	Create a file
****************************************************************************/
RCODE F_FileHdl::createFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	return( openOrCreate( pszFileName, uiIoFlags, TRUE));
}

/****************************************************************************
Desc:	Create a unique file name in the specified directory
****************************************************************************/
RCODE F_FileHdl::createUniqueFile(
	char *				pszDirName,
	const char *		pszFileExtension,
	FLMUINT				uiIoFlags)
{
	RCODE					rc = NE_FLM_OK;
	char *				pszTmp;
	FLMBOOL				bModext = TRUE;
	FLMUINT				uiBaseTime = 0;
	FLMBYTE				ucHighByte = 0;
	char					szFileName[ F_FILENAME_SIZE];
	char					szDirPath[ F_PATH_MAX_SIZE];
	char					szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT				uiCount;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_assert( !m_bFileOpened);
	f_assert( !m_pszFileName);
	
	szFileName[0] = '\0';
	szTmpPath[0] = '\0';

#if defined( FLM_UNIX)	
	if( !pszDirName || pszDirName[ 0] == '\0')
	{
		f_strcpy( szDirPath, "./");
	}
	else
#endif
	{
		f_strcpy( szDirPath, pszDirName);
	}

   // Truncate any trailing spaces

	pszTmp = (char *)szDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	
	while( pszTmp >= (char *)szDirPath && (*pszTmp == 0x20))
	{
		*pszTmp = 0;
		pszTmp--;
	}

	// Append a slash if one isn't already there

#if defined( FLM_UNIX)
	if( pszTmp >= (char *)szDirPath && *pszTmp != '/')
	{
		pszTmp++;
		*pszTmp++ = '/';
	}
#else
	if( pszTmp >= (char *)szDirPath && *pszTmp != '\\')
	{
		pszTmp++;
		*pszTmp++ = '\\';
	}
#endif
	else
	{
		pszTmp++;
	}
	
	*pszTmp = 0;

	if( pszFileExtension && f_strlen( pszFileExtension) >= 3)
	{
		bModext = FALSE;
	}

	uiCount = 0;
	do
	{
		pFileSystem->pathCreateUniqueName( &uiBaseTime, szFileName, 
			pszFileExtension, &ucHighByte, bModext);

		f_strcpy( szTmpPath, szDirPath);
		pFileSystem->pathAppend( szTmpPath, szFileName);
		
		rc = createFile( szTmpPath, uiIoFlags | FLM_IO_EXCL);
		
	} while( rc != NE_FLM_OK && (uiCount++ < 10));

   // Check if the path was created

   if( uiCount >= 10 && rc != NE_FLM_OK)
   {
		rc = RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }

	// Created file name needs to be returned
	
	f_strcpy( pszDirName, szTmpPath);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::getPreWriteExtendSize(
	FLMUINT64		ui64WriteOffset,
	FLMUINT			uiBytesToWrite,
	FLMUINT64 *		pui64CurrFileSize,
	FLMUINT *		puiTotalBytesToExtend)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiTotalBytesToExtend = 0;
	FLMUINT64		ui64CurrFileSize = 0;
	
	// Determine if the write will extend the file beyond its
	// current size.
	
	if( RC_BAD( rc = size( &ui64CurrFileSize)))
	{
		goto Exit;
	}
	
	if( ui64WriteOffset + uiBytesToWrite > ui64CurrFileSize)
	{
		if( (uiTotalBytesToExtend = m_uiExtendSize) != 0)
		{
			if( ui64CurrFileSize > m_uiMaxAutoExtendSize)
			{
				uiTotalBytesToExtend = 0;
			}
			else
			{
				// Don't extend beyond maximum file size.

				if( m_uiMaxAutoExtendSize - ui64CurrFileSize < uiTotalBytesToExtend)
				{
					uiTotalBytesToExtend = 
						(FLMUINT)(m_uiMaxAutoExtendSize - ui64CurrFileSize);
				}

				// If the extend size is not on a sector boundary, round it down.

				uiTotalBytesToExtend = 
					(FLMUINT)truncateToPrevSector( uiTotalBytesToExtend);
			}
		}
	}
	
Exit:

	*pui64CurrFileSize = ui64CurrFileSize;
	*puiTotalBytesToExtend = uiTotalBytesToExtend;

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FTKAPI F_FileAsyncClient::Release(
	FLMBOOL		bOkToReuse)
{
	FLMINT		iRefCnt;
	
	if( m_refCnt == 1)
	{
#if !defined( FLM_UNIX)
		FLMUINT		uiSignalCount = f_semGetSignalCount( m_hSem);
#endif

#if !defined( FLM_UNIX)
		f_assert( uiSignalCount <= 1);
#endif

		f_assert( !m_pIOBuffer);
		f_assert( !m_pFileHdl);

#if !defined( FLM_UNIX)
		if( uiSignalCount == 1)
		{
			// The application may not have cared to wait on this
			// individual write to complete.  Since the
			// semaphore has been signaled, we need to consume
			// the signal so that the next time this
			// async client is used, the semaphore will
			// block until that I/O operation is complete.

			f_semWait( m_hSem, F_WAITFOREVER);
		}
#endif

		if( m_pIOBuffer)
		{
			m_pIOBuffer->Release();
			m_pIOBuffer = NULL;
		}

		if( m_pFileHdl)
		{
			m_pFileHdl->Release();
			m_pFileHdl = NULL;
		}

		if( bOkToReuse)
		{
			f_mutexLock( F_FileHdl::m_hAsyncListMutex);
			
			if( F_FileHdl::m_uiAvailAsyncCount < 32)
			{
				f_assert( !m_pNext);
				m_pNext = F_FileHdl::m_pFirstAvailAsync;
				F_FileHdl::m_pFirstAvailAsync = this;
				F_FileHdl::m_uiAvailAsyncCount++;

				m_completionRc = NE_FLM_OK;
				m_uiBytesToDo = 0;
				m_uiBytesDone = 0;
				iRefCnt = m_refCnt;
			}
			else
			{
				iRefCnt = f_atomicDec( &m_refCnt);
			}
			
			f_mutexUnlock( F_FileHdl::m_hAsyncListMutex);
		}
		else
		{
			iRefCnt = f_atomicDec( &m_refCnt);
		}
	}
	else
	{
		iRefCnt = f_atomicDec( &m_refCnt);
	}
	
	if( !m_refCnt)
	{
		delete this;
	}

	return( iRefCnt);
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_FileAsyncClient::~F_FileAsyncClient()
{
	f_assert( !m_pNext);
	f_assert( !m_refCnt);
	
#if !defined( FLM_UNIX)
	if( m_hSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hSem);
	}
#endif
	
	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileAsyncClient::prepareForAsync(
	IF_IOBuffer *		pIOBuffer)
{
	RCODE					rc = NE_FLM_OK;

	if( m_pIOBuffer || !m_pFileHdl)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
#if !defined( FLM_UNIX)
	if( m_hSem == F_SEM_NULL)
	{
		if( RC_BAD( rc = f_semCreate( &m_hSem)))
		{
			goto Exit;
		}
	}

	f_assert( f_semGetSignalCount( m_hSem) == 0);
#endif

#if defined( FLM_UNIX) && defined( FLM_HAS_ASYNC_IO)
	f_memset( &m_aio, 0, sizeof( m_aio));
#endif
	
	m_completionRc = NE_FLM_IO_PENDING;
	m_uiBytesToDo = 0;
	m_uiBytesDone = 0;
	m_uiStartTime = FLM_GET_TIMER();
	m_uiEndTime = m_uiStartTime;
	
	if( pIOBuffer)
	{
		pIOBuffer->setAsyncClient( this);
		m_pIOBuffer = pIOBuffer;
		m_pIOBuffer->AddRef();
		m_pIOBuffer->setPending();
	}
	
	f_atomicInc( &m_pFileHdl->m_numAsyncPending);
	
Exit:

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileAsyncClient::waitToComplete( void)
{
	RCODE			completionRc = NE_FLM_OK;

#if defined( FLM_UNIX)
	FLMUINT		uiBytesDone = 0;

	#if defined( FLM_HAS_ASYNC_IO)
	{
		FLMINT						iAsyncResult;
		const struct aiocb *		ppAio[ 1];
		
		ppAio[ 0] = &m_aio;
	
		for( ;;)
		{
		#ifdef FLM_AIX
			aio_suspend( 1, ppAio);
		#else
			aio_suspend( ppAio, 1, NULL);
		#endif
			iAsyncResult = aio_error( &m_aio);
	
			if( !iAsyncResult)
			{
				if( (iAsyncResult = aio_return( &m_aio)) < 0)
				{
					f_assert( 0);
					completionRc = f_mapPlatformError( errno, NE_FLM_ASYNC_FAILED);
				}
				else
				{
					uiBytesDone = (FLMUINT)iAsyncResult;
				}
					
				break;
			}
				
			if( iAsyncResult == EINTR || iAsyncResult == EINPROGRESS)
			{
				continue;
			}
					
			f_assert( 0);
			completionRc = f_mapPlatformError( iAsyncResult, NE_FLM_ASYNC_FAILED);
		}
	}
	#endif

	m_completionRc = completionRc;	
	m_uiBytesDone = uiBytesDone;

#else
	if( RC_BAD( completionRc = f_semWait( m_hSem, F_WAITFOREVER)))
	{
		return( completionRc);
	}

	f_assert( f_semGetSignalCount( m_hSem) == 0);

#ifndef FLM_RING_ZERO_NLM
	if( m_pIOBuffer)
	{
		f_assert( !m_pIOBuffer->isPending());
	}
#endif

	f_assert( m_completionRc != NE_FLM_IO_PENDING);
	completionRc = m_completionRc;	
#endif

#if defined( FLM_UNIX) || defined( FLM_RING_ZERO_NLM)
	notifyComplete( m_completionRc, m_uiBytesDone);
#endif

	return( completionRc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI F_FileAsyncClient::notifyComplete(
	RCODE				completionRc,
	FLMUINT			uiBytesDone)
{
	f_assert( !m_pIOBuffer || m_pIOBuffer->isPending());
#if !defined( FLM_UNIX) && !defined( FLM_RING_ZERO_NLM)
	f_assert( m_completionRc == NE_FLM_IO_PENDING);
#endif

	AddRef();

	m_completionRc = completionRc;
	m_uiBytesDone = uiBytesDone;

	m_uiEndTime = FLM_GET_TIMER();

	m_completionRc = completionRc;
	m_uiBytesDone = uiBytesDone;
	
	if( m_pFileHdl)
	{
		f_assert( m_pFileHdl->m_numAsyncPending);
		f_atomicDec( &m_pFileHdl->m_numAsyncPending);
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}
	
	if( m_pIOBuffer)
	{
		IF_IOBuffer * pIOBuffer = m_pIOBuffer;

		m_pIOBuffer = NULL;
		pIOBuffer->notifyComplete( m_completionRc);
		pIOBuffer->Release();
		pIOBuffer = NULL;
	}

#if !defined( FLM_UNIX) && !defined( FLM_RING_ZERO_NLM)
	f_semSignal( m_hSem);
	f_assert( f_semGetSignalCount( m_hSem) == 1);
#endif

	f_assert( !m_pIOBuffer || !m_pIOBuffer->isPending());
	Release();
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FTKAPI F_FileAsyncClient::getElapsedTime( void)
{
	return( FLM_TIMER_UNITS_TO_MILLI( 
		FLM_ELAPSED_TIME( m_uiEndTime, m_uiStartTime)));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileAsyncClient::getCompletionCode( void)
{
	return( m_completionRc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::allocFileAsyncClient(
	F_FileAsyncClient **		ppAsyncClient)
{
	RCODE							rc = NE_FLM_OK;
	F_FileAsyncClient *		pAsyncClient = NULL;
	FLMBOOL						bMutexLocked = FALSE;
	
	f_mutexLock( m_hAsyncListMutex);
	bMutexLocked = TRUE;
	
	if( m_pFirstAvailAsync)
	{
		pAsyncClient = m_pFirstAvailAsync;
		m_pFirstAvailAsync = m_pFirstAvailAsync->m_pNext;
		pAsyncClient->m_pNext = NULL;
		m_uiAvailAsyncCount--;
	}
	else
	{
		f_mutexUnlock( m_hAsyncListMutex);
		bMutexLocked = FALSE;
	
		if( (pAsyncClient = f_new F_FileAsyncClient) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	
	pAsyncClient->m_pFileHdl = this;
	pAsyncClient->m_pFileHdl->AddRef();

	*ppAsyncClient = pAsyncClient;
	pAsyncClient = NULL;

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hAsyncListMutex);
	}

	if( pAsyncClient)
	{
		pAsyncClient->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Read from a file
****************************************************************************/
RCODE F_FileHdl::directRead(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,
   void *			pvBuffer,
	IF_IOBuffer *	pIOBuffer,
   FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesRead = 0;
	FLMUINT			uiCurrentBytesRead;
	FLMBYTE *		pucReadBuffer;
	FLMBYTE *		pucDestBuffer;
	FLMUINT			uiMaxBytesToRead;
	FLMBOOL			bHitEOF;
	
	if( !m_bFileOpened || !m_bDoDirectIO || !uiBytesToRead)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}
	else
	{
		m_ui64CurrentPos = ui64ReadOffset;
	}

	// This loop does multiple reads (if necessary) to get all of the
	// data.  It uses aligned buffers and reads at sector offsets.

	pucDestBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{

		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if ((ui64ReadOffset & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)((FLMUINT)pucDestBuffer)) & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)uiBytesToRead & m_ui64NotOnSectorBoundMask)))
		{
			if( m_bRequireAlignedIO)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_MISALIGNED_IO);
				goto Exit;
			}
			
			pucReadBuffer = m_pucAlignedBuff;

			// Must read enough bytes to cover all of the sectors that
			// contain the data we are trying to read.  The value of
			// (ui64ReadOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round that up to the next sector
			// to get the total number of bytes we are going to read.

			uiMaxBytesToRead = (FLMUINT)roundToNextSector( uiBytesToRead +
										(ui64ReadOffset & m_ui64NotOnSectorBoundMask));

			// Can't read more than the aligned buffer will hold.

			if (uiMaxBytesToRead > m_uiAlignedBuffSize)
			{
				uiMaxBytesToRead = m_uiAlignedBuffSize;
			}
		}
		else
		{
			uiMaxBytesToRead = (FLMUINT)roundToNextSector( uiBytesToRead);
			f_assert( uiMaxBytesToRead >= uiBytesToRead);
			pucReadBuffer = pucDestBuffer;
		}

		bHitEOF = FALSE;
		
		if( RC_BAD( rc = lowLevelRead( truncateToPrevSector( ui64ReadOffset),
			uiMaxBytesToRead, pucReadBuffer, pIOBuffer, &uiCurrentBytesRead)))
		{
			if( rc != NE_FLM_IO_END_OF_FILE)
			{
				goto Exit;
			}

			bHitEOF = TRUE;
			rc = NE_FLM_OK;
		}
		
		// If the offset we want to read from is not on a sector
		// boundary, increment the read buffer pointer to the
		// offset where the data we need starts and decrement the
		// bytes read by the difference between the start of the
		// sector and the actual read offset.

		if( uiCurrentBytesRead && (ui64ReadOffset & m_ui64NotOnSectorBoundMask))
		{
			pucReadBuffer += (ui64ReadOffset & m_ui64NotOnSectorBoundMask);
			f_assert( uiCurrentBytesRead >= m_uiBytesPerSector);
			uiCurrentBytesRead -= (FLMUINT)(ui64ReadOffset & m_ui64NotOnSectorBoundMask);
		}

		// If bytes read is more than we actually need, truncate it back
		// so that we only copy what we actually need.

		if( uiCurrentBytesRead > uiBytesToRead)
		{
			uiCurrentBytesRead = uiBytesToRead;
		}
		
		uiBytesToRead -= uiCurrentBytesRead;
		uiBytesRead += uiCurrentBytesRead;
		
		// If using a different buffer for reading, copy the
		// data read into the destination buffer.

		if( pucDestBuffer != pucReadBuffer)
		{
			f_memcpy( pucDestBuffer, pucReadBuffer, uiCurrentBytesRead);
		}
		
		if( !uiBytesToRead)
		{
			break;
		}

		// Still more to read - did we hit EOF above?

		if( bHitEOF)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			break;
		}
		
		pucDestBuffer += uiCurrentBytesRead;
		ui64ReadOffset += uiCurrentBytesRead;
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:		Write to a file using direct I/O
****************************************************************************/
RCODE F_FileHdl::directWrite(
	FLMUINT64				ui64WriteOffset,
	FLMUINT					uiBytesToWrite,
   const void *			pvBuffer,
	IF_IOBuffer *			pIOBuffer,
   FLMUINT *				puiBytesWritten)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucWriteBuffer;
	FLMBYTE *				pucSrcBuffer;
	FLMUINT					uiBytesWritten = 0;
	FLMUINT					uiMaxBytesToWrite;
	FLMUINT					uiBytesBeingOutput;
	FLMBOOL					bWaitForWrite = (pIOBuffer == NULL)
										? TRUE
										: FALSE;
	FLMUINT64				ui64LastWriteOffset;
	FLMUINT					uiLastWriteSize;
	FLMBOOL					bOffsetOnSectorBound;
	FLMBOOL					bBufferOnSectorBound;
	FLMBOOL					bSizeIsSectorMultiple;

	if( !m_bFileOpened || !m_bDoDirectIO || !uiBytesToWrite)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( pIOBuffer && pvBuffer && pvBuffer != pIOBuffer->getBufferPtr())
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}
	else
	{
		m_ui64CurrentPos = ui64WriteOffset;
	}

	if( pIOBuffer)
	{
		pucSrcBuffer = pIOBuffer->getBufferPtr();
	}
	else
	{
		pucSrcBuffer = (FLMBYTE *)pvBuffer;
	}
	
	for (;;)
	{
		bOffsetOnSectorBound = (ui64WriteOffset & m_ui64NotOnSectorBoundMask) 
												? FALSE 
												: TRUE;
		bBufferOnSectorBound = (((FLMUINT)pucSrcBuffer) & m_ui64NotOnSectorBoundMask)
												? FALSE
												: TRUE;
		bSizeIsSectorMultiple = (uiBytesToWrite & m_ui64NotOnSectorBoundMask)
												? FALSE
												: TRUE;

		if( !bOffsetOnSectorBound ||
			 !bBufferOnSectorBound ||
			 !bSizeIsSectorMultiple)
		{
			if( m_bRequireAlignedIO)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_MISALIGNED_IO);
				goto Exit;
			}

			bWaitForWrite = TRUE;
			pucWriteBuffer = m_pucAlignedBuff;

			// Must write enough bytes to cover all of the sectors that
			// contain the data we are trying to write out.  The value of
			// (ui64WriteOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round to the next sector to get the
			// total number of bytes we are going to write.

			uiMaxBytesToWrite = (FLMUINT)roundToNextSector( uiBytesToWrite +
									  (ui64WriteOffset & m_ui64NotOnSectorBoundMask));

			// Can't write more than the aligned buffer will hold.

			if( uiMaxBytesToWrite > m_uiAlignedBuffSize)
			{
				uiMaxBytesToWrite = m_uiAlignedBuffSize;
				uiBytesBeingOutput = (FLMUINT)(uiMaxBytesToWrite -
										(ui64WriteOffset & m_ui64NotOnSectorBoundMask));
			}
			else
			{
				uiBytesBeingOutput = uiBytesToWrite;
			}

			// If the write offset is not on a sector boundary, we must
			// read at least the first sector into the buffer.

			if( ui64WriteOffset & m_ui64NotOnSectorBoundMask)
			{

				// Read the first sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if( RC_BAD( rc = lowLevelRead(
					truncateToPrevSector( ui64WriteOffset),
					m_uiBytesPerSector, pucWriteBuffer, NULL, NULL)))
				{
					goto Exit;
				}
			}

			// If we are writing more than one sector, and the last sector's
			// worth of data we are writing out is only a partial sector,
			// we must read in this sector as well.

			if( (uiMaxBytesToWrite > m_uiBytesPerSector) &&
				 (uiMaxBytesToWrite > uiBytesToWrite))
			{

				// Read the last sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if( RC_BAD( rc = lowLevelRead(
					(truncateToPrevSector( ui64WriteOffset)) +
						(uiMaxBytesToWrite - m_uiBytesPerSector),
					m_uiBytesPerSector,
					(&pucWriteBuffer[ uiMaxBytesToWrite - m_uiBytesPerSector]),
					NULL, NULL)))
				{
					if (rc == NE_FLM_IO_END_OF_FILE)
					{
						rc = NE_FLM_OK;
						f_memset( &pucWriteBuffer [uiMaxBytesToWrite - m_uiBytesPerSector],
										0, m_uiBytesPerSector);
					}
					else
					{
						goto Exit;
					}
				}
			}

			// Finally, copy the data from the source buffer into the
			// write buffer.

			f_memcpy( &pucWriteBuffer[ ui64WriteOffset & m_ui64NotOnSectorBoundMask],
								pucSrcBuffer, uiBytesBeingOutput);
		}
		else
		{
			pucWriteBuffer = pucSrcBuffer;
			uiMaxBytesToWrite = uiBytesToWrite;
			uiBytesBeingOutput = uiBytesToWrite;
		}

		// Position the file to the nearest sector below the write offset.

		ui64LastWriteOffset = truncateToPrevSector( ui64WriteOffset);
		uiLastWriteSize = uiMaxBytesToWrite;
		
		if( !bWaitForWrite)
		{
			f_assert( pucWriteBuffer == pIOBuffer->getBufferPtr());
			
			if( RC_BAD( rc = lowLevelWrite( ui64LastWriteOffset, 
				uiMaxBytesToWrite, NULL, pIOBuffer, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			if( pIOBuffer)
			{
				pIOBuffer->setPending();
			}
			
			rc = lowLevelWrite( ui64LastWriteOffset, uiMaxBytesToWrite, 
				pucWriteBuffer, NULL, NULL);
				
			if( pIOBuffer)
			{
				pIOBuffer->notifyComplete( rc);
				pIOBuffer = NULL;
			}
				
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}

		uiBytesToWrite -= uiBytesBeingOutput;
		uiBytesWritten += uiBytesBeingOutput;
		
		if( !uiBytesToWrite)
		{
			break;
		}
		
		pucSrcBuffer += uiBytesBeingOutput;
		ui64WriteOffset += uiBytesBeingOutput;
	}

Exit:

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}

	return( rc);
}

/****************************************************************************
Desc:	Sets current position of file.
****************************************************************************/
RCODE FTKAPI F_FileHdl::seek(
	FLMUINT64			ui64Offset,
	FLMINT				iWhence,
	FLMUINT64 *			pui64NewOffset)
{
	RCODE	rc = NE_FLM_OK;

	switch (iWhence)
	{
		case FLM_IO_SEEK_CUR:
		{
			m_ui64CurrentPos += ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_SET:
		{
			m_ui64CurrentPos = ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_END:
		{
			if( RC_BAD( rc = size( &m_ui64CurrentPos )))
			{
				goto Exit;
			}
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if( pui64NewOffset)
	{
		*pui64NewOffset = m_ui64CurrentPos;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::tell(
	FLMUINT64 *		pui64Offset)
{
	*pui64Offset = m_ui64CurrentPos;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::read(
	FLMUINT64			ui64Offset,
	FLMUINT				uiLength,
	void *				pvBuffer,
	FLMUINT *			puiBytesRead)
{
	if( m_bDoDirectIO)
	{
		return( directRead( ui64Offset, uiLength, pvBuffer,
			NULL, puiBytesRead));
	}
	else
	{
		return( lowLevelRead( ui64Offset, uiLength, pvBuffer, 
			NULL, puiBytesRead));
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::read(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,
	IF_IOBuffer *	pIOBuffer)
{
	return( directRead( ui64ReadOffset, uiBytesToRead,
		NULL, pIOBuffer, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::write(
	FLMUINT64		ui64WriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWritten)
{
	if( m_bDoDirectIO)
	{
		return( directWrite( ui64WriteOffset, uiBytesToWrite, pvBuffer,
								NULL, puiBytesWritten));
	}
	else
	{
		return( lowLevelWrite( ui64WriteOffset, uiBytesToWrite, 
			pvBuffer, NULL, puiBytesWritten));
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::write(
	FLMUINT64			ui64WriteOffset,
	FLMUINT				uiBytesToWrite,
	IF_IOBuffer *		pIOBuffer)
{
	RCODE		rc = NE_FLM_OK;

	if( m_bDoDirectIO)
	{
		if( RC_BAD( rc = directWrite( ui64WriteOffset, uiBytesToWrite,
				NULL, pIOBuffer, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		pIOBuffer->setPending();
		
		rc = lowLevelWrite( ui64WriteOffset, uiBytesToWrite, 
			pIOBuffer->getBufferPtr(), NULL, NULL);

		pIOBuffer->notifyComplete( rc);

		if( RC_BAD( rc))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}
