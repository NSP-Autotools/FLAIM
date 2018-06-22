//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileHdl class on Windows platforms.
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

#include "ftksys.h"

#if defined( FLM_WIN)

extern FLMATOMIC						gv_openFiles;
extern SET_FILE_VALID_DATA_FUNC	gv_SetFileValidDataFunc;

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC VOID CALLBACK f_fileIOCompletionRoutine(
	DWORD 					dwErrorCode,
	DWORD 					dwNumberOfBytesTransfered,
	LPOVERLAPPED			lpOverlapped)
{
	F_FileAsyncClient *	pAsyncClient;
	
	pAsyncClient = (F_FileAsyncClient *)(lpOverlapped->hEvent);
	pAsyncClient->notifyComplete(
			f_mapPlatformError( dwErrorCode, NE_FLM_ASYNC_FAILED),
			(FLMUINT)dwNumberOfBytesTransfered);
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdl::F_FileHdl()
{
	initCommonData();
	m_hFile = INVALID_HANDLE_VALUE;
	m_bFlushRequired = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdl::~F_FileHdl()
{
	closeFile();
}

/***************************************************************************
Desc:	Open or create a file.
***************************************************************************/
RCODE F_FileHdl::openOrCreate(
	const char *		pszFileName,
   FLMUINT				uiIoFlags,
	FLMBOOL				bCreateFlag)
{
	RCODE					rc = NE_FLM_OK;
	DWORD					udAccessMode = 0;
	DWORD					udShareMode = 0;
	DWORD					udCreateMode = 0;
	DWORD					udAttrFlags = 0;
	DWORD					udErrCode;
	HANDLE      		hToken;
	TOKEN_PRIVILEGES	tp;
	TOKEN_PRIVILEGES	oldtp;
	DWORD					udTokenPrivSize = sizeof( TOKEN_PRIVILEGES);          
	LUID					luid;
	FLMBOOL				bOpenInAsyncMode = FALSE;
	FLMBOOL				bDoDirectIO;
	FLMBOOL				bRestoreTokenPrivileges = FALSE;
	FLMBOOL				bClosePrivToken = FALSE;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	f_assert( !m_bFileOpened);
	f_assert( !m_pszFileName);

	bDoDirectIO = (uiIoFlags & FLM_IO_DIRECT) ? TRUE : FALSE;

	// Save the file name

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
	{
		goto Exit;
	}

	f_strcpy( m_pszFileName, pszFileName);

	// If doing direct IO, need to get the sector size.

	if( bDoDirectIO)
	{
		if( RC_BAD( rc = pFileSystem->getSectorSize(
			pszFileName, &m_uiBytesPerSector)))
		{
			goto Exit;
		}
		
		m_ui64NotOnSectorBoundMask = m_uiBytesPerSector - 1;
		m_ui64GetSectorBoundMask = ~m_ui64NotOnSectorBoundMask;
	}

	// Only enable asynchronous writes if direct I/O is enabled.

	if( bDoDirectIO && f_getFileSysPtr()->canDoAsync())
	{
		bOpenInAsyncMode = TRUE;
	}

	// Set up the file characteristics requested by caller.

   if( uiIoFlags & FLM_IO_SH_DENYRW)
   {
      udShareMode = 0;
      uiIoFlags &= ~FLM_IO_SH_DENYRW;
   }
   else if( uiIoFlags & FLM_IO_SH_DENYWR)
   {
      udShareMode = FILE_SHARE_READ;
      uiIoFlags &= ~FLM_IO_SH_DENYWR;
   }
	else if (uiIoFlags & FLM_IO_SH_DENYNONE)
   {
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
      uiIoFlags &= ~FLM_IO_SH_DENYNONE;
   }
	else
	{
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
	}

	// Begin setting the CreateFile flags and fields

   udAttrFlags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS;
	
	if( bDoDirectIO)
	{
		// Specifying FILE_FLAG_NO_BUFFERING and FILE_FLAG_WRITE_THROUGH 
		// results in the data being immediately flushed to disk without
		// going through the system cache.  The operating system also 
		// requests a write-through the hard disk cache to persistent 
		// media. However, not all hardware supports this write-through
		// capability.

		udAttrFlags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
	}
	
	if( bOpenInAsyncMode)
	{
		udAttrFlags |= FILE_FLAG_OVERLAPPED;
	}

   if( bCreateFlag)
   {
   	if( uiIoFlags & FLM_IO_EXCL)
		{
	  		udCreateMode = CREATE_NEW;
		}
		else
		{
		   udCreateMode = CREATE_ALWAYS;
		}
   }
	else
   {
		udCreateMode = OPEN_EXISTING;
   }

   udAccessMode = GENERIC_READ | GENERIC_WRITE;

   if( (!bCreateFlag) && (uiIoFlags & FLM_IO_RDONLY))
	{
      udAccessMode = GENERIC_READ;
	}
	
	// Attempt to enable the "manage volume" privilege while the file is being
	// opened/created.  This will allow subsequent file extend operations to
	// be done via calls to SetFileValidData().

	if( gv_SetFileValidDataFunc)
	{
		if( OpenProcessToken( GetCurrentProcess(), 
			TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
		{
			bClosePrivToken = TRUE;
			
			if( LookupPrivilegeValue( NULL, SE_MANAGE_VOLUME_NAME, &luid))
			{
				ZeroMemory ( &tp, sizeof( tp));
				
				tp.PrivilegeCount = 1;
				tp.Privileges[0].Luid = luid;
				tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
				
				if( AdjustTokenPrivileges( hToken, FALSE, &tp, 
					sizeof( TOKEN_PRIVILEGES), &oldtp, &udTokenPrivSize))
				{
					bRestoreTokenPrivileges = TRUE;
				}
			}
		}
	}
			
Retry_Create:

	if( (m_hFile = CreateFile( (LPCTSTR)pszFileName, udAccessMode,
					udShareMode, NULL, udCreateMode,
					udAttrFlags, NULL)) == INVALID_HANDLE_VALUE)
	{
		udErrCode = GetLastError();
		if( (udErrCode == ERROR_PATH_NOT_FOUND) && (uiIoFlags & FLM_IO_CREATE_DIR))
		{
			char		szTemp[ F_PATH_MAX_SIZE];
			char		szDirPath[ F_PATH_MAX_SIZE];

			uiIoFlags &= ~FLM_IO_CREATE_DIR;

			// Remove the file name for which we are creating the directory.

			if( RC_OK( pFileSystem->pathReduce( m_pszFileName, 
				szDirPath, szTemp)))
			{
				if( RC_OK( rc = pFileSystem->createDir( szDirPath)))
				{
					goto Retry_Create;
				}
				else
				{
					goto Exit;
				}
			}
		}
		
		rc = f_mapPlatformError( udErrCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)(bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_CREATING_FILE
												: (RCODE)NE_FLM_CREATING_FILE)
								  : (RCODE)(bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_OPENING_FILE
												: (RCODE)NE_FLM_OPENING_FILE)));
		goto Exit;
	}
	
	if( uiIoFlags & FLM_IO_DELETE_ON_RELEASE)
	{
		m_bDeleteOnRelease = TRUE;
	}
	else
	{
		m_bDeleteOnRelease = FALSE;
	}

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = 64 * 1024;
	if( bDoDirectIO)
	{
		m_uiAlignedBuffSize = roundToNextSector( m_uiAlignedBuffSize);
	}

	if( RC_BAD( rc = f_allocAlignedBuffer( m_uiAlignedBuffSize, 
		&m_pucAlignedBuff)))
	{
		goto Exit;
	}
	
	if( bDoDirectIO)
	{
		if( uiIoFlags & FLM_IO_NO_MISALIGNED)
		{
			m_bRequireAlignedIO = TRUE;
		}
	}
	
	m_bFileOpened = TRUE;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & FLM_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;
	m_bDoDirectIO = bDoDirectIO;
	m_bOpenedInAsyncMode = bOpenInAsyncMode;
	m_bFlushRequired = FALSE;
	f_atomicInc( &gv_openFiles);

Exit:

	if( bRestoreTokenPrivileges)
	{
		AdjustTokenPrivileges( hToken, FALSE, &oldtp, 
			udTokenPrivSize, NULL, NULL);
	}
	
	if( bClosePrivToken)
	{
		CloseHandle( hToken);
	}

	if( RC_BAD( rc))
	{
		closeFile();
	}
	
   return( rc);
}

/****************************************************************************
Desc:	Close a file
****************************************************************************/
RCODE FTKAPI F_FileHdl::closeFile( void)
{
	if( m_bFlushRequired)
	{
		flush();
	}
	
	if( m_hFile != INVALID_HANDLE_VALUE)
	{
		CloseHandle( m_hFile);
		m_hFile = INVALID_HANDLE_VALUE;
	}
	
	if( m_bDeleteOnRelease)
	{
		DeleteFile( (LPTSTR)m_pszFileName);
		m_bDeleteOnRelease = FALSE;
	}
	
	if( m_bFileOpened)
	{
		f_atomicDec( &gv_openFiles);
	}
	
	freeCommonData();
	
	m_bFileOpened = FALSE;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = FALSE;
	m_bOpenedExclusive = FALSE;
	m_bDoDirectIO = FALSE;
	m_bOpenedInAsyncMode = FALSE;
	m_bFlushRequired = FALSE;
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::flush( void)
{
	RCODE		rc = NE_FLM_OK;

	if( !m_bDoDirectIO || m_bFlushRequired)
	{
		if( !FlushFileBuffers( m_hFile))
  		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_FLUSHING_FILE);
		}
		
		m_bFlushRequired = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::lowLevelRead(
	FLMUINT64				ui64ReadOffset,
	FLMUINT					uiBytesToRead,
	void *					pvBuffer,
	IF_IOBuffer *			pIOBuffer,
	FLMUINT *				puiBytesRead)
{
	RCODE						rc = NE_FLM_OK;
	DWORD						uiBytesRead = 0;
	OVERLAPPED *			pOverlapped = NULL;
	LARGE_INTEGER			liTmp;
	F_FileAsyncClient *	pAsyncClient = NULL;
	FLMBOOL					bWaitForRead = FALSE;

	if( pIOBuffer && pvBuffer && pvBuffer != pIOBuffer->getBufferPtr())
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
	
	if( !pvBuffer)
	{
		pvBuffer = pIOBuffer->getBufferPtr();
	}

	if( m_bOpenedInAsyncMode)
	{
		if( RC_BAD( rc = allocFileAsyncClient( &pAsyncClient)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pAsyncClient->prepareForAsync( pIOBuffer)))
		{
			goto Exit;
		}
		
		if( !pIOBuffer)
		{
			f_assert( pvBuffer);
			bWaitForRead = TRUE;
		}
			
		pAsyncClient->m_uiBytesToDo = uiBytesToRead;
		pOverlapped = &pAsyncClient->m_Overlapped;
		pOverlapped->Offset = (DWORD)(ui64ReadOffset & 0xFFFFFFFF);
		pOverlapped->OffsetHigh = (DWORD)(ui64ReadOffset >> 32);
		pOverlapped->hEvent = (HANDLE)pAsyncClient;
		pIOBuffer = NULL;
		
RetryRead:

		if( !ReadFileEx( m_hFile, pvBuffer, uiBytesToRead, 
			pOverlapped, f_fileIOCompletionRoutine))
		{
			DWORD		udErrCode = GetLastError();
			
			if( rc == ERROR_NOT_ENOUGH_MEMORY ||
				 rc == ERROR_INVALID_USER_BUFFER)
			{
				// The ReadFileEx function may fail, returning the messages 
				// ERROR_INVALID_USER_BUFFER or ERROR_NOT_ENOUGH_MEMORY if there 
				// are too many outstanding asynchronous I/O requests

				f_sleep( 10);
				goto RetryRead;
			}
			else if( udErrCode != ERROR_IO_PENDING)
			{
				rc = f_mapPlatformError( udErrCode, NE_FLM_READING_FILE);
				pAsyncClient->notifyComplete( rc, 0);
				goto Exit;
			}
		}
		
		if( bWaitForRead)
		{
			if( RC_BAD( rc = pAsyncClient->waitToComplete()))
			{
				if( rc != NE_FLM_IO_END_OF_FILE)
				{
					goto Exit;
				}
	
				rc = NE_FLM_OK;
			}
	
			uiBytesRead = pAsyncClient->m_uiBytesDone;
		}
		else
		{
			uiBytesRead = uiBytesToRead;
		}
	}
	else
	{
		if( pIOBuffer)
		{
			pIOBuffer->setPending();
		}
		
		liTmp.QuadPart = ui64ReadOffset;
		if( !SetFilePointerEx( m_hFile, liTmp, NULL, FILE_BEGIN))
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
		}
		else
		{
			if( !ReadFile( m_hFile, pvBuffer, uiBytesToRead, &uiBytesRead, NULL))
			{
				rc = f_mapPlatformError( GetLastError(), NE_FLM_READING_FILE);
			}
		}
		
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
	
	m_ui64CurrentPos += uiBytesRead;

	if( uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		goto Exit;
	}
	
Exit:

	f_assert( uiBytesRead || RC_BAD( rc));

	if( pAsyncClient)
	{
		pAsyncClient->Release();
	}

	if( pIOBuffer && !pIOBuffer->isPending())
	{
		f_assert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}
	
	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::lowLevelWrite(
	FLMUINT64				ui64WriteOffset,
	FLMUINT					uiBytesToWrite,
	const void *			pvBuffer,
	IF_IOBuffer *			pIOBuffer,
	FLMUINT *				puiBytesWritten)
{
	RCODE						rc = NE_FLM_OK;
	OVERLAPPED *			pOverlapped = NULL;
	LARGE_INTEGER			liTmp;
	DWORD						uiBytesWritten = 0;
	F_FileAsyncClient *	pAsyncClient = NULL;
	FLMBOOL					bWaitForWrite = FALSE;
	FLMUINT					uiTotalBytesToExtend;
	FLMUINT64				ui64CurrFileSize;
	
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
	
	if( m_bDoDirectIO && !m_numAsyncPending && m_uiExtendSize)
	{
		if( RC_BAD( rc = getPreWriteExtendSize( ui64WriteOffset, uiBytesToWrite,
			&ui64CurrFileSize, &uiTotalBytesToExtend)))
		{
			goto Exit;
		}
		
		if( uiTotalBytesToExtend)
		{
			if( RC_BAD( rc = extendFile( ui64CurrFileSize + uiTotalBytesToExtend)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if( RC_BAD( rc = size( &ui64CurrFileSize)))
		{
			goto Exit;
		}

		if( ui64CurrFileSize <= ui64WriteOffset + uiBytesToWrite)
		{
			// The file is being extended.  We must force a flush
			// to ensure that the directory entry is updated.

			m_bFlushRequired = TRUE;
		}
	}
	
	if( !pvBuffer)
	{
		pvBuffer = pIOBuffer->getBufferPtr();
	}

	if( m_bOpenedInAsyncMode)
	{
		if( RC_BAD( rc = allocFileAsyncClient( &pAsyncClient)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pAsyncClient->prepareForAsync( pIOBuffer)))
		{
			goto Exit;
		}
			
		if( !pIOBuffer)
		{
			bWaitForWrite = TRUE;
		}
			
		pAsyncClient->m_uiBytesToDo = uiBytesToWrite;
		pOverlapped = &pAsyncClient->m_Overlapped;
		pOverlapped->Offset = (DWORD)(ui64WriteOffset & 0xFFFFFFFF);
		pOverlapped->OffsetHigh = (DWORD)(ui64WriteOffset >> 32);
		pOverlapped->hEvent = (HANDLE)pAsyncClient;
		pIOBuffer = NULL;
		
RetryWrite:

		if( !WriteFileEx( m_hFile, pvBuffer,
			uiBytesToWrite, pOverlapped, f_fileIOCompletionRoutine))
		{
			DWORD		udErrCode = GetLastError();

			if( rc == ERROR_NOT_ENOUGH_MEMORY ||
				 rc == ERROR_INVALID_USER_BUFFER)
			{
				// The WriteFileEx function may fail, returning the messages 
				// ERROR_INVALID_USER_BUFFER or ERROR_NOT_ENOUGH_MEMORY if there 
				// are too many outstanding asynchronous I/O requests

				f_sleep( 10);
				goto RetryWrite;
			}
			else if( udErrCode != ERROR_IO_PENDING)
			{
				rc = f_mapPlatformError( udErrCode, NE_FLM_WRITING_FILE);
				pAsyncClient->notifyComplete( rc, 0);
				goto Exit;
			}
		}
		
		if( bWaitForWrite)
		{
			if( RC_BAD( rc = pAsyncClient->waitToComplete()))
			{
				if( rc != NE_FLM_IO_DISK_FULL)
				{
					goto Exit;
				}
	
				rc = NE_FLM_OK;
			}
				
			uiBytesWritten = pAsyncClient->m_uiBytesDone; 
		}
		else
		{
			uiBytesWritten = uiBytesToWrite;
		}
	}
	else
	{
		if( pIOBuffer)
		{
			pIOBuffer->setPending();
		}
		
		liTmp.QuadPart = ui64WriteOffset;
		if( !SetFilePointerEx( m_hFile, liTmp, NULL, FILE_BEGIN))
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
		}
		else
		{
			if( !WriteFile( m_hFile, pvBuffer, uiBytesToWrite, 
				&uiBytesWritten, NULL))
			{
				rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
			}
		}
	
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
	
	m_ui64CurrentPos += uiBytesWritten;

	if( uiBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
		goto Exit;
	}
	
Exit:

	if( pAsyncClient)
	{
		pAsyncClient->Release();
	}
	
	if( pIOBuffer && !pIOBuffer->isPending())
	{
		f_assert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Return the size of the file
****************************************************************************/
RCODE FTKAPI F_FileHdl::size(
	FLMUINT64 *		pui64Size)
{
	RCODE					rc = NE_FLM_OK;
	LARGE_INTEGER		liTmp;
	
	if( !GetFileSizeEx( m_hFile, &liTmp))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_GETTING_FILE_SIZE);
		goto Exit;
	}
	
	*pui64Size = liTmp.QuadPart;

Exit:

	return( rc);
}

/****************************************************************************
Desc: Truncate the file to the indicated size
****************************************************************************/
RCODE FTKAPI F_FileHdl::truncateFile(
	FLMUINT64			ui64NewSize)
{
	RCODE					rc = NE_FLM_OK;
	LARGE_INTEGER		liTmp;
	FLMUINT64			ui64CurrentSize;

	f_assert( m_bFileOpened);
	
	if( RC_BAD( rc = size( &ui64CurrentSize)))
	{
		goto Exit;
	}
	
	if( ui64NewSize >= ui64CurrentSize)
	{
		goto Exit;
	}

	liTmp.QuadPart = ui64NewSize;
	if( !SetFilePointerEx( m_hFile, liTmp, NULL, FILE_BEGIN))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
		goto Exit;
	}
		
	if( !SetEndOfFile( m_hFile))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_TRUNCATING_FILE);
		goto Exit;
	}
	
	m_bFlushRequired = TRUE;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::extendFile(
	FLMUINT64				ui64NewFileSize)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiBytesToWrite;
	DWORD						uiBytesWritten;
	FLMBYTE *				pucBuffer = NULL;
	FLMUINT					uiBufferSize;
	FLMUINT64				ui64FileSize;
	FLMUINT64				ui64TotalBytesToExtend = 0;
	LARGE_INTEGER			liTmp;
	F_FileAsyncClient *	pAsyncClient = NULL;
	
	// Get the current file size
	
	if( RC_BAD( rc = size( &ui64FileSize)))
	{
		goto Exit;
	}
	
	// File is already the requested size
	
	if( ui64FileSize >= ui64NewFileSize)
	{
		goto Exit;
	}
	
	// Try to extend the file using SetFileValidData.  This will allocate blocks
	// without zero-filling them.  This call is very fast, but is only available
	// on WinXP/2003 and newer systems and only if the file was opened while the
	// process had "volume manage" permission (which the open/create code in 
	// this file tries to acquire).  If we aren't able to extend the file this
	// way, the code will fall-back to the slower approach of zero-filling the 
	// extent to cause blocks to be allocated to the file.

	liTmp.QuadPart = ui64NewFileSize;
	if( !SetFilePointerEx( m_hFile, liTmp, NULL, FILE_BEGIN))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
		goto Exit;
	}
	
	if( !SetEndOfFile( m_hFile))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
		goto Exit;
	}
	
	if( gv_SetFileValidDataFunc)
	{
		if( (gv_SetFileValidDataFunc)( m_hFile, ui64NewFileSize))
		{
			goto Exit;
		}
	}

	// Determine the number of bytes to extend
	
	ui64TotalBytesToExtend = ui64NewFileSize - ui64FileSize;
	
	// Allocate a zero-filled buffer for extending

	uiBufferSize = 64 * 1024;
	if( RC_BAD( rc = f_allocAlignedBuffer( uiBufferSize, &pucBuffer)))
	{
		goto Exit;
	}
	
	f_memset( pucBuffer, 0, uiBufferSize);

	// Extend the file until we run out of bytes to write.

	while( ui64TotalBytesToExtend)
	{
		if( (uiBytesToWrite = uiBufferSize) > ui64TotalBytesToExtend)
		{
			uiBytesToWrite = (FLMUINT)ui64TotalBytesToExtend;
		}
		
		if( m_bOpenedInAsyncMode)
		{
			OVERLAPPED *		pOverlapped;
			
			if( RC_BAD( rc = allocFileAsyncClient( &pAsyncClient)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pAsyncClient->prepareForAsync( NULL)))
			{
				goto Exit;
			}
				
			pAsyncClient->m_uiBytesToDo = uiBytesToWrite;
			pOverlapped = &pAsyncClient->m_Overlapped;
			pOverlapped->Offset = (DWORD)(ui64FileSize & 0xFFFFFFFF);
			pOverlapped->OffsetHigh = (DWORD)(ui64FileSize >> 32);
			pOverlapped->hEvent = (HANDLE)pAsyncClient;
			
RetryWrite:

			if( !WriteFileEx( m_hFile, pucBuffer,
				uiBytesToWrite, pOverlapped, f_fileIOCompletionRoutine))
			{
				DWORD		udErrCode = GetLastError();
				
				if( rc == ERROR_NOT_ENOUGH_MEMORY ||
					rc == ERROR_INVALID_USER_BUFFER)
				{
					// The WriteFileEx function may fail, returning the messages 
					// ERROR_INVALID_USER_BUFFER or ERROR_NOT_ENOUGH_MEMORY if there 
					// are too many outstanding asynchronous I/O requests

					f_sleep( 10);
					goto RetryWrite;
				}
				else if( udErrCode != ERROR_IO_PENDING)
				{
					rc = f_mapPlatformError( udErrCode, NE_FLM_WRITING_FILE);
					pAsyncClient->notifyComplete( rc, 0);
					goto Exit;
				}
			}
			
			if( RC_BAD( rc = pAsyncClient->waitToComplete()))
			{
				goto Exit;
			}
					
			uiBytesWritten = (DWORD)pAsyncClient->m_uiBytesDone;
			pAsyncClient->Release();
			pAsyncClient = NULL;
		}
		else
		{
			LONG		lDummy = 0;

			if( SetFilePointer( m_hFile, (LONG)ui64FileSize,
						&lDummy, FILE_BEGIN) == 0xFFFFFFFF)
			{
				rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
				goto Exit;
			}
			
			if( !WriteFile( m_hFile, (LPVOID)pucBuffer,
						(DWORD)uiBytesToWrite, &uiBytesWritten, NULL))
			{
				rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
				goto Exit;
			}
		}

		// No more room on disk

		if( uiBytesWritten < (DWORD)uiBytesToWrite)
		{
			rc = RC_SET( NE_FLM_IO_DISK_FULL);
			goto Exit;
		}
		
		ui64TotalBytesToExtend -= uiBytesToWrite;
		ui64FileSize += uiBytesToWrite;
	}
	
	m_bFlushRequired = TRUE;

Exit:

	if( pucBuffer)
	{
		f_freeAlignedBuffer( &pucBuffer);
	}
	
	if( pAsyncClient)
	{
		pAsyncClient->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::lock( void)
{
	RCODE				rc = NE_FLM_OK;

	if( !LockFile( m_hFile, 0, 0, 1, 1))
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FTKAPI F_FileHdl::unlock( void)
{
	RCODE				rc = NE_FLM_OK;

	if( !UnlockFile( m_hFile, 0, 0, 1, 1))
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_yieldCPU( void)
{
	SleepEx( 0, true);
}

/**********************************************************************
Desc:
**********************************************************************/
RCODE FTKAPI f_chdir(
	const char *		pszDir)
{
	RCODE		rc = NE_FLM_OK;
	
	if( _chdir( pszDir) != 0)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/**********************************************************************
Desc:
**********************************************************************/
RCODE FTKAPI f_getcwd(
	char *			pszDir)
{
	RCODE		rc = NE_FLM_OK;
	
	if( _getcwd( pszDir, F_PATH_MAX_SIZE) == NULL)
	{
		*pszDir = 0;
		rc = f_mapPlatformError( GetLastError(), NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	return( rc);
}

#endif // FLM_WIN

/****************************************************************************
Desc:	Deletes a file
****************************************************************************/
#if defined( FLM_WATCOM_NLM) || defined( FLM_OSX)
int gv_ftkwinDummy(void)
{
	return( 0);
}
#endif
