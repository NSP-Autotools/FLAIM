//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileHdl class for UNIX.
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

#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

#ifdef FLM_AIX
	#ifndef _LARGE_FILES
		#define _LARGE_FILES
	#endif
	#include <stdio.h>
#endif

#include <sys/types.h>
#if !defined( FLM_OSX) && !defined( FLM_LIBC_NLM)
	#include <aio.h>
#endif

#include <fcntl.h>

#if defined( FLM_SOLARIS)
	#include <sys/statvfs.h>
#elif defined( FLM_LINUX)
	#include <sys/vfs.h>
#elif defined( FLM_OSF)

	// Tru64 4.0 does not have this declaration. Tru64 5.0 renames statfs
	// in vague ways, so we put these declarations before including
	// <sys/stat.h>

	// DSS NOTE: statfs declaration below conflicts with one found in
	// sys/mount.h header file, so I commented it out.  This was when I
	// compiled using the GNU compiler.

	struct statfs;
	#include <sys/mount.h>
#elif defined( FLM_LIBC_NLM)
	#define pread 			pread64
	#define pwrite 		pwrite64
	#define ftruncate		ftruncate64
#endif

#ifdef FLM_LINUX
	static FLMUINT					gv_uiLinuxMajorVer = 0;
	static FLMUINT					gv_uiLinuxMinorVer = 0;
	static FLMUINT					gv_uiLinuxRevision = 0;
#endif

#ifdef FLM_SOLARIS
	static lwp_mutex_t			gv_atomicMutex = DEFAULTMUTEX;
#else
	static pthread_mutex_t		gv_atomicMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

extern FLMATOMIC					gv_openFiles;

/******************************************************************************
Desc:
*******************************************************************************/
F_FileHdl::F_FileHdl()
{
	initCommonData();
	m_fd = -1;
	m_bFlushRequired = FALSE;
}

/******************************************************************************
Desc:
******************************************************************************/
F_FileHdl::~F_FileHdl()
{
	if( m_bFileOpened)
	{
		(void)closeFile();
	}
	
	freeCommonData();
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
	FLMBOOL				bDoDirectIO = FALSE;
	FLMBOOL				bUsingAsync = FALSE;
	int         		openFlags = O_RDONLY;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	f_assert( m_fd == -1);
	f_assert( !m_bFileOpened);
	f_assert( !m_pszFileName);

#if defined( FLM_HAS_DIRECT_IO)
	bDoDirectIO = (uiIoFlags & FLM_IO_DIRECT) ? TRUE : FALSE;
#endif

	// The Linux man pages *say* O_LARGEFILE is needed, although as of 
	// SUSE 9.1 it actually isn't.  Including this flag on Linux just it case...
	
#if defined( FLM_LINUX)
	openFlags |= O_LARGEFILE;
#endif

	// Save the file path

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
	{
		goto Exit;
	}

	f_strcpy( m_pszFileName, pszFileName);
	
	// Determine the i/o flags

   if( bCreateFlag)
   {
		openFlags |= O_CREAT;
		if( uiIoFlags & FLM_IO_EXCL)
		{
	  		openFlags |= O_EXCL;
		}
		else
		{
			openFlags |= O_TRUNC;
		}
	}

   if( !(uiIoFlags & FLM_IO_RDONLY))
	{
		openFlags |= O_RDWR;
	}
	
   if( !(uiIoFlags & FLM_IO_RDONLY))
	{
      openFlags |= O_RDWR;
	}
	
	// Determine if direct and async i/o are supported

	if( bDoDirectIO)
	{
	#if defined( FLM_LINUX)
		{
			FLMUINT		uiMajor;
			FLMUINT		uiMinor;
			FLMUINT		uiRevision;

			f_getLinuxKernelVersion( &uiMajor, &uiMinor, &uiRevision);

			if( uiMajor > 2 || (uiMajor == 2 && uiMinor > 6) ||
				(uiMajor == 2 && uiMinor == 6 && uiRevision >= 5))
			{
				openFlags |= O_DIRECT;
				bUsingAsync = TRUE;
			}
			else
			{
				bDoDirectIO = FALSE;
			}
			#ifdef O_NOATIME
			openFlags |= O_NOATIME;
			#endif
		}
	#elif defined( FLM_AIX)
		openFlags |= O_DIRECT;
		bUsingAsync = TRUE;
	#elif defined( FLM_HPUX)
		bUsingAsync = TRUE;
	#elif defined( FLM_SOLARIS) || defined( FLM_OSX)
		bUsingAsync = TRUE;
	#endif
	}
	
Retry_Create:

	// Try to create or open the file

	if ((m_fd = ::open( pszFileName, openFlags, 0600)) == -1)
	{
		if ((errno == ENOENT) && (uiIoFlags & FLM_IO_CREATE_DIR))
		{
			char	szTemp[ F_PATH_MAX_SIZE];
			char	szIoDirPath[ F_PATH_MAX_SIZE];

			uiIoFlags &= ~FLM_IO_CREATE_DIR;

			// Remove the file name for which we are creating the directory

			if( RC_OK( pFileSystem->pathReduce( 
				m_pszFileName, szIoDirPath, szTemp)))
			{
				if( RC_OK( rc = pFileSystem->createDir( szIoDirPath)))
				{
					goto Retry_Create;
				}
				else
				{
					goto Exit;
				}
			}
		}
#if defined( FLM_LINUX) || defined( FLM_AIX)
		else if( errno == EINVAL && bDoDirectIO)
		{
			openFlags &= ~O_DIRECT;
			bDoDirectIO = FALSE;
			bUsingAsync = FALSE;
			goto Retry_Create;
		}
#endif
		
		rc = f_mapPlatformError( errno, NE_FLM_OPENING_FILE);
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
	
#if defined( FLM_SOLARIS)
	if( bDoDirectIO)
	{
		directio( m_fd, DIRECTIO_ON);
	}
#endif

#if defined( FLM_HPUX)
	if( bDoDirectIO)
	{
		if( ioctl( m_fd, VX_SETCACHE, VX_DIRECT) == -1)
		{
			bDoDirectIO = FALSE;
			bUsingAsync = FALSE;
		}
	}
#endif

#if defined( FLM_OSX)
	if( bDoDirectIO)
	{
		if( fcntl( m_fd, F_NOCACHE, 1) == -1)
		{
			f_assert( 0);
			bDoDirectIO = FALSE;
			bUsingAsync = FALSE;
		}
	}
#endif

	// Get the sector size

#if defined( DEV_BSIZE)
	m_uiBytesPerSector = DEV_BSIZE;
#else
	{
		struct stat		filestats;
	
		if( fstat( m_fd, &filestats) != 0)
		{
			rc = f_mapPlatformError( errno, NE_FLM_OPENING_FILE);
			goto Exit;
		}
	
		m_uiBytesPerSector = (FLMUINT)filestats.st_blksize;
	}
#endif

	m_ui64NotOnSectorBoundMask = m_uiBytesPerSector - 1;
	m_ui64GetSectorBoundMask = ~m_ui64NotOnSectorBoundMask;

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = 64 * 1024;
	if( bDoDirectIO)
	{
		m_uiAlignedBuffSize = (FLMUINT)roundToNextSector( m_uiAlignedBuffSize);
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
	m_bDoDirectIO = bDoDirectIO;
	m_bOpenedInAsyncMode = bUsingAsync;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & FLM_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;
	f_atomicInc( &gv_openFiles);

Exit:

	if( RC_BAD( rc))
	{
		closeFile();
	}
	
   return( rc);
}

/******************************************************************************
Desc:	Close a file
******************************************************************************/
RCODE FTKAPI F_FileHdl::closeFile( void)
{
	if( m_fd != -1)
	{
		::close( m_fd);
		m_fd = -1;
	}
	
	if( m_bDeleteOnRelease)
	{
		f_getFileSysPtr()->deleteFile( m_pszFileName);
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

	return( NE_FLM_OK);
}

/******************************************************************************
Desc:	Make sure all file data is safely on disk
******************************************************************************/
RCODE FTKAPI F_FileHdl::flush( void)
{
	f_assert( m_bFileOpened);
	
#ifdef FLM_SOLARIS

	// Direct I/O on Solaris is ADVISORY, meaning that the
	// operating system may or may not actually honor the
	// option for some or all operations on a given file.
	// Thus, the only way to guarantee that writes are on disk
	// is to call fdatasync.
	//
	// If a process is killed (with SIGKILL or SIGTERM), the
	// dirty cache buffers associated with open files will be discarded unless
	// the process intercepts the signal and properly closes the files.
	//
	// NOTES FROM THE UNIX MAN PAGES ON SIGNALS
	//
	// When killing a process or series of processes, it is common sense
	// to start trying with the least dangerous signal, SIGTERM. That way,
	// programs that care about an orderly shutdown get the chance to follow
	// the procedures that they have been designed to execute when getting
	// the SIGTERM signal, such as cleaning up and closing open files.  If you
	// send a SIGKILL to a process, you remove any chance for the process
	// to do a tidy cleanup and shutdown, which might have unfortunate
	// consequences.

	if( fdatasync( m_fd) != 0)
	{
		 return( f_mapPlatformError( errno, NE_FLM_FLUSHING_FILE));
	}
	
#elif defined( FLM_OSX)

	// OS X doesn't support true direct I/O.  To force data all the way to the
	// disk platters, a call to fcntl with the F_FULLFSYNC flag is required.
	// However, fsync is MUCH faster, but only ensures that the data is delivered
	// to the drive.  If the drive's write-back cache is enabled (very common
	// with ATA drives), the data may not be written to the disk platters
	// until the drive determines an optimal time to do the write.

#ifdef FLM_OSX_FULL_FLUSH
	if( fcntl( m_fd, F_FULLFSYNC, 0) == -1)
#else
	if( fsync( m_fd) != 0)
#endif
	{
		 return( f_mapPlatformError( errno, NE_FLM_FLUSHING_FILE));
	}
	
#else

	if( !m_bDoDirectIO || m_bFlushRequired)
	{
		if( fdatasync( m_fd) != 0)
		{
			 return( f_mapPlatformError( errno, NE_FLM_FLUSHING_FILE));
		}
	}

#endif

	m_bFlushRequired = FALSE;
	return( NE_FLM_OK);
}

/******************************************************************************
Desc:	Return the size of the file
******************************************************************************/
RCODE FTKAPI F_FileHdl::size(
	FLMUINT64 *		pui64Size)
{
	RCODE				rc = NE_FLM_OK;
   struct stat 	statBuf;

	f_assert( m_bFileOpened);
	
   if( fstat( m_fd, &statBuf) == -1)
   {
      rc = f_mapPlatformError( errno, NE_FLM_GETTING_FILE_SIZE);
		goto Exit;
   }
	
	*pui64Size = statBuf.st_size;
	
Exit:

	return( rc);
}

/******************************************************************************
Desc:	Truncate the file to the indicated size
******************************************************************************/
RCODE FTKAPI F_FileHdl::truncateFile(
	FLMUINT64		ui64NewSize)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT64		ui64CurrentSize;

	f_assert( m_bFileOpened);

	if( RC_BAD( rc = size( &ui64CurrentSize)))
	{
		goto Exit;
	}
	
	if( ui64NewSize >= ui64CurrentSize)
	{
		goto Exit;
	}

	if( ftruncate( m_fd, ui64NewSize) == -1)
	{
		rc = f_mapPlatformError( errno, NE_FLM_TRUNCATING_FILE);
		goto Exit;
	}
	
	m_bFlushRequired = TRUE;

Exit:

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
	FLMUINT					uiBytesRead = 0;
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

#if defined( FLM_UNIX) && defined( FLM_HAS_ASYNC_IO)
	if( m_bOpenedInAsyncMode && pIOBuffer)
	{
		struct aiocb *		pAIO;
		
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
		pAIO = &pAsyncClient->m_aio;
		
	#ifdef FLM_AIX
		pAIO->aio_whence = SEEK_SET;
		pAIO->aio_fp = m_fd;
		pAIO->aio_flag = 0;
		pAIO->aio_handle = pAIO;
		pAIO->aio_offset = ui64ReadOffset;
		pAIO->aio_nbytes = uiBytesToRead;
		pAIO->aio_buf = (char *)pvBuffer;
	#else
		pAIO->aio_lio_opcode = LIO_READ;
		pAIO->aio_sigevent.sigev_notify = SIGEV_NONE;
		pAIO->aio_fildes = m_fd;
		pAIO->aio_offset = ui64ReadOffset;
		pAIO->aio_nbytes = uiBytesToRead;
		pAIO->aio_buf = pvBuffer;
	#endif

		pIOBuffer = NULL;
		
	#ifdef FLM_AIX
		if( aio_read( m_fd, pAIO) != 0)
	#else
		if( aio_read( pAIO) != 0)
	#endif
		{
			if( errno == EAGAIN || errno == ENOSYS || errno == EINVAL)
			{
				FLMINT		iBytesRead;
				
				if( (iBytesRead = pread( m_fd, pvBuffer, 
					uiBytesToRead, ui64ReadOffset)) == -1)
				{
					rc = f_mapPlatformError( errno, NE_FLM_READING_FILE);
				}
				else
				{
					uiBytesRead = (FLMUINT)iBytesRead;
					m_ui64CurrentPos += uiBytesRead;
				
					if( uiBytesRead < uiBytesToRead)
					{
						rc = RC_SET( NE_FLM_IO_END_OF_FILE);
					}
				}
			}
			else
			{
				f_assert( 0);
				rc = f_mapPlatformError( errno, NE_FLM_READING_FILE);
			}
			
			pAsyncClient->notifyComplete( rc, uiBytesRead);
			goto Exit;
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
#endif
	{
		FLMINT		iBytesRead;
		
		if( pIOBuffer)
		{
			pIOBuffer->setPending();
		}
	
		if( (iBytesRead = pread( m_fd, pvBuffer, 
			uiBytesToRead, ui64ReadOffset)) == -1)
		{
			rc = f_mapPlatformError( errno, NE_FLM_READING_FILE);
		}
		else
		{
			uiBytesRead = (FLMUINT)iBytesRead;
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
	FLMUINT					uiBytesWritten = 0;
	F_FileAsyncClient *	pAsyncClient = NULL;
	FLMBOOL					bWaitForWrite = FALSE;
	FLMBYTE *				pucExtendBuffer = NULL;
	
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
		FLMUINT64		ui64CurrFileSize;
		FLMUINT			uiTotalBytesToExtend;
		
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
	
	if( !pvBuffer)
	{
		pvBuffer = pIOBuffer->getBufferPtr();
	}

#if defined( FLM_UNIX) && defined( FLM_HAS_ASYNC_IO)
	if( m_bOpenedInAsyncMode)
	{
		struct aiocb *		pAIO;

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
		pAIO = &pAsyncClient->m_aio;
	#ifdef FLM_AIX
		pAIO->aio_whence = SEEK_SET;
		pAIO->aio_fp = m_fd;
		pAIO->aio_flag = 0;
		pAIO->aio_handle = pAIO;
		pAIO->aio_offset = ui64WriteOffset;
		pAIO->aio_nbytes = uiBytesToWrite;
		pAIO->aio_buf = (char *)pvBuffer;
	#else
		pAIO->aio_lio_opcode = LIO_WRITE;
		pAIO->aio_sigevent.sigev_notify = SIGEV_NONE;
		pAIO->aio_fildes = m_fd;
		pAIO->aio_offset = ui64WriteOffset;
		pAIO->aio_nbytes = uiBytesToWrite;
		pAIO->aio_buf = (void *)pvBuffer;
	#endif
		
		pIOBuffer = NULL;
		
	#ifdef FLM_AIX
		if( aio_write( m_fd, pAIO) != 0)
	#else
		if( aio_write( pAIO) != 0)
	#endif
		{
			if( errno == EAGAIN || errno == ENOSYS || errno == EINVAL)
			{
				FLMINT		iBytesWritten;

				for( ;;)
				{
					if( (iBytesWritten = pwrite( m_fd, 
						pvBuffer, uiBytesToWrite, ui64WriteOffset)) == -1)
					{
						if( errno == EINTR)
						{
							continue;
						}
						
						rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
					}
					else
					{
						uiBytesWritten = (FLMUINT)iBytesWritten;
						m_ui64CurrentPos += uiBytesWritten;
					
						if( uiBytesWritten < uiBytesToWrite)
						{
							rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
						}
					}
					
					break;
				}
			}
			else
			{
				f_assert( 0);
				rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
			}
			
			pAsyncClient->notifyComplete( rc, uiBytesWritten);
			goto Exit;
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
#endif
	{
		FLMINT		iBytesWritten;
		
		if( pIOBuffer)
		{
			pIOBuffer->setPending();
		}
		
		for( ;;)
		{
			if( (iBytesWritten = pwrite( m_fd, 
				pvBuffer, uiBytesToWrite, ui64WriteOffset)) == -1)
			{
				if( errno == EINTR)
				{
					continue;
				}
				
				rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
			}
			else
			{
				uiBytesWritten = (FLMUINT)iBytesWritten;
			}
			
			break;
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
	
	if( pucExtendBuffer)
	{
		f_freeAlignedBuffer( &pucExtendBuffer);
	}

	if( puiBytesWritten)
	{
		*puiBytesWritten = uiBytesWritten;
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdl::extendFile(
	FLMUINT64				ui64NewFileSize)
{
	RCODE						rc = NE_FLM_OK;
	FLMBYTE *				pucExtendBuffer = NULL;
	FLMUINT					uiExtendBufferSize;
	FLMUINT					uiCurrBytesToExtend;
	FLMINT					iBytesWritten;
	FLMUINT64				ui64FileSize;
	FLMUINT64				ui64TotalBytesToExtend = 0;
	
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
	
#if defined( FLM_HPUX)
	{
		struct vx_ext		extendInfo;
		struct stat			filestats;
	
		if( fstat( m_fd, &filestats) != 0)
		{
			rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
			goto Exit;
		}

		// If this is a JFS volume, we may be able to use an ioctl command
		// to extend the file.  If this doesn't work, we'll revert to
		// the slow way of extending the file.
		
		f_memset( &extendInfo, 0, sizeof( extendInfo));
		extendInfo.ext_size = 0;
		extendInfo.reserve = ui64NewFileSize / filestats.st_blksize;
		extendInfo.a_flags = VX_CHGSIZE;
		
		if( ioctl( m_fd, VX_SETEXT, &extendInfo) == 0)
		{
			// The call succeeded.  Our work is done.
			
			goto Exit;
		}
	}
#endif
	
	ui64TotalBytesToExtend = ui64NewFileSize - ui64FileSize;
	uiExtendBufferSize = (FLMUINT)f_min( ui64TotalBytesToExtend, 1024 * 1024);

	for( ;;)
	{
		if( RC_OK( rc = f_allocAlignedBuffer( 
			uiExtendBufferSize, &pucExtendBuffer)))
		{
			break;
		}

		if( uiExtendBufferSize <= (32 * 1024))
		{
			goto Exit;
		}

		uiExtendBufferSize >>= 1; 
	}
	
	if( ftruncate( m_fd, ui64NewFileSize) == -1)
	{
		rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
		goto Exit;
	}
	
	while( ui64TotalBytesToExtend)
	{
		uiCurrBytesToExtend = (FLMUINT)f_min( 
						ui64TotalBytesToExtend, uiExtendBufferSize);
		
		if( (iBytesWritten = pwrite( m_fd, pucExtendBuffer, 
			uiCurrBytesToExtend, ui64FileSize)) == -1)
		{
			if( errno == EINTR)
			{
				continue;
			}
			
			rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
			goto Exit;
		}
		
		ui64TotalBytesToExtend -= uiCurrBytesToExtend;
		ui64FileSize += uiCurrBytesToExtend;
	
		if( (FLMUINT)iBytesWritten < uiCurrBytesToExtend)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_IO_DISK_FULL);
			goto Exit;
		}
	}
	
	m_bFlushRequired = TRUE;
	
Exit:

	if( pucExtendBuffer)
	{
		f_freeAlignedBuffer( &pucExtendBuffer);
	}
	
	return( rc);
}
	
/******************************************************************************
Desc:	Attempts to lock byte 0 of the file.  This method is used to
		lock byte 0 of the .lck file to ensure that only one process
		has access to a database.
******************************************************************************/
RCODE FTKAPI F_FileHdl::lock( void)
{
	RCODE				rc = NE_FLM_OK;
	struct flock   LockStruct;

	f_assert( m_bFileOpened);
	
	// Lock the first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_WRLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Attempts to unlock byte 0 of the file.
******************************************************************************/
RCODE FTKAPI F_FileHdl::unlock( void)
{
	RCODE				rc = NE_FLM_OK;
	struct flock   LockStruct;

	f_assert( m_bFileOpened);
	
	// Unlock the first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_UNLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( NE_FLM_IO_FILE_UNLOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determines the kernel version of the linux system we are running on
***************************************************************************/
#ifdef FLM_LINUX
void f_getLinuxKernelVersion(
	FLMUINT *		puiMajor,
	FLMUINT *		puiMinor,
	FLMUINT *		puiRevision)
{
	int			fd = -1;
	int			iBytesRead;
	char			szBuffer [80];
	char *		pszVer;
	FLMUINT		uiMajorVer = 0;
	FLMUINT		uiMinorVer = 0;
	FLMUINT		uiRevision = 0;
	
	if( gv_uiLinuxMajorVer)
	{
		uiMajorVer = gv_uiLinuxMajorVer;
		uiMinorVer = gv_uiLinuxMinorVer;
		uiRevision = gv_uiLinuxRevision;
		goto Exit;
	}
	
	if( (fd = open( "/proc/version", O_RDONLY, 0600)) == -1)
	{
		goto Exit;
	}

	if( (iBytesRead = read( fd, szBuffer, sizeof( szBuffer))) == -1)
	{
		goto Exit;
	}
	if( (pszVer = f_strstr( szBuffer, "version ")) == NULL)
	{
		goto Exit;
	}
	pszVer += 8;

	while( *pszVer >= '0' && *pszVer <= '9')
	{
		uiMajorVer *= 10;
		uiMajorVer += (FLMUINT)(*pszVer - '0');
		pszVer++;
	}
	
	if( *pszVer == '.')
	{
		pszVer++;
		while (*pszVer >= '0' && *pszVer <= '9')
		{
			uiMinorVer *= 10;
			uiMinorVer += (FLMUINT)(*pszVer - '0');
			pszVer++;
		}
	}
	
	if( *pszVer == '.')
	{
		pszVer++;
		while (*pszVer >= '0' && *pszVer <= '9')
		{
			uiRevision *= 10;
			uiRevision += (FLMUINT)(*pszVer - '0');
			pszVer++;
		}
	}
	
Exit:

	if( fd != -1)
	{
		close( fd);
	}
	
	if( puiMajor)
	{
		*puiMajor = uiMajorVer;
	}
	
	if( puiMinor)
	{
		*puiMinor = uiMinorVer;
	}
	
	if( puiRevision)
	{
		*puiRevision = uiRevision;
	}
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_LINUX
void f_setupLinuxKernelVersion( void)
{
	f_getLinuxKernelVersion( &gv_uiLinuxMajorVer, 
		&gv_uiLinuxMinorVer, &gv_uiLinuxRevision);  
}
#endif

/***************************************************************************
Desc:	Determines if the linux system we are running on is 2.4 or greater.
***************************************************************************/
#ifdef FLM_LINUX
FLMUINT f_getLinuxMaxFileSize( void)
{
#ifdef FLM_32BIT
	return( FLM_MAXIMUM_FILE_SIZE);
#else
	FLMUINT	uiMaxFileSize = 0x7FF00000;
	
	f_assert( gv_uiLinuxMajorVer);
	
	// Version 2.4 or greater?

	if( gv_uiLinuxMajorVer > 2 || 
		(gv_uiLinuxMajorVer == 2 && gv_uiLinuxMinorVer >= 4))
	{
		uiMaxFileSize = FLM_MAXIMUM_FILE_SIZE;
	}
	
	return( uiMaxFileSize);
#endif
}
#endif

/****************************************************************************
Desc: This routine gets the block size for the file system a file belongs to.
****************************************************************************/
FLMUINT f_getFSBlockSize(
	FLMBYTE *	pszFileName)
{
	FLMUINT		uiFSBlkSize = 4096;
	FLMBYTE *	pszTmp = pszFileName + f_strlen( (const char *)pszFileName) - 1;
	FLMBYTE *	pszDir;
	FLMBYTE		ucRestoreByte = 0;

	while( pszTmp != pszFileName && *pszTmp != '/')
	{
		pszTmp--;
	}
	
	if( *pszTmp == '/')
	{
		if (pszTmp == pszFileName)
		{
			pszTmp++;
		}
		ucRestoreByte = *pszTmp;
		*pszTmp = 0;
		pszDir = pszFileName;
	}
	else
	{
		pszDir = (FLMBYTE *)".";
	}

#if defined( FLM_SOLARIS)
	struct statvfs statfsbuf;
	
	if (statvfs( (char *)pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
#elif defined( FLM_LINUX) || defined( FLM_OSX) || defined( FLM_AIX)
	struct statfs statfsbuf;
	
	if (statfs( (char *)pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
#endif

	if( ucRestoreByte)
	{
		*pszTmp = ucRestoreByte;
	}
	
	return( uiFSBlkSize);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SPARC) && !defined( FLM_SPARC_GENERIC) && !defined( FLM_SPARC_PLUS)
	#error This build will use mutex-based atomics.
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SPARC_PLUS)
extern "C" void sparc_atomic_code( void)
{
	asm( ".align 8");
	asm( ".global sparc_atomic_add_32");
	asm( ".type sparc_atomic_add_32, #function");
	asm( "sparc_atomic_add_32:");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "    ld [%o0], %o2");
	asm( "    add %o2, %o1, %o3");
	asm( "    cas [%o0], %o2, %o3");
	asm( "    cmp %o2, %o3");
	asm( "    bne sparc_atomic_add_32");
	asm( "    nop");
	asm( "    add %o3, %o1, %o0");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "retl");
	asm( "nop");
	
	asm( ".align 8");
	asm( ".global sparc_atomic_xchg_32");
	asm( ".type sparc_atomic_xchg_32, #function");
	asm( "sparc_atomic_xchg_32:");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "    ld [%o0], %o2");
	asm( "    mov %o1, %o3");
	asm( "    cas [%o0], %o2, %o3");
	asm( "    cmp %o2, %o3");
	asm( "    bne sparc_atomic_xchg_32");
	asm( "    nop");
	asm( "    mov %o2, %o0");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "retl");
	asm( "nop");
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_PPC) && defined( FLM_GNUC)
FLMATOMIC ppc_atomic_add(
	FLMATOMIC *		piTarget,
	FLMINT32			iDelta)
{
	long	result;

	__asm__ __volatile__(
		"1:\n"
		"lwarx		%0, 0, %2\n"
		"addc			%0, %0, %3\n"
		"stwcx.		%0, 0, %2\n"
		"bne-			1b\n"
		"isync"
			: "=&b" (result), "=m" (*piTarget)
			: "b" (piTarget), "r" (iDelta)
			: "cr0", "memory");

	return( result);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_PPC) && defined( FLM_GNUC)
FLMATOMIC ppc_atomic_xchg(
	FLMATOMIC *		piTarget,
	FLMATOMIC		iNewValue)
{
	long	iOldVal;

	__asm__ __volatile__(
		"1:\n"
		"lwarx		%0, 0, %2\n"
		"stwcx.		%3, 0, %2\n"
		"bne-			1b\n"
		"isync"
			: "=&b" (iOldVal), "=m" (*piTarget)
			: "b" (piTarget), "r" (iNewValue)
			: "cr0", "memory");

	return( iOldVal);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_yieldCPU( void)
{
#ifndef FLM_LIBC_NLM
	sched_yield();
#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE void posix_atomic_lock( void)
{
#if defined( FLM_SOLARIS)
	for( ;;)
	{
		if( _lwp_mutex_lock( &gv_atomicMutex) == 0)
		{
			break;
		}
	}
#else
	pthread_mutex_lock( &gv_atomicMutex);
#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE void posix_atomic_unlock( void)
{
#if defined( FLM_SOLARIS)
	_lwp_mutex_unlock( &gv_atomicMutex);
#else
	pthread_mutex_unlock( &gv_atomicMutex);
#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 posix_atomic_add_32(
	volatile FLMINT32 *		piTarget,
	FLMINT32						iDelta)
{
	FLMINT32		i32RetVal;

	posix_atomic_lock();
	(*piTarget) += iDelta;
	i32RetVal = *piTarget;
	posix_atomic_unlock();
	
	return( i32RetVal);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 posix_atomic_xchg_32(
	volatile FLMINT32 *		piTarget,
	FLMINT32						iNewValue)
{
	FLMINT32		i32RetVal;
	
	posix_atomic_lock();
	i32RetVal = *piTarget;
	*piTarget = iNewValue;
	posix_atomic_unlock();
	
	return( i32RetVal);
}

/**********************************************************************
Desc:
**********************************************************************/
RCODE FTKAPI f_chdir(
	const char *		pszDir)
{
	RCODE		rc = NE_FLM_OK;
	
	if( chdir( pszDir) != 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_IO_PATH_NOT_FOUND);
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
	
	if( getcwd( pszDir, F_PATH_MAX_SIZE) == NULL)
	{
		*pszDir = 0;
		rc = f_mapPlatformError( errno, NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	return( rc);
}

#else // FLM_UNIX

/**********************************************************************
Desc:
**********************************************************************/
int ftkunixDummy(void)
{
	return( 0);
}

#endif
