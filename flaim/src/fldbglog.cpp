//-------------------------------------------------------------------------
// Desc:	Debug logging routines.
// Tabs:	3
//
// Copyright (c) 1999-2003, 2005-2007 Novell, Inc. All Rights Reserved.
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

#ifdef FLM_DBG_LOG

FSTATIC void _flmDbgLogFlush( void);

FSTATIC void _flmDbgOutputMsg(
	char *			pszMsg);

F_MUTEX				gv_hDbgLogMutex = F_MUTEX_NULL;
IF_FileSystem *	gv_pFileSystem = NULL;
IF_FileHdl *		gv_pLogFile = NULL;
char *				gv_pszLogBuf = NULL;
FLMUINT				gv_uiLogBufOffset = 0;
FLMUINT				gv_uiLogFileOffset = 0;
FLMBOOL				gv_bDbgLogEnabled = TRUE;

#define DBG_LOG_BUFFER_SIZE		((FLMUINT)512000)

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogInit( void)
{
	RCODE				rc	= FERR_OK;
	char				szLogPath[ 256];

	flmAssert( gv_hDbgLogMutex == F_MUTEX_NULL);
	flmAssert( gv_pFileSystem == NULL);

	// Allocate a buffer for the log

	if( RC_BAD( rc = f_alloc( 
		DBG_LOG_BUFFER_SIZE + 1024, &gv_pszLogBuf)))
	{
		goto Exit;
	}

	// Create the mutex

	if( RC_BAD( f_mutexCreate( &gv_hDbgLogMutex)))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Create a new file system object
	
	if( RC_BAD( rc = FlmGetFileSystem( &gv_pFileSystem)))
	{
		goto Exit;
	}

	// Build the file path

#ifdef FLM_NLM
	f_strcpy( szLogPath, "SYS:\\FLMDBG.LOG");
#else
	f_sprintf( szLogPath, "FLMDBG.LOG");
#endif

	// Create the file.

	if( RC_BAD( rc = gv_pFileSystem->createFile( szLogPath, 
		FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
		&gv_pLogFile)))
	{

		// See if we can open the file and then truncate it.

		if( RC_OK( gv_pFileSystem->openFile( szLogPath,
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT, &gv_pLogFile)))
		{
			if( RC_BAD( rc = gv_pLogFile->truncate()))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	flmAssert( RC_OK( rc));
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogExit( void)
{
	if( gv_bDbgLogEnabled)
	{
		// Output "Log End" message
		f_mutexLock( gv_hDbgLogMutex);
		_flmDbgOutputMsg( "--- LOG END ---");
		f_mutexUnlock( gv_hDbgLogMutex);
		
		// Flush the log
		flmDbgLogFlush();
	}

	// Free all resources

	if( gv_hDbgLogMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hDbgLogMutex);
	}

	if( gv_pszLogBuf)
	{
		f_free( &gv_pszLogBuf);
	}

	if( gv_pLogFile)
	{
		gv_pLogFile->truncate( gv_uiLogFileOffset + gv_uiLogBufOffset);
		gv_pLogFile->close();
		gv_pLogFile->Release();
		gv_pLogFile = NULL;
	}

	if( gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}
	
	gv_bDbgLogEnabled = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogWrite(
	FLMUINT		uiFileId,
	FLMUINT		uiBlkAddress,
	FLMUINT		uiWriteAddress,
	FLMUINT		uiTransId,
	char *		pszEvent)
{
	char		pszTmpBuf[ 256];
	
	if( !gv_bDbgLogEnabled)
	{
		return;
	}

	if( !uiWriteAddress)
	{
		f_sprintf( pszTmpBuf, "f%u b=%X t%u %s",
			(unsigned)uiFileId,
			(unsigned)uiBlkAddress, (unsigned)uiTransId, pszEvent);
	}
	else
	{
		f_sprintf( pszTmpBuf, "f%u b=%X a=%X t%u %s",
				(unsigned)uiFileId,
    			(unsigned)uiBlkAddress, (unsigned)uiWriteAddress,
				(unsigned)uiTransId, pszEvent);
	}
	f_mutexLock( gv_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( gv_hDbgLogMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogUpdate(
	FLMUINT		uiFileId,
	FLMUINT		uiTransId,
	FLMUINT		uiContainer,	// Zero if logging transaction begin, commit, abort
	FLMUINT		uiDrn,			// Zero if logging transaction begin, commit, abort
	RCODE			rc,
	char *		pszEvent)
{
	char		pszTmpBuf[ 256];
	char		szErr [12];
	
	if (!gv_bDbgLogEnabled)
	{
		return;
	}
	if (RC_BAD( rc))
	{
		f_sprintf( szErr, " RC=%04X", (unsigned)rc);
	}
	else
	{
		szErr [0] = 0;
	}

	if( uiContainer)
	{
		f_sprintf( pszTmpBuf, "f%u t%u c%u d%u %s%s",
			(unsigned)uiFileId,
			(unsigned)uiTransId, (unsigned)uiContainer, 
			(unsigned)uiDrn, pszEvent, szErr);
	}
	else
	{
		f_sprintf( pszTmpBuf, "f%u t%u %s%s",
			(unsigned)uiFileId,
			(unsigned)uiTransId, pszEvent,
			szErr);
	}

	f_mutexLock( gv_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( gv_hDbgLogMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogMsg(
	char *		pszMsg)
{
	if (!gv_bDbgLogEnabled)
	{
		return;
	}
	
	f_mutexLock( gv_hDbgLogMutex);
	_flmDbgOutputMsg( pszMsg);
	f_mutexUnlock( gv_hDbgLogMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogFlush( void)
{
	f_mutexLock( gv_hDbgLogMutex);
	_flmDbgLogFlush();
	f_mutexUnlock( gv_hDbgLogMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void _flmDbgLogFlush( void)
{
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiBytesWritten;
	char *			pszBufPtr = gv_pszLogBuf;
	FLMUINT			uiTotalToWrite = gv_uiLogBufOffset;
	RCODE				rc = FERR_OK;
	FLMUINT			uiBufferSize = DBG_LOG_BUFFER_SIZE + 1024;

	while( uiTotalToWrite)
	{
		if( uiTotalToWrite > 0xFE00)
		{
			uiBytesToWrite = 0xFE00;
		}
		else
		{
			uiBytesToWrite = uiTotalToWrite;
		}

		if( RC_BAD( rc = gv_pLogFile->write( gv_uiLogFileOffset, uiBytesToWrite,
			pszBufPtr, &uiBytesWritten)))
		{
			goto Exit;
		}

		flmAssert( uiBytesToWrite == uiBytesWritten);
		gv_uiLogFileOffset += uiBytesWritten;
		pszBufPtr += uiBytesWritten;
		uiBufferSize -= uiBytesWritten;
		uiTotalToWrite -= uiBytesWritten;
	}

	if (gv_uiLogBufOffset & 0x1FF)
	{
		if (gv_uiLogBufOffset > 512)
		{
			f_memcpy( gv_pszLogBuf,
				&gv_pszLogBuf [gv_uiLogBufOffset & 0xFFFFFE00], 512);
			gv_uiLogBufOffset &= 0x1FF;
		}
		gv_uiLogFileOffset -= gv_uiLogBufOffset;
	}
	else
	{
		gv_uiLogBufOffset = 0;
	}

Exit:

	flmAssert( RC_OK( rc));
}

/****************************************************************************
Desc:
****************************************************************************/
void _flmDbgOutputMsg(
	char *		pszMsg)
{
	char *	pszBufPtr = &(gv_pszLogBuf[ gv_uiLogBufOffset]);

	f_sprintf( pszBufPtr, "%s\n", pszMsg);
	gv_uiLogBufOffset += f_strlen( pszBufPtr);

	if( gv_uiLogBufOffset >= DBG_LOG_BUFFER_SIZE)
	{
		_flmDbgLogFlush();
	}
}

#else	// #ifdef FLM_DBG_LOG

/****************************************************************************
Desc:
****************************************************************************/
void gv_fldbglog()
{
}

#endif
