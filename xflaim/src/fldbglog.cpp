//------------------------------------------------------------------------------
// Desc:	Contains the functions for debug logging.
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

#include "flaimsys.h"

#ifdef FLM_DBG_LOG

// Local prototypes

FSTATIC void _flmDbgLogFlush( void);

FSTATIC void _flmDbgOutputMsg(
	char *	pszMsg);

// Global data

F_MUTEX				g_hDbgLogMutex = F_MUTEX_NULL;
IF_FileHdl *		g_pLogFile = NULL;
char *				g_pszLogBuf = NULL;
FLMUINT				g_uiLogBufOffset = 0;
FLMUINT				g_uiLogFileOffset = 0;
FLMBOOL				g_bDbgLogEnabled = TRUE;

#define DBG_LOG_BUFFER_SIZE		((FLMUINT)512000)


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogInit( void)
{
	char	szLogPath[ 256];
	RCODE	rc	= NE_XFLM_OK;

	flmAssert( g_hDbgLogMutex == F_MUTEX_NULL);

	// Allocate a buffer for the log

	if (RC_BAD( rc = f_alloc( DBG_LOG_BUFFER_SIZE + 1024, &g_pszLogBuf)))
	{
		goto Exit;
	}

	// Create the mutex

	if (RC_BAD( rc = f_mutexCreate( &g_hDbgLogMutex)))
	{
		goto Exit;
	}

	// Build the file path

#ifdef FLM_NLM
	f_strcpy( szLogPath, "SYS:\\FLMDBG.LOG");
#else
	f_sprintf( szLogPath, "FLMDBG.LOG");
#endif

	// Create the file - truncate if it exists already.

	if( RC_BAD( rc = gv_XFlmSysData.pFileSystem->createFile( szLogPath, 
		gv_XFlmSysData.uiFileCreateFlags, &g_pLogFile)))
	{
		goto Exit;
	}

Exit:

	flmAssert( RC_OK( rc));
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogExit( void)
{
	if( g_bDbgLogEnabled)
	{
		// Output "Log End" message
		f_mutexLock( g_hDbgLogMutex);
		_flmDbgOutputMsg( "--- LOG END ---");
		f_mutexUnlock( g_hDbgLogMutex);
		
		// Flush the log
		flmDbgLogFlush();
	}

	// Free all resources

	if( g_hDbgLogMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &g_hDbgLogMutex);
	}

	if( g_pszLogBuf)
	{
		f_free( &g_pszLogBuf);
	}

	if( g_pLogFile)
	{
		g_pLogFile->Truncate( g_uiLogFileOffset + g_uiLogBufOffset);
		g_pLogFile->Close();
		g_pLogFile->Release();
		g_pLogFile = NULL;
	}

	g_bDbgLogEnabled = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogMsg(
	char *		pszMsg)
{
	if (!g_bDbgLogEnabled)
		return;
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszMsg);
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogFlush( void)
{
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgLogFlush();
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void _flmDbgLogFlush( void)
{
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiBytesWritten;
	char *			pszBufPtr = g_pszLogBuf;
	FLMUINT			uiTotalToWrite = g_uiLogBufOffset;
	RCODE				rc = NE_XFLM_OK;
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

		if( RC_BAD( rc = g_pLogFile->SectorWrite(
			g_uiLogFileOffset, uiBytesToWrite,
			pszBufPtr, uiBufferSize, NULL, &uiBytesWritten, FALSE)))
		{
			goto Exit;
		}

		flmAssert( uiBytesToWrite == uiBytesWritten);
		g_uiLogFileOffset += uiBytesWritten;
		pszBufPtr += uiBytesWritten;
		uiBufferSize -= uiBytesWritten;
		uiTotalToWrite -= uiBytesWritten;
	}

	if (g_uiLogBufOffset & 0x1FF)
	{
		if (g_uiLogBufOffset > 512)
		{
			f_memcpy( g_pszLogBuf,
				&g_pszLogBuf [g_uiLogBufOffset & 0xFFFFFE00],
					512);
			g_uiLogBufOffset &= 0x1FF;
		}
		g_uiLogFileOffset -= g_uiLogBufOffset;
	}
	else
	{
		g_uiLogBufOffset = 0;
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
	char *	pszBufPtr = (char *)(&(g_pszLogBuf[ g_uiLogBufOffset]));

	f_sprintf( (char *)pszBufPtr, "%s\n", pszMsg);
	g_uiLogBufOffset += f_strlen( pszBufPtr);

	if( g_uiLogBufOffset >= DBG_LOG_BUFFER_SIZE)
	{
		_flmDbgLogFlush();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogWrite(
	F_Database *	pDatabase,
	FLMUINT			uiBlkAddress,
	FLMUINT			uiWriteAddress,
	FLMUINT64		ui64TransId,
	char *			pszEvent)
{
	char		pszTmpBuf[ 256];
	
	if( !g_bDbgLogEnabled)
		return;

	if( !uiWriteAddress)
	{
		f_sprintf( (char *)pszTmpBuf, "d%X b=%X t%I64u %s",
			(unsigned)((FLMUINT)pDatabase),
			(unsigned)uiBlkAddress, ui64TransId, pszEvent);
	}
	else
	{
		f_sprintf( (char *)pszTmpBuf, "d%X b=%X a=%X t%I64u %s",
				(unsigned)((FLMUINT)pDatabase),
    			(unsigned)uiBlkAddress, (unsigned)uiWriteAddress,
				ui64TransId, pszEvent);
	}
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( g_hDbgLogMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogUpdate(
	F_Database *	pDatabase,
	FLMUINT64		ui64TransId,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	RCODE				rc,
	char *			pszEvent)
{
	char		pszTmpBuf[ 256];
	char		szErr [12];
	
	if (!g_bDbgLogEnabled)
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

	if (uiCollection)
	{
		f_sprintf( (char *)pszTmpBuf, "d%X t%I64u c%u n%I64u %s%s",
			(unsigned)((FLMUINT)pDatabase),
			ui64TransId, (unsigned)uiCollection, 
			ui64NodeId, pszEvent, szErr);
	}
	else
	{
		f_sprintf( (char *)pszTmpBuf, "d%X t%I64u %s%s",
			(unsigned)((FLMUINT)pDatabase),
			ui64TransId, pszEvent,
			szErr);
	}

	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( g_hDbgLogMutex);
}

#endif // FLM_DBG_LOG

/****************************************************************************
Desc:
****************************************************************************/
#ifndef FLM_DBG_LOG
void fldbglog_dummy()
{
}
#endif
