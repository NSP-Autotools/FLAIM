//------------------------------------------------------------------------------
// Desc:	Contains routines for logging messages from within FLAIM.
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

#include "ftksys.h"

static F_MUTEX					gv_hLoggerMutex = F_MUTEX_NULL;
static FLMUINT					gv_uiPendingLogMessages = 0;
static IF_LoggerClient *	gv_pLogger = NULL;

/***************************************************************************
Desc:
***************************************************************************/
class FTKEXP F_LogPrintfClient : public IF_PrintfClient
{
public:

#define MAX_LOG_BUF_CHARS	255

	F_LogPrintfClient( IF_LogMessageClient * pLogMsg)		
	{
		m_pLogMsg = pLogMsg;
		m_pLogMsg->AddRef();
		m_uiCharOffset = 0;
		m_eCurrentForeColor = FLM_BLACK;
		m_eCurrentBackColor = FLM_WHITE;
	}
	
	virtual ~F_LogPrintfClient()
	{
		if( m_pLogMsg)
		{
			if( m_uiCharOffset)
			{
				flushLogBuffer();
			}
			
			m_pLogMsg->Release();
			m_pLogMsg = NULL;
		}
	}

	FINLINE FLMINT FTKAPI outputChar(
		char				cChar,
		FLMUINT			uiCount)
	{
		FLMUINT			uiTmpCount;
		FLMINT			iBytesOutput = (FLMINT)uiCount;
		
		while( uiCount)
		{
			uiTmpCount = uiCount;
			
			if( m_uiCharOffset + uiTmpCount > MAX_LOG_BUF_CHARS)
			{
				uiTmpCount = MAX_LOG_BUF_CHARS - m_uiCharOffset;
			}
			
			f_memset( &m_szLogBuf [m_uiCharOffset], cChar, uiTmpCount);
			
			m_uiCharOffset += uiTmpCount;
			uiCount -= uiTmpCount;
			
			if (m_uiCharOffset == MAX_LOG_BUF_CHARS)
			{
				flushLogBuffer();
			}
		}
		
		return( iBytesOutput);
	}

	FINLINE FLMINT FTKAPI outputChar(
		char				cChar)
	{
		m_szLogBuf[ m_uiCharOffset++] = cChar;
		
		if( m_uiCharOffset == MAX_LOG_BUF_CHARS)
		{
			flushLogBuffer();
		}
		
		return( 1);
	}
		
	FINLINE FLMINT FTKAPI outputStr(
		const char *	pszStr,
		FLMUINT			uiLen)
	{
		FLMUINT			uiTmpLen;
		FLMINT			iBytesOutput = (FLMINT)uiLen;
		
		while( uiLen)
		{
			uiTmpLen = uiLen;
			
			if( m_uiCharOffset + uiTmpLen > MAX_LOG_BUF_CHARS)
			{
				uiTmpLen = MAX_LOG_BUF_CHARS - m_uiCharOffset;
			}
			
			f_memcpy( &m_szLogBuf [m_uiCharOffset], pszStr, uiTmpLen);
			
			m_uiCharOffset += uiTmpLen;
			uiLen -= uiTmpLen;
			pszStr += uiTmpLen;
			
			if (m_uiCharOffset == MAX_LOG_BUF_CHARS)
			{
				flushLogBuffer();
			}
		}
		
		return( iBytesOutput);
	}
		
	FLMINT FTKAPI colorFormatter(
		char				cFormatChar,
		eColorType		eColor,
		FLMUINT			uiFlags);

private:

	void flushLogBuffer( void);
	
	char							m_szLogBuf[ MAX_LOG_BUF_CHARS + 1];
	FLMUINT						m_uiCharOffset;
	IF_LogMessageClient *	m_pLogMsg;
	eColorType					m_eCurrentForeColor;
	eColorType					m_eCurrentBackColor;
};

/****************************************************************************
Desc:	Main entry point for printf functionality.
****************************************************************************/
void FTKAPI f_logPrintf(
	IF_LogMessageClient *	pLogMessage,
	const char *				pszFormatStr, ...)
{
	f_va_list					args;
	F_LogPrintfClient			printfClient( pLogMessage);

	f_va_start( args, pszFormatStr);
	f_vprintf( &printfClient, pszFormatStr, &args);
	f_va_end( args);
}

/****************************************************************************
Desc:
****************************************************************************/
void FTKAPI f_logPrintf(
	eLogMessageSeverity		msgSeverity,
	const char *				pszFormatStr, ...)
{
	f_va_list					args;
	IF_LogMessageClient *	pLogMsg = NULL;

	if( (pLogMsg = f_beginLogMessage( 0, msgSeverity)) != NULL)
	{
		F_LogPrintfClient			printfClient( pLogMsg);
		
		f_va_start( args, pszFormatStr);
		f_vprintf( &printfClient, pszFormatStr, &args);
		f_va_end( args);
		
		f_endLogMessage( &pLogMsg);
	}
}

/****************************************************************************
Desc:	Printf routine that accepts a va_list argument
****************************************************************************/
void FTKAPI f_logVPrintf(
	IF_LogMessageClient *	pLogMessage,
	const char *				pszFormatStr,
	f_va_list *					args)
{
	F_LogPrintfClient			printfClient( pLogMessage);

	f_vprintf( &printfClient, pszFormatStr, args);
}

/****************************************************************************
Desc:	Returns an IF_LogMessageClient object if logging is enabled for the
		specified message type
****************************************************************************/
IF_LogMessageClient * FTKAPI f_beginLogMessage(
	FLMUINT						uiMsgType,
	eLogMessageSeverity		eMsgSeverity)
{
	IF_LogMessageClient *	pNewMsg = NULL;

	f_mutexLock( gv_hLoggerMutex);
	
	if( !gv_pLogger)
	{
		goto Exit;
	}
		
	if( (pNewMsg = gv_pLogger->beginMessage( uiMsgType, eMsgSeverity)) != NULL)
	{
		gv_uiPendingLogMessages++;
	}
	
Exit:

	f_mutexUnlock( gv_hLoggerMutex);
	return( pNewMsg);
}

/****************************************************************************
Desc:		Logs information about an error
****************************************************************************/
void FTKAPI f_logError(
	RCODE				rc,
	const char *	pszDoing,
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	IF_LogMessageClient *	pLogMsg = NULL;

	if( (pLogMsg = f_beginLogMessage( 0, F_ERR_MESSAGE)) != NULL)
	{
		pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
		if( pszFileName)
		{
			f_logPrintf( pLogMsg, 
				"Error %s: %e, File=%s, Line=%d.\n",
				pszDoing, rc, pszFileName, (int)iLineNumber);
		}
		else
		{
			f_logPrintf( pLogMsg, "Error %s: %e.\n", pszDoing, rc);
		}
		
		f_endLogMessage( &pLogMsg);
	}
}

/****************************************************************************
Desc:	Ends a logging message
****************************************************************************/
void FTKAPI f_endLogMessage(
	IF_LogMessageClient **		ppLogMessage)
{
	if( *ppLogMessage)
	{
		f_mutexLock( gv_hLoggerMutex);
		f_assert( gv_uiPendingLogMessages);
		
		(*ppLogMessage)->endMessage();
		(*ppLogMessage)->Release();
		*ppLogMessage = NULL;
		
		gv_uiPendingLogMessages--;
		f_mutexUnlock( gv_hLoggerMutex);
	}
}

/****************************************************************************
Desc:	Initialize the toolkit logger
****************************************************************************/
RCODE f_loggerInit( void)
{
	RCODE		rc = NE_FLM_OK;

	if( RC_BAD( rc = f_mutexCreate( &gv_hLoggerMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Shutdown the toolkit logger
****************************************************************************/
void f_loggerShutdown( void)
{
	if( gv_pLogger)
	{
		gv_pLogger->Release();
		gv_pLogger = NULL;
	}
	
	if( gv_hLoggerMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hLoggerMutex);
	}
}

/****************************************************************************
Desc:	Set the toolkit logger client
****************************************************************************/
void FTKAPI f_setLoggerClient(
	IF_LoggerClient *	pLogger)
{
	f_mutexLock( gv_hLoggerMutex);
	
	if( gv_pLogger)
	{
		gv_pLogger->Release();
	}
	
	if( (gv_pLogger = pLogger) != NULL)
	{
		gv_pLogger->AddRef();
	}
	
	f_mutexUnlock( gv_hLoggerMutex);
}

/****************************************************************************
Desc:	Output the current log buffer - only called when logging.
****************************************************************************/
void F_LogPrintfClient::flushLogBuffer( void)
{
	if( m_uiCharOffset)
	{
		m_szLogBuf[ m_uiCharOffset] = 0;
		m_pLogMsg->appendString( m_szLogBuf);
		m_uiCharOffset = 0;
	}
}

/****************************************************************************
Desc:		Change colors - may only push or pop a color on to the color stack.
****************************************************************************/
FLMINT FTKAPI F_LogPrintfClient::colorFormatter(
	char			cFormatChar,
	eColorType	eColor,
	FLMUINT		uiFlags)
{
	// Color formatting is ignored if there is not a log message object.
	
	if( m_pLogMsg)
	{
		
		// Before changing colors, output the current log buffer.
		
		flushLogBuffer();
	
		if( cFormatChar == 'F')	// Foreground color
		{
			if( uiFlags & FLM_PRINTF_PLUS_FLAG)
			{
				m_pLogMsg->pushForegroundColor();
			}
			else if( uiFlags & FLM_PRINTF_MINUS_FLAG)
			{
				m_pLogMsg->popForegroundColor();
			}
			else if( m_eCurrentForeColor != eColor)
			{
				m_eCurrentForeColor = eColor;
				m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
			}
		}
		else	// cFormatChar == 'B' - background color
		{
			if( uiFlags & FLM_PRINTF_PLUS_FLAG)
			{
				m_pLogMsg->pushBackgroundColor();
			}
			else if( uiFlags & FLM_PRINTF_MINUS_FLAG)
			{
				m_pLogMsg->popBackgroundColor();
			}
			else if( m_eCurrentBackColor != eColor)
			{
				m_eCurrentBackColor = eColor;
				m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
			}
		}
	}
	
	return( 0);
}

