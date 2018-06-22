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

#include "flaimsys.h"

/****************************************************************************
Desc:	Returns an IF_LogMessageClient object if logging is enabled for the
		specified message type
****************************************************************************/
IF_LogMessageClient * flmBeginLogMessage(
	eLogMessageType	eMsgType)
{
	IF_LogMessageClient *		pNewMsg = NULL;

	f_mutexLock( gv_XFlmSysData.hLoggerMutex);
	
	if( !gv_XFlmSysData.pLogger)
	{
		goto Exit;
	}
		
	if( (pNewMsg = gv_XFlmSysData.pLogger->beginMessage( eMsgType)) != NULL)
	{
		gv_XFlmSysData.uiPendingLogMessages++;
	}
	
Exit:

	f_mutexUnlock( gv_XFlmSysData.hLoggerMutex);
	return( pNewMsg);
}

/****************************************************************************
Desc:		Logs information about an error
****************************************************************************/
void flmLogError(
	RCODE				rc,
	const char *	pszDoing,
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	IF_LogMessageClient *	pLogMsg = NULL;

	if( (pLogMsg = flmBeginLogMessage( XFLM_GENERAL_MESSAGE)) != NULL)
	{
		pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
		if( pszFileName)
		{
			f_logPrintf( pLogMsg,
				"Error %s: %e, File=%s, Line=%d.",
				pszDoing, rc, pszFileName, (int)iLineNumber);
		}
		else
		{
			f_logPrintf( pLogMsg, "Error %s: %e.", pszDoing, rc);
		}
		flmEndLogMessage( &pLogMsg);
	}
}

/****************************************************************************
Desc:	Ends a logging message
****************************************************************************/
void flmEndLogMessage(
	IF_LogMessageClient **		ppLogMessage)
{
	if( *ppLogMessage)
	{
		f_mutexLock( gv_XFlmSysData.hLoggerMutex);
		flmAssert( gv_XFlmSysData.uiPendingLogMessages);
		
		(*ppLogMessage)->endMessage();
		(*ppLogMessage)->Release();
		*ppLogMessage = NULL;
		
		gv_XFlmSysData.uiPendingLogMessages--;
		f_mutexUnlock( gv_XFlmSysData.hLoggerMutex);
	}
}

