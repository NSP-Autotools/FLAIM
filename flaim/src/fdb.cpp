//-------------------------------------------------------------------------
// Desc:	Routines for working with and FDB database handle structure.
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

FSTATIC void flmLogMustCloseReason(
	FFILE *			pFile,
	const char *	pszFileName,
	FLMINT			iLineNumber);

/****************************************************************************
Desc:	This function will use the FDB for use by the current thread.
		If another thread already has the FDB used, it will go into the
		debugger.
****************************************************************************/
#if defined( FLM_DEBUG)
void fdbUseCheck(
	FDB *		pDb)
{
	FLMUINT	uiMyThreadId = (FLMUINT)f_threadId();

	f_mutexLock( pDb->hMutex);
	if (!pDb->uiUseCount)
	{
		pDb->uiUseCount++;
		pDb->uiThreadId = uiMyThreadId;
	}
	else if (pDb->uiThreadId != uiMyThreadId)
	{
		flmAssert( 0);
	}
	else
	{
		pDb->uiUseCount++;
	}
	f_mutexUnlock( pDb->hMutex);
}
#endif

/****************************************************************************
Desc:	This function will unuse the FDB for use by the current thread.
****************************************************************************/
#if defined( FLM_DEBUG)
void fdbUnuse(
	FDB *		pDb)
{
	FLMUINT	uiMyThreadId = (FLMUINT)f_threadId();

	f_mutexLock( pDb->hMutex);
	if ((!pDb->uiUseCount) || (uiMyThreadId != pDb->uiThreadId))
	{
		flmAssert( 0);
	}
	else
	{
		pDb->uiUseCount--;
	}
	f_mutexUnlock( pDb->hMutex);
}
#endif

/****************************************************************************
Desc:	This function will init an FDB for a database that is being handled
		via a client/server connection.
****************************************************************************/
void fdbInitCS(
	FDB *		pDb)
{
	if (pDb)
	{
		fdbUseCheck( pDb);
		(void)flmResetDiag( pDb);
	}
}

/****************************************************************************
Desc:	This function will init an FDB for use.  It will also start
		the necessary type of transaction - if any.
****************************************************************************/
RCODE	fdbInit(
	FDB *				pDb,					// Pointer to database.
	FLMUINT			uiTransType,		// Type of transaction to start.
	FLMUINT			uiFlags,				// Flags for function.
	FLMUINT			uiAutoTrans,		// Auto transaction OK?  Used only if
												// uiTransType == FLM_UPDATE_TRANS.  This
												// also has the max lock wait time.
	FLMBOOL *		pbStartedTransRV	// Returns flag indicating whether or not
												// we started a transaction inside this
												// routine.
	)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiTransFlags = FLM_GET_TRANS_FLAGS( uiTransType);

	uiTransType = FLM_GET_TRANS_TYPE( uiTransType);

	if( pbStartedTransRV)
	{
		*pbStartedTransRV = FALSE;
	}

	fdbUseCheck( pDb);
	if (!pDb->uiInitNestLevel)
	{
		if (!(uiFlags & FDB_DONT_RESET_DIAG) && !pDb->uiInitNestLevel)
		{
			(void)flmResetDiag( pDb);
		}

		if (!gv_FlmSysData.Stats.bCollectingStats)
		{
			pDb->pStats = NULL;
			pDb->pDbStats = NULL;
		}
		else
		{
			pDb->pStats = &pDb->Stats;

			/*
			Statistics are being collected for the system.  Therefore,
			if we are not currently collecting statistics in the
			session, start.  If we were collecting statistics, but the
			start time was earlier than the start time in the system
			statistics structure, reset the statistics in the session.
			*/

			if (!pDb->Stats.bCollectingStats)
			{
				flmStatStart( &pDb->Stats);
			}
			else if (pDb->Stats.uiStartTime < gv_FlmSysData.Stats.uiStartTime)
			{
				flmStatReset( &pDb->Stats, FALSE, FALSE);
			}
			(void)flmStatGetDb( &pDb->Stats, pDb->pFile,
							0, &pDb->pDbStats, NULL, NULL);
			pDb->pLFileStats = NULL;
		}
	}

	pDb->uiInitNestLevel++;

	// Now that the nest level has been incremented, test
	// to see if the database is being forced to close.

	if( !(uiFlags & FDB_CLOSING_OK))
	{
		if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
		{
			goto Exit;
		}
	}

	/*
	If uiTransType == FLM_NO_TRANS, lock down the default dictionary
	if the FDB is not currently involved in a transaction.  If the
	FDB is already involved in a transaction, there is no need to
	lock down anything, because the FDICT structure and
	tables will already be locked down.
	*/

	if (uiTransType == FLM_NO_TRANS)
	{
		if (pDb->uiTransType == FLM_NO_TRANS)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			if (pDb->pFile->pDictList && pDb->pDict != pDb->pFile->pDictList)
			{
				flmLinkFdbToDict( pDb, pDb->pFile->pDictList);
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}
		goto Exit;
	}

	/*
	If they are requesting a read transaction, set the FLM_AUTO_TRANS
	bit to TRUE.
	*/

	if (uiTransType == FLM_READ_TRANS)
	{
		uiAutoTrans |= FLM_AUTO_TRANS;
	}
	else
	{
		if (pDb->uiTransType == FLM_UPDATE_TRANS)
		{
			pDb->bHadUpdOper = TRUE;
		}

		/* uiTransType == FLM_UPDATE_TRANS, make sure updates are OK. */

		if (pDb->uiFlags & FDB_FILE_LOCK_SHARED)
		{
			// There is a shared lock on the database.
			rc = RC_SET( FERR_PERMISSION);
			goto Exit;
		}
	}

Test_Trans:

	/* See if we already have a transaction going on the DB. */

	if (pDb->uiTransType != FLM_NO_TRANS)
	{
		/*
		If the transaction is an invisible transaction, we may need to
		abort it, depending on what is being asked for.
		*/

		if (pDb->uiFlags & FDB_INVISIBLE_TRANS)
		{
			/*
			Several conditions will cause us to abort an invisible
			transaction:

			  1. If it is NOT ok for a transaction to already be in progress.
			     That is, the caller does NOT want to piggy-back on an
				  already running transaction.

			  2. If it is a transaction that has been marked as being
			     required to abort because of some error.

			  3. If the transaction needed is an update transaction, but
			     the invisible transaction is a read transaction.

			  4. The flag requesting that we join invisible transactions
			     is not set.

			*/

			if ((!(uiFlags & FDB_TRANS_GOING_OK)) ||
			    (!(uiFlags & FDB_INVISIBLE_TRANS_OK)) ||
				 (flmCheckBadTrans( pDb)) ||
				 ((uiTransType == FLM_UPDATE_TRANS) &&
				  (pDb->uiTransType != FLM_UPDATE_TRANS)))
			{
				if (RC_BAD( rc = flmAbortDbTrans( pDb)))
				{
					goto Exit;
				}
				goto Test_Trans;
			}
		}
		else
		{
			if( !(uiFlags & FDB_TRANS_GOING_OK))
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}

			/* See if the transaction should be aborted. */

			if( flmCheckBadTrans( pDb))
			{
				rc = RC_SET( FERR_ABORT_TRANS);
				goto Exit;
			}

			/*
			If we need an update transaction, make sure that is what we
			currently have going.  Also, make sure we are not in read-only
			mode.
			*/

			if ((uiTransType == FLM_UPDATE_TRANS) &&
			 	 (pDb->uiTransType != FLM_UPDATE_TRANS))
			{
				rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
				goto Exit;
			}
		}
	}
	else if (!(uiAutoTrans & FLM_AUTO_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}
	else
	{

		// If we get to this point, we need to start a transaction on the
		// database.

		if( RC_BAD( rc = flmBeginDbTrans( pDb, uiTransType,
										(FLMUINT)(0x00FF & uiAutoTrans), uiTransFlags)))
		{
			goto Exit;
		}

		if( pbStartedTransRV)
		{
			*pbStartedTransRV = TRUE;
		}

		if (uiTransType == FLM_UPDATE_TRANS)
		{
			pDb->bHadUpdOper = TRUE;
		}
	}

Exit:

	// WARNING: The calling routine must call fdbExit or flmExit to unlock
	// any resources that are locked by fdbInit (even if this fdbInit
	// returns an error).

	return( rc);
}

/****************************************************************************
Desc:	This function will unlock an FDB.
****************************************************************************/
void fdbExit(
	FDB *	pDb)
{
	flmAssert( pDb);

	if( !pDb->pCSContext)
	{
		flmAssert( pDb->uiInitNestLevel);
		pDb->uiInitNestLevel--;
		if (!pDb->uiInitNestLevel)
		{
			if (pDb->pDict && pDb->uiTransType == FLM_NO_TRANS)
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);
				flmUnlinkFdbFromDict( pDb);
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}

			pDb->pStats = NULL;
		}
	}
	fdbUnuse( pDb);
}

/****************************************************************************
Desc: This function is used to determine if a function id corresponds to a
		cursor function.
****************************************************************************/
FINLINE FLMBOOL IsQueryFunc(
	FLMUINT		uiFuncId)
{
	return( (uiFuncId == FLM_CURSOR_CONFIG ||
				uiFuncId == FLM_CURSOR_NEXT ||
				uiFuncId == FLM_CURSOR_NEXT_DRN ||
				uiFuncId == FLM_CURSOR_PREV ||
				uiFuncId == FLM_CURSOR_PREV_DRN ||
				uiFuncId == FLM_CURSOR_FIRST ||
				uiFuncId == FLM_CURSOR_FIRST_DRN ||
				uiFuncId == FLM_CURSOR_LAST ||
				uiFuncId == FLM_CURSOR_LAST_DRN ||
				uiFuncId == FLM_CURSOR_MOVE_RELATIVE ||
				uiFuncId == FLM_CURSOR_REC_COUNT)
					? TRUE
					: FALSE);
}

/****************************************************************************
Desc: This function is used to determine if an error should
		require an update transaction to be aborted.
****************************************************************************/
FINLINE FLMBOOL IsAbortError(
	FLMUINT	uiFuncId,
	RCODE		rc)
{
	return( (rc != FERR_OK &&
				rc != FERR_END &&
				rc != FERR_BOF_HIT &&
				rc != FERR_EOF_HIT &&
				rc != FERR_EXISTS &&
				rc != FERR_NOT_FOUND &&
				rc != FERR_NOT_UNIQUE &&
				rc != FERR_BAD_FIELD_NUM &&
				rc != FERR_ABORT_TRANS &&
				rc != FERR_IO_FILE_LOCK_ERR &&
				rc != FERR_IO_ACCESS_DENIED &&
				rc != FERR_IO_PATH_NOT_FOUND &&
				rc != FERR_IO_INVALID_PATH &&
				rc != FERR_OLD_VIEW &&
				rc != FERR_PERMISSION &&
				rc != FERR_ILLEGAL_OP &&
				rc != FERR_ILLEGAL_TRANS_OP &&
				rc != FERR_DUPLICATE_DICT_REC &&
				rc != FERR_TIMEOUT &&
				rc != FERR_INDEX_OFFLINE &&
				(rc != FERR_BAD_IX ||
				 (!IsQueryFunc( uiFuncId) && uiFuncId != FLM_INDEX_STATUS)) &&
				(rc != FERR_CURSOR_SYNTAX || !IsQueryFunc( uiFuncId)))
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc: This function checks to see if an update transaction should be forced
		to abort.  It also resets the gedcom memory pool.
****************************************************************************/
void flmExit(
	eFlmFuncs	eFlmFuncId,
	FDB *			pDb,
	RCODE			rc)
{

	// There are a few functions that may not have an FDB

	if (pDb)
	{
		// If this is an update transaction, see if it should be aborted.

		if (pDb->uiTransType == FLM_UPDATE_TRANS &&
			 IsAbortError( eFlmFuncId, rc))
		{

			// Set the abort flag

			pDb->eAbortFuncId = eFlmFuncId;
			pDb->AbortRc = rc;
		}

		// Don't reset or free the temporary pool if FLAIM func was called
		// within a user call-back	that called another FLAIM functions.

		if (pDb->uiInFlmFunc == 0)
		{

			// Keep the main pool block around inbetween FLAIM calls.

			pDb->TempPool.poolReset();
		}
		fdbExit( pDb);
	}
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
	flmLogMessage( 
		F_DEBUG_MESSAGE,
		FLM_YELLOW,
		FLM_BLACK,
		pszFileName 
			? "Error %s: 0x%04X (%s), File=%s, Line=%d."
			: "Error %s: 0x%04X (%s).",
		pszDoing, (unsigned)rc, FlmErrorString( rc),
		pszFileName ? pszFileName : "",
		pszFileName ? (int)iLineNumber : 0);
}

/****************************************************************************
Desc:		Logs messages
****************************************************************************/
void flmLogMessage(
	eLogMessageSeverity 		eMsgSeverity,
	eColorType  				foreground,
	eColorType  				background,
	const char *				pszFormat,
	...)
{
	FLMINT						iLen;
	f_va_list					args;
	IF_LogMessageClient *	pLogMsg = NULL;
	char *						pszMsgBuf = NULL;
	
	if( !gv_FlmSysData.pLogger)
	{
		return;
	}
	
	if( (pLogMsg = gv_FlmSysData.pLogger->beginMessage( 
		FLM_GENERAL_MESSAGE, eMsgSeverity)) != NULL)
	{
		if( RC_OK( f_alloc( 1024, &pszMsgBuf)))
		{
			f_va_start( args, pszFormat);
			iLen = f_vsprintf( pszMsgBuf, pszFormat, &args);
			f_va_end( args);
	
			pLogMsg->changeColor( foreground, background);
			pLogMsg->appendString( pszMsgBuf);
		}
		
		pLogMsg->endMessage();
		pLogMsg->Release();
		pLogMsg = NULL;

		if( pszMsgBuf)
		{
			f_free( &pszMsgBuf);
		}
	}
}

/****************************************************************************
Desc:		Logs the reason for the "must close" flag being set
****************************************************************************/
FSTATIC void flmLogMustCloseReason(
	FFILE *			pFile,
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	// Log a message indicating why the "must close" flag was set

	flmLogMessage( 
			F_DEBUG_MESSAGE,
			FLM_YELLOW,
			FLM_BLACK,
			"Database (%s) must be closed because of a 0x%04X error, File=%s, Line=%d.",
				(pFile->pszDbPath
					? pFile->pszDbPath
					: ""),
				(unsigned)pFile->rcMustClose,
				pszFileName, (int)iLineNumber);
}

/****************************************************************************
Desc:		Checks to see if the database should be closed
****************************************************************************/
RCODE flmCheckDatabaseStateImp(
	FDB *				pDb,
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	RCODE			rc = FERR_OK;

	if( pDb && pDb->bMustClose)
	{
		flmLogMustCloseReason( pDb->pFile, pszFileName, iLineNumber);
		rc = RC_SET( FERR_CLOSING_DATABASE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Checks the FFILE state
****************************************************************************/
RCODE flmCheckFFileStateImp(
	FFILE *			pFile,
	const char *	pszFileName,
	FLMINT			iLineNumber)

{
	RCODE		rc = FERR_OK;

	if( pFile && pFile->bMustClose)
	{
		flmLogMustCloseReason( pFile, pszFileName, iLineNumber);
		rc = RC_SET( FERR_CLOSING_DATABASE);
		goto Exit;
	}

Exit:

	return( rc);
}
