//-------------------------------------------------------------------------
// Desc:	Database locking and unlocking.
// Tabs:	3
//
// Copyright (c) 1991, 1994-2007 Novell, Inc. All Rights Reserved.
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
Desc:	Obtains a a lock on the database.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbLock(
	HFDB				hDb,
	eLockType		lockType,
	FLMINT			iPriority,
	FLMUINT			uiTimeout)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bIgnore;
	FDB *		pDb = (FDB *)hDb;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_LOCK)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1, 
			(FLMUINT)lockType)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_SIGNED_NUMBER, 
			0, iPriority)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiTimeout)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// lockType better be exclusive or shared

	if ((lockType != FLM_LOCK_EXCLUSIVE) && (lockType != FLM_LOCK_SHARED))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Nesting of locks is not allowed - this test also keeps this call from
	// being executed inside an update transaction that implicitly acquired
	// the lock.

	if (pDb->uiFlags &
			(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED | FDB_FILE_LOCK_IMPLICIT))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Attempt to acquire the lock.

	if (RC_BAD( rc = pDb->pFile->pFileLockObj->lock( pDb->hWaitSem,
			(FLMBOOL)((lockType == FLM_LOCK_EXCLUSIVE)
					  ? (FLMBOOL)TRUE
					  : (FLMBOOL)FALSE),
			uiTimeout, iPriority, 
			pDb->pDbStats ? &pDb->pDbStats->LockStats : NULL)))
	{
		goto Exit;
	}
	
	pDb->uiFlags |= FDB_HAS_FILE_LOCK;
	
	if (lockType == FLM_LOCK_SHARED)
	{
		pDb->uiFlags |= FDB_FILE_LOCK_SHARED;
	}

Exit:

	flmExit( FLM_DB_LOCK, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	Releases a lock on the database
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbUnlock(
	HFDB		hDb)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = (FDB *)hDb;
	FLMBOOL	bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_UNLOCK)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:

		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If we don't have an explicit lock, can't do the unlock.  It is
	// also illegal to do the unlock during an update transaction.

	if (!(pDb->uiFlags & FDB_HAS_FILE_LOCK) ||
		 (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT) ||
		 (pDb->uiTransType == FLM_UPDATE_TRANS))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Unlock the file.

	if (RC_BAD( rc = pDb->pFile->pFileLockObj->unlock()))
	{
		goto Exit;
	}

	// Unset the flags that indicated the file was explicitly locked.

	pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED));

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_UNLOCK, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc : Returns information about the lock held by the specified database
		 handle.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetLockType(
	HFDB				hDb,
	eLockType *		pLockType,
	FLMBOOL *		pbImplicit)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bIgnore;

	if( pLockType)
	{
		*pLockType = FLM_LOCK_NONE;
	}

	if( pbImplicit)
	{
		*pbImplicit = FALSE;
	}

	if (IsInCSMode( hDb))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = (FDB *)hDb;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	if( pDb->uiFlags & FDB_HAS_FILE_LOCK)
	{
		if( pLockType)
		{
			if( pDb->uiFlags & FDB_FILE_LOCK_SHARED)
			{
				*pLockType = FLM_LOCK_SHARED;
			}
			else
			{
				*pLockType = FLM_LOCK_EXCLUSIVE;
			}
		}
		
		if( pbImplicit)
		{
			*pbImplicit = (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT) 
								? TRUE 
								: FALSE;
		}
	}

Exit:

	flmExit( FLM_DB_GET_LOCK_TYPE, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:	This routine locks a database for exclusive access.
****************************************************************************/
RCODE dbLock(
	FDB *		pDb,
	FLMUINT	uiMaxLockWait)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bGotFileLock = FALSE;
	FFILE *		pFile = pDb->pFile;

	// There must NOT be a shared lock on the file.

	if (pDb->uiFlags & FDB_FILE_LOCK_SHARED)
	{
		rc = RC_SET( FERR_PERMISSION);
		goto Exit;
	}

	// Must acquire an exclusive file lock first, if it hasn't been
	// acquired.

	if (!(pDb->uiFlags & FDB_HAS_FILE_LOCK))
	{
		if (RC_BAD( rc = pFile->pFileLockObj->lock( pDb->hWaitSem, TRUE,
			uiMaxLockWait, 0, pDb->pDbStats ? &pDb->pDbStats->LockStats : NULL)))
		{
			goto Exit;
		}
		bGotFileLock = TRUE;
		pDb->uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
	}

	if (RC_BAD( rc = pFile->pWriteLockObj->lock( pDb->hWaitSem, TRUE,
		uiMaxLockWait, 0, pDb->pDbStats ? &pDb->pDbStats->LockStats : NULL)))
	{
		goto Exit;
	}
		
	pDb->uiFlags |= FDB_HAS_WRITE_LOCK;

Exit:

	if (rc == FERR_IO_FILE_LOCK_ERR)
	{
		if (bGotFileLock)
		{
			(void)pFile->pFileLockObj->unlock();
			pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | 
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}

		if (pDb->uiTransType != FLM_NO_TRANS)
		{

			// Unlink the DB from the transaction.

			(void)flmUnlinkDbFromTrans( pDb, FALSE);
		}
	}
	else if (RC_BAD( rc))
	{
		if (bGotFileLock)
		{
			(void)pFile->pFileLockObj->unlock();
			pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | 
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine unlocks a database that was previously locked
		using the dbLock routine.
****************************************************************************/
RCODE dbUnlock(
	FDB *			pDb)
{
	RCODE	rc = FERR_OK;

	// If we have the write lock, unlock it first.

	flmAssert( pDb->uiFlags & FDB_HAS_WRITE_LOCK);

	pDb->pFile->pWriteLockObj->unlock();
	pDb->uiFlags &= ~FDB_HAS_WRITE_LOCK;

	// Give up the file lock, if it was acquired implicitly.
	
	if (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT)
	{
		if (RC_OK( rc = pDb->pFile->pFileLockObj->unlock()))
		{
			pDb->uiFlags &=
				(~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT));
		}
	}

	return( rc);
}
