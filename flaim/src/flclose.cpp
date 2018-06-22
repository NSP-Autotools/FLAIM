//-------------------------------------------------------------------------
// Desc:	Close a database
// Tabs:	3
//
// Copyright (c) 1990-1992, 1995-2003, 2005-2007 Novell, Inc.
// All Rights Reserved.
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
Desc :	Closes a FLAIM database.
****************************************************************************/
RCODE flmDbClose(
	HFDB *		phDbRV,
	FLMBOOL		bMutexLocked)
{
	FDB *			pDb;

	if ((!phDbRV) ||
		 ((pDb = (FDB *)*phDbRV) == NULL))
	{
		goto Exit;
	}

	if (IsInCSMode( pDb))
	{
		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		if( pCSContext->bConnectionGood)
		{

			// Send the request to close the database.

			if (RC_BAD( Wire.sendOp(
				FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_CLOSE)))
			{
				goto Finish_Close;
			}

			if (RC_BAD( Wire.sendTerminate()))
			{
				goto Transmission_Error;
			}

			/* Read the response. */
	
			if (RC_BAD( Wire.read()))
			{
				goto Transmission_Error;
			}

			Wire.getRCode();
			goto Finish_Close;
Transmission_Error:
			pDb->pCSContext->bConnectionGood = FALSE;
		}

Finish_Close:

		// Reset misc. variables.

		(void)flmCloseCSConnection( &pDb->pCSContext);
		pDb->pCSContext = NULL;
	}

	if (pDb->uiTransType != FLM_NO_TRANS)
	{

		// Force nested transactions to close.

		pDb->uiInFlmFunc++;
		(void)FlmDbTransAbort( (HFDB)pDb);
		pDb->uiInFlmFunc--;
	}

	// Free the super file.

	if( pDb->pSFileHdl)
	{
		// Opened files will be released back to the 
		// file handle manager
		pDb->pSFileHdl->Release();
	}

	// Unlink the FDB from the FFILE and FDICT structures.

	if (!bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}
	flmUnlinkFdbFromDict( pDb);
	flmUnlinkFdbFromFile( pDb);

	if (!bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// Free the temporary pools

	pDb->TempPool.poolFree();
	pDb->tmpKrefPool.poolFree();

	// Free up statistics.

	if (pDb->bStatsInitialized)
	{
		FlmFreeStats( &pDb->Stats);
	}

	// Get rid of mutex

#if defined( FLM_DEBUG)
	if (pDb->hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &pDb->hMutex);
	}
#endif

	// Free the semaphore
	
	if( pDb->hWaitSem != F_SEM_NULL)
	{
		f_semDestroy( &pDb->hWaitSem);
	}

	// Free the read buffer

	if( pDb->pucAlignedReadBuf)
	{
		f_freeAlignedBuffer( &pDb->pucAlignedReadBuf);
	}

	// Free the FDB.

	f_free( phDbRV);

Exit:

	return( FERR_OK);
}

/****************************************************************************
Desc:	Closes a FLAIM database.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbClose(
	HFDB *	phDbRV)
{
	return( flmDbClose( phDbRV, FALSE));
}
