//-------------------------------------------------------------------------
// Desc:	Retrieve record from database.
// Tabs:	3
//
// Copyright (c) 1990, 1994-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE	flmRecordRetrieveCS(
	FDB *				pDb,
	FLMUINT 			uiContainer,
	FLMUINT			uiDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecord,
	FLMUINT *		puiDrnRV);

/****************************************************************************
Desc:	Retrieves a single record from a container.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmRecordRetrieve(
	HFDB				hDb,
	FLMUINT 			uiContainer,
	FLMUINT			uiDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecord,
	FLMUINT *		puiDrnRV)
{
	FLMUINT			uiKeyRelPos;
	FLMBOOL			bTransStarted;
	FLMUINT			uiFoundDrn;
	LFILE *			pLFile;
	BTSK				stack[ BH_MAX_LEVELS];
	FLMBOOL			bStackInitialized = FALSE;
	BTSK *			pStack;
	FLMBYTE			pSearchBuf[ DRN_KEY_SIZ + 4];
	FLMBYTE			pKeyBuf[ DRN_KEY_SIZ + 4];
	FDB *				pDb = (FDB *)hDb;
	FLMUINT			uiSaveInitNestLevel = 0;
	DB_STATS *		pDbStats;
	RCODE				rc = FERR_OK;

#ifdef FLM_DEBUG
	flmAssert( hDb != NULL);
	flmAssert( uiContainer != 0);
#endif

	bTransStarted = FALSE;
	uiFoundDrn = 0;

	if( !uiDrn && (uiFlag & (FO_EXACT | FO_INCL)))
	{
		if( uiFlag & FO_INCL)
		{
			uiDrn = 1;
		}
		else
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmRecordRetrieveCS( pDb, uiContainer, uiDrn, uiFlag, 
			ppRecord, puiDrnRV);
		goto ExitCS;
	}

	// Remove the exclusive case and turn it into an inclusive search.
	
	if( uiFlag & FO_EXCL)
	{
		// For records, exclusive is same as inclusive drn+1
		
		uiDrn++;
		uiFlag &= ~FO_EXCL;
		uiFlag = FO_INCL;
	}

	uiSaveInitNestLevel = pDb->uiInitNestLevel;

	// Test both FO_FIRST and FO_LAST at the same time to save time.

	if (uiFlag & (FO_FIRST | FO_LAST))
	{
		if (uiFlag & FO_FIRST)
		{
			uiFlag = FO_INCL;
			uiDrn = 1;
		}
		else
		{
			uiFlag = FO_LAST;
			uiDrn = DRN_LAST_MARKER - 1;
			goto Search_Record;
		}
	}
	
	if( uiFlag & FO_EXACT)
	{
		if( RC_OK( rc = flmRcaRetrieveRec( pDb, &bTransStarted,
			uiContainer, uiDrn, TRUE, NULL, NULL, ppRecord)))
		{
			uiFoundDrn = uiDrn;
		}
		else if (rc != FERR_NOT_FOUND || ppRecord)
		{
			goto Exit;
		}
		// else do nothing - fall through, because even though we passed
		// TRUE in for the bOkToGetFromDisk parameter, flmRcaRetrieveRec
		// will NOT try to fetch from disk if ppRecord is NULL - so
		// we still need to try to fetch from disk.
	}
	else // only FO_INCL case can exist for the else case.
	{

		// Let's be optimistic and see if record is already in cache
		// before we search the b-tree.  Don't try to retrieve the record from
		// disk if it is not in cache ... the next record may not have a DRN
		// of uiDrn and we don't want to search the b-tree twice to fetch
		// the next record.

		if( RC_OK( rc = flmRcaRetrieveRec( pDb, &bTransStarted,
			uiContainer, uiDrn, FALSE, NULL, NULL, ppRecord)))
		{
			uiFoundDrn = uiDrn;
		}
		else if( rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
		// else - rc == FERR_NOT_FOUND, so we need to see if we can find the
		// next record on disk.
	}

	if( !uiFoundDrn)
	{
Search_Record:
		if( uiSaveInitNestLevel == pDb->uiInitNestLevel)
		{
			if ( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
						FDB_TRANS_GOING_OK, 0, &bTransStarted)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
		{
			goto Exit;
		}

		f_UINT32ToBigEndian( (FLMUINT32)uiDrn, pSearchBuf);
		FSInitStackCache( &stack [0], BH_MAX_LEVELS);
		pStack = &stack[0];
		bStackInitialized = TRUE;
		pStack->pKeyBuf = pKeyBuf;
			
		// Search the B-Tree for the key.
		
		if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, pSearchBuf, 4, 0)))
		{
			goto Exit;
		}

		uiKeyRelPos = pStack->uiCmpStatus;
		if( uiFlag & FO_EXACT)
		{
			if( uiKeyRelPos != BT_EQ_KEY)
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
		}
		else	// inclusive or FO_LAST
		{

			// Handle FO_LAST case - If FO_LAST bit was set, uiFlag will
			// have been changed above to simply be equal to FO_LAST.

			if (uiFlag == FO_LAST)
			{
				if (pStack->uiBlkAddr == BT_END)
				{
					rc = RC_SET( FERR_BOF_HIT);
					goto Exit;
				}

				if (uiKeyRelPos == BT_END_OF_DATA ||
					 f_bigEndianToUINT32( pKeyBuf) > uiDrn)
				{

					// Position to the last element in the block.

					if (RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
					{
						if (rc == FERR_BT_END_OF_DATA)
						{
							rc = RC_SET( FERR_BOF_HIT);
						}
						goto Exit;
					}

					// Position to beginning of record.

					while (BBE_NOT_FIRST( CURRENT_ELM( pStack )))
					{
						if (RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
						{
							if (rc == FERR_BT_END_OF_DATA)
							{ 
								rc = RC_SET( FERR_BTREE_ERROR);
							}
							
							goto Exit;
						}
					}
				}
			}
			else if( uiKeyRelPos == BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_EOF_HIT);
				goto Exit;
			}
		}
		
		uiFoundDrn = f_bigEndianToUINT32( pKeyBuf);
		
		if( uiFoundDrn == DRN_LAST_MARKER)
		{
			if( uiFlag & FO_EXACT)
			{
				rc = RC_SET( FERR_NOT_FOUND);
			}
			else
			{
				rc = RC_SET( FERR_EOF_HIT);
			}
			
			goto Exit;
		}

		// Need to return the record?
		
		if( ppRecord)
		{
			if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
					uiContainer, uiFoundDrn, TRUE, pStack, pLFile, ppRecord)))
			{
				goto Exit;
			}
		}
	}

	if( puiDrnRV)
	{
		*puiDrnRV = uiFoundDrn;
	}

	if( (pDbStats = pDb->pDbStats) != NULL)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->ui64NumRecordReads++;
	}

ExitCS:

	// Call record validator callback
	
	if( pDb->fnRecValidator)
	{
		FLMBOOL	bSavedInvisTrans;

		CB_ENTER( pDb, &bSavedInvisTrans);
		(void)(pDb->fnRecValidator)( FLM_RECORD_RETRIEVE, hDb,
				uiContainer, *ppRecord, NULL, pDb->RecValData, &rc);
		CB_EXIT( pDb, bSavedInvisTrans);
	}

Exit:
	
	if (bStackInitialized)
	{
		FSReleaseStackCache( stack, BH_MAX_LEVELS, FALSE);
	}

	if (bTransStarted)
	{
		RCODE 	rc2 = flmAbortDbTrans( pDb);
		
		if (RC_OK( rc))
		{
			rc = rc2;
		}
	}

	// Don't want it to call fdbExit if fdbInit wasn't called.

	if (pDb->uiInitNestLevel == uiSaveInitNestLevel)
	{
		pDb = NULL;
	}
	
	flmExit( FLM_RECORD_RETRIEVE, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:		Retrieves a record based on uiDrn and uiFlag client-server.
****************************************************************************/
FSTATIC RCODE flmRecordRetrieveCS(
	FDB *				pDb,
	FLMUINT 			uiContainer,
	FLMUINT			uiDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecord,
	FLMUINT *		puiDrnRV)
{
	RCODE				rc;
	CS_CONTEXT *	pCSContext = pDb->pCSContext;
	void *			pvMark = pCSContext->pool.poolMark();
	FCL_WIRE			Wire( pCSContext, pDb);

	// Set the record object so that it can be re-used,
	// if possible

	if( ppRecord)
	{
		Wire.setRecord( *ppRecord);
		if( *ppRecord)
		{
			(*ppRecord)->Release();
			*ppRecord = NULL;
		}
	}

	// Set the temporary pool

	Wire.setPool( &pCSContext->pool);

	// Send a request to retrieve the record

	if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_RECORD,
		FCS_OP_RECORD_RETRIEVE)))
	{
		goto Exit;
	}

	if (uiContainer)
	{
		if (RC_BAD( rc = Wire.sendNumber(
			WIRE_VALUE_CONTAINER_ID, uiContainer)))
		{
			goto Transmission_Error;
		}
	}

	if (uiDrn)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_DRN, uiDrn)))
		{
			goto Transmission_Error;
		}
	}

	if (uiFlag)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiFlag)))
		{
			goto Transmission_Error;
		}
	}

	if (ppRecord)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_BOOLEAN, TRUE)))
		{
			goto Transmission_Error;
		}
	}

	if (RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	// Read the response

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if( puiDrnRV)
	{
		*puiDrnRV = Wire.getDrn();
	}

	if( ppRecord)
	{
		if( (*ppRecord = Wire.getRecord()) != NULL)
		{
			(*ppRecord)->AddRef();
		}
	}

Exit:

	pCSContext->pool.poolReset( pvMark);
	return( rc);

Transmission_Error:

	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}
