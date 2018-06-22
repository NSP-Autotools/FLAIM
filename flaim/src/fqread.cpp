//-------------------------------------------------------------------------
// Desc:	Query record retrieval
// Tabs:	3
//
// Copyright (c) 1994-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmCurCSPerformRead(
	CURSOR *			pCursor,
	eFlmFuncs		eFlmFuncId,
	FlmRecord **	ppRecordRV,
	FLMUINT *		puiDrnRV,
	FLMUINT *		puiCountRV);

FSTATIC RCODE flmCurGetDRNRec(
	CURSOR * 		pCursor,
	FLMUINT			uiDRN,
	FlmRecord **	ppRecord);

/****************************************************************************
Desc:	Gets the requested record, DRN, or count over the CS line.
****************************************************************************/
FSTATIC RCODE flmCurCSPerformRead(
	CURSOR *			pCursor,
	eFlmFuncs		eFlmFuncId,
	FlmRecord **	ppRecordRV,
	FLMUINT *		puiDrnRV,
	FLMUINT *		puiCountRV)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext = pCursor->pCSContext;
	FCL_WIRE			Wire( pCSContext);
	void *			pvMark = pCSContext->pool.poolMark();
	FLMUINT			uiCSOp = 0;

	// If there is no VALID id for the cursor, get one.

	if (pCursor->uiCursorId == FCS_INVALID_ID)
	{
		if (RC_BAD( rc = flmInitCurCS( pCursor)))
		{
			goto Exit;
		}
	}

	Wire.setFDB( pCursor->pDb);

	// Set the temporary pool

	Wire.setPool( &pCSContext->pool);

	// Set the record object so that it can be re-used,
	// if possible

	if (ppRecordRV)
	{
		Wire.setRecord( *ppRecordRV);
		if (*ppRecordRV)
		{
			(*ppRecordRV)->Release();
			*ppRecordRV = NULL;
		}
	}

	// Map Function ID to CS Op
	switch (eFlmFuncId)
	{
	case FLM_CURSOR_REC_COUNT:
		uiCSOp = FCS_OP_ITERATOR_COUNT;
		break;
	case FLM_CURSOR_FIRST:
		uiCSOp = FCS_OP_ITERATOR_FIRST;
		break; 
	case FLM_CURSOR_LAST:
		uiCSOp = FCS_OP_ITERATOR_LAST;
		break;	
	case FLM_CURSOR_NEXT:
		uiCSOp = FCS_OP_ITERATOR_NEXT;
		break;
	case FLM_CURSOR_PREV:
		uiCSOp = FCS_OP_ITERATOR_PREV;
		break;
	case FLM_CURSOR_FIRST_DRN:
		uiCSOp = FCS_OP_ITERATOR_FIRST;
		break;
	case FLM_CURSOR_LAST_DRN:
		uiCSOp = FCS_OP_ITERATOR_LAST;
		break;
	case FLM_CURSOR_NEXT_DRN:
		uiCSOp = FCS_OP_ITERATOR_NEXT;
		break;
	case FLM_CURSOR_PREV_DRN:
		uiCSOp = FCS_OP_ITERATOR_PREV;
		break;
	default:	
		flmAssert( 0);					// Unsupported flaim function hit.
	}

	// Send a request to perform the read.

	if (RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_ITERATOR, uiCSOp)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber(
		WIRE_VALUE_ITERATOR_ID, pCursor->uiCursorId)))
	{
		goto Transmission_Error;
	}

	if (puiDrnRV && !ppRecordRV)
	{
		if (RC_BAD( rc = Wire.sendNumber(
			WIRE_VALUE_FLAGS, FCS_ITERATOR_DRN_FLAG)))
		{
			goto Transmission_Error;
		}
	}

	if (RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	// Read the response.

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if (puiCountRV)
	{
		*puiCountRV = (FLMUINT)Wire.getCount();
	}

	if (ppRecordRV)
	{
		if ((*ppRecordRV = Wire.getRecord()) != NULL)
		{
			(*ppRecordRV)->AddRef();
		}
	}

	if (puiDrnRV)
	{
		if (ppRecordRV && *ppRecordRV)
		{
			*puiDrnRV = (*ppRecordRV)->getID();
		}
		else
		{
			*puiDrnRV = Wire.getDrn();
		}
	}

	rc = Wire.getRCode();

Exit:
	
	pCSContext->pool.poolReset( pvMark);
	return( rc);

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:	Gets the requested record.
****************************************************************************/
FLMEXP RCODE FLMAPI flmCurPerformRead(
	eFlmFuncs		eFlmFuncId,
	HFCURSOR			hCursor,
	FLMBOOL			bReadForward,
	FLMBOOL			bFirstRead,
	FLMUINT *		puiSkipCount,
	FlmRecord **	ppRecord,
	FLMUINT *		puiDrn
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiDrn = 0;
	CURSOR *	pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	// Make sure the record is clear.

	if (ppRecord && *ppRecord)
	{
		(*ppRecord)->Release();
		*ppRecord = NULL;
	}

	if (pCursor->bEliminateDups)
	{
		if( pCursor->pDRNSet && (bFirstRead || !pCursor->bOptimized))
		{
			pCursor->pDRNSet->Release();
			pCursor->pDRNSet = NULL;
		}
	}

	if (!bFirstRead)
	{
		if (pCursor->ReadRc == FERR_EOF_HIT)
		{
			if (bReadForward)
			{
				rc = pCursor->ReadRc;
				goto Save_RecId;
			}
			else
			{
				bFirstRead = TRUE;
			}
		}
		else if (pCursor->ReadRc == FERR_BOF_HIT)
		{
			if (!bReadForward)
			{
				rc = pCursor->ReadRc;
				goto Save_RecId;
			}
			else
			{
				bFirstRead = TRUE;
			}
		}

		// No read has been performed yet - or the last
		// read returned an error besides eof or bof.

		else if (!pCursor->uiLastRecID)
		{
			bFirstRead = TRUE;
		}
	}

	pCursor->ReadRc = FERR_OK;

	if (pCursor->pCSContext)
	{
		rc = flmCurCSPerformRead( pCursor, eFlmFuncId,
						ppRecord, &uiDrn, NULL);
	}
	else
	{
		// Optimize the query if necessary.

		if (!pCursor->bOptimized)
		{
			bFirstRead = TRUE;
			if (RC_BAD( rc = flmCurPrep( pCursor)))
			{
				goto Exit;
			}
		}

		// If this is an empty query, return EOF or BOF.

		if (pCursor->bEmpty)
		{
			pCursor->rc =
			rc = (RCODE)((bReadForward)
							 ? RC_SET( FERR_EOF_HIT)
							 : RC_SET( FERR_BOF_HIT));
		}
		else
		{
			pCursor->rc = rc = flmCurSearch( eFlmFuncId, pCursor, bFirstRead,
										bReadForward, NULL,
										puiSkipCount, ppRecord, &uiDrn);
		}
	}

	if (RC_BAD( rc))
	{
		if (rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
		{
			pCursor->ReadRc = rc;
		}
		uiDrn = 0;
	}

Save_RecId:

	// Set a flag indicating that this cursor has been repositioned.

	pCursor->bUsePrcntPos = FALSE;
	pCursor->uiLastRecID = uiDrn;

Exit:
	if (puiDrn)
	{
		*puiDrn = uiDrn;
	}

	return( rc);
}

/****************************************************************************
Desc:	Gets the requested record given a DRN.
****************************************************************************/
FSTATIC RCODE flmCurGetDRNRec(
	CURSOR * 		pCursor,
	FLMUINT			uiDRN,
	FlmRecord **	ppRecord)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	LFILE *	pLFile;
			
	if (pCursor->pCSContext)
	{
		HFDB			hDb = (HFDB)pCursor->pDb;
		FLMUINT		uiContainer = pCursor->uiContainer;
		FCL_WIRE		Wire( pCursor->pCSContext);

		Wire.setFDB( (FDB *)hDb);
		for (;;)
		{
			rc = FlmRecordRetrieve( hDb, uiContainer, uiDRN, FO_EXACT, ppRecord, NULL);
			if (rc != FERR_OLD_VIEW)
			{
				break;
			}

			if (RC_BAD( rc = Wire.doTransOp( FCS_OP_TRANSACTION_RESET, 
				FLM_READ_TRANS, 0, 0)))
			{
				break;
			}
		}
		goto Exit;
	}

	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	rc = flmRcaRetrieveRec( pDb, NULL, pCursor->uiContainer,
				uiDRN, FALSE, NULL, NULL, ppRecord);
	if (rc == FERR_NOT_FOUND)
	{
		if (RC_BAD( rc = fdictGetContainer( pDb->pDict, 
					pCursor->uiContainer, &pLFile)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDRN,
								ppRecord, NULL, NULL)))
		{
			goto Exit;
		}
	}

Exit:
	if (pDb)
	{
		fdbExit( pDb);
	}
	return( rc);
}

/****************************************************************************
Desc:	Retrieves the record currently pointed to by a cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorCurrent(
	HFCURSOR 		hCursor,
	FlmRecord **	ppRecord)
{
	RCODE				rc = FERR_OK;
	CURSOR *			pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	
	*ppRecord = NULL;

	if (pCursor->uiLastRecID == 0)
	{
		if (RC_OK( rc = pCursor->ReadRc))
		{
			rc = RC_SET( FERR_BOF_HIT);
		}
	}
	else if (RC_OK( pCursor->rc))
	{
		if (RC_BAD( rc = flmCurGetDRNRec( pCursor, pCursor->uiLastRecID,
									ppRecord)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = pCursor->rc;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Retrieves the DRN of the current record in a set defined by a cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorCurrentDRN(
	HFCURSOR 	hCursor,
	FLMUINT * 	puiDrn)
{
	RCODE			rc = FERR_OK;
	CURSOR *		pCursor = (CURSOR *)hCursor;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	*puiDrn = 0;

	if (!pCursor->uiLastRecID)
	{
		if (RC_OK( rc = pCursor->ReadRc))
		{
			rc = RC_SET( FERR_BOF_HIT);
		}
	}
	else if (RC_OK( pCursor->rc))
	{
		*puiDrn = pCursor->uiLastRecID;
		rc = FERR_OK;
	}
	else
	{
		rc = pCursor->rc;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Positions the cursor to a next or previous item at an offset relative
		to the current item and retrieves that item from the database.
Note:	Requests that position beyond the end of the result set will
		cause an EOF_HIT error to be returned.  Likewise, requests that
		position before the beginning of the result set will cause a
		BOF_HIT error to be returned.  Passing a relative position of 0 is
		invalid and will cause ILLEGAL_OP to be returned.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorMoveRelative(
	HFCURSOR				hCursor,
	FLMINT *				piPosition,
	FlmRecord **		ppRecord)
{
	RCODE			rc = FERR_OK;
	FLMINT		iPosition;
	FLMUINT		uiTmpPos;

	if ((iPosition = *piPosition) == 0)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	uiTmpPos = (FLMUINT)((iPosition < 0)
							? (FLMUINT)(-iPosition) 
							: (FLMUINT)iPosition);

	rc = flmCurPerformRead( FLM_CURSOR_MOVE_RELATIVE, hCursor,
		(FLMBOOL)((iPosition > 0) ? TRUE : FALSE), FALSE,
		&uiTmpPos, ppRecord, NULL);

	*piPosition = (FLMINT)((iPosition < 0)
						? (FLMINT)(iPosition + uiTmpPos) 
						: (FLMINT)(iPosition - uiTmpPos));
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Returns the number of records in a set defined by a cursor.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorRecCount(
	HFCURSOR			hCursor,
	FLMUINT *		puiCount)
{
	RCODE				rc = FERR_OK;
	CURSOR *			pCursor = (CURSOR *)hCursor;
	RCODE				TmpRc;
	FDB *				pDb = NULL;
	FLMBOOL			bSavedPosition = FALSE;

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	*puiCount = 0;

	if (pCursor->pCSContext)
	{
		rc = flmCurCSPerformRead( pCursor, FLM_CURSOR_REC_COUNT, 
											NULL, NULL, puiCount);
		goto Exit2;
	}

	pDb = pCursor->pDb;
	if( RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}

	// Save current position so we can restore it after doing the count.

	bSavedPosition = TRUE;
	rc = flmCurSearch( FLM_CURSOR_REC_COUNT, pCursor, TRUE, TRUE,
								puiCount, NULL, NULL, NULL);
	if (rc == FERR_EOF_HIT)
	{
		rc = FERR_OK;
	}
	
Exit:

	// Restore saved cursor settings if necessary.

	if (bSavedPosition)
	{
		if (RC_BAD( TmpRc = flmCurRestorePosition( pCursor)))
		{
			if (RC_OK( rc))
			{
				rc = TmpRc;
			}
		}
	}

	flmExit( FLM_CURSOR_REC_COUNT, pDb, rc);
	pCursor->rc = rc;
Exit2:
	return( rc);
}
