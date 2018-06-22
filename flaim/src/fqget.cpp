//-------------------------------------------------------------------------
// Desc:	Query record retrieval
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE flmCurCSTestRec(
	CURSOR *			pCursor,
	FLMUINT			uiDrn,
	FlmRecord *		pTestRec,
	FLMBOOL *		pbIsMatchRV);

/****************************************************************************
Desc: Makes a SET_DEL from a record.
****************************************************************************/
RCODE flmCurMakeKeyFromRec(
	FDB *				pDb,
	IXD *				pIxd,
	F_Pool *			pPool,
	FlmRecord *		pRec,
	FLMBYTE **		ppucKeyBuffer,
	FLMUINT *		puiKeyLen)
{
	REC_KEY *	pKey = NULL;
	RCODE			rc = FERR_OK;

	// Set up CDL table and other KREF stuff in FDB.

	if (RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}

	// Parse the keys from the record, and verify that the record contains
	// only one key.

	rc = flmGetRecKeys( pDb, pIxd, pRec, pRec->getContainerID(),
								TRUE, pPool, &pKey);

	KYAbortCurrentRecord( pDb);

	if (RC_BAD( rc))
	{
		goto Exit;
	}
	else if (!pKey)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	if (pKey->pNextKey)
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Now allocate a key buffer to hold the key.

	if (!(*ppucKeyBuffer))
	{
		if( RC_BAD( rc = pPool->poolCalloc( MAX_KEY_SIZ, (void **)ppucKeyBuffer)))
		{
			goto Exit;
		}
	}

	// We pass in pRec->getContainerID(), because every key we generate
	// came from pRec, and we want to use its container number.

	if (RC_BAD( rc = KYTreeToKey( pDb, pIxd, pKey->pKey,
											pRec->getContainerID(),
											*ppucKeyBuffer,
											puiKeyLen, 0)))
	{
		goto Exit;
	}
Exit:

	while (pKey)
	{
		pKey->pKey->Release();
		pKey->pKey = NULL;
		pKey = pKey->pNextKey;
	}
	return( rc);
}

/****************************************************************************
Desc:	Sets a query's position from a passed-in DRN.
****************************************************************************/
RCODE flmCurSetPosFromDRN(
	CURSOR *			pCursor,
	FLMUINT			uiDRN
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	F_Pool *			pTempPool;
	FDB *				pDb = NULL;
	IXD *				pIxd;
	LFILE *			pLFile;
	SUBQUERY *		pSubQuery;
	FLMBOOL			bPositioned;
	FLMBYTE *		pucKeyBuffer = NULL;
	FLMUINT			uiKeyLen;

	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	pTempPool = &pDb->TempPool;

	// Read the record associated with the DRN, and construct an index key

	rc = flmRcaRetrieveRec( pDb, NULL, pCursor->uiContainer, uiDRN,
						FALSE, NULL, NULL, &pRec);

	if (rc == FERR_NOT_FOUND)
	{
		if (RC_BAD( rc = fdictGetContainer( pDb->pDict,
										pCursor->uiContainer, &pLFile)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDRN,
				&pRec, NULL, NULL)))
		{
			goto Exit;
		}
	}
	else if (RC_BAD( rc))
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

	// Go through the subquery list to find the sub-query into which the
	// DRN falls.

	bPositioned = FALSE;
	for( pSubQuery = pCursor->pSubQueryList;
			pSubQuery && !bPositioned;
			pSubQuery = pSubQuery->pNext)
	{
		FLMUINT	uiResult;

		// See if the record satisifies this sub-query's criteria.
		// If not, we cannot position into this sub-query.

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery, pRec,
											FALSE, &uiResult)))
		{
			goto Exit;
		}

		// Found a sub-query where this DRN passes.  Set this sub-query to
		// be the current sub-query on certain conditions.

		if (uiResult == FLM_TRUE)
		{
			switch (pSubQuery->OptInfo.eOptType)
			{
				case QOPT_USING_INDEX:
					if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
												pDb->pFile->bInLimitedMode,
												pSubQuery->OptInfo.uiIxNum, 
												NULL, &pIxd)))
					{
						goto Exit;
					}

					// If we cannot create a key for this index, go to the next
					// subquery.

					if (RC_BAD( rc = flmCurMakeKeyFromRec(	pDb, pIxd, pTempPool,
												pRec, &pucKeyBuffer, &uiKeyLen)))
					{
						if (rc == FERR_NOT_FOUND || rc == FERR_ILLEGAL_OP)
						{
							rc = FERR_OK;
							break;
						}
						goto Exit;
					}

					// Position to the key and drn - If we can't just go to the
					// next subquery.

					if (RC_BAD( rc = pSubQuery->pFSIndexCursor->positionTo( pDb,
												pucKeyBuffer, uiKeyLen, uiDRN)))
					{
						if (rc == FERR_NOT_FOUND ||
							 rc == FERR_EOF_HIT ||
							 rc == FERR_BOF_HIT)
						{
							rc = FERR_OK;
							break;
						}
						goto Exit;
					}

					// Retrieve the current key and DRN from the index cursor.

					rc = pSubQuery->pFSIndexCursor->currentKey( pDb,
											&pSubQuery->pRec, &pSubQuery->uiDrn);
					if (RC_BAD( rc))
					{
						goto Exit;
					}
					bPositioned = TRUE;
					pSubQuery->uiCurrKeyMatch = FLM_UNK;
					pSubQuery->bFirstReference = FALSE;

					// These should have been set by the call to currentKey.

					flmAssert( pSubQuery->pRec->getContainerID() ==
										pCursor->uiContainer);
					flmAssert( pSubQuery->pRec->getID() == pSubQuery->uiDrn);

					pSubQuery->bRecIsAKey = TRUE;
					break;
				case QOPT_USING_PREDICATE:
					// Can't position in a predicate - go to next sub-query.
					break;
				case QOPT_SINGLE_RECORD_READ:
					bPositioned = TRUE;
					pSubQuery->uiDrn = uiDRN;
					break;
				case QOPT_PARTIAL_CONTAINER_SCAN:
				case QOPT_FULL_CONTAINER_SCAN:
					rc = pSubQuery->pFSDataCursor->positionTo( pDb, uiDRN);
					if (RC_BAD( rc))
					{
						if (rc == FERR_NOT_FOUND ||
							 rc == FERR_EOF_HIT ||
							 rc == FERR_BOF_HIT)
						{
							rc = FERR_OK;
							break;
						}
						goto Exit;
					}
					bPositioned = TRUE;
					break;
				default:
					flmAssert( 0);
					break;
			}
			if (bPositioned)
			{
				pCursor->uiLastRecID = uiDRN;
				pCursor->pCurrSubQuery = pSubQuery;
				break;
			}
		}
	}
	if (!bPositioned)
	{
		rc = RC_SET( FERR_NOT_FOUND);
	}

Exit:

	if (pRec)
	{
		pRec->Release();
	}
	flmExit( FLM_CURSOR_CONFIG, pDb, rc);

	return( pCursor->rc = rc);
}

/****************************************************************************
Desc:	Given a cursor and two DRNs, this API does the following:
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorCompareDRNs(
	HFCURSOR		hCursor,
	FLMUINT		uiDRN1,
	FLMUINT		uiDRN2,
	FLMUINT		uiTimeLimit,
	FLMINT *		piCmpResult,
	FLMBOOL *	pbTimedOut,
	FLMUINT *	puiCount)
{
	RCODE					rc = FERR_OK;
	CURSOR *				pCursor = (CURSOR *)hCursor;
	FDB *					pDb = NULL;
	F_Pool *				pTempPool;
	FLMBYTE *			pucKey1;
	FLMUINT				uiKey1Len;
	FLMBYTE *			pucKey2;
	FLMUINT				uiKey2Len;
	FlmRecord *			pRec1 = NULL;
	FlmRecord *			pRec2 = NULL;
	IXD *					pIxd;
	FSIndexCursor *	pTmpFSIndexCursor = NULL;
	FSIndexCursor *	pSaveFSIndexCursor = NULL;
	FLMUINT				uiSaveTimeLimit = 0;
	LFILE *				pLFile;
	FLMUINT				uiIndexNum;
	FLMINT				iCmp;

	// Verify that return params are non-NULL.
	
	flmAssert( piCmpResult != NULL);
	flmAssert( pbTimedOut != NULL);
	flmAssert( puiCount != NULL);
	flmAssert( pCursor != NULL);

	if (!pCursor)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if (pCursor->pCSContext)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If the two DRNs are equal, there is no need to read the records to compare
	// their keys.  Furthermore, if they are in the cursor's result set, the
	// return count will be 2, otherwise it will be 0.
	
	if (uiDRN1 == uiDRN2)
	{
		*piCmpResult = 0;
		if (uiTimeLimit)
		{
			FLMBOOL	bInRSet;
			
			if (RC_BAD( rc = FlmCursorTestDRN( hCursor, uiDRN1, &bInRSet)))
			{
				goto Exit;
			}
			if (bInRSet)
			{
				*puiCount = 2;
			}
			else
			{
				*puiCount = 0;
			}
			*pbTimedOut = FALSE;
		}
		else
		{
			*pbTimedOut = TRUE;
			rc = FERR_OK;
		}
		goto Exit;
	}
	
	*puiCount = 0;
	*pbTimedOut = (FLMBOOL)((!uiTimeLimit)
									? TRUE
									: FALSE);
	
	// Read the records associated with the two DRNs, and construct index keys
	// from them to determine the directional relationship between them.
	
	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	pTempPool = &pDb->TempPool;

	// Get the records corresponding to the two DRNs.
			
	rc = flmRcaRetrieveRec( pDb, NULL, pCursor->uiContainer, uiDRN1,
					FALSE, NULL, NULL, &pRec1);
	if (rc == FERR_NOT_FOUND)
	{
		if (RC_BAD( rc = fdictGetContainer( pDb->pDict,
									pCursor->uiContainer, &pLFile)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDRN1,
						&pRec1, NULL, NULL)))
		{
			goto Exit;
		}
	}
	else if (RC_BAD( rc))
	{
		goto Exit;
	}

	// Retrieve the 2nd record.

	rc = flmRcaRetrieveRec( pDb, NULL, pCursor->uiContainer, uiDRN2,
		FALSE, NULL, NULL, &pRec2);
	if (rc == FERR_NOT_FOUND)
	{
		if (RC_BAD( rc = fdictGetContainer( pDb->pDict,
									pCursor->uiContainer, &pLFile)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDRN2,
									&pRec2, NULL, NULL)))
		{
			goto Exit;
		}
	}
	else if (RC_BAD( rc))
	{
		goto Exit;
	}

	// At this point, both DRNs have been found.  Now generate keys from each
	// of the records and perform the following actions:
	// 1) Verify that only one key exists in each record.  If there is more
	//		than one key, no clear comparison can be made.
	// 2)  Compare the two keys and set *piCmpResult as follows:
	//		  <0		First Key < Second Key
	//			0		First Key = Second Key
	//		  >0		First Key > Second Key

	// If necessary, optimize the query here. NOTE: this should be done only if
	// FLAIM is to choose an index for the query.

	if (!pCursor->bOptimized && pCursor->uiIndexNum == FLM_SELECT_INDEX)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}

	if (pCursor->bOptimized)
	{

		// If there are multiple subqueries, that means there are
		// multiple indexes.  Set uiIndexNum to zero in this
		// case so that we will return FERR_NOT_IMPLEMENTED
		// below.

		if (!pCursor->pSubQueryList ||
			 pCursor->pSubQueryList->pNext ||
			 pCursor->pSubQueryList->OptInfo.eOptType != QOPT_USING_INDEX)
		{
			uiIndexNum = 0;
		}
		else
		{
			uiIndexNum = pCursor->pSubQueryList->OptInfo.uiIxNum;
		}
	}
	else
	{
		uiIndexNum = pCursor->uiIndexNum;
	}

	// Index number of zero means that the user either did not set an
	// index, or no index was selected, or multiple indexes were
	// selected - all of which disqualify this query from being able
	// to compare to DRNs.

	if (!uiIndexNum)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIndexNum, NULL, &pIxd)))
	{
		goto Exit;
	}

	pucKey1 = NULL;
	if (RC_BAD( rc = flmCurMakeKeyFromRec(	pDb, pIxd, pTempPool,
											pRec1, &pucKey1, &uiKey1Len)))
	{
		goto Exit;
	}

	pucKey2 = NULL;
	if (RC_BAD( rc = flmCurMakeKeyFromRec(	pDb, pIxd, pTempPool,
											pRec2, &pucKey2, &uiKey2Len)))
	{
		goto Exit;
	}

	if (uiKey1Len > uiKey2Len)
	{
		if ((iCmp = f_memcmp( pucKey1, pucKey2, uiKey2Len)) == 0)
		{
			iCmp = 1;
		}
		else
		{
			iCmp = ( (iCmp > 0) ? 1 : -1);
		}
	}
	else if (uiKey1Len < uiKey2Len)
	{
		if ((iCmp = f_memcmp( pucKey1, pucKey2, uiKey1Len)) == 0)
		{
			iCmp = -1;
		}
		else
		{
			iCmp = ( (iCmp > 0) ? 1 : -1);
		}
	}
	else
	{
		if ((iCmp = f_memcmp( pucKey1, pucKey2, uiKey2Len)) != 0)
		{
			iCmp = ( (iCmp > 0) ? 1 : -1);
		}
		else
		{

			// Keys are equal, compare DRNs.

			iCmp = (FLMINT)((uiDRN1 > uiDRN2)
								 ? (FLMINT)-1
									: (FLMINT)((uiDRN1 < uiDRN2)
												  ? (FLMINT)1
												  : (FLMINT)0));
		}
	}
	*piCmpResult = iCmp;

	// If time limit is zero, don't need to do the count.

	if (!uiTimeLimit)
	{
		goto Exit;
	}

	// If keys are equal, simply return a count of two.

	if (!iCmp)
	{
		*puiCount = 2;
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

	// Create a new temporary index cursor.

	if ((pTmpFSIndexCursor = f_new FSIndexCursor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Count the entries between the two keys/references.

	if (iCmp > 0)
	{
		if (RC_BAD( rc = pTmpFSIndexCursor->setupKeys( pDb, pIxd,
									pucKey1, uiKey1Len, uiDRN1,
									pucKey2, uiKey2Len, uiDRN2, FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pTmpFSIndexCursor->setupKeys( pDb, pIxd,
									pucKey1, uiKey1Len, uiDRN1,
									pucKey2, uiKey2Len, uiDRN2, FALSE)))
		{
			goto Exit;
		}
	}

	// Intersect the temporary index cursor's key range with the
	// cursor's key range(es).

	pSaveFSIndexCursor = pCursor->pSubQueryList->pFSIndexCursor;
	uiSaveTimeLimit = pCursor->uiTimeLimit;
	if (RC_BAD( rc = pTmpFSIndexCursor->intersectKeys( pDb, pSaveFSIndexCursor)))
	{
		goto Exit;
	}

	// Set the query's index cursor to the temporary index cursor and count
	// the entries between them.

	if (uiTimeLimit == FLM_NO_LIMIT)
	{
		pCursor->uiTimeLimit = 0;
	}
	else
	{
		pCursor->uiTimeLimit = FLM_SECS_TO_TIMER_UNITS( uiTimeLimit);
	}

	// Perform the count operation.

	pCursor->pSubQueryList->pFSIndexCursor = pTmpFSIndexCursor;
	if (RC_BAD( rc = flmCurSearch( FLM_CURSOR_REC_COUNT, pCursor, TRUE,
								TRUE, puiCount, NULL, NULL, NULL)))
	{
		if (rc == FERR_EOF_HIT)
		{
			rc = FERR_OK;
		}
		else if (rc == FERR_TIMEOUT)
		{
			rc = FERR_OK;
			*pbTimedOut = TRUE;
		}
		goto Exit;
	}
Exit:

	if (pRec1)
	{
		pRec1->Release();
	}

	if (pRec2)
	{
		pRec2->Release();
	}

	if (pTmpFSIndexCursor)
	{
		pTmpFSIndexCursor->Release();
		pTmpFSIndexCursor = NULL;
	}

	// Restore saved index cursor if necessary.

	if (pSaveFSIndexCursor)
	{
		pCursor->pSubQueryList->pFSIndexCursor = pSaveFSIndexCursor;
		pCursor->uiTimeLimit = uiSaveTimeLimit;
	}
	flmExit( FLM_CURSOR_COMPARE_DRNS, pDb, rc);

	return( pCursor->rc = rc);
}

/****************************************************************************
Desc:	Does FlmCursorTestRec and FlmCursorTestDRN over the client/server
		line.
****************************************************************************/
FSTATIC RCODE flmCurCSTestRec(
	CURSOR *		pCursor,
	FLMUINT		uiDrn,
	FlmRecord *	pTestRec,
	FLMBOOL *	pbIsMatchRV
	)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext = pCursor->pCSContext;
	FCL_WIRE			Wire( pCSContext);

	// If there is no VALID id for the cursor, get one.

	if (pCursor->uiCursorId == FCS_INVALID_ID)
	{
		if (RC_BAD( rc = flmInitCurCS( pCursor)))
		{
			goto Exit;
		}
	}

	// Send a request to test the record or DRN.

	if (RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_ITERATOR, FCS_OP_ITERATOR_TEST_REC)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber(
		WIRE_VALUE_ITERATOR_ID, pCursor->uiCursorId)))
	{
		goto Transmission_Error;
	}

	if (pTestRec)
	{
		if (RC_BAD( rc = Wire.sendRecord( WIRE_VALUE_RECORD, pTestRec)))
		{
			goto Transmission_Error;
		}
	}
	else
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_DRN, uiDrn)))
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

	*pbIsMatchRV = Wire.getBoolean();
	rc = Wire.getRCode();

Exit:

	return( rc);

Transmission_Error:

	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}


/****************************************************************************
Desc:	Checks a record that has been retrieved from the database to see if
		satisfies the cursor selection criteria.
		IMPORTANT NOTE: pRec's containerID better be set to the container it
		came from, because flmCurEvalCriteria verifies that the record's
		container number matches the cursor's container number.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorTestRec(
	HFCURSOR			hCursor,
	FlmRecord * 	pRec,
	FLMBOOL * 		pbIsMatch)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = NULL;
	CURSOR *			pCursor = (CURSOR *)hCursor;
	SUBQUERY *		pSubQuery;

	flmAssert( pCursor != NULL);
	*pbIsMatch = FALSE;

	if (pCursor->pCSContext)
	{
		rc = flmCurCSTestRec( pCursor, 0, pRec, pbIsMatch);
		goto Exit2;
	}

	// Make sure that we don't have partially finished
	// query criteria.

	if (pCursor->QTInfo.uiNestLvl ||
		 ((pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND) &&
			 pCursor->QTInfo.pTopNode))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
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

	flmAssert( pCursor->pDb != NULL);

	pDb = pCursor->pDb;
	if (RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// Evaluate the record against all sub-queries until
	// we find a TRUE.
	// IMPORTANT NOTE: pRec's containerID better be set to the container it
	// came from, because flmCurEvalCriteria verifies that the record's
	// container number matches the cursor's container number.

	flmAssert( pRec->getContainerID() != 0);

	pSubQuery = pCursor->pSubQueryList;
	while (pSubQuery)
	{
		FLMUINT	uiResult;

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pRec, FALSE, &uiResult)))
		{
			goto Exit;
		}
		if (uiResult == FLM_TRUE)
		{
			*pbIsMatch = TRUE;
			break;
		}
		pSubQuery = pSubQuery->pNext;
	}

Exit:

	if (pDb)
	{
		fdbExit( pDb);
	}

Exit2:

	return( rc);
}

/****************************************************************************
Desc:		Retrieves the record identified by the passed-in DRN and checks it
			to see if it satisfies the cursor selection criteria.
Notes: 	This function is designed for use with cursors having only one
		 	associated source.  Multiple sources are not supported.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmCursorTestDRN(
	HFCURSOR			hCursor,
	FLMUINT 			uiDrn,
	FLMBOOL * 		pbIsMatch)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = NULL;
	CURSOR *			pCursor = (CURSOR *)hCursor;
	SUBQUERY *		pSubQuery;
	FlmRecord *		pRec = NULL;

	flmAssert( pCursor != NULL);
	*pbIsMatch = FALSE;
	if (pCursor->pCSContext)
	{
		rc = flmCurCSTestRec( pCursor, uiDrn, NULL, pbIsMatch);
		goto Exit2;
	}

	flmAssert( pCursor->pDb != NULL);
	if (RC_OK( rc = FlmRecordRetrieve( (HFDB)pCursor->pDb,
							pCursor->uiContainer,
							uiDrn, FO_EXACT, &pRec, NULL)))
	{

		// Optimize the subqueries as necessary

		if (!pCursor->bOptimized)
		{
			if (RC_BAD( rc = flmCurPrep( pCursor)))
			{
				goto Exit;
			}
		}

		pDb = pCursor->pDb;
		if (RC_BAD(rc = flmCurDbInit( pCursor)))
		{
			goto Exit;
		}

		// Evaluate the record against all sub-queries until
		// we find a TRUE.

		pSubQuery = pCursor->pSubQueryList;
		while (pSubQuery)
		{
			FLMUINT	uiResult;

			if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
										pRec, FALSE, &uiResult)))
			{
				goto Exit;
			}
			if (uiResult == FLM_TRUE)
			{
				*pbIsMatch = TRUE;
				break;
			}
			pSubQuery = pSubQuery->pNext;
		}

	}

Exit:

	if (pDb)
	{
		fdbExit( pDb);
	}

	if (pRec)
	{
		pRec->Release();
	}

Exit2:
	return( rc);
}
