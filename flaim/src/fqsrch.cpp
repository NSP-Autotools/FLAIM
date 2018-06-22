//-------------------------------------------------------------------------
// Desc:	Query search
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

// Call progress callback once every second

#define STATUS_CB_INTERVAL 1

FSTATIC RCODE flmCurSetSubQuery(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery);

FSTATIC int DRNCompareFunc(
	void *	pvData1,
	void *	pvData2,
	void *	pvUserValue);

FSTATIC RCODE flmCurRecValidate(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiSkipCount,
	FLMUINT *	puiCount,
	FLMBOOL *	pbReturnRecOK);

FSTATIC RCODE flmCurRetrieveRec(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FLMUINT		uiContainer);

FSTATIC RCODE flmCurSearchIndex(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount,
	FLMBOOL		bGettingRecord);

FSTATIC RCODE flmCurSearchPredicate(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount);

FSTATIC RCODE flmCurSearchContainer(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount);

FSTATIC RCODE flmCurEvalSingleRec(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount);

/****************************************************************************
Desc: This routine will do all the setup needed to establish a sub-query as
		the current subquery for the query.
****************************************************************************/
FSTATIC RCODE flmCurSetSubQuery(
	CURSOR *		pCursor,
	SUBQUERY *	pSubQuery
	)
{
	RCODE			rc = FERR_OK;

	pCursor->pCurrSubQuery = pSubQuery;

	// Do setup associated with a change of sub-queries.

	f_memset( &pSubQuery->SQStatus, 0, sizeof( FCURSOR_SUBQUERY_STATUS));
	pSubQuery->SQStatus.hDb = (HFDB)pCursor->pDb;
	pSubQuery->SQStatus.uiContainerNum = pCursor->uiContainer;
	if (pSubQuery->OptInfo.eOptType == QOPT_USING_INDEX)
	{
		pSubQuery->SQStatus.uiIndexNum = pSubQuery->OptInfo.uiIxNum;

		// Make sure that all of the keys have been committed.

		if (RC_BAD( rc = KYFlushKeys( pCursor->pDb)))
		{
			goto Exit;
		}
	}
	else if (pSubQuery->OptInfo.eOptType == QOPT_SINGLE_RECORD_READ)
	{
		pSubQuery->bRecReturned = FALSE;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc: DRN comparison callback function for dynamic result set.
****************************************************************************/
FSTATIC int DRNCompareFunc(
	void *	pvData1,
	void *	pvData2,
	void *	// pvUserValue
	)
{
	if( *((FLMUINT *)pvData1) < *((FLMUINT *)pvData2))
	{
		return -1;
	}
	else if( *((FLMUINT *)pvData1) > *((FLMUINT *)pvData2))
	{
		return 1;
	}
	return 0;
}

/****************************************************************************
Desc: Validate a record that has passed the search criteria.  This routine
		checks the record against the validation function and also checks it
		against the result set, if there is one.
****************************************************************************/
FSTATIC RCODE flmCurRecValidate(
	eFlmFuncs		eFlmFuncId,
	CURSOR *			pCursor,
	SUBQUERY *		pSubQuery,
	FLMUINT *		puiSkipCount,
	FLMUINT *		puiCount,
	FLMBOOL *		pbReturnRecOK)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bSavedInvisTrans;

	// At this point, we have a record that has passed all the selection
	// criteria in the query.  If we have a record validator callback,
	// call it.

	if (pCursor->fnRecValidator)
	{
		CB_ENTER( pCursor->pDb, &bSavedInvisTrans);
		*pbReturnRecOK = (pCursor->fnRecValidator)( eFlmFuncId,
						(HFDB)pCursor->pDb, pCursor->uiContainer,
						pSubQuery->pRec, NULL, pCursor->RecValData, &rc);
		CB_EXIT( pCursor->pDb, bSavedInvisTrans);
		if (!(*pbReturnRecOK))
		{
			pSubQuery->SQStatus.uiNumRejectedByCallback++;
			rc = FERR_OK;
			goto Exit;
		}
		else if (RC_BAD( rc))
		{
			goto Exit;
		}
	}

	// Record passed all criteria -- check for dups if necessary.

	if (pCursor->bEliminateDups)
	{
		// Setup the result set

		if( !pCursor->pDRNSet)
		{
			char		szTmpDir[ F_PATH_MAX_SIZE];

			szTmpDir[ 0] = 0;
			
			if ((pCursor->pDRNSet = f_new F_DynSearchSet) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( gv_FlmSysData.bTempDirSet && gv_FlmSysData.szTempDir[ 0])
			{
				if( RC_BAD( rc = flmGetTmpDir( szTmpDir)))
				{
					goto Exit;
				}
			}

			if( !szTmpDir[ 0])
			{
				if( RC_BAD( rc = gv_FlmSysData.pFileSystem->pathReduce( 
					pCursor->pDb->pFile->pszDbPath, szTmpDir, NULL)))
				{
					goto Exit;
				}
			}
			
			if (RC_BAD( rc = pCursor->pDRNSet->setup( szTmpDir, sizeof( FLMUINT))))
			{
				goto Exit;
			}

			pCursor->pDRNSet->setCompareFunc( DRNCompareFunc, NULL);
		}

		if (RC_BAD( rc = pCursor->pDRNSet->addEntry( &pSubQuery->uiDrn)))
		{
			if (rc == NE_FLM_EXISTS)
			{
				*pbReturnRecOK = FALSE;
				rc = FERR_OK;
				pSubQuery->SQStatus.uiDupsEliminated++;
			}
			goto Exit;
		}
	}

	// If we get here, the record passed all possible criteria.
	// However, it will not be returned if we are skipping
	// or counting.

	pSubQuery->SQStatus.uiMatchedCnt++;

	// If this is a request to skip records, make sure that the
	// correct number have been skipped.

	if (puiSkipCount && (--(*puiSkipCount) > 0))
	{

		// If we are skipping, we need to continue past this
		// record.

		*pbReturnRecOK = FALSE;
		goto Exit;
	}

	// If we are counting, we also want to continue past this
	// record.

	if (puiCount)
	{
		(*puiCount)++;
		*pbReturnRecOK = FALSE;
		goto Exit;
	}

	// If we get to here, we are going to keep the record and
	// pass it back out to the caller.

	*pbReturnRecOK = TRUE;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Does the actual reading of a record from cache or disk.
****************************************************************************/
FSTATIC RCODE flmCurRetrieveRec(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FLMUINT		uiContainer
	)
{
	RCODE	rc = FERR_OK;

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, uiContainer,
				pSubQuery->uiDrn, TRUE, NULL, NULL, &pSubQuery->pRec)))
	{
		goto Exit;
	}
	
	pSubQuery->bRecIsAKey = FALSE;

Exit:

	if (RC_BAD( rc) && pSubQuery->pRec)
	{
		pSubQuery->pRec->Release();
		pSubQuery->pRec = NULL;
		pSubQuery->uiDrn = 0;
	}
	
	return( rc);
}

/****************************************************************************
Desc: Searches an index for matching record.
****************************************************************************/
FSTATIC RCODE flmCurSearchIndex(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount,
	FLMBOOL		bGettingRecord
	)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = pCursor->pDb;
	FSIndexCursor *	pFSIndexCursor = pSubQuery->pFSIndexCursor;
	FLMUINT				uiCBTimer = 0;
	FLMUINT				uiCurrCBTime;
	FLMBOOL				bReturnRecOK;
	FLMBOOL				bDoKeyMatch;
	FLMBOOL				bDoRecMatch;
	FLMBOOL				bNeedToGetFullRec;
	FLMBOOL				bSaveDoRecMatch;
	FLMBOOL				bSaveNeedToGetFullRec;
	FLMUINT				uiCPUReleaseCnt = 0;
	FLMBOOL				bSavedInvisTrans;
	FLMUINT				uiStartTime = FLM_GET_TIMER();
	FLMUINT				uiCurrTime;
	FLMUINT				uiTimeLimit = pCursor->uiTimeLimit;

	if (pCursor->fnStatus)
	{
		uiCBTimer = FLM_SECS_TO_TIMER_UNITS( STATUS_CB_INTERVAL);
	}

	// Set up initial search parameters.  

	bSaveDoRecMatch = pSubQuery->OptInfo.bDoRecMatch;
	bDoKeyMatch = pSubQuery->OptInfo.bDoKeyMatch;
	bSaveNeedToGetFullRec = !bSaveDoRecMatch && bGettingRecord;
	pSubQuery->uiDrn = 0;

	// Loop to evaluate keys and records.

	for (;;)
	{
		bDoRecMatch = bSaveDoRecMatch;
		bNeedToGetFullRec = bSaveNeedToGetFullRec;

		//	Release the CPU periodically to prevent CPU hog

		if (((++uiCPUReleaseCnt) & 0x1F) == 0)
		{
			f_yieldCPU();
		}

		// See if we have timed out

		if (uiTimeLimit)
		{
			uiCurrTime = FLM_GET_TIMER();

			// Use greater than to compare because if the timeout was one
			// second, we want to be sure and give it at least one
			// full second.  We would rather give it an extra second
			// than shortchange it.
			
			if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) > uiTimeLimit)
			{
				rc = RC_SET( FERR_TIMEOUT);
				goto Exit;
			}
		}

		// Do the progress callback if enough time has elapsed.

		pSubQuery->SQStatus.uiProcessedCnt++;
		if (pCursor->fnStatus)
		{
			uiCurrCBTime = FLM_GET_TIMER();
			if (FLM_ELAPSED_TIME( uiCurrCBTime,
						pCursor->uiLastCBTime) > uiCBTimer)
			{
				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = (pCursor->fnStatus)( FLM_SUBQUERY_STATUS,
										(void *)&pSubQuery->SQStatus,
										(void *)FALSE,
										pCursor->StatusData);
				CB_EXIT( pDb, bSavedInvisTrans);

				if (RC_BAD( rc))
				{
					goto Exit;
				}
				pCursor->uiLastCBTime = FLM_GET_TIMER();
			}
		}

		// Get the next or previous key or reference

		if (bFirstRead)
		{

			// A value in uiCurrKeyMatch means we have not evaluated the
			// current key.

			pSubQuery->uiCurrKeyMatch = 0;
			rc = (RCODE)((bReadForward)
							 ? pFSIndexCursor->firstKey( pDb,
										&pSubQuery->pRec, &pSubQuery->uiDrn)
							 : pFSIndexCursor->lastKey( pDb,
										&pSubQuery->pRec, &pSubQuery->uiDrn));
			if (RC_OK( rc))
			{
				bFirstRead = FALSE;
			}
			pSubQuery->bFirstReference = TRUE;
		}
		else if (!pSubQuery->bFirstReference)
		{

			rc = (RCODE)((bReadForward)
							 ? pFSIndexCursor->nextRef( pDb,
									&pSubQuery->uiDrn)
							 : pFSIndexCursor->prevRef( pDb,
									&pSubQuery->uiDrn));
			if ((bReadForward && rc == FERR_EOF_HIT) ||
				 (!bReadForward && rc == FERR_BOF_HIT))
			{
				rc = FERR_OK;
				pSubQuery->bFirstReference = TRUE;
				goto Get_Key;
			}

			// If we don't have a current key, need to get it.
			// pSubQuery->pRec could be NULL if we returned the last
			// record that passed the criteria back to the caller.  In
			// that case we just need to get it from the index cursor.

			if (!pSubQuery->pRec || !pSubQuery->bRecIsAKey)
			{
				if (RC_BAD( rc = pFSIndexCursor->currentKey( pDb,
											&pSubQuery->pRec, &pSubQuery->uiDrn)))
				{
					goto Exit;
				}
				pSubQuery->bRecIsAKey = TRUE;

				// These should have been set by currentKey

				flmAssert( pSubQuery->pRec->getContainerID() ==
								pCursor->uiContainer);
				flmAssert( pSubQuery->pRec->getID() == pSubQuery->uiDrn);
			}
			else
			{

				// Container should have already been set.

				flmAssert( pSubQuery->pRec->getContainerID() ==
						pCursor->uiContainer);

				// Need to modify DRN to the newly retrieved DRN.

				pSubQuery->pRec->setID( pSubQuery->uiDrn);
			}
		}
		else
		{
Get_Key:

			// A value in uiCurrKeyMatch means we have not evaluated the
			// current key.

			pSubQuery->uiCurrKeyMatch = 0;
			rc = (RCODE)((bReadForward)
							 ? pFSIndexCursor->nextKey( pDb,
														&pSubQuery->pRec, &pSubQuery->uiDrn)
							 : pFSIndexCursor->prevKey( pDb,
														&pSubQuery->pRec, &pSubQuery->uiDrn));
		}
		if (RC_BAD( rc))
		{
			goto Exit;
		}
		if (pSubQuery->bFirstReference)
		{
			pSubQuery->SQStatus.uiKeysTraversed++;
			pSubQuery->bRecIsAKey = TRUE;
		}
		pSubQuery->SQStatus.uiRefsTraversed++;

		// Make sure the container of the key matches the container
		// of the query.  NOTE: flmCurEvalCriteria also does this, but
		// we need to do the test here in case it is not called for 
		// the key below.  In that case, we want to avoid retrieving
		// the record.  Furthermore, even if bHaveDrnFlds is TRUE,
		// there is no need to check each DRN of this key, because
		// it is guaranteed that none of them will match, so in this
		// case we simply want to skip the key entirely.

		if (pSubQuery->pRec->getContainerID() != pCursor->uiContainer)
		{
			pSubQuery->uiCurrKeyMatch = FLM_FALSE;

			// If bFirstReference is TRUE at this point, this is the
			// first reference of this key, so we need to increment
			// the uiNumKeysRejected counter as well as the
			// uiNumRefsRejected counter.

			if (pSubQuery->bFirstReference)
			{
				pSubQuery->SQStatus.uiKeysRejected++;
			}
			pSubQuery->SQStatus.uiRefsRejected++;

			// Set bFirstReference to TRUE so it will skip to the next
			// key.

			pSubQuery->bFirstReference = TRUE;
			continue;
		}

		// See if we need to evaluate the key against the criteria.

		if (bDoKeyMatch &&
			 (pSubQuery->bFirstReference || pSubQuery->bHaveDrnFlds))
		{

			if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pSubQuery->pRec, TRUE,
									&pSubQuery->uiCurrKeyMatch)))
			{
				if (rc == FERR_TRUNCATED_KEY)
				{
					rc = FERR_OK;
					pSubQuery->uiCurrKeyMatch = FLM_UNK;
				}
				else
				{
					goto Exit;
				}
			}
			flmAssert( pSubQuery->uiCurrKeyMatch != 0);

			if (pSubQuery->uiCurrKeyMatch == FLM_FALSE)
			{

				// If bFirstReference is TRUE at this point, this is the
				// first reference of this key, so we need to increment
				// the uiNumKeysRejected counter as well as the
				// uiNumRefsRejected counter.

				if (pSubQuery->bFirstReference)
				{
					pSubQuery->SQStatus.uiKeysRejected++;
				}
				pSubQuery->SQStatus.uiRefsRejected++;

				// If we must evaluate DRN fields, we need to go to the
				// next reference, so we set bFirstReference to FALSE.
				// Otherwise, we set it to TRUE so it will skip to the
				// next key.

				pSubQuery->bFirstReference = !pSubQuery->bHaveDrnFlds;

				// Continue loop to process next key or reference

				continue;
			}
		}

		// If we did a key match, see if it passed or was unknown.
		// If it passed, no need to do the record match.  If it
		// was unknown, must do record match.

		if (pSubQuery->uiCurrKeyMatch == FLM_TRUE)
		{
			bDoRecMatch = FALSE;
		}
		else if (pSubQuery->uiCurrKeyMatch == FLM_UNK)
		{
			bDoRecMatch = TRUE;
		}
		else
		{
			// NOTE: If uiCurrKeyMatch is FLM_FALSE, we should never get
			// to this point - it will continue up above.  Thus, at this
			// point, uiCurrKeyMatch should be zero, meaning that no
			// evaluation was done on the current key - which also means
			// that bDoKeyMatch better be FALSE at this point.

			// Even though bDoKeyMatch is FALSE, bDoRecMatch needs to be left
			// unchanged.  If bDoRecMatch is FALSE, we know from the query
			// criteria that all keys we read through will automatically
			// match.  If bDoRecMatch is TRUE, we have previously decided 
			// that we MUST NOT do a key match, but force a record match
			// instead.  This would be the case where the comparison involves
			// a substring that is not the first substring - you can't do a
			// key match in that case because it will come out FALSE.

			flmAssert( pSubQuery->uiCurrKeyMatch == 0 && !bDoKeyMatch);
		}

		// Set up to get next reference the next time around.

		pSubQuery->bFirstReference = FALSE;

		// Retrieve and evalute the data record if necessary.

		if (bDoRecMatch)
		{
			FLMUINT	uiRecMatch;

			// Because we are fetching the full record here, no need to below.

			bNeedToGetFullRec = FALSE;

			// Retrieve the record

			if (RC_OK( rc = flmCurRetrieveRec( pDb, pSubQuery,
										pCursor->uiContainer)))
			{
				pSubQuery->SQStatus.uiRecsFetchedForEval++;
			}
			else
			{
				if (rc == FERR_NOT_FOUND)
				{
					pSubQuery->SQStatus.uiRecsNotFound++;
					rc = RC_SET( FERR_NO_REC_FOR_KEY);
				}
				goto Exit;
			}

			// Evaluate the record against the query criteria

			if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pSubQuery->pRec, FALSE, &uiRecMatch)))
			{
				goto Exit;
			}

			if (uiRecMatch != FLM_TRUE)
			{
				pSubQuery->SQStatus.uiRecsRejected++;

				// Continue loop to process next key or reference

				continue;
			}
		}

		// If we must return full records, and we have not yet fetched
		// the record, retrieve the record now.

		if (bNeedToGetFullRec)
		{
			if (RC_OK( rc = flmCurRetrieveRec( pDb, pSubQuery,
									pCursor->uiContainer)))
			{
				pSubQuery->SQStatus.uiRecsFetchedForView++;
			}
			else
			{
				if (rc == FERR_NOT_FOUND)
				{
					pSubQuery->SQStatus.uiRecsNotFound++;
					rc = RC_SET( FERR_NO_REC_FOR_KEY);
				}
				goto Exit;
			}
		}

		// Record has passed all of the sub-query criteria.  See if
		// it passes our validation tests.

		if (RC_BAD( rc = flmCurRecValidate( eFlmFuncId, pCursor,
									pSubQuery, puiSkipCount, puiCount,
									&bReturnRecOK)))
		{
			goto Exit;
		}
		if (bReturnRecOK)
		{
			break;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc: Searches a user predicate for matching record.
****************************************************************************/
FSTATIC RCODE flmCurSearchPredicate(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount
	)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = pCursor->pDb;
	FlmUserPredicate *	pPredicate = pSubQuery->pPredicate;
	FLMBOOL					bSavedInvisTrans;
	FLMUINT					uiCBTimer = 0;
	FLMUINT					uiCurrCBTime;
	FLMBOOL					bReturnRecOK;
	FLMUINT					uiCPUReleaseCnt = 0;
	FLMUINT					uiStartTime = FLM_GET_TIMER();
	FLMUINT					uiCurrTime;
	FLMUINT					uiTimeLimit = pCursor->uiTimeLimit;
	FLMUINT					uiRecMatch;

	if (pCursor->fnStatus)
	{
		uiCBTimer = FLM_SECS_TO_TIMER_UNITS( STATUS_CB_INTERVAL);
	}

	// Set up initial search parameters.  

	pSubQuery->uiDrn = 0;

	// Loop to evaluate keys and records.

	for (;;)
	{
		//	Release the CPU periodically to prevent CPU hog

		if (((++uiCPUReleaseCnt) & 0x1F) == 0)
		{
			f_yieldCPU();
		}

		// See if we have timed out

		if (uiTimeLimit)
		{
			uiCurrTime = FLM_GET_TIMER();

			// Use greater than to compare because if the timeout was one
			// second, we want to be sure and give it at least one
			// full second.  We would rather give it an extra second
			// than shortchange it.
			
			if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) > uiTimeLimit)
			{
				rc = RC_SET( FERR_TIMEOUT);
				goto Exit;
			}
		}

		// Do the progress callback if enough time has elapsed.

		pSubQuery->SQStatus.uiProcessedCnt++;
		if (pCursor->fnStatus)
		{
			uiCurrCBTime = FLM_GET_TIMER();
			if (FLM_ELAPSED_TIME( uiCurrCBTime,
						pCursor->uiLastCBTime) > uiCBTimer)
			{
				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = (pCursor->fnStatus)( FLM_SUBQUERY_STATUS,
										(void *)&pSubQuery->SQStatus,
										(void *)FALSE,
										pCursor->StatusData);
				CB_EXIT( pDb, bSavedInvisTrans);

				if (RC_BAD( rc))
				{
					goto Exit;
				}
				pCursor->uiLastCBTime = FLM_GET_TIMER();
			}
		}

		// Can't go backwards with embedded predicates

		CB_ENTER( pDb, &bSavedInvisTrans);
		if (bFirstRead)
		{
			if (bReadForward)
			{
				rc = pPredicate->firstRecord( (HFDB)pDb,
										&pSubQuery->uiDrn, &pSubQuery->pRec);
			}
			else
			{
				rc = pPredicate->lastRecord( (HFDB)pDb,
										&pSubQuery->uiDrn, &pSubQuery->pRec);
			}
			if (RC_OK( rc))
			{
				bFirstRead = FALSE;
			}
		}
		else
		{
			if (bReadForward)
			{
				rc = pPredicate->nextRecord( (HFDB)pDb,
										&pSubQuery->uiDrn, &pSubQuery->pRec);
			}
			else
			{
				rc = pPredicate->prevRecord( (HFDB)pDb,
										&pSubQuery->uiDrn, &pSubQuery->pRec);
			}
		}
		CB_EXIT( pDb, bSavedInvisTrans);
		if (RC_BAD( rc))
		{
			goto Exit;
		}
		pSubQuery->SQStatus.uiRecsFetchedForEval++;
		pSubQuery->bRecIsAKey = FALSE;

		// Evaluate the record against the query criteria

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
								pSubQuery->pRec, FALSE, &uiRecMatch)))
		{
			goto Exit;
		}

		if (uiRecMatch != FLM_TRUE)
		{
			pSubQuery->SQStatus.uiRecsRejected++;

			// Continue loop to process next key or reference

			continue;
		}

		// Record has passed all of the sub-query criteria.  See if
		// it passes our validation tests.

		if (RC_BAD( rc = flmCurRecValidate( eFlmFuncId, pCursor,
									pSubQuery, puiSkipCount, puiCount,
									&bReturnRecOK)))
		{
			goto Exit;
		}
		if (bReturnRecOK)
		{
			break;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc: Searches a container for matching record.
****************************************************************************/
FSTATIC RCODE flmCurSearchContainer(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount
	)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = pCursor->pDb;
	FSDataCursor *		pFSDataCursor = pSubQuery->pFSDataCursor;
	FLMUINT				uiCBTimer = 0;
	FLMUINT				uiCurrCBTime;
	FLMBOOL				bReturnRecOK;
	FLMUINT				uiCPUReleaseCnt = 0;
	FLMBOOL				bSavedInvisTrans;
	FLMUINT				uiStartTime = FLM_GET_TIMER();
	FLMUINT				uiCurrTime;
	FLMUINT				uiTimeLimit = pCursor->uiTimeLimit;
	FLMUINT				uiRecMatch;

	if (pCursor->fnStatus)
	{
		uiCBTimer = FLM_SECS_TO_TIMER_UNITS( STATUS_CB_INTERVAL);
	}

	// Set up initial search parameters.  

	pSubQuery->uiDrn = 0;

	// Loop to evaluate keys and records.

	for (;;)
	{
		//	Release the CPU periodically to prevent CPU hog

		if (((++uiCPUReleaseCnt) & 0x1F) == 0)
		{
			f_yieldCPU();
		}

		// See if we have timed out

		if (uiTimeLimit)
		{
			uiCurrTime = FLM_GET_TIMER();

			// Use greater than to compare because if the timeout was one
			// second, we want to be sure and give it at least one
			// full second.  We would rather give it an extra second
			// than shortchange it.
			
			if (FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) > uiTimeLimit)
			{
				rc = RC_SET( FERR_TIMEOUT);
				goto Exit;
			}
		}

		// Do the progress callback if enough time has elapsed.

		pSubQuery->SQStatus.uiProcessedCnt++;
		if (pCursor->fnStatus)
		{
			uiCurrCBTime = FLM_GET_TIMER();
			if (FLM_ELAPSED_TIME( uiCurrCBTime,
						pCursor->uiLastCBTime) > uiCBTimer)
			{
				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = (pCursor->fnStatus)( FLM_SUBQUERY_STATUS,
										(void *)&pSubQuery->SQStatus,
										(void *)FALSE,
										pCursor->StatusData);
				CB_EXIT( pDb, bSavedInvisTrans);

				if (RC_BAD( rc))
				{
					goto Exit;
				}
				pCursor->uiLastCBTime = FLM_GET_TIMER();
			}
		}

		// Get the next or previous key or reference

		if (bFirstRead)
		{
			rc = (RCODE)((bReadForward)
							 ? pFSDataCursor->firstRec( pDb,
										&pSubQuery->pRec, &pSubQuery->uiDrn)
							 : pFSDataCursor->lastRec( pDb,
										&pSubQuery->pRec, &pSubQuery->uiDrn));
			if (RC_OK( rc))
			{
				bFirstRead = FALSE;
			}
		}
		else
		{
			rc = (RCODE)((bReadForward)
							 ? pFSDataCursor->nextRec( pDb,
														&pSubQuery->pRec, &pSubQuery->uiDrn)
							 : pFSDataCursor->prevRec( pDb,
														&pSubQuery->pRec, &pSubQuery->uiDrn));
		}
		if (RC_BAD( rc))
		{
			goto Exit;
		}
		pSubQuery->SQStatus.uiRecsFetchedForEval++;
		pSubQuery->bRecIsAKey = FALSE;

		// Evaluate the record against the query criteria

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
								pSubQuery->pRec, FALSE, &uiRecMatch)))
		{
			goto Exit;
		}

		if (uiRecMatch != FLM_TRUE)
		{
			pSubQuery->SQStatus.uiRecsRejected++;

			// Continue loop to process next record

			continue;
		}

		// Record has passed all of the sub-query criteria.  See if
		// it passes our validation tests.

		if (RC_BAD( rc = flmCurRecValidate( eFlmFuncId, pCursor,
									pSubQuery, puiSkipCount, puiCount,
									&bReturnRecOK)))
		{
			goto Exit;
		}
		if (bReturnRecOK)
		{
			break;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc: Retrieves a single record and evaluates it.
****************************************************************************/
FSTATIC RCODE flmCurEvalSingleRec(
	eFlmFuncs	eFlmFuncId,
	CURSOR *		pCursor,
	FLMBOOL		bFirstRead,
	FLMBOOL		bReadForward,
	SUBQUERY *	pSubQuery,
	FLMUINT *	puiCount,
	FLMUINT *	puiSkipCount
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = pCursor->pDb;
	FLMBOOL		bReturnRecOK;
	FLMUINT		uiRecMatch;

	// Return EOF or BOF if we have already returned the
	// record and it is not a first() or last() call.

	if (!bFirstRead && pSubQuery->bRecReturned)
	{
		rc = (RCODE)((bReadForward)
						 ? RC_SET( FERR_EOF_HIT)
						 : RC_SET( FERR_BOF_HIT));
		goto Exit;
	}

	// Set up initial search parameters.  

	pSubQuery->uiDrn = pSubQuery->OptInfo.uiDrn;

	if (RC_BAD( rc = flmCurRetrieveRec( pDb, pSubQuery, pCursor->uiContainer)))
	{
		goto Exit;
	}
	pSubQuery->SQStatus.uiRecsFetchedForEval++;

	// Evaluate the record against the query criteria

	if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
								pSubQuery->pRec, FALSE, &uiRecMatch)))
	{
		goto Exit;
	}

	if (uiRecMatch != FLM_TRUE)
	{
		pSubQuery->SQStatus.uiRecsRejected++;
		rc = (RCODE)((bReadForward)
						 ? (RCODE)FERR_EOF_HIT
						 : (RCODE)FERR_BOF_HIT);
		goto Exit;
	}

	if (RC_BAD( rc = flmCurRecValidate( eFlmFuncId, pCursor,
								pSubQuery, puiSkipCount, puiCount,
								&bReturnRecOK)))
	{
		goto Exit;
	}
	if (!bReturnRecOK)
	{
		rc = (RCODE)((bReadForward)
						 ? (RCODE)FERR_EOF_HIT
						 : (RCODE)FERR_BOF_HIT);
		goto Exit;
	}

	// Found a record, set bRecReturned to TRUE so that the
	// next time we come in on a next() or prev() call, we will
	// return EOF or BOF.

	pSubQuery->bRecReturned = TRUE;
Exit:
	return( rc);
}

/****************************************************************************
Desc: Performs the search operation.
****************************************************************************/
RCODE flmCurSearch(
	eFlmFuncs		eFlmFuncId,
	CURSOR *			pCursor,
	FLMBOOL			bFirstRead,
	FLMBOOL			bReadForward,
	FLMUINT *		puiCount,
	FLMUINT *		puiSkipCount,
	FlmRecord **	ppUserRecord,
	FLMUINT *		puiDrn
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb;
	DB_STATS *	pDbStats;
	RCODE			TmpRc;
	SUBQUERY *	pSubQuery = pCursor->pCurrSubQuery;
	FLMBOOL		bSavedInvisTrans;

	pDb = pCursor->pDb;
	if( RC_BAD( rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	if ((pDbStats = pDb->pDbStats) != NULL)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->ui64NumCursorReads++;
	}

	// If this is a first or last call, we need to reset the
	// current sub-query.

	if (bFirstRead)
	{
		pSubQuery = pCursor->pSubQueryList;

		// If reading backwards, position to the last subquery
		// in the list.

		if (!bReadForward)
		{
			while (pSubQuery->pNext)
			{
				pSubQuery = pSubQuery->pNext;
			}
		}
		if (RC_BAD( rc = flmCurSetSubQuery( pCursor, pSubQuery)))
		{
			goto Exit;
		}
	}

	// If counting, initialize the count to zero.

	if (puiCount)
	{
		*puiCount = 0;
	}

	// Loop through sub-queries

	for (;;)
	{
		switch (pSubQuery->OptInfo.eOptType)
		{
			case QOPT_USING_INDEX:
				rc = flmCurSearchIndex( eFlmFuncId,
							pCursor, bFirstRead, bReadForward,
							pSubQuery, puiCount, puiSkipCount,
							(FLMBOOL)((ppUserRecord && !pCursor->bOkToReturnKeys)
										 ? (FLMBOOL)TRUE
										 : (FLMBOOL)FALSE));
				break;
			case QOPT_USING_PREDICATE:
				rc = flmCurSearchPredicate( eFlmFuncId,
							pCursor, bFirstRead, bReadForward, pSubQuery,
							puiCount, puiSkipCount);
				break;
			case QOPT_SINGLE_RECORD_READ:
				rc = flmCurEvalSingleRec( eFlmFuncId,
							pCursor, bFirstRead, bReadForward, pSubQuery,
							puiCount, puiSkipCount);
				break;
			case QOPT_PARTIAL_CONTAINER_SCAN:
			case QOPT_FULL_CONTAINER_SCAN:
				rc = flmCurSearchContainer( eFlmFuncId,
							pCursor, bFirstRead, bReadForward,
							pSubQuery, puiCount, puiSkipCount);
				break;
			default:

				// Should never happen

				flmAssert( 0);
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
		}

		// If rc is FERR_OK, it means we got something that passed.

		if (RC_OK( rc))
		{

			// ppUserRecord will be NULL if this is a DRN only function or
			// the record count function.

			if (ppUserRecord)
			{
				flmAssert( pSubQuery->pRec != NULL);
				*ppUserRecord = pSubQuery->pRec;
				(*ppUserRecord)->AddRef();

				// We must release the record here since we are giving it
				// back to the caller.

				pSubQuery->pRec->Release();
				pSubQuery->pRec = NULL;
			}
			if (puiDrn)
			{
				*puiDrn = pSubQuery->uiDrn;
			}
			break;
		}

		// This is the place where we handle EOF and BOF.  We attempt to go to
		// the next or previous sub-query, if there is one.  If the error is not
		// EOF or BOF, we simply go to Exit.

		if (rc != (RCODE)((bReadForward)
								? (RCODE)FERR_EOF_HIT
								: (RCODE)FERR_BOF_HIT))
		{
			goto Exit;
		}

		// If there is a callback, call it to give the final status of
		// this sub-query.

		if (pCursor->fnStatus)
		{
			CB_ENTER( pDb, &bSavedInvisTrans);
			TmpRc = (pCursor->fnStatus)( FLM_SUBQUERY_STATUS,
									(void *)&pSubQuery->SQStatus,
									(void *)TRUE,
									pCursor->StatusData);
			CB_EXIT( pDb, bSavedInvisTrans);

			if (RC_BAD( TmpRc))
			{
				rc = TmpRc;
				goto Exit;
			}
			pCursor->uiLastCBTime = FLM_GET_TIMER();
		}

		// Reached BOF/EOF in the current subquery. Move on to the next one.

		pSubQuery = (SUBQUERY *)((bReadForward)
										 ? pSubQuery->pNext
										 : pSubQuery->pPrev);
		if (!pSubQuery)
		{

			// No more sub-queries.  If we were counting and have a non-zero
			// count, we can return FERR_OK.

			if (puiCount && *puiCount)
			{
				rc = FERR_OK;
			}
			//else rc should already be EOF or BOF.

			// If we are at the real EOF or BOF, and doing a move
			// relative, we must decrement the skip counter by one
			// more.  That way, if the skip count is set to one and
			// we hit EOF, the value returned to the user will be
			// one.

			if (puiSkipCount)
			{
				(*puiSkipCount)--;
			}
			goto Exit;
		}

		// Set up up to use the next or previous sub-query.

		if (RC_BAD( rc = flmCurSetSubQuery( pCursor, pSubQuery)))
		{
			goto Exit;
		}

		// For the next (or previous) sub-query, need to set the
		// bFirstRead flag to TRUE the first time in.

		bFirstRead = TRUE;
	}

Exit:

	if (pDb)
	{
		flmExit( eFlmFuncId, pDb, rc);
	}
	
	return( rc);
}
