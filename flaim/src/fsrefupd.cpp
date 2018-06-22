//-------------------------------------------------------------------------
// Desc:	Index reference updating.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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

extern FLMBYTE SENLenArray[];

#define INSERT_REF		0
#define DELETE_REF		1
#define SPLIT_90_10		0
#define SPLIT_50_50		1

FSTATIC RCODE FSUpdateIxCounts(
	FDB *				pDb,
	IXD *				pIxd,
	FLMBYTE			byFlags,
	FLMBOOL			bSingleRef);

FSTATIC RCODE FSOutputIxCounts(
	FDB *				pDb,
	IX_STATS *		pIxStats);

FSTATIC RCODE FSRefCreateRec(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack);

FSTATIC RCODE FSRefInsert(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack);

FSTATIC RCODE FSRefDelete(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack,
	FLMBOOL *		pbSingleRef);

/****************************************************************************
Desc:	Update (add or delete) a single reference
****************************************************************************/
RCODE FSRefUpdate(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry)
{
	RCODE				rc = FERR_OK;
	BTSK				stackBuf[BH_MAX_LEVELS];
	BTSK *			pStack = stackBuf;
	FLMUINT			uiDinDomain = DIN_DOMAIN( pKrefEntry->uiDrn) + 1;
	FLMBYTE			byFlags = (FLMBYTE) pKrefEntry->uiFlags;
	FLMBOOL			bSingleRef = FALSE;
	FLMBOOL			bAddReference = (byFlags & KREF_DELETE_FLAG) ? FALSE : TRUE;
	FLMBYTE			pKeyBuf[MAX_KEY_SIZ];

FSRU_try_again:

	if (pKrefEntry->uiFlags & KREF_ENCRYPTED_KEY)
	{

		// Can't allow updates whit these keys.

		flmAssert( pDb->pFile->bInLimitedMode);
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	FSInitStackCache( &stackBuf[0], BH_MAX_LEVELS);
	pStack = stackBuf;
	pStack->pKeyBuf = pKeyBuf;

	if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, (FLMBYTE*) &pKrefEntry[1],
				  pKrefEntry->ui16KeyLen, uiDinDomain)))
	{
		goto Exit;
	}

	// If pStack->bsStatus == REC_NOT_FOUND create a new element if found
	// then add the reference into the found element.

	if (pStack->uiCmpStatus == BT_EQ_KEY)
	{
		if ((byFlags & KREF_UNIQUE_KEY) && !(byFlags & KREF_DELETE_FLAG))
		{
			rc = RC_SET( FERR_NOT_UNIQUE);
			goto Exit;
		}

		if (pLFile->pIxd->uiFlags & IXD_POSITIONING)
		{
			if (RC_BAD( rc = FSChangeCount( pDb, pStack, bAddReference)))
			{
				goto Exit;
			}
		}

		bSingleRef = FALSE;

		if (bAddReference)
		{
			if (RC_BAD( rc = FSRefInsert( pDb, pLFile, pKrefEntry, pStack)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = FSRefDelete( pDb, pLFile, pKrefEntry, pStack,
						  &bSingleRef)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (!bAddReference)
		{

			// Already been deleted, ignore the error condition and go on.

			flmAssert( 0);
			rc = FERR_OK;
			goto Exit;
		}

		// The B-Tree may be empty

		if (pLFile->uiRootBlk == BT_END)
		{
			if (RC_BAD( rc = flmLFileInit( pDb, pLFile)))
			{
				goto Exit;
			}

			FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
			goto FSRU_try_again;
		}

		if (pLFile->pIxd->uiFlags & IXD_POSITIONING)
		{
			if (RC_BAD( rc = FSChangeCount( pDb, pStack, TRUE)))
			{
				goto Exit;
			}
		}

		// Add new key|reference element.

		bSingleRef = TRUE;
		if (RC_BAD( rc = FSRefCreateRec( pDb, pLFile, pKrefEntry, pStack)))
		{
			goto Exit;
		}
	}

Exit:

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	if (RC_OK( rc) && (pLFile->pIxd->uiFlags & IXD_COUNT))
	{
		rc = FSUpdateIxCounts( pDb, pLFile->pIxd, byFlags, bSingleRef);
	}

	return (rc);
}

/****************************************************************************
Desc:	Update the index reference and/or key counts in memory only. Counts
		are not written to the database until transaction commit time.
****************************************************************************/
FSTATIC RCODE FSUpdateIxCounts(
	FDB *			pDb,
	IXD *			pIxd,
	FLMBYTE		byFlags,
	FLMBOOL		bSingleRef)
{
	RCODE			rc = FERR_OK;
	IX_STATS *	pIxStat;

	// See if we have already created an IX_STATS structure in memory

	pIxStat = pDb->pIxStats;
	while (pIxStat && pIxStat->uiIndexNum != pIxd->uiIndexNum)
	{
		pIxStat = pIxStat->pNext;
	}

	// Allocate an IX_STATS if we didn't find one.

	if (!pIxStat)
	{
		if (RC_BAD( rc = f_calloc( sizeof(IX_STATS), &pIxStat)))
		{
			goto Exit;
		}

		pIxStat->uiIndexNum = pIxd->uiIndexNum;
		pIxStat->pNext = pDb->pIxStats;
		pDb->pIxStats = pIxStat;
	}

	// Determine whether to increment or decrement the counts

	if (byFlags & KREF_DELETE_FLAG)
	{
		if (bSingleRef)
		{
			pIxStat->iDeltaKeys--;
		}

		pIxStat->iDeltaRefs--;
	}
	else
	{
		if (bSingleRef)
		{
			pIxStat->iDeltaKeys++;
		}

		pIxStat->iDeltaRefs++;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Output the index reference and/or key counts by writing them to the
		tracker record.
****************************************************************************/
FSTATIC RCODE FSOutputIxCounts(
	FDB *			pDb,
	IX_STATS *	pIxStats)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bCreateRec;
	FlmRecord *		pRecord = NULL;
	FlmRecord *		pTmpRec = NULL;
	void *			pvField;
	FLMUINT			uiCount;
	LFILE *			pLFile;

	// If no counts changed, do nothing

	if (!pIxStats->iDeltaKeys && !pIxStats->iDeltaRefs)
	{
		goto Exit;
	}

	if (RC_BAD( rc = fdictGetContainer( pDb->pDict, FLM_TRACKER_CONTAINER,
				  &pLFile)))
	{
		goto Exit;
	}

	// Retrieve the tracker record from record cache.

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, FLM_TRACKER_CONTAINER,
				  pIxStats->uiIndexNum, TRUE, NULL, NULL, &pRecord)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}

		bCreateRec = TRUE;
		rc = FERR_OK;
	}
	else
	{
		bCreateRec = FALSE;
	}

	// If there was no record, create one. Otherwise, copy the record

	if (bCreateRec)
	{
		if ((pTmpRec = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		// Create at least the root node.

		if (RC_BAD( rc = pTmpRec->insertLast( 0, FLM_INDEX_TAG, FLM_CONTEXT_TYPE,
					  &pvField)))
		{
			goto Exit;
		}
	}
	else
	{
		if ((pTmpRec = pRecord->copy()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	// Update the key count.

	if (pIxStats->iDeltaKeys)
	{
		if ((pvField = pTmpRec->find( pTmpRec->root(), FLM_KEY_TAG)) == NULL)
		{

			// Cannot make key count negative.

			if (pIxStats->iDeltaKeys < 0)
			{
				flmAssert( 0);
				pIxStats->iDeltaKeys = 0;
			}

			if (RC_BAD( rc = pTmpRec->insert( pTmpRec->root(), INSERT_LAST_CHILD,
						  FLM_KEY_TAG, FLM_NUMBER_TYPE, &pvField)))
			{
				goto Exit;
			}

			uiCount = (FLMUINT) pIxStats->iDeltaKeys;
		}
		else
		{
			if (RC_BAD( rc = pTmpRec->getUINT( pvField, &uiCount)))
			{
				goto Exit;
			}

			if (pIxStats->iDeltaKeys < 0)
			{
				pIxStats->iDeltaKeys = -pIxStats->iDeltaKeys;

				if ((FLMUINT) pIxStats->iDeltaKeys <= uiCount)
				{
					uiCount -= (FLMUINT) pIxStats->iDeltaKeys;
				}
				else
				{

					// Key count cannot go negative.

					flmAssert( 0);
					uiCount = 0;
				}
			}
			else
			{
				uiCount += (FLMUINT) pIxStats->iDeltaKeys;
			}
		}

		if (RC_BAD( rc = pTmpRec->setUINT( pvField, uiCount)))
		{
			goto Exit;
		}
	}

	// Update the references count

	if (pIxStats->iDeltaRefs)
	{
		if ((pvField = pTmpRec->find( pTmpRec->root(), FLM_REFS_TAG)) == NULL)
		{

			// Cannot make reference count negative.

			if (pIxStats->iDeltaRefs < 0)
			{
				flmAssert( 0);
				pIxStats->iDeltaRefs = 0;
			}

			if (RC_BAD( rc = pTmpRec->insert( pTmpRec->root(), INSERT_LAST_CHILD,
						  FLM_REFS_TAG, FLM_NUMBER_TYPE, &pvField)))
			{
				goto Exit;
			}

			uiCount = (FLMUINT) pIxStats->iDeltaRefs;
		}
		else
		{
			if (RC_BAD( rc = pTmpRec->getUINT( pvField, &uiCount)))
			{
				goto Exit;
			}

			if (pIxStats->iDeltaRefs < 0)
			{
				pIxStats->iDeltaRefs = -pIxStats->iDeltaRefs;

				if ((FLMUINT) pIxStats->iDeltaRefs <= uiCount)
				{
					uiCount -= (FLMUINT) pIxStats->iDeltaRefs;
				}
				else
				{

					// Reference count cannot go negative.

					flmAssert( 0);
					uiCount = 0;
				}
			}
			else
			{
				uiCount += (FLMUINT) pIxStats->iDeltaRefs;
			}
		}

		if (RC_BAD( rc = pTmpRec->setUINT( pvField, uiCount)))
		{
			goto Exit;
		}
	}

	// Update or add the record.

	pTmpRec->setID( pIxStats->uiIndexNum);
	pTmpRec->setContainerID( FLM_TRACKER_CONTAINER);

	if (bCreateRec)
	{
		if (RC_BAD( rc = FSRecUpdate( pDb, pLFile, pTmpRec, pIxStats->uiIndexNum,
					  REC_UPD_ADD)))
		{
			goto Exit;
		}

		// Put the record into record cache.

		if (RC_BAD( rc = flmRcaInsertRec( pDb, pLFile,
					  pIxStats->uiIndexNum, pTmpRec)))
		{

			// Remove the record that was added.

			(void) FSRecUpdate( pDb, pLFile, NULL, pIxStats->uiIndexNum,
									 REC_UPD_DELETE);
			goto Exit;
		}
	}
	else
	{

		// Modify the record.

		if (RC_BAD( rc = FSRecUpdate( pDb, pLFile, pTmpRec, pIxStats->uiIndexNum,
					  REC_UPD_MODIFY)))
		{
			goto Exit;
		}

		// Put the modified record into record cache.

		if (RC_BAD( rc = flmRcaInsertRec( pDb, pLFile,
					  pIxStats->uiIndexNum, pTmpRec)))
		{

			// Undo the record that was modified - replace with original
			// record.

			(void) FSRecUpdate( pDb, pLFile, pRecord, pIxStats->uiIndexNum,
									 REC_UPD_MODIFY);
			goto Exit;
		}
	}

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}

	if (pTmpRec)
	{
		pTmpRec->Release();
	}

	return (rc);
}

/****************************************************************************
Desc:	Free the index stats structures for an FDB.
****************************************************************************/
void FSFreeIxCounts(
	FDB *			pDb)
{
	IX_STATS *	pNextIxStat;

	while (pDb->pIxStats)
	{
		pNextIxStat = pDb->pIxStats->pNext;
		f_free( &pDb->pIxStats);
		pDb->pIxStats = pNextIxStat;
	}
}

/****************************************************************************
Desc: Commit the index reference and/or key counts for a database. This
		routine is only called at commit time.
****************************************************************************/
RCODE FSCommitIxCounts(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	IX_STATS *	pNextIxStat;

	while (pDb->pIxStats)
	{
		pNextIxStat = pDb->pIxStats->pNext;
		if (RC_BAD( rc = FSOutputIxCounts( pDb, pDb->pIxStats)))
		{
			goto Exit;
		}

		f_free( &pDb->pIxStats);
		pDb->pIxStats = pNextIxStat;
	}

Exit:

	if (RC_BAD( rc))
	{
		FSFreeIxCounts( pDb);
	}

	return (rc);
}

/****************************************************************************
Desc:	Create a new record and add the reference to it.
Note:	The record size is the size of the only SEN value.  There is no 
		overhead for references for the entry compression
****************************************************************************/
FSTATIC RCODE FSRefCreateRec(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiElmSize;
	FLMUINT			uiKeyLen = pKrefEntry->ui16KeyLen;
	FLMUINT			uiRecLen;
	FLMBYTE *		pSen;
	FLMBYTE			byElmBuf[MAX_KEY_SIZ + BBE_KEY + SEN_MAX_SIZ];

	// Create the element overhead

	byElmBuf[BBE_PKC] = BBE_FIRST_FLAG | BBE_LAST_FLAG;
	BBE_SET_KL( byElmBuf, uiKeyLen);

	// Copy in the key

	f_memcpy( &byElmBuf[BBE_KEY], (FLMBYTE*) &pKrefEntry[1], uiKeyLen);
	uiElmSize = (BBE_KEY + uiKeyLen);

	pSen = &byElmBuf[uiElmSize];
	uiRecLen = SENPutNextVal( &pSen, pKrefEntry->uiDrn);
	uiElmSize += BBE_SET_RL( byElmBuf, uiRecLen);

	// Save the element

	if (RC_BAD( rc = FSBtInsert( pDb, pLFile, &pStack, byElmBuf, uiElmSize)))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Insert a reference into a key|reference list
Note:	The algorithm positions for an insert and inserts the reference while
		altering the block.  If a reference split is required refSplit() is
		called and FSRefInsert() could be recursively called.
****************************************************************************/
FSTATIC RCODE FSRefInsert(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack)
{
	RCODE				rc;
	FLMBYTE *		pElement = CURRENT_ELM( pStack);
	FLMUINT			uiElmLen = (FLMUINT) BBE_GET_RL( pElement);
	FLMUINT			uiElmSize;
	FLMBYTE			byElmBuf[MAX_KEY_SIZ + BBE_KEY + REF_SET_MAX_SIZ];

	// Build the element in byElmBuf[]

	FSSetElmOvhd( byElmBuf, BBE_KEY, 0, pStack->uiKeyLen, pElement);
	f_memcpy( &byElmBuf[BBE_KEY], pStack->pKeyBuf, pStack->uiKeyLen);

	if (uiElmLen > REF_SET_MAX_SIZ)
	{

		// May or may not split the reference set, but will always insert DRN

		rc = FSRefSplit( pDb, pLFile, &pStack, byElmBuf, pKrefEntry->uiDrn,
							 INSERT_REF, (FLMBYTE) (BBE_IS_FIRST( pElement) 
							 									? SPLIT_50_50 
																: SPLIT_90_10));
	}
	else
	{
		uiElmSize = BBE_KEY + pStack->uiKeyLen;

		if (FSSetInsertRef( &byElmBuf[uiElmSize], BBE_REC_PTR( pElement),
								 pKrefEntry->uiDrn, &uiElmLen))
		{
			rc = RC_SET( FERR_BTREE_ERROR);	// Entry is already there
			goto Exit;
		}

		BBE_SET_RL( byElmBuf, uiElmLen);
		rc = FSBtReplace( pDb, pLFile, &pStack, byElmBuf, uiElmSize + uiElmLen);
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Insert a single reference into a reference set. This is the code
		that supports the new DIN (Dual Integer Numbers) format for reference
		set compression.
****************************************************************************/
RCODE FSSetInsertRef(
	FLMBYTE *		pDestRef,
	FLMBYTE *		pSrcRef,
	FLMUINT			drn,
	FLMUINT *		puiSetLength)
{
	DIN_STATE		destState;
	DIN_STATE		srcState;
	FLMUINT			uiSetLength = *puiSetLength;
	FLMUINT			uiLastDrn;
	FLMUINT			uiOldDelta;
	FLMUINT			uiDelta;
	FLMUINT			uiNewDelta;
	FLMUINT			uiOneRun;
	FLMUINT			uiPrevOneRun;
	FLMUINT			uiLastSrcOfs;
	FLMUINT			uiPrevLastSrcOfs = 0;
	FLMUINT			uiMoveLen;
	FLMBYTE			byValue;

	// Initialization Section

	uiPrevOneRun = uiOldDelta = 0;
	uiLastSrcOfs = 0;
	
	RESET_DINSTATE( destState);
	RESET_DINSTATE( srcState);

	// Take care of the domain value.

	if (*pSrcRef == SEN_DOMAIN)
	{
		srcState.uiOffset = 1;
		DINNextVal( pSrcRef, &srcState);
		uiLastSrcOfs = srcState.uiOffset;
	}

	uiLastDrn = DINNextVal( pSrcRef, &srcState);

	if (drn > uiLastDrn)
	{

		// ADD TO THE FRONT. Set uiDelta and uiNewDelta

		uiNewDelta = (uiDelta = drn) - uiLastDrn;
		goto FSSIR_add_delta;
	}

	uiOldDelta = uiLastDrn;
	uiOneRun = 0;

	// Search through the set finding where the "drn" fits in.

	while (drn < uiLastDrn)
	{

		// Save previous last source offset & previous one runs

		uiPrevLastSrcOfs = uiLastSrcOfs;
		uiLastSrcOfs = srcState.uiOffset;
		uiPrevOneRun = uiOneRun;
		uiOneRun = 0;

		if (srcState.uiOffset >= uiSetLength)
		{

			// APPEND TO THE END

			uiDelta = uiLastDrn - drn;
			uiNewDelta = 0;
			goto FSSIR_add_delta;
		}

		// Check for a run of ONE's

		byValue = pSrcRef[srcState.uiOffset];
		if (DIN_IS_ONE_RUN( byValue))
		{

			// Read the number of one runs

			uiOneRun = DINOneRunVal( pSrcRef, &srcState);
			uiLastDrn -= uiOneRun;
			if (drn > uiLastDrn)
			{
				uiLastDrn = drn;
			}
		}
		else
		{
			uiOldDelta = DINNextVal( pSrcRef, &srcState);
			uiLastDrn -= uiOldDelta;
		}
	}

	// Check for duplicates

	if (drn == uiLastDrn)
	{
		// Duplicate found or inside of one run
		
		return (RC_SET( FERR_BTREE_ERROR));
	}

	uiDelta = uiLastDrn + uiOldDelta - drn;
	uiNewDelta = drn - uiLastDrn;

FSSIR_add_delta:

	// COMPRESS MULTIPLE ONE RUNS Special case if delta is 1. Check the
	// previous destinaion DIN for a 1 or a run of ones and combine to the
	// left. Also check to combine a one run on the right by jumping within
	// add_new_delta.

	if (uiDelta == 1)
	{
		uiOneRun = 1;
		if (uiPrevOneRun)
		{
			f_memcpy( pDestRef, pSrcRef, destState.uiOffset = uiPrevLastSrcOfs);
			uiOneRun += uiPrevOneRun;
		}
		else
		{
			f_memcpy( pDestRef, pSrcRef, destState.uiOffset = uiLastSrcOfs);
		}

		if (uiNewDelta == 1)
		{
			uiOneRun++;
			goto FSSIR_combine_right_one_runs;
		}

		// uiNewDelta is not 1 so write to destination and go on

		DINPutOneRunVal( pDestRef, &destState, uiOneRun);
	}
	else
	{
		f_memcpy( pDestRef, pSrcRef, destState.uiOffset = uiLastSrcOfs);

		// No one runs found so write the delta value

		DINPutNextVal( pDestRef, &destState, uiDelta);
	}

	// Add new delta value (uiOldDelta - uiDelta).  If the new delta value is
	// 1 then check next DIN for a 1 run and combine with the uiNewDelta
	// run. uiNewDelta only has a value when NOT appending to the end.
	
	if (uiNewDelta)
	{
		if (uiNewDelta != 1)
		{
			DINPutNextVal( pDestRef, &destState, uiNewDelta);
		}
		else
		{
			uiOneRun = 1;

FSSIR_combine_right_one_runs:

			if (srcState.uiOffset < uiSetLength)
			{

				// CONNECT ONE RUNS Check on the right side of the source for
				// another 1 run.

				byValue = pSrcRef[srcState.uiOffset];
				if (DIN_IS_ONE_RUN( byValue))
				{
					uiOneRun += DINOneRunVal( pSrcRef, &srcState);
				}
			}

			DINPutOneRunVal( pDestRef, &destState, uiOneRun);
		}
	}

	// FSSIR_move_rest_of_dins:

	if ((uiMoveLen = (FLMUINT) (uiSetLength - srcState.uiOffset)) != 0)
	{
		f_memcpy( &pDestRef[destState.uiOffset], &pSrcRef[srcState.uiOffset],
					uiMoveLen);
		destState.uiOffset += uiMoveLen;
	}

	*puiSetLength = destState.uiOffset;
	return (FERR_OK);
}

/****************************************************************************
Desc:	Delete a matching reference into a key|reference list.  If this was
		the last continuation element then remove the SEN_DOMAIN value from the
		previous element.
****************************************************************************/
FSTATIC RCODE FSRefDelete(
	FDB *				pDb,
	LFILE *			pLFile,
	KREF_ENTRY *	pKrefEntry,
	BTSK *			pStack,
	FLMBOOL *		pbSingleRef)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pElement = CURRENT_ELM( pStack);
	FLMBYTE *		pReference;
	FLMBYTE *		pDestRefPtr;
	FLMUINT			uiElmLen = BBE_GET_RL( pElement);
	FLMUINT			uiElmSize;
	FLMUINT			uiSetLength;
	FLMUINT			uiKeyLen;
	FLMUINT			uiSenLen;
	FLMBYTE			byElmBuf[MAX_KEY_SIZ + BBE_KEY + REF_SET_MAX_SIZ];

	// Build the element in byElmBuf[]

	FSSetElmOvhd( byElmBuf, BBE_KEY, 0, pStack->uiKeyLen, pElement);

	f_memcpy( &byElmBuf[BBE_KEY], pStack->pKeyBuf, pStack->uiKeyLen);
	pDestRefPtr = &byElmBuf[BBE_KEY + pStack->uiKeyLen];
	uiElmSize = (BBE_KEY + pStack->uiKeyLen);

	if (uiElmLen > REF_SET_MAX_SIZ)
	{

		// Straight deletions MAY overflow the record portion

		rc = FSRefSplit( pDb, pLFile, &pStack, byElmBuf, pKrefEntry->uiDrn,
							 DELETE_REF, SPLIT_50_50);
		return (rc);
	}

	pReference = BBE_REC_PTR( pElement);
	uiSetLength = uiElmLen;

	if (FSSetDeleteRef( pDestRefPtr, pReference, pKrefEntry->uiDrn, &uiSetLength))
	{
		goto Exit;
	}

	BBE_SET_RL( byElmBuf, uiSetLength);
	uiElmSize += uiSetLength;

	// Get the reference length and hasDomainFlag

	if (uiSetLength)
	{
		if (*pReference == SEN_DOMAIN)
		{
			uiSetLength--;
			uiSetLength -= SENValLen( pReference + 1);
		}
	}

	if (uiSetLength == 0)
	{
		FLMBOOL	bLastElm = BBE_IS_LAST( pElement);
		FLMBOOL	bFirstElm = BBE_IS_FIRST( pElement);

		// Delete the element

		if (RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
		{
			return (rc);
		}

		// Remove the LAST (only) reference in an element. There are 4 cases
		// to consider that deal with continuation elements:
		//		1. ONLY element - no list - ALL DONE 
		//		2. FIRST of many in a list - Set BBE_FIRST_FLAG in the
		//			next element 
		//		3. MIDDLE of a list - ALL DONE
		// 	4. LAST in a list Remove the domain from the previous element and
		// 		set the BBE_LAST_FLAG in the new last element.

		if (bFirstElm && bLastElm)					// Case 1 - ONLY element
		{
			*pbSingleRef = TRUE;
		}
		else if (bFirstElm && (!bLastElm))		// Case 2 - first of many
		{

			// Log the block before modifying it.

			if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				return (rc);
			}

			pElement = CURRENT_ELM( pStack);
			BBE_SET_FIRST( pElement);
		}
		else if (!bFirstElm && bLastElm)			// Case 4 - Last in a list
		{

			// Element was either the ONLY or LAST in a list.

			FSBtPrevElm( pDb, pLFile, pStack);	// Goto the previous element

			pElement = CURRENT_ELM( pStack);

			pReference = BBE_REC_PTR( pElement);
			uiSetLength = BBE_GET_RL( pElement);
			uiKeyLen = pStack->uiKeyLen;

			// Build the element in byElmBuf[] - sets BBE_FIRST_FLAG if set

			FSSetElmOvhd( byElmBuf, BBE_KEY, 0, uiKeyLen, pElement);
			BBE_SET_LAST( byElmBuf);				// Set last element flag

			f_memcpy( &byElmBuf[BBE_KEY], pStack->pKeyBuf, uiKeyLen);
			pDestRefPtr = &byElmBuf[BBE_KEY + uiKeyLen];

			// Parse past the element information and delete the domain data

			if (*pReference != SEN_DOMAIN)
			{
				return (RC_SET( FERR_BTREE_ERROR));
			}

			uiSenLen = (SENValLen( pReference + 1) + 1);
			pReference += uiSenLen;
			uiSetLength -= uiSenLen;
			f_memcpy( pDestRefPtr, pReference, uiSetLength);
			BBE_SET_RL( byElmBuf, uiSetLength);

			uiElmSize = (BBE_KEY + uiKeyLen + uiSetLength);

			// We could call FSBtReplace() except that if the element is the
			// last element in the block, the parent element will still
			// contain the 3 byte non-leaf DOMAIN number which should no
			// longer exist. In addition, the BBE_LAST_FLAG flag should be
			// set in the current element and replace doesn't do that.

			if (RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
			{
				return (rc);
			}

			// POSSIBLE CHECK YOU MAY FORGET... You could have deleted the
			// ONLY element in the tree. Check and if so init a new root
			// block and position for insert.

			if (pLFile->uiRootBlk == BT_END)
			{
				if (RC_BAD( rc = flmLFileInit( pDb, pLFile)))
				{
					return (rc);
				}

				// Call FSBtSearch to setup the pStack, would rather do a goto top;

				if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack,
							  (FLMBYTE*) &byElmBuf[BBE_KEY], uiKeyLen, 0)))
				{
					return (rc);
				}
			}
			else
			{

				if (RC_BAD( rc = FSBtScanTo( pStack, 
					&byElmBuf[BBE_KEY], uiKeyLen, 0)))
				{
					goto Exit;
				}
			}

			rc = FSBtInsert( pDb, pLFile, &pStack, byElmBuf, uiElmSize);
		}
	}
	else
	{

		// Replace the current element - wElmSize has had uiSetLength added
		// to it

		rc = FSBtReplace( pDb, pLFile, &pStack, byElmBuf, uiElmSize);
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Delete a single reference into a reference set. This is the code
		that supports the new DIN (Dual Integer Numbers) format for reference 
		set compression.
****************************************************************************/
RCODE FSSetDeleteRef(
	FLMBYTE *		pDestRef,
	FLMBYTE *		pSrcRef,
	FLMUINT			drn,
	FLMUINT *		puiSetLength)
{
	DIN_STATE		destState;
	DIN_STATE		srcState;
	FLMUINT			uiSetLength = *puiSetLength;
	FLMUINT			uiMoveLen;
	FLMUINT			uiLastSrcOfs;
	FLMUINT			uiLastDrn;
	FLMUINT			uiOldDelta;
	FLMUINT			uiOneRun;
	FLMUINT			uiTemp;
	FLMBYTE			byValue;

	RESET_DINSTATE( destState);
	RESET_DINSTATE( srcState);
	uiLastSrcOfs = 0;

	// Take care of the domain value

	if (*pSrcRef == SEN_DOMAIN)
	{
		srcState.uiOffset = 1;
		uiLastDrn = DINNextVal( pSrcRef, &srcState);
		uiLastSrcOfs = srcState.uiOffset;
	}

	uiLastDrn = DINNextVal( pSrcRef, &srcState);

	if (drn > uiLastDrn)
	{
		// Greater than the first reference
		
		return (RC_SET( FERR_KEY_NOT_FOUND));
	}

	if (drn == uiLastDrn)
	{

		// MATCHED THE FIRST REFERENCE Write the replacement starting DRN
		// and fix 1 run if there.

		if (uiLastSrcOfs)
		{
			// If domain - copy it
			
			f_memcpy( pDestRef, pSrcRef, destState.uiOffset = uiLastSrcOfs);
		}

		if (srcState.uiOffset >= uiSetLength)
		{
			// Only 1 reference
			
			goto FSSDR_done;
		}

		byValue = pSrcRef[srcState.uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{
			// Run of 2 or above
			
			DINPutNextVal( pDestRef, &destState, uiLastDrn - 1);
			DINPutOneRunVal( pDestRef, &destState,
								 DINOneRunVal( pSrcRef, &srcState) - 1);
		}
		else
		{
			uiLastDrn -= DINNextVal( pSrcRef, &srcState);
			DINPutNextVal( pDestRef, &destState, uiLastDrn);
		}

		goto FSSDR_move_rest_of_dins;
	}

	uiOldDelta = uiLastDrn;
	uiOneRun = 0;

	// Search through the set finding where the "din" fits in. Similar
	// while loop as the while loop in FSSetInsertRef() above.

	while (drn < uiLastDrn)
	{
		uiLastSrcOfs = srcState.uiOffset;
		uiOneRun = 0;

		if (srcState.uiOffset >= uiSetLength)
		{
			return (RC_SET( FERR_KEY_NOT_FOUND));
		}

		// Check for a run of ONE's

		byValue = pSrcRef[srcState.uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{

			// Read the number of one runs

			uiOneRun = DINOneRunVal( pSrcRef, &srcState);
			uiLastDrn -= uiOneRun;
		}
		else
		{
			uiOldDelta = DINNextVal( pSrcRef, &srcState);
			uiLastDrn -= uiOldDelta;
		}
	}

	f_memcpy( pDestRef, pSrcRef, destState.uiOffset = uiLastSrcOfs);

	if (uiOneRun)
	{

		// Divide out a one run. drn may match the first, middle or last
		// value of a one run. Copy up to the last source value to dest.

		uiLastDrn += uiOneRun;
		if (drn == uiLastDrn - 1)
		{

			// Remove the first value from the run - run is >= 2

			DINPutNextVal( pDestRef, &destState, 2);
			if (uiOneRun > 2)
			{
				DINPutOneRunVal( pDestRef, &destState, uiOneRun - 2);
			}

			goto FSSDR_move_rest_of_dins;
		}
		else
		{
			DINPutOneRunVal( pDestRef, &destState, (uiLastDrn - drn) - 1);

			if (drn > uiLastDrn - uiOneRun)
			{
				FLMUINT	oneRun2ndHalf = uiOneRun - (uiLastDrn - drn) - 1;

				DINPutNextVal( pDestRef, &destState, 2);
				if (oneRun2ndHalf)
				{
					DINPutOneRunVal( pDestRef, &destState, oneRun2ndHalf);
				}

				goto FSSDR_move_rest_of_dins;
			}
			else
			{

				// Delete from the END of a one run

				uiOldDelta = 1;
				goto FSSDR_combine_2_deltas;
			}
		}

		// Should never reach here!
	}

	// Check for duplicates

	else if (drn != uiLastDrn)
	{
		return (RC_SET( FERR_KEY_NOT_FOUND));
	}

	// Hit a non-one run where drn == uiLastDrn Check the next DIN value
	// and add uiOldDelta to it UNLESS it is a one run, then write
	// uiOldDelta+1 and if uiOneRun-1 is non-zero then write uiOneRun-1
	
	if (srcState.uiOffset < uiSetLength)
	{
		byValue = pSrcRef[srcState.uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{
			DINPutNextVal( pDestRef, &destState, uiOldDelta + 1);
			uiOneRun = DINOneRunVal( pSrcRef, &srcState) - 1;
			DINPutOneRunVal( pDestRef, &destState, uiOneRun);
		}
		else
		{
FSSDR_combine_2_deltas:

			if (srcState.uiOffset >= uiSetLength)
			{
				goto FSSDR_done;
			}

			uiTemp = DINNextVal( pSrcRef, &srcState);
			DINPutNextVal( pDestRef, &destState, uiOldDelta + uiTemp);
		}
	}

FSSDR_move_rest_of_dins:

	if ((uiMoveLen = (FLMUINT) (uiSetLength - srcState.uiOffset)) != 0)
	{
		f_memcpy( &pDestRef[destState.uiOffset], &pSrcRef[srcState.uiOffset],
					uiMoveLen);
		destState.uiOffset += uiMoveLen;
	}

FSSDR_done:

	*puiSetLength = destState.uiOffset;
	return (FERR_OK);
}

/****************************************************************************
Desc:	Put a SEN value into a buffer - return the length of storage used
****************************************************************************/
FLMUINT SENPutNextVal(
	FLMBYTE **		pSenRV,
	FLMUINT			senValue)
{
	FLMBYTE *		pSen = *pSenRV;
	FLMUINT			uiSenLen;

	if (senValue <= SEN_1B_VAL)
	{
		*pSen++ = (FLMBYTE) senValue;
	}
	else if (senValue <= SEN_2B_VAL)
	{
		*pSen++ = (FLMBYTE) (SEN_2B_CODE + 
					 (FLMBYTE) ((senValue >> 8) & SEN_2B_MASK));
		*pSen++ = (FLMBYTE) senValue;
	}
	else if (senValue <= SEN_3B_VAL)
	{
		*pSen++ = (FLMBYTE) (SEN_3B_CODE + 
					 (FLMBYTE) ((senValue >> 16) & SEN_3B_MASK));
		goto SENPV_2_bytes;
	}
	else if (senValue <= SEN_4B_VAL)
	{
		*pSen++ = (FLMBYTE) (SEN_4B_CODE + 
					 (FLMBYTE) ((senValue >> 24) & SEN_4B_MASK));
		goto SENPV_3_bytes;
	}
	else
	{
		*pSen++ = SEN_5B_CODE;
		*pSen++ = (FLMBYTE) (senValue >> 24);
		
SENPV_3_bytes:

		*pSen++ = (FLMBYTE) (senValue >> 16);
		
SENPV_2_bytes:

		*pSen++ = (FLMBYTE) (senValue >> 8);
		*pSen++ = (FLMBYTE) senValue;
	}

	uiSenLen = (FLMUINT) (pSen -*pSenRV);
	*pSenRV = pSen;
	return (uiSenLen);
}

/****************************************************************************
Desc: Put the next one run value - high level
****************************************************************************/
FLMUINT DINPutOneRunVal(
	FLMBYTE *		dinPtr,
	DIN_STATE *		state,
	FLMUINT			uiValue)
{
	FLMUINT			uiLength = 1;
	FLMUINT			uiOffset = state->uiOffset;
	FLMBYTE *		pOneRun;

	if (uiValue == 1)
	{
		dinPtr[uiOffset] = 1;
	}
	else if (uiValue <= DIN_MAX_1B_ONE_RUN)
	{
		dinPtr[uiOffset] = (FLMBYTE) (DIN_ONE_RUN_LV | (((FLMBYTE) uiValue) - 2));
	}
	else
	{
		dinPtr[uiOffset] = DIN_ONE_RUN_HV;
		pOneRun = &dinPtr[uiOffset + 1];

		uiLength += SENPutNextVal( &pOneRun, uiValue);
	}

	state->uiOffset += uiLength;
	return (uiLength);
}
