//-------------------------------------------------------------------------
// Desc:	Query positioning keys
// Tabs:	3
//
// Copyright (c) 1997-2007 Novell, Inc. All Rights Reserved.
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

#define DOMAIN_TO_DRN(uiDomain) \
	(FLMUINT)(((uiDomain) + 1) * 256 + 1)
	
#define DRN_TO_DOMAIN(uiDrn) \
	(FLMUINT)(((uiDrn) - 1) / 256 - 1)

FSTATIC FLMINT flmPosKeyCompare(
	POS_KEY *			pKey1,
	POS_KEY *			pKey2);

FSTATIC RCODE flmLoadPosKeys(
	CURSOR *				pCursor,
	POS_KEY *			pKeys,
	FLMUINT				uiNumKeys,
	FLMBOOL				bLeafLevel);

FSTATIC RCODE flmKeyIsMatch(
	CURSOR *				pCursor,
	IXD *					pIxd,
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiDrn,
	POS_KEY * *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMUINT				uiKeyArrayGrowSize);

FSTATIC RCODE flmExamineBlock(
	CURSOR *				pCursor,
	IXD *					pIxd,
	FLMBYTE *			pucBlk,
	FSIndexCursor *	pFSIndexCursor,
	FLMUINT **			ppuiChildBlockAddresses,
	FLMUINT *			puiNumChildBlocks,
	FLMUINT *			puiBlkAddressArrayAllocSize,
	POS_KEY * *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMBOOL *			pbHighKeyInRange);

FSTATIC RCODE flmGetLastKey(
	FDB *					pDb,
	CURSOR *				pCursor,
	IXD *					pIxd,
	LFILE *				pLFile,
	FLMUINT				uiBlockAddress,
	POS_KEY **			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize);

FSTATIC RCODE flmCurGetPosKeys(
	FDB *					pDb,
	CURSOR *				pCursor);

FSTATIC FLMBOOL flmFindWildcard(
	FLMBYTE *			pValue,
	FLMUINT *			puiCharPos);

FSTATIC RCODE flmAddKeyPiece(
	FLMUINT				uiMaxKeySize,
	IFD *					pIfd,
	FLMBOOL				bDoMatchBegin,
	FLMBYTE *			pFromKey,
	FLMUINT *			puiFromKeyPos,
	FLMBOOL				bFromAtFirst,
	FLMBYTE *			pUntilKey,
	FLMUINT *			puiUntilKeyPos,
	FLMBOOL				bUntilAtEnd,
	FLMBYTE *			pBuf,
	FLMUINT				uiBufLen,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbDoneBuilding);

FSTATIC RCODE flmAddTextPiece(
	FLMUINT				uiMaxKeySize,
	IFD *					pIfd,
	FLMBOOL				bCaseInsensitive,
	FLMBOOL				bDoMatchBegin,
	FLMBOOL				bDoFirstSubstring,
	FLMBOOL				bTrailingWildcard,
	FLMBYTE *			pFromKey,
	FLMUINT *			puiFromKeyPos,
	FLMBOOL				bFromAtFirst,
	FLMBYTE *			pUntilKey,
	FLMUINT *			puiUntilKeyPos,
	FLMBOOL				bUntilAtEnd,
	FLMBYTE *			pBuf,
	FLMUINT				uiBufLen,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbDoneBuilding,
	FLMBOOL *			pbOriginalCharsLost);

FSTATIC FLMBOOL flmSelectBestSubstr(
	FLMBYTE **			ppValue,
	FLMUINT *			puiValueLen,
	FLMUINT				uiIfdFlags,
	FLMBOOL *			pbTrailingWildcard);

FSTATIC FLMUINT flmCountCharacters(
	FLMBYTE *			pValue,
	FLMUINT				uiValueLen,
	FLMUINT				uiMaxToCount,
	FLMUINT				uiIfdFlags);

/****************************************************************************
Desc: Compares the contents of the key buffers for two cursor positioning keys,
		returning one of the following values:
				<0		Indicates that the first key is less than the second.
				 0		Indicates that the two keys are equal.
				>0		Indicates that the first key is greater then the second.
****************************************************************************/
FSTATIC FLMINT flmPosKeyCompare(
	POS_KEY *	pKey1,
	POS_KEY *	pKey2
	)
{
	FLMINT	iCmp;
	
	if (pKey1->uiKeyLen > pKey2->uiKeyLen)
	{
		if ((iCmp = f_memcmp( pKey1->pucKey, pKey2->pucKey,
								pKey2->uiKeyLen)) == 0)
		{
			iCmp = 1;
		}
	}
	else if( pKey1->uiKeyLen < pKey2->uiKeyLen)
	{
		if ((iCmp = f_memcmp( pKey1->pucKey, pKey2->pucKey,
								pKey1->uiKeyLen)) == 0)
		{
			iCmp = -1;
		}
	}
	else
	{
		if ((iCmp = f_memcmp( pKey1->pucKey,
							pKey2->pucKey, pKey2->uiKeyLen)) == 0)
		{
			// Compare DRNs if everything else is the same.  NOTE: DRNs are in
			// reverse order in the positioning key array.

			if (pKey1->uiDrn && pKey2->uiDrn)
			{
				if (pKey1->uiDrn > pKey2->uiDrn)
				{
					iCmp = -1;
				}
				else if (pKey1->uiDrn < pKey2->uiDrn)
				{
					iCmp = 1;
				}
			}
		}
	}
	return iCmp;
}

/****************************************************************************
Desc: Loads a set of positioning keys into a subquery's array, allocating it
		if necessary.
****************************************************************************/
FSTATIC RCODE flmLoadPosKeys(
	CURSOR *		pCursor,
	POS_KEY *	pKeys,
	FLMUINT		uiNumKeys,
	FLMBOOL		bLeafLevel
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiKeyCnt;
	FLMUINT	uiRFactor;
	FLMUINT	uiTotCnt;

	// If the B-tree was empty, the key array will be left NULL.
	
	if (!pKeys || !uiNumKeys)
	{
		goto Exit;
	}
	
	// Allocate the array of positioning keys in the subquery.
		
	uiKeyCnt = (uiNumKeys > FLM_MAX_POS_KEYS + 1)
				? FLM_MAX_POS_KEYS + 1
				: uiNumKeys;
	if (RC_BAD( rc = f_calloc( uiKeyCnt * sizeof( POS_KEY),
										&pCursor->pPosKeyArray)))
	{
		goto Exit;
	}
	pCursor->uiNumPosKeys = uiKeyCnt;
	pCursor->bLeafLevel = bLeafLevel;

	// If there are less keys than the number of slots in the positioning
	// key array, each key must be put into multiple slots.  Calculate how
	// many slots correspond to each key (uiSlots), and then set the keys
	// into their corresponding slots.  NOTE: it will often be the case that
	// the number of keys does not divide evenly into the number of slots in
	// the array.  In these cases thare will be a remainder, uiRFactor.  If
	// uiRFactor = n, the first n keys will be set into (uiSlots + 1) slots.
	
	if (uiNumKeys <= FLM_MAX_POS_KEYS + 1)
	{
		for (uiTotCnt = 0; uiTotCnt < uiKeyCnt; uiTotCnt++)
		{
			f_memcpy( &pCursor->pPosKeyArray[ uiTotCnt],
						 &pKeys[ uiTotCnt],
						 sizeof( POS_KEY));

			// NOTE: we're keeping this memory for the positioning key to which
			// it is being copied.
			pKeys [uiTotCnt].pucKey = NULL;
		}
	}

	// If there are more keys than the number of slots in the positioning
	// key array, a certain number of keys must be skipped for each key that
	// is set in the array.  Calculate how many keys must be skipped for each
	// slot (uiIntervalSize), and then iterate through the passed-in set of
	// keys, setting the appropriate ones into their corresponding slots.
	// NOTE: it will often be the case that the number of slots in the array
	// does not divide evenly into the number of keys.  In these cases there
	// will be a remainder (uiRFactor).  Where uiRFactor = n,
	// (uiIntervalSize + 1) keys will be skipped before each of the first n
	// slots in the array are filled.
	
	else
	{
		FLMUINT		uiLoopCnt;
		FLMUINT		uiIntervalSize = (uiNumKeys - 2) / (FLM_MAX_POS_KEYS - 1) - 1;
		
		uiRFactor = (uiNumKeys - 2) % (FLM_MAX_POS_KEYS - 1);

		f_memcpy( &pCursor->pPosKeyArray[ 0], &pKeys[ 0], sizeof( POS_KEY));
		f_memcpy( &pCursor->pPosKeyArray[ 1], &pKeys[ 1], sizeof( POS_KEY));
		
		// NOTE: we're keeping this memory for the positioning key to which
		// it is being copied.
		pKeys [0].pucKey = NULL;
		pKeys [1].pucKey = NULL;

		uiKeyCnt = 2;
		for( uiTotCnt = 2; uiTotCnt < FLM_MAX_POS_KEYS; uiTotCnt++)
		{
			for( uiLoopCnt = 0; uiLoopCnt < uiIntervalSize; uiLoopCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
				uiKeyCnt++;
			}
			
			if( uiRFactor)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
				uiKeyCnt++;
				uiRFactor--;
			}
			
			f_memcpy( &pCursor->pPosKeyArray[ uiTotCnt],
						&pKeys[ uiKeyCnt], sizeof( POS_KEY));
						
			// NOTE: we're keeping this memory for the positioning key to which
			// it is being copied.
			
			pKeys [uiKeyCnt].pucKey = NULL;
			uiKeyCnt++;
		}

		// Make sure the last key in the positioning key array is the last
		// key in the result set, then free the memory used for the pKey array.
		
		f_memcpy( &pCursor->pPosKeyArray[ FLM_MAX_POS_KEYS],
						&pKeys[ uiNumKeys - 1], sizeof( POS_KEY));
		pKeys [uiNumKeys - 1].pucKey = NULL;
		while (uiKeyCnt < uiNumKeys - 1)
		{
			f_free( &pKeys[ uiKeyCnt].pucKey);
			uiKeyCnt++;
		}
	}
	
Exit:
	return( rc);
}

/****************************************************************************
Desc: Evaluates an index key against selection criteria, and adds it to the
		passed-in key array.
****************************************************************************/
FSTATIC RCODE flmKeyIsMatch(
	CURSOR *				pCursor,
	IXD *					pIxd,
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiDrn,
	POS_KEY * *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMUINT				uiKeyArrayGrowSize
	)
{
	RCODE				rc = FERR_OK;
	SUBQUERY *		pSubQuery = pCursor->pSubQueryList;
	FlmRecord *		pKey = NULL;
	FLMBOOL			bHaveMatch = FALSE;
	FLMUINT			uiResult;
	POS_KEY *		pPosKey;
	
	// If pSubQuery->bDoKeyMatch is FALSE, the selection criteria for this
	// query are satisfied by a contiguous set of index keys.  Therefore,
	// there is no need to evaluate keys against the selection criteria.
	// We have already established that the passed-in key falls within
	// the range of keys that contains the result set of the query.
	// NOTE: bDoRecMatch cannot ever be set, otherwise, positioning is not
	// allowed.

	bHaveMatch = !pSubQuery->OptInfo.bDoKeyMatch;
	if (!bHaveMatch)
	{

		// Get the key in the form of a FlmRecord object.

		if (RC_BAD( rc = flmIxKeyOutput( pIxd, pucKey, uiKeyLen, &pKey, TRUE)))
		{
			goto Exit;
		}
		pKey->setID( uiDrn);

		// Evaluate the key against the subquery - there will only
		// be one at this point.

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pKey, TRUE, &uiResult)))
		{
			if (rc == FERR_TRUNCATED_KEY)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
		bHaveMatch = (uiResult == FLM_TRUE) ? TRUE : FALSE;
	}
			
	if (bHaveMatch)
	{
		if (*puiNumKeys == *puiKeyArrayAllocSize)
		{
			if (RC_BAD( rc = f_recalloc(
				(*puiKeyArrayAllocSize + uiKeyArrayGrowSize) * sizeof( POS_KEY),
				ppKeys)))
			{
				goto Exit;
			}
			(*puiKeyArrayAllocSize) += uiKeyArrayGrowSize;
		}
		pPosKey = &((*ppKeys)[*puiNumKeys]);
		if (RC_BAD( rc = f_calloc( uiKeyLen, &pPosKey->pucKey)))
		{
			goto Exit;
		}
		f_memcpy( pPosKey->pucKey, pucKey, uiKeyLen);
		pPosKey->uiKeyLen = uiKeyLen;
		pPosKey->uiDrn = uiDrn;
		(*puiNumKeys)++;
	}
	
Exit:
	if (pKey)
	{
		pKey->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Examines an index B-tree block to find the keys in it that could be
		used to position within a cursor's result set.
Visit:This code NEEDS to use the b-tree routines and NOT use the low level
		format codes to go to the next element or key.  Other problems include
		doing the same work for each element even though you are at the same
		level of the b-tree.
****************************************************************************/
FSTATIC RCODE flmExamineBlock(
	CURSOR *				pCursor,
	IXD *					pIxd,
	FLMBYTE *			pucBlk,
	FSIndexCursor *	pFSIndexCursor,
	FLMUINT **			ppuiChildBlockAddresses,
	FLMUINT *			puiNumChildBlocks,
	FLMUINT *			puiBlkAddressArrayAllocSize,
	POS_KEY * *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMBOOL *			pbHighKeyInRange
	)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucFromKey [MAX_KEY_SIZ];
	FLMUINT			uiFromKeyLen;
	FLMBYTE			ucUntilKey [MAX_KEY_SIZ];
	FLMUINT			uiUntilKeyLen;
	FLMUINT			uiUntilDrn = 0;
	FLMBOOL			bRangeOverlaps;
	FLMBOOL			bUntilKeyInSet;
	FLMBOOL			bUntilKeyPastEndOfKeys;
	FLMUINT			uiDomain;
	DIN_STATE		dinState;
	FLMBOOL			bFirstRef;
	FLMUINT			uiEndOfBlock = FB2UW( &pucBlk [BH_BLK_END]);
	FLMUINT			uiCurrElmOffset = BH_OVHD;
	FLMUINT			uiBlkType = (FLMUINT)BH_GET_TYPE( pucBlk);
	FLMUINT			uiElmLength;
	FLMBYTE *		pucElement;
	FLMBYTE *		pucElm;
	FLMBYTE *		pucElmKey;
	FLMBYTE *		pucElmRecord;
	FLMBYTE *		pucChildBlkAddr;
	FLMUINT			uiChildBlkAddr;
	FLMUINT			uiElmRecLen;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiElmPKCLen;
	FLMUINT			uiElmOvhd;

	// This loop moves across a database block from the leftmost element to the
	// rightmost.  Each contiguous pair of elements is viewed as a "key range",
	// where the first key in the pair is the start key and the second is the
	// end key.  In the loop, each key range is checked to see if it overlaps
	// with any part of the query's result set.  If it does, two things happen:
	// first, the down pointer from the end key is added to a passed-in list;
	// second, the end key is checked to see if it satisfies the query's
	// selection criteria.  If it does, it is added to a passed-in list of
	// positioning keys.
	// NOTE: until key is given a key length of 0 so that in the first iteration,
	// the key range will be from FO_FIRST to the leftmost key in the block.

	if( uiBlkType == BHT_LEAF)
	{
		uiElmOvhd = BBE_KEY;
	}
	else if( uiBlkType == BHT_NON_LEAF_DATA)
	{
		uiElmOvhd = BNE_DATA_OVHD;
	}
	else if( uiBlkType == BHT_NON_LEAF)
	{
		uiElmOvhd = BNE_KEY_START;
	}
	else
	{
		uiElmOvhd = BNE_KEY_COUNTS_START;
	}
	
	uiUntilKeyLen = 0;
	bUntilKeyPastEndOfKeys = FALSE;
	bFirstRef = TRUE;
	while (uiCurrElmOffset < uiEndOfBlock)
	{

		// Move the until key into the start key buffer.

		if (uiUntilKeyLen)
		{
			f_memcpy( ucFromKey, ucUntilKey, uiUntilKeyLen);
		}
		uiFromKeyLen = uiUntilKeyLen;

		pucElement = &pucBlk [uiCurrElmOffset];
		pucElm = pucElement;
		uiDomain = FSGetDomain( &pucElm, uiElmOvhd);

		if (uiBlkType == BHT_LEAF)
		{
			uiElmLength = (FLMUINT)(BBE_LEN( pucElement));
			pucElmKey = &pucElement [BBE_KEY];
			pucElmRecord = BBE_REC_PTR( pucElement);
			uiElmRecLen = BBE_GET_RL( pucElement);
			if (bFirstRef)
			{
				RESET_DINSTATE( dinState);
				uiUntilDrn = SENNextVal( &pucElm);
				bFirstRef = FALSE;
			}
			else
			{
				FLMUINT uiRefSize = uiElmRecLen -
											(FLMUINT)(pucElm - pucElmRecord);

				if (dinState.uiOffset < uiRefSize)
				{

					// Not at end, read current value.

					DINNextVal( pucElm, &dinState);
				}

				if (dinState.uiOffset >= uiRefSize)
				{
					uiCurrElmOffset += uiElmLength;
					bFirstRef = TRUE;

					// No need to go any further if we have run
					// off the end of the list of keys for the query.

					if (bUntilKeyPastEndOfKeys)
					{
						break;
					}
					else
					{
						continue;
					}
				}
				else
				{
					DIN_STATE	savedState;

					// Don't move the dinState, stay
					// put and get the next DIN value

					savedState.uiOffset = dinState.uiOffset;
					savedState.uiOnes   = dinState.uiOnes;
					uiUntilDrn -= DINNextVal( pucElm, &savedState);
				}
			}
		}
		else if (uiBlkType == BHT_NON_LEAF_DATA)
		{
			uiElmLength = uiElmOvhd;
			pucElmKey = pucElement;
			uiUntilDrn = DOMAIN_TO_DRN( uiDomain);
		}
		else
		{
			uiElmLength = BBE_GET_KL( pucElement ) + uiElmOvhd + 
							(BNE_IS_DOMAIN(pucElement) ? BNE_DOMAIN_LEN : 0);
			pucElmKey = &pucElement [uiElmOvhd];
			uiUntilDrn = DOMAIN_TO_DRN( uiDomain);
		}

		// See if we are on the last element.  If it is a leaf block,
		// it does NOT represent a key.  If it is a non-leaf block,
		// it represents the highest possible key, but there is no
		// data to extract fields from.

		if ((uiBlkType == BHT_LEAF) && (uiElmLength == uiElmOvhd))
		{
			goto Exit;		// Should return FERR_OK
		}

		if ((uiBlkType != BHT_LEAF) && (uiElmLength == uiElmOvhd))
		{
			uiElmKeyLen = uiElmPKCLen = uiUntilKeyLen = 0;
		}
		else
		{

			// Get the element key length and previous key count (PKC).

			uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pucElement));
			uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( pucElement));

			// Now copy the current partial key into the EndKey key buffer.
			
			f_memcpy( &ucUntilKey [uiElmPKCLen], pucElmKey, uiElmKeyLen);
			uiUntilKeyLen = uiElmKeyLen + uiElmPKCLen;
		}

		// Test for Overlap of from key (exclusive) to until key (inclusive)
		// with search keys.

		bRangeOverlaps = pFSIndexCursor->compareKeyRange(
									ucFromKey, uiFromKeyLen,
									(FLMBOOL)((uiFromKeyLen)
												 ? TRUE
												 : FALSE),
									ucUntilKey, uiUntilKeyLen, FALSE,
									&bUntilKeyInSet, &bUntilKeyPastEndOfKeys);

		// Does this range overlap a range of keys?

		if (bRangeOverlaps)
		{

			// If we are not at the leaf level, get and save child block address.

			if (uiBlkType != BHT_LEAF)
			{
				// THIS CODE SHOULD BE USING A STACK!!!!

				if (uiElmOvhd == BNE_DATA_OVHD)
				{
					pucChildBlkAddr = &pucElement[ BNE_DATA_CHILD_BLOCK];
				}
				else
				{
					pucChildBlkAddr = &pucElement [BNE_CHILD_BLOCK];
				}
				uiChildBlkAddr = FB2UD( pucChildBlkAddr );

				// Save uiChildBlkAddr to array of child block addresses.

				if (*puiNumChildBlocks == *puiBlkAddressArrayAllocSize)
				{
					if (RC_BAD( rc = f_recalloc(
						(*puiBlkAddressArrayAllocSize + FLM_ADDR_GROW_SIZE)
						* sizeof( FLMUINT), ppuiChildBlockAddresses)))
					{
						goto Exit;
					}
					(*puiBlkAddressArrayAllocSize) += FLM_ADDR_GROW_SIZE;
				}
				(*ppuiChildBlockAddresses)[ *puiNumChildBlocks] = uiChildBlkAddr;
				(*puiNumChildBlocks)++;
			}

			// If the last element in the block has just been processed, the key
			// will have a length of 0.  If it is somewhere within the range of
			// keys that contains the query's result set, return TRUE in
			// pbHighKeyInRange.  At a higher level, if only one more key is
			// needed to fill the array of positioning keys, the B-Tree will
			// then be traversed to the leaf level to retrieve and test the
			// rightmost key.

			if (!uiUntilKeyLen && bUntilKeyInSet)
			{
				*pbHighKeyInRange = TRUE;
			}

			// If the key falls into one of the key ranges that contain the
			// query's result set, see if it satisfies the selection criteria.
			// If so, increment the counter for the positioning key array and
			// put the key into the array.
			
			else if (bUntilKeyInSet)
			{
				if (RC_BAD( rc = flmKeyIsMatch( pCursor, pIxd,
														  ucUntilKey, uiUntilKeyLen,
														  uiUntilDrn,
														  ppKeys, puiNumKeys,
														  puiKeyArrayAllocSize,
														  FLM_KEYS_GROW_SIZE)))
				{
					goto Exit;
				}
			}
		}

		// If this is not the first reference, stay inside the element and
		// get the next reference.

		if (!bFirstRef)
		{
			continue;
		}

		uiCurrElmOffset += uiElmLength;

		// No need to go any further if we have run off the end of the list
		// of keys for the query.

		if (bUntilKeyPastEndOfKeys)
		{
			break;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc: Finds the rightmost key in the leaf level of a B-tree, and evaluates
		it against the selection criteria of the given subquery.
Visit:This routine must be rewritten to get rid of the low level BTREE
		definitions.  The next() btree calls should have been used.
****************************************************************************/
FSTATIC RCODE flmGetLastKey(
	FDB *				pDb,
	CURSOR *			pCursor,
	IXD *				pIxd,
	LFILE *			pLFile,
	FLMUINT			uiBlockAddress,
	POS_KEY * *		ppKeys,
	FLMUINT *		puiNumKeys,
	FLMUINT *		puiKeyArrayAllocSize
	)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucEndKey [MAX_KEY_SIZ];
	FLMUINT			uiEndKeyLen = 0;
	FLMUINT			uiEndDrn = 0;
	BTSK				stack;
	FLMBYTE			ucKeyBuf [MAX_KEY_SIZ];
	BTSK *			pStack = &stack;
	FLMUINT			uiEndOfBlock;
	FLMUINT			uiCurrElmOffset;
	FLMUINT			uiBlkType;
	FLMUINT			uiElmLength;
	FLMBYTE *		pucBlk;
	FLMBYTE *		pucElement = NULL;
	FLMBYTE *		pucElm;
	FLMBYTE *		pucElmKey;
	FLMBYTE *		pucElmRecord;
	FLMUINT			uiElmRecLen;
	FLMBYTE *		pucBlockAddress;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiElmPKCLen;
	FLMBOOL			bHaveLastKey = FALSE;
	FLMUINT			uiElmOvhd = 0;
	DIN_STATE		dinState;
	FLMUINT			uiRefSize;

	FSInitStackCache( pStack, 1);
	pStack->pKeyBuf = &ucKeyBuf [0];
	
	// uiBlockAddress contains the address of the rightmost B-Tree block at
	// some unspecified level of the B-Tree (usually not the leaf level).
	// This loop works down the right side of the B-Tree from the passed-in
	// block address until it reaches the rightmost block at the leaf level.
	// The rightmost key is then found in that block.
	
	for( ;;)
	{
		if (RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlockAddress, pStack)))
		{
			goto Exit;
		}
		pucBlk = pStack->pBlk;
		uiBlkType = (FLMUINT)(BH_GET_TYPE( pucBlk));
		uiEndOfBlock = (FLMUINT)pStack->uiBlkEnd;
		uiCurrElmOffset = BH_OVHD;
		
		// This loop works across a B-Tree block from the leftmost key to the
		// rightmost key.  At non-leaf levels of the B-Tree, the child block
		// address associated with the rightmost key is then used to progress
		// further down the right side of the B-Tree.

		while (uiCurrElmOffset < uiEndOfBlock)
		{

			pucElement = &pucBlk [uiCurrElmOffset];

			if (uiBlkType == BHT_LEAF)
			{
				uiElmOvhd = BBE_KEY;
				uiElmLength = (FLMUINT)(BBE_LEN( pucElement));
				pucElmKey = &pucElement [BBE_KEY];

				// See if we are on the last element.  If it is a leaf block,
				// it does NOT represent a key; the previous element that was
				// processed contained the last key, which means we're finished.
				
				if (uiElmLength == uiElmOvhd)
				{
					bHaveLastKey = TRUE;
					break;
				}

				// Get the last DRN in the element - in case this element is
				// the last one before the end.

				pucElmRecord = BBE_REC_PTR( pucElement);
				uiElmRecLen = BBE_GET_RL( pucElement);
				pucElm = pucElement;
				(void)FSGetDomain( &pucElm, uiElmOvhd);
				RESET_DINSTATE( dinState);
				uiEndDrn = SENNextVal( &pucElm);
				uiRefSize = uiElmRecLen -
											(FLMUINT)(pucElm - pucElmRecord);
				for (;;)
				{
					if (dinState.uiOffset < uiRefSize)
					{

						// Not at end, read current value.

						DINNextVal( pucElm, &dinState);
					}

					if (dinState.uiOffset >= uiRefSize)
					{
						break;
					}
					else
					{
						DIN_STATE	savedState;

						// Don't move the dinState, stay
						// put and get the next DIN value

						savedState.uiOffset = dinState.uiOffset;
						savedState.uiOnes   = dinState.uiOnes;
						uiEndDrn -= DINNextVal( pucElm, &savedState);
					}
				}
			}
			else if( uiBlkType == BHT_NON_LEAF_DATA)
			{
				uiElmOvhd = uiElmLength = BNE_DATA_OVHD;
				pucElmKey = pucElement;
			}
			else
			{
				uiElmOvhd = pStack->uiElmOvhd;

				uiElmLength = BBE_GET_KL( pucElement ) + uiElmOvhd + 
							(BNE_IS_DOMAIN(pucElement) ? BNE_DOMAIN_LEN : 0);	
				pucElmKey = &pucElement [uiElmOvhd];
			}

			if ((uiBlkType != BHT_LEAF) && (uiElmLength == uiElmOvhd))
			{
				uiElmKeyLen = uiElmPKCLen = uiEndKeyLen = 0;
			}
			else if (uiBlkType == BHT_NON_LEAF_DATA)
			{
				uiElmLength = BNE_DATA_OVHD;
				f_memcpy( ucEndKey, pucElmKey, DIN_KEY_SIZ);
			}
			else
			{

				/* Get the element key length and previous key count (PKC). */

				uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pucElement));
				uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( pucElement));

				f_memcpy( &ucEndKey [uiElmPKCLen], pucElmKey, uiElmKeyLen);
				uiEndKeyLen = (FLMUINT)(uiElmKeyLen + uiElmPKCLen);
			}
			uiCurrElmOffset += uiElmLength;
		}
		
		if (!bHaveLastKey)
		{

			// Get and save child block address.

			pucBlockAddress = (FLMBYTE *)((uiElmOvhd == BNE_DATA_OVHD)
													? &pucElement [BNE_DATA_CHILD_BLOCK]
													: &pucElement [BNE_CHILD_BLOCK]);
			uiBlockAddress = FB2UD( pucBlockAddress );
		}
		else
		{

			// We have reached the leaf level of the B-Tree, and we have the
			// rightmost key.  See if it satisfies the selection criteria for
			// the query. If so, put it into the passed-in array of positioning
			// keys.  Then break out of the loop; we're finished.

			if (RC_BAD( rc = flmKeyIsMatch( pCursor, pIxd,
											ucEndKey, uiEndKeyLen, uiEndDrn,
											ppKeys, puiNumKeys,
											puiKeyArrayAllocSize, 1)))
			{
				goto Exit;
			}

			break;
		}
	}
	
Exit:
	FSReleaseBlock( pStack, FALSE);
	return( rc);
}

/****************************************************************************
Desc: Frees the allocations associated with a subquery's array.
****************************************************************************/
void flmCurFreePosKeys(
	CURSOR *			pCursor
	)
{
	FLMUINT	uiLoopCnt;
	
	if (pCursor->pPosKeyArray)
	{
		for (uiLoopCnt = 0; uiLoopCnt < pCursor->uiNumPosKeys; uiLoopCnt++)
		{
			f_free( &pCursor->pPosKeyArray[ uiLoopCnt].pucKey);
		}
		f_free( &pCursor->pPosKeyArray);
		pCursor->uiNumPosKeys = 0;
	}
	pCursor->uiLastPrcntPos = 0;
	pCursor->uiLastPrcntOffs = 0;
	pCursor->bUsePrcntPos = FALSE;
}

/****************************************************************************
Desc: Gets a set of positioning keys for a particular subquery.
****************************************************************************/
FSTATIC RCODE flmCurGetPosKeys(
	FDB *				pDb,
	CURSOR *			pCursor
	)
{
	RCODE				rc = FERR_OK;
	BTSK				stack [BH_MAX_LEVELS];
	FLMBYTE			ucKeyBuf [MAX_KEY_SIZ];
	BTSK *			pStack = stack;
	LFILE *			pLFile;
	LFILE				TmpLFile;
	IXD *				pIxd;
	SUBQUERY *		pSubQuery;
	FLMUINT *		puiChildBlockAddresses = NULL;
	FLMUINT *		puiTmpBlocks = NULL;
	FLMUINT 			uiNumChildBlocks = 0;
	FLMUINT			uiNumTmpBlks;
	FLMUINT			uiBlkAddressArrayAllocSize = 0;
	POS_KEY *		pKeys = NULL;
	FLMUINT			uiNumKeys = 0;
	FLMUINT			uiKeyArrayAllocSize = 0;
	FLMBOOL			bHighKeyInRange = FALSE;

	FSInitStackCache( &stack[ 0], BH_MAX_LEVELS);

	// Check to verify that it is possible to set up an array of positioning keys
	// for this query.  The following conditions must be met:
	// 1) The query must use one and only one index
	// 2) The criteria must be solvable using only the index keys
	// 3)  The selection criteria cannot include DRNs.

	if (((pSubQuery = pCursor->pSubQueryList) == NULL) ||
		 pSubQuery->pNext ||
		 pSubQuery->OptInfo.eOptType != QOPT_USING_INDEX ||
		 pSubQuery->OptInfo.bDoRecMatch ||
		 pSubQuery->bHaveDrnFlds)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Free the existing key array, if there is one
	
	if (pCursor->pPosKeyArray)
	{
		flmCurFreePosKeys( pCursor);
	}

	// Get the necessary LFILE and IXD information from the subquery index.

	if (RC_BAD( rc = fdictGetIndex(
		pDb->pDict, pDb->pFile->bInLimitedMode,
		pSubQuery->OptInfo.uiIxNum, &pLFile, &pIxd)))
	{
		goto Exit;
	}

	// Set up a B-tree stack structure and get the root block in the index
	// B-tree.

	pStack->pKeyBuf = &ucKeyBuf [0];
	
	// If no root block returned from FSGetRootBlock, the array will be
	// returned empty, with rc set to success.
	
	if (RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}
		goto Exit;
	}
	uiNumTmpBlks = 1;

	// Extract the array of positioning keys by working down the B-tree
	// from the root block.  This loop will terminate when all levels of
	// the B-Tree have been processed, or when enough keys have been
	// found to populate the array.
	// NOTE: pSubQuery->pPosKeyPool has been initialized at a higher level.
	
	for(;;)
	{
		FLMUINT	uiBlkCnt = 0;
		
		// Work across the present level of the B-Tree from right to left.
		
		for(;;)
		{
		
			// This function moves across a database block from the leftmost
			// element to the rightmost, checking each key to see if it is
			// found in the query's result set.  If it is, it is added to a
			// list of possible positioning keys, and its pointers to child
			// blocks in the B-Tree are also kept.  In the event that not
			// enough keys are found at a given level in the B-Tree, the list
			// of child block pointers is used to work through the next level
			// of the B-Tree.
			
			if (RC_BAD( rc = flmExamineBlock( pCursor, pIxd, pStack->pBlk,
											 pSubQuery->pFSIndexCursor,
											 &puiChildBlockAddresses,
											 &uiNumChildBlocks,
											 &uiBlkAddressArrayAllocSize,
											 &pKeys, &uiNumKeys, &uiKeyArrayAllocSize,
											 &bHighKeyInRange)))
			{
				goto Exit;
			}
			uiBlkCnt++;
			
			// uiNumTmpBlks has the number of blocks to be processed at the
			// current level of the B-Tree.  When those have been processed,
			// break out of this loop and go to the next level of the B-Tree.
			
			if (uiBlkCnt == uiNumTmpBlks)
			{
				break;
			}
			if (RC_BAD( rc = FSGetBlock( pDb, pLFile, puiTmpBlocks[ uiBlkCnt],
										pStack )))
			{
				goto Exit;
			}
		}

		// If we're not on the leaf level, and we have at least
		// FLM_MIN_POS_KEYS - 1 keys, we need to go out and evaluate
		// the last key at the leaf level.
		
		if (uiNumKeys >= FLM_MIN_POS_KEYS - 1 &&
			 bHighKeyInRange && uiNumChildBlocks)
		{
			if (RC_BAD( rc = flmGetLastKey( pDb, pCursor, pIxd, pLFile,
										puiChildBlockAddresses [uiNumChildBlocks - 1],
										&pKeys, &uiNumKeys, &uiKeyArrayAllocSize)))
			{
				goto Exit;
			}
		}
		
		// If we have enough keys, or if we have reached the last level of the
		// B-tree, load up the subquery key array and quit.
			
		if ((uiNumKeys >= FLM_MIN_POS_KEYS) || !uiNumChildBlocks)
		{
			rc = flmLoadPosKeys( pCursor, pKeys, uiNumKeys,
								(FLMBOOL)((uiNumChildBlocks == 0)
											 ? TRUE
											 : FALSE));
			goto Exit;
		}
		
		// If not enough keys, go to the next level of the B-tree and traverse
		// it to find keys.  This should be done down to the last level.
		
		else
		{
			FLMUINT		uiKeyCnt;

			f_free( &puiTmpBlocks);
			puiTmpBlocks = puiChildBlockAddresses;
			uiNumTmpBlks = uiNumChildBlocks;
			puiChildBlockAddresses = NULL;
			uiNumChildBlocks = uiBlkAddressArrayAllocSize = 0;
			for (uiKeyCnt = 0; uiKeyCnt < uiNumKeys; uiKeyCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
			}
			f_free( &pKeys);
			pKeys = NULL;
			uiNumKeys = 0;
			uiKeyArrayAllocSize = 0;
			if (RC_BAD( rc = FSGetBlock( pDb, pLFile,
										 puiTmpBlocks[ 0],
										 pStack )))
			{
				goto Exit;
			}
			bHighKeyInRange = FALSE;
		}
	}
	
Exit:
	if ( pKeys)
	{
		if (RC_BAD( rc))
		{
			for ( FLMUINT uiKeyCnt = 0; uiKeyCnt < uiNumKeys; uiKeyCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
			}
		}
		f_free( &pKeys);
	}
	f_free( &puiChildBlockAddresses);
	f_free( &puiTmpBlocks);
	FSReleaseStackCache( stack, BH_MAX_LEVELS, FALSE);

	return( rc);
}

/****************************************************************************
Desc: Gets a set of positioning keys for a particular subquery.
****************************************************************************/
RCODE flmCurSetupPosKeyArray(
	CURSOR *	pCursor
	)
{
	RCODE	rc = FERR_OK;
	FDB *	pDb = NULL;
	
	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}
	
	// Set up the pDb

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	
	// Set up array of positioning keys.

	if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
	{
		goto Exit;
	}
Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_CONFIG, pDb, rc);
	}
	return( rc);
}

/****************************************************************************
Desc: Gets the approximate percentage position of a passed-in key within a
		cursor's result set.
****************************************************************************/
RCODE flmCurGetPercentPos(
	CURSOR *			pCursor,
	FLMUINT *		puiPrcntPos
	)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = NULL;
	IXD *				pIxd;
	POS_KEY *		pPosKeyArray;
	POS_KEY			CompKey;
	FLMUINT			uiLowOffset;
	FLMUINT			uiMidOffset;
	FLMUINT			uiHighOffset;
	FLMUINT			uiIntervalSize;
	FLMUINT			uiRFactor;
	FLMINT			iCmp;
	FLMUINT			uiContainer;

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
	
	// If no array of positioning keys exists in the subquery, set one up.

	if (!pCursor->uiNumPosKeys)
	{
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
			
		// If no positioning keys exist, either the index or the result set
		// is empty.  Return NOT_FOUND.
		
		if (!pCursor->uiNumPosKeys)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}

	// If the number of positioning keys is 1, the position is 0%.

	if (pCursor->uiNumPosKeys == 1)
	{
		*puiPrcntPos = 0;
		goto Exit;
	}

	pPosKeyArray = pCursor->pPosKeyArray;

	if (pCursor->uiNumPosKeys == 2)
	{
		uiIntervalSize = FLM_MAX_POS_KEYS;
		uiRFactor = 0;
	}
	else
	{
		uiIntervalSize = FLM_MAX_POS_KEYS / (pCursor->uiNumPosKeys - 1);
		uiRFactor = FLM_MAX_POS_KEYS % (pCursor->uiNumPosKeys - 1);
	}

	// DEFECT 84741 -- only want to return a position of 1 for the second key
	// if the positioning key array is full.
	
	// Get an IXD, then convert the passed-in key from GEDCOM format to a
	// buffer containing the key in the FLAIM internal format.
	
	if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
								pDb->pFile->bInLimitedMode,
								pCursor->pSubQueryList->OptInfo.uiIxNum,
								NULL, &pIxd)))
	{
		goto Exit;
	}

	if (pCursor->ReadRc == FERR_BOF_HIT)
	{
		*puiPrcntPos = 0;
		goto Exit;
	}
	if (pCursor->ReadRc == FERR_EOF_HIT)
	{
		*puiPrcntPos = FLM_MAX_POS_KEYS;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	if (RC_BAD( rc = pCursor->pSubQueryList->pFSIndexCursor->currentKeyBuf(
								pDb, &pDb->TempPool, &CompKey.pucKey,
								&CompKey.uiKeyLen, &CompKey.uiDrn, &uiContainer)))
	{
		if (rc == FERR_EOF_HIT || rc == FERR_BOF_HIT || rc == FERR_NOT_FOUND)
		{
			rc = FERR_OK;
			*puiPrcntPos = 0;
		}
		goto Exit;
	}
	flmAssert( uiContainer == pCursor->uiContainer);

	// If a set position call has been performed, and no reposisioning has
	// been done since, check the passed-in key to see if it matches the
	// key returned from the set position call. If so, return the percent
	// passed in on the set position call. This is to create some symmetry
	// where the user calls set position, then takes the resulting key and
	// passes it back into a get position call.

	if (pCursor->bUsePrcntPos &&
		 pCursor->uiLastPrcntPos <= FLM_MAX_POS_KEYS)
	{
		if (flmPosKeyCompare( &pPosKeyArray[ pCursor->uiLastPrcntOffs],
						&CompKey) == 0)
		{
			*puiPrcntPos = pCursor->uiLastPrcntPos;
			goto Exit;
		}
		pCursor->bUsePrcntPos = FALSE;
	}

	// Do a binary search in the array of positioning keys for the passed-in
	// key. NOTE: the point of this search is to find the closest key <= to
	// the passed- in key. The range of values returned is
	// 0 to FLM_MAX_POS_KEYS (currently defined to be 1000), where 0 and
	// FLM_MAX_POS_KEYS represent	the first and last keys in the query's
	// result set, respectively. Numbers between these two endpoints represent
	// intervals between two keys that are adjacent in the array, but which
	// may have any number of intervening keys in the index.
	
	uiLowOffset = 0;
	uiHighOffset = pCursor->uiNumPosKeys - 1;
	for(;;)
	{
		if (uiLowOffset == uiHighOffset)
		{
			uiMidOffset = uiLowOffset;

			// Defect #84741 (fix after failing regression test -
			// zeroeth object was always returning position 1).
			// Must do final comparison to determine which side of
			// the positioning key our key falls on.  Remember,
			// the positioning key represents all keys that are
			// LESS THAN OR EQUAL to it.  Thus, if this key is
			// greater than it, we should use the next positioning
			// key.

			if ((flmPosKeyCompare( &pPosKeyArray[ uiMidOffset],
							&CompKey) < 0) &&
				 (uiMidOffset < pCursor->uiNumPosKeys - 1))
			{
				uiMidOffset++;
			}
			break;
		}
		
		uiMidOffset = (FLMUINT)((uiHighOffset + uiLowOffset) / 2);
	
		iCmp = flmPosKeyCompare( &pPosKeyArray[ uiMidOffset], &CompKey);

		if( iCmp < 0)
		{
			uiLowOffset = uiMidOffset + 1;
		}
		else if( iCmp > 0)
		{
			if( uiMidOffset == uiLowOffset)
			{
				break;
			}
			else
			{
				uiHighOffset = uiMidOffset - 1;
			}
		}
		else
		{
			break;
		}
	}

	// DEFECT 84741 -- the first object should only return a position of 1
	// if there are FLM_MAX_POS_KEYS positioning keys in the array.

	if (uiMidOffset == 0 ||
		 (uiMidOffset == 1 && 
		  pCursor->uiNumPosKeys == FLM_MAX_POS_KEYS + 1))
	{
		*puiPrcntPos = uiMidOffset;
	}
	else if (uiMidOffset == pCursor->uiNumPosKeys - 1)
	{
		*puiPrcntPos = FLM_MAX_POS_KEYS;
	}
	else if (uiMidOffset <= uiRFactor)
	{
		*puiPrcntPos = uiMidOffset * (uiIntervalSize + 1);
	}
	else if (uiRFactor)
	{
		*puiPrcntPos = uiRFactor * (uiIntervalSize + 1) +
							(uiMidOffset - uiRFactor) * uiIntervalSize;
	}
	else
	{
		*puiPrcntPos = uiMidOffset * uiIntervalSize;
	}
	
Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_GET_CONFIG, pDb, rc);
	}
	
	return( rc);
}

/****************************************************************************
Desc: Sets a query's position to a percentage represented by one of an array
		of positioning keys.
****************************************************************************/
RCODE flmCurSetPercentPos(
	CURSOR *			pCursor,
	FLMUINT			uiPrcntPos
	)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = NULL;
	FLMUINT			uiPrcntOffs;
	FLMUINT			uiIntervalSize;
	FLMUINT			uiRFactor;
	SUBQUERY *		pSubQuery = NULL;
	POS_KEY *		pPosKey;
	
	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}
	
	// Check the value for the percentage position.  Should be between
	// 0 and FLM_MAX_POS_KEYS.
	
	flmAssert( uiPrcntPos <= FLM_MAX_POS_KEYS);
	
	// Initialize some variables
	
	pCursor->uiLastRecID = 0;
	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// If no array of positioning keys exists in the subquery, set one up.

	if (!pCursor->uiNumPosKeys)
	{
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
			
		// If no positioning keys exist, either the index or the result set
		// is empty.  Return BOF or EOF.
		
		if (!pCursor->uiNumPosKeys)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}

	pSubQuery = pCursor->pSubQueryList;

Retry:

	// Calculate the percent position using the following rules:
	//	1) If the number of positioning keys is 1, the position is 0%.
	//	2) If the number of positioning keys is 2, the position is either 0% or
	//		FLM_MAX_POS_KEYS.
	// 3) If there are more than 2 positioning keys, calculate the interval into
	//		which the percentage position falls.

	if (pCursor->uiNumPosKeys == 1)
	{
		uiPrcntOffs = 0;
	}
	else
	{
		if (pCursor->uiNumPosKeys == 2)
		{
			uiIntervalSize = FLM_MAX_POS_KEYS;
			uiRFactor = 0;
		}
		else
		{
			uiIntervalSize = FLM_MAX_POS_KEYS / (pCursor->uiNumPosKeys - 1);
			uiRFactor = FLM_MAX_POS_KEYS % (pCursor->uiNumPosKeys - 1);
		}
		
		// Convert passed-in number to an array offset.

		if (uiPrcntPos)
		{
			if (uiPrcntPos == 0 || pCursor->uiNumPosKeys == FLM_MAX_POS_KEYS + 1)
			{
				uiPrcntOffs = uiPrcntPos;
			}
			else if( uiPrcntPos == FLM_MAX_POS_KEYS)
			{
				uiPrcntOffs = pCursor->uiNumPosKeys - 1;
			}
			else if( uiPrcntPos <= uiRFactor * (uiIntervalSize + 1))
			{
				uiPrcntOffs = uiPrcntPos / (uiIntervalSize + 1);
			}
			else
			{
				uiPrcntOffs = uiRFactor +
									(uiPrcntPos - (uiIntervalSize + 1) * uiRFactor) /
									uiIntervalSize;
			}
		}
		else
		{
			uiPrcntOffs = 0;
		}
	}
	pPosKey = &pCursor->pPosKeyArray [uiPrcntOffs];

	// If the keys were generated from the leaf level, we can
	// position directly to them.  If not, we must call the
	// positionToDomain routine.

	if (pCursor->bLeafLevel)
	{
		rc = pSubQuery->pFSIndexCursor->positionTo( pDb, pPosKey->pucKey,
					pPosKey->uiKeyLen, pPosKey->uiDrn);
	}
	else
	{
		rc = pSubQuery->pFSIndexCursor->positionToDomain( pDb,
					pPosKey->pucKey, pPosKey->uiKeyLen,
					DRN_TO_DOMAIN( pPosKey->uiDrn));
	}

	if (RC_BAD( rc))
	{
		RCODE	saveRc;

		if (rc != FERR_BOF_HIT && rc != FERR_EOF_HIT && rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
			
		// If the positioning key was not found, the database has undergone
		// significant change since the array of positioning keys was generated.
		// Try to regenerate the array and reposition.

		saveRc = rc;
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
		if (pCursor->pPosKeyArray [0].pucKey == NULL)
		{
			rc = saveRc;
			goto Exit;
		}
		goto Retry;
	}

	// Retrieve the current key and DRN from the index cursor.

	if (RC_BAD( rc = pSubQuery->pFSIndexCursor->currentKey( pDb,
							&pSubQuery->pRec, &pSubQuery->uiDrn)))
	{
		goto Exit;
	}
	pSubQuery->bFirstReference = FALSE;
	pSubQuery->uiCurrKeyMatch = FLM_TRUE;

	// These should have already been set by the call to currentKey.

	flmAssert( pSubQuery->pRec->getContainerID() == pCursor->uiContainer);
	flmAssert( pSubQuery->pRec->getID() == pSubQuery->uiDrn);

	pSubQuery->bRecIsAKey = TRUE;
	
	// If we got this far, the positioning operation was a success.  Set
	// the query return code to success so it doesn't mess up subsequent
	// read operations.

	pCursor->uiLastRecID = pSubQuery->uiDrn;
	pCursor->rc = FERR_OK;
	pCursor->uiLastPrcntPos = uiPrcntPos;
	pCursor->uiLastPrcntOffs = uiPrcntOffs;
	pCursor->bUsePrcntPos = TRUE;

Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_CONFIG, pDb, rc);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Build the from and until keys given a field list with operators and
		values and an index.
Note:	The knowledge of query definitions is limited in these routines.
****************************************************************************/
RCODE flmBuildFromAndUntilKeys(
	IXD *					pIxd,
	QPREDICATE **		ppQPredicate,
	FLMBYTE *			pFromKey,
	FLMUINT *			puiFromKeyLen,
	FLMBYTE *			pUntilKey,
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbDoRecMatch,
	FLMBOOL *			pbDoKeyMatch,
	FLMBOOL *			pbExclusiveUntilKey)
{
	RCODE					rc = FERR_OK;
	QPREDICATE *		pCurPred;
	IFD *					pIfd = pIxd->pFirstIfd;
	FLMUINT				uiLanguage = pIxd->uiLanguage;
	FLMUINT				uiIfdCnt = pIxd->uiNumFlds;
	FLMUINT				uiFromKeyPos = 0;
	FLMUINT				uiUntilKeyPos = 0;
	FLMBOOL				bFromAtFirst;
	FLMBOOL				bUntilAtEnd;
	FLMBOOL				bDataTruncated;
	FLMBOOL				bDoneBuilding;
	FLMBOOL				bMustNotDoKeyMatch = FALSE;
	FLMBOOL				bDoKeyMatch = FALSE;
	FLMBOOL				bOriginalCharsLost;
	FLMBOOL				bDBCSLanguage;
	FLMBYTE				ucNumberBuf [ F_MAX_NUM64_BUF + 1];
	FLMUINT				uiTempLen;
	FLMUINT				uiMaxKeySize;

	bDataTruncated = FALSE;
	bDoneBuilding = FALSE;
	*puiFromKeyLen = 0; 
	*puiUntilKeyLen = 0;
	uiFromKeyPos = 0;
	uiUntilKeyPos = 0;
	*pbExclusiveUntilKey = TRUE;

	bDBCSLanguage = (uiLanguage >= FLM_FIRST_DBCS_LANG) && 
						 (uiLanguage <= FLM_LAST_DBCS_LANG) 
						 		? TRUE 
								: FALSE;
		
	uiMaxKeySize = (pIxd->uiContainerNum) 
							? MAX_KEY_SIZ 
							: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);
		
	for (; !bDoneBuilding && uiIfdCnt--; ppQPredicate++, pIfd++)
	{

		// Add the compound marker if not the first piece.

		if (pIfd->uiCompoundPos)
		{
			IFD *		pPrevIfd = (pIfd - 1);

			// Add the compound markers for this key piece.

			if (bDBCSLanguage &&
				 (IFD_GET_FIELD_TYPE( pPrevIfd) == FLM_TEXT_TYPE) &&
				 (!((pPrevIfd)->uiFlags & IFD_CONTEXT)))
			{
				pFromKey[uiFromKeyPos++] = 0;
				pUntilKey[uiUntilKeyPos++] = 0;
			}

			pFromKey[uiFromKeyPos++] = COMPOUND_MARKER;
			pUntilKey[uiUntilKeyPos++] = COMPOUND_MARKER;
		}

		bFromAtFirst = bUntilAtEnd = FALSE;
		pCurPred = *ppQPredicate;

		if (!pCurPred)
		{

			// There is not a predicate that matches this compound key piece.
			//
			// Done processing, yet may need to look for a predicate that
			// will force a doKeyMatch or a doRecMatch.

			if (RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE, pFromKey,
						  &uiFromKeyPos, TRUE, pUntilKey, &uiUntilKeyPos, TRUE, NULL,
						  0, &bDataTruncated, &bDoneBuilding)))
			{
				goto Exit;
			}

			continue;
		}

		// Handle special cases for indexing context and/or exists
		// predicate.

		else if (pIfd->uiFlags & IFD_CONTEXT)
		{

			// Indexed only the TAG. Simple to set the tag as the key.

			if (RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE, pFromKey,
						  &uiFromKeyPos, FALSE, pUntilKey, &uiUntilKeyPos, FALSE,
						  NULL, 0, &bDataTruncated, &bDoneBuilding)))
			{
				goto Exit;
			}

			// If we don't have an exists predicate we need to read the
			// record.

			if (pCurPred->eOperator != FLM_EXISTS_OP)
			{
				bMustNotDoKeyMatch = TRUE;
			}

			continue;
		}
		else
		{
			FLMBOOL		bMatchedBadOperator = FALSE;
			
			switch (pCurPred->eOperator)
			{
				case FLM_EXISTS_OP:
				case FLM_NE_OP:
				{
					bMatchedBadOperator = TRUE;
					bUntilAtEnd = TRUE;
					bFromAtFirst = TRUE;
					break;
				}
				
				default:
				{
					if (pCurPred->bNotted)
					{
						bMatchedBadOperator = TRUE;
						bUntilAtEnd = TRUE;
						bFromAtFirst = TRUE;
					}
					break;
				}
			}

			if (bMatchedBadOperator)
			{

				// Does exist is a FIRST to LAST for this piece.

				if (RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE,
							  pFromKey, &uiFromKeyPos, bFromAtFirst, pUntilKey,
							  &uiUntilKeyPos, bUntilAtEnd, NULL, 0, &bDataTruncated,
							  &bDoneBuilding)))
				{
					goto Exit;
				}

				continue;
			}
		}

		switch (IFD_GET_FIELD_TYPE( pIfd))
		{

			// Build TEXT type piece

			case FLM_TEXT_TYPE:
			{
				FLMBOOL		bCaseInsensitive;
				FLMBOOL		bDoFirstSubstring;
				FLMBOOL		bDoMatchBegin;
				FLMBOOL		bDoSubstringSearch;
				FLMBOOL		bTrailingWildcard;
				FLMBYTE *	pValue = (FLMBYTE *) pCurPred->pVal->val.pucBuf;
				FLMUINT		uiValueLen = pCurPred->pVal->uiBufLen;
				
				bCaseInsensitive = (FLMBOOL)((pCurPred->pVal->uiFlags & FLM_COMP_CASE_INSENSITIVE) 
														? TRUE 
														: FALSE);
														
				bDoFirstSubstring = (FLMBOOL)((pIfd->uiFlags & IFD_SUBSTRING) 
														? TRUE 
														: FALSE);
				

				bDoMatchBegin = FALSE;
				bDoSubstringSearch = FALSE;
				bTrailingWildcard = FALSE;
				
				switch (pCurPred->eOperator)
				{

					// The difference between MATCH and EQ_OP is that EQ does not
					// support wildcards inbedded in the search key.

					case FLM_MATCH_OP:
					case FLM_MATCH_BEGIN_OP:
					{
						if (pCurPred->eOperator == FLM_MATCH_BEGIN_OP)
						{
							bDoKeyMatch = bDoMatchBegin = TRUE;
						}

						if (pCurPred->pVal->uiFlags & FLM_COMP_WILD)
						{
							if (!bDoFirstSubstring)
							{
								FLMBOOL	bFoundWildcard = flmFindWildcard( pValue,
																						&uiValueLen);

								bDoKeyMatch = TRUE;

								if (pCurPred->eOperator == FLM_MATCH_OP)
								{
									bTrailingWildcard = bDoMatchBegin = bFoundWildcard;
								}
								else
								{
									bTrailingWildcard = bDoMatchBegin = TRUE;
								}
							}
							else
							{

								// If this is a substring index look for a better
								// 'contains' string to search for. We don't like
								// "A*BCDEFG" searches.

								bTrailingWildcard = 
									(pCurPred->eOperator == FLM_MATCH_BEGIN_OP) 
											? TRUE 
											: FALSE;

								if (flmSelectBestSubstr( &pValue, &uiValueLen,
													pIfd->uiFlags, &bTrailingWildcard))
								{
									bDoMatchBegin = bTrailingWildcard;
									bMustNotDoKeyMatch = TRUE;
									bDoFirstSubstring = FALSE;
								}
								else if (bTrailingWildcard)
								{
									bDoKeyMatch = bDoMatchBegin = TRUE;
								}
							}
						}
						break;
					}
					
					case FLM_CONTAINS_OP:
					case FLM_MATCH_END_OP:
					{

						// Normal text index this piece goes from first to last.

						if (!bDoFirstSubstring)
						{
							bFromAtFirst = TRUE;
							bUntilAtEnd = TRUE;
						}
						else
						{
							bDoFirstSubstring = TRUE;
							bDoSubstringSearch = TRUE;

							// SPACE/Hyphen rules on SUBSTRING index. If the search
							// string starts with " _asdf" then we must do a record
							// match so "Z asdf" matches and "Zasdf" doesn't. We
							// won't touch key match even though it MAY return
							// FLM_TRUE when in fact the key may or may not match.
							//
							// VISIT: MatchBegin and Contains could also optimize
							// the trailing space by adding the space ONLY to the
							// UNTIL key.

							if (uiValueLen && 
								 ((*pValue == ASCII_SPACE && 
									(pIfd->uiFlags & IFD_MIN_SPACES)) || 
								(*pValue == ASCII_UNDERSCORE && 
									(pIfd->uiFlags & IFD_NO_UNDERSCORE))))
							{
								*pbDoRecMatch = TRUE;
							}

							// Take the flags from the pVal and NOT from the
							// predicate.

							if (pCurPred->pVal->uiFlags & FLM_COMP_WILD)
							{

								// Select the best substring. The case of
								// "A*BCD*E*FGHIJKLMNOP" will look for "FGHIJKLMNOP".
								// and TURN OFF doKeyMatch and SET doRecMatch.

								bTrailingWildcard = 
									(pCurPred->eOperator == FLM_CONTAINS_OP) 
											? TRUE 
											: FALSE;

								if (flmSelectBestSubstr( &pValue, &uiValueLen,
											pIfd->uiFlags, &bTrailingWildcard))
								{
									bDoMatchBegin = bTrailingWildcard;
									bMustNotDoKeyMatch = TRUE;
									bDoFirstSubstring = FALSE;
								}

								if (bTrailingWildcard)
								{
									bDoKeyMatch = bDoMatchBegin = TRUE;
								}
							}

							if (bDoFirstSubstring)
							{

								// Setting bDoMatchBegin creates a UNTIL key with
								// trailing 0xFF values.

								if (pCurPred->eOperator == FLM_CONTAINS_OP)
								{
									bDoKeyMatch = TRUE;
									bDoMatchBegin = TRUE;
								}
							}

							// Special case: Single character contains/MEnd in a
							// substr ix.

							if (!bDBCSLanguage &&
								 flmCountCharacters( pValue, uiValueLen, 2,
										pIfd->uiFlags) < 2)
							{
								bDoKeyMatch = bFromAtFirst = bUntilAtEnd = TRUE;
							}
						}
						
						break;
					}

					// No wild card support for the operators below.

					case FLM_EQ_OP:
					{
						break;
					}
					
					case FLM_GE_OP:
					case FLM_GT_OP:
					{
						bUntilAtEnd = TRUE;
						break;
					}
					
					case FLM_LE_OP:
					{
						bFromAtFirst = TRUE;
						break;
					}
					
					case FLM_LT_OP:
					{
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					}
					
					default:
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
				}

				// If index is case insensitive, but search is case sensitive
				// we must NOT do a key match - we would fail things we should
				// not be failing.

				if ((pIfd->uiFlags & IFD_UPPER) && !bCaseInsensitive)
				{
					bMustNotDoKeyMatch = TRUE;
				}

				if (RC_BAD( rc = flmAddTextPiece( uiMaxKeySize, pIfd,
							  bCaseInsensitive, bDoMatchBegin, bDoFirstSubstring,
							  bTrailingWildcard, pFromKey, &uiFromKeyPos, bFromAtFirst,
							  pUntilKey, &uiUntilKeyPos, bUntilAtEnd, pValue,
							  uiValueLen, &bDataTruncated, &bDoneBuilding, 
							  &bOriginalCharsLost)))
				{
					goto Exit;
				}

				if (bOriginalCharsLost)
				{
					bMustNotDoKeyMatch = TRUE;
				}
				break;
			}

			// Build NUMBER or CONTEXT type piece VISIT: Add a true number type
			// so we don't have to build a NODE.

			case FLM_NUMBER_TYPE:
			case FLM_CONTEXT_TYPE:
			{
				switch (pCurPred->pVal->eType)
				{
					case FLM_INT32_VAL:
					{
						FLMINT	iValue = (FLMINT)pCurPred->pVal->val.i32Val;
						if (pCurPred->eOperator == FLM_GT_OP)
						{
							iValue++;
						}

						if (IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							uiTempLen = sizeof( ucNumberBuf);
							(void)FlmINT2Storage( iValue, &uiTempLen, ucNumberBuf);
						}
						else
						{
							UD2FBA( (FLMUINT32)iValue, ucNumberBuf);
							uiTempLen = 4;
						}
						break;
					}

					case FLM_INT64_VAL:
					{
						FLMINT64	i64Value = pCurPred->pVal->val.i64Val;
						if (pCurPred->eOperator == FLM_GT_OP)
						{
							i64Value++;
						}

						if (IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							uiTempLen = sizeof( ucNumberBuf);
							(void)FlmINT64ToStorage( i64Value, &uiTempLen, ucNumberBuf);
						}
						else
						{
							UD2FBA( (FLMUINT32)i64Value, ucNumberBuf);
							uiTempLen = 4;
						}
						break;
					}

					case FLM_UINT32_VAL:
					case FLM_REC_PTR_VAL:
					{
						FLMUINT	uiValue = (FLMUINT)pCurPred->pVal->val.ui32Val;
						if (pCurPred->eOperator == FLM_GT_OP)
						{
							uiValue++;
						}

						if (IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							uiTempLen = sizeof( ucNumberBuf);
							(void)FlmUINT2Storage( uiValue, &uiTempLen, ucNumberBuf);
						}
						else
						{
							UD2FBA( (FLMUINT32)uiValue, ucNumberBuf);
							uiTempLen = 4;
						}
						break;
					}

					case FLM_UINT64_VAL:
					{
						FLMUINT64	ui64Value = pCurPred->pVal->val.ui64Val;
						
						if (pCurPred->eOperator == FLM_GT_OP)
						{
							ui64Value++;
						}

						if (IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							uiTempLen = sizeof( ucNumberBuf);
							(void)FlmUINT64ToStorage( ui64Value, &uiTempLen, ucNumberBuf);
						}
						else
						{
							UD2FBA( (FLMUINT32)ui64Value, ucNumberBuf);
							uiTempLen = 4;
						}
						break;
					}

					default:
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
				}

				switch (pCurPred->eOperator)
				{
					case FLM_EQ_OP:
					{
						break;
					}
					
					case FLM_GE_OP:
					case FLM_GT_OP:
					{
						bUntilAtEnd = TRUE;
						break;
					}
					
					case FLM_LE_OP:
					{
						bFromAtFirst = TRUE;
						break;
					}
					
					case FLM_LT_OP:
					{
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					}
					
					default:
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
				}

				if (RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE, 
							pFromKey, &uiFromKeyPos, bFromAtFirst, pUntilKey, 
							&uiUntilKeyPos, bUntilAtEnd, ucNumberBuf, 
							uiTempLen, &bDataTruncated, &bDoneBuilding)))
				{
					goto Exit;
				}
				
				break;
			}

			case FLM_BINARY_TYPE:
			{
				FLMBOOL	bMatchBegin = FALSE;

				switch (pCurPred->eOperator)
				{
					case FLM_MATCH_BEGIN_OP:
					{
						bMatchBegin = TRUE;
						break;
					}
					
					case FLM_EQ_OP:
					{
						break;
					}
					
					case FLM_GE_OP:
					{
						bUntilAtEnd = TRUE;
						break;
					}
					
					case FLM_GT_OP:
					{
						bUntilAtEnd = TRUE;
						bDoKeyMatch = TRUE;
						break;
					}
					
					case FLM_LE_OP:
					{
						bFromAtFirst = TRUE;
						break;
					}
					
					case FLM_LT_OP:
					{
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					}
					
					default:
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
				}

				if (RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, bMatchBegin,
							  pFromKey, &uiFromKeyPos, bFromAtFirst, pUntilKey,
							  &uiUntilKeyPos, bUntilAtEnd, pCurPred->pVal->val.pucBuf,
							  pCurPred->pVal->uiBufLen, &bDataTruncated, 
							  &bDoneBuilding)))
				{
					goto Exit;
				}
				
				break;
			}

			default:
			{
				flmAssert( 0);
				break;
			}
		}

		if (bDataTruncated)
		{
			bMustNotDoKeyMatch = TRUE;
		}
	}

	// Really rare case where FROM/UNTIL keys are exactly the same.

	if (!bDoneBuilding && (uiIfdCnt + 1 == 0) &&
		 uiUntilKeyPos < uiMaxKeySize - 2)
	{

		// Always make the until key exclusive. pbExclusiveUntilKey = FALSE;

		pUntilKey[uiUntilKeyPos++] = 0xFF;
		pUntilKey[uiUntilKeyPos++] = 0xFF;
	}

Exit:

	if (bMustNotDoKeyMatch)
	{
		*pbDoKeyMatch = FALSE;
		*pbDoRecMatch = TRUE;
	}
	else if (bDoKeyMatch || !pIxd->uiContainerNum)
	{
		*pbDoKeyMatch = TRUE;
	}

	// Special case for building FIRST/LAST keys.

	if (!uiFromKeyPos)
	{
		*pFromKey = '\0';
		uiFromKeyPos = 1;
	}

	if (!uiUntilKeyPos)
	{
		f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
		uiUntilKeyPos = uiMaxKeySize - 2;
	}

	*puiFromKeyLen = uiFromKeyPos;
	*puiUntilKeyLen = uiUntilKeyPos;
	
	return (rc);
}

/****************************************************************************
Desc:	Truncate the length of the text buffer on the first wild card.
****************************************************************************/
FSTATIC FLMBOOL flmFindWildcard(
	FLMBYTE *		pVal,
	FLMUINT *		puiCharPos)
{
	FLMBOOL		bHaveChar = FALSE;
	FLMBYTE *	pSaveVal = pVal;
	FLMUINT		uiObjLength;
	FLMUINT		uiLen = *puiCharPos;

	for (; *pVal; pVal += uiObjLength, uiLen = (uiObjLength < uiLen) 
														? uiLen - uiObjLength 
														: 0)
	{
		switch ((FLMUINT) (flmTextObjType( *pVal)))
		{
			case ASCII_CHAR_CODE:	// 0nnnnnnn
			{
				if (*pVal == ASCII_WILDCARD)
				{
					bHaveChar = TRUE;
					goto Exit;
				}

				uiObjLength = 1;

				// Check for '*' or '\\' after an escape character.

				if (*pVal == ASCII_BACKSLASH &&
					 (*(pVal + 1) == ASCII_WILDCARD || 
					  *(pVal + 1) == ASCII_BACKSLASH))
				{
					uiObjLength++;
				}
				
				break;
			}
			
			case WHITE_SPACE_CODE:	// 110nnnnn
			{
				uiObjLength = 1;
				break;
			}
			
			case CHAR_SET_CODE:		// 10nnnnnn
			case UNK_EQ_1_CODE:
			case OEM_CODE:
			{
				uiObjLength = 2;
				break;
			}
			
			case UNICODE_CODE:		// Unconvertable UNICODE code
			case EXT_CHAR_CODE:
			{
				uiObjLength = 3;
				break;
			}
			
			case UNK_GT_255_CODE:
			{
				uiObjLength = 1 + sizeof(FLMUINT16) + FB2UW( pVal + 1);
				break;
			}
			
			case UNK_LE_255_CODE:
			{
				uiObjLength = 2 + (FLMUINT16) * (pVal + 1);
				break;
			}
			
			default:
			{
				// should NEVER happen: bug if does
				
				flmAssert( 0);
				uiObjLength = 1;
				break;
			}
		}
	}

Exit:

	*puiCharPos = (FLMUINT) (pVal - pSaveVal);
	return (bHaveChar);
}

/****************************************************************************
Desc:	Add a key piece to the from and until key. Text fields are not
		handled in this routine because of their complexity.
Note:	The goal of this code is to build a the collated compound piece for
		the 'from' and 'until' key only once instead of twice.
****************************************************************************/
FSTATIC RCODE flmAddKeyPiece(
	FLMUINT			uiMaxKeySize,
	IFD *				pIfd,
	FLMBOOL			bDoMatchBegin,
	FLMBYTE *		pFromKey,
	FLMUINT *		puiFromKeyPos,
	FLMBOOL			bFromAtFirst,
	FLMBYTE *		pUntilKey,
	FLMUINT *		puiUntilKeyPos,
	FLMBOOL			bUntilAtEnd,
	FLMBYTE *		pBuf,
	FLMUINT			uiBufLen,
	FLMBOOL *		pbDataTruncated,
	FLMBOOL *		pbDoneBuilding)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFromKeyPos = *puiFromKeyPos;
	FLMUINT		uiUntilKeyPos = *puiUntilKeyPos;
	FLMBYTE *	pDestKey;
	FLMUINT		uiDestKeyLen;

	if (pIfd->uiCompoundPos == 0 && bFromAtFirst && bUntilAtEnd)
	{

		// Special case for the first piece - FIRST to LAST - zero length
		// keys. so that the caller can get the number of references for the
		// entire index. VISIT: May want to set the from key to have 1 byte
		// and set high values for the until key. This way the caller never
		// checks this special case.

		*pbDoneBuilding = TRUE;
		goto Exit;
	}

	// Handle the CONTEXT exception here - this is not done in kyCollate.

	if (pIfd->uiFlags & IFD_CONTEXT)
	{
		pFromKey[uiFromKeyPos] = KY_CONTEXT_PREFIX;
		f_UINT16ToBigEndian( (FLMUINT16) pIfd->uiFldNum, &pFromKey[uiFromKeyPos + 1]);
		uiFromKeyPos += KY_CONTEXT_LEN;

		if (uiUntilKeyPos + KY_CONTEXT_LEN < uiMaxKeySize)
		{
			pUntilKey[uiUntilKeyPos] = KY_CONTEXT_PREFIX;
			f_UINT16ToBigEndian( (FLMUINT16) pIfd->uiFldNum, &pUntilKey[uiUntilKeyPos + 1]);
			uiUntilKeyPos += KY_CONTEXT_LEN;
		}

		goto Exit;
	}

	if (bFromAtFirst)
	{
		if (bUntilAtEnd)
		{

			// Not the first piece and need to go from first to last.

			*pbDoneBuilding = TRUE;

			if (uiUntilKeyPos < uiMaxKeySize - 2)
			{
				if (uiUntilKeyPos > 0)
				{

					// Instead of filling the key with 0xFF, increment the
					// marker.

					pUntilKey[uiUntilKeyPos - 1]++;
				}
				else
				{
					f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
					uiUntilKeyPos = uiMaxKeySize - 2;
				}
			}

			goto Exit;
		}

		if (uiUntilKeyPos >= uiMaxKeySize - 2)
		{
			goto Exit;
		}

		// Have a LAST key but no FROM key.

		pDestKey = pUntilKey + uiUntilKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiUntilKeyPos;
	}
	else
	{
		pDestKey = pFromKey + uiFromKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiFromKeyPos;
	}

	rc = KYCollateValue( pDestKey, &uiDestKeyLen, (FLMBYTE*) pBuf, uiBufLen,
							  pIfd->uiFlags, pIfd->uiLimit, NULL, NULL, 0, TRUE, FALSE,
							  FALSE, pbDataTruncated);

	if (rc == FERR_CONV_DEST_OVERFLOW)
	{
		rc = FERR_OK;
	}
	else if (RC_BAD( rc))
	{
		goto Exit;
	}

	// If we just built the FROM key, we may want to copy to the UNTIL key.

	if (pDestKey == pFromKey + uiFromKeyPos)
	{
		uiFromKeyPos += uiDestKeyLen;

		// Unless the UNTIL key is full, the length is at or less than FROM
		// key.

		if (!bUntilAtEnd)
		{
			if (uiUntilKeyPos + uiDestKeyLen <= uiMaxKeySize)
			{
				f_memcpy( &pUntilKey[uiUntilKeyPos], pDestKey, uiDestKeyLen);
				uiUntilKeyPos += uiDestKeyLen;
			}

			if (bDoMatchBegin)
			{
				flmAssert( IFD_GET_FIELD_TYPE( pIfd) == FLM_BINARY_TYPE);

				if (uiUntilKeyPos < MAX_KEY_SIZ - 2)
				{

					// Optimization - only need to set a single byte to 0xFF.
					// We can do this because this routine does not deal with
					// text key pieces and binary, number and context will
					// never have 0xFF bytes.

					pUntilKey[uiUntilKeyPos++] = 0xFF;
				}

				// We don't need to set *pbDoneBuilding = TRUE, because we
				// may be able to continue building the from key

			}
		}
		else
		{
			if (uiUntilKeyPos > 0)
			{

				// Instead of filling the key with 0xFF, increment the marker.

				pUntilKey[uiUntilKeyPos - 1]++;
			}
			else
			{

				// Optimization - only need to set a single byte to 0xFF. We
				// can do this because this routine does not deal with text
				// key pieces and binary, number and context will never have
				// 0xFF bytes.

				flmAssert( IFD_GET_FIELD_TYPE( pIfd) != FLM_TEXT_TYPE);

				*pUntilKey = 0xFF;
				uiUntilKeyPos++;
			}
		}
	}
	else
	{
		uiUntilKeyPos += uiDestKeyLen;
	}

Exit:

	// Set the FROM and UNTIL key length return values.

	*puiFromKeyPos = uiFromKeyPos;
	*puiUntilKeyPos = uiUntilKeyPos;
	return (rc);
}

/****************************************************************************
Desc:	Add a text piece to the from and until key.
****************************************************************************/
FSTATIC RCODE flmAddTextPiece(
	FLMUINT		uiMaxKeySize,
	IFD *			pIfd,
	FLMBOOL		bCaseInsensitive,
	FLMBOOL		bDoMatchBegin,
	FLMBOOL		bDoFirstSubstring,
	FLMBOOL		bTrailingWildcard,
	FLMBYTE *	pFromKey,
	FLMUINT *	puiFromKeyPos,
	FLMBOOL		bFromAtFirst,
	FLMBYTE *	pUntilKey,
	FLMUINT *	puiUntilKeyPos,
	FLMBOOL		bUntilAtEnd,
	FLMBYTE *	pBuf,
	FLMUINT		uiBufLen,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbDoneBuilding,
	FLMBOOL *	pbOriginalCharsLost)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFromKeyPos = *puiFromKeyPos;
	FLMUINT		uiUntilKeyPos = *puiUntilKeyPos;
	FLMUINT		uiLanguage = pIfd->pIxd->uiLanguage;
	FLMBYTE *	pDestKey;
	FLMUINT		uiDestKeyLen;
	FLMUINT		uiCollationLen = 0;
	FLMUINT		uiCaseLen;
	FLMBOOL		bIsDBCS;

	bIsDBCS = (uiLanguage >= FLM_FIRST_DBCS_LANG && uiLanguage <= FLM_LAST_DBCS_LANG) 
											? TRUE 
											: FALSE;
	*pbOriginalCharsLost = FALSE;
	if (pIfd->uiCompoundPos == 0 && bFromAtFirst && bUntilAtEnd)
	{

		// Special case for the first piece - FIRST to LAST - zero length
		// keys. so that the caller can get the number of references for the
		// entire index. VISIT: May want to set the from key to have 1 byte
		// and set high values for the until key. This way the caller never
		// checks this special case.

		*pbDoneBuilding = TRUE;
		goto Exit;
	}

	if (bFromAtFirst)
	{
		if (bUntilAtEnd)
		{

			// Not the first piece and need to go from first to last.

			*pbDoneBuilding = TRUE;

			if (uiUntilKeyPos < uiMaxKeySize - 2)
			{

				// Instead of filling the key with 0xFF, increment the marker.

				pUntilKey[uiUntilKeyPos - 1]++;
			}

			goto Exit;
		}

		if (uiUntilKeyPos >= uiMaxKeySize - 2)
		{
			goto Exit;
		}

		// Have a LAST key but no FROM key.

		pDestKey = pUntilKey + uiUntilKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiUntilKeyPos;
	}
	else	// Handle below if UNTIL key is LAST.
	{
		pDestKey = pFromKey + uiFromKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiFromKeyPos;
	}

	// Add IFD_ESC_CHAR to the ifd flags because the search string must
	// have BACKSLASHES and '*' escaped.

	rc = KYCollateValue( pDestKey, &uiDestKeyLen, (FLMBYTE*) pBuf, uiBufLen,
							  pIfd->uiFlags | IFD_ESC_CHAR, pIfd->uiLimit,
							  &uiCollationLen, &uiCaseLen, uiLanguage, TRUE,
							  bDoFirstSubstring, bTrailingWildcard, pbDataTruncated,
							  pbOriginalCharsLost);
							  
	if (rc == FERR_CONV_DEST_OVERFLOW)
	{
		rc = FERR_OK;
	}
	else if (RC_BAD( rc))
	{
		goto Exit;
	}

	if (pIfd->uiFlags & IFD_POST)
	{
		uiDestKeyLen -= uiCaseLen;
	}
	else
	{

		// Special case: The index is NOT an upper index and the search is
		// case-insensitive. The FROM key must have lower case values and
		// the UNTIL must be the upper case values. This will be true for
		// Asian indexes also.

		if (uiDestKeyLen &&
			 (bIsDBCS || (!(pIfd->uiFlags & IFD_UPPER) && bCaseInsensitive)))
		{

			// Subtract off all but the case marker. Remember that for DBCS
			// (Asian) the case marker is two bytes.

			uiDestKeyLen -= (uiCaseLen - ((FLMUINT) (bIsDBCS ? 2 : 1)));

			// NOTE: SC_LOWER is only used in GREEK indexes, which is why we
			// use it here instead of SC_MIXED.

			pDestKey[uiDestKeyLen - 1] = 
				(FLMBYTE) ((uiLanguage != (FLMUINT) FLM_GR_LANG) 
						? COLL_MARKER | SC_MIXED 
						: COLL_MARKER | SC_LOWER);

			// Once the FROM key has been approximated, we are done building.

			*pbDoneBuilding = TRUE;
		}
	}

	// Copy or move pieces of the FROM key into the UNTIL key.

	if (pDestKey == pFromKey + uiFromKeyPos)
	{
		if (uiUntilKeyPos < uiMaxKeySize - 2)
		{
			if (!bUntilAtEnd)
			{
				if (bDoMatchBegin)
				{
					if (uiCollationLen)
					{
						f_memcpy( &pUntilKey[uiUntilKeyPos], pDestKey, uiCollationLen);
						uiUntilKeyPos += uiCollationLen;
					}

					// Fill the rest of the key with high values.

					f_memset( &pUntilKey[uiUntilKeyPos], 0xFF,
								(uiMaxKeySize - 2) - uiUntilKeyPos);
					uiUntilKeyPos = uiMaxKeySize - 2;

					// Don't need to set the done building flag to TRUE.

				}
				else if (uiDestKeyLen)
				{
					if (!bDoFirstSubstring)
					{
						f_memcpy( &pUntilKey[uiUntilKeyPos], pDestKey, uiDestKeyLen);
						uiUntilKeyPos += uiDestKeyLen;
					}
					else
					{

						// Do two copies so that the first substring byte is
						// gone.

						f_memcpy( &pUntilKey[uiUntilKeyPos], pDestKey, uiCollationLen);
						uiUntilKeyPos += uiCollationLen;
						if (bIsDBCS)
						{
							uiCollationLen++;
						}

						uiCollationLen++;
						f_memcpy( &pUntilKey[uiUntilKeyPos], pDestKey + uiCollationLen,
									uiDestKeyLen - uiCollationLen);
						uiUntilKeyPos += (uiDestKeyLen - uiCollationLen);
					}

					// Special case again : raw case in index and search
					// comparison. Case has already been completely removed if
					// it is a post index, so no need to change the marker
					// byte.

					if (!(pIfd->uiFlags & IFD_POST) && (bIsDBCS || 
							(!(pIfd->uiFlags & IFD_UPPER) && bCaseInsensitive)))
					{

						// Add 1 to make sure the until key is higher than the
						// upper value.

						pUntilKey[uiUntilKeyPos - 1] = (COLL_MARKER | SC_UPPER) + 1;
					}
				}
			}
			else
			{
				if (uiUntilKeyPos > 0)
				{

					// Instead of filling the key with 0xFF, increment the
					// marker.

					pUntilKey[uiUntilKeyPos - 1]++;
				}
				else
				{

					// Keys can have 0xFF values in them, so it is not
					// sufficient to set only uiDestKeyLen bytes to 0xFF. We
					// must set the entire key.

					f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
					uiUntilKeyPos = uiMaxKeySize - 2;
				}
			}
		}

		uiFromKeyPos += uiDestKeyLen;
	}
	else
	{

		// We just built the UNTIL key. The FROM key doesn't need to be
		// built.

		uiUntilKeyPos += uiDestKeyLen;
	}

Exit:

	// Set the FROM and UNTIL keys

	*puiFromKeyPos = uiFromKeyPos;
	*puiUntilKeyPos = uiUntilKeyPos;
	return (rc);
}

/****************************************************************************
Desc:	Select the best substring for a CONTAINS or MATCH_END search. Look
		below for the algorithm.
****************************************************************************/
FSTATIC FLMBOOL flmSelectBestSubstr(
	FLMBYTE **		ppValue,
	FLMUINT *		puiValueLen,
	FLMUINT			uiIfdFlags,
	FLMBOOL *		pbTrailingWildcard)
{
	FLMBYTE *		pValue = *ppValue;
	FLMBYTE *		pCurValue;
	FLMBYTE *		pBest;
	FLMBOOL			bBestTerminatesWithWildCard = *pbTrailingWildcard;
	FLMUINT			uiCurLen;
	FLMUINT			uiBestNumChars;
	FLMUINT			uiBestValueLen;
	FLMUINT			uiWildcardPos = 0;
	FLMUINT			uiTargetNumChars;
	FLMUINT			uiNumChars;
	FLMBOOL			bNotUsingFirstOfString = FALSE;

	#define GOOD_ENOUGH_CHARS	16

	// There may not be any wildcards at all. Find the first one.

	if (flmFindWildcard( pValue, &uiWildcardPos))
	{
		bBestTerminatesWithWildCard = TRUE;
		pBest = pValue;
		pCurValue = pValue + uiWildcardPos + 1;
		uiCurLen = *puiValueLen - (uiWildcardPos + 1);

		uiBestValueLen = uiWildcardPos;
		uiBestNumChars = flmCountCharacters( pValue, uiWildcardPos,
														GOOD_ENOUGH_CHARS, uiIfdFlags);
		uiTargetNumChars = uiBestNumChars + uiBestNumChars;

		while (uiBestNumChars < GOOD_ENOUGH_CHARS && *pCurValue)
		{
			if (flmFindWildcard( pCurValue, &uiWildcardPos))
			{
				uiNumChars = flmCountCharacters( pCurValue, uiWildcardPos,
														  GOOD_ENOUGH_CHARS, uiIfdFlags);
				if (uiNumChars >= uiTargetNumChars)
				{
					pBest = pCurValue;
					uiBestValueLen = uiWildcardPos;
					uiBestNumChars = uiNumChars;
					uiTargetNumChars = uiNumChars + uiNumChars;
				}
				else
				{
					uiTargetNumChars += 2;
				}

				pCurValue = pCurValue + uiWildcardPos + 1;
				uiCurLen -= uiWildcardPos + 1;
			}
			else
			{

				// Check the last section that may or may not have trailing *.

				uiNumChars = flmCountCharacters( pCurValue, uiCurLen,
														  GOOD_ENOUGH_CHARS, uiIfdFlags);
				if (uiNumChars >= uiTargetNumChars)
				{
					pBest = pCurValue;
					uiBestValueLen = uiCurLen;
					bBestTerminatesWithWildCard = *pbTrailingWildcard;
				}
				break;
			}
		}

		if (pBest != *ppValue)
		{
			bNotUsingFirstOfString = TRUE;
		}

		*ppValue = pBest;
		*puiValueLen = uiBestValueLen;
		*pbTrailingWildcard = bBestTerminatesWithWildCard;
	}

	return (bNotUsingFirstOfString);
}

/****************************************************************************
Desc: Returns true if this text will generate a single characater key.
****************************************************************************/
FSTATIC FLMUINT flmCountCharacters(
	FLMBYTE *		pValue,
	FLMUINT 			uiValueLen,
	FLMUINT 			uiMaxToCount,
	FLMUINT 			uiIfdFlags)
{
	FLMUINT	uiNumChars = 0;
	FLMUINT	uiObjLength;

	for (uiObjLength = 0;
		  uiNumChars <= uiMaxToCount && uiValueLen;
		  pValue += uiObjLength, uiValueLen =
			  (uiValueLen >= uiObjLength) ? uiValueLen - uiObjLength : 0)
	{
		switch ((FLMUINT) (flmTextObjType( *pValue)))
		{
			case ASCII_CHAR_CODE:	// 0nnnnnnn
			{
				uiObjLength = 1;
				if (*pValue == ASCII_SPACE)
				{

					// Ignore if using space rules. VISIT: Need to address ending
					// spaces before a wildcard.

					if (uiIfdFlags & (IFD_MIN_SPACES | IFD_NO_SPACE))
					{
						break;
					}

					uiNumChars++;
				}
				else if (*pValue == ASCII_UNDERSCORE)
				{

					// Ignore if using the underscore space rules. VISIT: Need to
					// address ending spaces before a wildcard.

					if (uiIfdFlags & IFD_NO_UNDERSCORE)
					{
						break;
					}

					uiNumChars++;
				}
				else if (*pValue == ASCII_DASH)
				{
					if (uiIfdFlags & IFD_NO_DASH)
					{
						break;
					}

					uiNumChars++;
				}
				else if (*pValue == ASCII_BACKSLASH && 
						  (*(pValue + 1) == ASCII_WILDCARD || 
							*(pValue + 1) == ASCII_BACKSLASH))
				{
					uiObjLength++;
					uiNumChars++;
				}
				else
				{
					uiNumChars++;
				}
				
				break;
			}
			
			case WHITE_SPACE_CODE:	// 110nnnnn
			{
				uiObjLength = 1;
				uiNumChars++;
				break;
			}
			
			case CHAR_SET_CODE:		// 10nnnnnn
			case UNK_EQ_1_CODE:
			case OEM_CODE:
			{
				uiObjLength = 2;
				uiNumChars++;
				break;
			}
			
			case UNICODE_CODE:		// Unconvertable UNICODE code
			case EXT_CHAR_CODE:
			{
				uiObjLength = 3;
				uiNumChars++;
				break;
			}
			
			case UNK_GT_255_CODE:
			{
				uiObjLength = 1 + sizeof(FLMUINT16) + FB2UW( pValue + 1);
				break;
			}
			
			case UNK_LE_255_CODE:
			{
				uiObjLength = 2 + (FLMUINT16) * (pValue + 1);
				break;
			}
			
			default:
			{
				// should NEVER happen: bug if does
				
				flmAssert( 0);
				uiObjLength = 1;
				break;
			}
		}
	}

	return (uiNumChars);
}

