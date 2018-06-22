//-------------------------------------------------------------------------
// Desc:	Routines for calculating estimated costs for query optimization.
// Tabs:	3
//
// Copyright (c) 2000-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

extern FLMBYTE	SENLenArray[];

/***************************************************************************
Desc:		Compute the number of blocks between two stack positions.  
			These values may be estimated or actual.
*****************************************************************************/
RCODE FSComputeRecordBlocks(			// Returns WERR_OK or FERR_BTREE_ERROR
	BTSK *			pFromStack,			// [in] - be carefull not to change
												// anything in this structure.
	BTSK *			pUntilStack,		// [in]
	FLMUINT *		puiLeafBlocksBetween, // [out] blocks between the stacks
	FLMUINT *		puiTotalRecords,	// [out]	
	FLMBOOL *		pbTotalsEstimated)// [out] Set to TRUE when estimating.
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTotalRecords,
						uiTempRecordCount,
						uiEstRecordCount;
	FLMUINT			uiTotalBlocksBetween,
						uiEstBlocksBetween;
	FLMBYTE *		pBlk;
	
	uiTotalBlocksBetween = 0;
	*pbTotalsEstimated = FALSE;

	// Are the from and until positions in the same block? 

	if( pFromStack->uiBlkAddr == pUntilStack->uiBlkAddr)
	{
		rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
			pUntilStack->uiCurElm, &uiTotalRecords, NULL, NULL);
		goto Exit;
	}

	// Gather the counts in the from and until leaf blocks.

	if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
			pFromStack->uiBlkEnd, &uiTotalRecords, NULL, NULL)))
		goto Exit;

	if( RC_BAD( rc = FSBlockCounts( pUntilStack, BH_OVHD,
			pUntilStack->uiCurElm, &uiTempRecordCount, NULL, NULL)))
		goto Exit;

	uiTotalRecords += uiTempRecordCount;

	// Do the obvious check to see if the blocks are neighbors.

	pBlk = BLK_ELM_ADDR( pFromStack, BH_NEXT_BLK );
	if( FB2UD( pBlk ) == pUntilStack->uiBlkAddr)
	{
		goto Exit;
	}

	// Get (or estimate) the number of elements in the parent block.
	
	*pbTotalsEstimated = TRUE;
	if( RC_BAD( rc = FSBlockCounts( pFromStack, BH_OVHD, 
			pFromStack->uiBlkEnd, &uiEstRecordCount, NULL, NULL)))
		goto Exit;
	uiEstBlocksBetween = 1;

	for(;;)
	{
		FLMUINT		uiElementCount;
		FLMUINT		uiTempElementCount;
		FLMUINT		uiEstElementCount;

		// Go up a b-tree level and check out how far apart the elements are.
		pFromStack--;
		pUntilStack--;

		// Share the same block?
		if( pFromStack->uiBlkAddr == pUntilStack->uiBlkAddr)
		{
			if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
					pUntilStack->uiCurElm, NULL, &uiElementCount, NULL)))
				goto Exit;

			// Don't count the pFromStack current element.
			uiElementCount--;

			uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
			uiTotalRecords += uiEstRecordCount * uiElementCount;
			goto Exit;
		}

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
				pFromStack->uiBlkEnd, NULL, &uiElementCount, NULL)))
			goto Exit;

		// Don't count the first element.
		uiElementCount--;

		if( RC_BAD( rc = FSBlockCounts( pUntilStack, BH_OVHD,
				pUntilStack->uiCurElm, NULL, &uiTempElementCount, NULL)))
			goto Exit;

		uiElementCount += uiTempElementCount;

		uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
		uiTotalRecords += uiEstRecordCount * uiElementCount;

		// Do the obvious check to see if the blocks are neighbors.

		pBlk = BLK_ELM_ADDR( pFromStack, BH_NEXT_BLK );
		if( FB2UD( pBlk ) == pUntilStack->uiBlkAddr)
		{
			goto Exit;
		}


		// Recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		if( RC_BAD( rc = FSBlockCounts( pFromStack, BH_OVHD, 
				pFromStack->uiBlkEnd, NULL, &uiEstElementCount, NULL)))
			goto Exit;

		// Adjust the estimated key/ref count to be the counts from a complete
		// (not partial) block starting at this level going to the leaf.

		uiEstRecordCount *= uiEstElementCount;
		uiEstBlocksBetween *= uiEstElementCount;
	}

Exit:
	if( puiTotalRecords)
	{
		// Always include the UNTIL record.
		*puiTotalRecords = uiTotalRecords + 1;
	}
	if( puiLeafBlocksBetween)
	{
		*puiLeafBlocksBetween = uiTotalBlocksBetween;
	}
	return( rc);
}

/***************************************************************************
Desc:		Compute the key, element, reference and block counts 
			between two stack positions.  These values may be estimated or
			actual.
Notes:	There are two versions for this routine in the way of estimating
			how many keys and references there are in the unknown blocks between
			the from stack and the until stack.  The first version is implemented.
				The first version will estimate using the average number of keys
			and references in the pFromStack blocks.
				The second version will estimate using the average number of keys
			and references using pre-parsed stats gathered for the index in
			the LFILE.
*****************************************************************************/
RCODE FSComputeIndexCounts(					// Returns WERR_OK or FERR_BTREE_ERROR
	BTSK *			pFromStack,					// [in] - be carefull not to change
														// anything in this structure.
	BTSK *			pUntilStack,				// [in]
	FLMUINT *		puiLeafBlocksBetween, 	// [out] blocks between the stacks
	FLMUINT *		puiTotalKeys,				// [out] total number of keys inclusive
	FLMUINT *		puiTotalRefs,				// [out] total references inclusive
	FLMBOOL *		pbTotalsEstimated)		// [out] Set to TRUE when estimating.
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTotalKeys;
	FLMUINT			uiTempKeyCount;
	FLMUINT			uiEstKeyCount;
	FLMUINT			uiTotalRefs;
	FLMUINT			uiTempRefCount; 
	FLMUINT			uiEstRefCount;
	FLMUINT			uiTotalBlocksBetween;
	FLMUINT			uiEstBlocksBetween;
	FLMBYTE *		pBlk;
	
	uiTotalBlocksBetween = uiTotalKeys = uiTotalRefs = 0;
	*pbTotalsEstimated = FALSE;
	
	// Are the from and until positions in the same block? 

	if( pFromStack->uiBlkAddr == pUntilStack->uiBlkAddr)
	{
		rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
			pUntilStack->uiCurElm, &uiTotalKeys, NULL, &uiTotalRefs);

		goto Exit;
	}

	// Gather the counts in the from and until leaf blocks.

	if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
			pFromStack->uiBlkEnd, &uiTotalKeys, NULL, &uiTotalRefs)))
		goto Exit;

	if( RC_BAD( rc = FSBlockCounts( pUntilStack, BH_OVHD,
			pUntilStack->uiCurElm, &uiTempKeyCount, NULL, &uiTempRefCount)))
		goto Exit;

	uiTotalKeys += uiTempKeyCount;
	uiTotalRefs += uiTempRefCount;

	// Do the obvious check to see if the blocks are neighbors.

	pBlk = BLK_ELM_ADDR( pFromStack, BH_NEXT_BLK );
	if( FB2UD( pBlk ) == pUntilStack->uiBlkAddr)
	{
		goto Exit;
	}

	// Estimate number of keys/refs in the leaf block.
	// Estimate using just the left block.  The right block may be a right-most
	// block so will skew the results.
	//
	// Code for non-leaf child counts is easy - no need to estimate.
	
	if( (pFromStack-1)->uiBlkType != BHT_NON_LEAF_COUNTS)
	{
		*pbTotalsEstimated = TRUE;
		if( RC_BAD( rc = FSBlockCounts( pFromStack, BH_OVHD, 
				pFromStack->uiBlkEnd, &uiEstKeyCount, NULL, &uiEstRefCount)))
		{
			goto Exit;
		}
	}
	uiEstBlocksBetween = 1;

	for(;;)
	{
		FLMUINT		uiElementCount;
		FLMUINT		uiTempElementCount;
		FLMUINT		uiEstElementCount;
		FLMUINT		uiRefCount;
		FLMBYTE *	pCounts;

		// Go up a b-tree level and check out how far apart the elements are.
		
		pFromStack--;
		pUntilStack--;

		// Share the same block?
		
		if( pFromStack->uiBlkAddr == pUntilStack->uiBlkAddr)
		{
			if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
					pUntilStack->uiCurElm, NULL, &uiElementCount, &uiRefCount)))
			{
				goto Exit;
			}

			// Don't count the current element nor the ref counts.
			
			uiElementCount--;
			if( pFromStack->uiBlkType == BHT_NON_LEAF_COUNTS)
			{
				pCounts = pFromStack->pBlk + pFromStack->uiCurElm + BNE_CHILD_COUNT;
				uiRefCount -= FB2UD( pCounts);
				uiTotalRefs += uiRefCount;
				uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
				uiTotalKeys += uiEstKeyCount * uiElementCount;

				if( ((uiEstBlocksBetween != 1) && puiLeafBlocksBetween)
				 || puiTotalKeys)
				{
					*pbTotalsEstimated = TRUE;
				}
			}
			else
			{
				uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
				uiTotalKeys += uiEstKeyCount * uiElementCount;
				uiTotalRefs += uiEstRefCount * uiElementCount;
			}
			
			goto Exit;
		}

		// Gather the counts in the from and until non-leaf blocks.

		if( RC_BAD( rc = FSBlockCounts( pFromStack, pFromStack->uiCurElm,
				pFromStack->uiBlkEnd, NULL, &uiElementCount, &uiRefCount)))
		{
			goto Exit;
		}
		
		// Don't count the first element.
		
		uiElementCount--;

		if( pFromStack->uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			pCounts = pFromStack->pBlk + pFromStack->uiCurElm + BNE_CHILD_COUNT;
			uiRefCount -= FB2UD( pCounts);
			uiTotalRefs += uiRefCount;
		}

		if( RC_BAD( rc = FSBlockCounts( pUntilStack, BH_OVHD,
				pUntilStack->uiCurElm, NULL, &uiTempElementCount, &uiRefCount)))
		{
			goto Exit;
		}

		uiElementCount += uiTempElementCount;
		uiTotalBlocksBetween += uiEstBlocksBetween * uiElementCount;
		uiTotalKeys += uiEstKeyCount * uiElementCount;
		
		if( pUntilStack->uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			uiTotalRefs += uiRefCount;
			if( puiLeafBlocksBetween || puiTotalKeys)
			{
				*pbTotalsEstimated = TRUE;
			}
		}
		else
		{
			uiTotalRefs += uiEstRefCount * uiElementCount;
		}

		// Do the obvious check to see if the blocks are neighbors.

		pBlk = BLK_ELM_ADDR( pFromStack, BH_NEXT_BLK );
		if( FB2UD( pBlk ) == pUntilStack->uiBlkAddr)
		{
			goto Exit;
		}

		// We recompute the estimated element count on every b-tree level
		// because the compression is better the lower in the b-tree we go.

		if( RC_BAD( rc = FSBlockCounts( pFromStack, BH_OVHD, 
				pFromStack->uiBlkEnd, NULL, &uiEstElementCount, NULL)))
		{
			goto Exit;
		}

		// Adjust the estimated key/ref count to be the counts from a complete
		// (not partial) block starting at this level going to the leaf.

		uiEstKeyCount *= uiEstElementCount;
		uiEstRefCount *= uiEstElementCount;
		uiEstBlocksBetween *= uiEstElementCount;
	}

Exit:

	if( puiLeafBlocksBetween)
	{
		*puiLeafBlocksBetween = uiTotalBlocksBetween;
	}
	
	if( puiTotalKeys)
	{
		*puiTotalKeys = uiTotalKeys;
	}
	
	if( puiTotalRefs)
	{
		*puiTotalRefs = uiTotalRefs;
	}
	
	return( rc);
}

/***************************************************************************
Desc:		Returns the number of first keys (elements with the first flag),
			elements and references (for leaf blocks).  
*****************************************************************************/
RCODE FSBlockCounts(						// Returns WERR_OK currently.
	BTSK *			pStack,				// [in] - be careful not to change
												// anything in this structure.
	FLMUINT			uiFirstElement,	// [in] start at this element
	FLMUINT			uiLastElement,		// [in] Do not include reference counts
												// from this element.
	FLMUINT *		puiFirstKeyCount,	// [out] first key count or NULL
	FLMUINT *		puiElementCount,	// [out] element count or NULL
	FLMUINT *		puiRefCount)		// [out] reference count or NULL
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiFirstKeyCount;
	FLMUINT			uiElementCount;
	FLMUINT			uiRefCount;
	FLMBYTE *		pBlk;
	FLMBYTE *		pCounts;
	FLMBOOL			bHaveNonleafElementCounts;
	BTSK				tempStack;

	flmAssert( uiFirstElement <= uiLastElement);
	flmAssert( uiLastElement <= pStack->uiBlkEnd);

	uiFirstKeyCount = uiElementCount = uiRefCount = 0;

	// Set up the temporary stack so that the input stack doesn't get changed.

	tempStack.pBlk = pBlk = pStack->pBlk;
	tempStack.pSCache = pStack->pSCache;
	tempStack.uiBlkAddr = pStack->uiBlkAddr;
	FSBlkToStack( &tempStack);

	bHaveNonleafElementCounts = 
			(tempStack.uiBlkType == BHT_NON_LEAF_COUNTS) ? TRUE : FALSE;

	// Position to uiFirstElement (it could be bogus).
	
	tempStack.uiCurElm = uiFirstElement;

	// Loop gathering the statistics.
	
	while( tempStack.uiCurElm < uiLastElement)
	{
		uiElementCount++;

		if( puiFirstKeyCount)
		{
			if( pBlk[ tempStack.uiCurElm ] & BBE_FIRST_FLAG)
			{
				uiFirstKeyCount++;
			}
		}
		
		if( puiRefCount)
		{
			if( !bHaveNonleafElementCounts)
			{
				uiRefCount += FSElementRefCount( &tempStack);
			}
			else
			{
				pCounts = pBlk + tempStack.uiCurElm + BNE_CHILD_COUNT;
				uiRefCount += FB2UD( pCounts);
			}
		}

		// Next element.
		
		if( FSBlkNextElm( &tempStack) == FERR_BT_END_OF_DATA)
		{
			break;
		}
	}
	
	if( puiFirstKeyCount)
	{
		*puiFirstKeyCount = uiFirstKeyCount;
	}
	
	if( puiElementCount)
	{
		*puiElementCount = uiElementCount;
	}
	
	if( puiRefCount)
	{
		*puiRefCount = uiRefCount;
	}
	
	return( rc);
}

/***************************************************************************
Desc:		Returns the number of references at the current b-tree element.
			Leaf level blocks must be passed in and the block must be usable.
*****************************************************************************/
FLMUINT FSElementRefCount(					// Returns the number of references
	BTSK *			pStack)					// [in]
{
	FLMUINT			uiRefCount;
	FLMBYTE *		pCurRef;					// Points to current reference
	FLMBYTE *		pCurElm;					// Points to current element
	FLMUINT			uiRefSize;				// Size of the reference set
	DIN_STATE		tempState;

	// Check block type
	
	if( pStack->uiBlkType != BHT_LEAF)
	{
		uiRefCount = 0;
		goto Exit;
	}
	
	uiRefCount = 1;
	
	// Point to the start of the current reference skipping over domain info.
	
	pCurRef = pCurElm = CURRENT_ELM( pStack );
	(void) FSGetDomain( &pCurRef, pStack->uiElmOvhd );
	uiRefSize = (FLMUINT)(BBE_GET_RL(pCurElm) -
								(pCurRef - BBE_REC_PTR(pCurElm)));

	RESET_DINSTATE( tempState );

	// Read the first reference - there must be at least one reference.
	
	(void) DINNextVal( pCurRef, &tempState );	

	while( tempState.uiOffset < uiRefSize )
	{
		FLMUINT			uiNextLength;
	
		// Get the current byte to see what kind of item it is
		
		if( (uiNextLength = SENValLen( pCurRef + tempState.uiOffset)) == 0)
		{
			uiRefCount += DINOneRunVal( pCurRef, &tempState );
		}
		else
		{
			tempState.uiOffset += uiNextLength;
			uiRefCount++;
		}
	}
	
Exit:

	return (uiRefCount);
}

/***************************************************************************
Desc:		Read in the child block and set the counts in the input element.
			Does not go up the tree updating the counts.  Caller must do this.
*****************************************************************************/
RCODE FSUpdateAdjacentBlkCounts(
	FDB *				pDb,
	LFILE *			pLFile,
	BTSK * 			pStack,
	BTSK *			pNextBlkStk)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiNextBlkCount;
	BTSK *			pBaseStack = pStack;

	// Get the count of the next block and update up the tree.

	if( RC_BAD( rc = FSBlockCounts( pNextBlkStk, BH_OVHD,
			pNextBlkStk->uiBlkEnd, NULL, NULL, &uiNextBlkCount)))
	{
		goto Exit;
	}
	
	pStack = pBaseStack;
	pStack--;
	
	if( RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
	{
		if( rc == FERR_BT_END_OF_DATA)
		{
			rc = RC_SET( FERR_BTREE_ERROR);
		}
		
		goto Exit;
	}
	
	pStack = pBaseStack;
	
	if( RC_BAD( rc = FSUpdateBlkCounts( pDb, pStack, uiNextBlkCount)))
	{
		goto Exit;
	}
	
	pStack = pBaseStack;
	pStack--;
	
	if( RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
	{
		if( rc == FERR_BT_END_OF_DATA)
		{
			rc = RC_SET( FERR_BTREE_ERROR);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Update the block counts for a block adjusting all of the parent
			entry counts.
*****************************************************************************/
RCODE FSUpdateBlkCounts(
	FDB *				pDb,
	BTSK * 			pStack,
	FLMUINT			uiNewCount)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurElm;
	FLMUINT			uiCount;
	FLMINT			iDelta = 0;
	FLMBOOL			bFirstTime;

	// Go up the stack and update all parent block counts for the current blk.

	bFirstTime = TRUE;
	while( !BH_IS_ROOT_BLK( pStack->pBlk))
	{
		// Go to the parent and increment/decrement the counts.
		pStack--;

		pCurElm = pStack->pBlk + pStack->uiCurElm; 
		uiCount = FB2UD( &pCurElm[ BNE_CHILD_COUNT]);

		if( bFirstTime)
		{
			iDelta = uiCount - uiNewCount;
			bFirstTime = FALSE;

			// If the delta is zero there is nothing to do.
			
			if( !iDelta)
			{
				break;
			}
		}
		
		// Log the block.
		
		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			goto Exit;
		}
		
		// The block should be able to be used.

		uiCount = uiCount - iDelta;
		UD2FBA( (FLMUINT32)uiCount, &pCurElm[ BNE_CHILD_COUNT]);
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		For a positioning index update the count in all the parent elements.
			Splits and joins are very complex and will take care of redoing the
			counts.  This is why the calling code will increment/decrement 
			the counts before the key is added/deleted.
*****************************************************************************/
RCODE FSChangeCount(
	FDB *				pDb,
	BTSK * 			pStack,
	FLMBOOL			bAddReference)			// If FALSE, decrement the reference
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurElm;
	FLMUINT			uiCount;

	while( !BH_IS_ROOT_BLK( pStack->pBlk))
	{
		// Go to the parent and increment/decrement the counts.
		
		pStack--;

		// Log the block.
		
		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			goto Exit;
		}
		
		// The block should be able to be used.

		pCurElm = pStack->pBlk + pStack->uiCurElm; 
		uiCount = FB2UD( &pCurElm[ BNE_CHILD_COUNT]);
		
		if( bAddReference)
		{
			uiCount++;
		}
		else
		{
			// Don't allow value to be less than zero.
			
			if( uiCount)
			{
				uiCount--;
			}
		}
		
		UD2FBA( (FLMUINT32)uiCount, &pCurElm[ BNE_CHILD_COUNT]);
	}
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:		Through an insert or delete, change the block counts.
*****************************************************************************/
RCODE FSChangeBlkCounts(
	FDB *				pDb,
	BTSK * 			pStack,
	FLMINT			iDelta)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurElm;
	FLMUINT			uiCount;

	// Go up the stack and update all parent block counts for the current blk.

	while( !BH_IS_ROOT_BLK( pStack->pBlk))
	{
		// Go to the parent and increment/decrement the counts.
		
		pStack--;

		pCurElm = pStack->pBlk + pStack->uiCurElm; 
		uiCount = FB2UD( &pCurElm[ BNE_CHILD_COUNT]);

		uiCount = (((FLMINT)(uiCount + iDelta)) < 0) 
					? 0 : (FLMUINT) (uiCount + iDelta);

		// Log the block.
		
		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			goto Exit;
		}
		
		// The block should be able to be used.

		UD2FBA( (FLMUINT32)uiCount, &pCurElm[ BNE_CHILD_COUNT]);
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Given a stack, current element and a DINSTATE coupute the
			absolute position of the reference.  Must be called with a 
			positioning index.
*****************************************************************************/
RCODE FSGetBtreeRefPosition(
	FDB *				pDb,
	BTSK * 			pStack,
	DIN_STATE *		pDinState,
	FLMUINT *		puiRefPosition)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTotalCount;
	FLMUINT			uiRefCount;

	F_UNREFERENCED_PARM( pDb);
	
	// Compute the reference counts before all the current elements up the tree.
	
	if( RC_BAD( rc = FSBlockCounts( pStack, BH_OVHD, pStack->uiCurElm,
			NULL, NULL, &uiTotalCount)))
	{
		goto Exit;
	}

	// This must be a one-based number (first reference is 1).
	
	if( !pDinState->uiOffset)
	{
		uiTotalCount++;
	}
	else
	{
		FLMBYTE *		pCurRef;			// Points to current reference
		FLMBYTE *		pCurElm;			// Points to current element
		FLMUINT			uiRefSize;		// Size of the reference set
		DIN_STATE		tempState;

		// Compute the absolute position of this reference.

		uiRefCount = 2;
		RESET_DINSTATE( tempState );
		pCurRef = pCurElm = CURRENT_ELM( pStack );
		(void) FSGetDomain( &pCurRef, pStack->uiElmOvhd );
		uiRefSize = (FLMUINT)(BBE_GET_RL(pCurElm) -
									(pCurRef - BBE_REC_PTR(pCurElm)));

		// Read the first reference - there must be at least one reference.
		
		(void) DINNextVal( pCurRef, &tempState );	

		while( tempState.uiOffset < pDinState->uiOffset
			 && tempState.uiOffset < uiRefSize)
		{
			FLMUINT			uiNextLength;
	
			// Get the current byte to see what kind of item it is
		
			if( (uiNextLength = SENValLen( pCurRef + tempState.uiOffset)) == 0)
			{
				uiRefCount += DINOneRunVal( pCurRef, &tempState );
			}
			else
			{
				tempState.uiOffset += uiNextLength;
				uiRefCount++;
			}
		}
		
		if( tempState.uiOffset == pDinState->uiOffset && pDinState->uiOnes)
		{
			uiRefCount += pDinState->uiOnes;
		}
		
		uiTotalCount += uiRefCount;
	}

	// Go up the stack and keep the count up.
	
	while( !BH_IS_ROOT_BLK( pStack->pBlk))
	{
		// Go to the parent and increment/decrement the counts.
		
		pStack--;

		if( RC_BAD( rc = FSBlockCounts( pStack, BH_OVHD, pStack->uiCurElm,
				NULL, NULL, &uiRefCount)))
		{
			goto Exit;
		}
		
		uiTotalCount += uiRefCount;
	}

Exit:

	*puiRefPosition = uiTotalCount;
	return( rc);
}

/***************************************************************************
Desc:		Given a stack and a btree position, setup the b-tree to the
			current element and dinstate to the selected position.
			Must be called with a positioning index.
*****************************************************************************/
RCODE FSPositionSearch(
	FDB *				pDb,
	LFILE *			pLFile,
	FLMUINT			uiRefPosition,
	BTSK **			ppStack,
	FLMUINT *		puiRecordId,
	FLMUINT *		puiDomain,
	DIN_STATE *		pDinState)
{
	RCODE			rc;
	BTSK *		pStack = *ppStack;
	FLMBYTE *	pKeyBuf = pStack->pKeyBuf;
	FLMUINT		uiBlkAddr;
	LFILE			TmpLFile;

	if( RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}
		goto Exit;
	}

	pStack->uiCurElm = BH_OVHD;
	pStack->uiBlkEnd = (FLMUINT)FB2UW( &pStack->pBlk[ BH_ELM_END ] );

	for(;;)
	{
		pStack->uiFlags = FULL_STACK;
		pStack->uiKeyBufSize = MAX_KEY_SIZ;

		if( RC_BAD( rc = FSPositionScan( pStack, uiRefPosition, 
				&uiRefPosition, puiRecordId, puiDomain, pDinState)))
		{
			goto Exit;
		}
		
		if( !pStack->uiLevel)
		{
			break;
		}

		uiBlkAddr = FSChildBlkAddr( pStack );
		pStack++;
		pStack->pKeyBuf = pKeyBuf;

		if( RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack )))
		{
			goto Exit;
		}
	}
	
	*ppStack = pStack;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Position to the element given a position value relative to the block.
*****************************************************************************/
RCODE FSPositionScan(
	BTSK *			pStack,
	FLMUINT			uiRelativePosition,
	FLMUINT *		puiRelativePosInElement,
	FLMUINT *		puiRecordId,
	FLMUINT *		puiDomain,
	DIN_STATE *		pDinState)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiRefCount;
	FLMBYTE *		pCount;
	FLMBYTE *		pBlk = pStack->pBlk;
	FLMBYTE *		pKeyBuf = pStack->pKeyBuf;
	FLMBYTE *		pPrevKey;
	FLMBYTE *		pCurElm;
	FLMUINT			uiBlkType = pStack->uiBlkType;
	FLMUINT			uiElmOvhd = pStack->uiElmOvhd;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiPrevKeyCnt = 0;
	FLMUINT			uiPrevPrevKeyCnt = 0;
	FLMUINT			uiBytesToMove;
	FLMUINT			uiTotalElmLen;

	pStack->uiCurElm = BH_OVHD;
	pStack->uiBlkEnd = (FLMUINT)FB2UW( &pBlk[ BH_ELM_END]);
	pPrevKey = NULL;

	for(;;)
	{
		pCurElm = &pBlk[ pStack->uiCurElm ];

		if( uiBlkType != BHT_LEAF)
		{
			pCount = pCurElm + BNE_CHILD_COUNT;
			uiRefCount = FB2UD( pCount);
		}
		else
		{
			uiRefCount = FSElementRefCount( pStack);
		}

		uiElmKeyLen = BBE_GETR_KL( pCurElm );
		if( ( uiPrevKeyCnt = (BBE_GETR_PKC( pCurElm ))) > BBE_PKC_MAX)
		{
			uiElmKeyLen += (uiPrevKeyCnt & BBE_KL_HBITS) << BBE_KL_SHIFT_BITS;
			uiPrevKeyCnt &= BBE_PKC_MAX;
		}

		uiTotalElmLen = uiElmOvhd + uiElmKeyLen;
		if( uiBlkType != BHT_LEAF)
		{
			if( BNE_IS_DOMAIN( pCurElm))
			{
				uiTotalElmLen += BNE_DOMAIN_LEN;
			}
		}
		else
		{
			// Copy the key into the key buffer.
			
			if( uiPrevKeyCnt > uiPrevPrevKeyCnt)
			{
				uiBytesToMove = uiPrevKeyCnt - uiPrevPrevKeyCnt;
				f_memcpy( &pKeyBuf[ uiPrevPrevKeyCnt], pPrevKey, uiBytesToMove);
			}
			pPrevKey = pCurElm + uiElmOvhd;
			uiTotalElmLen += BBE_GET_RL( pCurElm);
		}
		
		if( uiRefCount >= uiRelativePosition)
		{
			pStack->uiKeyLen = uiElmKeyLen + uiPrevKeyCnt;
			pStack->uiPrevElmPKC = uiPrevPrevKeyCnt;
			pStack->uiPKC = uiPrevKeyCnt;

			if( uiBlkType == BHT_LEAF)
			{
				// Copy the remaining bytes of the key.  pPrevKey is current key.
				
				if( uiElmKeyLen)
				{
					f_memcpy( &pKeyBuf[ uiPrevKeyCnt], pPrevKey, uiElmKeyLen);
				}
				
				if( RC_BAD( rc = FSPositionToRef( pStack, uiRelativePosition,
					puiRecordId, puiDomain, pDinState)))
				{
					goto Exit;
				}
				
				uiRelativePosition = 0;
			}
			
			break;
		}

		uiPrevPrevKeyCnt = uiPrevKeyCnt;
		uiRelativePosition -= uiRefCount;
		pStack->uiCurElm += uiTotalElmLen;
		
		if( pStack->uiCurElm >= pStack->uiBlkEnd)
		{
			uiRelativePosition = 0;
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}

	*puiRelativePosInElement = uiRelativePosition;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:		Position to the element given a position value relative to the block.
*****************************************************************************/
RCODE FSPositionToRef(
	BTSK *			pStack,
	FLMUINT			uiRelativePosition,
	FLMUINT *		puiRecordId,
	FLMUINT *		puiDomain,
	DIN_STATE *		pDinState)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiRefCount;
	FLMUINT			uiRecordId;

	RESET_DINSTATE_p( pDinState);
	uiRefCount = FSElementRefCount( pStack);

	if( uiRelativePosition <= 1)
	{
		uiRecordId = FSRefFirst( pStack, pDinState, puiDomain);
	}
	else
	{
		// Find the position within the element.

		FLMBYTE *		pCurRef;					// Points to current reference
		FLMBYTE *		pCurElm;					// Points to current element
		FLMUINT			uiRefSize;				// Size of the reference set
		DIN_STATE		tempState;

		// Point to the start of the current reference skipping over domain info.
		
		pCurRef = pCurElm = CURRENT_ELM( pStack );
		*puiDomain = FSGetDomain( &pCurRef, pStack->uiElmOvhd) + 1;
		uiRefSize = (FLMUINT)(BBE_GET_RL(pCurElm) -
									(pCurRef - BBE_REC_PTR(pCurElm)));

		uiRecordId = DINNextVal( pCurRef, pDinState);
		uiRelativePosition--;
		
		while( uiRelativePosition > 1 && pDinState->uiOffset < uiRefSize)
		{
			uiRecordId -= DINNextVal( pCurRef, pDinState);
			uiRelativePosition--;
		}
		
		flmAssert( pDinState->uiOffset < uiRefSize);

		// Get the last value without moving pDinState.
		
		tempState.uiOffset = pDinState->uiOffset;
		tempState.uiOnes = pDinState->uiOnes;
		uiRecordId -= DINNextVal( pCurRef, &tempState );
	}
	
	*puiRecordId = uiRecordId;
	return( rc);
}
