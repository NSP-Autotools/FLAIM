//-------------------------------------------------------------------------
// Desc:	Delete element from b-tree block.
// Tabs:	3
//
// Copyright (c) 1991-2001, 2003-2007 Novell, Inc. All Rights Reserved.
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

/***************************************************************************
Desc:		Delete the current element from a b-tree block and write block.
			The stack must point to the next element after the deleted element.
			The next element could be in a different block so be careful.
Notes:	The order of the high level if's is VERY important.
			The order of changing the blocks is not important because of logging.
*****************************************************************************/
RCODE FSBtDelete(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **	 	pStackRV)
{
	RCODE			rc;
	BTSK *		pStack = *pStackRV;
	SCACHE *		pTmpSCache;
	FLMBYTE *	pBlk;
	FLMBOOL		bReleaseTmpCache = FALSE;
	FLMUINT		uiNextBlk;
	FLMUINT		uiPrevBlk;
	FLMUINT		uiLastElm;			// Offset of last element in a block
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMBYTE		pLEMBuffer[12];	// Last element marker buffer

	// Log block before modifying it

	if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
	{
		goto Exit;
	}

	// If this is a non-leaf positioning index, update parent counts.

	if (pLFile->pIxd && (pLFile->pIxd->uiFlags & IXD_POSITIONING))
	{
		if (pStack->uiLevel)
		{
			FLMUINT		uiRefCount;
			FLMBYTE*		pElm = pStack->pBlk + pStack->uiCurElm;

			// Reduce the counts from the parent up.

			uiRefCount = FB2UD( &pElm[BNE_CHILD_COUNT]);
			
			if (RC_BAD( rc = FSChangeBlkCounts( pDb, pStack, 
				(FLMINT) (0 - uiRefCount))))
			{
				goto Exit;
			}
		}
	}

	if (RC_BAD( rc = FSBlkDelElm( pStack)))
	{
		goto Exit;
	}

	// Take care of deletion of the ONLY element in the block.  There is NO
	// WAY this could be a root or right end block because of the last
	// element marker (LEM) takes up 2 or 4 bytes

	if (pStack->uiBlkEnd == BH_OVHD)
	{

		// Free up this empty block and fixup the NEXT/PREV links.

		uiNextBlk = FB2UD( &pStack->pBlk[BH_NEXT_BLK]);
		
		rc = FSBlockFixLinks( pDb, pLFile, pStack->pSCache);
		
		pStack->pSCache = NULL;
		pStack->pBlk = NULL;
		
		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// Remove the element that pointed to this block

		if (RC_OK( rc = FSDelParentElm( pDb, pLFile, &pStack)))
		{
			pStack->uiBlkAddr = uiNextBlk;

			// Read the next block and set stack to point to next element

			if (RC_OK( rc = FSGetBlock( pDb, pLFile, uiNextBlk, pStack)))
			{
				pStack->uiCurElm = BH_OVHD;
				FSBlkBuildPKC( pStack, pStack->pKeyBuf, FSBBPKC_AT_CURELM);
			}
		}

		*pStackRV = pStack;
	}

	// Deleting a RIGHT MOST B-tree block (along the right side). Check if
	// only the LEM (last element marker) is left in the block. A block
	// CANNOT contain just a LEM. Move the LEM and free the block. There are
	// 3 cases to watch for:
	//
	//			EMPTY B-TREE - free the root/block save next record in lfArea
	//			EMPTY ROOT - free root and goto child and make that root
	//			EMPTY BLOCK - cannot contain only a LEM - move to previous block
	//
	// Must not move last DRN Marker - there is no need to do this.

	else if ((FB2UD( &pStack->pBlk[BH_NEXT_BLK]) == BT_END) &&
				((pStack->uiBlkEnd == BH_OVHD + uiElmOvhd)))
	{

		// EMPTY B-TREE Level is not in the current LFILE version so check
		// if the root block is the same as this block.

		if ((pStack->uiBlkType == BHT_LEAF) &&
			 (pLFile->uiRootBlk == pStack->uiBlkAddr))	// Root is leaf
		{

			// Return the empty root block to the system
			// Code supports empty b-trees. We have gone back and forth on
			// returning empty root blocks to the system. Setup stack for
			// emtpy state and modify the LFILE on disk. The data record
			// b-tree can NEVER be empty because of the next record DRN
			// record should always hang around. ALL CALLING ROUTINES MUST
			// CHECK for STACK->uiBlkAddr == BT_END.

			pStack->uiBlkAddr = BT_END;
			{

				// Get the next DRN and save in the LFD

				FLMBYTE *		ptr = &pStack->pBlk[BH_OVHD];
				
				ptr += BBE_GETR_KL( ptr) + BBE_KEY;
				pLFile->uiNextDrn = (FLMUINT) FB2UD( ptr);
			}

			rc = FSBlockFree( pDb, pStack->pSCache);
			
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;
			
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			pLFile->uiRootBlk = BT_END;
			rc = flmLFileWrite( pDb, pLFile);
		}

		// EMPTY ROOT BLOCK Remove root block and set new root block to
		// child.

		else if (pLFile->uiRootBlk == pStack->uiBlkAddr)
		{

			// Obtain child block and set to lfArea to assign new root block

			uiNextBlk = FSChildBlkAddr( pStack);

			rc = FSBlockFree( pDb, pStack->pSCache);
			
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;

			if (RC_BAD( rc))
			{
				goto Exit;
			}

			pLFile->uiRootBlk = uiNextBlk;

			if (RC_BAD( rc = flmLFileWrite( pDb, pLFile)))
			{
				goto Exit;
			}

			// Assign the new root block

			if (RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiNextBlk, NULL,
						  &pTmpSCache)))
			{
				goto Exit;
			}

			bReleaseTmpCache = TRUE;

			if (RC_BAD( rc = ScaLogPhysBlk( pDb, &pTmpSCache)))
			{
				goto Exit;
			}

			// Set the root block flag on this block & mark dirty

			BH_SET_ROOT_BLK( pTmpSCache->pucBlk);
			ScaReleaseCache( pTmpSCache, FALSE);
			bReleaseTmpCache = FALSE;

			// It is possible to add an element so that a new root block
			// gets created AND THEN Deleted.  Because of this
			// the ROOT block in the stack MUST always be pStack[0]. The
			// three lines below do this.

			shiftN( (FLMBYTE*) pStack + sizeof(BTSK),
					 sizeof(BTSK) * pStack->uiLevel,			// think uiLevel + 1 - 1
					 0 - (sizeof(BTSK)));

			// We don't want a pointer to the same shared cache in two
			// different elements of the pStack. NULL out the one that was
			// moved down.

			(pStack + pStack->uiLevel + 1)->pSCache = NULL;
			(pStack + pStack->uiLevel + 1)->pBlk = NULL;
			pStack--;
			*pStackRV = pStack;

			// Don't worry about positioning to the next element - root gone
		}
		else
		{
			// ONLY LAST ELEMENT MARKER (LEM) REMAINS (leaf or (non-leaf but
			// NOT ROOT)). A block must NEVER contain only the last element
			// marker (LEM). This may free up a leaf block or a non-leaf
			// block but never a root block (root checks above.) This is the
			// most complex case. Move the last element marker (LEM) to the
			// end of the previous block. Be carefull to delete the correct
			// element in the parent block. The algorithm states that there
			// MUST ALWAYS be room to insert the LEM into a block without
			// splitting that block. Because of logging the block
			// modifications do not need to be flushed in a specific order to
			// prevent corruption.

			uiPrevBlk = FB2UD( &pStack->pBlk[BH_PREV_BLK]);

			// Save the last element marker, may be a non-leaf element

			f_memcpy( pLEMBuffer, &pStack->pBlk[BH_OVHD], uiElmOvhd);

			// Free the block

			rc = FSBlockFree( pDb, pStack->pSCache);
			
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;
			
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			// Pop stack - point to parent

			pStack--;

			// Modify the parent blocks last element marker (LEM) to point to
			// the new last block. THEN delete the previous element that also
			// points to the new last block.  Read the parent block.
			
			if (RC_BAD( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			// Change where element points to in the last element marker (LEM)

			FSSetChildBlkAddr( BLK_ELM_ADDR( pStack, pStack->uiCurElm), uiPrevBlk,
									pStack->uiElmOvhd);

			// FSBtPrevElm may return -1 which is NOT OK

			if (RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
			{
				rc = (rc == FERR_BT_END_OF_DATA) ? RC_SET( FERR_BTREE_ERROR) : rc;
				goto Exit;
			}

			// Delete the element that points to the new last block

			if (RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
			{
				goto Exit;
			}

			pStack++;

			// Read in the new last block Move in the old LEM to the end of
			// the block Position pStack to LEM

			if (RC_BAD( rc = FSGetBlock( pDb, pLFile, uiPrevBlk, pStack)))
			{
				goto Exit;
			}

			// Log block before modifying

			if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			{
				goto Exit;
			}

			pBlk = pStack->pBlk;

			// Set the pStack elements

			uiLastElm = pStack->uiBlkEnd;
			pStack->uiCurElm = uiLastElm;

			// Place the LEM (last element marker) in the block

			f_memcpy( &pBlk[uiLastElm], pLEMBuffer, uiElmOvhd);

			// Set next block pointer to BT_END

			UD2FBA( BT_END, &pBlk[BH_NEXT_BLK]);

			// Adjust end pointers

			pStack->uiBlkEnd = uiLastElm + uiElmOvhd;
			UW2FBA( (FLMUINT16) pStack->uiBlkEnd, &pBlk[BH_BLK_END]);

			*pStackRV = pStack;
		}
	}

	// Deleted the LAST (but not only) element in the block. The parent
	// block must have its current element changed to reflect the new last
	// element key in this block. The algoritm MUST insert the next last
	// element key and THEN delete the old key or you may have a new root
	// block which is your next block! Comentary: Some b-tree algorithms
	// have key markers that are shorter in the non-leaf blocks. This
	// right-end non-leaf trunctaion is a very good idea, but ALL of the
	// b-tree code has to conform to this rule and it is not a trivial
	// thing to support.

	else if (pStack->uiBlkEnd == pStack->uiCurElm)
	{
		pBlk = pStack->pBlk;

		// Double check in case of corrupt this is not a right most block

		if (FB2UD( &pBlk[BH_NEXT_BLK]) != BT_END)
		{
			rc = FSNewLastBlkElm( pDb, pLFile, &pStack, 
										 FSNLBE_LESS | FSNLBE_POSITION);
			*pStackRV = pStack;
		}
	}

	// Else - normal delete - everything should be correct and the stack is
	// positioned to the next element after the deleted element. The
	// pKeyBuf[] however, is NOT always set to be the current elms key.
	// Check to see if there is room to make a 3/2 combine and do it.

	else if (pStack->uiBlkEnd <=
					(FFILE_MIN_FILL * pDb->pFile->FileHdr.uiBlockSize / 100))
	{
		rc = FSCombineBlks( pDb, pLFile, &pStack);
		*pStackRV = pStack;	// Stack may have changed
	}

Exit:

	if (bReleaseTmpCache)
	{
		ScaReleaseCache( pTmpSCache, FALSE);
	}

	return (rc);
}

/****************************************************************************
Desc:  	Delete the parent element from where you are at in the stack.
Notes: 	In the future we should either pin down the current block or
			reread it in.
****************************************************************************/
RCODE FSDelParentElm(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **	 	pStackRV)
{
	RCODE		rc;
	BTSK *	pStack = *pStackRV;

	pStack--;
	
	if (RC_BAD( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
	{
		goto Exit;
	}

	// Ignore status value from FSBtScanTo - just position for delete

	if (RC_BAD( rc = FSBtScanTo( pStack, NULL, 0, (FLMUINT) 0)))
	{
		goto Exit;
	}

	// Call will position to the next element which pts to the next blk
	
	rc = FSBtDelete( pDb, pLFile, &pStack);

Exit:

	pStack++;							// Go down the pStack to the original level
	*pStackRV = pStack;
	
	return (rc);
}

/****************************************************************************
Desc:  	There is a new last block element, store key in parent and
			remove current element if( uiFlags & FSNLBE_LESS or GREATER is set
Notes:	Take note of the uiFlags - they can change what is done
****************************************************************************/
RCODE FSNewLastBlkElm(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **	 	pStackRV,
	FLMUINT		uiFlags) // FSNLBE_GREATER, *_LESS, CK_COMBINE, POSITION or 0
{
	RCODE			rc;
	BTSK*			pStack = *pStackRV;
	FLMBYTE *	pBlk = pStack->pBlk;
	FLMBYTE *	pCurElm;							// Points to the current last element
	FLMUINT		uiOldCurElm = pStack->uiCurElm;
	FLMUINT		uiNextBlk = 0;
	FLMUINT		uiDomain = 0;					// Domain is index reference range
	FLMUINT		uiElmLen;						// Element length
	FLMUINT		uiKeyLen;
	FLMUINT		uiNewElmOvhd;
	FLMUINT		uiRefCount;
	FLMBYTE *	pKey;
	FLMBYTE		pElmBuffer[MAX_KEY_SIZ + BNE_KEY_COUNTS_START + BNE_DOMAIN_LEN];

	uiNewElmOvhd = (pStack - 1)->uiElmOvhd;
	uiElmLen = uiNewElmOvhd;

	if (uiNewElmOvhd == BNE_DATA_OVHD)
	{
		pKey = pElmBuffer;
	}
	else
	{
		pElmBuffer[BBE_PKC] = 0;				// Set PKC, DOMAIN to zero
		pElmBuffer[BBE_KL] = 0;
		pKey = pElmBuffer + uiNewElmOvhd;

		// Only update the counts if the inserting a key and not replacing.

		if (uiNewElmOvhd == BNE_KEY_COUNTS_START)
		{
			if (RC_BAD( rc = FSBlockCounts( pStack, BH_OVHD, pStack->uiBlkEnd,
						  NULL, NULL, &uiRefCount)))
			{
				goto Exit;
			}

			UD2FBA( (FLMUINT32)uiRefCount, &pElmBuffer[BNE_CHILD_COUNT]);
		}
	}

	// Build the pElmBuffer[] with the new last key in the block

	FSSetChildBlkAddr( pElmBuffer, pStack->uiBlkAddr, uiNewElmOvhd);

	if ((uiNextBlk = FB2UD( &pBlk[BH_NEXT_BLK])) == BT_END)
	{
		if (uiNewElmOvhd == BNE_DATA_OVHD)
		{
			UD2FBA( 0xFFFFFFFF, pElmBuffer);
		}

		uiKeyLen = 0;
		uiElmLen = uiNewElmOvhd;
		goto no_domain;
	}

	// else - fall through

	pStack->uiCurElm = pStack->uiBlkEnd;	// Position past last element
	FSBtPrevElm( pDb, pLFile, pStack);		// Build full key in pKeyBuf[]
	uiKeyLen = pStack->uiKeyLen;

	// Copy the key

	if (uiNewElmOvhd == BNE_DATA_OVHD)
	{
		flmCopyDrnKey( pElmBuffer, pStack->pKeyBuf);
	}
	else if (uiKeyLen)
	{
		f_memcpy( &pElmBuffer[uiNewElmOvhd], pStack->pKeyBuf, uiKeyLen);
		BBE_SET_KL( pElmBuffer, uiKeyLen);
		uiElmLen += uiKeyLen;
	}

	// If this is a boundqed index reference set then store DOMAIN

	pCurElm = CURRENT_ELM( pStack);

	uiDomain = (pLFile->uiLfType == LF_INDEX) 
							? FSGetDomain( &pCurElm, pStack->uiElmOvhd) 
							: (FLMUINT) 0;
							
	if (uiDomain)
	{
		BNE_SET_DOMAIN( pElmBuffer);
		
		pElmBuffer[uiElmLen++] = (FLMBYTE) (uiDomain >> 16);
		pElmBuffer[uiElmLen++] = (FLMBYTE) (uiDomain >> 8);
		pElmBuffer[uiElmLen++] = (FLMBYTE) (uiDomain & 0xFF);
	}

no_domain:

	// Go to the parent block, insert new element and delete the old last
	// element in the block. Pinning the current block is not suggested
	// because you could pin number of (levels - 1) blocks.

	pStack--;
	if (RC_BAD( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
	{
		goto Exit;
	}

	// If greater you should be positioned AFTER the matching element

	if (pStack->uiBlkEnd > BH_OVHD)			// Don't call if NO elements
	{

		// Set up the pStack elements for insert - passing keyLen sets up
		// pStack

		if (RC_BAD( rc = FSBtScanTo( pStack, pKey, uiKeyLen, uiDomain)))
		{
			goto Exit;
		}
	}
	else
	{
		pStack->uiPrevElmPKC = pStack->uiPKC = 0;
	}

	// Insert the element into the parent block - watch for splits!

	if (RC_OK( rc = FSBtInsert( pDb, pLFile, &pStack, pElmBuffer, uiElmLen)))
	{
		if (uiFlags & FSNLBE_LESS)
		{
			
			// Go to the next element, may read a new block!

			if (RC_OK( rc = FSBtNextElm( pDb, pLFile, pStack)))
			{
				if (RC_OK( rc = FSBtDelete( pDb, pLFile, &pStack)))
				{
					
					// Now position to the current element - back one

					if (!(uiFlags & FSNLBE_POSITION))
					{
						rc = FSBtPrevElm( pDb, pLFile, pStack);
						rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;
					}
				}
				else
				{
					rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;
				}
			}
		}
		else if (uiFlags & FSNLBE_GREATER)
		{
			if (RC_OK( rc = FSBtPrevElm( pDb, pLFile, pStack)))
			{
				if (RC_OK( rc = FSBtDelete( pDb, pLFile, &pStack)))
				{

					// Position to the next element if flag is set

					if (uiFlags & FSNLBE_POSITION)
					{
						rc = FSBtNextElm( pDb, pLFile, pStack);
						rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;
					}
				}
			}
		}
	}

Exit:

	// Pop the pStack and position to next element in the block that you
	// expect to be positioned to. You should be positioned to the correct
	// parent element.

	pStack++;
	*pStackRV = pStack;							// Update caller's pStack
	
	if (RC_OK( rc))
	{
		if ((uiFlags & FSNLBE_POSITION) && (uiNextBlk != BT_END))
		{
			pStack->uiBlkAddr = uiNextBlk;
			uiOldCurElm = BH_OVHD;
		}

		// Read the next block and set pStack.

		if (RC_OK( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
		{
			pStack->uiCurElm = uiOldCurElm;	// Restore original curElm value
			FSBlkBuildPKC( pStack, pStack->pKeyBuf, FSBBPKC_AT_CURELM);
		}
	}

	return (rc);
}

/***************************************************************************
Desc:		Delete the current element from a b-tree block without writing block.
			The pStack must point to the next element after the deleted element.
Notes:	Code handles deletion of an element at any level of the b-tree.
*****************************************************************************/
RCODE FSBlkDelElm(
	BTSK *		pStack)					// Stack of variables for each level
{
	RCODE			rc;
	FLMBYTE*		pDelElm;					// Points to deleted element
	FLMBYTE*		pCurElm;					// Points to current elm to move down
	FLMBYTE*		pBlk = pStack->pBlk; // Points to block for speed

	FLMUINT		uiCurElmOfs;			// Current (next) element's offset
	FLMUINT		uiDelElmPkc;			// # of carry bytes for deleted elm
	FLMUINT		uiCurElmPkc;			// Current element's Prev key count
	FLMUINT		uiCurElmKeyLen;		// Current element's key length
	FLMUINT		uiOldCurElm = pStack->uiCurElm;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMINT		iDelElmSize;			// Deleted element's size
	FLMINT		iPkcLost;				// Number bytes to expand for next elm

	pDelElm = &pBlk[pStack->uiCurElm];

	if (RC_OK( rc = FSBlkNextElm( pStack)))
	{

		// Setup to remove what pDelElm is pointing to. This is NOT the last
		// element in the block so move down the rest of the block data.

		uiCurElmOfs = pStack->uiCurElm;
		iDelElmSize = (FLMINT) (uiCurElmOfs - uiOldCurElm);
		pCurElm = &pBlk[uiCurElmOfs];

		if (pStack->uiBlkType != BHT_NON_LEAF_DATA)
		{
			uiDelElmPkc = (FLMUINT) (BBE_GET_PKC( pDelElm));
			uiCurElmPkc = (FLMUINT) (BBE_GET_PKC( pCurElm));

			// If current element uses bytes from the deleted element...

			if (uiCurElmPkc > uiDelElmPkc)
			{
				iPkcLost = (FLMINT) (uiCurElmPkc - uiDelElmPkc);

				// Create the new deleted element and setup for the shiftN()
				// below moving pCurElm.

				uiCurElmPkc -= iPkcLost;
				uiCurElmKeyLen = (FLMUINT) (BBE_GET_KL( pCurElm) + iPkcLost);

				BBE_SET_PKC( pDelElm, uiCurElmPkc); // Clears all bits
				BBE_SET_KL( pDelElm, uiCurElmKeyLen);
				*pDelElm |= (FLMBYTE) (BBE_IS_FIRST_LAST( pCurElm));

				if (pStack->uiBlkType == BHT_LEAF)
				{
					BBE_SET_RL( pDelElm, BBE_GET_RL( pCurElm));
				}
				else
				{
					f_memcpy( pDelElm + BNE_CHILD_BLOCK, pCurElm + BNE_CHILD_BLOCK,
								uiElmOvhd - BNE_CHILD_BLOCK);
				}

				// Adjust iDelElmSize and uiCurElmOfs to delete current element
				// overhead

				iDelElmSize -= iPkcLost;
				uiCurElmOfs += uiElmOvhd;

				// Move any extra bytes from the deleted element refered in
				// curElm

				f_memcpy( pDelElm + uiElmOvhd, &pStack->pKeyBuf[uiDelElmPkc],
							iPkcLost);

				// Fall through and copy the rest of the block down

			}
		}

		// Shift down - starting at pCurElm

		shiftN( &pBlk[uiCurElmOfs], pStack->uiBlkEnd - uiCurElmOfs,
				 (FLMINT) (0 - iDelElmSize));

		pStack->uiBlkEnd -= iDelElmSize;
	}
	else if (rc == FERR_BT_END_OF_DATA)
	{
		rc = FERR_OK;
		if (pStack->uiCurElm == pStack->uiBlkEnd)
		{	
			// Deleted last element in blk
			
			pStack->uiBlkEnd = uiOldCurElm;
		}
		else
		{

			// Need to move the last element marker (LEM) down;
			// This should only be used on leaf blocks
			
			pBlk[uiOldCurElm] = BBE_FIRST_FLAG | BBE_LAST_FLAG;
			pBlk[uiOldCurElm + BBE_KL] = pBlk[uiOldCurElm + BBE_RL] = 0;
			pStack->uiBlkEnd = uiOldCurElm + uiElmOvhd;
		}
	}
	else
	{
		goto Exit;
	}

	// Modify the element end offset & restore curElm & prevElm on pStack

	UW2FBA( (FLMUINT16) pStack->uiBlkEnd, &pBlk[BH_BLK_END]);
	pStack->uiCurElm = uiOldCurElm;
	
Exit:

	return (rc);
}

/****************************************************************************
Desc:  	Sets the parent element's child block value
****************************************************************************/
void FSSetChildBlkAddr(
	FLMBYTE *	pElement,
	FLMUINT		uiBlkAddr,
	FLMUINT		uiBlkOvhd)
{
	FLMBYTE *	pChildAddr;

	if (uiBlkOvhd == BNE_KEY_START || uiBlkOvhd == BNE_KEY_COUNTS_START)
	{
		pChildAddr = pElement + BNE_CHILD_BLOCK;
		UD2FBA( (FLMUINT32)uiBlkAddr, pChildAddr);
	}
	else if (uiBlkOvhd == BNE_DATA_OVHD)
	{
		pChildAddr = pElement + BNE_DATA_CHILD_BLOCK;
		UD2FBA( (FLMUINT32)uiBlkAddr, pChildAddr);
	}

	return;
}
