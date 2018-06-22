//-------------------------------------------------------------------------
// Desc:	Insert an element into a b-tree block.
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

FSTATIC RCODE FSBlkInsElm(
	BTSK *			stk,
	FLMBYTE *		pElm,
	FLMUINT			uiElmLen,
	FLMUINT			uiBlkSize);

/****************************************************************************
Desc: Replace the current element with the input element. Both elements
		must contain EXACTLY the same key. element[] must contain full key.
		This gets complex if the input element causes a block split. 
		If so, then call FSBtDelete() and then call FSBtInsert() to split
		the block. 
****************************************************************************/
RCODE FSBtReplace(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **		 pStackRV,
	FLMBYTE *	pElement,
	FLMUINT		uiElmLen)
{
	RCODE			rc;
	BTSK *		pStack = *pStackRV;
	FLMBYTE *	pCurElm = CURRENT_ELM( pStack);
	FLMBYTE *	pMovePoint;
	FLMUINT		uiCurRecOfs = (FLMUINT) BBE_REC_OFS( pCurElm);
	FLMUINT		uiCurRecLen = (FLMUINT) BBE_GET_RL( pCurElm);
	FLMUINT		uiElmRecOfs = BBE_REC_OFS( pElement);
	FLMUINT		uiElmRecLen = BBE_GET_RL( pElement);
	FLMUINT		uiBytesFree;
	FLMUINT		uiArea;
	FLMINT		iDistance;

	// Set bsBlkEnd because of bug somewhere in the system. This MUST be
	// found as soon as possible. April 23, 1992 (SWP)

	pStack->uiBlkEnd = (FLMUINT) FB2UW( &pStack->pBlk[BH_ELM_END]);
	uiBytesFree = pDb->pFile->FileHdr.uiBlockSize - 
							(pStack->uiBlkEnd + BBE_LEM_LEN);

	// Code around signed compare problems to see if pElement will fit

	if ((uiElmRecLen <= uiCurRecLen) ||
		 (uiBytesFree >= (uiElmRecLen - uiCurRecLen)))
	{

		// Log the block before modifying it.

		if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			return (rc);
		}

		pCurElm = CURRENT_ELM( pStack);

		// There is room to move things around in the block

		iDistance = (FLMINT) (uiElmRecLen - uiCurRecLen);
		uiArea = pStack->uiBlkEnd - (pStack->uiCurElm + uiCurRecOfs);
		pMovePoint = &pCurElm[uiCurRecOfs];

		if (uiElmRecLen < uiCurRecLen)
		{
			uiArea += iDistance;			// iDistance is negitive
			pMovePoint -= iDistance;	// Add |distance| to pMovePoint
		}

		if (iDistance)
		{
			shiftN( pMovePoint, uiArea, iDistance);
			pStack->uiBlkEnd += iDistance;
			UW2FBA( (FLMUINT16) pStack->uiBlkEnd, BLK_ELM_ADDR( pStack, BH_ELM_END));
		}

		// Change the record length

		BBE_SET_RL( pCurElm, BBE_GET_RL( pElement));
		f_memcpy( &pCurElm[uiCurRecOfs], &pElement[uiElmRecOfs], uiElmRecLen);
	}
	else
	{
		FLMUINT	uiKeyLen = (FLMUINT) BBE_GET_KL( pElement);

		if (RC_BAD( rc = FSBtDelete( pDb, pLFile, &pStack)))
		{
			return (rc);
		}

		// Setup the pStack and bsKeyBuf[] for the insert

		if (RC_BAD( rc = FSBtScanTo( pStack, &pElement[BBE_KEY], uiKeyLen, 0)))
		{
			goto Exit;
		}

		rc = FSBtInsert( pDb, pLFile, &pStack, pElement, uiElmLen);
		*pStackRV = pStack;
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc: Insert an pElement/key into a logical b-tree with split support.
		Supports insertion of a new leaf element in the b-tree structure.
****************************************************************************/
RCODE FSBtInsert(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK **		pStackRV,
	FLMBYTE *	pElement,
	FLMUINT		uiElmLen)
{
	RCODE			rc;
	BTSK *		pStack = *pStackRV;
	FLMUINT		uiBlkSize = pDb->pFile->FileHdr.uiBlockSize;

	// Log the block before modifying it.

	if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
	{
		goto Exit;
	}

	// See if there is enough room for the insertion in the block

	if (RC_OK( rc = FSBlkInsElm( pStack, pElement, uiElmLen, uiBlkSize)))
	{

		// If this is a non-leaf positioning index, update parent counts.

		if (pLFile->pIxd && (pLFile->pIxd->uiFlags & IXD_POSITIONING))
		{
			if (pStack->uiLevel)
			{
				rc = FSChangeBlkCounts( pDb, pStack,
											  FB2UD( &pElement[BNE_CHILD_COUNT]));
			}
		}
	}
	else if (rc == FERR_BLOCK_FULL)
	{

		// No room to insert, split the block and reinsert

		rc = FSBlkSplit( pDb, pLFile, pStackRV, pElement, uiElmLen);
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Insert an pElement (any type) into a block & compress the key.
****************************************************************************/
FSTATIC RCODE FSBlkInsElm(
	BTSK *		pStack,		// Stack holding all state info
	FLMBYTE *	pElement,	// The input element to insert
	FLMUINT		uiElmLen,	// Length of the element
	FLMUINT		uiBlkSize)	// Size of the stack block
{
	RCODE			rc = RC_SET( FERR_BLOCK_FULL);
	FLMBYTE *	pCurElm;
	FLMUINT		uiShiftLen;
	FLMUINT		uiShiftDist;
	FLMUINT		uiNewElmPkc;
	FLMUINT		uiNewElmLen = uiElmLen - pStack->uiPrevElmPKC;
	FLMUINT		uiNewElmKeyLen;
	FLMUINT		uiCurElmPkc;
	FLMUINT		uiCurElmKeyLen;
	FLMUINT		uiDiff;
	FLMUINT		uiBlkEnd = pStack->uiBlkEnd;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;

	uiNewElmPkc = pStack->uiPrevElmPKC;
	uiNewElmKeyLen = BBE_GET_KL( pElement) - uiNewElmPkc;

	// If there is room, then compress current element and insert new
	// element. There must ALWAYS be room for the last element marker
	// (LEM). "uiDiff" not in computation because of the complexity of
	// determining uiDiff before this compare;
	// thus perform a split that wasn't needed. This is OK because blocks
	// should have some breathing room anyway.

	if ((uiBlkEnd + uiElmOvhd + uiNewElmLen) <= uiBlkSize)
	{
		// There is enough space in the block for the element. Shift up
		// higher elements compressing more the current element. The key in
		// keyBuf must be the curElm key and NOT prevElm! pStack->wPKC and
		// pStackPvElmPKC be valid from btScan()

		pCurElm = CURRENT_ELM( pStack);

		if (pStack->uiCurElm < uiBlkEnd)
		{

			// uiDiff = additional bytes to compress on the current element.
			// There is no way diff can be negative (unless buggy code).

			if (uiElmOvhd != BNE_DATA_OVHD)
			{
				uiCurElmPkc = BBE_GET_PKC( pCurElm);
			}
			else
			{
				uiCurElmPkc = 0;
			}

			if ((uiDiff = (FLMUINT) (pStack->uiPKC - uiCurElmPkc)) >= MAX_KEY_SIZ)
			{
				return (RC_SET( FERR_BTREE_ERROR));
			}

			if (uiDiff == 0)
			{
				
				// If there is no "diff" then current element does not change.
				// Move element down so many bytes and go on.

				if (uiBlkEnd <= pStack->uiCurElm)
				{
					return (RC_SET( FERR_BTREE_ERROR));
				}

				shiftN( pCurElm, (FLMUINT) (uiBlkEnd - pStack->uiCurElm),
						 uiNewElmLen);
			}
			else
			{

				// Move from the current element down so many bytes to fit the
				// new element.

				uiCurElmKeyLen = (FLMUINT) BBE_GET_KL( pCurElm);
				uiCurElmPkc += uiDiff;

				// Check if uiCurElmPkc has overflowed the max. value

				if (uiCurElmPkc > BBE_PKC_MAX)
				{
					uiDiff -= uiCurElmPkc - BBE_PKC_MAX;	// Could set diff to 0
					uiCurElmPkc = BBE_PKC_MAX;
				}

				uiCurElmKeyLen -= uiDiff;						// Remove diff bytes
																		
				// Shift from the current element's key+diff to end of block

				uiShiftLen = (uiBlkEnd - (pStack->uiCurElm + uiElmOvhd + uiDiff));
				uiShiftDist = (FLMUINT) (uiNewElmLen - uiDiff);

				// Change block end value to compensate for uiDiff and curElm
				// change

				uiBlkEnd -= uiDiff;

				// Move up the current element in two statments to re-compress

				shiftN( pCurElm + uiElmOvhd + uiDiff, uiShiftLen, uiShiftDist);

				// Output the new current element overhead

				FSSetElmOvhd( pCurElm + uiNewElmLen, uiElmOvhd, uiCurElmPkc,
								 uiCurElmKeyLen, pCurElm);
			}
		}
		else
		{

			// Else insert at the end of the block. These are special
			// controlled inserts

			if (pStack->uiCurElm != uiBlkEnd)
			{
				return (RC_SET( FERR_BTREE_ERROR));
			}
		}

		// Create the new elements element overhead portion

		FSSetElmOvhd( pCurElm, uiElmOvhd, uiNewElmPkc, uiNewElmKeyLen, pElement);

		// Move the part of the key and the rest of the record

		if (uiElmLen - (uiElmOvhd + uiNewElmPkc))
		{
			f_memcpy( pCurElm + uiElmOvhd, &pElement[uiElmOvhd + uiNewElmPkc],
						uiElmLen - (uiElmOvhd + uiNewElmPkc));
		}

		// Reset the block end

		uiBlkEnd += uiNewElmLen;
		pStack->uiBlkEnd = uiBlkEnd;

		UW2FBA( (FLMUINT16)uiBlkEnd, BLK_ELM_ADDR( pStack, BH_ELM_END));
		rc = FERR_OK;
	}

	return (rc);
}

/****************************************************************************
Desc: Set the element overhead given the needed values
****************************************************************************/
FLMUINT FSSetElmOvhd(
	FLMBYTE *	pElement,
	FLMUINT		uiElmOvhd,
	FLMUINT		uiPkc,
	FLMUINT		uiKeyLen,
	FLMBYTE *	origElm)
{
	FLMBYTE		byFirstByte;

	if (uiElmOvhd == BBE_KEY)
	{
		byFirstByte = (FLMBYTE) ((*origElm & 
			(BBE_FIRST_FLAG | BBE_LAST_FLAG)) + uiPkc);
		if (uiKeyLen > 0xFF)
		{
			byFirstByte |= (FLMBYTE) (((uiKeyLen) >> BBE_KL_SHIFT_BITS) & 
													BBE_KL_HBITS);
		}

		*pElement++ = byFirstByte;
		*pElement++ = (FLMBYTE) uiKeyLen;
		*pElement++ = origElm[BBE_RL];
	}
	else if (uiElmOvhd == BNE_DATA_OVHD)
	{
		f_memcpy( pElement, origElm, BNE_DATA_OVHD);
	}
	else
	{
		byFirstByte = (FLMBYTE) ((*origElm & 
			(BBE_FIRST_FLAG | BBE_LAST_FLAG)) + uiPkc);
			
		if (uiKeyLen > 0xFF)
		{
			byFirstByte |= (FLMBYTE) (((uiKeyLen) >> 
				BBE_KL_SHIFT_BITS) & BBE_KL_HBITS);
		}

		*pElement++ = byFirstByte;
		*pElement++ = (FLMBYTE) uiKeyLen;

		// Will copy 3 bytes for 2x dbs and 4 bytes ro 3x dbs.

		f_memcpy( pElement, &origElm[BNE_CHILD_BLOCK], 
				uiElmOvhd - BNE_CHILD_BLOCK);
	}

	return (uiElmOvhd);
}
