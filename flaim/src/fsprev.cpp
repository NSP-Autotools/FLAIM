//-------------------------------------------------------------------------
// Desc:	Traverse to previous element in a b-tree.
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
Desc:	Go to the previous element in the logical b-tree while building key
***************************************************************************/
RCODE FSBtPrevElm(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pStack)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBlkAddr;
	FLMUINT		uiTargetElm;
	FLMUINT		uiPrevElm = 0;
	FLMUINT		uiPrevKeyCnt = 0;
	FLMUINT		uiElmKeyLen = 0;
	FLMUINT		uiKeyBufSize = pStack->uiKeyBufSize;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMBYTE *	pCurElm;
	FLMBYTE *	pBlk;

	// Check if you are at or before the first element in the block

	if (pStack->uiCurElm <= BH_OVHD)
	{
		pBlk = BLK_PTR( pStack);

		// YES - read in the previous block & go to the last element

		if ((uiBlkAddr = (FLMUINT) FB2UD( &pBlk[BH_PREV_BLK])) == BT_END)
		{

			// Unless you are at the end

			rc = FERR_BT_END_OF_DATA;
		}
		else
		{
			if (RC_OK( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
			{

				// Set blkEnd & curElm and adjust parent block to previous
				// element

				pBlk = pStack->pBlk;
				pStack->uiCurElm = pStack->uiBlkEnd = pStack->uiBlkEnd;
				
				if (pStack->uiFlags & FULL_STACK)
				{
					rc = FSAdjustStack( pDb, pLFile, pStack, FALSE);
				}
			}
		}
	}

	// Move down 1 before the current element

	if (RC_OK( rc))
	{
		if (pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			pStack->uiCurElm -= BNE_DATA_OVHD;
			pBlk = pStack->pBlk;
			pCurElm = &pBlk[pStack->uiCurElm];
			flmCopyDrnKey( pStack->pKeyBuf, pCurElm);
			goto Exit;
		}

		// Set up pointing to first element in the block

		uiTargetElm = pStack->uiCurElm;
		pStack->uiCurElm = BH_OVHD;
		pBlk = pStack->pBlk;

		while (pStack->uiCurElm < uiTargetElm)
		{
			pCurElm = &pBlk[pStack->uiCurElm];
			uiPrevKeyCnt = (FLMUINT) (BBE_GET_PKC( pCurElm));
			uiElmKeyLen = (FLMUINT) (BBE_GET_KL( pCurElm));

			if (uiElmKeyLen + uiPrevKeyCnt > uiKeyBufSize)
			{
				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}

			if (uiElmKeyLen)
			{
				f_memcpy( &pStack->pKeyBuf[uiPrevKeyCnt], &pCurElm[uiElmOvhd],
							uiElmKeyLen);
			}

			uiPrevElm = pStack->uiCurElm;
			if (RC_BAD( rc = FSBlkNextElm( pStack)))
			{
				rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;
				break;
			}
		}

		pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen;
		pStack->uiCurElm = uiPrevElm;
	}

Exit:

	return (rc);
}

/***************************************************************************
Desc: Return the last DIN in the current element's reference list
***************************************************************************/
FLMUINT FSRefLast(
	BTSK *			pStack,			// Small stack to hold btree variables
	DIN_STATE *		pState,			// Holds offset, one run number, etc.
	FLMUINT *		puiDomainRV)	// Returns the elements domain
{
	FLMBYTE *		pCurElm = CURRENT_ELM( pStack);
	FLMBYTE *		pCurRef;
	FLMUINT			uiRefSize;

	// Point past the domain, ignore return value

	pCurRef = pCurElm;
	*puiDomainRV = FSGetDomain( &pCurRef, pStack->uiElmOvhd);
	uiRefSize = (FLMUINT) (BBE_GET_RL( pCurElm) - 
						(pCurRef - BBE_REC_PTR( pCurElm)));

	return (FSGetPrevRef( pCurRef, pState, uiRefSize));
}

/***************************************************************************
Desc: Position and return the previous reference saving the state
***************************************************************************/
FLMUINT FSGetPrevRef(
	FLMBYTE *		pCurRef,
	DIN_STATE *		pState,
	FLMUINT			uiTarget)
{
	FLMUINT			uiDin;
	FLMUINT			uiOneRuns = 0;
	FLMUINT			uiDelta = 0;
	FLMUINT			uiLastOffset = 0;
	FLMBYTE			byValue;

	RESET_DINSTATE_p( pState);
	uiDin = DINNextVal( pCurRef, pState);

	while (pState->uiOffset < uiTarget)
	{

		// Get the current byte to see what kind of item it is

		byValue = (FLMBYTE) pCurRef[uiLastOffset = pState->uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{
			uiDelta = 0;
			uiOneRuns = DINOneRunVal( pCurRef, pState);
			uiDin -= uiOneRuns;
		}
		else
		{
			uiDelta = DINNextVal( pCurRef, pState);
			uiDin -= uiDelta;
		}
	}

	// Hit the end of the reference set for the current element. The
	// current din is a correct return value. The pState structure must be
	// setup to refer to the last entry using uiLastOffset.

	if ((pState->uiOffset = uiLastOffset) != 0)
	{
		if (uiDelta == 0)
		{

			// One runs was the last entry, setup for one run state *

			uiOneRuns--;
			pState->uiOnes = uiOneRuns;
		}
	}

	return (uiDin);
}

/***************************************************************************
Desc: Go to the previous reference given a valid cursor.
***************************************************************************/
RCODE FSRefPrev(
	FDB *				pDb,
	LFILE *			pLFile,
	BTSK *			pStack,
	DIN_STATE *		pState,
	FLMUINT *		puiDinRV)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurRef;
	FLMBYTE *		pCurElm;
	FLMUINT			uiDin = *puiDinRV;
	FLMUINT			uiDummyDomain;
	FLMBYTE			byValue;

	// Point to the start of the current reference

	pCurRef = pCurElm = CURRENT_ELM( pStack);
	FSGetDomain( &pCurRef, pStack->uiElmOvhd);

	// Was this the first reference

	if (pState->uiOffset == 0)
	{

		// Read in the previous element or return FERR_BT_END_OF_DATA if
		// first

		if (BBE_IS_FIRST( pCurElm))
		{
			return (FERR_BT_END_OF_DATA);
		}

		if (RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
		{
			return (rc);
		}

		uiDin = FSRefLast( pStack, pState, &uiDummyDomain);
	}
	else
	{

		// Get current byte - could be a 1 run

		byValue = pCurRef[pState->uiOffset];

		if (DIN_IS_REAL_ONE_RUN( byValue) && pState->uiOnes)
		{
			uiDin++;
			pState->uiOnes--;
		}
		else
		{
			uiDin = FSGetPrevRef( pCurRef, pState, pState->uiOffset);
		}
	}

	*puiDinRV = uiDin;
	return (FERR_OK);
}
