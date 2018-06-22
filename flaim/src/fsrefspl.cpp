//-------------------------------------------------------------------------
// Desc:	Index reference splitting routines.
// Tabs:	3
//
// Copyright (c) 1991-2000, 2002-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC FLMUINT FSSplitRefSet(
	FLMBYTE *		leftBuf,
	FLMUINT *		leftLenRV,
	FLMBYTE *		rightBuf,
	FLMUINT *		rightLenRV,
	FLMBYTE *		refPtr,
	FLMUINT			uiRefLen,
	FLMUINT			uiSplitFactor);

/***************************************************************************
Desc:	Try to split a reference set. The size is over the first threshold.
		If you split this update the b-tree with the new element and position
		to the current element for insert of the din.
***************************************************************************/
RCODE FSRefSplit(
	FDB *				pDb,
	LFILE *			pLFile,
	BTSK **		 	pStackRV,
	FLMBYTE *		pElmBuf,
	FLMUINT			din,
	FLMUINT			uiDeleteFlag,
	FLMUINT			uiSplitFactor)
{
	RCODE				rc = FERR_OK;
	BTSK *			pStack = *pStackRV;
	FLMBYTE *		pCurElm = CURRENT_ELM( pStack);
	FLMINT			iElmLen;
	FLMBYTE			leftBuf[MAX_REC_ELM];
	FLMUINT			leftDomain;
	FLMUINT			leftLen;
	FLMBYTE			rightBuf[MAX_REC_ELM];
	FLMUINT			rightDomain;
	FLMUINT			rightLen;
	FLMBYTE *		refPtr;
	FLMBYTE *		recPtr;
	FLMUINT			uiRefLen;
	FLMUINT			firstFlag = 0;

	refPtr = pCurElm;
	recPtr = BBE_REC_PTR( pCurElm);
	rightDomain = FSGetDomain( &refPtr, (FLMBYTE) pStack->uiElmOvhd);
	uiRefLen = (FLMUINT) (BBE_GET_RL( pCurElm) - (FLMUINT) (refPtr - recPtr));
	
FSRS_try_again:

	leftDomain = FSSplitRefSet( leftBuf, &leftLen, rightBuf, &rightLen, refPtr,
										uiRefLen, uiSplitFactor);

	if (leftDomain == 0)
	{
		// Split failed, setup to add
		//
		// Try again using a different split factor - OK to fail above;
		// In the future, should just handle no splitting and go on
		
		if (uiSplitFactor == SPLIT_50_50)
		{
			uiSplitFactor = SPLIT_90_10;
			goto FSRS_try_again;
		}

		// Setup for inserting the din into the right buffer and call
		// replace

		leftDomain = DIN_DOMAIN( din) + 1;
		f_memcpy( rightBuf, refPtr, rightLen = uiRefLen);
		leftLen = 0;
	}

	// Write the right element's references. Write the right domain if
	// non-zero and replace element

	iElmLen = (FLMINT) (BBE_REC_OFS( pElmBuf));
	refPtr = recPtr = &pElmBuf[iElmLen];
	
	if (rightDomain)
	{
		*refPtr++ = SEN_DOMAIN;
		SENPutNextVal( &refPtr, rightDomain);
	}

	if (DIN_DOMAIN( din) < leftDomain)
	{

		// Build element inserting the input din

		if (uiDeleteFlag)
		{
			if (FSSetDeleteRef( refPtr, rightBuf, din, &rightLen))
			{

				// rightLen should not have changed if error found

				return (RC_SET( FERR_KEY_NOT_FOUND));
			}
		}
		else if (FSSetInsertRef( refPtr, rightBuf, din, &rightLen))
		{

			// Reference there so give up and return success

			goto Exit;
		}
	}
	else
	{
		f_memcpy( refPtr, rightBuf, rightLen);
	}

	// The other flags and lengths are been set by the caller

	iElmLen += BBE_SET_RL( pElmBuf, rightLen + (FLMUINT) (refPtr - recPtr));

	if (BBE_IS_FIRST( pElmBuf) && leftLen)
	{
		firstFlag++;
		BBE_CLR_FIRST( pElmBuf);

		// Log the block before modifying it.

		if (RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			goto Exit;
		}

		pCurElm = CURRENT_ELM( pStack);
		BBE_CLR_FIRST( pCurElm);

		// Call replace below because FIRST flag is now clear
	}

	// Can call replace because FIRST flag was NOT set

	if (RC_BAD( rc = FSBtReplace( pDb, pLFile, &pStack, pElmBuf, iElmLen)))
	{
		goto Exit;
	}

	// Write the left buffer Should be positioned to the right buffer

	if (leftLen)
	{
		
		// Adjust variables to build and point to the left buffers
		// references. Set the domain and insert into the b-tree. Then go to
		// the next element

		BBE_CLR_LAST( pElmBuf);
		if (firstFlag)
		{
			BBE_SET_FIRST( pElmBuf);
		}

		iElmLen = (FLMINT) (BBE_REC_OFS( pElmBuf));
		refPtr = recPtr = &pElmBuf[iElmLen];
		*refPtr++ = SEN_DOMAIN;
		SENPutNextVal( &refPtr, leftDomain);

		if (DIN_DOMAIN( din) >= leftDomain)
		{

			// Build element inserting the input din

			if (uiDeleteFlag)
			{
				if (FSSetDeleteRef( refPtr, leftBuf, din, &leftLen))
				{
					return (RC_SET( FERR_KEY_NOT_FOUND));
				}
			}
			else
			{
				if (FSSetInsertRef( refPtr, leftBuf, din, &leftLen))
				{
					f_memcpy( refPtr, leftBuf, leftLen);
				}
			}
		}
		else
		{
			f_memcpy( refPtr, leftBuf, leftLen);
		}

		iElmLen += BBE_SET_RL( pElmBuf, leftLen + (FLMUINT) (refPtr - recPtr));

		// Setup the pStack and bsKeyBuf[] for the insert

		if (RC_BAD( rc = FSBtScanTo( pStack, &pElmBuf[BBE_KEY],
					  (FLMUINT) (BBE_GET_KL( pElmBuf)), 0)))
		{
			goto Exit;
		}

		rc = FSBtInsert( pDb, pLFile, &pStack, pElmBuf, iElmLen);
	}

Exit:

	return (rc);
}

/***************************************************************************
Desc: Split a reference set within a domain value. If buffer cannot be
		split then will return a leftDomain value of ZERO. Must have a 
		minimum of 2 references in left and right buffers.
***************************************************************************/
FSTATIC FLMUINT FSSplitRefSet(
	FLMBYTE *		leftBuf,
	FLMUINT *		leftLenRV,
	FLMBYTE *		rightBuf,
	FLMUINT *		rightLenRV,
	FLMBYTE *		refPtr,
	FLMUINT			uiRefLen,
	FLMUINT			uiSplitFactor)
{
	FLMUINT			leftDomain = 0;
	FLMUINT			din = 0;
	FLMUINT			oneRuns = 0;
	FLMUINT			delta;
	FLMUINT			rightLen;
	FLMUINT			offsetTarget;
	DIN_STATE		leftState;
	DIN_STATE		rightState;
	DIN_STATE		refState;
	FLMBYTE			byValue;
	FLMUINT			uiLeftCnt;

	RESET_DINSTATE( leftState);
	RESET_DINSTATE( rightState);
	RESET_DINSTATE( refState);

	offsetTarget = (uiSplitFactor == SPLIT_90_10) 
								? REF_SPLIT_90_10 
								: REF_SPLIT_50_50;
	
	// Read the first din value

	din = DINNextVal( refPtr, &refState);
	DINPutNextVal( leftBuf, &leftState, din);
	uiLeftCnt = 1;

	// Must have at least 2 in the left buffer.

	while (refState.uiOffset < offsetTarget || uiLeftCnt < 2)
	{
		byValue = refPtr[refState.uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{
			oneRuns = DINOneRunVal( refPtr, &refState);
			DINPutOneRunVal( leftBuf, &leftState, oneRuns);
			din -= oneRuns;
		}
		else
		{
			delta = DINNextVal( refPtr, &refState);
			DINPutNextVal( leftBuf, &leftState, delta);
			din -= delta;
		}

		uiLeftCnt++;
	}

	// Made it past the target point - find where domain changes

	leftDomain = DIN_DOMAIN( din);

	// Don't parse past the end

	while (refState.uiOffset < uiRefLen)
	{
		byValue = refPtr[refState.uiOffset];
		if (DIN_IS_REAL_ONE_RUN( byValue))
		{
			oneRuns = DINOneRunVal( refPtr, &refState);
			if (DIN_DOMAIN( din - oneRuns) != leftDomain)
			{

				// This is tricky, write only correct number of one runs

				delta = din & 0xFF;
				if (delta)
				{
					DINPutOneRunVal( leftBuf, &leftState, delta);
				}

				// Increment delta because setting up for next element

				delta++;
				oneRuns -= delta;

				// Write din and one runs below

				din -= delta;
				break;
			}

			DINPutOneRunVal( leftBuf, &leftState, oneRuns);
			din -= oneRuns;
		}
		else
		{
			delta = DINNextVal( refPtr, &refState);
			din -= delta;
			if (DIN_DOMAIN( din) != leftDomain)
			{
				oneRuns = 0;
				break;
			}

			DINPutNextVal( leftBuf, &leftState, delta);
		}
	}

	if (refState.uiOffset == uiRefLen)
	{
		 // Cannot split, caller take care of
		
		return (0);
	}

	// Start writing to the right side, compare /w uiRefLen proves > 2 refs

	DINPutNextVal( rightBuf, &rightState, din);
	if (oneRuns)
	{
		DINPutOneRunVal( rightBuf, &rightState, oneRuns);
	}

	*leftLenRV = leftState.uiOffset;
	rightLen = (FLMUINT) (uiRefLen - refState.uiOffset);

	f_memcpy( &rightBuf[rightState.uiOffset], &refPtr[refState.uiOffset],
				rightLen);

	*rightLenRV = rightLen + rightState.uiOffset;
	return (leftDomain);
}
