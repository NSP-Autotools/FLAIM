//-------------------------------------------------------------------------
// Desc:	Traverse to next element in a b-tree.
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

// The SENLenArray[] is used to find the length of an unsigned SEN value.

FLMBYTE	SENLenArray[] = { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 0 };

/***************************************************************************
Desc: Go to the next data record given a stack.
***************************************************************************/
RCODE FSNextRecord(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pStack)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pCurElm;

	pStack->uiFlags = NO_STACK;
	pStack->uiKeyBufSize = DIN_KEY_SIZ;
	pCurElm = CURRENT_ELM( pStack);

	// Scan over the current record till 'does continue' flag NOT set

	while (BBE_NOT_LAST( pCurElm))
	{

		// First go to the next element - rc may return FERR_BT_END_OF_DATA

		if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
		{
			if (rc == FERR_BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_BTREE_ERROR);
			}

			goto Exit;
		}

		pCurElm = CURRENT_ELM( pStack);
	}

	// Now go to the next element.

	if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
	{
		if (rc == FERR_BT_END_OF_DATA)
		{
			rc = RC_SET( FERR_EOF_HIT);
		}

		goto Exit;
	}

Exit:

	return (rc);
}

/***************************************************************************
Desc:	Go to the next reference given a valid cursor.
***************************************************************************/
RCODE FSRefNext(
	FDB *				pDb,
	LFILE *			pLFile,
	BTSK *			pStack,
	DIN_STATE *		pState,
	FLMUINT *		puiDin)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurRef;
	FLMBYTE *		pCurElm;
	FLMUINT			uiRefSize;
	FLMUINT			uiHasDomain;
	FLMUINT			uiDin = *puiDin;
	DIN_STATE		savedState;

	// Point to the start of the current reference

	pCurRef = pCurElm = CURRENT_ELM( pStack);
	uiHasDomain = FSGetDomain( &pCurRef, pStack->uiElmOvhd);
	uiRefSize = (FLMUINT) (BBE_GET_RL( pCurElm) - 
							(pCurRef - BBE_REC_PTR( pCurElm)));

	if (pState->uiOffset < uiRefSize)
	{
		DINNextVal( pCurRef, pState);
	}

	if (pState->uiOffset >= uiRefSize)
	{

		// Read in the next element if a domain was found else
		// FERR_BT_END_OF_DATA

		if (!uiHasDomain)
		{	
			// May use the DOES_CONT element flag
			
			return (FERR_BT_END_OF_DATA);
		}

		if (RC_BAD( rc = FSBtNextElm( pDb, pLFile, pStack)))
		{
			return (rc);
		}

		uiDin = FSRefFirst( pStack, pState, &uiHasDomain);
	}
	else
	{

		// Don't move the pState, stay put and get the next DIN value

		savedState.uiOffset = pState->uiOffset;
		savedState.uiOnes = pState->uiOnes;

		uiDin -= DINNextVal( pCurRef, &savedState);
	}

	*puiDin = uiDin;
	return (FERR_OK);
}

/***************************************************************************
Desc:	Search for and position to the current (or next) reference
***************************************************************************/
RCODE FSRefSearch(
	BTSK *			pStack,
	DIN_STATE *		pState,
	FLMUINT *		puiDin)
{
	FLMBYTE *		pCurRef;
	FLMBYTE *		pCurElm;
	FLMUINT			uiRefSize;
	FLMUINT			uiLastOffset;
	FLMUINT			uiDelta;
	FLMUINT			uiTargetDin = *puiDin;
	FLMUINT			uiDin;
	FLMUINT			uiOneRuns;
	DIN_STATE		tempState;
	FLMBYTE			byValue;

	// Point to the start of the current reference

	pCurRef = pCurElm = CURRENT_ELM( pStack);
	(void) FSGetDomain( &pCurRef, pStack->uiElmOvhd);
	uiRefSize = (FLMUINT) (BBE_GET_RL( pCurElm) - 
							(pCurRef - BBE_REC_PTR( pCurElm)));

	RESET_DINSTATE_p( pState);
	RESET_DINSTATE( tempState);

	uiLastOffset = tempState.uiOffset;
	uiDin = DINNextVal( pCurRef, &tempState);

	if (uiDin > uiTargetDin)
	{
		while (tempState.uiOffset < uiRefSize)
		{

			// Get the current byte to see what kind of item it is

			byValue = (FLMBYTE) pCurRef[uiLastOffset = tempState.uiOffset];

			if (DIN_IS_REAL_ONE_RUN( byValue))
			{
				uiOneRuns = DINOneRunVal( pCurRef, &tempState);

				// Check if one run is includes the target din

				if ((uiDin - uiOneRuns) <= uiTargetDin)
				{
					uiOneRuns = (uiDin - uiTargetDin) - 1;
					pState->uiOffset = uiLastOffset;
					pState->uiOnes = uiOneRuns;
					uiDin = uiTargetDin;
					break;
				}

				uiDin -= uiOneRuns;
			}
			else
			{
				uiDelta = DINNextVal( pCurRef, &tempState);

				// Check if next din value is equal or less than target din

				uiDin -= uiDelta;
				if (uiDin <= uiTargetDin)
				{
					pState->uiOffset = uiLastOffset;
					break;
				}
			}
		}

		pState->uiOffset = uiLastOffset;
	}

	*puiDin = uiDin;
	return ((uiDin == uiTargetDin) ? FERR_OK : RC_SET( FERR_FAILURE));
}

/***************************************************************************
Desc:	Get the next DIN value from the DIN set.
***************************************************************************/
FLMUINT DINNextVal(
	FLMBYTE *		puiDin,
	DIN_STATE *		pState)
{
	FLMBYTE *		pOneRun;
	FLMBYTE *		pCurDin;
	FLMUINT			uiTemp = 0;
	FLMUINT			uiOneRun;
	FLMUINT			uiStateOneRuns;

	pCurDin = puiDin + pState->uiOffset;

	switch (SENValLen( pCurDin))
	{
		case 0:
		{
			uiOneRun = 0;
			pOneRun = pCurDin + 1;
			
			if (*pCurDin < DIN_ONE_RUN_HV)
			{
				uiOneRun = (*pCurDin - DIN_ONE_RUN_LV) + 2;
			}
			else if (*pCurDin == DIN_ONE_RUN_HV)
			{
				uiOneRun = SENNextVal( &pOneRun);
			}
			else
			{

				// Invalid code found

				pCurDin++;
				uiTemp = 0;
				break;
			}

			// Handle the position of the one run

			uiStateOneRuns = pState->uiOnes;

			// return 1 unless on last one run value
			
			uiTemp = 1;
			uiStateOneRuns++;
			
			if (uiStateOneRuns >= uiOneRun)
			{
				pCurDin = pOneRun;				// Set to after the one runs
				uiStateOneRuns = 0;				// This sets state of ones to zero
			}

			pState->uiOnes = uiStateOneRuns;
			break;
		}
		
		case 1:
		{
			uiTemp = *pCurDin++;
			break;
		}
		
		case 2:
		{
			uiTemp = ((FLMUINT) (SEN_2B_MASK &*pCurDin++)) << 8;
			goto DINNV_1_byte;
		}
		
		case 3:
		{
			uiTemp = ((FLMUINT) (SEN_3B_MASK & (*pCurDin++))) << 16;
			goto DINNV_2_bytes;
		}
		
		case 4:
		{
			uiTemp = ((FLMUINT) (SEN_4B_MASK & (*pCurDin++))) << 24;
			goto DINNV_3_bytes;
		}
		
		case 5:
		{
			pCurDin++;
			uiTemp = ((FLMUINT) * pCurDin++) << 24;
	DINNV_3_bytes:

			uiTemp += ((FLMUINT) * pCurDin++) << 16;
	DINNV_2_bytes:

			uiTemp += ((FLMUINT) * pCurDin++) << 8;
	DINNV_1_byte:

			uiTemp += *pCurDin++;
			break;
		}
	}

	// Set the offset to point to the next reference

	pState->uiOffset = (FLMUINT) (pCurDin - puiDin);
	return (uiTemp);
}

/***************************************************************************
Desc:	Get the next one run value and update the state information
***************************************************************************/
FLMUINT DINOneRunVal(
	FLMBYTE *		puiDin,
	DIN_STATE *		pState)
{
	FLMBYTE *		pOneRun;
	FLMBYTE *		pCurDin;
	FLMUINT			uiOneRun;

	pCurDin = puiDin + pState->uiOffset;

	if (*pCurDin == 1)
	{
		pState->uiOffset++;
		uiOneRun = 1;
	}
	else
	{
		uiOneRun = 0;
		pOneRun = pCurDin + 1;
		
		if (*pCurDin < DIN_ONE_RUN_HV)
		{
			uiOneRun = (*pCurDin - DIN_ONE_RUN_LV) + 2;
		}
		else if (*pCurDin == DIN_ONE_RUN_HV)
		{
			uiOneRun = SENNextVal( &pOneRun);
		}
		else
		{
			// Invald code found
			
			uiOneRun = 0;
		}

		pState->uiOffset = (FLMUINT) (pOneRun - puiDin);
	}

	return (uiOneRun);
}

/***************************************************************************
Desc:	Return the next SEN value.
***************************************************************************/
FLMUINT SENNextVal(
	FLMBYTE **	 	pSenRV)
{
	FLMUINT			uiTemp;
	FLMBYTE *		pSen = *pSenRV;

	switch (SENValLen( pSen))
	{
		case 1:
		{
			uiTemp = *pSen;
			break;
		}
		
		case 2:
		{
			uiTemp = ((FLMUINT) (SEN_2B_MASK &*pSen++)) << 8;
			uiTemp += *pSen;
			break;
		}
		
		case 3:
		{
			uiTemp = ((FLMUINT) (SEN_3B_MASK & (*pSen++))) << 16;
			goto SENNV_2_bytes;
		}
		
		case 4:
		{
			uiTemp = ((FLMUINT) (SEN_4B_MASK & (*pSen++))) << 24;
			goto SENNV_3_bytes;
		}
		
		case 5:
		{
			pSen++;
			uiTemp = ((FLMUINT) * pSen++) << 24;
	SENNV_3_bytes:

			uiTemp += ((FLMUINT) * pSen++) << 16;
	SENNV_2_bytes:

			uiTemp += ((FLMUINT) * pSen++) << 8;
			uiTemp += *pSen;
			break;
		}
		
		default:
		{
			uiTemp = 0;
			break;
		}
	}

	*pSenRV = pSen + 1;
	return (uiTemp);
}

/***************************************************************************
Desc: Get the domain from a block type and current element pointer
***************************************************************************/
FLMUINT FSGetDomain(
	FLMBYTE **		curElmRV,
	FLMUINT			uiElmOvhd)
{
	FLMUINT			uiDinDomain = 0;
	FLMBYTE *		curElm = *curElmRV;

	if (uiElmOvhd == BBE_KEY)
	{

		// Normal leaf block, parse element to see if DOMAIN flag is present

		curElm += BBE_REC_OFS( curElm);

		// Skip past the update version information

		if (*curElm == SEN_DOMAIN)
		{
			curElm++;
			uiDinDomain = SENNextVal( &curElm);
		}
	}
	else
	{
		if (BNE_IS_DOMAIN( curElm))
		{
			curElm += BBE_GET_KL( curElm) + uiElmOvhd;
			uiDinDomain = ((FLMUINT) * curElm++) << 16;
			uiDinDomain |= (FLMUINT)(((FLMUINT16) * curElm++) << 8);
			uiDinDomain |= *curElm++;
		}
	}

	*curElmRV = curElm;
	return (uiDinDomain);
}
