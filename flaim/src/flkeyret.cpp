//-------------------------------------------------------------------------
// Desc:	Retrieve keys from an index.
// Tabs:	3
//
// Copyright (c) 1998-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC RCODE	flmKeyRetrieveCS(
	FDB *				pDb,
	FLMUINT			uiIndex,
	FLMUINT			uiContainer,
	FlmRecord *		pKeyTree,
	FLMUINT			uiRefDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecordRV,
	FLMUINT *		puiDrnRV);

FSTATIC RCODE flmNextKey(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pStack,
	FLMUINT *	pudRefDrn);


/****************************************************************************
Desc:		Retrieves a key from an index based on a passed-in GEDCOM tree and DRN.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmKeyRetrieve(
	HFDB				hDb,
	FLMUINT			uiIndex,
	FLMUINT			uiContainer,
	FlmRecord *		pKeyTree,
	FLMUINT			uiRefDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecordRV,
	FLMUINT *		puiDrnRV
	)
{
	BTSK			stack[ BH_MAX_LEVELS];
	FLMBOOL		bStackInitialized = FALSE;
	BTSK *		pStack = &stack[0];
	DIN_STATE	dinState;
	FDB *			pDb = (FDB *)hDb;
	IXD *			pIxd = NULL;
	LFILE *		pLFile;
	FLMBYTE *	pSearchKeyBuf = NULL;
	FLMBYTE *	pKeyBuf = NULL;
	void *		pvMark = NULL;
	FLMUINT		uiFoundDrn = 0;
	FLMUINT		uiDomain;
	FLMUINT		uiElmDomain;
	FLMUINT		uiSearchKeyLen;
	FLMBOOL		bImplicitTrans = FALSE;
	FLMBYTE		pSmallBuf[8];
	FLMUINT		uiKeyRelPos;
	RCODE			rc;

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	pvMark = pDb->TempPool.poolMark();

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		rc = flmKeyRetrieveCS( pDb, uiIndex, uiContainer,
			pKeyTree, uiRefDrn, uiFlag, 
			ppRecordRV, puiDrnRV);
		goto Exit_CS;
	}

	if( pKeyTree || (uiFlag & FO_LAST))
	{
		if( RC_BAD( rc = pDb->TempPool.poolAlloc( MAX_KEY_SIZ + 4, 
			(void **)&pSearchKeyBuf)))
		{
			goto Exit;
		}
	}
	else
	{
		pSearchKeyBuf = pSmallBuf;
	}
	
	if( RC_BAD( rc = pDb->TempPool.poolAlloc( MAX_KEY_SIZ + 4, 
		(void **)&pKeyBuf)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
										FDB_TRANS_GOING_OK, 0, &bImplicitTrans)))
	{
		goto Exit;
	}
	if( RC_BAD( rc = fdictGetIndex(
			pDb->pDict, pDb->pFile->bInLimitedMode,
			uiIndex, &pLFile, &pIxd)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = KYFlushKeys( pDb)))
	{
		goto Exit;
	}

	if (uiFlag & (FO_LAST | FO_FIRST))
	{
		if (uiFlag & FO_FIRST)
		{
			pSearchKeyBuf[0] = 0;
			uiSearchKeyLen = 1;

			// Get rid of other bits that may have been set.
			// Handle like FO_INCL from here on out.
			// FO_FIRST takes precedence over other bits that
			// may have been set.

			uiFlag = FO_INCL;
		}
		else
		{
			f_memset( pSearchKeyBuf, 0xFF, MAX_KEY_SIZ);
			uiSearchKeyLen = MAX_KEY_SIZ;

			// Get rid of other bits that may have been set.

			uiFlag = FO_LAST;
		}
		uiRefDrn = 0;
	}
	else if (pKeyTree)
	{
		if( RC_BAD( rc = KYTreeToKey( pDb, pIxd, pKeyTree,
				uiContainer,
				pSearchKeyBuf, &uiSearchKeyLen, 0)))
		{
			goto Exit;
		}
	}
	else
	{
		pSearchKeyBuf[0] = 0;
		uiSearchKeyLen = 1;
	}

	uiDomain = (FLMUINT)((uiRefDrn) ? DIN_DOMAIN( uiRefDrn) + 1
											  : 0);

	FSInitStackCache( &stack[0], BH_MAX_LEVELS);
	bStackInitialized = TRUE;
	pStack->pKeyBuf = pKeyBuf;
		
	/* Search the B-Tree for the key. */
	
	if (RC_BAD( rc = FSBtSearch( pDb, pLFile, &pStack, pSearchKeyBuf,
										  uiSearchKeyLen, uiDomain)))
	{
		goto Exit;
	}

	uiKeyRelPos = pStack->uiCmpStatus;

	// Handle FO_LAST case - If FO_LAST bit was set, uiFlag will
	// have been changed above to simply be equal to FO_LAST.

	if (uiFlag == FO_LAST)
	{
		FLMUINT		uiTmp;
		DIN_STATE	DinState;

		if (pStack->uiBlkAddr == BT_END)
		{
			rc = RC_SET( FERR_BOF_HIT);
			goto Exit;
		}

		// Should be positioned at end of data in the B-tree

		flmAssert( uiKeyRelPos == BT_END_OF_DATA);

		// Position to the last element in the block.

		if (RC_BAD( rc = FSBtPrevElm( pDb, pLFile, pStack)))
		{
			if (rc == BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_BOF_HIT);
			}
			goto Exit;
		}
		RESET_DINSTATE( DinState);
		uiFoundDrn = FSRefLast( pStack, &DinState, &uiTmp);
		goto Make_Key;
	}
	if( (uiKeyRelPos == BT_END_OF_DATA) ||
		 ((uiFlag & (FO_EXACT | FO_KEY_EXACT)) && (uiKeyRelPos != BT_EQ_KEY)))
	{
		rc = (uiFlag & FO_EXACT) 
			? RC_SET( FERR_NOT_FOUND)
			: RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	// NOTE: dinState will be initialized by FSRefFirst - no
	// need to memset prior to calling.
	uiFoundDrn = FSRefFirst( pStack, &dinState, &uiElmDomain);

	// Key positioning returns the first reference.
	if( !uiRefDrn)
	{

		// If exclusive and NOT key exact
		if( (uiFlag & (FO_EXCL | FO_KEY_EXACT)) == FO_EXCL)
		{
			if( uiKeyRelPos == BT_EQ_KEY)
			{
				if( RC_BAD( rc = flmNextKey( pDb, pLFile, pStack, &uiFoundDrn)))
				{
					goto Exit;
				}
			}
		}
		// else already positioned to key and set uiFoundDrn.
		// Cases are FO_INCL, (FO_KEY_EXACT | FO_INCL), (FO_KEY_EXACT | FO_EXCL)
	}
	else	// uiRefDrn != 0
	{
		// Code below falls through with rc.

		/*
		Handles both cases of FO_KEY_EXACT being set or not set.

		FO_KEY_EXACT cases.
			Position to the exact key and FO flags act on the uiRefDrn value.

			FO_EXACT - exact on uiRefDrn
			FO_EXCL - Go exclusive of the input uiRefDrn or first ref if 0.
				Returns FERR_NOT_FOUND when no more DRN references follow.
			FO_INCL - Go inclusive of the current or next reference.
				Returns FERR_NOT_FOUND when no more DRN references follow.
		*/

		if( uiFlag & FO_EXACT)
		{
			/*
			Reading the current element, position to or after uiFoundDrn.
			Anything but success means we did not find the exact DRN.
			*/
		
			uiFoundDrn = uiRefDrn;
			if( RC_BAD( rc = FSRefSearch( pStack, &dinState, &uiFoundDrn)))
			{
				rc = RC_SET( FERR_NOT_FOUND);
			}
		}
		else if (uiFlag & FO_EXCL)
		{
			if (uiKeyRelPos == BT_EQ_KEY)
			{
				/* Need to find reference and position to the next one */

				uiFoundDrn = uiRefDrn;
				rc = FSRefSearch( pStack, &dinState, &uiFoundDrn);

				/*
				SUCCESS means we found the DRN, need to position to
				one past it.  FERR_FAILURE means we are positioned
				to a DRN that is SMALLER than the one we are
				looking for or we are at the end of the reference set.
				*/

				if( RC_OK( rc))
				{
					if( (rc = FSRefNext( pDb, pLFile, pStack, &dinState,
										&uiFoundDrn)) == FERR_BT_END_OF_DATA)
					{
						if (uiFlag & FO_KEY_EXACT)
						{
							rc = RC_SET( FERR_EOF_HIT);
						}
						else
						{
							rc = flmNextKey( pDb, pLFile, pStack, &uiFoundDrn);
						}
					}
				}
				else if( rc == FERR_FAILURE)
				{
					/*
					If FSRefSearch returns a non-zero reference,
					we are positioned on a DRN that is SMALLER
					the one we searched for.  Otherwise, we are
					at the end of that key's reference set and we
					need to go to the next key.
					*/

					if (uiFoundDrn)
					{
						rc = FERR_OK;
					}
					else if (uiFlag & FO_KEY_EXACT)
					{
						rc = RC_SET( FERR_EOF_HIT);
					}
					else
					{
						rc = flmNextKey( pDb, pLFile, pStack, &uiFoundDrn);
					}
				}
			}
			else if (uiFlag & FO_KEY_EXACT)
			{
				rc = RC_SET( FERR_EOF_HIT);
			}
			// else already positioned on the next key and uiFoundDrn is set.
		}
		else		// FO_INCL
		{
			if (uiKeyRelPos == BT_EQ_KEY)
			{

				/* Need to find reference if possible. */

				uiFoundDrn = uiRefDrn;
				rc = FSRefSearch( pStack, &dinState, &uiFoundDrn);

				/*
				SUCCESS means we found the DRN.  FERR_FAILURE means we
				are either positioned past the DRN or we are at the
				end of the key's reference set.
				*/

				if( rc == FERR_FAILURE)
				{
					/*
					If FSRefSearch returns a non-zero reference,
					we are positioned on a DRN that is SMALLER than
					the one we searched for.  Otherwise, we are
					at the end of that key's reference set and we
					need to go to the next key.
					*/

					if (uiFoundDrn)
					{
						rc = FERR_OK;
					}
					else if (uiFlag & FO_KEY_EXACT)
					{
						rc = RC_SET( FERR_EOF_HIT);
					}
					else
					{
						rc = flmNextKey( pDb, pLFile, pStack, &uiFoundDrn);
					}
				}
			}
			else if (uiFlag & FO_KEY_EXACT)
			{
				rc = RC_SET( FERR_EOF_HIT);
			}
			// else already positioned on the next key and uiFoundDrn is set.
		}
	}

	/*
	If everything went OK, render the key in GEDCOM form and return it together
	with the reference DRN.
	*/

Make_Key:
	if( RC_OK( rc) && ppRecordRV)
	{

		/*
		VISIT: We must build a fat tree and not a flat tree. (TRUE in last parm is fat)
		Need to visit all of smi\fixcalls.cpp.
		*/
		if( RC_OK( rc = flmIxKeyOutput( pIxd,
							pStack->pKeyBuf, pStack->uiKeyLen, ppRecordRV, TRUE)))
		{
			(*ppRecordRV)->setID( uiFoundDrn);
		}
	}

Exit:
	if( RC_OK(rc) && puiDrnRV)
	{
		*puiDrnRV = uiFoundDrn;
	}

	if( bStackInitialized)
	{
		FSReleaseStackCache( stack, BH_MAX_LEVELS, FALSE);
	}

	/* If we started an implicit transaction, abort it here */

	if( bImplicitTrans)
	{
		(void)flmAbortDbTrans( pDb);
	}

Exit_CS:

	pDb->TempPool.poolReset( pvMark);
	flmExit( FLM_KEY_RETRIEVE, pDb, rc);

	return( rc);
}

/****************************************************************************
Desc:		Retrieves a key from an index based on a passed-in GEDCOM tree and DRN.
VISIT:	On reading data records and FO_EXCL, increment the DRN instead of
			positioning to the DRN and then scanning to the next record.
****************************************************************************/
FSTATIC RCODE	flmKeyRetrieveCS(
	FDB *				pDb,
	FLMUINT			uiIndex,
	FLMUINT			uiContainer,
	FlmRecord *		pKeyTree,
	FLMUINT			uiRefDrn,
	FLMUINT			uiFlag,
	FlmRecord **	ppRecordRV,
	FLMUINT *		puiDrnRV)
{
	RCODE				rc;
	CS_CONTEXT *	pCSContext = pDb->pCSContext;
	FCL_WIRE			Wire( pCSContext, pDb);
	void *			pvMark = pCSContext->pool.poolMark();

	/*
	Set the record object so that it can be re-used,
	if possible
	*/

	if( ppRecordRV)
	{
		Wire.setRecord( *ppRecordRV);
		if( *ppRecordRV)
		{
			(*ppRecordRV)->Release();
			*ppRecordRV = NULL;
		}
	}

	/*
	Set the temporary pool
	*/

	Wire.setPool( &pCSContext->pool);

	/*
	Send the request
	*/

	if( RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_RECORD, FCS_OP_KEY_RETRIEVE)))
	{
		goto Exit;
	}

	if( pCSContext->uiServerFlaimVer >= FLM_FILE_FORMAT_VER_4_50)
	{
		if( uiIndex)
		{
			if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_INDEX_ID, uiIndex)))
			{
				goto Transmission_Error;
			}
		}

		if( uiContainer)
		{
			if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_CONTAINER_ID, uiContainer)))
			{
				goto Transmission_Error;
			}
		}
	}
	else
	{

		// Older versions of server expect the index in the CONTAINER tag.

		if( uiIndex)
		{
			if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_CONTAINER_ID, uiIndex)))
			{
				goto Transmission_Error;
			}
		}
	}

	if( pKeyTree)
	{
		if (RC_BAD( rc = Wire.sendHTD( WIRE_VALUE_HTD, pKeyTree)))
		{
			goto Transmission_Error;
		}
	}

	if( uiRefDrn)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_DRN, uiRefDrn)))
		{
			goto Transmission_Error;
		}
	}

	if( uiFlag)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS,	uiFlag)))
		{
			goto Transmission_Error;
		}
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if( RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if( ppRecordRV)
	{
		if( (*ppRecordRV = Wire.getRecord()) != NULL)
		{
			(*ppRecordRV)->AddRef();
		}
	}

	if( puiDrnRV)
	{
		*puiDrnRV = Wire.getDrn();
	}

Exit:
	
	pCSContext->pool.poolReset( pvMark);
	return( rc);

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc: 	Go to the next key given a valid cursor.  Get & position to reference
			if you care about references
****************************************************************************/
FSTATIC RCODE flmNextKey(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK *		pStack,
	FLMUINT *	puiRefDrn)
{
	RCODE			rc;
	FLMBYTE *	pCurElm;

	/* The stack should be set up and is pointing to a valid block */
	
	pStack->uiFlags = NO_STACK;
	pStack->uiKeyBufSize = MAX_KEY_SIZ;

	pCurElm = CURRENT_ELM( pStack);

	/* Scan over the current record till 'does continue' flag NOT set */

	while( BBE_NOT_LAST( pCurElm))
	{
		/* First go to the next element - rc may return FERR_BT_END_OF_DATA */
		
		if( RC_BAD(rc = FSBtNextElm( pDb, pLFile, pStack)))
		{
			if( rc == FERR_BT_END_OF_DATA)		/* b-tree corrupt if FERR_BT_END_OF_DATA */
				rc = RC_SET( FERR_BTREE_ERROR);
			goto Exit;
		}
		pCurElm = CURRENT_ELM( pStack);
	}

	/* Now go to the next element */

	if( RC_BAD(rc = FSBtNextElm( pDb, pLFile, pStack)))
	{
		if( rc == FERR_BT_END_OF_DATA)
			rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	if (puiRefDrn)
	{
		pCurElm = CURRENT_ELM( pStack);
		(void) FSGetDomain( &pCurElm, BBE_KEY);

		if( puiRefDrn)
		{
			*puiRefDrn = SENNextVal( &pCurElm);
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Given an input key tree a FLAIM collated key will be built and returned
		to the user.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmKeyBuild(
	HFDB			hDb,
	FLMUINT		uiIxNum,
	FLMUINT		uiContainer,
	FlmRecord *	pRecord,
	FLMUINT		uiFlag,
	FLMBYTE * 	pKeyBuf,
	FLMUINT *	puiKeyLenRV)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	IXD *			pIxd;
	FLMBOOL		bImplicitTrans = FALSE;

	if( RC_OK( rc = fdbInit( pDb, FLM_READ_TRANS,
										TRUE, 0, &bImplicitTrans)))
	{
		if( RC_OK( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIxNum, NULL, &pIxd)))
		{
	
			/* Build the collated key */

			rc = KYTreeToKey( pDb, pIxd, pRecord, uiContainer,
					pKeyBuf, puiKeyLenRV, uiFlag );
		}
	}

	if( bImplicitTrans)
	{
		(void)flmAbortDbTrans( pDb);
	}
	
	(void)fdbExit( pDb);
	return( rc);
}
