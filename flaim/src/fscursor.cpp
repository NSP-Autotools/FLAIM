//-------------------------------------------------------------------------
// Desc:	Routines used during query to traverse through index b-trees.
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC FLMINT FSCompareKeys(
	FLMBOOL			bKey1IsUntilKey,
	FLMBYTE *		pKey1,
	FLMUINT			uiKeyLen1,
	FLMBOOL			bExclusiveKey1,
	FLMBOOL			bKey2IsUntilKey,
	FLMBYTE *		pKey2,
	FLMUINT			uiKeyLen2,
	FLMBOOL			bExclusiveKey2);

#define FS_COMPARE_KEYS( bKey1IsUntilKey,pKeyPos1,bKey2IsUntilKey,pKeyPos2) \
	FSCompareKeys( (bKey1IsUntilKey), \
			(pKeyPos1)->pKey, (pKeyPos1)->uiKeyLen, (pKeyPos1)->bExclusiveKey, \
			(bKey2IsUntilKey), \
			(pKeyPos2)->pKey, (pKeyPos2)->uiKeyLen, (pKeyPos2)->bExclusiveKey)

/****************************************************************************
Desc:
****************************************************************************/
FSIndexCursor::FSIndexCursor() 
{
	m_pFirstSet = m_pCurSet = NULL;
	m_pSavedPos = NULL;
	m_curKeyPos.bStackInUse = FALSE;
	reset();
}

/****************************************************************************
Desc:
****************************************************************************/
FSIndexCursor::~FSIndexCursor() 
{
	releaseBlocks();
	freeSets();
}

/****************************************************************************
Desc:	Resets any allocations, keys, state, etc.
****************************************************************************/
void FSIndexCursor::reset( void)
{
	releaseBlocks();
	freeSets();

	m_uiIndexNum = m_uiCurrTransId = m_uiBlkChangeCnt = 0;
	m_pFirstSet = m_pCurSet = NULL;
	m_pSavedPos = NULL;
	m_curKeyPos.bExclusiveKey = FALSE;

	m_DefaultSet.pNext = m_DefaultSet.pPrev = NULL;
	m_DefaultSet.fromKey.bStackInUse = 
		m_DefaultSet.untilKey.bStackInUse = FALSE;
	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
}


/****************************************************************************
Desc:	Resets to a new transaction that may change the read consistency of
		the query.  This is usually an old view error internal or external of
		this class.
****************************************************************************/
RCODE FSIndexCursor::resetTransaction( 
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	KEYSET *		pTmpSet;

	if( RC_BAD( rc = fdictGetIndex(
			pDb->pDict, pDb->pFile->bInLimitedMode,
			m_uiIndexNum, &m_pLFile, &m_pIxd)))
	{	
		goto Exit;
	}
	
	m_uiCurrTransId = pDb->LogHdr.uiCurrTransID;
	m_uiBlkChangeCnt = pDb->uiBlkChangeCnt;
	m_bIsUpdateTrans = (pDb->uiTransType == FLM_UPDATE_TRANS) ? TRUE : FALSE;

	// Need to release all stacks that are currently in use.

	for( pTmpSet = m_pFirstSet; pTmpSet; pTmpSet = pTmpSet->pNext)
	{
		releaseKeyBlocks( &pTmpSet->fromKey);
		releaseKeyBlocks( &pTmpSet->untilKey);
	}

	releaseKeyBlocks( &m_DefaultSet.fromKey);
	releaseKeyBlocks( &m_DefaultSet.untilKey);

	if( m_pSavedPos)
	{
		releaseKeyBlocks( m_pSavedPos);
	}

	releaseKeyBlocks( &m_curKeyPos);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Free all of the allocated sets.
****************************************************************************/
void FSIndexCursor::freeSets( void)
{
	KEYSET *		pCurSet;			// Current set.
	KEYSET *		pNextSet;		// Current set.

	for( pCurSet = m_pFirstSet; pCurSet; pCurSet = pNextSet)
	{
		pNextSet = pCurSet->pNext;
		
		if( pCurSet != &m_DefaultSet)
		{
			f_free( &pCurSet);
		}
	}
	
	m_pFirstSet = m_pCurSet = NULL;

	if( m_pSavedPos)
	{
		releaseKeyBlocks( m_pSavedPos);
		f_free( &m_pSavedPos);
		m_pSavedPos = NULL;
	}
}

/****************************************************************************
Desc:	Releases the cache blocks back to cache.
****************************************************************************/
void FSIndexCursor::releaseBlocks( void)
{
	KEYSET *		pCurSet;

	// Unuse all of the cache blocks in the from and until keys.

	for( pCurSet = m_pFirstSet; pCurSet; pCurSet = pCurSet->pNext)
	{
		releaseKeyBlocks( &pCurSet->fromKey);
		releaseKeyBlocks( &pCurSet->untilKey);
	}
	
	releaseKeyBlocks( &m_curKeyPos);
}

/****************************************************************************
Desc: Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the index.
		This code does not work with multiple key sets of FROM/UNTIL keys.
****************************************************************************/
RCODE FSIndexCursor::setupKeys(
	FDB *				pDb,
	IXD *				pIxd,
	QPREDICATE **	ppQPredicateList,
	FLMBOOL *		pbDoRecMatch,			// [out] Leave alone or set to TRUE.
	FLMBOOL *		pbDoKeyMatch,			// [out] Set to TRUE or FALSE
	FLMUINT *		puiLeafBlocksBetween,// [out] blocks between the stacks
	FLMUINT *		puiTotalKeys,			// [out] total number of keys
	FLMUINT *		puiTotalRefs,			// [out] total references
	FLMBOOL *		pbTotalsEstimated)	// [out] set to TRUE when estimating.
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiUntilKeyLen;
	FLMBYTE		pUntilKey [MAX_KEY_SIZ + 4];
	DIN_STATE	dinState;

	RESET_DINSTATE( dinState);
	m_uiIndexNum = pIxd->uiIndexNum;
	
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	m_DefaultSet.fromKey.uiRefPosition = 0;
	m_DefaultSet.untilKey.uiRefPosition = 0;
	m_DefaultSet.fromKey.bExclusiveKey = FALSE;
	m_DefaultSet.untilKey.bExclusiveKey = TRUE;

	if( RC_BAD( rc = flmBuildFromAndUntilKeys( pIxd, ppQPredicateList,
			m_DefaultSet.fromKey.pKey, &m_DefaultSet.fromKey.uiKeyLen, 
			m_DefaultSet.untilKey.pKey, &m_DefaultSet.untilKey.uiKeyLen,
			pbDoRecMatch, pbDoKeyMatch, &m_DefaultSet.untilKey.bExclusiveKey)))
	{
		goto Exit;
	}
	
	// Here is a rundown of how the data is setup after this call.
	// Default.FROM key - block address and current element offset and 
	//							 generated FROM key.
	// Default.UNTIL key - full stack setup and the generated UNTIL key
	// curKeyPos - pKey is the first key positioned the full stack is setup.

	f_memcpy( m_curKeyPos.pKey, m_DefaultSet.fromKey.pKey, 
		m_curKeyPos.uiKeyLen = m_DefaultSet.fromKey.uiKeyLen);
	f_memcpy( pUntilKey, m_DefaultSet.untilKey.pKey, 
		uiUntilKeyLen = m_DefaultSet.untilKey.uiKeyLen);

	m_pCurSet = m_pFirstSet = &m_DefaultSet;
	m_DefaultSet.fromKey.uiRecordId = m_curKeyPos.uiRecordId = 0;
	m_DefaultSet.fromKey.uiDomain = m_curKeyPos.uiDomain = MAX_DOMAIN;
	m_DefaultSet.untilKey.uiRecordId = 0;
	m_DefaultSet.untilKey.uiDomain = ZERO_DOMAIN;

	// Want any of the counts back?
	
	if( puiLeafBlocksBetween || puiTotalKeys || puiTotalRefs)
	{
		if( RC_OK( rc = setKeyPosition( pDb, TRUE, 
					&m_DefaultSet.fromKey, &m_curKeyPos)))
		{
			// Copy the b-tree information to the from key.
			
			m_DefaultSet.fromKey.uiBlockAddr = m_curKeyPos.uiBlockAddr;
			m_DefaultSet.fromKey.uiDomain = m_curKeyPos.uiDomain;
			m_DefaultSet.fromKey.uiBlockTransId = m_curKeyPos.uiBlockTransId;
			m_DefaultSet.fromKey.uiCurElm = m_curKeyPos.uiCurElm;
	
			// All keys bewteen FROM and UNTIL may be gone.
			
			if( FS_COMPARE_KEYS( FALSE, &m_curKeyPos, 
										TRUE, &m_DefaultSet.untilKey) <= 0)
			{
				rc = setKeyPosition( pDb, TRUE, 
						&m_DefaultSet.untilKey, &m_DefaultSet.untilKey);
				rc = (rc == FERR_EOF_HIT) ? FERR_OK: rc;

				// Restore the original UNTIL key - throws away what the last key is.
				
				f_memcpy( m_DefaultSet.untilKey.pKey, pUntilKey,
					m_DefaultSet.untilKey.uiKeyLen = uiUntilKeyLen);
			}
			else
			{
				rc = RC_SET( FERR_BOF_HIT);
			}
		}
		else
		{
			if( rc == FERR_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			m_bAtBOF = FALSE;
		}

		if( RC_BAD( rc))
		{
			// Empty tree or empty set case.
			
			if( rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
			{
				if( puiLeafBlocksBetween)
					*puiLeafBlocksBetween = 0;
				if( puiTotalKeys)
					*puiTotalKeys = 0;
				if( puiTotalRefs)
					*puiTotalRefs = 0;
				if( pbTotalsEstimated)
					*pbTotalsEstimated = FALSE;
				rc = FERR_OK;
			}
			goto Exit;
		}
		else
		{
			// If this is a positioning index, set the ref positions.

			if( pIxd->uiFlags & IXD_POSITIONING)
			{
				if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, 
							m_curKeyPos.pStack, &dinState, 
							&m_DefaultSet.fromKey.uiRefPosition)))
				{
					goto Exit;
				}
				if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, 
							m_DefaultSet.untilKey.pStack, &dinState, 
							&m_DefaultSet.untilKey.uiRefPosition)))
				{
					goto Exit;
				}
			}
			if( RC_BAD( rc = FSComputeIndexCounts( m_curKeyPos.pStack,
				m_DefaultSet.untilKey.pStack, puiLeafBlocksBetween, puiTotalKeys,
				puiTotalRefs, pbTotalsEstimated)))
			{
				goto Exit;
			}
		}
	}
	m_bAtBOF = TRUE;

Exit:
	return( rc);
}

/****************************************************************************
Public: 	setupKeys
Desc: 	Setup a cursor with just a single FROM/UNTIL key.
****************************************************************************/

RCODE	FSIndexCursor::setupKeys(
	FDB *			pDb,
	IXD *       pIxd,
	FLMBYTE *	pFromKey,
	FLMUINT		uiFromKeyLen,
	FLMUINT		uiFromRecordId,
	FLMBYTE *	pUntilKey,
	FLMUINT		uiUntilKeyLen,
	FLMUINT		uiUntilRecordId,
	FLMBOOL		bExclusiveUntil)
{
	RCODE			rc;

	m_uiIndexNum = pIxd->uiIndexNum;
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	m_DefaultSet.pNext = m_DefaultSet.pPrev = NULL;

	// FROM key
	
	m_DefaultSet.fromKey.uiRecordId = uiFromRecordId;
	m_DefaultSet.fromKey.uiDomain = 
		uiFromRecordId ? DRN_DOMAIN( uiFromRecordId) + 1: MAX_DOMAIN;
	
	m_DefaultSet.fromKey.uiKeyLen = uiFromKeyLen;
	f_memcpy( m_DefaultSet.fromKey.pKey, pFromKey, uiFromKeyLen);
	m_DefaultSet.fromKey.bExclusiveKey = FALSE;

	// UNTIL key
	
	m_DefaultSet.untilKey.uiRecordId = uiUntilRecordId;
	m_DefaultSet.untilKey.uiDomain = 
		uiUntilRecordId ? DRN_DOMAIN( uiUntilRecordId) + 1: ZERO_DOMAIN;
	
	m_DefaultSet.untilKey.uiKeyLen = uiUntilKeyLen;
	f_memcpy( m_DefaultSet.untilKey.pKey, pUntilKey, uiUntilKeyLen);
	m_DefaultSet.untilKey.bExclusiveKey = bExclusiveUntil;

	m_pFirstSet = &m_DefaultSet;
	m_bAtBOF = TRUE;
	m_pCurSet = NULL;

	// If this is a positioning index we need to setup the FROM/UNTIL
	// btrees to get the FROM/UNTIL reference positions.
	
	if( pIxd->uiFlags & IXD_POSITIONING)
	{
		if( RC_BAD( rc = setupForPositioning( pDb)))
		{
			goto Exit;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc: 	Read down all of the b-tree sets (FROM/UNTIL) so the all absolute
			positioning values can be set.
****************************************************************************/
RCODE	FSIndexCursor::setupForPositioning(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	DIN_STATE	dinState;
	FLMUINT		uiTempKeyLen;
	FLMBYTE		pTempKey [MAX_KEY_SIZ + 4];
	KEYSET *		pSrcSet;			// Current source set

	// Read all of the FROM/UNTIL keys going from the last set to the first.
	// This way m_curKeyPos will be positioned to the first key.

	for( pSrcSet = m_pFirstSet; pSrcSet->pNext; pSrcSet = pSrcSet->pNext)
	{
		;
	}

	for( ; pSrcSet; pSrcSet = pSrcSet->pPrev)
	{

		f_memcpy( pTempKey, pSrcSet->untilKey.pKey,
			uiTempKeyLen = pSrcSet->untilKey.uiKeyLen);

		if( RC_BAD( rc = setKeyPosition( pDb, TRUE, 
					&pSrcSet->untilKey, &pSrcSet->untilKey)))
		{
			goto Exit;
		}
		
		// Copy the key back.
		
		f_memcpy( pSrcSet->untilKey.pKey, pTempKey, 
			pSrcSet->untilKey.uiKeyLen = uiTempKeyLen);

		if( m_pIxd->uiFlags & IXD_POSITIONING)
		{
			if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, 
						pSrcSet->untilKey.pStack, &dinState,
						&pSrcSet->untilKey.uiRefPosition)))
			{
				goto Exit;
			}
		}

		if( pSrcSet == m_pFirstSet)
		{
			if( RC_BAD( rc = setKeyPosition( pDb, TRUE, 
						&pSrcSet->fromKey, &m_curKeyPos)))
			{
				goto Exit;
			}
			m_pCurSet = m_pFirstSet;
			
			// Copy the b-tree information to the from key.
			
			m_pCurSet->fromKey.uiBlockAddr = m_curKeyPos.uiBlockAddr;
			m_pCurSet->fromKey.uiDomain = m_curKeyPos.uiDomain;
			m_pCurSet->fromKey.uiBlockTransId = m_curKeyPos.uiBlockTransId;
			m_pCurSet->fromKey.uiCurElm = m_curKeyPos.uiCurElm;
			
			if( m_pIxd->uiFlags & IXD_POSITIONING)
			{
				if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, 
							m_curKeyPos.pStack, &dinState,
							&pSrcSet->fromKey.uiRefPosition)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			f_memcpy( pTempKey, pSrcSet->fromKey.pKey,
				uiTempKeyLen = pSrcSet->fromKey.uiKeyLen);

			if( RC_BAD( rc = setKeyPosition( pDb, TRUE, 
						&pSrcSet->fromKey, &pSrcSet->fromKey)))
			{
				goto Exit;
			}
			
			// Copy the key back.
			
			f_memcpy( pSrcSet->fromKey.pKey, pTempKey, 
				pSrcSet->fromKey.uiKeyLen = uiTempKeyLen);
				
			if( m_pIxd->uiFlags & IXD_POSITIONING)
			{
				if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, 
							pSrcSet->fromKey.pStack, &dinState,
							&pSrcSet->fromKey.uiRefPosition)))
				{
					goto Exit;
				}
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Merge the input cursors fromUntil sets as a result of a UNION.
****************************************************************************/
RCODE FSIndexCursor::unionKeys( 
	FSIndexCursor * pFSCursor)
{
	RCODE			rc = FERR_OK;
	KEYSET *		pInputSet;				// Sets input via pFSCursor.
	KEYSET *		pSrcSet;					// Current set
	KEYSET *		pDestSet = NULL;		// Newly allocated set
	KEYSET *		pCurDestSet = NULL;	// New current set
	KEYSET *		pPrevDestSet;

	// We need to release all of the blocks.  Too complex
	
	pFSCursor->releaseBlocks();
	releaseBlocks();
	pInputSet = pFSCursor->getFromUntilSets();
	pSrcSet = m_pFirstSet;

	while( pSrcSet || pInputSet)
	{
		FLMBOOL		bFromKeyLessThan;
		FLMBOOL		bUntilKeyGreaterThan;

		pPrevDestSet = pCurDestSet;
		if( RC_BAD( rc = f_calloc( sizeof( KEYSET), &pCurDestSet)))
		{
			goto Exit;
		}
		
		if( !pSrcSet)
		{
			f_memcpy( pCurDestSet, pInputSet, sizeof( KEYSET));
			pInputSet = pInputSet->pNext;

		}
		else if( !pInputSet)
		{
			f_memcpy( pCurDestSet, pSrcSet, sizeof( KEYSET));
			pSrcSet = pSrcSet->pNext;
		}
		else if( !FSCompareKeyPos( pInputSet, pSrcSet, 
					&bFromKeyLessThan, &bUntilKeyGreaterThan))
		{
			if( bFromKeyLessThan)
			{
				f_memcpy( pCurDestSet, pInputSet, sizeof( KEYSET));
				pInputSet = pInputSet->pNext;
			}
			else
			{
				f_memcpy( pCurDestSet, pSrcSet, sizeof( KEYSET));
				pSrcSet = pSrcSet->pNext;
			}
		}
		else
		{
			f_memcpy( &pCurDestSet->fromKey, 
				bFromKeyLessThan ? &pInputSet->fromKey : &pSrcSet->fromKey,
				sizeof( KEYPOS));
			
			for(;;)
			{
				// Keys overlap - take the lowest FROM key and highest UNTIL key
				// walking through the lower UNTIL key set while the keys overlap.

				if( bUntilKeyGreaterThan)
				{
					if( ((pSrcSet = pSrcSet->pNext) == NULL) || 
						!FSCompareKeyPos( pInputSet, pSrcSet, 
								&bFromKeyLessThan, &bUntilKeyGreaterThan))
					{
						f_memcpy( &pCurDestSet->untilKey, &pInputSet->untilKey, 
							sizeof( KEYPOS));
						pInputSet = pInputSet->pNext;
						break;
					}
				}
				else
				{
					if( ((pInputSet = pInputSet->pNext) == NULL) ||  
						!FSCompareKeyPos( pInputSet, pSrcSet, 
								&bFromKeyLessThan, &bUntilKeyGreaterThan))
					{
						f_memcpy( &pCurDestSet->untilKey, 
							&pSrcSet->untilKey, sizeof( KEYPOS));
						pSrcSet = pSrcSet->pNext;
						break;
					}
				}
			}
		}

		// Link in.
		
		pCurDestSet->pNext = NULL;
		if( !pDestSet)
		{
			pDestSet = pCurDestSet;
			pCurDestSet->pPrev = NULL;
		}
		else
		{
			pPrevDestSet->pNext = pCurDestSet;
			pCurDestSet->pPrev = pPrevDestSet;
		}
	}
	
	// We went to the trouble of having a default set allocated with this class.
	// Undo the last allocation.  if( pDestSet) then pCurDestSet can be used.

	freeSets();
	if( pDestSet)
	{
		f_memcpy( &m_DefaultSet, pCurDestSet, sizeof( KEYSET));
		if( pCurDestSet->pPrev)
		{
			pCurDestSet->pPrev->pNext = &m_DefaultSet;
			m_pFirstSet = pDestSet;
		}
		else
		{
			m_pFirstSet = &m_DefaultSet;
		}
		f_free( &pCurDestSet);
	}
	
	m_bAtBOF = TRUE;
	m_pCurSet = NULL;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Intersect the from/until key sets of pFSCursor into 'this'.
****************************************************************************/
RCODE FSIndexCursor::intersectKeys( 
	FDB *			pDb,
	FSIndexCursor * pFSCursor)
{
	RCODE			rc = FERR_OK;
	KEYSET *		pInputSet;				// Sets input via pFSCursor.
	KEYSET *		pSrcSet;					// Current set
	KEYSET *		pDestSet = NULL;		// Newly allocated set
	KEYSET *		pCurDestSet = NULL;	// New current set
	KEYSET *		pPrevDestSet;

	// Create a new destiniation set and throw away the current set.
	
	pFSCursor->releaseBlocks();
	releaseBlocks();
	pInputSet = pFSCursor->getFromUntilSets();
	pSrcSet = m_pFirstSet;

	while( pSrcSet && pInputSet)
	{
		FLMBOOL		bFromKeyLessThan;
		FLMBOOL		bUntilKeyGreaterThan;

		if( !FSCompareKeyPos( pInputSet, pSrcSet, 
					&bFromKeyLessThan, &bUntilKeyGreaterThan))
		{
			// Keys do NOT overlap - go to the next set section.
			
			if( bFromKeyLessThan)
			{
				pInputSet = pInputSet->pNext;
			}
			else
			{
				pSrcSet = pSrcSet->pNext;
			}
		}
		else
		{
			// Values overlap - see the two boolean values on how they overlap.

			pPrevDestSet = pCurDestSet;
			if( RC_BAD( rc = f_calloc( sizeof( KEYSET), &pCurDestSet)))
			{
				goto Exit;
			}
			
			pCurDestSet->pNext = NULL;
			
			if( !pDestSet)
			{
				pDestSet = pCurDestSet;
				pCurDestSet->pPrev = NULL;
			}
			else
			{
				pCurDestSet->pPrev = pPrevDestSet;
				pPrevDestSet->pNext = pCurDestSet;
			}

			// Take the highest FROM key.
			
			f_memcpy( &pCurDestSet->fromKey, 
				bFromKeyLessThan ? &pSrcSet->fromKey : &pInputSet->fromKey, 
				sizeof( KEYPOS));

			// Take the lowest until key and position to the next set.
			
			if( bUntilKeyGreaterThan)
			{
				f_memcpy( &pCurDestSet->untilKey, &pSrcSet->untilKey, sizeof( KEYPOS));
				pSrcSet = pSrcSet->pNext;
			}
			else
			{
				f_memcpy( &pCurDestSet->untilKey, &pInputSet->untilKey, sizeof( KEYPOS));
				pInputSet = pInputSet->pNext;
			}
		}
	}

	// We went to the trouble of having a default set allocated with this class.
	// Undo the last allocation.  if( pDestSet) then pCurDestSet can be used.
	
	freeSets();
	
	if( pDestSet)
	{
		f_memcpy( &m_DefaultSet, pCurDestSet, sizeof( KEYSET));
		if( pCurDestSet->pPrev)
		{
			pCurDestSet->pPrev->pNext = &m_DefaultSet;
			m_pFirstSet = pDestSet;
		}
		else
		{
			m_pFirstSet = &m_DefaultSet;
		}
		
		f_free( &pCurDestSet);
	}
	
	m_bAtBOF = TRUE;
	m_pCurSet = NULL;

	// If this is a positioning index we need to setup the FROM/UNTIL
	// btrees to get the FROM/UNTIL reference positions.
	
	if( m_pIxd->uiFlags & IXD_POSITIONING)
	{
		if( RC_BAD( rc = setupForPositioning( pDb)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Specific compare for a key range against the key ranges in this
		cursor.
****************************************************************************/

FLMBOOL FSIndexCursor::compareKeyRange(	// Returns TRUE if keys overlap.
	FLMBYTE *	pFromKey,
	FLMUINT		uiFromKeyLen,
	FLMBOOL		bExclusiveFrom,
	FLMBYTE *	pUntilKey,
	FLMUINT		uiUntilKeyLen,
	FLMBOOL		bExclusiveUntil,
	FLMBOOL *	pbUntilKeyInSet,			// Truth Table: T   F	F 
	FLMBOOL *	pbUntilGreaterThan)		//              F   T	F
{
	FLMBOOL		bKeysOverlap = FALSE;
	KEYSET *		pSrcSet;						// Current source set
	FLMINT		iFromCmp;
	FLMINT		iUntilCmp;
	FLMINT		iFromUntilCmp;
	FLMINT		iUntilFromCmp;

	for( pSrcSet = m_pFirstSet; pSrcSet; pSrcSet = pSrcSet->pNext)
	{
		iFromCmp = FSCompareKeys( FALSE, pFromKey,
											uiFromKeyLen,
											bExclusiveFrom, FALSE,
											pSrcSet->fromKey.pKey,
											pSrcSet->fromKey.uiKeyLen,
											pSrcSet->fromKey.bExclusiveKey);

		if( iFromCmp < 0)
		{
			// Move from left-most to right-most to see where the overlap is.
			
			iUntilFromCmp = FSCompareKeys( TRUE,
													pUntilKey, uiUntilKeyLen,
													bExclusiveUntil,
													FALSE,
													pSrcSet->fromKey.pKey,
													pSrcSet->fromKey.uiKeyLen,
													pSrcSet->fromKey.bExclusiveKey);
													
			if( iUntilFromCmp < 0)
			{
				*pbUntilKeyInSet = FALSE;
				*pbUntilGreaterThan = FALSE;
				goto Exit;
			}
			
			if( iUntilFromCmp == 0)
			{
				bKeysOverlap = TRUE;
				*pbUntilKeyInSet = TRUE;
				*pbUntilGreaterThan = FALSE;
				goto Exit;
			}
			
			// UNTIL > pSrcSet->fromKey
			
			iUntilCmp = FSCompareKeys( TRUE, pUntilKey, uiUntilKeyLen,
												bExclusiveUntil, TRUE,
												pSrcSet->untilKey.pKey,
												pSrcSet->untilKey.uiKeyLen,
												pSrcSet->untilKey.bExclusiveKey);
												
			if( iUntilCmp <= 0)
			{
				bKeysOverlap = TRUE;
				*pbUntilKeyInSet = TRUE;
				*pbUntilGreaterThan = FALSE;
				goto Exit;
			}
			
			bKeysOverlap = TRUE;
			
			// Try the next source set to see if the UNTIL key is in a set.
		}
		else
		{
			// Move from left-most to right-most to see where the overlap is.
			
			if( iFromCmp == 0)
			{
				bKeysOverlap = TRUE;
			}
			else
			{
				iFromUntilCmp = FSCompareKeys( FALSE, pFromKey, uiFromKeyLen,
														 bExclusiveFrom,
														 TRUE, pSrcSet->untilKey.pKey,
														 pSrcSet->untilKey.uiKeyLen,
														 pSrcSet->untilKey.bExclusiveKey);
														 
				if( iFromUntilCmp <= 0)
				{
					bKeysOverlap = TRUE;
				}
				else
				{
					continue;
				}
			}
			
			iUntilCmp = FSCompareKeys( TRUE, pUntilKey, uiUntilKeyLen,
												bExclusiveFrom,
												TRUE, pSrcSet->untilKey.pKey,
												pSrcSet->untilKey.uiKeyLen,
												pSrcSet->untilKey.bExclusiveKey);
												
			if( iUntilCmp <= 0)
			{
				bKeysOverlap = TRUE;
				*pbUntilKeyInSet = TRUE;
				*pbUntilGreaterThan = FALSE;
				goto Exit;
			}
		}
	}
	
	*pbUntilKeyInSet = FALSE;
	*pbUntilGreaterThan = TRUE;
	
Exit:

	return bKeysOverlap;
}	

/****************************************************************************
Desc:	Compare two From/Until key positions.
****************************************************************************/
FLMBOOL FSIndexCursor::FSCompareKeyPos(		// TRUE if keys overlap
	KEYSET *			pSet1,
	KEYSET *			pSet2,
	FLMBOOL *		pbFromKeyLessThan,			// pSet1->from < pSet2->from
	FLMBOOL *		pbUntilKeyGreaterThan)		// pSet1->until > pSet2->until
{

	if( FS_COMPARE_KEYS( TRUE, &pSet1->untilKey, FALSE, &pSet2->fromKey) < 0)
	{
		*pbFromKeyLessThan = TRUE;
		pbUntilKeyGreaterThan = FALSE;
		return FALSE;
	}
	
	if( FS_COMPARE_KEYS( FALSE, &pSet1->fromKey, TRUE, &pSet2->untilKey) > 0)
	{
		*pbFromKeyLessThan = FALSE;
		*pbUntilKeyGreaterThan = TRUE;
		return FALSE;
	}

	// Keys overlap.  Two more compares needed

	*pbFromKeyLessThan = (FLMBOOL) ((FS_COMPARE_KEYS( FALSE, &pSet1->fromKey, 
				FALSE, &pSet2->fromKey) < 0) ? TRUE : FALSE);

	*pbUntilKeyGreaterThan = (FLMBOOL) ((FS_COMPARE_KEYS( TRUE, &pSet1->untilKey, 
				TRUE, &pSet2->untilKey) > 0) ? TRUE : FALSE);

	return (TRUE);
}

/****************************************************************************
Desc:	Set information from the block into the KEYPOS structure.
****************************************************************************/
FINLINE void setKeyItemsFromBlock(
	KEYPOS *			pKeyPos)
{
	pKeyPos->uiBlockAddr = pKeyPos->pStack->uiBlkAddr;
	pKeyPos->uiCurElm = pKeyPos->pStack->uiCurElm;
	pKeyPos->uiKeyLen = pKeyPos->pStack->uiKeyLen;
	pKeyPos->uiBlockTransId = (pKeyPos->uiBlockAddr != BT_END) 
							  ? FB2UD( &pKeyPos->pStack->pBlk[ BH_TRANS_ID])
							  : 0;
}

/****************************************************************************
Desc:	Set the key position given some KEYPOS structure.
		Please note that the blocks in the stack may or may not be used.
****************************************************************************/
RCODE FSIndexCursor::setKeyPosition(
	FDB *				pDb,
	FLMBOOL			bGoingForward,
	KEYPOS *			pInKeyPos,		// Input key position
	KEYPOS *			pOutKeyPos)		// Output to setup the stack/key buffer.
											// It is ok for Input==Output
{
	RCODE				rc;
	FLMBYTE *		pSearchKey;
	FLMBYTE			pTempKey[ MAX_KEY_SIZ + 4];
	FLMUINT			uiTargetRecordId = pInKeyPos->uiRecordId;

	// May have to unuse the b-tree blocks.  Then setup the stack.
	
	if( !pOutKeyPos->bStackInUse)
	{
		FSInitStackCache( pOutKeyPos->Stack, BH_MAX_LEVELS);
		pOutKeyPos->bStackInUse = TRUE;
	}

	if( pInKeyPos == pOutKeyPos)
	{
		pSearchKey = pTempKey;
		f_memcpy( pTempKey, pInKeyPos->pKey, pInKeyPos->uiKeyLen);
	}
	else
	{
		pSearchKey = pInKeyPos->pKey;
	}
	
	// Setup the stack.
	
	pOutKeyPos->pStack = pOutKeyPos->Stack;
	pOutKeyPos->Stack[0].pKeyBuf = pOutKeyPos->pKey;

	// All of the variables should be setup for the search.
	
	if( RC_BAD( rc = FSBtSearch( pDb, m_pLFile, &pOutKeyPos->pStack,
				pSearchKey, pInKeyPos->uiKeyLen, 
				uiTargetRecordId
					? DRN_DOMAIN( uiTargetRecordId) + 1
					: pInKeyPos->uiDomain)))
	{
		goto Exit;
	}
	
	pOutKeyPos->uiBlockAddr = pOutKeyPos->pStack->uiBlkAddr;
	pOutKeyPos->uiCurElm = pOutKeyPos->pStack->uiCurElm;

	if( pOutKeyPos->pStack->uiBlkAddr == BT_END)
	{
		pOutKeyPos->bStackInUse = FALSE;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	
	pOutKeyPos->uiKeyLen = pOutKeyPos->pStack->uiKeyLen;

	if( bGoingForward)
	{
		if( pOutKeyPos->pStack->uiCmpStatus == BT_END_OF_DATA)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}
	else
	{
		BTSK *			pStack;

		// Going backwards or to the last.  May have positioned too far.
		
		if( pOutKeyPos->pStack->uiCmpStatus == BT_END_OF_DATA || 
			 FS_COMPARE_KEYS( TRUE, pOutKeyPos, TRUE, pInKeyPos) > 0)
		{
			uiTargetRecordId = 0;
			pStack = pOutKeyPos->pStack;

			// The stack should be set up and is pointing to a valid block.

			if( pStack->uiCmpStatus != BT_END_OF_DATA)
			{
				while( BBE_NOT_FIRST( CURRENT_ELM( pStack)))
				{
					if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
					{
						if( rc == FERR_BT_END_OF_DATA)	
						{
							rc = RC_SET( FERR_BTREE_ERROR);
						}
						
						goto Exit;
					}
				}
			}
			
			// Position to the last continuation element of the previous element.
			
			if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
			{
				if( rc == FERR_BT_END_OF_DATA)
				{
					rc = RC_SET( FERR_BOF_HIT);
				}
				
				goto Exit;
			}
		}
	}
	
	// When using the positioning index, we must always have a complete stack.
	
	if( !(m_pIxd->uiFlags & IXD_POSITIONING))
	{
		pOutKeyPos->pStack->uiFlags = NO_STACK;
	}

	// Get the record ID at the leaf level of the b-tree.
	
	if( uiTargetRecordId)
	{
		pOutKeyPos->uiRecordId = pInKeyPos->uiRecordId;
		rc = FSRefSearch( pOutKeyPos->pStack,
						&pOutKeyPos->DinState, &pOutKeyPos->uiRecordId);
						
		if( rc == FERR_FAILURE)
		{
			rc = FERR_OK;
		}
	}
	else
	{
		if( bGoingForward)
		{
			pOutKeyPos->uiRecordId = FSRefFirst( pOutKeyPos->pStack, 
						&pOutKeyPos->DinState, &pOutKeyPos->uiDomain);
			pOutKeyPos->uiDomain++;
		}
		else
		{
			pOutKeyPos->uiRecordId = FSRefLast( pOutKeyPos->pStack, 
						&pOutKeyPos->DinState, &pOutKeyPos->uiDomain);
		}
	}

Exit:

	// Save state only on a good return value.
	
	if( RC_OK( rc) || 
		((rc == FERR_EOF_HIT || rc == FERR_BOF_HIT) && pOutKeyPos->bStackInUse))
	{
		setKeyItemsFromBlock( pOutKeyPos);
	}
	else
	{
		releaseKeyBlocks( pOutKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Return the current record and record id.
VISIT:	We may want to return BOF/EOF when positioned on an endpoint.
****************************************************************************/
RCODE FSIndexCursor::currentKey(
	FDB *				pDb,
	FlmRecord **	ppRecordKey,		// Will replace what is there
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( m_bAtBOF)
	{
		rc = RC_SET( FERR_BOF_HIT);
		goto Exit;
	}
	
	if( m_bAtEOF)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	if( !m_curKeyPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone, 
													 FALSE, FALSE, &bRefGone)))
		{
			// The current key is gone.  Returns FERR_NOT_FOUND.
			goto Exit;
		}
	}

	// If this assert happens we have to code for it.
	
	flmAssert( m_curKeyPos.uiRecordId != 0);

	if( ppRecordKey)
	{
		if( RC_BAD( rc = flmIxKeyOutput( m_pIxd, m_curKeyPos.pKey, 
			m_curKeyPos.uiKeyLen, ppRecordKey, TRUE)))
		{
			goto Exit;
		}
		
		(*ppRecordKey)->setID( m_curKeyPos.uiRecordId);
	}
	
	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Return the current record and record id.
VISIT:	We may want to return BOF/EOF when positioned on an endpoint.
****************************************************************************/
RCODE FSIndexCursor::currentKeyBuf(
	FDB *				pDb,
	F_Pool *			pPool,
	FLMBYTE **		ppKeyBuf,
	FLMUINT *		puiKeyLen,
	FLMUINT *		puiRecordId,
	FLMUINT *		puiContainerId)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( m_bAtBOF)
	{
		rc = RC_SET( FERR_BOF_HIT);
		goto Exit;
	}
	
	if( m_bAtEOF)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	if( !m_curKeyPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone, 
													 FALSE, FALSE, &bRefGone)))
		{
			// The current key is gone.  Returns FERR_NOT_FOUND.
			goto Exit;
		}
	}

	// If this assert happens we have to code for it.
	
	flmAssert( m_curKeyPos.uiRecordId != 0);

	if( ppKeyBuf)
	{
		// If they passed in a non-null key buffer pointer, they also
		// need to pass in a non-null return length.

		flmAssert( puiKeyLen != NULL);

		if( (*puiKeyLen = m_curKeyPos.uiKeyLen) != 0)
		{
			if( RC_BAD( rc = pPool->poolAlloc( m_curKeyPos.uiKeyLen,
				(void **)ppKeyBuf)))
			{
				goto Exit;
			}
			
			f_memcpy( *ppKeyBuf, m_curKeyPos.pKey, m_curKeyPos.uiKeyLen);
		}
		else
		{
			*ppKeyBuf = NULL;
		}
	}
	
	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}
	
	if (puiContainerId)
	{
		if ((*puiContainerId = m_pIxd->uiContainerNum) == 0)
		{
			flmAssert( m_curKeyPos.uiKeyLen > getIxContainerPartLen( m_pIxd));
			*puiContainerId = getContainerFromKey( m_curKeyPos.pKey,
											m_curKeyPos.uiKeyLen);
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Position to and return the first record.
			This is hard because positioning using the first key may actually
			position past or into another FROM/UNTIL set in the list.
****************************************************************************/
RCODE FSIndexCursor::firstKey(
	FDB *				pDb,
	FlmRecord **	ppRecordKey,		// Will replace what is there
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( !m_pFirstSet)
	{
		m_bAtBOF = FALSE;
		m_bAtEOF = TRUE;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	// If at BOF and stack is in use, the key is the first key!
	// This should only happen when firstKey is called the first time.
	
	if( m_bAtBOF && m_curKeyPos.bStackInUse && m_pCurSet)
	{
		m_bAtBOF = FALSE;
	}
	else
	{
		m_pCurSet = m_pFirstSet;
		m_bAtBOF = m_bAtEOF = FALSE;
		
		if( RC_BAD( rc = setKeyPosition( pDb, TRUE, &m_pCurSet->fromKey, 
				&m_curKeyPos)))
		{
			if( rc == FERR_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			
			goto Exit;
		}
	}

	// Check to see if the key is within one of the FROM/UNTIL sets.
	
	for(;;)
	{
		// Check to see if the current key <= UNTIL key.

		if( FS_COMPARE_KEYS( FALSE, &m_curKeyPos, TRUE, 
			&m_pCurSet->untilKey) <= 0)
		{
			break;
		}
		
		// Nope - UntilKey < current key.
		
		if( !m_pCurSet->pNext)
		{
			rc = RC_SET( FERR_EOF_HIT);
			m_bAtEOF = TRUE;
			goto Exit;
		}
		
		m_pCurSet = m_pCurSet->pNext;

		// If (current key < FROM key of the next set) we need to reposition.

		if( FS_COMPARE_KEYS( FALSE, &m_curKeyPos, FALSE, &m_pCurSet->fromKey) < 0)
		{
			if( RC_BAD( rc = setKeyPosition( pDb, TRUE, &m_pCurSet->fromKey, 
						&m_curKeyPos)))
			{
				if( rc == FERR_EOF_HIT)
				{
					m_bAtEOF = TRUE;
				}
				
				goto Exit;
			}
		}
	}
	
	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}
	
	if( ppRecordKey)
	{
		if( RC_BAD( rc = flmIxKeyOutput( m_pIxd, m_curKeyPos.pKey, 
			m_curKeyPos.uiKeyLen, ppRecordKey, TRUE)))
		{
			goto Exit;
		}
		
		(*ppRecordKey)->setID( m_curKeyPos.uiRecordId);
	}

Exit:

	if( m_bAtEOF)
	{
		// The saved state is not pointing anywhere specific in the B-tree.
		
		releaseKeyBlocks( &m_curKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Position to the next key and the first reference of that key.
****************************************************************************/
RCODE FSIndexCursor::nextKey(
	FDB *				pDb,
	FlmRecord **	ppRecordKey,
	FLMUINT *		puiRecordId)
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;
	BTSK *			pStack;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( m_bAtEOF)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	
	if( m_bAtBOF)
	{
		rc = firstKey( pDb, ppRecordKey, puiRecordId);
		goto Exit;
	}

	bKeyGone = bRefGone = FALSE;

	// Takes care of any re-read of a block if we changed transactions.

	if( !m_curKeyPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, TRUE, FALSE, &bKeyGone, 
											  TRUE, FALSE, &bRefGone)))
		{
			if( rc == FERR_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			
			// May return FERR_EOF_HIT if all remaining keys are deleted.
			
			goto Exit;
		}
	}

	pStack = m_curKeyPos.pStack;
	for(;;)
	{
		FLMINT			iCmp;
		if( !bKeyGone)
		{
			FLMBYTE *		pCurElm;

			pCurElm = CURRENT_ELM( pStack);
			while( BBE_NOT_LAST( pCurElm))
			{
				if( RC_BAD( rc = FSBtNextElm( pDb, m_pLFile, pStack)))
				{
					// b-tree corrupt if FERR_BT_END_OF_DATA
					
					if( rc == FERR_BT_END_OF_DATA)		
					{
						rc = RC_SET( FERR_BTREE_ERROR);
					}
					
					goto Exit;
				}
				pCurElm = CURRENT_ELM( pStack);
			}
			
			// Now go to the next element.
			
			if( RC_BAD( rc = FSBtNextElm( pDb, m_pLFile, pStack)))
			{
				if( rc == FERR_BT_END_OF_DATA)		
				{
					m_bAtEOF = TRUE;
					rc = RC_SET( FERR_EOF_HIT);
				}
				
				goto Exit;
			}
			
			bKeyGone = TRUE;
			m_curKeyPos.uiKeyLen = m_curKeyPos.pStack->uiKeyLen;
		}

		// Could have positioned after the current sets until key before
		// the FROM key of the next set.
		
		iCmp = FS_COMPARE_KEYS( FALSE, &m_curKeyPos, TRUE, &m_pCurSet->untilKey);
		if( iCmp <= 0)
		{
			setKeyItemsFromBlock( &m_curKeyPos);
			m_curKeyPos.uiRecordId = 
				FSRefFirst( pStack, &m_curKeyPos.DinState, &m_curKeyPos.uiDomain);
			break;
		}
		
		// Go to the next set if there is one.
		
		if( !m_pCurSet->pNext)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
		
		m_pCurSet = m_pCurSet->pNext;
		
		// The key may fit in the next set.
		
		iCmp = FS_COMPARE_KEYS( FALSE, &m_curKeyPos, FALSE, &m_pCurSet->fromKey);
		if( iCmp < 0)
		{
			// Reposition using the from key.
			if( RC_BAD( rc = setKeyPosition( pDb, TRUE, 
				&m_pCurSet->fromKey, &m_curKeyPos)))
			{
				if( rc == FERR_EOF_HIT)
				{
					m_bAtEOF = TRUE;
				}
				goto Exit;
			}
		}
	}

	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}
	
	if( ppRecordKey)
	{
		if( RC_BAD( rc = flmIxKeyOutput( m_pIxd, m_curKeyPos.pKey, 
			m_curKeyPos.uiKeyLen, ppRecordKey, TRUE)))
		{
			goto Exit;
		}
		
		(*ppRecordKey)->setID( m_curKeyPos.uiRecordId);
	}

Exit:

	if( m_bAtEOF)
	{
		// The saved state is not pointing anywhere specific in the B-tree.
		
		releaseKeyBlocks( &m_curKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 	Position to the next referece of the current key.
****************************************************************************/
RCODE FSIndexCursor::nextRef(			// FERR_OK, FERR_EOF_HIT or error
	FDB *				pDb,
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	flmAssert( m_pCurSet != NULL);
	bKeyGone = bRefGone = FALSE;

	if( !m_curKeyPos.bStackInUse)
	{
		// Take care of any re-read of a block if we changed transactions.
		
		if( RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone, 
											  TRUE, FALSE, &bRefGone)))
		{
			// May return FERR_EOF_HIT if all remaining references are deleted.
			goto Exit;
		}

		flmAssert( !bKeyGone);
	}

	if( !bRefGone)
	{
		if( RC_BAD( rc = FSRefNext( pDb, m_pLFile, m_curKeyPos.pStack,
				&m_curKeyPos.DinState, &m_curKeyPos.uiRecordId)))
		{
			if( rc == FERR_BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_EOF_HIT);
			}
			
			goto Exit;
		}
		else
		{
			setKeyItemsFromBlock( &m_curKeyPos);
		}
	}
	
	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Position to and return the last record.
		This is hard because positioning using the first key may actually
		position past or into another FROM/UNTIL set in the list.
****************************************************************************/
RCODE FSIndexCursor::lastKey(
	FDB *				pDb,
	FlmRecord **	ppRecordKey,		// Will replace what is there
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	m_pCurSet = m_pFirstSet;
	if( !m_pCurSet)
	{
		m_bAtBOF = TRUE;
		m_bAtEOF = FALSE;
		rc = RC_SET( FERR_BOF_HIT);
		goto Exit;
	}

	// Position to the last set.
	
	while( m_pCurSet->pNext)
	{
		m_pCurSet = m_pCurSet->pNext;
	}
	
	m_bAtBOF = m_bAtEOF = FALSE;

	for(;;)
	{
		if( RC_BAD( rc = setKeyPosition( pDb, FALSE, &m_pCurSet->untilKey, 
				&m_curKeyPos)))
		{
			// Returns an error or FERR_EOF_HIT.
			
			if( rc != FERR_EOF_HIT)
			{
				goto Exit;
			}
				
			// Position to the previous key if possible.
			
			if( m_curKeyPos.pStack->uiBlkAddr == BT_END)
			{
				m_bAtBOF = TRUE;
				rc = RC_SET( FERR_BOF_HIT);
				goto Exit;
			}
			
			// Went past the until key.  Go back one.
			
			if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, m_curKeyPos.pStack)))
			{
				if( rc == FERR_BT_END_OF_DATA)
				{
					rc = RC_SET( FERR_BOF_HIT);
				}
				
				goto Exit;
			}
		}

		// We may have positioned before the FROM key of this set.

		if( FS_COMPARE_KEYS( FALSE, &m_pCurSet->fromKey, 
			FALSE, &m_curKeyPos) <= 0)
		{
			break;
		}
		
		if( !m_pCurSet->pPrev)
		{
			rc = RC_SET( FERR_BOF_HIT);
			m_bAtBOF = TRUE;
			goto Exit;
		}
		
		m_pCurSet = m_pCurSet->pPrev;
	}

	// Now we are positioned somewhere.  Return the key and first record id.

	m_curKeyPos.uiRecordId = FSRefLast( m_curKeyPos.pStack,
			&m_curKeyPos.DinState, &m_curKeyPos.uiDomain);
	setKeyItemsFromBlock( &m_curKeyPos);

	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}
	
	if( ppRecordKey)
	{
		if( RC_BAD( rc = flmIxKeyOutput( m_pIxd, m_curKeyPos.pKey, 
			m_curKeyPos.uiKeyLen, ppRecordKey, TRUE)))
		{
			goto Exit;
		}
		
		(*ppRecordKey)->setID( m_curKeyPos.uiRecordId);
	}

Exit:

	if( rc == FERR_BOF_HIT)
	{
		// The saved state is not pointing anywhere specific in the B-tree.
		releaseKeyBlocks( &m_curKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc: Position to the PREVIOUS key and the LAST reference of that key.
****************************************************************************/
RCODE FSIndexCursor::prevKey(
	FDB *				pDb,
	FlmRecord **	ppRecordKey,
	FLMUINT *		puiRecordId)
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;
	BTSK *			pStack = m_curKeyPos.pStack;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( m_bAtBOF)
	{
		rc = RC_SET( FERR_BOF_HIT);
		goto Exit;
	}
	
	if( !m_pCurSet || m_bAtEOF)
	{
		rc = lastKey( pDb, ppRecordKey, puiRecordId);
		goto Exit;
	}

	bKeyGone = bRefGone = FALSE;

	// Takes care of any re-read of a block if we changed transactions.

	if( !m_curKeyPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, FALSE, TRUE, &bKeyGone, 
											  FALSE, FALSE, &bRefGone)))
		{
			// May return FERR_EOF_HIT if all remaining keys are deleted.
			
			if( rc != FERR_BOF_HIT && rc != FERR_EOF_HIT)
			{
				goto Exit;
			}
			
			m_bAtBOF = TRUE;
			rc = RC_SET( FERR_BOF_HIT);
		}
	}
	
	for(;;)
	{
		if( !bKeyGone)
		{
			FLMBYTE *		pCurElm;

			pCurElm = CURRENT_ELM( pStack);
			while( BBE_NOT_FIRST( pCurElm))
			{
				if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
				{
					// b-tree corrupt if FERR_BT_END_OF_DATA
					
					if( rc == FERR_BT_END_OF_DATA)		
					{
						rc = RC_SET( FERR_BTREE_ERROR);
					}
					
					goto Exit;
				}
				
				pCurElm = CURRENT_ELM( pStack);
			}

			// Now go to the previous element.

			if( RC_BAD(rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
			{
				// b-tree corrupt if FERR_BT_END_OF_DATA
				
				if( rc == FERR_BT_END_OF_DATA)		
				{
					m_bAtBOF = TRUE;
					rc = RC_SET( FERR_BOF_HIT);
				}
				
				goto Exit;
			}
			
			bKeyGone = TRUE;
			m_curKeyPos.uiKeyLen = m_curKeyPos.pStack->uiKeyLen;
		}
		
		// Could have positioned after the current sets until key before
		// the FROM key of the next set.
		
		if( FS_COMPARE_KEYS( FALSE, &m_curKeyPos, TRUE, &m_pCurSet->fromKey) >= 0)
		{
			setKeyItemsFromBlock( &m_curKeyPos);
			m_curKeyPos.uiRecordId = 
				FSRefLast( pStack, &m_curKeyPos.DinState, &m_curKeyPos.uiDomain);
			break;
		}
		
		// Go to the previous set if there is one.
		
		if( !m_pCurSet->pPrev)
		{
			m_bAtBOF = TRUE;
			rc = RC_SET( FERR_BOF_HIT);
			goto Exit;
		}
		
		m_pCurSet = m_pCurSet->pPrev;

		// The key may fit in the previous set.
		
		if( FS_COMPARE_KEYS( FALSE, &m_curKeyPos, FALSE, &m_pCurSet->fromKey) > 0)
		{
			// Reposition using the from key.
			
			if( RC_BAD( rc = setKeyPosition( pDb, FALSE, 
						&m_pCurSet->untilKey, &m_curKeyPos)))
			{
				if( rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
				{
					m_bAtBOF = TRUE;
					rc = RC_SET( FERR_BOF_HIT);
				}
				
				goto Exit;
			}
		}
	}

	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}
	
	if( ppRecordKey)
	{
		if( RC_BAD( rc = flmIxKeyOutput( m_pIxd, m_curKeyPos.pKey, 
			m_curKeyPos.uiKeyLen, ppRecordKey, TRUE)))
		{
			goto Exit;
		}
		
		(*ppRecordKey)->setID( m_curKeyPos.uiRecordId);
	}

Exit:

	if( rc == FERR_BOF_HIT)
	{
		// The saved state is not pointing anywhere specific in the B-tree.
		releaseKeyBlocks( &m_curKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Position to the previous referece of the current key.
****************************************************************************/
RCODE FSIndexCursor::prevRef(
	FDB *				pDb,
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;

	flmAssert( m_pCurSet != NULL);
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	bKeyGone = bRefGone = FALSE;

	if( !m_curKeyPos.bStackInUse)
	{
		// Take care of any re-read of a block if we changed transactions.
		
		if( RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone, 
											  FALSE, TRUE, &bRefGone)))
		{
			// May return FERR_EOF_HIT if all preceeding references are deleted.
			
			if( rc == FERR_EOF_HIT)
			{
				rc = RC_SET( FERR_BOF_HIT);
			}
			
			goto Exit;
		}

		flmAssert( !bKeyGone);
	}
	
	if( !bRefGone)
	{
		if( RC_BAD( rc = FSRefPrev( pDb, m_pLFile, m_curKeyPos.pStack,
					&m_curKeyPos.DinState, &m_curKeyPos.uiRecordId)))
		{
			if( rc == FERR_BT_END_OF_DATA)
			{
				rc = RC_SET( FERR_BOF_HIT);
			}
			
			goto Exit;
		}
		else
		{
			setKeyItemsFromBlock( &m_curKeyPos);
		}
	}
	
	if( puiRecordId)
	{
		*puiRecordId = m_curKeyPos.uiRecordId;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reposition to the current key + recordId.  If the current key
		is gone we may reposition past the UNTIL key and should check.
****************************************************************************/
RCODE FSIndexCursor::reposition(
	FDB *				pDb,
	FLMBOOL			bCanPosToNextKey,	// May be TRUE if bPosToPrevKey is FALSE
	FLMBOOL			bCanPosToPrevKey,	// May be TRUE if bPosToNextKey is FALSE
	FLMBOOL *		pbKeyGone,			// [out] cannot be NULL
	FLMBOOL			bCanPosToNextRef,	// May be TRUE if bPosToPrevRef is FALSE
	FLMBOOL			bCanPosToPrevRef,	// May be TRUE if bPosToNextRef is FALSE
	FLMBOOL *		pbRefGone)			// [out] cannot be NULL
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBlkTransId = 0;
	FLMBOOL			bReread = FALSE;
	FLMUINT			uiTargetRecordId;

	uiTargetRecordId = m_curKeyPos.uiRecordId;
	
	// May have to unuse the b-tree blocks.  Then setup the stack again.

	flmAssert( !m_curKeyPos.bStackInUse);
	*pbKeyGone = *pbRefGone = FALSE;

	// Re-read the block and see if it is the same block.
	
	if( m_curKeyPos.uiBlockAddr == BT_END)
	{
		bReread = TRUE;
	}
	else
	{
		if( RC_BAD( rc = FSGetBlock( pDb, m_pLFile, 
					m_curKeyPos.uiBlockAddr, m_curKeyPos.pStack)))
		{
			if( rc != FERR_DATA_ERROR)
			{
				goto Exit;
			}
			
			rc = FERR_OK;
			bReread = TRUE;
		}
		else
		{
			uiBlkTransId = FB2UD( &m_curKeyPos.pStack->pBlk[ BH_TRANS_ID]);
			m_curKeyPos.bStackInUse = TRUE;
		}
	}

	if( m_curKeyPos.uiBlockTransId != uiBlkTransId)
	{
		bReread = TRUE;
	}
	else if( pDb->uiTransType == FLM_UPDATE_TRANS) 
	{
		bReread = TRUE;
	}

	if( bReread) 
	{
		FLMBYTE		pKey [MAX_KEY_SIZ + 4];
		FLMUINT		uiKeyLen = m_curKeyPos.uiKeyLen;
		FLMUINT		uiSaveRecId = m_curKeyPos.uiRecordId;

		f_memcpy( pKey, m_curKeyPos.pKey, uiKeyLen);

		// This may be a new read transaction.  Call BTSearch.
		// The current reference or current key may go away on

		if( RC_BAD( rc = setKeyPosition( pDb, 
			bCanPosToPrevKey ? FALSE : TRUE, &m_curKeyPos, &m_curKeyPos)))
		{
			if( rc != FERR_EOF_HIT && rc != FERR_BOF_HIT)
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc) ||
			 uiKeyLen != m_curKeyPos.uiKeyLen ||
			 f_memcmp( pKey, m_curKeyPos.pKey, uiKeyLen))
		{
			// It is OK that we may not be positioned inside of the current set.

			*pbKeyGone = TRUE;
			*pbRefGone = TRUE;

			if( !bCanPosToNextKey && !bCanPosToPrevKey)
			{
				if( uiKeyLen)
				{
					f_memcpy( m_curKeyPos.pKey, pKey, uiKeyLen);
				}

				m_curKeyPos.uiKeyLen = uiKeyLen;
				m_curKeyPos.uiRecordId = uiSaveRecId;

				releaseKeyBlocks( &m_curKeyPos);
				m_curKeyPos.uiBlockAddr = BT_END;

				if( bCanPosToNextKey || bCanPosToNextRef)
				{
					rc = RC_SET( FERR_EOF_HIT);
				}
				else if( bCanPosToPrevKey || bCanPosToPrevRef)
				{
					rc = RC_SET( FERR_BOF_HIT);
				}
				else
				{
					rc = RC_SET( FERR_NOT_FOUND);
				}
			}

			goto Exit;
		}
	}
	else
	{
		m_curKeyPos.bStackInUse = TRUE;
	}

	if (uiTargetRecordId &&
		 uiTargetRecordId != m_curKeyPos.uiRecordId)
	{
		*pbRefGone = TRUE;

		if( bCanPosToPrevRef && m_curKeyPos.uiRecordId < uiTargetRecordId)
		{
			if (RC_OK( rc = FSRefPrev( pDb, m_pLFile, m_curKeyPos.pStack,
				&m_curKeyPos.DinState, &m_curKeyPos.uiRecordId)))
			{
				setKeyItemsFromBlock( &m_curKeyPos);
			}
		}
		else if( !bCanPosToNextRef)
		{
			// We have an error positioning.  Return NOT_FOUND
			
			rc = RC_SET( FERR_NOT_FOUND);
		}
	}		

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Save the current key position.
****************************************************************************/
void FSIndexCursor::saveCurrKeyPos(
	KEYPOS *		pSaveKeyPos)
{
	f_memcpy( pSaveKeyPos->pKey, m_curKeyPos.pKey, m_curKeyPos.uiKeyLen);
	pSaveKeyPos->uiKeyLen = m_curKeyPos.uiKeyLen;
	pSaveKeyPos->uiRecordId = m_curKeyPos.uiRecordId;
	pSaveKeyPos->uiDomain = m_curKeyPos.uiDomain;
}

/****************************************************************************
Desc:	Restore the current key position.
****************************************************************************/
void FSIndexCursor::restoreCurrKeyPos(
	KEYPOS *		pSaveKeyPos)
{
	f_memcpy( m_curKeyPos.pKey, pSaveKeyPos->pKey, pSaveKeyPos->uiKeyLen);
	m_curKeyPos.uiKeyLen = pSaveKeyPos->uiKeyLen;
	m_curKeyPos.uiRecordId = pSaveKeyPos->uiRecordId;
	m_curKeyPos.uiDomain = pSaveKeyPos->uiDomain;
}

/****************************************************************************
Desc: 	Find the key set for the passed in key.
****************************************************************************/
RCODE FSIndexCursor::getKeySet(
	FLMBYTE *	pKey,
	FLMUINT		uiKeyLen,
	KEYSET **	ppKeySet)
{
	RCODE			rc = FERR_OK;
	KEYSET *		pKeySet;

	pKeySet = m_pFirstSet;
	while (pKeySet)
	{

		// Compare this key against the from key.  If it is less than the
		// from key, we are not inside one of the key ranges.

		if( FSCompareKeys( FALSE, pKey, uiKeyLen, FALSE,
									 FALSE, pKeySet->fromKey.pKey,
									 pKeySet->fromKey.uiKeyLen,
									 pKeySet->fromKey.bExclusiveKey) < 0)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}

		// Key is >= from key, see how it compares to until key.

		if( FSCompareKeys( FALSE, pKey, uiKeyLen, FALSE,
									TRUE, pKeySet->untilKey.pKey,
									pKeySet->untilKey.uiKeyLen,
									pKeySet->untilKey.bExclusiveKey) <= 0)
		{
			break;
		}

		// Go to the next key range.

		pKeySet = pKeySet->pNext;
	}
	
	if( !pKeySet)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	*ppKeySet = pKeySet;
	return( rc);
}

/****************************************************************************
Desc:	Position to the input key + recordId.
****************************************************************************/
RCODE FSIndexCursor::positionTo(
	FDB *				pDb,
	FLMBYTE *		pKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRecordId)
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;
	KEYSET *			pKeySet;
	KEYPOS *			pSaveKeyPos = NULL;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	// A key with a non-zero key length must be passed in.
	
	flmAssert( uiKeyLen != 0);

	// Make sure we fall inside one of the key ranges and
	// save the current key position so we can restore it
	// if the reposition call fails.

	if( RC_BAD( rc = f_alloc( sizeof( KEYPOS),
								&pSaveKeyPos)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = getKeySet( pKey, uiKeyLen, &pKeySet)))
	{
		goto Exit;
	}
	
	saveCurrKeyPos( pSaveKeyPos);
	releaseKeyBlocks( &m_curKeyPos);
	m_curKeyPos.uiKeyLen = uiKeyLen;
	f_memcpy( m_curKeyPos.pKey, pKey, uiKeyLen);
	m_curKeyPos.uiRecordId = uiRecordId;
	m_curKeyPos.uiDomain = DRN_DOMAIN( uiRecordId) + 1;
	m_curKeyPos.uiBlockAddr = BT_END;

	if (RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone,
								FALSE, FALSE, &bRefGone)))
	{
		restoreCurrKeyPos( pSaveKeyPos);
		goto Exit;
	}
	
	m_bAtEOF = FALSE;
	m_bAtBOF = FALSE;
	m_pCurSet = pKeySet;
	
Exit:

	if (pSaveKeyPos)
	{
		f_free( &pSaveKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Position to the input key + domain
****************************************************************************/
RCODE FSIndexCursor::positionToDomain(
	FDB *				pDb,
	FLMBYTE *		pKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDomain)
{
	RCODE				rc;
	FLMBOOL			bKeyGone;
	FLMBOOL			bRefGone;
	KEYSET *			pKeySet;
	KEYPOS *			pSaveKeyPos = NULL;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	// A key with a non-zero key length must be passed in.

	flmAssert( uiKeyLen != 0);

	// Make sure we fall inside one of the key ranges and
	// save the current key position so we can restore it
	// if the reposition call fails.

	if( RC_BAD( rc = f_alloc( sizeof( KEYPOS), &pSaveKeyPos)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = getKeySet( pKey, uiKeyLen, &pKeySet)))
	{
		goto Exit;
	}
	
	saveCurrKeyPos( pSaveKeyPos);
	releaseKeyBlocks( &m_curKeyPos);
	m_curKeyPos.uiKeyLen = uiKeyLen;
	f_memcpy( m_curKeyPos.pKey, pKey, uiKeyLen);
	m_curKeyPos.uiRecordId = 0;
	m_curKeyPos.uiDomain = uiDomain;
	m_curKeyPos.uiBlockAddr = BT_END;

	if (RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone,
								FALSE, FALSE, &bRefGone)))
	{
		restoreCurrKeyPos( pSaveKeyPos);
		goto Exit;
	}
	
	m_bAtEOF = FALSE;
	m_bAtBOF = FALSE;
	m_pCurSet = pKeySet;
	
Exit:

	if (pSaveKeyPos)
	{
		f_free( &pSaveKeyPos);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Save the current position.
****************************************************************************/
RCODE FSIndexCursor::savePosition( void)
{
	RCODE			rc = FERR_OK;

	if( !m_pSavedPos)
	{
		if( RC_BAD( rc = f_calloc( sizeof( KEYPOS), &m_pSavedPos)))
		{
			goto Exit;
		}	
	}
	
	f_memcpy( m_pSavedPos, &m_curKeyPos, sizeof( KEYPOS));
	m_curKeyPos.bStackInUse = FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FSIndexCursor::restorePosition( void)
{
	if( m_pSavedPos)
	{
		releaseKeyBlocks( &m_curKeyPos);
		f_memcpy( &m_curKeyPos, m_pSavedPos, sizeof(KEYPOS));
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:	Compare two key positions as they would be ordered in the index.
****************************************************************************/
FSTATIC FLMINT FSCompareKeys(			// negative is '<', 0 is '=', position='>'
	FLMBOOL			bKey1IsUntilKey,	// TRUE (until key) or FALSE (from key)
	FLMBYTE *		pKey1,
	FLMUINT			uiKeyLen1,
	FLMBOOL			bExclusiveKey1,
	FLMBOOL			bKey2IsUntilKey,
	FLMBYTE *		pKey2,
	FLMUINT			uiKeyLen2,
	FLMBOOL			bExclusiveKey2)
{	
	FLMINT			iCmp;

	// Handle the first key and last key issues.

	if( !uiKeyLen1)						// FROM or UNTIL key at end point?
	{
		if( !bKey1IsUntilKey)			// FROM key
		{
			iCmp = !uiKeyLen2 && !bKey2IsUntilKey ? 0 : -1;
		}
		else									// UNTIL key
		{
			iCmp = !bKey2IsUntilKey ? 1 : (!uiKeyLen2 ? 0 : 1);
		}
	}
	else if( !uiKeyLen2)
	{
		if( !bKey2IsUntilKey)			// FROM key
		{
			iCmp = !uiKeyLen1 && !bKey1IsUntilKey ? 0 : 1;
		}
		else									// UNTIL key
		{
			iCmp = !bKey1IsUntilKey ? -1 : (!uiKeyLen1 ? 0 : -1);
		}
		
	}
	else if( uiKeyLen1 > uiKeyLen2)
	{
		// Compare the key buffers.  No FIRST or LAST key now.
		
		if( (iCmp = f_memcmp( pKey1, pKey2, uiKeyLen2)) == 0)
		{
			iCmp = 1;
		}
	}
	else if( uiKeyLen1 < uiKeyLen2)
	{
		if( (iCmp = f_memcmp( pKey1, pKey2, uiKeyLen1)) == 0)
		{
			iCmp = -1;
		}
	}
	else
	{
		if( (iCmp = f_memcmp( pKey1, pKey2, uiKeyLen1)) == 0)
		{
			// The keys are EQUAL.  
			// bExclusiveKey ONLY applies to an UNTIL key.

			// Check the exclusive flag and THEN if needed the uiRecordId.

			if( !bKey1IsUntilKey)
			{
				if( bKey2IsUntilKey && bExclusiveKey2)
				{
					iCmp = 1;
				}
			}
			else if( !bKey2IsUntilKey)
			{
				if( bKey1IsUntilKey && bExclusiveKey1)
				{
					iCmp = -1;
				}
			}
			else // both are until keys.
			{
				if( bExclusiveKey1 != bExclusiveKey2)
				{
					iCmp = bExclusiveKey1 ? -1 : 1;
				}
			}
		}
	}
	
	return (iCmp);
}

/****************************************************************************
Desc:	Allocate and return the first and last keys in an index.
****************************************************************************/
RCODE	FSIndexCursor::getFirstLastKeys(	
	FLMBYTE **		ppFirstKey,
	FLMUINT *		puiFirstKeyLen,
	FLMBYTE **		ppLastKey,
	FLMUINT *		puiLastKeyLen,
	FLMBOOL *		pbLastKeyExclusive)
{
	RCODE				rc = FERR_OK;
	KEYSET *			pCurSet = m_pFirstSet;

	if( !pCurSet)
	{
		*ppFirstKey = *ppLastKey = NULL;
		*puiFirstKeyLen = 0;
		*pbLastKeyExclusive = TRUE;
		goto Exit;
	}
	
	if( RC_BAD( rc = f_alloc( pCurSet->fromKey.uiKeyLen, 
		ppFirstKey)))
	{
		goto Exit;
	}
	
	*puiFirstKeyLen = pCurSet->fromKey.uiKeyLen;
	f_memcpy( *ppFirstKey, pCurSet->fromKey.pKey, *puiFirstKeyLen);

	while( pCurSet->pNext)
	{
		pCurSet = pCurSet->pNext;
	}

	if( RC_BAD( rc = f_alloc( pCurSet->untilKey.uiKeyLen, 
		ppLastKey)))
	{
		if( *ppFirstKey)
		{
			f_free( ppFirstKey);
		}
		
		goto Exit;
	}
	
	*puiLastKeyLen = pCurSet->untilKey.uiKeyLen;
	f_memcpy( *ppLastKey, pCurSet->untilKey.pKey, *puiLastKeyLen);
	*pbLastKeyExclusive = pCurSet->untilKey.bExclusiveKey;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set the absolute position in a positioning index relative to the
		FROM/UNTIL keys in all of the key sets. Supports multiple key sets.
****************************************************************************/
RCODE FSIndexCursor::setAbsolutePosition(
	FDB *				pDb,
	FLMUINT			uiRefPosition)		// 0 -> BOF, -1 -> EOF points, one based
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBtreeRefPosition;
	KEYSET *			pTempSet;

	if( !isAbsolutePositionable())
	{
		flmAssert(0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	// Zero value position to BOF
	
	if( uiRefPosition == 0 || uiRefPosition == 1)
	{
		// Go to the first reference.
		
		if( RC_OK( rc = firstKey( pDb, NULL, NULL)))
		{
			if( uiRefPosition == 0)
			{
				// Should return FERR_BOF_HIT
				
				rc = prevKey( pDb, NULL, NULL);
			}
		}
		
		goto Exit;
	}

	// High-values value position to EOF
	
	if( uiRefPosition == (FLMUINT) -1)
	{
		// Position to EOF
		
		if( RC_OK( rc = lastKey( pDb, NULL, NULL)))
		{
			// Should return FERR_EOF_HIT
			
			rc = nextKey( pDb, NULL, NULL);
		}
		
		goto Exit;
	}

	// Find the set that contains this absolute position.

	uiBtreeRefPosition = 0;
	for( pTempSet = m_pFirstSet; pTempSet; pTempSet = pTempSet->pNext)
	{
		FLMUINT			uiTotalInSet;

		uiTotalInSet = pTempSet->untilKey.uiRefPosition - 
							pTempSet->fromKey.uiRefPosition;

		if( uiTotalInSet < uiRefPosition)
		{
			uiRefPosition -= uiTotalInSet;
		}
		else
		{
			m_pCurSet = pTempSet;
			uiBtreeRefPosition = pTempSet->fromKey.uiRefPosition + uiRefPosition - 1;
			break;
		}
	}
	
	if( !uiBtreeRefPosition)
	{
		// Position to EOF
		
		if( RC_OK( rc = lastKey( pDb, NULL, NULL)))
		{
			// Should return FERR_EOF_HIT
			
			rc = nextKey( pDb, NULL, NULL);
		}
		
		goto Exit;
	}

	m_curKeyPos.pStack = m_curKeyPos.Stack;
	RESET_DINSTATE( m_curKeyPos.DinState);
	m_curKeyPos.Stack[0].pKeyBuf = m_curKeyPos.pKey;

	if( RC_BAD( rc = FSPositionSearch( pDb, m_pLFile, uiBtreeRefPosition,
			&m_curKeyPos.pStack, &m_curKeyPos.uiRecordId, 
			&m_curKeyPos.uiDomain, &m_curKeyPos.DinState)))
	{
		goto Exit;
	}
	
	m_curKeyPos.bStackInUse = TRUE;
	setKeyItemsFromBlock( &m_curKeyPos);
	m_bAtBOF = m_bAtEOF = FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get the absolute position in a positioning index.
		Supports multiple sets.
****************************************************************************/
RCODE FSIndexCursor::getAbsolutePosition(
	FDB *				pDb,
	FLMUINT *		puiRefPosition)		// 0 -> BOF, -1 -> EOF points, one based
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiRefPosition = 0;	// Position relative to the FROM key sets.
	FLMUINT			uiBtreeRefPosition;	// True absolute position in the btree.
	KEYSET *			pTempSet;

	if( !isAbsolutePositionable())
	{
		flmAssert(0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	if( m_bAtBOF)
	{
		*puiRefPosition = 0;
		goto Exit;
	}
	
	if( m_bAtEOF)
	{
		*puiRefPosition = (FLMUINT) -1;
		goto Exit;
	}

	if( !m_curKeyPos.bStackInUse)
	{
		FLMBOOL		bKeyGone;
		FLMBOOL		bRefGone;
		
		if( RC_BAD( rc = reposition( pDb, FALSE, FALSE, &bKeyGone, 
													 FALSE, FALSE, &bRefGone)))
		{
			if( rc != FERR_NOT_FOUND)
				goto Exit;
			rc = FERR_OK;
		}
	}

	// Compute where we are relative the the reference position of the current set.

	if( RC_BAD( rc = FSGetBtreeRefPosition( pDb, m_curKeyPos.pStack,
				&m_curKeyPos.DinState, &uiBtreeRefPosition)))
	{
		goto Exit;
	}
	
	uiRefPosition = (uiBtreeRefPosition - m_pCurSet->fromKey.uiRefPosition) + 1;

	for( pTempSet = m_pCurSet->pPrev; pTempSet; pTempSet = pTempSet->pPrev)
	{
		FLMUINT			uiTemp;

		uiTemp = pTempSet->untilKey.uiRefPosition - pTempSet->fromKey.uiRefPosition;
		uiRefPosition += uiTemp;
	}

Exit:

	if( RC_OK( rc))
	{
		*puiRefPosition = uiRefPosition;
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Get the total number of reference for all of the sets.
****************************************************************************/
RCODE FSIndexCursor::getTotalReferences(
	FDB *				pDb,
	FLMUINT *		puiTotalRefs,
	FLMBOOL *		pbTotalsEstimated)
{
	RCODE				rc = FERR_OK;
	KEYSET *			pKeySet;			// Current source set
	KEYSET *			pTempSet = NULL;
	FLMUINT			uiTotalRefs = 0;
	FLMUINT			uiRefCount;
	FLMUINT			uiStackPos;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	*pbTotalsEstimated = FALSE;

	// We have to be careful not to change the current position in any way.

	for( pKeySet = m_pFirstSet; pKeySet; pKeySet = pKeySet->pNext)
	{
		// If this is an positioning index, the ref positions MAY be set up.

		if( pKeySet->fromKey.uiRefPosition && pKeySet->untilKey.uiRefPosition)
		{
			uiTotalRefs += pKeySet->untilKey.uiRefPosition - 
								pKeySet->fromKey.uiRefPosition;
			continue;
		}
		else if( !pTempSet)
		{
			if( RC_BAD( rc = f_calloc( sizeof( KEYSET), &pTempSet)))
			{
				goto Exit;
			}	
		}
		
		// Compute the counts the old fashion way.

		f_memcpy( pTempSet, pKeySet, sizeof( KEYSET));
		for( uiStackPos = 0; uiStackPos < BH_MAX_LEVELS; uiStackPos++)
		{
			pTempSet->fromKey.Stack[uiStackPos].pSCache = NULL;
			pTempSet->fromKey.Stack[uiStackPos].pBlk = NULL;
			pTempSet->untilKey.Stack[uiStackPos].pSCache = NULL;
			pTempSet->untilKey.Stack[uiStackPos].pBlk = NULL;
		}
		
		pTempSet->fromKey.bStackInUse = FALSE;
		pTempSet->untilKey.bStackInUse = FALSE;

		// Search down the tree and get the counts.
		
		if( RC_OK( rc = setKeyPosition( pDb, TRUE, 
					&pTempSet->fromKey, &pTempSet->fromKey)))
		{
			// All keys bewteen low and high may be gone.
			
			if( FS_COMPARE_KEYS( FALSE, &pTempSet->fromKey,
										TRUE, &pKeySet->untilKey) > 0)
			{
				rc = RC_SET( FERR_BOF_HIT);
			}
			else
			{
				rc = setKeyPosition( pDb, FALSE, 
					&pTempSet->untilKey, &pTempSet->untilKey);
			}
		}
		
		if( RC_BAD( rc))
		{
			// Empty tree case.
			
			if( rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = FSComputeIndexCounts( pTempSet->fromKey.pStack,
				pTempSet->untilKey.pStack, NULL, NULL, 
				&uiRefCount, pbTotalsEstimated)))
			{
				goto Exit;
			}
			
			uiTotalRefs += uiRefCount;
		}
		
		releaseKeyBlocks( &pTempSet->fromKey);
		releaseKeyBlocks( &pTempSet->untilKey);
	}

Exit:

	if( pTempSet)
	{
		releaseKeyBlocks( &pTempSet->fromKey);
		releaseKeyBlocks( &pTempSet->untilKey);
		f_free( &pTempSet);
	}
	
	if( RC_OK(rc))
	{
		*puiTotalRefs = uiTotalRefs;
	}
	
	return( rc);
}
