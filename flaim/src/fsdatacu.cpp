//-------------------------------------------------------------------------
// Desc:	Routines used during query to traverse through container b-trees.
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

/****************************************************************************
Desc:
****************************************************************************/
FSDataCursor::FSDataCursor() 
{
	m_pFirstSet = m_pCurSet = NULL;
	m_pSavedPos = NULL;
	m_curRecPos.bStackInUse = FALSE;
	m_uiContainer = FLM_DATA_CONTAINER;
	reset();
}

/****************************************************************************
Desc:
****************************************************************************/
FSDataCursor::~FSDataCursor() 
{
	releaseBlocks();
	freeSets();
}

/****************************************************************************
Desc:	Resets any allocations, keys, state, etc.
****************************************************************************/
void FSDataCursor::reset( void)
{
	releaseBlocks();
	freeSets();

	m_uiContainer = m_uiCurrTransId = m_uiBlkChangeCnt = 0;
	m_pFirstSet = m_pCurSet = &m_DefaultSet;
	m_DefaultSet.fromKey.uiRecordId = 1;
	m_DefaultSet.untilKey.uiRecordId = DRN_LAST_MARKER - 1;
	m_DefaultSet.pNext = m_DefaultSet.pPrev = NULL;
	m_DefaultSet.fromKey.bStackInUse = m_DefaultSet.untilKey.bStackInUse = FALSE;

	m_curRecPos.bStackInUse = FALSE;
	m_curRecPos.uiBlockAddr = BT_END;
	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
}


/****************************************************************************
Desc: Resets to a new transaction that may change the read consistency of
		the query.  This is usually an old view error internal or external of
		this class.
****************************************************************************/
RCODE FSDataCursor::resetTransaction( 
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	RECSET *		pTmpSet;

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, m_uiContainer, 
			&m_pLFile)))
	{	
		goto Exit;
	}
	m_uiCurrTransId = pDb->LogHdr.uiCurrTransID;
	m_uiBlkChangeCnt = pDb->uiBlkChangeCnt;
	m_bIsUpdateTrans = (pDb->uiTransType == FLM_UPDATE_TRANS) ? TRUE : FALSE;

	// Need to release all stacks that are currently in use.

	for( pTmpSet = m_pFirstSet; pTmpSet; pTmpSet = pTmpSet->pNext)
	{
		releaseRecBlocks( &pTmpSet->fromKey);
		releaseRecBlocks( &pTmpSet->untilKey);
	}

	releaseRecBlocks( &m_DefaultSet.fromKey);
	releaseRecBlocks( &m_DefaultSet.untilKey);

	if( m_pSavedPos)
	{
		releaseRecBlocks( m_pSavedPos);
	}

	releaseRecBlocks( &m_curRecPos);

Exit:
	return( rc);
}

/****************************************************************************
Desc: Free all of the allocated sets.
****************************************************************************/
void FSDataCursor::freeSets( void)
{
	RECSET *		pCurSet;			// Current set.
	RECSET *		pNextSet;		// Current set.

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
		releaseRecBlocks( m_pSavedPos);
		f_free( &m_pSavedPos);
		m_pSavedPos = NULL;
	}
	return;
}


/****************************************************************************
Desc:	Releases the cache blocks back to cache.
****************************************************************************/
void FSDataCursor::releaseBlocks( void)
{
	RECSET *		pCurSet;

	// Unuse all of the cache blocks in the from and until keys.

	for( pCurSet = m_pFirstSet; pCurSet; pCurSet = pCurSet->pNext)
	{
		releaseRecBlocks( &pCurSet->fromKey);
		releaseRecBlocks( &pCurSet->untilKey);
	}
	releaseRecBlocks( &m_curRecPos);

	return;
}

/****************************************************************************
Desc:	Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the index.
		This code does not work with multiple key sets of FROM/UNTIL keys.
****************************************************************************/
RCODE FSDataCursor::setupRange(
	FDB *			pDb,
	FLMUINT		uiContainer,
	FLMUINT		uiLowRecordId,
	FLMUINT		uiHighRecordId,
	FLMUINT *	puiLeafBlocksBetween,// [out] blocks between the stacks
	FLMUINT *	puiTotalRecords,		// [out]
	FLMBOOL *	pbTotalsEstimated)	// [out] set to TRUE when estimating.
{
	RCODE			rc = FERR_OK;

	if( uiLowRecordId == DRN_LAST_MARKER)
	{
		uiLowRecordId = DRN_LAST_MARKER - 1;
	}
	if( uiHighRecordId == DRN_LAST_MARKER)
	{
		uiHighRecordId = DRN_LAST_MARKER - 1;
	}
	m_uiContainer = uiContainer;
	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	m_pCurSet = m_pFirstSet = &m_DefaultSet;
	m_DefaultSet.fromKey.uiRecordId = uiLowRecordId;
	m_DefaultSet.untilKey.uiRecordId = uiHighRecordId;

	// Want any of the counts back?
	if( puiLeafBlocksBetween || puiTotalRecords)
	{
		// Even though this is INCORRECT, the UNTIL record is inclusive and
		// we are NOT going to count the blocks the UNTIL record spans.
		// If we positioned to the (UNTIL record + 1) then we may do a lot
		// of extra block reads for nothing.

		if( uiLowRecordId == uiHighRecordId)
		{
			if( puiLeafBlocksBetween)
			{
				*puiLeafBlocksBetween = 0;
			}
			if( puiTotalRecords)
			{
				*puiTotalRecords = 0;
			}
		}
		else
		{
			// Position to the FROM and UNTIL key so we can get the stats.
			if( RC_OK( rc = setRecPosition( pDb, TRUE,
				&m_DefaultSet.fromKey, &m_DefaultSet.fromKey)))
			{
				// All records between LOW and HIGH may be gone - check.
				if( m_DefaultSet.fromKey.uiRecordId > uiHighRecordId)
				{
					rc = RC_SET( FERR_BOF_HIT);
				}
				else
				{
					m_DefaultSet.fromKey.uiRecordId = uiLowRecordId;
				
					rc = setRecPosition( pDb, FALSE,
						&m_DefaultSet.untilKey, &m_DefaultSet.untilKey);
					m_DefaultSet.untilKey.uiRecordId = uiHighRecordId;
				}
			}
			if( RC_BAD( rc))
			{
				if( rc == FERR_EOF_HIT || rc == FERR_BOF_HIT)
				{
					if( puiLeafBlocksBetween)
						*puiLeafBlocksBetween = 0;
					if( puiTotalRecords)
						*puiTotalRecords = 0;
					if( pbTotalsEstimated)
						*pbTotalsEstimated = FALSE;
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = FSComputeRecordBlocks( 
					m_DefaultSet.fromKey.pStack, m_DefaultSet.untilKey.pStack, 
					puiLeafBlocksBetween, puiTotalRecords, pbTotalsEstimated)))
				{
					goto Exit;
				}
			}
		}
	}
	m_bAtBOF = TRUE;
	m_pCurSet = NULL;

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Merge the input cursors fromUntil sets as a result of a UNION.
****************************************************************************/
RCODE FSDataCursor::unionRange( 
	FSDataCursor * pFSCursor)
{
	RCODE			rc = FERR_OK;
	RECSET *		pInputSet;				// Sets input via pFSCursor.
	RECSET *		pSrcSet;					// Current set
	RECSET *		pDestSet = NULL;		// Newly allocated set
	RECSET *		pCurDestSet = NULL;	// New current set
	RECSET *		pPrevDestSet;
	FLMBOOL		bKeysOverlap;

	pInputSet = pFSCursor->getFromUntilSets();
	pSrcSet = m_pFirstSet;
	while( pSrcSet || pInputSet)
	{
		FLMBOOL		bFromKeyLessThan;
		FLMBOOL		bUntilKeyGreaterThan = FALSE;

		pPrevDestSet = pCurDestSet;
		if( RC_BAD( rc = f_calloc( sizeof( RECSET), &pCurDestSet)))
		{
			goto Exit;
		}	
		if( !pSrcSet)
		{	
			bKeysOverlap = FALSE;
			bFromKeyLessThan = TRUE;
		}
		else if( !pInputSet)
		{
			bKeysOverlap = FALSE;
			bFromKeyLessThan = FALSE;
		}
		else
		{
			bKeysOverlap = FSCompareRecPos( pInputSet, pSrcSet, 
						&bFromKeyLessThan, &bUntilKeyGreaterThan);

			// Do a special optimization to join two adjacent sets.
			if( !bKeysOverlap)
			{
				if( bFromKeyLessThan)
				{
					if( pInputSet->untilKey.uiRecordId + 1 == 
						 pSrcSet->fromKey.uiRecordId)
					{
						bKeysOverlap = TRUE;
					}
				}
				else
				{
					if( pInputSet->fromKey.uiRecordId - 1 == 
						 pSrcSet->untilKey.uiRecordId)
					{
						bKeysOverlap = TRUE;
					}
				}
			}
		}
		if( !bKeysOverlap)
		{
			if( bFromKeyLessThan)
			{
				pCurDestSet->fromKey.uiRecordId = pInputSet->fromKey.uiRecordId;
				pCurDestSet->untilKey.uiRecordId = pInputSet->untilKey.uiRecordId;
				pInputSet = pInputSet->pNext;
			}
			else
			{
				pCurDestSet->fromKey.uiRecordId = pSrcSet->fromKey.uiRecordId;
				pCurDestSet->untilKey.uiRecordId = pSrcSet->untilKey.uiRecordId;
				pSrcSet = pSrcSet->pNext;
			}
		}
		else
		{
			// Keys overlap.  Change the FROM or UNTIL recordId if needed.

			pCurDestSet->fromKey.uiRecordId = bFromKeyLessThan ? 
				pInputSet->fromKey.uiRecordId : pSrcSet->fromKey.uiRecordId;
			
			for(;;)
			{
				if( bUntilKeyGreaterThan)
				{
					if( ((pSrcSet = pSrcSet->pNext) == NULL)
					 || !FSCompareRecPos( pInputSet, pSrcSet,
								&bFromKeyLessThan, &bUntilKeyGreaterThan))
					{
						pCurDestSet->untilKey.uiRecordId = 
							pInputSet->untilKey.uiRecordId;
						pInputSet = pInputSet->pNext;
						break;
					}
				}
				else
				{
					if( ((pInputSet = pInputSet->pNext) == NULL)
					 || !FSCompareRecPos( pInputSet, pSrcSet,
								&bFromKeyLessThan, &bUntilKeyGreaterThan))
					{
						pCurDestSet->untilKey.uiRecordId = 
							pSrcSet->untilKey.uiRecordId;
						pSrcSet = pSrcSet->pNext;
						break;
					}
				}
			}
		}
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
		f_memcpy( &m_DefaultSet, pCurDestSet, sizeof( RECSET));
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

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Intersect the from/until key sets of pFSCursor into 'this'.
****************************************************************************/
RCODE FSDataCursor::intersectRange( 
	FSDataCursor * pFSCursor)
{
	RCODE			rc = FERR_OK;
	RECSET *		pInputSet;				// Sets input via pFSCursor.
	RECSET *		pSrcSet;					// Current set
	RECSET *		pDestSet = NULL;		// Newly allocated set
	RECSET *		pCurDestSet = NULL;	// New current set
	RECSET *		pPrevDestSet;

	pInputSet = pFSCursor->getFromUntilSets();
	pSrcSet = m_pFirstSet;

	while( pSrcSet && pInputSet)
	{
		FLMBOOL		bFromKeyLessThan;
		FLMBOOL		bUntilKeyGreaterThan;

		if( !FSCompareRecPos( pInputSet, pSrcSet, 
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
			if( RC_BAD( rc = f_calloc( sizeof( RECSET), &pCurDestSet)))
			{
				goto Exit;
			}	
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

			// Take the highest FROM key
			pCurDestSet->fromKey.uiRecordId = bFromKeyLessThan 
				? pSrcSet->fromKey.uiRecordId : pInputSet->fromKey.uiRecordId;

			// Take the lowest until key and position to the next set.
			if( bUntilKeyGreaterThan)
			{
				pCurDestSet->untilKey.uiRecordId = pSrcSet->untilKey.uiRecordId;
				pSrcSet = pSrcSet->pNext;
			}
			else
			{
				pCurDestSet->untilKey.uiRecordId = pInputSet->untilKey.uiRecordId;
				pInputSet = pInputSet->pNext;
			}
		}
	}
	// We went to the trouble of having a default set allocated with this class.
	// Undo the last allocation.  if( pDestSet) then pCurDestSet can be used.
	freeSets();
	if( pDestSet)
	{
		f_memcpy( &m_DefaultSet, pCurDestSet, sizeof( RECSET));
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

Exit:
	return( rc);
}

/****************************************************************************
Desc: Compare two From/Until key positions.
****************************************************************************/
FLMBOOL FSDataCursor::FSCompareRecPos(			// TRUE if keys overlap
	RECSET *		pSet1,
	RECSET *		pSet2,
	FLMBOOL *	pbFromKeyLessThan,		// pSet1->from < pSet2->from
	FLMBOOL *	pbUntilKeyGreaterThan)	// pSet1->until > pSet2->until
{

	if( pSet1->untilKey.uiRecordId < pSet2->fromKey.uiRecordId)
	{
		*pbFromKeyLessThan = TRUE;
		pbUntilKeyGreaterThan = FALSE;
		return FALSE;
	}
	if( pSet1->fromKey.uiRecordId > pSet2->untilKey.uiRecordId)
	{
		*pbFromKeyLessThan = FALSE;
		*pbUntilKeyGreaterThan = TRUE;
		return FALSE;
	}

	// Keys overlap.  Two more compares needed

	*pbFromKeyLessThan = (FLMBOOL) 
		((pSet1->fromKey.uiRecordId < pSet2->fromKey.uiRecordId) ? TRUE : FALSE);

	*pbUntilKeyGreaterThan = (FLMBOOL)
		((pSet1->untilKey.uiRecordId > pSet2->untilKey.uiRecordId) ? TRUE : FALSE);

	return TRUE;
}

/****************************************************************************
Desc: 	Return the current record and record id.
VISIT:	We may want to return BOF/EOF when positioned on an endpoint.
****************************************************************************/
RCODE FSDataCursor::currentRec(		// FERR_OK, FERR_EOF_HIT or error
	FDB *				pDb,
	FlmRecord **	ppRecord,			// Will replace what is there
	FLMUINT *		puiRecordId)		// Set the record ID
{
	RCODE				rc;

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

	if( puiRecordId)
	{
		*puiRecordId = m_curRecPos.uiRecordId;
	}
	if( ppRecord)
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, m_uiContainer, 
				m_curRecPos.uiRecordId, TRUE, m_curRecPos.pStack,
				m_pLFile, ppRecord)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Position to and return the first record.
		This is hard because positioning using the first key may actually
		position past or into another FROM/UNTIL set in the list.
****************************************************************************/
RCODE FSDataCursor::firstRec(
	FDB *				pDb,
	FlmRecord **	ppRecord,		// Will replace what is there
	FLMUINT *		puiRecordId)	// Set the record ID
{
	RCODE				rc;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	m_pCurSet = m_pFirstSet;
	if( !m_pCurSet)
	{
		m_bAtBOF = FALSE;
		m_bAtEOF = TRUE;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	m_bAtBOF = m_bAtEOF = FALSE;
	m_curRecPos.uiRecordId = 0;
	for(;;)
	{
		if( m_curRecPos.uiRecordId < m_pCurSet->fromKey.uiRecordId)
		{
			if( RC_BAD( rc = setRecPosition( pDb, TRUE,
				&m_pCurSet->fromKey, &m_curRecPos)))
			{
				if( rc == FERR_BOF_HIT)
				{
					rc = RC_SET( FERR_EOF_HIT);
				}
				m_bAtEOF = TRUE;
				goto Exit;
			}
		}
		// Check to see if the current record is within some set.

		if( m_curRecPos.uiRecordId <= m_pCurSet->untilKey.uiRecordId)
		{
			break;
		}
		if( !m_pCurSet->pNext)
		{
			m_bAtEOF = TRUE;
			rc = FERR_EOF_HIT;
			goto Exit;
		}
		m_pCurSet = m_pCurSet->pNext;
	}

	// Return the data.
	if( puiRecordId)
	{
		*puiRecordId = m_curRecPos.uiRecordId;
	}
	if( ppRecord)
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, m_uiContainer, 
				m_curRecPos.uiRecordId, TRUE, m_curRecPos.pStack,
				m_pLFile, ppRecord)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Sets the record ID, block address, and block transaction ID for a
		record position.
****************************************************************************/
FINLINE void setItemsFromBlock(
	RECPOS *	pRecPos
	)
{
	pRecPos->uiRecordId = f_bigEndianToUINT32( pRecPos->pKey);
	pRecPos->uiBlockAddr = pRecPos->pStack->uiBlkAddr;
	pRecPos->uiBlockTransId = (pRecPos->uiBlockAddr != BT_END) 
									  ? FB2UD( &pRecPos->pStack->pBlk[ BH_TRANS_ID])
									  : 0;
}

/****************************************************************************
Protected:	setRecPosition
Desc: 		Set the key position given some RECPOS structure.
				Please note that the blocks in the stack may or may not be used.
****************************************************************************/
RCODE FSDataCursor::setRecPosition(
	FDB *				pDb,
	FLMBOOL			bGoingForward,
	RECPOS *			pInRecPos,		// Input key position
	RECPOS *			pOutRecPos)		// Output to setup the stack/keybuffer.
											// It is ok for Input==Output
{
	RCODE				rc;
	FLMUINT			uiRecordId;
	FLMBYTE			buf[ DIN_KEY_SIZ];

	// May have to unuse the b-tree blocks.  Then setup the stack.
	if( !pOutRecPos->bStackInUse)
	{
		FSInitStackCache( pOutRecPos->Stack, BH_MAX_LEVELS);
		pOutRecPos->bStackInUse = TRUE;
	}

	// Setup the stack.
	pOutRecPos->pStack = pOutRecPos->Stack;
	pOutRecPos->Stack[0].pKeyBuf = pOutRecPos->pKey;
	uiRecordId = pInRecPos->uiRecordId;
	f_UINT32ToBigEndian( (FLMUINT32)uiRecordId, buf);

	// All of the variables should be setup for the search.
	if( RC_BAD( rc = FSBtSearch( pDb, m_pLFile, &pOutRecPos->pStack,
				buf, DIN_KEY_SIZ, 0)))
	{
		goto Exit;
	}

	if( pOutRecPos->pStack->uiBlkAddr == BT_END)
	{
		pOutRecPos->bStackInUse = FALSE;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	if( bGoingForward)
	{
		if( pOutRecPos->pStack->uiCmpStatus == BT_END_OF_DATA ||
			f_bigEndianToUINT32( pOutRecPos->pKey) == DRN_LAST_MARKER)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}
	else
	{
		if( (pOutRecPos->pStack->uiCmpStatus == BT_END_OF_DATA) ||
			 (f_bigEndianToUINT32( pOutRecPos->pKey) > uiRecordId))
		{
			// Went a little too far - go back to the previous record.
			// Need to position back one element.
			if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pOutRecPos->pStack)))
			{
					if( rc == FERR_BT_END_OF_DATA)
					{ 
						rc = RC_SET( FERR_BOF_HIT);
					}
					goto Exit;
			}
			while( BBE_NOT_FIRST( CURRENT_ELM( pOutRecPos->pStack )))
			{
				if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pOutRecPos->pStack)))
				{
					if( rc == FERR_BT_END_OF_DATA)
					{ 
						rc = RC_SET( FERR_BTREE_ERROR);
					}
					goto Exit;
				}
			}
		}
	}

	// Allow us to navigate at the low level of the b-tree.
	pOutRecPos->pStack->uiFlags = NO_STACK;
	
	// Obtain the recordId of where we are positioned.
	setItemsFromBlock( pOutRecPos);
Exit:
	return( rc);
}

/****************************************************************************
Desc: Position to and return the last record.
		This is hard because positioning using the first key may actually
		position past or into another FROM/UNTIL set in the list.
****************************************************************************/
RCODE FSDataCursor::lastRec(
	FDB *				pDb,
	FlmRecord **	ppRecord,			// Will replace what is there
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
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	while( m_pCurSet->pNext)
	{
		m_pCurSet = m_pCurSet->pNext;
	}

	m_bAtBOF = m_bAtEOF = FALSE;
	m_curRecPos.uiRecordId = DRN_LAST_MARKER;
	for(;;)
	{
		if( m_curRecPos.uiRecordId > m_pCurSet->untilKey.uiRecordId)
		{
			if( RC_BAD( rc = setRecPosition( pDb, FALSE,
					&m_pCurSet->untilKey, &m_curRecPos)))
			{
				if( rc == FERR_EOF_HIT)
				{
					// Look at empty case verses end of file case.
					if( m_curRecPos.uiBlockAddr != BT_END)
					{
						rc = FERR_OK;
					}
					else
					{
						m_bAtBOF = TRUE;
						rc = RC_SET( FERR_BOF_HIT);
					}
				}
				if( RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
		// Check to see if the current record is within some set.

		if( m_curRecPos.uiRecordId > m_pCurSet->untilKey.uiRecordId)
		{
			// Position to the previous record.
			BTSK *		pStack = m_curRecPos.pStack;

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
			// Need to position back one element.
			if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
			{
					if( rc == FERR_BT_END_OF_DATA)
					{ 
						m_bAtBOF = TRUE;
						rc = RC_SET( FERR_BOF_HIT);
					}
					goto Exit;
			}
			while( BBE_NOT_FIRST( CURRENT_ELM( pStack )))
			{
				if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack )))
				{
					if( rc == FERR_BT_END_OF_DATA)
					{ 
						rc = RC_SET( FERR_BTREE_ERROR);
					}
					goto Exit;
				}
			}
			setItemsFromBlock( &m_curRecPos);
		}
		if( m_curRecPos.uiRecordId <= m_pCurSet->untilKey.uiRecordId
		 && m_curRecPos.uiRecordId >= m_pCurSet->fromKey.uiRecordId)
		{
			break;
		}
		if( !m_pCurSet->pPrev)
		{
			m_bAtBOF = TRUE;
			rc = FERR_BOF_HIT;
			goto Exit;
		}
		m_pCurSet = m_pCurSet->pPrev;
	}

	// Return the data.
	
	if( puiRecordId)
	{
		*puiRecordId = m_curRecPos.uiRecordId;
	}
	
	if( ppRecord)
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, m_uiContainer, 
				m_curRecPos.uiRecordId, TRUE, m_curRecPos.pStack,
				m_pLFile, ppRecord)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Position to the next key and the first reference of that key.
****************************************************************************/
RCODE FSDataCursor::nextRec(
	FDB *				pDb,
	FlmRecord **	ppRecord,
	FLMUINT *		puiRecordId)
{
	RCODE				rc;
	FLMBOOL			bRecordGone;
	BTSK *			pStack = m_curRecPos.pStack;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	if( m_bAtEOF)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	if( !m_pCurSet || m_bAtBOF)
	{
		rc = firstRec( pDb, ppRecord, puiRecordId);
		goto Exit;
	}

	bRecordGone = FALSE;

	// Takes care of any re-read of a block if we changed transactions.

	if( !m_curRecPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, TRUE, FALSE, &bRecordGone)))
		{
			// May return FERR_EOF_HIT if all remaining keys are deleted.
			goto Exit;
		}
	}
	for(;;)
	{
		if( !bRecordGone)
		{
			// Optimization:
			// If we are on the UNTIL record no reason to go forward.
			if( m_curRecPos.uiRecordId < m_pCurSet->untilKey.uiRecordId)
			{
				FLMBYTE *		pCurElm;

				pCurElm = CURRENT_ELM( pStack);
				while( BBE_NOT_LAST( pCurElm))
				{
					if( RC_BAD( rc = FSBtNextElm( pDb, m_pLFile, pStack)))
					{
						// b-tree corrupt if FERR_BT_END_OF_DATA .
						if( rc == FERR_BT_END_OF_DATA)		
						{
							rc = RC_SET( FERR_BTREE_ERROR);
						}
						goto Exit;
					}
					pCurElm = CURRENT_ELM( pStack);
				}

				// Now go to the next element.

				if( RC_BAD(rc = FSBtNextElm( pDb, m_pLFile, pStack)))
				{
					// b-tree corrupt if FERR_BT_END_OF_DATA .
					if( rc == FERR_BT_END_OF_DATA)		
					{
						rc = RC_SET( FERR_EOF_HIT);
					}
					goto Exit;
				}
				bRecordGone = TRUE;
				if( f_bigEndianToUINT32( m_curRecPos.pKey) <=
						m_pCurSet->untilKey.uiRecordId)
				{
					setItemsFromBlock( &m_curRecPos);
					break;
				}
			}
		}
		else
		{
			if( m_curRecPos.uiRecordId <= m_pCurSet->untilKey.uiRecordId)
			{
				break;
			}
		}
		if( !m_pCurSet->pNext)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
		m_pCurSet = m_pCurSet->pNext;
		if( m_curRecPos.uiRecordId < m_pCurSet->fromKey.uiRecordId)
		{
			// Re-read the record resetting all of the blocks.
			if( RC_BAD( rc = setRecPosition( pDb, TRUE, 
					&m_pCurSet->fromKey, &m_curRecPos)))
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
		*puiRecordId = m_curRecPos.uiRecordId;
	}
	if( ppRecord)
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, m_uiContainer, 
				m_curRecPos.uiRecordId, TRUE, m_curRecPos.pStack,
				m_pLFile, ppRecord)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Position to the PREVIOUS record.
****************************************************************************/
RCODE FSDataCursor::prevRec(
	FDB *				pDb,
	FlmRecord **	ppRecord,
	FLMUINT *		puiRecordId)
{
	RCODE				rc;
	FLMBOOL			bRecordGone;
	BTSK *			pStack = m_curRecPos.pStack;

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
		rc = lastRec( pDb, ppRecord, puiRecordId);
		goto Exit;
	}

	bRecordGone = FALSE;

	// Takes care of any re-read of a block if we changed transactions.

	if( !m_curRecPos.bStackInUse)
	{
		if( RC_BAD( rc = reposition( pDb, FALSE, TRUE, &bRecordGone)))
		{
			// May return FERR_EOF_HIT if all remaining keys are deleted.
			goto Exit;
		}
	}
	for(;;)
	{
		if( !bRecordGone)
		{
			// Optimization:
			// If we are on the FROM record no reason to go backwards.
			if( m_curRecPos.uiRecordId > m_pCurSet->fromKey.uiRecordId)
			{
				FLMBYTE *		pCurElm;

				pCurElm = CURRENT_ELM( pStack);
				while( BBE_NOT_FIRST( pCurElm))
				{
					if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
					{
						// b-tree corrupt if FERR_BT_END_OF_DATA .
						if( rc == FERR_BT_END_OF_DATA)		
						{
							rc = RC_SET( FERR_BTREE_ERROR);
						}
						goto Exit;
					}
					pCurElm = CURRENT_ELM( pStack);
				}
				// Now go back one previous element.

				if( RC_BAD(rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
				{
					// b-tree corrupt if FERR_BT_END_OF_DATA .
					if( rc == FERR_BT_END_OF_DATA)		
					{
						m_bAtBOF = TRUE;
						rc = RC_SET( FERR_BOF_HIT);
					}
					goto Exit;
				}
				pCurElm = CURRENT_ELM( pStack);
				while( BBE_NOT_FIRST( pCurElm))
				{
					if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, pStack)))
					{
						// b-tree corrupt if FERR_BT_END_OF_DATA .
						if( rc == FERR_BT_END_OF_DATA)		
						{
							rc = RC_SET( FERR_BTREE_ERROR);
						}
						goto Exit;
					}
					pCurElm = CURRENT_ELM( pStack);
				}
				bRecordGone = TRUE;
				if( f_bigEndianToUINT32( m_curRecPos.pKey) >= 
						m_pCurSet->fromKey.uiRecordId)
				{
					setItemsFromBlock( &m_curRecPos);
					break;
				}
			}
		}
		else
		{
			if( m_curRecPos.uiRecordId >= m_pCurSet->fromKey.uiRecordId)
			{
				break;
			}
		}
		if( !m_pCurSet->pPrev)
		{
			m_bAtBOF = TRUE;
			rc = RC_SET( FERR_BOF_HIT);
			goto Exit;
		}
		m_pCurSet = m_pCurSet->pPrev;
		if( m_curRecPos.uiRecordId > m_pCurSet->untilKey.uiRecordId)
		{
			// Re-read the record resetting all of the blocks.
			if( RC_BAD( rc = setRecPosition( pDb, FALSE,
				&m_pCurSet->untilKey, &m_curRecPos)))
			{
				if( rc == FERR_EOF_HIT)
				{
					m_bAtBOF = TRUE;
				}
				goto Exit;
			}
			if( m_curRecPos.uiRecordId != m_pCurSet->untilKey.uiRecordId)
			{
				bRecordGone = TRUE;
			}
		}
	}

	if( puiRecordId)
	{
		*puiRecordId = m_curRecPos.uiRecordId;
	}
	if( ppRecord)
	{
		if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, m_uiContainer, 
				m_curRecPos.uiRecordId, TRUE, m_curRecPos.pStack,
				m_pLFile, ppRecord)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: 	Reposition to the current key + recordId.  If the current key
			is gone we may reposition past the UNTIL key and should check.
			Called only from next/prev.
****************************************************************************/
RCODE FSDataCursor::reposition(
	FDB *				pDb,
	FLMBOOL			bCanPosToNextRec,	// May be TRUE if bPosToPrevRec is FALSE
	FLMBOOL			bCanPosToPrevRec,	// May be TRUE if bPosToNextRec is FALSE
	FLMBOOL *		pbRecordGone)
{
	RCODE				rc;
	FLMBOOL			bReread = FALSE;
	FLMUINT			uiBlkTransId;
	FLMUINT			uiRecordId = m_curRecPos.uiRecordId;

	flmAssert( !m_curRecPos.bStackInUse);

	// Re-read the block and see if it is the same block.

	if( m_curRecPos.uiBlockAddr == BT_END ||
		m_curRecPos.uiBlockAddr != m_curRecPos.pStack->uiBlkAddr)
	{
		bReread = TRUE;
	}
	else
	{
		if( RC_BAD( rc = FSGetBlock( pDb, m_pLFile, 
					m_curRecPos.uiBlockAddr, m_curRecPos.pStack)))
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
			uiBlkTransId = FB2UD( &m_curRecPos.pStack->pBlk[ BH_TRANS_ID]);
			m_curRecPos.bStackInUse = TRUE;
		}
	}

	/*
	Four cases for re-positioning down the B-Tree:

	1) Block address was BT_END (bReread == TRUE)
	2) FSGetBlock returned FERR_OLD_VIEW (bReread == TRUE)
	3) The transaction ID on the block changed from the last time we read it
	4) We are in an update transaction and the block has been updated
	   (we don't know if it has been updated since we last read it, but
		we take the safe approach of re-positioning down the B-Tree)
	*/

	if( bReread ||
		m_curRecPos.uiBlockTransId != uiBlkTransId ||
		pDb->uiTransType == FLM_UPDATE_TRANS)
	{
		// This may be a new read transaction. 
		if( RC_BAD( rc = setRecPosition( pDb, 
				bCanPosToPrevRec ? FALSE : TRUE, &m_curRecPos, &m_curRecPos)))
		{
			if( rc != FERR_EOF_HIT && rc != FERR_BOF_HIT)
			{
				goto Exit;
			}
		}
		if( RC_BAD( rc) || uiRecordId != m_curRecPos.uiRecordId)
		{
			*pbRecordGone = TRUE;
			if( !bCanPosToNextRec && !bCanPosToPrevRec)
			{
				rc = RC_SET( FERR_NOT_FOUND);
			}

			// Allowed to position to previous record?
			if( bCanPosToPrevRec)
			{
				// Positioned to the next record.  Go the the previous record.
				if( RC_BAD( rc = FSBtPrevElm( pDb, m_pLFile, m_curRecPos.pStack)))
				{
					// b-tree corrupt if FERR_BT_END_OF_DATA.
					if( rc == FERR_BT_END_OF_DATA)
					{
						rc = RC_SET( FERR_BOF_HIT);
						goto Exit;
					}
				}
			}
		}
	}
	m_curRecPos.bStackInUse = TRUE;
	setItemsFromBlock( &m_curRecPos);

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Position to the input key + recordId.
****************************************************************************/
RCODE FSDataCursor::positionTo(
	FDB *				pDb,
	FLMUINT			uiRecordId)
{
	RCODE				rc;
	FLMUINT			uiTempRecordId = uiRecordId;
	RECSET *			pSaveRecSet = m_pCurSet;
	FLMUINT			uiSaveRecId = m_curRecPos.uiRecordId;

	if( RC_BAD( rc = positionToOrAfter( pDb, &uiTempRecordId)))
	{
		if( rc == FERR_EOF_HIT)
		{
			rc = RC_SET( FERR_NOT_FOUND);
		}
		goto Exit;
	}
	if( m_curRecPos.uiRecordId != uiRecordId)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:
	if (RC_BAD( rc))
	{
		m_pCurSet = pSaveRecSet;
		m_curRecPos.uiRecordId = uiSaveRecId;
	}
	return( rc);
};

/****************************************************************************
Desc:	Position to the input key + recordId.
****************************************************************************/
RCODE FSDataCursor::positionToOrAfter(
	FDB *				pDb,
	FLMUINT *		puiRecordId)
{
	RCODE				rc;
	FLMUINT			uiRecordId = *puiRecordId;
	RECSET *			pSaveRecSet = m_pCurSet;
	FLMUINT			uiSaveRecId = m_curRecPos.uiRecordId;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	// See which set the record id is in.

	for( m_pCurSet = m_pFirstSet; m_pCurSet; m_pCurSet = m_pCurSet->pNext)
	{
		if( m_pCurSet->fromKey.uiRecordId <= uiRecordId
		 && m_pCurSet->untilKey.uiRecordId >= uiRecordId)
		{
			break;
		}
	}
	if (!m_pCurSet)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

	m_curRecPos.uiRecordId = uiRecordId;
	if( RC_BAD( rc = setRecPosition( pDb, TRUE, &m_curRecPos, &m_curRecPos)))
	{
		// Could return FERR_EOF_HIT.
		goto Exit;
	}
	*puiRecordId = m_curRecPos.uiRecordId;

Exit:
	if (RC_BAD( rc))
	{
		m_pCurSet = pSaveRecSet;
		m_curRecPos.uiRecordId = uiSaveRecId;
	}
	return( rc);
};

/****************************************************************************
Desc:	Save the current position.
****************************************************************************/
RCODE FSDataCursor::savePosition( void)
{
	RCODE			rc = FERR_OK;

	if( !m_pSavedPos)
	{
		if( RC_BAD( rc = f_calloc( sizeof( RECPOS), &m_pSavedPos)))
		{
			goto Exit;
		}	
	}
	releaseRecBlocks( &m_curRecPos);
	f_memcpy( m_pSavedPos, &m_curRecPos, sizeof( RECPOS));

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Save the current position.
****************************************************************************/
RCODE FSDataCursor::restorePosition( void)
{
	if( m_pSavedPos)
	{
		releaseRecBlocks( &m_curRecPos);
		f_memcpy( &m_curRecPos, m_pSavedPos, sizeof(RECPOS));
	}

	return FERR_OK;
}
