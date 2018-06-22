//------------------------------------------------------------------------------
// Desc:	Cursor routines to get the complexity of the file system out 
//			of the search code.
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
#include "fscursor.h"

FSTATIC RCODE copyKey(
	KEYPOS *			pDestKey,
	F_DataVector *	pDestSrchKey,
	KEYPOS *			pSrcKey,
	F_DataVector *	pSrcSrchKey);
	
/****************************************************************************
Desc:
****************************************************************************/
FSIndexCursor::FSIndexCursor() 
{
	m_pbTree = NULL;
	m_bTreeOpen = FALSE;
	m_pucCurKeyDataBuf = NULL;
	m_uiCurKeyDataBufSize = 0;
	m_uiCurKeyDataLen = 0;
	m_pIndex = NULL;
	m_pLFile = NULL;
	m_pDb = NULL;
	m_eTransType = SFLM_NO_TRANS;
	resetCursor();
}

/****************************************************************************
Desc:
****************************************************************************/
FSIndexCursor::~FSIndexCursor() 
{
	closeBTree();
	if (m_pucCurKeyDataBuf)
	{
		f_free( &m_pucCurKeyDataBuf);
	}
	if (m_pbTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &m_pbTree);
	}
}

/****************************************************************************
Desc:	Resets any allocations, keys, state, etc.
****************************************************************************/
void FSIndexCursor::resetCursor( void)
{
	closeBTree();

	m_uiIndexNum = 0;
	m_uiBlkChangeCnt = 0;
	m_ui64CurrTransId = 0;
	m_curKey.uiKeyLen = 0;
	m_uiCurKeyDataLen = 0;
	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
	m_bSetup = FALSE;
	m_bDoRowMatch = FALSE;
	m_bCanCompareOnKey = TRUE;
	m_ui64Cost = 0;
	m_ui64LeafBlocksBetween = 0;
	m_ui64TotalRefs = 0;
	m_bTotalsEstimated = FALSE;
}

/****************************************************************************
Desc: Resets to a new transaction that may change the read consistency of
		the query.
****************************************************************************/
RCODE FSIndexCursor::resetTransaction( 
	F_Db *		pDb)
{
	RCODE			rc = NE_SFLM_OK;
	LFILE *		pLFile;
	F_INDEX *	pIndex;

	if ((pIndex = pDb->m_pDict->getIndex( m_uiIndexNum)) == NULL)
	{
		rc = RC_SET( NE_SFLM_INVALID_INDEX_NUM);
		goto Exit;
	}
	if (pIndex->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED))
	{
		rc = RC_SET( NE_SFLM_INDEX_OFFLINE);
		goto Exit;
	}
	pLFile = &pIndex->lfInfo;
	if (m_pDb != pDb || pLFile != m_pLFile || pIndex != m_pIndex)
	{
		m_pLFile = pLFile;
		m_pIndex = pIndex;
		if (m_bTreeOpen)
		{
			closeBTree();
		}
		m_pDb = pDb;
		m_eTransType = pDb->m_eTransType;
	}
	m_ixCompare.setIxInfo( pDb, m_pIndex);
	m_ui64CurrTransId = pDb->m_ui64CurrTransID;
	m_uiBlkChangeCnt = pDb->m_uiBlkChangeCnt;

Exit:

	return( rc);
}

/****************************************************************************
Desc: Estimate the cost of going between the from and until key and
		fetching every row.
****************************************************************************/
RCODE FSIndexCursor::calculateCost( void)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pUntilBTree = NULL;
	FLMINT		iCompare;

	// Get a btree object

	if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &pUntilBTree)))
	{
		goto Exit;
	}

	if (RC_OK( rc = setKeyPosition( m_pDb, TRUE, FALSE,
				&m_fromExtKey, &m_fromKey, &m_curKey, TRUE, NULL, NULL, NULL)))
	{

		// All keys between FROM and UNTIL may be gone.
		
		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
									&m_untilExtKey, NULL,
									m_curKey.ucKey, m_curKey.uiKeyLen,
									&m_untilExtKey, NULL,
									m_untilKey.ucKey, m_untilKey.uiKeyLen,
									&iCompare)))
		{
			goto Exit;
		}
		if (iCompare <= 0)
		{
			KEYPOS	tmpUntilKey;

			if (RC_OK( rc = pUntilBTree->btOpen( m_pDb, m_pLFile,
									isAbsolutePositionable(), FALSE,
									&m_ixCompare)))
			{

				// Going forward so position is exclusive

				rc = setKeyPosition( m_pDb, FALSE, FALSE,
					&m_untilExtKey, &m_untilKey,
					&tmpUntilKey, FALSE, NULL, pUntilBTree, NULL);
			}
		}
		else
		{
			rc = RC_SET( NE_SFLM_BOF_HIT);
			m_bAtBOF = TRUE;
			m_bAtEOF = FALSE;
		}
	}
	else
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			m_bAtEOF = TRUE;
			m_bAtBOF = FALSE;
		}
		else if (rc == NE_SFLM_BOF_HIT)
		{
			m_bAtEOF = FALSE;
			m_bAtBOF = TRUE;
		}
	}

	if (RC_BAD( rc))
	{

		// Empty tree or empty set case.

		if (rc == NE_SFLM_EOF_HIT || rc == NE_SFLM_BOF_HIT)
		{
			m_ui64Cost = 0;
			m_ui64LeafBlocksBetween = 0;
			m_ui64TotalRefs = 0;
			m_bTotalsEstimated = FALSE;
			rc = NE_SFLM_OK;
		}
		goto Exit;
	}
	else
	{
		if (RC_BAD( rc = m_pbTree->btComputeCounts( pUntilBTree,
									&m_ui64LeafBlocksBetween, &m_ui64TotalRefs,
									&m_bTotalsEstimated,
									(m_pDb->m_pDatabase->m_uiBlockSize * 3) / 4)))
		{
			goto Exit;
		}
		if ((m_ui64Cost = m_ui64LeafBlocksBetween + m_ui64TotalRefs) == 0)
		{
			m_ui64Cost = 1;
		}
	}

Exit:

	if (pUntilBTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pUntilBTree);
	}

	return( rc);
}

/****************************************************************************
Desc: Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the index.
****************************************************************************/
RCODE FSIndexCursor::setupKeys(
	F_Db *			pDb,
	F_INDEX *		pIndex,
	F_TABLE *		pTable,
	SQL_PRED **		ppKeyComponents)
{
	RCODE	rc = NE_SFLM_OK;

	m_uiIndexNum = pIndex->uiIndexNum;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmBuildFromAndUntilKeys( pDb->getDict(), pIndex, pTable,
			ppKeyComponents,
			&m_fromExtKey, m_fromKey.ucKey, &m_fromKey.uiKeyLen, 
			&m_untilExtKey, m_untilKey.ucKey, &m_untilKey.uiKeyLen,
			&m_bDoRowMatch, &m_bCanCompareOnKey)))
	{
		goto Exit;
	}

	m_curKey.uiKeyLen = 0;
	m_bSetup = TRUE;

	if (RC_BAD( rc = calculateCost()))
	{
		goto Exit;
	}
	m_bAtBOF = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set the destination key equal to the source key.
****************************************************************************/
FSTATIC RCODE copyKey(
	KEYPOS *			pDestKey,
	F_DataVector *	pDestSrchKey,
	KEYPOS *			pSrcKey,
	F_DataVector *	pSrcSrchKey)
{
	RCODE	rc = NE_SFLM_OK;
	
	// Copy the key buffer.
	
	if ((pDestKey->uiKeyLen = pSrcKey->uiKeyLen) != 0)
	{
		f_memcpy( pDestKey->ucKey, pSrcKey->ucKey, pSrcKey->uiKeyLen);
	}
	
	// Copy the data vector.

	if (RC_BAD( rc = pDestSrchKey->copyVector( pSrcSrchKey)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Union the keys from another index cursor into this one.
****************************************************************************/
RCODE FSIndexCursor::unionKeys(
	F_Db *				pDb,
	FSIndexCursor *	pFSIndexCursor,
	FLMBOOL *			pbUnioned,
	FLMINT *				piCompare)
{
	RCODE		rc = NE_SFLM_OK;
	FLMINT	iCompare;
	FLMBOOL	bIncomingLessThan = FALSE;
	FLMBOOL	bIncomingGreaterThan = FALSE;
	
	*pbUnioned = FALSE;
	
	// Make sure both cursors are referencing the same index.  If not,
	// they cannot be unioned.
	
	if (pFSIndexCursor->m_uiIndexNum != m_uiIndexNum)
	{
		goto Exit;	// Will return NE_SFLM_OK
	}
	
	// Both cursors better already be setup
	
	flmAssert( m_bSetup && pFSIndexCursor->m_bSetup);
	
	// Set both cursors to have their m_pDb, m_pIndex, etc. set up.
	// These are needed for comparison of keys.
	
	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pFSIndexCursor->checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	// Compare the two from keys.
	
	if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
								&pFSIndexCursor->m_fromExtKey, NULL,
								pFSIndexCursor->m_fromKey.ucKey,
								pFSIndexCursor->m_fromKey.uiKeyLen,
								&m_fromExtKey, NULL,
								m_fromKey.ucKey,
								m_fromKey.uiKeyLen,
								&iCompare)))
	{
		goto Exit;
	}
	
	if (iCompare < 0)
	{
		
		// If the incoming from key is less than our from key,
		// see if the incoming until key is >= our from key.
		// If so, they overlap.
		
		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
									&pFSIndexCursor->m_untilExtKey, NULL,
									pFSIndexCursor->m_untilKey.ucKey,
									pFSIndexCursor->m_untilKey.uiKeyLen,
									&m_fromExtKey, NULL,
									m_fromKey.ucKey,
									m_fromKey.uiKeyLen,
									&iCompare)))
		{
			goto Exit;
		}
		
		if (iCompare < 0)
		{
			
			// Incoming until key is less than our from key - no union.
			
			*piCompare = -1;
			goto Exit;
		}
		
		// Incoming from key is < our from key, and incoming until key
		// is >= our from key - definitely an overlap, and we need to
		// set our from key equal to the incoming from key.
			
		*pbUnioned = TRUE;
		if (RC_BAD( rc = copyKey( &m_fromKey, &m_fromExtKey,
										  &pFSIndexCursor->m_fromKey,
										  &pFSIndexCursor->m_fromExtKey)))
		{
			goto Exit;
		}
		bIncomingLessThan = TRUE;
		
		// Setup so that we will have to reposition on a call to
		// firstKey or nextKey.
		
		m_curKey.uiKeyLen = 0;
		m_bAtBOF = TRUE;
		
		// If the incoming until key was > our from key, see if it is
		// also greater than our  until key.  If so, our until key will
		// be set to be equal to the incoming until key.
		
		if (iCompare > 0)
		{
			goto Compare_Until_Keys;
		}
	}
	else
	{
		
		if (iCompare > 0)
		{
			// The incoming from key is > than our from key,
			// see if the incoming from key is > our until key there
			// is no overlap.
			
			if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
										&pFSIndexCursor->m_fromExtKey, NULL,
										pFSIndexCursor->m_fromKey.ucKey,
										pFSIndexCursor->m_fromKey.uiKeyLen,
										&m_untilExtKey, NULL,
										m_untilKey.ucKey,
										m_untilKey.uiKeyLen,
										&iCompare)))
			{
				goto Exit;
			}
			
			if (iCompare > 0)
			{
				
				// Incoming from key is also greater than our until key - no overlap.
				
				*piCompare = 1;
				goto Exit;
			}
		}
		
		// From key is between our from and until key - definitely a union case.
		// Just need to see if the incoming until key is > our until key.
		// If so, set our until key to be equal to the incoming until key.
		
		*pbUnioned = TRUE;
		
Compare_Until_Keys:

		// Compare the until keys.
		
		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
									&pFSIndexCursor->m_untilExtKey, NULL,
									pFSIndexCursor->m_untilKey.ucKey,
									pFSIndexCursor->m_untilKey.uiKeyLen,
									&m_untilExtKey, NULL,
									m_untilKey.ucKey,
									m_untilKey.uiKeyLen,
									&iCompare)))
		{
			goto Exit;
		}
		if (iCompare > 0)
		{
		
			// Incoming until key is > our until key, set our until key to be
			// equal to the incoming until key.
		
			if (RC_BAD( rc = copyKey( &m_untilKey, &m_untilExtKey,
											  &pFSIndexCursor->m_untilKey,
											  &pFSIndexCursor->m_untilExtKey)))
			{
				goto Exit;
			}
			bIncomingGreaterThan = TRUE;
		}
		// else iCompare <= 0
		// Incoming until key is <= our until key - no need to change our
		// until key.
		
	}
	
	// If we need to recalculate the cost, do so here.
	
	if (bIncomingLessThan && bIncomingGreaterThan)
	{
		
		// Our range was a complete subset of the incoming range, so
		// we should be able to just use whatever cost was calculated
		// for it, unless no cost was calculated, in which case we
		// need to calculate a cost.
		
		if (pFSIndexCursor->m_ui64Cost)
		{
			m_ui64Cost = pFSIndexCursor->m_ui64Cost;
			m_ui64LeafBlocksBetween = pFSIndexCursor->m_ui64LeafBlocksBetween;
			m_ui64TotalRefs = pFSIndexCursor->m_ui64TotalRefs;
			m_bTotalsEstimated = pFSIndexCursor->m_bTotalsEstimated;
		}
		else
		{
			if (RC_BAD( rc = calculateCost()))
			{
				goto Exit;
			}
		}
	}
	
	// If we have not calculated a cost, or the union created a larger
	// range of keys on the from or until side, then we need to
	// recalculate the cost.
	
	else if (bIncomingLessThan || bIncomingGreaterThan || !m_ui64Cost)
	{
		if (RC_BAD( rc = calculateCost()))
		{
			goto Exit;
		}
	}
	
	// If we unioned the keys, set the do row match and can compare on key
	// flags to the worst case.  If the incoming cursor required us to
	// match on the row, then we need to set this cursor to match on the row.
	// If the incoming cursor did not allow us to compare on the key, then
	// we don't want to allow this cursor to compare on the key.
	
	if (*pbUnioned)
	{
		if (pFSIndexCursor->m_bDoRowMatch)
		{
			m_bDoRowMatch = TRUE;
		}
		if (!pFSIndexCursor->m_bCanCompareOnKey)
		{
			m_bCanCompareOnKey = FALSE;
		}
		
	}
							
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Open the F_Btree object if not already open.
****************************************************************************/
RCODE FSIndexCursor::openBTree(
	F_Db *	pDb
	)
{
	RCODE	rc = NE_SFLM_OK;

	if (!m_bTreeOpen)
	{
		if ( !m_pbTree)
		{
			if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &m_pbTree)))
			{
				goto Exit;
			}
		}
Open_Btree:
		if (RC_BAD( rc = m_pbTree->btOpen( pDb, m_pLFile,
								isAbsolutePositionable(), FALSE,
								&m_ixCompare)))
		{
			goto Exit;
		}
		m_bTreeOpen = TRUE;
		m_pDb = pDb;
		m_eTransType = pDb->m_eTransType;
		m_ixCompare.setIxInfo( m_pDb, m_pIndex);
	}
	else
	{
		if (pDb != m_pDb || pDb->m_eTransType != m_eTransType)
		{
			closeBTree();
			goto Open_Btree;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get a key's data part.
****************************************************************************/
RCODE FSIndexCursor::getKeyData(
	F_Btree *	pBTree,
	FLMUINT		uiDataLen)
{
	RCODE	rc = NE_SFLM_OK;

	m_uiCurKeyDataLen = 0;

	// See if there is a data part

	if (m_pIndex->pDataIcds && uiDataLen)
	{

		// If the data will fit in the search key buffer, just
		// reuse it since we are not going to do anything with
		// it after this.  Otherwise, allocate a new buffer.

		if (uiDataLen > m_uiCurKeyDataBufSize)
		{
			FLMBYTE *	pucNewBuf;
			FLMUINT		uiNewLen = uiDataLen;

			if (uiNewLen < 256)
			{
				uiNewLen = 256;
			}

			if (RC_BAD( rc = f_alloc( uiNewLen, &pucNewBuf)))
			{
				goto Exit;
			}
			if (m_pucCurKeyDataBuf)
			{
				f_free( &m_pucCurKeyDataBuf);
			}
			m_pucCurKeyDataBuf = pucNewBuf;
			m_uiCurKeyDataBufSize = uiNewLen;
		}

		// Retrieve the data

		if (RC_BAD( rc = pBTree->btGetEntry(
			m_curKey.ucKey, SFLM_MAX_KEY_SIZE, m_curKey.uiKeyLen,
			m_pucCurKeyDataBuf, m_uiCurKeyDataBufSize,
			&m_uiCurKeyDataLen)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Set the key position given some KEYPOS structure.
		Please note that the blocks in the stack may or may not be used.
****************************************************************************/
RCODE FSIndexCursor::setKeyPosition(
	F_Db *			pDb,
	FLMBOOL			bGoingForward,
	FLMBOOL			bExcludeKey,
	F_DataVector *	pExtSrchKey,
	KEYPOS *			pSearchKey,		// Search key
	KEYPOS *			pFoundKey,
	FLMBOOL			bGetKeyData,
	FLMUINT *		puiDataLen,
	F_Btree *		pBTree,			// BTree to use.  NULL means use our
											// internal one.
	FLMUINT *		puiAbsolutePos)
{
	RCODE				rc = NE_SFLM_OK;
	FLMUINT			uiDataLen;
	FLMINT			iCompare = 0;

	// if pBTree is NULL, we are to use m_pbTree.  Otherwise, we
	// need to open the pBTree and use it.

	if (!pBTree)
	{
		if (RC_BAD( rc = openBTree( pDb)))
		{
			goto Exit;
		}
		pBTree = m_pbTree;
	}

	if (pFoundKey != pSearchKey)
	{
		f_memcpy( pFoundKey->ucKey, pSearchKey->ucKey,
						pSearchKey->uiKeyLen);
		pFoundKey->uiKeyLen = pSearchKey->uiKeyLen;
	}
	
	m_ixCompare.setSearchKey( pExtSrchKey);
	m_ixCompare.setCompareRowId( pExtSrchKey ? FALSE : TRUE);
	if (RC_BAD( rc = pBTree->btLocateEntry( pFoundKey->ucKey, SFLM_MAX_KEY_SIZE,
										&pFoundKey->uiKeyLen,
										(bGoingForward && bExcludeKey)
										? FLM_EXCL
										: FLM_INCL,
										puiAbsolutePos, &uiDataLen, NULL, NULL)))
	{
		if (rc != NE_SFLM_EOF_HIT)
		{
			goto Exit;
		}
	}

	if (bGoingForward)
	{
		if (rc == NE_SFLM_EOF_HIT)
		{
			goto Exit;
		}
	}
	else
	{

		// Going backwards or to last.  See if we positioned too far.

		if (rc == NE_SFLM_EOF_HIT)
		{

			// Position to last key in tree.

			if (RC_BAD( rc = pBTree->btLastEntry( pFoundKey->ucKey, SFLM_MAX_KEY_SIZE,
														&pFoundKey->uiKeyLen,
														&uiDataLen, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else
		{

			// We want to go to the previous key if we went past the key
			// we were aiming for, or if we landed on that key, but we
			// are doing an exclusive lookup.

			if (!bExcludeKey)
			{
				if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex,
											pExtSrchKey ? FALSE : TRUE,
											pExtSrchKey, NULL,
											pFoundKey->ucKey, pFoundKey->uiKeyLen,
											pExtSrchKey, NULL,
											pSearchKey->ucKey, pSearchKey->uiKeyLen,
											&iCompare)))
				{
					goto Exit;
				}
			}
	
			if (bExcludeKey || iCompare > 0)
			{

				// Position to the previous key.

				if (RC_BAD( rc = pBTree->btPrevEntry( pFoundKey->ucKey, SFLM_MAX_KEY_SIZE,
																	&pFoundKey->uiKeyLen,
																	&uiDataLen, NULL, NULL)))
				{
					goto Exit;
				}
			}
		}
	}

	// See if there is any data to get

	if (bGetKeyData)
	{
		flmAssert( pFoundKey == &m_curKey);

		if (RC_BAD( rc = getKeyData( pBTree, uiDataLen)))
		{
			goto Exit;
		}
	}
	if (puiDataLen)
	{
		*puiDataLen = uiDataLen;
	}

Exit:

	if (RC_BAD( rc))
	{
		pFoundKey->uiKeyLen = 0;
		if (pBTree == m_pbTree)
		{
			closeBTree();
		}
	}
	return( rc);
}

/****************************************************************************
Desc:	Populate a key from the current key.
****************************************************************************/
RCODE FSIndexCursor::populateKey(
	F_DataVector *	pKey)
{
	RCODE	rc = NE_SFLM_OK;

	pKey->reset();
	if (RC_BAD( rc = pKey->inputKey( m_pDb, m_uiIndexNum,
											m_curKey.ucKey, m_curKey.uiKeyLen)))
	{
		goto Exit;
	}

	// See if there is a data part

	if (m_pIndex->pDataIcds && m_uiCurKeyDataLen)
	{

		// Parse the data

		if (RC_BAD( rc = pKey->inputData( m_pDb, m_uiIndexNum, m_pucCurKeyDataBuf,
										m_uiCurKeyDataLen)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Return the current record and record id.
****************************************************************************/
RCODE FSIndexCursor::currentKey(
	F_Db *			pDb,
	F_DataVector *	pKey)
{
	RCODE		rc = NE_SFLM_OK;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	if (m_bAtBOF)
	{
		rc = RC_SET( NE_SFLM_BOF_HIT);
		goto Exit;
	}
	if (m_bAtEOF)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}

	// If we get this far, we are positioned on some key.

	flmAssert( m_curKey.uiKeyLen);

	if (RC_BAD( rc = populateKey( pKey)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Make sure the current key is positioned in a key set.  If not,
		move to the next or previous key set until it is.
****************************************************************************/
RCODE FSIndexCursor::checkIfKeyInRange(
	FLMBOOL	bPositionForward)
{
	RCODE		rc = NE_SFLM_OK;
	FLMINT	iCompare;

	if (bPositionForward)
	{

		// See if the key is within the bounds of the current key set.
		// It is already guaranteed to be >= the fromKey, we just need to
		// make sure it is <= the until key.

		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
								&m_untilExtKey, NULL,
								m_curKey.ucKey, m_curKey.uiKeyLen,
								&m_untilExtKey, NULL,
								m_untilKey.ucKey, m_untilKey.uiKeyLen,
								&iCompare)))
		{
			goto Exit;
		}
	
		if (iCompare > 0)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( NE_SFLM_EOF_HIT);
			goto Exit;
		}
	}
	else
	{

		// See if the key is within the bounds of the current key set.
		// It is already guaranteed to be <= the untilKey, we just need to
		// make sure it is >= the from key.

		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIndex, FALSE,
								&m_fromExtKey, NULL,
								m_curKey.ucKey, m_curKey.uiKeyLen,
								&m_fromExtKey, NULL,
								m_fromKey.ucKey, m_fromKey.uiKeyLen,
								&iCompare)))
		{
			goto Exit;
		}
	
		if (iCompare < 0)
		{
			m_bAtBOF = TRUE;
			rc = RC_SET( NE_SFLM_BOF_HIT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Position to and return the first key.
****************************************************************************/
RCODE FSIndexCursor::firstKey(
	F_Db *			pDb,
	F_DataVector *	pKey)
{
	RCODE		rc = NE_SFLM_OK;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// If at BOF and we have a key, then we are positioned on the first
	// key already - this would have happened if we had positioned to
	// calculate a cost.  Rather than do the positioning again, we simply
	// set m_bAtBOF to FALSE.

	if (m_bAtBOF && m_curKey.uiKeyLen)
	{
		m_bAtBOF = FALSE;
	}
	else
	{
		m_bAtBOF = m_bAtEOF = FALSE;
		if (RC_BAD( rc = setKeyPosition( pDb, TRUE, FALSE,
				&m_fromExtKey, &m_fromKey, 
				&m_curKey, TRUE, NULL, NULL, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			goto Exit;
		}
	}

	// Make sure the current key is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfKeyInRange( TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = populateKey( pKey)))
	{
		goto Exit;
	}


Exit:

	if (RC_BAD( rc))
	{
		m_curKey.uiKeyLen = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the next key.
****************************************************************************/
RCODE FSIndexCursor::nextKey(
	F_Db *			pDb,
	F_DataVector *	pKey,
	FLMBOOL			bSkipCurrKey)
{
	RCODE				rc = NE_SFLM_OK;
	KEYPOS			saveCurrentKey;
	FLMUINT			uiDataLen;
	FLMINT			iCompare;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	flmAssert( m_bSetup);
	if (m_bAtEOF)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		goto Exit;
	}
	if (m_bAtBOF || !m_curKey.uiKeyLen)
	{
		rc = firstKey( pDb, pKey);
		goto Exit;
	}

	// Save the current key, so we can tell when we have gone beyond it.

	if (bSkipCurrKey)
	{
		getCurrKey( &saveCurrentKey);
	}
	
	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}
	for (;;)
	{

		// checkTransaction may have closed the B-tree, in which case we
		// need to reopen the b-tree and position exclusive of the current key. 
	
		if (!m_bTreeOpen)
		{
			if (RC_BAD( rc = setKeyPosition( pDb, TRUE, TRUE,
										NULL, &m_curKey, &m_curKey, FALSE, &uiDataLen,
										NULL, NULL)))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					m_bAtEOF = TRUE;
				}
				goto Exit;
			}
		}
		else
		{

			// set the compare object's search key to NULL because m_curKey
			// should always be something we obtained from the index, and
			// we should not need a search key for doing extended
			// comparisons on truncated keys.
			
			m_ixCompare.setSearchKey( NULL);
			m_ixCompare.setCompareRowId( TRUE);
			
			// Get the next key, if any
	
			if (RC_BAD( rc = m_pbTree->btNextEntry( m_curKey.ucKey, SFLM_MAX_KEY_SIZE,
												&m_curKey.uiKeyLen, &uiDataLen, NULL, NULL)))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					m_bAtEOF = TRUE;
				}
				goto Exit;
			}
		}
		if (!bSkipCurrKey)
		{
Check_Key:
			if (RC_BAD( rc = getKeyData( m_pbTree, uiDataLen)))
			{
				goto Exit;
			}

			// We got to the next key, make sure it is in the key range.

			if (RC_BAD( rc = checkIfKeyInRange( TRUE)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = populateKey( pKey)))
			{
				goto Exit;
			}

			break;
		}

		// If the bSkipCurrKey flag is TRUE, we want to skip keys until
		// we come to one where the key part is different.
		
		if (RC_BAD( rc = ixKeyCompare( pDb, m_pIndex, FALSE,
									NULL, NULL,
									m_curKey.ucKey, m_curKey.uiKeyLen,
									NULL, NULL,
									saveCurrentKey.ucKey, saveCurrentKey.uiKeyLen,
									&iCompare)))
		{
			goto Exit;
		}

		// See if the key part is the same.

		if (iCompare != 0)
		{
			goto Check_Key;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		m_curKey.uiKeyLen = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to and return the last key in the range.
****************************************************************************/
RCODE FSIndexCursor::lastKey(
	F_Db *			pDb,
	F_DataVector *	pKey)
{
	RCODE		rc = NE_SFLM_OK;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// Position to the until key.

	m_bAtBOF = m_bAtEOF = FALSE;
	if (RC_BAD( rc = setKeyPosition( pDb, FALSE, FALSE,
			&m_untilExtKey, &m_untilKey, 
			&m_curKey, TRUE, NULL, NULL, NULL)))
	{
		if (rc == NE_SFLM_BOF_HIT)
		{
			m_bAtBOF = TRUE;
		}
		goto Exit;
	}

	// Make sure the current key is within one of the FROM/UNTIL sets.

	if (RC_BAD( rc = checkIfKeyInRange( FALSE)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = populateKey( pKey)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_curKey.uiKeyLen = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the PREVIOUS key in the range.
****************************************************************************/
RCODE FSIndexCursor::prevKey(
	F_Db *			pDb,
	F_DataVector *	pKey,
	FLMBOOL			bSkipCurrKey)
{
	RCODE				rc = NE_SFLM_OK;
	KEYPOS			saveCurrentKey;
	FLMUINT			uiDataLen;
	FLMINT			iCompare;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	
	flmAssert( m_bSetup);
	
	if (m_bAtBOF)
	{
		rc = RC_SET( NE_SFLM_BOF_HIT);
		goto Exit;
	}
	
	if (m_bAtEOF || !m_curKey.uiKeyLen)
	{
		rc = lastKey( pDb, pKey);
		goto Exit;
	}

	// Save the current key, so we can tell when we have gone beyond it.

	if (bSkipCurrKey)
	{
		getCurrKey( &saveCurrentKey);
	}

	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}
	for (;;)
	{

		// checkTransaction may have closed the B-tree, in which case we
		// need to reopen the b-tree and position exclusive of the current key. 
	
		if (!m_bTreeOpen)
		{
			if (RC_BAD( rc = setKeyPosition( pDb, FALSE, TRUE,
										NULL, &m_curKey, &m_curKey,
										FALSE, &uiDataLen, NULL, NULL)))
			{
				if (rc == NE_SFLM_BOF_HIT)
				{
					m_bAtBOF = TRUE;
				}
				goto Exit;
			}
		}
		else
		{
			
			// set the compare object's search key to NULL because m_curKey
			// should always be something we obtained from the index, and
			// we should not need a search key for doing extended
			// comparisons on truncated keys.
			
			m_ixCompare.setSearchKey( NULL);
			m_ixCompare.setCompareRowId( TRUE);
		
			// Get the previous key, if any
	
			if (RC_BAD( rc = m_pbTree->btPrevEntry( m_curKey.ucKey, SFLM_MAX_KEY_SIZE,
												&m_curKey.uiKeyLen, &uiDataLen, NULL, NULL)))
			{
				if (rc == NE_SFLM_BOF_HIT)
				{
					m_bAtBOF = TRUE;
				}
				goto Exit;
			}
		}
		if (!bSkipCurrKey)
		{
Check_Key:
			if (RC_BAD( rc = getKeyData( m_pbTree, uiDataLen)))
			{
				goto Exit;
			}

			// We got to the previous key, make sure it is in the key range.

			if (RC_BAD( rc = checkIfKeyInRange( FALSE)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = populateKey( pKey)))
			{
				goto Exit;
			}

			break;
		}

		// If the bSkipCurrKey flag is TRUE, we want to skip keys until
		// we come to one where the key part is different.

		if (RC_BAD( rc = ixKeyCompare( pDb, m_pIndex, FALSE,
									NULL, NULL,
									m_curKey.ucKey, m_curKey.uiKeyLen,
									NULL, NULL,
									saveCurrentKey.ucKey, saveCurrentKey.uiKeyLen,
									&iCompare)))
		{
			goto Exit;
		}

		// See if the key part is the same.

		if (iCompare != 0)
		{
			goto Check_Key;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		m_curKey.uiKeyLen = 0;
	}

	return( rc);
}

