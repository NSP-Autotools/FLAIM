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

FSTATIC int nodeIdCompFn(
	void *	pvData1,
	void *	pvData2,
	void *	pvUserData);

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
	m_pIxd = NULL;
	m_pLFile = NULL;
	m_pDb = NULL;
	m_eTransType = XFLM_NO_TRANS;
	m_pNodeIdSet = NULL;
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
		gv_XFlmSysData.pBtPool->btpReturnBtree( &m_pbTree);
	}
	if (m_pNodeIdSet)
	{
		m_pNodeIdSet->Release();
		m_pNodeIdSet = NULL;
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
	if (m_pNodeIdSet)
	{
		m_pNodeIdSet->Release();
		m_pNodeIdSet = NULL;
	}
	m_bElimDups = FALSE;
	m_bMovingForward = TRUE;
}

/****************************************************************************
Desc:	Comparison function for comparing node ids.
****************************************************************************/
FSTATIC int nodeIdCompFn(
	void *	pvData1,
	void *	pvData2,
	void *	// pvUserData
	)
{
	if (*((FLMUINT64 *)pvData1) < *((FLMUINT64 *)pvData2))
	{
		return( -1);
	}
	else if (*((FLMUINT64 *)pvData1) > *((FLMUINT64 *)pvData2))
	{
		return( 1);
	}
	else
	{
		return( 0);
	}
}

/****************************************************************************
Desc:	Allocate a result set for duplicate checking.
****************************************************************************/
RCODE FSIndexCursor::allocDupCheckSet( void)
{
	RCODE			rc = NE_XFLM_OK;
	char			szTmpDir [F_PATH_MAX_SIZE];

	// If it is not a compound index, no need for a result set.

	if (!m_bElimDups)
	{
		goto Exit;
	}
	if (m_pNodeIdSet)
	{
		m_pNodeIdSet->Release();
		m_pNodeIdSet = NULL;
	}
	if ((m_pNodeIdSet = f_new F_DynSearchSet) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = gv_pXFlmDbSystem->getTempDir( szTmpDir)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if (!szTmpDir [0] && m_pDb)
	{
		if (RC_BAD( rc = gv_XFlmSysData.pFileSystem->pathReduce( 
					m_pDb->m_pDatabase->m_pszDbPath, szTmpDir, NULL)))
		{
			goto Exit;
		}
	}

	if (RC_BAD( rc = m_pNodeIdSet->setup( szTmpDir, sizeof( FLMUINT64))))
	{
		goto Exit;
	}

	m_pNodeIdSet->setCompareFunc( nodeIdCompFn, NULL);

Exit:

	if (RC_BAD( rc))
	{
		if (m_pNodeIdSet)
		{
			m_pNodeIdSet->Release();
			m_pNodeIdSet = NULL;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Check to see if we have already returned the node.
****************************************************************************/
RCODE FSIndexCursor::checkIfDup(
	FLMUINT64	ui64NodeId,
	FLMBOOL *	pbDup
	)
{
	RCODE	rc = NE_XFLM_OK;

	*pbDup = FALSE;

	// Should only be called if m_bElimDups is TRUE

	flmAssert( m_bElimDups);

	// If we have not yet allocated the result set, do it now.

	if (!m_pNodeIdSet)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
	}

	// See if we can add the node id to the result set

	if (RC_BAD( rc = m_pNodeIdSet->addEntry( &ui64NodeId)))
	{
		if (rc == NE_FLM_EXISTS)
		{
			*pbDup = TRUE;
			rc = NE_XFLM_OK;
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Resets to a new transaction that may change the read consistency of
		the query.
****************************************************************************/
RCODE FSIndexCursor::resetTransaction( 
	F_Db *		pDb)
{
	RCODE		rc = NE_XFLM_OK;
	LFILE *	pLFile;
	IXD *		pIxd;

	if (RC_BAD( rc = pDb->m_pDict->getIndex( m_uiIndexNum, 
			&pLFile, &pIxd)))
	{	
		goto Exit;
	}
	if (m_pDb != pDb || pLFile != m_pLFile || pIxd != m_pIxd)
	{
		m_pLFile = pLFile;
		m_pIxd = pIxd;
		if (m_bTreeOpen)
		{
			closeBTree();
		}
		m_pDb = pDb;
		m_eTransType = pDb->m_eTransType;
	}
	m_ixCompare.setIxInfo( pDb, m_pIxd);
	m_ui64CurrTransId = pDb->m_ui64CurrTransID;
	m_uiBlkChangeCnt = pDb->m_uiBlkChangeCnt;

Exit:

	return( rc);
}

/****************************************************************************
Desc: Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the index.
****************************************************************************/
RCODE FSIndexCursor::setupKeys(
	F_Db *			pDb,
	IXD *				pIxd,
	PATH_PRED *		pPred,
	FLMBOOL *		pbDoNodeMatch,
	FLMBOOL *		pbCanCompareOnKey,
	FLMUINT *		puiLeafBlocksBetween,// [out] blocks between the stacks
	FLMUINT *		puiTotalRefs,			// [out] total references
	FLMBOOL *		pbTotalsEstimated)	// [out] set to TRUE when estimating.
{
	RCODE				rc = NE_XFLM_OK;
	F_Btree *		pUntilBTree = NULL;
	FLMINT			iCompare;

	m_uiIndexNum = pIxd->uiIndexNum;

	// Need to eliminate dups if we are working with
	// a compound index or an index that has one or more
	// data components along with the key.  This is because
	// the query code only looks at the 0th key component
	// of keys that are returned.  So when we have compound
	// keys or keys with data components, we could get back
	// multiple keys from the index with the same 0th
	// component (but different 1st, 2nd, etc. components).

	m_bElimDups = (pIxd->uiNumKeyComponents > 1 ||
						  pIxd->uiNumDataComponents)
						  ? TRUE
						  : FALSE;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmBuildFromAndUntilKeys( pIxd, pPred,
			&m_fromExtKey, m_fromKey.ucKey, &m_fromKey.uiKeyLen, 
			&m_untilExtKey, m_untilKey.ucKey, &m_untilKey.uiKeyLen,
			pbDoNodeMatch, pbCanCompareOnKey)))
	{
		goto Exit;
	}

	m_curKey.uiKeyLen = 0;
	m_bSetup = TRUE;

	// Want any of the counts back?

	if (puiLeafBlocksBetween || puiTotalRefs)
	{

		// Get a btree object

		if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pUntilBTree)))
		{
			goto Exit;
		}

		if (RC_OK( rc = setKeyPosition( pDb, TRUE, FALSE,
					&m_fromExtKey, &m_fromKey, &m_curKey, TRUE, NULL, NULL, NULL)))
		{

			// All keys between FROM and UNTIL may be gone.
			
			if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIxd, &m_untilExtKey,
										NULL, NULL,
										FALSE, FALSE,
										m_curKey.ucKey, m_curKey.uiKeyLen,
										m_untilKey.ucKey, m_untilKey.uiKeyLen,
										&iCompare)))
			{
				goto Exit;
			}
			if (iCompare <= 0)
			{
				KEYPOS	tmpUntilKey;

				if (RC_OK( rc = pUntilBTree->btOpen( pDb, m_pLFile,
										isAbsolutePositionable(), FALSE,
										&m_ixCompare)))
				{

					// Going forward so position is exclusive

					rc = setKeyPosition( pDb, FALSE, FALSE,
						&m_untilExtKey, &m_untilKey,
						&tmpUntilKey, FALSE, NULL, pUntilBTree, NULL);
				}
			}
			else
			{
				rc = RC_SET( NE_XFLM_BOF_HIT);
				m_bAtBOF = TRUE;
				m_bAtEOF = FALSE;
			}
		}
		else
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
				m_bAtBOF = FALSE;
			}
			else if (rc == NE_XFLM_BOF_HIT)
			{
				m_bAtEOF = FALSE;
				m_bAtBOF = TRUE;
			}
		}

		if (RC_BAD( rc))
		{

			// Empty tree or empty set case.

			if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BOF_HIT)
			{
				if (puiLeafBlocksBetween)
				{
					*puiLeafBlocksBetween = 0;
				}
				if (puiTotalRefs)
				{
					*puiTotalRefs = 0;
				}
				if (pbTotalsEstimated)
				{
					*pbTotalsEstimated = FALSE;
				}
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
		else
		{
			if (RC_BAD( rc = m_pbTree->btComputeCounts( pUntilBTree,
										puiLeafBlocksBetween, puiTotalRefs,
										pbTotalsEstimated,
										(pDb->m_pDatabase->m_uiBlockSize * 3) / 4)))
			{
				goto Exit;
			}
		}
	}
	m_bAtBOF = TRUE;

Exit:

	if (pUntilBTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pUntilBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Open the F_Btree object if not already open.
****************************************************************************/
RCODE FSIndexCursor::openBTree(
	F_Db *	pDb
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (!m_bTreeOpen)
	{
		if ( !m_pbTree)
		{
			if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &m_pbTree)))
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
		m_ixCompare.setIxInfo( m_pDb, m_pIxd);
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
	RCODE	rc = NE_XFLM_OK;

	m_uiCurKeyDataLen = 0;

	// See if there is a data part

	if (m_pIxd->pFirstData && uiDataLen)
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
			m_curKey.ucKey, XFLM_MAX_KEY_SIZE, m_curKey.uiKeyLen,
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
	RCODE				rc = NE_XFLM_OK;
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
	m_ixCompare.setCompareDocId( pExtSrchKey ? FALSE : TRUE);
	m_ixCompare.setCompareNodeIds( pExtSrchKey ? FALSE : TRUE);
	if (RC_BAD( rc = pBTree->btLocateEntry( pFoundKey->ucKey, XFLM_MAX_KEY_SIZE,
										&pFoundKey->uiKeyLen,
										(bGoingForward && bExcludeKey)
										? XFLM_EXCL
										: XFLM_INCL,
										puiAbsolutePos, &uiDataLen, NULL, NULL)))
	{
		if (rc != NE_XFLM_EOF_HIT)
		{
			goto Exit;
		}
	}

	if (bGoingForward)
	{
		if (rc == NE_XFLM_EOF_HIT)
		{
			goto Exit;
		}
	}
	else
	{

		// Going backwards or to last.  See if we positioned too far.

		if (rc == NE_XFLM_EOF_HIT)
		{

			// Position to last key in tree.

			if (RC_BAD( rc = pBTree->btLastEntry( pFoundKey->ucKey, XFLM_MAX_KEY_SIZE,
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
				if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIxd, pExtSrchKey,
											NULL, NULL,
											pExtSrchKey ? FALSE : TRUE,
											pExtSrchKey ? FALSE : TRUE,
											pFoundKey->ucKey, pFoundKey->uiKeyLen,
											pSearchKey->ucKey, pSearchKey->uiKeyLen,
											&iCompare)))
				{
					goto Exit;
				}
			}
	
			if (bExcludeKey || iCompare > 0)
			{

				// Position to the previous key.

				if (RC_BAD( rc = pBTree->btPrevEntry( pFoundKey->ucKey, XFLM_MAX_KEY_SIZE,
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
	F_DataVector *	pKey
	)
{
	RCODE	rc = NE_XFLM_OK;

	pKey->reset();
	if (RC_BAD( rc = pKey->inputKey( m_pIxd, m_curKey.ucKey, m_curKey.uiKeyLen)))
	{
		goto Exit;
	}

	// See if there is a data part

	if (m_pIxd->pFirstData && m_uiCurKeyDataLen)
	{

		// Parse the data

		if (RC_BAD( rc = pKey->inputData( m_pIxd, m_pucCurKeyDataBuf,
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
	RCODE		rc = NE_XFLM_OK;

	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	if (m_bAtBOF)
	{
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
	if (m_bAtEOF)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
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
	RCODE		rc = NE_XFLM_OK;
	FLMINT	iCompare;

	if (bPositionForward)
	{

		// See if the key is within the bounds of the current key set.
		// It is already guaranteed to be >= the fromKey, we just need to
		// make sure it is <= the until key.

		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIxd, &m_untilExtKey,
								NULL, NULL,
								FALSE, FALSE,
								m_curKey.ucKey, m_curKey.uiKeyLen,
								m_untilKey.ucKey, m_untilKey.uiKeyLen,
								&iCompare)))
		{
			goto Exit;
		}
	
		if (iCompare > 0)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}
	}
	else
	{

		// See if the key is within the bounds of the current key set.
		// It is already guaranteed to be <= the untilKey, we just need to
		// make sure it is >= the from key.

		if (RC_BAD( rc = ixKeyCompare( m_pDb, m_pIxd, &m_fromExtKey,
								NULL, NULL,
								FALSE, FALSE,
								m_curKey.ucKey, m_curKey.uiKeyLen,
								m_fromKey.ucKey, m_fromKey.uiKeyLen,
								&iCompare)))
		{
			goto Exit;
		}
	
		if (iCompare < 0)
		{
			m_bAtBOF = TRUE;
			rc = RC_SET( NE_XFLM_BOF_HIT);
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
	RCODE		rc = NE_XFLM_OK;

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
			if (rc == NE_XFLM_EOF_HIT)
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

	// On firstKey, clear out the duplicate elimination result
	// set, if there is one.

	if (m_bElimDups)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = populateKey( pKey)))
	{
		goto Exit;
	}

	if (m_bElimDups)
	{
		FLMUINT64	ui64NodeId = pKey->getID( 0);

		if (RC_BAD( rc = m_pNodeIdSet->addEntry( &ui64NodeId)))
		{
			flmAssert( rc != NE_FLM_EXISTS);
			goto Exit;
		}
	}
	m_bMovingForward = TRUE;

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
	RCODE				rc = NE_XFLM_OK;
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
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}
	if (m_bAtBOF || !m_curKey.uiKeyLen)
	{
		rc = firstKey( pDb, pKey);
		goto Exit;
	}

	// On change of direction, clear out the duplicate elimination result
	// set, if there is one.

	if (m_bElimDups && !m_bMovingForward)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
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
				if (rc == NE_XFLM_EOF_HIT)
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
			m_ixCompare.setCompareNodeIds( TRUE);
			m_ixCompare.setCompareDocId( TRUE);
			
			// Get the next key, if any
	
			if (RC_BAD( rc = m_pbTree->btNextEntry( m_curKey.ucKey, XFLM_MAX_KEY_SIZE,
												&m_curKey.uiKeyLen, &uiDataLen, NULL, NULL)))
			{
				if (rc == NE_XFLM_EOF_HIT)
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

			if (m_bElimDups)
			{
				FLMBOOL	bDup;

				if (RC_BAD( rc = checkIfDup( pKey->getID( 0), &bDup)))
				{
					goto Exit;
				}
				if (bDup)
				{
					continue;
				}
			}
			break;
		}

		// If the bSkipCurrKey flag is TRUE, we want to skip keys until
		// we come to one where the key part is different.
		
		if (RC_BAD( rc = ixKeyCompare( pDb, m_pIxd, NULL,
									NULL, NULL,
									FALSE, FALSE,
									m_curKey.ucKey, m_curKey.uiKeyLen,
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

	m_bMovingForward = TRUE;

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
	RCODE		rc = NE_XFLM_OK;

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
		if (rc == NE_XFLM_BOF_HIT)
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
	
	// On firstKey, clear out the duplicate elimination result
	// set, if there is one.

	if (m_bElimDups)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = populateKey( pKey)))
	{
		goto Exit;
	}

	if (m_bElimDups)
	{
		FLMUINT64	ui64NodeId = pKey->getID( 0);

		if (RC_BAD( rc = m_pNodeIdSet->addEntry( &ui64NodeId)))
		{
			flmAssert( rc != NE_FLM_EXISTS);
			goto Exit;
		}
	}
	m_bMovingForward = FALSE;

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
	RCODE				rc = NE_XFLM_OK;
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
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}
	
	if (m_bAtEOF || !m_curKey.uiKeyLen)
	{
		rc = lastKey( pDb, pKey);
		goto Exit;
	}

	// On change of direction, clear out the duplicate elimination result
	// set, if there is one.

	if (m_bElimDups && m_bMovingForward)
	{
		if (RC_BAD( rc = allocDupCheckSet()))
		{
			goto Exit;
		}
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
				if (rc == NE_XFLM_BOF_HIT)
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
			m_ixCompare.setCompareNodeIds( TRUE);
			m_ixCompare.setCompareDocId( TRUE);
		
			// Get the previous key, if any
	
			if (RC_BAD( rc = m_pbTree->btPrevEntry( m_curKey.ucKey, XFLM_MAX_KEY_SIZE,
												&m_curKey.uiKeyLen, &uiDataLen, NULL, NULL)))
			{
				if (rc == NE_XFLM_BOF_HIT)
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

			if (m_bElimDups)
			{
				FLMBOOL	bDup;

				if (RC_BAD( rc = checkIfDup( pKey->getID( 0), &bDup)))
				{
					goto Exit;
				}
				if (bDup)
				{
					continue;
				}
			}
			break;
		}

		// If the bSkipCurrKey flag is TRUE, we want to skip keys until
		// we come to one where the key part is different.

		
		if (RC_BAD( rc = ixKeyCompare( pDb, m_pIxd, NULL,
									NULL, NULL,
									FALSE, FALSE,
									m_curKey.ucKey, m_curKey.uiKeyLen,
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

	m_bMovingForward = FALSE;

Exit:

	if (RC_BAD( rc))
	{
		m_curKey.uiKeyLen = 0;
	}

	return( rc);
}

