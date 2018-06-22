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

/****************************************************************************
Desc:
****************************************************************************/
FSTableCursor::FSTableCursor() 
{
	m_pbTree = NULL;
	m_bTreeOpen = FALSE;
	m_pTable = NULL;
	m_pLFile = NULL;
	m_pDb = NULL;
	m_eTransType = SFLM_NO_TRANS;
	resetCursor();
}

/****************************************************************************
Desc:
****************************************************************************/
FSTableCursor::~FSTableCursor() 
{
	closeBTree();
	if (m_pbTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &m_pbTree);
	}
}

/****************************************************************************
Desc:	Resets any allocations, keys, state, etc.
****************************************************************************/
void FSTableCursor::resetCursor( void)
{
	closeBTree();
	m_uiTableNum = 0;
	m_uiBlkChangeCnt = 0;
	m_ui64CurrTransId = 0;
	m_ui64CurRowId = 0;
	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
	m_bSetup = FALSE;
}

/****************************************************************************
Desc:	Resets to a new transaction that may change the read consistency of
		the query.
****************************************************************************/
RCODE FSTableCursor::resetTransaction( 
	F_Db *	pDb)
{
	RCODE			rc = NE_SFLM_OK;
	F_TABLE *	pTable;

	if ((pTable = pDb->m_pDict->getTable( m_uiTableNum)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_INVALID_TABLE_NUM);
		goto Exit;
	}
	if (pTable != m_pTable)
	{
		m_pTable = pTable;
		m_pLFile = &pTable->lfInfo;
		if (m_bTreeOpen)
		{
			closeBTree();
		}
		m_pDb = pDb;
		m_eTransType = pDb->m_eTransType;
	}
	m_ui64CurrTransId = pDb->m_ui64CurrTransID;
	m_uiBlkChangeCnt = pDb->m_uiBlkChangeCnt;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Open the F_Btree object if not already open.
****************************************************************************/
RCODE FSTableCursor::openBTree(
	F_Db *	pDb
	)
{
	RCODE	rc = NE_SFLM_OK;

	if (!m_bTreeOpen)
	{
		if (!m_pbTree)
		{
			if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree( &m_pbTree)))
			{
				goto Exit;
			}
		}
Open_Btree:
		if (RC_BAD( rc = m_pbTree->btOpen( pDb, m_pLFile, FALSE, FALSE)))
		{
			goto Exit;
		}
		m_bTreeOpen = TRUE;
		m_pDb = pDb;
		m_eTransType = pDb->m_eTransType;
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
Desc:	Set the row position.
****************************************************************************/
RCODE FSTableCursor::setRowPosition(
	F_Db *			pDb,
	FLMBOOL			bGoingForward,
	FLMUINT64		ui64RowId,
	FLMBOOL			bPopulateCurRowId,
	F_Btree *		pBTree			// BTree to use.  NULL means use our
											// internal one.
	)
{
	RCODE				rc = NE_SFLM_OK;
	FLMBOOL			bNeg;
	FLMBYTE			ucRowKey [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT			uiRowKeyLen;
	FLMBYTE *		pucRowKey;
	FLMUINT *		puiRowKeyLen;
	FLMUINT			uiKeyBufSize;
	FLMUINT			uiBytesProcessed;
	FLMUINT64		ui64TmpRowId;

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
	
	if (bPopulateCurRowId)
	{
		pucRowKey = &m_ucCurRowKey [0];
		puiRowKeyLen = &m_uiCurRowKeyLen;
		uiKeyBufSize = sizeof( m_ucCurRowKey);
	}
	else
	{
		pucRowKey = &ucRowKey [0];
		puiRowKeyLen = &uiRowKeyLen;
		uiKeyBufSize = sizeof( ucRowKey);
	}

	*puiRowKeyLen = uiKeyBufSize;
	if (RC_BAD( rc = flmNumber64ToStorage( ui64RowId, puiRowKeyLen,
									pucRowKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pBTree->btLocateEntry( pucRowKey, uiKeyBufSize,
								puiRowKeyLen, FLM_INCL)))
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

		if (rc == NE_SFLM_BOF_HIT || rc == NE_SFLM_EOF_HIT)
		{

			// Position to last key in tree.

			if (RC_BAD( rc = pBTree->btLastEntry( pucRowKey, uiKeyBufSize,
														puiRowKeyLen,
														NULL, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = flmCollation2Number( *puiRowKeyLen, pucRowKey,
										&ui64TmpRowId, &bNeg, &uiBytesProcessed)))
			{
				goto Exit;
			}
			if (ui64TmpRowId > ui64RowId)
			{

				// Position to the previous key.

				if (RC_BAD( rc = pBTree->btPrevEntry( pucRowKey, uiKeyBufSize,
														puiRowKeyLen, NULL, NULL, NULL)))
				{
					goto Exit;
				}
			}
		}
	}

	if (bPopulateCurRowId)
	{
		if (RC_BAD( rc = flmCollation2Number( *puiRowKeyLen, pucRowKey,
									&m_ui64CurRowId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		if (pBTree == m_pbTree)
		{
			closeBTree();
		}
	}
	return( rc);
}

/****************************************************************************
Desc:	Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the table.
****************************************************************************/
RCODE FSTableCursor::setupRange(
	F_Db *			pDb,
	FLMUINT			uiTableNum,
	FLMUINT64		ui64LowRowId,
	FLMUINT64		ui64HighRowId,
	FLMBOOL			bEstimateCost)
{
	RCODE			rc = NE_SFLM_OK;
	F_Btree *	pUntilBTree = NULL;

	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
	m_uiTableNum = uiTableNum;
	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	m_ui64FromRowId = ui64LowRowId;
	m_ui64UntilRowId = ui64HighRowId;
	m_bSetup = TRUE;
	m_ui64CurRowId = 0;

	// Want any of the counts back?

	if (bEstimateCost)
	{

		// Position to the FROM and UNTIL key so we can get the stats.

		if (RC_BAD( rc = setRowPosition( pDb, TRUE,
								m_ui64FromRowId, TRUE, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				m_ui64Cost = 0;
				m_ui64LeafBlocksBetween = 0;
				m_ui64TotalRows = 0;
				m_bTotalsEstimated = FALSE;
				rc = NE_SFLM_OK;
			}
			goto Exit;
		}

		// All nodes between FROM and UNTIL may be gone.

		if (m_ui64CurRowId < m_ui64UntilRowId)
		{

			// Get a btree object

			if (RC_BAD( rc = gv_SFlmSysData.pBtPool->btpReserveBtree(
												&pUntilBTree)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pUntilBTree->btOpen( pDb, m_pLFile,
									FALSE, FALSE)))
			{
				goto Exit;
			}

			// We better be able to at least find m_ui64CurRowId going
			// backward from m_ui64UntilRowId.

			if (RC_BAD( rc = setRowPosition( pDb, FALSE,
							m_ui64UntilRowId, FALSE, pUntilBTree)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = m_pbTree->btComputeCounts( pUntilBTree,
										&m_ui64LeafBlocksBetween, &m_ui64TotalRows,
										&m_bTotalsEstimated,
										(pDb->m_pDatabase->m_uiBlockSize * 3) / 4)))
			{
				goto Exit;
			}
			if ((m_ui64Cost = m_ui64LeafBlocksBetween) == 0)
			{
				m_ui64Cost = 1;
			}
		}
		else
		{
			m_ui64Cost = 0;
			m_ui64LeafBlocksBetween = 0;
			m_ui64TotalRows = 0;
			m_bTotalsEstimated = FALSE;
		}
	}
	else
	{
		m_ui64Cost = 0;
		m_ui64LeafBlocksBetween = 0;
		m_ui64TotalRows = 0;
		m_bTotalsEstimated = FALSE;
	}

Exit:

	if (pUntilBTree)
	{
		gv_SFlmSysData.pBtPool->btpReturnBtree( &pUntilBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Return the current row.
****************************************************************************/
RCODE FSTableCursor::currentRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId
	)
{
	RCODE	rc = NE_SFLM_OK;

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

	flmAssert( m_ui64CurRowId);

	if (RC_BAD( rc = populateRow( pDb, ppRow, pui64RowId)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Fetch the current row.  B-tree object is assumed to be positioned on
		the row.
****************************************************************************/
RCODE FSTableCursor::populateRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId)
{
	if (pui64RowId)
	{
		*pui64RowId = m_ui64CurRowId;
	}
	if (ppRow)
	{
		return( gv_SFlmSysData.pRowCacheMgr->retrieveRow( pDb, m_uiTableNum,
							m_ui64CurRowId, ppRow));
	}
	return( NE_SFLM_OK);
}

/****************************************************************************
Desc:	Make sure the current node is positioned in the range for the cursor.
****************************************************************************/
RCODE FSTableCursor::checkIfRowInRange(
	FLMBOOL	bPositionForward)
{
	RCODE		rc = NE_SFLM_OK;

	if (bPositionForward)
	{
		if (m_ui64CurRowId > m_ui64UntilRowId)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( NE_SFLM_EOF_HIT);
			goto Exit;
		}
	}
	else
	{
		if (m_ui64CurRowId < m_ui64FromRowId)
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
Desc:	Position to and return the first row.
****************************************************************************/
RCODE FSTableCursor::firstRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId)
{
	RCODE		rc = NE_SFLM_OK;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// If at BOF and we have a node, then we are positioned on the first
	// node already - this would have happened if we had positioned to
	// calculate a cost.  Rather than do the positioning again, we simply
	// set m_bAtBOF to FALSE.

	if (m_bAtBOF && m_ui64CurRowId)
	{
		m_bAtBOF = FALSE;
	}
	else
	{
		m_bAtBOF = m_bAtEOF = FALSE;
		if (RC_BAD( rc = setRowPosition( pDb, TRUE, m_ui64FromRowId,
				TRUE, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			goto Exit;
		}
	}

	// Make sure the current row ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfRowInRange( TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateRow( pDb, ppRow, pui64RowId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurRowId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the next row.
****************************************************************************/
RCODE FSTableCursor::nextRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId)
{
	RCODE		rc = NE_SFLM_OK;

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
	if (m_bAtBOF || !m_ui64CurRowId)
	{
		rc = firstRow( pDb, ppRow, pui64RowId);
		goto Exit;
	}

	// Get the next row, if any

	if (m_ui64CurRowId == FLM_MAX_UINT64)
	{
		rc = RC_SET( NE_SFLM_EOF_HIT);
		m_bAtEOF = TRUE;
		goto Exit;
	}
	
	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}
	
	// checkTransaction may have closed the B-tree, in which case we
	// need to reopen the b-tree and position to the next row.

	if (!m_bTreeOpen)
	{
		if (RC_BAD( rc = setRowPosition( pDb, TRUE,
									m_ui64CurRowId + 1, TRUE, NULL)))
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
		FLMBOOL	bNeg;
		FLMUINT	uiBytesProcessed;

		// If we have a b-tree open, it is more efficient to call btNextEntry
		// directly.  This may allow us to avoid repositioning down the b-tree.
		
		if (RC_BAD( rc = m_pbTree->btNextEntry( m_ucCurRowKey,
											sizeof( m_ucCurRowKey),
											&m_uiCurRowKeyLen, NULL, NULL, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			goto Exit;
		}
		if (RC_BAD( rc = flmCollation2Number( m_uiCurRowKeyLen, m_ucCurRowKey,
									&m_ui64CurRowId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}
	}

	// Make sure the current row ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfRowInRange( TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateRow( pDb, ppRow, pui64RowId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurRowId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to and return the last row.
****************************************************************************/
RCODE FSTableCursor::lastRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId)
{
	RCODE		rc = NE_SFLM_OK;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// Position to the until node

	m_bAtBOF = m_bAtEOF = FALSE;
	if (RC_BAD( rc = setRowPosition( pDb, FALSE, m_ui64UntilRowId,
			TRUE, NULL)))
	{
		if (rc == NE_SFLM_BOF_HIT)
		{
			m_bAtBOF = TRUE;
		}
		goto Exit;
	}

	// Make sure the current row ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfRowInRange( FALSE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateRow( pDb, ppRow, pui64RowId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurRowId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the previous row.
****************************************************************************/
RCODE FSTableCursor::prevRow(
	F_Db *		pDb,
	F_Row **		ppRow,
	FLMUINT64 *	pui64RowId)
{
	RCODE	rc = NE_SFLM_OK;

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
	if (m_bAtEOF || !m_ui64CurRowId)
	{
		rc = lastRow( pDb, ppRow, pui64RowId);
		goto Exit;
	}

	// Get the previous row, if any

	if (m_ui64CurRowId == 1)
	{
		rc = RC_SET( NE_SFLM_BOF_HIT);
		m_bAtBOF = TRUE;
		goto Exit;
	}
	
	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}

	// checkTransaction may have closed the B-tree, in which case we
	// need to reopen the b-tree and position to the previous row.

	if (!m_bTreeOpen)
	{
		if (RC_BAD( rc = setRowPosition( pDb, TRUE,
									m_ui64CurRowId - 1, TRUE, NULL)))
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
		FLMBOOL	bNeg;
		FLMUINT	uiBytesProcessed;

		// If we have a b-tree open, it is more efficient to call btPrevEntry
		// directly.  This may allow us to avoid repositioning down the b-tree.
		
		if (RC_BAD( rc = m_pbTree->btPrevEntry( m_ucCurRowKey,
											sizeof( m_ucCurRowKey),
											&m_uiCurRowKeyLen, NULL, NULL, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			goto Exit;
		}
		if (RC_BAD( rc = flmCollation2Number( m_uiCurRowKeyLen, m_ucCurRowKey,
									&m_ui64CurRowId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}
	}

	// Make sure the current row ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfRowInRange( FALSE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateRow( pDb, ppRow, pui64RowId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurRowId = 0;
	}
	return( rc);
}

