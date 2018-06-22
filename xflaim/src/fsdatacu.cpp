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
FSCollectionCursor::FSCollectionCursor() 
{
	m_pbTree = NULL;
	m_bTreeOpen = FALSE;
	m_pCollection = NULL;
	m_bDocumentIds = FALSE;
	m_pLFile = NULL;
	m_pDb = NULL;
	m_eTransType = XFLM_NO_TRANS;
	resetCursor();
}

/****************************************************************************
Desc:
****************************************************************************/
FSCollectionCursor::~FSCollectionCursor() 
{
	closeBTree();
	if (m_pbTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &m_pbTree);
	}
}

/****************************************************************************
Desc:	Resets any allocations, keys, state, etc.
****************************************************************************/
void FSCollectionCursor::resetCursor( void)
{
	closeBTree();
	m_uiCollection = 0;
	m_bDocumentIds = FALSE;
	m_uiBlkChangeCnt = 0;
	m_ui64CurrTransId = 0;
	m_ui64CurNodeId = 0;
	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
	m_bSetup = FALSE;
}

/****************************************************************************
Desc:	Resets to a new transaction that may change the read consistency of
		the query.
****************************************************************************/
RCODE FSCollectionCursor::resetTransaction( 
	F_Db *	pDb)
{
	RCODE				rc = NE_XFLM_OK;
	F_COLLECTION *	pCollection;

	if (RC_BAD( rc = pDb->m_pDict->getCollection( m_uiCollection, 
			&pCollection)))
	{	
		goto Exit;
	}
	if (pCollection != m_pCollection)
	{
		m_pCollection = pCollection;
		m_pLFile = &pCollection->lfInfo;
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
RCODE FSCollectionCursor::openBTree(
	F_Db *	pDb
	)
{
	RCODE	rc = NE_XFLM_OK;

	if (!m_bTreeOpen)
	{
		if (!m_pbTree)
		{
			if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &m_pbTree)))
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
Desc:	Set the node position.
****************************************************************************/
RCODE FSCollectionCursor::setNodePosition(
	F_Db *			pDb,
	FLMBOOL			bGoingForward,
	FLMUINT64		ui64NodeId,
	FLMUINT64 *		pui64FoundNodeId,
	F_Btree *		pBTree			// BTree to use.  NULL means use our
											// internal one.
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucKey [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT			uiKeyLen;
	FLMBOOL			bNeg;
	FLMUINT			uiBytesProcessed;
	FLMUINT64		ui64TmpNodeId;
	IF_DOMNode *	pNode = NULL;

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

	uiKeyLen = sizeof( ucKey);
	if (RC_BAD( rc = flmNumber64ToStorage( ui64NodeId, &uiKeyLen,
									ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pBTree->btLocateEntry(
								ucKey, sizeof( ucKey), &uiKeyLen, XFLM_INCL)))
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

		if (rc == NE_XFLM_BOF_HIT || rc == NE_XFLM_EOF_HIT)
		{

			// Position to last key in tree.

			if (RC_BAD( rc = pBTree->btLastEntry( ucKey, sizeof( ucKey),
														&uiKeyLen,
														NULL, NULL, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
										&ui64TmpNodeId, &bNeg, &uiBytesProcessed)))
			{
				goto Exit;
			}
			if (ui64TmpNodeId > ui64NodeId)
			{

				// Position to the previous key.

				if (RC_BAD( rc = pBTree->btPrevEntry( ucKey, sizeof( ucKey),
														&uiKeyLen, NULL, NULL, NULL)))
				{
					goto Exit;
				}
			}
		}
	}

	if (!m_bDocumentIds)
	{
		if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
									pui64FoundNodeId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}
	}
	else
	{

		// Need to position to a document ID if we are looking only for document
		// root nodes.

		for (;;)
		{
			if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
										&ui64TmpNodeId, &bNeg, &uiBytesProcessed)))
			{
				goto Exit;
			}

			// The following code tests to see if we have gone past the
			// from or until node id.  It is an optimization that isn't
			// actually necessary for the code to work properly, because
			// the outside code also makes this check.  This just allows
			// us to quit earlier than we otherwise might while looking
			// for a document root node.

			if (bGoingForward)
			{

				// If we have gone past the until node id, we are done.

				if (ui64TmpNodeId > m_ui64UntilNodeId)
				{
					rc = RC_SET( NE_XFLM_EOF_HIT);
					goto Exit;
				}
			}
			else
			{

				// If we have gone past the from node id, we are done.

				if (ui64TmpNodeId < m_ui64FromNodeId)
				{
					rc = RC_SET( NE_XFLM_BOF_HIT);
					goto Exit;
				}
			}

			if (RC_BAD( rc = pDb->getNode( m_uiCollection, ui64TmpNodeId, &pNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{

					// Better be able to find the node at this point!

					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}

			// If the node is a root node, we have a document we can
			// process.

			if (((F_DOMNode *)pNode)->isRootNode())
			{
				*pui64FoundNodeId = ui64TmpNodeId;
				break;
			}

			// Need to go to the next or previous node.

			if (bGoingForward)
			{
				if (RC_BAD( rc = pBTree->btNextEntry( ucKey, uiKeyLen, &uiKeyLen)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = pBTree->btPrevEntry( ucKey, uiKeyLen, &uiKeyLen)))
				{
					goto Exit;
				}
			}
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
	if (pNode)
	{
		pNode->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Setup the from and until keys in the cursor.  Return counts
		after positioning to the from and until key in the index.
		This code does not work with multiple key sets of FROM/UNTIL keys.
****************************************************************************/
RCODE FSCollectionCursor::setupRange(
	F_Db *		pDb,
	FLMUINT		uiCollection,
	FLMBOOL		bDocumentIds,
	FLMUINT64	ui64LowNodeId,
	FLMUINT64	ui64HighNodeId,
	FLMUINT *	puiLeafBlocksBetween,// [out] blocks between the stacks
	FLMUINT *	puiTotalNodes,			// [out]
	FLMBOOL *	pbTotalsEstimated)	// [out] set to TRUE when estimating.
{
	RCODE			rc = NE_XFLM_OK;
	F_Btree *	pUntilBTree = NULL;

	m_bAtBOF = TRUE;
	m_bAtEOF = FALSE;
	m_uiCollection = uiCollection;
	m_bDocumentIds = bDocumentIds;
	if (RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}

	m_ui64FromNodeId = ui64LowNodeId;
	m_ui64UntilNodeId = ui64HighNodeId;
	m_bSetup = TRUE;
	m_ui64CurNodeId = 0;

	// Want any of the counts back?

	if (puiLeafBlocksBetween || puiTotalNodes)
	{
		if (puiLeafBlocksBetween)
		{
			*puiLeafBlocksBetween = 0;
		}
		if (puiTotalNodes)
		{
			*puiTotalNodes = 0;
		}
		if (pbTotalsEstimated)
		{
			*pbTotalsEstimated = FALSE;
		}

		// Position to the FROM and UNTIL key so we can get the stats.

		if (RC_BAD( rc = setNodePosition( pDb, TRUE,
								m_ui64FromNodeId, &m_ui64CurNodeId, NULL)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}

		// All nodes between FROM and UNTIL may be gone.

		if (m_ui64CurNodeId < m_ui64UntilNodeId)
		{
			FLMUINT64	ui64TmpNodeId;

			// Get a btree object

			if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree(
												&pUntilBTree)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = pUntilBTree->btOpen( pDb, m_pLFile,
									FALSE, FALSE)))
			{
				goto Exit;
			}

			// We better be able to at least find m_ui64CurNodeId going
			// backward from m_ui64UntilNodeId.

			if (RC_BAD( rc = setNodePosition( pDb, FALSE,
							m_ui64UntilNodeId, &ui64TmpNodeId, pUntilBTree)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = m_pbTree->btComputeCounts( pUntilBTree,
										puiLeafBlocksBetween, puiTotalNodes,
										pbTotalsEstimated,
										(pDb->m_pDatabase->m_uiBlockSize * 3) / 4)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (pUntilBTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pUntilBTree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Return the current node.
****************************************************************************/
RCODE FSCollectionCursor::currentNode(
	F_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT64 *		pui64NodeId
	)
{
	RCODE	rc = NE_XFLM_OK;

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

	flmAssert( m_ui64CurNodeId);

	if (RC_BAD( rc = populateNode( pDb, ppNode, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Make sure the current node is positioned in the range for the cursor.
****************************************************************************/
RCODE FSCollectionCursor::checkIfNodeInRange(
	FLMBOOL	bPositionForward)
{
	RCODE		rc = NE_XFLM_OK;

	if (bPositionForward)
	{
		if (m_ui64CurNodeId > m_ui64UntilNodeId)
		{
			m_bAtEOF = TRUE;
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}
	}
	else
	{
		if (m_ui64CurNodeId < m_ui64FromNodeId)
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
Desc:	Position to and return the first node.
****************************************************************************/
RCODE FSCollectionCursor::firstNode(
	F_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT64 *		pui64NodeId
	)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// If at BOF and we have a node, then we are positioned on the first
	// node already - this would have happened if we had positioned to
	// calculate a cost.  Rather than do the positioning again, we simply
	// set m_bAtBOF to FALSE.

	if (m_bAtBOF && m_ui64CurNodeId)
	{
		m_bAtBOF = FALSE;
	}
	else
	{
		m_bAtBOF = m_bAtEOF = FALSE;
		if (RC_BAD( rc = setNodePosition( pDb, TRUE, m_ui64FromNodeId,
				&m_ui64CurNodeId, NULL)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				m_bAtEOF = TRUE;
			}
			goto Exit;
		}
	}

	// Make sure the current node ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfNodeInRange( TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateNode( pDb, ppNode, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurNodeId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the next node.
****************************************************************************/
RCODE FSCollectionCursor::nextNode(
	F_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT64 *		pui64NodeId
	)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = NULL;

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
	if (m_bAtBOF || !m_ui64CurNodeId)
	{
		rc = firstNode( pDb, ppNode, pui64NodeId);
		goto Exit;
	}

	if (m_bDocumentIds)
	{
		if (RC_BAD( rc = pDb->getNode( m_uiCollection, m_ui64CurNodeId,
										&pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;

				// Allow to just fall through to below and call setNodePosition.
			}
			else
			{
				goto Exit;
			}
		}
		else if (((F_DOMNode *)pNode)->isRootNode())
		{
			if (RC_BAD( rc = pNode->getNextDocument( pDb, &pNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_EOF_HIT;
					m_bAtEOF = TRUE;
				}
			}
			else
			{
				if( RC_BAD( rc = pNode->getNodeId( m_pDb, &m_ui64CurNodeId)))
				{
					goto Exit;
				}
				
				if (RC_OK( rc = checkIfNodeInRange( TRUE)))
				{
					if (pui64NodeId)
					{
						*pui64NodeId = m_ui64CurNodeId;
					}
					if (ppNode)
					{
						if (*ppNode)
						{
							(*ppNode)->Release();
						}
						*ppNode = pNode;
						pNode = NULL;
					}
				}
			}
			goto Exit;
		}
		// Fall through to below to call setNodePosition.
	}

	// Get the next node, if any

	if (m_ui64CurNodeId == ~((FLMUINT64)0))
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		m_bAtEOF = TRUE;
		goto Exit;
	}
	
	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}

	if (RC_BAD( rc = setNodePosition( pDb, TRUE,
								m_ui64CurNodeId + 1, &m_ui64CurNodeId, NULL)))
	{
		if (rc == NE_XFLM_EOF_HIT)
		{
			m_bAtEOF = TRUE;
		}
		goto Exit;
	}

	// Make sure the current node ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfNodeInRange( TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateNode( pDb, ppNode, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (RC_BAD( rc))
	{
		m_ui64CurNodeId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to and return the last node.
****************************************************************************/
RCODE FSCollectionCursor::lastNode(
	F_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT64 *		pui64NodeId
	)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = checkTransaction( pDb)))
	{
		goto Exit;
	}
	flmAssert( m_bSetup);

	// Position to the until node

	m_bAtBOF = m_bAtEOF = FALSE;
	if (RC_BAD( rc = setNodePosition( pDb, FALSE, m_ui64UntilNodeId,
			&m_ui64CurNodeId, NULL)))
	{
		if (rc == NE_XFLM_BOF_HIT)
		{
			m_bAtBOF = TRUE;
		}
		goto Exit;
	}

	// Make sure the current node ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfNodeInRange( FALSE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateNode( pDb, ppNode, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc))
	{
		m_ui64CurNodeId = 0;
	}
	return( rc);
}

/****************************************************************************
Desc:	Position to the previous node.
****************************************************************************/
RCODE FSCollectionCursor::prevNode(
	F_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT64 *		pui64NodeId
	)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = NULL;

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
	if (m_bAtEOF || !m_ui64CurNodeId)
	{
		rc = lastNode( pDb, ppNode, pui64NodeId);
		goto Exit;
	}

	if (m_bDocumentIds)
	{
		if (RC_BAD( rc = pDb->getNode( m_uiCollection, m_ui64CurNodeId,
										&pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;

				// Allow to just fall through to below and call setNodePosition.
			}
			else
			{
				goto Exit;
			}
		}
		else if (((F_DOMNode *)pNode)->isRootNode())
		{
			if (RC_BAD( rc = pNode->getPreviousDocument( pDb, &pNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_BOF_HIT;
					m_bAtBOF = TRUE;
				}
			}
			else
			{
				if( RC_BAD( rc = pNode->getNodeId( m_pDb, &m_ui64CurNodeId)))
				{
					goto Exit;
				}

				if (RC_OK( rc = checkIfNodeInRange( FALSE)))
				{
					if (pui64NodeId)
					{
						*pui64NodeId = m_ui64CurNodeId;
					}
					if (ppNode)
					{
						if (*ppNode)
						{
							(*ppNode)->Release();
						}
						*ppNode = pNode;
						pNode = NULL;
					}
				}
			}
			goto Exit;
		}
		// Fall through to below to call setNodePosition.
	}

	// Get the previous node, if any

	if (m_ui64CurNodeId == 1)
	{
		rc = RC_SET( NE_XFLM_BOF_HIT);
		m_bAtBOF = TRUE;
		goto Exit;
	}
	
	// See if we need to reset the b-tree object we are using

	if (m_bTreeOpen &&
		 (pDb != m_pDb || pDb->m_eTransType != m_eTransType))
	{
		closeBTree();
	}

	if (RC_BAD( rc = setNodePosition( pDb, FALSE,
								m_ui64CurNodeId - 1, &m_ui64CurNodeId, NULL)))
	{
		if (rc == NE_XFLM_BOF_HIT)
		{
			m_bAtBOF = TRUE;
		}
		goto Exit;
	}

	// Make sure the current node ID is within the FROM/UNTIL range.

	if (RC_BAD( rc = checkIfNodeInRange( FALSE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = populateNode( pDb, ppNode, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (RC_BAD( rc))
	{
		m_ui64CurNodeId = 0;
	}
	return( rc);
}
