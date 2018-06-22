//------------------------------------------------------------------------------
// Desc:	Contains the code to F_Db::sweep method
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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

FSTATIC ELM_ATTR_STATE_INFO * sweepFindState(
	ELM_ATTR_STATE_INFO *	pStateTbl,
	FLMUINT						uiNumItems,
	FLMUINT						uiDictType,
	FLMUINT						uiDictNum,
	FLMUINT *					puiTblSlot);

/****************************************************************************
Desc:	Provides the ability to scan a FLAIM database to delete or check
		for usage of elements and attributes.
****************************************************************************/
RCODE F_Db::sweep(
	IF_Thread *					pThread)
{
	RCODE							rc = NE_XFLM_OK;
	FLMBOOL						bStartedTrans = FALSE;
	ELM_ATTR_STATE_INFO *	pStateTbl = NULL;
	FLMUINT						uiNumItems = 0;
	F_COLLECTION *				pCollection;
	FLMUINT						uiCollection;
	F_Btree *					pbtree = NULL;
	FLMBYTE						ucKey [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT						uiKeyLen = 0;
	FLMUINT64					ui64NodeId;
	FLMBOOL						bNeg;
	FLMUINT						uiBytesProcessed;
	FLMUINT64					ui64SavedTransId;
	F_DOMNode *					pNode = NULL;
	eDomNodeType				eNodeType;

	// See if the database is being forced to close

	if (RC_BAD( rc = checkState( __FILE__, __LINE__)))
	{
		goto Exit;
	}

	// Must not be a transaction going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// Start a read transaction

	if (RC_BAD( rc = beginTrans( XFLM_READ_TRANS, 
		FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Determine which elements and attributes have been marked for
	// purging or checking.

	if (RC_BAD( rc = sweepGatherList( &pStateTbl, &uiNumItems)))
	{
		goto Exit;
	}

	// If there were no items to check or purge, we are done

	if (!uiNumItems)
	{
		goto Exit;	// Will return NE_XFLM_OK
	}

	// Walk through every node in the database.

	uiCollection = 0;
	pCollection = NULL;
	for (;;)
	{
		if( pThread->getShutdownFlag())
		{
			goto Exit;
		}

		// Get the next collection if necessary.

		if (!pCollection)
		{
			pCollection = m_pDict->getNextCollection( uiCollection, TRUE);
			if (!pCollection)
			{
				break;
			}
			uiCollection = pCollection->lfInfo.uiLfNum;

			if (pbtree)
			{
				pbtree->btClose();
			}
			else
			{

				// Get a btree

				if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbtree)))
				{
					goto Exit;
				}
			}

			if (RC_BAD( rc = pbtree->btOpen( this, &pCollection->lfInfo,
										FALSE, TRUE)))
			{
				goto Exit;
			}

			uiKeyLen = sizeof( ucKey);
			if (RC_BAD( rc = flmNumber64ToStorage( 1, &uiKeyLen,
											ucKey, FALSE, TRUE)))
			{
				goto Exit;
			}
			if( RC_BAD( rc = pbtree->btLocateEntry(
										ucKey, sizeof( ucKey), &uiKeyLen, XFLM_INCL)))
			{
				if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
				{
					rc = NE_XFLM_OK;

					// Go to the next collection, if any.

					pCollection = NULL;
					continue;
				}

				goto Exit;
			}
		}

		// Get the node ID

		if (RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
									&ui64NodeId, &bNeg, &uiBytesProcessed)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = getNode( uiCollection, ui64NodeId, &pNode)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{

				// Better be able to find the node at this point!

				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		ui64SavedTransId = getTransID();
		eNodeType = pNode->getNodeType();
		
		if( eNodeType == ELEMENT_NODE)
		{
			if( RC_BAD( rc = sweepCheckElementState( pNode, 
				pStateTbl, &uiNumItems, &bStartedTrans)))
			{
				goto Exit;
			}
		}

		if( !uiNumItems)
		{
			goto Exit;
		}
		
		// If the transaction changed, it was due to a dictionary update.
		// Need to refresh the b-tree because it is using an out-of-date
		// lfile.

		if( getTransID() != ui64SavedTransId)
		{
			if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
			{
				if( rc == NE_XFLM_BAD_COLLECTION)
				{
					rc = NE_XFLM_OK;

					// Go to the next collection, if any.

					pCollection = NULL;
					continue;
				}
				goto Exit;
			}

			pbtree->btClose();

			if( RC_BAD( rc = pbtree->btOpen( this, &pCollection->lfInfo,
				FALSE, TRUE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pbtree->btLocateEntry( ucKey, sizeof( ucKey),
				&uiKeyLen, XFLM_EXCL)))
			{
				if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					pCollection = NULL;
					continue;
				}

				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = pbtree->btNextEntry( ucKey, 
				sizeof( ucKey), &uiKeyLen)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
					pCollection = NULL;
					continue;
				}

				goto Exit;
			}
		}
	}

	// Now go through all of the items we gathered at the beginning and
	// if they are still in the state we first looked at them, remove
	// them.

	if( RC_BAD( rc = sweepFinalizeStates( pStateTbl, uiNumItems,
		&bStartedTrans)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		(void)abortTrans();
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( pStateTbl)
	{
		f_free( &pStateTbl);
	}
	
	if( pbtree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbtree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Gather the list of elements and attributes that are marked as
		needed to be checked or purged.  This routine assumes that a read
		transaction is already going.
****************************************************************************/
RCODE F_Db::sweepGatherList(
	ELM_ATTR_STATE_INFO **	ppStateTbl,
	FLMUINT *					puiNumItems)
{
	RCODE							rc = NE_XFLM_OK;
	FLMUINT						uiDictType;
	FLMUINT						uiDictNum;
	FLMUINT						uiStateTblSize = 0;
	ELM_ATTR_STATE_INFO *	pStateInfo;
	F_DOMNode *					pDictDoc = NULL;
	F_AttrElmInfo				defInfo;

	flmAssert( *puiNumItems == 0);
	flmAssert( *ppStateTbl == NULL);

	// Gather the elements and attributes that have been marked for
	// purging or checking.

	uiDictType = ELM_ELEMENT_TAG;
	uiDictNum = 0;
	for (;;)
	{
		if (uiDictType == ELM_ELEMENT_TAG)
		{
			if (RC_BAD( rc = m_pDict->getNextElement( this, &uiDictNum,
								&defInfo)))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
					uiDictNum = 0;
					uiDictType = ELM_ATTRIBUTE_TAG;
					continue;
				}
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = m_pDict->getNextAttribute( this, &uiDictNum,
												&defInfo)))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
					break;
				}
				goto Exit;
			}
		}
		
		if (defInfo.m_uiState == ATTR_ELM_STATE_CHECKING ||
			 defInfo.m_uiState == ATTR_ELM_STATE_PURGE)
		{
			// Add to the state table, increase table size if needed.

			if (*puiNumItems == uiStateTblSize)
			{
				ELM_ATTR_STATE_INFO *	pNewTbl;
				FLMUINT						uiNewSize;

				// Increase by 100 at a time - should be plenty, because
				// applications are not going to be checking 100s of
				// elements or attributes at a time.

				uiNewSize = uiStateTblSize + 100;

				if (RC_BAD( rc = f_calloc( uiNewSize *
											sizeof( ELM_ATTR_STATE_INFO),
											&pNewTbl)))
				{
					goto Exit;
				}
				
				if (uiStateTblSize)
				{
					f_memcpy( pNewTbl, *ppStateTbl,
									sizeof( ELM_ATTR_STATE_INFO) * uiStateTblSize);
					f_free( ppStateTbl);
				}
				
				*ppStateTbl = pNewTbl;
				uiStateTblSize = uiNewSize;
			}
			
			pStateInfo = &((*ppStateTbl)[*puiNumItems]);
			pStateInfo->uiDictType = uiDictType;
			pStateInfo->uiDictNum = uiDictNum;
			pStateInfo->uiState = defInfo.m_uiState;

			// Read the dictionary item and get its state change count.

			if (RC_BAD( rc = getDictionaryDef( uiDictType, uiDictNum,
										(IF_DOMNode **)&pDictDoc)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = pDictDoc->getAttributeValueUINT64( this,
									ATTR_STATE_CHANGE_COUNT_TAG,
									&pStateInfo->ui64StateChangeCount)))
			{
				goto Exit;
			}
			
			(*puiNumItems)++;
		}

		defInfo.resetInfo();
	}

Exit:

	if (pDictDoc)
	{
		pDictDoc->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Find an element or attribute's state.
****************************************************************************/
FSTATIC ELM_ATTR_STATE_INFO * sweepFindState(
	ELM_ATTR_STATE_INFO *	pStateTbl,
	FLMUINT						uiNumItems,
	FLMUINT						uiDictType,
	FLMUINT						uiDictNum,
	FLMUINT *					puiTblSlot
	)
{
	ELM_ATTR_STATE_INFO *	pStateInfo = NULL;
	FLMUINT						uiTblSize;
	FLMUINT						uiLow;
	FLMUINT						uiMid;
	FLMUINT						uiHigh;
	FLMUINT						uiTblDictType;
	FLMUINT						uiTblDictNum;
	FLMINT						iCmp;

	// Do binary search in the table

	if ((uiTblSize = uiNumItems) == 0)
	{
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		uiTblDictType = pStateTbl [uiMid].uiDictType;
		uiTblDictNum = pStateTbl [uiMid].uiDictNum;
		if (uiDictType == uiTblDictType)
		{
			if (uiDictNum == uiTblDictNum)
			{

				// Found Match

				pStateInfo = &pStateTbl [uiMid];
				*puiTblSlot = uiMid;
				goto Exit;
			}
			else if (uiDictNum < uiTblDictNum)
			{
				iCmp = -1;
			}
			else
			{
				iCmp = 1;
			}
		}
		else if (uiDictType < uiTblDictType)
		{
			iCmp = -1;
		}
		else
		{
			iCmp = 1;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pStateInfo);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::sweepCheckElementState(
	F_DOMNode *					pElementNode,
	ELM_ATTR_STATE_INFO *	pStateTbl,
	FLMUINT *					puiNumItems,
	FLMBOOL *					pbStartedTrans)
{
	RCODE							rc = NE_XFLM_OK;
	ELM_ATTR_STATE_INFO *	pStateInfo;
	FLMUINT						uiNameId;
	FLMUINT						uiTblSlot;
	F_DOMNode *					pDictDoc = NULL;
	FLMUINT64					ui64StateChangeCount;
	F_AttrElmInfo				defInfo;

	if( RC_BAD( rc = pElementNode->getNameId( this, &uiNameId)))
	{
		goto Exit;
	}
	
	if( !uiNameId)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	pStateInfo = sweepFindState( pStateTbl, *puiNumItems, 
		ELM_ELEMENT_TAG, uiNameId, &uiTblSlot);

	if( pStateInfo)
	{
		// Stop the read transaction and start an update
		// transaction.

		if( RC_BAD( rc = abortTrans()))
		{
			goto Exit;
		}
		
		*pbStartedTrans = FALSE;

		if( RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
		{
			goto Exit;
		}

		*pbStartedTrans = TRUE;

		// Get the current state to see if it has changed.

		if( RC_BAD( rc = m_pDict->getElement( this, uiNameId, &defInfo)))
		{
			if( rc != NE_XFLM_BAD_ELEMENT_NUM)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
		}

		// Read the dictionary item and get its state change count.

		if( RC_BAD( rc = getDictionaryDef( ELM_ELEMENT_TAG, uiNameId,
									(IF_DOMNode **)&pDictDoc)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDictDoc->getAttributeValueUINT64( this,
			ATTR_STATE_CHANGE_COUNT_TAG, &ui64StateChangeCount)))
		{
			goto Exit;
		}
		
		if( ui64StateChangeCount != pStateInfo->ui64StateChangeCount)
		{
			defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
		}

		// If the item's state is still 'checking' set it to
		// active.

		if( pStateInfo->uiState == ATTR_ELM_STATE_CHECKING)
		{
			if( defInfo.m_uiState == ATTR_ELM_STATE_CHECKING)
			{
				if( RC_BAD( rc = changeItemState( ELM_ELEMENT_TAG, uiNameId,
					XFLM_ACTIVE_OPTION_STR)))
				{
					goto Exit;
				}

				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
		}
		else
		{
			// If the state is not still purge, don't do anything more
			// on this element - set state to active, so no more purges
			// will take place.

			if( defInfo.m_uiState != ATTR_ELM_STATE_PURGE)
			{
				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
			else
			{
				if( RC_BAD( rc = pElementNode->deleteNode( this)))
				{
					if( rc != NE_XFLM_DOM_NODE_DELETED)
					{
						goto Exit;
					}

					rc = NE_XFLM_OK;
				}

				pElementNode = NULL;
			}
		}

		// Commit the transaction

		*pbStartedTrans = FALSE;
		if( RC_BAD( rc = commitTrans( 0, FALSE)))
		{
			goto Exit;
		}

		// If the state got changed to active, remove the thing from the
		// array and decrement the item count.  It means we have stopped
		// processing this item.

		if( pStateInfo->uiState != defInfo.m_uiState)
		{
			if( uiTblSlot < *puiNumItems - 1)
			{
				f_memmove( &pStateTbl [uiTblSlot], &pStateTbl [uiTblSlot + 1],
							sizeof( ELM_ATTR_STATE_INFO) *
							(*puiNumItems - 1 - uiTblSlot));
			}

			(*puiNumItems)--;
		}

		// Restart the read transaction.

		if( RC_BAD( rc = beginTrans( XFLM_READ_TRANS, 
			FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}

		*pbStartedTrans = TRUE;
	}

	// Check the element's attributes

	if( pElementNode)
	{
		if( RC_BAD( rc = sweepCheckAttributeStates( pElementNode, 
			pStateTbl, puiNumItems, pbStartedTrans)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDictDoc)
	{
		pDictDoc->Release();
	}

	if( RC_BAD( rc) && *pbStartedTrans)
	{
		abortTrans();
		*pbStartedTrans = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Db::sweepCheckAttributeStates(
	F_DOMNode *					pElementNode,
	ELM_ATTR_STATE_INFO *	pStateTbl,
	FLMUINT *					puiNumItems,
	FLMBOOL *					pbStartedTrans)
{
	RCODE							rc = NE_XFLM_OK;
	ELM_ATTR_STATE_INFO *	pStateInfo;
	FLMUINT						uiTblSlot;
	FLMUINT						uiNameId;
	F_DOMNode *					pDictDoc = NULL;
	IF_DOMNode *				pAttrNode = NULL;
	IF_DOMNode *				pNextAttrNode = NULL;
	FLMUINT64					ui64StateChangeCount;
	F_AttrElmInfo				defInfo;
	FLMBOOL						bModifiedDatabase = FALSE;

	flmAssert( pElementNode->getNodeType() == ELEMENT_NODE);

	if( !pElementNode->hasAttributes())
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pElementNode->getFirstAttribute( this, &pAttrNode)))
	{
		flmAssert( rc != NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	for( ;;)	
	{
		if( RC_BAD( rc = pAttrNode->getNameId( this, &uiNameId)))
		{
			goto Exit;
		}
	
		pStateInfo = sweepFindState( pStateTbl, *puiNumItems, 
			ELM_ATTRIBUTE_TAG, uiNameId, &uiTblSlot);
	
		// No need to do anything if there is no state info.
	
		if( !pStateInfo)
		{
			if( RC_BAD( rc = pAttrNode->getNextSibling( this, &pAttrNode)))
			{
				if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				
				rc = NE_XFLM_OK;
				break;
			}
			
			continue;
		}
	
		// Stop the read transaction and start an update
		// transaction.
		
		if( getTransType() != XFLM_UPDATE_TRANS)
		{
			if( RC_BAD( rc = abortTrans()))
			{
				goto Exit;
			}
			
			*pbStartedTrans = FALSE;
	
			if( RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
			{
				goto Exit;
			}
		
			*pbStartedTrans = TRUE;
			bModifiedDatabase = TRUE;
		}
		
		// Get the current state to see if it has changed.
	
		if( RC_BAD( rc = m_pDict->getAttribute( this, uiNameId, &defInfo)))
		{
			if( rc != NE_XFLM_BAD_ATTRIBUTE_NUM)
			{
				goto Exit;
			}
	
			rc = NE_XFLM_OK;
			defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
		}
	
		// Read the dictionary item and get its state change count.
	
		if( RC_BAD( rc = getDictionaryDef( ELM_ATTRIBUTE_TAG, 
			uiNameId, (IF_DOMNode **)&pDictDoc)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDictDoc->getAttributeValueUINT64( this,
			ATTR_STATE_CHANGE_COUNT_TAG, &ui64StateChangeCount)))
		{
			goto Exit;
		}
		
		if( ui64StateChangeCount != pStateInfo->ui64StateChangeCount)
		{
			defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
		}

		// Get the next attribute before doing anything to our
		// current attribute - because we may end up deleting
		// pAttrNode below.
	
		if( RC_BAD( rc = pAttrNode->getNextSibling( this, 
			&pNextAttrNode)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
		}
				
		// If the item's state is still 'checking' set it to
		// active.
	
		if( pStateInfo->uiState == ATTR_ELM_STATE_CHECKING)
		{
			if( defInfo.m_uiState == ATTR_ELM_STATE_CHECKING)
			{
				if( RC_BAD( rc = changeItemState( ELM_ATTRIBUTE_TAG,
					uiNameId, XFLM_ACTIVE_OPTION_STR)))
				{
					goto Exit;
				}
	
				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
		}
		else
		{
			// If the state is not still purge, don't do anything more
			// on this attribute - set state to active, so no more purges
			// will take place.
	
			if( defInfo.m_uiState != ATTR_ELM_STATE_PURGE)
			{
				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
			else
			{
				if( RC_BAD( rc = pAttrNode->deleteNode( this)))
				{
					if( rc != NE_XFLM_DOM_NODE_DELETED)
					{
						goto Exit;
					}
	
					rc = NE_XFLM_OK;
				}
			}
		}
	
		// If the state got changed to active, remove the thing from the
		// array and decrement the item count.  It means we have stopped
		// processing this item.
	
		if( pStateInfo->uiState != defInfo.m_uiState)
		{
			if( uiTblSlot < *puiNumItems - 1)
			{
				f_memmove( &pStateTbl [uiTblSlot], &pStateTbl [uiTblSlot + 1],
							sizeof( ELM_ATTR_STATE_INFO) *
							(*puiNumItems - 1 - uiTblSlot));
			}
	
			(*puiNumItems)--;
			
			if( *puiNumItems == 0)
			{
				break;
			}
		}
		
		pAttrNode->Release();
		pAttrNode = NULL;

		// Point pAttrNode to pNextAttrNode and steal its AddRef()

		if( (pAttrNode = pNextAttrNode) == NULL)
		{
			break;
		}
		pNextAttrNode = NULL;
	}
	
	if( bModifiedDatabase)
	{
		// Commit the transaction
	
		*pbStartedTrans = FALSE;
		if( RC_BAD( rc = commitTrans( 0, FALSE)))
		{
			goto Exit;
		}
	
		// Restart the read transaction.
	
		if( RC_BAD( rc = beginTrans( XFLM_READ_TRANS, 
			FLM_NO_TIMEOUT, XFLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
	
		*pbStartedTrans = TRUE;
	}

Exit:

	if( pDictDoc)
	{
		pDictDoc->Release();
	}
	
	if( pAttrNode)
	{
		pAttrNode->Release();
	}
	
	if( pNextAttrNode)
	{
		pNextAttrNode->Release();
	}

	if( RC_BAD( rc) && *pbStartedTrans)
	{
		abortTrans();
		*pbStartedTrans = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:	Go through items in the element/attribute table and finalize the
		state for each item.
****************************************************************************/
RCODE F_Db::sweepFinalizeStates(
	ELM_ATTR_STATE_INFO *	pStateTbl,
	FLMUINT						uiNumItems,
	FLMBOOL *					pbStartedTrans)
{
	RCODE							rc = NE_XFLM_OK;
	ELM_ATTR_STATE_INFO *	pStateInfo;
	F_DOMNode *					pNode = NULL;
	F_DOMNode *					pDictDoc = NULL;
	FLMUINT						uiLoop;
	FLMUINT64					ui64StateChangeCount;
	F_AttrElmInfo				defInfo;

	m_bItemStateUpdOk = TRUE;

	// Stop the read transaction and start an update transaction.

	abortTrans();
	*pbStartedTrans = FALSE;
	
	if( RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	*pbStartedTrans = TRUE;

	// Check the state of all items in the table.

	for (uiLoop = 0, pStateInfo = pStateTbl;
		  uiLoop < uiNumItems;
		  uiLoop++, pStateInfo++)
	{
		if (pStateInfo->uiDictType == ELM_ELEMENT_TAG)
		{
			if (RC_BAD( rc = m_pDict->getElement( this,
				pStateInfo->uiDictNum, &defInfo)))
			{
				// Element has gone away.

				if( rc != NE_XFLM_BAD_ELEMENT_NUM)
				{
					goto Exit;
				}
				
				rc = NE_XFLM_OK;
				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
		}
		else
		{
			if (RC_BAD( rc = m_pDict->getAttribute( this,
				pStateInfo->uiDictNum, &defInfo)))
			{

				// Attribute has gone away.

				if( rc != NE_XFLM_BAD_ATTRIBUTE_NUM)
				{
					goto Exit;
				}
				
				rc = NE_XFLM_OK;
				defInfo.m_uiState = ATTR_ELM_STATE_ACTIVE;
			}
		}

		// Read the dictionary item and get its state change count.

		if (RC_BAD( rc = getDictionaryDef( pStateInfo->uiDictType,
									pStateInfo->uiDictNum,
									(IF_DOMNode **)&pDictDoc)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pDictDoc->getAttributeValueUINT64( this,
								ATTR_STATE_CHANGE_COUNT_TAG,
								&ui64StateChangeCount)))
		{
			goto Exit;
		}

		// If the state is unchanged, purge the definition document

		if (defInfo.m_uiState == pStateInfo->uiState &&
			 ui64StateChangeCount == pStateInfo->ui64StateChangeCount)
		{
			// First make sure the element or attribute is not
			// referenced from an index definition.

			if (pStateInfo->uiDictType == ELM_ELEMENT_TAG)
			{
				if( RC_BAD( rc = m_pDict->checkElementReferences(
					pStateInfo->uiDictNum)))
				{
					if( rc != NE_XFLM_CANNOT_DEL_ELEMENT)
					{
						goto Exit;
					}
					
					rc = NE_XFLM_OK;
					pStateInfo->uiState = ATTR_ELM_STATE_ACTIVE;
				}
			}
			else
			{
				if( RC_BAD( rc = m_pDict->checkAttributeReferences(
					pStateInfo->uiDictNum)))
				{
					if( rc != NE_XFLM_CANNOT_DEL_ATTRIBUTE)
					{
						goto Exit;
					}
					
					rc = NE_XFLM_OK;
					pStateInfo->uiState = ATTR_ELM_STATE_ACTIVE;
				}
			}

			if( pStateInfo->uiState == ATTR_ELM_STATE_ACTIVE)
			{
				// Change the state to active since it is referenced
				// from an index definition

				if (RC_BAD( rc = changeItemState( pStateInfo->uiDictType,
											pStateInfo->uiDictNum,
											XFLM_ACTIVE_OPTION_STR)))
				{
					goto Exit;
				}
			}
			else
			{
				F_DataVector	srchKey;
				F_DataVector	foundKey;
				
				// Find and purge the definition document.

				if (RC_BAD( rc = srchKey.setUINT( 0, pStateInfo->uiDictType)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = srchKey.setUINT( 1, pStateInfo->uiDictNum)))
				{
					goto Exit;
				}

				if (RC_BAD( rc = keyRetrieve( XFLM_DICT_NUMBER_INDEX,
										&srchKey, XFLM_EXACT, &foundKey)))
				{
					if (rc == NE_XFLM_NOT_FOUND)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					goto Exit;
				}

				if (RC_BAD( rc = getNode( XFLM_DICT_COLLECTION,
											foundKey.getDocumentID(), &pNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					goto Exit;
				}
				
				if (RC_BAD( rc = pNode->deleteNode( this)))
				{
					goto Exit;
				}
			}
		}

		defInfo.resetInfo();
	}

	// Commit the transaction.

	*pbStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, FALSE)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc) && *pbStartedTrans)
	{
		abortTrans();
		*pbStartedTrans = FALSE;
	}

	m_bItemStateUpdOk = FALSE;

	if (pNode)
	{
		pNode->Release();
	}

	if (pDictDoc)
	{
		pDictDoc->Release();
	}
	
	return( rc);
}
