//------------------------------------------------------------------------------
// Desc:	Class for gathering node information.
// Tabs:	3
//
// Copyright (c) 2005-2007 Novell, Inc. All Rights Reserved.
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

/*****************************************************************************
Desc:	Get node information and add it to the node information object.
******************************************************************************/
RCODE XFLAPI F_NodeInfo::addNodeInfo(
	IF_Db *			ifpDb,
	IF_DOMNode *	pNode,
	FLMBOOL			bDoNodeSubTree,
	FLMBOOL			bDoSelf)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pCurNode = NULL;
	IF_DOMNode *	pTmpNode = NULL;
	FLMUINT64		ui64MyNodeId;
	F_CachedNode *	pCachedNode = ((F_DOMNode *)pNode)->m_pCachedNode;
	FLMUINT			uiCollection = pCachedNode->getCollection();
	FLMBOOL			bStartedTrans = FALSE;
	FLMBOOL			bDoNode;
	F_Db *			pDb = (F_Db *)ifpDb;

	// Start a read transaction, if no other transaction is going.

	if (ifpDb->getTransType() == XFLM_NO_TRANS)
	{
		if (RC_BAD( rc = ifpDb->transBegin( XFLM_READ_TRANS)))
		{
			goto Exit;
		}
		bStartedTrans = TRUE;
	}

	pCurNode = pNode;
	pCurNode->AddRef();
	
	// Need special case handling of attribute nodes, because
	// the pCachedNode will be pointing to the element node for
	// the attribute.  Furthermore, it will be the only node
	// we do, because it can have no children to do.
	
	if (pCurNode->getNodeType() == ATTRIBUTE_NODE)
	{
		if (bDoSelf)
		{
			F_AttrItem *	pAttrItem = NULL;
			FLMUINT			uiSizeNeeded;
	
			if (pCachedNode->m_uiAttrCount)
			{
				pAttrItem = pCachedNode->getAttribute(
										((F_DOMNode *)pCurNode)->m_uiAttrNameId, NULL);
			}
			if (!pAttrItem)
			{
				rc = RC_SET( NE_XFLM_DOM_NODE_DELETED);
				goto Exit;
			}
			pAttrItem->getAttrSizeNeeded( pCachedNode->m_ppAttrList [0]->m_uiNameId,
					&m_nodeInfo, NULL, &uiSizeNeeded);
		}
		
		// There can be no child nodes to do, so we are done.
		
		goto Exit;
	}
	
	// Traverse the sub-tree and get info. on all nodes below and including the
	// node we are starting on.

	ui64MyNodeId = pCachedNode->getNodeId();
	bDoNode = bDoSelf;
	for (;;)
	{
		
		// Add in statistics for the current node.
		
		if (bDoNode)
		{
			if (RC_BAD( rc = pCachedNode->headerToBuf(
							pCachedNode->getModeFlags() & FDOM_FIXED_SIZE_HEADER
							? TRUE
							: FALSE, NULL, NULL, &m_nodeInfo, pDb)))
			{
				goto Exit;
			}
		}
		if (!bDoNodeSubTree)
		{
			break;
		}
		
		// If the node has an annotation, do that node.
		
		if (pCachedNode->getAnnotationId() && bDoNode)
		{
			F_CachedNode *	pTmpCachedNode;

			if (RC_BAD( rc = ifpDb->getNode( uiCollection, pCachedNode->getAnnotationId(),
									&pTmpNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
				}
				goto Exit;
			}
			pTmpCachedNode = ((F_DOMNode *)pTmpNode)->m_pCachedNode;
			if (RC_BAD( rc = pTmpCachedNode->headerToBuf(
							pTmpCachedNode->getModeFlags() & FDOM_FIXED_SIZE_HEADER
							? TRUE
							: FALSE, NULL, NULL, &m_nodeInfo, pDb)))
			{
				goto Exit;
			}
		}
		
		// If the node has a child node, go to it.
		
		if (pCachedNode->getFirstChildId())
		{
			if (RC_BAD( rc = ifpDb->getNode( uiCollection, pCachedNode->getFirstChildId(),
									&pCurNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
				}
				goto Exit;
			}
			pCachedNode = ((F_DOMNode *)pCurNode)->m_pCachedNode;
			bDoNode = TRUE;
			continue;
		}

		for(;;)
		{
			
			// If we are on the node we started on, there is nothing more to do.
			
			if (pCachedNode->getNodeId() == ui64MyNodeId)
			{
				goto Exit;		// Should return NE_XFLM_OK
			}
		
			// If node has a sibling node, go to it.
			
			if (pCachedNode->getNextSibId())
			{
				if (RC_BAD( rc = ifpDb->getNode( uiCollection, pCachedNode->getNextSibId(),
										&pCurNode)))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = RC_SET( NE_XFLM_DATA_ERROR);
					}
					goto Exit;
				}
				pCachedNode = ((F_DOMNode *)pCurNode)->m_pCachedNode;
				bDoNode = TRUE;
				break;
			}
		
			// Traverse back up the tree to parent node.
			// Better be a parent node at this point - because we know we
			// are not on the node we started on.
			
			flmAssert( pCachedNode->getParentId());
			if (RC_BAD( rc = ifpDb->getNode( uiCollection, pCachedNode->getParentId(),
									&pCurNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET( NE_XFLM_DATA_ERROR);
				}
				goto Exit;
			}
			pCachedNode = ((F_DOMNode *)pCurNode)->m_pCachedNode;
		}
	}

Exit:

	if (pCurNode)
	{
		pCurNode->Release();
	}
	if (pTmpNode)
	{
		pTmpNode->Release();
	}
	if (bStartedTrans)
	{
		(void)ifpDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:	Create an empty node info. object and return it's interface...
****************************************************************************/
RCODE XFLAPI F_DbSystem::createIFNodeInfo(
	IF_NodeInfo **	ppNodeInfo)
{
	RCODE	rc = NE_XFLM_OK;

	if ((*ppNodeInfo = f_new F_NodeInfo) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
}
