//------------------------------------------------------------------------------
// Desc:	Document list object implementation
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
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

#define MAX_PENDING_NODES		255

/****************************************************************************
Desc: 
****************************************************************************/
FLMBOOL F_NodeList::findNode(
	FLMUINT		uiCollection,
	FLMUINT64	ui64Document,
	FLMUINT64	ui64NodeId,
	FLMUINT *	puiPos
	)
{
	FLMBOOL		bFound = FALSE;
	FLMUINT		uiTblSize;
	FLMUINT		uiLow;
	FLMUINT		uiMid;
	FLMUINT		uiHigh;
	FLMUINT		uiTblCollection;
	FLMUINT64	ui64TblDocument;
	FLMUINT64	ui64TblNodeId;
	FLMINT		iCmp;

	// Do binary search in the table

	if ((uiTblSize = m_uiNumNodes) == 0)
	{
		*puiPos = 0;
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;

		uiTblCollection = m_pNodeTbl[ uiMid].uiCollection;
		ui64TblDocument = m_pNodeTbl[ uiMid].ui64Document;
		ui64TblNodeId = m_pNodeTbl[ uiMid].ui64NodeId;

		if( uiCollection == uiTblCollection)
		{
			if( ui64Document == ui64TblDocument)
			{
				if( ui64NodeId == ui64TblNodeId)
				{
					iCmp = 0;
				}
				else if( ui64NodeId < ui64TblNodeId)
				{
					iCmp = -1;
				}
				else
				{
					iCmp = 1;
				}
			}
			else if( ui64Document < ui64TblDocument)
			{
				iCmp = -1;
			}
			else
			{
				iCmp = 1;
			}
		}
		else if( uiCollection < uiTblCollection)
		{
			iCmp = -1;
		}
		else
		{
			iCmp = 1;
		}

		if (!iCmp)
		{
			// Found Match

			bFound = TRUE;
			*puiPos = uiMid;
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{
			// Done, item not found

			*puiPos = (iCmp < 0 
							? uiMid 
							: uiMid + 1);
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				*puiPos = 0;
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				*puiPos = uiMid + 1;
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( bFound);
}

/*****************************************************************************
Desc:	Add a node to the node list.  If it is already there, it is ok.
******************************************************************************/
RCODE F_NodeList::addNode(
	FLMUINT		uiCollection,
	FLMUINT64	ui64Document,
	FLMUINT64	ui64NodeId)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiInsertPos;

	// Cannot allow collection or document to be zero

	flmAssert( uiCollection && ui64Document);

	if( m_uiLastCollection == uiCollection &&
		m_ui64LastDocument == ui64Document &&
		m_ui64LastNodeId == ui64NodeId)
	{
		goto Exit;
	}

	if( !findNode( uiCollection, ui64Document, ui64NodeId, &uiInsertPos))
	{
		// Have we reached the limit of the number of documents we will
		// keep pending in a transaction?

		if( m_uiNumNodes == MAX_PENDING_NODES)
		{
			rc = RC_SET( NE_XFLM_TOO_MANY_PENDING_NODES);
			goto Exit;
		}

		// See if we need to allocate the table

		if( !m_pNodeTbl)
		{
			if (RC_BAD( rc = f_alloc( 
				sizeof( NODE_LIST_ITEM) * MAX_PENDING_NODES, &m_pNodeTbl)))
			{
				goto Exit;
			}

			m_uiNodeTblSize = MAX_PENDING_NODES;
		}

		flmAssert( uiInsertPos <= m_uiNumNodes);

		// Make room for the new node ID

		if( uiInsertPos < m_uiNumNodes)
		{
			f_memmove( &m_pNodeTbl[ uiInsertPos+1],
						  &m_pNodeTbl[ uiInsertPos],
						  sizeof( NODE_LIST_ITEM) * (m_uiNumNodes - uiInsertPos));
		}

		m_pNodeTbl[ uiInsertPos].uiCollection = uiCollection;
		m_pNodeTbl[ uiInsertPos].ui64Document = ui64Document;
		m_pNodeTbl[ uiInsertPos].ui64NodeId = ui64NodeId;
		m_uiNumNodes++;
	}

	// Save collection and document id - this is an optimization
	// that will keep us from calling findNode too much if
	// we are working inside the same document.

	m_uiLastPosition = uiInsertPos;
	m_uiLastCollection = uiCollection;
	m_ui64LastDocument = ui64Document;
	m_ui64LastNodeId = ui64NodeId;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Remove a node from the node list.  If it is not there, it is ok.
******************************************************************************/
void F_NodeList::removeNode(
	FLMUINT		uiCollection,
	FLMUINT64	ui64Document,
	FLMUINT64	ui64NodeId)
{
	FLMUINT	uiPos;

	// Cannot allow collection or document to be zero

	flmAssert( uiCollection && ui64Document);

	if( m_uiLastCollection == uiCollection && 
		 m_ui64LastDocument == ui64Document &&
		 m_ui64LastNodeId == ui64NodeId)
	{
		flmAssert( m_uiLastPosition < m_uiNumNodes);
		removeNode( m_uiLastPosition);
	}
	else
	{
		if( findNode( uiCollection, ui64Document, ui64NodeId, &uiPos))
		{
			flmAssert( uiPos < m_uiNumNodes);
			removeNode( uiPos);
		}
	}
}
