//------------------------------------------------------------------------------
// Desc:	This file contains the main routines for building of index keys,
//			and adding them to the database.
// Tabs:	3
//
// Copyright (c) 1990-1992, 1994-2007 Novell, Inc. All Rights Reserved.
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

#define STACK_DATA_BUF_SIZE		64

typedef struct NodeTraverseTag *	NODE_TRAV_p;
typedef struct AnchorNodeTag *	ANCHOR_NODE_p;

typedef struct AnchorNodeTag
{
	FLMUINT64			ui64AnchorNodeId;
	FLMUINT				uiAnchorNameId;
	eDomNodeType		eNodeType;
	FLMBOOL				bSeeIfRepeatingSibs;
	ANCHOR_NODE_p		pNext;
} ANCHOR_NODE;

typedef struct NodeTraverseTag
{
	ANCHOR_NODE *	pAnchorNode;
	F_DOMNode *		pNode;
	FLMBOOL			bInNodeSubtree;
	ICD *				pIcd;
	FLMUINT			uiSibIcdAttrs;
	FLMUINT			uiSibIcdElms;
	FLMBOOL			bTraverseChildren;
	FLMBOOL			bTraverseSibs;
	NODE_TRAV_p		pParent;
	NODE_TRAV_p		pChild;
} NODE_TRAV;

// Local function prototypes

FSTATIC RCODE kyAddIDsToKey(
	FLMUINT64		ui64DocumentID,
	IXD *				pIxd,
	CDL_HDR *		pCdlTbl,
	FLMBYTE *		pucKeyBuf,
	FLMUINT			uiIDBufSize,
	FLMUINT *		puiIDLen);

FSTATIC RCODE kySeeIfRepeatingSibs(
	F_Db *			pDb,
	F_DOMNode *		pNode,
	FLMBOOL *		pbHadRepeatingSib);

FSTATIC RCODE kyFindChildNode(
	F_Db *			pDb,
	F_Pool *			pPool,
	NODE_TRAV **	ppTrav,
	FLMBOOL *		pbGotChild,
	FLMBOOL *		pbHadRepeatingSib);

FSTATIC RCODE kyFindSibNode(
	F_Db *			pDb,
	NODE_TRAV *		pTrav,
	FLMBOOL			bTestFirstNode,
	FLMBOOL *		pbGotSib,
	FLMBOOL *		pbHadRepeatingSib);

/****************************************************************************
Desc:	Append node IDs to the key buffer.
****************************************************************************/
FSTATIC RCODE kyAddIDsToKey(
	FLMUINT64		ui64DocumentID,
	IXD *				pIxd,
	CDL_HDR *		pCdlTbl,
	FLMBYTE *		pucKeyBuf,
	FLMUINT			uiIDBufSize,
	FLMUINT *		puiIDLen)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucIDs = pucKeyBuf;
	FLMUINT			uiSenLen;
	FLMUINT			uiIDLen = 0;
	ICD *				pIcd;
	FLMUINT64		ui64Id;
	FLMBYTE			ucTmpSen [FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE *		pucTmpSen;

	// Our ID buffer can never exceed MAX_ID_SIZE bytes (which is always
	// defined to be 256), because it has a trailing
	// length byte that we put on it - which can only represent up to
	// 255 bytes.

	if (uiIDBufSize > MAX_ID_SIZE)
	{
		uiIDBufSize = MAX_ID_SIZE;
	}

	// Put document ID into buffer.  If there is room for at least nine
	// bytes, we can encode the ID right into the buffer safely.  Otherwise,
	// we have to use a temporary buffer and see if there is room.
	
	if (uiIDBufSize - uiIDLen >= 9)
	{
		uiIDLen += f_encodeSEN( ui64DocumentID, &pucIDs);
	}
	else
	{
		pucTmpSen = &ucTmpSen [0];
		uiSenLen = f_encodeSEN( ui64DocumentID, &pucTmpSen);
		if (uiSenLen + uiIDLen > uiIDBufSize)
		{
			rc = RC_SET( NE_XFLM_KEY_OVERFLOW);
			goto Exit;
		}
		f_memcpy( pucIDs, ucTmpSen, uiSenLen);
		uiIDLen += uiSenLen;
		pucIDs += uiSenLen;
	}

	// Append the key component NODE IDs to the key

	for (pIcd = pIxd->pFirstKey; pIcd; pIcd = pIcd->pNextKeyComponent)
	{
		ui64Id = (FLMUINT64)(pCdlTbl [pIcd->uiCdl].pCdlList &&
									pCdlTbl [pIcd->uiCdl].pCdlList->pNode
									? pCdlTbl [pIcd->uiCdl].pCdlList->pNode->getIxNodeId()
									: (FLMUINT64)0);
									
		// Put node ID into buffer.  If there is room for at least nine
		// bytes, we can encode the ID right into the buffer safely.  Otherwise,
		// we have to use a temporary buffer and see if there is room.
		
		if (uiIDBufSize - uiIDLen >= 9)
		{
			uiIDLen += f_encodeSEN( ui64Id, &pucIDs);
		}
		else
		{
			pucTmpSen = &ucTmpSen [0];
			uiSenLen = f_encodeSEN( ui64Id, &pucTmpSen);
			if (uiSenLen + uiIDLen > uiIDBufSize)
			{
				rc = RC_SET( NE_XFLM_KEY_OVERFLOW);
				goto Exit;
			}
			f_memcpy( pucIDs, ucTmpSen, uiSenLen);
			uiIDLen += uiSenLen;
			pucIDs += uiSenLen;
		}
	}

	// Append the data NODE IDs to the key

	for (pIcd = pIxd->pFirstData; pIcd; pIcd = pIcd->pNextDataComponent)
	{
		ui64Id = (FLMUINT64)(pCdlTbl [pIcd->uiCdl].pCdlList &&
									pCdlTbl [pIcd->uiCdl].pCdlList->pNode
					? pCdlTbl [pIcd->uiCdl].pCdlList->pNode->getIxNodeId()
					: (FLMUINT64)0);

		// Put node ID into buffer.  If there is room for at least nine
		// bytes, we can encode the ID right into the buffer safely.  Otherwise,
		// we have to use a temporary buffer and see if there is room.
		
		if (uiIDBufSize - uiIDLen >= 9)
		{
			uiIDLen += f_encodeSEN( ui64Id, &pucIDs);
		}
		else
		{
			pucTmpSen = &ucTmpSen [0];
			uiSenLen = f_encodeSEN( ui64Id, &pucTmpSen);
			if (uiSenLen + uiIDLen > uiIDBufSize)
			{
				rc = RC_SET( NE_XFLM_KEY_OVERFLOW);
				goto Exit;
			}
			f_memcpy( pucIDs, ucTmpSen, uiSenLen);
			uiIDLen += uiSenLen;
			pucIDs += uiSenLen;
		}
	}

	// Append the context NODE IDs to the key

	for (pIcd = pIxd->pFirstContext; pIcd; pIcd = pIcd->pNextKeyComponent)
	{
		ui64Id = (FLMUINT64)(pCdlTbl [pIcd->uiCdl].pCdlList &&
									pCdlTbl [pIcd->uiCdl].pCdlList->pNode
					? pCdlTbl [pIcd->uiCdl].pCdlList->pNode->getIxNodeId()
					: (FLMUINT64)0);

		// Put node ID into buffer.  If there is room for at least nine
		// bytes, we can encode the ID right into the buffer safely.  Otherwise,
		// we have to use a temporary buffer and see if there is room.
		
		if (uiIDBufSize - uiIDLen >= 9)
		{
			uiIDLen += f_encodeSEN( ui64Id, &pucIDs);
		}
		else
		{
			pucTmpSen = &ucTmpSen [0];
			uiSenLen = f_encodeSEN( ui64Id, &pucTmpSen);
			if (uiSenLen + uiIDLen > uiIDBufSize)
			{
				rc = RC_SET( NE_XFLM_KEY_OVERFLOW);
				goto Exit;
			}
			f_memcpy( pucIDs, ucTmpSen, uiSenLen);
			uiIDLen += uiSenLen;
			pucIDs += uiSenLen;
		}
	}

	*puiIDLen = uiIDLen;

Exit:

	return( rc);
}

/*****************************************************************************
Desc: F_OldNodeList destructor
*****************************************************************************/
F_OldNodeList::~F_OldNodeList()
{
	if (m_pNodeList)
	{
		f_free( &m_pNodeList);
	}
	m_pool.poolFree();
}

/*****************************************************************************
Desc: Find a node in the old node list.
*****************************************************************************/
FLMBOOL F_OldNodeList::findNodeInList(
	eDomNodeType	eNodeType,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	FLMUINT			uiNameId,
	FLMBYTE **		ppucData,
	FLMUINT *		puiDataLen,
	FLMUINT *		puiInsertPos)
{
	FLMBOOL			bFound = FALSE;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	eDomNodeType	eTblNodeType;
	FLMUINT			uiTblCollection;
	FLMUINT			uiTblNameId;
	FLMUINT64		ui64TblNodeId;

	// Do binary search in the table

	if ((uiTblSize = m_uiNodeCount) == 0)
	{
		*puiInsertPos = 0;
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		eTblNodeType = m_pNodeList[ uiMid].eNodeType;
		uiTblCollection = m_pNodeList[ uiMid].uiCollection;
		ui64TblNodeId = m_pNodeList[ uiMid].ui64NodeId;
		uiTblNameId = m_pNodeList[ uiMid].uiNameId;

		flmAssert( eTblNodeType != INVALID_NODE);
		flmAssert( uiTblCollection);
		flmAssert( ui64TblNodeId);
		flmAssert( uiTblNameId);

		if( eTblNodeType == eNodeType &&
			 uiTblCollection == uiCollection && 
			 ui64TblNodeId == ui64NodeId &&
			 uiTblNameId == uiNameId)
		{
			bFound = TRUE;
			*ppucData = m_pNodeList [uiMid].pucData;
			*puiDataLen = m_pNodeList [uiMid].uiDataLen;
			*puiInsertPos = uiMid;
			goto Exit;
		}

		// Check if we are done

		if( uiLow >= uiHigh)
		{
			// Done, item not found

			if( eNodeType >= eTblNodeType)
			{
				goto CmpGreaterEq;
			}

			if( uiCollection >= uiTblCollection)
			{
				goto CmpGreaterEq;
			}

			if( ui64NodeId >= ui64TblNodeId)
			{
				goto CmpGreaterEq;
			}

			if( uiNameId >= uiTblNameId)
			{
CmpGreaterEq:
				*puiInsertPos = uiMid + 1;
				goto Exit;
			}

			*puiInsertPos = uiMid;
			goto Exit;
		}

		if( eNodeType >= eTblNodeType)
		{
			goto CmpGreaterEq2;
		}

		if( uiCollection >= uiTblCollection)
		{
			goto CmpGreaterEq2;
		}

		if( ui64NodeId >= ui64TblNodeId)
		{
			goto CmpGreaterEq2;
		}

		if( uiNameId >= uiTblNameId)
		{
CmpGreaterEq2:
			if (uiMid == uiTblSize)
			{
				*puiInsertPos = uiMid + 1;
				goto Exit;
			}
			uiLow = uiMid + 1;
			continue;
		}

		if (uiMid == 0)
		{
			*puiInsertPos = 0;
			goto Exit;
		}
		uiHigh = uiMid - 1;
		continue;
	}

Exit:

	return( bFound);
}

/*****************************************************************************
Desc: Add an old node to the old node list.
*****************************************************************************/
RCODE F_OldNodeList::addNodeToList(
	F_Db *			pDb,
	F_DOMNode *		pNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiInsertPos;
	FLMBYTE *			pucData;
	FLMUINT				uiDataType;
	FLMUINT				uiDataLen;
	FLMUINT				uiChars;
	FLMUINT				uiBufSize;
	FLMUINT				uiCollection;
	FLMUINT				uiNameId;
	FLMUINT64			ui64NodeId;
	
	if( RC_BAD( rc = pNode->getCollection( pDb, &uiCollection)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNode->getDataType( pDb, &uiDataType)))
	{
		goto Exit;
	}

	ui64NodeId = pNode->getIxNodeId();
	uiNameId = pNode->getNameId();
	
	// See if the node is already there
	
	if( !findNodeInList( pNode->getNodeType(),
		uiCollection, ui64NodeId,
		uiNameId, &pucData, &uiDataLen, &uiInsertPos))
	{
		// Expand the size of the table.
		
		if (m_uiNodeCount == m_uiListSize)
		{
			if (RC_BAD( rc = f_realloc( (m_uiListSize + 20) *
												sizeof( OLD_NODE_DATA), &m_pNodeList)))
			{
				goto Exit;
			}
			m_uiListSize += 20;
		}

		if (uiInsertPos < m_uiNodeCount)
		{
			f_memmove( &m_pNodeList [uiInsertPos + 1],
						  &m_pNodeList [uiInsertPos],
						  sizeof( OLD_NODE_DATA) * (m_uiNodeCount - uiInsertPos));
		}

		m_pNodeList [uiInsertPos].eNodeType = pNode->getNodeType();
		m_pNodeList [uiInsertPos].uiCollection = uiCollection;
		m_pNodeList [uiInsertPos].ui64NodeId = ui64NodeId;
		m_pNodeList [uiInsertPos].uiNameId = uiNameId;
		m_uiNodeCount++;
		
		// Set up the data - either unicode or binary.
		
		if( uiDataType == XFLM_BINARY_TYPE)
		{
			// Get the length first.
			
			if( RC_BAD( rc = pNode->getDataLength( pDb, &uiBufSize)))
			{
				goto Exit;
			}
			
			// Allocate the space needed.
			
			if (RC_BAD( rc = m_pool.poolAlloc( uiBufSize,
										(void **)&m_pNodeList [uiInsertPos].pucData)))
			{
				goto Exit;
			}
			
			// Go back again and get the data now.

			if( RC_BAD( rc = pNode->getBinary( pDb, m_pNodeList [uiInsertPos].pucData,
				0, uiBufSize, NULL)))
			{
				goto Exit;
			}

			m_pNodeList [uiInsertPos].uiDataLen = uiBufSize;
		}
		else
		{
			flmAssert( uiDataType == XFLM_TEXT_TYPE);
			
			// Get the length first.

			if( RC_BAD( rc = pNode->getUnicodeChars( pDb, &uiChars)))
			{
				goto Exit;
			}
			
			// Allocate the space needed.
			
			uiBufSize = (uiChars + 1) * sizeof( FLMUNICODE);
			if (RC_BAD( rc = m_pool.poolAlloc( uiBufSize,
										(void **)&m_pNodeList [uiInsertPos].pucData)))
			{
				goto Exit;
			}
			
			// Go back again and get the data now.

			if( RC_BAD( rc = pNode->getUnicode( pDb, 
				(FLMUNICODE *)m_pNodeList [uiInsertPos].pucData,
				uiBufSize, 0, FLM_MAX_UINT)))
			{
				goto Exit;
			}

			m_pNodeList [uiInsertPos].uiDataLen = uiBufSize;
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc: Release all of the nodes in the list.
*****************************************************************************/
void F_OldNodeList::resetList( void)
{
	m_pool.poolReset( NULL);
	m_uiNodeCount = 0;
}

/****************************************************************************
Desc:	Add an index key to the buffers
****************************************************************************/
RCODE F_Db::addToKrefTbl(
	FLMUINT	uiKeyLen,
	FLMUINT	uiDataLen)
{
	RCODE				rc = NE_XFLM_OK;
	KREF_ENTRY *	pKref;
	FLMUINT			uiSizeNeeded;
	FLMBYTE *		pucDest;

	// If the table is FULL, expand the table

	if (m_uiKrefCount == m_uiKrefTblSize)
	{
		FLMUINT		uiAllocSize;
		FLMUINT		uiOrigKrefTblSize = m_uiKrefTblSize;

		if (m_uiKrefTblSize > 0x8000 / sizeof( KREF_ENTRY *))
		{
			m_uiKrefTblSize += 4096;
		}
		else
		{
			m_uiKrefTblSize *= 2;
		}

		uiAllocSize = m_uiKrefTblSize * sizeof( KREF_ENTRY *);

		rc = f_realloc( uiAllocSize, &m_pKrefTbl);
		if (RC_BAD(rc))
		{
			m_uiKrefTblSize = uiOrigKrefTblSize;
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}

	// Allocate memory for the key's KREF and the key itself.
	// We allocate one extra byte so we can zero terminate the key
	// below.  The extra zero character is to ensure that the compare
	// in the qsort routine will work.

	uiSizeNeeded = sizeof( KREF_ENTRY) + uiKeyLen + 1 + uiDataLen;

	if (RC_BAD( rc = m_pKrefPool->poolAlloc( uiSizeNeeded,
										(void **)&pKref)))
	{
		goto Exit;
	}
	
	m_pKrefTbl [ m_uiKrefCount++] = pKref;
	m_uiTotalKrefBytes += uiSizeNeeded;

	// Fill in all of the fields in the KREF structure.

	pKref->ui16IxNum = (FLMUINT16)m_keyGenInfo.pIxd->uiIndexNum;
	pKref->bDelete = m_keyGenInfo.bAddKeys ? FALSE : TRUE;
	pKref->ui16KeyLen = (FLMUINT16)uiKeyLen;
	pKref->uiSequence = m_uiKrefCount;
	pKref->uiDataLen = uiDataLen;

	// Copy the key to just after the KREF structure.

	pucDest = (FLMBYTE *)(&pKref [1]);
 	f_memcpy( pucDest, m_keyGenInfo.pucKeyBuf, uiKeyLen);

	// Null terminate the key so compare in qsort will work.

	pucDest [uiKeyLen] = 0;
	if (uiDataLen)
	{
		f_memcpy( pucDest + uiKeyLen + 1, m_keyGenInfo.pucData, uiDataLen);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Verifies that all of the nodes in the CDL list are in contexts that
		are related according to the index definition.
****************************************************************************/
RCODE F_Db::verifyKeyContext(
	FLMBOOL *		pbVerified)
{
	RCODE				rc = NE_XFLM_OK;
	CDL *				pCdl;
	CDL *				pParentCdl;
	ICD *				pIcd;

	*pbVerified = FALSE;
	pIcd = m_keyGenInfo.pIxd->pIcdTree;

	// Do in-order traversal, from leaf ICDs up.

	while (pIcd->pFirstChild)
	{
		pIcd = pIcd->pFirstChild;
	}

	for (;;)
	{
		if ((pCdl = m_keyGenInfo.pCdlTbl [pIcd->uiCdl].pCdlList) != NULL)
		{

			// If this is a "missing" placeholder and the
			// component is required, we cannot build the key

			if (!pCdl->pNode && pIcd->uiKeyComponent &&
				(pIcd->uiFlags & ICD_REQUIRED_PIECE))
			{
				goto Exit;
			}

			// If the ICD has a parent, see if the parent has the
			// correct node id.

			if (pIcd->pParent)
			{
				pParentCdl = m_keyGenInfo.pCdlTbl [pIcd->pParent->uiCdl].pCdlList;
				
				if( !pParentCdl || !pParentCdl->pNode)
				{
					goto Exit;
				}
				
				if( pParentCdl->pNode->getNodeId() != pCdl->ui64ParentId)
				{
					goto Exit;
				}
			}
		}
		else
		{
			if (pIcd->uiKeyComponent && (pIcd->uiFlags & ICD_REQUIRED_PIECE))
			{
				// This better already have been checked

				flmAssert( 0);
				goto Exit;
			}
		}

		// See if there is a sibling

		if (pIcd->pNextSibling)
		{
			pIcd = pIcd->pNextSibling;
			while (pIcd->pFirstChild)
			{
				pIcd = pIcd->pFirstChild;
			}
		}
		else
		{
			if ((pIcd = pIcd->pParent) == NULL)
			{
				break;
			}
		}
	}
	*pbVerified = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Build the data part for a key.
Notes:	This routine is recursive in nature.  Will recurse the number of
			data components defined in the index.
****************************************************************************/
RCODE F_Db::buildData(
	ICD *			pIcd,
	FLMUINT		uiKeyLen,
	FLMUINT		uiDataLen
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCdl;
	CDL *			pFirstCdl;
	CDL *			pCdl;
	FLMUINT		uiDataComponentLen;
	F_DOMNode *	pNode = NULL;
	FLMBYTE		ucTmpSen [FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE *	pucTmpSen;
	FLMUINT		uiSENLen;
	FLMBOOL		bVerified;
	FLMUINT		uiIDLen;

	uiCdl = pIcd->uiCdl;
	pFirstCdl = m_keyGenInfo.pCdlTbl [uiCdl].pCdlList;
	pCdl = pFirstCdl;
	if (!m_keyGenInfo.bUseSubtreeNodes)
	{

		// If we are not using nodes in the sub-tree, skip any
		// that are in the sub-tree.

		while (pCdl && pCdl->bInNodeSubtree)
		{
			pCdl = pCdl->pNext;
		}
	}

	// Go through all of the data CDL - even the ones that are NULL

	for (;;)
	{

		// Data components cannot be root tags.

		flmAssert( pIcd->uiDictNum != ELM_ROOT_TAG);

		if (pNode)
		{
			pNode->Release();
			pNode = NULL;
		}

		m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pCdl;

		if (pCdl)
		{

			// NOTE: pNode could be NULL because it is a "missing" placeholder

			pNode = pCdl->pNode;
		}

		if (pNode)
		{
			pNode->AddRef();
			if (RC_BAD( rc = pNode->getDataLength( this, &uiDataComponentLen)))
			{
				goto Exit;
			}
		}
		else
		{
			uiDataComponentLen = 0;
		}

		// Output the length of the data as a SEN value

		pucTmpSen = &ucTmpSen [0];
		uiSENLen = f_encodeSEN( uiDataComponentLen, &pucTmpSen);
		if (uiDataComponentLen + uiSENLen + uiDataLen > m_keyGenInfo.uiDataBufSize)
		{
			FLMUINT	uiNewSize = uiDataComponentLen + uiSENLen + uiDataLen + 512;

			// Allocate the data buffer if it has not been allocated.  Otherwise,
			// realloc it.

			if (!m_keyGenInfo.bDataBufAllocated)
			{
				FLMBYTE *	pucNewData;

				if (RC_BAD( rc = f_alloc( uiNewSize, &pucNewData)))
				{
					goto Exit;
				}

				if( uiDataLen)
				{
					f_memcpy( pucNewData, m_keyGenInfo.pucData, uiDataLen);
				}
				m_keyGenInfo.pucData = pucNewData;
				m_keyGenInfo.bDataBufAllocated = TRUE;
			}
			else
			{
				// Reallocate the buffer.

				if (RC_BAD( rc = f_realloc( uiNewSize, &m_keyGenInfo.pucData)))
				{
					goto Exit;
				}
			}

			m_keyGenInfo.uiDataBufSize = uiNewSize;
		}
		f_memcpy( m_keyGenInfo.pucData + uiDataLen, ucTmpSen, uiSENLen);
		if (uiDataComponentLen)
		{
			if (RC_BAD( rc = pNode->getData( this,
												m_keyGenInfo.pucData + uiDataLen + uiSENLen,
												&uiDataComponentLen)))
			{
				goto Exit;
			}
		}

		// If this is the last data CDL, append IDs to the
		// key and output the key and data to the KREF.
		// Otherwise, recurse down.

		if (pIcd->pNextDataComponent)
		{
			if (RC_BAD( rc = buildData( pIcd->pNextDataComponent, uiKeyLen,
										uiDataLen + uiDataComponentLen + uiSENLen)))
			{
				goto Exit;
			}
		}
		else if (m_keyGenInfo.pIxd->pFirstContext)
		{
			if (RC_BAD( rc = buildContext( m_keyGenInfo.pIxd->pFirstContext, uiKeyLen,
									uiDataLen + uiDataComponentLen + uiSENLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = verifyKeyContext( &bVerified)))
			{
				goto Exit;
			}
			if (bVerified)
			{
				if (RC_BAD( rc = kyAddIDsToKey( m_keyGenInfo.ui64DocumentID,
											m_keyGenInfo.pIxd,
											m_keyGenInfo.pCdlTbl,
											&m_keyGenInfo.pucKeyBuf [uiKeyLen],
											XFLM_MAX_KEY_SIZE - uiKeyLen, &uiIDLen)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = addToKrefTbl( uiKeyLen + uiIDLen,
										uiDataLen + uiDataComponentLen + uiSENLen)))
				{
					goto Exit;
				}
			}
		}

		// Get the next CDL, if any

		if (pCdl)
		{
			pCdl = pCdl->pNext;
			if (!m_keyGenInfo.bUseSubtreeNodes)
			{

				// If we are not using nodes in the sub-tree, skip any
				// that are in the sub-tree.

				while (pCdl && pCdl->bInNodeSubtree)
				{
					pCdl = pCdl->pNext;
				}
			}
		}
		if (!pCdl)
		{
			goto Exit;
		}
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pFirstCdl;

	return( rc);
}

/****************************************************************************
Desc:		Go through the context components of a key.
Notes:	This routine is recursive in nature.  Will recurse the number of
			context components defined in the index.
****************************************************************************/
RCODE F_Db::buildContext(
	ICD *			pIcd,
	FLMUINT		uiKeyLen,
	FLMUINT		uiDataLen
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCdl;
	CDL *			pFirstCdl;
	CDL *			pCdl;
	FLMUINT		uiIDLen;
	F_DOMNode *	pNode = NULL;
	FLMBOOL		bVerified;

	uiCdl = pIcd->uiCdl;
	pFirstCdl = m_keyGenInfo.pCdlTbl [uiCdl].pCdlList;
	pCdl = pFirstCdl;
	if (!m_keyGenInfo.bUseSubtreeNodes)
	{

		// If we are not using nodes in the sub-tree, skip any
		// that are in the sub-tree.

		while (pCdl && pCdl->bInNodeSubtree)
		{
			pCdl = pCdl->pNext;
		}
	}

	// Go through all of the context CDLs - even the ones that are NULL

	for (;;)
	{
		if (pNode)
		{
			pNode->Release();
			pNode = NULL;
		}
		m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pCdl;
		if (pCdl)
		{

			// NOTE: pNode could be NULL because it is a "missing" placeholder

			if ((pNode = pCdl->pNode) != NULL)
			{
				pNode->AddRef();
			}
		}

		// If this is the last context CDL, append IDs to the
		// key and output the key and data to the KREF.
		// Otherwise, recurse down.

		if (pIcd->pNextKeyComponent)
		{
			if (RC_BAD( rc = buildContext( pIcd->pNextKeyComponent,
										uiKeyLen, uiDataLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = verifyKeyContext( &bVerified)))
			{
				goto Exit;
			}
			if (bVerified)
			{
				if (RC_BAD( rc = kyAddIDsToKey( m_keyGenInfo.ui64DocumentID,
											m_keyGenInfo.pIxd,
											m_keyGenInfo.pCdlTbl,
											&m_keyGenInfo.pucKeyBuf [uiKeyLen],
											XFLM_MAX_KEY_SIZE - uiKeyLen, &uiIDLen)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = addToKrefTbl( uiKeyLen + uiIDLen, uiDataLen)))
				{
					goto Exit;
				}
			}
		}

		// Get the next CDL, if any

		if (pCdl)
		{
			pCdl = pCdl->pNext;
			if (!m_keyGenInfo.bUseSubtreeNodes)
			{

				// If we are not using nodes in the sub-tree, skip any
				// that are in the sub-tree.

				while (pCdl && pCdl->bInNodeSubtree)
				{
					pCdl = pCdl->pNext;
				}
			}
		}
		if (!pCdl)
		{
			goto Exit;
		}
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pFirstCdl;

	return( rc);
}

/****************************************************************************
Desc:	Finish the current key component.  If there is a next one call
		build keys.  Otherwise, go on to doing data and context pieces.
****************************************************************************/
RCODE F_Db::finishKeyComponent(
	ICD *		pIcd,
	FLMUINT	uiKeyLen
	)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (pIcd->pNextKeyComponent)
	{
		flmAssert( m_keyGenInfo.bIsCompound);
		if (RC_BAD( rc = buildKeys( pIcd->pNextKeyComponent, uiKeyLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if (m_keyGenInfo.pIxd->pFirstData)
		{
			if (RC_BAD( rc = buildData( m_keyGenInfo.pIxd->pFirstData, uiKeyLen, 0)))
			{
				goto Exit;
			}
		}
		else if (m_keyGenInfo.pIxd->pFirstContext)
		{
			if (RC_BAD( rc = buildContext( m_keyGenInfo.pIxd->pFirstContext, uiKeyLen, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			FLMBOOL	bVerified;
			
			if (RC_BAD( rc = verifyKeyContext( &bVerified)))
			{
				goto Exit;
			}
			if (bVerified)
			{
				FLMUINT	uiIDLen;

				if (RC_BAD( rc = kyAddIDsToKey( m_keyGenInfo.ui64DocumentID,
											m_keyGenInfo.pIxd,
											m_keyGenInfo.pCdlTbl,
											&m_keyGenInfo.pucKeyBuf [uiKeyLen],
											XFLM_MAX_KEY_SIZE - uiKeyLen,
											&uiIDLen)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = addToKrefTbl( uiKeyLen + uiIDLen, 0)))
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
Desc:	Generate the keys for a text component.
****************************************************************************/
RCODE F_Db::genTextKeyComponents(
	F_DOMNode *	pNode,
	ICD *			pIcd,
	FLMUINT		uiKeyLen,
	FLMBYTE **	ppucTmpBuf,
	FLMUINT *	puiTmpBufSize,
	void **		ppvMark)
{
	RCODE						rc = NE_XFLM_OK;
	IF_PosIStream *		pIStream = NULL;
	FLMUINT					uiNumChars;
	FLMUINT					uiStrBytes;
	FLMUINT					uiSubstrChars;
	FLMUINT					uiMeta;
	FLMBOOL					bEachWord = FALSE;
	FLMBOOL					bMetaphone = FALSE;
	FLMBOOL					bSubstring = FALSE;
	FLMBOOL					bWholeString = FALSE;
	FLMBOOL					bHadAtLeastOneString = FALSE;
	FLMBOOL					bDataTruncated;
	FLMUINT					uiSaveKeyLen;
	FLMUINT					uiElmLen;
	FLMUINT					uiKeyLenPos = uiKeyLen;
	FLMUINT					uiCompareRules = pIcd->uiCompareRules;
	F_NodeBufferIStream	nodeBufferIStream;
	IF_BufferIStream *	pBufferIStream = NULL;
	
	uiKeyLen += 2;
	uiSaveKeyLen = uiKeyLen;
	
	if (!pNode)
	{
		goto No_Strings;
	}
	
	if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pNode->getTextIStream( this, 
		&nodeBufferIStream, &pIStream, &uiNumChars)))
	{
		goto Exit;
	}
	
	if (!uiNumChars)
	{
No_Strings:

		// Save the key component length
		
		UW2FBA( 0, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);

		if( pIStream)
		{
			pIStream->Release();
			pIStream = NULL;
		}
		
		rc = finishKeyComponent( pIcd, uiKeyLen);
		goto Exit;
	}

	if (pIcd->uiFlags & ICD_EACHWORD)
	{
		bEachWord = TRUE;
		
		// OR in the compressing of spaces, because we only want to treat
		// spaces as delimiters between words.
	
		uiCompareRules |= XFLM_COMP_COMPRESS_WHITESPACE; 
	}
	else if (pIcd->uiFlags & ICD_METAPHONE)
	{
		bMetaphone = TRUE;
	}
	else if (pIcd->uiFlags & ICD_SUBSTRING)
	{
		bSubstring = TRUE;
	}
	else
	{
		bWholeString = TRUE;
	}

	// Loop on each word or substring in the value

	for (;;)
	{
		uiKeyLen = uiSaveKeyLen;
		bDataTruncated = FALSE;

		if (bWholeString)
		{
			uiElmLen = XFLM_MAX_KEY_SIZE - uiKeyLen;
			if( RC_BAD( rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
							&uiElmLen,
							pIStream, XFLM_TEXT_TYPE,
							pIcd->uiFlags, pIcd->uiCompareRules,
							pIcd->uiLimit, NULL, NULL,
							m_keyGenInfo.pIxd->uiLanguage,
							FALSE, FALSE,
							&bDataTruncated, NULL)))
			{
				goto Exit;
			}
		}
		else if (bEachWord)
		{
			if (*ppucTmpBuf == NULL)
			{
				*ppvMark = m_tempPool.poolMark();
				*puiTmpBufSize = (FLMUINT)XFLM_MAX_KEY_SIZE + 8;
				if (RC_BAD( rc = m_tempPool.poolAlloc( *puiTmpBufSize,
												(void **)ppucTmpBuf)))
				{
					goto Exit;
				}
			}
	
			uiStrBytes = *puiTmpBufSize;
			if( RC_BAD( rc = KYEachWordParse( pIStream, &uiCompareRules,
				pIcd->uiLimit, *ppucTmpBuf, &uiStrBytes)))
			{
				goto Exit;
			}

			if (!uiStrBytes)
			{
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)*ppucTmpBuf, uiStrBytes)))
			{
				goto Exit;
			}
			
			// Pass 0 for compare rules because KYEachWordParse will already
			// have taken care of them - except for XFLM_COMP_CASE_INSENSITIVE.

			uiElmLen = XFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										pBufferIStream, XFLM_TEXT_TYPE,
										pIcd->uiFlags,
										pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIxd->uiLanguage,
										FALSE, FALSE, &bDataTruncated, NULL);
			pBufferIStream->closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}
		else if (bMetaphone)
		{
			FLMBYTE	ucStorageBuf[ FLM_MAX_NUM_BUF_SIZE];
			FLMUINT	uiStorageLen;

			if( RC_BAD( rc = f_getNextMetaphone( pIStream, &uiMeta)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

			uiStorageLen = FLM_MAX_NUM_BUF_SIZE;
			if( RC_BAD( rc = flmNumber64ToStorage( uiMeta,
				&uiStorageLen, ucStorageBuf, FALSE, FALSE)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)ucStorageBuf, uiStorageLen)))
			{
				goto Exit;
			}

			// Pass 0 for compare rules - only applies to strings.
			
			uiElmLen = XFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										pBufferIStream, XFLM_NUMBER_TYPE,
										pIcd->uiFlags, 0,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIxd->uiLanguage,
										FALSE, FALSE, NULL, NULL);
			pBufferIStream->closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}
		else
		{
			flmAssert( bSubstring);
			if (*ppucTmpBuf == NULL)
			{
				*ppvMark = m_tempPool.poolMark();
				*puiTmpBufSize = (FLMUINT)XFLM_MAX_KEY_SIZE + 8;
				if (RC_BAD( rc = m_tempPool.poolAlloc( *puiTmpBufSize,
												(void **)ppucTmpBuf)))
				{
					goto Exit;
				}
			}
			uiStrBytes = *puiTmpBufSize;
			
			if( RC_BAD( rc = KYSubstringParse( pIStream, &uiCompareRules,
				pIcd->uiLimit, *ppucTmpBuf, &uiStrBytes, &uiSubstrChars)))
			{
				goto Exit;
			}

			if (!uiStrBytes)
			{
				if (!bHadAtLeastOneString)
				{
					goto No_Strings;
				}
				break;
			}

			if (bHadAtLeastOneString && uiSubstrChars == 1 && !m_keyGenInfo.bIsAsia)
			{
				break;
			}

			if (RC_BAD( rc = pBufferIStream->openStream( 
				(const char *)*ppucTmpBuf, uiStrBytes)))
			{
				goto Exit;
			}
			
			// Pass 0 for compare rules, because KYSubstringParse has already
			// taken care of them, except for XFLM_COMP_CASE_INSENSITIVE

			uiElmLen = XFLM_MAX_KEY_SIZE - uiKeyLen;
			rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
										&uiElmLen,
										pBufferIStream, XFLM_TEXT_TYPE,
										pIcd->uiFlags,
										pIcd->uiCompareRules & XFLM_COMP_CASE_INSENSITIVE,
										pIcd->uiLimit,
										NULL, NULL,
										m_keyGenInfo.pIxd->uiLanguage,
										bHadAtLeastOneString ? FALSE : TRUE, FALSE,
										&bDataTruncated, NULL);

			pBufferIStream->closeStream();

			if( RC_BAD( rc))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			bHadAtLeastOneString = TRUE;
		}

		uiKeyLen += uiElmLen;
		
		// Save the key component length
		
		if (!bDataTruncated)
		{
			UW2FBA( (FLMUINT16)uiElmLen, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}
		else
		{
			UW2FBA( (FLMUINT16)(uiElmLen | TRUNCATED_FLAG),
								&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
								
			// If we are deleting, save the node into the list of "old" nodes
			// We do this so that the compare routine can look for the node
			// if it has to compare the full value.
			
			if (!m_keyGenInfo.bAddKeys)
			{
				if (!m_pOldNodeList)
				{
					if ((m_pOldNodeList = f_new F_OldNodeList) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}
				}
				if (RC_BAD( rc = m_pOldNodeList->addNodeToList( this, pNode)))
				{
					goto Exit;
				}
			}
		}

		if (RC_BAD( rc = finishKeyComponent( pIcd, uiKeyLen)))
		{
			goto Exit;
		}

		if (bWholeString)
		{
			break;
		}
	}
	
Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
	}

	if (pIStream)
	{
		pIStream->Release();
		pIStream = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:	Generate the keys for other data types besides text.
****************************************************************************/
RCODE F_Db::genOtherKeyComponent(
	F_DOMNode *	pNode,
	ICD *			pIcd,
	FLMUINT		uiKeyLen)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiElmLen;
	FLMUINT					uiKeyLenPos = uiKeyLen;
	FLMBOOL					bDataTruncated;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	
	uiKeyLen += 2;
	if (!pNode)
	{
		
No_Data:

		// Save the key component length
		
		UW2FBA( 0, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		
		rc = finishKeyComponent( pIcd, uiKeyLen);
		goto Exit;
	}
	
	if (pIcd->uiFlags & ICD_PRESENCE)
	{
		FLMUINT	uiNameId = pIcd->uiDictNum;
		
		// If we are indexing ELM_ROOT_TAG, we
		// need to get the name id from the node.
		
		if (uiNameId == ELM_ROOT_TAG)
		{
			if (RC_BAD( rc = pNode->getNameId( this, &uiNameId)))
			{
				goto Exit;
			}
		}
		
		f_UINT32ToBigEndian( (FLMUINT32)uiNameId, 
			&m_keyGenInfo.pucKeyBuf [uiKeyLen]);
		uiKeyLen += 4;
		
		// Save the key component length.
		
		UW2FBA( 4, &m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
	}
	else
	{
		// If it is not a presence index, we cannot be indexing
		// ELM_ROOT_TAG on an element node.
		
		flmAssert( pIcd->uiDictNum != ELM_ROOT_TAG);
					  
		if (RC_BAD( rc = pNode->getIStream( this, &bufferIStream, &pIStream)))
		{
			goto Exit;
		}

		if (!pIStream->remainingSize())
		{
			goto No_Data;
		}
			
		// Compute number of bytes left

		uiElmLen = XFLM_MAX_KEY_SIZE - uiKeyLen;
		bDataTruncated = FALSE;
		
		// Pass zero for compare rules - these are not strings.
		
		if( RC_BAD( rc = KYCollateValue( &m_keyGenInfo.pucKeyBuf [uiKeyLen],
						&uiElmLen, pIStream, icdGetDataType( pIcd), pIcd->uiFlags,
						0, pIcd->uiLimit, NULL, NULL,
						m_keyGenInfo.pIxd->uiLanguage,
						FALSE, FALSE,
						&bDataTruncated, NULL)))
		{
			goto Exit;
		}
		uiKeyLen += uiElmLen;
		
		// Save the key component length.
		
		if (!bDataTruncated)
		{
			UW2FBA( (FLMUINT16)uiElmLen,
						&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
		}
		else
		{
			UW2FBA( (FLMUINT16)(uiElmLen | TRUNCATED_FLAG),
						&m_keyGenInfo.pucKeyBuf [uiKeyLenPos]);
						
			// If we are deleting, save the node into the list of "old" nodes
			// We do this so that the compare routine can look for the node
			// if it has to compare the full value.
			
			if (!m_keyGenInfo.bAddKeys)
			{
				if (!m_pOldNodeList)
				{
					if ((m_pOldNodeList = f_new F_OldNodeList) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}
				}
				
				if (RC_BAD( rc = m_pOldNodeList->addNodeToList( this, pNode)))
				{
					goto Exit;
				}
			}
		}

		// Better have an F_IStream at this point!

		flmAssert( pIStream);
		pIStream->Release();
		pIStream = NULL;
	}
	
	if (RC_BAD( rc = finishKeyComponent( pIcd, uiKeyLen)))
	{
		goto Exit;
	}
	
Exit:

	if (pIStream)
	{
		pIStream->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:		Build all compound keys from the CDL table.
Notes:	This routine is recursive in nature.  Will recurse the number of
			key components defined in the index.
****************************************************************************/
RCODE F_Db::buildKeys(
	ICD *				pIcd,
	FLMUINT			uiKeyLen
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCdl = pIcd->uiCdl;
	CDL *					pFirstCdl = m_keyGenInfo.pCdlTbl [uiCdl].pCdlList;
	CDL *					pCdl = pFirstCdl;
	FLMBYTE *	   	pucTmpBuf = NULL;
	void *				pvMark = NULL;
	FLMUINT				uiTmpBufSize = 0;
	F_DOMNode *			pNode = NULL;

	flmAssert( m_keyGenInfo.bIsCompound || pIcd->uiKeyComponent == 1);

	// Do each CDL.  If there is no CDL for this level, must
	// still do at least once so that if there are other
	// components or data after this component, we will
	// do a recursive call or generate a key.

	if (!m_keyGenInfo.bUseSubtreeNodes)
	{

		// Skip any CDLs in the subtree if we are not using
		// sub-tree nodes at this point.

		while (pCdl && pCdl->bInNodeSubtree)
		{
			pCdl = pCdl->pNext;
		}
	}
	for (;;)
	{
		
		// Need to set the current CDL into the table so that
		// when we append the IDs we will have the right ones.
		// This will be restored to the first item in the
		// table at Exit.

		if (pNode)
		{
			pNode->Release();
			pNode = NULL;
		}
		m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pCdl;
		if (pCdl)
		{

			// pNode could be NULL because it is a placeholder for a
			// "missing" value

			if ((pNode = pCdl->pNode) != NULL)
			{
				pNode->AddRef();
			}
		}

		// Generate the key component

		if (icdGetDataType( pIcd) == XFLM_TEXT_TYPE &&
			 !(pIcd->uiFlags & ICD_PRESENCE))
		{
			if (RC_BAD( rc = genTextKeyComponents( pNode, pIcd, uiKeyLen,
										&pucTmpBuf, &uiTmpBufSize, &pvMark)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = genOtherKeyComponent( pNode, pIcd, uiKeyLen)))
			{
				goto Exit;
			}
		}
		
		// Go to the next CDL, if any

		if (pCdl)
		{
			pCdl = pCdl->pNext;
			if (!m_keyGenInfo.bUseSubtreeNodes)
			{

				// Skip any CDLs in the subtree if we are not using
				// sub-tree nodes at this point.

				while (pCdl && pCdl->bInNodeSubtree)
				{
					pCdl = pCdl->pNext;
				}
			}
		}
		if (!pCdl)
		{
			break;
		}
	}

Exit:

	if (pNode)
	{
		pNode->Release();
	}

	if (pvMark)
	{
		m_tempPool.poolReset( pvMark);
	}

	// Restore the CDL table entry to point to the
	// beginning of the list

	m_keyGenInfo.pCdlTbl [uiCdl].pCdlList = pFirstCdl;
	return( rc);
}

/****************************************************************************
Desc:	Build all keys from combinations of CDLs.  Add keys to KREF table.
****************************************************************************/
RCODE F_Db::buildKeys(
	FLMUINT64	ui64DocumentID,
	IXD *			pIxd,
	CDL_HDR *	pCdlTbl,
	FLMBOOL		bUseSubtreeNodes,
	FLMBOOL		bAddKeys)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiNumKeysHavingCdl;
	CDL *			pCdl;
	ICD *			pIcd;
	FLMBYTE		ucDataBuf [STACK_DATA_BUF_SIZE];

	m_keyGenInfo.bDataBufAllocated = FALSE;
	
	// Do a quick check to make sure we have all required pieces

	uiNumKeysHavingCdl = 0;
	pIcd = pIxd->pFirstKey;
	while (pIcd)
	{
		pCdl = pCdlTbl [pIcd->uiCdl].pCdlList;
		if (!bUseSubtreeNodes)
		{

			// Skip any nodes in the subtree if we are not using
			// sub-tree nodes at this point.  Also skip any
			// "missing" placeholders.

			while (pCdl && (pCdl->bInNodeSubtree || !pCdl->pNode))
			{
				pCdl = pCdl->pNext;
			}
		}
		else
		{

			// Skip any "missing" placeholders.

			while (pCdl && !pCdl->pNode)
			{
				pCdl = pCdl->pNext;
			}
		}
		if (!pCdl)
		{
			if (pIcd->uiFlags & ICD_REQUIRED_PIECE)
			{
				goto Exit;	// Nothing to generate, a required piece is missing.
			}
		}
		else
		{
			uiNumKeysHavingCdl++;
		}
		pIcd = pIcd->pNextKeyComponent;
	}

	// If none of the key pieces had a CDL, we cannot generate a key either.

	if (!uiNumKeysHavingCdl)
	{
		goto Exit;
	}

	// Build all of the keys
	
							  
	m_keyGenInfo.ui64DocumentID = ui64DocumentID;
	m_keyGenInfo.pIxd = pIxd;
	m_keyGenInfo.bIsAsia = (FLMBOOL)(pIxd->uiLanguage >= FLM_FIRST_DBCS_LANG &&
							  					pIxd->uiLanguage <= FLM_LAST_DBCS_LANG)
											  ? TRUE
											  : FALSE;
	m_keyGenInfo.bIsCompound = pIxd->uiNumKeyComponents > 1 ? TRUE : FALSE;
	m_keyGenInfo.pCdlTbl = pCdlTbl;
	m_keyGenInfo.bUseSubtreeNodes = bUseSubtreeNodes;
	m_keyGenInfo.bAddKeys = bAddKeys;
	m_keyGenInfo.pucKeyBuf = m_pucKrefKeyBuf;
	m_keyGenInfo.pucData = &ucDataBuf [0];
	m_keyGenInfo.uiDataBufSize = sizeof( ucDataBuf);
	m_keyGenInfo.bDataBufAllocated = FALSE;
	if (RC_BAD( rc = buildKeys( pIxd->pFirstKey, 0)))
	{
		goto Exit;
	}

Exit:

	if (m_keyGenInfo.bDataBufAllocated)
	{
		f_free( &m_keyGenInfo.pucData);
		m_keyGenInfo.bDataBufAllocated = FALSE;
	}

	return( rc);
}

/****************************************************************************
Desc:	See if the passed in node has any other siblings with the same ID.
		The passed in node must be an element node.
****************************************************************************/
FSTATIC RCODE kySeeIfRepeatingSibs(
	F_Db *		pDb,
	F_DOMNode *	pNode,
	FLMBOOL *	pbHadRepeatingSib)
{
	RCODE			rc = NE_XFLM_OK;
	F_DOMNode *	pTmpNode = NULL;
	FLMUINT		uiNameId;
	FLMUINT		uiTmpNameId;

	flmAssert( pNode->getNodeType() == ELEMENT_NODE);

	if (RC_BAD( rc = pNode->getNameId( pDb, &uiNameId)))
	{
		goto Exit;
	}

	// Traverse next siblings

	pTmpNode = pNode;
	pTmpNode->AddRef();
	for (;;)
	{
		if( RC_BAD( rc = pTmpNode->getNextSibling( pDb,
										(IF_DOMNode **)&pTmpNode)))
		{
			if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			rc = NE_XFLM_OK;
			break;
		}
		
		if( pTmpNode->getNodeType() == ELEMENT_NODE)
		{
			if (RC_BAD( rc = pTmpNode->getNameId( pDb, &uiTmpNameId)))
			{
				goto Exit;
			}
			if (uiTmpNameId == uiNameId)
			{
				*pbHadRepeatingSib = TRUE;
				goto Exit;
			}
		}
	}

	// Traverse previous siblings

	pTmpNode->Release();
	pTmpNode = pNode;
	pTmpNode->AddRef();
	
	for (;;)
	{
		if( RC_BAD( rc = pTmpNode->getPreviousSibling( pDb,
										(IF_DOMNode **)&pTmpNode)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			rc = NE_XFLM_OK;
			break;
		}
		
		if (pTmpNode->getNodeType() == ELEMENT_NODE)
		{
			if (RC_BAD( rc = pTmpNode->getNameId( pDb, &uiTmpNameId)))
			{
				goto Exit;
			}
			
			if (uiTmpNameId == uiNameId)
			{
				*pbHadRepeatingSib = TRUE;
				goto Exit;
			}
		}
	}

Exit:

	if (pTmpNode)
	{
		pTmpNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Get a child node for current traversal node.
****************************************************************************/
FSTATIC RCODE kyFindChildNode(
	F_Db *			pDb,
	F_Pool *			pPool,
	NODE_TRAV **	ppTrav,
	FLMBOOL *		pbGotChild,
	FLMBOOL *		pbHadRepeatingSib)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	ICD *				pIcd;
	FLMUINT			uiNameId;
	NODE_TRAV *		pTrav = *ppTrav;
	FLMUINT			uiElmIcdCount = 0;
	FLMUINT			uiAttrIcdCount = 0;
	FLMBOOL			bTraverseElms;
	FLMBOOL			bTraverseAttrs;
	ANCHOR_NODE *	pAnchorNode = pTrav->pAnchorNode;
	FLMUINT			uiAttrNameId = 0;
	
	if( pAnchorNode && 
		pTrav->pNode->getNodeId() == pAnchorNode->ui64AnchorNodeId)
	{
		pAnchorNode = pAnchorNode->pNext;
	}
	else
	{
		pAnchorNode = NULL;
	}

	*pbGotChild = FALSE;

	// Determine if we need to traverse elements, attributes,
	// or both.

	pIcd = pTrav->pIcd->pFirstChild;
	while (pIcd)
	{
		if (pIcd->uiFlags & ICD_IS_ATTRIBUTE)
		{
			uiAttrNameId = pIcd->uiDictNum;
			uiAttrIcdCount++;
		}
		else
		{
			uiElmIcdCount++;
		}
		pIcd = pIcd->pNextSibling;

	}
	bTraverseElms = (uiElmIcdCount) ? TRUE : FALSE;
	bTraverseAttrs = (uiAttrIcdCount) ? TRUE : FALSE;

	flmAssert( pTrav->pNode->getNodeType() == ELEMENT_NODE);

	if (bTraverseElms)
	{
		bTraverseElms = FALSE;
		if (RC_BAD( rc = pTrav->pNode->getFirstChild( pDb, 
			(IF_DOMNode **)&pNode)) && rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			if (bTraverseAttrs)
			{
				bTraverseAttrs = FALSE;
				rc = pTrav->pNode->getFirstAttribute( pDb,
								(IF_DOMNode **)&pNode);
			}
			else
			{
				rc = NE_XFLM_OK;
				goto Exit;
			}
		}
	}
	else
	{
		flmAssert( bTraverseAttrs);
		bTraverseAttrs = FALSE;
		
		// If we only have a single attribute we are searching for,
		// it will be quicker to call getAttribute than to cycle through
		// all of the attributes.
		
		if (uiAttrIcdCount == 1)
		{
			rc = pTrav->pNode->getAttribute( (IF_Db *)pDb, uiAttrNameId,
									(IF_DOMNode **)&pNode); 
		}
		else
		{
			rc = pTrav->pNode->getFirstAttribute( pDb, (IF_DOMNode **)&pNode);
		}
	}
	if (RC_BAD( rc))
	{
		if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Follow siblings until we have a match to any of the sibling ICDs
	// Attribute nodes are also considered siblings.

	for (;;)
	{
		eDomNodeType		eNodeType = pNode->getNodeType();

		// Skip any nodes that are not element or attribute nodes.

		if( eNodeType != ELEMENT_NODE && eNodeType != ATTRIBUTE_NODE)
		{
			goto Next_Node;
		}

		if (RC_BAD( rc = pNode->getNameId( pDb, &uiNameId)))
		{
			goto Exit;
		}

		// If the node has the same name id as the anchor node,
		// we need to skip it, unless it is the anchor node itself

		if( pAnchorNode &&
			 uiNameId == pAnchorNode->uiAnchorNameId &&
			 pNode->getNodeType() == pAnchorNode->eNodeType &&
			 pNode->getIxNodeId() != pAnchorNode->ui64AnchorNodeId)
		{

			// Better not be repeated attribute nodes

			flmAssert( pAnchorNode->eNodeType != ATTRIBUTE_NODE);
			if (pAnchorNode->bSeeIfRepeatingSibs)
			{
				*pbHadRepeatingSib = TRUE;
			}
			pIcd = NULL;
		}
		else
		{
			pIcd = pTrav->pIcd->pFirstChild;
			if (pNode->getNodeType() == ELEMENT_NODE)
			{
				while (pIcd &&
						 (pIcd->uiDictNum != uiNameId ||
						  (pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pNextSibling;
				}
			}
			else // pNode->getNodeType() == ATTRIBUTE_NODE
			{
				while (pIcd &&
						 (pIcd->uiDictNum != uiNameId ||
						  !(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pNextSibling;
				}
			}
		}
		if (pIcd)
		{
			if (!pTrav->pChild)
			{
				NODE_TRAV *	pNewTrav;

				if (RC_BAD( rc = pPool->poolCalloc( sizeof( NODE_TRAV),
											(void **)&pNewTrav)))
				{
					goto Exit;
				}
				pNewTrav->pParent = pTrav;
				pTrav->pChild = pNewTrav;
			}
			else
			{
				f_memset( pTrav->pChild, 0, sizeof( NODE_TRAV));
				pTrav->pChild->pParent = pTrav;
			}

			*ppTrav = pTrav = pTrav->pChild;
			pTrav->pNode = pNode;
			pTrav->pNode->AddRef();
			pTrav->pAnchorNode = pAnchorNode;
			pTrav->uiSibIcdElms = uiElmIcdCount;
			pTrav->uiSibIcdAttrs = uiAttrIcdCount;
			pTrav->pIcd = pIcd;
			if (pNode->getNodeType() == ATTRIBUTE_NODE)
			{

				// There will only be one occurrence of any given
				// attribute node, so once we have found it, there
				// is no need to traverse any other siblings if
				// there are no other sibling ICDs that are
				// attributes.  At this point, the number of ICDs
				// that are elements is irrelevant, because we would
				// have already traversed through all of them.

				pTrav->bTraverseSibs = uiAttrIcdCount > 1 ? TRUE : FALSE;

				// Attributes don't have children

				pTrav->bTraverseChildren = FALSE;

				flmAssert( !pIcd->pFirstChild);
			}
			else
			{
				FLMBOOL	bIsAnchor = FALSE;
						
				if( pAnchorNode)
				{
					if( pNode->getNodeId() == pAnchorNode->ui64AnchorNodeId)
					{
						bIsAnchor = TRUE;
					}
				}

				// If there are attribute ICDs or more than one
				// element ICD, or there is exactly one element
				// ICD, but this is the anchor node, then no
				// need to traverse siblings.

				if (uiAttrIcdCount || uiElmIcdCount > 1 || !bIsAnchor)
				{
					pTrav->bTraverseSibs = TRUE;
				}
				else
				{
					pTrav->bTraverseSibs = FALSE;

					// No need to specially traverse siblings searching
					// for repeating siblings if bTraverseSibs is set
					// to TRUE.

					if (bIsAnchor && pAnchorNode->bSeeIfRepeatingSibs &&
						 !(*pbHadRepeatingSib))
					{
						if (RC_BAD( rc = kySeeIfRepeatingSibs( pDb, pNode,
													pbHadRepeatingSib)))
						{
							goto Exit;
						}
					}
				}

				// If we have a child ICD, we need to traverse child nodes.

				pTrav->bTraverseChildren = pIcd->pFirstChild
													? TRUE
													: FALSE;
			}
			*pbGotChild = TRUE;
			goto Exit;	// Will return NE_XFLM_OK
		}

Next_Node:

		// Try the next sibling node

		if (RC_BAD( rc = pNode->getNextSibling( pDb, (IF_DOMNode **)&pNode)))
		{
			if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			rc = NE_XFLM_OK;

			// If we have not yet traversed attributes, do it now.

			if (!bTraverseAttrs)
			{
				break;
			}
			bTraverseAttrs = FALSE;
			
			// If there is only one attribute, it is quicker to call
			// getAttribute than it is to cycle through the attributes.
			
			if (uiAttrIcdCount == 1)
			{
				rc = pTrav->pNode->getAttribute( (IF_Db *)pDb, uiAttrNameId,
										(IF_DOMNode **)&pNode);
			}
			else
			{
				rc = pTrav->pNode->getFirstAttribute( pDb,
												(IF_DOMNode **)&pNode);
			}
			if (RC_BAD( rc))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}
		}
	}

Exit:

	if (!(*pbGotChild))
	{
		pTrav->bTraverseChildren = FALSE;
	}

	if (pNode)
	{
		pNode->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	Get a sibling node for current traversal node.
****************************************************************************/
FSTATIC RCODE kyFindSibNode(
	F_Db *		pDb,
	NODE_TRAV *	pTrav,
	FLMBOOL		bTestFirstNode,
	FLMBOOL *	pbGotSib,
	FLMBOOL *	pbHadRepeatingSib
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	ICD *				pIcd;
	FLMUINT			uiNameId;
	FLMBOOL			bTraverseAttrs = pTrav->uiSibIcdAttrs ? TRUE : FALSE;
	FLMBOOL			bTraverseElms = pTrav->uiSibIcdElms ? TRUE : FALSE;
	FLMBOOL			bGetNextNode;
	ANCHOR_NODE *	pAnchorNode = pTrav->pAnchorNode;
	eDomNodeType	eNodeType;

	*pbGotSib = FALSE;

	pNode = pTrav->pNode;
	pNode->AddRef();
	eNodeType = pNode->getNodeType();

	if( eNodeType == ELEMENT_NODE)
	{
		flmAssert( bTraverseElms);
	}
	else
	{
		flmAssert( eNodeType != DATA_NODE);
		flmAssert( bTraverseAttrs);
		bTraverseAttrs = FALSE;
	}
	bTraverseElms = FALSE;

	// Follow siblings until we have a match to any of the sibling ICDs
	// Attribute nodes are also considered siblings.

	bGetNextNode = bTestFirstNode ? FALSE : TRUE;
	for (;;)
	{
		if (bGetNextNode)
		{
			if (RC_BAD( rc = pNode->getNextSibling( pDb, (IF_DOMNode **)&pNode)))
			{
				if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;

				// If we have not yet traversed attributes, do it now.

				if (!bTraverseAttrs)
				{
					break;
				}
				bTraverseAttrs = FALSE;
				if (RC_OK( rc = pTrav->pNode->getParentNode( pDb,
														(IF_DOMNode **)&pNode)))
				{
					rc = pNode->getFirstAttribute( pDb, (IF_DOMNode **)&pNode);
				}
				if (RC_BAD( rc))
				{
					if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = NE_XFLM_OK;
					}
					goto Exit;
				}
			}
		}
		else
		{

			// Set to TRUE for next time around.

			bGetNextNode = TRUE;
		}

		// Skip any nodes that are not element or attribute nodes

		eNodeType = pNode->getNodeType();

		if( eNodeType != ELEMENT_NODE &&
			 eNodeType != ATTRIBUTE_NODE)
		{
			continue;
		}

		if (RC_BAD( rc = pNode->getNameId( pDb, &uiNameId)))
		{
			goto Exit;
		}

		// If the node has the same name id as the anchor node,
		// we need to skip it, unless it is the anchor node itself

		if (pAnchorNode &&
			 eNodeType == pAnchorNode->eNodeType &&
			 uiNameId == pAnchorNode->uiAnchorNameId &&
			 pNode->getIxNodeId() != pAnchorNode->ui64AnchorNodeId)
		{

			// Better not be repeated attribute nodes

			flmAssert( pAnchorNode->eNodeType != ATTRIBUTE_NODE);
			if (pAnchorNode->bSeeIfRepeatingSibs)
			{
				*pbHadRepeatingSib = TRUE;
			}
			pIcd = NULL;
		}
		else if( eNodeType == ELEMENT_NODE)
		{

			// Search forward for a matching ICD.

			pIcd = pTrav->pIcd;
			while (pIcd &&
					 (pIcd->uiDictNum != uiNameId ||
					  (pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
			{
				pIcd = pIcd->pNextSibling;
			}

			// If didn't find a matching ICD in the forward direction,
			// search backward.

			if (!pIcd)
			{
				pIcd = pTrav->pIcd->pPrevSibling;
				while (pIcd &&
						 (pIcd->uiDictNum != uiNameId ||
						  (pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pPrevSibling;
				}
			}
		}
		else
		{
			flmAssert( eNodeType == ATTRIBUTE_NODE);

			// Search forward for a matching ICD.

			pIcd = pTrav->pIcd;
			while (pIcd &&
					 (pIcd->uiDictNum != uiNameId ||
					  !(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
			{
				pIcd = pIcd->pNextSibling;
			}

			// If didn't find a matching ICD in the forward direction,
			// search backward.

			if (!pIcd)
			{
				pIcd = pTrav->pIcd->pPrevSibling;
				while (pIcd &&
						 (pIcd->uiDictNum != uiNameId ||
						  !(pIcd->uiFlags & ICD_IS_ATTRIBUTE)))
				{
					pIcd = pIcd->pPrevSibling;
				}
			}
		}

		if (pIcd)
		{
			pTrav->pNode->Release();
			pTrav->pNode = pNode;
			pTrav->pNode->AddRef();
			pTrav->pIcd = pIcd;

			if( eNodeType == ATTRIBUTE_NODE)
			{

				// Attributes don't have children

				pTrav->bTraverseChildren = FALSE;
				flmAssert( !pIcd->pFirstChild);
			}
			else
			{
				// If we have a child ICD, we need to traverse child nodes.

				pTrav->bTraverseChildren = pIcd->pFirstChild
													? TRUE
													: FALSE;
			}

			*pbGotSib = TRUE;
			goto Exit;
		}
	}

Exit:

	if (!(*pbGotSib))
	{
		pTrav->bTraverseSibs = FALSE;
	}

	if (pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Release all nodes a CDL table is pointing to.
****************************************************************************/
void kyReleaseCdls(
	IXD *			pIxd,
	CDL_HDR *	pCdlTbl
	)
{
	FLMUINT	uiLoop;
	CDL *		pCdl;

	if (pCdlTbl)
	{
		for (uiLoop = 0; uiLoop < pIxd->uiNumIcds; uiLoop++)
		{
			pCdl = pCdlTbl [uiLoop].pCdlList;
			while (pCdl)
			{

				// NOTE: pNode can be NULL because it may be a
				// placeholder for a "missing" component

				if (pCdl->pNode)
				{
					pCdl->pNode->Release();
				}
				pCdl = pCdl->pNext;
			}
			pCdlTbl [uiLoop].pCdlList = NULL;
		}
	}
}

/****************************************************************************
Desc:	Generate index keys from a given node for a particular
		index.
****************************************************************************/
RCODE F_Db::genIndexKeys(
	FLMUINT64			ui64DocumentID,
	F_DOMNode *			pIxNode,
	IXD *					pIxd,
	ICD *					pIcd,
	IxAction				eAction)
{
	RCODE					rc = NE_XFLM_OK;
	ICD *					pTmpIcd;
	ICD *					pChildIcd;
	CDL_HDR *			pCdlTbl = NULL;
	CDL *					pCdl;
	ANCHOR_NODE *		pAnchorNodeList;
	ANCHOR_NODE *		pAnchorNode;
	NODE_TRAV *			pTrav = NULL;
	FLMBOOL				bGotNode;
	F_DOMNode *			pParent = NULL;
	F_DOMNode *			pTmpNode = NULL;
	FLMUINT				uiParentNameId;
	FLMUINT				uiIxNodeName;
	FLMBOOL				bInNodeSubtree;
	FLMBOOL				bHadRepeatingSib = FALSE;
	eDomNodeType		eIxNodeType = pIxNode->getNodeType();
	FLMUINT64			ui64IxNodeId = pIxNode->getIxNodeId();

	pAnchorNodeList = NULL;

	if( RC_BAD( rc = pIxNode->getNameId( this, &uiIxNodeName)))
	{
		goto Exit;
	}
	
	// Follow links back up to highest parent.  If the path doesn't
	// match what we have in the index definition, then this node
	// is irrelevant to generating keys in this index.

	if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( ANCHOR_NODE),
												(void **)&pAnchorNode)))
	{
		goto Exit;
	}
	
	pAnchorNode->eNodeType = eIxNodeType;
	pAnchorNode->ui64AnchorNodeId = ui64IxNodeId;
	pAnchorNode->uiAnchorNameId = uiIxNodeName;

	// If the action is link/unlink, and the node is an element node
	// because it is possible for elements to be repeated, we need
	// to see if there is another sibling node of this same name.
	// If there is another sibling with this same name, we do not
	// need to add the "without" keys back in on the unlink action.
	// Nor do we need to delete the "without" keys on the link
	// action.

	if ((eAction == IX_UNLINK_NODE || eAction == IX_LINK_NODE) &&
		 eIxNodeType == ELEMENT_NODE)
	{
		pAnchorNode->bSeeIfRepeatingSibs = TRUE;
	}
	
	pAnchorNodeList = pAnchorNode;
	pTmpIcd = pIcd;
	pParent = pIxNode;
	pParent->AddRef();
	
	while (pTmpIcd->pParent)
	{
		pTmpIcd = pTmpIcd->pParent;
		if (RC_BAD( rc = pParent->getParentNode( this, (IF_DOMNode **)&pParent)))
		{
			if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
		
		if (RC_BAD( rc = pParent->getNameId( this, &uiParentNameId)))
		{
			goto Exit;
		}
		
		if (pTmpIcd->uiDictNum == ELM_ROOT_TAG)
		{
			// If this node has a parent, it is not the root node, so
			// we don't match the ICD.

			if (pParent->getParentId())
			{
				goto Exit;		// Will return NE_XFLM_OK
			}

			// Better not be any more parent ICDs.

			flmAssert( !pTmpIcd->pParent);
		}
		else
		{
			if (pTmpIcd->uiDictNum != uiParentNameId)
			{
				goto Exit;		// Will return NE_XFLM_OK
			}
		}
		
		if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( ANCHOR_NODE),
													(void **)&pAnchorNode)))
		{
			goto Exit;
		}
		
		pAnchorNode->eNodeType = pParent->getNodeType();
		flmAssert( pAnchorNode->eNodeType == ELEMENT_NODE);
		pAnchorNode->ui64AnchorNodeId = pParent->getNodeId();
		pAnchorNode->uiAnchorNameId = uiParentNameId;
		pAnchorNode->pNext = pAnchorNodeList;
		pAnchorNodeList = pAnchorNode;
	}

	// Allocate a CDL table for the index.

	if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( CDL_HDR) *
								pIxd->uiNumIcds, (void **)&pCdlTbl)))
	{
		goto Exit;
	}

	// Create a traversal node for the root node we arrived at.

	if (RC_BAD( rc = m_tempPool.poolCalloc( sizeof( NODE_TRAV),
											(void **)&pTrav)))
	{
		goto Exit;
	}

	pTrav->pNode = pParent;
	pTrav->pNode->AddRef();
	pTrav->pIcd = pTmpIcd;
	pTrav->pAnchorNode = pAnchorNodeList;

	// Count the number of sibling ICDs that are attributes versus elements

	while (pTmpIcd)
	{
		if (pTmpIcd->uiFlags & ICD_IS_ATTRIBUTE)
		{
			pTrav->uiSibIcdAttrs++;
		}
		else
		{
			pTrav->uiSibIcdElms++;
		}
		pTmpIcd = pTmpIcd->pNextSibling;
	}

	pTmpIcd = pTrav->pIcd->pPrevSibling;

	while (pTmpIcd)
	{
		if (pTmpIcd->uiFlags & ICD_IS_ATTRIBUTE)
		{
			pTrav->uiSibIcdAttrs++;
		}
		else
		{
			pTrav->uiSibIcdElms++;
		}

		pTmpIcd = pTmpIcd->pPrevSibling;
	}

#ifdef FLM_DEBUG
	if (pTrav->pIcd->uiDictNum == ELM_ROOT_TAG)
	{
		flmAssert( !pTrav->uiSibIcdAttrs && pTrav->uiSibIcdElms == 1);
		flmAssert( pTrav->pNode->getNodeType() == ELEMENT_NODE);
	}
#endif

	// See if we need to do this node's siblings.  If it is an
	// attribute node and there are no other attribute ICDs and
	// no element ICDs, or if it is an element node and there are
	// no other element ICDs and no attribute ICDs, we don't need
	// to do the node's siblings.  Otherwise, we do, so we will
	// go up to the node's parent and back down to the first
	// child element or first attribute.

	pTrav->bTraverseSibs = FALSE;
	if ((pTrav->pNode->getNodeType() == ATTRIBUTE_NODE &&
		  (pTrav->uiSibIcdAttrs > 1 || pTrav->uiSibIcdElms)) ||
		 (pTrav->pNode->getNodeType() == ELEMENT_NODE &&
		  (pTrav->uiSibIcdAttrs || pTrav->uiSibIcdElms > 1)))
	{
		if (RC_BAD( rc = pTrav->pNode->getParentNode( this,
			(IF_DOMNode **)&pTmpNode)))
		{
			if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			rc = NE_XFLM_OK;
		}
		else if ((pTrav->pNode->getNodeType() == ATTRIBUTE_NODE &&
					 pTrav->uiSibIcdElms) ||
					(pTrav->pNode->getNodeType() == ELEMENT_NODE &&
					pTrav->uiSibIcdElms > 1))
		{
			if (RC_BAD( rc = pTmpNode->getFirstChild( this,
				(IF_DOMNode **)&pTmpNode)))
			{
				if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				rc = NE_XFLM_OK;

				// No sibling elements, do we need to do attributes?

				if ((pTrav->pNode->getNodeType() == ATTRIBUTE_NODE &&
					  pTrav->uiSibIcdAttrs > 1) ||
					 (pTrav->pNode->getNodeType() == ELEMENT_NODE &&
					  pTrav->uiSibIcdAttrs))
				{
					goto Get_First_Attribute;
				}
			}
			else
			{
				pTrav->pNode->Release();
				pTrav->pNode = pTmpNode;
				pTrav->pNode->AddRef();
				pTmpNode->Release();
				pTmpNode = NULL;
				pTrav->bTraverseSibs = TRUE;
			}
		}
		else
		{
Get_First_Attribute:
			if (RC_BAD( rc = pTmpNode->getFirstAttribute( this,
											(IF_DOMNode **)&pTmpNode)))
			{
				if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}
			else
			{
				pTrav->pNode->Release();
				pTrav->pNode = pTmpNode;
				pTrav->pNode->AddRef();
				pTmpNode->Release();
				pTmpNode = NULL;
				pTrav->bTraverseSibs = TRUE;
			}
		}
	}

	// Position to a sibling node to start on.  At this point, we should
	// be guaranteed to at least find the node we started on

	if (pTrav->bTraverseSibs)
	{
		if (RC_BAD( rc = kyFindSibNode( this, pTrav, TRUE, &bGotNode,
													&bHadRepeatingSib)))
		{
			goto Exit;
		}
		flmAssert( bGotNode);
	}
	else if (pTrav->pNode->getNodeType() == ATTRIBUTE_NODE)
	{
		pTrav->bTraverseChildren = FALSE;
	}
	else	// ELEMENT_NODE
	{

		// If we have a child ICD, we need to traverse child nodes.

		pTrav->bTraverseChildren = pTrav->pIcd->pFirstChild
											? TRUE
											: FALSE;

		// If this is our primary anchor node, we need to see
		// if there are repeating siblings.  No need to do it
		// in the other above cases, because if bTraversSibs is
		// TRUE, it will happen inside kyFindSibNode, and if
		// the node type is an attribute node, they should never
		// have repeated siblings.

		if (pTrav->pAnchorNode &&
			 pTrav->pAnchorNode->bSeeIfRepeatingSibs)
		{
			if (RC_BAD( rc = kySeeIfRepeatingSibs( this, pTrav->pNode,
										&bHadRepeatingSib)))
			{
				goto Exit;
			}
		}
	}

	// Follow all links from the current node and ICD,
	// populating CDL tables as we go.
	
	if( pTrav->pNode->getNodeType() == eIxNodeType && 
		 pTrav->pNode->getNameId() == uiIxNodeName &&
		 pTrav->pNode->getIxNodeId() == ui64IxNodeId)
	{
		pTrav->bInNodeSubtree = TRUE;
	}
	
	for (;;)
	{

		pCdl = pCdlTbl [pTrav->pIcd->uiCdl].pCdlList;
		if (pCdl && !pCdl->pNode &&
			 pTrav->pNode->getParentId() &&
			 pCdl->ui64ParentId == pTrav->pNode->getParentId())
		{
			pCdl->pNode = pTrav->pNode;
			pCdl->bInNodeSubtree = pTrav->bInNodeSubtree;
			pCdl->pNode->AddRef();
		}
		else
		{
			if (RC_BAD( rc = m_tempPool.poolAlloc( sizeof( CDL),
											(void **)&pCdl)))
			{
				goto Exit;
			}

			pCdl->pNode = pTrav->pNode;
			pCdl->ui64ParentId = pTrav->pNode->getParentId();
			pCdl->bInNodeSubtree = pTrav->bInNodeSubtree;
			pCdl->pNode->AddRef();
			pCdl->pNext = pCdlTbl [pTrav->pIcd->uiCdl].pCdlList;
			pCdlTbl [pTrav->pIcd->uiCdl].pCdlList = pCdl;
		}

		// Add "missing" place-holders for any child ICDs

		pChildIcd = pTrav->pIcd->pFirstChild;
		while (pChildIcd)
		{
			if (RC_BAD( rc = m_tempPool.poolAlloc(
										sizeof( CDL), (void **)&pCdl)))
			{
				goto Exit;
			}
			
			pCdl->pNode = NULL;
			pCdl->bInNodeSubtree = pTrav->bInNodeSubtree;
			pCdl->ui64ParentId = pTrav->pNode->getNodeId();
			pCdl->pNext = pCdlTbl [pChildIcd->uiCdl].pCdlList;
			pCdlTbl [pChildIcd->uiCdl].pCdlList = pCdl;
			pChildIcd = pChildIcd->pNextSibling;
		}

Next_Node:

		if (pTrav->bTraverseChildren)
		{
			bInNodeSubtree = pTrav->bInNodeSubtree;
			if (RC_BAD( rc = kyFindChildNode( this, &m_tempPool,
									&pTrav, &bGotNode, &bHadRepeatingSib)))
			{
				goto Exit;
			}
			
			if (!bGotNode)
			{
				goto Next_Node;
			}
			
			if( bInNodeSubtree || 
				 (pTrav->pNode->getNodeType() == eIxNodeType &&
				  pTrav->pNode->getNameId() == uiIxNodeName &&
				  pTrav->pNode->getIxNodeId() == ui64IxNodeId))
			{
				pTrav->bInNodeSubtree = TRUE;
			}
		}
		else if (pTrav->bTraverseSibs)
		{
			// Here we are using bInNodeSubtree to indicate whether the entire
			// sibling list is in the node's subtree.  It could be that the
			// current node is the subtree, but that does not mean that the
			// entire sibling list is in the sub-tree.

			if (!pTrav->bInNodeSubtree)
			{
				bInNodeSubtree = FALSE;
			}
			else
			{
				// Is this sibling list in the node subtree?
				// If bInNodeSubtree is TRUE, and this is not the original
				// sub-tree node, then this sibling list has to be subordinate
				// to the original sub-tree node.

				if( pTrav->pNode->getNodeType() != eIxNodeType || 
					pTrav->pNode->getNameId() != uiIxNodeName ||
					pTrav->pNode->getIxNodeId() != ui64IxNodeId)
				{
					bInNodeSubtree = TRUE;
				}
				else
				{
					bInNodeSubtree = FALSE;
				}
			}
			
			if (RC_BAD( rc = kyFindSibNode( this, pTrav, FALSE, &bGotNode,
										&bHadRepeatingSib)))
			{
				goto Exit;
			}
			
			if (!bGotNode)
			{
				goto Next_Node;
			}

			// If the entire sibling list was in the node sub-tree or this
			// node is the subtree, then we are in the subtree.

			if( bInNodeSubtree || 
				 (pTrav->pNode->getNodeType() == eIxNodeType &&
				  pTrav->pNode->getNameId() == uiIxNodeName &&
				  pTrav->pNode->getIxNodeId() == ui64IxNodeId))
			{
				pTrav->bInNodeSubtree = TRUE;
			}
			else
			{
				pTrav->bInNodeSubtree = FALSE;
			}
		}
		else
		{
			if (pTrav->pParent)
			{
				pTrav->pNode->Release();
				pTrav->pNode = NULL;
				pTrav = pTrav->pParent;
				pTrav->bTraverseChildren = FALSE;
				goto Next_Node;
			}
			else
			{
				break;
			}
		}
	}

	// Generate the keys from the combinations of CDLs.

	switch (eAction)
	{
		case IX_UNLINK_NODE:
		{
			if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
													TRUE, FALSE)))
			{
				goto Exit;
			}

			if (!bHadRepeatingSib)
			{
				if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
														FALSE, TRUE)))
				{
					goto Exit;
				}
			}
			break;
		}

		case IX_LINK_NODE:
		{
			if( !bHadRepeatingSib)
			{
				if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
														FALSE, FALSE)))
				{
					goto Exit;
				}
			}

			if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
													TRUE, TRUE)))
			{
				goto Exit;
			}
			break;
		}

		case IX_DEL_NODE_VALUE:
		{
			if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
										TRUE, FALSE)))
			{
				goto Exit;
			}
			break;
		}

		case IX_ADD_NODE_VALUE:
		{
			if (RC_BAD( rc = buildKeys( ui64DocumentID, pIxd, pCdlTbl,
										TRUE, TRUE)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

Exit:

	// Release the nodes CDL table

	kyReleaseCdls( pIxd, pCdlTbl);

	// Release any nodes in the traversal list.

	if (pTrav)
	{

		// Go to top of list

		while (pTrav->pParent)
		{
			pTrav = pTrav->pParent;
		}

		// Visit each NODE_TRAV node and release whatever
		// DOM node it is pointing to.

		while (pTrav)
		{
			if (pTrav->pNode)
			{
				pTrav->pNode->Release();
				pTrav->pNode = NULL;
			}
			pTrav = pTrav->pChild;
		}
	}

	if (pParent)
	{
		pParent->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Routine that is called before and after changing a node - so we can
		delete and add any keys from the database.
****************************************************************************/
RCODE F_Db::updateIndexKeys(
	FLMUINT			uiCollectionNum,
	F_DOMNode *		pIxNode,
	IxAction			eAction,
	FLMBOOL			bStartOfUpdate,
	FLMBOOL *		pbIsIndexed)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bIsRoot;
	FLMBOOL			bIsIndexed = FALSE;
	ICD *				pIcd;
	FLMUINT			uiIcdDictNum;
	IXD *				pIxd;
	void *			pvMark = NULL;
	FLMUINT64		ui64DocumentID;
	FLMUINT			uiNameId;
	F_DOMNode *		pNode = NULL;
	F_AttrElmInfo	defInfo;
	IxAction			eTmpAction;

	if( bStartOfUpdate)
	{
		if (RC_BAD( rc = krefCntrlCheck()))
		{
			goto Exit;
		}
		if (m_pOldNodeList)
		{
			m_pOldNodeList->resetList();
		}
	}
	
	if( RC_BAD( rc = pIxNode->getNameId( this, &uiNameId)))
	{
		goto Exit;
	}

	if( !uiNameId)
	{
		goto Exit;
	}

	pNode = pIxNode;
	pNode->AddRef();

	// Lookup the node's ICD list

	switch( pNode->getNodeType())
	{
		case ELEMENT_NODE:
		{
			if (RC_BAD( rc = m_pDict->getElement( this, uiNameId, &defInfo)))
			{
				goto Exit;
			}
			
			break;
		}

		case ATTRIBUTE_NODE:
		{
			if (RC_BAD( rc = m_pDict->getAttribute( this, uiNameId, &defInfo)))
			{
				goto Exit;
			}
			
			break;
		}

		case DATA_NODE:
		{
			// Need to operate on the parent node, which should be an element
			// node.

			if (RC_BAD( rc = m_pDict->getElement( this, uiNameId, &defInfo)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = pNode->getParentNode( this, (IF_DOMNode **)&pNode)))
			{
				if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					// Data node not yet linked to a parent node, so
					// it won't affect indexing.

					rc = NE_XFLM_OK;
				}
				
				goto Exit;
			}

			// Name ID of element better be the same as what we found in the
			// data node, and node type better be an element node.

			flmAssert( pNode->getNameId() == uiNameId);
			flmAssert( pNode->getNodeType() == ELEMENT_NODE);

			// Action cannot be link/unlink on data nodes.  Higher level needs
			// to take care of.  They should first delete keys on the
			// parent node, then unlink/link the node, then call add keys on
			// the parent node.

			flmAssert( eAction != IX_LINK_NODE && eAction != IX_UNLINK_NODE);
			break;
		}

		default:
		{

			// If the node is not an element or an attribute, it will not have
			// any effect on indexing.

			goto Exit;
		}
	}

	bIsRoot = pNode->isRootNode();

	if( (pIcd = defInfo.m_pFirstIcd) == NULL)
	{
		if( !bIsRoot || !m_pDict->m_pRootIcdList)
		{
			goto Exit;
		}
		else
		{
			pIcd = m_pDict->m_pRootIcdList;
		}
	}

	// The node may be indexed so we need to process the ICD list

	pvMark = m_tempPool.poolMark();
	bIsIndexed = TRUE;
	ui64DocumentID = pNode->getDocumentId();

	// Process each index

	for (;;)
	{
		pIxd = pIcd->pIxd;
		uiIcdDictNum = pIcd->uiDictNum;

		// See if this ICD's index is on the collection we are doing.
		// Also, make sure if we are doing a specific index, that this is
		// that index.

		if (pIxd->uiCollectionNum != uiCollectionNum)
		{
			goto Next_Index;
		}

		// See if this document has been indexed for this index.

		if (ui64DocumentID > pIxd->ui64LastDocIndexed)
		{
			goto Next_Index;
		}

		if ((eTmpAction = eAction) == IX_LINK_AND_ADD_NODE)
		{
			if (!pIcd->pParent && !pIcd->pNextSibling && !pIcd->pPrevSibling)
			{
				if (!pIcd->uiKeyComponent && !pIcd->uiDataComponent)
				{

					// If this ICD is not a key or data component, it has no
					// effect on index keys if we are only modifying its value.

					goto Next_Index;
				}
				else
				{
					eTmpAction = IX_ADD_NODE_VALUE;
				}
			}
			else
			{
				eTmpAction = IX_LINK_NODE;
			}
		}
		else if (eTmpAction == IX_LINK_NODE || eTmpAction == IX_UNLINK_NODE)
		{

			// IX_LINK_NODE and IX_UNLINK_NODE are called when a node is linked
			// to its parent.  Thus, if this ICD is the root ICD, it will have
			// no effect on index keys.

			if (!pIcd->pParent && !pIcd->pNextSibling && !pIcd->pPrevSibling)
			{
				goto Next_Index;
			}
		}
		else
		{
			if (!pIcd->uiKeyComponent && !pIcd->uiDataComponent)
			{

				// If this ICD is not a key or data component, it has no
				// effect on index keys if we are only modifying its value.

				goto Next_Index;
			}
		}

		// Generate index keys for this index.

		if (RC_BAD( rc = genIndexKeys( ui64DocumentID, pNode, pIxd,
									pIcd, eTmpAction)))
		{
			goto Exit;
		}

Next_Index:

		m_tempPool.poolReset( pvMark);
		if ((pIcd = pIcd->pNextInChain) == NULL)
		{
			if (!bIsRoot || uiIcdDictNum == ELM_ROOT_TAG)
			{
				break;
			}

			// When done processing regular ICDs, process the
			// root ICD list, if this node is a root node.

			if ((pIcd = m_pDict->m_pRootIcdList) == NULL)
			{
				break;
			}
		}
	}

Exit:

	if( pvMark)
	{
		m_tempPool.poolReset( pvMark);
	}

	if (pNode)
	{
		pNode->Release();
	}
	
	if( pbIsIndexed)
	{
		*pbIsIndexed = bIsIndexed;
	}

	return( rc);
}
