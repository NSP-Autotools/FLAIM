//------------------------------------------------------------------------------
// Desc:	DOM node implementation
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

// Local constants

#define SEN_RESERVE_BYTES					5

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE F_Db::attrIsInIndexDef(
	FLMUINT			uiAttrNameId,
	FLMBOOL *		pbIsInIndexDef)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrElmInfo	defInfo;

	if( RC_BAD( rc = m_pDict->getAttribute( this, uiAttrNameId, &defInfo)))
	{
		return( rc);
	}

	*pbIsInIndexDef = defInfo.m_pFirstIcd ? TRUE : FALSE;
	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc: This class converts a text stream of Unicode or UTF8 to an ASCII
		text stream.
******************************************************************************/
class F_AsciiIStream : public IF_IStream
{
public:

	F_AsciiIStream(
		FLMBYTE *		pucText,
		FLMUINT			uiNumBytesInBuffer,
		eXFlmTextType	eTextType)
	{
		m_pucText = pucText;
		m_pucCurrPtr = pucText;
		m_uiCurrChar = 0;
		m_eTextType = eTextType;
		
		if( uiNumBytesInBuffer)
		{
			m_pucEnd = &pucText[ uiNumBytesInBuffer];
		}
		else
		{
			m_pucEnd = NULL;
		}
	}
	
	virtual ~F_AsciiIStream()
	{
	}
		
	RCODE XFLAPI read(
		void *					pvBuffer,
		FLMUINT					uiBytesToRead,
		FLMUINT *				puiBytesRead);

	FINLINE RCODE XFLAPI closeStream( void)
	{
		return( NE_XFLM_OK);
	}
	
private:

	const FLMBYTE *	m_pucText;
	const FLMBYTE *	m_pucCurrPtr;
	const FLMBYTE *	m_pucEnd;
	FLMUINT				m_uiCurrChar;
	eXFlmTextType		m_eTextType;
};

/*****************************************************************************
Desc:
******************************************************************************/
FLMINT XFLAPI F_BTreeIStream::Release( void)
{
	FLMATOMIC	refCnt = --m_refCnt;
	
	if (m_refCnt == 0)
	{
		closeStream();
		if( gv_XFlmSysData.pNodePool)
		{
			m_refCnt = 1;
			gv_XFlmSysData.pNodePool->insertBTreeIStream( this);
			return( 0);
		}
		else
		{
			delete this;
		}
	}
	
	return( refCnt);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::canSetValue(
	F_Db *				pDb,
	FLMUINT				uiDataType)
{
	RCODE					rc = NE_XFLM_OK;
	F_Database *		pDatabase = pDb->m_pDatabase;
	IF_DOMNode *		pNode = NULL;
	eDomNodeType		eNodeType = getNodeType();
	
	if( eNodeType < ELEMENT_NODE || eNodeType > ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Cannot set a value without a data type

	if( uiDataType == XFLM_NODATA_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_DATA_TYPE);
		goto Exit;
	}

	// If the node is read-only, don't allow it to be changed
	
	if( getModeFlags() & FDOM_READ_ONLY)
	{
		rc = RC_SET( NE_XFLM_READ_ONLY);
		goto Exit;
	}

	// If this is a comment or CDATA node, only allow text values

	if (uiDataType != XFLM_TEXT_TYPE &&
		 (eNodeType == COMMENT_NODE || eNodeType == CDATA_SECTION_NODE))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// If the node is a data node and it has already been linked
	// into the document, its data type cannot be changed.

	if( getParentId() && eNodeType == DATA_NODE && uiDataType != getDataType())
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// Cannot allow a value to be set on this node if a pending input stream
	// is still open.
	
	if( pDatabase->m_pPendingInput &&
		 pDatabase->m_pPendingInput != m_pCachedNode)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit;
	}
	
Exit:

	if( pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT XFLAPI F_DOMNode::Release( void)
{
	FLMINT	iRefCnt = --m_refCnt;
	
	if (iRefCnt == 0)
	{
		if( gv_XFlmSysData.pNodeCacheMgr)
		{
			m_refCnt = 1;
			gv_XFlmSysData.pNodeCacheMgr->insertDOMNode( this);
			return( 0);
		}
		else
		{
			delete this;
		}
	}

	return( iRefCnt);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::isChildTypeValid(
	eDomNodeType	eChildNodeType)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bTypeValid = FALSE;

	if( !m_pCachedNode)
	{
		rc = RC_SET( NE_XFLM_DOM_INVALID_CHILD_TYPE);
		goto Exit;
	}

	switch( getNodeType())
	{
		case ELEMENT_NODE:
		{
			if (eChildNodeType == ELEMENT_NODE ||
				 (eChildNodeType == DATA_NODE &&
				  getDataType() != XFLM_NODATA_TYPE && getDataLength() == 0) ||
				 eChildNodeType == COMMENT_NODE ||
				 eChildNodeType == PROCESSING_INSTRUCTION_NODE ||
				 eChildNodeType == CDATA_SECTION_NODE)
			{
				bTypeValid = TRUE;
			}
			
			break;
		}
		
		case DOCUMENT_NODE:
		{
			if (eChildNodeType == ELEMENT_NODE ||
				 eChildNodeType == PROCESSING_INSTRUCTION_NODE ||
				 eChildNodeType == COMMENT_NODE)
			{
				bTypeValid = TRUE;
			}
			break;
		}

		case ATTRIBUTE_NODE:		
		case DATA_NODE:
		case CDATA_SECTION_NODE:
		case PROCESSING_INSTRUCTION_NODE:
		case COMMENT_NODE:
		{
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	if (!bTypeValid)
	{
		rc = RC_SET( NE_XFLM_DOM_INVALID_CHILD_TYPE);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::isDescendantOf(
	F_Db *		pDb,
	F_DOMNode *	pAncestor,
	FLMBOOL *	pbDescendant)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pParent = NULL;
	FLMUINT64		ui64AncestorId;
	FLMUINT64		ui64ThisParentId;

	*pbDescendant = FALSE;

	if( !m_pCachedNode)
	{
		goto Exit;
	}
	
	ui64AncestorId = pAncestor->getNodeId();
	ui64ThisParentId = getParentId();

	if( ui64ThisParentId == ui64AncestorId)
	{
		*pbDescendant = TRUE;
		goto Exit;
	}

	if( !ui64ThisParentId ||
		(ui64AncestorId != ui64ThisParentId && 
		 ui64ThisParentId == getDocumentId()) || !pAncestor->getFirstChildId())
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = getParentNode( pDb, (IF_DOMNode **)&pParent)))
	{
		goto Exit;
	}

	while( pParent)
	{
		if( pParent->getNodeId() == ui64AncestorId)
		{
			*pbDescendant = TRUE;
			goto Exit;
		}

		if( RC_BAD( rc = pParent->getParentNode( pDb, (IF_DOMNode **)&pParent)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			goto Exit;
		}
	}

Exit:

	if( pParent)
	{
		pParent->Release();
	}

	return( rc);
}

/*****************************************************************************
Notes:	When an node is unlinked, its document or root node ID is not
			changed.  Once unlinked, the node can be deleted or re-linked
			elsewhere within the same document.
			This routine assumes that the caller has checked the cannot delete
			bits and read-only bits if necessary.
******************************************************************************/
RCODE F_DOMNode::unlinkNode(
	F_Db *		pDb,
	FLMUINT		uiFlags)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pTmpNode = NULL;
	F_COLLECTION *		pCollection = NULL;
	FLMUINT64			ui64OldPrevSib = 0;
	FLMUINT64			ui64OldNextSib = 0;
	FLMBOOL				bChangedThisHeader = FALSE;
	eDomNodeType		eNodeType;

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();

	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// Unlink the node from its siblings and parent

	if( (ui64OldPrevSib = getPrevSibId()) != 0)
	{
		if( RC_BAD( rc = pDb->getNode( getCollection(), getPrevSibId(),
			&pTmpNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}

			goto Exit;
		}

		if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		pTmpNode->setNextSibId( getNextSibId());

		if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, uiFlags)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		setPrevSibId( 0);
		bChangedThisHeader = TRUE;
	}

	if( (ui64OldNextSib = getNextSibId()) != 0)
	{
		if( RC_BAD( rc = pDb->getNode( getCollection(), getNextSibId(),
			&pTmpNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		pTmpNode->setPrevSibId( ui64OldPrevSib);

		if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, uiFlags)))
		{
			goto Exit;
		}

		if( !bChangedThisHeader)
		{
			if( RC_BAD( rc = makeWriteCopy( pDb)))
			{
				goto Exit;
			}
		}
		
		setNextSibId( 0);
		bChangedThisHeader = TRUE;
	}

	if( getParentId())
	{
		FLMBOOL		bChangedTmpHeader = FALSE;

		if( RC_BAD( rc = pDb->getNode( getCollection(), getParentId(),
			&pTmpNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND ||
				 rc == NE_XFLM_DOM_NODE_DELETED)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}
		
		if( eNodeType == ANNOTATION_NODE)
		{
			if (!bChangedTmpHeader)
			{
				if (RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
				{
					goto Exit;
				}
				bChangedTmpHeader = TRUE;
			}
			
			pTmpNode->setAnnotationId( 0);
		}
		else
		{
			// If the parent node is one whose child elements must all
			// be unique, we must remove the node from the node list of the
			// parent.
			
			if( pTmpNode->getModeFlags() & FDOM_HAVE_CELM_LIST)
			{
				FLMUINT	uiElmOffset;
				
				if( !bChangedTmpHeader)
				{
					if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
					{
						goto Exit;
					}
					bChangedTmpHeader = TRUE;
				}
				
				// Only element nodes should be child nodes of this parent.
				
				flmAssert( eNodeType == ELEMENT_NODE);
				
				if( !pTmpNode->findChildElm( getNameId(), &uiElmOffset))
				{
					// Child node should have been found.
					
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
				
				if( RC_BAD( rc = pTmpNode->removeChildElm( uiElmOffset)))
				{
					goto Exit;
				}
			}
			
			// If this is a data node, we need to update the parent element's
			// data node count
			
			if( eNodeType == DATA_NODE)
			{
				if( !bChangedTmpHeader)
				{
					if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
					{
						goto Exit;
					}
					bChangedTmpHeader = TRUE;
				}
				
				flmAssert( pTmpNode->getDataChildCount());
				pTmpNode->setDataChildCount( pTmpNode->getDataChildCount() - 1);
			}

			if( !ui64OldPrevSib)
			{
				if( !bChangedTmpHeader)
				{
					if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
					{
						goto Exit;
					}
					bChangedTmpHeader = TRUE;
				}
				
				flmAssert( pTmpNode->canHaveChildren());
				pTmpNode->setFirstChildId( ui64OldNextSib);
			}
	
			if( !ui64OldNextSib)
			{
				if( !bChangedTmpHeader)
				{
					if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
					{
						goto Exit;
					}
					bChangedTmpHeader = TRUE;
				}
				
				flmAssert( pTmpNode->canHaveChildren());
				pTmpNode->setLastChildId( ui64OldPrevSib);
			}
		}

		if( bChangedTmpHeader)
		{
			if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, uiFlags)))
			{
				goto Exit;
			}
		}
		
		if( !bChangedThisHeader)
		{
			if( RC_BAD( rc = makeWriteCopy( pDb)))
			{
				goto Exit;
			}
		}

		setParentId( 0);
		bChangedThisHeader = TRUE;
	}

	// If this is a root node, the document list pointers may need
	// to be updated

	if( isRootNode())
	{
		if( eNodeType != DOCUMENT_NODE && eNodeType != ELEMENT_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		// Get a pointer to the collection

		if( RC_BAD( rc = pDb->m_pDict->getCollection(
			getCollection(), &pCollection)))
		{
			goto Exit;
		}

		if( pCollection->ui64FirstDocId == getNodeId() ||
			pCollection->ui64LastDocId == getNodeId())
		{
			// Clone the dictionary since the first/last document ID
			// values of the collection will be changed

			if( !(pDb->m_uiFlags & FDB_UPDATED_DICTIONARY))
			{
				if( RC_BAD( rc = pDb->dictClone()))
				{
					goto Exit;
				}
			}

			// Get a pointer to the new collection

			if( RC_BAD( rc = pDb->m_pDict->getCollection(
				getCollection(), &pCollection)))
			{
				goto Exit;
			}

			// Change the first and/or last document IDs

			if( pCollection->ui64FirstDocId == getNodeId())
			{
				pCollection->ui64FirstDocId = ui64OldNextSib;
				pCollection->bNeedToUpdateNodes = TRUE;
			}

			if( pCollection->ui64LastDocId == getNodeId())
			{
				pCollection->ui64LastDocId = ui64OldPrevSib;
				pCollection->bNeedToUpdateNodes = TRUE;
			}
		}
	}

	// If this is a data node, clear its name tag

	if( eNodeType == DATA_NODE)
	{
		if( !bChangedThisHeader)
		{
			if( RC_BAD( rc = makeWriteCopy( pDb)))
			{
				goto Exit;
			}
		}
		
		setNameId( 0);
		bChangedThisHeader = TRUE;
	}

	if( bChangedThisHeader)
	{
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, uiFlags)))
		{
			goto Exit;
		}
	}
	
Exit:

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::addModeFlags(
	F_Db *			pDb,
	FLMUINT			uiFlags)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	F_Rfl *			pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT			uiRflToken = 0;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// Only need to set flags if they are not all currently set.

	if( (getModeFlags() & uiFlags) != uiFlags)
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		if( getNodeType() == ATTRIBUTE_NODE)
		{
			if( RC_BAD( rc = m_pCachedNode->addModeFlags( 
				pDb, m_uiAttrNameId, uiFlags)))
			{
				goto Exit;
			}
		}
		else
		{
			m_pCachedNode->setFlags( uiFlags);
		}
		
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0))) 
		{
			goto Exit;
		}

		pRfl->enableLogging( &uiRflToken);

		if( RC_BAD( pRfl->logNodeFlagsUpdate( 
			pDb, getCollection(), m_pCachedNode->getNodeId(), 
			m_uiAttrNameId, uiFlags, TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::removeModeFlags(
	F_Db *			pDb,
	FLMUINT			uiFlags)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	F_Rfl *			pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT			uiRflToken = 0;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// Only need to remove the flags if any of them are currently set.
	
	if( getModeFlags() & uiFlags)
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		if( getNodeType() == ATTRIBUTE_NODE)
		{
			if( RC_BAD( rc = m_pCachedNode->removeModeFlags( 
				pDb, m_uiAttrNameId, uiFlags)))
			{
				goto Exit;
			}
		}
		else
		{
			m_pCachedNode->unsetFlags( uiFlags);
		}
		
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		pRfl->enableLogging( &uiRflToken);

		if( RC_BAD( pRfl->logNodeFlagsUpdate( 
			pDb, getCollection(), m_pCachedNode->getNodeId(), 
			m_uiAttrNameId, uiFlags, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getData(
	F_Db *			pDb,
	FLMBYTE *		pucBuffer,
	FLMUINT *		puiLength)
{
	RCODE						rc = NE_XFLM_OK;
	IF_PosIStream *		pIStream = NULL;
	FLMBOOL					bStartedTrans = FALSE;
	F_NodeBufferIStream	bufferIStream;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// If a NULL buffer is passed in, just return the
	// data length

	if( !pucBuffer)
	{
		rc = getDataLength( pDb, puiLength);
		goto Exit;
	}

	if( RC_BAD( rc = getIStream( pDb, &bufferIStream, &pIStream)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmReadStorageAsBinary( pIStream, pucBuffer,
		*puiLength, 0, puiLength)))
	{
		goto Exit;
	}

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getDataLength(
	IF_Db *			ifpDb,
	FLMUINT *		puiLength)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	eDomNodeType	eNodeType;
	F_Db *			pDb = (F_Db *)ifpDb;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	eNodeType = getNodeType();

	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getDataLength(
			m_uiAttrNameId, puiLength)))
		{
			goto Exit;
		}
	}
	else if( (*puiLength = getDataLength()) == 0)
	{
		// If this is an element node, we will automatically search
		// for the first data node (if any)

		if( eNodeType == ELEMENT_NODE && getDataChildCount())
		{
			F_DOMNode *	pNode = NULL;

			if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				
				goto Exit;
			}

			*puiLength = pNode->getDataLength();
			pNode->Release();
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::insertBefore(
	IF_Db *				ifpDb,
	IF_DOMNode *		ifpNewChild,
	IF_DOMNode *		ifpRefChild)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pFirstNewChild = NULL;
	F_DOMNode *			pLastNewChild = NULL;
	F_DOMNode *			pCurNewNode = NULL;
	F_DOMNode *			pNextSib = NULL;
	F_DOMNode *			pInsertBefore = NULL;
	F_DOMNode *			pTmpNode = NULL;
	FLMBOOL				bDescendant;
	FLMBOOL				bDone = FALSE;
	FLMUINT64			ui64Tmp;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bStartOfUpdate;
	F_DOMNode *			pDataElementNode = NULL;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	F_DOMNode *			pNewChild = (F_DOMNode *)ifpNewChild;
	F_DOMNode *			pRefChild = (F_DOMNode *)ifpRefChild;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bUpdatedNode = FALSE;
	FLMUINT				uiRflToken = 0;
	FLMUINT64			ui64RefChildId = 0;
	eDomNodeType		eThisNodeType;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);
	
	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eThisNodeType = getNodeType();

	if( eThisNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	if( !pNewChild)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = pNewChild->syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( pRefChild)
	{
		if( RC_BAD( rc = pRefChild->syncFromDb( pDb)))
		{
			goto Exit;
		}
		
		if( pRefChild->getNodeType() == ATTRIBUTE_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
		
		ui64RefChildId = pRefChild->getNodeId();
	}
	
	if( pNewChild->getDatabase() != getDatabase() ||
		pNewChild->getCollection() != getCollection())
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( pNewChild->getNodeType() == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// If this is a node whose children are all supposed to be unique,
	// make sure that is the case, and find out where to insert the node.
	
	if( getModeFlags() & FDOM_HAVE_CELM_LIST)
	{
		FLMUINT	uiInsertPos;
		
		// Only element child nodes are allowed.
		
		if( pNewChild->getNodeType() != ELEMENT_NODE)
		{
			rc = RC_SET( NE_XFLM_DOM_INVALID_CHILD_TYPE);
			goto Exit;
		}
		
		// All of the element names must be unique.
		
		if( m_pCachedNode->findChildElm( pNewChild->getNameId(),
										&uiInsertPos))
		{
			rc = RC_SET( NE_XFLM_DOM_DUPLICATE_ELEMENT);
			goto Exit;
		}
		
		// Element was not found, insert into the list of elements.
		
		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		bUpdatedNode = TRUE;
		
		if( RC_BAD( rc = m_pCachedNode->insertChildElm( uiInsertPos,
										pNewChild->getNameId(),
										pNewChild->getNodeId())))
		{
			goto Exit;
		}
	}

	// If a non-NULL reference child was passed in,
	// do some basic sanity checks

	if( pRefChild)
	{
		if( pRefChild->getDatabase() != getDatabase() ||
			pRefChild->getCollection() != getCollection())
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}

		if( pNewChild->getNodeId() == pRefChild->getNodeId())
		{
			rc = RC_SET( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
			goto Exit;
		}

		if( pRefChild->getParentId() != getNodeId())
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
	}

	pFirstNewChild = pNewChild;
	pFirstNewChild->AddRef();

	pLastNewChild = pNewChild;
	pLastNewChild->AddRef();

	if( pRefChild)
	{
		pInsertBefore = pRefChild;
		pInsertBefore->AddRef();
	}

	pCurNewNode = pFirstNewChild;
	pCurNewNode->AddRef();

	bStartOfUpdate = TRUE;
	for( ;;)
	{
		// Make sure it is legal for the new child node to be
		// linked to this node

		if( RC_BAD( rc = isChildTypeValid( pCurNewNode->getNodeType())))
		{
			goto Exit;
		}

		// If the node being inserted is not from the same document, 
		// we cannot perform the operation

		if( pCurNewNode->getDocumentId() != getDocumentId())
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_WRONG_DOCUMENT_ERR);
			goto Exit;
		}

		// A document node can only have one element node and one document type
		// node.

		if( eThisNodeType == DOCUMENT_NODE)
		{
			if( pCurNewNode->getNodeType() == ELEMENT_NODE && 
				 getFirstChildId())
			{
				if( RC_BAD( rc = getChild( ifpDb, ELEMENT_NODE, 
					(IF_DOMNode **)&pTmpNode)))
				{
					if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						goto Exit;
					}

					rc = NE_XFLM_OK;
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
					goto Exit;
				}
			}
		}

		// Get the next sibling node (if any) before we
		// change the tree

		if( pNextSib)
		{
			pNextSib->Release();
			pNextSib = NULL;
		}
		
		if( RC_BAD( rc = pCurNewNode->getNextSibling( pDb,
			(IF_DOMNode **)&pNextSib)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				bDone = TRUE;
			}
			else
			{
				goto Exit;
			}
		}

		bMustAbortOnError = TRUE;
		
		if( pDataElementNode)
		{
			pDataElementNode->Release();
			pDataElementNode = NULL;
		}

		// Do indexing work before making changes

		if( pCurNewNode->getNameId())
		{
			if( pCurNewNode->getNodeType() == DATA_NODE)
			{
				if( pCurNewNode->getParentId())
				{
					if( RC_BAD( rc = pCurNewNode->getParentNode( pDb,
												(IF_DOMNode **)&pDataElementNode)))
					{
						goto Exit;
					}
					
					if( RC_BAD( rc = pDb->updateIndexKeys(
								getCollection(), pDataElementNode,
								IX_DEL_NODE_VALUE, bStartOfUpdate)))
					{
						goto Exit;
					}
					
					bStartOfUpdate = FALSE;
				}

				if( !getFirstChildId())
				{
					if( RC_BAD( rc = pDb->updateIndexKeys(
						getCollection(), this, IX_DEL_NODE_VALUE, bStartOfUpdate)))
					{
						goto Exit;
					}
					bStartOfUpdate = FALSE;
				}
			}
			else
			{
				if( RC_BAD( rc = pDb->updateIndexKeys(
					getCollection(), pCurNewNode, IX_UNLINK_NODE, bStartOfUpdate)))
				{
					goto Exit;
				}
				
				bStartOfUpdate = FALSE;
			}
		}

		// Remove pCurNewNode from the tree

		if( pCurNewNode->getModeFlags() & (FDOM_CANNOT_DELETE | FDOM_READ_ONLY))
		{
			rc = RC_SET( NE_XFLM_DELETE_NOT_ALLOWED);
			goto Exit;
		}
	
		if( RC_BAD( rc = pCurNewNode->unlinkNode( pDb, 0)))
		{
			goto Exit;
		}

		if( pDataElementNode)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys(
				getCollection(), pDataElementNode, IX_ADD_NODE_VALUE, 
				bStartOfUpdate)))
			{
				goto Exit;
			}
			
			bStartOfUpdate = FALSE;
			
			if( RC_BAD( rc = pDb->updateIndexKeys(
				getCollection(), this, IX_DEL_NODE_VALUE, bStartOfUpdate)))
			{
				goto Exit;
			}
		}

		// Make sure that "this" node is not a child of the node
		// being inserted

		if( RC_BAD( rc = isDescendantOf( pDb, pCurNewNode, &bDescendant)))
		{
			goto Exit;
		}

		if( bDescendant)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
			goto Exit;
		}

		// If the node being inserted isn't from the same document,
		// we cannot perform the operation

		if( pCurNewNode->getDocumentId() != getDocumentId())
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_WRONG_DOCUMENT_ERR);
			goto Exit;
		}

		if( RC_BAD( rc = pCurNewNode->makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		if( pInsertBefore)
		{
			if( RC_BAD( rc = pInsertBefore->makeWriteCopy( pDb)))
			{
				goto Exit;
			}
			
			// Insert the new child before the ref child

			if( !pInsertBefore->getPrevSibId())
			{
				if( !bUpdatedNode)
				{
					if( RC_BAD( rc = makeWriteCopy( pDb)))
					{
						goto Exit;
					}
					bUpdatedNode = TRUE;
				}
				
				// Change the parent's first child pointer

				setFirstChildId( pCurNewNode->getNodeId());
			}
			else
			{
				pCurNewNode->setPrevSibId( pInsertBefore->getPrevSibId());

				// Get the prev sib and set its next sib value

				if( RC_BAD( rc = pDb->getNode(
					getCollection(), pInsertBefore->getPrevSibId(), &pTmpNode)))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
				{
					goto Exit;
				}

				pTmpNode->setNextSibId( pCurNewNode->getNodeId());

				if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, 
					FLM_UPD_INTERNAL_CHANGE)))
				{
					goto Exit;
				}

				pTmpNode->Release();
				pTmpNode = NULL;
			}

			pInsertBefore->setPrevSibId( pCurNewNode->getNodeId());
			pCurNewNode->setNextSibId( pInsertBefore->getNodeId());

			if( RC_BAD( rc = pDb->updateNode( pInsertBefore->m_pCachedNode, 
				FLM_UPD_INTERNAL_CHANGE)))
			{
				goto Exit;
			}
		}
		else
		{
			if( (ui64Tmp = getLastChildId()) != 0)
			{
				// Get the prev sib and set its next sib value

				if( RC_BAD( rc = pDb->getNode(
					getCollection(), getLastChildId(), &pTmpNode)))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
				{
					goto Exit;
				}

				pTmpNode->setNextSibId( pCurNewNode->getNodeId());
				pCurNewNode->setPrevSibId( getLastChildId());

				if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, 
					FLM_UPD_INTERNAL_CHANGE)))
				{
					goto Exit;
				}

				pTmpNode->Release();
				pTmpNode = NULL;
			}

			pCurNewNode->setPrevSibId( getLastChildId());
				
			if( !bUpdatedNode)
			{
				if( RC_BAD( rc = makeWriteCopy( pDb)))
				{
					goto Exit;
				}
				bUpdatedNode = TRUE;
			}
			
			setLastChildId( pCurNewNode->getNodeId());

			if( !ui64Tmp)
			{
				setFirstChildId( pCurNewNode->getNodeId());
			}
		}
		
		// Need to increment the data child node count

		if( pCurNewNode->getNodeType() == DATA_NODE &&
			eThisNodeType == ELEMENT_NODE)
		{
			setDataChildCount( getDataChildCount() + 1);
			bUpdatedNode = TRUE;
		}
		
		if( bUpdatedNode)
		{
			if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 
				FLM_UPD_INTERNAL_CHANGE)))
			{
				goto Exit;
			}
			
			// Reset to FALSE for next time around in loop.
			
			bUpdatedNode = FALSE;
		}

		pCurNewNode->setParentId( getNodeId());

		// Need to set the naming tag and data type of the node.

		if( pCurNewNode->getNodeType() == DATA_NODE &&
			eThisNodeType == ELEMENT_NODE)
		{
			if( pCurNewNode->getDataType() == XFLM_NODATA_TYPE)
			{
				pCurNewNode->setDataType( getDataType());
			}
			else if( getDataType() != pCurNewNode->getDataType())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
				goto Exit;
			}

			pCurNewNode->setNameId( getNameId());
		}
		
		// An additional restriction on unique child elements is that they
		// must have a node ID greater than the parent element.  This allows
		// them to be stored using a very compact representation.
		
		if( getModeFlags() & FDOM_HAVE_CELM_LIST)
		{
			if( pCurNewNode->getNodeId() < getNodeId())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_CHILD_ELM_NODE_ID);
				goto Exit;
			}
		}

		if( RC_BAD( rc = pDb->updateNode( pCurNewNode->m_pCachedNode, 
			FLM_UPD_INTERNAL_CHANGE)))
		{
			goto Exit;
		}

		if( pCurNewNode->getNodeId() == pLastNewChild->getNodeId())
		{
			bDone = TRUE;
		}

		// Do post-link indexing work

		if( pCurNewNode->getNameId())
		{
			if( pCurNewNode->getNodeType() == DATA_NODE)
			{
				if( RC_BAD( rc = pDb->updateIndexKeys( 
					getCollection(), this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = pDb->updateIndexKeys(
					getCollection(), pCurNewNode, IX_LINK_NODE, bStartOfUpdate)))
				{
					goto Exit;
				}
			}

			bStartOfUpdate = FALSE;
		}

		pCurNewNode->Release();
		pCurNewNode = pNextSib;
		pNextSib = NULL;

		if( bDone)
		{
			break;
		}
	}
	
	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logInsertBefore( 
		pDb, getCollection(), getNodeId(), pNewChild->getNodeId(),
		ui64RefChildId)))
	{
		goto Exit;
	}

Exit:

	// Release any nodes we are still holding

	if( pFirstNewChild)
	{
		pFirstNewChild->Release();
	}

	if( pLastNewChild)
	{
		pLastNewChild->Release();
	}

	if( pCurNewNode)
	{
		pCurNewNode->Release();
	}

	if( pDataElementNode)
	{
		pDataElementNode->Release();
	}

	if( pNextSib)
	{
		pNextSib->Release();
	}

	if( pInsertBefore)
	{
		pInsertBefore->Release();
	}

	if( pTmpNode)
	{
		pTmpNode->Release();
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setNumber64(
	IF_Db *			ifpDb,
	FLMINT64			i64Value,
	FLMUINT64		ui64Value,
	FLMUINT			uiEncDefId)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bNeg = FALSE;
	F_Db *				pDb = (F_Db*)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	IF_DOMNode *		pNode = NULL;
	FLMUINT				uiCollection;
	FLMUINT				uiRflToken = 0;
	FLMUINT				uiNodeDataType;
	FLMUINT				uiValLen;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bIsIndexed = TRUE;
	FLMBOOL				bStartOfUpdate = TRUE;
	eDomNodeType		eNodeType;
	
	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = canSetValue( pDb, XFLM_NUMBER_TYPE)))
	{
		goto Exit;
	}
	
	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Grab some information about the node

	uiCollection = getCollection();
	eNodeType = getNodeType();
	
	// Special case for element nodes
	
	if( eNodeType == ELEMENT_NODE && getDataChildCount())
	{
		if( RC_BAD( rc = getChild( pDb, DATA_NODE, &pNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}
		
		bMustAbortOnError = TRUE;
		rc = ((F_DOMNode *)pNode)->setNumber64( pDb, i64Value, 
					ui64Value, uiEncDefId);
		goto Exit;
	}
	
	// If the number is less than zero, invert the sign
	
	if( i64Value < 0)
	{
		bNeg = TRUE;
		ui64Value = (FLMUINT64)(-i64Value);
	}
	else if( i64Value)
	{
		ui64Value = (FLMUINT64)i64Value;
	}
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = makeWriteCopy( (F_Db *)ifpDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
			this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
			
		if( RC_BAD( rc = m_pCachedNode->setNumber64( pDb,
			m_uiAttrNameId, ui64Value, bNeg, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( bIsIndexed)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				uiCollection, this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
			{
				goto Exit;
			}

			bStartOfUpdate = FALSE;
		}

		// Update the node

		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb, 
			m_pCachedNode, m_uiAttrNameId))) 
		{
			goto Exit;
		}
	
		goto Exit;
	}
	
	// Generate the storage value

	uiNodeDataType = getDataType();

	if( uiNodeDataType == XFLM_NUMBER_TYPE ||
		uiNodeDataType == XFLM_NODATA_TYPE)
	{
		if( bNeg)
		{
			if( (getModeFlags() & FDOM_SIGNED_QUICK_VAL) &&
				i64Value == getQuickINT64())
			{
				goto Exit;
			}
		}
		else if( (getModeFlags() & FDOM_UNSIGNED_QUICK_VAL) &&
			ui64Value == getQuickUINT64())
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
				this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
			{
				goto Exit;
			}

			bStartOfUpdate = FALSE;
		}
		else
		{
			bIsIndexed = FALSE;
		}
	
		if( RC_BAD( rc = makeWriteCopy( (F_Db *)ifpDb)))
		{
			goto Exit;
		}

		if( getDataBufSize() < FLM_MAX_NUM_BUF_SIZE)
		{
			if( RC_BAD( rc = resizeDataBuffer( FLM_MAX_NUM_BUF_SIZE, FALSE)))
			{
				goto Exit;
			}
		}

		uiValLen = FLM_MAX_NUM_BUF_SIZE;
		if( RC_BAD( rc = flmNumber64ToStorage( ui64Value,
			&uiValLen, getDataPtr(), bNeg, FALSE)))
		{
			goto Exit;
		}

		if( uiNodeDataType == XFLM_NODATA_TYPE)
		{
			uiNodeDataType = XFLM_NUMBER_TYPE;
			setDataType( uiNodeDataType);
		}
	}
	else if( uiNodeDataType == XFLM_TEXT_TYPE)
	{
		FLMBYTE		ucNumBuf[ 64];
		FLMBYTE *	pucSen;
		FLMUINT		uiSenLen;
		
		if( !bNeg)
		{
			f_ui64toa( ui64Value, (char *)&ucNumBuf[ 1]);
		}
		else
		{
			f_i64toa( i64Value, (char *)&ucNumBuf[ 1]);
		}
		
		uiValLen = f_strlen( (const char *)ucNumBuf);
		pucSen = &ucNumBuf[ 0];
		uiSenLen = f_encodeSEN( uiValLen, &pucSen, (FLMUINT)0);
		flmAssert( uiSenLen == 1);
		uiValLen += uiSenLen + 1;

		// If the value isn't being changed, there is no need to continue
		
		if( getDataLength() == uiValLen &&
			f_memcmp( getDataPtr(), ucNumBuf, uiValLen) == 0)
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
				this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
			{
				goto Exit;
			}

			bStartOfUpdate = FALSE;
		}
		else
		{
			bIsIndexed = FALSE;
		}

		if( RC_BAD( rc = makeWriteCopy( (F_Db *)ifpDb)))
		{
			goto Exit;
		}

		// Allocate or re-allocate the buffer

		if( calcDataBufSize( uiValLen) > getDataBufSize())
		{
			if( RC_BAD( rc = resizeDataBuffer( uiValLen, FALSE)))
			{
				goto Exit;
			}
		}

		if( uiValLen)
		{
			f_memcpy( getDataPtr(), ucNumBuf, uiValLen);
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_DATA_TYPE);
		goto Exit;
	}
	
	setEncDefId( uiEncDefId);
	setDataLength( uiValLen);

	// Clear the "on disk" flag (if set), as well as the quick nums

	unsetFlags( FDOM_VALUE_ON_DISK |
					FDOM_FIXED_SIZE_HEADER |
					FDOM_UNSIGNED_QUICK_VAL |
					FDOM_SIGNED_QUICK_VAL);

	// Update the node

	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		goto Exit;
	}

	if( bIsIndexed)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
			this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
	}

	if( !bNeg)
	{
		setUINT64( ui64Value);
	}
	else
	{
		setINT64( i64Value);
	}
	
	// Log the value to the RFL
	
	pRfl->enableLogging( &uiRflToken);

	if( !uiEncDefId)
	{
		if( RC_BAD( rc = pRfl->logNodeSetNumberValue( pDb, 
			uiCollection, getNodeId(), ui64Value, bNeg))) 
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pRfl->logEncryptedNodeUpdate( pDb, m_pCachedNode))) 
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setStorageValue(
	F_Db *			pDb,
	void *			pvValue,
	FLMUINT			uiValueLen,
	FLMUINT			uiEncDefId,
	FLMBOOL			bLast)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCollection;
	eDomNodeType		eNodeType;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bIsIndexed = TRUE;
	F_Database *		pDatabase = pDb->m_pDatabase;
	FLMBOOL				bFirst = pDatabase->m_pPendingInput ? FALSE : TRUE;
	
	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if( getDataType() == XFLM_UNKNOWN_TYPE || (bStartedTrans && !bLast))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	bMustAbortOnError = TRUE;
	uiCollection = getCollection();
	eNodeType = getNodeType();

	if( eNodeType == ELEMENT_NODE ||
		 eNodeType == DATA_NODE)
	{
		if( !getNameId())
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
	}

	if( bFirst)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
			this, IX_DEL_NODE_VALUE, TRUE, &bIsIndexed)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
	}

	if( eNodeType == ELEMENT_NODE ||
		 eNodeType == DATA_NODE ||
		 eNodeType == COMMENT_NODE)
	{
		FLMUINT		uiBytesToCopy;
		FLMBYTE *	pucValue = (FLMBYTE *)pvValue;

		if( bFirst)
		{
			setEncDefId( uiEncDefId);
		}

		if( bFirst && bLast)
		{
			if( calcDataBufSize( uiValueLen) != getDataBufSize())
			{
				if( RC_BAD( rc = resizeDataBuffer( uiValueLen, FALSE)))
				{
					goto Exit;
				}
			}
				
			setDataLength( uiValueLen);

			if( uiValueLen)
			{
				f_memcpy( getDataPtr(), pvValue, uiValueLen);
			}
		}
		else
		{
			if( bFirst)
			{
				if( RC_BAD( rc = m_pCachedNode->openPendingInput( pDb, 
					getDataType())))
				{
					goto Exit;
				}
			}

			while( uiValueLen)
			{
				uiBytesToCopy = pDatabase->m_uiUpdBufferSize - 
									pDatabase->m_uiUpdByteCount;

				uiBytesToCopy = uiBytesToCopy > uiValueLen 
											? uiValueLen 
											: uiBytesToCopy;

				if( !uiBytesToCopy)
				{
					if( RC_BAD( rc = m_pCachedNode->flushPendingInput( pDb, FALSE)))
					{
						goto Exit;
					}

					continue;
				}

				f_memcpy( &pDatabase->m_pucUpdBuffer[ pDatabase->m_uiUpdByteCount],
					pucValue, uiBytesToCopy);
				pucValue += uiBytesToCopy;
				uiValueLen -= uiBytesToCopy;
				pDatabase->m_uiUpdByteCount += uiBytesToCopy;
			}

			if( bLast)
			{
				if( pDatabase->m_bUpdFirstBuf)
				{
					if( RC_BAD( rc = m_pCachedNode->flushPendingInput( pDb, FALSE)))
					{
						goto Exit;
					}
				}

				if( RC_BAD( rc = m_pCachedNode->flushPendingInput( pDb, TRUE)))
				{
					goto Exit;
				}

				pDatabase->endPendingInput();
			}
		}
	}
	else if( eNodeType == ATTRIBUTE_NODE)
	{
		if( !bLast)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}

		if( RC_BAD( rc = m_pCachedNode->setStorageValue( 
			pDb, m_uiAttrNameId, pvValue, uiValueLen, uiEncDefId)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( bLast)
	{
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		if( bIsIndexed)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
				this, IX_ADD_NODE_VALUE, FALSE)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		if( bMustAbortOnError)
		{
			pDb->setMustAbortTrans( rc);
		}

		pDatabase->endPendingInput();
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::openPendingInput(
	F_Db *			pDb,
	FLMUINT			uiNewDataType)
{
	RCODE				rc = NE_XFLM_OK;
	F_Database *	pDatabase = pDb->m_pDatabase;
	eDomNodeType	eNodeType = getNodeType();

	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	// Set up the database to point to this node.  Only one node
	// is allowed to stream data into the B-Tree at a time.

	if( RC_BAD( rc = pDatabase->startPendingInput( uiNewDataType, this)))
	{
		goto Exit;
	}

	// If the naming tag has already been set, make sure the
	// new data type is compatible with the defined type for
	// the tag.

	if( getNameId())
	{
		switch( eNodeType)
		{
			case ELEMENT_NODE:
			case DATA_NODE:
			{
				F_AttrElmInfo	elmInfo;	

				if( RC_BAD( rc = pDb->m_pDict->getElement(
					pDb, m_nodeInfo.uiNameId, &elmInfo)))
				{
					goto Exit;
				}

				if( elmInfo.getDataType() != uiNewDataType)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
					goto Exit;
				}
				break;
			}

			case ANNOTATION_NODE:
			{
				break;
			}

			default:
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
				goto Exit;
			}
		}
	}

	// Better be the latest version for update.
	
	flmAssert( m_ui64LowTransId == pDb->m_ui64CurrTransID);

	m_nodeInfo.uiDataLength = 0;
	m_nodeInfo.uiDataType = uiNewDataType;
	unsetFlags( FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	setFlags( FDOM_VALUE_ON_DISK | FDOM_FIXED_SIZE_HEADER);

	flmAssert( !pDatabase->m_uiUpdByteCount);
	flmAssert( !pDatabase->m_uiUpdCharCount);

Exit:

	if( RC_BAD( rc))
	{
		pDatabase->endPendingInput();
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::flushPendingInput(
	F_Db *			pDb,
	FLMBOOL			bLast)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection = NULL;
	FLMBYTE				ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE				ucHeader[ MAX_DOM_HEADER_SIZE + 16];
	FLMUINT				uiKeyLen;
	FLMUINT				uiHeaderStorageSize;
	F_Database *		pDatabase = pDb->m_pDatabase;
	FLMUINT				uiOutputLength;
	FLMUINT				uiLeftoverLength;

	uiKeyLen = sizeof( ucKey);
	if( RC_BAD( rc = flmNumber64ToStorage( m_nodeInfo.ui64NodeId, 
		&uiKeyLen, ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Open the B-Tree

	if( !pDatabase->m_pPendingBTree)
	{
		if( !pDatabase->m_bUpdFirstBuf)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}

		if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree(
			&pDatabase->m_pPendingBTree)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pDb->m_pDict->getCollection(
			m_nodeInfo.uiCollection, &pCollection)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pDatabase->m_pPendingBTree->btOpen(
			pDb, &pCollection->lfInfo, FALSE, TRUE)))
		{
			goto Exit;
		}
	}

	// Better be the latest version for update.
	
	flmAssert( m_ui64LowTransId == pDb->m_ui64CurrTransID);
	uiOutputLength = pDatabase->m_uiUpdByteCount;
	uiLeftoverLength = 0;

	// Output the node header

	if( pDatabase->m_bUpdFirstBuf)
	{
		FLMUINT	uiIVLen = 0;

		// This routine is designed to handle multi-block data streams.
		// It shouldn't be called if everything fits within the a single
		// update buffer.

		flmAssert( !bLast);
		
		// There shouldn't be any attributes at this point, because
		// this shouldn't be an element node
		
		flmAssert( !hasAttributes());
		flmAssert( m_nodeInfo.eNodeType != ELEMENT_NODE);

		// Build the node header

		if( RC_BAD( rc = headerToBuf( TRUE, ucHeader, 
			&uiHeaderStorageSize, NULL, NULL)))
		{
			goto Exit;
		}
		
		if( getEncDefId())
		{
			F_ENCDEF *	pEncDef;
			
			if( RC_BAD( rc = pDb->m_pDict->getEncDef( getEncDefId(), &pEncDef)))
			{
				goto Exit;
			}
			
			uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( uiIVLen == 8 || uiIVLen == 16);
			
			if( RC_BAD( rc = pEncDef->pCcs->generateIV( uiIVLen, pDatabase->m_ucIV)))
			{
				goto Exit;
			}
			
			f_memcpy( &ucHeader[ uiHeaderStorageSize], pDatabase->m_ucIV, uiIVLen);
		}
		
		// Output the header

		if( nodeIsNew())
		{
			// If this is a new entry, we need to insert it into the
			// b-tree.

			if( RC_BAD( rc = pDatabase->m_pPendingBTree->btInsertEntry(
							ucKey, uiKeyLen,
							ucHeader, uiHeaderStorageSize + uiIVLen, TRUE, FALSE,
							&m_ui32BlkAddr, &m_uiOffsetIndex)))
			{
				if( rc == NE_XFLM_NOT_UNIQUE)
				{
					rc = RC_SET( NE_XFLM_EXISTS);
				}
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pDatabase->m_pPendingBTree->btReplaceEntry(
				ucKey, uiKeyLen,
				ucHeader, uiHeaderStorageSize + uiIVLen, TRUE, FALSE, TRUE,
				&m_ui32BlkAddr, &m_uiOffsetIndex)))
			{
				goto Exit;
			}
		}

		pDatabase->m_bUpdFirstBuf = FALSE;
	}

	// Output the data buffer

	if( pDatabase->m_uiUpdByteCount || bLast)
	{
		// Encrypt the buffer if this node is encrypted.  If encrypted,
		// uiOutputLength will hold the length of the encrypted data.
		// If not encrypted, it will hold the length of the non-encrypted data,
		// i.e. pDatabase->m_uiUpdByteCount.  the encrypted data will be returned
		// in the input buffer.

		if( getEncDefId())
		{
			// If this is not the last buffer, only encrypt to the nearest
			// FLM_ENCRYPT_CHUNK_SIZE byte boundary.  Move whatever we don't
			// encrypt down to the beginning of the buffer after writing the
			// encrypted part out to the B-tree (see below).  This is necessary
			// because the decryption algorithm assumes that all of the 
			// encrypted data is in FLM_ENRYPT_CHUNK_SIZE byte chunks - except
			// for the last chunk, which may be encrypted to the nearest 16 byte
			// boundary.

			if( !bLast && (uiOutputLength & (FLM_ENCRYPT_CHUNK_SIZE - 1)))
			{
				uiLeftoverLength = uiOutputLength & (FLM_ENCRYPT_CHUNK_SIZE - 1);
				uiOutputLength -= uiLeftoverLength;
			}

			if( RC_BAD( rc = pDb->encryptData( getEncDefId(), pDatabase->m_ucIV, 
				pDatabase->m_pucUpdBuffer, pDatabase->m_uiUpdBufferSize,
				uiOutputLength, &uiOutputLength)))
			{
				goto Exit;
			}
		}
		
		if( nodeIsNew())
		{
			// If this is a new entry, we need to continue calling the
			// insert method to stream its data into the b-tree

			if( RC_BAD( rc = pDatabase->m_pPendingBTree->btInsertEntry(
							ucKey, uiKeyLen,
							pDatabase->m_pucUpdBuffer, uiOutputLength,
							FALSE, bLast,
							&m_ui32BlkAddr, &m_uiOffsetIndex)))
			{
				if( rc == NE_XFLM_NOT_UNIQUE)
				{
					rc = RC_SET( NE_XFLM_EXISTS);
				}
				
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = 	pDatabase->m_pPendingBTree->btReplaceEntry(
				ucKey, uiKeyLen,
				pDatabase->m_pucUpdBuffer, uiOutputLength,
				FALSE, bLast, TRUE,
				&m_ui32BlkAddr, &m_uiOffsetIndex)))
			{
				goto Exit;
			}
		}
	}

	m_nodeInfo.uiDataLength += uiOutputLength;
	if( (pDatabase->m_uiUpdByteCount = uiLeftoverLength) != 0)
	{
		f_memmove( pDatabase->m_pucUpdBuffer, 
			pDatabase->m_pucUpdBuffer + uiOutputLength, uiLeftoverLength);
	}

	if( bLast)
	{
		// Clear the dirty flag and the new flag.

		unsetNodeDirtyAndNew( pDb);
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::setMetaValue(
	IF_Db *					ifpDb,
	FLMUINT64				ui64Value)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db*)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	eDomNodeType		eNodeType;
	
	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();
	
	// Only allow meta values on element nodes.
	
	if( eNodeType != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If the value isn't changing, don't do anything

	if( ui64Value == getMetaValue())
	{
		goto Exit;
	}

	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Make sure the node can be updated

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}

	// Set the value

	setMetaValue( ui64Value);

	// Update the node

	if( RC_BAD( rc = pDb->updateNode( 
		m_pCachedNode, FLM_UPD_INTERNAL_CHANGE)))
	{
		goto Exit;
	}

	// Log the value to the RFL
	
	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = pRfl->logNodeSetMetaValue( pDb, 
		getCollection(), getNodeId(), ui64Value)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getMetaValue(
	IF_Db *					ifpDb,
	FLMUINT64 *				pui64Value)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db*)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	*pui64Value = getMetaValue();
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::isDataLocalToNode(
	IF_Db *					ifpDb,
	FLMBOOL *				pbDataIsLocal)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db*)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (getNodeType() == ATTRIBUTE_NODE)
	{
		*pbDataIsLocal = TRUE;
	}
	else
	{
		*pbDataIsLocal = getDataLength() ? TRUE : FALSE;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc: Read from a text stream, outputting ASCII.  If we encounter a
		character that cannot be output as ASCII, we will return an error.
******************************************************************************/
RCODE F_AsciiIStream::read(
	void *		pvBuffer,
	FLMUINT		uiBytesToRead,
	FLMUINT *	puiBytesRead)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUNICODE	uzChar;
	
	*puiBytesRead = 0;
	while( *puiBytesRead < uiBytesToRead)
	{
		if( m_eTextType == XFLM_UNICODE_TEXT)
		{
			if( (m_pucEnd && ((FLMUINT)(m_pucEnd - m_pucCurrPtr) < 
					sizeof( FLMUNICODE))) ||
				(uzChar = *((FLMUNICODE *)m_pucCurrPtr)) == 0)
			{
				break;
			}
			
			if( uzChar > 127)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}
			
			*pucBuffer++ = (FLMBYTE)uzChar;
			m_pucCurrPtr += sizeof( FLMUNICODE);
		}
		else	// UTF8
		{
			if( RC_BAD( rc = f_getCharFromUTF8Buf( 
				&m_pucCurrPtr, m_pucEnd, &uzChar)))
			{
				goto Exit;
			}
			
			if( !uzChar)
			{
				break;
			}
			
			if( uzChar > 127)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}
			
			*pucBuffer++ = (FLMBYTE)uzChar;
		}
		
		m_uiCurrChar++;
		(*puiBytesRead)++;
	}
	
	if( *puiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc: This class converts a storage text stream to an ASCII
		text stream.
******************************************************************************/
class F_AsciiStorageStream : public IF_IStream
{
public:

	F_AsciiStorageStream()
	{
		m_pIStream = NULL;
	}
	
	~F_AsciiStorageStream()
	{
		closeStream();
	}
		
	RCODE XFLAPI read(
		void *					pvBuffer,
		FLMUINT					uiBytesToRead,
		FLMUINT *				puiBytesRead);

	FINLINE RCODE XFLAPI closeStream( void)
	{
		if( m_pIStream)
		{
			m_pIStream->Release();
			m_pIStream = NULL;
		}
		
		return( NE_XFLM_OK);
	}
	
	RCODE openStream(
		IF_IStream *		pIStream);
		
private:

	IF_IStream *		m_pIStream;
};

/*****************************************************************************
Desc: Open an Ascii storage stream.
******************************************************************************/
RCODE F_AsciiStorageStream::openStream(
	IF_IStream *	pIStream)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBYTE	ucSENBuf [16];
	FLMUINT	uiLen;
	FLMUINT	uiSENLen;

	closeStream();
	m_pIStream = pIStream;
	m_pIStream->AddRef();
	
	// Skip over the SEN

	uiLen = 1;
	if( RC_BAD( rc = m_pIStream->read( (char *)&ucSENBuf [0], uiLen, &uiLen)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	if( (uiSENLen = f_getSENLength( ucSENBuf[ 0])) > 1)
	{
		uiLen = uiSENLen - 1;
		if( RC_BAD( rc = m_pIStream->read( 
			(char *)&ucSENBuf [1], uiLen, &uiLen)))
		{
			goto Exit;
		}
	}
	
	// We are now positioned to read UTF8

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc: Read from a storage text stream, outputting ASCII.  If we encounter a
		character that cannot be output as ASCII, we will return an error.
******************************************************************************/
RCODE F_AsciiStorageStream::read(
	void *		pvBuffer,
	FLMUINT		uiBytesToRead,
	FLMUINT *	puiBytesRead)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUNICODE	uzChar;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT		uiBytesRead = 0;
	
	// Better have a stream that has been positioned by a call to open().
	
	flmAssert( m_pIStream);

	while( uiBytesRead < uiBytesToRead)
	{
		if( RC_BAD( rc = f_readUTF8CharAsUnicode( m_pIStream, &uzChar)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		
		if( uzChar <= 127)
		{
			*pucBuffer++ = (FLMBYTE)uzChar;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
		
		uiBytesRead++;
	}
	
	if( uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}
	
Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/*****************************************************************************
Desc: This class converts a binary stream to an ASCII text storage
		stream.
******************************************************************************/
class F_BinaryToTextStream : public IF_IStream
{
public:

	F_BinaryToTextStream()
	{
		m_pEncoderStream = NULL;
	}
	
	~F_BinaryToTextStream()
	{
		closeStream();
	}
		
	RCODE XFLAPI read(
		void *					pvBuffer,
		FLMUINT					uiBytesToRead,
		FLMUINT *				puiBytesRead);

	FINLINE RCODE XFLAPI closeStream( void)
	{
		if( m_pEncoderStream)
		{
			m_pEncoderStream->Release();
			m_pEncoderStream = NULL;
		}
		
		return( NE_XFLM_OK);
	}
	
	RCODE openStream(
		IF_IStream *		pIStream,
		FLMUINT				uiDataLen,
		FLMUINT *			puiTextLength);
		
private:

	FLMBYTE					m_ucSENBuf [16];
	FLMUINT					m_uiSENLen;
	FLMUINT					m_uiCurrOffset;
	IF_IStream *			m_pEncoderStream;
};

/*****************************************************************************
Desc: Open a binary-to-text stream.
******************************************************************************/
RCODE F_BinaryToTextStream::openStream(
	IF_IStream *	pIStream,
	FLMUINT			uiDataLen,
	FLMUINT *		puiTextLength)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucSen;
	FLMUINT		uiOutputLen;

	closeStream();
	
	// Set up the SEN buffer.  Calculate length to be 4 bytes for every 3
	// binary bytes.
	
	uiOutputLen = (uiDataLen / 3) * 4;
	
	// If number of bytes is not an exact multiple of 3, we will need 4
	// more bytes.
	
	if( uiDataLen % 3)
	{
		uiOutputLen += 4;
	}
	
	pucSen = &m_ucSENBuf [0];
	m_uiSENLen = f_encodeSEN( (FLMUINT64)uiOutputLen, &pucSen, (FLMUINT)0);
	m_uiCurrOffset = 0;
	
	// Need to include the data length, SEN length, and the terminating null
	// character.
	
	*puiTextLength = uiOutputLen + m_uiSENLen + 1;
	
	// Set up the encoder stream
	
	if( RC_BAD( rc = FlmOpenBase64EncoderIStream( pIStream, 
		FALSE, &m_pEncoderStream)))
	{
		goto Exit;
	}
	
Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc: Read from a binary stream, outputting base 64 encoded ASCII.
******************************************************************************/
RCODE F_BinaryToTextStream::read(
	void *		pvBuffer,
	FLMUINT		uiBytesToRead,
	FLMUINT *	puiBytesRead)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT		uiBytesRead;
	
	// Better have a stream that has been positioned by a call to open().
	
	flmAssert( m_pEncoderStream);
	*puiBytesRead = 0;
	
	if( m_uiCurrOffset < m_uiSENLen)
	{
		if( uiBytesToRead >= m_uiSENLen - m_uiCurrOffset)
		{
			f_memcpy( pucBuffer, &m_ucSENBuf [m_uiCurrOffset],
						m_uiSENLen - m_uiCurrOffset);
			(*puiBytesRead) += (m_uiSENLen - m_uiCurrOffset);
			pucBuffer += (m_uiSENLen - m_uiCurrOffset);
			m_uiCurrOffset = m_uiSENLen;
		}
		else
		{
			f_memcpy( pucBuffer, &m_ucSENBuf [m_uiCurrOffset],
						uiBytesToRead);
			(*puiBytesRead) += uiBytesToRead;
			m_uiCurrOffset += uiBytesToRead;
			pucBuffer += uiBytesToRead;
		}
	}
	
	// If we didn't get everything from the SEN buffer, read from the
	// decoding stream.
	
	if( *puiBytesRead < uiBytesToRead)
	{
		if( RC_BAD( rc = m_pEncoderStream->read( pucBuffer,
									uiBytesToRead - *puiBytesRead, &uiBytesRead)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				(*puiBytesRead) += uiBytesRead;
			}
			goto Exit;
		}
		else
		{
			(*puiBytesRead) += uiBytesRead;
		}
	}
	
	if( *puiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc: Store text as a number.
******************************************************************************/
RCODE F_DOMNode::storeTextAsNumber(
	F_Db *					pDb,
	void *					pvValue,
	FLMUINT					uiNumBytesInBuffer,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE					ucChar;
	FLMUINT					uiIncrAmount = 0;
	FLMBOOL					bNeg = FALSE;
	FLMBOOL					bHex = FALSE;
	FLMUINT64				ui64Num = 0;
	FLMUINT					uiBytesRead;
	FLMBOOL					bFirstChar = TRUE;
	F_AsciiIStream			asciiStream( (FLMBYTE *)pvValue, uiNumBytesInBuffer, XFLM_UTF8_TEXT);
	
	// Convert the text to a number.
	
	for (;;)
	{
		if( RC_BAD( rc = asciiStream.read( &ucChar, 1, &uiBytesRead)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
	
		// See if we can convert to a number.
			
		if( ucChar >= ASCII_ZERO && ucChar <= ASCII_NINE)
		{
			uiIncrAmount = (FLMUINT)(ucChar - '0');
		}
		else if( ucChar >= ASCII_UPPER_A && ucChar <= ASCII_UPPER_F)
		{
			if( bHex)
			{
				uiIncrAmount = (FLMUINT)(ucChar - ASCII_UPPER_A + 10);
			}
			else
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
		}
		else if( ucChar >= ASCII_LOWER_A && ucChar <= ASCII_LOWER_F)
		{
			if( bHex)
			{
				uiIncrAmount = (FLMUINT)(ucChar - ASCII_LOWER_A + 10);
			}
			else
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
		}
		else if( ucChar == ASCII_LOWER_X || ucChar == ASCII_UPPER_X)
		{
			if( !ui64Num && !bHex)
			{
				bHex = TRUE;
			}
			else
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
				goto Exit;
			}
		}
		else if( ucChar == ASCII_DASH && bFirstChar)
		{
			bNeg = TRUE;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
			goto Exit;
		}

		if( !bHex)
		{
			if( ui64Num > (~(FLMUINT64)0) / 10)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}
			ui64Num *= (FLMUINT64)10;
		}
		else
		{
			if( ui64Num > (~(FLMUINT64)0) / 16)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
				goto Exit;
			}
			ui64Num *= (FLMUINT64)16;
		}
	
		if( ui64Num > (~(FLMUINT64)0) - (FLMUINT64)uiIncrAmount)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}

		ui64Num += (FLMUINT64)uiIncrAmount;
		bFirstChar = FALSE;
	}
		
	// If the number is negative, make sure it doesn't
	// overflow the maximum negative number.
	
	if( bNeg)
	{
		if( ui64Num > gv_ui64MaxSignedIntVal + 1)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_UNDERFLOW);
			goto Exit;
		}
		
		if( RC_BAD( rc = setINT64( (IF_Db *)pDb, 
			-((FLMINT64)ui64Num), uiEncDefId)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = setUINT64( (IF_Db *)pDb, ui64Num, uiEncDefId)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc: Store text as a binary value.
******************************************************************************/
RCODE F_DOMNode::storeTextAsBinary(
	F_Db *					pDb,
	const void *			pvValue,
	FLMUINT					uiNumBytesInBuffer,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE					ucBuf[ 64];
	F_AsciiIStream			asciiStream( (FLMBYTE *)pvValue, 
														uiNumBytesInBuffer, XFLM_UTF8_TEXT);
	IF_IStream *			pDecoderStream = NULL;
	FLMBYTE					ucDynaBuf[ 64];
	F_DynaBuf				dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));
	FLMUINT					uiBytesRead;
	
	if( RC_BAD( rc = FlmOpenBase64DecoderIStream( 
		&asciiStream, &pDecoderStream)))
	{
		goto Exit;
	}
	
	for( ;;)
	{
		if( RC_BAD( rc = pDecoderStream->read( ucBuf, sizeof( ucBuf), &uiBytesRead)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			break;
		}

		if( RC_BAD( rc = dynaBuf.appendData( ucBuf, uiBytesRead)))
		{
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = setBinary( 
		(IF_Db *)pDb, dynaBuf.getBufferPtr(), 
		dynaBuf.getDataLength(), TRUE, uiEncDefId)))
	{
		goto Exit;
	}
	
Exit:

	if( pDecoderStream)
	{
		pDecoderStream->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc: Store binary as a text value (base 64 encoded)
******************************************************************************/
RCODE F_DOMNode::storeBinaryAsText(
	F_Db *				pDb,
	const void *		pvValue,
	FLMUINT				uiLength,
	FLMUINT				uiEncDefId)
{
	RCODE							rc = NE_XFLM_OK;
	FLMBYTE						ucBuf[ 64];
	IF_PosIStream *			pIStream = NULL;
	IF_IStream * 				pEncoderStream = NULL;
	FLMBYTE						ucDynaBuf[ 64];
	F_DynaBuf					dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));
	FLMUINT						uiBytesRead;
	
	if( RC_BAD( rc = FlmOpenBufferIStream( 
		(const char *)pvValue, uiLength, &pIStream)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmOpenBase64EncoderIStream( pIStream, 
		FALSE, &pEncoderStream)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = pEncoderStream->read( 
			ucBuf, sizeof( ucBuf), &uiBytesRead)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			break;
		}
	
		if( RC_BAD( rc = dynaBuf.appendData( ucBuf, uiBytesRead)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = setTextFastPath( pDb, dynaBuf.getBufferPtr(), 
		dynaBuf.getDataLength(), XFLM_UTF8_TEXT, uiEncDefId)))
	{
		goto Exit;
	}
	
Exit:

	if( pEncoderStream)
	{
		pEncoderStream->Release();
	}
	
	if( pIStream)
	{
		pIStream->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setTextStreaming(
	F_Db *					pDb,
	const void *			pvValue,
	FLMUINT					uiNumBytesInBuffer,
	eXFlmTextType			eTextType,
	FLMBOOL					bLast,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiLen;
	FLMBYTE *				pucUpdBuffer;
	FLMUINT					uiUpdBufferSize;
	FLMBYTE *				pucTmp;
	FLMUINT					uiSenLen;
	FLMBYTE					ucTmpSen[ FLM_MAX_NUM_BUF_SIZE];
	F_Database *			pDatabase = pDb->m_pDatabase;
	F_Rfl *					pRfl = pDatabase->m_pRfl;
	F_DOMNode *				pNode = NULL;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bMustAbortOnError = FALSE;
	FLMBOOL					bFirst = pDatabase->m_pPendingInput ? FALSE : TRUE;
	FLMUINT					uiRflToken = 0;
	eDomNodeType			eNodeType;
	
	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();
	
	// Not supported on attributes
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If this is an element node, we need to find or create
	// the child data node

	if( eNodeType == ELEMENT_NODE)
	{
		// The streaming interface is not directly supported on element nodes.
		// If the node already has a value and does not already have a child data
		// node, we can clear the value on the element, create a data node, and
		// allow the operation to continue.
		
		if( getDataLength())
		{
			if( getDataChildCount())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			
			bMustAbortOnError = TRUE;
			
			if( RC_BAD( rc = clearNodeValue( pDb)))
			{
				goto Exit;
			}
		}
		else if( getDataChildCount())
		{
			if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				
				goto Exit;
			}
		}
	
		if( !pNode)
		{
			if( RC_BAD( rc = createNode( (IF_Db *)pDb, DATA_NODE, 
				getNameId(), XFLM_LAST_CHILD, (IF_DOMNode **)&pNode)))
			{
				goto Exit;
			}

			bMustAbortOnError = TRUE;
		}

		if( RC_BAD( rc = pNode->setTextStreaming( pDb, 
			pvValue, uiNumBytesInBuffer, eTextType, bLast, uiEncDefId)))
		{
			goto Exit;
		}

		goto Exit;
	}
	
	// Make sure the state of the node and database
	// allow a value to be set.

	if( RC_BAD( rc = canSetValue( pDb, XFLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	pRfl->disableLogging( &uiRflToken);

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}
	
	unsetFlags( FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	bMustAbortOnError = TRUE;

	// If we are in the middle of streaming data into a node,
	// any error will probably leave the database in a bad
	// state.  Because of this, we want to force the
	// transaction to abort.

	if( pDatabase->m_pPendingInput)
	{
		if( pDatabase->m_pPendingInput != m_pCachedNode)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
	}

	pucUpdBuffer = pDatabase->m_pucUpdBuffer;

	// NOTE: So that flushNode would not have to allocate a buffer for
	// writing out data with a non-padded header, we reserve enough
	// room in pDatabase->m_pucUpdBuffer so that it could hold a maximum
	// header plus all of the data if it needed to.
	// Also, make sure there is room to encrypt the data.  That is why
	// we subtract another 16 bytes.

	uiUpdBufferSize = pDatabase->m_uiUpdBufferSize - 
								MAX_DOM_HEADER_SIZE - ENCRYPT_MIN_CHUNK_SIZE;

	// Update the index keys

	if( bFirst)
	{
		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				getCollection(), this, IX_DEL_NODE_VALUE, TRUE)))
			{
				goto Exit;
			}
		}
	}

	// Open the pending input stream

	if( bFirst)
	{
		if( RC_BAD( rc = openPendingInput( pDb, XFLM_TEXT_TYPE)))
		{
			goto Exit;
		}

		// Reserve 5 bytes for a SEN indicating the number of characters
		// in the string.  The SEN (representing a 32-bit number) will
		// never require more than 5 bytes and will typically only
		// require 1 or 2 bytes.

		pDatabase->m_uiUpdByteCount += SEN_RESERVE_BYTES;
		*pucUpdBuffer = 0;

		// Save the encryption scheme

		setEncDefId( uiEncDefId);
	}

	// Set the value

	if( pvValue)
	{
		// If a zero was passed for the character count, change it to
		// the UINT high value.  We will output characters until a terminating
		// null is found.

		if( !uiNumBytesInBuffer)
		{
			uiNumBytesInBuffer = FLM_MAX_UINT;
		}

		switch( eTextType)
		{
			case XFLM_UNICODE_TEXT:
			{
				FLMUNICODE *	puCurChar = (FLMUNICODE *)pvValue;

				while( *puCurChar && uiNumBytesInBuffer >= sizeof( FLMUNICODE))
				{
					uiLen = uiUpdBufferSize - pDatabase->m_uiUpdByteCount;
					if( RC_BAD( rc = f_uni2UTF8( *puCurChar,
						&pucUpdBuffer[ pDatabase->m_uiUpdByteCount], &uiLen)))
					{
						if( rc == NE_XFLM_CONV_DEST_OVERFLOW)
						{
							if( RC_BAD( rc = flushPendingInput( pDb, FALSE)))
							{
								goto Exit;
							}
							continue;
						}

						goto Exit;
					}

					pDatabase->m_uiUpdByteCount += uiLen;
					pDatabase->m_uiUpdCharCount++;
					uiNumBytesInBuffer -= sizeof( FLMUNICODE);
					puCurChar++;
				}

				break;
			}

			case XFLM_UTF8_TEXT:
			{
				FLMBYTE *	pucCurByte = (FLMBYTE *)pvValue;
				FLMBYTE *	pucEnd = (FLMBYTE *)pvValue + uiNumBytesInBuffer;

				for( ;;)
				{
					if( (uiUpdBufferSize - pDatabase->m_uiUpdByteCount) < 3)
					{
						if( RC_BAD( rc = flushPendingInput( pDb, FALSE)))
						{
							goto Exit;
						}
					}
					
					if( RC_BAD( rc = f_getUTF8CharFromUTF8Buf( &pucCurByte,
						pucEnd, &pucUpdBuffer[ pDatabase->m_uiUpdByteCount], &uiLen)))
					{
						goto Exit;
					}
					
					if( !uiLen)
					{
						break;
					}

					pDatabase->m_uiUpdByteCount += uiLen;
					pDatabase->m_uiUpdCharCount++;
					uiNumBytesInBuffer -= uiLen;
				}
				
				break;
			}

			default:
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
				goto Exit;
			}
		}
	}

	if( bLast)
	{
		// Terminate the buffer with a null byte

		if( pDatabase->m_uiUpdCharCount)
		{
			// See if there is room for a single zero byte

			if( pDatabase->m_uiUpdByteCount == uiUpdBufferSize)
			{
				if( RC_BAD( rc = flushPendingInput( pDb, FALSE)))
				{
					goto Exit;
				}
			}

			pucUpdBuffer[ pDatabase->m_uiUpdByteCount++] = 0;
		}

		if( pDatabase->m_bUpdFirstBuf)
		{
			if( pDatabase->m_uiUpdCharCount)
			{
				FLMUINT		uiValLen;

				// Five bytes were reserved to encode the number of characters
				// in the string.  Since we didn't have to use multiple buffers
				// to write the string to the database, we won't have to
				// output a padded SEN.

				pucTmp = ucTmpSen;
				f_encodeSEN( pDatabase->m_uiUpdCharCount, &pucTmp);
				uiSenLen = (FLMUINT)(pucTmp - ucTmpSen);

				// Copy the value into the node

				uiValLen = (pDatabase->m_uiUpdByteCount -
										SEN_RESERVE_BYTES) + uiSenLen;

				if( calcDataBufSize( uiValLen) > getDataBufSize())
				{
					if( RC_BAD( rc = resizeDataBuffer( uiValLen, FALSE)))
					{
						goto Exit;
					}
				}

				setDataLength( uiValLen);
				f_memcpy( getDataPtr(), ucTmpSen, uiSenLen);
				f_memcpy( getDataPtr() + uiSenLen,
					&pucUpdBuffer[ SEN_RESERVE_BYTES],
					pDatabase->m_uiUpdByteCount - SEN_RESERVE_BYTES);
			}
			else
			{
				setDataLength( 0);
			}

			// Clear flags

			unsetFlags( FDOM_VALUE_ON_DISK | FDOM_FIXED_SIZE_HEADER);

			// Update the node

			if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
			{
				setDataLength( 0);
				goto Exit;
			}
		}
		else
		{
			FLMBYTE		ucKey[ FLM_MAX_NUM_BUF_SIZE];
			FLMUINT		uiKeyLen;
			FLMUINT		uiHeaderStorageSize;
			FLMUINT32	ui32BlkAddr;
			FLMUINT		uiOffsetIndex;

			// Flush anything that is in the current buffer and end the
			// pending input stream.

			if( RC_BAD( rc = flushPendingInput( pDb, TRUE)))
			{
				goto Exit;
			}

			pucTmp = ucTmpSen;
			f_encodeSEN( pDatabase->m_uiUpdCharCount,
				&pucTmp, SEN_RESERVE_BYTES);
			flmAssert( (FLMUINT)(pucTmp - ucTmpSen) == SEN_RESERVE_BYTES);
			
			// Output the header
			
			if( RC_BAD( rc = headerToBuf( TRUE, pDatabase->m_pucUpdBuffer, 
				&uiHeaderStorageSize, NULL, NULL)))
			{
				goto Exit;
			}

			// Copy the SEN into the update buffer

			f_memcpy( &pDatabase->m_pucUpdBuffer[ uiHeaderStorageSize],
				ucTmpSen, SEN_RESERVE_BYTES);

			// Replace the header

			uiKeyLen = sizeof( ucKey);
			if( RC_BAD( rc = flmNumber64ToStorage( getNodeId(), &uiKeyLen,
				ucKey, FALSE, TRUE)))
			{
				goto Exit;
			}
			
			ui32BlkAddr = getBlkAddr();
			uiOffsetIndex = getOffsetIndex();

			if( RC_BAD( rc = 	pDatabase->m_pPendingBTree->btReplaceEntry(
				ucKey, uiKeyLen,
				pDatabase->m_pucUpdBuffer, uiHeaderStorageSize + SEN_RESERVE_BYTES,
				TRUE, TRUE, FALSE, &ui32BlkAddr, &uiOffsetIndex)))
			{
				goto Exit;
			}
			
			setBlkAddr( ui32BlkAddr);
			setOffsetIndex( uiOffsetIndex);

			// Clear the dirty flag and the new flag.

			unsetNodeDirtyAndNew( pDb);
		}
		
		pDatabase->endPendingInput();

		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				getCollection(), this, IX_ADD_NODE_VALUE, FALSE)))
			{
				goto Exit;
			}
		}
		
		// Log the node to the RFL
		
		pRfl->enableLogging( &uiRflToken);

		if( RC_BAD( rc = pRfl->logNodeSetValue( pDb, 
			RFL_NODE_SET_TEXT_VALUE_PACKET, m_pCachedNode))) 
		{
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( RC_BAD( rc))
	{
		pDatabase->endPendingInput();

		if( bMustAbortOnError)
		{
			pDb->setMustAbortTrans( rc);
		}
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}
	
	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setTextFastPath(
	F_Db *					pDb,
	const void *			pvValue,
	FLMUINT					uiNumBytesInBuffer,
	eXFlmTextType			eTextType,
	FLMUINT					uiEncDefId)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiNumCharsInBuffer = 0;
	FLMBYTE				ucUTFCharBuf[ 3];
	F_Database *		pDatabase = pDb->m_pDatabase;
	F_Rfl *				pRfl = pDatabase->m_pRfl;
	F_DOMNode *			pNode = NULL;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMUINT				uiRflToken = 0;
	FLMUINT				uiValLen;
	FLMUINT				uiReadLen;
	FLMBOOL				bIsIndexed = TRUE;
	FLMBYTE				ucDynaBuf[ 64];
	F_DynaBuf			dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));
	FLMBOOL				bStartOfUpdate = TRUE;
	eDomNodeType		eNodeType;
	
	// Make sure a transaction is active

	if( RC_BAD( rc = pDb->checkTransaction(
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Cannot set a value if input is still pending
	
	if( pDatabase->m_pPendingInput)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	eNodeType = getNodeType();

	// Convert the text to UTF-8 if needed

	if( eTextType == XFLM_UNICODE_TEXT)
	{
		FLMUNICODE *	puCurChar = (FLMUNICODE *)pvValue;

		if( puCurChar)
		{
			if( !uiNumBytesInBuffer)
			{
				uiNumBytesInBuffer = FLM_MAX_UINT;
			}

			while( *puCurChar && uiNumBytesInBuffer >= sizeof( FLMUNICODE))
			{
				if( *puCurChar <= 0x007F)
				{
					if( RC_BAD( rc = dynaBuf.appendByte( (FLMBYTE)*puCurChar)))
					{
						goto Exit;
					}
				}
				else
				{
					uiReadLen = sizeof( ucUTFCharBuf);
					if( RC_BAD( rc = f_uni2UTF8( *puCurChar, 
						ucUTFCharBuf, &uiReadLen)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = dynaBuf.appendData( ucUTFCharBuf, uiReadLen)))
					{
						goto Exit;
					}
				}

				puCurChar++;
				uiNumBytesInBuffer -= sizeof( FLMUNICODE);
				uiNumCharsInBuffer++;
			}
		}

		if( RC_BAD( rc = dynaBuf.appendByte( 0)))
		{
			goto Exit;
		}

		eTextType = XFLM_UTF8_TEXT;
		pvValue = dynaBuf.getBufferPtr();
		uiNumBytesInBuffer = dynaBuf.getDataLength();
	}
	else if( eTextType == XFLM_UTF8_TEXT)
	{
		if( RC_BAD( rc = f_getUTF8Length( (FLMBYTE *)pvValue, uiNumBytesInBuffer,
			&uiNumBytesInBuffer, &uiNumCharsInBuffer)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	// If this is an element node, we need to see if there is a
	// child data node

	if( eNodeType == ELEMENT_NODE && getDataChildCount())
	{
		if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}

		bMustAbortOnError = TRUE;
		
		if( getDataChildCount() == 1)
		{
			// If this node only has one data child, delete the child and
			// store the value directly on the element
			
			if( RC_BAD( rc = pNode->deleteNode( pDb)))
			{
				goto Exit;
			}
			
			pNode->Release();
			pNode = NULL;
		}
		else
		{
			rc = pNode->setTextFastPath( pDb, pvValue, uiNumBytesInBuffer,
				eTextType, uiEncDefId);
			goto Exit;
		}
	}

	// Make sure the states of the node and database
	// allow a value to be set.

	if( RC_BAD( rc = canSetValue( pDb, XFLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	// If this is an attribute node, need to use the
	// attribute list object to set the value

	if( eNodeType == ATTRIBUTE_NODE)
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = pDb->updateIndexKeys( 
			getCollection(), this, IX_DEL_NODE_VALUE, bStartOfUpdate, 
			&bIsIndexed)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;

		if( RC_BAD( rc = m_pCachedNode->setUTF8( pDb,
			m_uiAttrNameId, pvValue, uiNumBytesInBuffer,
			uiNumCharsInBuffer, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( bIsIndexed)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				getCollection(), this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
			{
				goto Exit;
			}
		}

		// Update the node

		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			setDataLength( 0);
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb,
			m_pCachedNode, m_uiAttrNameId)))
		{
			goto Exit;
		}
	
		goto Exit;
	}

	pRfl->disableLogging( &uiRflToken);

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}
	
	unsetFlags( FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	bMustAbortOnError = TRUE;

	// Update the index keys

	if( getNameId())
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( 
			getCollection(), this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
	}
	else
	{
		bIsIndexed = FALSE;
	}

	// Verify and save the encryption scheme

	if( uiEncDefId)
	{
		if( RC_BAD( rc = pDb->m_pDict->getEncDef( uiEncDefId, NULL)))
		{
			flmAssert( 0);
			goto Exit;
		}
	}

	setEncDefId( uiEncDefId);

	// Set the value

	uiValLen = 0;

	if( pvValue && uiNumBytesInBuffer)
	{
		FLMUINT		uiSenLen;
		FLMBYTE *	pucTmp;
		FLMBYTE *	pucValue = (FLMBYTE *)pvValue;
		FLMBOOL		bNullTerminate = FALSE;
		
		if( pucValue[ uiNumBytesInBuffer - 1] != 0)
		{
			bNullTerminate = TRUE;
		}
		
		uiSenLen = f_getSENByteCount( uiNumCharsInBuffer);
		uiValLen = uiNumBytesInBuffer + uiSenLen + (bNullTerminate ? 1 : 0);

		if( calcDataBufSize( uiValLen) > getDataBufSize())
		{
			if( RC_BAD( rc = resizeDataBuffer( uiValLen, FALSE)))
			{
				goto Exit;
			}
		}

		pucTmp = getDataPtr();
		f_encodeSENKnownLength( uiNumCharsInBuffer, uiSenLen, &pucTmp);
		f_memcpy( pucTmp, pucValue, uiNumBytesInBuffer);
		
		if( bNullTerminate)
		{
			pucTmp[ uiNumBytesInBuffer] = 0;
		}
	}

	setDataLength( uiValLen);

	// Clear flags

	unsetFlags( FDOM_VALUE_ON_DISK | FDOM_FIXED_SIZE_HEADER);

	// Update the node

	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		setDataLength( 0);
		goto Exit;
	}

	if( bIsIndexed)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( 
			getCollection(), this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
	}
	
	// Log the node to the RFL
	
	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logNodeSetValue( pDb, 
		RFL_NODE_SET_TEXT_VALUE_PACKET, m_pCachedNode))) 
	{
		goto Exit;
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( RC_BAD( rc))
	{
		if( bMustAbortOnError)
		{
			pDb->setMustAbortTrans( rc);
		}
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}
	
	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setBinaryStreaming(
	IF_Db *					ifpDb,
	const void *			pvValue,
	FLMUINT					uiLength,
	FLMBOOL					bLast,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	F_Db *					pDb = (F_Db *)ifpDb;
	F_Database *			pDatabase = pDb->m_pDatabase;
	F_Rfl *					pRfl = pDatabase->m_pRfl;
	FLMBYTE *				pucValue = (FLMBYTE *)pvValue;
	FLMBYTE *				pucUpdBuffer;
	FLMUINT					uiUpdBufferSize;
	FLMUINT					uiTmp;
	FLMBOOL					bStartedTrans = FALSE;
	F_DOMNode *				pNode = NULL;
	FLMBOOL					bMustAbortOnError = FALSE;
	FLMUINT					uiRflToken = 0;
	eDomNodeType			eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();

	// Not supported on attribute nodes
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If this is an element node, we need to find or create
	// the child data node

	if( eNodeType == ELEMENT_NODE)
	{
		// The streaming interface is not directly supported on element nodes.
		// If the node already has a value and does not already have a child data
		// node, we can clear the value on the element, create a data node, and
		// allow the operation to continue.
		
		if( getDataLength())
		{
			if( getDataChildCount())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			
			bMustAbortOnError = TRUE;
			
			if( RC_BAD( rc = clearNodeValue( pDb)))
			{
				goto Exit;
			}
		}
		else if( getDataChildCount())
		{
			if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				
				goto Exit;
			}
		}
	
		if( !pNode)
		{
			if( RC_BAD( rc = createNode( (IF_Db *)pDb, DATA_NODE, 
				getNameId(), XFLM_LAST_CHILD, (IF_DOMNode **)&pNode)))
			{
				goto Exit;
			}

			bMustAbortOnError = TRUE;
		}

		if( RC_BAD( rc = pNode->setBinary( ifpDb, pucValue, uiLength, bLast,
													  uiEncDefId)))
		{
			goto Exit;
		}

		goto Exit;
	}
	
	// Make sure the state of the node and database
	// allow a value to be set.

	if( RC_BAD( rc = canSetValue( pDb, XFLM_BINARY_TYPE)))
	{
		goto Exit;
	}

	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);
	
	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}

	unsetFlags( FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	bMustAbortOnError = TRUE;
	
	// If we are in the middle of streaming data into a node,
	// any error will probably leave the database in a bad
	// state.  Because of this, we want to force the
	// transaction to abort.

	if( pDatabase->m_pPendingInput)
	{
		if( pDatabase->m_pPendingInput != m_pCachedNode)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
	}

	pucUpdBuffer = pDatabase->m_pucUpdBuffer;

	// NOTE: So that flushNode would not have to allocate a buffer for
	// writing out data with a non-padded header, we reserve enough
	// room in pDatabase->m_pucUpdBuffer so that it could hold a maximum
	// header plus all of the data if it needed to.
	//
	// Also, make sure there is room to encrypt the data.  That is why
	// we subtract another 16 bytes.

	uiUpdBufferSize = pDatabase->m_uiUpdBufferSize - 
								MAX_DOM_HEADER_SIZE - ENCRYPT_MIN_CHUNK_SIZE;

	// Open the pending input stream

	if( !pDatabase->m_pPendingInput)
	{
		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
				this, IX_DEL_NODE_VALUE, TRUE)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = openPendingInput( pDb, XFLM_BINARY_TYPE)))
		{
			goto Exit;
		}
	}

	// Save the encryption scheme

	setEncDefId( uiEncDefId);

	while( uiLength)
	{
		if( (uiTmp = f_min( uiLength,
			uiUpdBufferSize - pDatabase->m_uiUpdByteCount)) == 0)
		{
			if( RC_BAD( rc = flushPendingInput( pDb, FALSE)))
			{
				goto Exit;
			}
			continue;
		}

		f_memcpy( &pucUpdBuffer[ pDatabase->m_uiUpdByteCount], pucValue, uiTmp);
		pDatabase->m_uiUpdByteCount += uiTmp;
		pucValue += uiTmp;
		uiLength -= uiTmp;
	}

	if( bLast)
	{
		if( pDatabase->m_bUpdFirstBuf)
		{
			if( pDatabase->m_uiUpdByteCount)
			{
				if( calcDataBufSize( pDatabase->m_uiUpdByteCount) > getDataBufSize())
				{
					if( RC_BAD( rc = resizeDataBuffer( 
						pDatabase->m_uiUpdByteCount, FALSE)))
					{
						goto Exit;
					}
				}
			}

			setDataLength( pDatabase->m_uiUpdByteCount);

			if( pDatabase->m_uiUpdByteCount)
			{
				f_memcpy( getDataPtr(), pucUpdBuffer, pDatabase->m_uiUpdByteCount);
			}

			// Clear unwanted flags

			unsetFlags( FDOM_VALUE_ON_DISK | FDOM_FIXED_SIZE_HEADER);

			// Update the node

			if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			FLMBYTE		ucKey[ FLM_MAX_NUM_BUF_SIZE];
			FLMUINT		uiKeyLen;
			FLMUINT		uiHeaderStorageSize;
			FLMUINT32	ui32BlkAddr;
			FLMUINT		uiOffsetIndex;

			// Flush anything that is in the current buffer

			if( RC_BAD( rc = flushPendingInput( pDb, TRUE)))
			{
				goto Exit;
			}

			// Output the header
			
			if( RC_BAD( rc = headerToBuf( TRUE, pDatabase->m_pucUpdBuffer, 
				&uiHeaderStorageSize, NULL, NULL)))
			{
				goto Exit;
			}

			// Replace the header

			uiKeyLen = sizeof( ucKey);
			if( RC_BAD( rc = flmNumber64ToStorage( getNodeId(), &uiKeyLen,
				ucKey, FALSE, TRUE)))
			{
				goto Exit;
			}
			
			ui32BlkAddr = getBlkAddr();
			uiOffsetIndex = getOffsetIndex();

			if( RC_BAD( rc = 	pDatabase->m_pPendingBTree->btReplaceEntry(
				ucKey, uiKeyLen,
				pDatabase->m_pucUpdBuffer, uiHeaderStorageSize,
				TRUE, TRUE, FALSE, &ui32BlkAddr, &uiOffsetIndex)))
			{
				goto Exit;
			}
			
			setBlkAddr( ui32BlkAddr);
			setOffsetIndex( uiOffsetIndex);

			// Clear the dirty flag and the new flag.

			unsetNodeDirtyAndNew( pDb);
		}
		
		pDatabase->endPendingInput();

		if( getNameId())
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
				this, IX_ADD_NODE_VALUE, FALSE)))
			{
				goto Exit;
			}
		}
		
		// Log the node to the RFL
		
		pRfl->enableLogging( &uiRflToken);

		if( RC_BAD( rc = pRfl->logNodeSetValue( pDb, 
			RFL_NODE_SET_BINARY_VALUE_PACKET, m_pCachedNode))) 
		{
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDatabase->endPendingInput();
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setBinaryFastPath(
	IF_Db *					ifpDb,
	const void *			pvValue,
	FLMUINT					uiLength,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	F_Db *					pDb = (F_Db *)ifpDb;
	F_Database *			pDatabase = pDb->m_pDatabase;
	F_Rfl *					pRfl = pDatabase->m_pRfl;
	FLMBYTE *				pucValue = (FLMBYTE *)pvValue;
	FLMBOOL					bStartedTrans = FALSE;
	F_DOMNode *				pNode = NULL;
	FLMBOOL					bMustAbortOnError = FALSE;
	FLMUINT					uiRflToken = 0;
	eDomNodeType			eNodeType;
	FLMBOOL					bIsIndexed = TRUE;
	FLMBOOL					bStartOfUpdate = TRUE;

	// Make sure a transaction is active
	
	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Cannot set a value if input is still pending
	
	if( pDatabase->m_pPendingInput)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();

	// If this is an element node, we need to find or create
	// the child data node

	if( eNodeType == ELEMENT_NODE && getDataChildCount())
	{
		if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			
			goto Exit;
		}
		
		bMustAbortOnError = TRUE;
		
		if( getDataChildCount() == 1)
		{
			// If this node only has one data child, delete the child and
			// store the value directly on the element
			
			if( RC_BAD( rc = pNode->deleteNode( ifpDb)))
			{
				goto Exit;
			}
			
			pNode->Release();
			pNode = NULL;
		}
		else
		{
			rc = pNode->setBinaryFastPath( ifpDb, pucValue, uiLength, uiEncDefId);
			goto Exit;
		}
	}
	else if( eNodeType == ATTRIBUTE_NODE)
	{
		// Make sure the state of the node and database
		// allow a value to be set.
	
		if( RC_BAD( rc = canSetValue( pDb, XFLM_BINARY_TYPE)))
		{
			goto Exit;
		}
		
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( (F_Db *)ifpDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
			this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
			
		if( RC_BAD( rc = m_pCachedNode->setBinary( 
			pDb, m_uiAttrNameId, pvValue, uiLength, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( bIsIndexed)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				getCollection(), this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
			{
				goto Exit;
			}

			bStartOfUpdate = FALSE;
		}

		// Update the node

		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb, 
			m_pCachedNode, m_uiAttrNameId))) 
		{
			goto Exit;
		}
	
		goto Exit;
	}
	
	// If node is a text data type, convert the binary to base 64 encoding
	// and store as text.
	
	if( getDataType() == XFLM_TEXT_TYPE)
	{
		// Convert to base64 text and call the routine to set as native.
		
		rc = storeBinaryAsText( (F_Db *)ifpDb, pvValue, uiLength, uiEncDefId);
		goto Exit;
	}
	
	// Make sure the state of the node and database
	// allow a value to be set.

	if( RC_BAD( rc = canSetValue( pDb, XFLM_BINARY_TYPE)))
	{
		goto Exit;
	}

	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);
	
	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;
	unsetFlags( FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);

	if( getNameId())
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
			this, IX_DEL_NODE_VALUE, bStartOfUpdate, &bIsIndexed)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
	}
	else
	{
		bIsIndexed = FALSE;
	}
	
	setEncDefId( uiEncDefId);
	
	if( calcDataBufSize( uiLength) > getDataBufSize())
	{
		if( RC_BAD( rc = resizeDataBuffer( uiLength, FALSE)))
		{
			goto Exit;
		}
	}
		
	setDataLength( uiLength);

	if( uiLength)
	{
		f_memcpy( getDataPtr(), pvValue, uiLength);
	}

	// Clear unwanted flags

	unsetFlags( FDOM_VALUE_ON_DISK | FDOM_FIXED_SIZE_HEADER);

	// Update the node

	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		goto Exit;
	}
		
	if( bIsIndexed)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
			this, IX_ADD_NODE_VALUE, bStartOfUpdate)))
		{
			goto Exit;
		}

		bStartOfUpdate = FALSE;
	}
	
	// Log the node to the RFL
	
	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logNodeSetValue( pDb, 
		RFL_NODE_SET_BINARY_VALUE_PACKET, m_pCachedNode))) 
	{
		goto Exit;
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::clearNodeValue(
	F_Db *					pDb)
{
	RCODE						rc = NE_XFLM_OK;
	F_Database *			pDatabase = pDb->m_pDatabase;
	F_Rfl *					pRfl = pDatabase->m_pRfl;
	FLMBOOL					bStartedTrans = FALSE;
	FLMBOOL					bMustAbortOnError = FALSE;
	FLMUINT					uiRflToken = 0;

	// Make sure a transaction is active
	
	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Cannot clear a value if input is still pending
	
	if( pDatabase->m_pPendingInput)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);
	bMustAbortOnError = TRUE;
	
	if( RC_BAD( rc = setStorageValue( pDb, NULL, 0, 0, TRUE)))
	{
		goto Exit;
	}

	// Log the update to the RFL
	
	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logNodeClearValue( pDb, getCollection(),
		getNodeId(), m_uiAttrNameId))) 
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getIStream(
	F_Db *						pDb,
	F_NodeBufferIStream *	pStackStream,
	IF_PosIStream **			ppIStream,
	FLMUINT *					puiDataType,
	FLMUINT *					puiDataLength)
{
	RCODE					rc = NE_XFLM_OK;
	IF_PosIStream *	pIStream = NULL;
	F_DOMNode *			pNode = NULL;
	F_CachedNode *		pCachedNode = this;
	eDomNodeType		eNodeType = getNodeType();

	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	if( m_uiFlags & FDOM_VALUE_ON_DISK)
	{
		F_BTreeIStream *	pBTreeIStream;
		F_ENCDEF *			pEncDef;
		FLMUINT				uiIVLen;
		FLMUINT				uiLen;
		F_NODE_INFO			nodeInfo;
		FLMUINT				uiTmpFlags;

		if( eNodeType == ELEMENT_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->flushDirtyNode( this)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = gv_XFlmSysData.pNodePool->allocBTreeIStream(
											&pBTreeIStream)))
		{
			goto Exit;
		}
		pIStream = pBTreeIStream;

		if( RC_BAD( rc = pBTreeIStream->openStream( pDb, getCollection(),
			getNodeId(), getBlkAddr(), getOffsetIndex())))
		{
			goto Exit;
		}

		// Skip the node header

		if( RC_BAD( rc = flmReadNodeInfo( getCollection(), getNodeId(), 
			pBTreeIStream, (FLMUINT)pBTreeIStream->remainingSize(), TRUE,
			&nodeInfo, &uiTmpFlags)))
		{
			goto Exit;
		}

		// Read the IV if data is encrypted
		
		if( getEncDefId())
		{
			if( RC_BAD( rc = pDb->m_pDict->getEncDef( getEncDefId(), &pEncDef)))
			{
				goto Exit;
			}
			
			uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( uiIVLen == 8 || uiIVLen == 16);
			
			if( RC_BAD( rc = pBTreeIStream->read( (char *)pBTreeIStream->m_ucIV,
										uiIVLen, &uiLen)))
			{
				goto Exit;
			}
			
			flmAssert( uiLen == uiIVLen);
			pBTreeIStream->m_bDataEncrypted = TRUE;
		}
		
		pBTreeIStream->m_uiEncDefId = getEncDefId();
		pBTreeIStream->m_uiDataLength = getDataLength();
	}
	else
	{
		F_NodeBufferIStream *	pNodeBufferIStream;
		FLMUINT64					ui64TmpNodeId;

		if( eNodeType == ELEMENT_NODE && getDataChildCount())
		{
			ui64TmpNodeId = getFirstChildId();
			
			for( ;;)
			{
				if( !ui64TmpNodeId)
				{
					break;
				}
		
				if( RC_BAD( rc = ((IF_Db *)pDb)->getNode(
					getCollection(), ui64TmpNodeId, (IF_DOMNode **)&pNode)))
				{
					goto Exit;
				}
		
				if( pNode->getNodeType() == DATA_NODE)
				{
					pCachedNode = pNode->m_pCachedNode;
					break;
				}
		
				ui64TmpNodeId = pNode->getNextSibId();
			}
		}
		
		if( pStackStream)
		{
			pNodeBufferIStream = pStackStream;
			pStackStream->AddRef();
			flmAssert( !pStackStream->m_pCachedNode);
		}
		else
		{
			if( (pNodeBufferIStream = f_new F_NodeBufferIStream) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
		}
		
		pIStream = pNodeBufferIStream;

		if( RC_BAD( rc = pNodeBufferIStream->openStream( 
			(const char *)pCachedNode->getDataPtr(), 
			pCachedNode->getDataLength())))
		{
			goto Exit;
		}
		
		if( !pStackStream)
		{
			pNodeBufferIStream->m_pCachedNode = pCachedNode;
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
			pCachedNode->incrNodeUseCount();
			pCachedNode->incrStreamUseCount();
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		}
	}

	if( puiDataType)
	{
		*puiDataType = pCachedNode->getDataType();
	}
	
	if( puiDataLength)
	{
		*puiDataLength = pCachedNode->getDataLength();
	}

	*ppIStream = pIStream;
	pIStream = NULL;

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( pNode)
	{
		pNode->Release();
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getRawIStream(
	F_Db *				pDb,
	IF_PosIStream **	ppIStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_PosIStream *	pIStream = NULL;
	F_BTreeIStream *	pBTreeIStream;
	eDomNodeType		eNodeType = getNodeType();
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	// Flush all dirty nodes out to the B-Tree
	
	if( RC_BAD( rc = pDb->flushDirtyNode( this)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_XFlmSysData.pNodePool->allocBTreeIStream(
		&pBTreeIStream)))
	{
		goto Exit;
	}
	pIStream = pBTreeIStream;

	if( RC_BAD( rc = pBTreeIStream->openStream( pDb, getCollection(),
		getNodeId(), getBlkAddr(), getOffsetIndex())))
	{
		goto Exit;
	}

	*ppIStream = pIStream;
	pIStream = NULL;

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getIStream(
	F_Db *						pDb,
	F_NodeBufferIStream *	pStackStream,
	IF_PosIStream **			ppIStream,
	FLMUINT *					puiDataType,
	FLMUINT *					puiDataLength)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pNode = NULL;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, NULL)))
	{
		goto Exit;
	}

	// Sync the node to make sure it is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	switch( getNodeType())
	{
		case DATA_NODE:
		case COMMENT_NODE:
		case ANNOTATION_NODE:
		case CDATA_SECTION_NODE:
		{
			if( RC_BAD( rc = m_pCachedNode->getIStream( pDb, pStackStream,
				ppIStream, puiDataType, puiDataLength)))
			{
				goto Exit;
			}

			break;
		}

		case ELEMENT_NODE:
		{
			if( getDataChildCount())
			{
				if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
				{
					if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					
					goto Exit;
				}
				
				if( RC_BAD( rc = pNode->m_pCachedNode->getIStream( pDb,
					pStackStream, ppIStream, puiDataType, puiDataLength)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = m_pCachedNode->getIStream( pDb, 
					pStackStream, ppIStream, puiDataType, puiDataLength)))
				{
					goto Exit;
				}
			}
			
			break;
		}

		case ATTRIBUTE_NODE:
		{
			if( RC_BAD( rc = m_pCachedNode->getIStream( 
				pDb, m_uiAttrNameId, pStackStream, ppIStream, 
				puiDataType, puiDataLength)))
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

	if( pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getTextIStream(
	F_Db *						pDb,
	F_NodeBufferIStream *	pStackStream,
	IF_PosIStream **			ppIStream,
	FLMUINT *					puiNumChars)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiDataType;

	*ppIStream = NULL;
	*puiNumChars = 0;

	if( RC_BAD( rc = getIStream( pDb, pStackStream, ppIStream, &uiDataType)))
	{
		goto Exit;
	}

	if( uiDataType != XFLM_TEXT_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_DATA_TYPE);
		goto Exit;
	}

	// Skip the leading SEN so that the stream is positioned to
	// read raw utf8.

	if( (*ppIStream)->remainingSize())
	{
		if( RC_BAD( rc = f_readSEN( *ppIStream, puiNumChars)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc) && *ppIStream)
	{
		(*ppIStream)->Release();
		*ppIStream = NULL;
		*puiNumChars = 0;
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getNumber64(
	F_Db *				pDb,
	FLMUINT64 *			pui64Num,
	FLMBOOL *			pbNeg)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiDataType;
	FLMUINT64				ui64Num;
	FLMBOOL					bNeg;
	F_DOMNode *				pNode = NULL;
	eDomNodeType			eNodeType;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();

	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getNumber64( 
			pDb, m_uiAttrNameId, &ui64Num, &bNeg)))
		{
			goto Exit;
		}
	}
	else if ( !getQuickNumber64( &ui64Num, &bNeg))
	{
		if( eNodeType == ELEMENT_NODE && getDataChildCount())
		{
			if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				
				goto Exit;
			}
			
			rc = pNode->getNumber64( pDb, pui64Num, pbNeg);
			goto Exit;
		}
		else
		{
			if( RC_BAD( rc = getIStream( pDb, &bufferIStream, 
				&pIStream, &uiDataType)))
			{
				goto Exit;
			}
		
			if( RC_BAD( rc = flmReadStorageAsNumber( pIStream, uiDataType,
				&ui64Num, &bNeg)))
			{
				goto Exit;
			}
		}
	}

	if( pui64Num)
	{
		*pui64Num = ui64Num;
	}
	
	if( pbNeg)
	{
		*pbNeg = bNeg;
	}

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}
	
	if( pNode)
	{
		pNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:		Allocate data for a unicode element and retrieve it.
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getUnicode(
	IF_Db *			ifpDb,
	FLMUNICODE **	ppuzUnicode)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLen;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// Get the unicode length (does not include NULL terminator)

	if( RC_BAD( rc = getUnicodeChars( pDb, &uiLen)))
	{
		goto Exit;
	}

	if( uiLen)
	{
		FLMUINT	uiBufSize = (uiLen + 1) * sizeof( FLMUNICODE);

		if( RC_BAD( rc = f_alloc( uiBufSize, ppuzUnicode)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getUnicode( pDb, 
			*ppuzUnicode, uiBufSize, 0, uiLen, &uiLen)))
		{
			goto Exit;
		}
	}
	else
	{
		*ppuzUnicode = NULL;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getUnicode(
	IF_Db *					ifpDb,
	FLMUNICODE *			puzBuffer,
	FLMUINT					uiBufSize,
	FLMUINT					uiCharOffset,
	FLMUINT					uiMaxCharsRequested,
	FLMUINT *				puiCharsReturned,
	FLMUINT *				puiBufferBytesUsed)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiDataType;
	FLMUINT					uiDataLength;
	F_NodeBufferIStream	bufferStream;
	IF_PosIStream *		pIStream = NULL;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getIStream( pDb, &bufferStream, 
		&pIStream, &uiDataType, &uiDataLength)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmReadStorageAsText(
		pIStream, NULL, uiDataLength, uiDataType, puzBuffer,
		uiBufSize, XFLM_UNICODE_TEXT,
		uiMaxCharsRequested, uiCharOffset, puiCharsReturned,
		puiBufferBytesUsed)))
	{
		goto Exit;
	}

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getUnicode(
	IF_Db *			ifpDb,
	F_DynaBuf *		pBuffer)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMUINT			uiBufSize;
	FLMUINT			uiChars;
	void *			pvBuffer = NULL;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	pBuffer->truncateData( 0);
	
	if( RC_BAD( rc = getUnicode( ifpDb, NULL, 0, 0, ~((FLMUINT)0), &uiChars)))
	{
		goto Exit;
	}
	
	uiBufSize = (uiChars + 1) * sizeof( FLMUNICODE);
	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getUnicode( ifpDb, (FLMUNICODE *)pvBuffer, uiBufSize, 0,
		~((FLMUINT)0), NULL)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getUTF8(
	IF_Db *			ifpDb,
	FLMBYTE *		pszValue,
	FLMUINT 			uiBufferSize,
	FLMUINT			uiCharOffset,
	FLMUINT			uiMaxCharsRequested,
	FLMUINT *		puiCharsReturned,
	FLMUINT *		puiBufferBytesUsed)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiDataType;
	FLMUINT					uiDataLength;
	F_DOMNode *				pNode = NULL;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	switch( getNodeType())
	{
		case DATA_NODE:
		case COMMENT_NODE:
		case ANNOTATION_NODE:
		case CDATA_SECTION_NODE:
		{
			pNode = this;
			pNode->AddRef();
			break;
		}

		case ATTRIBUTE_NODE:
		{
			pNode = this;
			pNode->AddRef();
			goto SlowDecode;
		}

		case ELEMENT_NODE:
		{
			if( getDataChildCount())
			{
				if( RC_BAD( rc = getChild( pDb, DATA_NODE, (IF_DOMNode **)&pNode)))
				{
					if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					}
					
					goto Exit;
				}
			}
			else
			{
				pNode = this;
				pNode->AddRef();
			}
			
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	if( (pNode->getModeFlags() & FDOM_VALUE_ON_DISK) || 
			pNode->getDataType() != XFLM_TEXT_TYPE || uiCharOffset)
	{
SlowDecode:

		if( RC_BAD( rc = pNode->getIStream( pDb, &bufferIStream, &pIStream, 
			&uiDataType, &uiDataLength)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmReadStorageAsText( 
			pIStream, NULL, uiDataLength,
			uiDataType, pszValue, uiBufferSize, XFLM_UTF8_TEXT,
			uiMaxCharsRequested, uiCharOffset, puiCharsReturned, 
			puiBufferBytesUsed)))
		{
			goto Exit;
		}
	}
	else
	{
		const FLMBYTE *	pucBuffer = pNode->getDataPtr();
		const FLMBYTE *	pucEnd = pucBuffer + pNode->getDataLength();
		FLMUINT				uiCharCount = 0;
		FLMUINT				uiStrByteLen = 0;

		if( pucBuffer)
		{
			if( RC_BAD( rc = f_decodeSEN( &pucBuffer, pucEnd, &uiCharCount)))
			{
				goto Exit;
			}

			uiStrByteLen = (FLMUINT)(pucEnd - pucBuffer);
		}

		if( uiCharCount > uiMaxCharsRequested || 
			(pszValue && uiBufferSize < uiStrByteLen))
		{
			goto SlowDecode;
		}

		if( pszValue)
		{
			if( uiStrByteLen)
			{
				f_memcpy( pszValue, pucBuffer, uiStrByteLen);
			}
			else if( uiBufferSize > 0)
			{
				*pszValue = 0;
			}
		}

		if( puiCharsReturned)
		{
			*puiCharsReturned = uiCharCount;
		}

		if( puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = uiStrByteLen;
		}
	}

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( pNode)
	{
		pNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getUTF8(
	IF_Db *			ifpDb,
	FLMBYTE **		ppszUTF8)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMUINT			uiBufSize;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = getUTF8( ifpDb, NULL, 0, 0,
		FLM_MAX_UINT, NULL, &uiBufSize)))
	{
		goto Exit;
	}
	
	if( uiBufSize)
	{
		if( RC_BAD( rc = f_alloc( uiBufSize, ppszUTF8)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = getUTF8( ifpDb, *ppszUTF8, uiBufSize, 0,
			FLM_MAX_UINT, NULL, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		*ppszUTF8 = NULL;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getUTF8(
	IF_Db *			ifpDb,
	F_DynaBuf *		pBuffer)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMUINT			uiBufSize;
	void *			pvBuffer = NULL;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	pBuffer->truncateData( 0);
	
	if( RC_BAD( rc = getUTF8( ifpDb, NULL, 0, 0,
		FLM_MAX_UINT, NULL, &uiBufSize)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getUTF8( ifpDb, (FLMBYTE *)pvBuffer, uiBufSize, 0,
		FLM_MAX_UINT, NULL, NULL)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getBinary(
	IF_Db *					ifpDb,
	void *					pvValue,
	FLMUINT					uiByteOffset,
	FLMUINT					uiBytesRequested,
	FLMUINT *				puiBytesReturned)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE *				pucValue = (FLMBYTE *)pvValue;
	IF_PosIStream *		pIStream = NULL;
	IF_IStream *			pDecoderStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMUINT					uiTmp;
	FLMUINT					uiDataType;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// If a NULL buffer is passed in, just return the
	// data length

	if( !pucValue)
	{
		if( RC_BAD( rc = getDataLength( pDb, &uiTmp)))
		{
			goto Exit;
		}

		if( uiByteOffset <= uiTmp)
		{
			*puiBytesReturned = uiTmp - uiByteOffset;
		}
		else
		{
			*puiBytesReturned = 0;
		}

		goto Exit;
	}
	
	if( RC_BAD( rc = getIStream( pDb, &bufferIStream, &pIStream, &uiDataType)))
	{
		goto Exit;
	}

	if( uiDataType == XFLM_TEXT_TYPE)
	{
		F_AsciiStorageStream	asciiStream;
		
		if( RC_BAD( rc = asciiStream.openStream( pIStream)))
		{
			goto Exit;
		}
		else
		{
			if( RC_BAD( rc = FlmOpenBase64DecoderIStream( 
				&asciiStream, &pDecoderStream)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = flmReadStorageAsBinary( 
				pDecoderStream, (FLMBYTE *)pucValue,
				uiBytesRequested, uiByteOffset, puiBytesReturned)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if( RC_BAD( rc = flmReadStorageAsBinary( 
			pIStream, (FLMBYTE *)pucValue,
			uiBytesRequested, uiByteOffset, puiBytesReturned)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDecoderStream)
	{
		pDecoderStream->Release();
	}

	if( pIStream)
	{
		pIStream->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getBinary(
	IF_Db *			ifpDb,
	F_DynaBuf *		pBuffer)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMUINT			uiBufSize;
	void *			pvBuffer = NULL;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	pBuffer->truncateData( 0);
	
	if( RC_BAD( rc = getBinary( ifpDb, NULL, 0, FLM_MAX_UINT, &uiBufSize)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getBinary( ifpDb, pvBuffer, 0, uiBufSize, NULL)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getAttributeValueNumber(
	F_Db *		pDb,
	FLMUINT		uiAttrName,
	FLMUINT64 *	pui64Num,
	FLMBOOL *	pbNeg)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pCachedNode->getNumber64( pDb, uiAttrName, 
		pui64Num, pbNeg)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getAttributeValueText(
	IF_Db *					ifpDb,
	FLMUINT					uiAttrName,
	eXFlmTextType			eTextType,
	void *					pvBuffer,
	FLMUINT					uiBufSize,
	FLMUINT *				puiCharsReturned,
	FLMUINT *				puiBufferBytesUsed)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBYTE *				pucStorageData = NULL;
	FLMUINT					uiDataType;
	FLMUINT					uiDataLength;
	F_AttrItem *			pAttrItem;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( (pAttrItem = m_pCachedNode->getAttribute( uiAttrName, NULL)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	if( !pAttrItem->m_uiEncDefId)
	{
		pucStorageData = pAttrItem->getAttrDataPtr();
		uiDataLength = pAttrItem->getAttrDataLength();
		uiDataType = pAttrItem->m_uiDataType;
		
		if( uiDataType == XFLM_TEXT_TYPE && eTextType == XFLM_UTF8_TEXT)
		{
			const FLMBYTE *	pucStart = pucStorageData;
			const FLMBYTE *	pucEnd = pucStart + uiDataLength;
			FLMUINT				uiCharCount = 0;
			FLMUINT				uiStrByteLen = 0;
	
			if( pucStart)
			{
				if( RC_BAD( rc = f_decodeSEN( &pucStart, pucEnd, &uiCharCount)))
				{
					goto Exit;
				}
	
				uiStrByteLen = (FLMUINT)(pucEnd - pucStart);
				
				if( uiBufSize < uiStrByteLen)
				{
					goto SlowDecode;
				}
				
				f_memcpy( pvBuffer, pucStart, uiStrByteLen);
			}
	
			if( puiCharsReturned)
			{
				*puiCharsReturned = uiCharCount;
			}
	
			if( puiBufferBytesUsed)
			{
				*puiBufferBytesUsed = uiStrByteLen;
			}
			
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pCachedNode->getIStream( pDb, uiAttrName, 
			&bufferIStream, &pIStream, &uiDataType, &uiDataLength))) 
		{
			goto Exit;
		}
	}
	
SlowDecode:
	
	if( RC_BAD( rc = flmReadStorageAsText( 
		pIStream, pucStorageData, uiDataLength, uiDataType, pvBuffer,
		uiBufSize, eTextType, FLM_MAX_UINT, 0, puiCharsReturned,
		puiBufferBytesUsed)))
	{
		goto Exit;
	}

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueUnicode(
	IF_Db *			ifpDb,
	FLMUINT			uiAttrName,
	F_DynaBuf *		pBuffer)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiBufSize;
	void *			pvBuffer = NULL;
	
	pBuffer->truncateData( 0);
	
	if( RC_BAD( rc = getAttributeValueUnicode( ifpDb, uiAttrName, 
		NULL, 0, NULL, &uiBufSize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getAttributeValueUnicode( ifpDb, uiAttrName, 
		(FLMUNICODE *)pvBuffer, uiBufSize, NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueUnicode(
	IF_Db *					ifpDb,
	FLMUINT					uiAttrName,
	FLMUNICODE **			ppuzUnicode)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiBufSize;

	if( RC_BAD( rc = getAttributeValueUnicode( ifpDb, uiAttrName, 
		NULL, 0, NULL, &uiBufSize)))
	{
		goto Exit;
	}
	
	if( uiBufSize)
	{
		if( RC_BAD( rc = f_alloc( uiBufSize, ppuzUnicode)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getAttributeValueUnicode( ifpDb, uiAttrName, 
			*ppuzUnicode, uiBufSize)))
		{
			goto Exit;
		}
	}
	else
	{
		*ppuzUnicode = NULL;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueUTF8(
	IF_Db *					ifpDb,
	FLMUINT					uiAttrName,
	FLMBYTE **				ppszValue)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiBufSize;

	if( RC_BAD( rc = getAttributeValueUTF8( ifpDb, uiAttrName, 
		NULL, 0, NULL, &uiBufSize)))
	{
		goto Exit;
	}
	
	if( uiBufSize)
	{
		if( RC_BAD( rc = f_alloc( uiBufSize, ppszValue)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getAttributeValueUTF8( ifpDb, uiAttrName, 
			*ppszValue, uiBufSize)))
		{
			goto Exit;
		}
	}
	else
	{
		*ppszValue = NULL;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueUTF8(
	IF_Db *			ifpDb,
	FLMUINT			uiAttrName,
	F_DynaBuf *		pBuffer)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiBufSize;
	void *			pvBuffer = NULL;
	
	pBuffer->truncateData( 0);
	
	if( RC_BAD( rc = getAttributeValueUTF8( ifpDb, uiAttrName, 
		NULL, 0, NULL, &uiBufSize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getAttributeValueUTF8( ifpDb, uiAttrName, 
		(FLMBYTE *)pvBuffer, uiBufSize)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueBinary(
	IF_Db *					ifpDb,
	FLMUINT					uiAttrName,
	void *					pvValue,
	FLMUINT					uiBufferSize,
	FLMUINT *				puiLength)
{
	RCODE						rc = NE_XFLM_OK;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pCachedNode->getBinary( pDb, uiAttrName, 
		pvValue, uiBufferSize, puiLength)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->abortTrans();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getAttributeValueBinary(
	IF_Db *				ifpDb,
	FLMUINT				uiAttrName,
	F_DynaBuf *			pBuffer)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiBufSize;
	void *				pvBuffer = NULL;
	
	pBuffer->truncateData( 0);

	if( RC_BAD( rc = getAttributeValueBinary( ifpDb, uiAttrName, NULL, 0, 
		&uiBufSize)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiBufSize, &pvBuffer)))
	{
		goto Exit;
	}
		
	if( RC_BAD( rc = getAttributeValueBinary( ifpDb, uiAttrName, pvBuffer, 
		uiBufSize, &uiBufSize)))
	{
		goto Exit;
	}
	
Exit:

	if( bStartedTrans)
	{
		ifpDb->transAbort();
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::setAttributeValueNumber(
	IF_Db *				ifpDb,
	FLMUINT				uiAttrName,
	FLMINT64				i64Value,
	FLMUINT64			ui64Value,
	FLMUINT				uiEncDefId)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pAttribute = NULL;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bNeg = FALSE;
	FLMBOOL				bIsInIndexDef = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->attrIsInIndexDef( uiAttrName, &bIsInIndexDef)))
	{
		goto Exit;
	}

	if( bIsInIndexDef)
	{
		if( RC_BAD( rc = createAttribute( (IF_Db *)pDb, uiAttrName,
			(IF_DOMNode **)&pAttribute)))
		{
			goto Exit;
		}
		
		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = pAttribute->setNumber64( pDb, i64Value,
			ui64Value, uiEncDefId)))
		{
			goto Exit;
		}
	}
	else
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( !ui64Value)
		{
			if( i64Value < 0)
			{
				bNeg = TRUE;
				ui64Value = (FLMUINT64)-i64Value;
			}
			else
			{
				ui64Value = (FLMUINT64)i64Value;
			}
		}

		if( RC_BAD( rc = m_pCachedNode->setNumber64( pDb,
			uiAttrName, ui64Value, bNeg, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb,
			m_pCachedNode, uiAttrName))) 
		{
			goto Exit;
		}
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if( pAttribute)
	{
		pAttribute->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::setAttributeValueUnicode(
	IF_Db *					ifpDb,
	FLMUINT					uiAttrName,
	const FLMUNICODE *	puzValue,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	F_DOMNode *				pAttribute = NULL;
	FLMBOOL					bStartedTrans = FALSE;
	F_Db *					pDb = (F_Db *)ifpDb;
	FLMBOOL					bMustAbortOnError = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = createAttribute( (IF_Db *)pDb, uiAttrName,
		(IF_DOMNode **)&pAttribute)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttribute->setUnicode( (IF_Db *)pDb, puzValue, 0,
		TRUE, uiEncDefId)))
	{
		goto Exit;
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if( pAttribute)
	{
		pAttribute->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::setAttributeValueBinary(
	IF_Db *				ifpDb,
	FLMUINT				uiAttrName,
	const void *		pvValue,
	FLMUINT				uiLength,
	FLMUINT				uiEncDefId)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pAttribute = NULL;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bIsInIndexDef = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->attrIsInIndexDef( uiAttrName, &bIsInIndexDef)))
	{
		goto Exit;
	}

	if( bIsInIndexDef)
	{
		if( RC_BAD( rc = createAttribute( (IF_Db *)pDb, uiAttrName,
			(IF_DOMNode **)&pAttribute)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttribute->setBinary( 
			(IF_Db *)pDb, (FLMBYTE *)pvValue, uiLength, TRUE, uiEncDefId)))
		{
			goto Exit;
		}
	}
	else
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = m_pCachedNode->setBinary( pDb,
			uiAttrName, pvValue, uiLength, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb,
			m_pCachedNode, uiAttrName))) 
		{
			goto Exit;
		}
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if( pAttribute)
	{
		pAttribute->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}


/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::setAttributeValueUTF8(
	IF_Db *				ifpDb,
	FLMUINT				uiAttrName,
	const FLMBYTE *	pucValue,
	FLMUINT				uiLength,
	FLMUINT				uiEncDefId)
{
	RCODE					rc = NE_XFLM_OK;
	F_DOMNode *			pAttribute = NULL;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiNumCharsInBuffer;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bIsInIndexDef = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBOOL				bMustAbortOnError = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->attrIsInIndexDef( uiAttrName, &bIsInIndexDef)))
	{
		goto Exit;
	}

	if( bIsInIndexDef)
	{
		if( RC_BAD( rc = createAttribute( (IF_Db *)pDb, uiAttrName,
			(IF_DOMNode **)&pAttribute)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttribute->setUTF8( 
			(IF_Db *)pDb, pucValue, uiLength, TRUE, uiEncDefId)))
		{
			goto Exit;
		}
	}
	else
	{
		pRfl->disableLogging( &uiRflToken);

		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}

		bMustAbortOnError = TRUE;

		if( RC_BAD( rc = f_getUTF8Length( pucValue, uiLength,
			&uiLength, &uiNumCharsInBuffer)))
		{
			goto Exit;
		}

		if( RC_BAD( m_pCachedNode->setUTF8( pDb, uiAttrName, pucValue, 
			uiLength, uiNumCharsInBuffer, uiEncDefId)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
		{
			goto Exit;
		}

		// Log the value to the RFL
		
		pRfl->enableLogging( &uiRflToken);
	
		if( RC_BAD( rc = pRfl->logAttrSetValue( pDb,
			m_pCachedNode, uiAttrName))) 
		{
			goto Exit;
		}
	}

	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if( pAttribute)
	{
		pAttribute->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:	Delete a node and all of its child/descendant-nodes.
NOTE:	If the cannot-delete bit or read-only bit is set on the node, the delete
		is not allowed.  However, the child/descendant-nodes cannot-delete and
		read-only bits will NOT be checked.  If a parent node can be deleted,
		then by definition all of its child/descendant nodes can also be deleted.
******************************************************************************/
RCODE F_DOMNode::deleteNode(
	IF_Db *		ifpDb)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64CurNode;
	F_DOMNode *			pCurNode = NULL;
	F_DOMNode *			pParentNode = NULL;
	F_DOMNode *			pTmpNode = NULL;
	FLMBOOL				bMustAbortOnError = FALSE;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMBOOL				bStartOfUpdate;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiCollection;
	FLMUINT				uiFlags = 0;
	FLMUINT				uiRflToken = 0;
	FLMUINT64			ui64MyNodeId;
	FLMBOOL				bIsIndexed;
	eDomNodeType		eNodeType;

	// Start a transaction if necessary

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	uiCollection = getCollection();
	eNodeType = getNodeType();
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = pDb->getNode( uiCollection, getParentId(), &pCurNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}
		
		rc = pCurNode->deleteAttribute( pDb, m_uiAttrNameId);
		goto Exit;
	}

	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// See if the node can be deleted

	if( getModeFlags() & (FDOM_READ_ONLY | FDOM_CANNOT_DELETE))
	{
		rc = RC_SET( NE_XFLM_DELETE_NOT_ALLOWED);
		goto Exit;
	}

	if( isRootNode())
	{
		// Set flags to FLM_UPD_INTERNAL_CHANGE to prevent the node
		// from being added to the document list or constraint
		// checking list - no need since we are deleting the root node.
		
		uiFlags = FLM_UPD_INTERNAL_CHANGE;

		// If we are deleting the root node of a document in the dictionary
		// collection, before deleting it, we must allow the dictionary
		// to be updated.
	
		if( uiCollection == XFLM_DICT_COLLECTION)
		{
			// Call dictDocumentDone with bDeleting flag set to TRUE.

			if( RC_BAD( rc = pDb->dictDocumentDone( 
				getNodeId(), TRUE, NULL)))
			{
				goto Exit;
			}
			
			pDb->m_pDatabase->m_DocumentList.removeNode( uiCollection, 
				getNodeId(), 0);
		}
	}
	bMustAbortOnError = TRUE;

	// Traverse the tree and delete all nodes below and including the
	// node we are starting on.

	ui64MyNodeId = ui64CurNode = getNodeId();
	bStartOfUpdate = TRUE;
	for (;;)
	{
		if (RC_BAD( rc = pDb->getNode( uiCollection, ui64CurNode, &pCurNode)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		// If the current node has children, go to those children
		
		if( pCurNode->getLastChildId())
		{
			ui64CurNode = pCurNode->getLastChildId();
		}
		else if( pCurNode->hasAttributes())
		{
			if( pCurNode->getNodeType() != ELEMENT_NODE)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}

			flmAssert( pCurNode->m_pCachedNode->m_uiAttrCount);

			if( RC_BAD( rc = pCurNode->deleteAttributes( pDb, 0, uiFlags)))
			{
				goto Exit;
			}
		}
		else if (pCurNode->getAnnotationId())
		{
			ui64CurNode = pCurNode->getAnnotationId();
		}
		else
		{
			// Node has no children, no attributes, and no annotations.  It is
			// therefore a leaf node that can be purged.
		
			FLMUINT64	ui64ParentId;
			FLMBOOL		bWasDataNode = FALSE;
			
			// Save the node's parent node before purging it.  That is the
			// node we want to return to.
			
			if( RC_BAD( rc = pCurNode->getParentId( pDb, &ui64ParentId)))
			{
				goto Exit;
			}

			// Update the index

			if( pCurNode->getNodeType() == DATA_NODE)
			{
				bWasDataNode = TRUE;
				
				// Data nodes MUST be children to an element node.
				
				flmAssert( ui64ParentId);
				if (RC_BAD( rc = pDb->getNode( uiCollection, ui64ParentId,
												(IF_DOMNode **)&pParentNode)))
				{
					goto Exit;
				}
					
				if (RC_BAD( rc = pDb->updateIndexKeys( 
					uiCollection, pParentNode, IX_DEL_NODE_VALUE,
					bStartOfUpdate)))
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = pDb->updateIndexKeys( 
					uiCollection, pCurNode, IX_DEL_NODE_VALUE,
					bStartOfUpdate, &bIsIndexed)))
				{
					goto Exit;
				}
				bStartOfUpdate = FALSE;
				
				if( bIsIndexed)
				{
					if (RC_BAD( rc = pDb->updateIndexKeys( 
						uiCollection, pCurNode, IX_UNLINK_NODE,
						bStartOfUpdate)))
					{
						goto Exit;
					}
				}
			}
			
			bStartOfUpdate = FALSE;
			flmAssert( pCurNode->getNodeType() != ATTRIBUTE_NODE);

			if (RC_BAD( rc = pCurNode->unlinkNode( pDb, uiFlags)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDb->purgeNode( uiCollection, ui64CurNode)))
			{
				goto Exit;
			}

			pCurNode->Release();
			pCurNode = NULL;

			if( bWasDataNode)
			{
				flmAssert( pParentNode);
				if( RC_BAD( rc = pDb->updateIndexKeys( 
					uiCollection, pParentNode, IX_ADD_NODE_VALUE, 
					bStartOfUpdate)))
				{
					goto Exit;
				}
				
				bStartOfUpdate = FALSE;
			}

			// Did we just delete the primary target or root node?
			// Do NOT access m_pCachedNode after this point, because it
			// may have been set to NULL by the call to purgeNode.

			if (ui64CurNode == ui64MyNodeId || !ui64ParentId)
			{
				break;
			}
			
			// Go back to the parent node.
			
			ui64CurNode = ui64ParentId;
		}
	}

	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = pRfl->logNodeDelete( pDb, uiCollection, ui64MyNodeId)))
	{
		goto Exit;
	}

Exit:

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( pCurNode)
	{
		pCurNode->Release();
	}

	if( pParentNode)
	{
		pParentNode->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::deleteChildren(
	IF_Db *		ifpDb,
	FLMUINT		uiNameId)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT64			ui64NextNode;
	F_DOMNode *			pCurNode = NULL;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMUINT				uiCollection;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiRflToken = 0;
	eDomNodeType		eNodeType;

	// Start a transaction if necessary

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	uiCollection = getCollection();
	eNodeType = getNodeType();
	
	// Not supported on attribute nodes
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// See if the node can be deleted

	if( getModeFlags() & (FDOM_READ_ONLY | FDOM_CANNOT_DELETE))
	{
		rc = RC_SET( NE_XFLM_DELETE_NOT_ALLOWED);
		goto Exit;
	}
	
	// Turn of RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Iterate over the children

	bMustAbortOnError = TRUE;
	ui64NextNode = getFirstChildId();

	while( ui64NextNode)
	{
		if( RC_BAD( rc = pDb->getNode( uiCollection, 
			ui64NextNode, XFLM_EXACT, &pCurNode)))
		{
			goto Exit;
		}

		ui64NextNode = pCurNode->getNextSibId();

		if( !uiNameId || uiNameId == pCurNode->getNameId())
		{
			if( RC_BAD( rc = pCurNode->deleteNode( pDb)))
			{
				goto Exit;
			}
		}
	}

	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = pRfl->logNodeChildrenDelete( 
		pDb, uiCollection, getNodeId(), uiNameId)))
	{
		goto Exit;
	}
	
Exit:

	if( pCurNode)
	{
		pCurNode->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}
	
	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::_syncFromDb(
	F_Db *			pDb)
{
	RCODE			rc = NE_XFLM_OK;
	F_DOMNode *	pDOMNode = this;	

	// If we get to this point, we are going to read the node
	// from the database.  This instance of the node should 
	// not be dirty.

	flmAssert( !nodeIsDirty());
	
	// Should not have any input streams open on the cached node
	
	if( getStreamUseCount())
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->retrieveNode( pDb,
		getCollection(), m_pCachedNode->getNodeId(), &pDOMNode)))
	{
		if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_DELETED);
		}
		goto Exit;
	}
	
	if( m_uiAttrNameId)
	{
		if( !m_pCachedNode->m_uiAttrCount || 
			 !m_pCachedNode->getAttribute( m_uiAttrNameId, NULL))
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_DELETED);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getFirstAttribute(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppAttr)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;
	F_DOMNode *			pAttrNode = NULL;
	F_AttrItem *		pAttrItem;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( (pAttrItem = m_pCachedNode->getFirstAttribute()) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttrNode)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttrNode->m_pCachedNode = m_pCachedNode;
	m_pCachedNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	pAttrNode->m_uiAttrNameId = pAttrItem->m_uiNameId;

	if( ifppAttr)
	{
		if( *ifppAttr)
		{
			(*ifppAttr)->Release();
		}
	
		*ifppAttr = (IF_DOMNode *)pAttrNode;
		pAttrNode = NULL;
	}
	
Exit:

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getLastAttribute(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppAttr)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;
	F_DOMNode *			pAttrNode = NULL;
	F_AttrItem *		pAttrItem;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( (pAttrItem = m_pCachedNode->getLastAttribute()) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttrNode)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttrNode->m_pCachedNode = m_pCachedNode;
	m_pCachedNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	pAttrNode->m_uiAttrNameId = pAttrItem->m_uiNameId;

	if( ifppAttr)
	{
		if( *ifppAttr)
		{
			(*ifppAttr)->Release();
		}
	
		*ifppAttr = (IF_DOMNode *)pAttrNode;
		pAttrNode = NULL;
	}
	
Exit:

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::deleteAttribute(
	IF_Db *			ifpDb,
	FLMUINT			uiAttrName)
{
	RCODE					rc = NE_XFLM_OK;

	if( !uiAttrName)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = deleteAttributes( (F_Db *)ifpDb, 
		uiAttrName, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::deleteAttributes(
	F_Db *			pDb,
	FLMUINT			uiAttrToDelete,
	FLMUINT			uiFlags)
{
	RCODE					rc = NE_XFLM_OK;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiCollection;
	FLMUINT				uiAttrName;
	F_DOMNode *			pAttrNode = NULL;
	F_AttrItem *		pAttrItem;
	FLMUINT				uiPos;
	FLMBOOL				bIsIndexed;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Disable logging
	
	pRfl->disableLogging( &uiRflToken);
	
	// Make sure the node is current
	
	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	if( !m_pCachedNode->m_uiAttrCount)
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;
	uiCollection = getCollection();

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttrNode)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttrNode->m_pCachedNode = m_pCachedNode;
	m_pCachedNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);

	for( ;;)
	{
		if( !uiAttrToDelete)
		{
			pAttrItem = m_pCachedNode->getFirstAttribute();
			uiPos = 0;
		}
		else
		{
			if( (pAttrItem = m_pCachedNode->getAttribute( uiAttrToDelete,
											&uiPos)) == NULL)
			{
				break;
			}
		}

		if( uiAttrToDelete && 
			(pAttrItem->m_uiFlags & (FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
		{
			rc = RC_SET( NE_XFLM_DELETE_NOT_ALLOWED);
			goto Exit;
		}
		
		uiAttrName = pAttrItem->m_uiNameId;
		pAttrNode->m_uiAttrNameId = uiAttrName;
		
		if( RC_BAD( rc = pDb->updateIndexKeys( 
			uiCollection, pAttrNode, IX_DEL_NODE_VALUE,
			TRUE, &bIsIndexed)))
		{
			goto Exit;
		}
			
		if( bIsIndexed)
		{
			if( RC_BAD( rc = pDb->updateIndexKeys( 
				uiCollection, pAttrNode, IX_UNLINK_NODE, FALSE)))
			{
				goto Exit;
			}
		}
		
		// Free the attribute

		if (RC_BAD( rc = m_pCachedNode->freeAttribute( pAttrItem, uiPos)))
		{
			goto Exit;
		}
			
		pRfl->enableLogging( &uiRflToken);
		
		if( RC_BAD( rc = pRfl->logAttributeDelete( pDb, uiCollection, 
			getNodeId(), uiAttrName)))
		{
			goto Exit;
		}

		pRfl->disableLogging( &uiRflToken);

		if( !m_pCachedNode->m_uiAttrCount)
		{
			break;
		}
	}
	
	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, uiFlags)))
	{
		goto Exit;
	}
	
	if( bStartedTrans)
	{
		bStartedTrans = FALSE;
		if( RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
	}
	
Exit:

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::hasAttribute(
	IF_Db *				ifpDb,
	FLMUINT				uiNameId,
	IF_DOMNode **		ifppAttr)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_DOMNode *			pAttrNode = NULL;
	F_AttrItem *		pAttrItem;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = checkAttrList()))
	{
		goto Exit;
	}
	
	if( (pAttrItem = m_pCachedNode->getAttribute( uiNameId, NULL)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	if( ifppAttr)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttrNode)))
		{
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			goto Exit;
		}
		
		pAttrNode->m_pCachedNode = m_pCachedNode;
		m_pCachedNode->incrNodeUseCount();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		pAttrNode->m_uiAttrNameId = pAttrItem->m_uiNameId;

		if( *ifppAttr)
		{
			(*ifppAttr)->Release();
		}
	
		*ifppAttr = (IF_DOMNode *)pAttrNode;
		pAttrNode = NULL;
	}
	
Exit:

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::insertChildElm(
	FLMUINT		uiChildElmOffset,
	FLMUINT		uiChildElmNameId,
	FLMUINT64	ui64ChildElmNodeId)
{
	RCODE			rc = NE_XFLM_OK;
	NODE_ITEM *	pChildElmNode;
	
	if( RC_BAD( rc = resizeChildElmList( m_nodeInfo.uiChildElmCount + 1, FALSE)))
	{
		goto Exit;
	}

	// Remember, m_nodeInfo.uiChildElmCount has been incremented by
	// resizeChildElmList, so there really isn't anything in the
	// m_nodeInfo.uiChildElmCount - 1 slot.
	
	pChildElmNode = &m_pNodeList [ uiChildElmOffset];
	if( m_nodeInfo.uiChildElmCount > 1 && 
		 uiChildElmOffset < m_nodeInfo.uiChildElmCount - 1)
	{
		f_memmove( &m_pNodeList [ uiChildElmOffset + 1], pChildElmNode,
			sizeof( NODE_ITEM) * (m_nodeInfo.uiChildElmCount - 
				uiChildElmOffset - 1));
						
	}
	
	pChildElmNode->uiNameId = uiChildElmNameId;
	pChildElmNode->ui64NodeId = ui64ChildElmNodeId;
	
Exit:

	return( rc);
}
	
/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::createAttribute(
	IF_Db *				ifpDb,
	FLMUINT				uiNameId,
	IF_DOMNode **		ifppAttr)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	F_AttrElmInfo		attrInfo;
	F_DOMNode *			pAttr = NULL;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	FLMUINT				uiRflToken = 0;
	F_AttrItem *		pAttrItem = NULL;
	FLMBOOL				bCreatedNewAttr = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Disable logging
	
	pRfl->disableLogging( &uiRflToken);

	// Make sure our copy of the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	// If this isn't an element node, return an error

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// Force the transaction to abort on error beyond this point
	
	bMustAbortOnError = TRUE;
	
	// Check the attribute state
	
	if( RC_BAD( rc = pDb->checkAndUpdateState( 
		ATTRIBUTE_NODE, uiNameId)))
	{
		goto Exit;
	}

	// Retrieve or create the attribute list node

	if( !m_pCachedNode->m_uiAttrCount ||
		 (pAttrItem = m_pCachedNode->getAttribute( uiNameId, NULL)) == NULL)
	{
		if( RC_BAD( rc = makeWriteCopy( pDb)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pCachedNode->createAttribute( 
			pDb, uiNameId, &pAttrItem)))
		{
			goto Exit;
		}

		bCreatedNewAttr = TRUE;
	}
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttr)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttr->m_pCachedNode = m_pCachedNode;
	m_pCachedNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	pAttr->m_uiAttrNameId = uiNameId;

	if( bCreatedNewAttr)
	{
		// Update the element.

		if( RC_BAD( rc = pDb->updateNode( m_pCachedNode,
			FLM_UPD_INTERNAL_CHANGE)))
		{
			goto Exit;
		}

		// Update the indexes

		if( RC_BAD( rc = pDb->updateIndexKeys( getCollection(),
			pAttr, IX_LINK_AND_ADD_NODE, TRUE)))
		{
			goto Exit;
		}

		// Log the attribute create
		
		pRfl->enableLogging( &uiRflToken);
		
		if( RC_BAD( rc = pRfl->logAttributeCreate( 
			pDb, getCollection(), getNodeId(), uiNameId, 0)))
		{
			goto Exit;
		}
	}
	
	if( ifppAttr)
	{
		if( *ifppAttr)
		{
			(*ifppAttr)->Release();
		}
	
		*ifppAttr = (IF_DOMNode *)pAttr;
		pAttr = NULL;
	}
	
Exit:

	if( pAttr)
	{
		pAttr->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}
	
	if( RC_BAD( rc) && bMustAbortOnError)
	{
		pDb->setMustAbortTrans( rc);
	}
	
	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::createNode(
	IF_Db *				ifpDb,
	eDomNodeType		eNodeType,
	FLMUINT				uiNameId,
	eNodeInsertLoc		eLocation,
	IF_DOMNode **		ifppNewNode,
	FLMUINT64 *			pui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	F_DOMNode *			pNewNode = NULL;
	F_CachedNode *		pNewCachedNode;
	F_DOMNode *			pRefNode = NULL;
	F_DOMNode *			pNewParent = NULL;
	FLMUINT				uiDataType = XFLM_NODATA_TYPE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiRflToken = 0;

	// Not supported for attributes
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	// Make sure an update transaction is active
	
	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of this node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Make sure the node type is valid

	if( eLocation == XFLM_FIRST_CHILD || eLocation == XFLM_LAST_CHILD)
	{
		if( RC_BAD( rc = isChildTypeValid( eNodeType)))
		{
			goto Exit;
		}
	}
	else if( eLocation == XFLM_PREV_SIB || eLocation == XFLM_NEXT_SIB)
	{
		if( eNodeType != ELEMENT_NODE && eNodeType != DATA_NODE &&
			eNodeType != COMMENT_NODE && eNodeType != CDATA_SECTION_NODE &&
			eNodeType != PROCESSING_INSTRUCTION_NODE)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If the user is requesting a specific nodeId, then make sure
	// the node is not already in use.

	if( pui64NodeId)
	{
		if( *pui64NodeId && (pDb->m_uiFlags & FDB_REBUILDING_DATABASE))
		{
			if( RC_BAD( rc = pDb->getNode( getCollection(), *pui64NodeId, &pNewNode)))
			{
				if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				rc = NE_XFLM_OK;
			}
			else
			{
				// Already in use

				rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
				goto Exit;
			}
		}
		else if( *pui64NodeId)
		{
			// Set to zero so we don't use it.  We will return the
			// new nodeId.

			*pui64NodeId = 0;
		}
	}

	// Look at the node's state (checking, etc.) and verify that
	// the node's name ID is valid
	//
	// IMPORTANT NOTE: checkAndUpdateState may change m_pDict if it ends
	// up calling changeItemState

	if( RC_BAD( rc = pDb->checkAndUpdateState( eNodeType, uiNameId)))
	{
		goto Exit;
	}

	// Create the new node.

	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->createNode( pDb,
									getCollection(),
									(FLMUINT64)(pui64NodeId
													? *pui64NodeId
													: (FLMUINT64)0),
									&pNewNode)))
	{
		goto Exit;
	}

	pNewCachedNode = pNewNode->m_pCachedNode;
	if( eNodeType == DATA_NODE)
	{
		flmAssert( getNodeType() == ELEMENT_NODE);
		uiNameId = getNameId();
	}

	if( eNodeType == ELEMENT_NODE || eNodeType == DATA_NODE)
	{
		F_AttrElmInfo		elmInfo;

		if( RC_BAD( rc = pDb->m_pDict->getElement( pDb, uiNameId, &elmInfo)))
		{
			goto Exit;
		}
		uiDataType = elmInfo.m_uiDataType;

		// Is this a node whose child elements must all be unique?
	
		if( eNodeType == ELEMENT_NODE &&
			elmInfo.m_uiFlags & ATTR_ELM_UNIQUE_SUBELMS)
		{
			flmAssert( uiDataType == XFLM_NODATA_TYPE);
			pNewCachedNode->setFlags( FDOM_HAVE_CELM_LIST);
		}
	}
	else
	{
		uiDataType = XFLM_NODATA_TYPE;
		uiNameId = 0;
	}
	
	pNewCachedNode->setNodeType( eNodeType);
	pNewCachedNode->setDocumentId( getDocumentId());
	pNewCachedNode->setDataType( uiDataType);
	
	if( uiNameId)
	{
		pNewCachedNode->setNameId( uiNameId);
	}

	if( RC_BAD( rc = pDb->updateNode( pNewCachedNode, FLM_UPD_ADD)))
	{
		goto Exit;
	}

	if( eNodeType == ELEMENT_NODE)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( 
			pNewCachedNode->getCollection(),
			pNewNode, IX_ADD_NODE_VALUE, TRUE)))
		{
			goto Exit;
		}
	}

	switch( eLocation)
	{
		case XFLM_FIRST_CHILD:
		{
			pNewParent = this;
			pNewParent->AddRef();

			if( getFirstChildId())
			{
				if( RC_BAD( rc = pDb->getNode(
					getCollection(),
					getFirstChildId(), &pRefNode)))
				{
					goto Exit;
				}
			}

			break;
		}

		case XFLM_LAST_CHILD:
		{
			pNewParent = this;
			pNewParent->AddRef();
			break;
		}

		case XFLM_PREV_SIB:
		{
			if( !getParentId())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
				goto Exit;
			}

			if( RC_BAD( rc = pDb->getNode( getCollection(), getParentId(),
				&pNewParent)))
			{
				goto Exit;
			}

			pRefNode = this;
			pRefNode->AddRef();
			break;
		}

		case XFLM_NEXT_SIB:
		{
			if( !getParentId())
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
				goto Exit;
			}

			if( RC_BAD( rc = pDb->getNode( getCollection(), getParentId(),
				&pNewParent)))
			{
				goto Exit;
			}

			if( getNextSibId())
			{
				if( RC_BAD( rc = pDb->getNode( getCollection(), 
					getNextSibId(), &pRefNode)))
				{
					goto Exit;
				}
			}
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}			
	}

	if( RC_BAD( rc = pNewParent->insertBefore( pDb, pNewNode, pRefNode)))
	{
		goto Exit;
	}

	if( pui64NodeId)
	{
		*pui64NodeId = pNewCachedNode->getNodeId();
	}

	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logNodeCreate( 
		pDb, pNewNode->getCollection(), getNodeId(), 
		eNodeType, uiNameId, eLocation, pNewNode->getNodeId())))
	{
		goto Exit;
	}
	
	if( ifppNewNode)
	{
		if( *ifppNewNode)
		{
			(*ifppNewNode)->Release();
		}
	
		*ifppNewNode = (IF_DOMNode *)pNewNode;
		pNewNode = NULL;
	}
	
Exit:

	if( pNewNode)
	{
		pNewNode->Release();
	}

	if( pRefNode)
	{
		pRefNode->Release();
	}

	if( pNewParent)
	{
		pNewParent->Release();
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::createChildElement(
	IF_Db *				ifpDb,
	FLMUINT				uiNameId,
	eNodeInsertLoc		eLocation,
	IF_DOMNode **		ifppNewNode,
	FLMUINT64 *			pui64NodeId)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	F_Rfl *				pRfl = pDb->m_pDatabase->m_pRfl;
	F_DOMNode *			pTmpNode = NULL;
	F_DOMNode *			pNewNode = NULL;
	F_CachedNode *		pNewCachedNode;
	F_AttrElmInfo		elmInfo;
	eDomNodeType		eThisNodeType;
	FLMUINT				uiCollection;
	FLMUINT				uiDataType = XFLM_NODATA_TYPE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiRflToken = 0;
	FLMBOOL				bIsIndexed;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of this node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	// Make sure the insert location is supported
	
	if( eLocation != XFLM_FIRST_CHILD && eLocation != XFLM_LAST_CHILD)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Make sure the node type is valid

	eThisNodeType = getNodeType();

	if( eThisNodeType != ELEMENT_NODE && eThisNodeType != DOCUMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// A document node can only have one element node

	if( eThisNodeType == DOCUMENT_NODE && getFirstChildId())
	{
		if( RC_BAD( rc = getChild( ifpDb, ELEMENT_NODE, 
			(IF_DOMNode **)&pTmpNode)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR);
			goto Exit;
		}
	}

	// Setup misc. variables

	uiCollection = getCollection();

	// If the user is requesting a specific nodeId, then make sure
	// the node is not already in use.

	if( pui64NodeId)
	{
		if( *pui64NodeId && (pDb->m_uiFlags & FDB_REBUILDING_DATABASE))
		{
			if( RC_OK( rc = pDb->getNode( uiCollection, *pui64NodeId, &pNewNode)))
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
				goto Exit;
			}
			else if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else if( *pui64NodeId)
		{
			*pui64NodeId = 0;
		}
	}

	// Check the element's state
	
	if( RC_BAD( rc = pDb->checkAndUpdateState( ELEMENT_NODE, uiNameId)))
	{
		goto Exit;
	}
		
	// Create the new node.

	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->createNode( pDb,
									uiCollection,
									(FLMUINT64)(pui64NodeId
													? *pui64NodeId
													: (FLMUINT64)0),
									&pNewNode)))
	{
		goto Exit;
	}

	pNewCachedNode = pNewNode->m_pCachedNode;
	
	// Make sure the parent node (this) can be updated

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}
		
	// Does the parent expect all children to be unique?

	if( getModeFlags() & FDOM_HAVE_CELM_LIST)
	{
		FLMUINT	uiInsertPos;
		
		if( m_pCachedNode->findChildElm( uiNameId, &uiInsertPos))
		{
			rc = RC_SET( NE_XFLM_DOM_DUPLICATE_ELEMENT);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pCachedNode->insertChildElm( uiInsertPos,
			uiNameId, pNewCachedNode->getNodeId())))
		{
			goto Exit;
		}
	}

	// Update the element's state

	if( RC_BAD( rc = pDb->m_pDict->getElement( pDb, uiNameId, &elmInfo)))
	{
		goto Exit;
	}
	
	uiDataType = elmInfo.m_uiDataType;

	// Is this a node whose children must all be unique?

	if( elmInfo.m_uiFlags & ATTR_ELM_UNIQUE_SUBELMS)
	{
		flmAssert( uiDataType == XFLM_NODATA_TYPE);
		pNewCachedNode->setFlags( FDOM_HAVE_CELM_LIST);
	}
	
	pNewCachedNode->setNodeType( ELEMENT_NODE);
	pNewCachedNode->setParentId( getNodeId());
	pNewCachedNode->setDocumentId( getDocumentId());
	pNewCachedNode->setDataType( uiDataType);
	pNewCachedNode->setNameId( uiNameId);

	// Set the sibling pointers

	if( eLocation == XFLM_LAST_CHILD)
	{
		if( getLastChildId())
		{
			if( RC_BAD( rc = pDb->getNode( uiCollection, 
				getLastChildId(), &pTmpNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
			
			flmAssert( pTmpNode->getNextSibId() == 0);
			
			if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
			{
				goto Exit;
			}
			
			pTmpNode->setNextSibId( pNewCachedNode->getNodeId());
			pNewCachedNode->setPrevSibId( getLastChildId());
			
			if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			setFirstChildId( pNewCachedNode->getNodeId());
		}
		
		setLastChildId( pNewCachedNode->getNodeId());
	}
	else
	{
		flmAssert( eLocation == XFLM_FIRST_CHILD);
		
		if( getFirstChildId())
		{
			if( RC_BAD( rc = pDb->getNode( 
				uiCollection, getFirstChildId(), &pTmpNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
			
			flmAssert( pTmpNode->getPrevSibId() == 0);
			
			if( RC_BAD( rc = pTmpNode->makeWriteCopy( pDb)))
			{
				goto Exit;
			}
			
			pTmpNode->setPrevSibId( pNewCachedNode->getNodeId());
			pNewCachedNode->setNextSibId( getFirstChildId());
			
			if( RC_BAD( rc = pDb->updateNode( pTmpNode->m_pCachedNode, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			setLastChildId( pNewCachedNode->getNodeId());
		}
		
		setFirstChildId( pNewCachedNode->getNodeId());
	}
		

	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->updateNode( pNewCachedNode, FLM_UPD_ADD)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
		pNewNode, IX_ADD_NODE_VALUE, TRUE, &bIsIndexed)))
	{
		goto Exit;
	}
	
	if( bIsIndexed)
	{
		if( RC_BAD( rc = pDb->updateIndexKeys( uiCollection,
			pNewNode, IX_LINK_NODE, FALSE, &bIsIndexed)))
		{
			goto Exit;
		}
	}
	
	if( pui64NodeId)
	{
		*pui64NodeId = pNewCachedNode->getNodeId();
	}

	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logNodeCreate( pDb, uiCollection, 
		pNewCachedNode->getParentId(), 
		ELEMENT_NODE, uiNameId, XFLM_LAST_CHILD, pNewCachedNode->getNodeId())))
	{
		goto Exit;
	}
	
	if( ifppNewNode)
	{
		if( *ifppNewNode)
		{
			(*ifppNewNode)->Release();
		}
	
		*ifppNewNode = (IF_DOMNode *)pNewNode;
		pNewNode = NULL;
	}
	
Exit:

	if( pNewNode)
	{
		pNewNode->Release();
	}

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::createAnnotation(
	IF_Db *					ifpDb,
	IF_DOMNode **			ifppAnnotation,
	FLMUINT64 *				pui64NodeId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	F_CachedNode *	pCachedNode;
	FLMBOOL			bMustAbortOnError = FALSE;
	F_Db *			pDb = (F_Db *)ifpDb;
	F_DOMNode **	ppAnnotation = (F_DOMNode **)ifppAnnotation;
	FLMBOOL			bStartedTrans = FALSE;
	eDomNodeType	eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure our copy of this node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();

	// Not supported on attribute nodes
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// If the node already has an annotation, return an error

	if( getAnnotationId())
	{
		rc = RC_SET( NE_XFLM_EXISTS);
		goto Exit;
	}

	// If the user is requesting a specific nodeId, then make sure
	// the node is not already in use.

	if( pui64NodeId)
	{
		if( *pui64NodeId && (pDb->m_uiFlags & FDB_REBUILDING_DATABASE))
		{
			if( RC_BAD( rc = pDb->getNode( getCollection(), *pui64NodeId, &pNode)))
			{
				if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				rc = NE_XFLM_OK;
			}
			else
			{
				// Already in use

				rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
				goto Exit;
			}
		}
		else if( *pui64NodeId)
		{
			// Set to zero so we don't use it.  We will return the
			// new nodeId.
			*pui64NodeId = 0;
		}
	}

	// Create the new node.

	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->createNode( pDb,
									getCollection(),
									(FLMUINT64)(pui64NodeId
													? *pui64NodeId
													: (FLMUINT64)0),
									&pNode)))
	{
		goto Exit;
	}

	pCachedNode = pNode->m_pCachedNode;
	pCachedNode->setNodeType( ANNOTATION_NODE);
	pCachedNode->setDocumentId( getDocumentId());
	pCachedNode->setParentId( getNodeId());
	pCachedNode->setDataType( XFLM_NODATA_TYPE);

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = pDb->updateNode( pCachedNode, FLM_UPD_ADD)))
	{
		goto Exit;
	}

	// Link the annotation to this node

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}
	
	setAnnotationId( pCachedNode->getNodeId());
	
	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		goto Exit;
	}

	if( bStartedTrans)
	{
		if(  RC_BAD( rc = pDb->transCommit()))
		{
			goto Exit;
		}
		bStartedTrans = FALSE;
	}

	if( pui64NodeId)
	{
		*pui64NodeId = pCachedNode->getNodeId();
	}

	// Release any node that the passed-in parameter may be
	// pointing at

	if( *ppAnnotation)
	{
		(*ppAnnotation)->Release();
	}

	*ppAnnotation = pNode;
	pNode = NULL;

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( RC_BAD( rc))
	{
		if(  bMustAbortOnError)
		{
			pDb->setMustAbortTrans( rc);		
		}
	
		if(  bStartedTrans)
		{
			pDb->transAbort();
		}
	
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::hasAnnotation(
	IF_Db *				ifpDb,
	FLMBOOL *			pbHasAnnotation)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	*pbHasAnnotation = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( (F_Db *)ifpDb)))
	{
		goto Exit;
	}

	if( getAnnotationId())
	{
		*pbHasAnnotation = TRUE;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getAnnotation(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppAnnotation)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( (F_Db *)ifpDb)))
	{
		goto Exit;
	}

	if( !getAnnotationId())
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = ifpDb->getNode( getCollection(), getAnnotationId(),
		ifppAnnotation)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getDocumentId(
	IF_Db *				ifpDb,
	FLMUINT64 *			pui64DocId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	*pui64DocId = m_pCachedNode->getDocumentId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getNodeId(
	IF_Db *			ifpDb,
	FLMUINT64 *		pui64NodeId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if( getNodeType() == ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pui64NodeId = m_pCachedNode->getNodeId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:	puiAttrNameId returns 0 if the node isn't an attribute.
******************************************************************************/
RCODE F_DOMNode::getNodeId(
	F_Db *			pDb,
	FLMUINT64 *		pui64NodeId,
	FLMUINT *		puiAttrNameId)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	*pui64NodeId = m_pCachedNode->getNodeId();
	
	if( getNodeType() == ATTRIBUTE_NODE)
	{
		*puiAttrNameId = m_uiAttrNameId;
	}
	else
	{
		*puiAttrNameId = 0;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getParentId(
	IF_Db *			ifpDb,
	FLMUINT64 *		pui64ParentId)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	*pui64ParentId = getParentId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:	COM version of the getPrevSibId method.  This method ensures that
		the DOM node is up-to-date.
******************************************************************************/
RCODE XFLAPI F_DOMNode::getPrevSibId(
	IF_Db *				ifpDb,
	FLMUINT64 *			pui64PrevSibId)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pui64PrevSibId = getPrevSibId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return rc;
}


/*****************************************************************************
Desc:	COM version of the getNextSibId method.  This method ensures that
		the DOM node is up-to-date.
******************************************************************************/
RCODE XFLAPI F_DOMNode::getNextSibId(
	IF_Db *				ifpDb,
	FLMUINT64 *			pui64NextSibId)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pui64NextSibId = getNextSibId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return rc;
}

/*****************************************************************************
Desc:	COM version of the getFirstChildId method.  This method ensures that
		the DOM node is up-to-date.
******************************************************************************/
RCODE XFLAPI F_DOMNode::getFirstChildId(
	IF_Db *				ifpDb,
	FLMUINT64 *			pui64FirstChildId)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pui64FirstChildId = getFirstChildId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return rc;
}


/*****************************************************************************
Desc:	COM version of the getLastChildId method.  This method ensures that
		the DOM node is up-to-date.
******************************************************************************/
RCODE XFLAPI F_DOMNode::getLastChildId(
	IF_Db *				ifpDb,
	FLMUINT64 *			pui64LastChildId)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pui64LastChildId = getLastChildId();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return rc;
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::isNamespaceDecl(
	IF_Db *				ifpDb,
	FLMBOOL *			pbIsNamespaceDecl)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	*pbIsNamespaceDecl = isNamespaceDecl();

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::hasChildren(
	IF_Db *			ifpDb,
	FLMBOOL *		pbHasChildren)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( (F_Db *)ifpDb)))
	{
		goto Exit;
	}

	if (getNodeType() == ATTRIBUTE_NODE)
	{
		*pbHasChildren = FALSE;
	}
	else
	{
		*pbHasChildren = getFirstChildId() ? TRUE : FALSE;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getNameId(
	IF_Db *			ifpDb,
	FLMUINT *		puiNameId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	eDomNodeType	eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		*puiNameId = m_uiAttrNameId;
	}
	else if( m_pCachedNode)
	{
		*puiNameId = getNameId();
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getEncDefId(
	IF_Db *			ifpDb,
	FLMUINT *		puiEncDefNumber)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if( getNodeType() == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getEncDefId( 
			m_uiAttrNameId, puiEncDefNumber)))
		{
			goto Exit;
		}
	}
	else if( m_pCachedNode)
	{
		*puiEncDefNumber = getEncDefId();
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getAnnotationId(
	IF_Db *			ifpDb,
	FLMUINT64 *		pui64AnnotationId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	if( getNodeType() == ATTRIBUTE_NODE)
	{
		*pui64AnnotationId = 0;
	}
	else if( m_pCachedNode)
	{
		*pui64AnnotationId = getAnnotationId();
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::getDataType(
	IF_Db *			ifpDb,
	FLMUINT *		puiDataType)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getDataType( 
			m_uiAttrNameId, puiDataType)))
		{
			goto Exit;
		}
	}
	else
	{
		*puiDataType = m_pCachedNode->getDataType();
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getPrefixId(
	IF_Db *			ifpDb,
	FLMUINT *		puiPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMUINT			uiPrefix = 0;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getPrefixId( 
			m_uiAttrNameId, &uiPrefix)))
		{
			goto Exit;
		}
	}
	else
	{
		if( (uiPrefix = m_pCachedNode->getPrefixId()) != 0)
		{
			if( RC_BAD( rc = pDb->m_pDict->getPrefix( uiPrefix, NULL)))
			{
				if( rc != NE_XFLM_BAD_PREFIX)
				{
					goto Exit;
				}
	
				rc = NE_XFLM_OK;
				uiPrefix = 0;
			}
		}
	}

	*puiPrefixId = uiPrefix;

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::hasAttributes(
	IF_Db *			ifpDb,
	FLMBOOL *		pbHasAttrs)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		*pbHasAttrs = FALSE;
		goto Exit;
	}

	*pbHasAttrs = m_pCachedNode->m_uiAttrCount ? TRUE : FALSE;
	
Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::hasNextSibling(
	IF_Db *			ifpDb,
	FLMBOOL *		pbHasNextSibling)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pbHasNextSibling = (m_pCachedNode->getNextSibId() &&
								getParentId())
								? TRUE 
								: FALSE;

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_DOMNode::hasPreviousSibling(
	IF_Db *			ifpDb,
	FLMBOOL *		pbHasPreviousSibling)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	*pbHasPreviousSibling = (m_pCachedNode->getPrevSibId() &&
									 getParentId())
									 ? TRUE 
									 : FALSE;

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getAncestorElement(
	IF_Db *					ifpDb,
	FLMUINT					uiNameId,
	IF_DOMNode **			ifppAncestor)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	F_DOMNode *		pTmpNode = NULL;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiCollection;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	pTmpNode = this;
	pTmpNode->AddRef();
	uiCollection = getCollection();

	while( pTmpNode)
	{
		if( !pTmpNode->getParentId())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->getNode(
			uiCollection, pTmpNode->getParentId(), &pTmpNode)))
		{
			goto Exit;
		}
		
		if( pTmpNode->getNameId() == uiNameId)
		{
			break;
		}
	}
	
	if( !pTmpNode)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	if( *ifppAncestor)
	{
		(*ifppAncestor)->Release();
	}
	
	*ifppAncestor = pTmpNode;
	pTmpNode = NULL;

Exit:

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getDescendantElement(
	IF_Db *			ifpDb,
	FLMUINT			uiNameId,
	IF_DOMNode **	ifppDescendant)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	F_DOMNode *		pContextNode = NULL;
	F_DOMNode *		pFoundNode = NULL;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiCollection;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	pContextNode = this;
	pContextNode->AddRef();

	uiCollection = getCollection();
	while( pContextNode)
	{
		if( pContextNode->getFirstChildId())
		{
			if( RC_BAD( rc = pDb->getNode( uiCollection,
				pContextNode->getFirstChildId(), &pContextNode)))
			{
				goto Exit;
			}
		}
		else if( pContextNode->getNextSibId())
		{
Get_Next_Sib:

			if( pContextNode->getNodeId() == getNodeId())
			{
				break;
			}
			
			if( RC_BAD( rc = pDb->getNode( uiCollection,
				pContextNode->getNextSibId(), &pContextNode)))
			{
				goto Exit;
			}
		}
		else
		{
			if( pContextNode->getNodeId() == getNodeId())
			{
				break;
			}
			
			if( RC_BAD( rc = pDb->getNode( uiCollection,
				pContextNode->getParentId(), &pContextNode)))
			{
				goto Exit;
			}

			goto Get_Next_Sib;
		}
		
		if( pContextNode->getNodeType() != ELEMENT_NODE)
		{
			continue;
		}
		
		if( pContextNode->getNameId() == uiNameId)
		{
			pFoundNode = pContextNode;
			pFoundNode->AddRef();
			break;
		}
	}
	
	if( *ifppDescendant)
	{
		(*ifppDescendant)->Release();
	}
	
	*ifppDescendant = pFoundNode;
	pFoundNode = NULL;

Exit:

	if( pContextNode)
	{
		pContextNode->Release();
	}
	
	if( pFoundNode)
	{
		pFoundNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getDocumentNode(
	IF_Db *					ifpDb,
	IF_DOMNode **			ifppDoc)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( isRootNode())
	{
		IF_DOMNode *	pTmpNode = *ifppDoc;

		*ifppDoc = this;
		(*ifppDoc)->AddRef();
		
		if( pTmpNode)
		{
			pTmpNode->Release();
		}

		goto Exit;
	}

	if( RC_BAD( rc = ifpDb->getNode( getCollection(),
		getDocumentId(), ifppDoc)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getParentNode(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppParent)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( !getParentId())
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = ifpDb->getNode(
		getCollection(), getParentId(), ifppParent)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getFirstChild(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppChild)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( !getFirstChildId())
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = ifpDb->getNode( getCollection(),
		getFirstChildId(), ifppChild)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getLastChild(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppChild)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( !getLastChildId())
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = ifpDb->getNode( getCollection(),
		getLastChildId(), ifppChild)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getChild(
	IF_Db *			ifpDb,
	eDomNodeType	eNodeType,
	IF_DOMNode **	ppChild)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;
	F_DOMNode *	pCurNode = NULL;
	FLMUINT64	ui64NodeId;
	F_Db *		pDb = (F_Db *)ifpDb;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// We can do a quick lookup if this node is an element where we are
	// maintaining a child element list.

	if( eNodeType == ELEMENT_NODE &&
		 (getModeFlags() & FDOM_HAVE_CELM_LIST))
	{
		if( getChildElmCount())
		{
			if( RC_BAD( rc = ((IF_Db *)pDb)->getNode(
				getCollection(), getChildElmNodeId( 0), ppChild)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
			}
		}
		else
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		}

		goto Exit;
	}

	ui64NodeId = getFirstChildId();
	
	for( ;;)
	{
		if( !ui64NodeId)
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		if( RC_BAD( rc = ((IF_Db *)pDb)->getNode(
			getCollection(), ui64NodeId, (IF_DOMNode **)&pCurNode)))
		{
			goto Exit;
		}

		if( pCurNode->getNodeType() == eNodeType)
		{
			if( *ppChild)
			{
				(*ppChild)->Release();
			}

			*ppChild = pCurNode;
			pCurNode = NULL;
			break;
		}

		ui64NodeId = pCurNode->getNextSibId();
	}

Exit:

	if( pCurNode)
	{
		pCurNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getChildElement(
	IF_Db *				ifpDb,
	FLMUINT				uiNameId,
	IF_DOMNode **		ppChild,
	FLMUINT				uiFlags)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;
	F_DOMNode *	pCurNode = NULL;
	FLMUINT64	ui64NodeId;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMUINT		uiCollection;
	FLMUINT		uiElmPos;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	// We can do a quick lookup if this node is an element where we are
	// maintaining a child element list.

	if( getModeFlags() & FDOM_HAVE_CELM_LIST)
	{
		if( !getChildElmCount())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
		
		if( !findChildElm( uiNameId, &uiElmPos) &&
			 (!uiFlags || (uiFlags & XFLM_EXACT) ||
			 uiElmPos >= getChildElmCount()))
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		// At this point, if we did an exact match, we will be on the node.
		// If we did an inclusive match, we will be either on the node or
		// past it.  If we did an exclusive match, we need to determine if
		// we are on the node or past it.  If we are on the node, we need
		// to try to move past it, unless we are at the end of the list, in
		// which case we cannot go exclusive.

		if( uiFlags & XFLM_EXCL)
		{
			// If we found the node, we need to go one past it, if there
			// is one past it to go to.  If not, we must return
			// not found.

			 if( getChildElmNameId( uiElmPos) == uiNameId)
			 {
				 if( uiElmPos == getChildElmCount() - 1)
				 {
					rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
					goto Exit;
				 }
				 else
				 {
					 uiElmPos++;
				 }
			 }
		}
		if( RC_BAD( rc = ((IF_Db *)pDb)->getNode( getCollection(),
				getChildElmNodeId( uiElmPos), ppChild)))
		{
			if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}
	}
	else
	{
		// Cannot set uiFlags for nodes that are not unique-child nodes.

		if( uiFlags)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_FLAG);
			goto Exit;
		}

		ui64NodeId = getFirstChildId();
		uiCollection = getCollection();
		for( ;;)
		{
			if( !ui64NodeId)
			{
				rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
				goto Exit;
			}
	
			if( RC_BAD( rc = ((IF_Db *)pDb)->getNode(
				uiCollection, ui64NodeId, (IF_DOMNode **)&pCurNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				}
				goto Exit;
			}
	
			if( pCurNode->getNodeType() == ELEMENT_NODE &&
				pCurNode->getNameId() == uiNameId)
			{
				if( *ppChild)
				{
					(*ppChild)->Release();
				}
	
				*ppChild = pCurNode;
				pCurNode = NULL;
				break;
			}
	
			ui64NodeId = pCurNode->getNextSibId();
		}
	}

Exit:

	if( pCurNode)
	{
		pCurNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getSiblingElement(
	IF_Db *				ifpDb,
	FLMUINT				uiNameId,
	FLMBOOL				bNext,
	IF_DOMNode **		ppSibling)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bStartedTrans = FALSE;
	F_DOMNode *	pCurNode = NULL;
	FLMUINT64	ui64NodeId;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMUINT		uiCollection;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( bNext)
	{
		ui64NodeId = getNextSibId();
	}
	else
	{
		ui64NodeId = getPrevSibId();
	}

	uiCollection = getCollection();
	for( ;;)
	{
		if( !ui64NodeId)
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		if( RC_BAD( rc = ((IF_Db *)pDb)->getNode(
			uiCollection, ui64NodeId, (IF_DOMNode **)&pCurNode)))
		{
			goto Exit;
		}

		if( pCurNode->getNodeType() == ELEMENT_NODE &&
			pCurNode->getNameId() == uiNameId)
		{
			if( *ppSibling)
			{
				(*ppSibling)->Release();
			}

			*ppSibling = pCurNode;
			pCurNode = NULL;
			break;
		}

		if( bNext)
		{
			ui64NodeId = pCurNode->getNextSibId();
		}
		else
		{
			ui64NodeId = pCurNode->getPrevSibId();
		}
	}

Exit:

	if( pCurNode)
	{
		pCurNode->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getPreviousSibling(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppSib)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() == ATTRIBUTE_NODE)
	{
		if( !(*ifppSib))
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pCachedNode->getPrevSiblingNode(
			m_uiAttrNameId, ifppSib)))
		{
			goto Exit;
		}
	}
	else
	{
		if( !getPrevSibId() || !getParentId())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		if( RC_BAD( rc = ifpDb->getNode(
			getCollection(), getPrevSibId(), ifppSib)))
		{
			goto Exit;
		}
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getNextSibling(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppSib)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( getNodeType() == ATTRIBUTE_NODE)
	{
		if( !(*ifppSib))
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pCachedNode->getNextSiblingNode(
			m_uiAttrNameId, ifppSib)))
		{
			goto Exit;
		}
	}
	else
	{
		if( !getNextSibId() || !getParentId())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}

		if( RC_BAD( rc = ifpDb->getNode( getCollection(), 
			getNextSibId(), ifppSib)))
		{
			goto Exit;
		}
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getPreviousDocument(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppDoc)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	F_DOMNode *		pNode = NULL;
	FLMBYTE			ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT			uiKeyLen;
	FLMUINT64		ui64StartDocId;
	FLMUINT64		ui64DocumentId;
	FLMBOOL			bNeg;
	FLMUINT			uiBytesProcessed;
	F_Btree *		pBTree = NULL;
	FLMUINT			uiCollection;
	F_COLLECTION *	pCollection;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	uiCollection = getCollection();
	
	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		if( rc != NE_XFLM_DOM_NODE_DELETED)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pBTree)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->m_pDict->getCollection( uiCollection, &pCollection)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pBTree->btOpen( pDb, &pCollection->lfInfo, FALSE, TRUE)))
		{
			goto Exit;
		}

		ui64DocumentId = ui64StartDocId = getDocumentId(); 
		uiKeyLen = sizeof( ucKey);
		if( RC_BAD( rc = flmNumber64ToStorage( ui64StartDocId, &uiKeyLen,
										ucKey, FALSE, TRUE)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pBTree->btLocateEntry(
									ucKey, sizeof( ucKey), &uiKeyLen, XFLM_INCL)))
		{
			if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			}
	
			goto Exit;
		}
	
		for (;;)
		{
			// Need to go to the previous node.
	
			if( RC_BAD( rc = pBTree->btPrevEntry( ucKey, uiKeyLen, &uiKeyLen)))
			{
				if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_BOF_HIT)
				{
					rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
				}
				goto Exit;
			}
			
			if( RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
										&ui64DocumentId, &bNeg, &uiBytesProcessed)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDb->getNode( uiCollection, ui64DocumentId, &pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
	
					// Better be able to find the node at this point!
	
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
	
			// If the node is a root node, we have a document we can
			// process.
	
			if( pNode->isRootNode() && pNode->getNodeId() < ui64StartDocId)
			{
				if( *ifppDoc)
				{
					(*ifppDoc)->Release();
				}
				
				// Just use the reference on pNode for *ifppDoc
				
				*ifppDoc = pNode;
				pNode = NULL;
				goto Exit;
			}
		}
	}
	else
	{
		// If we are not at the root node of the document,
		// jump to the root.
	
		if( !isRootNode())
		{
			if( RC_BAD( rc = pDb->getNode( uiCollection, getDocumentId(), &pNode)))
			{
				goto Exit;
			}
		}
		else
		{
			pNode = this;
			pNode->AddRef();
		}
	
		if( !pNode->getPrevSibId())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
	
		if( RC_BAD( rc = ifpDb->getNode( uiCollection,
			pNode->getPrevSibId(), ifppDoc)))
		{
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pBTree)
	{
		pBTree->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_DOMNode::getNextDocument(
	IF_Db *				ifpDb,
	IF_DOMNode **		ifppDoc)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	F_DOMNode *		pNode = NULL;
	FLMBYTE			ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT			uiKeyLen;
	FLMUINT64		ui64DocumentId;
	FLMBOOL			bNeg;
	FLMUINT			uiBytesProcessed;
	F_Btree *		pBTree = NULL;
	FLMUINT			uiCollection;
	F_COLLECTION *	pCollection;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	uiCollection = getCollection();
	
	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		if( rc != NE_XFLM_DOM_NODE_DELETED)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pBTree)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pDb->m_pDict->getCollection( uiCollection, &pCollection)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pBTree->btOpen( pDb, &pCollection->lfInfo, FALSE, TRUE)))
		{
			goto Exit;
		}
		
		ui64DocumentId = getDocumentId();
		uiKeyLen = sizeof( ucKey);
		if( RC_BAD( rc = flmNumber64ToStorage( ui64DocumentId, &uiKeyLen,
										ucKey, FALSE, TRUE)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pBTree->btLocateEntry(
									ucKey, sizeof( ucKey), &uiKeyLen, XFLM_EXCL)))
		{
			if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			}
	
			goto Exit;
		}
	
		for (;;)
		{
			if( RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
										&ui64DocumentId, &bNeg, &uiBytesProcessed)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pDb->getNode( 
				uiCollection, ui64DocumentId, &pNode)))
			{
				if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
				{
	
					// Better be able to find the node at this point!
	
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
	
			// If the node is a root node, we have a document we can
			// process.
	
			if( pNode->isRootNode())
			{
				if( *ifppDoc)
				{
					(*ifppDoc)->Release();
				}
				
				// Just use the reference on pNode for *ifppDoc
				
				*ifppDoc = pNode;
				pNode = NULL;
				goto Exit;
			}
	
			// Need to go to the next node.
	
			if( RC_BAD( rc = pBTree->btNextEntry( ucKey, uiKeyLen, &uiKeyLen)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
				}
				goto Exit;
			}
		}
	}
	else
	{
		// If we are not at the root node of the document,
		// jump to the root.
	
		if( !isRootNode())
		{
			if( RC_BAD( rc = pDb->getNode( uiCollection, getDocumentId(), &pNode)))
			{
				goto Exit;
			}
		}
		else
		{
			pNode = this;
			pNode->AddRef();
		}
	
		if( !pNode->getNextSibId())
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
			goto Exit;
		}
	
		if( RC_BAD( rc = ifpDb->getNode(
			uiCollection, getNextSibId(), ifppDoc)))
		{
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pBTree)
	{
		pBTree->Release();
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmReadStorageAsText(
	IF_IStream *		pIStream,
	FLMBYTE *			pucStorageData,
	FLMUINT				uiDataLen,
	FLMUINT				uiDataType,
	void *				pvBuffer,
	FLMUINT 				uiBufLen,
	eXFlmTextType		eTextType,
	FLMUINT				uiMaxCharsToRead,
	FLMUINT				uiCharOffset,
	FLMUINT *			puiCharsRead,
	FLMUINT *			puiBufferBytesUsed)
{
	RCODE							rc = NE_XFLM_OK;
	FLMBYTE						ucByte;
	FLMUINT						uiCharsDecoded = 0;
	FLMUINT						uiSENLen;
	FLMUINT						uiNumChars;
	FLMUINT						uiCharsOutput = 0;
	FLMUNICODE *				puzOutBuf = NULL;
	FLMBYTE *					pszOutBuf = NULL;
	void *						pvEnd = ((char *)pvBuffer) + uiBufLen;
	const FLMBYTE *			pucTmp;
	FLMUINT						uiLen;
	FLMBYTE						ucSENBuf[ 16];
	FLMBYTE						ucConvBuf[ 64];
	FLMUINT						uiLastUTFLen = 0;
	IF_IStream *				pStream = pIStream;
	IF_BufferIStream *		pConvStream = NULL;
	F_BinaryToTextStream		binaryToTextStream;

	// If the value is a number, convert to text

	if( uiDataType == XFLM_NUMBER_TYPE)
	{
		FLMBYTE			ucNumBuf[ FLM_MAX_NUM_BUF_SIZE];
		FLMUINT			uiNumBufLen;

		// Read the entire number into the temporary buffer.
		// NOTE: Numbers are not encoded with a length.  It
		// is expected that the number of bytes remaining
		// in the stream will be equal to the exact number of
		// bytes representing the number.  If this is not the case,
		// either a corruption error or an incorrect value will
		// be returned.

		uiNumBufLen = sizeof( ucNumBuf);
		
		if( pStream)
		{
			if( RC_BAD( rc = pStream->read( (char *)ucNumBuf, 
				uiNumBufLen, &uiNumBufLen)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}
		}
		else
		{
			f_memcpy( ucNumBuf, pucStorageData, f_max( uiNumBufLen, uiDataLen));
		}

		// Convert the storage number to storage text

		uiDataLen = sizeof( ucConvBuf);
		if( RC_BAD( rc = flmStorageNum2StorageText( ucNumBuf, uiNumBufLen,
			ucConvBuf, &uiDataLen)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = FlmAllocBufferIStream( &pConvStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pConvStream->openStream( 
			(const char *)ucConvBuf, uiDataLen)))
		{
			goto Exit;
		}

		pStream = pConvStream;
		pucStorageData = NULL;
	}
	else if( uiDataType == XFLM_BINARY_TYPE)
	{
		if( !pStream)
		{
			if( RC_BAD( rc = FlmAllocBufferIStream( &pConvStream)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pConvStream->openStream( 
				(const char *)pucStorageData, uiDataLen)))
			{
				goto Exit;
			}
			
			pStream = pConvStream;
		}
		
		if( RC_BAD( rc = binaryToTextStream.openStream( pStream, 
			uiDataLen, &uiDataLen)))
		{
			goto Exit;
		}
		
		pStream = &binaryToTextStream;
		pucStorageData = NULL;
	}
	else if( uiDataType == XFLM_NODATA_TYPE)
	{
		goto Empty_String;
	}
	else if( uiDataType != XFLM_TEXT_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
		goto Exit;
	}

	// Determine the SEN length

	if( pStream)
	{
		uiLen = 1;
		if( RC_BAD( rc = pStream->read( (char *)&ucSENBuf[ 0], uiLen, &uiLen)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
Empty_String:
				rc = NE_XFLM_OK;
				if( eTextType == XFLM_UTF8_TEXT)
				{
					if( pvBuffer)
					{
						*((FLMBYTE *)pvBuffer) = 0;
					}
					if( puiBufferBytesUsed)
					{
						*puiBufferBytesUsed = 1;
					}
				}
				else
				{
					flmAssert( eTextType == XFLM_UNICODE_TEXT);
					if( pvBuffer)
					{
						*((FLMUNICODE *)pvBuffer) = 0;
					}
					if( puiBufferBytesUsed)
					{
						*puiBufferBytesUsed = sizeof( FLMUNICODE);
					}
				}
			}
			goto Exit;
		}
		uiDataLen -= uiLen;
	}
	else
	{
		if( !uiDataLen)
		{
			goto Empty_String;
		}
		
		ucSENBuf[ 0] = *pucStorageData++;
		uiDataLen--;
	}

	if( (uiSENLen = f_getSENLength( ucSENBuf[ 0])) > 1)
	{
		uiLen = uiSENLen - 1;
		
		if( pStream)
		{
			if( RC_BAD( rc = pStream->read( 
				(char *)&ucSENBuf[ 1], uiLen, &uiLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if( uiDataLen < uiLen)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_EOF_HIT);
				goto Exit;
			}
			
			f_memcpy( &ucSENBuf[ 1], pucStorageData, uiLen);
			pucStorageData += uiLen;
		}
		
		uiDataLen -= uiLen;
	}

	pucTmp = &ucSENBuf[ 0];
	if( RC_BAD( rc = f_decodeSEN(
		&pucTmp, &ucSENBuf[ uiSENLen], &uiNumChars)))
	{
		goto Exit;
	}

	// If only a length is needed (number of bytes), we can
	// return that without parsing the string

	if( !pvBuffer)
	{
		uiCharsOutput = uiCharOffset >= uiNumChars
							? 0
							: uiNumChars - uiCharOffset;
							
		if( puiBufferBytesUsed)
		{
			if( eTextType == XFLM_UNICODE_TEXT)
			{
				*puiBufferBytesUsed = (uiCharsOutput + 1) * sizeof( FLMUNICODE);
			}
			else // UTF-8
			{
				*puiBufferBytesUsed = uiDataLen;
			}
		}
		
		goto Exit;
	}

	if( eTextType == XFLM_UTF8_TEXT)
	{
		if( !uiBufLen)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		pszOutBuf = (FLMBYTE *)pvBuffer;
	}
	else
	{
		flmAssert( eTextType == XFLM_UNICODE_TEXT);
		if( uiBufLen < sizeof( FLMUNICODE))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		puzOutBuf = (FLMUNICODE *)pvBuffer;
	}

	// If we have a zero-length string, jump to exit.

	if( !uiNumChars)
	{
		// Read the null terminator

		if( pStream)
		{
			uiLen = 1;
			if( RC_BAD( rc = pStream->read( (char *)&ucByte, uiLen, &uiLen)))
			{
				goto Exit;
			}
		}
		else
		{
			if( !uiDataLen)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			
			ucByte = *pucStorageData++;
			uiDataLen--;
		}

		if( ucByte != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
		
		goto Empty_String;
	}

	// Parse through the string, outputting data to the buffer as we go.

	uiCharsDecoded = 0;
	if( eTextType == XFLM_UNICODE_TEXT)
	{
		FLMUNICODE		uChar;

		while( uiCharsOutput < uiMaxCharsToRead)
		{
			if( pStream)
			{
				if( RC_BAD( rc = f_readUTF8CharAsUnicode( pStream, &uChar)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
Unicode_EOF_Hit:
						if( uiCharsDecoded != uiNumChars)
						{
							rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
							goto Exit;
						}
						
						rc = NE_XFLM_OK;
						break;
					}
					
					goto Exit;
				}
			}
			else
			{
				FLMBYTE	ucTmpUni[ 3];	
				
				if( !uiDataLen)
				{
					goto Unicode_EOF_Hit;
				}
				
				ucTmpUni[ 0] = *pucStorageData++;
				uiDataLen--;
				
				if( ucTmpUni[ 0] <= 0x7F)
				{
					if( !ucTmpUni[ 0])
					{
						goto Unicode_EOF_Hit;
					}
					
					uChar = (FLMUNICODE)ucTmpUni[ 0];
				}
				else
				{
					if( !uiDataLen)
					{
						goto Unicode_EOF_Hit;
					}
					
					ucTmpUni[ 1] = *pucStorageData++;
					uiDataLen--;
				
					if( (ucTmpUni[ 1] >> 6) != 0x02)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_UTF8);
						goto Exit;
					}
				
					if( (ucTmpUni[ 0] >> 5) == 0x06)
					{
						uChar = ((FLMUNICODE)( ucTmpUni[ 0] - 0xC0) << 6) +
											(FLMUNICODE)(ucTmpUni[ 1] - 0x80);
					}
					else
					{
						if( !uiDataLen)
						{
							goto Unicode_EOF_Hit;
						}
						
						ucTmpUni[ 2] = *pucStorageData++;
						uiDataLen--;
					
						if( (ucTmpUni[ 0] >> 4) != 0x0E || 
							 (ucTmpUni[ 2] >> 6) != 0x02)
						{
							rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_UTF8);
							goto Exit;
						}
					
						uChar = ((FLMUNICODE)(ucTmpUni[ 0] - 0xE0) << 12) +
											((FLMUNICODE)(ucTmpUni[ 1] - 0x80) << 6) +
											(FLMUNICODE)(ucTmpUni[ 2] - 0x80);
					}
				}
			}

			if( ++uiCharsDecoded > uiNumChars)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}

			if( uiCharOffset)
			{
				uiCharOffset--;
				continue;
			}

			if( puzOutBuf)
			{
				if( puzOutBuf + 1 >= pvEnd)
				{
					goto Overflow_Error;
				}
				
				*puzOutBuf++ = uChar;
				uiCharsOutput++;
			}
			else
			{
				if( uChar <= 0xFF)
				{
					if( pszOutBuf + 1 >= pvEnd)
					{
						goto Overflow_Error;
					}
					*pszOutBuf++ = f_tonative( (FLMBYTE)uChar);
					uiCharsOutput++;
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
					goto Exit;
				}
			}
		}
	}
	else // UTF-8
	{
		flmAssert( eTextType == XFLM_UTF8_TEXT);
		while( uiCharsOutput < uiMaxCharsToRead)
		{
			if( (uiLen = ((FLMBYTE *)pvEnd) - pszOutBuf) == 0)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Overflow_Error;
			}
			
			if( pStream)
			{
				if( RC_BAD( rc = f_readUTF8CharAsUTF8(
					pStream, pszOutBuf, &uiLen)))
				{
					if( rc == NE_XFLM_CONV_DEST_OVERFLOW)
					{
						goto Overflow_Error;
					}
					
					if( rc == NE_XFLM_EOF_HIT)
					{
UTF8_EOF_Hit:
						if( uiCharsDecoded != uiNumChars)
						{
							rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
							goto Exit;
						}
						
						rc = NE_XFLM_OK;
						break;
					}
	
					goto Exit;
				}
			}
			else
			{
				if( !uiDataLen)
				{
					goto UTF8_EOF_Hit;
				}
				
				pszOutBuf[ 0] = *pucStorageData++;
				uiDataLen--;
			
				if( pszOutBuf[ 0] <= 0x7F)
				{
					if( !pszOutBuf[ 0])
					{
						goto UTF8_EOF_Hit;
					}
					
					uiLen = 1;
				}
				else
				{
					if( uiLen < 2)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Overflow_Error;
					}
					
					if( !uiDataLen)
					{
						goto UTF8_EOF_Hit;
					}
					
					pszOutBuf[ 1] = *pucStorageData++;
					uiDataLen--;
					
					if( (pszOutBuf[ 1] >> 6) != 0x02)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_UTF8);
						goto Exit;
					}
				
					if( (pszOutBuf[ 0] >> 5) == 0x06)
					{
						uiLen = 2;
					}
					else
					{
						if( uiLen < 3)
						{
							rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
							goto Overflow_Error;
						}
					
						if( !uiDataLen)
						{
							goto UTF8_EOF_Hit;
						}
						
						pszOutBuf[ 2] = *pucStorageData++;
						uiDataLen--;
						
						if( (pszOutBuf[ 0] >> 4) != 0x0E || 
							(pszOutBuf[ 2] >> 6) != 0x02)
						{
							rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_UTF8);
							goto Exit;
						}
					
						uiLen = 3;
					}
				}
			}

			if( ++uiCharsDecoded > uiNumChars)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
				goto Exit;
			}

			if( uiCharOffset)
			{
				uiCharOffset--;
				continue;
			}

			if( pszOutBuf + uiLen >= pvEnd)
			{
				goto Overflow_Error;
			}

			pszOutBuf += uiLen;
			uiLastUTFLen = uiLen;
			uiCharsOutput++;
		}
	}

	// There is room for the 0 terminating character, but we
	// will not increment the return length, but we will output the
	// number of buffer bytes used

	if( eTextType == XFLM_UTF8_TEXT && pszOutBuf < pvEnd)
	{
		*pszOutBuf = 0;
		if( puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = (FLMUINT)(pszOutBuf - (FLMBYTE *)pvBuffer) + 1;
		}
	}
	else if( eTextType == XFLM_UNICODE_TEXT && &puzOutBuf[ 1] <= pvEnd)
	{
		*puzOutBuf = 0;
		if( puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = (FLMUINT)((FLMBYTE *)puzOutBuf - 
											(FLMBYTE *)pvBuffer) + sizeof( FLMUNICODE);
		}
	}
	else
	{
Overflow_Error:

		if( uiCharsOutput)
		{
			uiCharsOutput--;
			if( puzOutBuf)
			{
				*(puzOutBuf - 1) = 0;
				if( puiBufferBytesUsed)
				{
					*puiBufferBytesUsed = (FLMUINT)((FLMBYTE *)puzOutBuf - 
						(FLMBYTE *)pvBuffer);
				}
			}
			else
			{
				pszOutBuf -= uiLastUTFLen;
				*pszOutBuf = 0;
				if( puiBufferBytesUsed)
				{
					*puiBufferBytesUsed = (FLMUINT)(pszOutBuf - 
						(FLMBYTE *)pvBuffer) + 1;
				}
			}
		}
		else if( puiBufferBytesUsed)
		{
			*puiBufferBytesUsed = 0;
		}
		
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

Exit:

	if( pConvStream)
	{
		pConvStream->Release();
	}

	if( puiCharsRead)
	{
		*puiCharsRead = uiCharsOutput;
	}
	
	if( rc == NE_XFLM_EOF_HIT)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmReadStorageAsBinary(
	IF_IStream *	pIStream,
	void *			pvBuffer,
	FLMUINT 			uiBufLen,
	FLMUINT			uiByteOffset,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

	// Position to the requested offset

	if( uiByteOffset)
	{
		if( RC_BAD( rc = pIStream->read( 
			NULL, uiByteOffset, &uiByteOffset)))
		{
			goto Exit;
		}
	}

	// Read the requested bytes

	rc = pIStream->read( pucBuffer, uiBufLen, &uiBufLen);

	if( puiBytesRead)
	{
		*puiBytesRead = uiBufLen;
	}

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmReadStorageAsNumber(
	IF_IStream *	pIStream,
	FLMUINT			uiDataType,
	FLMUINT64 *		pui64Number,
	FLMBOOL *		pbNeg)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bNeg = FALSE;
	FLMUINT64	ui64Num = 0;

	switch( uiDataType)
	{
		case XFLM_NUMBER_TYPE :
		{
			FLMBYTE		ucNumBuf[ FLM_MAX_NUM_BUF_SIZE];
			FLMUINT		uiNumBufLen;

			// Read the entire number into the temporary buffer.
			// NOTE: Numbers are not encoded with a length.  It
			// is expected that the number of bytes remaining
			// in the stream will be equal to the exact number of
			// bytes representing the number.  If this is not the case,
			// either a corruption error or an incorrect value will
			// be returned.

			uiNumBufLen = sizeof( ucNumBuf);
			if( RC_BAD( rc = pIStream->read( 
				(char *)ucNumBuf, uiNumBufLen, &uiNumBufLen)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					goto Exit;
				}
				rc = NE_XFLM_OK;
			}

			if( RC_BAD( rc = flmStorageNumberToNumber( 
				ucNumBuf, uiNumBufLen, &ui64Num, &bNeg)))
			{
				goto Exit;
			}

			break;
		}

		case XFLM_TEXT_TYPE :
		{
			FLMUNICODE		uChar;
			FLMUINT			uiLoop;
			FLMBOOL			bHex = FALSE;
			FLMUINT			uiIncrAmount = 0;

			// Skip the character count

			if( RC_BAD( rc = f_readSEN64( pIStream, NULL, NULL)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					// Empty string

					rc = NE_XFLM_OK;
				}
				goto Exit;
			}

			for( uiLoop = 0;; uiLoop++)
			{
				if( RC_BAD( rc = f_readUTF8CharAsUnicode( 
					pIStream, &uChar)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						break;
					}
					else
					{
						goto Exit;
					}
				}

				if( uChar >= FLM_UNICODE_0 && uChar <= FLM_UNICODE_9)
				{
					uiIncrAmount = (FLMUINT)(uChar - FLM_UNICODE_0);
				}
				else if( uChar >= FLM_UNICODE_A && uChar <= FLM_UNICODE_F)
				{
					if( bHex)
					{
						uiIncrAmount = (FLMUINT)(uChar - FLM_UNICODE_A + 10);
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if( uChar >= FLM_UNICODE_a && uChar <= FLM_UNICODE_f)
				{
					if( bHex)
					{
						uiIncrAmount = (FLMUINT)(uChar - FLM_UNICODE_a + 10);
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if( uChar == FLM_UNICODE_X || uChar == FLM_UNICODE_x)
				{
					if( !ui64Num && !bHex)
					{
						bHex = TRUE;
					}
					else
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
						goto Exit;
					}
				}
				else if( uChar == FLM_UNICODE_HYPHEN && !uiLoop)
				{
					bNeg = TRUE;
				}
				else
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
					goto Exit;
				}

				if( !bHex)
				{
					if( ui64Num > (~(FLMUINT64)0) / 10)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
						goto Exit;
					}
	
					ui64Num *= (FLMUINT64)10;
				}
				else
				{
					if( ui64Num > (~(FLMUINT64)0) / 16)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
						goto Exit;
					}
	
					ui64Num *= (FLMUINT64)16;
				}
	
				if( ui64Num > (~(FLMUINT64)0) - (FLMUINT64)uiIncrAmount)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_NUM_OVERFLOW);
					goto Exit;
				}

				ui64Num += (FLMUINT64)uiIncrAmount;
			}

			break;
		}

		default :
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_BAD_DIGIT);
			goto Exit;
		}
	}

	*pui64Number = ui64Num;
	*pbNeg = bNeg;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE flmReadLine(
	IF_IStream *	pIStream,
	FLMBYTE *		pszBuffer,
	FLMUINT *		puiSize)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiMaxBytes = *puiSize;
	FLMUINT		uiOffset = 0;
	FLMBYTE		ucByte;

	*puiSize = 0;

	for( ;;)
	{
		if( RC_BAD( rc = pIStream->read( (char *)&ucByte, 1, NULL)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}

		if( ucByte == 0x0A || ucByte == 0x0D)
		{
			if( uiOffset)
			{
				break;
			}
			continue;
		}
		else
		{
			if( (uiOffset + 1) == uiMaxBytes)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_BUFFER_OVERFLOW);
				goto Exit;
			}

			pszBuffer[ uiOffset++] = (char)ucByte;
		}
	}

	pszBuffer[ uiOffset] = 0;
	*puiSize = uiOffset;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::openStream(
	F_Db *			pDb,
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId,
	FLMUINT32		ui32BlkAddr,
	FLMUINT			uiOffsetIndex)
{
	RCODE						rc = NE_XFLM_OK;
	F_COLLECTION *			pCollection;
	F_Dict *					pDict = pDb->m_pDict;
	F_Btree *				pBTree = NULL;
	
	if( RC_BAD( rc = pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pBTree)))
	{
		goto Exit;
	}

	// Set up the btree object

	if( RC_BAD( rc = pBTree->btOpen( pDb, &pCollection->lfInfo, FALSE, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = openStream( pDb, pBTree, XFLM_EXACT, 
		uiCollection, ui64NodeId, ui32BlkAddr, uiOffsetIndex)))
	{
		goto Exit;
	}

	pBTree = NULL;
	m_bReleaseBTree = TRUE;

Exit:

	if( pBTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pBTree);
	}

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::openStream(
	F_Db *			pDb,
	F_Btree *		pBTree,
	FLMUINT			uiFlags,
	FLMUINT			uiCollection,
	FLMUINT64 		ui64NodeId,
	FLMUINT32		ui32BlkAddr,
	FLMUINT			uiOffsetIndex)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( !m_pBTree);

	m_pDb = pDb;
	m_uiCollection = uiCollection;
	m_pBTree = pBTree;

	// Save the key and key length

	m_uiKeyLength = sizeof( m_ucKey);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64NodeId, &m_uiKeyLength,
									m_ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	m_ui32BlkAddr = ui32BlkAddr;
	m_uiOffsetIndex = uiOffsetIndex;

	if( RC_BAD( rc = m_pBTree->btLocateEntry(
		m_ucKey, sizeof( m_ucKey), &m_uiKeyLength, uiFlags,
		NULL, &m_uiStreamSize, &m_ui32BlkAddr, &m_uiOffsetIndex)))
	{
		if( rc == NE_XFLM_NOT_FOUND)
		{
			rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		}
		goto Exit;
	}

	if( uiFlags == XFLM_EXACT)
	{
		m_ui64NodeId = ui64NodeId;
	}
	else
	{
		if( RC_BAD( rc = flmCollation2Number( m_uiKeyLength, m_ucKey,
			&m_ui64NodeId, NULL, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc))
	{
		closeStream();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_BTreeIStream::positionTo(
	FLMUINT64		ui64Position)
{
	RCODE				rc = NE_XFLM_OK;

	if( ui64Position >= m_uiBufferStartOffset &&
		 ui64Position <= m_uiBufferStartOffset + m_uiBufferBytes)
	{
		m_uiBufferOffset = (FLMUINT)(ui64Position - m_uiBufferStartOffset);
	}
	else
	{
		if( !m_bDataEncrypted)
		{
			if( RC_BAD( rc = m_pBTree->btSetReadPosition( m_ucKey,
									m_uiKeyLength, (FLMUINT)ui64Position)))
			{
				goto Exit;
			}

			m_uiBufferStartOffset = (FLMUINT)ui64Position;
			m_uiBufferOffset = 0;
			m_uiBufferBytes = 0;
		}
		else
		{
			// When the data is encrypted, we can't just position the btree to a
			// new read position.  We must read the data chunk by chunk until we
			// get to the buffer that holds the specified position.  Then we can
			// decrypt the buffer and set the correct position.

			m_bBufferDecrypted = FALSE;

			if(  ui64Position > m_uiBufferStartOffset + m_uiBufferBytes)
			{
				while( ui64Position > m_uiBufferStartOffset + m_uiBufferBytes)
				{
					m_uiBufferStartOffset += m_uiBufferBytes;
				}
			}
			else
			{
				while( ui64Position < m_uiBufferStartOffset)
				{
					m_uiBufferStartOffset -= f_min( m_uiBufferStartOffset,
															  FLM_ENCRYPT_CHUNK_SIZE);
				}
			}

			flmAssert( ui64Position >= m_uiBufferStartOffset &&
						  ui64Position <= m_uiBufferStartOffset + m_uiBufferBytes);

			// If the new position uis out of range, we will get an error returned.

			if( RC_BAD( rc = m_pBTree->btSetReadPosition( m_ucKey,
									m_uiKeyLength, m_uiBufferStartOffset)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pBTree->btGetEntry(
				m_ucKey, m_uiKeyLength, m_uiKeyLength,
				m_pucBuffer, m_uiBufferSize, &m_uiBufferBytes)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					if( !m_uiBufferBytes)
					{
						goto Exit;
					}
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}

			flmAssert( m_uiBufferBytes <= FLM_ENCRYPT_CHUNK_SIZE);

			if( RC_BAD( rc = m_pDb->decryptData( m_uiEncDefId, m_ucIV,
				m_pucBuffer, m_uiBufferBytes, m_pucBuffer, m_uiBufferSize)))
			{
				goto Exit;
			}

			// Check to see if we are at the end of the encrypted data.

			if( m_uiBufferStartOffset + m_uiBufferBytes >= m_uiDataLength)
			{
				// Trim back to the valid decrypted data.

				m_uiBufferBytes -= (ENCRYPT_MIN_CHUNK_SIZE - 
												extraEncBytes( m_uiDataLength));
			}

			m_bBufferDecrypted = TRUE;
			m_uiBufferOffset = (FLMUINT)(ui64Position - m_uiBufferStartOffset);
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_BTreeIStream::read(
	void *			pvBuffer,
	FLMUINT			uiBytesToRead,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT			uiBufBytesAvail;
	FLMUINT			uiTmp;
	FLMUINT			uiOffset = 0;

	flmAssert( m_pBTree);
	
	if( m_bDataEncrypted && !m_bBufferDecrypted)
	{
		if( (m_uiBufferBytes - m_uiBufferOffset) > 0)
		{
			// Since we are now looking at encrypted data that was not
			// decrypted, we will need to move the data to the front of the
			// buffer, then read in enough data to fill the buffer (if there
			// is any there) before we can decrypt it.
			
			f_memmove( &m_ucBuffer[0], &m_ucBuffer[ m_uiBufferOffset], 
						 (m_uiBufferBytes - m_uiBufferOffset));
			m_uiBufferBytes -= m_uiBufferOffset;
			m_uiBufferOffset = 0;
			
			// Fill the remainder of the buffer
			
			if( RC_BAD( rc = m_pBTree->btGetEntry(
					m_ucKey, m_uiKeyLength, m_uiKeyLength,
					&m_ucBuffer[ m_uiBufferBytes], m_uiBufferSize - m_uiBufferBytes,
					&uiTmp)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			
			m_uiBufferBytes += uiTmp;
			
			flmAssert( extraEncBytes( m_uiBufferBytes) == 0);
			flmAssert( m_uiBufferBytes <= FLM_ENCRYPT_CHUNK_SIZE);
			
			// Now decrypt what we have.

			uiTmp = m_uiBufferBytes;
			if( RC_BAD( rc = m_pDb->decryptData( m_uiEncDefId, m_ucIV,
				m_ucBuffer, m_uiBufferBytes, m_ucBuffer, m_uiBufferSize)))
			{
				goto Exit;
			}
			
			// Check for the end of the data.
			
			if( m_uiBufferStartOffset + m_uiBufferBytes > m_uiDataLength)
			{
				// Trim back the buffer to valid data only.
				m_uiBufferBytes -= (ENCRYPT_MIN_CHUNK_SIZE - 
							extraEncBytes( m_uiDataLength));
			}
			
			m_bBufferDecrypted = TRUE;
		}
	}

	while( uiBytesToRead)
	{
		if( (uiBufBytesAvail = m_uiBufferBytes - m_uiBufferOffset) != 0)
		{
			uiTmp = f_min( uiBufBytesAvail, uiBytesToRead);
			if( pucBuffer)
			{
				f_memcpy( &pucBuffer[ uiOffset], 
					&m_pucBuffer[ m_uiBufferOffset], uiTmp);
			}
			m_uiBufferOffset += uiTmp;
			uiOffset += uiTmp;
			if( (uiBytesToRead -= uiTmp) == 0)
			{
				break;
			}
		}
		else
		{
			m_uiBufferStartOffset += m_uiBufferBytes;
			m_uiBufferOffset = 0;

			if( RC_BAD( rc = m_pBTree->btGetEntry(
				m_ucKey, m_uiKeyLength, m_uiKeyLength,
				m_pucBuffer, m_uiBufferSize, &m_uiBufferBytes)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					if( !m_uiBufferBytes)
					{
						goto Exit;
					}
					rc = NE_XFLM_OK;
					continue;
				}
				goto Exit;
			}
			
			// Check for encryption.
			
			if( m_bDataEncrypted)
			{
				flmAssert( m_uiBufferBytes <= FLM_ENCRYPT_CHUNK_SIZE);

				if( RC_BAD( rc = m_pDb->decryptData( m_uiEncDefId, m_ucIV,
					m_pucBuffer, m_uiBufferBytes, m_pucBuffer, m_uiBufferSize)))
				{
					goto Exit;
				}
				
				// Check to see if we are at the end of the encrypted data.

				if( m_uiBufferStartOffset + m_uiBufferBytes > m_uiDataLength) 
				{
					// Trim back to the valid decrypted data.

					m_uiBufferBytes -= (ENCRYPT_MIN_CHUNK_SIZE -
										extraEncBytes( m_uiDataLength));
				}

				m_bBufferDecrypted = TRUE;
			}
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiOffset;
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::decryptData(
	FLMUINT			uiEncDefId,
	FLMBYTE *		pucIV,
	void *			pvInBuf,
	FLMUINT			uiInLen,
	void *			pvOutBuf,
	FLMUINT			uiOutBufSize)
{
	RCODE				rc = NE_XFLM_OK;
	F_Dict *			pDict;
	F_ENCDEF *		pEncDef = NULL;
	FLMBYTE *		pucInBuf = (FLMBYTE *)pvInBuf;
	FLMBYTE *		pucOutBuf = (FLMBYTE *)pvOutBuf;
	FLMUINT			uiEncLen;
	FLMUINT			uiOutLen;
	
	if( m_pDatabase->m_bInLimitedMode)
	{
		rc = RC_SET( m_pDatabase->m_rcLimitedCode);
		goto Exit;
	}
	
	flmAssert( extraEncBytes( uiInLen) == 0);
	flmAssert( uiEncDefId);

	// Need the dictionary and encryption definition.
	
	if( RC_BAD( rc = getDictionary( &pDict)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDict->getEncDef( uiEncDefId, &pEncDef)))
	{
		goto Exit;
	}
	
	flmAssert( pEncDef);
	flmAssert( pEncDef->pCcs);
	flmAssert( pEncDef->uiEncKeySize);

	while( uiInLen)
	{
		uiEncLen = f_min( uiInLen, FLM_ENCRYPT_CHUNK_SIZE);
		uiOutLen = uiOutBufSize;

		if( RC_BAD( rc = pEncDef->pCcs->decryptFromStore( 
			pucInBuf, uiEncLen, pucOutBuf, &uiOutLen, pucIV)))
		{
			goto Exit;
		}

		flmAssert( uiOutLen == uiEncLen);

		pucInBuf += uiEncLen;
		uiInLen -= uiEncLen;

		pucOutBuf += uiEncLen;
		uiOutBufSize -= uiEncLen;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
F_NodePool::~F_NodePool()
{
	F_BTreeIStream *			pTmpBTreeIStream;

	while( (pTmpBTreeIStream = m_pFirstBTreeIStream) != NULL)
	{
		m_pFirstBTreeIStream = m_pFirstBTreeIStream->m_pNextInPool;
		pTmpBTreeIStream->m_refCnt = 0;
		pTmpBTreeIStream->m_pNextInPool = NULL;
		delete pTmpBTreeIStream;
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_NodePool::setup( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_NodeCacheMgr::insertDOMNode(
	F_DOMNode *			pNode)
{
	
	flmAssert( pNode->m_refCnt == 1);

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	pNode->resetDOMNode( TRUE);
	pNode->m_pNextInPool = m_pFirstNode;
	m_pFirstNode = pNode;
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_NodePool::allocBTreeIStream(
	F_BTreeIStream **	ppBTreeIStream)
{
	RCODE			rc = NE_XFLM_OK;

	if( m_pFirstBTreeIStream)
	{
		f_mutexLock( m_hMutex);
		if( !m_pFirstBTreeIStream)
		{
			f_mutexUnlock( m_hMutex);
		}
		else
		{
			f_resetStackInfo( m_pFirstBTreeIStream);
			*ppBTreeIStream = m_pFirstBTreeIStream;
			m_pFirstBTreeIStream = m_pFirstBTreeIStream->m_pNextInPool;
			(*ppBTreeIStream)->m_pNextInPool = NULL;
	
			f_mutexUnlock( m_hMutex);
			goto Exit;
		}
	}

	if( (*ppBTreeIStream = f_new F_BTreeIStream) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_NodePool::insertBTreeIStream(
	F_BTreeIStream *			pBTreeIStream)
{
	flmAssert( pBTreeIStream->m_refCnt == 1);

	pBTreeIStream->reset();
	f_mutexLock( m_hMutex);
	pBTreeIStream->m_pNextInPool = m_pFirstBTreeIStream;
	m_pFirstBTreeIStream = pBTreeIStream;
	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_DOMNode::getNamespaceURI(
	FLMBOOL			bUnicode,
	IF_Db *			ifpDb,
	void *			pvNamespaceURI,
	FLMUINT			uiBufSize,
	FLMUINT *		puiCharsReturned)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiTag;
	FLMUNICODE *	puzNamespaceURI;
	char *			pszNamespaceURI;
	FLMUINT			uiChars = 0;
	F_NameTable *	pNameTable = NULL;
	FLMUINT			uiNameId;
	eDomNodeType	eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	pNameTable = pDb->m_pDict->getNameTable();
	eNodeType = getNodeType();

	if( eNodeType == ELEMENT_NODE)
	{
		uiTag = ELM_ELEMENT_TAG;
		uiNameId = getNameId();
	}
	else if( eNodeType == ATTRIBUTE_NODE)
	{
		uiTag = ELM_ATTRIBUTE_TAG;
		uiNameId = m_uiAttrNameId;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	uiChars = uiBufSize;
	if( bUnicode)
	{
		puzNamespaceURI = (FLMUNICODE *)pvNamespaceURI;
		pszNamespaceURI = NULL;
	}
	else
	{
		puzNamespaceURI = NULL;
		pszNamespaceURI = (char *)pvNamespaceURI;
	}
	
	if( RC_BAD( rc = pNameTable->getFromTagTypeAndNum(
		pDb, uiTag, uiNameId, NULL, NULL, NULL, NULL,
		puzNamespaceURI, pszNamespaceURI, &uiChars, FALSE)))
	{
		goto Exit;
	}

Exit:

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiChars;
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE F_DOMNode::getLocalName(
	FLMBOOL			bUnicode,
	IF_Db *			ifpDb,
	void *			pvLocalName,
	FLMUINT			uiBufSize,
	FLMUINT *		puiCharsReturned)
{
	RCODE					rc = NE_XFLM_OK;
	F_Db *				pDb = (F_Db *)ifpDb;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiTag;
	F_NameTable *		pNameTable = NULL;
	FLMUINT				uiChars = 0;
	FLMUINT				uiNameId;
	eDomNodeType		eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	pNameTable = pDb->m_pDict->getNameTable();
	eNodeType = getNodeType();

	if( eNodeType == ELEMENT_NODE)
	{
		uiTag = ELM_ELEMENT_TAG;
		uiNameId = getNameId();
	}
	else if( eNodeType == ATTRIBUTE_NODE)
	{
		uiTag = ELM_ATTRIBUTE_TAG;
		uiNameId = m_uiAttrNameId;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	uiChars = uiBufSize;

	if( bUnicode)
	{
		if( RC_BAD( rc = pNameTable->getFromTagTypeAndNum( pDb,
					uiTag, uiNameId, (FLMUNICODE *)pvLocalName, NULL, &uiChars)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pNameTable->getFromTagTypeAndNum( pDb,
					uiTag, uiNameId, NULL, (char *)pvLocalName, &uiChars)))
		{
			goto Exit;
		}
	}

Exit:

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiChars;
	}

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:	Return the prefix name, either as Unicode or as Native.
*****************************************************************************/
RCODE F_DOMNode::getPrefix(
	FLMBOOL			bUnicode,
	IF_Db *			ifpDb,
	void *			pvPrefix,
	FLMUINT			uiBufSize,
	FLMUINT *		puiCharsReturned)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiPrefix;
	FLMUNICODE *	puzPrefix = (FLMUNICODE *)pvPrefix;
	char *			pszPrefix = (char *)pvPrefix;
	eDomNodeType	eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}
	
	eNodeType = getNodeType();
	
	if( eNodeType == ELEMENT_NODE)
	{
		uiPrefix = getPrefixId();
	}
	else if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getPrefixId(
			m_uiAttrNameId, &uiPrefix)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( uiPrefix)
	{
		if( bUnicode)
		{
			if( RC_BAD( rc = pDb->m_pDict->getPrefix(
				uiPrefix, puzPrefix, uiBufSize, puiCharsReturned)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pDb->m_pDict->getPrefix(
				uiPrefix, pszPrefix, uiBufSize, puiCharsReturned)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if( uiBufSize)
		{
			if( puzPrefix)
			{
				*puzPrefix = 0;
			}
		}

		if( puiCharsReturned)
		{
			*puiCharsReturned = 0;
		}

		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getQualifiedName(
	IF_Db *			ifpDb,
	FLMUNICODE *	puzQualifiedName,
	FLMUINT			uiBufSize,
	FLMUINT *		puiCharsReturned)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;
	FLMUINT		uiCharsReturned;
	FLMUINT		uiTmp;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getPrefix( ifpDb, puzQualifiedName,
		uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}

	if( uiCharsReturned)
	{
		if( puzQualifiedName)
		{
			uiBufSize -= (sizeof( FLMUNICODE) * uiCharsReturned);
			if( uiBufSize < sizeof( FLMUNICODE) * 2)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
			puzQualifiedName[ uiCharsReturned] = FLM_UNICODE_COLON;
			uiCharsReturned++;
			puzQualifiedName += uiCharsReturned;
			uiBufSize -= sizeof( FLMUNICODE);
		}
		else
		{
			uiCharsReturned++;
		}
	}

	if( RC_BAD( rc = getLocalName( ifpDb, puzQualifiedName, uiBufSize, &uiTmp)))
	{
		goto Exit;
	}

	uiCharsReturned += uiTmp;

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiCharsReturned;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::getQualifiedName(
	IF_Db *			ifpDb,
	char *			pszQualifiedName,
	FLMUINT			uiBufSize,
	FLMUINT *		puiCharsReturned)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = (F_Db *)ifpDb;
	FLMBOOL		bStartedTrans = FALSE;
	FLMUINT		uiCharsReturned;
	FLMUINT		uiTmp;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getPrefix( ifpDb, 
		pszQualifiedName, uiBufSize, &uiCharsReturned)))
	{
		goto Exit;
	}

	if( uiCharsReturned)
	{
		if( pszQualifiedName)
		{
			uiBufSize -= uiCharsReturned;
			if( uiBufSize < 2)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			pszQualifiedName[ uiCharsReturned] = ASCII_COLON;
			uiCharsReturned++;
			pszQualifiedName += uiCharsReturned;
			uiBufSize--;
		}
		else
		{
			uiCharsReturned++;
		}
	}

	if( RC_BAD( rc = getLocalName( ifpDb, 
		pszQualifiedName, uiBufSize, &uiTmp)))
	{
		goto Exit;
	}

	uiCharsReturned += uiTmp;

	if( puiCharsReturned)
	{
		*puiCharsReturned = uiCharsReturned;
	}

Exit:

	if( bStartedTrans)
	{
		pDb->transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:	Method to set the prefix on a DOM node.  Either a unicode or native
		buffer may be used. ** private **
*****************************************************************************/
RCODE F_DOMNode::setPrefix(
	FLMBOOL			bUnicode,
	IF_Db *			ifpDb,
	void *			pvPrefix)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiPrefixId;
	F_Db *			pDb = (F_Db *)ifpDb;
	FLMBOOL			bStartedTrans = FALSE;
	char *			pszPrefix = (char *)pvPrefix;
	FLMUNICODE *	puzPrefix = (FLMUNICODE *)pvPrefix;
	eDomNodeType	eNodeType;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// If the node is read-only, don't allow it to be changed

	if( getModeFlags() & FDOM_READ_ONLY)
	{
		rc = RC_SET( NE_XFLM_READ_ONLY);
		goto Exit;
	}

	eNodeType = getNodeType();
	
	// Make sure this is an element or attribute node

	if( eNodeType != ELEMENT_NODE && eNodeType != ATTRIBUTE_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	uiPrefixId = 0;
	if( puzPrefix && bUnicode)
	{
		// Find the prefix ID

		if( RC_BAD( rc = pDb->m_pDict->getPrefixId( pDb,
			puzPrefix, &uiPrefixId)))
		{
			if( rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PREFIX);
			}
			goto Exit;
		}
	}
	else if( pszPrefix)
	{
		// Find the prefix ID

		if( RC_BAD( rc = pDb->m_pDict->getPrefixId( pDb,
			pszPrefix, &uiPrefixId)))
		{
			if( rc == NE_XFLM_NOT_FOUND)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PREFIX);
			}
			goto Exit;
		}
	}

	// Set the prefix

	if( RC_BAD( rc = setPrefixId( ifpDb, uiPrefixId)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_DOMNode::setPrefixId(
	IF_Db *			ifpDb,
	FLMUINT			uiPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Db *			pDb = (F_Db *)ifpDb;
	F_Rfl *			pRfl = pDb->m_pDatabase->m_pRfl;
	FLMBOOL			bStartedTrans = FALSE;
	FLMBOOL			bMustAbortOnError = FALSE;
	FLMUINT			uiRflToken = 0;
	eDomNodeType	eNodeType;
	FLMUINT			uiTmp;

	if( RC_BAD( rc = pDb->checkTransaction( 
		XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Make sure the node is current

	if( RC_BAD( rc = syncFromDb( pDb)))
	{
		goto Exit;
	}

	// If the node is read-only, don't allow it to be changed

	if( getModeFlags() & FDOM_READ_ONLY)
	{
		rc = RC_SET( NE_XFLM_READ_ONLY);
		goto Exit;
	}

	eNodeType = getNodeType();
	
	// If the prefix isn't changing, don't do anything

	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->getPrefixId( 
			m_uiAttrNameId, &uiTmp)))
		{
			goto Exit;
		}
		
		if( uiPrefixId == uiTmp)
		{
			goto Exit;
		}
	}
	else if( eNodeType == ELEMENT_NODE)
	{
		if( uiPrefixId == getPrefixId())
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Verify that the prefix is valid

	if( RC_BAD( rc = pDb->m_pDict->getPrefix( 
		uiPrefixId, (char *)NULL, 0, NULL)))
	{
		goto Exit;
	}
	
	// Disable RFL logging
	
	pRfl->disableLogging( &uiRflToken);

	// Set the new prefix

	if( RC_BAD( rc = makeWriteCopy( pDb)))
	{
		goto Exit;
	}
	
	bMustAbortOnError = TRUE;
	
	if( eNodeType == ATTRIBUTE_NODE)
	{
		if( RC_BAD( rc = m_pCachedNode->setPrefixId( pDb,
			m_uiAttrNameId, uiPrefixId)))
		{
			goto Exit;
		}
	}
	else
	{
		setPrefixId( uiPrefixId);
	}

	// Update the node

	if( RC_BAD( rc = pDb->updateNode( m_pCachedNode, 0)))
	{
		goto Exit;
	}
	
	// Log the update
	
	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = pRfl->logNodeSetPrefixId( pDb, 
		getCollection(), getIxNodeId(), 
		m_uiAttrNameId, uiPrefixId)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		if( bMustAbortOnError)
		{
			pDb->setMustAbortTrans( rc);
		}
	}
	
	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			pDb->transAbort();
		}
		else
		{
			rc = pDb->transCommit();
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Method to compare two dom nodes.
*****************************************************************************/
FLMUINT XFLAPI F_DOMNode::compareNode(
	IF_DOMNode *	pNode,
	IF_Db *			pDb1,
	IF_Db *			pDb2,
	char *			pszErrBuff,
	FLMUINT			uiErrBuffLen)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiEqual = 0;
	F_DOMNode *		pLeftNode = (F_DOMNode *)this;
	F_DOMNode *		pRightNode = (F_DOMNode *)pNode;
	char				szBuffer[ 100];
	FLMUNICODE *	puzVal1 = NULL;
	FLMUNICODE *	puzVal2 = NULL;	
	char *			pucBinary1 = NULL;
	char *			pucBinary2 = NULL;
	FLMUINT			uiBytesReturned1;
	FLMUINT			uiBytesReturned2;
	FLMUINT			uiDataLen1;
	FLMUINT			uiDataLen2;
	FLMUINT			uiTmp1;
	FLMUINT			uiTmp2;
	FLMUINT64		ui64Tmp1;
	FLMUINT64		ui64Tmp2;

#define NODE_NOT_EQUAL 1

	szBuffer[0] = '\0';

	if( pLeftNode->getNodeType() != pRightNode->getNodeType())
	{
		f_sprintf( szBuffer, "Node Type mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}
	
	if( RC_BAD( rc = pLeftNode->getDataType( pDb1, &uiTmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getDataType( pDb2, &uiTmp2)))
	{
		goto Exit;
	}

	if( uiTmp1 != uiTmp2)
	{
		f_sprintf( szBuffer, "Data Type mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( pLeftNode->getCollection() != pRightNode->getCollection())
	{
		f_sprintf( szBuffer, "Collection mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getPrefixId( pDb1, &uiTmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getPrefixId( pDb2, &uiTmp2)))
	{
		goto Exit;
	}
	
	if( uiTmp1 != uiTmp2)
	{
		f_sprintf( szBuffer, "Prefix mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getNameId( pDb1, &uiTmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getNameId( pDb2, &uiTmp2)))
	{
		goto Exit;
	}
	
	if( uiTmp1 != uiTmp2)
	{
		f_sprintf( szBuffer, "Name Id mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getEncDefId( pDb1, &uiTmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getEncDefId( pDb2, &uiTmp2)))
	{
		goto Exit;
	}
	
	if( uiTmp1 != uiTmp2)
	{
		f_sprintf( szBuffer, "Encryption Id mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( (pLeftNode->getModeFlags() & FDOM_PERSISTENT_FLAGS) != 
		 (pRightNode->getModeFlags() & FDOM_PERSISTENT_FLAGS))
	{
		f_sprintf( szBuffer, "Flags mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getNodeId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getNodeId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Node Id mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getDocumentId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getDocumentId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Root Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getParentId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getParentId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Parent Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getFirstChildId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getFirstChildId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "First Child Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getLastChildId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getLastChildId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Last Child Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getPrevSibId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getPrevSibId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Previous Sibling Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getNextSibId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getNextSibId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Next Sibling Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( RC_BAD( rc = pLeftNode->getAnnotationId( pDb1, &ui64Tmp1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getAnnotationId( pDb2, &ui64Tmp2)))
	{
		goto Exit;
	}
	
	if( ui64Tmp1 != ui64Tmp2)
	{
		f_sprintf( szBuffer, "Annotation Node mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}
	
	if( RC_BAD( rc = pLeftNode->getDataLength( pDb1, &uiDataLen1)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pRightNode->getDataLength( pDb2, &uiDataLen2)))
	{
		goto Exit;
	}
	
	if( uiDataLen1 != uiDataLen2)
	{
		f_sprintf( szBuffer, "Data Length mismatch");
		uiEqual = NODE_NOT_EQUAL;
		goto Exit;
	}

	if( uiDataLen1)
	{
		switch( pLeftNode->getDataType())
		{
			case XFLM_NODATA_TYPE:
			{
				goto Exit;
			}

			case XFLM_TEXT_TYPE:
			{
			
				if( RC_BAD( rc = pLeftNode->getUnicode( pDb1, &puzVal1)))
				{
					f_sprintf( szBuffer, "getUnicode failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if(  RC_BAD( rc = pRightNode->getUnicode( pDb2, &puzVal2)))
				{
					f_sprintf( szBuffer, "getUnicode failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if( f_unicmp( puzVal1, puzVal2) != 0)
				{
					f_sprintf( szBuffer, "Data Value mismatch");
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				break;
			}

			case XFLM_NUMBER_TYPE:
			{
				if( RC_BAD( rc = pLeftNode->getUINT64( pDb1, &ui64Tmp1)))
				{
					f_sprintf( szBuffer, "getUINT64 failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}
			
				if( RC_BAD( rc = pRightNode->getUINT64( pDb2, &ui64Tmp2)))
				{
					f_sprintf( szBuffer, "getUINT64 failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if( ui64Tmp1 != ui64Tmp2)
				{
					f_sprintf( szBuffer, "Data Value mismatch");
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				break;
			}

			case XFLM_BINARY_TYPE:
			{
				if( RC_BAD( rc = f_alloc( uiDataLen1 + 1, &pucBinary1)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = f_alloc( uiDataLen2 + 1, &pucBinary2)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pLeftNode->getBinary( pDb1, pucBinary1,
					0, uiDataLen1, &uiBytesReturned1)))
				{
					f_sprintf( szBuffer, "getBinary failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if( RC_BAD( rc = pRightNode->getBinary(
					pDb2, pucBinary2, 0, uiDataLen2, &uiBytesReturned2)))
				{
					f_sprintf( szBuffer, "getBinary failed with rc==0x%04X.",
						(unsigned)rc);
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if( uiBytesReturned1 != uiBytesReturned2)
				{
					f_sprintf( szBuffer, "Return data length mismatch");
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				if( f_memcmp( pucBinary1, pucBinary2, uiBytesReturned1) != 0)
				{
					f_strcpy( szBuffer, "Data Value mismatch");
					uiEqual = NODE_NOT_EQUAL;
					goto Exit;
				}

				break;
			}
			
			default:
			{
				f_strcpy( szBuffer, "Invalid Data Type");
				uiEqual = NODE_NOT_EQUAL;
				goto Exit;
			}
		}
	}

Exit:

	f_memcpy( pszErrBuff, szBuffer, f_min( uiErrBuffLen, f_strlen( szBuffer)));
	pszErrBuff[ f_min( uiErrBuffLen, f_strlen( szBuffer))] = '\0';

	if( puzVal1)
	{
		f_free( &puzVal1);
	}

	if( puzVal2)
	{
		f_free( &puzVal2);
	}

	if( pucBinary1)
	{
		f_free( &pucBinary1);
	}

	if( pucBinary2)
	{
		f_free( &pucBinary2);
	}

	return( uiEqual);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_Db::getDictionaryDef(
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	IF_DOMNode **	ppDocumentNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	F_DataVector		searchKey;
	F_DataVector		foundKey;

	searchKey.reset();
	foundKey.reset();

	if( RC_BAD( rc = checkTransaction( 
		XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = searchKey.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = searchKey.setUINT( 1, uiDictNumber)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = keyRetrieve( XFLM_DICT_NUMBER_INDEX,
		&searchKey, XFLM_EXACT, &foundKey)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNode( XFLM_DICT_COLLECTION,
		foundKey.getDocumentID(), ppDocumentNode)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getElementNameId(
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzElementName,
	FLMUINT *				puiElementNameId)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_ELEMENT_TAG, puzElementName, NULL,
		TRUE, puzNamespaceURI, puiElementNameId)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getElementNameId(
	const char *			pszNamespaceURI,
	const char *			pszElementName,
	FLMUINT *				puiElementNameId)
{
	RCODE					rc	= NE_XFLM_OK;
	FLMUNICODE *		puzNamespaceURI = NULL;
	FLMUNICODE *		puzTmp;
	const char *		pszTmp;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( pszNamespaceURI && *pszNamespaceURI)
	{
		if( RC_BAD( rc = f_alloc( (f_strlen( pszNamespaceURI) + 1) *
									sizeof( FLMUNICODE), &puzNamespaceURI)))
		{
			goto Exit;
		}
		
		pszTmp = pszNamespaceURI;
		puzTmp = puzNamespaceURI;
		
		while (*pszTmp)
		{
			*puzTmp = (FLMUNICODE)(*pszTmp);
			pszTmp++;
			puzTmp++;
		}
		
		*puzTmp = 0;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_ELEMENT_TAG, NULL, pszElementName,
		TRUE, puzNamespaceURI, puiElementNameId)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	if( puzNamespaceURI)
	{
		f_free( &puzNamespaceURI);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getAttributeNameId(
	const FLMUNICODE *	puzNamespaceURI,
	const FLMUNICODE *	puzAttributeName,
	FLMUINT *				puiAttributeNameId)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_ATTRIBUTE_TAG, puzAttributeName, NULL,
		TRUE, puzNamespaceURI, puiAttributeNameId)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getAttributeNameId(
	const char *			pszNamespaceURI,
	const char *			pszAttributeName,
	FLMUINT *				puiAttributeNameId)
{
	RCODE					rc	= NE_XFLM_OK;
	FLMUNICODE *		puzNamespaceURI = NULL;
	FLMUNICODE *		puzTmp;
	const char *		pszTmp;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( pszNamespaceURI && *pszNamespaceURI)
	{
		if( RC_BAD( rc = f_alloc( (f_strlen( pszNamespaceURI) + 1) *
									sizeof( FLMUNICODE), &puzNamespaceURI)))
		{
			goto Exit;
		}
		
		pszTmp = pszNamespaceURI;
		puzTmp = puzNamespaceURI;
		
		while (*pszTmp)
		{
			*puzTmp = (FLMUNICODE)(*pszTmp);
			pszTmp++;
			puzTmp++;
		}
		
		*puzTmp = 0;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_ATTRIBUTE_TAG, NULL, pszAttributeName,
		TRUE, puzNamespaceURI, puiAttributeNameId)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	if( puzNamespaceURI)
	{
		f_free( &puzNamespaceURI);
	}

	return( rc);
}

/*****************************************************************************
Desc: Get a collection number from collection name.
******************************************************************************/
RCODE XFLAPI F_Db::getCollectionNumber(
	const char *			pszCollectionName,
	FLMUINT *				puiCollectionNumber)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_COLLECTION_TAG, NULL, pszCollectionName,
		FALSE, NULL, puiCollectionNumber)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc: Get a collection number from collection name.
******************************************************************************/
RCODE XFLAPI F_Db::getCollectionNumber(
	const FLMUNICODE *	puzCollectionName,
	FLMUINT *				puiCollectionNumber)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_COLLECTION_TAG, puzCollectionName, NULL,
		FALSE, NULL, puiCollectionNumber)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc: Get an index number from index name.
******************************************************************************/
RCODE XFLAPI F_Db::getIndexNumber(
	const char *			pszIndexName,
	FLMUINT *				puiIndexNumber)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_INDEX_TAG, NULL, pszIndexName,
		FALSE, NULL, puiIndexNumber)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc: Get an index number from index name.
******************************************************************************/
RCODE XFLAPI F_Db::getIndexNumber(
	const FLMUNICODE *	puzIndexName,
	FLMUINT *				puiIndexNumber)
{
	RCODE					rc	= NE_XFLM_OK;
	F_NameTable *		pNameTable = NULL;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNameTable( &pNameTable)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->getFromTagTypeAndName(
		this, ELM_INDEX_TAG, puzIndexName, NULL,
		FALSE, NULL, puiIndexNumber)))
	{
		goto Exit;
	}

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createDocument(
	FLMUINT					uiCollection,
	IF_DOMNode **			ppDocument,
	FLMUINT64 *				pui64NodeId)
{
	RCODE			rc = NE_XFLM_OK;

	if( uiCollection == XFLM_MAINT_COLLECTION)
	{
		// Users are not allowed to create documents in the
		// maintenance collection

		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = createRootNode( uiCollection, 0,
		DOCUMENT_NODE, (F_DOMNode **)ppDocument, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createRootElement(
	FLMUINT					uiCollection,
	FLMUINT					uiNameId,
	IF_DOMNode **			ppElement,
	FLMUINT64 *				pui64NodeId)
{
	RCODE		rc = NE_XFLM_OK;

	if( uiCollection == XFLM_MAINT_COLLECTION)
	{
		// Users are not allowed to create documents in the
		// maintenance collection

		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = createRootNode( uiCollection, uiNameId, 
			ELEMENT_NODE, (F_DOMNode **)ppElement, pui64NodeId)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::checkAndUpdateState(
	eDomNodeType		eNodeType,
	FLMUINT				uiNameId)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bUserDefined = FALSE;
	FLMUINT				uiDictType = 0;
	F_AttrElmInfo		defInfo;

	// The F_AttrElmInfo constructor should have set the state
	// to active.

	flmAssert( defInfo.m_uiState == ATTR_ELM_STATE_ACTIVE);

	if( eNodeType == ATTRIBUTE_NODE)
	{
		uiDictType = ELM_ATTRIBUTE_TAG;
		if( RC_BAD( rc = m_pDict->getAttribute( this, uiNameId, &defInfo)))
		{
			goto Exit;
		}

		bUserDefined = attributeIsUserDefined( uiNameId);
	}
	else if( eNodeType == ELEMENT_NODE)
	{
		uiDictType = ELM_ELEMENT_TAG;
		if( RC_BAD( rc = m_pDict->getElement( this, uiNameId, &defInfo)))
		{
			goto Exit;
		}

		bUserDefined = elementIsUserDefined( uiNameId);
	}
	else if( eNodeType == DATA_NODE)
	{
		if( uiNameId)
		{
			uiDictType = ELM_ELEMENT_TAG;
			if( RC_BAD( rc = m_pDict->getElement( this, uiNameId, &defInfo)))
			{
				goto Exit;
			}

			bUserDefined = elementIsUserDefined( uiNameId);
		}
	}
	else if( eNodeType != COMMENT_NODE &&
		eNodeType != DOCUMENT_NODE &&
		eNodeType != ANNOTATION_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	if( bUserDefined && defInfo.m_uiState != ATTR_ELM_STATE_ACTIVE)
	{
		if( defInfo.m_uiState == ATTR_ELM_STATE_PURGE )
		{
			// Marked as 'purged'. So, user is not allowed to add
			// new instances of this field.

			rc = (RCODE)(eNodeType == ELEMENT_NODE
							 ? RC_SET( NE_XFLM_ELEMENT_PURGED)
							 : RC_SET( NE_XFLM_ATTRIBUTE_PURGED));
			goto Exit;
		}
		else if( defInfo.m_uiState == ATTR_ELM_STATE_CHECKING)
		{
			// Because an occurance is being added, update the
			// state to be 'active'

			if( RC_BAD( rc = changeItemState( uiDictType, uiNameId,
										XFLM_ACTIVE_OPTION_STR)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::_updateNode(
	F_CachedNode *		pCachedNode,
	FLMUINT				uiFlags)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiCollection = pCachedNode->getCollection();
	F_COLLECTION *		pCollection;
	FLMBOOL				bMustAbortOnError = FALSE;
	FLMBOOL				bAdd = (uiFlags & FLM_UPD_ADD) ? TRUE : FALSE;
	F_AttrElmInfo		defInfo;
	
	// Logging should be done at a higher level

	flmAssert( !m_pDatabase->m_pRfl->isLoggingEnabled());
	flmAssert( uiCollection);
	
	// Mark the node as being dirty
	
	pCachedNode->setNodeDirty( this, bAdd);
	
	if( bAdd)
	{
		// Get a pointer to the collection

		if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
		{
			goto Exit;
		}

		// If the nodeId is greater than or equal to the next nodeId, we
		// will set the next nodeId to 1 greater than the new nodeId to avoid
		// running into the same nodeId later.
		//
		// Node ID should already be set at this point.
		
		flmAssert( pCachedNode->getNodeId());
		
		if( pCachedNode->getNodeId() >= pCollection->ui64NextNodeId)
		{
			pCollection->ui64NextNodeId = pCachedNode->getNodeId() + 1;
			pCollection->bNeedToUpdateNodes = TRUE;
		}

		bMustAbortOnError = TRUE;
	}
	else if( !pCachedNode->getNodeId())
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_TRANS_OP);
		goto Exit;
	}


	// Add the document to the list of documents that need to have
	// documentDone called at commit time.

	if( !(uiFlags & FLM_UPD_INTERNAL_CHANGE) && 
		uiCollection == XFLM_DICT_COLLECTION)
	{
		if( RC_BAD( rc = m_pDatabase->m_DocumentList.addNode(
				pCachedNode->getCollection(), pCachedNode->getDocumentId(), 0)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::getCachedBTree(
	FLMUINT				uiCollection,
	F_Btree **			ppBTree)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection;

	if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	if( m_pCachedBTree)
	{
		flmAssert( m_pCachedBTree->getRefCount() == 1);
		m_pCachedBTree->btClose();
	}
	else
	{
		// Reserve a B-Tree from the pool

		if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( 
			&m_pCachedBTree)))
		{
			goto Exit;
		}
	}

	// Set up the btree object

	if( RC_BAD( rc = m_pCachedBTree->btOpen( this,
		&pCollection->lfInfo, FALSE, TRUE)))
	{
		goto Exit;
	}

	m_pCachedBTree->AddRef();
	*ppBTree = m_pCachedBTree;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::flushNode(
	F_Btree *			pBTree,
	F_CachedNode *		pCachedNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection;
	FLMUINT64			ui64NodeId = pCachedNode->getNodeId();
	FLMBYTE				ucKeyBuf[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				uiKeyLen;
	FLMUINT				uiHeaderStorageSize;
	F_DynaBuf			dynaBuf( m_pDatabase->m_pucUpdBuffer, m_pDatabase->m_uiUpdBufferSize);
	FLMBOOL				bMustAbortOnError = FALSE;
	IF_PosIStream *	pIStream = NULL;
	FLMBYTE *			pucSrc = NULL;
	FLMBYTE *			pucTmp;
	FLMBOOL				bOutputNodeData;
	FLMBOOL				bTruncateOnReplace;
	FLMUINT32			ui32BlkAddr;
	FLMUINT				uiOffsetIndex;
	FLMBYTE *			pucIV = NULL;
	FLMUINT				uiIVLen = 0;
	FLMUINT				uiEncDefId = pCachedNode->getEncDefId();
	eDomNodeType		eNodeType = pCachedNode->getNodeType();
	FLMUINT				uiNodeDataLength;

	// Node should be dirty

	flmAssert( pCachedNode->nodeIsDirty());

	// Transaction IDs should match

	if( pCachedNode->getLowTransId() != getTransID())
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}
	
	uiNodeDataLength = pCachedNode->m_nodeInfo.uiDataLength;
	
	// Output the header

	if( (pCachedNode->getModeFlags() & FDOM_FIXED_SIZE_HEADER) == 0)
	{
		if( RC_BAD( rc = dynaBuf.allocSpace( MAX_DOM_HEADER_SIZE, 
			(void **)&pucTmp)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pCachedNode->headerToBuf( FALSE, 
			pucTmp, &uiHeaderStorageSize, NULL, NULL)))
		{
			goto Exit;
		}
		
		dynaBuf.truncateData( uiHeaderStorageSize);
	}
	else
	{
		if( RC_BAD( rc = dynaBuf.allocSpace( FIXED_DOM_HEADER_SIZE, 
			(void **)&pucTmp)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pCachedNode->headerToBuf( TRUE, pucTmp, 
			&uiHeaderStorageSize, NULL, NULL)))
		{
			goto Exit;
		}
	}

	// Get a pointer to the collection

	if( RC_BAD( rc = m_pDict->getCollection( pCachedNode->getCollection(),
							&pCollection)))
	{
		goto Exit;
	}

	uiKeyLen = sizeof( ucKeyBuf);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64NodeId, 
		&uiKeyLen, ucKeyBuf, FALSE, TRUE)))
	{
		goto Exit;
	}
	
	if( eNodeType == ELEMENT_NODE)
	{
		FLMUINT			uiLoop;
		NODE_ITEM *		pNodeItem;
		FLMUINT			uiNodeCount = pCachedNode->getChildElmCount();
		FLMUINT			uiTmpOffset;
		FLMUINT			uiPrevNameId = 0;
		FLMUINT64		ui64ElmNodeId = pCachedNode->getNodeId();
		
		// Go through the child element list and output them to the buffer
		// Note that the child element node count has already been output
		// as part of the node header.
		
		pNodeItem = pCachedNode->m_pNodeList;
		for( uiLoop = 0; uiLoop < uiNodeCount; pNodeItem++, uiLoop++)
		{
			uiTmpOffset = dynaBuf.getDataLength();
			
			if( RC_BAD( rc = dynaBuf.allocSpace( FLM_MAX_SEN_LEN * 2,
				(void **)&pucTmp)))
			{
				goto Exit;
			}
			
			flmAssert( pNodeItem->uiNameId > uiPrevNameId);
			flmAssert( pNodeItem->ui64NodeId > ui64ElmNodeId);
			
			uiTmpOffset += f_encodeSEN( pNodeItem->uiNameId - uiPrevNameId,
												  &pucTmp);
												  
			uiTmpOffset += f_encodeSEN( pNodeItem->ui64NodeId - ui64ElmNodeId, 
												  &pucTmp);

			uiPrevNameId = pNodeItem->uiNameId;												  
			dynaBuf.truncateData( uiTmpOffset);
		}
		
		// Export any attributes on the element
		
		if( pCachedNode->m_uiAttrCount)
		{
			if( RC_BAD( rc = pCachedNode->exportAttributeList( this, 
				&dynaBuf, NULL)))
			{
				goto Exit;
			}
		}
	}
	
	// Set up to output data
	
	if( uiNodeDataLength)
	{
		if( pCachedNode->getModeFlags() & FDOM_VALUE_ON_DISK)
		{
			if( pCachedNode->getModeFlags() & FDOM_FIXED_SIZE_HEADER)
			{
				bOutputNodeData = FALSE;
				bTruncateOnReplace = FALSE;
			}
			else
			{
				// If the value is on disk and we don't have a fixed-size header,
				// we'll have to read and output the entire node value.
				
				flmAssert( eNodeType != ELEMENT_NODE);
		
				if( RC_BAD( rc = pCachedNode->getIStream( this, NULL, &pIStream)))
				{
					goto Exit;
				}
				
				bOutputNodeData = TRUE;
				bTruncateOnReplace = TRUE;
			}
		}
		else
		{
			pucSrc = pCachedNode->getDataPtr();
			bOutputNodeData = TRUE;
			bTruncateOnReplace = TRUE;
		}
	}
	else
	{
		bOutputNodeData = FALSE;
		bTruncateOnReplace = TRUE;
	}
	
	if( bOutputNodeData)
	{
		FLMUINT	uiDataOutputSize = uiNodeDataLength;

		if( uiEncDefId)
		{
			F_ENCDEF *	pEncDef;
			
			if( RC_BAD( rc = m_pDict->getEncDef( uiEncDefId, &pEncDef)))
			{
				goto Exit;
			}
			
			uiIVLen = pEncDef->pCcs->getIVLen();
			flmAssert( uiIVLen == 8 || uiIVLen == 16);
			
			if( RC_BAD( rc = dynaBuf.allocSpace( uiIVLen, (void **)&pucIV)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pEncDef->pCcs->generateIV( uiIVLen, pucIV)))
			{
				goto Exit;
			}
			
			uiDataOutputSize = getEncLen( uiNodeDataLength);
		}
												
		if( RC_BAD( rc = dynaBuf.allocSpace( uiDataOutputSize, 
			(void **)&pucTmp)))
		{
			goto Exit;
		}
		
		if( pIStream)
		{
			if( RC_BAD( rc = pIStream->read( pucTmp, uiNodeDataLength, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			f_memcpy( pucTmp, pucSrc, uiNodeDataLength);
		}
		
		if( uiEncDefId)
		{
			if( RC_BAD( rc = encryptData( uiEncDefId, pucIV, 
				pucTmp, uiDataOutputSize, uiNodeDataLength, &uiDataOutputSize)))
			{
				goto Exit;
			}
		}
	}
	
	ui32BlkAddr = pCachedNode->getBlkAddr();
	uiOffsetIndex = pCachedNode->getOffsetIndex();
	
	if( pCachedNode->nodeIsNew())
	{
		// If this is a new node, the value will not be on disk.
		// This routine is only called for values that we are
		// keeping in memory.  If we stream a value out through
		// multiple buffers, it is written right away and the
		// node's dirty and new flags will be unset as soon as
		// the writing is done.

		if( pCachedNode->getModeFlags() & FDOM_VALUE_ON_DISK)
		{
			// This shouldn't happen

			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}
		
		if( RC_BAD( rc = pBTree->btInsertEntry(
						ucKeyBuf, uiKeyLen,
						dynaBuf.getBufferPtr(), dynaBuf.getDataLength(),
						TRUE, TRUE, &ui32BlkAddr, &uiOffsetIndex)))
		{
			if( rc == NE_XFLM_NOT_UNIQUE)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_EXISTS);
			}
			
			goto Exit;
		}
	}
	else
	{
		// Replace the node on disk.

		if( RC_BAD( rc = pBTree->btReplaceEntry(
						ucKeyBuf, uiKeyLen,
						dynaBuf.getBufferPtr(), dynaBuf.getDataLength(),
						TRUE, TRUE, bTruncateOnReplace, &ui32BlkAddr,
						&uiOffsetIndex)))
		{
			goto Exit;
		}
	}
	
	pCachedNode->setBlkAddr( ui32BlkAddr);
	pCachedNode->setOffsetIndex( uiOffsetIndex);
	
	// Clear the dirty flag and the new flag.

	pCachedNode->unsetNodeDirtyAndNew( this);

Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:	
******************************************************************************/
RCODE F_Db::encryptData(
	FLMUINT				uiEncDefId,
	FLMBYTE *			pucIV,
	FLMBYTE *			pucBuffer,
	FLMUINT				uiBufferSize,
	FLMUINT				uiDataLen,
	FLMUINT *			puiOutputLength)
{
	RCODE					rc = NE_XFLM_OK;
	F_Dict *				pDict;
	F_ENCDEF *			pEncDef;
	FLMUINT				uiEncLen;
	FLMUINT				uiTmpLen;
	FLMUINT				uiEncBuffLen;
	FLMUINT				uiDataToEncrypt = uiDataLen;
	FLMUINT				uiChunkLen;
	FLMBYTE *			pucEncTmp;
	FLMBYTE				ucEncryptBuffer[ FLM_ENCRYPT_CHUNK_SIZE];

	if( m_pDatabase->m_bInLimitedMode)
	{
		*puiOutputLength = uiDataLen;
		rc = RC_SET( m_pDatabase->m_rcLimitedCode);
		goto Exit;
	}
	
	// Need to retrieve the encryption key.
	
	if( RC_BAD( rc = getDictionary( &pDict)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDict->getEncDef( uiEncDefId, &pEncDef)))
	{
		goto Exit;
	}

	flmAssert( pEncDef);
	flmAssert( pEncDef->pCcs);
	
	// Check first to make sure we wil be able to encrypt the entire buffer
	// since we must return the encrypted data in the source buffer.
	
	uiEncLen = getEncLen( uiDataLen);

	if( uiEncLen > uiBufferSize)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Encrypt the buffer in chunks

	uiEncBuffLen = 0;
	pucEncTmp = &ucEncryptBuffer[0];

	while( uiDataToEncrypt)
	{
		FLMUINT		uiDataEncrypted;
		
		uiDataEncrypted = uiChunkLen = ((uiDataToEncrypt > FLM_ENCRYPT_CHUNK_SIZE)
																	? FLM_ENCRYPT_CHUNK_SIZE
																	: uiDataToEncrypt);
								
		if( extraEncBytes( uiChunkLen) != 0)
		{
			// If we are padding, we  *MUST* be on the last piece to encrypt!

			flmAssert( uiChunkLen == uiDataToEncrypt);
			uiChunkLen += (ENCRYPT_MIN_CHUNK_SIZE - extraEncBytes( uiChunkLen));
		}

		flmAssert( uiChunkLen <= FLM_ENCRYPT_CHUNK_SIZE);

		uiTmpLen = uiChunkLen;

		if( RC_BAD( rc = pEncDef->pCcs->encryptToStore( 
			pucBuffer, uiChunkLen, pucEncTmp, &uiTmpLen, pucIV)))
		{
			goto Exit;
		}
		
		flmAssert( uiTmpLen == uiChunkLen);
		f_memcpy( pucBuffer, pucEncTmp, uiChunkLen);
		
		pucBuffer += uiChunkLen;
		uiDataToEncrypt -= uiDataEncrypted;
		uiEncBuffLen += uiChunkLen;
		
		flmAssert( uiEncBuffLen <= uiEncLen);
	}

	flmAssert( uiEncBuffLen == uiEncLen);
	*puiOutputLength = uiEncLen;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createRootNode(
	FLMUINT					uiCollection,
	FLMUINT					uiElementNameId,
	eDomNodeType			eNodeType,
	F_DOMNode **			ppNewNode,
	FLMUINT64 *				pui64NodeId)
{
	RCODE						rc = NE_XFLM_OK;
	F_Rfl *					pRfl = m_pDatabase->m_pRfl;
	F_DOMNode *				pPrevSib = NULL;
	F_DOMNode *				pNewNode = NULL;
	F_CachedNode *			pCachedNode;
	F_COLLECTION *			pCollection;
	FLMBOOL					bMustAbortOnError = FALSE;
	FLMBOOL					bStartedTrans = FALSE;
	FLMUINT					uiRflToken = 0;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	if( eNodeType != ELEMENT_NODE && eNodeType != DOCUMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// If a specific node Id is requested, check to make sure it is not
	// already in use.

	if( pui64NodeId)
	{
		if( *pui64NodeId && (m_uiFlags & FDB_REBUILDING_DATABASE))
		{
			if( RC_OK( rc = getNode( uiCollection, *pui64NodeId, &pNewNode)))
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
				goto Exit;
			}
			else if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			// Set the value to zero so we don't use it.  We will
			// return the new nodeId.

			*pui64NodeId = 0;
		}
	}
	
	bMustAbortOnError = TRUE;
	
	// Check the state if this is a root element
	
	if( eNodeType == ELEMENT_NODE)
	{
		if( RC_BAD( rc = checkAndUpdateState( ELEMENT_NODE, uiElementNameId)))
		{
			goto Exit;
		}
	}

	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);

	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->createNode( this,
									uiCollection,
									(FLMUINT64)(pui64NodeId
													? *pui64NodeId
													: (FLMUINT64)0),
									&pNewNode)))
	{
		goto Exit;
	}
	
	// Clone the dictionary since the first/last document ID
	// values of the collection will be changed

	if( !(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if( RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}

	// Get a pointer to the collection

	if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	pCachedNode = pNewNode->m_pCachedNode;
	
	if( !pCollection->ui64FirstDocId)
	{
		pCollection->ui64FirstDocId = pCollection->ui64NextNodeId;
		pCollection->ui64LastDocId = pCollection->ui64NextNodeId;
	}
	else
	{
		pCachedNode->setPrevSibId( pCollection->ui64LastDocId);
		pCollection->ui64LastDocId = pCollection->ui64NextNodeId;
	}
	pCollection->bNeedToUpdateNodes = TRUE;
	
	if( eNodeType == ELEMENT_NODE)
	{
		F_AttrElmInfo	elmInfo;

		if( RC_BAD( rc = m_pDict->getElement( this, uiElementNameId, &elmInfo)))
		{
			goto Exit;
		}

		pCachedNode->setNameId( uiElementNameId);
		pCachedNode->setDataType( elmInfo.m_uiDataType);
		
		// Is this a node whose child elements must all be unique?
	
		if( elmInfo.m_uiFlags & ATTR_ELM_UNIQUE_SUBELMS)
		{
			flmAssert( elmInfo.m_uiDataType == XFLM_NODATA_TYPE);
			pCachedNode->setFlags( FDOM_HAVE_CELM_LIST);
		}
	}
	else
	{
		pCachedNode->setDataType( XFLM_NODATA_TYPE);
	}
	
	pCachedNode->setNodeType( eNodeType);
	pCachedNode->setDocumentId( pCachedNode->getNodeId());

	// Link the document into the document list

	bMustAbortOnError = TRUE;

	if( pui64NodeId && *pui64NodeId)
	{
		pCollection->ui64LastDocId = *pui64NodeId;
		pCollection->bNeedToUpdateNodes = TRUE;
	}

	// Output the node

	if( RC_BAD( rc = updateNode( pCachedNode, FLM_UPD_ADD)))
	{
		goto Exit;
	}

	// Retrieve the previous sibling and set its next sibling
	// value

	if( pCachedNode->getPrevSibId())
	{
		if( RC_BAD( rc = getNode( uiCollection,
			pCachedNode->getPrevSibId(), &pPrevSib)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pPrevSib->makeWriteCopy( this)))
		{
			goto Exit;
		}

		pPrevSib->setNextSibId( pCachedNode->getNodeId());

		if( RC_BAD( rc = updateNode( pPrevSib->m_pCachedNode,
			FLM_UPD_INTERNAL_CHANGE)))
		{
			goto Exit;
		}
	}

	// Check indexing if it is an element

	if( eNodeType == ELEMENT_NODE)
	{
		if( RC_BAD( rc = updateIndexKeys( 
			uiCollection, pNewNode, IX_ADD_NODE_VALUE, TRUE)))
		{
			goto Exit;
		}
	}

	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = pRfl->logNodeCreate( this, pCachedNode->getCollection(),
		pCachedNode->getNodeId(), eNodeType, uiElementNameId, 
		XFLM_ROOT, pNewNode->getNodeId())))
	{
		goto Exit;
	}

	if( pui64NodeId)
	{
		*pui64NodeId = pCachedNode->getNodeId();
	}

	if( ppNewNode)
	{
		if( *ppNewNode)
		{
			(*ppNewNode)->Release();
		}

		*ppNewNode = pNewNode;
		pNewNode = NULL;
	}

Exit:

	if( pNewNode)
	{
		pNewNode->Release();
	}

	if( pPrevSib)
	{
		pPrevSib->Release();
	}

	if( RC_BAD( rc) && bMustAbortOnError)
	{
		setMustAbortTrans( rc);
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::findNode(
	FLMUINT				uiCollection,
	FLMUINT64 *			pui64NodeId,
	FLMUINT				uiFlags)
{
	RCODE					rc = NE_XFLM_OK;
	F_Btree *			pBTree = NULL;
	F_BTreeIStream		btreeIStream;
		
	// Determine the node's B-Tree address

	if( RC_BAD( rc = getCachedBTree( uiCollection, &pBTree)))
	{
		goto Exit;
	}
	
	// At this point, we know that uiFlags is NOT XFLM_EXACT, but is
	// XFLM_INCL, XFLM_EXCL, XFLM_FIRST, or XFLM_LAST.  So we will
	// need to reassign ui64NodeId once we have located the node.

	if( RC_BAD( rc = btreeIStream.openStream( this, pBTree,
		uiFlags, uiCollection, *pui64NodeId, 0, 0)))
	{
		goto Exit;
	}
	*pui64NodeId = btreeIStream.m_ui64NodeId;

	// Close the input stream

	btreeIStream.closeStream();

Exit:

	if( pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::getNode(
	FLMUINT					uiCollection,
	FLMUINT64				ui64NodeId,
	FLMUINT					uiFlags,
	F_DOMNode **			ppNode)
{
	RCODE						rc = NE_XFLM_OK;
	F_DOMNode *				pNode = NULL;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Quick check to see if the user is re-retrieving the same
	// node that is currently pointed to by pNode

	if( uiFlags == XFLM_EXACT)
	{
		F_DOMNode *	pTmpNode;

		if( ((pTmpNode = *ppNode) != NULL) && 
			 pTmpNode->m_pCachedNode && pTmpNode->m_uiAttrNameId == 0)
		{
			if( pTmpNode->getNodeId() == ui64NodeId &&
				 pTmpNode->getCollection() == uiCollection &&
				 pTmpNode->getDatabase() == m_pDatabase)
			{
				if( RC_BAD( rc = pTmpNode->syncFromDb( this)))
				{
					if( rc == NE_XFLM_DOM_NODE_DELETED)
					{
						rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
					}
				}
				
				goto Exit;
			}
		}
	}
	else
	{
		if( getTransType() == XFLM_UPDATE_TRANS)
		{
			// Need to flush dirty nodes through to the B-Tree so that
			// look-ups will work correctly

			if( RC_BAD( rc = flushDirtyNodes()))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = findNode( uiCollection, &ui64NodeId, uiFlags)))
		{
			goto Exit;
		}
	}

	// Retrieve the node into cache.

	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->retrieveNode( this,
									uiCollection, ui64NodeId, &pNode)))
	{
		goto Exit;
	}

	if( *ppNode)
	{
		(*ppNode)->Release();
	}
	
	*ppNode = pNode;
	pNode = NULL;

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getAttribute(
	FLMUINT					uiCollection,
	FLMUINT64				ui64ElementId,
	FLMUINT					uiAttrName,
	IF_DOMNode **			ifppAttr)
{
	RCODE						rc = NE_XFLM_OK;
	F_DOMNode *				pElementNode = NULL;
	F_DOMNode *				pAttrNode = NULL;
	F_AttrItem *			pAttrItem;
	FLMBOOL					bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getNode( uiCollection, ui64ElementId,
		XFLM_EXACT, &pElementNode)))
	{
		goto Exit;
	}

	if( pElementNode->getNodeType() != ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( (pAttrItem = pElementNode->m_pCachedNode->getAttribute( uiAttrName,
								NULL)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttrNode)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttrNode->m_pCachedNode = pElementNode->m_pCachedNode;
	pAttrNode->m_pCachedNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	pAttrNode->m_uiAttrNameId = pAttrItem->m_uiNameId;

	if( ifppAttr)
	{
		if( *ifppAttr)
		{
			(*ifppAttr)->Release();
		}
	
		*ifppAttr = (IF_DOMNode *)pAttrNode;
		pAttrNode = NULL;
	}

Exit:

	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	if( pElementNode)
	{
		pElementNode->Release();
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getFirstDocument(
	FLMUINT					uiCollection,
	IF_DOMNode **			ppDocumentNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	if(  !pCollection->ui64FirstDocId)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = getNode( uiCollection, pCollection->ui64FirstDocId,
										ppDocumentNode)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getLastDocument(
	FLMUINT					uiCollection,
	IF_DOMNode **			ppDocumentNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection;
	FLMBOOL				bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	if(  !pCollection->ui64LastDocId)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = getNode( uiCollection, pCollection->ui64LastDocId,
							ppDocumentNode)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE XFLAPI F_Db::getDocument(
	FLMUINT					uiCollection,
	FLMUINT					uiFlags,
	FLMUINT64				ui64DocumentId,
	IF_DOMNode **			ppDocumentNode)
{
	RCODE					rc = NE_XFLM_OK;
	F_COLLECTION *		pCollection;
	FLMBOOL				bStartedTrans = FALSE;
	F_Btree *			pBTree = NULL;
	F_DOMNode *			pNode = NULL;
	FLMBYTE				ucKey [FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				uiKeyLen;
	FLMBOOL				bNeg;
	FLMUINT				uiBytesProcessed;
	FLMUINT64			ui64NodeId;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getCollection( uiCollection, &pCollection)))
	{
		goto Exit;
	}

	switch (uiFlags)
	{
		case XFLM_FIRST:
		{
			if( RC_BAD( rc = getNode( uiCollection, pCollection->ui64FirstDocId,
												ppDocumentNode)))
			{
				goto Exit;
			}

			break;
		}

		case XFLM_LAST:
		{
			if( RC_BAD( rc = getNode( uiCollection, pCollection->ui64LastDocId,
											ppDocumentNode)))
			{
				goto Exit;
			}

			break;
		}

		case XFLM_EXACT:
		{
			if( RC_BAD( rc = getNode( uiCollection, ui64DocumentId, &pNode)))
			{
				goto Exit;
			}

			if( !pNode->isRootNode())
			{
				rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
				goto Exit;
			}

			if( *ppDocumentNode)
			{
				(*ppDocumentNode)->Release();
			}

			// No need to do an AddRef on ppDocumentNode - just use the reference
			// from pNode and set pNode to NULL so it won't be released
			
			*ppDocumentNode = pNode;
			pNode = NULL;
			break;
		}

		case XFLM_INCL:
		case XFLM_EXCL:
		{
			if( getTransType() == XFLM_UPDATE_TRANS)
			{
				// Need to flush dirty nodes through to the B-Tree so that
				// look-ups will work correctly

				if( RC_BAD( rc = flushDirtyNodes()))
				{
					goto Exit;
				}
			}

			// Get a btree

			if( RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pBTree)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pBTree->btOpen( this, &pCollection->lfInfo,
										FALSE, TRUE)))
			{
				goto Exit;
			}

			uiKeyLen = sizeof( ucKey);
			if( RC_BAD( rc = flmNumber64ToStorage( ui64DocumentId, &uiKeyLen,
											ucKey, FALSE, TRUE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pBTree->btLocateEntry(
										ucKey, sizeof( ucKey), &uiKeyLen, XFLM_INCL)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
				}
				goto Exit;
			}

			// Make sure we hit a root node.  If not, continue reading until we do
			// or until we hit the end.  Root nodes are always linked together in
			// ascending order, so if there is another document, we will find it
			// simply by searching forward from where we are.  Then we can follow
			// document links.

			for (;;)
			{
				if( RC_BAD( rc = flmCollation2Number( uiKeyLen, ucKey,
											&ui64NodeId, &bNeg, &uiBytesProcessed)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = getNode( uiCollection, ui64NodeId, &pNode)))
				{
					if( rc == NE_XFLM_DOM_NODE_NOT_FOUND)
					{

						// Better be able to find the node at this point!

						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
						goto Exit;
					}
				}

				// If the node is a root node, we have a document we can
				// process.

				if( pNode->isRootNode())
				{
					if( uiFlags == XFLM_EXCL && ui64NodeId == ui64DocumentId)
					{
						if( RC_BAD( rc = pNode->getNextDocument( this, ppDocumentNode)))
						{
							goto Exit;
						}
					}
					else
					{
						if( *ppDocumentNode)
						{
							(*ppDocumentNode)->Release();
						}

						// No need to do an AddRef on ppDocumentNode - just use the reference
						// from pNode and set pNode to NULL so it won't be released
						
						*ppDocumentNode = pNode;
						pNode = NULL;
					}
					
					break;
				}

				// Need to go to the next node.

				if( RC_BAD( rc = pBTree->btNextEntry( ucKey, uiKeyLen, &uiKeyLen)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
					}
					goto Exit;
				}
			}
			
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
			goto Exit;
		}
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( pBTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pBTree);
	}

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:
Notes:	This routine assumes that the node has been unlinked.  It doesn't
			perform any checks to guarantee that removing the node won't cause
			referential integrity problems.
******************************************************************************/
RCODE F_Db::purgeNode(
	FLMUINT			uiCollection,
	FLMUINT64		ui64NodeId)
{
	RCODE				rc = NE_XFLM_OK;
	F_Btree *		pBTree = NULL;
	FLMUINT			uiKeyLen;
	FLMBYTE			ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMBOOL			bStartedTrans = FALSE;
	FLMBOOL			bMustAbortOnError = FALSE;

	// Make sure we're in an update transaction

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Remove the node from the collection

	if( RC_BAD( rc = getCachedBTree( uiCollection, &pBTree)))
	{
		goto Exit;
	}

	uiKeyLen = sizeof( ucKey);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64NodeId,
								&uiKeyLen, ucKey, FALSE, TRUE)))
	{
		goto Exit;
	}

	bMustAbortOnError = TRUE;

	if( RC_BAD( rc = pBTree->btRemoveEntry( ucKey, uiKeyLen)))
	{
		if( rc != NE_XFLM_NOT_FOUND)
		{
			goto Exit;
		}

		// Item may not have been flushed from cache yet

		rc = NE_XFLM_OK;
	}

	// Remove the node from the cache

	gv_XFlmSysData.pNodeCacheMgr->removeNode( this, uiCollection, ui64NodeId);

Exit:

	if( pBTree)
	{
		pBTree->Release();
	}

	if( RC_BAD( rc))
	{
		if( bMustAbortOnError)
		{
			setMustAbortTrans( rc);
		}
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::documentDone(
	FLMUINT			uiCollection,
	FLMUINT64		ui64DocId)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pNode = NULL;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiRflToken = 0;
	F_Rfl *			pRfl = m_pDatabase->m_pRfl;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( !m_pDatabase->m_DocumentList.isNodeInList( uiCollection, ui64DocId, 0))
	{
		goto Exit;
	}
	
	pRfl->disableLogging( &uiRflToken);

	if( uiCollection == XFLM_DICT_COLLECTION)
	{
		FLMUINT		uiDictDefType;
		
		if( RC_BAD( rc = dictDocumentDone( ui64DocId, FALSE, &uiDictDefType)))
		{
			goto Exit;
		}
		
		// If this is an encryption definition, we need to log the encryption
		// key

		if( uiDictDefType == ELM_ENCDEF_TAG && !(m_uiFlags & FDB_REPLAYING_RFL))
		{
			FLMBYTE			ucDynaBuf[ 64];
			F_DynaBuf		keyBuffer( ucDynaBuf, sizeof( ucDynaBuf));
			FLMUINT			uiKeySize;
			
			if( RC_BAD( rc = getNode( uiCollection, ui64DocId,
				XFLM_EXACT, &pNode)))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			
			if( RC_BAD( rc = pNode->getAttributeValueBinary( this, 
				ATTR_ENCRYPTION_KEY_TAG, &keyBuffer)))
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
			
			if( RC_BAD( rc = pNode->getAttributeValueUINT( this,
				ATTR_ENCRYPTION_KEY_SIZE_TAG, &uiKeySize)))
			{
				goto Exit;
			}
			
			pRfl->enableLogging( &uiRflToken);
		
			if( RC_BAD( rc = pRfl->logEncDefKey( this, (FLMUINT)ui64DocId, 
				keyBuffer.getBufferPtr(), keyBuffer.getDataLength(), uiKeySize)))
			{
				goto Exit;
			}
			
			pRfl->disableLogging( &uiRflToken);
		}
	}

	m_pDatabase->m_DocumentList.removeNode( uiCollection, ui64DocId, 0);
	pRfl->enableLogging( &uiRflToken);

	if( RC_BAD( rc = pRfl->logDocumentDone( this, uiCollection, ui64DocId)))
	{
		goto Exit;
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::documentDone(
	IF_DOMNode *	pDocNode)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	FLMUINT			uiCollection;
	FLMUINT64		ui64DocId;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDocNode->getCollection( this, &uiCollection)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDocNode->getDocumentId( this, &ui64DocId)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = documentDone( uiCollection, ui64DocId)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::import(
	IF_IStream *			ifpStream,
	FLMUINT					uiCollection,
	IF_DOMNode *			pNodeToLinkTo,
	eNodeInsertLoc			eInsertLoc,
	XFLM_IMPORT_STATS *	pImportStats)
{
	RCODE				rc = NE_XFLM_OK;
	F_XMLImport		xmlImport;
	F_DOMNode *		pNode = NULL;
	F_DOMNode *		pNewNode = NULL;

	if( RC_BAD( rc = xmlImport.setup()))
	{
		goto Exit;
	}
	
	if( (pNode = (F_DOMNode *)pNodeToLinkTo) != NULL)
	{
		pNode->AddRef();
	}

	for( ;;)
	{
		if( RC_BAD( rc = xmlImport.import( ifpStream, this, uiCollection,
			FLM_XML_COMPRESS_WHITESPACE_FLAG |
			FLM_XML_TRANSLATE_ESC_FLAG |
			FLM_XML_EXTEND_DICT_FLAG, pNode, eInsertLoc,
			&pNewNode, pImportStats)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			
			goto Exit;
		}

		if( RC_BAD( rc = documentDone( pNewNode)))
		{
			goto Exit;
		}
		
		// If pNode is NULL, we are creating separate documents
		// and pNode should remain NULL.  Otherwise, it should be
		// set to the newly created node, and all subsequent trees
		// should be linked as next siblings after this one.
		
		if( pNode)
		{
			pNode->Release();
			pNode = pNewNode;
			
			// No need to do pNode->AddRef() and pNewNode->Release() since
			// there is already a reference on pNewNode.
			
			pNewNode = NULL;
			eInsertLoc = XFLM_NEXT_SIB;
		}
		else
		{
			pNewNode->Release();
			pNewNode = NULL;
		}

		xmlImport.reset();
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pNewNode)
	{
		pNewNode->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::importDocument(
	IF_IStream *			ifpStream,
	FLMUINT					uiCollection,
	IF_DOMNode **			ppDoc,
	XFLM_IMPORT_STATS *	pImportStats)
{
	RCODE				rc = NE_XFLM_OK;
	F_XMLImport		xmlImport;
	F_DOMNode *		pNode = NULL;

	if( pNode)
	{
		pNode->AddRef();
	}

	if( RC_BAD( rc = xmlImport.setup()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = xmlImport.import( ifpStream, this, uiCollection,
		FLM_XML_COMPRESS_WHITESPACE_FLAG |
		FLM_XML_TRANSLATE_ESC_FLAG |
		FLM_XML_EXTEND_DICT_FLAG, NULL, XFLM_LAST_CHILD, &pNode, pImportStats)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = documentDone( pNode)))
	{
		goto Exit;
	}

	if( ppDoc)
	{
		if( *ppDoc)
		{
			(*ppDoc)->Release();
		}

		*ppDoc = pNode;
		pNode = NULL;
	}

Exit:

	if( pNode)
	{
		pNode->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createElemOrAttrDef(
	FLMBOOL				bElement,
	FLMBOOL				bUnicode,
	const void *		pvNamespaceURI,
	const void *		pvLocalName,
	FLMUINT				uiDataType,
	FLMBOOL				bUniqueChildElms,
	FLMUINT * 			puiNameId,
	F_DOMNode **		ppDocumentNode)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = createRootNode( XFLM_DICT_COLLECTION,
		bElement ? ELM_ELEMENT_TAG : ELM_ATTRIBUTE_TAG, ELEMENT_NODE, &pDoc)))
	{
		goto Exit;
	}

	if( pvNamespaceURI)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_TARGET_NAMESPACE_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( bUnicode)
		{
			if( RC_BAD( rc = pAttr->setUnicode(
				this, (FLMUNICODE *)pvNamespaceURI)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvNamespaceURI)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = pDoc->createAttribute( this, ATTR_NAME_TAG,
										(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( bUnicode)
	{
		if( RC_BAD( rc = pAttr->setUnicode( this, (FLMUNICODE *)pvLocalName)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvLocalName)))
		{
			goto Exit;
		}
	}

	if( puiNameId && *puiNameId)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_DICT_NUMBER_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( this, *puiNameId)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pDoc->createAttribute( this,
		ATTR_TYPE_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pAttr->setUTF8( this, 
		(const FLMBYTE *)fdictGetDataTypeStr( uiDataType))))
	{
		goto Exit;
	}

	if( bUniqueChildElms && bElement)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_UNIQUE_SUB_ELEMENTS_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)"true")))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = documentDone( pDoc)))
	{
		goto Exit;
	}

	if( puiNameId)
	{
		if( RC_BAD( rc = pDoc->getAttribute( this, ATTR_DICT_NUMBER_TAG,
										(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT( this, puiNameId)))
		{
			goto Exit;
		}
	}

	if( ppDocumentNode)
	{
		if( (*ppDocumentNode))
		{
			(*ppDocumentNode)->Release();
		}

		*ppDocumentNode = pDoc;
		pDoc = NULL;
	}

Exit:

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pAttr)
	{
		flmAssert( pAttr->m_refCnt <= 2);
		pAttr->Release();
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}
	else if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createPrefixDef(
	FLMBOOL				bUnicode,
	const void *		pvPrefixName,
	FLMUINT * 			puiPrefixNumber)
{
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	F_DOMNode *		pNumAttr = NULL;
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = createRootNode( XFLM_DICT_COLLECTION,
		ELM_PREFIX_TAG, ELEMENT_NODE, &pDoc)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDoc->createAttribute( this, ATTR_NAME_TAG,
										(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( bUnicode)
	{
		if( RC_BAD( rc = pAttr->setUnicode( this, (FLMUNICODE *)pvPrefixName)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvPrefixName)))
		{
			goto Exit;
		}
	}

	// Create the prefix number attribute if passed in.

	if( puiPrefixNumber && *puiPrefixNumber)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_DICT_NUMBER_TAG, (IF_DOMNode **)&pNumAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pNumAttr->setUINT( this, *puiPrefixNumber)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = documentDone( pDoc)))
	{
		goto Exit;
	}

	// Change the modes on the definition so that the name cannot be modified
	// and the definition cannot be deleted.  This needs to be done after
	// calling documentDone() because it sets the flags on the nodes of the
	// definition according to a set of default rules that do not correspond
	// to what we need to have in this case.

	if( RC_BAD( rc = pAttr->addModeFlags( 
		this, FDOM_READ_ONLY | FDOM_CANNOT_DELETE)))
	{
		goto Exit;
	}

	if( puiPrefixNumber)
	{
		if( RC_BAD( rc = pDoc->getAttribute( this, ATTR_DICT_NUMBER_TAG,
											(IF_DOMNode **)&pNumAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pNumAttr->getUINT( this, puiPrefixNumber)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( pNumAttr)
	{
		pNumAttr->Release();
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}
	else if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::createEncDef(
	FLMBOOL			bUnicode,
	const void *	pvEncType,
	const void *	pvEncName,
	FLMUINT			uiKeySize,
	FLMUINT * 		puiEncDefNumber)
{
	RCODE				rc = NE_XFLM_OK;
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = createRootNode( XFLM_DICT_COLLECTION,
		ELM_ENCDEF_TAG, ELEMENT_NODE, &pDoc)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDoc->createAttribute( this, ATTR_NAME_TAG,
										(IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( bUnicode)
	{
		if( RC_BAD( rc = pAttr->setUnicode( this, (FLMUNICODE *)pvEncName)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvEncName)))
		{
			goto Exit;
		}
	}

	// Create the encdef id attribute if passed in.

	if( puiEncDefNumber && *puiEncDefNumber)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_DICT_NUMBER_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( this, *puiEncDefNumber)))
		{
			goto Exit;
		}
	}

	// Create the algorithm attribute

	if( RC_BAD( rc = pDoc->createAttribute( this,
		ATTR_TYPE_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( bUnicode)
	{
		if( RC_BAD( rc = pAttr->setUnicode( this, (FLMUNICODE *)pvEncType)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvEncType)))
		{
			goto Exit;
		}
	}

	// Create the key size attribute

	if( uiKeySize)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_ENCRYPTION_KEY_SIZE_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( this, uiKeySize)))
		{
			goto Exit;
		}
	}

	// Call documentDone to complete it all.

	if( RC_BAD( rc = documentDone( pDoc)))
	{
		goto Exit;
	}

	if( puiEncDefNumber)
	{
		if( RC_BAD( rc = pDoc->getAttribute( this, ATTR_DICT_NUMBER_TAG,
											(IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT( this, puiEncDefNumber)))
		{
			goto Exit;
		}
	}

Exit:

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}
	else if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}


	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_Db::getPrefixId(
	const FLMUNICODE *	puzPrefixName,
	FLMUINT *				puiPrefixNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getPrefixId( 
		this, puzPrefixName, puiPrefixNumber)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_Db::getPrefixId(
	const char *			pszPrefixName,
	FLMUINT *				puiPrefixNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getPrefixId( 
		this, pszPrefixName, puiPrefixNumber)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_Db::getEncDefId(
	const char *	pszEncDefName,
	FLMUINT *		puiEncDefNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getEncDefId( this, 
		pszEncDefName, puiEncDefNumber)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE XFLAPI F_Db::getEncDefId(
	const FLMUNICODE *	puzEncDefName,
	FLMUINT *				puiEncDefNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;

	if( RC_BAD( rc = checkTransaction( XFLM_READ_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDict->getEncDefId( 
		this, puzEncDefName, puiEncDefNumber)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	return( rc);
}

/*****************************************************************************
Desc:	Create a new c ollection definition in the dictionary.
******************************************************************************/
RCODE F_Db::createCollectionDef(
	FLMBOOL				bUnicode,
	const void *		pvCollectionName,
	FLMUINT * 			puiCollectionNumber,
	FLMUINT				uiEncNumber)
{
	F_DOMNode *		pDoc = NULL;
	F_DOMNode *		pAttr = NULL;
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	
	if( RC_BAD( rc = checkTransaction( XFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// First, create the root element to hold the collection definition

	if( RC_BAD( rc = createRootNode( XFLM_DICT_COLLECTION,
		ELM_COLLECTION_TAG, ELEMENT_NODE, &pDoc)))
	{
		goto Exit;
	}

	// Create the collection name attruibute
	
	if( RC_BAD( rc = pDoc->createAttribute( 
		this, ATTR_NAME_TAG, (IF_DOMNode **)&pAttr)))
	{
		goto Exit;
	}

	if( bUnicode)
	{
		if( RC_BAD( rc = pAttr->setUnicode( 
			this, (FLMUNICODE *)pvCollectionName)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pAttr->setUTF8( this, (FLMBYTE *)pvCollectionName)))
		{
			goto Exit;
		}
	}

	// Create the collection number attribute if passed in.
	
	if( puiCollectionNumber && *puiCollectionNumber)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_DICT_NUMBER_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( this, *puiCollectionNumber)))
		{
			goto Exit;
		}
	}

	if( uiEncNumber)
	{
		if( RC_BAD( rc = pDoc->createAttribute( this,
			ATTR_ENCRYPTION_ID_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->setUINT( this, uiEncNumber)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = documentDone( pDoc)))
	{
		goto Exit;
	}

	if( puiCollectionNumber)
	{
		if( RC_BAD( rc = pDoc->getAttribute( 
			this, ATTR_DICT_NUMBER_TAG, (IF_DOMNode **)&pAttr)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pAttr->getUINT( this, puiCollectionNumber)))
		{
			goto Exit;
		}
	}

Exit:

	if( bStartedTrans)
	{
		if( RC_BAD( rc))
		{
			transAbort();
		}
		else
		{
			rc = transCommit();
		}
	}

	if( pDoc)
	{
		pDoc->Release();
	}

	if( pAttr)
	{
		pAttr->Release();
	}

	if( RC_BAD( rc))
	{
		setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::flushDirtyNodes( void)
{
	RCODE					rc = NE_XFLM_OK;
	F_CachedNode *		pNode;
	F_Btree *			pBtree = NULL;
	FLMUINT				uiRflToken = 0;
	FLMUINT				uiCollection = 0;
	
	if( !m_uiDirtyNodeCount)
	{
		goto Exit;
	}

	// Disable RFL logging

	m_pDatabase->m_pRfl->disableLogging( &uiRflToken);

	// All of the dirty nodes should be at the front of the list.

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	while ((pNode = m_pDatabase->m_pFirstNode) != NULL)
	{
		if( !pNode->nodeIsDirty())
		{
			break;
		}
		
		// Flushing the node should remove it from the front of the list.
		// Need to increment the use count on the node to prevent it from
		// being moved while we are flushing it to disk.
		
		pNode->incrNodeUseCount();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);

		if( uiCollection != pNode->getCollection())
		{
			if( pBtree)
			{
				pBtree->Release();
			}

			uiCollection = pNode->getCollection();
			if( RC_BAD( rc = getCachedBTree( uiCollection, &pBtree)))
			{
				goto Exit;
			}
		}

		rc = flushNode( pBtree, pNode);

		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		pNode->decrNodeUseCount();

		if( rc == NE_XFLM_NOT_FOUND)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		}

		if( RC_BAD( rc))
		{
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			goto Exit;
		}
	}

	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	flmAssert( !m_uiDirtyNodeCount);

Exit:

	if( uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if( pBtree)
	{
		pBtree->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Db::flushDirtyNode(
	F_CachedNode *		pNode)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	FLMUINT				uiRflToken = 0;
	F_Btree *			pBTree = NULL;

	m_pDatabase->m_pRfl->disableLogging( &uiRflToken);

	if( !pNode->nodeIsDirty())
	{
		goto Exit;
	}
		
	// Get a B-Tree object

	if( RC_BAD( rc = getCachedBTree( pNode->getCollection(), &pBTree)))
	{
		goto Exit;
	}
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;

	pNode->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = FALSE;

	rc = flushNode( pBTree, pNode);

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;

	pNode->decrNodeUseCount();

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	if( uiRflToken)
	{
		m_pDatabase->m_pRfl->enableLogging( &uiRflToken);
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	if( pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_Database::startPendingInput(
	FLMUINT				uiPendingType,
	F_CachedNode *		pPendingNode)
{
	RCODE			rc = NE_XFLM_OK;

	if( m_pPendingInput)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INPUT_PENDING);
		goto Exit;
	}

	// Not supported on element nodes
	
	if( pPendingNode->getNodeType() == ELEMENT_NODE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	
	m_uiPendingType = uiPendingType;
	m_pPendingInput = pPendingNode;
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	m_pPendingInput->incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	m_uiUpdCharCount = 0;
	m_bUpdFirstBuf = TRUE;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
void F_Database::endPendingInput( void)
{
	if( m_pPendingInput)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		m_pPendingInput->decrNodeUseCount();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		m_pPendingInput = NULL;
	}

	if( m_pPendingBTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &m_pPendingBTree);
		m_pPendingBTree = NULL;
	}

	m_uiPendingType = XFLM_NODATA_TYPE;
	m_bUpdFirstBuf = TRUE;
	m_uiUpdByteCount = 0;
	m_uiUpdCharCount = 0;
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::createAttribute(
	F_Db *			pDb,
	FLMUINT			uiAttrNameId,
	F_AttrItem **	ppAttrItem)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem = NULL;
	FLMUINT			uiInsertPos;

	// Attribute should not exist

	if (getAttribute( uiAttrNameId, &uiInsertPos) != NULL)
	{
		flmAssert( 0);
	}
	else
	{
	
		// Logging should be done by the caller
		
		flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	
		// Allocate the new attribute
		
		if( RC_BAD( rc = allocAttribute( pDb, uiAttrNameId, NULL, uiInsertPos,
									&pAttrItem, FALSE)))
		{
			goto Exit;
		}
	}

Exit:

	if( ppAttrItem)
	{
		*ppAttrItem = pAttrItem;
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_AttrItem * F_CachedNode::getAttribute(
	FLMUINT		uiAttrNameId,
	FLMUINT *	puiInsertPos)
{
	FLMUINT			uiLoop;
	F_AttrItem *	pAttrItem = NULL;
	FLMUINT			uiTblSize;
	FLMUINT			uiLow;
	FLMUINT			uiMid;
	FLMUINT			uiHigh;
	FLMUINT			uiTblNameId;
	
	if (!m_uiAttrCount)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	// If the attribute count is <= 4, do a sequential search through
	// the array.  Otherwise, do a binary search.
	
	if ((uiTblSize = m_uiAttrCount) <= 4)
	{
		for (uiLoop = 0; uiLoop < m_uiAttrCount; uiLoop++)
		{
			pAttrItem = m_ppAttrList [uiLoop];
			if (pAttrItem->m_uiNameId == uiAttrNameId)
			{
				break;
			}
			else if (pAttrItem->m_uiNameId > uiAttrNameId)
			{
				pAttrItem = NULL;
				break;
			}
		}
		if (uiLoop == m_uiAttrCount)
		{
			pAttrItem = NULL;
		}
		if (puiInsertPos)
		{
			*puiInsertPos = uiLoop;
		}
	}
	else
	{
		uiHigh = --uiTblSize;
		uiLow = 0;
		for (;;)
		{
			uiMid = (uiLow + uiHigh) / 2;
			uiTblNameId = m_ppAttrList [uiMid]->m_uiNameId;
			if (uiTblNameId == uiAttrNameId)
			{
				// Found Match
	
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid;
				}
				pAttrItem = m_ppAttrList [uiMid];
				goto Exit;
			}
	
			// Check if we are done
	
			if (uiLow >= uiHigh)
			{
				// Done, item not found
	
				if (puiInsertPos)
				{
					*puiInsertPos = (uiAttrNameId < uiTblNameId)
											 ? uiMid
											 : uiMid + 1;
				}
				goto Exit;
			}
	
			if (uiAttrNameId < uiTblNameId)
			{
				if (uiMid == 0)
				{
					if (puiInsertPos)
					{
						*puiInsertPos = 0;
					}
					goto Exit;
				}
				uiHigh = uiMid - 1;
			}
			else
			{
				if (uiMid == uiTblSize)
				{
					if (puiInsertPos)
					{
						*puiInsertPos = uiMid + 1;
					}
					goto Exit;
				}
				uiLow = uiMid + 1;
			}
		}
	}
	
Exit:

	return( pAttrItem);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getPrevSiblingNode(
	FLMUINT					uiCurrentNameId,
	IF_DOMNode **			ppSib)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	F_DOMNode *		pAttr = NULL;
	
	if( (pAttrItem = getPrevSibling( uiCurrentNameId)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttr)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttr->m_uiAttrNameId = pAttrItem->m_uiNameId;
	pAttr->m_pCachedNode = this;
	incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	
	if( ppSib)
	{
		if( *ppSib)
		{
			(*ppSib)->Release();
		}
		
		*ppSib = (IF_DOMNode *)pAttr;
		pAttr = NULL;
	}
	
Exit:

	if( pAttr)
	{
		pAttr->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getNextSiblingNode(
	FLMUINT					uiCurrentNameId,
	IF_DOMNode **			ppSib)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	F_DOMNode *		pAttr = NULL;
	
	if( (pAttrItem = getNextSibling( uiCurrentNameId)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->allocDOMNode( &pAttr)))
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		goto Exit;
	}
	
	pAttr->m_uiAttrNameId = pAttrItem->m_uiNameId;
	pAttr->m_pCachedNode = this;
	incrNodeUseCount();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	
	if( ppSib)
	{
		if( *ppSib)
		{
			(*ppSib)->Release();
		}
		
		*ppSib = (IF_DOMNode *)pAttr;
		pAttr = NULL;
	}
	
Exit:

	if( pAttr)
	{
		pAttr->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::allocAttribute(
	F_Db *			pDb,
	FLMUINT			uiAttrNameId,
	F_AttrItem *	pCopyFromItem,
	FLMUINT			uiInsertPos,
	F_AttrItem **	ppAttrItem,
	FLMBOOL			bMutexAlreadyLocked)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrElmInfo	defInfo;
	F_AttrItem *	pAttrItem = NULL;
	FLMUINT			uiSize;
	FLMBOOL			bMutexLocked = FALSE;

	if( RC_BAD( rc = pDb->m_pDict->getAttribute( 
		pDb, uiAttrNameId, &defInfo)))
	{
		flmAssert( 0);
		goto Exit;
	}
	
	if( !bMutexAlreadyLocked)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		bMutexLocked = TRUE;
	}

	if( (pAttrItem = new F_AttrItem) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if( pCopyFromItem)
	{
		pAttrItem->copyFrom( pCopyFromItem);
	}
	else
	{
		if( defInfo.getFlags() & ATTR_ELM_NS_DECL)
		{
			pAttrItem->m_uiFlags |= FDOM_NAMESPACE_DECL;
		}
	}

	// Find where this thing goes in the attribute list
	// If we are copying, we don't need to resize the list as it will
	// already be the correct size.
	
	if (!pCopyFromItem)
	{
		if (RC_BAD( rc = resizeAttrList( m_uiAttrCount + 1, bMutexLocked)))
		{
			goto Exit;
		}
		
		// NOTE: m_uiAttrCount will have been incremented to accommodate
		// the new attribute.
		
		// Move everything above the [uiInsertPos] slot up.  Remember, no need to
		// preserve the item at [m_uiAttrCount - 1], because we just increased
		// the size of the array, and there is nothing there right now.
		
		if (uiInsertPos < m_uiAttrCount - 1)
		{
			f_memmove( &m_ppAttrList [uiInsertPos + 1],
							&m_ppAttrList [uiInsertPos],
							sizeof( F_AttrItem *) * (m_uiAttrCount - uiInsertPos - 1));
		}
	}
	m_ppAttrList [uiInsertPos] = pAttrItem;

	pAttrItem->m_pCachedNode = this;
	pAttrItem->m_uiDataType = defInfo.getDataType();
	pAttrItem->m_uiNameId = uiAttrNameId;
	*ppAttrItem = pAttrItem;

	uiSize = pAttrItem->memSize();
	m_uiTotalAttrSize += uiSize;

	if (m_ui64HighTransId != FLM_MAX_UINT64)
	{
		gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiSize;
	}
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiSize;

	// Set to NULL so it won't get deleted below.

	pAttrItem = NULL;

Exit:

	if (pAttrItem)
	{
		delete pAttrItem;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	return( rc);
}
	
/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::freeAttribute(
	F_AttrItem *	pAttrItem,
	FLMUINT			uiPos)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bMutexLocked = FALSE;
	
	// Move everything above the [uiPos] slot down.
	
	if (uiPos < m_uiAttrCount - 1)
	{
		f_memmove( &m_ppAttrList [uiPos],
						&m_ppAttrList [uiPos + 1],
						sizeof( F_AttrItem *) * (m_uiAttrCount - uiPos - 1));
	}
	
	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	bMutexLocked = TRUE;
	if (RC_BAD( rc = resizeAttrList( m_uiAttrCount - 1, bMutexLocked)))
	{
		goto Exit;
	}

	delete pAttrItem;

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::setNumber64(
	F_Db *					pDb,
	FLMUINT					uiAttrNameId,
	FLMUINT64				ui64Value,
	FLMBOOL					bNeg,
	FLMUINT					uiEncDefId)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	FLMUINT			uiValLen = 0;
	FLMUINT			uiEncryptedLen;
	FLMBYTE			ucNumBuf[ 32];

	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	
	// Get a pointer to the attribute list item

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		if( RC_BAD( rc = createAttribute( pDb, uiAttrNameId, &pAttrItem)))
		{
			goto Exit;
		}
	}
	else
	{
		if( pAttrItem->m_uiFlags & FDOM_READ_ONLY)
		{
			rc = RC_SET( NE_XFLM_READ_ONLY);
			goto Exit;
		}
		pAttrItem->m_uiFlags &= ~(FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	}
	
	pAttrItem->m_ui64QuickVal = ui64Value;

	if( bNeg)
	{
		pAttrItem->m_uiFlags |= FDOM_SIGNED_QUICK_VAL;
	}
	else
	{
		pAttrItem->m_uiFlags |= FDOM_UNSIGNED_QUICK_VAL;
	}

	switch( pAttrItem->m_uiDataType)
	{
		case XFLM_NUMBER_TYPE:
		{
			if( ui64Value <= 0x7F && !uiEncDefId)
			{
				if( RC_BAD( rc = pAttrItem->setupAttribute( pDb, 0, 1, FALSE,
										FALSE)))
				{
					goto Exit;
				}

				*(pAttrItem->getAttrDataPtr()) = (FLMBYTE)ui64Value;
				goto Exit;
			}
			else
			{
				uiValLen = sizeof( ucNumBuf);
				if( RC_BAD( rc = flmNumber64ToStorage( 
					ui64Value, &uiValLen, ucNumBuf, bNeg, FALSE)))
				{
					goto Exit;
				}
			}
			
			break;
		}
		
		case XFLM_TEXT_TYPE:
		{
			FLMBYTE *	pucSen;
			FLMUINT		uiSenLen;
			
			if( bNeg)
			{
				ucNumBuf[ 1] = '-';
				f_ui64toa( ui64Value, (char *)&ucNumBuf[ 2]);
			}
			else
			{
				f_ui64toa( ui64Value, (char *)&ucNumBuf[ 1]);
			}
			
			uiValLen = f_strlen( (const char *)&ucNumBuf[ 1]);
			pucSen = &ucNumBuf [0];
			uiSenLen = f_encodeSEN( uiValLen, &pucSen, (FLMUINT)0);
			flmAssert( uiSenLen == 1);
			uiValLen += uiSenLen + 1;
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_DATA_TYPE);
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pAttrItem->setupAttribute( pDb, uiEncDefId,
		uiValLen, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	if( uiValLen)
	{
		f_memcpy( pAttrItem->getAttrDataPtr(), ucNumBuf, uiValLen);

		if( uiEncDefId)
		{
			if( RC_BAD( rc = pDb->encryptData( uiEncDefId, 
				pAttrItem->getAttrIVPtr(), pAttrItem->getAttrDataPtr(),
				pAttrItem->getAttrDataBufferSize(), uiValLen, &uiEncryptedLen)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		pAttrItem->m_uiPayloadLen = 0;
	}

	pAttrItem->m_uiDecryptedDataLen = uiValLen;
	
Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getNumber64(
	F_Db *					pDb,
	FLMUINT					uiAttrNameId,
	FLMUINT64 *				pui64Value,
	FLMBOOL *				pbNeg)
{
	RCODE						rc = NE_XFLM_OK;
	FLMBOOL					bNeg;
	FLMUINT64				ui64Value;
	F_AttrItem *			pAttrItem;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	
	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	else if( pAttrItem->m_uiFlags & FDOM_UNSIGNED_QUICK_VAL)
	{
		*pbNeg = FALSE;
		*pui64Value = pAttrItem->m_ui64QuickVal;
	}
	else if( pAttrItem->m_uiFlags & FDOM_SIGNED_QUICK_VAL)
	{
		*pbNeg = TRUE;
		*pui64Value = pAttrItem->m_ui64QuickVal;
	}
	else
	{
		if( RC_BAD( rc = getIStream( pDb, uiAttrNameId, &bufferIStream, &pIStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = flmReadStorageAsNumber( 
			pIStream, pAttrItem->m_uiDataType, &ui64Value, &bNeg)))
		{
			goto Exit;
		}

		*pui64Value = ui64Value;
		*pbNeg = bNeg;
	}
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::setBinary(
	F_Db *				pDb,
	FLMUINT				uiAttrNameId,
	const void *		pvValue,
	FLMUINT				uiValueLen,
	FLMUINT				uiEncDefId)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	FLMUINT			uiDecryptedValLen = 0;
	FLMUINT			uiEncryptedLen;

	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	
	// Get a pointer to the attribute list item

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		if( RC_BAD( rc = createAttribute( pDb, uiAttrNameId, &pAttrItem)))
		{
			goto Exit;
		}
	}
	else
	{
		if( pAttrItem->m_uiFlags & FDOM_READ_ONLY)
		{
			rc = RC_SET( NE_XFLM_READ_ONLY);
			goto Exit;
		}
		pAttrItem->m_uiFlags &= ~(FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	}
	
	switch( pAttrItem->m_uiDataType)
	{
		case XFLM_BINARY_TYPE:
		{
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = pAttrItem->setupAttribute( pDb, uiEncDefId, 
		uiValueLen, TRUE, FALSE)))
	{
		goto Exit;
	}
		
	if( uiValueLen)
	{
		f_memcpy( pAttrItem->getAttrDataPtr(), pvValue, uiValueLen);
		
		if( uiEncDefId)
		{
			uiDecryptedValLen = uiValueLen;
			if( RC_BAD( rc = pDb->encryptData( uiEncDefId, 
				pAttrItem->getAttrIVPtr(), pAttrItem->getAttrDataPtr(),
				pAttrItem->getAttrDataBufferSize(), uiValueLen, &uiEncryptedLen)))
			{
				goto Exit;
			}
			
			flmAssert( uiEncryptedLen == pAttrItem->getAttrDataBufferSize());
		}
	}
	
	pAttrItem->m_uiDecryptedDataLen = uiValueLen;
	
Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::getBinary(
	F_Db *				pDb,
	FLMUINT				uiAttrNameId,
	void *				pvBuffer,
	FLMUINT				uiBufferLen,
	FLMUINT *			puiDataLen)
{
	RCODE						rc = NE_XFLM_OK;
	F_AttrItem *			pAttrItem;
	IF_PosIStream *		pIStream = NULL;
	F_NodeBufferIStream	bufferIStream;
	
	// If a NULL buffer is passed in, just return the
	// data length

	if( !pvBuffer)
	{
		rc = getDataLength( uiAttrNameId, puiDataLen);
		goto Exit;
	}
	
	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	if( !pAttrItem->m_uiEncDefId)
	{
		FLMUINT	uiCopySize = f_min( uiBufferLen, 
										pAttrItem->getAttrDataLength()); 

		if( uiCopySize)
		{
			f_memcpy( pvBuffer, pAttrItem->getAttrDataPtr(), uiCopySize);
		}
		
		if( puiDataLen)
		{
			*puiDataLen = uiCopySize;
		}
	}
	else
	{
		if( RC_BAD( rc = getIStream( pDb, uiAttrNameId, &bufferIStream, &pIStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = flmReadStorageAsBinary( pIStream, 
			pvBuffer, uiBufferLen, 0, puiDataLen)))
		{
			goto Exit;
		}
	}
	
Exit:

	if( pIStream)
	{
		pIStream->Release();
	}

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::setUTF8(
	F_Db *					pDb,
	FLMUINT					uiAttrNameId,
	const void *			pvValue,
	FLMUINT					uiNumBytesInValue,
	FLMUINT					uiNumCharsInValue,
	FLMUINT					uiEncDefId)
{
	RCODE						rc = NE_XFLM_OK;
	F_AttrItem *			pAttrItem;
	FLMUINT					uiValLen = 0;
	FLMUINT					uiDecryptedValLen = 0;
	FLMUINT					uiEncryptedLen;
	FLMUINT					uiSenLen;
	FLMBYTE *				pucValue = (FLMBYTE *)pvValue;
	FLMBOOL					bNullTerminate = FALSE;

	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	
	// Get a pointer to the attribute list item

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		if( RC_BAD( rc = createAttribute( pDb, uiAttrNameId, &pAttrItem)))
		{
			goto Exit;
		}
	}
	else
	{
		if( pAttrItem->m_uiFlags & FDOM_READ_ONLY)
		{
			rc = RC_SET( NE_XFLM_READ_ONLY);
			goto Exit;
		}
		pAttrItem->m_uiFlags &= ~(FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	}
	
	switch( pAttrItem->m_uiDataType)
	{
		case XFLM_TEXT_TYPE:
		{
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if( pvValue && uiNumBytesInValue)
	{
		if( pucValue[ uiNumBytesInValue - 1] != 0)
		{
			bNullTerminate = TRUE;
		}
		
		uiSenLen = f_getSENByteCount( uiNumCharsInValue);
		uiValLen = uiNumBytesInValue + uiSenLen + (bNullTerminate ? 1 : 0);
	}
	else
	{
		uiSenLen = 0;
		uiValLen = 0;
	}

	if( RC_BAD( rc = pAttrItem->setupAttribute( pDb, uiEncDefId,
		uiValLen, TRUE, FALSE)))
	{
		goto Exit;
	}
	
	if( uiValLen)
	{
		FLMBYTE *	pucTmp = pAttrItem->getAttrDataPtr();
		
		f_encodeSENKnownLength( uiNumCharsInValue, uiSenLen, &pucTmp);
		f_memcpy( pucTmp, pucValue, uiNumBytesInValue);
		
		if( bNullTerminate)
		{
			pucTmp[ uiNumBytesInValue] = 0;
		}
		
		if( uiEncDefId)
		{
			uiDecryptedValLen = uiValLen;
			if( RC_BAD( rc = pDb->encryptData( uiEncDefId, 
				pAttrItem->getAttrIVPtr(), pAttrItem->getAttrDataPtr(),
				pAttrItem->getAttrDataBufferSize(), uiValLen, &uiEncryptedLen)))
			{
				goto Exit;
			}
		}
	}

	pAttrItem->m_uiDecryptedDataLen = uiValLen;
	
Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::setStorageValue(
	F_Db *				pDb,
	FLMUINT				uiAttrNameId,
	const void *		pvValue,
	FLMUINT				uiValueLen,
	FLMUINT				uiEncDefId)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	FLMUINT			uiDecryptedValLen = 0;
	FLMUINT			uiEncryptedLen;

	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());
	
	// Get a pointer to the attribute list item

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		if( RC_BAD( rc = createAttribute( pDb, uiAttrNameId, &pAttrItem)))
		{
			goto Exit;
		}
	}
	else
	{
		pAttrItem->m_uiFlags &= ~(FDOM_UNSIGNED_QUICK_VAL | FDOM_SIGNED_QUICK_VAL);
	}

	if( pAttrItem->m_uiDataType == XFLM_UNKNOWN_TYPE)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = pAttrItem->setupAttribute( pDb, uiEncDefId, 
		uiValueLen, TRUE, FALSE)))
	{
		goto Exit;
	}
		
	if( uiValueLen)
	{
		f_memcpy( pAttrItem->getAttrDataPtr(), pvValue, uiValueLen);
		
		if( uiEncDefId)
		{
			uiDecryptedValLen = uiValueLen;
			if( RC_BAD( rc = pDb->encryptData( uiEncDefId, 
				pAttrItem->getAttrIVPtr(), pAttrItem->getAttrDataPtr(),
				pAttrItem->getAttrDataBufferSize(), uiValueLen, &uiEncryptedLen)))
			{
				goto Exit;
			}
			
			flmAssert( uiEncryptedLen == pAttrItem->getAttrDataBufferSize());
		}
	}
	
	pAttrItem->m_uiDecryptedDataLen = uiValueLen;
	
Exit:

	if( RC_BAD( rc))
	{
		pDb->setMustAbortTrans( rc);
	}
	
	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::addModeFlags(
	F_Db *	pDb,
	FLMUINT	uiAttrNameId,
	FLMUINT	uiFlags)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;

	F_UNREFERENCED_PARM( pDb);
	
	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	pAttrItem->m_uiFlags |= uiFlags;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::removeModeFlags(
	F_Db *	pDb,
	FLMUINT	uiAttrNameId,
	FLMUINT	uiFlags)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;
	
	F_UNREFERENCED_PARM( pDb);
	
	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	pAttrItem->m_uiFlags &= ~uiFlags;
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CachedNode::setPrefixId(
	F_Db *	pDb,
	FLMUINT	uiAttrNameId,
	FLMUINT	uiPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	F_AttrItem *	pAttrItem;

	F_UNREFERENCED_PARM( pDb);
	
	// Logging should be done by the caller
	
	flmAssert( !pDb->m_pDatabase->m_pRfl->isLoggingEnabled());

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}

	pAttrItem->m_uiPrefixId = uiPrefixId;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CachedNode::getIStream(
	F_Db *						pDb,
	FLMUINT						uiAttrNameId,
	F_NodeBufferIStream *	pStackStream,
	IF_PosIStream **			ppIStream,
	FLMUINT *					puiDataType,
	FLMUINT *					puiDataLength)
{
	RCODE							rc = NE_XFLM_OK;
	F_AttrItem *				pAttrItem;
	F_NodeBufferIStream *	pNodeBufferIStream = NULL;
	FLMBYTE *					pucAllocatedBuffer = NULL;

	if( (pAttrItem = getAttribute( uiAttrNameId, NULL)) == NULL)
	{
		rc = RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND);
		goto Exit;
	}
	
	if( pStackStream)
	{
		pNodeBufferIStream = pStackStream;
		pStackStream->AddRef();
		flmAssert( !pStackStream->m_pCachedNode);
	}
	else
	{
		if( (pNodeBufferIStream = f_new F_NodeBufferIStream) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	
	if( pAttrItem->m_uiEncDefId)
	{
		flmAssert( pAttrItem->m_uiIVLen);
		
		if( RC_BAD( rc = pNodeBufferIStream->openStream( NULL, 
			pAttrItem->getAttrDataBufferSize(), (char **)&pucAllocatedBuffer)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = pDb->decryptData(
			pAttrItem->m_uiEncDefId, pAttrItem->getAttrIVPtr(),
			pAttrItem->getAttrDataPtr(), pAttrItem->getAttrDataBufferSize(),
			pucAllocatedBuffer, (FLMUINT)pNodeBufferIStream->totalSize())))
		{
			goto Exit;
		}

		pNodeBufferIStream->truncate( pAttrItem->getAttrDataLength());
	}
	else
	{
		if( RC_BAD( rc = pNodeBufferIStream->openStream(
			(const char *)pAttrItem->getAttrDataPtr(),
			pAttrItem->getAttrDataLength())))
		{
			goto Exit;
		}
	}
	
	if( !pStackStream)
	{
		pNodeBufferIStream->m_pCachedNode = this;
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		incrNodeUseCount();
		incrStreamUseCount();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}
	
	if( puiDataType)
	{
		*puiDataType = pAttrItem->m_uiDataType;
	}

	if( puiDataLength)
	{
		*puiDataLength = (FLMUINT)pNodeBufferIStream->remainingSize();
	}
	
	if( *ppIStream)
	{
		(*ppIStream)->Release();
	}
	
	*ppIStream = pNodeBufferIStream;
	pNodeBufferIStream = NULL;

Exit:
	
	if( pNodeBufferIStream)
	{
		pNodeBufferIStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_AttrItem::resizePayloadBuffer(
	FLMUINT			uiTotalNeeded,
	FLMBOOL			bMutexAlreadyLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiCurrentSize = m_uiPayloadLen;
	FLMBOOL		bMutexLocked = FALSE;
	
	if( uiCurrentSize != uiTotalNeeded)
	{
		FLMUINT	uiNewSize;
		FLMUINT	uiSize;
		FLMUINT	uiOldSize;
		
		if (!bMutexAlreadyLocked)
		{
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
			bMutexLocked = TRUE;
		}

		uiOldSize = memSize();

		if( uiTotalNeeded <= sizeof( FLMBYTE *))
		{
			if( uiCurrentSize && uiCurrentSize > sizeof( FLMBYTE *))
			{
				m_pucPayload -= sizeof( F_AttrItem *);
				gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->freeBuf( m_uiPayloadLen + 
					sizeof( F_AttrItem *), &m_pucPayload);
			}
			else
			{
				// NOTE: Mutex is NOT locked here, because
				// nothing will be changed that requires the
				// mutex to be locked.  Nor will the size
				// change.  If the size were to change, we
				// would want to lock the mutex because we
				// would be incrementing/decrementing size
				// counts below.
				
				m_pucPayload = NULL;
			}
		}
		else
		{
			F_AttrItem *	pAttrItem = this;

			if( uiCurrentSize && uiCurrentSize > sizeof( FLMBYTE *))
			{
				m_pucPayload -= sizeof( F_AttrItem *);
				if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->reallocBuf(
					&gv_XFlmSysData.pNodeCacheMgr->m_attrBufferRelocator,
					m_uiPayloadLen + sizeof( F_AttrItem *), 
					uiTotalNeeded + sizeof( F_AttrItem *), 
					&pAttrItem, sizeof( F_AttrItem *), &m_pucPayload)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->m_pBufAllocator->allocBuf(
					&gv_XFlmSysData.pNodeCacheMgr->m_attrBufferRelocator,
					uiTotalNeeded + sizeof( F_AttrItem *),
					&pAttrItem, sizeof( F_AttrItem *), &m_pucPayload)))
				{
					goto Exit;
				}
			}

			flmAssert( *((F_AttrItem **)m_pucPayload) == this);
			m_pucPayload += sizeof( F_AttrItem *);
		}

		m_uiPayloadLen = uiTotalNeeded;
		uiNewSize = memSize();

		if( uiNewSize > uiOldSize)
		{
			uiSize = uiNewSize - uiOldSize;
			m_pCachedNode->m_uiTotalAttrSize += uiSize;

			if (m_pCachedNode->m_ui64HighTransId != FLM_MAX_UINT64)
			{
				gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes += uiSize;
			}

			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount += uiSize;
		}
		else if (uiNewSize < uiOldSize)
		{
			uiSize = uiOldSize - uiNewSize;
			
			flmAssert( m_pCachedNode->m_uiTotalAttrSize >= uiSize);
			m_pCachedNode->m_uiTotalAttrSize -= uiSize;

			if (m_pCachedNode->m_ui64HighTransId != FLM_MAX_UINT64)
			{
				flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes >= uiSize);
				gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiOldVerBytes -= uiSize;
			}

			flmAssert( gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount >= uiSize);
			gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiByteCount -= uiSize;
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_AttrItem::setupAttribute(
	F_Db *			pDb,
	FLMUINT			uiEncDefId,
	FLMUINT			uiDataSizeNeeded,
	FLMBOOL			bOkToGenerateIV,
	FLMBOOL			bMutexAlreadyLocked)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiTotalNeeded = uiDataSizeNeeded;
	FLMUINT			uiIVLen = 0;
	FLMBOOL			bGenerateIV = FALSE;
	F_ENCDEF *		pEncDef = NULL;

	if( uiEncDefId)
	{
		if( RC_BAD( rc = pDb->m_pDict->getEncDef( 
			uiEncDefId, &pEncDef)))
		{
			goto Exit;
		}
		
		uiIVLen = pEncDef->pCcs->getIVLen();
		flmAssert( uiIVLen == 8 || uiIVLen == 16);
		m_uiEncDefId = uiEncDefId;
		m_uiIVLen = uiIVLen;

		if( bOkToGenerateIV)
		{
			bGenerateIV = TRUE;
		}

		uiTotalNeeded += m_uiIVLen + 
									(getEncLen( uiDataSizeNeeded) - uiDataSizeNeeded);
	}
	else
	{
		m_uiEncDefId = 0;
		m_uiIVLen = 0;
	}

#ifdef FLM_DEBUG
	if( uiEncDefId)
	{
		flmAssert( uiTotalNeeded >= 8 + uiDataSizeNeeded);
	}
#endif

	if( RC_BAD( rc = resizePayloadBuffer( uiTotalNeeded, bMutexAlreadyLocked)))
	{
		goto Exit;
	}

	if( bGenerateIV)
	{
		if( RC_BAD( rc = pEncDef->pCcs->generateIV( uiIVLen, getAttrIVPtr())))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLAPI F_NodeBufferIStream::openStream(
	const char *		pucBuffer,
	FLMUINT				uiLength,
	char **				ppucAllocatedBuffer)
{
	RCODE						rc = NE_XFLM_OK;
	IF_BufferIStream *	pBufferIStream = NULL;
	
	if( RC_BAD( rc = FlmAllocBufferIStream( &pBufferIStream)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBufferIStream->openStream( pucBuffer, 
		uiLength, ppucAllocatedBuffer)))
	{
		goto Exit;
	}
	
	m_pBufferIStream = pBufferIStream;
	pBufferIStream = NULL;
	
Exit:

	if( pBufferIStream)
	{
		pBufferIStream->Release();
	}
	
	return( rc);
}
