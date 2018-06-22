//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DOMNode class
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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

#include "xflaim.h"

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC void XFLAPI xflaim_DOMNode_Release(
	IF_DOMNode *	pThisNode)
{
	if (pThisNode)
	{
		pThisNode->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_createNode(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NodeType,
	FLMUINT32		ui32NameId,
	FLMUINT32		ui32InsertLoc,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->createNode( pDb, (eDomNodeType)ui32NodeType, (FLMUINT)ui32NameId,
								(eNodeInsertLoc)ui32InsertLoc, ppNode, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_createChildElement(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32ChildElementNameId,
	FLMBOOL			bFirstChild,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->createChildElement( pDb, (FLMUINT)ui32ChildElementNameId,
								(eNodeInsertLoc)(bFirstChild ? XFLM_FIRST_CHILD : XFLM_LAST_CHILD),
								ppNode, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_deleteNode(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb)
{
	return( pThisNode->deleteNode( pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_deleteChildren(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb)
{
	return( pThisNode->deleteChildren( pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC FLMUINT32 XFLAPI xflaim_DOMNode_getNodeType(
	IF_DOMNode *	pThisNode)
{
	return( (FLMUINT32)pThisNode->getNodeType());
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_isDataLocalToNode(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbLocal)
{
	return( pThisNode->isDataLocalToNode( pDb, pbLocal));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_createAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->createAttribute( pDb, (FLMUINT)ui32AttrNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getFirstAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getFirstAttribute( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLastAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getLastAttribute( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getAttribute( pDb, (FLMUINT)ui32AttrNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_deleteAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId)
{
	return( pThisNode->deleteAttribute( pDb, (FLMUINT)ui32AttrNameId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasAttribute(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMBOOL *		pbHasAttr)
{
	RCODE		rc;

	rc = pThisNode->hasAttribute( pDb, (FLMUINT)ui32AttrNameId, NULL);

	if (RC_OK( rc))
	{
		*pbHasAttr = TRUE;
	}
	else if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		*pbHasAttr = FALSE;
		rc = NE_XFLM_OK;
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasAttributes(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbHasAttrs)
{
	return( pThisNode->hasAttributes( pDb, pbHasAttrs));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasNextSibling(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbHasNextSibling)
{
	return( pThisNode->hasNextSibling( pDb, pbHasNextSibling));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasPreviousSibling(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbHasPreviousSibling)
{
	return( pThisNode->hasPreviousSibling( pDb, pbHasPreviousSibling));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasChildren(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbHasChildren)
{
	return( pThisNode->hasChildren( pDb, pbHasChildren));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_isNamespaceDecl(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbIsNamespaceDecl)
{
	return( pThisNode->isNamespaceDecl( pDb, pbIsNamespaceDecl));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getParentId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64ParentId)
{
	return( pThisNode->getParentId( pDb, pui64ParentId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNodeId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64NodeId)
{
	return( pThisNode->getNodeId( pDb, pui64NodeId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getDocumentId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64DocumentId)
{
	return( pThisNode->getDocumentId( pDb, pui64DocumentId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPrevSibId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64PrevSibId)
{
	return( pThisNode->getPrevSibId( pDb, pui64PrevSibId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNextSibId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64NextSibId)
{
	return( pThisNode->getNextSibId( pDb, pui64NextSibId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getFirstChildId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64FirstChildId)
{
	return( pThisNode->getFirstChildId( pDb, pui64FirstChildId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLastChildId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64LastChildId)
{
	return( pThisNode->getLastChildId( pDb, pui64LastChildId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNameId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32NameId)
{
	RCODE		rc;
	FLMUINT	uiNameId;

	rc = pThisNode->getNameId( pDb, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setULong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64		ui64Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setUINT64( pDb, ui64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueULong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMUINT64		ui64Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
			ui64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setLong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMINT64			i64Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setINT64( pDb, i64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueLong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMINT64			i64Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
			i64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setUInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setUINT( pDb, (FLMUINT)ui32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueUInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMUINT32		ui32Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
			(FLMUINT)ui32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMINT32			i32Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setINT( pDb, (FLMINT)i32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMINT32			i32Value,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
			(FLMINT)i32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setString(
	IF_DOMNode *			pThisNode,
	IF_Db *			pDb,
	const FLMUNICODE *	puzValue,
	FLMBOOL					bLast,
	FLMUINT32				ui32EncId)
{
	return( pThisNode->setUnicode( pDb, puzValue, 0, bLast,
							(FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueString(
	IF_DOMNode *			pThisNode,
	IF_Db *			pDb,
	FLMUINT32				ui32AttrNameId,
	const FLMUNICODE *	puzValue,
	FLMUINT32				ui32EncId)
{
	return( pThisNode->setAttributeValueUnicode( pDb, (FLMUINT)ui32AttrNameId,
							puzValue, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setBinary(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	const void *	pvValue,
	FLMUINT32		ui32Len,
	FLMBOOL			bLast,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setBinary( pDb, pvValue, (FLMUINT)ui32Len, bLast,
							(FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setAttributeValueBinary(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	const void *	pvValue,
	FLMUINT32		ui32Len,
	FLMUINT32		ui32EncId)
{
	return( pThisNode->setAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId,
							pvValue, (FLMUINT)ui32Len, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getDataLength(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32DataLength)
{
	RCODE		rc;
	FLMUINT	uiDataLength;

	rc = pThisNode->getDataLength( pDb, &uiDataLength);
	*pui32DataLength = (FLMUINT32)uiDataLength;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getDataType(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32DataType)
{
	RCODE		rc;
	FLMUINT	uiDataType;

	rc = pThisNode->getDataType( pDb, &uiDataType);
	*pui32DataType = (FLMUINT32)uiDataType;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getULong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64Value)
{
	return( pThisNode->getUINT64( pDb, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueULong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMBOOL			bDefaultOk,
	FLMUINT64		ui64DefaultToUse,
	FLMUINT64 *		pui64Value)
{
	RCODE		rc;
	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
								pui64Value, ui64DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
								pui64Value);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMINT64 *		pi64Value)
{
	return( pThisNode->getINT64( pDb, pi64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueLong(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMBOOL			bDefaultOk,
	FLMINT64			i64DefaultToUse,
	FLMINT64 *		pi64Value)
{
	RCODE		rc;

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
								pi64Value, i64DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
								pi64Value);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getUInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32Value)
{
	RCODE		rc;
	FLMUINT	uiValue;

	rc = pThisNode->getUINT( pDb, &uiValue);
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueUInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMBOOL			bDefaultOk,
	FLMUINT32		ui32DefaultToUse,
	FLMUINT32 *		pui32Value)
{
	RCODE		rc;
	FLMUINT	uiValue;

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
								&uiValue, (FLMUINT)ui32DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
								&uiValue);
	}
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMINT32 *		pi32Value)
{
	RCODE		rc;
	FLMINT	iValue;

	rc = pThisNode->getINT( pDb, &iValue);
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueInt(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMBOOL			bDefaultOk,
	FLMINT32			i32DefaultToUse,
	FLMINT32 *		pi32Value)
{
	RCODE		rc;
	FLMINT	iValue;

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
								&iValue, (FLMINT)i32DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
								&iValue);
	}
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getString(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32StartPos,
	FLMUINT32		ui32NumChars,
	FLMUNICODE **	ppuzValue)
{
	RCODE		rc;
	FLMUINT	uiNumChars;
	FLMUINT	uiBufSize;

	*ppuzValue = NULL;
	if (RC_BAD( rc = pThisNode->getUnicodeChars( pDb, &uiNumChars)))
	{
		goto Exit;
	}
	if ((FLMUINT)ui32StartPos >= uiNumChars)
	{
		if (RC_BAD( rc = f_alloc( sizeof( FLMUNICODE), ppuzValue)))
		{
			goto Exit;	
		}
		(*ppuzValue) [0] = 0;
		goto Exit;
	}
	uiNumChars -= (FLMUINT)ui32StartPos;
	if (ui32NumChars && (FLMUINT)ui32NumChars < uiNumChars)
	{
		uiNumChars = (FLMUINT)ui32NumChars;
	}

	uiBufSize = (uiNumChars + 1) * sizeof( FLMUNICODE);
	if (RC_BAD( rc = f_alloc( uiBufSize, ppuzValue)))
	{
		goto Exit;	
	}

	if (RC_BAD( rc = pThisNode->getUnicode( pDb, *ppuzValue, uiBufSize,
											(FLMUINT)ui32StartPos, uiNumChars, NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueString(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMUNICODE **	ppuzValue)
{
	return( pThisNode->getAttributeValueUnicode( pDb, (FLMUINT)ui32AttrNameId,
									ppuzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getStringLen(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32NumChars)
{
	RCODE		rc;
	FLMUINT	uiNumChars;

	rc = pThisNode->getUnicodeChars( pDb, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getBinary(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32StartPos,
	FLMUINT32		ui32NumBytes,
	void *			pvValue)
{
	return( pThisNode->getBinary( pDb, pvValue, (FLMUINT)ui32StartPos,
											(FLMUINT)ui32NumBytes, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueDataLength(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMUINT32 *		pui32DataLen)
{
	RCODE		rc;
	FLMUINT	uiDataLen;

	rc = pThisNode->getAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId,
												NULL, 0, &uiDataLen);
	*pui32DataLen = (FLMUINT32)uiDataLen;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAttributeValueBinary(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32AttrNameId,
	FLMUINT32		ui32Len,
	void *			pvValue)
{
	return( pThisNode->getAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId, pvValue,
								(FLMUINT)ui32Len, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getDocumentNode(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getDocumentNode( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getParentNode(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getParentNode( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getFirstChild(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getFirstChild( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLastChild(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getLastChild( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getChild(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NodeType,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getChild( pDb, (eDomNodeType)ui32NodeType, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getChildElement(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32ElementNameId,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getChildElement( pDb, (FLMUINT)ui32ElementNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getSiblingElement(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32ElementNameId,
	FLMBOOL			bNext,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getSiblingElement( pDb, (FLMUINT)ui32ElementNameId,
							bNext, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAncestorElement(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32ElementNameId,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getAncestorElement( pDb, (FLMUINT)ui32ElementNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getDescendantElement(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32ElementNameId,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getDescendantElement( pDb, (FLMUINT)ui32ElementNameId, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPreviousSibling(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getPreviousSibling( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNextSibling(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getNextSibling( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPreviousDocument(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getPreviousDocument( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNextDocument(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getNextDocument( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPrefixChars(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *	pui32NumChars)
{
	RCODE		rc;
	FLMUINT	uiNumChars;

	rc = pThisNode->getPrefix( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPrefix(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzPrefix)
{
	FLMUINT	uiNumChars;

	return( pThisNode->getPrefix( pDb, puzPrefix, (FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE),
											&uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getPrefixId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32PrefixId)
{
	RCODE		rc;
	FLMUINT	uiPrefixId;

	rc = pThisNode->getPrefixId( pDb, &uiPrefixId);
	*pui32PrefixId = (FLMUINT32)uiPrefixId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getEncDefId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32EncDefId)
{
	RCODE		rc;
	FLMUINT	uiEncDefId;

	rc = pThisNode->getEncDefId( pDb, &uiEncDefId);
	*pui32EncDefId = (FLMUINT32)uiEncDefId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setPrefix(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	const FLMUNICODE *	puzPrefix)
{
	return( pThisNode->setPrefix( pDb, puzPrefix));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setPrefixId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32PrefixId)
{
	return( pThisNode->setPrefixId( pDb, (FLMUINT)ui32PrefixId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNamespaceURIChars(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32NumChars)
{
	RCODE		rc;
	FLMUINT	uiNumChars;

	rc = pThisNode->getNamespaceURI( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getNamespaceURI(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzNamespaceURI)
{
	FLMUINT	uiNumChars;

	return( pThisNode->getNamespaceURI( pDb, puzNamespaceURI,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLocalNameChars(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32NumChars)
{
	RCODE		rc;
	FLMUINT	uiNumChars;

	rc = pThisNode->getLocalName( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getLocalName(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzLocalName)
{
	FLMUINT	uiNumChars;

	return( pThisNode->getLocalName( pDb, puzLocalName,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getQualifiedNameChars(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32NumChars)
{
	RCODE		rc;
	FLMUINT	uiNumChars;

	rc = pThisNode->getQualifiedName( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getQualifiedName(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzQualifiedName)
{
	FLMUINT	uiNumChars;

	return( pThisNode->getQualifiedName( pDb, puzQualifiedName,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getCollection(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT32 *		pui32Collection)
{
	RCODE		rc;
	FLMUINT	uiCollection;

	rc = pThisNode->getCollection( pDb, &uiCollection);
	*pui32Collection = (FLMUINT32)uiCollection;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_createAnnotation(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->createAnnotation( pDb, ppNode, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAnnotation(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	IF_DOMNode **	ppNode)
{
	return( pThisNode->getAnnotation( pDb, ppNode));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getAnnotationId(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64AnnotationId)
{
	return( pThisNode->getAnnotationId( pDb, pui64AnnotationId));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_hasAnnotation(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMBOOL *		pbHasAnnotation)
{
	return( pThisNode->hasAnnotation( pDb, pbHasAnnotation));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_getMetaValue(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64 *		pui64Value)
{
	return( pThisNode->getMetaValue( pDb, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
XFLXPC RCODE XFLAPI xflaim_DOMNode_setMetaValue(
	IF_DOMNode *	pThisNode,
	IF_Db *			pDb,
	FLMUINT64		ui64Value)
{
	return( pThisNode->setMetaValue( pDb, ui64Value));
}
