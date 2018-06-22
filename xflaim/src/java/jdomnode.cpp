//------------------------------------------------------------------------------
// Desc:
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

#include "xflaim.h"
#include "xflaim_DOMNode.h"
#include "jniftk.h"

#define THIS_NODE() ((IF_DOMNode *)((FLMUINT)lThis))

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_DOMNode *	pNode = THIS_NODE();
	
	if( pNode)
	{
		pNode->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNodeType,
	jint				iNameId,
	jint				iInsertLoc,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->createNode( pDb, (eDomNodeType)iNodeType, 
			(FLMUINT)iNameId, (eNodeInsertLoc)iInsertLoc, &pNewNode, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createChildElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iChildElementNameId,
	jboolean			bFirstChild,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->createChildElement( pDb, 
			(FLMUINT)iChildElementNameId,
			(eNodeInsertLoc)(bFirstChild ? XFLM_FIRST_CHILD : XFLM_LAST_CHILD),
			&pNewNode, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	
	if (RC_BAD( rc = pNode->deleteNode( pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteChildren(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	
	if (RC_BAD( rc = pNode->deleteChildren( pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getNodeType(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_NODE()->getNodeType());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1isDataLocalToNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bLocal = FALSE;
	
	if (RC_BAD( rc = pThisNode->isDataLocalToNode( pDb, &bLocal)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( bLocal ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->createAttribute( pDb, 
		(FLMUINT)iNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if ( RC_BAD( rc = pThisNode->getFirstAttribute( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if ( RC_BAD( rc = pThisNode->getLastAttribute( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttributeId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->getAttribute( pDb, (FLMUINT)iAttributeId,
											   &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttributeId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	
	if (RC_BAD( rc = pThisNode->deleteAttribute( pDb, (FLMUINT)iAttributeId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttributeId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	jboolean			bRv = JNI_FALSE;
	
	rc = pThisNode->hasAttribute( pDb, (FLMUINT)iAttributeId, NULL);
	
	if (RC_OK( rc))
	{
		bRv = JNI_TRUE;
	}
	else if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( bRv);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAttributes(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bHasAttr = FALSE;
	
	if( RC_BAD( rc = pThisNode->hasAttributes( pDb, &bHasAttr)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( bHasAttr ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasNextSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bHasNextSib = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasNextSibling( pDb, &bHasNextSib)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasNextSib ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasPreviousSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bHasPreviousSib = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasPreviousSibling( pDb, &bHasPreviousSib)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasPreviousSib ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasChildren(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bHasChild = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasChildren( pDb, &bHasChild)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasChild ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1isNamespaceDecl(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bIsDecl = FALSE;
	
	if (RC_BAD( rc = pThisNode->isNamespaceDecl( pDb, &bIsDecl)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bIsDecl ? JNI_TRUE : JNI_FALSE);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getParentId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getParentId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNodeId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getNodeId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDocumentId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getDocumentId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}
 
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPrevSibId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getPrevSibId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextSibId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getNextSibId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstChildId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getFirstChildId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastChildId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getLastChildId( pDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiId;
	
	if (RC_BAD( rc = pThisNode->getNameId( pDb, &uiId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiId);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lValue,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);

	if (RC_BAD( rc = pThisNode->setINT64( pDb, (FLMINT64)lValue,
											(FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
	}
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setAttributeValueLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId,
	jlong				lValue,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);

	if (RC_BAD( rc = pThisNode->setAttributeValueINT64( pDb, (FLMUINT)iAttrNameId,
									(FLMINT64)lValue, (FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
	}
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jstring			sValue,
	jboolean			bLast,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	jchar *			pszValue = NULL;
	FLMUINT			uiLength = 0;

	if (sValue)
	{
		pszValue = (jchar *)pEnv->GetStringCritical( sValue, NULL);
		uiLength = (FLMUINT)pEnv->GetStringLength( sValue);
	}
	
	if (RC_BAD( rc = pThisNode->setUnicode( pDb, pszValue, uiLength,
		bLast ? TRUE : FALSE, (FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pszValue)
	{
		pEnv->ReleaseStringCritical( sValue, pszValue);
	}
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setAttributeValueString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId,
	jstring			sValue,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBYTE			ucValue [256];
	F_DynaBuf		valueBuf( ucValue, sizeof( ucValue));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sValue, &valueBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pThisNode->setAttributeValueUTF8( pDb, (FLMUINT)iAttrNameId,
		(const FLMBYTE *)valueBuf.getBufferPtr(), 0, (FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jbyteArray		Value,
	jboolean			bLast,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiLength = pEnv->GetArrayLength( Value);
	void *			pvValue = NULL;
	jboolean			bIsCopy = JNI_FALSE;
	
	if( (pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if( RC_BAD( rc = pThisNode->setBinary(pDb, pvValue, uiLength,
								bLast ? TRUE : FALSE,
								(FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvValue)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setAttributeValueBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId,
	jbyteArray		Value,
	jint				iEncId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiLength = pEnv->GetArrayLength( Value);
	void *			pvValue = NULL;
	jboolean			bIsCopy = JNI_FALSE;
	
	if( (pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if( RC_BAD( rc = pThisNode->setAttributeValueBinary(pDb, (FLMUINT)iAttrNameId,
								pvValue, uiLength, (FLMUINT)iEncId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvValue)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDataLength(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiLength;
	
	if (RC_BAD( rc = pThisNode->getDataLength( pDb, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)uiLength);
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getDataType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiType;
	
	if (RC_BAD( rc = pThisNode->getDataType( pDb, &uiType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiType);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pThisNode = THIS_NODE();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMINT64				i64Val;
	
	if (RC_BAD( rc = pThisNode->getINT64( pDb, &i64Val)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)i64Val);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAttributeValueLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId,
	jboolean			bDefaultOk,
	jlong				lDefaultToUse)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pThisNode = THIS_NODE();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMINT64				i64Val;
	
	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)iAttrNameId,
								&i64Val, (FLMINT64)lDefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)iAttrNameId,
								&i64Val);
	}
	
	if (RC_BAD( rc))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)i64Val);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iStartPos,
	jint				iNumChars)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUNICODE		uzBuffer[ 128];
	FLMUNICODE *	puzBuf = uzBuffer;
	FLMUINT			uiBufSize = sizeof(uzBuffer);
	FLMUINT			uiNumChars;
	jstring			sBuf = NULL;
	
	if (RC_BAD( rc = pThisNode->getUnicodeChars( pDb, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if ((FLMUINT)iStartPos >= uiNumChars)
	{
		goto Exit;
	}
	uiNumChars -= (FLMUINT)iStartPos;
	if (iNumChars && (FLMUINT)iNumChars < uiNumChars)
	{
		uiNumChars = (FLMUINT)iNumChars;
	}

	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, &puzBuf)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}

	if (RC_BAD( rc = pThisNode->getUnicode( pDb, puzBuf, uiBufSize,
											(FLMUINT)iStartPos, uiNumChars, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sBuf = pEnv->NewString( puzBuf, (jsize)uiNumChars);	
	
Exit:

	if (puzBuf != uzBuffer)
	{
		f_free( &puzBuf);
	}
	
	return( sBuf);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getAttributeValueString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBYTE			ucBuf [256];
	F_DynaBuf		dynaBuf( ucBuf, sizeof( ucBuf));
	jstring			sBuf = NULL;

	if (RC_BAD( rc = pThisNode->getAttributeValueUnicode( pDb,
								(FLMUINT)iAttrNameId, &dynaBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	sBuf = pEnv->NewString( dynaBuf.getUnicodePtr(),
						(jsize)dynaBuf.getUnicodeLength());
	
Exit:

	return( sBuf);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getStringLen(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiVal;
	
	if (RC_BAD(rc = pThisNode->getUnicodeChars( pDb, &uiVal)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiVal);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DOMNode__1getBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iStartPos,
	jint				iNumBytes)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiLength;
	jbyteArray		Data = NULL;
	void *			pvData = NULL;
	jboolean			bIsCopy = JNI_FALSE;
	
	if (RC_BAD(rc = pThisNode->getDataLength( pDb, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if ((FLMUINT)iStartPos >= uiLength)
	{
		goto Exit;
	}
	uiLength -= (FLMUINT)iStartPos;
	if (iNumBytes && (FLMUINT)iNumBytes < uiLength)
	{
		uiLength = (FLMUINT)iNumBytes;
	}

	Data = pEnv->NewByteArray( (jsize)uiLength);
	
	if ( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pThisNode->getBinary( pDb, pvData, (FLMUINT)iStartPos,
											uiLength, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvData)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
			pEnv->DeleteLocalRef( Data);
			Data = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, 0);
		}
	}

	return( Data);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DOMNode__1getAttributeValueBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iAttrNameId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiLength;
	jbyteArray		Data = NULL;
	void *			pvData = NULL;
	jboolean			bIsCopy = JNI_FALSE;
	
	if (RC_BAD(rc = pThisNode->getAttributeValueBinary( pDb, (FLMUINT)iAttrNameId,
												NULL, 0, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	Data = pEnv->NewByteArray( (jsize)uiLength);
	
	if ( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pThisNode->getAttributeValueBinary( pDb, (FLMUINT)iAttrNameId,
											pvData, uiLength, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvData)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
			pEnv->DeleteLocalRef( Data);
			Data = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, 0);
		}
	}

	return( Data);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDocumentNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getDocumentNode( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getParentNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getParentNode( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstChild(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getFirstChild( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastChild(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getLastChild( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getChild(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNodeType,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getChild( pDb, (eDomNodeType)iNodeType, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getChildElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getChildElement( pDb, 
		(FLMUINT)iNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getSiblingElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNameId,
	jboolean			bNext,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getSiblingElement( pDb, 
		(FLMUINT)iNameId, bNext ? TRUE : FALSE, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAncestorElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getAncestorElement( pDb, 
		(FLMUINT)iNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDescendantElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getDescendantElement( pDb,
		(FLMUINT)iNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPreviousSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getPreviousSibling( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getNextSibling( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPreviousDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getPreviousDocument( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pThisNode->getNextDocument( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getPrefix(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUNICODE		uzPrefix[ 128];
	FLMUNICODE *	puzPrefix = uzPrefix;
	FLMUINT			uiBufSize = sizeof( uzPrefix);
	FLMUINT			uiNumChars;
	jstring			sPrefix = NULL;
	
	if (RC_BAD( rc = pThisNode->getPrefix( pDb, 
		(FLMUNICODE *)NULL, 0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof( FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzPrefix)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getPrefix( pDb, puzPrefix, uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sPrefix = pEnv->NewString( puzPrefix, (jsize)uiNumChars);
	
Exit:

	if (puzPrefix != uzPrefix)
	{
		f_free( &puzPrefix);
	}	
	
	return( sPrefix);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getPrefixId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiPrefixId = 0;
	
	if (RC_BAD( rc = pThisNode->getPrefixId( pDb, &uiPrefixId))) 
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiPrefixId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getEncDefId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiEncDefId = 0;
	
	if (RC_BAD( rc = pThisNode->getEncDefId( pDb, &uiEncDefId))) 
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiEncDefId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setPrefix(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jstring			sPrefix)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBYTE			ucPrefix [256];
	F_DynaBuf		prefixBuf( ucPrefix, sizeof( ucPrefix));

	if (RC_BAD( rc = getUniString( pEnv, sPrefix, &prefixBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = pThisNode->setPrefix( pDb, prefixBuf.getUnicodePtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setPrefixId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jint				iPrefixId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);

	if (RC_BAD( rc = pThisNode->setPrefixId( pDb, (FLMUINT)iPrefixId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getNamespaceURI(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)	
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUNICODE		uzNamespaceURI[ 128];
	FLMUNICODE *	puzNamespaceURI = uzNamespaceURI;
	FLMUINT			uiBufSize = sizeof( uzNamespaceURI);
	FLMUINT			uiNumChars;
	jstring			sNamespaceURI = NULL;
	
	if (RC_BAD( rc = pThisNode->getNamespaceURI( pDb, 
		(FLMUNICODE *)NULL, 0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzNamespaceURI)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getNamespaceURI( pDb, puzNamespaceURI,
						uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sNamespaceURI = pEnv->NewString( puzNamespaceURI, (jsize)uiNumChars);
	
Exit:

	if (puzNamespaceURI != uzNamespaceURI)
	{
		f_free( &puzNamespaceURI);
	}	
	
	return( sNamespaceURI);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getLocalName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUNICODE		uzLocalName[ 128];
	FLMUNICODE *	puzLocalName = uzLocalName;
	FLMUINT			uiBufSize = sizeof(uzLocalName);
	FLMUINT			uiNumChars;
	jstring			sLocalName = NULL;
	
	if (RC_BAD( rc = pThisNode->getLocalName( pDb, (FLMUNICODE *)NULL, 
		0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}

	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzLocalName)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getLocalName( pDb, puzLocalName,
		uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sLocalName = pEnv->NewString( puzLocalName, (jsize)uiNumChars);
	
Exit:

	if (puzLocalName != uzLocalName)
	{
		f_free( &puzLocalName);
	}	
	
	return( sLocalName);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getQualifiedName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUNICODE		uzQualName[ 128];
	FLMUNICODE *	puzQualName = uzQualName;
	FLMUINT			uiNumChars;
	FLMUINT			uiBufSize = sizeof( uzQualName);
	jstring			sLocalName = NULL;
	
	if (RC_BAD( rc = pThisNode->getQualifiedName( pDb, (FLMUNICODE *)NULL,
		0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize =  (uiNumChars + 1)* sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzQualName)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getQualifiedName( pDb, puzQualName,
		uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sLocalName = pEnv->NewString( puzQualName, (jsize)uiNumChars);
	
Exit:

	if (puzQualName != uzQualName)
	{
		f_free( &puzQualName);
	}	
	
	return( sLocalName);				
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getCollection(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiColNum;
	
	if (RC_BAD( rc = pThisNode->getCollection( pDb, &uiColNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiColNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createAnnotation(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->createAnnotation( pDb, &pNewNode, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAnnotation(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if( RC_BAD( rc = pThisNode->getAnnotation( pDb, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAnnotationId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64AnnotationId;
	
	if( RC_BAD( rc = pThisNode->getAnnotationId( pDb, &ui64AnnotationId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64AnnotationId);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAnnotation(
 	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBOOL			bHasAnnotation;
	
	if( RC_BAD( rc = pThisNode->hasAnnotation( pDb, &bHasAnnotation)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (bHasAnnotation ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getMetaValue(
 	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT64		ui64Value;
	
	if( RC_BAD( rc = pThisNode->getMetaValue( pDb, &ui64Value)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Value);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setMetaValue(
 	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDbRef,
	jlong				lValue)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	
	if( RC_BAD( rc = pThisNode->setMetaValue( pDb, (FLMUINT64)lValue)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

