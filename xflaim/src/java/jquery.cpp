//------------------------------------------------------------------------------
// Desc:
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

#include "xflaim_Query.h"
#include "xflaim_OptInfo.h"
#include "xflaim_ResultSetCounts.h"
#include "flaimsys.h"
#include "jniftk.h"

#define THIS_QUERY() ((F_Query *)((FLMUINT)lThis))

// Fields in the OptInfo class
	
static jfieldID	fid_OptInfo_iOptType = NULL;
static jfieldID	fid_OptInfo_iCost = NULL;
static jfieldID	fid_OptInfo_lNodeId = NULL;
static jfieldID	fid_OptInfo_lEndNodeId = NULL;
static jfieldID	fid_OptInfo_iIxNum = NULL;
static jfieldID	fid_OptInfo_sIxName = NULL;
static jfieldID	fid_OptInfo_bMustVerifyPath = NULL;
static jfieldID	fid_OptInfo_bDoNodeMatch = NULL;
static jfieldID	fid_OptInfo_bCanCompareOnKey = NULL;
static jfieldID	fid_OptInfo_lKeysRead = NULL;
static jfieldID	fid_OptInfo_lKeyHadDupDoc = NULL;
static jfieldID	fid_OptInfo_lKeysPassed = NULL;
static jfieldID	fid_OptInfo_lNodesRead = NULL;
static jfieldID	fid_OptInfo_lNodesTested = NULL;
static jfieldID	fid_OptInfo_lNodesPassed = NULL;
static jfieldID	fid_OptInfo_lDocsRead = NULL;
static jfieldID	fid_OptInfo_lDupDocsEliminated = NULL;
static jfieldID	fid_OptInfo_lNodesFailedValidation = NULL;
static jfieldID	fid_OptInfo_lDocsFailedValidation = NULL;
static jfieldID	fid_OptInfo_lDocsPassed = NULL;

// Fields in the ResultSetCounts class
	
static jfieldID	fid_ResultSetCounts_iReadCount = NULL;
static jfieldID	fid_ResultSetCounts_iPassedCount = NULL;
static jfieldID	fid_ResultSetCounts_iPositionableToCount = NULL;
static jfieldID	fid_ResultSetCounts_bDoneBuildingResultSet = NULL;
	
FSTATIC jobject NewOptInfo(
	JNIEnv *				pEnv,
	XFLM_OPT_INFO *	pOptInfo,
	jclass				jOptInfoClass);
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_OptInfo_initIDs(
	JNIEnv *	pEnv,
	jclass	jOptInfoClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_OptInfo_iOptType = pEnv->GetFieldID( jOptInfoClass,
								"iOptType", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_iCost = pEnv->GetFieldID( jOptInfoClass,
								"iCost", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lNodeId = pEnv->GetFieldID( jOptInfoClass,
								"lNodeId", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lEndNodeId = pEnv->GetFieldID( jOptInfoClass,
								"lEndNodeId", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_iIxNum = pEnv->GetFieldID( jOptInfoClass,
								"iIxNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_sIxName = pEnv->GetFieldID( jOptInfoClass,
								"sIxName", "Ljava/lang/String;")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_bMustVerifyPath = pEnv->GetFieldID( jOptInfoClass,
								"bMustVerifyPath", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_bDoNodeMatch = pEnv->GetFieldID( jOptInfoClass,
								"bDoNodeMatch", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_bCanCompareOnKey = pEnv->GetFieldID( jOptInfoClass,
								"bCanCompareOnKey", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lKeysRead = pEnv->GetFieldID( jOptInfoClass,
								"lKeysRead", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lKeyHadDupDoc = pEnv->GetFieldID( jOptInfoClass,
								"lKeyHadDupDoc", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lKeysPassed = pEnv->GetFieldID( jOptInfoClass,
								"lKeysPassed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lNodesRead = pEnv->GetFieldID( jOptInfoClass,
								"lNodesRead", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lNodesTested = pEnv->GetFieldID( jOptInfoClass,
								"lNodesTested", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lNodesPassed = pEnv->GetFieldID( jOptInfoClass,
								"lNodesPassed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lDocsRead = pEnv->GetFieldID( jOptInfoClass,
								"lDocsRead", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lDupDocsEliminated = pEnv->GetFieldID( jOptInfoClass,
								"lDupDocsEliminated", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lNodesFailedValidation = pEnv->GetFieldID( jOptInfoClass,
								"lNodesFailedValidation", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lDocsFailedValidation = pEnv->GetFieldID( jOptInfoClass,
								"lDocsFailedValidation", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_OptInfo_lDocsPassed = pEnv->GetFieldID( jOptInfoClass,
								"lDocsPassed", "J")) == NULL)
	{
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_ResultSetCounts_initIDs(
	JNIEnv *	pEnv,
	jclass	jResultSetCountsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_ResultSetCounts_iReadCount = pEnv->GetFieldID( jResultSetCountsClass,
								"iReadCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ResultSetCounts_iPassedCount = pEnv->GetFieldID( jResultSetCountsClass,
								"iPassedCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ResultSetCounts_iPositionableToCount = pEnv->GetFieldID( jResultSetCountsClass,
								"iPositionableToCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ResultSetCounts_bDoneBuildingResultSet = pEnv->GetFieldID( jResultSetCountsClass,
								"bDoneBuildingResultSet", "Z")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Query *	pQuery = THIS_QUERY();
	
	if (pQuery)
	{
		pQuery->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Query__1createQuery(
	JNIEnv *				pEnv,
	jobject,				// obj
	jint					iCollection)
{
	RCODE			rc = NE_XFLM_OK;
	F_Query *	pQuery;

	if ((pQuery = f_new F_Query) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	pQuery->setCollection( (FLMUINT)iCollection);
	
Exit:
	
	return( (jlong)((FLMUINT)pQuery));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1setLanguage(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iLanguage)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();

	if (RC_BAD( rc = pQuery->setLanguage( (FLMUINT)iLanguage)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1setupQueryExpr(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jstring				sQuery)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	IF_Db *		pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBYTE		ucQuery [512];
	F_DynaBuf	queryBuf( ucQuery, sizeof( ucQuery));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sQuery, &queryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pQuery->setupQueryExpr( pDb, (const char *)queryBuf.getBufferPtr())))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1copyCriteria(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lQueryToCopy)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	IF_Query *	pQueryToCopy = (IF_Query *)((FLMUINT)lQueryToCopy);
	
	if (RC_BAD( rc = pQuery->copyCriteria( pQueryToCopy)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addXPathComponent(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iXPathAxis,
	jint					iNodeType,
	jint					iNameId)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addXPathComponent( (eXPathAxisTypes)iXPathAxis,
										(eDomNodeType)iNodeType, (FLMUINT)iNameId,
										NULL)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addOperator(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iOperator,
	jint					iCompareRules)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addOperator( (eQueryOperators)iOperator,
										(FLMUINT)iCompareRules, NULL)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addStringValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jstring				sValue)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	FLMBYTE		ucValue [256];
	F_DynaBuf	valueBuf( ucValue, sizeof( ucValue));
	
	if (RC_BAD( rc = getUniString( pEnv, sValue, &valueBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pQuery->addUnicodeValue( valueBuf.getUnicodePtr())))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addBinaryValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jbyteArray			Value)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	FLMUINT		uiLength = pEnv->GetArrayLength( Value);
	void *		pvValue = NULL;
	jboolean		bIsCopy = JNI_FALSE;
	
	if ((pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pQuery->addBinaryValue( pvValue, uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvValue)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
	
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1addLongValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lValue)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addINT64Value( (FLMINT64)lValue)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addBoolean(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jboolean				bValue,
	jboolean				bUnknown)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addBoolean( bValue ? TRUE : FALSE,
													 bUnknown ? TRUE : FALSE)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1getFirst(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->getFirst( pDb, &pNewNode, (FLMUINT)iTimeLimit)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1getLast(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->getLast( pDb, &pNewNode, (FLMUINT)iTimeLimit)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1getNext(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit,
	jint					iNumToSkip)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->getNext( pDb, &pNewNode, (FLMUINT)iTimeLimit,
								(FLMUINT)iNumToSkip, NULL)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1getPrev(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit,
	jint					iNumToSkip)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->getPrev( pDb, &pNewNode, (FLMUINT)iTimeLimit,
								(FLMUINT)iNumToSkip, NULL)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1getCurrent(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->getCurrent( pDb, &pNewNode)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1resetQuery(
	JNIEnv *,			// pEnv,
	jobject,				// obj
	jlong					lThis)
{
	THIS_QUERY()->resetQuery();
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC jobject NewOptInfo(
	JNIEnv *				pEnv,
	XFLM_OPT_INFO *	pOptInfo,
	jclass				jOptInfoClass)
{
	jobject	jOptInfo = NULL;
	jstring	jIxName;
	
	if ((jIxName = pEnv->NewStringUTF( (const char *)pOptInfo->szIxName)) == NULL)
	{
		goto Exit;
	}
	
	// Allocate and populate the opt info object
	
	if ((jOptInfo = pEnv->AllocObject( jOptInfoClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetIntField( jOptInfo, fid_OptInfo_iOptType,
		(jint)pOptInfo->eOptType);
	pEnv->SetIntField( jOptInfo, fid_OptInfo_iCost,
		(jint)pOptInfo->uiCost);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lNodeId,
		(jlong)pOptInfo->ui64NodeId);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lEndNodeId,
		(jlong)pOptInfo->ui64EndNodeId);
	pEnv->SetIntField( jOptInfo, fid_OptInfo_iIxNum,
		(jint)pOptInfo->uiIxNum);
	pEnv->SetObjectField( jOptInfo, fid_OptInfo_sIxName,
		jIxName);
	pEnv->SetBooleanField( jOptInfo, fid_OptInfo_bMustVerifyPath,
		(jboolean)(pOptInfo->bMustVerifyPath ? JNI_TRUE : JNI_FALSE));
	pEnv->SetBooleanField( jOptInfo, fid_OptInfo_bDoNodeMatch,
		(jboolean)(pOptInfo->bDoNodeMatch ? JNI_TRUE : JNI_FALSE));
	pEnv->SetBooleanField( jOptInfo, fid_OptInfo_bCanCompareOnKey,
		(jboolean)(pOptInfo->bCanCompareOnKey ? JNI_TRUE : JNI_FALSE));
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lKeysRead,
		(jlong)pOptInfo->ui64KeysRead);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lKeyHadDupDoc,
		(jlong)pOptInfo->ui64KeyHadDupDoc);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lKeysPassed,
		(jlong)pOptInfo->ui64KeysPassed);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lNodesRead,
		(jlong)pOptInfo->ui64NodesRead);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lNodesTested,
		(jlong)pOptInfo->ui64NodesTested);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lNodesPassed,
		(jlong)pOptInfo->ui64NodesPassed);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lDocsRead,
		(jlong)pOptInfo->ui64DocsRead);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lDupDocsEliminated,
		(jlong)pOptInfo->ui64DupDocsEliminated);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lNodesFailedValidation,
		(jlong)pOptInfo->ui64NodesFailedValidation);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lDocsFailedValidation,
		(jlong)pOptInfo->ui64DocsFailedValidation);
	pEnv->SetLongField( jOptInfo, fid_OptInfo_lDocsPassed,
		(jlong)pOptInfo->ui64DocsPassed);

Exit:

	return( jOptInfo);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobjectArray JNICALL Java_xflaim_Query__1getStatsAndOptInfo(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = THIS_QUERY();
	XFLM_OPT_INFO *	pOptInfoArray = NULL;
	FLMUINT				uiNumOptInfos = 0;
	jclass				jOptInfoClass = NULL;
	jobjectArray		jOptInfoArray = NULL;
	
	if (RC_BAD( rc = pQuery->getStatsAndOptInfo( &uiNumOptInfos, &pOptInfoArray)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Get the OptInfo class.
	
	if ((jOptInfoClass = pEnv->FindClass( "xflaim/OptInfo")) == NULL)
	{
		goto Exit;
	}
	
	// Allocate an array of OptInfo objects.
	
	if (uiNumOptInfos)
	{
		XFLM_OPT_INFO *	pOptInfo;
		FLMUINT				uiLoop;
		jobject				jOptInfo;
		
		if ((jOptInfoArray = pEnv->NewObjectArray( (jsize)uiNumOptInfos,
							jOptInfoClass, NULL)) == NULL)
		{
			goto Exit;
		}
		
		// Populate the OptInfo array
		
		for (uiLoop = 0, pOptInfo = pOptInfoArray;
			  uiLoop < uiNumOptInfos;
			  uiLoop++, pOptInfo++)
		{
			
			// Allocate an opt info object.
			
			if ((jOptInfo = NewOptInfo( pEnv, pOptInfo, jOptInfoClass)) == NULL)
			{
				goto Exit;
			}
			
			// Put the opt info object into the array of
			// opt info objects.
			
			pEnv->SetObjectArrayElement( jOptInfoArray, (jsize)uiLoop, jOptInfo);
		}
	}
	
Exit:

	if (pOptInfoArray)
	{
		pQuery->freeStatsAndOptInfo( &pOptInfoArray);
	}

	return( jOptInfoArray);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1setDupHandling(
	JNIEnv *,			// pEnv,
	jobject,				// obj
	jlong					lThis,
	jboolean				bRemoveDups)
{
	THIS_QUERY()->setDupHandling( bRemoveDups ? TRUE : FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1setIndex(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iIndex)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->setIndex( (FLMUINT)iIndex)))
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
JNIEXPORT jint JNICALL Java_xflaim_Query__1getIndex(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiIndex;
	FLMBOOL			bHaveMultiple;
	
	if (RC_BAD( rc = pQuery->getIndex( pDb, &uiIndex, &bHaveMultiple)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiIndex);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Query__1usesMultipleIndexes(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT			uiIndex;
	FLMBOOL			bHaveMultiple;
	
	if (RC_BAD( rc = pQuery->getIndex( pDb, &uiIndex, &bHaveMultiple)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jboolean)(bHaveMultiple ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Query__1addSortKey(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lSortKeyContext,
	jboolean				bChildToContext,
	jboolean				bElement,
	jint					iNameId,
	jint					iCompareRules,
	jint					iLimit,
	jint					iKeyComponent,
	jboolean				bSortDescending,
	jboolean				bSortMissingHigh)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	void *			pvContext = (void *)((FLMUINT)lSortKeyContext);
	void *			pvReturnedContext = NULL;
	
	if (RC_BAD( rc = pQuery->addSortKey( pvContext,
								(FLMBOOL)(bChildToContext ? TRUE : FALSE),
								(FLMBOOL)(bElement ? TRUE : FALSE),
								(FLMUINT)iNameId,
								(FLMUINT)iCompareRules,
								(FLMUINT)iLimit,
								(FLMUINT)iKeyComponent,
								(FLMBOOL)(bSortDescending ? TRUE : FALSE),
								(FLMBOOL)(bSortMissingHigh ? TRUE : FALSE),
								&pvReturnedContext)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)pvReturnedContext));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1enablePositioning(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->enablePositioning()))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1positionTo(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit,
	jint					iPosition)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Query *		pQuery = THIS_QUERY();
	IF_Db *			pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DOMNode *	pNewNode = lReusedNodeRef
									  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
									  : NULL;
	
	if (RC_BAD( rc = pQuery->positionTo( pDb, &pNewNode, (FLMUINT)iTimeLimit,
								(FLMUINT)iPosition)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Query__1positionTo2(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jlong					lReusedNodeRef,
	jint					iTimeLimit,
	jlong					lSearchKeyRef,
	jint					iFlags)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = THIS_QUERY();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	IF_DataVector *	pSearchKey = (IF_DataVector *)((FLMUINT)lSearchKeyRef);
	IF_DOMNode *		pNewNode = lReusedNodeRef
										  ? (IF_DOMNode *)((FLMUINT)lReusedNodeRef)
										  : NULL;
	
	if (RC_BAD( rc = pQuery->positionTo( pDb, &pNewNode, (FLMUINT)iTimeLimit,
								pSearchKey, (FLMUINT)iFlags)))
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
JNIEXPORT jint JNICALL Java_xflaim_Query__1getPosition(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = THIS_QUERY();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMUINT				uiPosition = 0;
	
	if (RC_BAD( rc = pQuery->getPosition( pDb, &uiPosition)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiPosition);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1buildResultSet(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jint					iTimeLimit)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = THIS_QUERY();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	
	if (RC_BAD( rc = pQuery->buildResultSet( pDb, (FLMUINT)iTimeLimit)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1stopBuildingResultSet(
	JNIEnv *,			// pEnv,
	jobject,				// obj
	jlong					lThis)
{
	THIS_QUERY()->stopBuildingResultSet();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1enableResultSetEncryption(
	JNIEnv *,			// pEnv,
	jobject,				// obj
	jlong					lThis)
{
	THIS_QUERY()->enableResultSetEncryption();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Query__1getResultSetCounts(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jint					iTimeLimit,
	jboolean				bPartialCountOk)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = THIS_QUERY();
	IF_Db *				pDb = (IF_Db *)((FLMUINT)lDbRef);
	jclass				jResultSetCountsClass = NULL;
	jobject				jCounts = NULL;
	FLMUINT				uiReadCount;
	FLMUINT				uiPassedCount;
	FLMUINT				uiPositionableToCount;
	FLMBOOL				bDoneBuildingResultSet;
	
	if (RC_BAD( rc = pQuery->getCounts( pDb, (FLMUINT)iTimeLimit,
								(FLMBOOL)(bPartialCountOk ? TRUE : FALSE),
								&uiReadCount, &uiPassedCount,
								&uiPositionableToCount, &bDoneBuildingResultSet)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if ((jResultSetCountsClass = pEnv->FindClass( "xflaim/ResultSetCounts")) == NULL)
	{
		goto Exit;
	}
	
	if ((jCounts = pEnv->AllocObject( jResultSetCountsClass)) == NULL)
	{
		goto Exit;
	}
	pEnv->SetIntField( jCounts, fid_ResultSetCounts_iReadCount,
		(jint)uiReadCount);
	pEnv->SetIntField( jCounts, fid_ResultSetCounts_iPassedCount,
		(jint)uiPassedCount);
	pEnv->SetIntField( jCounts, fid_ResultSetCounts_iPositionableToCount,
		(jint)uiPositionableToCount);
	pEnv->SetBooleanField( jCounts, fid_ResultSetCounts_bDoneBuildingResultSet,
		(jboolean)(bDoneBuildingResultSet ? JNI_TRUE : JNI_FALSE));
	
Exit:

	return( jCounts);
}

