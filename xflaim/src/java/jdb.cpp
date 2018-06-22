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
#include "xflaim_Db.h"
#include "xflaim_IndexStatus.h"
#include "xflaim_ImportStats.h"
#include "xflaim_CheckpointInfo.h"
#include "xflaim_LockUser.h"
#include "jniftk.h"

// Field IDs for the IndexStatus class in JAVA.

static jfieldID	fid_IndexStatus_iIndexNum = NULL;
static jfieldID	fid_IndexStatus_iState = NULL;
static jfieldID	fid_IndexStatus_iStartTime = NULL;
static jfieldID	fid_IndexStatus_lLastDocumentIndexed = NULL;
static jfieldID	fid_IndexStatus_lKeysProcessed = NULL;
static jfieldID	fid_IndexStatus_lDocumentsProcessed = NULL;
static jfieldID	fid_IndexStatus_lTransactions = NULL;

// Field IDs for the ImportStats class in JAVA.

static jfieldID	fid_ImportStats_iLines = NULL;
static jfieldID	fid_ImportStats_iChars = NULL;
static jfieldID	fid_ImportStats_iAttributes = NULL;
static jfieldID	fid_ImportStats_iElements = NULL;
static jfieldID	fid_ImportStats_iText = NULL;
static jfieldID	fid_ImportStats_iDocuments = NULL;
static jfieldID	fid_ImportStats_iErrLineNum = NULL;
static jfieldID	fid_ImportStats_iErrLineOffset = NULL;
static jfieldID	fid_ImportStats_iErrorType = NULL;
static jfieldID	fid_ImportStats_iErrLineFilePos = NULL;
static jfieldID	fid_ImportStats_iErrLineBytes = NULL;
static jfieldID	fid_ImportStats_bUTF8Encoding = NULL;

// Field IDs for the CheckpointInfo class in JAVA
	
static jfieldID	fid_CheckpointInfo_bRunning = NULL;
static jfieldID	fid_CheckpointInfo_iRunningTime = NULL;
static jfieldID	fid_CheckpointInfo_bForcingCheckpoint = NULL;
static jfieldID	fid_CheckpointInfo_iForceCheckpointRunningTime = NULL;
static jfieldID	fid_CheckpointInfo_iForceCheckpointReason = NULL;
static jfieldID	fid_CheckpointInfo_bWritingDataBlocks = NULL;
static jfieldID	fid_CheckpointInfo_iLogBlocksWritten = NULL;
static jfieldID	fid_CheckpointInfo_iDataBlocksWritten = NULL;
static jfieldID	fid_CheckpointInfo_iDirtyCacheBytes = NULL;
static jfieldID	fid_CheckpointInfo_iBlockSize = NULL;
static jfieldID	fid_CheckpointInfo_iWaitTruncateTime = NULL;

// Field IDs for the LockUser class in JAVA

static jfieldID	fid_LockUser_iThreadId = NULL;
static jfieldID	fid_LockUser_iTime = NULL;

#define THIS_FDB() ((IF_Db *)(FLMUINT)lThis)

#define ELEMENT_DEF		1		// Maps to ELM_ELEMENT_TAG
#define ATTRIBUTE_DEF	2		// Maps to ELM_ATTRIBUTE_TAG
#define PREFIX_DEF		3		// Maps to ELM_PREFIX_TAG
#define ENCRYPTION_DEF	4		// Maps to ELM_ENCDEF_TAG
#define COLLECTION_DEF	5		// Maps to ELM_COLLECTION_TAG
#define INDEX_DEF			6		// Maps to ELM_INDEX_TAG

FSTATIC RCODE mapDictType(
	jint			iDictType,
	FLMUINT *	puiDictType);
	
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	jint				iDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf);
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CheckpointInfo_initIDs(
	JNIEnv *	pEnv,
	jclass	jCheckpointInfoClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_CheckpointInfo_bRunning = pEnv->GetFieldID( jCheckpointInfoClass,
								"bRunning", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iRunningTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iRunningTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_bForcingCheckpoint = pEnv->GetFieldID( jCheckpointInfoClass,
								"bForcingCheckpoint", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iForceCheckpointRunningTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iForceCheckpointRunningTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iForceCheckpointReason = pEnv->GetFieldID( jCheckpointInfoClass,
								"iForceCheckpointReason", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_bWritingDataBlocks = pEnv->GetFieldID( jCheckpointInfoClass,
							"bWritingDataBlocks", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iLogBlocksWritten = pEnv->GetFieldID( jCheckpointInfoClass,
								"iLogBlocksWritten", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iDataBlocksWritten = pEnv->GetFieldID( jCheckpointInfoClass,
								"iDataBlocksWritten", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iDirtyCacheBytes = pEnv->GetFieldID( jCheckpointInfoClass,
								"iDirtyCacheBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iBlockSize = pEnv->GetFieldID( jCheckpointInfoClass,
								"iBlockSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_CheckpointInfo_iWaitTruncateTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iWaitTruncateTime", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_IndexStatus_initIDs(
	JNIEnv *	pEnv,
	jclass	jIndexStatusClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_IndexStatus_iIndexNum = pEnv->GetFieldID( jIndexStatusClass,
								"iIndexNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_iState = pEnv->GetFieldID( jIndexStatusClass,
								"iState", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_iStartTime = pEnv->GetFieldID( jIndexStatusClass,
								"iStartTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_lLastDocumentIndexed = pEnv->GetFieldID( jIndexStatusClass,
								"lLastDocumentIndexed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_lKeysProcessed = pEnv->GetFieldID( jIndexStatusClass,
								"lKeysProcessed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_lDocumentsProcessed = pEnv->GetFieldID( jIndexStatusClass,
							"lDocumentsProcessed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fid_IndexStatus_lTransactions = pEnv->GetFieldID( jIndexStatusClass,
								"lTransactions", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_ImportStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jImportStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_ImportStats_iLines = pEnv->GetFieldID( jImportStatsClass,
								"iLines", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iChars = pEnv->GetFieldID( jImportStatsClass,
								"iChars", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iAttributes = pEnv->GetFieldID( jImportStatsClass,
								"iAttributes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iElements = pEnv->GetFieldID( jImportStatsClass,
								"iElements", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iText = pEnv->GetFieldID( jImportStatsClass,
								"iText", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iDocuments = pEnv->GetFieldID( jImportStatsClass,
							"iDocuments", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iErrLineNum = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iErrLineOffset = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineOffset", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iErrorType = pEnv->GetFieldID( jImportStatsClass,
								"iErrorType", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iErrLineFilePos = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineFilePos", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_iErrLineBytes = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_ImportStats_bUTF8Encoding = pEnv->GetFieldID( jImportStatsClass,
								"bUTF8Encoding", "Z")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_LockUser_initIDs(
	JNIEnv *	pEnv,
	jclass	jIndexStatusClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fid_LockUser_iThreadId = pEnv->GetFieldID( jIndexStatusClass,
								"iThreadId", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fid_LockUser_iTime = pEnv->GetFieldID( jIndexStatusClass,
								"iTime", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_FDB()->Release();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JIII(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iTransactionType,
	jint				iMaxLockWait,
	jint				iFlags)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->transBegin( (eDbTransType)iTransactionType,
		(FLMUINT)iMaxLockWait, (FLMUINT)iFlags)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lSrcDb)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Db *			pSrcDb = ((IF_Db *)(FLMUINT)lSrcDb);

	if (RC_BAD( rc = pDb->transBegin( pSrcDb)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1transCommit(
	JNIEnv *			pEnv,
	jobject			obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	(void)obj;
	
	if (RC_BAD( rc = pDb->transCommit()))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1transAbort(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->transAbort()))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getTransType(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *			pDb = THIS_FDB();
	
	return( (jint)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1doCheckpoint(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->doCheckpoint( (FLMUINT)iTimeout)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1dbLock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iLockType,
	jint				iPriority,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbLock( (eLockType)iLockType,
									(FLMINT)iPriority, (FLMUINT)iTimeout)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1dbUnlock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbUnlock()))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockType(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (int)lockType);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getLockImplicit(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( bImplicit ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockThreadId(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( 0, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiThreadId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumExclQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( 0, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumExclQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumSharedQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( 0, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumSharedQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockPriorityCount(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiPriorityCount);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1indexSuspend(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexSuspend( (FLMUINT)iIndexNum)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1indexResume(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexResume( (FLMUINT)iIndexNum)))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1indexGetNext(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iCurrIndex)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiCurrIndex = (FLMUINT)iCurrIndex;
	
	if (RC_BAD( rc = pDb->indexGetNext( &uiCurrIndex)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiCurrIndex);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1indexStatus(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndex)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Db *				pDb = THIS_FDB();
	XFLM_INDEX_STATUS	ixStatus;
	jclass				jIndexStatusClass = NULL;
	jobject				jIndexStatus = NULL;
	
	if (RC_BAD( rc = pDb->indexStatus( (FLMUINT)iIndex, &ixStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Find the IndexStatus class

	if ((jIndexStatusClass = pEnv->FindClass( "xflaim/IndexStatus")) == NULL)
	{
		goto Exit;
	}

	// Allocate an index status class.
	
	if ((jIndexStatus = pEnv->AllocObject( jIndexStatusClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetIntField( jIndexStatus, fid_IndexStatus_iIndexNum, iIndex);
	pEnv->SetIntField( jIndexStatus, fid_IndexStatus_iState, (jint)ixStatus.eState);
	pEnv->SetIntField( jIndexStatus, fid_IndexStatus_iStartTime, (jint)ixStatus.ui32StartTime);
	pEnv->SetLongField( jIndexStatus, fid_IndexStatus_lLastDocumentIndexed,
						(jlong)ixStatus.ui64LastDocumentIndexed);
	pEnv->SetLongField( jIndexStatus, fid_IndexStatus_lKeysProcessed,
						(jlong)ixStatus.ui64KeysProcessed);
	pEnv->SetLongField( jIndexStatus, fid_IndexStatus_lDocumentsProcessed,
						(jlong)ixStatus.ui64DocumentsProcessed);
	pEnv->SetLongField( jIndexStatus, fid_IndexStatus_lTransactions,
						(jlong)ixStatus.ui64Transactions);
	
Exit:

	return( jIndexStatus);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1reduceSize(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iCount)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiCount = 0;
	
	if (RC_BAD( rc = pDb->reduceSize( (FLMUINT)iCount, &uiCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiCount);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1keyRetrieve(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iIndex,
	jlong				lSearchKey,
	jint				iSearchFlags,
	jlong				lFoundKey)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = (IF_DataVector *)((FLMUINT)lSearchKey);
	IF_DataVector *	pFoundKey = (IF_DataVector *)((FLMUINT)lFoundKey);
	IF_Db *				pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->keyRetrieve( (FLMUINT)iIndex,
							pSearchKey, (FLMUINT)iSearchFlags, pFoundKey)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createDocument((FLMUINT)iCollection, &pNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createRootElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iElementNameId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createRootElement((FLMUINT)iCollection,
			(FLMUINT)iElementNameId, &pNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getFirstDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);
	
	if (RC_BAD( rc = pDb->getFirstDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getLastDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);
	
	if (RC_BAD( rc = pDb->getLastDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iSearchFlags,
	jlong				lDocumentId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);
	
	if (RC_BAD( rc = pDb->getDocument( (FLMUINT)iCollection,
									(FLMUINT)iSearchFlags, (FLMUINT64)lDocumentId,
									&pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JIJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lDocumentId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->documentDone( (FLMUINT)iCollection,
									(FLMUINT64)lDocumentId)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lNode)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)lNode);
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->documentDone( pNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createElementDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createElementDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createUniqueElmDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createUniqueElmDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											&uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getElementNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = 0;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getElementNameId( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createAttributeDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createAttributeDef( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getAttributeNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = 0;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getAttributeNameId( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createPrefixDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createPrefixDef( prefixName.getUnicodePtr(),
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getPrefixId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = 0;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getPrefixId( prefixName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createEncDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncType,
	jstring			sEncName,
	jint				iKeySize,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucEncTypeBuf [200];
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encType( ucEncTypeBuf, sizeof( ucEncTypeBuf));
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncType);
	if (RC_BAD( rc = getUniString( pEnv, sEncType, &encType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createEncDef( encType.getUnicodePtr(),
														encName.getUnicodePtr(),
														(FLMUINT)iKeySize,
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getEncDefId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = 0;
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getEncDefId( encName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createCollectionDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName,
	jint				iEncNumber,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucCollectionNameBuf [200];
	F_DynaBuf	collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createCollectionDef( collectionName.getUnicodePtr(),
														&uiNameId, (FLMUINT)iEncNumber)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getCollectionNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiNameId = 0;
	FLMBYTE			ucCollectionNameBuf [200];
	F_DynaBuf		collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getCollectionNumber( collectionName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getIndexNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sIndexName)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiNameId = 0;
	FLMBYTE			ucIndexNameBuf [200];
	F_DynaBuf		indexName( ucIndexNameBuf, sizeof( ucIndexNameBuf));
	
	flmAssert( sIndexName);
	if (RC_BAD( rc = getUniString( pEnv, sIndexName, &indexName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getIndexNumber( indexName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE mapDictType(
	jint			iDictType,
	FLMUINT *	puiDictType)
{
	RCODE	rc = NE_XFLM_OK;
	
	switch (iDictType)
	{
		case ELEMENT_DEF:
			*puiDictType = ELM_ELEMENT_TAG;
			break;
		case ATTRIBUTE_DEF:
			*puiDictType = ELM_ATTRIBUTE_TAG;
			break;
		case PREFIX_DEF:
			*puiDictType = ELM_PREFIX_TAG;
			break;
		case ENCRYPTION_DEF:
			*puiDictType = ELM_ENCDEF_TAG;
			break;
		case COLLECTION_DEF:
			*puiDictType = ELM_COLLECTION_TAG;
			break;
		case INDEX_DEF:
			*puiDictType = ELM_INDEX_TAG;
			break;
		default:
			rc = RC_SET( NE_XFLM_INVALID_PARM);
			goto Exit;
	}
	
Exit:

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDictionaryDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);
	FLMUINT			uiDictType = 0;

	if (RC_BAD( rc = mapDictType( iDictType, &uiDictType)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getDictionaryDef( uiDictType,
									(FLMUINT)iDictNumber, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	jint				iDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNameSize = 0;
	FLMUNICODE *	puzName = NULL;
	FLMUINT			uiDictType = 0;
	
	if (RC_BAD( rc = mapDictType( iDictType, &uiDictType)))
	{
		goto Exit;
	}
	
	// Determine how much space is needed to get the name.

	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											(FLMUNICODE *)NULL, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
		
	// uiNameSize comes back as number of characters, so to
	// get the buffer size needed, we need to add one for a null
	// terminator, and then multiply by the size of a unicode character.
	
	uiNameSize++;
	uiNameSize *= sizeof( FLMUNICODE);
	
	if (RC_BAD( rc = pDynaBuf->allocSpace( uiNameSize, (void **)&puzName)))
	{
		goto Exit;
	}
	
	// Now get the name.
	
	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											puzName, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											puzName, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
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
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getDictionaryName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, iDictType, (FLMUINT)iDictNumber,
									FALSE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getElementNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ELEMENT_DEF, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getAttributeNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ATTRIBUTE_DEF, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lNodeId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);

	if (RC_BAD( rc = pDb->getNode( (FLMUINT)iCollection, (FLMUINT64)lNodeId,
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
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lElementNodeId,
	jint				iAttrNameId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)((FLMUINT)lOldNodeRef)
														  : NULL);

	if (RC_BAD( rc = pDb->getAttribute( (FLMUINT)iCollection,
									(FLMUINT64)lElementNodeId,
								   (FLMUINT)iAttrNameId, &pNewNode)))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDataType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiDataType = 0;
	FLMUINT			uiDictType = 0;
	
	if (RC_BAD( rc = mapDictType( iDictType, &uiDictType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getDataType( uiDictType, (FLMUINT)iDictNumber,
									&uiDataType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiDataType);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1backupBegin(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iBackupType,
	jint				iTransType,
	jint				iMaxLockWait,
	jlong				lReusedRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Backup *		pBackup = lReusedRef
									 ? (IF_Backup *)((FLMUINT)lReusedRef)
									 : NULL;
	
	if (RC_BAD( rc = pDb->backupBegin( (eDbBackupType)iBackupType, 
			(eDbTransType)iTransType, (FLMUINT)iMaxLockWait, &pBackup)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
		
Exit:

	return( (jlong)((FLMUINT)pBackup));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1import(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jint				iCollection,
	jlong				lNodeToLinkTo,
	jint				iInsertLoc)
{
	RCODE						rc = NE_XFLM_OK;
	IF_IStream *			pIStream = (IF_IStream *)((FLMUINT)lIStream);
	IF_DOMNode *			pNodeToLinkTo = (IF_DOMNode *)((FLMUINT)lNodeToLinkTo);
	IF_Db *					pDb = THIS_FDB();
	XFLM_IMPORT_STATS		importStats;
	jclass					jImportStatsClass = NULL;
	jobject					jImportStats = NULL;
	
	if (!pIStream)
	{
		ThrowError( NE_XFLM_FAILURE, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pDb->import( pIStream, (FLMUINT)iCollection,
									pNodeToLinkTo, (eNodeInsertLoc)iInsertLoc,
									&importStats)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	// Find the ImportStats class

	if ((jImportStatsClass = pEnv->FindClass( "xflaim/ImportStats")) == NULL)
	{
		goto Exit;
	}

	// Allocate an import stats class.
	
	if ((jImportStats = pEnv->AllocObject( jImportStatsClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetIntField( jImportStats, fid_ImportStats_iLines, (jint)importStats.uiLines);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iChars, (jint)importStats.uiChars);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iAttributes, (jint)importStats.uiAttributes);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iElements, (jint)importStats.uiElements);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iText, (jint)importStats.uiText);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iDocuments, (jint)importStats.uiDocuments);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iErrLineNum, (jint)importStats.uiErrLineNum);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iErrLineOffset, (jint)importStats.uiErrLineOffset);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iErrorType, (jint)importStats.eErrorType);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iErrLineFilePos, (jint)importStats.uiErrLineFilePos);
	pEnv->SetIntField( jImportStats, fid_ImportStats_iErrLineBytes, (jint)importStats.uiErrLineBytes);
	pEnv->SetBooleanField( jImportStats, fid_ImportStats_bUTF8Encoding,
		(jboolean)(importStats.eXMLEncoding == XFLM_XML_UTF8_ENCODING
					  ? JNI_TRUE
					  : JNI_FALSE));
Exit:

	return( jImportStats);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1changeItemState(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNum,
	jstring			sState)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucState [80];
	F_DynaBuf	stateBuf( ucState, sizeof( ucState));
	FLMUINT		uiDictType = 0;
	
	if (RC_BAD( rc = mapDictType( iDictType, &uiDictType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = getUTF8String( pEnv, sState, &stateBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->changeItemState( uiDictType, (FLMUINT)iDictNum,
									(const char *)stateBuf.getBufferPtr())))
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
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getRflFileName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iFileNum,
	jboolean			bBaseOnly)
{
	IF_Db *		pDb = THIS_FDB();
	char			szRflFileName [F_PATH_MAX_SIZE];
	FLMUINT		uiFileNameBufSize = sizeof( szRflFileName);
	
	pDb->getRflFileName( (FLMUINT)iFileNum,
									(FLMBOOL)(bBaseOnly ? TRUE : FALSE),
									szRflFileName, &uiFileNameBufSize, NULL);
	return( pEnv->NewStringUTF( szRflFileName));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setNextNodeId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lNextNodeId)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setNextNodeId( (FLMUINT)iCollection,
									(FLMUINT64)lNextNodeId)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1setNextDictNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iNextDictNumber)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiDictType = 0;
	
	if (RC_BAD( rc = mapDictType( iDictType, &uiDictType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->setNextDictNum( uiDictType, (FLMUINT)iNextDictNumber)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflKeepFilesFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bKeep)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setRflKeepFilesFlag( (FLMBOOL)(bKeep ? TRUE : FALSE))))
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
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getRflKeepFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bKeep = FALSE;
	
	if (RC_BAD( rc = pDb->getRflKeepFlag( &bKeep)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( bKeep ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sRflDir)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucDirBuf [F_PATH_MAX_SIZE];
	F_DynaBuf	dirBuf( ucDirBuf, sizeof( ucDirBuf));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &dirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->setRflDir( (const char *)dirBuf.getBufferPtr())))
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
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getRflDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	char			szRflDir [F_PATH_MAX_SIZE];
	
	pDb->getRflDir( szRflDir);
	return( pEnv->NewStringUTF( szRflDir));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getRflFileNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiFileNum = 0;
	
	if (RC_BAD( rc = pDb->getRflFileNum( &uiFileNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiFileNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getHighestNotUsedRflFileNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiFileNum = 0;
	
	if (RC_BAD( rc = pDb->getHighestNotUsedRflFileNum( &uiFileNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiFileNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflFileSizeLimits(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iMinRflSize,
	jint				iMaxRflSize)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setRflFileSizeLimits( (FLMUINT)iMinRflSize,
									(FLMUINT)iMaxRflSize)))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMinRflFileSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiMinFileSize = 0;
	FLMUINT		uiMaxFileSize = 0;
	
	if (RC_BAD( rc = pDb->getRflFileSizeLimits( &uiMinFileSize, &uiMaxFileSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiMinFileSize);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMaxRflFileSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiMinFileSize = 0;
	FLMUINT		uiMaxFileSize = 0;
	
	if (RC_BAD( rc = pDb->getRflFileSizeLimits( &uiMinFileSize, &uiMaxFileSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiMaxFileSize);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1rflRollToNextFile(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->rflRollToNextFile()))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1setKeepAbortedTransInRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bKeep)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setKeepAbortedTransInRflFlag( (FLMBOOL)(bKeep ? TRUE : FALSE))))
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
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getKeepAbortedTransInRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bKeep = FALSE;
	
	if (RC_BAD( rc = pDb->getKeepAbortedTransInRflFlag( &bKeep)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jboolean)(bKeep ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setAutoTurnOffKeepRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bAutoTurnOff)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setAutoTurnOffKeepRflFlag( (FLMBOOL)(bAutoTurnOff ? TRUE : FALSE))))
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
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getAutoTurnOffKeepRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bAutoTurnOff = FALSE;
	
	if (RC_BAD( rc = pDb->getAutoTurnOffKeepRflFlag( &bAutoTurnOff)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jboolean)(bAutoTurnOff ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setFileExtendSize(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iFileExtendSize)
{
	IF_Db *		pDb = THIS_FDB();
	
	pDb->setFileExtendSize( (FLMUINT)iFileExtendSize);
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getFileExtendSize(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getFileExtendSize());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDbVersion(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getDbVersion());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getBlockSize(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getBlockSize());
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDefaultLanguage(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getDefaultLanguage());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getTransID(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jlong)pDb->getTransID());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getDbControlFileName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	char			szFileName [F_PATH_MAX_SIZE];
	
	szFileName [0] = 0;
	if (RC_BAD( rc = pDb->getDbControlFileName( szFileName, sizeof( szFileName))))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( pEnv->NewStringUTF( szFileName));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getLastBackupTransID(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64LastBackupTransID = 0;
	
	if (RC_BAD( rc = pDb->getLastBackupTransID( &ui64LastBackupTransID)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64LastBackupTransID);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getBlocksChangedSinceBackup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiBlocksChangedSinceBackup = 0;
	
	if (RC_BAD( rc = pDb->getBlocksChangedSinceBackup( &uiBlocksChangedSinceBackup)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiBlocksChangedSinceBackup);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getNextIncBackupSequenceNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNextIncBackupSequenceNum = 0;
	
	if (RC_BAD( rc = pDb->getNextIncBackupSequenceNum( &uiNextIncBackupSequenceNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiNextIncBackupSequenceNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceDataSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( &ui64Size, NULL, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceRollbackSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( NULL, &ui64Size, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceRflSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( NULL, NULL, &ui64Size)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceTotalSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64DataSize = 0;
	FLMUINT64	ui64RollbackSize = 0;
	FLMUINT64	ui64RflSize = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( &ui64DataSize, &ui64RollbackSize,
										&ui64RflSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(ui64DataSize + ui64RollbackSize + ui64RflSize));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMustCloseRC(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	return( (jint)pDb->getMustCloseRC());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getAbortRC(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	return( (jint)pDb->getAbortRC());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setMustAbortTrans(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iRc)
{
	IF_Db *		pDb = THIS_FDB();
	pDb->setMustAbortTrans( (RCODE)iRc);
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1enableEncryption(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->enableEncryption()))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1wrapKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPassword)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucPassword [200];
	F_DynaBuf	passwordBuf( ucPassword, sizeof( ucPassword));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->wrapKey( (const char *)passwordBuf.getBufferPtr())))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1rollOverDbKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->rollOverDbKey()))
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
JNIEXPORT jbyteArray JNICALL Java_xflaim_Db__1getSerialNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	char			ucSerialNumber [XFLM_SERIAL_NUM_SIZE];
	jbyteArray	jSerialNumber;
	
	pDb->getSerialNumber( ucSerialNumber);
	
	if ((jSerialNumber = pEnv->NewByteArray( (jsize)XFLM_SERIAL_NUM_SIZE)) == NULL)
	{
		goto Exit;
	}
	
   pEnv->SetByteArrayRegion( jSerialNumber, (jsize)0, (jsize)XFLM_SERIAL_NUM_SIZE,
					(jbyte *)ucSerialNumber);

Exit:

	return( jSerialNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1getCheckpointInfo(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *					pDb = THIS_FDB();
	XFLM_CHECKPOINT_INFO	checkpointInfo;
	jclass					jCheckpointInfoClass = NULL;
	jobject					jCheckpointInfo = NULL;
	
	pDb->getCheckpointInfo( &checkpointInfo);

	// Find the IndexStatus class

	if ((jCheckpointInfoClass = pEnv->FindClass( "xflaim/CheckpointInfo")) == NULL)
	{
		goto Exit;
	}

	// Allocate an index status class.
	
	if ((jCheckpointInfo = pEnv->AllocObject( jCheckpointInfoClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetBooleanField( jCheckpointInfo, fid_CheckpointInfo_bRunning,
		(jboolean)(checkpointInfo.bRunning ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iRunningTime,
		(jint)checkpointInfo.ui32RunningTime);
	pEnv->SetBooleanField( jCheckpointInfo, fid_CheckpointInfo_bForcingCheckpoint,
		(jboolean)(checkpointInfo.bForcingCheckpoint ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iForceCheckpointRunningTime,
		(jint)checkpointInfo.ui32ForceCheckpointRunningTime);
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iForceCheckpointReason,
		(jint)checkpointInfo.ui32ForceCheckpointReason);
	pEnv->SetBooleanField( jCheckpointInfo, fid_CheckpointInfo_bWritingDataBlocks,
		(jboolean)(checkpointInfo.bWritingDataBlocks ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iLogBlocksWritten,
		(jint)checkpointInfo.ui32LogBlocksWritten);
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iDataBlocksWritten,
		(jint)checkpointInfo.ui32DataBlocksWritten);
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iDirtyCacheBytes,
		(jint)checkpointInfo.ui32DirtyCacheBytes);
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iBlockSize,
		(jint)checkpointInfo.ui32BlockSize);
	pEnv->SetIntField( jCheckpointInfo, fid_CheckpointInfo_iWaitTruncateTime,
		(jint)checkpointInfo.ui32WaitTruncateTime);
	
Exit:

	return( jCheckpointInfo);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1exportXML__JJLjava_lang_String_2I(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lStartNode,
	jstring			sFileName,
	jint				iFormat)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pStartNode = (IF_DOMNode *)((FLMUINT)lStartNode);
	IF_OStream *	pOStream = NULL;
	FLMBYTE			ucFileName [F_PATH_MAX_SIZE];
	F_DynaBuf		fileNameBuf( ucFileName, sizeof( ucFileName));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sFileName, &fileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = FlmOpenFileOStream( (const char *)fileNameBuf.getBufferPtr(),
								TRUE, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->exportXML( pStartNode, pOStream,
							(eExportFormatType)iFormat)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pOStream)
	{
		pOStream->Release();
	}

	return;
}

/****************************************************************************
Desc:	Output stream that writes to a dynamic buffer.
****************************************************************************/
class DynaBufOStream : public IF_OStream
{
public:

	DynaBufOStream(
		F_DynaBuf *	pDynaBuf)
	{
		m_pDynaBuf = pDynaBuf;
		m_pDynaBuf->truncateData( 0);
	}
	
	virtual ~DynaBufOStream()
	{
	}
	
	RCODE XFLAPI write(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite,
		FLMUINT *		puiBytesWritten = NULL)
	{
		RCODE	rc = NE_XFLM_OK;
		
		if (RC_BAD( rc = m_pDynaBuf->appendData( pvBuffer, uiBytesToWrite)))
		{
			goto Exit;
		}
		if (puiBytesWritten)
		{
			*puiBytesWritten = uiBytesToWrite;
		}
	Exit:
		return( rc);
	}
	
	RCODE XFLAPI closeStream( void)
	{
		return( m_pDynaBuf->appendByte( 0));
	}
	
private:

	F_DynaBuf *	m_pDynaBuf;
	
};
	

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1exportXML__JJI(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lStartNode,
	jint				iFormat)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pStartNode = (IF_DOMNode *)((FLMUINT)lStartNode);
	FLMBYTE			ucBuffer [512];
	F_DynaBuf		dynaBuf( ucBuffer, sizeof( ucBuffer));
	DynaBufOStream	dynaOStream( &dynaBuf);
	jstring			jXML = NULL;
	
	if (RC_BAD( rc = pDb->exportXML( pStartNode, &dynaOStream,
							(eExportFormatType)iFormat)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	dynaOStream.closeStream();
	
	// Create a string and return it.

	jXML = pEnv->NewStringUTF( (const char *)dynaBuf.getBufferPtr());

Exit:

	return( jXML);
}

/****************************************************************************
Desc:	Client for getting lock information.
****************************************************************************/
class	JavaLockInfoClient : public IF_LockInfoClient
{
public:

	JavaLockInfoClient()
	{
		m_pLockUsers = NULL;
		m_uiNumLockUsers = 0;
	}
	
	virtual ~JavaLockInfoClient()
	{
		if (m_pLockUsers)
		{
			f_free( &m_pLockUsers);
		}
	}
	
	FLMBOOL XFLAPI setLockCount(
		FLMUINT	uiTotalLocks)
	{
		if (RC_BAD( f_calloc( sizeof( F_LOCK_USER) * uiTotalLocks,
								&m_pLockUsers)))
		{
			return( FALSE);
		}
		else
		{
			m_uiNumLockUsers = uiTotalLocks;
			return( TRUE);
		}
	}
	
	FLMBOOL XFLAPI addLockInfo(
		FLMUINT		uiLockNum,
		FLMUINT		uiThreadID,
		FLMUINT		uiTime)
	{
		if (uiLockNum < m_uiNumLockUsers)
		{
			m_pLockUsers [uiLockNum].uiThreadId = uiThreadID;
			m_pLockUsers [uiLockNum].uiTime = uiTime;
		}
		return( TRUE);
	}
	
	FINLINE FLMUINT getLockCount( void)
	{
		return( m_uiNumLockUsers);
	}
	
	FINLINE F_LOCK_USER * getLockUsers( void)
	{
		return( m_pLockUsers);
	}
	
private:

	F_LOCK_USER *	m_pLockUsers;
	FLMUINT			m_uiNumLockUsers;
};
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobjectArray JNICALL Java_xflaim_Db__1getLockWaiters(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Db *					pDb = THIS_FDB();
	jclass					jLockUserClass = NULL;
	jobjectArray			jLockUsers = NULL;
	jobject					jLockUser = NULL;
	JavaLockInfoClient	lockInfoClient;
	F_LOCK_USER *			pLockUser;
	FLMUINT					uiLoop;
	
	if (RC_BAD( rc = pDb->getLockWaiters( &lockInfoClient)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Find the LockUser class

	if ((jLockUserClass = pEnv->FindClass( "xflaim/LockUser")) == NULL)
	{
		goto Exit;
	}

	// Allocate the array of lock users.
	
	if ((jLockUsers = pEnv->NewObjectArray( (jsize)lockInfoClient.getLockCount(),
										jLockUserClass, NULL)) == NULL)
	{
		goto Exit;
	}
	
	// Now allocate each lock user and set its members.
	
	for (uiLoop = 0, pLockUser = lockInfoClient.getLockUsers();
		  uiLoop < lockInfoClient.getLockCount();
		  uiLoop++, pLockUser++)
	{
		// Allocate a lock user object.
		
		if ((jLockUser = pEnv->AllocObject( jLockUserClass)) == NULL)
		{
			goto Exit;
		}
		pEnv->SetIntField( jLockUser, fid_LockUser_iThreadId, (jint)pLockUser->uiThreadId);
		pEnv->SetIntField( jLockUser, fid_LockUser_iTime, (jint)pLockUser->uiTime);
		pEnv->SetObjectArrayElement( jLockUsers, (jsize)uiLoop, jLockUser);
    }

Exit:

	return( jLockUsers);
}

/****************************************************************************
Desc: Delete status callback
****************************************************************************/
class JavaDeleteStatus : public IF_DeleteStatus
{
public:

	JavaDeleteStatus(
		JNIEnv *		pEnv,
		jobject		jDeleteStatusObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jDeleteStatusObject = pEnv->NewGlobalRef( jDeleteStatusObject);
		m_jReportDeleteMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDeleteStatusObject),
													"reportDelete",
													"(II)I");
	}
	
	virtual ~JavaDeleteStatus()
	{
		if (m_jDeleteStatusObject)
		{
			m_pEnv->DeleteGlobalRef( m_jDeleteStatusObject);
		}
	}
			
	RCODE XFLAPI reportDelete(
		FLMUINT	uiBlocksDeleted,
		FLMUINT	uiBlockSize)
	{
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setDeleteStatusObject method was called.  It will not
		// work to set the delete status object in one thread, but then do
		// the delete operation in another thread.
		
		return( (RCODE)m_pEnv->CallIntMethod( m_jDeleteStatusObject,
										m_jReportDeleteMethodId, (jint)uiBlocksDeleted,
										(jint)uiBlockSize));
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jDeleteStatusObject;
	jmethodID	m_jReportDeleteMethodId;
	
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setDeleteStatusObject(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jobject			jDeleteStatusObject)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Db *					pDb = THIS_FDB();
	JavaDeleteStatus *	pDeleteStatusObj = NULL;
	
	if (jDeleteStatusObject)
	{
		if ((pDeleteStatusObj = f_new JavaDeleteStatus( pEnv,
													jDeleteStatusObject)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}

	pDb->setDeleteStatusObject( pDeleteStatusObj);
	
Exit:

	return;
}

/****************************************************************************
Desc: Indexing client callback
****************************************************************************/
class JavaIxClient : public IF_IxClient
{
public:

	JavaIxClient(
		JNIEnv *		pEnv,
		jobject		jIxClientObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jIxClientObject = pEnv->NewGlobalRef( jIxClientObject);
		m_jDoIndexingMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jIxClientObject),
													"doIndexing",
													"(IIJ)I");
	}
	
	virtual ~JavaIxClient()
	{
		if (m_jIxClientObject)
		{
			m_pEnv->DeleteGlobalRef( m_jIxClientObject);
		}
	}
			
	RCODE XFLAPI doIndexing(
		IF_Db *			pDb,
		FLMUINT			uiIndexNum,
		FLMUINT			uiCollection,
		IF_DOMNode *	pDocNode)
	{
		RCODE			rc = NE_XFLM_OK;
		FLMUINT64	ui64NodeId;
		
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setIndexingClientObject method was called.  It will not
		// work to set the index client object in one thread, but then do
		// the index operation in another thread.
		
		if (RC_BAD( rc = pDocNode->getNodeId( pDb, &ui64NodeId)))
		{
			goto Exit;
		}
		rc = (RCODE)m_pEnv->CallIntMethod( m_jIxClientObject,
										m_jDoIndexingMethodId, (jint)uiIndexNum,
										(jint)uiCollection, (jlong)ui64NodeId);
	Exit:
	
		return( rc);
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jIxClientObject;
	jmethodID	m_jDoIndexingMethodId;
	
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setIndexingClientObject(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jobject			jIxClientObject)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Db *					pDb = THIS_FDB();
	JavaIxClient *			pIxClientObj = NULL;
	
	if (jIxClientObject)
	{
		if ((pIxClientObj = f_new JavaIxClient( pEnv,
													jIxClientObject)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}

	pDb->setIndexingClientObject( pIxClientObj);
	
Exit:

	return;
}

/****************************************************************************
Desc: Indexing status callback
****************************************************************************/
class JavaIxStatus : public IF_IxStatus
{
public:

	JavaIxStatus(
		JNIEnv *		pEnv,
		jobject		jIxStatusObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jIxStatusObject = pEnv->NewGlobalRef( jIxStatusObject);
		m_jReportIndexMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jIxStatusObject),
													"reportIndex",
													"(J)I");
	}
	
	virtual ~JavaIxStatus()
	{
		if (m_jIxStatusObject)
		{
			m_pEnv->DeleteGlobalRef( m_jIxStatusObject);
		}
	}
			
	RCODE XFLAPI reportIndex(
		FLMUINT64		ui64LastDocumentId)
	{
		
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setIndexingStatusObject method was called.  It will not
		// work to set the index status object in one thread, but then do
		// the index operation in another thread.
		
		return( (RCODE)m_pEnv->CallIntMethod( m_jIxStatusObject,
									m_jReportIndexMethodId, (jlong)ui64LastDocumentId));
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jIxStatusObject;
	jmethodID	m_jReportIndexMethodId;
	
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setIndexingStatusObject(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jobject			jIxStatusObject)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Db *					pDb = THIS_FDB();
	JavaIxStatus *			pIxStatusObj = NULL;
	
	if (jIxStatusObject)
	{
		if ((pIxStatusObj = f_new JavaIxStatus( pEnv,
													jIxStatusObject)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}

	pDb->setIndexingStatusObject( pIxStatusObj);
	
Exit:

	return;
}

/****************************************************************************
Desc: Commit client callback
****************************************************************************/
class JavaCommitClient : public IF_CommitClient
{
public:

	JavaCommitClient(
		JNIEnv *		pEnv,
		jobject		dbObj,
		jobject		jCommitClientObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jCommitClientObject = pEnv->NewGlobalRef( jCommitClientObject);
		m_jDbObj = pEnv->NewGlobalRef( dbObj);
		m_jCommitMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jCommitClientObject),
													"commit",
													"(Lxflaim/Db;)V");
	}
	
	virtual ~JavaCommitClient()
	{
		if (m_jCommitClientObject)
		{
			m_pEnv->DeleteGlobalRef( m_jCommitClientObject);
		}
		if (m_jDbObj)
		{
			m_pEnv->DeleteGlobalRef( m_jDbObj);
		}
	}
			
	void XFLAPI commit(
		IF_Db *			// pDb
		)
	{
		
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setCommitClientObject method was called.  It will not
		// work to set the commit client object in one thread, but then do
		// the commit operation in another thread.
		
		m_pEnv->CallVoidMethod( m_jCommitClientObject, m_jCommitMethodId, m_jDbObj);
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jDbObj;
	jobject		m_jCommitClientObject;
	jmethodID	m_jCommitMethodId;
	
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setCommitClientObject(
	JNIEnv *			pEnv,
	jobject			dbObj,
	jlong				lThis,
	jobject			jCommitClientObject)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Db *					pDb = THIS_FDB();
	JavaCommitClient *	pCommitClientObj = NULL;
	
	if (jCommitClientObject)
	{
		if ((pCommitClientObj = f_new JavaCommitClient( pEnv, dbObj,
													jCommitClientObject)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}

	pDb->setCommitClientObject( pCommitClientObj);
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1upgrade(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE		rc = NE_XFLM_OK;
	IF_Db *	pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->upgrade( NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

